import pandas as pd
import numpy as np
import dgl
import torch


class AnomalEGraphBuilder:
    """
    Constructs PyTorch/DGL compatible graphs from preprocessed NetFlow dataframes.
    Assigns constant features to nodes and network flow statistics to edges.

    This corresponds to the final step of Fig. 4 in the Anomal-E paper
    ("Train & Test Graph Generation"), run AFTER the AnomalEPreprocessor
    pipeline. Input dataframes must already contain the 'h' column (the
    normalised edge feature vector built in apply_normalization).

    IMPLEMENTATION NOTE (why no NetworkX): the original implementation built
    the graph via nx.from_pandas_edgelist(...) + .to_directed() +
    dgl.from_networkx(...). NetworkX represents every single edge as its own
    Python dictionary object, and .to_directed() copies the entire graph
    while doubling the edge count -- for a NetFlow dataset with hundreds of
    thousands to millions of rows, this is both extremely slow and memory-
    hungry (it was exhausting Colab's RAM / taking a very long time even
    after upstream memory fixes). This version builds the same graph
    structure directly with pandas (for IP -> integer node-ID mapping) and
    DGL/PyTorch tensor operations (for edge duplication and feature
    stacking), which are all vectorised, array-based operations instead of
    a Python object per edge. The resulting graph is structurally identical
    to what the NetworkX version produced (same nodes, same directed edges,
    same edge/node feature values and shapes) -- only the construction path
    changed.
    """
    def __init__(self):
        pass

    def _build_single_graph(self, df):
        """
        Internal method to convert a single dataframe into a DGL graph,
        without going through NetworkX.

        Mirrors the exact graph the previous NetworkX-based implementation
        produced:
            - Each row of df is one directed flow: IPV4_SRC_ADDR -> IPV4_DST_ADDR.
            - Because the previous version built an undirected MultiGraph and
              then called .to_directed(), every original flow ended up
              represented as TWO directed edges: src->dst AND dst->src, both
              carrying the SAME edge attributes (h, Label, Attack). We
              reproduce that here explicitly (see Step 2 below) rather than
              relying on any implicit undirected-to-directed conversion.
            - Node features are set to a constant vector of ones, matching
              the paper's "node features are set to a constant vector
              containing ones with the same dimensions as those of the edge
              features" (Section 4.2/4.3 of the Anomal-E paper).
        """
        # Step 1: Map every distinct IP address string in this dataframe to
        # a small-integer node ID. pandas' factorize() does this in one
        # vectorised pass (no per-row Python object creation, unlike
        # NetworkX's node/edge dictionaries).
        #
        # IMPORTANT (train/test independence): factorize() is run separately
        # for train_df and test_df (this method is called once per graph, see
        # generate_graphs() below), so an IP address that appears in BOTH
        # train and test data gets an UNRELATED node ID in each graph -- e.g.
        # node 5 in train_g and node 5 in test_g are not "the same" node in
        # any meaningful sense, they just happen to share a numeric ID. This
        # intentionally matches the original NetworkX-based implementation,
        # where train_g and test_g were built as two fully independent graphs
        # with no shared node identity, consistent with the paper's strict
        # train/test separation to avoid data leakage.
        all_ips = pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]], ignore_index=True)
        node_ids, unique_ips = pd.factorize(all_ips)
        num_nodes = len(unique_ips)
        num_rows = len(df)

        src_ids = node_ids[:num_rows]
        dst_ids = node_ids[num_rows:]

        # Step 2: Build directed edges in BOTH directions (src->dst and
        # dst->src) for every flow, with identical edge features on both
        # copies. This reproduces what the old
        # MultiGraph(undirected) -> to_directed() pipeline did: an
        # undirected edge {u, v} becomes two directed edges u->v and v->u
        # when converted to a directed graph.
        src_all = np.concatenate([src_ids, dst_ids])
        dst_all = np.concatenate([dst_ids, src_ids])

        # Step 3: Stack the edge feature vectors ('h', built in
        # apply_normalization) into one (num_rows, feat_dim) float32 array,
        # then duplicate it for the reverse-direction copy of every edge.
        # np.stack on the already-float32 per-row arrays from
        # apply_normalization is a single vectorised allocation, not a
        # per-edge Python object.
        h_values = np.stack(df["h"].values).astype(np.float32)
        h_all = np.concatenate([h_values, h_values], axis=0)

        # Step 4: Same duplication for Label and Attack, which trainer.py's
        # evaluate() and encode_labels() rely on being present as edge
        # attributes, in the same edge order as the feature tensor.
        label_values = df["Label"].to_numpy()
        label_all = np.concatenate([label_values, label_values], axis=0)

        attack_values = df["Attack"].to_numpy()
        attack_all = np.concatenate([attack_values, attack_values], axis=0)

        # Step 5: Build the DGL graph directly from integer edge endpoints --
        # this is the same graph.DGLGraph structure dgl.from_networkx() would
        # have produced, just built without ever materialising a NetworkX
        # graph in between.
        dgl_g = dgl.graph(
            (torch.from_numpy(src_all).long(), torch.from_numpy(dst_all).long()),
            num_nodes=num_nodes,
        )

        dgl_g.edata['h'] = torch.from_numpy(h_all)
        dgl_g.edata['Label'] = torch.from_numpy(label_all)
        dgl_g.edata['Attack'] = torch.from_numpy(attack_all)

        # Step 6: Initialize constant node features
        # (vector of 1s matching edge feature dimensions).
        #
        # IMPORTANT: this does NOT touch or overwrite edge features in any
        # way. dgl_g.ndata (node data) and dgl_g.edata (edge data) are two
        # completely separate structures. The real flow statistics stay
        # untouched in dgl_g.edata['h']; here we are only creating a
        # brand-new feature for the NODES, which never had any real per-IP
        # attributes to begin with -- a NetFlow record only describes the
        # flow between two IPs, never a standalone property of a single IP.
        # So we assign every node a placeholder vector of 1s, exactly as the
        # paper specifies.
        edge_feat_dim = dgl_g.edata['h'].shape[1]
        dgl_g.ndata['h'] = torch.ones([dgl_g.number_of_nodes(), edge_feat_dim])

        # Step 7: Reshape node and edge features for E-GraphSAGE compatibility.
        # Adding an extra middle dimension (size 1) so tensors go from shape
        # (num_nodes, feat_dim) to (num_nodes, 1, feat_dim), and likewise for
        # edges. This does not change any values -- it only repackages the
        # tensor shape to match what the E-GraphSAGE/DGL message-passing
        # layers expect as input format. Identical to the previous
        # implementation's Step 5.
        dgl_g.ndata['h'] = torch.reshape(
            dgl_g.ndata['h'],
            (dgl_g.ndata['h'].shape[0], 1, dgl_g.ndata['h'].shape[1])
        )

        dgl_g.edata['h'] = torch.reshape(
            dgl_g.edata['h'],
            (dgl_g.edata['h'].shape[0], 1, dgl_g.edata['h'].shape[1])
        )

        return dgl_g

    def generate_graphs(self, train_df, test_df):
        """
        Takes preprocessed Train and Test dataframes and returns DGL graphs.

        Train and test graphs are built completely independently (separate
        nodes and edges, no mixing) -- consistent with the paper's emphasis
        on keeping train/test strictly separate throughout preprocessing to
        avoid any data leakage. An IP address appearing in both train_df and
        test_df will get an unrelated node ID in each resulting graph, same
        as in the previous NetworkX-based implementation.
        """
        print("[INFO] Building Training Graph...")
        train_g = self._build_single_graph(train_df)

        print("[INFO] Building Testing Graph...")
        test_g = self._build_single_graph(test_df)

        print(f"[SUCCESS] Graphs generated. Train Nodes: {train_g.number_of_nodes()}, Train Edges: {train_g.number_of_edges()}")
        return train_g, test_g