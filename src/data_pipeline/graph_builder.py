import networkx as nx
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
    """
    def __init__(self):
        pass

    def _build_single_graph(self, df):
        """
        Internal method to convert a single dataframe into a DGL graph.
        """
        # Step 1: Create a NetworkX MultiGraph from pandas edgelist.
        # Each row of df becomes one EDGE: source/target columns become the
        # two node endpoints (IPs), and edge_attr lists which columns are
        # stored as edge attributes -- 'h' (the flow feature vector we built
        # during preprocessing), plus 'Label' and 'Attack' for later
        # evaluation.
        #
        # MultiGraph (not Graph) is required because the SAME pair of IPs can
        # have several distinct flows between them (e.g. one benign flow and
        # one attack flow between 10.0.0.1 and 10.0.0.5). A plain Graph only
        # allows a single edge per node pair and would silently overwrite
        # earlier flows; MultiGraph keeps every flow as its own parallel edge.
        nx_g = nx.from_pandas_edgelist(
            df, 
            source="IPV4_SRC_ADDR", 
            target="IPV4_DST_ADDR",
            edge_attr=["h", "Label", "Attack"], 
            create_using=nx.MultiGraph()
        )
        
        # Step 2: Convert to directed graph representing bidirectional flows.
        # A network flow is inherently directional (A->B is not the same
        # event as B->A, e.g. a request vs. its response), so we need a
        # directed representation rather than an undirected one.
        nx_g = nx_g.to_directed()
        
        # Step 3: Convert NetworkX graph to DGL graph.
        # This is purely a format conversion -- same graph, same edges, same
        # attributes -- but now in a data structure DGL/PyTorch can run GNN
        # operations (like the AGG aggregation from the paper's formulas) on.
        dgl_g = dgl.from_networkx(nx_g, edge_attrs=['h', 'Attack', 'Label'])
        
        # Step 4: Initialize constant node features
        # (vector of 1s matching edge feature dimensions).
        #
        # IMPORTANT: this does NOT touch or overwrite edge features in any
        # way. dgl_g.ndata (node data) and dgl_g.edata (edge data) are two
        # completely separate structures. The real flow statistics (e.g.
        # [0.6, 0.8, 0.1]) stay untouched in dgl_g.edata['h']; here we are
        # only creating a brand-new feature for the NODES, which never had
        # any real per-IP attributes to begin with -- a NetFlow record only
        # describes the flow between two IPs, never a standalone property of
        # a single IP. So we assign every node a placeholder vector of 1s,
        # exactly as the paper specifies: "node features are set to a
        # constant vector containing ones with the same dimensions as those
        # of the edge features".
        edge_feat_dim = len(dgl_g.edata['h'][0])
        nfeat_weight = torch.ones([dgl_g.number_of_nodes(), edge_feat_dim])
        dgl_g.ndata['h'] = nfeat_weight
        
        # Step 5: Reshape node and edge features for E-GraphSAGE compatibility.
        # Adding an extra middle dimension (size 1) so tensors go from shape
        # (num_nodes, feat_dim) to (num_nodes, 1, feat_dim), and likewise for
        # edges. This does not change any values -- it only repackages the
        # tensor shape to match what the upcoming E-GraphSAGE/DGL message-
        # passing layers expect as input format.
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
        avoid any data leakage.
        """
        print("[INFO] Building Training Graph...")
        train_g = self._build_single_graph(train_df)
        
        print("[INFO] Building Testing Graph...")
        test_g = self._build_single_graph(test_df)
        
        print(f"[SUCCESS] Graphs generated. Train Nodes: {train_g.number_of_nodes()}, Train Edges: {train_g.number_of_edges()}")
        return train_g, test_g