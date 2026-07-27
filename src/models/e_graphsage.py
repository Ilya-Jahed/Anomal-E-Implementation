import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class AnomalESAGELayer(nn.Module):
    """
    Custom E-GraphSAGE Layer for Anomal-E.
    Aggregates edge features into node representations, and updates edge representations
    based on the updated source and destination nodes.

    This implements Equations (4), (2), and (5) from the Anomal-E paper (Algorithm 1):
      - Eq. (4): neighbourhood aggregation using EDGE features (not node features)
      - Eq. (2): combining a node's own representation with its aggregated neighbourhood
      - Eq. (5): building the final edge embedding by concatenating the two endpoint
                 node embeddings (here extended with an extra learnable linear layer,
                 W_edge, rather than a raw concatenation as in the paper).
    """
    def __init__(self, ndim_in, edims, ndim_out, edge_out_dim=256, activation=F.relu):
        """
        Args:
            ndim_in (int): Input dimensionality of node features.
            edims (int): Dimensionality of edge features.
            ndim_out (int): Output dimensionality of node features.
            edge_out_dim (int): Output dimensionality of edge features.
            activation (callable): Activation function to apply.
        """
        super(AnomalESAGELayer, self).__init__()
        
        # Linear layer for updating node features (combines original node feats + aggregated edge feats).
        # This is W^k from Eq. (2) of the paper. Input size is ndim_in + edims because it
        # receives a CONCATENATED vector: [node's own features | aggregated neighbour info].
        self.W_apply = nn.Linear(ndim_in + edims, ndim_out)
        self.activation = activation
        
        # Linear layer for updating edge features (combines source and destination node feats).
        # The paper's Eq. (5) just concatenates z_u and z_v with no extra weight matrix;
        # this implementation adds a learnable linear layer on top of that concatenation,
        # so the input size is ndim_out * 2 (the two endpoint embeddings, each ndim_out long).
        self.W_edge = nn.Linear(ndim_out * 2, edge_out_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters for better convergence using Xavier Uniform.
        Standard practice for stabilising the very start of training -- avoids
        gradients that are too large or too small right from initialisation.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_apply.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_edge.weight, gain=gain)

    def message_func(self, edges):
        """
        Message function: Sends edge features to destination nodes.

        DGL splits message passing into two parts: a "message function" (what
        gets sent along each edge) and a "reduce function" (how messages
        arriving at a node get combined). Here the message is simply the
        edge's own feature vector -- this is exactly e_uv from the paper's
        Eq. (4): each edge hands its flow-feature vector to its destination
        node so it can be aggregated there.
        """
        return {'m': edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
        """
        Forward pass for the E-GraphSAGE layer.
        
        Args:
            g_dgl (DGLGraph): The input DGL graph.
            nfeats (Tensor): Node features of shape (N, 1, ndim_in).
            efeats (Tensor): Edge features of shape (E, 1, edims).
            
        Returns:
            updated_nfeats (Tensor): Updated node features.
            updated_efeats (Tensor): Updated edge features.

        Worked numeric example (2 nodes A, B; 1 edge A->B):
            h_A = [1, 1], h_B = [1, 1]   (constant node features, ndim_in=2)
            h_AB = [0.5, 0.3]             (edge feature, edims=2)

            Step 1 (aggregation): B has one incoming edge, so
                h_neigh_B = mean([0.5, 0.3]) = [0.5, 0.3]
            Step 2 (node update): concat h_B with h_neigh_B ->
                node_cat_B = [1, 1, 0.5, 0.3]  (length ndim_in+edims=4)
                updated_h_B = ReLU(W_apply @ node_cat_B)
            Step 3 (edge update): concat updated h_A and updated h_B ->
                edge_cat_AB = [updated_h_A | updated_h_B]  (length ndim_out*2)
                updated_h_AB = W_edge @ edge_cat_AB
        """
        # Use local_scope so we don't permanently modify the original graph during message passing.
        # This matters because the SAME graph object is reused for both the real
        # forward pass and the corrupted forward pass (see AnomalESAGEEncoder below) --
        # local_scope guarantees whatever we write into g.ndata/g.edata here is
        # discarded once this function returns, leaving the original graph untouched.
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            
            # Step 1: Message Passing & Aggregation.
            # Send edge features ('h') as messages ('m'), and average them at the
            # destination node ('h_neigh'). This is Eq. (4) of the paper: neighbourhood
            # aggregation performed over EDGE features (not node features), with
            # mean() as the chosen AGG_k aggregator function.
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            
            # Step 2: Node Update.
            # Concatenate original node features with the aggregated neighborhood features
            # along dimension 2 (the feature dimension -- dims 0 and 1 are node-index and
            # the singleton dimension from preprocessing's reshape step).
            # Shape becomes: (N, 1, ndim_in + edims)
            # This is Eq. (2): h_v^k = sigma(W^k . CONCAT(h_v^(k-1), h_N(v)^k))
            node_cat = torch.cat([g.ndata['h'], g.ndata['h_neigh']], dim=2)
            updated_nfeats = self.activation(self.W_apply(node_cat))
            g.ndata['h'] = updated_nfeats

            # Step 3: Edge Update.
            # Extract source (u) and destination (v) node indices for all edges.
            u, v = g.edges()
            
            # Concatenate the UPDATED features of the source and destination nodes.
            # Shape becomes: (E, 1, ndim_out * 2)
            # This is Eq. (5): z_uv^K = CONCAT(z_u^K, z_v^K) -- extended here with a
            # learnable linear layer (W_edge) on top of the raw concatenation.
            edge_cat = torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), dim=2)
            updated_efeats = self.W_edge(edge_cat)
            
            return updated_nfeats, updated_efeats


class AnomalESAGEEncoder(nn.Module):
    """
    The main GraphSAGE Encoder for Anomal-E.
    Stacks E-GraphSAGE layers and provides a corruption mechanism for contrastive learning (DGI).

    The `corrupt` flag implements exactly the DGI corruption function C(G) discussed
    earlier: edge features are randomly shuffled among edges while the graph's
    topology (adjacency / which nodes are connected) is left completely unchanged.
    This encoder is called ONCE on the real graph (corrupt=False) and ONCE on the
    same graph again (corrupt=True) during each training step -- both passes share
    the exact same weights (self.layers), which is what forces the discriminator
    (implemented elsewhere, in the DGI training loop) to learn a real vs. corrupted
    distinction based on graph content, not based on any difference between models.
    """
    def __init__(self, ndim_in, edims, hidden_dim, edge_hidden_dim=256, activation=F.relu):
        """
        Args:
            ndim_in (int): Dimension of input node features.
            edims (int): Dimension of input edge features.
            hidden_dim (int): Output dimension for node embeddings.
            edge_hidden_dim (int): Output dimension for edge embeddings.
            activation (callable): Activation function for the layer.
        """
        super(AnomalESAGEEncoder, self).__init__()
        self.layers = nn.ModuleList()
        
        # Currently, Anomal-E uses a single E-GraphSAGE layer as per the reference notebook.
        # This matches the paper's own design choice (Section 4.3): a 1-layer model is
        # used instead of 2 layers, because the DGI objective benefits from a wider
        # rather than deeper encoder. This can be expanded to multiple layers if required.
        self.layers.append(
            AnomalESAGELayer(
                ndim_in=ndim_in, 
                edims=edims, 
                ndim_out=hidden_dim, 
                edge_out_dim=edge_hidden_dim, 
                activation=activation
            )
        )

    def forward(self, g, nfeats, efeats, corrupt=False):
        """
        Args:
            g (DGLGraph): The input graph.
            nfeats (Tensor): Node features.
            efeats (Tensor): Edge features.
            corrupt (bool): If True, randomly permutes edge features to generate negative samples.
            
        Returns:
            nfeats_sum (Tensor): Summarized node embeddings across dimension 1.
            efeats_sum (Tensor): Summarized edge embeddings across dimension 1.

        Worked numeric example of corruption (3 edges):
            efeats = [[0.5, 0.3],   # edge 0
                      [0.1, 0.9],   # edge 1
                      [0.7, 0.2]]   # edge 2
            e_perm = randperm(3) -> e.g. [2, 0, 1]
            efeats[e_perm] = [[0.7, 0.2],   # now sitting on edge 0
                               [0.5, 0.3],   # now sitting on edge 1
                               [0.1, 0.9]]   # now sitting on edge 2
            The graph's adjacency (who is connected to whom) is completely
            untouched -- only which feature vector sits on which edge changes.
        """
        if corrupt:
            # Generate a random permutation of indices for the edges to create negative samples.
            # This is the DGI corruption function C(G): shuffle edge features among edges,
            # keep the adjacency matrix A identical (paper: "we use corruption function C,
            # which shuffles the edge features such that X~ != X, but retains the
            # adjacency matrix A~ = A").
            e_perm = torch.randperm(g.number_of_edges())
            efeats = efeats[e_perm]
            
        for layer in self.layers:
            nfeats, e_feats = layer(g, nfeats, efeats)
            
        # Summing along dimension 1 to collapse the unsqueezed extra dimension 
        # (originally added during data preprocessing for PyTorch compatibility).
        # That middle dimension has size 1, so summing over it is a no-op on the
        # values themselves -- it purely reshapes (N, 1, dim) back down to (N, dim).
        return nfeats.sum(1), e_feats.sum(1)