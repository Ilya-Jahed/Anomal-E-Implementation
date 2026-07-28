import math
import torch
import torch.nn as nn
from .e_graphsage import AnomalESAGEEncoder

class Discriminator(nn.Module):
    """
    Bilinear discriminator for Deep Graph Infomax (DGI).
    Compares edge representations with a global graph summary to score them as Real or Fake.

    This implements the bilinear scoring function from the paper's Eq. (6)/(7):
        D(z_uv, s_bar) = sigma( z_uv^T . w . s_bar )
    with one deliberate difference: this implementation returns the RAW score
    (no sigmoid applied here) -- see the note in `forward` below for why.
    """
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        # The weight matrix for the bilinear scoring function.
        # This is exactly 'w' from the paper's Eq. (6)/(7) -- a single learnable
        # square matrix (n_hidden x n_hidden) shared by both the real-embedding
        # scoring pass and the corrupted-embedding scoring pass.
        #
        # Note: built with the lower-level nn.Parameter (a raw learnable tensor)
        # rather than nn.Linear, because a bilinear form z^T . W . s needs the
        # SAME matrix W sandwiched between two different vectors (z and s) --
        # nn.Linear only implements a simple W @ x + bias, which doesn't fit
        # this two-vector pattern.
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        """
        Fills `tensor` in-place with values drawn uniformly from
        [-bound, bound], where bound = 1/sqrt(size). This is a standard,
        simple weight-initialisation scheme (distinct from the Xavier
        initialisation used in e_graphsage.py) -- appropriate here since this
        is a single square matrix rather than a layer with separate
        fan_in/fan_out sizes.
        """
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        """
        Initializes self.weight using the uniform scheme above, with
        size = n_hidden (the matrix is square, so its size(0) is n_hidden).
        Runs once, at construction time, before any training happens.
        """
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        """
        Args:
            features (Tensor): The embedding vectors (e.g., edge embeddings),
                shape (num_edges, n_hidden). Passed in once for the REAL
                (positive) edge embeddings and once for the CORRUPTED
                (negative) edge embeddings -- see AnomalEDGI.forward.
            summary (Tensor): The global summary vector of the graph,
                shape (n_hidden,). Always built from the REAL embeddings only
                (see AnomalEDGI.forward) -- corrupted embeddings are scored
                AGAINST this same real summary, never used to build their own.
        Returns:
            Tensor: Real/Fake scores for each feature, shape (num_edges,).

        Worked numeric example (n_hidden=2, 1 edge embedding):
            features = [0.8, 0.2]         (one edge embedding, e.g. pos_edge_emb[0])
            weight   = [[0.5, 0.1],
                        [0.2, 0.4]]
            summary  = [0.6, 0.3]

            Step A: weight @ summary
                = [0.5*0.6 + 0.1*0.3, 0.2*0.6 + 0.4*0.3]
                = [0.33, 0.24]
            Step B: features @ (weight @ summary)
                = 0.8*0.33 + 0.2*0.24 = 0.312
            -> raw score = 0.312 (a single scalar per edge; no sigmoid applied here)
        """
        # Bilinear scoring: features^T * W * summary
        # NOTE ON MISSING SIGMOID: the paper's Eq. (6)/(7) wraps this in sigma()
        # to turn it into a 0-1 probability. Here the raw (pre-sigmoid) score is
        # returned instead, because AnomalEDGI uses nn.BCEWithLogitsLoss, which
        # expects raw logits and applies sigmoid internally in a numerically more
        # stable, combined way than computing sigmoid and then plain BCE
        # separately. So the sigmoid from the paper's formula still happens --
        # just inside the loss function rather than here.
        scores = torch.matmul(features, torch.matmul(self.weight, summary))
        return scores


class AnomalEDGI(nn.Module):
    """
    Deep Graph Infomax module for Anomal-E.
    Trains the E-GraphSAGE encoder in a self-supervised manner using Negative Sampling.

    This class ties together everything discussed earlier as the DGI training
    step (the "for epoch..." loop, Algorithm 2 lines 2-9 of the paper):
      1. Run the encoder on the real graph AND on a corrupted version of it,
         both through the exact same shared weights (self.encoder).
      2. Build a single global summary vector from the real embeddings only.
      3. Score both real and corrupted embeddings against that same summary.
      4. Combine both scores into one binary cross-entropy loss that pushes
         real scores up and corrupted scores down.
    """
    def __init__(self, ndim_in, edims, ndim_out, edge_out_dim=256, activation=nn.ReLU()):
        super(AnomalEDGI, self).__init__()
        
        # 1. Initialize the custom E-GraphSAGE encoder.
        # This is g(G, theta) from Algorithm 2 -- the same encoder class fully
        # broken down in e_graphsage.py. A single instance is used for BOTH the
        # real and corrupted forward passes below (see forward()), which is
        # exactly what forces the model to distinguish real vs. corrupted
        # based on graph CONTENT rather than on having two different networks.
        self.encoder = AnomalESAGEEncoder(
            ndim_in=ndim_in, 
            edims=edims, 
            hidden_dim=ndim_out, 
            edge_hidden_dim=edge_out_dim, 
            activation=activation
        )
        
        # 2. Discriminator specifically for the edge embeddings.
        # n_hidden = edge_out_dim because the discriminator compares EDGE
        # embeddings (dimension edge_out_dim) against the global summary,
        # which is itself built from edge embeddings (see forward() below) --
        # so both vectors going into the bilinear form share this dimension.
        self.discriminator = Discriminator(edge_out_dim)
        
        # 3. Binary Cross Entropy loss for distinguishing real vs corrupted edges.
        # BCEWithLogitsLoss = sigmoid + binary cross-entropy fused into one
        # numerically stable operation. This is exactly what implements the
        # paper's L_DGI loss (Algorithm 2, line 8).
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, n_features, e_features):
        """
        Args:
            g (DGLGraph): Input graph.
            n_features (Tensor): Node features.
            e_features (Tensor): Edge features.
            
        Returns:
            loss (Tensor): The self-supervised DGI objective loss.

        This single forward() call is exactly one iteration of the paper's
        training loop (Algorithm 2, lines 3-8), condensed into 5 steps.
        """
        # Step 1: Generate real embeddings (corrupt=False).
        # We only keep the edge embeddings (pos_edge_emb) -- the node
        # embeddings (the '_') aren't needed again after this point, since
        # both the summary and the discriminator operate on edge embeddings.
        # This is z_uv^K from Algorithm 2, line 3.
        _, pos_edge_emb = self.encoder(g, n_features, e_features, corrupt=False)
        
        # Step 2: Generate fake embeddings from corrupted graph (corrupt=True).
        # Same encoder, same weights as Step 1 -- only the edge features fed
        # in get internally shuffled (see AnomalESAGEEncoder.forward). This is
        # z~_uv^K from Algorithm 2, line 4.
        _, neg_edge_emb = self.encoder(g, n_features, e_features, corrupt=True)

        # Step 3: Create global graph summary based on REAL edge embeddings.
        # We take the mean across all edges (dim=0) and apply sigmoid.
        # This is s_bar from Algorithm 2, line 5 -- note it is built ONLY from
        # pos_edge_emb; neg_edge_emb is never used to build a summary of its
        # own, it is only ever scored against this one real summary in Step 4.
        summary = torch.sigmoid(pos_edge_emb.mean(dim=0))

        # Step 4: Score both real and fake embeddings using the Discriminator.
        # Both calls use the exact same self.discriminator (same weight matrix
        # 'w'), scored against the exact same summary -- this is
        # Algorithm 2, lines 6-7 (Eq. 6 for pos_scores, Eq. 7 for neg_scores).
        pos_scores = self.discriminator(pos_edge_emb, summary)
        neg_scores = self.discriminator(neg_edge_emb, summary)

        # Step 5: Calculate BCE loss.
        # Real targets should be 1 (torch.ones_like), Fake targets should be 0 (torch.zeros_like).
        # l1 pushes pos_scores toward "real" (label 1); l2 pushes neg_scores
        # toward "fake" (label 0). Adding them together gives one combined
        # loss -- this is L_DGI from Algorithm 2, line 8. Calling .backward()
        # on the returned value (elsewhere, in the training loop) computes
        # gradients for BOTH self.encoder and self.discriminator at once,
        # since both were used to produce pos_scores/neg_scores.
        l1 = self.loss(pos_scores, torch.ones_like(pos_scores))
        l2 = self.loss(neg_scores, torch.zeros_like(neg_scores))

        return l1 + l2
