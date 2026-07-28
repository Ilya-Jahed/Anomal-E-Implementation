# Anomal-E: Discriminator & Deep Graph Infomax (DGI) Module
### Complete walkthrough of [`dgi_module.py`](../src/models/dgi_module.py)

This document explains, in full detail with running numeric examples, how the `Discriminator` and `AnomalEDGI` classes implement the paper's **Algorithm 2** (the DGI training step) on top of the `AnomalESAGEEncoder` from `e_graphsage.py`.

---

## 0. Where this file sits in the pipeline

```
[preprocessor.py]      → cleaned train/test dataframes with edge feature 'h'
        │
        ▼
[graph_builder.py]     → DGL graph (constant node features, real edge features)
        │
        ▼
[e_graphsage.py]       → AnomalESAGEEncoder: g(G, θ) — produces node & edge embeddings
        │
        ▼
[dgi_module.py]        ← WE ARE HERE
        │   (trains the encoder above using a self-supervised real-vs-corrupted signal)
        ▼
[anomaly_detectors.py] → PCA / IF / CBLOF / HBOS (future phase)
```

This file implements exactly **Algorithm 2, lines 2–9** of the paper: one training epoch's forward passes, scoring, and loss computation. It does **not** update weights itself — that happens in an external training loop that calls `.backward()` and `optimizer.step()` on the loss this file returns.

---

## 1. The Big Picture: What is DGI and Why?

**DGI (Deep Graph Infomax)** is a **self-supervised** training method — it trains the encoder without needing any attack/benign labels. Instead, it teaches the encoder by asking it to distinguish between:

- A **real graph** — actual flows between actual IPs with their true, correlated feature statistics
- A **corrupted graph** — same topology (same IPs connected the same way), but edge features randomly shuffled among edges

The core insight: a good encoder should produce **different embeddings** for real vs corrupted graphs, because real network flows have coherent statistical patterns between connected IPs, while shuffled features destroy that coherence.

```
Real graph:
    A → B  with features [0.5, 0.3]   (e.g. slow HTTP flow: low bytes, low duration)
    C → B  with features [0.1, 0.9]   (e.g. fast DNS: low bytes, high packet rate)
    Coherent: the features "make sense" for this A→B and C→B connection.

Corrupted graph:
    A → B  with features [0.1, 0.9]   ← DNS stats sitting on the HTTP connection
    C → B  with features [0.5, 0.3]   ← HTTP stats sitting on the DNS connection
    Incoherent: topology unchanged, but features no longer reflect the real connection behaviour.
```

---

## 2. Why Two Classes?

| Class | Role | Paper equivalent |
|---|---|---|
| `Discriminator` | Scores one embedding against the global summary using a bilinear form | $D(z_{uv}, \bar{s})$, Eq. (6)/(7) |
| `AnomalEDGI` | Orchestrates the whole training step: encoder × 2, summary, scoring, loss | Full `for epoch...` loop body, Algorithm 2 lines 3–8 |

---

## 3. Class `Discriminator` — Full Breakdown

### 3.1 Constructor: `__init__(n_hidden)`

```python
self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
self.reset_parameters()
```

#### Why `nn.Parameter` and not `nn.Linear`?

This is the same question as in `e_graphsage.py` — but here the answer is different. Throughout `e_graphsage.py`, `nn.Linear` was used because it implements `output = W @ input + bias`, which fits the node/edge update operations perfectly. Here, the paper's bilinear scoring formula is:

$$D(z_{uv}, \bar{s}) = \sigma\!\left(z_{uv}^T \cdot w \cdot \bar{s}\right)$$

This is a **two-vector sandwich**: matrix `w` sits *between* two different vectors — `z_uv` (an edge embedding) on the left and `s̄` (the global summary) on the right. `nn.Linear` has no way to express this pattern — it only knows how to multiply one matrix against one input vector. So the code drops to `nn.Parameter`, which is just a **raw learnable tensor with no built-in forward logic**. The two `matmul` calls in `forward()` implement the sandwich manually.

`self.weight` shape: `(n_hidden, n_hidden)` — square because it maps an `n_hidden`-dimensional summary vector into a space that can be dot-producted against an `n_hidden`-dimensional embedding, and both sides need to match.

> [!NOTE]
> `nn.Parameter` IS still tracked by PyTorch's autograd and included in `model.parameters()` — it is just a tensor without built-in matrix-multiply logic. The difference from `nn.Linear` is only in what forward operation is used, not in whether it gets trained.

### 3.2 Weight Initialisation: `uniform` and `reset_parameters`

```python
def uniform(self, size, tensor):
    bound = 1.0 / math.sqrt(size)
    tensor.data.uniform_(-bound, bound)

def reset_parameters(self):
    size = self.weight.size(0)
    self.uniform(size, self.weight)
```

Every entry of `self.weight` is drawn uniformly from $[-\frac{1}{\sqrt{n}}, +\frac{1}{\sqrt{n}}]$.

**Worked example:** if `n_hidden = 256`, then $\text{bound} = \frac{1}{\sqrt{256}} = \frac{1}{16} = 0.0625$. All $256 \times 256 = 65{,}536$ entries start in `[-0.0625, +0.0625]`.

#### Why not Xavier (as used in `e_graphsage.py`)?

Xavier specifically corrects for different fan-in/fan-out sizes and for the signal shrinkage caused by a following ReLU activation. Here:
- `self.weight` is square — fan-in equals fan-out, so Xavier's correction is a constant factor
- There is no activation function directly applied to the bilinear output (the raw score goes into `BCEWithLogitsLoss`)

A simpler symmetric uniform scheme — the standard default in the original DGI reference implementation — is perfectly appropriate here.

### 3.3 `forward(features, summary)` — The Bilinear Score

```python
def forward(self, features, summary):
    scores = torch.matmul(features, torch.matmul(self.weight, summary))
    return scores
```

#### Tensor shapes

```
summary               shape: (n_hidden,)
self.weight           shape: (n_hidden, n_hidden)
weight @ summary      shape: (n_hidden,)            ← inner matmul
features              shape: (num_edges, n_hidden)
features @ result     shape: (num_edges,)            ← outer matmul, one score per edge
```

#### Worked numeric example (`n_hidden = 2`, 1 edge)

```
features = [0.8, 0.2]
weight   = [[0.5, 0.1],
            [0.2, 0.4]]
summary  = [0.6, 0.3]
```

**Inner matmul** — `weight @ summary`:

$$\begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 0.6 \\ 0.3 \end{bmatrix} = \begin{bmatrix} 0.5 \times 0.6 + 0.1 \times 0.3 \\ 0.2 \times 0.6 + 0.4 \times 0.3 \end{bmatrix} = \begin{bmatrix} 0.33 \\ 0.24 \end{bmatrix}$$

**Outer matmul** — `features @ [0.33, 0.24]`:

$$0.8 \times 0.33 + 0.2 \times 0.24 = 0.264 + 0.048 = \mathbf{0.312}$$

Raw score for this edge = **0.312** (one scalar). For `num_edges` embeddings, this produces a tensor of shape `(num_edges,)` — one score per edge, all computed in parallel.

### 3.4 Why No `sigmoid()` Here?

The paper's Eq. (6)/(7) explicitly wraps the bilinear score in $\sigma(\ldots)$ to produce a 0–1 probability. This implementation returns the **raw logit** instead.

Reason: `AnomalEDGI` uses `nn.BCEWithLogitsLoss`, which fuses sigmoid + binary cross-entropy into one numerically stable operation. Computing `sigmoid(x)` then `BCELoss` as two separate steps can cause floating-point precision issues when scores are very large or very small (sigmoid saturates near 0 or 1, making the log in BCE unstable). The sigmoid from the paper's formula still happens — it has just moved from `Discriminator.forward` into `self.loss`.

---

## 4. Class `AnomalEDGI` — Full Breakdown

### 4.1 Constructor: `__init__`

```python
self.encoder      = AnomalESAGEEncoder(ndim_in, edims, hidden_dim=ndim_out, ...)
self.discriminator = Discriminator(edge_out_dim)
self.loss          = nn.BCEWithLogitsLoss()
```

Three components assembled:

| Component | What it is | Paper symbol |
|---|---|---|
| `self.encoder` | The E-GraphSAGE encoder from `e_graphsage.py` | $g(G, \theta)$ |
| `self.discriminator` | The bilinear scorer above | $D(\cdot, \cdot)$, weight $w$ (= $\omega$) |
| `self.loss` | Sigmoid + BCE fused | $\mathcal{L}_{\text{DGI}}$ |

#### Why one shared encoder, not two?

This is the most important design decision in DGI. If two separate encoder instances existed (one for real, one for corrupted), the model could trivially learn to tell them apart by memorising *which network produced which output* — without learning anything about real network structure. With one shared encoder and shared weights, the **only** information the discriminator can use to distinguish real from corrupted is whether the embeddings reflect genuine topology-feature correlations.

#### Why `Discriminator(edge_out_dim)` and not `Discriminator(ndim_out)`?

The discriminator compares **edge embeddings** against the global summary, and the global summary is also built from **edge embeddings**. Both vectors entering the bilinear form must have the same dimension — `edge_out_dim`, which is the output dimension of `AnomalESAGELayer`'s `W_edge` transformation.

### 4.2 `forward(g, n_features, e_features)` — One Full DGI Training Step

This method is called once per training iteration and returns a single scalar loss. Each step maps directly to a line in Algorithm 2.

---

#### Step 1 — Real embeddings (Algorithm 2, line 3)

```python
_, pos_edge_emb = self.encoder(g, n_features, e_features, corrupt=False)
```

Runs the encoder on the real graph. Node embeddings (`_`) are discarded — both the summary and the discriminator work at the edge level. `pos_edge_emb` shape: `(num_edges, edge_out_dim)`.

---

#### Step 2 — Corrupted embeddings (Algorithm 2, line 4)

```python
_, neg_edge_emb = self.encoder(g, n_features, e_features, corrupt=True)
```

Same encoder object, same weights as Step 1. The `corrupt=True` flag causes `AnomalESAGEEncoder` to shuffle `e_features` via `efeats[torch.randperm(num_edges)]` before running the aggregation steps. `neg_edge_emb` shape: `(num_edges, edge_out_dim)`.

> [!IMPORTANT]
> The graph `g` is passed to both calls unchanged. The corruption only touches the `e_features` tensor (creating a shuffled copy), never the graph object itself. `AnomalESAGELayer`'s `local_scope()` ensures even the temporary graph-state writes are cleaned up after each call.

---

#### Step 3 — Global summary (Algorithm 2, line 5)

```python
summary = torch.sigmoid(pos_edge_emb.mean(dim=0))
```

**What `mean(dim=0)` does:** averages across all edges for each embedding dimension independently.

```
pos_edge_emb = [[0.9, -0.2],    # edge 0
                 [0.5,  0.1],    # edge 1
                 [0.7,  0.3]]    # edge 2

mean(dim=0) = [(0.9+0.5+0.7)/3,  (-0.2+0.1+0.3)/3]
            = [0.700,              0.067]
```

Then sigmoid is applied element-wise:

```
summary = [sigmoid(0.700), sigmoid(0.067)]
        ≈ [0.668,           0.517]         shape: (edge_out_dim,)
```

This is $\bar{s}$ from the paper — a single vector summarising the entire graph's real edge embedding distribution.

> [!IMPORTANT]
> `summary` is built **exclusively from `pos_edge_emb`** (real embeddings). `neg_edge_emb` never contributes to any summary of its own — it is only ever *scored against* this real summary. This asymmetry is what makes the discriminator's task meaningful: "does this embedding look consistent with the real graph's overall character, or not?"

---

#### Step 4 — Scoring (Algorithm 2, lines 6–7)

```python
pos_scores = self.discriminator(pos_edge_emb, summary)
neg_scores = self.discriminator(neg_edge_emb, summary)
```

Both calls use the **exact same `self.discriminator`** (same `weight` matrix) and the **exact same `summary`** vector. The only thing that differs is which embeddings are being scored.

```
pos_scores shape: (num_edges,)  — one real/fake score per real edge
neg_scores shape: (num_edges,)  — one real/fake score per corrupted edge
```

---

#### Step 5 — Loss (Algorithm 2, line 8)

```python
l1 = self.loss(pos_scores, torch.ones_like(pos_scores))
l2 = self.loss(neg_scores, torch.zeros_like(neg_scores))
return l1 + l2
```

- `torch.ones_like(pos_scores)` → target label **1** ("these should be classified as REAL")
- `torch.zeros_like(neg_scores)` → target label **0** ("these should be classified as FAKE")

`BCEWithLogitsLoss` internally computes: $\text{loss} = -[y \cdot \log\sigma(x) + (1-y) \cdot \log(1-\sigma(x))]$

**Worked numeric example** (continuing from Section 3.3 where raw score = 0.312):

For a positive edge with raw score `0.312`, target `1`:
$$l1 = -\log(\sigma(0.312)) = -\log(0.577) \approx 0.550$$

For a corrupted edge with raw score `-0.4`, target `0`:
$$l2 = -\log(1 - \sigma(-0.4)) = -\log(0.599) \approx 0.512$$

Combined loss: $l1 + l2 \approx 1.062$

Gradients from this single loss flow back through **both** `self.discriminator.weight` and all of `self.encoder`'s layers simultaneously (since both were involved in producing `pos_scores` and `neg_scores`). This is Algorithm 2, line 9: $\theta, \omega \leftarrow \text{Adam}(\mathcal{L}_{\text{DGI}})$ — updating both encoder and discriminator in one backward pass. The `.backward()` and `optimizer.step()` calls happen outside this file, in the training loop.

---

## 5. Complete Algorithm 2 Mapping

| Algorithm 2 | Paper notation | Code |
|---|---|---|
| Line 3 | $z_{uv}^K, z_v^K = g(G, \theta)$ | `_, pos_edge_emb = self.encoder(g, n_features, e_features, corrupt=False)` |
| Line 4 | $\tilde{z}_{uv}^K = g(\tilde{G}, \theta)$ | `_, neg_edge_emb = self.encoder(g, n_features, e_features, corrupt=True)` |
| Line 5 | $\bar{s} = \sigma\!\left(\frac{1}{n}\sum z_v^{(K)}\right)$ | `summary = torch.sigmoid(pos_edge_emb.mean(dim=0))` |
| Line 6 | $D(z_{uv}^K, \bar{s}) = \sigma(z_{uv}^{KT} \cdot w \cdot \bar{s})$ | `pos_scores = self.discriminator(pos_edge_emb, summary)` — sigmoid deferred to loss |
| Line 7 | $D(\tilde{z}_{uv}^K, \bar{s}) = \sigma(\tilde{z}_{uv}^{KT} \cdot w \cdot \bar{s})$ | `neg_scores = self.discriminator(neg_edge_emb, summary)` — sigmoid deferred to loss |
| Line 8 | $\mathcal{L}_\text{DGI} = -\frac{1}{2n}\sum[\mathbb{E}_G \log D + \mathbb{E}_{\tilde{G}}\log(1-D)]$ | `l1 + l2` via `BCEWithLogitsLoss` |
| Line 9 | $\theta, \omega \leftarrow \text{Adam}(\mathcal{L}_\text{DGI})$ | `.backward()` + `optimizer.step()` in external training loop |

---

## 6. Two Deliberate Deviations from the Paper

| Paper says | Code does | Why it is correct |
|---|---|---|
| Discriminator applies $\sigma(\ldots)$ directly (Eq. 6/7) | Returns raw logits; sigmoid happens inside `BCEWithLogitsLoss` | Numerically equivalent but more stable — fused form avoids floating-point precision issues near 0/1 |
| Summary formula written over $z_v$ (node-style notation) | Summary built from `pos_edge_emb` (edge embeddings) | Anomal-E is edge-centric — edge embeddings are what the anomaly detectors ultimately score |

Neither is a bug. Both are intentional engineering choices that preserve the paper's mathematical intent.

---

## 7. Quick Reference: All Tensor Shapes in One Place

```
Input:
  n_features        (N, ndim_in)          node features (all-ones from graph_builder)
  e_features        (E, edims)            edge features (normalized flow stats)

After encoder (corrupt=False):
  pos_edge_emb      (E, edge_out_dim)     real edge embeddings

After encoder (corrupt=True):
  neg_edge_emb      (E, edge_out_dim)     corrupted edge embeddings

Summary:
  summary           (edge_out_dim,)       one vector representing the whole real graph

Discriminator:
  self.weight       (edge_out_dim, edge_out_dim)   the bilinear weight matrix w
  pos_scores        (E,)                  one real/fake logit per real edge
  neg_scores        (E,)                  one real/fake logit per corrupted edge

Loss:
  l1, l2            scalar               BCE for real and corrupted respectively
  return value      scalar               l1 + l2 — the full DGI loss
```

---

## 8. One-Paragraph Mental Model

`Discriminator` answers one narrow question: *"given one edge embedding and the graph's overall summary, how compatible do they look?"* — computed with a single learnable bilinear matrix $w$, shared across every embedding it scores. `AnomalEDGI` sets up the entire self-supervised training signal: it runs the **same** encoder on the real graph and a shuffled copy of it, builds one summary from the real run only, asks the discriminator to score both real and shuffled embeddings against that summary, and combines *"did it correctly call real REAL and fake FAKE?"* into one scalar loss. That loss drives both the encoder's and discriminator's weight updates — no attack/benign labels involved anywhere.
