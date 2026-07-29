# `AnomalETrainer` — Execution Engine of the Anomal-E Pipeline

> This document explains what `trainer.py` does, why it exists, how it fits into
> the rest of the Anomal-E pipeline, and how to use it — tied back to the
> original paper (Caville, Lo, Layeghy & Portmann, *"Anomal-E: A self-supervised
> network intrusion detection system based on graph neural networks"*,
> Knowledge-Based Systems 258, 2022).

---

## 1. Where This Fits in the Pipeline

```
Raw NetFlow CSV
      │
      ▼
preprocessor.py        (AnomalEPreprocessor)      → clean train/test dataframes
      │
      ▼
graph_builder.py        (AnomalEGraphBuilder)      → DGL graphs (train_g, test_g)
      │
      ▼
e_graphsage.py           (AnomalESAGEEncoder)       → g(G, θ): node & edge embeddings
      │
      ▼
dgi_module.py             (AnomalEDGI, Discriminator) → self-supervised loss (Algorithm 2)
      │
      ▼
anomaly_detector.py        (AnomalEDetector)          → PCA / HBOS / CBLOF / IForest
      │
      ▼
trainer.py                  (AnomalETrainer)  <-- THIS FILE
      │
      ▼
main.py                      wires everything together and runs it
```

Every other file in the project implements one *component* (an encoder layer, a
loss function, a detector). `AnomalETrainer` is the only file that **calls all of
them in the right order with the right data** — hence "conductor." It has no
learnable parameters of its own; it only orchestrates.

---

## 2. Why Training and Evaluation Are Two Separate Methods

This split is not just code style — it mirrors a hard boundary the whole paper
depends on:

| | `train()` | `evaluate()` |
|---|---|---|
| Uses attack/benign labels? | **Never** | **Yes — but only to score results, never to fit anything upstream** |
| What it touches | `dgi_model` (encoder + discriminator) | `dgi_model.encoder` (frozen, eval mode) + `detector` |
| Paper reference | Algorithm 2, lines 2-9 | Section 4.4, Tables 3-8 |
| Can be called on... | training graph only | any graph (typically the held-out test graph) |

Keeping these in separate methods makes the label-free guarantee visually
obvious in the code: `train()`'s signature doesn't even accept a label-related
parameter, and `evaluate()` reads labels from exactly one place (the graph
itself — see §4a).

---

## 3. Class Reference

### `AnomalETrainer(dgi_model, detector, optimizer, epochs=50, log_every=5)`

| Parameter | Type | Meaning |
|---|---|---|
| `dgi_model` | `AnomalEDGI` | The self-supervised model: wraps `AnomalESAGEEncoder` + `Discriminator` + `BCEWithLogitsLoss`. Calling `dgi_model(g, n_features, e_features)` runs one full DGI step and returns a scalar loss (see `DGI_MODULE_EXPLANATION.md`). |
| `detector` | `AnomalEDetector` | The classical unsupervised detector (`'pca'`, `'hbos'`, `'cblof'`, or `'iforest'`) used only in `evaluate()`. |
| `optimizer` | `torch.optim.Optimizer` | Must already be constructed over `dgi_model.parameters()` (e.g. `Adam`). The trainer never creates or modifies the optimizer's parameter groups — it only calls `.zero_grad()` / `.step()`. |
| `epochs` | `int` | Number of full passes over the graph during training. Raises `ValueError` if not positive. |
| `log_every` | `int` | Print a progress line every N epochs, plus always at epoch 1. Raises `ValueError` if not positive. |

### `train(g, n_features, e_features) -> List[float]`

Runs the DGI training loop for `self.epochs` iterations.

- **Input:** the training graph and its node/edge feature tensors — typically
  `train_g`, `train_g.ndata['h']`, `train_g.edata['h']`.
- **Each epoch:** `zero_grad() → dgi_model(g, n_features, e_features) → backward() → step()`.
  The forward call internally does two encoder passes (real + corrupted graph),
  builds the summary vector, scores both with the discriminator, and returns
  `l1 + l2` — see `DGI_MODULE_EXPLANATION.md` §4.2 for the full derivation.
- **No labels appear anywhere in this method** — the method's signature has no
  label-related parameter at all.
- **Returns:** the list of per-epoch loss values (also cached in
  `self.loss_history`). Logged at the end: total wall-clock time, final loss,
  and the best (lowest) loss with the epoch it occurred at.

### `evaluate(g, n_features, e_features, label_key='Label') -> Dict[str, Optional[float]]`

Runs the trained encoder once (no gradient), fits the detector, and scores it.

1. **Freeze embeddings:** `dgi_model.eval()` + `torch.no_grad()`, then a single
   encoder call with `corrupt=False` — this is the "clean" pass, i.e. the real
   embeddings the encoder learned to produce, not the corrupted-negative pass
   used only during training.
2. **Read labels from the graph itself, exclusively:** `g.edata[label_key]`
   (default `'Label'`). There is no separately-supplied labels parameter —
   `evaluate()` only ever reads ground truth from the same graph object the
   embeddings were just computed from. See §4a below for exactly why this is
   the only safe source.
3. **Fit the detector:** `detector.fit(edge_embeddings)`. Still fully
   unsupervised — `contamination` is a prior guess, not a label lookup (see
   `ANOMAL-E_DETECTOR_EXPLANATION.md` §1).
4. **Score:** `detector.predict()` for hard 0/1 labels, `detector.get_anomaly_scores()`
   for continuous severity, both compared against the labels from step 2 —
   the *only* place ground truth is read in this entire class.
5. **Metrics:** ROC AUC (from continuous scores), F1 / Precision / Recall
   (from hard predictions). ROC AUC is wrapped in its own `try/except` because
   it requires both classes to be present in the labels — this can
   legitimately fail on a small `sanity_check=True` subset; if so, the other
   three metrics still get computed and printed, and `"auc"` comes back as
   `None` instead of crashing the whole evaluation.

Returns a dict (`{"auc", "f1", "precision", "recall"}`) instead of a bare tuple
so callers can log, compare across detector hyperparameter sweeps, or write to
a results file without depending on positional order.

### `save_checkpoint(path)` / `load_checkpoint(path, map_location=None)`

Persist/restore `dgi_model`'s and `optimizer`'s state dicts plus
`loss_history`. Useful for resuming a long training run or reusing a trained
encoder across multiple detector experiments (Table 2's grid search) without
retraining the GNN each time. Purely additive utilities — no other method
depends on them.

---

## 4a. Why Labels Are Read From the Graph, Not From the Dataframe

### The mechanism (confirmed against `graph_builder.py` and the reference notebook)

`AnomalEGraphBuilder._build_single_graph` does, in order:

```python
nx_g = nx.from_pandas_edgelist(df, source=..., target=..., edge_attr=["h", "Label", "Attack"],
                                create_using=nx.MultiGraph())
nx_g = nx_g.to_directed()
dgl_g = dgl.from_networkx(nx_g, edge_attrs=["h", "Attack", "Label"])
```

`nx.MultiGraph.to_directed()` doesn't add a `V→U` mirror alongside each
`U→V` edge in dataframe order — it rebuilds the edge list by walking node
adjacency (grouped by source node), and because undirected multigraph
adjacency is stored symmetrically, this can even produce more directed
edges than original rows whenever a node pair has flows recorded in *both*
directions (verified empirically: `A→B` and a separate real `B→A` flow
between the same pair combine into 4 directed edges, two of which pair the
wrong direction with the wrong feature/label). A labels array pulled
separately from the dataframe's row order is therefore **not guaranteed to
line up with `dgl_g`'s edge order** — not even approximately, and not
fixable by any positional slicing.

**This is not a bug specific to this project** — it's confirmed to be
exactly how the original Anomal-E reference implementation (the authors'
own notebook) builds its graphs too, using the identical
`MultiGraph → to_directed() → dgl.from_networkx` sequence. Reproducing it
faithfully here is the correct choice for results to be comparable to the
paper.

### The fix: read labels off the graph, not off the dataframe

`edge_attr=["h", "Label", "Attack"]` means `Label` travels alongside `h`
through every conversion step — `from_pandas_edgelist` → `to_directed()`
(which deep-copies edge data onto every directed copy it creates) →
`dgl.from_networkx`. Whatever order `dgl_g`'s edges end up in,
`dgl_g.edata['Label']` and `dgl_g.edata['h']` were built from the *same*
iteration and are therefore always mutually consistent — and, by
extension, consistent with `edge_embeddings`, which the encoder computes
straight from `g.edata['h']`.

`evaluate()` uses this directly:

```python
labels_to_use = g.edata[label_key]        # label_key defaults to "Label"
if isinstance(labels_to_use, torch.Tensor):
    labels_to_use = labels_to_use.detach().cpu().numpy()
```

**This matches the reference notebook exactly** — its evaluation cells
also read `train_g.edata['Label']` / `test_g.edata['Label']` directly
rather than tracking a separately-extracted labels array, for precisely
this reason. No slicing, no assumption about edge ordering — it works
regardless of how NetworkX/DGL order or duplicate edges internally,
because the labels were never separated from the edges in the first
place.

### There is no `true_labels` override parameter — and that's deliberate

An earlier version of this method accepted an optional `true_labels`
parameter, honored only when its length happened to match `g`'s edge
count. That parameter has been **removed entirely**, not just left unused
by default. The reasoning: a length match is a *necessary* but not
*sufficient* condition for a label array to actually be aligned with
`edge_embeddings` — as shown above, `to_directed()`'s reordering means a
same-length array built from the dataframe's original row order can still
be silently misaligned with the graph's actual edge order. Keeping that
parameter around, even as an opt-in override, preserved exactly the
failure mode this design is meant to eliminate: a caller could pass a
plausible-looking but wrongly-ordered array and get back confidently
wrong metrics with no error at all.

Removing the parameter closes that door completely. There is now exactly
**one** way to supply labels to `evaluate()` — `g.edata[label_key]` — and
it is the one structurally guaranteed to be correct. If a genuine need to
evaluate against a different label source ever comes up, the safer path
is a dedicated helper that verifies the array was built from the same
graph object (not just checks its length), rather than reintroducing a
general-purpose override.

If a different ground-truth column is needed — e.g. the multi-class
`'Attack'` column instead of the binary `'Label'` column — `label_key` is
still the parameter to use for that, since both columns are carried
through the same graph-construction pipeline and are equally
order-safe.

### A known characteristic worth being aware of, not a bug to chase

Because `to_directed()` mirrors every edge (and, for node pairs with
flows in both real directions, produces a few mismatched combinations too,
as shown above), evaluation runs over roughly `2×` the original row count,
including edges that don't correspond to an independently-observed flow.
The reference implementation does not filter these out, and neither does
this codebase — deviating from that would produce metrics that are no
longer comparable to the paper's Tables 3-8. This is simply how the
Anomal-E method evaluates, not a defect introduced by this reimplementation.

---

## 4. Design Notes / Things to Watch

- **Label alignment is handled by reading `g.edata['Label']` directly, and
  only that** — see §4a for why a dataframe-order array can't be trusted
  here, and why the earlier `true_labels` override parameter was removed
  rather than kept as an opt-in. This matches the original Anomal-E
  reference notebook's own evaluation cells exactly, and confirms
  `graph_builder.py` itself needs no changes — the `MultiGraph → to_directed()`
  pattern it uses is a faithful reproduction of the reference implementation,
  not a defect. The reference `main.py` no longer extracts labels from
  `test_df` at all; it just calls `trainer.evaluate(test_g, ...)`.
- **`detector.model_name` is read defensively** via `getattr(..., default=type(...).__name__)`
  purely for the log line in step 2 of `evaluate()` — if the detector doesn't
  expose that attribute, evaluation still proceeds normally.
- **No labels leak into `train()`** by construction — the method's signature
  simply has no parameter through which a label could be passed.
- **GPU/device placement is not handled inside the trainer.** Tensors and the
  model must already be on the same device before being passed in; this
  mirrors the rest of the current pipeline (`preprocessor.py` /
  `graph_builder.py` are CPU-only), but is worth revisiting if the full
  (non-`sanity_check`) dataset is trained on GPU.

---

## 5. Usage Example

```python
from src.engine.trainer import AnomalETrainer

trainer = AnomalETrainer(dgi_model, detector, optimizer, epochs=50)

# Training — no labels involved.
trainer.train(train_g, train_g.ndata['h'], train_g.edata['h'])

# Evaluation — labels are read straight from test_g.edata['Label'], no
# separately-extracted array needed or accepted.
metrics = trainer.evaluate(test_g, test_g.ndata['h'], test_g.edata['h'])
print(metrics["f1"])

# Optional: persist the trained encoder + discriminator for later reuse.
trainer.save_checkpoint("checkpoints/dgi_run1.pt")
```

Reproducing a detector hyperparameter sweep (Table 2) without retraining the
encoder each time:

```python
trainer.train(train_g, train_g.ndata['h'], train_g.edata['h'])   # train once

for contamination in [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]:
    detector = AnomalEDetector('iforest', contamination=contamination)
    trainer.detector = detector
    metrics = trainer.evaluate(test_g, test_g.ndata['h'], test_g.edata['h'])
    print(contamination, metrics)
```

Evaluating against the multi-class `Attack` column instead of the binary
`Label` column:

```python
metrics = trainer.evaluate(test_g, test_g.ndata['h'], test_g.edata['h'], label_key='Attack')
```

---

## 6. Quick Reference Summary

| Concept | What it is | Where in this file |
|---|---|---|
| `loss_history` | Per-epoch DGI loss values from the last `train()` call | Populated in `train`, usable for plotting convergence |
| `train()` | Runs Algorithm 2's training step for `epochs` iterations, label-free by signature | Core training loop |
| `evaluate()` | Freezes encoder → detector.fit → predict/score → metrics vs. ground truth | The only place ground-truth labels are read |
| Label sourcing | Reads `g.edata[label_key]` directly — the only accepted source, no override parameter | `evaluate()`, step 2 — see §4a for why |
| `save_checkpoint` / `load_checkpoint` | Persist/restore model + optimizer + loss history | Additive utility, not required for a single run |
| ROC AUC guard | `try/except ValueError` around `roc_auc_score` | Prevents a single-class label split from crashing evaluation |

---

## 7. One-Paragraph Mental Model

`AnomalETrainer` is the thin layer that turns a pile of already-defined
components — an encoder, a discriminator, an optimizer, a classical outlier
detector — into an actual experiment. `train()` repeatedly asks the DGI model
"how well can you currently tell real graphs from shuffled ones?" and lets
gradient descent push that answer higher, never once looking at a label —
its signature doesn't even leave room for one. Only after training is frozen
does `evaluate()` hand the resulting edge embeddings to a classical detector
and, for the first and only time in this class, compare its guesses against
ground truth read directly off the graph — producing the numbers that
actually appear in the paper's result tables.