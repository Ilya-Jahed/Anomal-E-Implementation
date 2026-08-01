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

### `train(g, n_features, e_features, checkpoint_path=None, checkpoint_every=10, start_epoch=0) -> List[float]`

Runs the DGI training loop for `self.epochs` iterations.

- **Input:** the training graph and its node/edge feature tensors — typically
  `train_g`, `train_g.ndata['h']`, `train_g.edata['h']`.
- **Each epoch:** `zero_grad() → dgi_model(g, n_features, e_features) → backward() → step()`.
  The forward call internally does two encoder passes (real + corrupted graph),
  builds the summary vector, scores both with the discriminator, and returns
  `l1 + l2` — see `DGI_MODULE_EXPLANATION.md` §4.2 for the full derivation.
- **No labels appear anywhere in this method** — the method's signature has no
  label-related parameter at all.
- **Mid-training checkpointing (`checkpoint_path`, `checkpoint_every`):** if
  `checkpoint_path` is given, a checkpoint is saved every `checkpoint_every`
  epochs, and unconditionally on the final epoch — so a run that finishes
  cleanly always ends with an up-to-date checkpoint on disk, and an
  interrupted run loses at most `checkpoint_every - 1` epochs of progress.
- **Resuming (`start_epoch`):** typically the return value of a prior
  `load_checkpoint()` call. Lets the epoch counter and log messages continue
  from where a previous run left off, instead of restarting at 1. This
  parameter only affects looping/logging — it does **not** by itself restore
  model weights; `load_checkpoint()` must be called separately, before
  `train()`, to actually restore state. If `start_epoch >= self.epochs`,
  the method logs that there's nothing left to train and returns immediately.
- **Returns:** the list of per-epoch loss values (also cached in
  `self.loss_history`, reset at the start of every `train()` call — so on a
  resumed run, this list only contains the current call's losses, not the
  previous run's). Logged at the end: total wall-clock time, final loss,
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

### `save_checkpoint(path, epoch=None)` / `load_checkpoint(path, map_location=None) -> int`

Persist/restore `dgi_model`'s and `optimizer`'s state dicts plus
`loss_history` and the completed-epoch count. `save_checkpoint`'s `epoch`
parameter defaults to `self.epochs` (i.e. "assume training is fully complete")
when not given — this matters when called manually outside of `train()`'s own
mid-training checkpointing loop. `load_checkpoint` returns the completed-epoch
count read from the checkpoint (`0` if that field is missing, for backward
compatibility with checkpoints saved before mid-training checkpointing was
added), meant to be passed straight into `train()`'s `start_epoch` argument.
Useful for resuming a long training run or reusing a trained encoder across
multiple detector experiments (Table 2's grid search) without retraining the
GNN each time. Purely additive utilities — no other method depends on them.

---

## 4a. Why Labels Are Read From the Graph, Not From the Dataframe

### The mechanism (confirmed against `graph_builder.py`, no NetworkX involved)

`AnomalEGraphBuilder._build_single_graph` does **not** use NetworkX at all —
an earlier version did (`nx.from_pandas_edgelist` → `.to_directed()` →
`dgl.from_networkx`), but that path was replaced because NetworkX represents
every edge as its own Python dictionary object, which was too slow and
memory-hungry once memory-safe chunked CSV loading made larger `fraction`
values practical. The current implementation builds the same graph structure
directly with `pandas.factorize` and vectorised NumPy/PyTorch operations:

```python
all_ips = pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]], ignore_index=True)
node_ids, unique_ips = pd.factorize(all_ips)
src_ids = node_ids[:num_rows]
dst_ids = node_ids[num_rows:]

# Build directed edges in BOTH directions for every flow:
src_all = np.concatenate([src_ids, dst_ids])
dst_all = np.concatenate([dst_ids, src_ids])

h_values = np.stack(df["h"].values).astype(np.float32)
h_all = np.concatenate([h_values, h_values], axis=0)   # duplicated the same way as src_all/dst_all

label_values = df["Label"].to_numpy()
label_all = np.concatenate([label_values, label_values], axis=0)  # same duplication pattern
```

`src_all`/`dst_all` are built as `[forward edges, reverse edges]` — every
original flow `u→v` produces both `u→v` (real) and `v→u` (manufactured,
carrying identical `h`/`Label`/`Attack` values). `h_all` and `label_all` are
concatenated in that exact same `[forward, reverse]` order, so entry `i` and
entry `i + num_rows` of every array always correspond to the same pair of
directed edges. This is a deliberate design choice rather than an
NetworkX-adjacency side effect, but it produces the same practical outcome
described below: if a node pair has *real*, independently-observed flows in
*both* directions (e.g. a real `A→B` attack flow and a separate real `B→A`
benign flow), the resulting graph ends up with two edges in the `A→B`
direction carrying contradictory labels — one from the real `A→B` flow, one
manufactured as a copy of the real `B→A` flow's features. `dgl.graph(...)`
supports parallel/multi-edges by default, so both are kept rather than one
overwriting the other; nothing in this construction path filters or resolves
that contradiction.

A labels array pulled separately from the dataframe's original row order
is therefore **not guaranteed to line up with `dgl_g`'s edge order** — the
graph has up to `2×` the row count, in a `[forward, reverse]` order that a
plain dataframe-order array does not follow, and no positional slicing of
the dataframe recovers that order.

**This duplication pattern is not a bug specific to this project** — it's
confirmed to reproduce exactly what the original Anomal-E reference
implementation (the authors' own notebook, via `MultiGraph → to_directed()`)
produces. Reproducing it faithfully here is the correct choice for results to
be comparable to the paper, even after the NetworkX-based construction path
itself was replaced for performance reasons.

### The fix: read labels off the graph, not off the dataframe

Because `h_all`, `label_all`, and `attack_all` are all built from the exact
same `[forward, reverse]` concatenation as `src_all`/`dst_all` (Step 3-4 of
`_build_single_graph`, see `DATA_PIPELINE_EXPLANATION.md` Part B), whatever
order `dgl_g`'s edges end up in, `dgl_g.edata['Label']` and `dgl_g.edata['h']`
were built from the *same* construction pass and are therefore always
mutually consistent — and, by extension, consistent with `edge_embeddings`,
which the encoder computes straight from `g.edata['h']`.

`evaluate()` uses this directly:

```python
labels_to_use = g.edata[label_key]        # label_key defaults to "Label"
if isinstance(labels_to_use, torch.Tensor):
    labels_to_use = labels_to_use.detach().cpu().numpy()
```

**This matches the reference notebook's evaluation behaviour** — its
evaluation cells also read `train_g.edata['Label']` / `test_g.edata['Label']`
directly rather than tracking a separately-extracted labels array, for
precisely this reason. No slicing, no assumption about edge ordering — it
works regardless of how the graph-construction step orders or duplicates
edges internally, because the labels were never separated from the edges in
the first place.

### There is no `true_labels` override parameter — and that's deliberate

An earlier version of this method accepted an optional `true_labels`
parameter, honored only when its length happened to match `g`'s edge
count. That parameter has been **removed entirely**, not just left unused
by default. The reasoning: a length match is a *necessary* but not
*sufficient* condition for a label array to actually be aligned with
`edge_embeddings` — as shown above, the forward+reverse edge duplication
means a same-length array built from the dataframe's original row order can
still be silently misaligned with the graph's actual edge order. Keeping
that parameter around, even as an opt-in override, preserved exactly the
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

Because every original flow is duplicated into a forward and a manufactured
reverse edge (and, for node pairs with flows in both real directions,
produces a couple of label-contradicting edges too, as shown above),
evaluation runs over roughly `2×` the original row count, including edges
that don't correspond to an independently-observed flow. The reference
implementation does not filter these out, and neither does this codebase —
deviating from that would produce metrics that are no longer comparable to
the paper's Tables 3-8. This is simply how the Anomal-E method evaluates,
not a defect introduced by this reimplementation. (For a deeper look at
whether this duplication pattern helps or hurts reported results, see the
discussion in `DATA_PIPELINE_EXPLANATION.md`.)

---

## 4. Design Notes / Things to Watch

- **Label alignment is handled by reading `g.edata['Label']` directly, and
  only that** — see §4a for why a dataframe-order array can't be trusted
  here, and why the earlier `true_labels` override parameter was removed
  rather than kept as an opt-in. This matches the original Anomal-E
  reference notebook's own evaluation cells exactly. The current
  `graph_builder.py` (no NetworkX, `pandas.factorize`-based) reproduces the
  same forward+reverse edge duplication pattern as the reference
  implementation, just via a faster construction path — the label-ordering
  guarantee holds either way, because both paths keep `h`/`Label`/`Attack`
  concatenated in lockstep. The reference `main.py` no longer extracts
  labels from `test_df` at all; it just calls `trainer.evaluate(test_g, ...)`.
- **`detector.model_name` is read defensively** via `getattr(..., default=type(...).__name__)`
  purely for the log line in step 2 of `evaluate()` — if the detector doesn't
  expose that attribute, evaluation still proceeds normally.
- **No labels leak into `train()`** by construction — the method's signature
  simply has no parameter through which a label could be passed.
- **GPU/device placement is not handled inside the trainer.** Tensors and the
  model must already be on the same device before being passed in — `main.py`
  now moves `train_g`/`test_g` (and therefore their `ndata`/`edata`) to
  `device` before calling into the trainer, so this is handled by the caller,
  not by `AnomalETrainer` itself.

---

## 5. Usage Example

```python
from src.engine.trainer import AnomalETrainer

trainer = AnomalETrainer(dgi_model, detector, optimizer, epochs=50)

# Training — no labels involved. Optionally checkpoint every 10 epochs.
trainer.train(
    train_g, train_g.ndata['h'], train_g.edata['h'],
    checkpoint_path="checkpoints/anomal_e_dgi.pt", checkpoint_every=10,
)

# Evaluation — labels are read straight from test_g.edata['Label'], no
# separately-extracted array needed or accepted.
metrics = trainer.evaluate(test_g, test_g.ndata['h'], test_g.edata['h'])
print(metrics["f1"])

# Optional: persist the trained encoder + discriminator for later reuse.
trainer.save_checkpoint("checkpoints/dgi_run1.pt")
```

Resuming a previously-checkpointed run:

```python
start_epoch = trainer.load_checkpoint("checkpoints/anomal_e_dgi.pt", map_location=device)
trainer.train(train_g, train_g.ndata['h'], train_g.edata['h'], start_epoch=start_epoch)
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
| `train()` | Runs Algorithm 2's training step for `epochs` iterations, label-free by signature, with optional mid-training checkpointing and resume support | Core training loop |
| `evaluate()` | Freezes encoder → detector.fit → predict/score → metrics vs. ground truth | The only place ground-truth labels are read |
| Label sourcing | Reads `g.edata[label_key]` directly — the only accepted source, no override parameter | `evaluate()`, step 2 — see §4a for why |
| `save_checkpoint` / `load_checkpoint` | Persist/restore model + optimizer + loss history + completed-epoch count | Additive utility; `load_checkpoint`'s return value feeds `train()`'s `start_epoch` |
| ROC AUC guard | `try/except ValueError` around `roc_auc_score` | Prevents a single-class label split from crashing evaluation |

---

## 7. One-Paragraph Mental Model

`AnomalETrainer` is the thin layer that turns a pile of already-defined
components — an encoder, a discriminator, an optimizer, a classical outlier
detector — into an actual experiment. `train()` repeatedly asks the DGI model
"how well can you currently tell real graphs from shuffled ones?" and lets
gradient descent push that answer higher, never once looking at a label —
its signature doesn't even leave room for one — while optionally saving its
progress to disk along the way so a long run can be resumed if interrupted.
Only after training is frozen does `evaluate()` hand the resulting edge
embeddings to a classical detector and, for the first and only time in this
class, compare its guesses against ground truth read directly off the graph
— producing the numbers that actually appear in the paper's result tables.