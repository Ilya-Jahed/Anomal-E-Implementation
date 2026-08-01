# `plot_comparison.py` — Raw Features vs. Embeddings Comparison & Loss Plots

> This document explains every part of `plot_comparison.py`, the script that
> reproduces the paper's Fig. 5-8-style comparison charts (Raw Features vs.
> Anomal-E Embeddings, across all four anomaly detectors, at two
> contamination levels) plus a training-loss curve. Reading this top to
> bottom should let you reconstruct exactly what the script does and why,
> without needing to re-read `main.py` or the model modules separately.

---

## 1. What This Script Does, In One Picture

```
Saved checkpoint (from a previous `python main.py` run)
        │
        ▼  Rebuild the SAME preprocessed data + graphs main.py used
train_g, test_g
        │
        ▼  Load trained AnomalEDGI encoder from checkpoint (NO retraining)
        ▼  Extract train/test edge embeddings (encoder in eval mode)
train_embeddings, test_embeddings   +   train_raw = train_g.edata['h'], test_raw = test_g.edata['h']
        │
        ▼  For BOTH raw features and embeddings, for BOTH contamination scenarios,
        ▼  fit + evaluate all 4 classical detectors (PCA / IF / CBLOF / HBOS)
raw_results, embed_results  (Macro F1-Score per algorithm per scenario)
        │
        ▼  plot_loss_curve()          -> plots/training_loss.png
        ▼  plot_comparison() x 2      -> plots/comparison_0pct_contamination.png
                                       -> plots/comparison_natural_contamination.png
```

This script is deliberately **separate from `main.py`** and does **not**
retrain the GNN encoder. It loads the already-trained model from a
checkpoint, so re-running it to tweak a plot (colors, contamination
threshold, which algorithms to include) takes seconds, not the full DGI
training time.

---

## 2. Imports

```python
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from src.data_pipeline.preprocessor import AnomalEPreprocessor
from src.data_pipeline.graph_builder import AnomalEGraphBuilder
from src.models.dgi_module import AnomalEDGI
from src.models.anomaly_detectors import AnomalEDetector
from src.engine.trainer import AnomalETrainer
```

| Import | Role |
|---|---|
| `os` | Checkpoint/plots directory handling, reading the checkpoint-dir environment variable. |
| `numpy` | Building the x-axis positions for bar groups, computing natural contamination, masking benign rows. |
| `torch` | Device selection, `torch.no_grad()` for embedding extraction, boolean-mask indexing. |
| `matplotlib.pyplot` | All plotting (bar charts, loss curve). |
| `sklearn.metrics.f1_score` | Computing Macro F1-Score, the same metric the paper leads with in Tables 3-6. |
| `AnomalEPreprocessor`, `AnomalEGraphBuilder` | Rebuild the exact same `train_g`/`test_g` graphs `main.py` used (see Section 5 for why this is necessary and what to keep in sync). |
| `AnomalEDGI` | The encoder class, re-instantiated here so its trained weights can be loaded from the checkpoint. |
| `AnomalEDetector` | The same PyOD wrapper used in `main.py`, instantiated fresh here for each of the four algorithms. |
| `AnomalETrainer` | Reused only for its `load_checkpoint()` method (see Section 6) -- not for training or evaluating in this script. |

---

## 3. Module-Level Constants

```python
ALGORITHMS = ["pca", "iforest", "cblof", "hbos"]
ALGORITHM_DISPLAY_NAMES = {"pca": "PCA", "iforest": "IF", "cblof": "CBLOF", "hbos": "HBOS"}
```

- `ALGORITHMS`: the internal `model_name` strings `AnomalEDetector` expects,
  iterated over everywhere four-algorithm comparisons are run.
- `ALGORITHM_DISPLAY_NAMES`: maps those internal strings to the exact
  abbreviations the paper itself uses on its chart x-axes (PCA, IF, CBLOF,
  HBOS), so the plots visually match Fig. 5-8.

---

## 4. Helper Function: `fit_and_score()`

```python
def fit_and_score(model_name, contamination, fit_features, eval_features, eval_labels):
    detector = AnomalEDetector(model_name=model_name, contamination=contamination)
    detector.fit(fit_features)
    predictions = detector.predict(eval_features)
    return f1_score(eval_labels, predictions, average="macro")
```

The smallest unit of work in this script: build **one** detector, fit it on
**one** feature set (unsupervised -- `fit_features` never includes labels),
predict on a separate evaluation set, and return a single Macro F1 number.
Called 16 times in total per feature space (4 algorithms × 2 contamination
scenarios), by `run_all_algorithms()` below.

`average="macro"` matches the paper's own metric choice (Macro F1, described
in Section 5.2 of the paper as combating class imbalance by treating both
classes' F1 equally rather than weighting by class frequency).

---

## 5. Helper Function: `run_all_algorithms()`

```python
def run_all_algorithms(fit_features_0pct, fit_features_natural, eval_features, eval_labels,
                        contamination_natural):
    results = {"0pct": {}, "natural": {}}
    for algo in ALGORITHMS:
        results["0pct"][algo] = fit_and_score(
            algo, contamination=0.001, fit_features=fit_features_0pct,
            eval_features=eval_features, eval_labels=eval_labels,
        )
        results["natural"][algo] = fit_and_score(
            algo, contamination=contamination_natural, fit_features=fit_features_natural,
            eval_features=eval_features, eval_labels=eval_labels,
        )
    return results
```

Loops over all four algorithms and, for each one, calls `fit_and_score()`
twice -- once per contamination scenario:

- **`"0pct"`**: fits on `fit_features_0pct` (the caller passes in ONLY
  benign-row features here -- see Section 9) with a near-zero
  `contamination=0.001` (PyOD requires a small positive value, not exactly
  0, to define a decision threshold).
- **`"natural"`**: fits on `fit_features_natural` (the full training set,
  whatever real attack ratio survived preprocessing) with
  `contamination=contamination_natural` (computed once in `main()`, see
  Section 9).

Both scenarios are always evaluated against the **same** `eval_features`/
`eval_labels` (the test set) -- only what the detector was fit on changes
between the two scenarios, mirroring the paper's own protocol of varying
training-time contamination while keeping the test set fixed.

This function is called **twice** in `main()`: once with raw features passed
in for `fit_features_*`/`eval_features`, and once with GNN embeddings passed
in instead -- the function itself has no idea which feature space it's
working with, which is what keeps the raw-vs-embeddings comparison
symmetric and bug-resistant (both code paths run through the exact same
sixteen `fit_and_score()` calls).

---

## 6. Plotting Function: `plot_comparison()`

```python
def plot_comparison(raw_results, embed_results, scenario_key, scenario_title, output_path):
    labels = [ALGORITHM_DISPLAY_NAMES[a] for a in ALGORITHMS]
    raw_scores = [raw_results[scenario_key][a] * 100 for a in ALGORITHMS]
    embed_scores = [embed_results[scenario_key][a] * 100 for a in ALGORITHMS]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_raw = ax.bar(x - width / 2, raw_scores, width, label="Raw Features", color="#ED7D31")
    bars_embed = ax.bar(x + width / 2, embed_scores, width, label="Embeddings", color="#4472C4")
    ...
```

Builds exactly one bar chart, matching the paper's Fig. 5-8 layout:

- **x-axis**: one group per algorithm (`PCA`, `IF`, `CBLOF`, `HBOS`), in the
  same left-to-right order the paper uses.
- **Two bars per group**: "Raw Features" (orange, `#ED7D31`) and
  "Embeddings" (blue, `#4472C4`) -- these specific colors were chosen to
  visually echo the paper's own Fig. 5-8 color scheme.
- **`x - width/2` / `x + width/2`**: standard matplotlib grouped-bar-chart
  positioning -- offsets each pair of bars symmetrically around its integer
  tick position so they sit side by side without overlapping.
- **y-axis fixed to `(0, 100)`**: Macro F1-Score as a percentage, consistent
  scale across all generated plots so multiple runs/plots are visually
  comparable.
- **`ax.annotate(...)` loop**: writes the exact numeric value above each bar
  (e.g. `"88.45"`), matching the paper's own figures which label every bar
  with its precise score rather than leaving the reader to estimate from
  the y-axis alone.
- **`fig.savefig(output_path, dpi=150)`**: saves a PNG file at a resolution
  suitable for viewing/embedding in a report, before...
- **`plt.show()`**: ...also displaying it inline. In a Colab notebook, this
  renders the figure directly in the output cell -- both saving to disk and
  displaying inline happen from this single call, so you always get a
  file in `plots/` even in an environment where inline display isn't
  visible (e.g. a plain terminal).

Called **twice** in `main()` -- once per contamination scenario (`"0pct"`
and `"natural"`), producing two separate PNG files.

---

## 7. Plotting Function: `plot_loss_curve()`

```python
def plot_loss_curve(loss_history, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color="#4472C4", linewidth=1.5)
    ...
```

A simple line plot: x-axis is epoch number (1-indexed, matching the `Epoch
NNN/50` numbering already printed by `trainer.py` during training), y-axis
is the DGI loss value recorded that epoch. `ax.grid(alpha=0.3)` adds a faint
grid, useful for reading off approximate loss values without cluttering the
line itself.

`loss_history` here comes from the checkpoint (see Section 8) -- it is
**not** freshly computed, so this plot reflects whatever training run
produced the checkpoint being loaded, not necessarily every epoch the model
has ever seen if you've retrained from scratch multiple times without
clearing old checkpoints.

---

## 8. `main()` — Step by Step

### 8.1. Paths and early exit

```python
dataset_path = "data/raw/NF-CSE-CIC-IDS2018-v2.csv"
checkpoint_dir = os.environ.get("ANOMAL_E_CHECKPOINT_DIR", "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "anomal_e_dgi.pt")
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

if not os.path.exists(checkpoint_path):
    print(f"[ERROR] No checkpoint found at '{checkpoint_path}'. "
          f"Run `python main.py` first to train and save a checkpoint.")
    return
```

- Reuses the **same** `ANOMAL_E_CHECKPOINT_DIR` environment-variable
  convention as `main.py`, so if you trained with checkpoints pointed at
  Google Drive, this script finds them at the same location automatically
  -- no separate configuration needed.
- Fails fast with a clear, actionable message if no checkpoint exists yet,
  rather than crashing later with a less obvious `FileNotFoundError` deep
  inside `torch.load`.
- `plots/` is created (if missing) up front, before any potentially slow
  work (preprocessing, embedding extraction) happens.

### 8.2. Device setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Identical logic to `main.py` -- automatically uses GPU if available.

### 8.3. Rebuilding the data and graphs

```python
preprocessor = AnomalEPreprocessor()
train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=False, fraction=0.05)

graph_builder = AnomalEGraphBuilder()
train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
train_g = train_g.to(device)
test_g = test_g.to(device)
```

**⚠️ This is the single most important thing to keep in sync with
`main.py`.** The checkpoint being loaded contains weights for a specific
`AnomalEDGI` instance, whose input dimensions (`ndim_in`, `edims`) were
determined by whatever graph `main.py` built when it trained. If this
script's `fraction`/`sanity_check` arguments differ from what `main.py`
used, the rebuilt graph will have different node/edge counts and
potentially different feature dimensions, and loading the checkpoint's
`state_dict` into a differently-shaped encoder can either error out (shape
mismatch) or -- more dangerously -- silently load into a graph it was never
actually trained on, producing meaningless embeddings and misleading plots.

**Practical rule**: whatever `fraction` value `main.py` used for the run
that produced the checkpoint you're about to load, this script's call must
use the exact same value.

Everything else about the graph-building and device-placement code here is
identical in behaviour to the corresponding lines in `main.py` (see
`MAIN_EXPLANATION.md`, Sections 5 and 7, for the full breakdown of what
`process_pipeline()` and `.to(device)` do).

### 8.4. Loading the trained encoder from checkpoint

```python
ndim_in = train_g.ndata['h'].shape[2]
edims = train_g.edata['h'].shape[2]
hidden_dim = 128
edge_hidden_dim = 256

dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim).to(device)
optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)  # required by load_checkpoint's API
trainer = AnomalETrainer(dgi_model, detector=None, optimizer=optimizer, epochs=50)

print(f"[INFO] Loading checkpoint from '{checkpoint_path}'...")
completed_epochs = trainer.load_checkpoint(checkpoint_path, map_location=device)
print(f"[INFO] Loaded encoder trained for {completed_epochs} epochs.")
```

- `hidden_dim`/`edge_hidden_dim` are hardcoded to the same values `main.py`
  uses (128/256) -- these, too, must match whatever was used to produce the
  checkpoint, for the same shape-mismatch reasons as `fraction` above.
- A fresh `AnomalEDGI` is constructed with the correct shape, then
  immediately overwritten with the checkpoint's saved weights via
  `load_checkpoint()` -- the freshly-initialised (random) weights are
  discarded entirely; only the shape/architecture of this fresh instance
  matters, not its initial values.
- An `optimizer` is built here purely because `AnomalETrainer.load_checkpoint()`
  expects one to restore its state dict into (`AnomalETrainer` was designed
  around always having a paired optimizer). It is never actually stepped in
  this script -- no training happens here, so the optimizer's restored
  state is simply unused dead weight in this context, kept only to satisfy
  the trainer class's constructor signature.
- `detector=None` is passed to `AnomalETrainer(...)` because this script
  never calls `trainer.evaluate()` -- it builds its own `AnomalEDetector`
  instances directly (see Section 5), one per algorithm, rather than using
  the single detector `AnomalETrainer` was originally designed around.
- `completed_epochs` (returned by `load_checkpoint`) is only used here for
  an informational print statement -- unlike in `main.py`, it is not fed
  into any further training call, since this script never trains.

### 8.5. Plotting the loss curve

```python
if trainer.loss_history:
    plot_loss_curve(trainer.loss_history, os.path.join(plots_dir, "training_loss.png"))
else:
    print("[WARNING] No loss_history found in checkpoint -- skipping loss plot.")
```

`load_checkpoint()` restores `trainer.loss_history` from the checkpoint file
(see `trainer.py`'s own checkpoint format). If, for some reason, an older
checkpoint format without a stored `loss_history` is loaded, this guards
against calling `plot_loss_curve()` on an empty list (which would otherwise
produce a blank/misleading plot) and instead prints a clear warning.

### 8.6. Extracting embeddings (no training)

```python
dgi_model.eval()
with torch.no_grad():
    _, train_embeddings = dgi_model.encoder(
        train_g, train_g.ndata['h'], train_g.edata['h'], corrupt=False
    )
    _, test_embeddings = dgi_model.encoder(
        test_g, test_g.ndata['h'], test_g.edata['h'], corrupt=False
    )
```

- `dgi_model.eval()`: switches the model to evaluation mode (relevant if any
  layers behave differently during training vs. inference, e.g. dropout --
  the current encoder has none, but this is good practice regardless).
- `torch.no_grad()`: disables gradient tracking, since nothing here will be
  backpropagated through -- saves memory and compute.
- `corrupt=False`: runs the encoder on the **real** (uncorrupted) graph,
  producing genuine edge embeddings -- the same call `trainer.evaluate()`
  makes internally in `main.py`, just done here for both the train graph
  AND the test graph (whereas `main.py`'s `evaluate()` only ever needed the
  test graph, since the training graph's raw features/labels weren't
  previously extracted for a raw-vs-embeddings comparison).
- The first return value (node embeddings, `_`) is discarded -- only edge
  embeddings are needed for anomaly detection, consistent with every other
  use of this encoder in the project.

### 8.7. Extracting raw features

```python
train_raw = train_g.edata['h'].squeeze(1)
test_raw = test_g.edata['h'].squeeze(1)
```

`edata['h']` has shape `(E, 1, edims)` (that middle dimension of size 1 was
added deliberately in `graph_builder.py` to match what the E-GraphSAGE
layers expect as input). `.squeeze(1)` removes that middle dimension,
giving shape `(E, edims)` -- matching the shape convention `train_embeddings`
already has (since `AnomalESAGEEncoder.forward()` internally does
`.sum(1)` on its outputs, collapsing that same dimension). Keeping both
feature spaces in the same `(E, feat_dim)` shape convention means
`AnomalEDetector` can be handed either one without any special-casing.

### 8.8. Labels and natural contamination

```python
train_labels = train_g.edata['Label'].detach().cpu().numpy()
test_labels = test_g.edata['Label'].detach().cpu().numpy()

natural_contamination = float(np.clip(train_labels.mean(), 0.001, 0.5))
print(f"[INFO] Natural attack contamination in training data: {natural_contamination * 100:.2f}%")
```

- Labels are read directly from `edata['Label']`, for the exact same reason
  explained in `trainer.py`'s `evaluate()` docstring and `MAIN_EXPLANATION.md`
  Section 9: this is the only array structurally guaranteed to be in the
  same edge order as the features/embeddings, regardless of how graph
  construction reordered or duplicated edges relative to the source
  dataframe.
- `train_labels.mean()`: since `Label` is 0/1, the mean is exactly the
  fraction of attack edges in the training graph -- this is the "natural"
  contamination level (whatever real attack ratio survived downsampling),
  as opposed to the paper's separate experiment where contamination was
  deliberately controlled to exactly 4%.
- `np.clip(..., 0.001, 0.5)`: PyOD's `contamination` parameter must be
  strictly greater than 0 and at most 0.5 -- this guards against a
  pathological edge case (e.g. an extremely small/unlucky downsample
  producing literally zero attack edges in `train_g`) that would otherwise
  make `AnomalEDetector.__init__` raise an error from an invalid
  `contamination=0.0`.

### 8.9. Building the "0% contamination" fit sets

```python
benign_mask_train = (train_labels == 0)
train_raw_benign = train_raw[torch.from_numpy(benign_mask_train)]
train_embeddings_benign = train_embeddings[torch.from_numpy(benign_mask_train)]
```

For the `"0pct"` scenario (matching the paper's 0%-contamination
experiments in Tables 3/5), the detector should be fit **only** on rows
that are actually benign -- this boolean-mask indexing filters both
`train_raw` and `train_embeddings` down to just the `Label == 0` rows,
which are then passed as `fit_features_0pct` into `run_all_algorithms()`
(Section 9 below). The full (unfiltered) `train_raw`/`train_embeddings`
are separately passed as `fit_features_natural`, for the other scenario.

### 8.10. Running all algorithms on both feature spaces

```python
raw_results = run_all_algorithms(
    fit_features_0pct=train_raw_benign,
    fit_features_natural=train_raw,
    eval_features=test_raw,
    eval_labels=test_labels,
    contamination_natural=natural_contamination,
)

embed_results = run_all_algorithms(
    fit_features_0pct=train_embeddings_benign,
    fit_features_natural=train_embeddings,
    eval_features=test_embeddings,
    eval_labels=test_labels,
    contamination_natural=natural_contamination,
)
```

Two calls to the same function (Section 5), one for each feature space.
Both use the same `test_raw`/`test_embeddings`... wait -- note `eval_features`
differs between the two calls (`test_raw` vs. `test_embeddings`), matching
whichever feature space is being evaluated; `eval_labels` (`test_labels`)
and `contamination_natural` are shared, since those don't depend on which
feature space is in use.

### 8.11. Console results table

```python
for scenario_key, scenario_label in [("0pct", "0% contamination"), ("natural", "Natural contamination")]:
    print(f"\n-- {scenario_label} --")
    print(f"{'Algorithm':<10} {'Raw Features':>15} {'Embeddings':>15}")
    for algo in ALGORITHMS:
        raw_f1 = raw_results[scenario_key][algo] * 100
        embed_f1 = embed_results[scenario_key][algo] * 100
        print(f"{ALGORITHM_DISPLAY_NAMES[algo]:<10} {raw_f1:>14.2f}% {embed_f1:>14.2f}%")
```

A plain-text table printed to the console, mirroring the structure of the
paper's Tables 3/5 (though limited to Macro F1, since that's the only
metric this script computes -- Accuracy/DR are not currently included, see
Section 10 below). Useful for quickly reading off exact numbers without
having to zoom into a saved image.

### 8.12. Generating the plots

```python
plot_comparison(raw_results, embed_results, "0pct", "(0% contamination)",
                 os.path.join(plots_dir, "comparison_0pct_contamination.png"))
plot_comparison(raw_results, embed_results, "natural",
                 f"(natural contamination, ~{natural_contamination * 100:.1f}%)",
                 os.path.join(plots_dir, "comparison_natural_contamination.png"))
```

Two calls to `plot_comparison()` (Section 6), one per scenario, each saving
a distinct PNG file under `plots/`. The "natural" scenario's title
dynamically includes the actual measured contamination percentage (e.g.
`"~3.8%"`), rather than a hardcoded `"4%"`, since this project's natural
attack ratio is whatever survived preprocessing rather than a value chosen
to match the paper's fixed 4% experiment.

---

## 9. Output Files Produced

Running this script produces (all under `plots/`, created if missing):

| File | Contents |
|---|---|
| `plots/training_loss.png` | DGI loss vs. epoch, from the loaded checkpoint's `loss_history`. |
| `plots/comparison_0pct_contamination.png` | Bar chart: Macro F1 for PCA/IF/CBLOF/HBOS, Raw Features vs. Embeddings, detector fit on benign-only training data. |
| `plots/comparison_natural_contamination.png` | Same bar-chart layout, but the detector is fit on the full training set (whatever real attack ratio survived downsampling). |

All three are also displayed inline (via `plt.show()`) when run in an
environment that supports it, such as a Colab notebook cell.

---

## 10. Known Limitations / What This Script Does NOT Do

- **No Accuracy or Detection Rate (DR)**: only Macro F1 is computed and
  plotted, unlike the paper's Tables 3-6 which also report Accuracy and DR
  side by side. Extending `fit_and_score()` to also return
  `accuracy_score`/recall-based DR would be a straightforward addition if
  needed later.
- **No GraphSAGE/DGI baseline comparison**: the paper's Tables 7-8 compare
  Anomal-E against plain GraphSAGE and vanilla DGI (using node features
  instead of edge features) as baselines. This script only compares Raw
  Features vs. Anomal-E Embeddings -- it does not train or evaluate those
  alternative baseline encoders.
- **Requires manually keeping `fraction`/`sanity_check`/`hidden_dim`/
  `edge_hidden_dim` in sync with whatever `main.py` run produced the
  checkpoint being loaded** (Section 8.3/8.4) -- there is no automatic
  check that verifies these match; a silent mismatch would not necessarily
  raise an error and could produce misleading plots.
- **Does not retrain if no checkpoint exists** -- by design (see the
  module-level docstring), this script only ever loads an existing
  checkpoint; running `python main.py` first is a hard prerequisite.