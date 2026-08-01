# `plot_comparison.py` — Raw Features vs. Embeddings Comparison & Loss Plots

> This document explains every part of `plot_comparison.py`, the script that
> reproduces the paper's Fig. 5-8-style comparison charts (Raw Features vs.
> Anomal-E Embeddings, across all four anomaly detectors, at two
> contamination-fit scenarios) plus a training-loss curve. Reading this top
> to bottom should let you reconstruct exactly what the script does and why,
> without needing to re-read `main.py` or the model modules separately.

---

## 1. What This Script Does, In One Picture

```
Saved checkpoint (from a previous `python main.py` run, trained for 4000 epochs)
        │
        ▼  Rebuild the SAME preprocessed data + graphs main.py used (fraction=0.1)
train_g, test_g
        │
        ▼  Load trained AnomalEDGI encoder from checkpoint (NO retraining)
        ▼  Extract train/test edge embeddings (encoder in eval mode)
train_embeddings, test_embeddings   +   train_raw = train_g.edata['h'], test_raw = test_g.edata['h']
        │
        ▼  For BOTH raw features and embeddings, for BOTH fit scenarios (0% / natural),
        ▼  GRID-SEARCH each of the 4 classical detectors (PCA / IF / CBLOF / HBOS)
        ▼  over its own hyperparameter grid x a shared contamination grid,
        ▼  keeping the BEST Macro F1 found
raw_results, embed_results  (best Macro F1-Score per algorithm per scenario)
        │
        ▼  plot_loss_curve()          -> plots/training_loss.png
        ▼  plot_comparison() x 2      -> plots/comparison_0pct_contamination.png
                                       -> plots/comparison_natural_contamination.png
```

This script is deliberately **separate from `main.py`** and does **not**
retrain the GNN encoder. It loads the already-trained model from a
checkpoint, so re-running it to tweak a plot (colors, which algorithms to
include, grid ranges) takes minutes, not the full 4000-epoch DGI training
time.

---

## 2. Imports

```python
import itertools
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
| `itertools` | `itertools.product(...)` builds every (hyperparameter, contamination) combination for each algorithm's grid search. |
| `os` | Checkpoint/plots directory handling, reading the checkpoint-dir environment variable. |
| `numpy` | Building the x-axis positions for bar groups, masking benign rows. |
| `torch` | Device selection, `torch.no_grad()` for embedding extraction, boolean-mask indexing. |
| `matplotlib.pyplot` | All plotting (bar charts, loss curve). |
| `sklearn.metrics.f1_score` | Computing Macro F1-Score, the same metric the paper leads with in Tables 3-6. |
| `AnomalEPreprocessor`, `AnomalEGraphBuilder` | Rebuild the exact same `train_g`/`test_g` graphs `main.py` used (see Section 8.3 for why this is necessary and what to keep in sync). |
| `AnomalEDGI` | The encoder class, re-instantiated here so its trained weights can be loaded from the checkpoint. |
| `AnomalEDetector` | The same PyOD wrapper used in `main.py`, instantiated fresh here for every point in the grid search. |
| `AnomalETrainer` | Reused only for its `load_checkpoint()` method (see Section 8.4) -- not for training or evaluating in this script. |

---

## 3. Module-Level Constants

```python
ALGORITHMS = ["pca", "iforest", "cblof", "hbos"]
ALGORITHM_DISPLAY_NAMES = {"pca": "PCA", "iforest": "IF", "cblof": "CBLOF", "hbos": "HBOS"}

ALGORITHM_PARAM_GRIDS = {
    "pca":     {"n_components": [5, 10, 15, 20, 25, 30]},
    "iforest": {"n_estimators": [20, 50, 100, 150]},
    "cblof":   {"n_clusters":   [2, 3, 5, 7, 9, 10]},
    "hbos":    {"n_bins":       [5, 10, 15, 20, 25, 30]},
}
CONTAMINATION_GRID = [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]
```

- `ALGORITHMS` / `ALGORITHM_DISPLAY_NAMES`: as before -- internal `model_name`
  strings vs. the exact abbreviations the paper uses on its chart x-axes.
- **`ALGORITHM_PARAM_GRIDS`**: each algorithm's own hyperparameter search
  space, **copied exactly from the reference Anomal-E notebook's own
  grid-search cells** (its Cell 34 for CBLOF's `n_est = [2,3,5,7,9,10]`,
  Cell 42 for HBOS's `n_est = [5,10,15,20,25,30]`, Cell 46 for PCA's
  `n_est = [5,10,15,20,25,30]` used as `n_components`, and Cell 50 for
  Isolation Forest's `n_est = [20,50,100,150]` used as `n_estimators`). The
  dict key in each entry (e.g. `"n_components"`) matches the exact kwarg
  name `AnomalEDetector.__init__`'s `**kwargs` forwards straight through to
  the underlying PyOD model constructor.
- **`CONTAMINATION_GRID`**: the same `[0.001, 0.01, 0.04, 0.05, 0.1, 0.2]`
  list used in *every one* of the reference notebook's grid-search cells
  (34 through 53) -- shared across all four algorithms, crossed with each
  algorithm's own grid via `itertools.product` (Section 4).

Using these specific grids (rather than each algorithm's PyOD default
hyperparameters) matters: this project's earlier version fit each detector
with library defaults and a single fixed `contamination`, which meant its
Macro F1 numbers were not directly comparable to the paper's Tables 3-6 --
the paper's own numbers come from exactly this kind of per-algorithm grid
search, keeping only the best score found. Matching that search process is
necessary, not just matching `fraction`/`epochs`, for a fair comparison.

---

## 4. Helper Function: `grid_search_best_f1()`

```python
def grid_search_best_f1(model_name, fit_features, eval_features, eval_labels):
    param_grid = ALGORITHM_PARAM_GRIDS[model_name]
    param_name, param_values = next(iter(param_grid.items()))

    best_f1 = -1.0
    best_params = None

    for param_value, contamination in itertools.product(param_values, CONTAMINATION_GRID):
        kwargs = {param_name: param_value}
        detector = AnomalEDetector(model_name=model_name, contamination=contamination, **kwargs)
        detector.fit(fit_features)
        predictions = detector.predict(eval_features)
        f1 = f1_score(eval_labels, predictions, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_params = {param_name: param_value, "contamination": contamination}

    return best_f1, best_params
```

This is the smallest unit of work in the script, replacing the earlier
single-fit `fit_and_score()` helper. For **one** algorithm:

1. Looks up that algorithm's own hyperparameter name/grid from
   `ALGORITHM_PARAM_GRIDS` (e.g. for `"hbos"`, `param_name = "n_bins"`,
   `param_values = [5, 10, 15, 20, 25, 30]`).
2. `itertools.product(param_values, CONTAMINATION_GRID)` yields every
   `(param_value, contamination)` pair -- 6×6=36 combinations for
   PCA/CBLOF/HBOS, 4×6=24 for Isolation Forest.
3. For each combination: build a fresh `AnomalEDetector` with that
   algorithm-specific kwarg **and** that contamination value, `fit()` it on
   `fit_features` (unsupervised throughout -- no labels used in fitting),
   `predict()` on `eval_features`, and score with Macro F1 against
   `eval_labels`.
4. Keeps only the single best `(f1, params)` pair seen across the whole
   grid, matching the reference notebook's own `if new_score > score: ...`
   pattern in each of its grid-search cells.

`average="macro"` matches the paper's own metric choice (Macro F1, combating
class imbalance by treating both classes' F1 equally rather than weighting
by class frequency).

**Cost**: this function alone runs 36 (or 24, for IForest) full fit+predict
cycles. Multiplied across 4 algorithms × 2 fit-scenarios × 2 feature spaces
(raw/embeddings) in `run_all_algorithms()` (Section 5) called twice in
`main()`, the full script performs `(36×3 + 24) × 2 × 2 = 528` total
detector fit+predict cycles -- still fast in absolute terms since none of
these classical detectors involve GPU/gradient computation, but noticeably
more work than the single-fit version this replaced.

---

## 5. Helper Function: `run_all_algorithms()`

```python
def run_all_algorithms(fit_features_0pct, fit_features_natural, eval_features, eval_labels):
    results = {"0pct": {}, "natural": {}}
    best_params_log = {"0pct": {}, "natural": {}}

    for algo in ALGORITHMS:
        f1, params = grid_search_best_f1(algo, fit_features_0pct, eval_features, eval_labels)
        results["0pct"][algo] = f1
        best_params_log["0pct"][algo] = params

        f1, params = grid_search_best_f1(algo, fit_features_natural, eval_features, eval_labels)
        results["natural"][algo] = f1
        best_params_log["natural"][algo] = params

    return results, best_params_log
```

Loops over all four algorithms and, for each one, calls
`grid_search_best_f1()` twice -- once per fit scenario:

- **`"0pct"`**: grid-searches using `fit_features_0pct` (the caller passes in
  ONLY benign-row features here -- see Section 8.9) as the fit set.
- **`"natural"`**: grid-searches using `fit_features_natural` (the full
  training set, whatever real attack ratio survived preprocessing) as the
  fit set.

**What changed from the earlier version**: previously, `"0pct"` used a fixed
`contamination=0.001` and `"natural"` used a single computed
`contamination_natural` value, with no other hyperparameter search at all.
Now, **both** scenarios search the full `CONTAMINATION_GRID` (not a single
fixed value) *in addition to* each algorithm's own hyperparameter -- what
still distinguishes the two scenarios is purely **which rows the detector is
fit on** (benign-only vs. the full training set), not which contamination
values are allowed to be tried. This matches the reference notebook, whose
grid-search cells for the "benign-only" and "normal" (full-training-set) fit
sets both sweep the identical `contamination = [0.001, 0.01, 0.04, 0.05,
0.1, 0.2]` grid.

Both scenarios are always evaluated against the **same** `eval_features`/
`eval_labels` (the test set) -- only what the detector was fit on changes
between the two scenarios, mirroring the paper's own protocol of varying
training-time contamination composition while keeping the test set fixed.

This function is called **twice** in `main()`: once with raw features passed
in for `fit_features_*`/`eval_features`, and once with GNN embeddings passed
in instead -- the function itself has no idea which feature space it's
working with, which is what keeps the raw-vs-embeddings comparison
symmetric and bug-resistant (both code paths run through the exact same
grid-search logic).

`best_params_log` is new: it records which `(hyperparameter, contamination)`
combination actually achieved the best score for each `(scenario, algorithm)`
pair, which `main()` prints alongside the results table (Section 8.11) for
reproducibility -- so the exact settings behind any reported number are
always visible, not just the final score.

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

Unchanged from the earlier version -- builds exactly one bar chart, matching
the paper's Fig. 5-8 layout:

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
  the y-axis alone. Note these bar heights are now each the *best* score
  found across a full grid search, not a single-hyperparameter run.
- **`fig.savefig(output_path, dpi=150)`**: saves a PNG file at a resolution
  suitable for viewing/embedding in a report, before...
- **`plt.show()`**: ...also displaying it inline.

Called **twice** in `main()` -- once per scenario (`"0pct"` and `"natural"`),
producing two separate PNG files. **The `"natural"` call's title is now a
fixed string** (`"(natural contamination)"`) rather than dynamically
including a measured percentage -- see Section 8.12 for why.

---

## 7. Plotting Function: `plot_loss_curve()`

Unchanged from the earlier version:

```python
def plot_loss_curve(loss_history, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color="#4472C4", linewidth=1.5)
    ...
```

A simple line plot: x-axis is epoch number (1-indexed, matching the `Epoch
NNNN/4000` numbering printed by `trainer.py` during training), y-axis is the
DGI loss value recorded that epoch. With `epochs=4000` (rather than the
earlier `50`), this curve now has 80x more points, which should make
convergence behaviour (e.g. whether loss has plateaued well before epoch
4000, or is still decreasing near the end) considerably easier to read than
with the earlier short run.

`loss_history` here comes from the checkpoint -- it is **not** freshly
computed, so this plot reflects whatever training run produced the
checkpoint being loaded.

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

Unchanged from the earlier version -- reuses the same
`ANOMAL_E_CHECKPOINT_DIR` convention as `main.py`, fails fast with a clear
message if no checkpoint exists yet, and creates `plots/` up front.

### 8.2. Device setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Identical logic to `main.py`.

### 8.3. Rebuilding the data and graphs

```python
preprocessor = AnomalEPreprocessor()
train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=False, fraction=0.1)

graph_builder = AnomalEGraphBuilder()
train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
train_g = train_g.to(device)
test_g = test_g.to(device)
```

**⚠️ This is the single most important thing to keep in sync with
`main.py`.** The checkpoint being loaded contains weights for a specific
`AnomalEDGI` instance, whose input dimensions (`ndim_in`, `edims`) were
determined by whatever graph `main.py` built when it trained. **`fraction`
here is now `0.1`**, matching the value `main.py` currently uses (see
`MAIN_EXPLANATION.md` Section 5) -- if this script's `fraction`/
`sanity_check` arguments differ from what `main.py` used to produce the
loaded checkpoint, the rebuilt graph will have different node/edge counts,
and loading the checkpoint's `state_dict` into a differently-shaped encoder
can either error out (shape mismatch) or -- more dangerously -- silently
load into a graph it was never actually trained on, producing meaningless
embeddings and misleading plots.

**Practical rule**: whatever `fraction` value `main.py` used for the run
that produced the checkpoint you're about to load, this script's call must
use the exact same value. If you trained an older checkpoint with the
earlier `fraction=0.05`, either retrain with `fraction=0.1` first, or
temporarily change this script's `fraction` argument back to `0.05` to match
that specific older checkpoint.

### 8.4. Loading the trained encoder from checkpoint

```python
ndim_in = train_g.ndata['h'].shape[2]
edims = train_g.edata['h'].shape[2]
hidden_dim = 128
edge_hidden_dim = 256

dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim).to(device)
optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)  # required by load_checkpoint's API
trainer = AnomalETrainer(dgi_model, detector=None, optimizer=optimizer, epochs=4000)

print(f"[INFO] Loading checkpoint from '{checkpoint_path}'...")
completed_epochs = trainer.load_checkpoint(checkpoint_path, map_location=device)
print(f"[INFO] Loaded encoder trained for {completed_epochs} epochs.")
```

- `hidden_dim`/`edge_hidden_dim` are hardcoded to the same values `main.py`
  uses (128/256) -- these, too, must match whatever was used to produce the
  checkpoint, for the same shape-mismatch reasons as `fraction` above.
- `epochs=4000` here (raised from the earlier `50`) matches `main.py`'s
  current setting purely for consistent logging -- `trainer.epochs` is used
  in `save`/`load` messages; the actual completed-epoch count is restored
  separately from the checkpoint file itself via `load_checkpoint()`.
- A fresh `AnomalEDGI` is constructed with the correct shape, then
  immediately overwritten with the checkpoint's saved weights via
  `load_checkpoint()` -- the freshly-initialised (random) weights are
  discarded entirely; only the shape/architecture of this fresh instance
  matters, not its initial values.
- An `optimizer` is built here purely because `AnomalETrainer.load_checkpoint()`
  expects one to restore its state dict into. It is never actually stepped
  in this script -- no training happens here.
- `detector=None` is passed to `AnomalETrainer(...)` because this script
  never calls `trainer.evaluate()` -- it builds its own `AnomalEDetector`
  instances directly inside `grid_search_best_f1()` (Section 4), many times
  over, rather than using the single detector `AnomalETrainer` was
  originally designed around.
- `completed_epochs` (returned by `load_checkpoint`) is only used here for
  an informational print statement.

### 8.5. Plotting the loss curve

```python
if trainer.loss_history:
    plot_loss_curve(trainer.loss_history, os.path.join(plots_dir, "training_loss.png"))
else:
    print("[WARNING] No loss_history found in checkpoint -- skipping loss plot.")
```

Unchanged -- guards against an empty `loss_history` (e.g. from an older
checkpoint format) producing a blank/misleading plot.

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

Unchanged -- `corrupt=False` runs the encoder on the real (uncorrupted)
graph for both train and test graphs, producing genuine edge embeddings
used both for the grid search's fit sets (train) and its shared evaluation
set (test).

### 8.7. Extracting raw features

```python
train_raw = train_g.edata['h'].squeeze(1)
test_raw = test_g.edata['h'].squeeze(1)
```

Unchanged -- `.squeeze(1)` removes the singleton middle dimension, matching
the shape convention `train_embeddings`/`test_embeddings` already have.

### 8.8. Labels

```python
train_labels = train_g.edata['Label'].detach().cpu().numpy()
test_labels = test_g.edata['Label'].detach().cpu().numpy()
```

Labels are read directly from `edata['Label']`, for the same reason
explained in `trainer.py`'s `evaluate()` docstring and `MAIN_EXPLANATION.md`
Section 9: this is the only array structurally guaranteed to be in the same
edge order as the features/embeddings.

**What changed from the earlier version**: the earlier
`natural_contamination = float(np.clip(train_labels.mean(), 0.001, 0.5))`
computation has been **removed**. Grid search now sweeps the full
`CONTAMINATION_GRID` for every scenario (Section 5) instead of fixing a
single natural-contamination value derived from `train_labels.mean()` -- so
that computation is no longer needed. `train_labels`/`test_labels` are still
used for the benign-only mask (Section 8.9) and as the ground truth passed
into every grid-search evaluation.

### 8.9. Building the "0% contamination" fit sets

```python
benign_mask_train = (train_labels == 0)
train_raw_benign = train_raw[torch.from_numpy(benign_mask_train)]
train_embeddings_benign = train_embeddings[torch.from_numpy(benign_mask_train)]
```

Unchanged -- for the `"0pct"` scenario (matching the paper's
0%-contamination experiments in Tables 3/5), the detector should be fit
**only** on rows that are actually benign. This boolean-mask indexing
filters both `train_raw` and `train_embeddings` down to just the
`Label == 0` rows, passed as `fit_features_0pct` into `run_all_algorithms()`.
The full (unfiltered) `train_raw`/`train_embeddings` are separately passed
as `fit_features_natural`, for the other scenario.

### 8.10. Running all algorithms on both feature spaces

```python
raw_results, raw_best_params = run_all_algorithms(
    fit_features_0pct=train_raw_benign,
    fit_features_natural=train_raw,
    eval_features=test_raw,
    eval_labels=test_labels,
)

embed_results, embed_best_params = run_all_algorithms(
    fit_features_0pct=train_embeddings_benign,
    fit_features_natural=train_embeddings,
    eval_features=test_embeddings,
    eval_labels=test_labels,
)
```

Two calls to the same function (Section 5), one for each feature space.
**`contamination_natural` is no longer a parameter** -- both calls now rely
entirely on `run_all_algorithms()`'s internal grid search (Section 5) to
choose the best contamination per algorithm per scenario. `eval_features`
differs between the two calls (`test_raw` vs. `test_embeddings`), matching
whichever feature space is being evaluated; `eval_labels` (`test_labels`)
is shared, since it doesn't depend on which feature space is in use. Each
call now also returns a second value -- `raw_best_params`/`embed_best_params`
-- the per-`(scenario, algorithm)` best hyperparameters found (Section 5).

### 8.11. Console results table

```python
for scenario_key, scenario_label in [("0pct", "0% contamination (fit on benign only)"),
                                      ("natural", "Natural contamination (fit on full train set)")]:
    print(f"\n-- {scenario_label} --")
    print(f"{'Algorithm':<10} {'Raw Features':>15} {'Embeddings':>15}")
    for algo in ALGORITHMS:
        raw_f1 = raw_results[scenario_key][algo] * 100
        embed_f1 = embed_results[scenario_key][algo] * 100
        print(f"{ALGORITHM_DISPLAY_NAMES[algo]:<10} {raw_f1:>14.2f}% {embed_f1:>14.2f}%")
    print(f"\n  Best hyperparameters found (Raw Features): {raw_best_params[scenario_key]}")
    print(f"  Best hyperparameters found (Embeddings):   {embed_best_params[scenario_key]}")
```

A plain-text table printed to the console, mirroring the structure of the
paper's Tables 3/5 (though still limited to Macro F1, since that's the only
metric this script computes -- Accuracy/DR are not currently included, see
Section 10). Scenario labels were clarified (`"0% contamination (fit on
benign only)"` / `"Natural contamination (fit on full train set)"`) to make
explicit that -- now that both scenarios search the same contamination
grid -- what actually distinguishes them is the fit set, not a fixed
contamination number. **New**: each scenario's block now also prints the
exact `(hyperparameter, contamination)` combination that achieved the best
score for raw features and for embeddings, sourced from
`best_params_log` (Section 5) -- useful for reproducibility and for
directly citing "which settings" behind any number you report elsewhere.

### 8.12. Generating the plots

```python
plot_comparison(raw_results, embed_results, "0pct", "(0% contamination)",
                 os.path.join(plots_dir, "comparison_0pct_contamination.png"))
plot_comparison(raw_results, embed_results, "natural", "(natural contamination)",
                 os.path.join(plots_dir, "comparison_natural_contamination.png"))
```

Two calls to `plot_comparison()` (Section 6), one per scenario, each saving
a distinct PNG file under `plots/`. **The `"natural"` scenario's title is
now a fixed string**, rather than the earlier version's dynamically-computed
`f"(natural contamination, ~{natural_contamination * 100:.1f}%)"` -- since
`natural_contamination` (Section 8.8) no longer exists as a single number
(the scenario now searches a full contamination grid rather than fixing one
value), there is no single percentage left to display in the title.

---

## 9. Output Files Produced

Running this script produces (all under `plots/`, created if missing):

| File | Contents |
|---|---|
| `plots/training_loss.png` | DGI loss vs. epoch (1 to however many epochs the loaded checkpoint completed, up to 4000), from the loaded checkpoint's `loss_history`. |
| `plots/comparison_0pct_contamination.png` | Bar chart: best grid-searched Macro F1 for PCA/IF/CBLOF/HBOS, Raw Features vs. Embeddings, detector fit on benign-only training data. |
| `plots/comparison_natural_contamination.png` | Same bar-chart layout, but the detector is fit on the full training set (whatever real attack ratio survived downsampling). |

All three are also displayed inline (via `plt.show()`) when run in an
environment that supports it, such as a Colab notebook cell. The console
output additionally now includes the best hyperparameters found for every
bar in both charts (Section 8.11).

---

## 10. Known Limitations / What This Script Does NOT Do

- **No Accuracy or Detection Rate (DR)**: only Macro F1 is computed and
  plotted, unlike the paper's Tables 3-6 which also report Accuracy and DR
  side by side. Extending `grid_search_best_f1()` to also track
  `accuracy_score`/recall-based DR for whichever combination achieves the
  best F1 would be a straightforward addition if needed later.
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
- **Grid search here still won't exactly reproduce the paper's own reported
  numbers**, even with matching `fraction`, `epochs`, and hyperparameter
  grids -- see the discussion of remaining differences (encoder convergence
  variance, random seeds, dataset version/download source) if you're
  comparing directly against the paper's Tables 3-6.
- **Does not retrain if no checkpoint exists** -- by design, this script
  only ever loads an existing checkpoint; running `python main.py` first is
  a hard prerequisite.