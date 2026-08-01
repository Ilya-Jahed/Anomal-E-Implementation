# `main.py` — Anomal-E Pipeline Entry Point

> This document explains every part of `main.py`, the single script that runs
> the entire Anomal-E pipeline end to end: data preprocessing, graph
> construction, self-supervised GNN training (with resumable checkpointing),
> and anomaly-detection evaluation. Reading this top to bottom should let you
> reconstruct exactly what the code does and why, without needing to trace
> through every module it imports.

---

## 1. What This Script Does, In One Picture

```
Raw NetFlow CSV
      │
      ▼  AnomalEPreprocessor.process_pipeline()
Cleaned, encoded, L2-normalised train_df / test_df
      │
      ▼  AnomalEGraphBuilder.generate_graphs()
train_g, test_g  (DGL graphs, moved to GPU/CPU via .to(device))
      │
      ▼  AnomalEDGI (E-GraphSAGE encoder + DGI discriminator)
      ▼  AnomalETrainer.train()  -- resumable, checkpointed every 100 epochs
Trained encoder + 256-dim edge embeddings
      │
      ▼  AnomalETrainer.evaluate()  -- fits AnomalEDetector (HBOS), scores against labels
ROC AUC / F1 / Precision / Recall printed to console
```

`main.py` itself contains no model logic -- it is purely a **conductor**: it
builds each component in order, wires them together, and calls their
public methods. All of the actual algorithms (E-GraphSAGE, DGI, HBOS, etc.)
live in the modules it imports.

---

## 2. Imports

```python
import os
import torch
from src.data_pipeline.preprocessor import AnomalEPreprocessor
from src.data_pipeline.graph_builder import AnomalEGraphBuilder
from src.models.dgi_module import AnomalEDGI
from src.models.anomaly_detectors import AnomalEDetector
from src.engine.trainer import AnomalETrainer
```

| Import | Role |
|---|---|
| `os` | File-existence checks, directory creation, reading environment variables. |
| `torch` | Device selection (`cuda`/`cpu`), building the optimizer. |
| `AnomalEPreprocessor` | Phase 1: cleans the raw CSV, downsamples, target-encodes, L2-normalises, produces the final `train_df`/`test_df` with an `'h'` edge-feature column. |
| `AnomalEGraphBuilder` | Phase 1 (end): converts `train_df`/`test_df` into DGL graphs (`train_g`, `test_g`). |
| `AnomalEDGI` | Phase 2: the self-supervised model -- an E-GraphSAGE encoder + a DGI discriminator, trained without labels. |
| `AnomalEDetector` | Phase 3: wraps a classical PyOD anomaly detector (PCA/HBOS/CBLOF/IForest) that scores the GNN's edge embeddings. |
| `AnomalETrainer` | Phase 4: the "conductor" class that runs the DGI training loop and the final evaluation, tying the encoder and the detector together. |

---

## 3. Dataset Existence Check

```python
dataset_path = "data/raw/NF-CSE-CIC-IDS2018-v2.csv"    

if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset not found at: {dataset_path}")
    return
```

Fails fast with a clear message if the CSV hasn't been placed at the expected
path yet, instead of letting `pandas.read_csv` raise a less obvious
`FileNotFoundError` deep inside the preprocessor.

---

## 4. Device Setup (CPU vs. GPU)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Using device: {device} ===")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

- Automatically detects whether a CUDA-capable GPU is available and picks
  it; otherwise falls back to CPU so the exact same script also runs
  unmodified on a machine with no GPU.
- `device` is a plain `torch.device` object, later passed to every
  `.to(device)` call in this script (graphs, model) and to
  `load_checkpoint(..., map_location=device)`, so that a checkpoint saved on
  one device (e.g. GPU) can still be correctly loaded on another (e.g. CPU),
  with `map_location` handling that translation.
- The detector (HBOS/PCA/etc., Section 8 below) intentionally stays on CPU
  regardless of `device` -- PyOD/scikit-learn have no CUDA support, and
  `AnomalEDetector._prepare_data()` already `.detach().cpu().numpy()`s
  whatever tensor it receives before fitting/scoring.

---

## 5. Phase 1: Data Preprocessing

```python
print("=== Starting Phase 1: Data Pipeline ===")
preprocessor = AnomalEPreprocessor()
train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=False, fraction=0.1)
```

- `sanity_check=False` means the full pipeline runs on the real dataset
  (not the 50,000-row quick-smoke-test subset).
- **`fraction=0.1`**: the fraction of rows kept after stratified
  downsampling (stratified by the `Attack` column, so rare attack types
  aren't wiped out). This value was deliberately set to **match the
  reference Anomal-E notebook exactly** (its Cell 6:
  `data.groupby(by='Attack').sample(frac=0.1, random_state=13)`), rather
  than an earlier, smaller value (`0.05`) this project used at one point.

  The earlier `0.05` value existed purely to fit a specific constraint:
  `AnomalESAGEEncoder` runs full-batch (the entire graph processed in a
  single forward pass, no mini-batching), so the number of edges in the
  graph directly determines peak GPU memory during training, and
  `fraction=0.1` produces a graph with roughly 5.29M edges -- on a 14.56GB
  GPU, that combination triggered a `CUDA OutOfMemoryError`, so `fraction`
  was temporarily halved to `0.05` to fit that specific hardware.

  On a GPU with substantially more VRAM (e.g. an 80GB A100), the same
  ~5.29M-edge graph fits comfortably, so `fraction=0.1` is used here to
  keep this project's Macro-F1 results **meaningfully comparable** to the
  paper's own reported numbers -- comparing against a paper that trained on
  10% of the data while this project trained on only 5% would confound any
  difference in results with a difference in training-set size, on top of
  whatever the actual comparison of interest is (raw features vs.
  embeddings, etc.). If you are running on a smaller GPU and hit an
  out-of-memory error again, lower `fraction` back down (e.g. to `0.05` or
  `0.02`) -- just be aware that doing so makes results less directly
  comparable to the paper's Tables 3-6.
  
  Two separate memory optimisations elsewhere in the pipeline (not affected
  by this `fraction` choice) also help fraction=0.1 stay feasible on modest
  hardware: `AnomalEPreprocessor.load_and_clean_data()` reads the CSV in
  memory-safe chunks and downsamples each chunk immediately rather than
  loading the entire multi-million-row file into RAM at once, and
  `apply_normalization()` stores the final `'h'` feature vectors as
  `float32` NumPy row-arrays rather than converting them to full Python
  lists of Python floats.

`train_df`/`test_df` come back fully preprocessed: categorical columns
target-encoded, numeric columns L2-normalised, and packed into a single
`'h'` column per row -- this is exactly the edge feature vector `e_uv`
referred to throughout the Anomal-E paper.

> **Note on labels**: the code comment right after this call explains why no
> separate `test_labels` array is extracted here anymore -- see Section 9
> below (labels are read directly from the graph's edge data instead, to
> guarantee correct alignment with the edge embeddings).

---

## 6. Checkpoint Setup

```python
checkpoint_dir = os.environ.get("ANOMAL_E_CHECKPOINT_DIR", "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "anomal_e_dgi.pt")
checkpoint_every = 100  # save a mid-training checkpoint every N epochs
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"[INFO] Checkpoint path: {checkpoint_path} (saved every {checkpoint_every} epochs)")
```

- `checkpoint_dir` defaults to a local `checkpoints/` folder inside the repo,
  but can be overridden via the `ANOMAL_E_CHECKPOINT_DIR` environment
  variable to point at a persistent location instead -- e.g. a mounted
  Google Drive path in Colab, so training progress survives even if the
  Colab runtime itself is torn down:
  ```bash
  ANOMAL_E_CHECKPOINT_DIR="/content/drive/MyDrive/anomal_e_checkpoints" python main.py
  ```
- **`checkpoint_every = 100`**: raised from an earlier value of `10` now
  that `epochs=4000` (Section 8) -- checkpointing every 10 epochs out of
  4000 would mean 400 checkpoint writes over the course of one run, which
  is unnecessary disk I/O for a run this long. Checkpointing every 100
  epochs instead still bounds the worst case tightly (at most 99 epochs of
  progress lost if a session is interrupted) while writing 40 checkpoints
  total instead of 400. `AnomalETrainer.train()` also always saves a
  checkpoint on the final epoch regardless of this interval, so a run that
  finishes cleanly never ends without an up-to-date checkpoint on disk.
- This local `checkpoints/` directory should be listed in `.gitignore` --
  it's a local run artifact, not part of the source code, and checkpoint
  files (containing full model + optimizer state) can be large.

---

## 7. Phase 1 (continued): Graph Construction + Device Placement

```python
graph_builder = AnomalEGraphBuilder()
train_g, test_g = graph_builder.generate_graphs(train_df, test_df)

train_g = train_g.to(device)
test_g = test_g.to(device)
```

- `generate_graphs()` builds two **completely independent** DGL graphs from
  `train_df` and `test_df` -- separate node ID spaces, no shared identity
  even for an IP address appearing in both splits. This intentionally
  matches the paper's strict train/test separation to avoid data leakage.
- `g.to(device)` moves both the graph structure and everything stored in
  `g.ndata`/`g.edata` to the target device in one call -- so the
  `ndata['h']`/`edata['h']` tensors indexed below are already on `device`
  by the time they're used.

---

## 8. Phase 2 & 3: Model, Optimizer, Detector, Trainer

```python
print("\n=== Starting Final Phase: Execution Engine ===")
ndim_in = train_g.ndata['h'].shape[2]
edims = train_g.edata['h'].shape[2]
hidden_dim = 128
edge_hidden_dim = 256

# 1. GNN Model
print("Initializing DGI Model...")
dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim).to(device)

# 2. Optimizer
optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)

# 3. Anomaly Detector (HBOS with 10% contamination)
print("Initializing HBOS Detector...")
detector = AnomalEDetector(model_name='hbos', contamination=0.10)

# 4. Trainer
trainer = AnomalETrainer(dgi_model, detector, optimizer, epochs=4000)
```

Line by line:

- `ndim_in`, `edims`: read directly off the graph's feature tensors (shape
  `(N, 1, ndim_in)` and `(E, 1, edims)` respectively), so the model's input
  dimensions always automatically match whatever the preprocessing pipeline
  produced -- no hardcoded feature-count constants to keep in sync manually.
- `hidden_dim = 128`, `edge_hidden_dim = 256`: match the paper's
  hyperparameters (Table 1) -- 128-unit hidden node representations, and
  256-dim final edge embeddings (node embeddings concatenated, doubling the
  size, per Eq. 5 of the paper).
- `dgi_model = AnomalEDGI(...).to(device)`: builds the encoder +
  discriminator and moves all their parameters to the training device.
- `optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)`: built
  **after** `.to(device)`, deliberately -- constructing the optimizer before
  moving the model would bind it to the pre-move (e.g. CPU) parameter
  copies, and it would then be optimizing stale tensors instead of the ones
  actually used in the forward pass.
- `detector = AnomalEDetector(model_name='hbos', contamination=0.10)`: the
  classical anomaly-scoring algorithm that will later be fit on the GNN's
  edge embeddings, used only by this script's own `trainer.evaluate()` call
  in Section 9. `contamination=0.10` here is a fixed, hand-picked value --
  contrast this with `plot_comparison.py`, which instead grid-searches
  `contamination` (and each algorithm's own hyperparameter) automatically;
  see `PLOTCOMPARISON_EXPLANATION.md` for why that script needs the more
  thorough search and this one doesn't (it exists mainly as an end-to-end
  smoke test / single quick metric readout, not the source of the
  paper-comparable numbers).
- **`trainer = AnomalETrainer(dgi_model, detector, optimizer, epochs=4000)`**:
  `epochs=4000` was raised from an earlier value of `50`, again to **match
  the reference notebook exactly** (its Cell 19: `epochs = 4000`). The
  previous, much shorter `epochs=50` setting was a fast-iteration value
  used while getting the pipeline itself working end-to-end; with the full
  4000-epoch run, the encoder gets substantially more opportunity to
  converge, which matters for how close this project's embeddings-based
  results can get to the paper's own reported numbers (see
  `PLOTCOMPARISON_EXPLANATION.md`'s discussion of what still differs
  between this project's results and the paper's, even after matching
  `fraction` and `epochs`).

---

## 9. Phase 4: Resumable Training + Evaluation

```python
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"\n[INFO] Found existing checkpoint at '{checkpoint_path}' -- resuming from it.")
    start_epoch = trainer.load_checkpoint(checkpoint_path, map_location=device)

trainer.train(
    train_g,
    train_g.ndata['h'],
    train_g.edata['h'],
    checkpoint_path=checkpoint_path,
    checkpoint_every=checkpoint_every,
    start_epoch=start_epoch,
)

trainer.evaluate(test_g, test_g.ndata['h'], test_g.edata['h'])
```

### Resuming from a checkpoint

- If a checkpoint file already exists at `checkpoint_path` (from an earlier
  run that was interrupted, finished, or is just being re-used),
  `trainer.load_checkpoint(...)` restores the encoder's and optimizer's
  state dicts **and returns how many epochs were already completed**
  (`start_epoch`).
- `map_location=device` ensures a checkpoint saved on one device (e.g. GPU
  during a previous Colab session) loads correctly even if this run happens
  on a different device (e.g. CPU), by remapping tensor storage locations
  during deserialisation.
- If no checkpoint exists yet, `start_epoch` stays `0` and training starts
  fresh from epoch 1.
- With `epochs=4000` (Section 8), resuming matters more than it did at
  `epochs=50` -- a 4000-epoch run is far more likely to span multiple
  sessions on a shared/time-limited GPU, and losing at most
  `checkpoint_every - 1` = 99 epochs of progress (Section 6) rather than
  the entire run is what makes that practical.

### Training call

- `trainer.train(...)` runs the self-supervised DGI training loop (see
  `AnomalEDGI.forward()` for what happens inside each epoch: real vs.
  corrupted graph embeddings, scored by a discriminator against a global
  graph summary, combined into one BCE loss). No labels are read or used
  anywhere in this call.
- Passing `checkpoint_path`/`checkpoint_every` here means `train()` will
  itself call `save_checkpoint(...)` every 100 epochs (and on the final
  epoch), so this single call handles both fresh runs and resumed runs
  uniformly -- the only difference is what `start_epoch` was set to above.
- If `start_epoch >= 4000` (i.e. training was already fully completed in a
  previous run), `train()` detects this, prints an informational message,
  and returns immediately without doing any further work or re-saving a
  checkpoint -- it does **not** silently redo all 4000 epochs, nor does it
  error out.

### Evaluation call

- `trainer.evaluate(test_g, ...)` puts the encoder in eval mode, extracts
  edge embeddings from the **test** graph (no gradient tracking), fits the
  detector (`self.detector.fit(edge_embeddings)`) on those embeddings, and
  scores the predictions against ground-truth labels.
- **No labels array is passed into this call.** `evaluate()` reads labels
  directly from `test_g.edata['Label']` internally. This is deliberate: the
  graph-construction step (`AnomalEGraphBuilder`) can duplicate/reorder
  edges relative to the original dataframe's row order, so a label array
  pulled straight from `test_df['Label'].values` is not guaranteed to line
  up with `edge_embeddings` by position -- a same-length-but-wrong-order
  array would silently produce wrong metrics with no error raised. Reading
  labels from `test_g.edata['Label']` instead is structurally guaranteed to
  match the edge order the encoder just embedded, because `'Label'` is
  carried through graph construction as an edge attribute alongside `'h'`.
- Printed metrics: ROC AUC (skipped with a warning if only one class is
  present in the labels, which can happen on very small/degenerate splits),
  Macro-unweighted F1, Precision, and Recall -- computed via
  scikit-learn's `f1_score`/`precision_score`/`recall_score`/`roc_auc_score`
  against the detector's binary predictions and continuous anomaly scores.

---

## 10. Final Message

```python
print("\n🎉 === ANOMAL-E PIPELINE SUCCESSFULLY COMPLETED === 🎉")
```

Purely a friendly confirmation that every phase (preprocessing → graph
construction → training/resuming → evaluation) ran without raising an
exception.

---

## 11. Quick Reference: How to Re-run / Resume

| Situation | What happens when you run `python main.py` |
|---|---|
| First run, no checkpoint exists | Preprocesses full dataset (10% of rows), builds graphs, trains from epoch 1 to 4000, saving checkpoints every 100 epochs, then evaluates. |
| Session interrupted mid-training (e.g. epoch 2300) | Preprocessing and graph-building re-run from scratch (they are not checkpointed), but `AnomalEDGI` training resumes from the last saved checkpoint (epoch 2200, in this example) instead of epoch 1. |
| Training already fully completed (checkpoint at epoch 4000) | `train()` detects `start_epoch >= epochs`, skips training entirely, and goes straight to `evaluate()` -- useful for re-running only the evaluation step (e.g. after changing the detector) without retraining. |
| You change `fraction` (data size) between runs | **Delete the old checkpoint first.** A checkpoint's saved encoder weights match a specific input feature/graph configuration; loading an old checkpoint against a differently-sized graph can error out or silently misbehave, since the graph structure itself (not just the model weights) changed. This also applies if you switch back to the older `fraction=0.05` value -- it will not load against a checkpoint trained with `fraction=0.1`. |

---

## 12. Things Deliberately Left Out of `main.py`

- **Mini-batch training**: the encoder runs full-batch (entire graph per
  forward pass), matching the paper's own design. This is why `fraction`
  needs to be kept low enough to fit in available GPU memory -- there is no
  batching logic here to fall back on if the graph is too large.
- **Detector hyperparameter search**: `contamination` is hardcoded to a
  single value (`0.10`) here, and each PyOD detector uses its own library
  default for algorithm-specific parameters (e.g. `n_components` for PCA).
  The thorough per-algorithm grid search matching the reference notebook
  lives in `plot_comparison.py` instead, which is the script whose numbers
  should actually be compared against the paper's Tables 3-6 -- see
  `PLOTCOMPARISON_EXPLANATION.md`.
- **Plotting**: `main.py` only prints final metrics to the console. Loss
  curves and Raw-Features-vs-Embeddings comparison bar charts (mirroring
  the paper's Fig. 5-8) are produced by a separate script
  (`plot_comparison.py`), which loads the checkpoint saved here rather than
  retraining, to keep plotting iterations fast.