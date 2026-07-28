# `AnomalEDetector` — Anomaly Detection Stage of the Anomal-E Pipeline

> This document explains what `anomaly_detector.py` does, why it exists, how it fits
> into the rest of the Anomal-E pipeline, and how to use it — including the meaning
> of every parameter, tied back to the original paper (Caville, Lo, Layeghy &
> Portmann, *"Anomal-E: A self-supervised network intrusion detection system based
> on graph neural networks"*, Knowledge-Based Systems 258, 2022).

---

## 1. Where This Fits in the Pipeline

Anomal-E is a **four-stage pipeline**. This file implements only the **last stage**.

```
Raw NetFlow CSV (IPV4_SRC_ADDR, IPV4_DST_ADDR, byte counts, ...)
        │
        ▼
 preprocessor.py        cleaning, target-encoding, L2 normalisation
        │
        ▼
 graph_builder.py       builds a DGL graph: IPs = nodes, flows = edges
        │
        ▼
 AnomalESAGEEncoder     self-supervised E-GraphSAGE + DGI training
        │                (no labels used anywhere in this step)
        ▼
   256-dim edge embeddings z_uv, one per network flow
        │
        ▼
 anomaly_detector.py     <-- THIS FILE
        │
        ▼
 Anomaly score / binary label (benign vs. attack) per flow
```

By the time data reaches `AnomalEDetector`, all the "hard" representation-learning
work is already done. The GNN encoder's entire job was to transform raw, noisy
flow statistics into a 256-dimensional space where benign and malicious flows are
*easier to separate*. This module does the actual separating, using simple,
fast, well-understood classical outlier-detection algorithms rather than more GNN
machinery. This mirrors Algorithm 2 (lines 11–14) and Section 4.4 of the paper.

**Crucially, nothing in this pipeline — including this file — ever needs attack
labels to train.** `contamination` is only a *prior belief* about how dirty the
data might be, not a label column.

---

## 2. Why Classical Algorithms, Not Another Neural Net?

The paper deliberately hands off from the GNN to classical unsupervised
detectors for two reasons:

1. **Modularity** — any embedding-producing encoder (E-GraphSAGE, plain
   GraphSAGE, DGI, etc.) can be swapped in without touching the detection
   logic, since these detectors only ever see a plain `(n_samples, n_features)`
   matrix.
2. **They are the standard, cheap, interpretable baselines for unsupervised
   anomaly detection.** Comparing "raw flow features + these detectors" vs.
   "Anomal-E embeddings + these same detectors" isolates exactly how much the
   GNN embedding step helps (Tables 3–6 in the paper), without any confound
   from also changing the detection algorithm.

---

## 3. The Four Supported Algorithms

| `model_name` | Algorithm | Core idea | Paper ref. |
|---|---|---|---|
| `'pca'` | PCA-based anomaly detector | Learns the principal components of "mostly normal" data, then flags points whose reconstruction from those components deviates strongly. Note: this is **not** ordinary PCA for dimensionality reduction — it's an anomaly-scoring extension of it. | [7] |
| `'hbos'` | Histogram-Based Outlier Score | Builds one histogram per feature dimension; a point's score is derived from how rare its bin is, combined across all dimensions. Assumes feature independence; very fast. | [9] |
| `'cblof'` | Cluster-Based Local Outlier Factor | Clusters the embeddings (e.g. k-means), labels clusters as "large/normal" or "small/anomalous", then scores points by cluster size and distance to the nearest large cluster. | [8] |
| `'iforest'` | Isolation Forest | Builds an ensemble of random decision trees. Anomalies are isolated in fewer splits (closer to the root) than normal points, so shallow average path length ⇒ high anomaly score. | [6] |

All four are wrapped through [PyOD](https://pyod.readthedocs.io/) (Python Outlier
Detection), which gives every model the same `fit` / `predict` /
`decision_function` interface — this is exactly why `AnomalEDetector` can treat
them interchangeably behind one class.

---

## 4. Class Reference

### `AnomalEDetector(model_name='pca', contamination=0.1, random_state=42, **kwargs)`

| Parameter | Type | Meaning |
|---|---|---|
| `model_name` | `str` | Which of the four algorithms to use: `'pca'`, `'hbos'`, `'cblof'`, `'iforest'`. Case-insensitive. |
| `contamination` | `float` | Expected fraction of anomalies in the fitting data, in `(0, 0.5)`. PyOD uses this to pick where the anomaly-score cutoff sits — it does **not** require label access, just a prior guess. The paper's grid search sweeps `[0.001, 0.01, 0.04, 0.05, 0.1, 0.2]` (see Table 2), and uses **0.04** specifically to match the natural attack ratio of NF-UNSW-NB15-v2 in the "contaminated" experiments. |
| `random_state` | `int` | Seed for reproducibility, for the algorithms that have randomness (PCA's solver, CBLOF's clustering, IForest's random splits). **Not accepted by HBOS**, which is a deterministic histogram method — passing it there would raise a `TypeError`, which is why the code special-cases it. |
| `**kwargs` | — | Any extra PyOD hyperparameter for the chosen model, corresponding to the other axes of the paper's Table 2 grid search: `n_components` (PCA), `n_bins` (HBOS), `n_clusters` (CBLOF), `n_estimators` (IForest). |

Raises `ValueError` if `model_name` isn't one of the four supported strings.

### `fit(X)`

Trains the chosen PyOD model on a matrix of embeddings `X`.

- **Input:** `X` — a `torch.Tensor` or `np.ndarray` of shape `(n_samples, n_features)`
  (e.g. `(n_train_flows, 256)` when using Anomal-E's default 256-dim edge
  embeddings).
- **No labels are passed or needed.** In the paper's "0% contamination"
  experiments, `X` contains embeddings of benign flows only; in the "4%
  contamination" experiments, ~4% of `X`'s rows are secretly attack flows
  mixed in with benign ones — but the method itself never sees which is which.
- **Returns:** `self`, so calls can be chained:
  ```python
  detector = AnomalEDetector('iforest', contamination=0.04).fit(z_train)
  ```

### `predict(X)`

Returns a **hard binary label** per row of `X`, using the threshold PyOD
derived internally from `contamination` at fit time.

- **Returns:** `np.ndarray` of shape `(n_samples,)`, integers: `0` = benign,
  `1` = anomaly / predicted attack. This is what gets compared against ground
  truth labels to compute Accuracy, Macro F1, and Detection Rate (Tables 3–8
  of the paper) — but note the ground-truth comparison happens *outside* this
  class, purely for evaluation; the model itself trains and predicts
  label-free.

### `get_anomaly_scores(X)`

Returns the **continuous anomaly score** per row of `X`, without collapsing
it to a 0/1 label.

- Higher score = more anomalous. PyOD standardises the sign/direction across
  all four algorithms, so `'pca'`, `'hbos'`, `'cblof'`, and `'iforest'` scores
  are all comparable in direction even though their internal math differs
  (reconstruction error vs. histogram rarity vs. cluster distance vs. inverse
  path length).
- Useful for ranking flows by severity, plotting score distributions, or
  computing ROC curves without needing to refit at a different
  `contamination` value.

---

## 5. Internal Helper: `_prepare_data(X)`

Edge embeddings usually arrive straight out of `AnomalESAGEEncoder.forward()`
as a `torch.Tensor` — possibly still attached to the autograd graph, and
possibly living on a GPU. PyOD's models are scikit-learn-based and only
understand NumPy arrays. `_prepare_data` bridges this gap:

```python
if isinstance(X, torch.Tensor):
    return X.detach().cpu().numpy()
return np.array(X)
```

1. **`.detach()`** — severs the tensor from the autograd computation graph, so
   we don't accidentally keep gradient-tracking history alive in memory just
   to score embeddings.
2. **`.cpu()`** — moves data off the GPU, since NumPy can't read GPU memory.
3. **`.numpy()`** — converts to a plain NumPy array (cheap/near-zero-copy once
   the tensor is on CPU and detached).

If `X` is already a NumPy array or plain list, it's passed through
`np.array(X)` unchanged in spirit — this keeps `fit`, `predict`, and
`get_anomaly_scores` agnostic to whether embeddings come straight from the
GNN or from a saved `.npy` file.

---

## 6. Usage Example

```python
from anomaly_detector import AnomalEDetector

# z_train, z_test: (N, 256) edge embeddings from a trained AnomalESAGEEncoder,
# generated by running the encoder once more in eval mode after DGI training
# (Algorithm 2, line 10 of the paper): z_uv, _ = encoder(g, nfeats, efeats)

detector = AnomalEDetector(model_name='iforest', contamination=0.04, n_estimators=100)
detector.fit(z_train)

labels = detector.predict(z_test)              # 0 = benign, 1 = attack
scores = detector.get_anomaly_scores(z_test)    # continuous severity, higher = worse
```

To reproduce the paper's grid search over an algorithm's own hyperparameter
(Table 2), sweep `**kwargs` alongside `contamination`:

```python
best_f1 = -1
for n_bins in [5, 10, 15, 20, 25, 30]:
    for contamination in [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]:
        detector = AnomalEDetector('hbos', contamination=contamination, n_bins=n_bins)
        detector.fit(z_train)
        preds = detector.predict(z_test)
        f1 = macro_f1(y_test, preds)   # your own evaluation function
        if f1 > best_f1:
            best_f1, best_params = f1, (n_bins, contamination)
```

---

## 7. Quick Reference Summary

| Concept | What it is | Where in this file |
|---|---|---|
| `model_name` | Selects one of 4 PyOD detectors | `__init__` |
| `contamination` | Prior belief on anomaly fraction; sets decision threshold; no labels needed | `__init__` |
| `random_state` | Reproducibility seed (all models except HBOS) | `__init__` |
| `_prepare_data` | Torch tensor / list → detached, CPU, NumPy array | Internal helper |
| `fit(X)` | Trains the detector on embeddings, unsupervised | Corresponds to paper Algorithm 2, lines 11–14 |
| `predict(X)` | Hard 0/1 labels using the contamination-derived threshold | Feeds Accuracy / Macro F1 / DR in Tables 3–8 |
| `get_anomaly_scores(X)` | Continuous severity score, higher = more anomalous | For ranking / ROC curves / custom thresholds |
| PyOD | Library providing a consistent API across PCA, HBOS, CBLOF, IForest | Imported at top of file |