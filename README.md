# Anomal-E: Self-Supervised Network Anomaly Detection using E-GraphSAGE

> A modular, research-grade implementation of the **Anomal-E** architecture — an edge-centric Graph Neural Network trained entirely without attack labels, using Deep Graph Infomax (DGI) for self-supervised learning on NetFlow data.

---

## What is Anomal-E?

Most network intrusion detection systems rely on labelled datasets — they need to know in advance which flows are attacks. **Anomal-E** removes this requirement entirely.

It works by modelling a network as a **graph** — IP addresses become nodes, network flows become edges — and learning embeddings for each flow (edge) using a custom GNN. Those embeddings are then scored by classical unsupervised detectors (PCA, Isolation Forest, etc.) to flag anomalies without ever seeing an attack label during training.

The key technical innovations:

- **E-GraphSAGE**: Unlike standard GraphSAGE which aggregates node features from neighbours, E-GraphSAGE aggregates **edge features** (bytes, packet counts, duration) — because in NetFlow data, the meaningful information lives on the connection, not on the IP address itself.
- **DGI self-supervised training**: The encoder is trained by comparing real graph embeddings against embeddings of a corrupted graph (same topology, shuffled edge features). No labels required.
- **Classical unsupervised scoring**: The trained edge embeddings are handed off to well-understood detectors (PCA, HBOS, CBLOF, Isolation Forest) that never see a label either — keeping the whole pipeline label-free end to end.

---

## 🏗️ Project Architecture

The project follows Clean Architecture principles — each component has exactly one responsibility and communicates only through clearly defined interfaces.

```
Anomal-E-Implementation/
│
├── data/
│   ├── raw/                        # Original NF-CSE-CIC-IDS2018-v2.csv goes here
│   └── processed/                  # Intermediate outputs (future use)
│
├── src/
│   ├── data_pipeline/
│   │   ├── preprocessor.py         # Cleaning, target encoding, L2 normalisation
│   │   └── graph_builder.py        # NetFlow DataFrame → DGL bidirectional graph
│   │
│   ├── models/
│   │   ├── e_graphsage.py          # AnomalESAGELayer + AnomalESAGEEncoder
│   │   ├── dgi_module.py           # Discriminator + AnomalEDGI training module
│   │   └── anomaly_detectors.py    # AnomalEDetector: PCA / IF / CBLOF / HBOS
│   │
│   └── engine/
│       ├── trainer.py              # DGI training loop              [Phase 4 — coming soon]
│       └── evaluator.py            # Embedding evaluation           [Phase 4 — coming soon]
│
├── docs/
│   ├── DATA_PIPELINE_EXPLANATION.md          # Deep-dive on preprocessor + graph builder
│   ├── E_GRAPHSAGE_EXPLANATION.md            # Deep-dive on AnomalESAGELayer + Encoder
│   ├── DGI_MODULE_EXPLANATION.md             # Deep-dive on Discriminator + AnomalEDGI
│   └── ANOMAL-E_DETECTOR_EXPLANATION.md      # Deep-dive on AnomalEDetector (PCA/HBOS/CBLOF/IForest)
│
├── configs/                        # Hyperparameter configs (future use)
├── notebooks/                      # Exploratory notebooks (future use)
├── main.py                         # End-to-end pipeline entry point
└── requirements.txt
```

---

## 🔬 How the Pipeline Works

```
Raw CSV (NetFlow data)
        │
        ▼  preprocessor.py
   Drop ports → Stratified downsample → Train/Test split
   → Target-encode categorical columns → L2 normalise
   → Pack all numeric columns into edge feature vector 'h'
        │
        ▼  graph_builder.py
   Each row → one directed edge (src IP → dst IP)
   Node features = constant vector of 1s (same dim as edge features)
   Result: DGL bidirectional graph with ndata['h'] and edata['h']
        │
        ▼  e_graphsage.py  (AnomalESAGELayer)
   1. Each edge sends its feature vector to its destination node
   2. Each node averages all incoming edge features (E-GraphSAGE aggregation)
   3. Node updates its embedding: concat(own feats, aggregated) → Linear + ReLU
   4. Edge updates its embedding: concat(updated src, updated dst) → Linear
        │
        ▼  dgi_module.py  (AnomalEDGI)
   Real pass:      encoder(graph, real edge features)   → pos_edge_emb
   Corrupt pass:   encoder(graph, SHUFFLED edge features) → neg_edge_emb
   Summary:        sigmoid(mean(pos_edge_emb))
   Loss:           BCE(discriminator(pos) vs 1) + BCE(discriminator(neg) vs 0)
        │
        ▼  anomaly_detectors.py  (AnomalEDetector)
   Fit PCA / IF / CBLOF / HBOS on trained edge embeddings — unsupervised
   Score each flow → anomaly score / benign-attack label
```

---

## 🚀 Setup and Installation

This project uses `conda` to maintain a strict and reproducible environment.

**1. Create the environment:**
```bash
conda create -n anomal-e python=3.10 -y
```

**2. Activate the environment:**
```bash
conda activate anomal-e
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

> The `requirements.txt` pins PyTorch 2.1.2 (CPU) and DGL to a compatible wheel. If you want GPU support, replace the `torch` and `dgl` lines with the appropriate CUDA variants from [pytorch.org](https://pytorch.org) and [dgl.ai](https://www.dgl.ai).

**4. Place the dataset:**

Download the original `NF-CSE-CIC-IDS2018-v2.csv` from the [University of Queensland NIDS dataset](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) and place it at:
```
data/raw/NF-CSE-CIC-IDS2018-v2.csv
```

> ⚠️ **Use the original CSV, not Parquet.** Converted Parquet versions (e.g. from Kaggle) frequently drop the `IPV4_SRC_ADDR` and `IPV4_DST_ADDR` columns, which are required for graph construction.

---

## ▶️ Running the Pipeline

```bash
python main.py
```

This runs the full Phase 1 + Phase 2 pipeline in **sanity-check mode** (50,000 rows), confirming end-to-end correctness before committing to a full multi-million-row run. Expected output:

```
=== Starting Phase 1: Data Pipeline ===
[INFO] Loading FULL dataset from CSV...
[INFO] Dropped Source and Destination Ports.
[INFO] Downsampling data to 10.0%...
[INFO] Splitting data into Train and Test sets...
[INFO] Applying Target Encoding...
[INFO] Applying L2 Normalization...
[SUCCESS] Data Preprocessing Pipeline Completed.

[INFO] Building Training Graph...
[INFO] Building Testing Graph...
[SUCCESS] Graphs generated. Train Nodes: ..., Train Edges: ...

=== Phase 1 Execution Successful ===

[INSPECTION] Training Graph Overview:
...
Node Features (ndata['h']) shape: (N, 1, F)
Edge Features (edata['h']) shape: (E, 1, F)
```

---

## 📖 Documentation

Each implemented module has a corresponding deep-dive document in `docs/` that maps every line of code back to the paper's equations and algorithms:

| Document | Covers |
|---|---|
| [`DATA_PIPELINE_EXPLANATION.md`](docs/DATA_PIPELINE_EXPLANATION.md) | Preprocessing steps, target encoding, L2 normalisation, graph construction, `ndata`/`edata` structure |
| [`E_GRAPHSAGE_EXPLANATION.md`](docs/E_GRAPHSAGE_EXPLANATION.md) | `AnomalESAGELayer` (message passing, node update, edge update), `AnomalESAGEEncoder` (DGI corruption, layer stacking), `g.ndata` deep-dive |
| [`DGI_MODULE_EXPLANATION.md`](docs/DGI_MODULE_EXPLANATION.md) | `Discriminator` (bilinear form, `nn.Parameter` vs `nn.Linear`), `AnomalEDGI` (full Algorithm 2 mapping, loss computation) |
| [`ANOMAL-E_DETECTOR_EXPLANATION.md`](docs/ANOMAL-E_DETECTOR_EXPLANATION.md) | `AnomalEDetector` (PCA / HBOS / CBLOF / Isolation Forest wrapping via PyOD, `contamination` meaning, `fit`/`predict`/`get_anomaly_scores`) |

---

## 🗺️ Development Roadmap

- [x] **Phase 1 — Data Pipeline & Graph Builder**
  - NetFlow CSV loading, port dropping, stratified downsampling
  - Target encoding of categorical features, L2 normalisation
  - DGL bidirectional graph construction with per-edge feature vectors

- [x] **Phase 2 — Core GNN Architecture & DGI Module**
  - `AnomalESAGELayer`: edge-feature aggregation, node update (Eq. 4 & 2), edge update (Eq. 5)
  - `AnomalESAGEEncoder`: DGI corruption mechanism, layer stacking
  - `Discriminator`: bilinear scoring (Eq. 6/7)
  - `AnomalEDGI`: full self-supervised training objective (Algorithm 2)

- [x] **Phase 3 — Unsupervised Anomaly Detectors**
  - `AnomalEDetector`: unified PyOD wrapper for PCA, HBOS, CBLOF, Isolation Forest
  - Fitted directly on trained edge embeddings — fully label-free
  - Per-flow binary label (`predict`) and continuous severity score (`get_anomaly_scores`)

- [ ] **Phase 4 — Execution Engine**
  - `Trainer`: DGI training loop with optimizer, scheduler, checkpointing
  - `Evaluator`: AUC-ROC, F1, precision/recall on test graph embeddings

---

## 📄 Reference

This implementation is based on:

> **Anomal-E: A Self-Supervised Network Intrusion Detection System based on Graph Neural Networks**  
> Caville et al., 2022  
> [arXiv:2207.06819](https://arxiv.org/abs/2207.06819)

The E-GraphSAGE encoder extends the original GraphSAGE architecture from:

> **Inductive Representation Learning on Large Graphs**  
> Hamilton et al., NeurIPS 2017  
> [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

The self-supervised DGI training objective is from:

> **Deep Graph Infomax**  
> Veličković et al., ICLR 2019  
> [arXiv:1809.10341](https://arxiv.org/abs/1809.10341)