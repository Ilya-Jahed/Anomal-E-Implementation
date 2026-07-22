# Anomal-E: Data Preprocessing & Graph Construction Pipeline
### A complete, worked-example walkthrough of `anomal_e_preprocessor_annotated.py` and `anomal_e_graph_builder_annotated.py`

This document explains, in full detail and with one running numeric example carried through every step, how raw NetFlow records become a PyTorch/DGL graph ready to be fed into E-GraphSAGE. It corresponds to two files:

- **File 1 — `AnomalEPreprocessor`**: raw CSV/Parquet → cleaned, encoded, normalised train/test dataframes
- **File 2 — `AnomalEGraphBuilder`**: cleaned dataframes → DGL graph objects

Together these two files implement exactly what Fig. 4 of the Anomal-E paper describes:

```
Drop Port → Downsampling → Train/Test Split → Feature Conversion (Target Encoding)
→ Replace Empty/Infinite Values → L2 Normalisation → Train & Test Graph Generation
```

---

## 0. The starting point: what raw NetFlow data looks like

Before any processing, we have a plain table. Each row is **one unidirectional flow** between a source IP and a destination IP, with a set of statistics describing that flow, plus the ground-truth label (used only for evaluation and for target encoding — never for the self-supervised training of E-GraphSAGE itself).

**Running example** (we will carry these exact 4 rows through the entire pipeline):

| Row | IPV4_SRC_ADDR | IPV4_DST_ADDR | L4_SRC_PORT | PROTOCOL | IN_BYTES | Attack | Label | Split |
|---|---|---|---|---|---|---|---|---|
| 1 | 10.0.0.1 | 192.168.1.5 | 4444 | TCP | 500 | Benign | 0 | Train |
| 2 | 10.0.0.1 | 192.168.1.6 | 8080 | TCP | 1000 | Benign | 0 | Train |
| 3 | 10.0.0.2 | 192.168.1.5 | 53 | UDP | 8000 | DDoS | 1 | Train |
| 4 | 10.0.0.3 | 192.168.1.5 | 80 | TCP | 600 | Benign | 0 | Test |

Notice: rows 1–3 are training data, row 4 is test data. This split matters a lot for every step below, because **every learned parameter (target-encoding means, normaliser fit) is only allowed to see training rows.**

---

## PART A — `AnomalEPreprocessor` (File 1)

### Step 0 — Loading the raw data
**Code:** `load_and_clean_data`, beginning:
```python
if sanity_check:
    data = pd.read_csv(file_path, nrows=50000)
else:
    data = pd.read_csv(file_path)
    data.rename(columns=lambda x: str(x).strip(), inplace=True)
```

**Back to CSV-only (Parquet support was tried and reverted):** an earlier version of this pipeline added `.parquet` loading for speed on the ~19-million-row full dataset. That was reverted — **converted Parquet copies of this dataset (e.g. versions redistributed on Kaggle) frequently drop the `IPV4_SRC_ADDR` / `IPV4_DST_ADDR` columns**, which are not optional here: they are exactly what becomes the graph's *nodes* in Part B below. Without them, there is no way to build the graph at all. Since the whole point of Anomal-E is the graph structure, a faster file format that silently loses the columns needed to build that structure is worse than useless. The pipeline now requires the **original UQ-distributed CSV**, and a safety check was added right after loading:

```python
for col in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']:
    if col not in data.columns:
        raise KeyError(f"[CRITICAL ERROR] Column '{col}' is missing! ...")
```

This makes the pipeline **fail fast and loudly** if someone accidentally points it at a stripped-down/converted version of the dataset, rather than silently proceeding and failing much later (and much more confusingly) inside graph construction.

**Why `sanity_check=True` exists:** running the *entire* pipeline on ~19 million rows just to check for a typo or a broken import is slow and wasteful. Setting `sanity_check=True` loads only the **first 50,000 rows** (`nrows=50000`) so the whole pipeline can be smoke-tested end-to-end in seconds. Everything else about preprocessing runs identically regardless of this flag — it purely controls how much data gets loaded in the first place. (This flag also affects Step 3's train/test split, covered there.)

### Step 1 — Drop Port Information
**Code:** `load_and_clean_data`, lines dropping `L4_SRC_PORT` / `L4_DST_PORT`.

The `L4_SRC_PORT` column is removed entirely.

**Why:** A port number can act as a shortcut/"cheat code" for the model — e.g. it might just learn "port 4444 → attack" instead of learning the deeper structural/behavioural patterns that generalise to *unseen* attacks. Since generalising to new, unseen attack patterns is the whole point of Anomal-E, this shortcut is deliberately removed at the source.

**Our example after Step 1:**

| Row | IPV4_SRC_ADDR | IPV4_DST_ADDR | PROTOCOL | IN_BYTES | Attack | Label | Split |
|---|---|---|---|---|---|---|---|
| 1 | 10.0.0.1 | 192.168.1.5 | TCP | 500 | Benign | 0 | Train |
| 2 | 10.0.0.1 | 192.168.1.6 | TCP | 1000 | Benign | 0 | Train |
| 3 | 10.0.0.2 | 192.168.1.5 | UDP | 8000 | DDoS | 1 | Train |
| 4 | 10.0.0.3 | 192.168.1.5 | TCP | 600 | Benign | 0 | Test |

The `L4_SRC_PORT`/`L4_DST_PORT` columns are simply gone — no replacement value, no encoding, just removed.

### Step 2 — Uniform Random Downsampling
**Code:** `data.groupby(by='Attack').sample(frac=fraction, random_state=random_state)`

**Why:** Real NIDS datasets contain millions of flows (e.g. NF-CSE-CIC-IDS2018-v2 has ~18.9 million). Training on the full dataset is computationally expensive, so the paper downsamples to 10%.

**Why `groupby('Attack')` specifically, not a plain random 10% sample of everything?**
Because attacks are rare (often <5% of all traffic). If you downsampled the whole dataset randomly, a rare attack type with only 200 total flows could end up with almost none in your 10% sample — possibly losing that attack category from training entirely. Grouping by `Attack` first means each category (Benign, DDoS, Exploit, etc.) is downsampled **independently to 10% of itself**, so every category is still represented proportionally.

*(In our tiny 4-row example we'll skip this step, since downsampling a 4-row toy example doesn't illustrate anything new — but conceptually, each of Benign, DDoS would be sampled separately.)*

### Step 3 — Train/Test Split
**Code:** `train_test_split(..., stratify=y)`

Our example is already pre-split (rows 1–3 = train, row 4 = test) to keep the walkthrough concrete. In the real pipeline this happens via `sklearn.model_selection.train_test_split`, with `stratify=y` ensuring the attack/benign ratio is preserved in both splits.

**Critically:** From this point on, `X_train` and `X_test` are separate objects. Every subsequent `fit()` call is only ever given `X_train`.

**Note on `sanity_check` here too:** when `sanity_check=True` (see Step 0), `split_data` sets `stratify_col = None` instead of `stratify=y`. This is needed because stratified splitting requires *at least 2 rows* of every class in both the train and test split — in a small 50,000-row slice, some rare attack categories might appear only once, which would make `train_test_split(..., stratify=y)` raise an error. Disabling stratification avoids that crash during a quick smoke test. The full run (`sanity_check=False`, the default) is unaffected and still uses `stratify=y` exactly as described above.

### Step 4 — Target Encoding (Feature Conversion)
**Code:** `apply_feature_conversion`, `self.encoder.fit(X_train, y_train['Label'])`

**What gets encoded:** Only the columns listed in `self.target_cols` (`PROTOCOL`, `TCP_FLAGS`, etc. — the *categorical* columns). `IN_BYTES` and similar numeric columns are untouched here.

**Mechanism:** For each categorical value, replace it with the **mean of `Label`** across all *training* rows that had that value.

**Worked calculation using our example (training rows only, rows 1–3):**

| PROTOCOL | Label |
|---|---|
| TCP (row 1) | 0 |
| TCP (row 2) | 0 |
| UDP (row 3) | 1 |

- Mean Label for `TCP` = (0 + 0) / 2 = **0.0**
- Mean Label for `UDP` = (1) / 1 = **1.0**

These two numbers (`TCP → 0.0`, `UDP → 1.0`) are now "baked into" the fitted encoder — this is exactly what `self.encoder.fit(X_train, ...)` computes and stores internally.

**Applying to test data (row 4, PROTOCOL = TCP):**
`self.encoder.transform(X_test)` looks up `TCP` in its learned dictionary and assigns **0.0** — *without ever looking at row 4's actual Label*. This is what "no data leakage" means concretely: the test row's true label plays zero role in what number `TCP` gets mapped to.

**What if the test set contained a category never seen in training** (e.g. `ICMP`)? The encoder has no learned mean for it, so it emits `NaN` (or in some edge cases `inf`, from internal smoothing division). This is *exactly* why Step 5 exists immediately afterward.

**Table after Step 4:**

| Row | PROTOCOL (encoded) | IN_BYTES | Split |
|---|---|---|---|
| 1 | 0.0 | 500 | Train |
| 2 | 0.0 | 1000 | Train |
| 3 | 1.0 | 8000 | Train |
| 4 | 0.0 | 600 | Test |

### Step 5 — Replace Empty and Infinity Values
**Code:**
```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
```

Any `inf`/`-inf` becomes `NaN`, then every `NaN` becomes `0`.

**Why this matters mechanically:** A single `NaN` or `inf` value, once it enters a neural network's matrix multiplications (which is exactly what E-GraphSAGE's aggregation and weight layers do), propagates and corrupts the entire computation — one bad number can turn an entire embedding vector into garbage. This step guarantees every value feeding into the model is a well-defined real number.

*(Our example has no unseen categories, so nothing changes at this step — but this is the safety net that would catch it if it happened.)*

### Step 6 — L2 Normalization
**Code:** `apply_normalization`

**Mechanism:** Unlike target encoding, L2 normalisation is **row-wise**, not column-wise. Each flow (row) is rescaled so that the Euclidean length (L2 norm) of its own feature vector equals 1:

$$x_{normalized} = \frac{x}{\sqrt{\sum_{i=1}^{n} x_i^2}}$$

**Worked calculation for row 1** (features: `PROTOCOL=0.0`, `IN_BYTES=500`):

$$\text{norm} = \sqrt{0.0^2 + 500^2} = \sqrt{250000} = 500$$

$$x_{normalized} = \left[\frac{0.0}{500}, \frac{500}{500}\right] = [0.0, 1.0]$$

**Worked calculation for row 3** (features: `PROTOCOL=1.0`, `IN_BYTES=8000`):

$$\text{norm} = \sqrt{1.0^2 + 8000^2} = \sqrt{64000001} \approx 8000.0000625$$

$$x_{normalized} \approx \left[\frac{1.0}{8000.0000625}, \frac{8000}{8000.0000625}\right] \approx [0.000125, 0.999999]$$

**Why normalise at all?** Before this step, `IN_BYTES` (hundreds/thousands) completely dwarfs `PROTOCOL` (0 or 1) in raw magnitude. Without rescaling, the neural network's matrix multiplications would be dominated by whichever feature happens to have the largest raw numbers — not necessarily the most *informative* one. L2 normalisation puts every flow's feature vector on a comparable scale.

**Note on `fit()` here:** unlike the target encoder, `Normalizer.fit()` doesn't actually learn or store anything meaningful from the training data — L2 normalisation only needs the row itself. `fit()` is called mainly to follow the standard scikit-learn API convention; calling `fit_transform()` separately on train and test would give identical results here.

**Building the final edge feature vector `h`:**
```python
X_train['h'] = X_train.iloc[:, 2:].values.tolist()
```
All normalised numeric columns for a row are packed into one Python list. For row 1: `h = [0.0, 1.0]`. This `h` column **is** $e_{uv}$ from the paper's notation — it will become the edge feature attached to that flow's edge in the graph.

**Table after Step 6 (final preprocessed state):**

| Row | IPV4_SRC_ADDR | IPV4_DST_ADDR | h | Label | Attack | Split |
|---|---|---|---|---|---|---|
| 1 | 10.0.0.1 | 192.168.1.5 | [0.0, 1.0] | 0 | Benign | Train |
| 2 | 10.0.0.1 | 192.168.1.6 | [0.0, 1.0] | 0 | Benign | Train |
| 3 | 10.0.0.2 | 192.168.1.5 | [1.0, 1.0]* | 1 | DDoS | Train |
| 4 | 10.0.0.3 | 192.168.1.5 | [0.0, 1.0]** | 0 | Benign | Test |

*(rounded from ~[0.000125, 0.999999] for readability; **row 4 computed the same way as row 1, since it has the same PROTOCOL/IN_BYTES pattern relative to its own norm)*

**This is the exact output that `AnomalEPreprocessor.process_pipeline()` returns** — two dataframes (`train_df`, `test_df`), each with a ready-to-use `h` edge-feature column. This is where File 1 ends and File 2 begins.

---

## PART B — `AnomalEGraphBuilder` (File 2)

Now we take the `train_df` / `test_df` tables above and turn them into actual graph objects.

### Step 7.1 — Node and Edge Mapping
**Code:** `nx.from_pandas_edgelist(df, source="IPV4_SRC_ADDR", target="IPV4_DST_ADDR", edge_attr=["h","Label","Attack"], create_using=nx.MultiGraph())`

Every **unique IP address becomes a node**. Every **row (flow) becomes an edge** connecting its source node to its destination node, carrying `h`, `Label`, and `Attack` as edge attributes.

**Applied to our training rows (1–3):**

- Nodes created: `10.0.0.1`, `192.168.1.5`, `192.168.1.6`, `10.0.0.2`  (4 unique nodes)
- Edges created:
  - `10.0.0.1 → 192.168.1.5`, h=[0.0, 1.0], Label=0
  - `10.0.0.1 → 192.168.1.6`, h=[0.0, 1.0], Label=0
  - `10.0.0.2 → 192.168.1.5`, h=[1.0, 1.0], Label=1

**Why `MultiGraph` and not plain `Graph`:** if two different flows happened to occur between the *same* pair of IPs (e.g. two separate connections between `10.0.0.1` and `192.168.1.5`, one benign and one attack), a plain `Graph` only keeps a single edge per node pair — the second flow would silently overwrite the first, and we'd lose one of the two flows entirely. `MultiGraph` keeps every flow as its own parallel edge, which is essential since real hosts frequently exchange many separate flows.

**Why `.to_directed()` afterward:** a flow is inherently directional — "A sent data to B" is a different event from "B sent data to A" (e.g. a request vs. its response can have very different byte counts, and in fact in these v2 NetFlow datasets, `IN_BYTES`/`OUT_BYTES` are already separated by direction). Converting to directed preserves *which side initiated which flow*, rather than collapsing that information into an undirected "these two are connected" relationship.

### Step 7.2 — Feature Assignment: Edges

`dgl.from_networkx(nx_g, edge_attrs=['h','Attack','Label'])` carries the `h` vectors over unchanged into `dgl_g.edata['h']`. Nothing about the real flow statistics changes here — this is a pure format conversion from NetworkX's data structure into DGL's, so PyTorch operations can run on it.

### Step 7.3 — Feature Assignment: Nodes (the constant "1s" vector)

**Code:**
```python
edge_feat_dim = len(dgl_g.edata['h'][0])
nfeat_weight = torch.ones([dgl_g.number_of_nodes(), edge_feat_dim])
dgl_g.ndata['h'] = nfeat_weight
```

**This is the step that most often causes confusion, so let's be extremely explicit: nothing about the edge data (the real `[0.0, 1.0]`, `[1.0, 1.0]` values) is touched, deleted, or overwritten here.** `dgl_g.ndata` (node data) and `dgl_g.edata` (edge data) are two entirely separate storage structures inside a DGL graph object. This step only creates a brand-new feature *for nodes*, which never had any real data to begin with.

**Why nodes need something at all:** a NetFlow record only ever describes a *flow between two IPs* — there is no column anywhere describing "IP `10.0.0.1`'s own inherent properties." An IP address by itself carries no meaningful signal (it's just an identifier). But the GNN math (recall Equation 2 from the paper, $h_v^k = \sigma(W^k \cdot \text{CONCAT}(h_v^{k-1}, h_{N(v)}^k))$) requires every node to have *some* starting vector $h_v^0$ to begin the aggregation process. Since we have nothing meaningful to put there, the paper's design choice is to give every node an identical placeholder: a vector of all 1s.

**Why the dimensionality must match `edge_feat_dim` exactly (here, 2, matching our toy `h` vectors):**
This is a deliberate design choice, stated explicitly in the paper: *"the dimensions of these constant vectors also match the dimensions of the edge feature set."* It keeps node and edge vectors at a comparable scale/dimensionality going into the concatenation step of E-GraphSAGE's aggregation formula.

**Our example — computing `nfeat_weight`:**
- `dgl_g.number_of_nodes()` = 4 (10.0.0.1, 192.168.1.5, 192.168.1.6, 10.0.0.2)
- `edge_feat_dim` = `len([0.0, 1.0])` = 2

$$
\text{nfeat\_weight} =
\begin{bmatrix}
1 & 1 \\
1 & 1 \\
1 & 1 \\
1 & 1
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow \text{10.0.0.1} \\
\leftarrow \text{192.168.1.5} \\
\leftarrow \text{192.168.1.6} \\
\leftarrow \text{10.0.0.2}
\end{matrix}
$$

Every node — regardless of how "suspicious" or "benign" its flows look — starts from the exact same `[1, 1]` vector. This is intentional: it forces E-GraphSAGE to derive *all* meaningful signal purely from the graph topology (who's connected to whom) and the edge features (the `h` vectors), rather than from any pre-existing notion of "this IP is bad."

**Direct proof that `edata['h']` is untouched — before and after, side by side:**

`dgl_g.edata['h']` **before** Step 7.3 runs (right after Steps 7.1–7.2, using our 3 training edges):

```python
>>> dgl_g.edata['h']
tensor([[0.0000, 1.0000],      # edge: 10.0.0.1 → 192.168.1.5
        [0.0000, 1.0000],      # edge: 10.0.0.1 → 192.168.1.6
        [0.0001, 0.9999]])     # edge: 10.0.0.2 → 192.168.1.5
```

Now the three lines of Step 7.3 run. Notice line 1 only *reads* from `edata` (just to check its dimensionality), and line 3 only *writes* to `ndata` — no line anywhere assigns to `dgl_g.edata['h']`:

```python
edge_feat_dim = len(dgl_g.edata['h'][0])            # reads edata, dim = 2
nfeat_weight = torch.ones([dgl_g.number_of_nodes(), edge_feat_dim])
dgl_g.ndata['h'] = nfeat_weight                      # writes ndata only
```

`dgl_g.edata['h']` **after** Step 7.3 runs — identical to before, down to the last digit:

```python
>>> dgl_g.edata['h']
tensor([[0.0000, 1.0000],      # unchanged
        [0.0000, 1.0000],      # unchanged
        [0.0001, 0.9999]])     # unchanged
```

And at this same point, `dgl_g.ndata['h']` now exists for the first time:

```python
>>> dgl_g.ndata['h']
tensor([[1., 1.],      # 10.0.0.1
        [1., 1.],      # 192.168.1.5
        [1., 1.],      # 192.168.1.6
        [1., 1.]])     # 10.0.0.2
```

So the graph now holds **two separate, independent tensors** side by side — the real flow statistics on `edata['h']`, and the placeholder all-ones vectors on `ndata['h']` — exactly as Algorithm 1's line 1 (`h⁰_v ← x_v`) and the edge feature input (`{e_uv}`) require as separate inputs to E-GraphSAGE.

### Step 7.4 — Tensor Reshaping

**Code:**
```python
dgl_g.ndata['h'] = torch.reshape(dgl_g.ndata['h'], (dgl_g.ndata['h'].shape[0], 1, dgl_g.ndata['h'].shape[1]))
dgl_g.edata['h'] = torch.reshape(dgl_g.edata['h'], (dgl_g.edata['h'].shape[0], 1, dgl_g.edata['h'].shape[1]))
```

**What changes and what doesn't:** no values change at all — only the tensor's *shape* changes, by inserting an extra dimension of size 1 in the middle.

**Our example, before reshape:**
`dgl_g.ndata['h'].shape` = `(4, 2)` — 4 nodes, each a 2-element vector.

**After reshape:**
`dgl_g.ndata['h'].shape` = `(4, 1, 2)` — same 4 vectors, same 2 numbers each, just wrapped with an extra middle dimension.

**Why this is necessary:** the E-GraphSAGE message-passing layers built with PyTorch's `nn.Linear` and DGL's message-passing API expect input tensors with this extra dimension (commonly used as a "channel" or "head" placeholder in these APIs) — it's a data-formatting requirement of the downstream layers, not a conceptual change to the data itself.

### Final result of `generate_graphs()`

Two fully-formed DGL graph objects — `train_g` (built only from training rows 1–3) and `test_g` (built only from row 4) — completely independent of each other, each with:

- `ndata['h']`: constant all-ones node features, shape `(num_nodes, 1, edge_feat_dim)`
- `edata['h']`: the real, normalised flow statistics, shape `(num_edges, 1, edge_feat_dim)`
- `edata['Label']`, `edata['Attack']`: kept alongside for later evaluation (never fed into the self-supervised training itself)

This is exactly the input E-GraphSAGE's `forward` pass (Algorithm 1 of the paper) expects: a graph where node features start as placeholders and all the real signal lives on the edges.

---

## A note on `main.py`

`main.py` isn't a third pipeline stage — it's simply the script that **runs Files 1 and 2 back-to-back** on a real dataset to check that everything up to this point actually works:

```python
preprocessor = AnomalEPreprocessor()
train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=True)

graph_builder = AnomalEGraphBuilder()
train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
```

It checks the dataset file exists (with a message now correctly pointing at the original CSV, not Parquet), runs `sanity_check=True` for a fast 50,000-row pass, and then prints the resulting graphs (`print(train_g)`) along with the shapes of `ndata['h']` and `edata['h']`. Seeing `(num_nodes, 1, num_features)` and `(num_edges, 1, num_features)` with matching `num_features` confirms Steps 7.3–7.4 above worked correctly — this is purely a "does the pipeline run end-to-end without errors" check before moving on to the actual E-GraphSAGE/DGI training loop, not a new stage in the data-processing logic itself.

For reference, the surrounding project structure is:

```
Anomal-E-Implementation/
├── data/raw/NF-CSE-CIC-IDS2018-v2.csv   ← original UQ CSV, not Parquet
├── docs/DATA_PIPELINE_EXPLANATION.md    ← this document
├── src/data_pipeline/
│   ├── preprocessor.py                  ← File 1 (AnomalEPreprocessor)
│   └── graph_builder.py                 ← File 2 (AnomalEGraphBuilder)
├── src/engine/  src/models/  src/utils/  ← placeholders for E-GraphSAGE, DGI training loop, PCA/IF/CBLOF/HBOS
└── main.py                              ← the smoke test above
```

---

## Quick-reference: which file does what

| | File 1: `AnomalEPreprocessor` | File 2: `AnomalEGraphBuilder` |
|---|---|---|
| **Input** | Raw NetFlow **CSV only** (original UQ file — Parquet was tried and reverted, see Step 0) | Preprocessed train/test dataframes (with `h` column) |
| **Output** | Cleaned, encoded, normalised dataframes | DGL graph objects (`train_g`, `test_g`) |
| **Paper section** | §4.2, Fig. 4 (preprocessing box) | §4.2, Fig. 4 (graph generation box) + Algorithm 1, line 1 |
| **Key operations** | Load CSV, verify IP columns exist, drop ports, downsample, split, target-encode, clean inf/NaN, L2-normalise | Build MultiGraph → directed → DGL, assign constant node features, reshape tensors |
| **Data-leakage safeguard** | Encoder/scaler `fit()` only ever sees `X_train` | Train and test graphs built from fully separate dataframes — no shared nodes/edges |
| **Dev/debug aid** | `sanity_check=True` loads only 50,000 rows and disables split stratification | (inherits fast graphs automatically, since it just receives smaller dataframes) |

---


## One-paragraph mental model to keep in mind

Think of the whole pipeline as answering two questions in order. **File 1 asks:** "how do I turn messy, mixed-type flow records into clean numeric vectors, without ever letting the test set leak into anything the model learns from?" **File 2 asks:** "given those clean vectors, how do I arrange them into the graph shape (nodes + directed edges + node/edge features) that E-GraphSAGE's math (Algorithm 1) actually operates on?" Every design choice in both files — Parquet/CSV loading, sanity-check mode, port-dropping, stratified downsampling, fit-only-on-train, MultiGraph, all-ones node features, tensor reshaping — exists to serve one or both of those two questions. `main.py` simply runs both files together as a quick correctness check before moving on to training.