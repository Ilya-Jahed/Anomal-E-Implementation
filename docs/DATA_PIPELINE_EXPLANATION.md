# Anomal-E: Data Preprocessing & Graph Construction Pipeline
### A complete, worked-example walkthrough of `preprocessor.py` and `graph_builder.py`

This document explains, in full detail and with one running numeric example carried through every step, how raw NetFlow records become a PyTorch/DGL graph ready to be fed into E-GraphSAGE. It corresponds to two files:

- **File 1 — `AnomalEPreprocessor`** (`src/data_pipeline/preprocessor.py`): raw CSV → cleaned, encoded, normalised train/test dataframes
- **File 2 — `AnomalEGraphBuilder`** (`src/data_pipeline/graph_builder.py`): cleaned dataframes → DGL graph objects

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

### Step 0 — Loading the raw data (chunked, memory-safe)

**Code:** `load_and_clean_data`

```python
if sanity_check:
    data = pd.read_csv(file_path, nrows=50000)
    data.rename(columns=lambda x: str(x).strip(), inplace=True)
else:
    reader = pd.read_csv(file_path, chunksize=chunksize)
    for i, chunk in enumerate(reader):
        chunk.rename(columns=lambda x: str(x).strip(), inplace=True)
        chunk_sample = chunk.groupby(by='Attack', group_keys=False).sample(
            frac=fraction, random_state=random_state
        )
        sampled_chunks.append(chunk_sample)
    data = pd.concat(sampled_chunks, ignore_index=True)
```

**Why chunked reading, not `pd.read_csv(file_path)` in one shot:** the full NF-CSE-CIC-IDS2018-v2 CSV has ~19 million rows. Loading it whole into RAM *before* any downsampling happens is what was exhausting memory on Colab and similar-RAM machines. Instead, the file is read `chunksize` rows (default 500,000) at a time, and **each chunk is immediately downsampled** (stratified by `Attack`, see Step 2 below) before the next chunk is read. At any point in time, only one raw chunk plus the running list of already-sampled rows are held in memory — never the full ~19M-row file at once.

**Why this still gives a representative sample:** sampling `fraction` from each chunk independently, then concatenating, approximates sampling `fraction` from the whole file at once — as long as attack types are reasonably spread out across the file rather than concentrated in one contiguous block. This holds for the standard NF-CSE-CIC-IDS2018-v2 release (rows are not attack-sorted), so the resulting class balance closely matches what a single whole-file stratified sample would give, at a fraction of the peak memory usage. A group with very few rows in a single chunk can occasionally sample 0 rows from that chunk — this is fine, since the same attack type reappears in later chunks.

**Only CSV is supported, deliberately:** converted Parquet copies of this dataset (e.g. versions redistributed on Kaggle) frequently drop the `IPV4_SRC_ADDR` / `IPV4_DST_ADDR` columns, which are not optional — they become the graph's *nodes* in Part B. A safety check right after loading enforces this:

```python
for col in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']:
    if col not in data.columns:
        raise KeyError(f"[CRITICAL ERROR] Column '{col}' is missing! ...")
```

This makes the pipeline fail fast and loudly if someone points it at a stripped-down dataset, instead of failing later and more confusingly inside graph construction.

**Why `sanity_check=True` exists:** running the entire pipeline on ~19 million rows just to check for a typo or a broken import is slow and wasteful. `sanity_check=True` loads only the first 50,000 rows (`nrows=50000`, no chunking needed at that size) so the whole pipeline can be smoke-tested end-to-end in seconds. Everything else about preprocessing runs identically regardless of this flag. (This flag also affects Step 3's train/test split, covered there.)

### Step 1 — Drop Port Information

**Code:** `load_and_clean_data`, after the IP columns are cast to string:

```python
if "L4_SRC_PORT" in data.columns and "L4_DST_PORT" in data.columns:
    data.drop(columns=["L4_SRC_PORT", "L4_DST_PORT"], inplace=True)
```

**Why:** a port number can act as a shortcut/"cheat code" for the model — e.g. it might just learn "port 4444 → attack" instead of learning the deeper structural/behavioural patterns that generalise to *unseen* attacks. Since generalising to new, unseen attack patterns is the whole point of Anomal-E, this shortcut is deliberately removed at the source.

**Our example after Step 1:**

| Row | IPV4_SRC_ADDR | IPV4_DST_ADDR | PROTOCOL | IN_BYTES | Attack | Label | Split |
|---|---|---|---|---|---|---|---|
| 1 | 10.0.0.1 | 192.168.1.5 | TCP | 500 | Benign | 0 | Train |
| 2 | 10.0.0.1 | 192.168.1.6 | TCP | 1000 | Benign | 0 | Train |
| 3 | 10.0.0.2 | 192.168.1.5 | UDP | 8000 | DDoS | 1 | Train |
| 4 | 10.0.0.3 | 192.168.1.5 | TCP | 600 | Benign | 0 | Test |

The `L4_SRC_PORT`/`L4_DST_PORT` columns are simply gone — no replacement value, no encoding, just removed.

### Step 2 — Stratified Downsampling

**Code (non-sanity-check path):** downsampling happens **per chunk**, inside the loop in Step 0:

```python
chunk_sample = chunk.groupby(by='Attack', group_keys=False).sample(
    frac=fraction, random_state=random_state
)
```

**Code (sanity-check path):** downsampling happens **once**, after the whole 50,000-row slice is loaded:

```python
if sanity_check:
    data = data.groupby(by='Attack').sample(frac=fraction, random_state=random_state)
```

**Why these two paths exist separately:** the non-sanity-check path already downsampled every chunk as it was read (Step 0). Downsampling *again* here would double-downsample and shrink the final dataset far below the requested `fraction`. So the `if sanity_check:` block at the end of `load_and_clean_data` only runs the group-and-sample step for the sanity-check path, where no per-chunk sampling happened yet.

**Why `groupby('Attack')` specifically, not a plain random sample of everything:** attacks are rare (often <5% of all traffic). A plain random sample could wipe out a rare attack category almost entirely. Grouping by `Attack` first means each category (Benign, DDoS, Exploit, etc.) is downsampled independently, so every category stays represented proportionally.

*(In our tiny 4-row example we'll skip this step, since downsampling a 4-row toy example doesn't illustrate anything new — but conceptually, each of Benign, DDoS would be sampled separately, whether that happens per-chunk or all at once.)*

### Step 3 — Train/Test Split

**Code:** `split_data`, using `sklearn.model_selection.train_test_split(..., stratify=stratify_col)`

Our example is already pre-split (rows 1–3 = train, row 4 = test) to keep the walkthrough concrete. In the real pipeline, `stratify=y` ensures the attack/benign ratio is preserved in both splits.

**Critically:** from this point on, `X_train` and `X_test` are separate objects. Every subsequent `fit()` call is only ever given `X_train`.

**Note on `sanity_check` here too:** when `sanity_check=True`, `split_data` sets `stratify_col = None` instead of `stratify=y`. This is needed because stratified splitting requires at least 2 rows of every class in both the train and test split — in a small 50,000-row slice, some rare attack categories might appear only once, which would make a stratified split raise an error. The full run (`sanity_check=False`, the default) is unaffected and always uses `stratify=y`.

### Step 4 — Target Encoding (Feature Conversion)

**Code:** `apply_feature_conversion`, `self.encoder.fit(X_train, y_train['Label'])`

**What gets encoded:** only the columns listed in `self.target_cols` (`PROTOCOL`, `TCP_FLAGS`, etc. — the categorical columns). `IN_BYTES` and similar numeric columns are untouched here.

**Mechanism:** for each categorical value, replace it with the mean of `Label` across all training rows that had that value.

**Worked calculation using our example (training rows only, rows 1–3):**

| PROTOCOL | Label |
|---|---|
| TCP (row 1) | 0 |
| TCP (row 2) | 0 |
| UDP (row 3) | 1 |

- Mean Label for `TCP` = (0 + 0) / 2 = **0.0**
- Mean Label for `UDP` = (1) / 1 = **1.0**

These two numbers (`TCP → 0.0`, `UDP → 1.0`) are now "baked into" the fitted encoder — this is exactly what `self.encoder.fit(X_train, y_train['Label'])` computes and stores internally.

**Applying to test data (row 4, PROTOCOL = TCP):**
`self.encoder.transform(X_test)` looks up `TCP` in its learned dictionary and assigns **0.0** — without ever looking at row 4's actual Label. This is what "no data leakage" means concretely: the test row's true label plays zero role in what number `TCP` gets mapped to.

**What if the test set contained a category never seen in training** (e.g. `ICMP`)? The encoder has no learned mean for it, so it emits `NaN` (or in some edge cases `inf`, from internal smoothing division). This is exactly why Step 5 exists immediately afterward.

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
for df in [X_train_enc, X_test_enc]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
```

Any `inf`/`-inf` becomes `NaN`, then every `NaN` becomes `0`.

**Why this matters mechanically:** a single `NaN` or `inf` value, once it enters a neural network's matrix multiplications (which is exactly what E-GraphSAGE's aggregation and weight layers do), propagates and corrupts the entire computation — one bad number can turn an entire embedding vector into garbage. This step guarantees every value feeding into the model is a well-defined real number.

*(Our example has no unseen categories, so nothing changes at this step — but this is the safety net that would catch it if it happened.)*

### Step 6 — L2 Normalization

**Code:** `apply_normalization`

**Mechanism:** unlike target encoding, L2 normalisation is row-wise, not column-wise. Each flow (row) is rescaled so that the Euclidean length (L2 norm) of its own feature vector equals 1:

$$x_{normalized} = \frac{x}{\sqrt{\sum_{i=1}^{n} x_i^2}}$$

**Worked calculation for row 1** (features: `PROTOCOL=0.0`, `IN_BYTES=500`):

$$\text{norm} = \sqrt{0.0^2 + 500^2} = \sqrt{250000} = 500$$

$$x_{normalized} = \left[\frac{0.0}{500}, \frac{500}{500}\right] = [0.0, 1.0]$$

**Worked calculation for row 3** (features: `PROTOCOL=1.0`, `IN_BYTES=8000`):

$$\text{norm} = \sqrt{1.0^2 + 8000^2} = \sqrt{64000001} \approx 8000.0000625$$

$$x_{normalized} \approx \left[\frac{1.0}{8000.0000625}, \frac{8000}{8000.0000625}\right] \approx [0.000125, 0.999999]$$

**Why normalise at all?** Before this step, `IN_BYTES` (hundreds/thousands) completely dwarfs `PROTOCOL` (0 or 1) in raw magnitude. Without rescaling, the neural network's matrix multiplications would be dominated by whichever feature happens to have the largest raw numbers — not necessarily the most informative one. L2 normalisation puts every flow's feature vector on a comparable scale.

**Note on `fit()` here:** unlike the target encoder, `Normalizer.fit()` doesn't actually learn or store anything meaningful from the training data — L2 normalisation only needs the row itself. `fit()` is called mainly to follow the standard scikit-learn API convention; calling `fit_transform()` separately on train and test gives an identical result here, since each row is normalised independently of every other row.

**Building the final edge feature vector `h` (memory-optimised):**

```python
train_values = X_train.iloc[:, 2:].values.astype(np.float32)
test_values = X_test.iloc[:, 2:].values.astype(np.float32)
X_train['h'] = list(train_values)
X_test['h'] = list(test_values)
```

**Why `.astype(np.float32)` + `list(train_values)`, instead of the more obvious `X_train.iloc[:, 2:].values.tolist()`:** the naive `.tolist()` approach does two expensive things: (1) builds a float64 NumPy array, then (2) converts *every single element* of it into an individual Python `float` object. Step (2) is what's expensive — a NumPy float64 is 8 raw bytes, but a Python float object carries ~24 bytes of interpreter overhead each, so a matrix that's already large as a NumPy array can end up 3-4x larger once turned into nested Python lists, while briefly holding both representations in memory at once. This is what was exhausting RAM even at moderate `fraction` values. The fix keeps values as float32 (half the memory of float64, still ample precision for L2-normalised flow features) and stores one NumPy row-array per cell instead of a Python list per cell — DGL/PyTorch accept NumPy arrays exactly as readily as Python lists when building the edge feature tensor downstream, so nothing else needs to change.

All normalised numeric columns for a row are packed into one array. For row 1: `h = [0.0, 1.0]`. This `h` column **is** $e_{uv}$ from the paper's notation — it will become the edge feature attached to that flow's edge in the graph.

**Table after Step 6 (final preprocessed state):**

| Row | IPV4_SRC_ADDR | IPV4_DST_ADDR | h | Label | Attack | Split |
|---|---|---|---|---|---|---|
| 1 | 10.0.0.1 | 192.168.1.5 | [0.0, 1.0] | 0 | Benign | Train |
| 2 | 10.0.0.1 | 192.168.1.6 | [0.0, 1.0] | 0 | Benign | Train |
| 3 | 10.0.0.2 | 192.168.1.5 | [1.0, 1.0]* | 1 | DDoS | Train |
| 4 | 10.0.0.3 | 192.168.1.5 | [0.0, 1.0]** | 0 | Benign | Test |

*(rounded from ~[0.000125, 0.999999] for readability; **row 4 computed the same way as row 1, since it has the same PROTOCOL/IN_BYTES pattern relative to its own norm)*

**This is the exact output that `AnomalEPreprocessor.process_pipeline()` returns** — two dataframes (`train_df`, `test_df`), each with a ready-to-use `h` edge-feature column. This is where File 1 ends and File 2 begins.

### Step 6.5 — Encoding Attack labels to integers

**Code:** `encode_labels`

```python
self.label_encoder.fit(pd.concat([train_df["Attack"], test_df["Attack"]]))
train_df["Attack"] = self.label_encoder.transform(train_df["Attack"])
test_df["Attack"] = self.label_encoder.transform(test_df["Attack"])
```

This maps text labels (`"Benign"`, `"DDoS"`, ...) to plain integers. Unlike the target encoder above, this is fit on the **combined** train+test labels, not train-only — but this does not introduce data leakage, because it's purely a text-to-integer lookup table for the final evaluation label, not a statistic derived from `Label` that feeds into the model's learned representation.

---

## PART B — `AnomalEGraphBuilder` (File 2)

Now we take the `train_df` / `test_df` tables above and turn them into actual graph objects.

**Why this file no longer uses NetworkX at all:** an earlier version built the graph via `nx.from_pandas_edgelist(...)` → `.to_directed()` → `dgl.from_networkx(...)`. NetworkX represents every single edge as its own Python dictionary object, and `.to_directed()` copies the entire graph while doubling the edge count — for a NetFlow dataset with hundreds of thousands to millions of rows, this was both extremely slow and memory-hungry, exhausting available RAM even after the chunked-loading fix in Part A. The current version builds the *same* graph structure directly with `pandas.factorize` (for IP → integer node-ID mapping) and NumPy/PyTorch tensor operations (for edge duplication and feature stacking) — all vectorised, array-based operations instead of a Python object per edge. The resulting graph is structurally identical to what the NetworkX version produced (same nodes, same directed edges, same edge/node feature values and shapes) — only the construction path changed.

### Step 7.1 — Mapping IP addresses to integer node IDs

**Code:** `_build_single_graph`, Step 1:

```python
all_ips = pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]], ignore_index=True)
node_ids, unique_ips = pd.factorize(all_ips)
num_nodes = len(unique_ips)
num_rows = len(df)

src_ids = node_ids[:num_rows]
dst_ids = node_ids[num_rows:]
```

`pd.factorize` assigns every distinct string in `all_ips` a small integer, in the order it first encounters them, and does this in one vectorised pass — no per-row Python object creation, unlike NetworkX's node/edge dictionaries.

**Applied to our training rows (1–3):** `all_ips` is built by first stacking all `IPV4_SRC_ADDR` values, then all `IPV4_DST_ADDR` values:

```
index 0: 10.0.0.1       (row 1 src)
index 1: 10.0.0.1       (row 2 src)
index 2: 10.0.0.2       (row 3 src)
index 3: 192.168.1.5    (row 1 dst)
index 4: 192.168.1.6    (row 2 dst)
index 5: 192.168.1.5    (row 3 dst)
```

`pd.factorize` walks this list top to bottom and assigns each *new* string the next integer, reusing the same integer for repeats:

| unique_ips index | IP |
|---|---|
| 0 | 10.0.0.1 |
| 1 | 10.0.0.2 |
| 2 | 192.168.1.5 |
| 3 | 192.168.1.6 |

`node_ids = [0, 0, 1, 2, 3, 2]` — one integer per entry in `all_ips`, in the same order.

Since `num_rows = 3`, the first 3 entries of `node_ids` are the source IDs and the last 3 are the destination IDs:

```
src_ids = [0, 0, 1]   # rows 1,2,3's IPV4_SRC_ADDR
dst_ids = [2, 3, 2]   # rows 1,2,3's IPV4_DST_ADDR
```

So: row 1 is node 0 → node 2, row 2 is node 0 → node 3, row 3 is node 1 → node 2 — matching `10.0.0.1→192.168.1.5`, `10.0.0.1→192.168.1.6`, `10.0.0.2→192.168.1.5` exactly.

**Train/test independence:** `_build_single_graph` is called once for `train_df` and once for `test_df` (see `generate_graphs` below), so `pd.factorize` runs separately for each. An IP address that appears in *both* train and test data gets an **unrelated** node ID in each graph — node 0 in `train_g` and node 0 in `test_g` are not "the same" node in any meaningful sense, they just happen to share a number. This is consistent with the paper's strict train/test separation to avoid data leakage.

### Step 7.2 — Building directed edges in both directions

**Code:** `_build_single_graph`, Step 2:

```python
src_all = np.concatenate([src_ids, dst_ids])
dst_all = np.concatenate([dst_ids, src_ids])
```

This reproduces exactly what the previous `MultiGraph(undirected) → .to_directed()` pipeline did: an undirected edge `{u, v}` becomes two directed edges `u→v` and `v→u`. Here, that's done directly with array concatenation instead of relying on NetworkX's implicit conversion — `src_all`/`dst_all` first contain the original `src_ids→dst_ids` direction, then the same pairs reversed.

**Continuing our example**, concatenating the arrays from Step 7.1:

```
src_all = [0, 0, 1,  2, 3, 2]
dst_all = [2, 3, 2,  0, 0, 1]
          ^forward^  ^reverse^
```

Written out as IP pairs, this gives **6 directed edges** from the 3 original flows:

| # | Edge | Direction |
|---|---|---|
| 1 | 10.0.0.1 → 192.168.1.5 | forward (real flow, row 1) |
| 2 | 10.0.0.1 → 192.168.1.6 | forward (real flow, row 2) |
| 3 | 10.0.0.2 → 192.168.1.5 | forward (real flow, row 3) |
| 4 | 192.168.1.5 → 10.0.0.1 | **reverse (manufactured, copy of row 1)** |
| 5 | 192.168.1.6 → 10.0.0.1 | **reverse (manufactured, copy of row 2)** |
| 6 | 192.168.1.5 → 10.0.0.2 | **reverse (manufactured, copy of row 3)** |

Edges 4–6 did not exist as independent flows in the raw dataset — they are generated purely by this step, carrying the *same* `h`/`Label`/`Attack` values as their forward counterpart (see Step 7.3).

### Step 7.3 — Duplicating edge features (`h`, `Label`, `Attack`) to match

**Code:** `_build_single_graph`, Steps 3–4:

```python
h_values = np.stack(df["h"].values).astype(np.float32)
h_all = np.concatenate([h_values, h_values], axis=0)

label_values = df["Label"].to_numpy()
label_all = np.concatenate([label_values, label_values], axis=0)

attack_values = df["Attack"].to_numpy()
attack_all = np.concatenate([attack_values, attack_values], axis=0)
```

Since `src_all`/`dst_all` above are exactly `[forward edges, reverse edges]`, the feature arrays are duplicated the same way — `h_all`, `label_all`, `attack_all` all end up with `2 × num_rows` entries, where entry `i` and entry `i + num_rows` carry identical feature/label/attack values, one for the forward edge and one for its manufactured reverse.

**Continuing our example:**

```
h_all     = [ [0.0,1.0], [0.0,1.0], [1.0,1.0],   [0.0,1.0], [0.0,1.0], [1.0,1.0] ]
label_all = [     0,          0,         1,          0,          0,         1    ]
attack_all= [ Benign,     Benign,     DDoS,       Benign,     Benign,     DDoS   ]
             ^-------------- forward --------------^  ^-------------- reverse -------------^
```

So edge 4 (`192.168.1.5 → 10.0.0.1`, the manufactured reverse of row 1) carries `Label=0` (Benign) — identical to row 1's forward edge — because it is a direct copy, not an independently observed flow.

### Step 7.4 — Building the DGL graph

**Code:** `_build_single_graph`, Step 5:

```python
dgl_g = dgl.graph(
    (torch.from_numpy(src_all).long(), torch.from_numpy(dst_all).long()),
    num_nodes=num_nodes,
)
dgl_g.edata['h'] = torch.from_numpy(h_all)
dgl_g.edata['Label'] = torch.from_numpy(label_all)
dgl_g.edata['Attack'] = torch.from_numpy(attack_all)
```

This constructs the graph directly from the integer edge endpoint arrays — the same `DGLGraph` structure `dgl.from_networkx()` would have produced, just built without ever materialising a NetworkX graph in between. `DGLGraph` supports parallel/multi-edges by default (like the `MultiGraph` the old pipeline used), so if the same directed pair appears more than once (e.g. a real edge and a manufactured reverse edge landing on the same direction), both are kept as separate edges rather than one overwriting the other.

### Step 7.5 — Feature Assignment: Nodes (the constant "1s" vector)

**Code:** `_build_single_graph`, Step 6:

```python
edge_feat_dim = dgl_g.edata['h'].shape[1]
dgl_g.ndata['h'] = torch.ones([dgl_g.number_of_nodes(), edge_feat_dim])
```

**This step does not touch, delete, or overwrite edge data in any way.** `dgl_g.ndata` (node data) and `dgl_g.edata` (edge data) are two entirely separate storage structures inside a DGL graph object. This step only creates a brand-new feature *for nodes*, which never had any real data to begin with.

**Why nodes need something at all:** a NetFlow record only ever describes a flow between two IPs — there is no column anywhere describing "IP `10.0.0.1`'s own inherent properties." An IP address by itself carries no meaningful signal (it's just an identifier). But the GNN math (recall Equation 2 from the paper, $h_v^k = \sigma(W^k \cdot \text{CONCAT}(h_v^{k-1}, h_{N(v)}^k))$) requires every node to have some starting vector $h_v^0$ to begin the aggregation process. Since there's nothing meaningful to put there, the paper's design gives every node an identical placeholder: a vector of all 1s, dimensioned to match the edge features exactly — as stated in the paper: *"the dimensions of these constant vectors also match the dimensions of the edge feature set."*

**Our example — computing the node feature matrix:**
- `dgl_g.number_of_nodes()` = 4 (`10.0.0.1`, `10.0.0.2`, `192.168.1.5`, `192.168.1.6`, per the `unique_ips` ordering from Step 7.1)
- `edge_feat_dim` = 2 (length of each `h` vector)

$$
\text{ndata['h']} =
\begin{bmatrix}
1 & 1 \\
1 & 1 \\
1 & 1 \\
1 & 1
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow \text{node 0 (10.0.0.1)} \\
\leftarrow \text{node 1 (10.0.0.2)} \\
\leftarrow \text{node 2 (192.168.1.5)} \\
\leftarrow \text{node 3 (192.168.1.6)}
\end{matrix}
$$

Every node — regardless of how "suspicious" or "benign" its flows look — starts from the exact same `[1, 1]` vector. This forces E-GraphSAGE to derive all meaningful signal purely from the graph topology (who's connected to whom) and the edge features (the `h` vectors), rather than from any pre-existing notion of "this IP is bad."

**`edata['h']` is unaffected — before and after, side by side:**

Before Step 7.5 (right after Steps 7.1–7.4, all 6 edges from our example):

```python
>>> dgl_g.edata['h']
tensor([[0.0000, 1.0000],   # edge 1: 10.0.0.1 -> 192.168.1.5 (forward, real)
        [0.0000, 1.0000],   # edge 2: 10.0.0.1 -> 192.168.1.6 (forward, real)
        [0.0001, 0.9999],   # edge 3: 10.0.0.2 -> 192.168.1.5 (forward, real)
        [0.0000, 1.0000],   # edge 4: 192.168.1.5 -> 10.0.0.1 (reverse, manufactured)
        [0.0000, 1.0000],   # edge 5: 192.168.1.6 -> 10.0.0.1 (reverse, manufactured)
        [0.0001, 0.9999]])  # edge 6: 192.168.1.5 -> 10.0.0.2 (reverse, manufactured)
```

After Step 7.5 runs — identical, down to the last digit, since the step only reads `edata['h'].shape[1]` and writes to `ndata['h']`:

```python
>>> dgl_g.edata['h']
tensor([[0.0000, 1.0000],   # unchanged
        [0.0000, 1.0000],   # unchanged
        [0.0001, 0.9999],   # unchanged
        [0.0000, 1.0000],   # unchanged
        [0.0000, 1.0000],   # unchanged
        [0.0001, 0.9999]])  # unchanged
```

The graph now holds two separate, independent tensors side by side — the (real + manufactured) flow statistics on `edata['h']`, and the placeholder all-ones vectors on `ndata['h']` — exactly as Algorithm 1's line 1 (`h⁰_v ← x_v`) and the edge feature input (`{e_uv}`) require as separate inputs to E-GraphSAGE.

### Step 7.6 — Tensor Reshaping

**Code:** `_build_single_graph`, Step 7:

```python
dgl_g.ndata['h'] = torch.reshape(
    dgl_g.ndata['h'], (dgl_g.ndata['h'].shape[0], 1, dgl_g.ndata['h'].shape[1])
)
dgl_g.edata['h'] = torch.reshape(
    dgl_g.edata['h'], (dgl_g.edata['h'].shape[0], 1, dgl_g.edata['h'].shape[1])
)
```

**What changes and what doesn't:** no values change at all — only the tensor's shape changes, by inserting an extra dimension of size 1 in the middle.

**Our example, before reshape:** `dgl_g.ndata['h'].shape` = `(4, 2)` — 4 nodes, each a 2-element vector. `dgl_g.edata['h'].shape` = `(6, 2)` — 6 edges (3 real + 3 manufactured), each a 2-element vector.

**After reshape:** `dgl_g.ndata['h'].shape` = `(4, 1, 2)`. `dgl_g.edata['h'].shape` = `(6, 1, 2)` — same values, just wrapped with an extra middle dimension.

**Why this is necessary:** the E-GraphSAGE message-passing layers built with PyTorch's `nn.Linear` and DGL's message-passing API expect input tensors with this extra dimension — a data-formatting requirement of the downstream layers, not a conceptual change to the data itself.

### Final result of `generate_graphs()`

**Code:**

```python
def generate_graphs(self, train_df, test_df):
    train_g = self._build_single_graph(train_df)
    test_g = self._build_single_graph(test_df)
    return train_g, test_g
```

Two fully-formed DGL graph objects — `train_g` (built only from training rows 1–3, giving 4 nodes / 6 edges as worked out above) and `test_g` (built only from row 4) — completely independent of each other, each with:

- `ndata['h']`: constant all-ones node features, shape `(num_nodes, 1, edge_feat_dim)`
- `edata['h']`: the flow statistics (real + manufactured reverse copies), shape `(num_edges, 1, edge_feat_dim)`
- `edata['Label']`, `edata['Attack']`: kept alongside for later evaluation (never fed into the self-supervised training itself)

This is exactly the input E-GraphSAGE's forward pass (Algorithm 1 of the paper) expects: a graph where node features start as placeholders and all the real signal lives on the edges.

---

## A note on `main.py`

`main.py` isn't a third pipeline stage — it's simply the script that runs Files 1 and 2 back-to-back on a real dataset:

```python
preprocessor = AnomalEPreprocessor()
train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=False, fraction=0.05)

graph_builder = AnomalEGraphBuilder()
train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
```

It checks the dataset file exists, then hands off to the trainer (E-GraphSAGE + DGI training loop) and the anomaly detector for the rest of the pipeline — this file itself is purely orchestration, not a new stage in the data-processing logic.

For reference, the surrounding project structure is:

```
Anomal-E-Implementation/
├── data/raw/NF-CSE-CIC-IDS2018-v2.csv   ← original UQ CSV, not Parquet
├── docs/DATA_PIPELINE_EXPLANATION.md    ← this document
├── src/data_pipeline/
│   ├── preprocessor.py                  ← File 1 (AnomalEPreprocessor)
│   └── graph_builder.py                 ← File 2 (AnomalEGraphBuilder, no NetworkX)
├── src/engine/  src/models/  src/utils/ ← E-GraphSAGE, DGI training loop, PCA/IF/CBLOF/HBOS
└── main.py                              ← orchestration entry point
```

---

## Quick-reference: which file does what

| | File 1: `AnomalEPreprocessor` | File 2: `AnomalEGraphBuilder` |
|---|---|---|
| **Input** | Raw NetFlow **CSV only** (original UQ file — Parquet was tried and reverted, see Step 0) | Preprocessed train/test dataframes (with `h` column) |
| **Output** | Cleaned, encoded, normalised dataframes | DGL graph objects (`train_g`, `test_g`) |
| **Paper section** | §4.2, Fig. 4 (preprocessing box) | §4.2, Fig. 4 (graph generation box) + Algorithm 1, line 1 |
| **Key operations** | Chunked CSV loading, verify IP columns exist, drop ports, per-chunk stratified downsampling, split, target-encode, clean inf/NaN, L2-normalise | Map IPs to node IDs via `pd.factorize`, build directed edges in both directions, assign constant node features, reshape tensors — no NetworkX |
| **Data-leakage safeguard** | Encoder/scaler `fit()` only ever sees `X_train` | Train and test graphs built from fully separate dataframes — no shared node identity, even for IPs common to both |
| **Dev/debug aid** | `sanity_check=True` loads only 50,000 rows (no chunking) and disables split stratification | (inherits fast graphs automatically, since it just receives smaller dataframes) |

---

## One-paragraph mental model to keep in mind

Think of the whole pipeline as answering two questions in order. **File 1 asks:** "how do I turn messy, mixed-type flow records into clean numeric vectors — reading a 19-million-row file without blowing up RAM — without ever letting the test set leak into anything the model learns from?" **File 2 asks:** "given those clean vectors, how do I arrange them into the graph shape (nodes + directed edges + node/edge features) that E-GraphSAGE's math (Algorithm 1) actually operates on, without materialising a slow, memory-heavy NetworkX graph along the way?" Every design choice in both files — chunked CSV loading, CSV-only enforcement, sanity-check mode, port-dropping, per-chunk stratified downsampling, fit-only-on-train, `pd.factorize`-based node mapping, forward+reverse edge duplication, all-ones node features, tensor reshaping — exists to serve one or both of those two questions. `main.py` simply runs both files together before handing off to training.