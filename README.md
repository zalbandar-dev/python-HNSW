# HNSW + Dimensionality Reduction Benchmarking

Benchmarking approximate nearest-neighbor search using `hnswlib` on standard ANN benchmark datasets.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Important Fixes Before Running](#4-important-fixes-before-running)
5. [Running the Benchmark](#5-running-the-benchmark)
6. [Running the Plotting Script](#6-running-the-plotting-script)
7. [Main Hyperparameters](#7-main-hyperparameters)
8. [Current Parameter Grid](#8-current-parameter-grid)
9. [Output JSON Schema](#9-output-json-schema)
10. [Function Documentation](#10-function-documentation)
11. [How to Modify the Benchmark](#11-how-to-modify-the-benchmark)
12. [Recommended Next Improvements](#12-recommended-next-improvements)
13. [Troubleshooting](#13-troubleshooting)
14. [Suggested Learning Resources](#14-suggested-learning-resources)
15. [Minimal Workflow for a New Student](#15-minimal-workflow-for-a-new-student)
16. [Current Project Status](#16-current-project-status)

---

## 1. Project Overview

This project benchmarks **approximate nearest-neighbor search** using `hnswlib` on two ANN benchmark datasets:

- **SIFT1M** — 128-dimensional SIFT vectors, Euclidean / L2 distance.
- **DEEP image embeddings subset** — 96-dimensional vectors, cosine / angular-style distance.

The benchmark pipeline is:

1. Download / load benchmark datasets.
2. Optionally reduce dimensionality with **PCA** or **Gaussian Random Projection**.
3. Build an **HNSW** index with configurable parameters.
4. Time per-query nearest-neighbor search.
5. Save results as JSON.
6. Aggregate saved results and plot query-time distributions by hyperparameter.

`hnswlib` supports creating an index with `Index(space='l2'/'cosine'/'ip', dim=...)`, initializing with `ef_construction` and `M`, adding items, and controlling search quality/speed with `set_ef(...)`. Higher `ef` generally improves recall but slows search; larger `M` increases memory use and can improve accuracy.

The datasets used here come from ANN-Benchmarks-style HDF5 files: `sift-128-euclidean` and `deep-image-96-angular`.

---

## 2. Repository Structure

```text
project/
├── benchmark_hnsw.py          # Main benchmark script
├── plot_results.py            # Aggregation + plotting script
├── requirements.txt           # Python dependencies
├── datasets/                  # Downloaded HDF5 datasets live here
├── results_data/              # JSON benchmark outputs
├── esrp_plots/                # Saved PNG plots
└── README.md                  # This guide
```

If you're starting from the notebook-style code, split it into two scripts:

- **`benchmark_hnsw.py`** — dataset loading, dimensionality reduction, index building, search timing, benchmark loops, result saving.
- **`plot_results.py`** — JSON loading, aggregation, visualization.

---

## 3. Environment Setup

### 3.1 Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3.2 Install dependencies

Create `requirements.txt`:

```txt
numpy
hnswlib
h5py
requests
scikit-learn
tqdm
matplotlib
pandas
scipy
```

Install:

```bash
pip install -r requirements.txt
```

This project uses:

- `h5py` for reading HDF5 benchmark files.
- `scikit-learn` for `PCA` and `GaussianRandomProjection`.
- `hnswlib` for ANN indexing.
- `pandas` + `matplotlib` + `scipy` for plotting/aggregation.

---

## 4. Important Fixes Before Running

The pasted code contains a few notebook-export artifacts that must be cleaned up.

### 4.1 Fix script entry point

Replace:

```python
if name == "main":
```

with:

```python
if __name__ == "__main__":
```

Do this in **both** scripts.

### 4.2 Remove stray `text` lines

Lines like:

```python
text
```

will raise `NameError`. Delete them.

### 4.3 Turn section headers into comments

Replace:

```text
============================================================
HYPERPARAMETERS
============================================================
```

with:

```python
# ============================================================
# HYPERPARAMETERS
# ============================================================
```

### 4.4 Fix the encoding line

Use:

```python
#!/usr/bin/env python
# coding: utf-8
```

not:

```python
#!/usr/bin/env python

coding: utf-8
```

---

## 5. Running the Benchmark

After cleaning the script:

```bash
python benchmark_hnsw.py
```

The script will:

1. Create `./datasets` and `./results_data` if needed.
2. Download benchmark HDF5 files if missing.
3. Load train / query / ground-truth arrays.
4. Loop through the hyperparameter grid.
5. Run dimensionality reduction.
6. Build an HNSW index.
7. Time each query.
8. Save JSON result files to `./results_data`.

Example output files:

```text
results_data/
├── sift_results_20260504_153000.json
└── deep_results_20260504_153000.json
```

---

## 6. Running the Plotting Script

After at least one benchmark run:

```bash
python plot_results.py
```

This script will:

1. Load all `.json` files from `./results_data`.
2. Separate SIFT and DEEP results based on filename.
3. Aggregate per-query timing samples.
4. Plot query-time distributions grouped by hyperparameter quartiles.
5. Save plots to `./esrp_plots`.

Example output:

```text
esrp_plots/
├── query_time_dist_sift1m_20260504_160000.png
└── query_time_dist_deep1b_20260504_160000.png
```

---

## 7. Main Hyperparameters

```python
EF_CONSTRUCTION = 200
HNSW_GRAPH_DEGREE = 16
EF_SEARCH = 100
TARGET_DIMENSION = 32
DIM_REDUCTION_METHOD = "PCA"
NUM_QUERIES = 1000
K_NEIGHBORS = 10
MAX_ELEMENTS_DEEP1B = 1000000
WORST_X_PERCENT = 0.03
```

| Parameter | Meaning |
|---|---|
| `ef_construction` | HNSW build-time search width. Higher → better graph quality, slower build. |
| `M` / `graph_degree` | HNSW graph connectivity. Larger → more memory, possibly higher accuracy. |
| `ef_search` | Search-time candidate list size. Higher → better accuracy, slower queries. |
| `target_dim` | Number of dimensions after PCA / random projection. |
| `k_neighbors` | Number of nearest neighbors returned per query. |
| `NUM_QUERIES` | Number of query vectors benchmarked. |
| `MAX_ELEMENTS_DEEP1B` | Caps DEEP training vectors for memory/time. |
| `WORST_X_PERCENT` | Fraction of slowest queries used for tail-latency stats. |

---

## 8. Current Parameter Grid

```python
PARAM_GRID = {
    "ef_construction": range(100, 150, 25),
    "graph_degree":    range(10, 100, 20),
    "ef_search":       range(50, 300, 50),
    "target_dim":      range(15, 150, 35),
    "k_neighbors":     range(2, 100, 20),
}
```

Expands to:

```text
2 ef_construction values
× 5 graph_degree values
× 5 ef_search values
× 4 target_dim values
× 5 k_neighbors values
= 1000 parameter combinations
```

Because the script runs both SIFT and DEEP, that's up to **2000 benchmark attempts**. DEEP has original dim 96, so `target_dim=120` is skipped automatically.

For a quick smoke test:

```python
PARAM_GRID = {
    "ef_construction": [100],
    "graph_degree":    [16],
    "ef_search":       [50, 100],
    "target_dim":      [32],
    "k_neighbors":     [10],
}

NUM_QUERIES = 100
MAX_ELEMENTS_DEEP1B = 100000
NUM_RUNS = 1
```

---

## 9. Output JSON Schema

```json
{
  "dataset": "SIFT1M",
  "k": 10,
  "metric": "l2",
  "reduction_method": "PCA",
  "dimensions": 32,
  "ef_construction": 100,
  "M": 16,
  "ef_search": 100,
  "query_times": [0.00012, 0.00013, "..."],
  "worst_query_times": [0.00042, 0.00045, "..."],
  "average_worst_time": 0.00044
}
```

> Timing values are stored in **seconds**. The plotting script multiplies by `1000` to display milliseconds.

---

## 10. Function Documentation

### `download_file(url, filepath)`

Downloads a file with a progress bar.

- Skips download if `filepath` exists.
- Streams the response to disk in chunks.
- Uses `tqdm` for progress.

---

### `load_sift1m()`

Loads the SIFT1M dataset.

```python
{
    "name": "SIFT1M",
    "train": train_data,
    "test": test_data,
    "ground_truth": ground_truth,
    "original_dim": 128,
    "metric": "l2"
}
```

- Downloads `sift-128-euclidean.hdf5` if needed.
- Uses the first `NUM_QUERIES` test vectors.
- Slices ground-truth to `K_NEIGHBORS`.

---

### `load_deep1b()`

Loads the DEEP image dataset.

```python
{
    "name": "DEEP1B",
    "train": train_data,
    "test": test_data,
    "ground_truth": ground_truth,
    "original_dim": 96,
    "metric": "cosine"
}
```

- Downloads `deep-image-96-angular.hdf5` if needed.
- Caps training data at `MAX_ELEMENTS_DEEP1B`.

> **Warning:** if you compute recall on a subset of DEEP, ground-truth neighbors may reference items outside the subset.

---

### `apply_dimensionality_reduction(train_data, test_data, target_dim, method="PCA")`

Projects train/query vectors into a lower-dimensional space.

- Skips reduction if `target_dim >= original_dim`.
- Fits the reducer on at most the first `100000` training vectors.
- Returns `(reduced_train, reduced_test, reducer)`.
- Prints PCA explained variance.

---

### `build_hnsw_index(data, dim, ef_construction, M, metric="l2", num_threads=4)`

Builds an HNSW index using `hnswlib.Index`, initializes with `max_elements`, `ef_construction`, and `M`, then adds vectors with integer IDs.

---

### `search_hnsw(index, queries, k, ef_search)`

Times HNSW queries:

- Sets `index.set_ef(ef_search)`.
- Performs one warm-up query batch.
- Times each query individually using `time.perf_counter()`.
- Returns a list of per-query times in seconds.

---

### `run_benchmark(...)`

Runs a full benchmark for one parameter combination:

1. Reduce dimensionality.
2. Build HNSW index.
3. Search all queries.
4. Sort query times.
5. Compute slowest `WORST_X_PERCENT` tail-latency.
6. Return result dictionary.

---

### `save_results(all_sift_results, all_deep_results, save_dir=RESULTS_DIR)`

Saves results to timestamped files:

```text
sift_results_YYYYMMDD_HHMMSS.json
deep_results_YYYYMMDD_HHMMSS.json
```

---

### `load_all_results(results_dir="./results_data")`

- Scans `results_dir`.
- Loads every `.json` file.
- Routes files containing `"sift"` to SIFT results.
- Routes files containing `"deep"` to DEEP results.

---

### `plot_variable_impact(results, dataset_name="SIFT1M", save_dir="./esrp_plots")`

For each hyperparameter:

1. Bucket runs into quartiles using `pd.qcut`.
2. Concatenate all query timings in each bucket.
3. Plot log-scaled histograms.
4. Overlay KDE curves.
5. Save the figure as PNG.

---

## 11. How to Modify the Benchmark

### Change the dimensionality-reduction method

```python
DIM_REDUCTION_METHOD = "PCA"
# or
DIM_REDUCTION_METHOD = "GaussianRandomProjection"
```

### Change the number of queries

```python
NUM_QUERIES = 100   # fast
NUM_QUERIES = 1000  # full
```

### Change DEEP subset size

```python
MAX_ELEMENTS_DEEP1B = 100000     # low memory
MAX_ELEMENTS_DEEP1B = 1000000    # full subset
```

### Add more repeated runs

```python
NUM_RUNS = 3
```

### Add another dataset

1. Write a loader function similar to `load_sift1m()` / `load_deep1b()`.
2. Return the same dictionary shape:

```python
{
    "name": "NEW_DATASET",
    "train": train_data,
    "test": test_data,
    "ground_truth": ground_truth,
    "original_dim": train_data.shape[1],
    "metric": "l2"  # or "cosine" / "ip"
}
```

3. Call `run_benchmark(...)` inside the main loop.

---

## 12. Recommended Next Improvements

### 12.1 Add recall measurement

Currently the script measures **latency only**. Add:

```python
def compute_recall(predicted_labels, ground_truth, k):
    """Compute average recall@k."""
    recalls = []
    for pred, true in zip(predicted_labels, ground_truth):
        pred_set = set(pred[:k])
        true_set = set(true[:k])
        recalls.append(len(pred_set & true_set) / k)
    return float(np.mean(recalls))
```

Modify `search_hnsw` to return labels too:

```python
def search_hnsw(index, queries, k, ef_search):
    """Search HNSW and return labels, distances, per-query times."""
    index.set_ef(ef_search)
    index.knn_query(queries[:10], k=k)  # warm-up

    labels_all, distances_all, per_query_times = [], [], []

    for i in tqdm(range(len(queries)), desc="Querying HNSW", leave=False):
        query = queries[i:i + 1]
        start = time.perf_counter()
        labels, distances = index.knn_query(query, k=k)
        elapsed = time.perf_counter() - start

        labels_all.append(labels[0])
        distances_all.append(distances[0])
        per_query_times.append(elapsed)

    return np.array(labels_all), np.array(distances_all), per_query_times
```

> If you evaluate `k_neighbors` larger than the original `K_NEIGHBORS`, load more ground-truth columns.

### 12.2 Cache reduced datasets

PCA is currently refit for every parameter combination. Cache by `(dataset_name, reduction_method, target_dim)`:

```python
reduction_cache = {}
cache_key = (dataset_info["name"], reduction_method, target_dim)

if cache_key not in reduction_cache:
    reduction_cache[cache_key] = apply_dimensionality_reduction(...)

train_reduced, test_reduced, reducer = reduction_cache[cache_key]
```

### 12.3 Save build / reduction time

```python
"reduction_time": reduction_time,
"build_time": build_time,
"mean_query_time": float(np.mean(query_times)),
"median_query_time": float(np.median(query_times)),
"p95_query_time": float(np.percentile(query_times, 95)),
"p99_query_time": float(np.percentile(query_times, 99))
```

### 12.4 Optionally save raw labels/distances

Useful for debugging recall, but increases JSON size:

```python
"labels": labels.tolist(),
"distances": distances.tolist()
```

---

## 13. Troubleshooting

### `NameError: name 'name' is not defined`

Use `if __name__ == "__main__":`.

### `NameError: name 'text' is not defined`

Delete all standalone `text` lines.

### Benchmark takes too long

Use a smaller grid:

```python
PARAM_GRID = {
    "ef_construction": [100],
    "graph_degree":    [16],
    "ef_search":       [50],
    "target_dim":      [32],
    "k_neighbors":     [10],
}

NUM_QUERIES = 100
MAX_ELEMENTS_DEEP1B = 100000
```

### Out of memory

- Reduce `MAX_ELEMENTS_DEEP1B`.
- Use smaller `graph_degree` values.
- Avoid very large `M`.

### Plotting fails — no JSON files

Run the benchmark first:

```bash
python benchmark_hnsw.py
python plot_results.py
```

### `target_dim` is skipped

By design: skipped if `target_dim >= original_dim`. For DEEP, `target_dim=120` is skipped (original = 96).

---

## 14. Suggested Learning Resources

- **hnswlib Python examples** — index creation, `init_index`, `add_items`, `set_ef`, querying, parameter notes.
- **scikit-learn PCA documentation** — PCA parameters and projection behavior.
- **scikit-learn random projection documentation** — Gaussian random projection and Johnson–Lindenstrauss motivation.
- **h5py documentation** — HDF5 dataset loading.
- **pandas user guide** — manipulating benchmark results as tabular data.
- **Matplotlib documentation** — modifying plots and saving figures.
- **ANN-Benchmarks website** — benchmark datasets and ANN algorithm comparisons.

---

## 15. Minimal Workflow for a New Student

```bash
git clone <project-repo>
cd <project-repo>

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# or .venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

Smoke test by reducing the grid:

```python
PARAM_GRID = {
    "ef_construction": [100],
    "graph_degree":    [16],
    "ef_search":       [50],
    "target_dim":      [32],
    "k_neighbors":     [10],
}

NUM_QUERIES = 100
MAX_ELEMENTS_DEEP1B = 100000
```

Run:

```bash
python benchmark_hnsw.py
python plot_results.py
```

Check:

```text
results_data/
esrp_plots/
```

If those contain JSON files and PNG plots, the pipeline is working.

---

## 16. Current Project Status

### ✅ What works

- Dataset download / loading.
- PCA or Gaussian random projection.
- HNSW index construction.
- Per-query latency measurement.
- Worst-tail query-time calculation.
- JSON result saving.
- Aggregated plotting across runs.

### 🚧 What still needs improvement

- Recall@k calculation.
- PCA / reduction caching.
- Better memory accounting.
- Saving build / reduction / search summary statistics.
- Cleaner experiment configuration (YAML / JSON / CLI args).
- More robust handling of DEEP ground truth when using subsets.
- Optional multiprocessing or job scheduling for large parameter grids.

---

*Last updated: 2026-05-04*
