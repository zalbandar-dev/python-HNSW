#!/usr/bin/env python
# coding: utf-8

import numpy as np
import hnswlib
import time
import os
import struct
import h5py
import requests
import tarfile
import gzip
import shutil
import json
from datetime import datetime
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import itertools
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# ============================================================
# HYPERPARAMETERS
# ============================================================

EF_CONSTRUCTION = 200
HNSW_GRAPH_DEGREE = 16
EF_SEARCH = 100
TARGET_DIMENSION = 32
DIM_REDUCTION_METHOD = "PCA"
NUM_QUERIES = 1000
K_NEIGHBORS = 10
MAX_ELEMENTS_DEEP1B = 1000000
WORST_X_PERCENT = 0.03

PARAM_GRID = {
    "ef_construction": range(100, 150, 25),
    "graph_degree":    range(10,  100, 20),
    "ef_search":       range(50,  300, 50),
    "target_dim":      range(15,  150, 35),
    "k_neighbors":     range(2,   100, 20),
}

NUM_RUNS = 1  # how many times to repeat each parameter combination

RESULTS_DIR = "./results_data"
DATA_DIR    = "./datasets"

print("=" * 60)
print("📋 HYPERPARAMETER SUMMARY:")
print("=" * 60)
print(f"  EF Construction:        {EF_CONSTRUCTION}")
print(f"  HNSW Graph Degree (M):  {HNSW_GRAPH_DEGREE}")
print(f"  EF Search:              {EF_SEARCH}")
print(f"  Target Dimension:       {TARGET_DIMENSION}")
print(f"  Reduction Method:       {DIM_REDUCTION_METHOD}")
print(f"  Num Queries:            {NUM_QUERIES}")
print(f"  K Neighbors:            {K_NEIGHBORS}")
print("=" * 60)


# ============================================================
# DATASET UTILITIES
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def download_file(url, filepath):
    """Download a file with progress bar."""
    if os.path.exists(filepath):
        print(f"  ✅ File already exists: {filepath}")
        return
    print(f"  ⬇️  Downloading: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"  ✅ Downloaded: {filepath}")


def read_fvecs(filename):
    """Read .fvecs file format."""
    with open(filename, 'rb') as f:
        data = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            data.append(vec)
    return np.array(data, dtype=np.float32)


def read_ivecs(filename):
    """Read .ivecs file format."""
    with open(filename, 'rb') as f:
        data = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            data.append(vec)
    return np.array(data, dtype=np.int32)


def read_bvecs(filename, max_elements=None):
    """Read .bvecs file format."""
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        dim = struct.unpack('i', dim_bytes)[0]
        f.seek(0)

        vec_size   = 4 + dim
        file_size  = os.path.getsize(filename)
        total_vecs = file_size // vec_size

        if max_elements:
            total_vecs = min(total_vecs, max_elements)

        data = np.zeros((total_vecs, dim), dtype=np.float32)
        for i in tqdm(range(total_vecs), desc="Reading bvecs"):
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            d   = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('B' * d, f.read(d))
            data[i] = vec
    return data


def load_sift1m():
    """Download and load SIFT1M dataset."""
    print("\n📦 Loading SIFT1M Dataset...")
    sift_dir = os.path.join(DATA_DIR, "sift")

    if not os.path.exists(sift_dir):
        alt_url   = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
        hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")

        download_file(alt_url, hdf5_path)

        with h5py.File(hdf5_path, 'r') as f:
            train_data   = np.array(f['train'],     dtype=np.float32)
            test_data    = np.array(f['test'],      dtype=np.float32)
            ground_truth = np.array(f['neighbors'], dtype=np.int32)

        print(f"  📊 Train shape: {train_data.shape}")
        print(f"  📊 Test shape:  {test_data.shape}")
        print(f"  📊 GT shape:    {ground_truth.shape}")

        return {
            'name':         'SIFT1M',
            'train':        train_data,
            'test':         test_data[:NUM_QUERIES],
            'ground_truth': ground_truth[:NUM_QUERIES, :K_NEIGHBORS],
            'original_dim': train_data.shape[1],
            'metric':       'l2'
        }

    train_data   = read_fvecs(os.path.join(sift_dir, "sift_base.fvecs"))
    test_data    = read_fvecs(os.path.join(sift_dir, "sift_query.fvecs"))
    ground_truth = read_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))

    return {
        'name':         'SIFT1M',
        'train':        train_data,
        'test':         test_data[:NUM_QUERIES],
        'ground_truth': ground_truth[:NUM_QUERIES, :K_NEIGHBORS],
        'original_dim': train_data.shape[1],
        'metric':       'l2'
    }


def load_deep1b():
    """Download and load DEEP1B dataset (subset)."""
    print("\n📦 Loading DEEP1B Dataset...")

    hdf5_path = os.path.join(DATA_DIR, "deep-image-96-angular.hdf5")
    url       = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"

    download_file(url, hdf5_path)

    with h5py.File(hdf5_path, 'r') as f:
        train_data   = np.array(f['train'],     dtype=np.float32)
        test_data    = np.array(f['test'],      dtype=np.float32)
        ground_truth = np.array(f['neighbors'], dtype=np.int32)

    if MAX_ELEMENTS_DEEP1B and train_data.shape[0] > MAX_ELEMENTS_DEEP1B:
        print(f"  ⚠️  Limiting DEEP1B from {train_data.shape[0]} to {MAX_ELEMENTS_DEEP1B} elements")
        train_data = train_data[:MAX_ELEMENTS_DEEP1B]

    print(f"  📊 Train shape: {train_data.shape}")
    print(f"  📊 Test shape:  {test_data.shape}")
    print(f"  📊 GT shape:    {ground_truth.shape}")

    return {
        'name':         'DEEP1B',
        'train':        train_data,
        'test':         test_data[:NUM_QUERIES],
        'ground_truth': ground_truth[:NUM_QUERIES, :K_NEIGHBORS],
        'original_dim': train_data.shape[1],
        'metric':       'cosine'
    }


print("✅ Dataset utilities defined!")


# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================

def apply_dimensionality_reduction(train_data, test_data, target_dim, method="PCA"):
    """Project high-dimensional data to lower dimensions."""
    original_dim = train_data.shape[1]

    if target_dim >= original_dim:
        print(f"  ⚠️  Target dim ({target_dim}) >= original dim ({original_dim}). Skipping.")
        return None, None, None

    if method == "PCA":
        reducer = PCA(n_components=target_dim, random_state=42)
    elif method == "GaussianRandomProjection":
        reducer = GaussianRandomProjection(n_components=target_dim, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    start_time = time.time()

    fit_size = min(100000, train_data.shape[0])
    print(f"  Fitting reducer on {fit_size} samples...")
    reducer.fit(train_data[:fit_size])

    reduced_train = reducer.transform(train_data).astype(np.float32)
    reduced_test  = reducer.transform(test_data).astype(np.float32)

    elapsed = time.time() - start_time

    if method == "PCA":
        explained_var = sum(reducer.explained_variance_ratio_) * 100
        print(f"  PCA Explained Variance: {explained_var:.2f}%")

    print(f"  Reduction time: {elapsed:.2f}s")
    print(f"  New shapes — Train: {reduced_train.shape}, Test: {reduced_test.shape}")

    return reduced_train, reduced_test, reducer


print("✅ Dimensionality reduction utilities defined!")


# ============================================================
# HNSW INDEX
# ============================================================

def build_hnsw_index(data, dim, ef_construction, M, metric='l2', num_threads=4):
    """Build HNSW index with specified hyperparameters."""
    num_elements = data.shape[0]

    print(f"\n  HNSW Index parameters")
    print(f"    Elements:        {num_elements:,}")
    print(f"    Dimensions:      {dim}")
    print(f"    EF Construction: {ef_construction}")
    print(f"    M (Graph Degree):{M}")
    print(f"    Space:           {metric}")

    index = hnswlib.Index(space=metric, dim=dim)
    index.init_index(
        max_elements=num_elements,
        ef_construction=ef_construction,
        M=M
    )
    index.set_num_threads(num_threads)

    start_time = time.time()
    index.add_items(data, np.arange(num_elements))
    build_time = time.time() - start_time

    print(f"  Build time:   {build_time:.2f}s")
    print(f"  Memory usage: ~{index.element_count * dim * 4 / (1024**2):.1f} MB (vectors only)")

    return index


def search_hnsw(index, queries, k, ef_search):
    """Search HNSW index and return per-query times."""
    index.set_ef(ef_search)

    # warm-up run (results discarded)
    index.knn_query(queries, k=k)

    per_query_times = []
    for i in tqdm(range(len(queries)), desc="Querying HNSW", unit="query", leave=False):
        query = queries[i:i+1]
        start = time.perf_counter()
        index.knn_query(query, k=k)
        elapsed = time.perf_counter() - start
        per_query_times.append(elapsed)

    return per_query_times


print("✅ HNSW index utilities defined!")


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_benchmark(dataset_info, ef_construction, M, target_dim,
                  reduction_method, ef_search, k):
    """Run a complete benchmark for one parameter combination."""

    train_data   = dataset_info['train']
    test_data    = dataset_info['test']
    metric       = dataset_info['metric']
    original_dim = dataset_info['original_dim']

    # ---- Dimensionality Reduction ----
    train_reduced, test_reduced, reducer = apply_dimensionality_reduction(
        train_data, test_data, target_dim, method=reduction_method
    )

    if train_reduced is None:
        return None  # invalid target_dim, skip

    # ---- Build Index ----
    index = build_hnsw_index(
        data=train_reduced,
        dim=target_dim,
        ef_construction=ef_construction,
        M=M,
        metric=metric
    )

    # ---- Search ----
    query_times = search_hnsw(
        index=index,
        queries=test_reduced,
        k=k,
        ef_search=ef_search
    )

    # ---- Worst-X% stats ----
    sorted_times = sorted(query_times)
    top_count    = max(1, round(len(sorted_times) * WORST_X_PERCENT))
    worst_times  = sorted_times[-top_count:]
    avg_worst    = float(np.mean(worst_times))

    result = {
        'dataset':          dataset_info['name'],
        'k':                k,
        'metric':           metric,
        'reduction_method': reduction_method,
        'dimensions':       target_dim,
        'ef_construction':  ef_construction,
        'M':                M,
        'ef_search':        ef_search,
        'query_times':      [float(t) for t in sorted_times],
        'worst_query_times':[float(t) for t in worst_times],
        'average_worst_time': avg_worst,
    }

    return result


print("✅ Benchmark runner defined!")


# ============================================================
# SAVE UTILITIES
# ============================================================

def save_results(all_sift_results, all_deep_results, save_dir=RESULTS_DIR):
    """Save benchmark results to timestamped JSON files."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_sift_results:
        clean     = [r for r in all_sift_results if r is not None]
        sift_path = os.path.join(save_dir, f"sift_results_{timestamp}.json")
        with open(sift_path, "w") as f:
            json.dump(clean, f)
        print(f"💾 Saved SIFT results ({len(clean)} runs) → {sift_path}")
    else:
        print("⚠️  No SIFT results to save.")

    if all_deep_results:
        clean     = [r for r in all_deep_results if r is not None]
        deep_path = os.path.join(save_dir, f"deep_results_{timestamp}.json")
        with open(deep_path, "w") as f:
            json.dump(clean, f)
        print(f"💾 Saved DEEP results ({len(clean)} runs) → {deep_path}")
    else:
        print("⚠️  No DEEP results to save.")


print("✅ Save utilities defined!")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # ── Load datasets once ──
    sift_data = load_sift1m()
    deep_data = load_deep1b()

    # ── Build parameter combinations ──
    param_product = list(itertools.product(
        PARAM_GRID["ef_construction"],
        PARAM_GRID["graph_degree"],
        PARAM_GRID["ef_search"],
        PARAM_GRID["target_dim"],
        PARAM_GRID["k_neighbors"],
        range(NUM_RUNS)
    ))

    total = len(param_product)
    print(f"\n🔁 Total parameter combinations: {total}")

    all_sift_results = []
    all_deep_results = []

    for ef_construction, graph_degree, ef_search, target_dim, k_neighbors, run \
            in tqdm(param_product, desc="Overall progress"):

        tqdm.write(
            f"\n▶ ef_c={ef_construction} M={graph_degree} "
            f"ef_s={ef_search} dim={target_dim} k={k_neighbors} run={run}"
        )

        # ── SIFT ──
        sift_result = run_benchmark(
            dataset_info    = sift_data,
            ef_construction = ef_construction,
            M               = graph_degree,
            target_dim      = target_dim,
            reduction_method= DIM_REDUCTION_METHOD,
            ef_search       = ef_search,
            k               = k_neighbors
        )
        all_sift_results.append(sift_result)

        # ── DEEP ──
        deep_result = run_benchmark(
            dataset_info    = deep_data,
            ef_construction = ef_construction,
            M               = graph_degree,
            target_dim      = target_dim,
            reduction_method= DIM_REDUCTION_METHOD,
            ef_search       = ef_search,
            k               = k_neighbors
        )
        all_deep_results.append(deep_result)

    # ── Save ──
    save_results(all_sift_results, all_deep_results, save_dir=RESULTS_DIR)

    print("\n" + "=" * 70)
    print("✅ ALL BENCHMARKS COMPLETE!")
    print(f"   SIFT runs: {len([r for r in all_sift_results if r is not None])}")
    print(f"   DEEP runs: {len([r for r in all_deep_results if r is not None])}")
    print("=" * 70)