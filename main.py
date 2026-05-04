#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
import pandas as pd
from datetime import datetime

# ============================================================
# LOAD & AGGREGATE
# ============================================================

def load_all_results(results_dir="./results_data"):
    """
    Scans directory for all JSON result files and aggregates them.
    Returns all_sift_results, all_deep_results as flat lists of run dicts.
    """
    all_sift_results = []
    all_deep_results = []

    if not os.path.exists(results_dir):
        print(f"❌ Directory not found: {results_dir}")
        return [], []

    files = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])

    if not files:
        print(f"⚠️  No JSON files found in {results_dir}")
        return [], []

    print(f"📂 Found {len(files)} result file(s) in {results_dir}")

    for filename in files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        # filter out nulls
        data = [r for r in data if r is not None]

        if "sift" in filename.lower():
            all_sift_results.extend(data)
            print(f"  ✅ Loaded {len(data):>5} SIFT runs  ← {filename}")
        elif "deep" in filename.lower():
            all_deep_results.extend(data)
            print(f"  ✅ Loaded {len(data):>5} DEEP runs  ← {filename}")

    print(f"\n📊 Total — SIFT: {len(all_sift_results)}, DEEP: {len(all_deep_results)}")
    return all_sift_results, all_deep_results


# ============================================================
# PLOT
# ============================================================

def plot_variable_impact(results, dataset_name="SIFT1M", save_dir="./esrp_plots"):
    """
    For each hyperparameter, plots how its value affects query time distribution.
    Each color = a quartile bucket of that variable's values.
    All other variables are averaged out.
    """

    # ─────────────────────────────────────────
    #         PLOT CONFIGURATION
    # ─────────────────────────────────────────
    VARIABLES = {
        "ef_construction": "EF Construction",
        "ef_search":       "EF Search",
        "k":               "K Neighbors",
        "dimensions":      "Target Dimensions",
        "M":               "Graph Degree (M)",
    }
    N_BUCKETS   = 4        # quartile groups per variable
    ALPHA       = 0.25     # histogram transparency
    BINS        = 60       # histogram bins
    FIGURE_COLS = 2        # subplot grid columns
    COLORMAP    = "plasma"
    # ─────────────────────────────────────────

    if not results:
        print(f"⚠️  No results provided for {dataset_name}.")
        return

    df = pd.DataFrame(results)

    # ── Check which variables exist and have enough unique values ──
    available_vars = {}
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} — {len(df)} total runs loaded")
    print(f"Variable ranges:")
    for k, v in VARIABLES.items():
        if k not in df.columns:
            print(f"  ⚠️  Skipping '{k}': column not found")
            continue
        n_unique = df[k].nunique()
        print(f"  {k:20s}: {n_unique} unique values  "
              f"(min={df[k].min()}, max={df[k].max()})")
        if n_unique < 2:
            print(f"             ⚠️  Skipping — need at least 2 unique values")
            continue
        available_vars[k] = v

    if not available_vars:
        print("\n❌ No variables have enough unique values to plot.")
        print(f"   → Run more benchmark iterations with varied parameters.")
        return

    # ── Shared log-scale bin edges from ALL query times ──
    all_times_ms = np.concatenate([
        np.array(r["query_times"]) * 1000
        for r in results
        if "query_times" in r and r["query_times"] and len(r["query_times"]) > 0
    ])
    print(f"\nTotal query time samples : {len(all_times_ms):,}")
    print(f"Time range               : "
          f"{all_times_ms.min():.4f}ms – {all_times_ms.max():.4f}ms")

    log_min   = np.log10(np.percentile(all_times_ms, 0.5))
    log_max   = np.log10(np.percentile(all_times_ms, 99.5))
    bin_edges = np.logspace(log_min, log_max, BINS)

    # ── Build subplot grid ──
    n_vars = len(available_vars)
    n_cols = FIGURE_COLS
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    fig.suptitle(
        f"Query Time Distribution by Hyperparameter — {dataset_name}\n"
        f"({len(df)} total runs aggregated)",
        fontsize=15, fontweight="bold", y=1.02
    )

    for ax_idx, (var_key, var_label) in enumerate(available_vars.items()):
        ax = axes[ax_idx]
        print(f"\n── Plotting: {var_label} ──")

        try:
            df["_bucket"] = pd.qcut(df[var_key], q=N_BUCKETS, duplicates="drop")
        except Exception as e:
            print(f"   ⚠️  Could not bucket {var_key}: {e}")
            ax.set_visible(False)
            continue

        buckets = sorted([b for b in df["_bucket"].unique() if pd.notna(b)])
        if not buckets:
            ax.set_visible(False)
            continue

        colors = plt.cm.get_cmap(COLORMAP)(np.linspace(0.15, 0.85, len(buckets)))
        plotted_any = False

        for color, bucket in zip(colors, buckets):
            bucket_df = df[df["_bucket"] == bucket]

            times_list = [
                np.array(row["query_times"]) * 1000
                for _, row in bucket_df.iterrows()
                if row.get("query_times") and len(row["query_times"]) > 0
            ]

            if not times_list:
                continue

            bucket_times = np.concatenate(times_list)
            n_runs = len(bucket_df)
            print(f"   Bucket {bucket}: {n_runs} runs, {len(bucket_times):,} samples")

            # histogram
            ax.hist(
                bucket_times,
                bins=bin_edges,
                density=True,
                alpha=ALPHA,
                color=color,
            )

            # KDE curve
            try:
                log_times = np.log10(bucket_times[bucket_times > 0])
                kde      = gaussian_kde(log_times, bw_method=0.15)
                x_range  = np.linspace(log_min, log_max, 500)
                kde_vals = kde(x_range) / (np.log(10) * 10**x_range)
                ax.plot(
                    10**x_range, kde_vals,
                    color=color, linewidth=2,
                    label=f"{bucket}\n(n={n_runs} runs)"
                )
                plotted_any = True
            except Exception as e:
                print(f"   ⚠️  KDE failed for bucket {bucket}: {e}")

        if not plotted_any:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

        ax.set_xscale("log")
        ax.set_xlabel("Query Time (ms)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Impact of {var_label}", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.3g}ms")
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(fontsize=7, framealpha=0.85, loc="upper right")

    # hide any leftover empty axes
    for i in range(len(available_vars), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        save_dir, f"query_time_dist_{dataset_name.lower()}_{timestamp}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n💾 Saved → {save_path}")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    RESULTS_DIR = "./results_data"
    PLOTS_DIR   = "./esrp_plots"

    all_sift_results, all_deep_results = load_all_results(RESULTS_DIR)

    plot_variable_impact(all_sift_results, dataset_name="SIFT1M", save_dir=PLOTS_DIR)
    plot_variable_impact(all_deep_results, dataset_name="DEEP1B", save_dir=PLOTS_DIR)