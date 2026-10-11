"""Render every figure for E1 / E3 / E6 with the new bold style.

* E1 QPS + recall per dataset (glue / sift / glove) — 6 PNGs.
* E3 QPS + recall bar chart — 2 PNGs.
* E6 cross-substrate (DuckDB vs PG) — 2 PNGs (one for QPS, one for recall).

The plots deliberately use ``plot_style`` so the figures are
visually scannable on screen (linewidth 2.6, markersize 10,
axis labels 14pt).
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps

FIGS = "E:/CloudStation/MosaicOpt/docs/SIGMOD/figures"
os.makedirs(FIGS, exist_ok=True)

CATEGORIES = ["Q1_VTC", "Q2_VGC", "Q3_VTGC", "Q4_VTf",
              "Q5_Vonly", "Q6_Tonly", "Q7_MV", "Q8_Join"]
CATEGORY_LABELS = {
    "Q1_VTC": "Q1 VTC", "Q2_VGC": "Q2 VGC", "Q3_VTGC": "Q3 VTGC",
    "Q4_VTf": "Q4 VTf", "Q5_Vonly": "Q5 V-only", "Q6_Tonly": "Q6 T-only",
    "Q7_MV": "Q7 MV", "Q8_Join": "Q8 Join",
}
SYSTEMS = ["mosaicopt", "post_filter", "pre_filter", "navix", "acorn", "bvb_native"]

# Map plot-script names → palette names in plot_style (with renames)
SYSTEM_KEY = {
    "mosaicopt":   "mosaicopt",
    "post_filter": "postfilter",
    "pre_filter":  "prefilter",
    "navix":       "navix",
    "acorn":       "acorn",
    "bvb_native":  "bvb_native",
}


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


# ============================================================
# E1 per-dataset (QPS + recall)
# ============================================================
def plot_e1_per_dataset(rows, dataset_name, suffix, dim):
    by_cat_sys_sel = defaultdict(list)
    for r in rows:
        by_cat_sys_sel[(r["category"], r["system"],
                        float(r["target_selectivity"]))].append(
            (float(r["qps"]), float(r["recall"])))

    def _plot(metric_key, ylabel, fname, ylog=False, ylim=None):
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        for ci, cat in enumerate(CATEGORIES):
            ax = axes[ci // 4, ci % 4]
            for sys_name in SYSTEMS:
                key = SYSTEM_KEY[sys_name]
                sels = sorted({float(r["target_selectivity"]) for r in rows
                                if r["category"] == cat and r["system"] == sys_name})
                vals_med = []
                for sel in sels:
                    cells = by_cat_sys_sel.get((cat, sys_name, sel), [])
                    vals_med.append(np.median([v[metric_key] for v in cells]) if cells else 0)
                ax.plot(sels, vals_med,
                        marker=ps.MARKER[key],
                        color=ps.PALETTE[key],
                        label=ps.LABEL[key],
                        linewidth=ps.mpl.rcParams["lines.linewidth"],
                        markersize=ps.mpl.rcParams["lines.markersize"])
            ax.set_xscale("log")
            ax.set_xlabel("target selectivity $s$")
            ax.set_ylabel(ylabel)
            ax.set_title(CATEGORY_LABELS[cat])
            ax.grid(True, alpha=0.30, linewidth=0.7)
            if ylog:
                ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(*ylim)
            if ci == 0:
                ax.legend(loc="best", ncol=1)
        fig.suptitle(f"E1 ({dataset_name}, n=10k, d={dim}): {ylabel} vs target selectivity",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot(0, "QPS",  f"{FIGS}/e1_real_qps_{suffix}.png",  ylog=True)
    _plot(1, "recall", f"{FIGS}/e1_real_recall_{suffix}.png", ylim=(-0.05, 1.10))
    print(f"  wrote {suffix} QPS+recall")


# ============================================================
# E3 bar charts
# ============================================================
def plot_e3(rows):
    by_cfg_cat = defaultdict(list)
    for r in rows:
        by_cfg_cat[(r["config"], r["category"])].append(
            (float(r["qps"]), float(r["recall"])))

    CONFIGS = ["default_0.5", "with_stats", "with_rho", "with_adaptive", "full"]
    CFG_LABELS = {
        "default_0.5": "default\\_0.5",
        "with_stats":  "with\\_stats",
        "with_rho":    "with\\_$\\rho_f$",
        "with_adaptive": "with\\_adaptive",
        "full":        "full",
    }
    E3_CATS = ["Q1_VTC", "Q3_VTGC", "Q5_Vonly", "Q8_Join"]
    E3_LABELS = {
        "Q1_VTC": "Q1 VTC", "Q3_VTGC": "Q3 VTGC",
        "Q5_Vonly": "Q5 V-only", "Q8_Join": "Q8 Join",
    }

    bar_colors = ["#003049", "#669BBC", "#E5C595", "#C1121F"]

    # QPS
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(CONFIGS))
    width = 0.20
    for ci, cat in enumerate(E3_CATS):
        qps_med = []
        for cfg in CONFIGS:
            vals = by_cfg_cat.get((cfg, cat), [])
            qps_med.append(np.median([q for q, _ in vals]) if vals else 0)
        ax.bar(x + (ci - 1.5) * width, qps_med, width,
               label=E3_LABELS[cat], color=bar_colors[ci],
               edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CFG_LABELS[c] for c in CONFIGS])
    ax.set_ylabel("QPS")
    ax.set_xlabel("CBO configuration")
    ax.set_title("E3 (BigVectorBench-style, n=10k, d=128): QPS by CBO configuration",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.30, axis="y", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/e3_real_cbo_qps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Recall
    fig, ax = plt.subplots(figsize=(11, 6))
    for ci, cat in enumerate(E3_CATS):
        rec_med = []
        for cfg in CONFIGS:
            vals = by_cfg_cat.get((cfg, cat), [])
            rec_med.append(np.median([r for _, r in vals]) if vals else 0)
        ax.bar(x + (ci - 1.5) * width, rec_med, width,
               label=E3_LABELS[cat], color=bar_colors[ci],
               edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CFG_LABELS[c] for c in CONFIGS])
    ax.set_ylabel("recall")
    ax.set_xlabel("CBO configuration")
    ax.set_title("E3 (BigVectorBench-style, n=10k, d=128): recall by CBO configuration",
                 fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.6, linewidth=1.6,
               label="recall floor = 0.95")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.30, axis="y", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/e3_real_cbo_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote E3 QPS+recall")


# ============================================================
# E6 cross-substrate (DuckDB vs PG)
# ============================================================
def plot_e6(rows):
    """rows: dict-like with keys substrate, category, system, selectivity,
       qps, recall, latency_ms, bulk_load_ms
    """
    by_cat_sub = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cat_sub[r["category"]][r["substrate"]].append(
            (float(r["selectivity"]), float(r["qps"]), float(r["recall"])))

    SUBS = ["duckdb", "postgres"]
    E6_CATS = ["Q1_VTC", "Q7_MV", "Q8_Join"]
    E6_LABELS = {
        "Q1_VTC": "Q1 VTC", "Q7_MV": "Q7 MV", "Q8_Join": "Q8 Join",
    }
    SYSTEMS_E6 = ["mosaicopt", "pre_filter", "post_filter", "navix"]

    SUBS_ALL = ["duckdb", "sqlite", "postgres"]
    SUB_LINESTYLE = {"duckdb": "-", "sqlite": "--", "postgres": ":"}
    SUB_LABEL = {
        "duckdb":   "DuckDB",
        "sqlite":   "SQLite",
        "postgres": "PostgreSQL",
    }

    def _matches():
        return [r for r in rows if r["substrate"] in SUBS_ALL]

    # QPS: 3 substrates × 1 figure with 3 panels (Q1, Q7, Q8).
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    for ci, cat in enumerate(E6_CATS):
        ax = axes[ci]
        # Iterate substrate-sorted then system-sorted for stable legend
        for sub in SUBS_ALL:
            for sys_name in SYSTEMS_E6:
                sub_rows = [r for r in rows
                            if r["category"] == cat
                            and r["substrate"] == sub
                            and r["system"] == sys_name]
                if not sub_rows:
                    continue
                sub_rows.sort(key=lambda r: float(r["selectivity"]))
                sels = [float(r["selectivity"]) for r in sub_rows]
                qps = [float(r["qps"]) for r in sub_rows]
                ls = SUB_LINESTYLE[sub]
                mk = ps.MARKER[sys_name]
                color = ps.PALETTE[sys_name]
                label = f"{ps.LABEL[sys_name]} @ {SUB_LABEL[sub]}"
                ax.plot(sels, qps, marker=mk, color=color, linestyle=ls,
                        linewidth=2.6, markersize=10, label=label,
                        alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("target selectivity $s$")
        ax.set_ylabel("QPS")
        ax.set_title(E6_LABELS[cat])
        ax.grid(True, alpha=0.30, linewidth=0.7)
        if ci == 0:
            ax.legend(loc="lower left", ncol=1, framealpha=0.95,
                      fontsize=10)
    fig.suptitle("E6 (cross-substrate, n=10k): QPS on DuckDB / SQLite / PostgreSQL",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/e6_cross_substrate_qps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Recall
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    for ci, cat in enumerate(E6_CATS):
        ax = axes[ci]
        for sub in SUBS_ALL:
            for sys_name in SYSTEMS_E6:
                sub_rows = [r for r in rows
                            if r["category"] == cat
                            and r["substrate"] == sub
                            and r["system"] == sys_name]
                if not sub_rows:
                    continue
                sub_rows.sort(key=lambda r: float(r["selectivity"]))
                sels = [float(r["selectivity"]) for r in sub_rows]
                rec = [float(r["recall"]) for r in sub_rows]
                ls = SUB_LINESTYLE[sub]
                mk = ps.MARKER[sys_name]
                color = ps.PALETTE[sys_name]
                label = f"{ps.LABEL[sys_name]} @ {SUB_LABEL[sub]}"
                ax.plot(sels, rec, marker=mk, color=color, linestyle=ls,
                        linewidth=2.6, markersize=10, label=label,
                        alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("target selectivity $s$")
        ax.set_ylabel("recall")
        ax.set_title(E6_LABELS[cat])
        ax.set_ylim(-0.05, 1.10)
        ax.grid(True, alpha=0.30, linewidth=0.7)
        if ci == 0:
            ax.legend(loc="lower left", ncol=1, framealpha=0.95,
                      fontsize=10)
    fig.suptitle("E6 (cross-substrate, n=10k): recall on DuckDB / SQLite / PostgreSQL",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/e6_cross_substrate_recall.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote E6 QPS+recall")


# ============================================================
# Main
# ============================================================
def main():
    print("== E1 per-dataset ==")
    datasets = [
        ("BigVectorBench-style", "e1_real_n10k_d128.csv",  "glue",  128),
        ("SIFT1M-extended",      "e1_real_sift_n10k_d128.csv", "sift", 128),
        ("GloVe-100-extended",   "e1_real_glove_n10k_d100.csv", "glove", 100),
    ]
    for ds_label, fname, suffix, dim in datasets:
        rows = load_csv(f"E:/CloudStation/MosaicOpt/prototype/results/{fname}")
        if not rows:
            print(f"  skip {ds_label}: no CSV")
            continue
        plot_e1_per_dataset(rows, ds_label, suffix, dim)

    print("== E3 ==")
    rows = load_csv("E:/CloudStation/MosaicOpt/prototype/results/e3_real_cbo_n10k.csv")
    if rows:
        plot_e3(rows)
    else:
        print("  skip E3: no CSV")

    print("== E6 cross-substrate ==")
    rows = load_csv("E:/CloudStation/MosaicOpt/prototype/results/e6_cross_substrate.csv")
    if rows:
        plot_e6(rows)
    else:
        print("  skip E6: no CSV (run prototype first)")


if __name__ == "__main__":
    main()
