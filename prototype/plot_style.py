"""Shared plotting style: thick lines, large markers, large fonts.

The figures are meant to be readable on a screen at modest zoom.
The defaults below are about 1.7x larger than matplotlib's default
in every relevant dimension (linewidth, markersize, font sizes).
All E1/E2/E3/E5/E6 plot scripts import this.
"""
import matplotlib as mpl

# Master style: bigger than matplotlib defaults.
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.6,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "lines.linewidth": 2.6,         # was matplotlib default 1.5
    "lines.markersize": 10,          # was default 6
    "lines.markeredgewidth": 1.0,
    "figure.dpi": 110,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Substrate / system palettes (high contrast for screen)
PALETTE = {
    "mosaicopt":   "#C1121F",   # Crimson (ours)
    "postfilter":  "#780000",
    "prefilter":   "#E5C595",   # Tan
    "navix":       "#003049",   # Dark blue
    "acorn":       "#669BBC",   # Light blue
    "bvb_native":  "#8D99AE",
    # Substrate comparison:
    "duckdb":      "#003049",
    "postgres":    "#C1121F",
}

MARKER = {
    "mosaicopt":   "o",
    "postfilter":  "v",
    "prefilter":   "^",
    "navix":       "s",
    "acorn":       "x",
    "bvb_native":  "D",
    "duckdb":      "o",
    "postgres":    "s",
}

LINESTYLE = {
    "duckdb":      "-",
    "postgres":    "--",
}

LABEL = {
    "mosaicopt":   "MosaicOpt",
    "postfilter":  "post-filter",
    "prefilter":   "pre-filter",
    "navix":       "NaviX",
    "acorn":       "ACORN-$\\sigma$",
    "bvb_native":  "BVB-native",
    "duckdb":      "DuckDB",
    "postgres":    "PostgreSQL",
}
