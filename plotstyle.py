"""
plotstyle.py
============

Shared IEEE two-column publication style for every figure script in this study.

Import this module FIRST (before pyplot is used) in any figure script:

    import plotstyle          # sets matplotlib rcParams + selects the serif font
    from plotstyle import ALGO_COLORS, ALGO_LS, SERIES, savefig_both, new_fig, FIGDIR

Design goals
------------
* Single-column IEEE width = 3.5 in; 300 dpi; serif body font metrically
  compatible with Times (Nimbus Roman on Linux; falls back gracefully).
* Colours are colourblind-safe (Okabe-Ito) AND each series also carries a
  distinct linestyle, so the figures survive black-and-white printing.
* One canonical colour/linestyle per algorithm, reused across EVERY figure, so
  "same colour == same algorithm" everywhere in the paper.

The selected font is printed at import so a silent DejaVu fallback is visible.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =============================================================================
# FONT SELECTION  (Times New Roman is rarely installed on Linux)
# =============================================================================
# Preference order: Nimbus Roman (metrically identical to Times), then the
# other Times-like/serif options. Whatever is actually installed wins; we print
# the choice so a silent fallback to DejaVu Serif is never invisible.
_SERIF_PREFERENCE = [
    "Times New Roman",   # if the user happens to have the real thing
    "Nimbus Roman",      # URW clone, metrically compatible with Times
    "Liberation Serif",  # RedHat clone, also Times-metric-compatible
    "DejaVu Serif",      # matplotlib's bundled fallback (last resort)
]


def _select_serif():
    """Return the best available serif family from the preference list."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _SERIF_PREFERENCE:
        if name in installed:
            return name, installed
    return "DejaVu Serif", installed          # guaranteed to exist


SELECTED_SERIF, _INSTALLED = _select_serif()

_is_fallback = SELECTED_SERIF == "DejaVu Serif"
print(f"[plotstyle] serif font selected: '{SELECTED_SERIF}'"
      + ("  (Times-metric-compatible)" if SELECTED_SERIF in
         ("Times New Roman", "Nimbus Roman", "Liberation Serif")
         else "  <-- FALLBACK, not Times-compatible; consider `apt install "
              "fonts-liberation`"))
if _is_fallback:
    print("[plotstyle] WARNING: fell back to DejaVu Serif; install "
          "fonts-liberation for Times-compatible metrics.")


# =============================================================================
# rcParams  (IEEE single-column: 3.5 in wide)
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    # put the selected family first, keep the rest as explicit fallbacks
    "font.serif": [SELECTED_SERIF, "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.figsize": (3.5, 2.62),
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # embed real (Type-1/TrueType) fonts in the PDF, never bitmaps:
    "pdf.fonttype": 42,          # TrueType (Type-42) glyphs
    "ps.fonttype": 42,
})


# =============================================================================
# CANONICAL PALETTE  (Okabe-Ito, colourblind-safe) + LINESTYLES
# =============================================================================
# Same colour AND linestyle per algorithm across every figure.
ALGO_COLORS = {
    "PSO":  "#0072B2",   # blue
    "GWO":  "#D55E00",   # vermillion
    "ACOR": "#009E73",   # bluish green
}
ALGO_LS = {
    "PSO":  "-",
    "GWO":  "--",
    "ACOR": "-.",
}
ALGO_ORDER = ["PSO", "GWO", "ACOR"]

# Extra named series that appear in the H-infinity / tracking figures. GWO keeps
# its canonical colour so it reads the same as in the optimizer figures.
SERIES = {
    "Hinf": {"color": "#CC79A7", "ls": "-",  "label": r"H-$\infty$"},   # reddish purple
    "GWO":  {"color": ALGO_COLORS["GWO"], "ls": ALGO_LS["GWO"], "label": "GWO-best"},
    "PD":   {"color": "#555555", "ls": ":",  "label": "PD-only"},        # gray
    "ref":  {"color": "#000000", "ls": (0, (1, 1)), "label": r"$q_{\mathrm{ref}}$"},  # black dotted
}


# =============================================================================
# OUTPUT HELPERS
# =============================================================================
# Relative to the current working directory by default; override with the
# FIGDIR environment variable (e.g. `FIGDIR=/scratch/figs python3 make_figures.py`)
# so nothing in this repo depends on a machine-specific absolute path.
FIGDIR = os.environ.get("FIGDIR", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def new_fig(width=3.5, height=2.62, nrows=1, **kw):
    """A figure sized in inches; width is capped at the 3.5 in IEEE column."""
    width = min(width, 3.5)
    return plt.subplots(nrows, 1, figsize=(width, height), **kw)


def savefig_both(fig, stem):
    """Save `fig` as both a vector PDF and a 300-dpi PNG in FIGDIR.

    Returns the (pdf_path, png_path) written.
    """
    pdf = os.path.join(FIGDIR, stem + ".pdf")
    png = os.path.join(FIGDIR, stem + ".png")
    for path in (pdf, png):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf, png
