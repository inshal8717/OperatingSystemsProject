"""
app.py — Virtual Memory Simulation Engine · Streamlit Dashboard
===============================================================
Clean, minimal dashboard.  Sidebar → simulate → unified metrics table
+ two side-by-side charts (fault rate and memory utilization).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core import BasePageReplacementAlgorithm
from standard_algorithms import FIFOAlgorithm, LRUAlgorithm, SecondChanceAlgorithm
from advanced_algorithms import LRUKAlgorithm, MFUIAlgorithm, WSClockAlgorithm


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VM Simulation Engine",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_reference_string(raw: str) -> Tuple[List[int], Optional[str]]:
    """
    Parse a comma-separated string of page IDs into a list of non-negative ints.
    Returns (pages, None) on success or ([], error_message) on failure.
    """
    raw = raw.strip()
    if not raw:
        return [], "Reference string is empty."

    pages: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
        except ValueError:
            return [], f"Token {token!r} is not a valid integer."
        if val < 0:
            return [], f"Page ID {val} is negative; all IDs must be ≥ 0."
        pages.append(val)

    if not pages:
        return [], "No valid page IDs found after parsing."
    return pages, None


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_all(pages: List[int], capacity: int, tau: int) -> Dict[str, BasePageReplacementAlgorithm]:
    """
    Instantiate and drive all six algorithms over the identical reference
    sequence.  Returns a dict mapping algorithm name → finished instance.
    """
    algos: List[BasePageReplacementAlgorithm] = [
        FIFOAlgorithm(capacity),
        LRUAlgorithm(capacity),
        SecondChanceAlgorithm(capacity),
        LRUKAlgorithm(capacity),
        MFUIAlgorithm(capacity),          # MFUI with decay
        WSClockAlgorithm(capacity, tau=tau),
    ]
    for algo in algos:
        for page_id in pages:
            algo.access_page(page_id, is_write=False)
    return {algo.name: algo for algo in algos}


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

_CHART_BG   = "#0f1117"
_BAR_FAULT  = "#e05252"   # red-ish for fault rate
_BAR_UTIL   = "#4a9eff"   # blue for utilization
_TEXT_COLOR = "#c9d1d9"
_TITLE_COLOR = "#f0c040"


def _base_fig(ax: plt.Axes, title: str) -> None:
    """Apply shared dark-theme styling to a single Axes object."""
    ax.set_facecolor("#161b26")
    ax.set_title(title, color=_TITLE_COLOR, fontsize=10, fontfamily="monospace", pad=6)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2d3748")
        spine.set_linewidth(0.5)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_fontsize(9)


def plot_charts(summaries: List[dict]) -> plt.Figure:
    """
    Return a Figure with two side-by-side bar charts:
      left  — Page Fault Rate (%) per algorithm
      right — Memory Utilization Efficiency (%) per algorithm
    """
    names  = [s["Strategy"] for s in summaries]
    frates = [s["Page Fault Rate (%)"] for s in summaries]
    utils  = [s["Memory Utilization (%)"] for s in summaries]
    x      = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    fig.patch.set_facecolor(_CHART_BG)

    # ── Left: fault rate ──
    bars1 = ax1.bar(x, frates, color=_BAR_FAULT, width=0.55, edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=22, ha="right", fontsize=7.5,
                        color=_TEXT_COLOR, fontfamily="monospace")
    ax1.set_ylabel("Fault Rate (%)")
    ax1.set_ylim(0, max(frates) * 1.22 if max(frates) > 0 else 10)
    _base_fig(ax1, "Page Fault Rate (%) per Algorithm")
    for bar, val in zip(bars1, frates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, color=_TEXT_COLOR, fontfamily="monospace")

    # ── Right: memory utilization ──
    bars2 = ax2.bar(x, utils, color=_BAR_UTIL, width=0.55, edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=22, ha="right", fontsize=7.5,
                        color=_TEXT_COLOR, fontfamily="monospace")
    ax2.set_ylabel("Utilization (%)")
    ax2.set_ylim(0, 110)
    _base_fig(ax2, "Memory Utilization Efficiency (%) per Algorithm")
    for bar, val in zip(bars2, utils):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, color=_TEXT_COLOR, fontfamily="monospace")

    plt.tight_layout(pad=1.4)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🖥️ Virtual Memory · Page Replacement Simulator")
    st.caption("Six-algorithm comparative analysis — FIFO · LRU · Clock · LRU-2 · MFUI · WSClock")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        raw_ref = st.text_area(
            "Reference String (comma-separated page IDs)",
            value="1, 2, 3, 2, 1, 5, 2, 1, 6, 2, 5, 6, 3, 1, 3",
            height=90,
        )

        capacity = st.slider("Physical Frame Capacity", min_value=2, max_value=8, value=3)

        tau = st.slider(
            "WSClock Window Size (τ)",
            min_value=1, max_value=20, value=4,
            help="Pages with age > τ and R-bit=0 are outside the working set and eligible for eviction.",
        )

        run_clicked = st.button("▶  Simulate", use_container_width=True)

    # ── Parse input ───────────────────────────────────────────────────────
    pages, err = parse_reference_string(raw_ref)
    if err:
        st.error(f"Input error: {err}")
        return

    # ── Run on button press, persist result in session_state ──────────────
    if run_clicked:
        st.session_state["results"] = run_all(pages, capacity, tau)
        st.session_state["run_cfg"] = (len(pages), len(set(pages)), capacity, tau)

    if "results" not in st.session_state:
        st.info("Set parameters in the sidebar and press **▶ Simulate**.")
        return

    results: Dict[str, BasePageReplacementAlgorithm] = st.session_state["results"]
    n_refs, n_unique, cap, tw = st.session_state["run_cfg"]

    # ── Quick-stat strip ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("References", n_refs)
    c2.metric("Unique Pages", n_unique)
    c3.metric("Frames Allocated", cap)

    st.markdown("---")

    # ── Unified summary table ─────────────────────────────────────────────
    st.subheader("Summary Metrics")
    summaries = [algo.summary() for algo in results.values()]
    df = pd.DataFrame(summaries).set_index("Strategy")

    # Colour-grade the two key numeric columns for quick visual scanning.
    styled_df = (
        df.style
        .background_gradient(subset=["Page Fault Rate (%)"],     cmap="RdYlGn_r", vmin=0, vmax=100)
        .background_gradient(subset=["Memory Utilization (%)"],  cmap="Blues",    vmin=0, vmax=100)
        .format({
            "Page Fault Rate (%)":    "{:.2f}%",
            "Memory Utilization (%)": "{:.2f}%",
        })
    )
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")

    # ── Two side-by-side charts ───────────────────────────────────────────
    st.subheader("Comparative Charts")
    fig = plot_charts(summaries)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


if __name__ == "__main__":
    main()