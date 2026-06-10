"""
DESRI Unified Dispatch Report
==============================
Reads the optimized CSV produced by DesriOptimizer.py and generates a single
self-contained HTML report with:
  - All 5 interactive Plotly charts (embedded, no external files)
  - Per-chart AI interpretation via the Claude API
  - Executive summary and recommendations
  - KPI strip, revenue table, and behavioral analysis

The output is one .html file you can open in any browser or send as a report.

USAGE:
    1. Run DesriPipeline.py → DesriOptimizer.py first (or use DesriRun.py)
    2. Set OPTIMIZED_CSV below
    3. Run this script
    4. Open {project}_report_{dates}.html in any browser

REQUIREMENTS:
    pip install pandas numpy plotly requests
"""
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import os
import json
import math
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OPTIMIZED_CSV = r"Enter_path_to_your_optimized_csv_here.csv"
OUTPUT_DIR    = r"Enter_path_to_output_directory_here"

# Claude API key — set in environment (recommended) or paste fallback here
#   PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "Enter_your_Anthropic_API_key_here")
MODEL = "claude-sonnet-4-6" # Most capable Claude model as of June 2024. Adjust if newer models are available.

# Chart appearance — light theme to match DESRI brand
TEMPLATE     = "plotly_white"
CHART_WIDTH  = 1300
CHART_HEIGHT = 550

# ── DESRI brand palette ────────────────────────────────────────────────────────
# Navy = primary brand color (from website header)
# Mid blue = secondary / hover
# All chart lines use colors that read clearly on a white background
DESRI_NAVY   = "#1B3A6B"
DESRI_MID    = "#2A5298"
DESRI_LIGHT  = "#E8EEF8"
DESRI_BORDER = "#D0DAF0"

# Blue-forward palette: navy is the hero (actual), optimal strategies step
# through a blue/teal family so the chart reads as DESRI blue-and-white. Teal
# is reserved for the realistic DA benchmark (the "good" target). Warm orange
# is used ONLY for price/LMP elsewhere, so the strategy lines never compete
# with the price signal.
STRATEGY_CONFIG = {
    "actual": {
        "label": "Actual Dispatch",
        "color": "#1B3A6B", "dash": "solid", "width": 2.5,
    },
    "arbitrage_real_time": {
        "label": "RT Arbitrage (ceiling)",
        "color": "#8FB4E0", "dash": "dot", "width": 1.5,
    },
    "arbitrage_day_ahead": {
        "label": "DA Arbitrage (realistic optimal)",
        "color": "#2A9D8F", "dash": "dash", "width": 1.8,
    },
    "scarcity": {
        "label": "Scarcity / Grid Stress",
        "color": "#5C6F8A", "dash": "dashdot", "width": 1.5,
    },
    "hybrid": {
        "label": "Hybrid",
        "color": "#3E6FB0", "dash": "longdash", "width": 1.5,
    },
}

STRATEGY_NAMES = {
    "arbitrage_real_time": "RT Arbitrage (perfect foresight ceiling)",
    "arbitrage_day_ahead": "DA Arbitrage (realistic optimal)",
    "scarcity":            "Scarcity / Grid Stress",
    "hybrid":              "Hybrid (stress + price)",
}

# ── DESRI logo SVG (black paths → white for dark header use) ──────────────────
# This is the full inline SVG with fill colors changed to white for the header.
DESRI_LOGO_WHITE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 775.51202 225.21468" height="48" width="165" xml:space="preserve"><defs><clipPath id="clipPath18" clipPathUnits="userSpaceOnUse"><path d="M 0,168.911 H 581.634 V 0 H 0 Z"/></clipPath></defs><g transform="matrix(1.3333333,0,0,-1.3333333,0,225.21467)"><g><g clip-path="url(#clipPath18)"><g transform="translate(329.5901,79.5743)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 V -20.304 H -3.463 C -6.986,-5.773 -9.29,-3.853 -21.282,-3.853 h -13.197 v -31.205 c 16.512,0.236 19.059,2.863 19.523,12.18 h 3.78 v -28.079 h -3.78 c -0.464,9.257 -3.011,11.938 -19.523,12.039 v -35.631 h 2.176 c 25.519,0 28.456,5.942 30.316,19.699 H 1.826 L -0.115,-78.389 H -56.42 v 3.84 h 9.997 V -3.853 H -56.42 V 0 Z"/></g><g transform="translate(226.3698,72.9994)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 c -5.08,2.371 -12.591,2.715 -18.324,2.715 h -5.699 v -70.689 h 5.699 c 5.733,0 13.244,0.324 18.324,2.682 9.115,4.257 12.982,13.897 12.982,32.666 C 12.982,-13.912 9.108,-4.238 0,0 m 2.755,-69.665 c -6.117,-1.832 -11.917,-2.162 -21.079,-2.162 h -27.614 v 3.853 h 9.971 V 2.715 h -9.971 v 3.86 h 27.614 c 9.162,0 14.962,-0.357 21.079,-2.169 15.481,-4.622 23.437,-17.132 23.437,-37.032 0,-19.921 -7.956,-32.431 -23.437,-37.039"/></g><g transform="translate(404.2126,77.0143)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 c -5.127,2.284 -12.685,3.584 -19.745,3.584 -16.465,0 -25.876,-7.949 -25.876,-20.507 0,-11.203 6.05,-16.942 16.505,-21.24 0,0 1.731,-0.782 10.684,-4.487 8.906,-3.733 11.082,-8.677 11.082,-15.872 0,-9.33 -5.268,-14.518 -15.191,-14.518 -9.97,0 -17.926,7.35 -20.958,20.642 h -3.355 l 1.711,-20.453 c 5.747,-2.58 12.234,-4.163 20.756,-4.163 16.828,0 28.038,8.629 28.038,23.524 0,12.538 -6.824,17.583 -17.34,21.827 0,0 -6.892,2.796 -10.361,4.251 -7.175,2.917 -10.476,6.623 -10.476,14.559 0,8.238 5.875,12.429 13.373,12.429 9.734,0 15.225,-5.956 17.455,-17.448 l 3.698,0 z"/></g><g transform="translate(457.0245,42.8387)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h -8.317 v 32.877 l 7.236,-0.004 c 8.284,0 17.648,-1.54 17.648,-16.547 C 16.567,4.561 11.045,0 0,0 m 27.851,-23.41 c -1.32,10.083 -4.682,18.368 -16.806,21.249 v 0.239 c 10.684,2.042 18.368,8.524 18.368,19.328 0,10.684 -5.403,19.329 -24.972,19.329 l -34.646,-0.004 v -3.854 h 9.998 v -70.695 h -9.998 v -3.84 H 1.681 v 3.84 h -9.998 v 33.616 h 7.236 c 6.962,0 12.606,-2.161 14.645,-10.324 0.841,-3.242 2.163,-12.126 3.362,-17.288 1.681,-7.323 6.124,-9.844 13.086,-9.844 h 9.484 v 3.722 c -8.763,0.96 -10.565,6.843 -11.645,14.526"/></g><g transform="translate(540.3575,58.5975)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 c -1.851,2.326 -3.196,4.148 -3.828,5.254 v 0.001 c -2.57,4.5 -1.935,10.505 -0.484,15.722 h -7.632 v -74.549 h -9.971 v -3.853 H 9.971 v 3.853 H 0 Z"/></g><g transform="translate(509.3339,100.1634)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 c -6.826,0.03 -45.978,-8.347 -61.439,-11.716 -0.969,-0.211 -0.844,-1.63 0.147,-1.673 26.268,-1.136 52.026,-1.333 77.035,-0.15 0.673,0.032 1.039,0.799 0.644,1.344 C 13.961,-8.846 6.919,-0.03 0,0"/></g><g transform="translate(539.2891,65.4297)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 c 3.387,-5.926 30.218,-35.644 40.866,-47.35 0.667,-0.733 1.834,0.085 1.375,0.964 C 30.091,-23.07 17.383,-0.664 3.854,20.403 3.49,20.97 2.642,20.904 2.367,20.289 0.68,16.513 -3.434,6.008 0,0"/></g><g transform="translate(554.3918,108.7385)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 c 3.439,5.896 15.76,43.991 20.574,59.066 0.301,0.944 -0.991,1.545 -1.523,0.709 C 4.933,37.594 -8.117,15.386 -19.596,-6.864 c -0.309,-0.599 0.172,-1.3 0.842,-1.23 C -14.641,-7.668 -3.486,-5.977 0,0"/></g><g transform="translate(540.6107,87.7858)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 c -2.02,-3.464 -6.466,-4.635 -9.93,-2.614 -3.465,2.02 -4.635,6.466 -2.615,9.93 2.021,3.464 6.467,4.635 9.931,2.614 C 0.85,7.91 2.02,3.464 0,0"/></g><g transform="translate(137.9898,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(119.2419,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(155.805,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(128.8489,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g><g transform="translate(124.0854,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(105.3374,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(141.9005,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(114.9445,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g><g transform="translate(110.1809,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(91.433,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(127.9961,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(101.0401,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g><g transform="translate(92.2733,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(73.5254,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(110.0885,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(83.1324,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g><g transform="translate(78.3689,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(59.6209,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(96.184,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(69.228,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g><g transform="translate(64.4644,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(45.7165,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(82.2796,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(55.3235,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g><g transform="translate(46.5568,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(27.8089,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(64.372,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(37.4159,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g><g transform="translate(32.6524,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(13.9044,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(50.4675,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(23.5115,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g><g transform="translate(18.7479,41.3787)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.902 l 8.209,17.603 H 8.208 Z"/></g><g transform="translate(0,1.1725)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="m 0,0 h 11.901 l 8.675,18.603 H 8.674 Z"/></g><g transform="translate(36.5631,79.5847)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 -8.675,-18.603 H 3.228 L 11.903,0 Z"/></g><g transform="translate(9.6071,21.7756)"><path style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none" d="M 0,0 H 11.902 L 20.11,17.603 H 8.208 Z"/></g></g></g></g></svg>"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def call_claude(system_prompt, user_prompt):
    if not ANTHROPIC_API_KEY:
        return "<p><em>Claude API key not set — set ANTHROPIC_API_KEY environment variable to enable AI interpretation.</em></p>"
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model":    MODEL,
        "max_tokens": 1200,
        "system":   system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=body, timeout=120,
        )
        if resp.status_code != 200:
            return f"<p><em>Claude API error {resp.status_code}: {resp.text[:300]}</em></p>"
        return resp.json()["content"][0]["text"]
    except Exception as e:
        return f"<p><em>Claude API call failed: {e}</em></p>"


def fmt_dollars(val):
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"${val:,.0f}"
    return f"${val:.2f}"


def fig_to_html(fig, div_id=None):
    # A stable div_id lets the page's vanilla JS find each plot to (a) resize it
    # when its tab becomes visible / on full-screen toggle / window resize, and
    # (b) relayout its x-range from the shared date-range picker.
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
        div_id=div_id,
    )


# Shared chart layout defaults for the light DESRI theme
# Legend styling kept separate from CHART_LAYOUT so per-chart calls can merge
# in orientation/position without passing `legend` twice to update_layout().
LEGEND_STYLE = dict(
    bgcolor="white",
    bordercolor=DESRI_BORDER,
    borderwidth=1,
    font=dict(color="#1B3A6B", size=11),
)

# Horizontal legend across the top, with the shared styling merged in.
LEGEND_TOP = dict(
    LEGEND_STYLE,
    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
)

# Horizontal legend BELOW the plot. Multi-series time charts (Chart 1, Chart 3)
# put their legend here so it can never collide with the title or the subplot
# titles at the top — the overlap problem with LEGEND_TOP. Pair it with extra
# bottom margin in the finalize() call so the legend has room.
LEGEND_BOTTOM = dict(
    LEGEND_STYLE,
    orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0,
)

# Base layout shared by every chart. We deliberately keep ONLY keys that no
# chart ever overrides individually. `legend`, `xaxis`, `yaxis`, and `title`
# are NOT set here: charts pass those explicitly, and having them in both this
# dict and an explicit keyword would crash update_layout() with a duplicate
# keyword argument (this was the original Chart 4 failure). Axis grid/line
# colors are applied uniformly in finalize() via update_xaxes/update_yaxes,
# which merges cleanly with any per-chart axis settings.
CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#F7F9FC",
    font=dict(family="Segoe UI, system-ui, sans-serif", color=DESRI_NAVY, size=12),
)

AXIS_STYLE = dict(gridcolor=DESRI_BORDER, linecolor=DESRI_BORDER, zerolinecolor=DESRI_BORDER)


def finalize(fig, *, legend=None, div_id=None, **layout):
    """Apply the shared DESRI theme to a figure and return its embedded HTML.

    Collapses the repeated update_layout / update_xaxes / update_yaxes /
    fig_to_html boilerplate that lived in every chart block. Pass per-chart
    options (title, height, width, hovermode, yaxis_title, etc.) as keywords;
    pass a legend dict via `legend=` only when the chart shows one.
    """
    opts = dict(
        CHART_LAYOUT,
        template=TEMPLATE,
        margin=dict(l=60, r=40, t=80, b=40),
    )
    if legend is not None:
        opts["legend"] = legend
    opts.update(layout)               # per-chart overrides win
    fig.update_layout(**opts)
    fig.update_xaxes(**AXIS_STYLE)     # merges with any per-chart axis settings
    fig.update_yaxes(**AXIS_STYLE)
    return fig_to_html(fig, div_id=div_id)

CHART_SYSTEM = """You are an expert energy analyst specializing in battery energy storage systems
(BESS) and electricity market optimization. You work for DESRI, a renewable energy developer.
You are writing concise chart interpretations for a professional dispatch analysis report.

Rules:
- Use the exact numbers given. Do not estimate or round differently.
- Return exactly 3 ultra-concise bullets. Each is a terse fragment, NOT a full
  sentence — max ~12 words, no filler, no preamble. Lead with the number/finding.
- Think headline, not prose. Example style: "<strong>$1.2M/yr</strong> leakage vs DA optimal."
- Write in HTML using ONLY <ul> and <li> tags, with <strong> for key figures.
  Do NOT use <p>, headers, or any other tags. Start with <ul>, end with </ul>.
- Audience: utility executives and DESRI analysts."""


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("DESRI Unified Dispatch Report")
print("="*60)

print("\nLoading optimized CSV...")
df = pd.read_csv(OPTIMIZED_CSV)
df["time"]       = pd.to_datetime(df["time"], utc=True)
df["time_local"] = df["time"].dt.tz_convert("America/Los_Angeles")
df["hour"]       = df["time_local"].dt.hour
df["date"]       = df["time_local"].dt.date

project   = df["project"].iloc[0]
start_str = str(df["time_local"].min().date())
end_str   = str(df["time_local"].max().date())
n_days    = (df["time_local"].max() - df["time_local"].min()).days + 1
cap_mw    = float(df["battery_mw"].iloc[0])
cap_mwh   = float(df["battery_mwh"].iloc[0])
rte       = float(df["rte"].iloc[0])
max_cy    = float(df["max_cycles_year"].iloc[0])

print(f"  {project}  |  {start_str} to {end_str}  |  {n_days} days")

# ── Revenue ────────────────────────────────────────────────────────────────────
rev = {}
if "actual_revenue_usd" in df.columns:
    rev["actual"] = df["actual_revenue_usd"].sum()
for key in STRATEGY_NAMES:
    col = f"optimal_{key}_revenue_usd"
    if col in df.columns:
        rev[key] = df[col].sum()

if not [k for k in rev if k != "actual"]:
    raise SystemExit(
        "No optimal strategy revenue columns found in the CSV "
        "(expected optimal_<strategy>_revenue_usd). Run the optimizer first."
    )

best_strategy      = max((k for k in rev if k != "actual"), key=lambda k: rev[k])
actual_rev         = rev.get("actual", 0)
n_days             = max(n_days, 1)  # guard against single-interval/zero spans
leakage_vs_best    = rev[best_strategy] - actual_rev
leakage_per_day    = leakage_vs_best / n_days
leakage_annualized = leakage_per_day * 365

# ── Behavioral stats ───────────────────────────────────────────────────────────
actual_kw       = df["power_kw"].fillna(0)
charging_pct    = (actual_kw < -100).sum() / len(df) * 100
discharging_pct = (actual_kw >  100).sum() / len(df) * 100
idle_pct        = (actual_kw.abs() <= 100).sum() / len(df) * 100
soc_mean        = df["soc_pct"].mean()
lmp_rt_mean     = df["lmp_rt"].mean()
lmp_rt_max      = df["lmp_rt"].max()

high_lmp_threshold  = df["lmp_rt"].quantile(0.75)
discharge_intervals = df[df["power_kw"] > 1000]
discharge_at_high_lmp = (
    (discharge_intervals["lmp_rt"] > high_lmp_threshold).mean() * 100
    if len(discharge_intervals) > 0 else 0
)

top_lmp_intervals = df.nlargest(50, "lmp_rt")
soc_during_peaks  = top_lmp_intervals["soc_pct"].mean()

# ── Strategy match scores ──────────────────────────────────────────────────────
match_scores = {}
for key in STRATEGY_NAMES:
    opt_col = f"optimal_{key}_kw"
    if opt_col in df.columns:
        mask = actual_kw.notna() & df[opt_col].notna()
        if mask.sum() > 10:
            match_scores[key] = float(np.corrcoef(actual_kw[mask], df[opt_col][mask])[0, 1])

best_match = max(match_scores, key=lambda k: match_scores[k]
                 if not np.isnan(match_scores[k]) else -999) if match_scores else None

# ── Available strategies ───────────────────────────────────────────────────────
available = [s for s in STRATEGY_CONFIG if s != "actual"
             and f"optimal_{s}_kw" in df.columns]
mw = 1 / 1000
DT_HOURS = 1 / 12   # 5-minute interval expressed in hours


# ══════════════════════════════════════════════════════════════════════════════
# TOP METRICS — energy throughput, RTE, equivalent full cycles  (Change #2)
# ══════════════════════════════════════════════════════════════════════════════
# All derived from the actual net power column. Convention (stated in the
# glossary): power_kw is net, discharge positive / charge negative.
#   MWh per interval = power_kw / 1000 * dt
discharged_mwh = float((actual_kw.clip(lower=0) / 1000 * DT_HOURS).sum())
charged_mwh    = float((-actual_kw.clip(upper=0) / 1000 * DT_HOURS).sum())  # >= 0
mwh_per_day    = discharged_mwh / n_days

# Equivalent full cycles = total energy discharged / usable energy capacity.
# (Discharge-throughput convention — one full cycle = one battery-capacity worth
#  of energy delivered. Stated in the glossary.)
eq_cycles       = discharged_mwh / cap_mwh if cap_mwh else 0.0
cycles_per_day  = eq_cycles / n_days
cycles_per_year = cycles_per_day * 365
cycle_util_pct  = (cycles_per_year / max_cy * 100) if max_cy else 0.0
cycle_over_warranty = cycles_per_year > max_cy if max_cy else False


# ══════════════════════════════════════════════════════════════════════════════
# CHARGING vs DISCHARGING LMP  (Change #5)
# ══════════════════════════════════════════════════════════════════════════════
# Classify each interval by SOC slope (rising = charging, falling = discharging),
# with a small dead-band so flat/idle intervals are excluded. power_kw sign is
# used only as a cross-check (reported for QA), per the glossary rule.
SOC_SLOPE_DEADBAND = 0.05   # %-SOC change per 5-min below which we call it idle
soc_slope          = df["soc_pct"].diff()
charging_mask      = soc_slope >  SOC_SLOPE_DEADBAND
discharging_mask   = soc_slope < -SOC_SLOPE_DEADBAND

def _safe_mean(series):
    return float(series.mean()) if len(series) and series.notna().any() else float("nan")

lmp_when_charging    = _safe_mean(df.loc[charging_mask, "lmp_rt"])
lmp_when_discharging = _safe_mean(df.loc[discharging_mask, "lmp_rt"])
cd_spread            = (lmp_when_discharging - lmp_when_charging
                        if not (math.isnan(lmp_when_charging) or math.isnan(lmp_when_discharging))
                        else float("nan"))

# How well does the timing match the ideal (charge cheap, discharge dear)?
median_lmp = float(df["lmp_rt"].median())
chg_below_median = (df.loc[charging_mask, "lmp_rt"] < median_lmp).mean() * 100 \
    if charging_mask.any() else float("nan")
dis_above_median = (df.loc[discharging_mask, "lmp_rt"] > median_lmp).mean() * 100 \
    if discharging_mask.any() else float("nan")

# Cross-check: agreement between SOC-slope class and power-sign class.
_pwr_chg = (df["power_kw"] < -100)
_pwr_dis = (df["power_kw"] >  100)
cd_crosscheck_pct = float(
    ((charging_mask & _pwr_chg) | (discharging_mask & _pwr_dis)).sum()
    / max((charging_mask | discharging_mask).sum(), 1) * 100
)


# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL DRIVERS — three models compared  (Change #6)
# ══════════════════════════════════════════════════════════════════════════════
# Primary source: the optimizer's sidecar JSON (Decision 2). If it is missing
# (e.g. report run standalone on an old CSV), recompute Models 1 & 2 directly
# from the CSV columns so the Analysis tab never shows a blank; Model 3 (the
# multivariate logistic model) is only available from the sidecar.
DRIVER_SIGNALS = ["lmp_rt", "lmp_da", "irradiance_wm2", "grid_stress_mwh"]
DRIVER_LABELS = {
    "lmp_rt":          "RT price (LMP)",
    "lmp_da":          "DA price (LMP)",
    "irradiance_wm2":  "Irradiance (solar)",
    "grid_stress_mwh": "Grid stress",
    "soc_pct":         "State of charge",
    "hour_sin":        "Hour-of-day",
    "hour_cos":        "Hour-of-day",
}

def _report_safe_corr(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return float("nan")
    aa, bb = a[m], b[m]
    if aa.std() == 0 or bb.std() == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])

def _load_drivers_sidecar():
    path = (OPTIMIZED_CSV[:-4] if OPTIMIZED_CSV.lower().endswith(".csv")
            else OPTIMIZED_CSV) + "_drivers.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), path
    except Exception:
        return None, path

drivers_sidecar, drivers_path = _load_drivers_sidecar()
if drivers_sidecar:
    print(f"  Loaded behavioral-drivers sidecar: {os.path.basename(drivers_path)}")
    model1 = drivers_sidecar.get("model1_pearson", {})
    model2 = drivers_sidecar.get("model2_regime", {})
    model3 = drivers_sidecar.get("model3_logistic")
else:
    print("  No drivers sidecar found — recomputing Models 1 & 2 in-report.")
    model1 = {c: _report_safe_corr(df["power_kw"], df[c])
              for c in DRIVER_SIGNALS if c in df.columns}
    price_hi_rep = df["lmp_rt"].quantile(0.75)
    hp = df["lmp_rt"] > price_hi_rep
    hs = df["soc_pct"] > 70
    model2 = {c: {"all":      _report_safe_corr(df["power_kw"],     df[c]),
                  "hi_price": _report_safe_corr(df["power_kw"][hp], df[c][hp]),
                  "hi_soc":   _report_safe_corr(df["power_kw"][hs], df[c][hs])}
              for c in DRIVER_SIGNALS if c in df.columns}
    model3 = None


def _is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

def fmt_corr(x):
    return "n/a" if _is_nan(x) else f"{x:+.3f}"

def fmt_lmp(x):
    return "n/a" if _is_nan(x) else f"${x:.2f}"

def fmt_pct1(x):
    return "n/a" if _is_nan(x) else f"{x:.1f}%"

def _strongest_driver(corr_map):
    """(signal, corr) with the largest absolute correlation, ignoring NaN."""
    best, best_abs = None, -1.0
    for sig, c in corr_map.items():
        if not _is_nan(c) and abs(c) > best_abs:
            best, best_abs = sig, abs(c)
    return best, (corr_map.get(best) if best else float("nan"))


def build_drivers_summary():
    """
    Compare the three driver models and return (verdict_html, table_html).
    Deterministic, data-driven text (not AI) — this is a headline insight, so it
    must always render with the exact numbers. Explicitly calls out irradiance.
    """
    m3_ok = isinstance(model3, dict) and "coefs" in model3 and model3["coefs"]
    auc   = model3.get("auc") if isinstance(model3, dict) else None

    # Model 1 / Model 2 headline drivers
    m1_sig, m1_corr = _strongest_driver(model1) if model1 else (None, float("nan"))

    # Largest regime-conditioned correlation across all drivers (Model 2)
    m2_best = (None, None, float("nan"))   # (signal, regime, corr)
    for sig, regimes in (model2 or {}).items():
        for regime, c in regimes.items():
            if regime == "all" or _is_nan(c):
                continue
            if abs(c) > (abs(m2_best[2]) if not _is_nan(m2_best[2]) else -1):
                m2_best = (sig, regime, c)

    # ── Decide which model is most informative ──────────────────────────────
    if m3_ok and auc is not None and not math.isnan(auc) and auc >= 0.60:
        chosen = "model3"
    elif (not _is_nan(m2_best[2]) and not _is_nan(m1_corr)
          and abs(m2_best[2]) - abs(m1_corr) > 0.10):
        chosen = "model2"
    else:
        chosen = "model1"

    # ── Irradiance role ─────────────────────────────────────────────────────
    irr_line = ""
    if m3_ok:
        coefs = model3["coefs"]
        irr = next(((i, feat, coef) for i, (feat, coef) in enumerate(coefs)
                    if feat == "irradiance_wm2"), None)
        if irr:
            rank, _, coef = irr
            direction = "less" if coef < 0 else "more"
            irr_line = (f"<strong>Irradiance</strong> ranks #{rank+1} of {len(coefs)} "
                        f"drivers in the multivariate model (standardized coef "
                        f"{coef:+.2f}): the battery discharges <strong>{direction}</strong> "
                        f"as on-site solar rises — a behavioral tell that dispatch is "
                        f"shaped by the co-located PV profile, not price alone.")
    if not irr_line:
        irr_c = model1.get("irradiance_wm2") if model1 else None
        if not _is_nan(irr_c):
            sign = "negatively" if irr_c < 0 else "positively"
            irr_line = (f"<strong>Irradiance</strong> correlates {sign} with dispatch "
                        f"(r = {fmt_corr(irr_c)}), indicating solar output is a "
                        f"measurable driver of when the battery charges and discharges.")
        else:
            irr_line = ("<strong>Irradiance</strong> is included as a candidate driver; "
                        "its correlation was not estimable on this sample.")

    # ── Verdict sentence per chosen model ───────────────────────────────────
    if chosen == "model3":
        top3 = ", ".join(f"{DRIVER_LABELS.get(f, f)} ({c:+.2f})"
                         for f, c in model3["coefs"][:3])
        auc_txt = f" (holdout AUC {auc:.2f})" if auc is not None and not math.isnan(auc) else ""
        verdict = (f"The <strong>multivariate driver model (Model 3)</strong> best explains "
                   f"the utility's dispatch{auc_txt}. Ranked by influence, the strongest "
                   f"predictors of the discharge decision are: {top3}.")
    elif chosen == "model2":
        verdict = (f"The <strong>regime-conditioned correlation (Model 2)</strong> is most "
                   f"informative: dispatch aligns with <strong>"
                   f"{DRIVER_LABELS.get(m2_best[0], m2_best[0])}</strong> far more strongly "
                   f"in the {m2_best[1].replace('_',' ')} regime "
                   f"(r = {fmt_corr(m2_best[2])}) than overall — the relationship is "
                   f"conditional, not constant.")
    else:
        verdict = (f"The <strong>plain correlation (Model 1)</strong> is the clearest read "
                   f"on this sample: actual dispatch tracks <strong>"
                   f"{DRIVER_LABELS.get(m1_sig, m1_sig) if m1_sig else 'n/a'}</strong> most "
                   f"closely (r = {fmt_corr(m1_corr)}).")

    verdict_html = f"<p>{verdict}</p><p>{irr_line}</p>"

    # ── Comparison table ────────────────────────────────────────────────────
    def _badge(active):
        return ' <span class="model-pick">◄ most informative</span>' if active else ""

    m1_cell = (f"{DRIVER_LABELS.get(m1_sig, m1_sig)} (r = {fmt_corr(m1_corr)})"
               if m1_sig else "n/a")
    m2_cell = (f"{DRIVER_LABELS.get(m2_best[0], m2_best[0])} in "
               f"{m2_best[1].replace('_',' ')} (r = {fmt_corr(m2_best[2])})"
               if m2_best[0] else "n/a")
    if m3_ok:
        auc_disp = f"{auc:.2f}" if auc is not None and not math.isnan(auc) else "n/a"
        top_feat = model3["coefs"][0][0]
        m3_cell = f"{DRIVER_LABELS.get(top_feat, top_feat)} top; holdout AUC {auc_disp}"
    elif isinstance(model3, dict) and model3.get("error"):
        m3_cell = f"unavailable ({model3['error']})"
    else:
        m3_cell = "unavailable (run optimizer for sidecar)"

    table_html = f"""
      <table class="model-table">
        <thead><tr><th>Model</th><th>What it measures</th><th>Headline result</th></tr></thead>
        <tbody>
          <tr><td><strong>1 · Plain correlation</strong>{_badge(chosen=='model1')}</td>
              <td>Pearson of dispatch vs each driver</td><td>{m1_cell}</td></tr>
          <tr><td><strong>2 · Regime-conditioned</strong>{_badge(chosen=='model2')}</td>
              <td>Same, sliced by price/SOC regime</td><td>{m2_cell}</td></tr>
          <tr><td><strong>3 · Multivariate logistic</strong>{_badge(chosen=='model3')}</td>
              <td>Joint feature importance on the discharge decision</td><td>{m3_cell}</td></tr>
        </tbody>
      </table>"""
    return verdict_html, table_html

drivers_verdict_html, drivers_table_html = build_drivers_summary()


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — DISPATCH COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding charts...")
print("  Chart 1: Dispatch comparison")

fig1 = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.06,
    subplot_titles=["Battery Dispatch: Actual vs Optimal Strategies (MW)", "RT LMP ($/MWh)"],
)

fig1.add_trace(go.Scatter(
    x=df["time_local"], y=df["power_kw"] * mw,
    name=STRATEGY_CONFIG["actual"]["label"],
    line=dict(color=STRATEGY_CONFIG["actual"]["color"],
              width=STRATEGY_CONFIG["actual"]["width"]),
    hovertemplate="%{x}<br>Actual: %{y:.1f} MW<extra></extra>",
), row=1, col=1)

for name in available:
    cfg = STRATEGY_CONFIG[name]
    fig1.add_trace(go.Scatter(
        x=df["time_local"], y=df[f"optimal_{name}_kw"] * mw,
        name=cfg["label"],
        line=dict(color=cfg["color"], width=cfg["width"], dash=cfg["dash"]),
        hovertemplate=f"%{{x}}<br>{cfg['label']}: %{{y:.1f}} MW<extra></extra>",
        opacity=0.85,
    ), row=1, col=1)

fig1.add_hline(y=0, line_width=0.5, line_color=DESRI_BORDER, row=1, col=1)
fig1.add_trace(go.Scatter(
    x=df["time_local"], y=df["lmp_rt"], name="RT LMP",
    line=dict(color="#E05A2B", width=1),
    fill="tozeroy", fillcolor="rgba(224,90,43,0.08)",
    hovertemplate="%{x}<br>RT LMP: $%{y:.2f}/MWh<extra></extra>",
), row=2, col=1)

html_chart1 = finalize(
    fig1,
    div_id="chart1",
    legend=LEGEND_BOTTOM,
    height=CHART_HEIGHT + 170, width=CHART_WIDTH,
    hovermode="x unified",
    yaxis_title="Power (MW)  [+ = discharge, − = charge]",
    yaxis2_title="$/MWh",
    margin=dict(l=70, r=40, t=70, b=90),
)

print("  Requesting AI interpretation for Chart 1...")
rev_table = "\n".join(
    f"  {STRATEGY_NAMES.get(k, k)}: {fmt_dollars(rev[k])}"
    for k in rev if k != "actual"
)
interp1 = call_claude(CHART_SYSTEM, f"""
Interpret this battery dispatch comparison chart for the project {project}.

Key data:
- Period: {start_str} to {end_str} ({n_days} days)
- Battery: {cap_mw} MW / {cap_mwh} MWh, RTE {rte*100:.1f}%
- Actual revenue: {fmt_dollars(actual_rev)}
- Best optimal strategy: {STRATEGY_NAMES[best_strategy]} at {fmt_dollars(rev[best_strategy])}
- Revenue leakage: {fmt_dollars(leakage_vs_best)} ({fmt_dollars(leakage_per_day)}/day)
- Actual dispatch profile: {discharging_pct:.1f}% discharging, {charging_pct:.1f}% charging, {idle_pct:.1f}% idle
- % of discharge events during top-quartile LMP hours: {discharge_at_high_lmp:.1f}%
- Mean SOC during top 50 price spikes: {soc_during_peaks:.1f}%
- Closest strategy match: {STRATEGY_NAMES.get(best_match, 'n/a')} (corr: {match_scores.get(best_match, float('nan')):.3f})

Strategy revenues:
{rev_table}

The chart shows actual dispatch vs all optimal strategies over time, with RT LMP below.
Focus on: how well does actual dispatch track optimal? When does it diverge most?
What does the LMP panel reveal about missed vs captured opportunities?
""")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — REVENUE WATERFALL
# ══════════════════════════════════════════════════════════════════════════════

print("  Chart 2: Revenue waterfall")

rev_items = [("actual", rev.get("actual", 0))]
for name in sorted((k for k in rev if k != "actual"), key=lambda k: rev[k]):
    rev_items.append((name, rev[name]))

labels  = [STRATEGY_CONFIG[n]["label"] for n, _ in rev_items]
values  = [v for _, v in rev_items]
colors  = [STRATEGY_CONFIG[n]["color"] for n, _ in rev_items]
gap_lbl = []
for name, val in rev_items:
    if name == "actual":
        gap_lbl.append("Actual")
    else:
        gap = val - actual_rev
        gap_lbl.append(f"+{fmt_dollars(gap)} vs actual")

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=labels, y=values, marker_color=colors,
    marker_line=dict(color="white", width=1),
    text=[fmt_dollars(v) for v in values],
    textposition="outside",
    textfont=dict(color=DESRI_NAVY, size=11),
    customdata=gap_lbl,
    hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<br>%{customdata}<extra></extra>",
))
fig2.add_hline(
    y=actual_rev, line_dash="dash", line_color=DESRI_NAVY, line_width=1.5,
    annotation_text=f"Actual: {fmt_dollars(actual_rev)}",
    annotation_position="top right",
    annotation_font=dict(color=DESRI_NAVY),
)
html_chart2 = finalize(
    fig2,
    div_id="chart2",
    title=dict(text=f"{project} — Revenue by Dispatch Strategy<br>"
               f"<sup>{start_str} to {end_str}  |  {n_days} days</sup>",
               font=dict(size=15, color=DESRI_NAVY)),
    height=CHART_HEIGHT, width=900,
    yaxis_title="Total Revenue (USD)",
    showlegend=False,
    margin=dict(l=60, r=40, t=80, b=60),
)

print("  Requesting AI interpretation for Chart 2...")
interp2 = call_claude(CHART_SYSTEM, f"""
Interpret this revenue comparison chart for {project}.

Revenue by strategy over {n_days} days ({start_str} to {end_str}):
{chr(10).join(f"  {STRATEGY_NAMES.get(k,'Actual') if k != 'actual' else 'Actual'}: {fmt_dollars(v)}" for k, v in rev.items())}

Revenue leakage vs best strategy ({STRATEGY_NAMES[best_strategy]}): {fmt_dollars(leakage_vs_best)}
Per day: {fmt_dollars(leakage_per_day)}
Annualized estimate: {fmt_dollars(leakage_annualized)}/year

Focus on: How large is the gap? Is it consistent underperformance across all strategies?
What does the annualized leakage imply for the utility relationship?
""")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — SOC + LMP OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

print("  Chart 3: SOC + LMP overlay")

fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Scatter(
    x=df["time_local"], y=df["soc_pct"], name="State of Charge (%)",
    line=dict(color=DESRI_NAVY, width=1.5),
    fill="tozeroy", fillcolor="rgba(27,58,107,0.08)",
    hovertemplate="%{x}<br>SOC: %{y:.1f}%<extra></extra>",
), secondary_y=False)
fig3.add_trace(go.Scatter(
    x=df["time_local"], y=df["lmp_rt"], name="RT LMP ($/MWh)",
    line=dict(color="#E05A2B", width=1.5),
    hovertemplate="%{x}<br>RT LMP: $%{y:.2f}/MWh<extra></extra>",
), secondary_y=True)

neg_lmp = df[df["lmp_rt"] < 0]
if len(neg_lmp) > 0:
    fig3.add_trace(go.Scatter(
        x=pd.concat([neg_lmp["time_local"], neg_lmp["time_local"].iloc[::-1]]),
        y=pd.concat([neg_lmp["lmp_rt"], pd.Series([0] * len(neg_lmp))]),
        fill="toself", fillcolor="rgba(42,157,143,0.15)",
        line=dict(width=0), name="Negative LMP (free charge window)",
        showlegend=True, hoverinfo="skip",
    ), secondary_y=True)

fig3.update_yaxes(title_text="State of Charge (%)", secondary_y=False, range=[0, 110])
fig3.update_yaxes(title_text="RT LMP ($/MWh)", secondary_y=True)
html_chart3 = finalize(
    fig3,
    div_id="chart3",
    legend=LEGEND_BOTTOM,
    title=dict(text=f"{project} — State of Charge vs RT LMP<br>"
               f"<sup>Green shading = negative LMP (grid oversupply, ideal charge window)</sup>",
               font=dict(size=15, color=DESRI_NAVY)),
    height=CHART_HEIGHT, width=CHART_WIDTH, hovermode="x unified",
    margin=dict(l=60, r=60, t=90, b=90),
)

print("  Requesting AI interpretation for Chart 3...")
neg_lmp_pct = (df["lmp_rt"] < 0).sum() / len(df) * 100
interp3 = call_claude(CHART_SYSTEM, f"""
Interpret this State of Charge vs RT LMP overlay chart for {project}.

Key data:
- Mean SOC: {soc_mean:.1f}%
- SOC range: {df["soc_pct"].min():.1f}% to {df["soc_pct"].max():.1f}%
- Mean RT LMP: ${lmp_rt_mean:.2f}/MWh
- Peak RT LMP: ${lmp_rt_max:.2f}/MWh
- Mean SOC during top 50 highest-price intervals: {soc_during_peaks:.1f}%
- Negative LMP intervals (free charging opportunities): {neg_lmp_pct:.1f}% of the period
- Avg RT LMP while CHARGING (SOC rising): {fmt_lmp(lmp_when_charging)}/MWh
- Avg RT LMP while DISCHARGING (SOC falling): {fmt_lmp(lmp_when_discharging)}/MWh
- Charge-to-discharge price spread: {fmt_lmp(cd_spread)}/MWh (positive = buys low, sells high)
- Discharge intervals above median price: {fmt_pct1(dis_above_median)}; charge intervals below median: {fmt_pct1(chg_below_median)}

Focus on: Is the battery available (high SOC) when prices are highest?
Does it charge during negative/low LMP windows and discharge into high prices?
Comment on the charge-vs-discharge price spread and what it says about timing quality.
""")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — HOURLY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

print("  Chart 4: Hourly behavior heatmap")

df["weekday"] = df["time_local"].dt.day_name()
heatmap_data = (
    df.groupby(["date", "hour"])["power_kw"]
    .mean().reset_index()
)
heatmap_pivot = heatmap_data.pivot(index="date", columns="hour", values="power_kw")
heatmap_pivot = heatmap_pivot * mw

def _fmt_day(d):
    try:
        ts = pd.Timestamp(str(d))
        return f"{ts.strftime('%a %b')} {ts.day}"
    except Exception:
        return str(d)

row_labels = [_fmt_day(d) for d in heatmap_pivot.index]
_finite = heatmap_pivot.values[~np.isnan(heatmap_pivot.values)]
max_abs = max(abs(_finite).max(), 1) if _finite.size else 1

fig4 = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=[f"{h:02d}:00" for h in heatmap_pivot.columns],
    y=row_labels,
    # Navy = charging, white = idle, orange = discharging (readable on light bg)
    colorscale=[[0.0, "#1B3A6B"], [0.5, "#F0F4FA"], [1.0, "#E05A2B"]],
    zmid=0, zmin=-max_abs, zmax=max_abs,
    colorbar=dict(
        # Modern Plotly: title font lives under title.font, not `titlefont`.
        title=dict(text="MW", font=dict(color=DESRI_NAVY)),
        tickvals=[-max_abs, 0, max_abs],
        ticktext=[f"−{max_abs:.0f} charge", "0", f"+{max_abs:.0f} discharge"],
        tickfont=dict(color=DESRI_NAVY),
    ),
    hovertemplate="<b>%{y}  %{x}</b><br>Avg power: %{z:.1f} MW<extra></extra>",
))
html_chart4 = finalize(
    fig4,
    div_id="chart4",
    title=dict(text=f"{project} — Dispatch Behavior by Hour of Day<br>"
               f"<sup>Navy = charging  |  Orange = discharging  |  "
               f"Consistent columns reveal time-of-day dispatch rules</sup>",
               font=dict(size=15, color=DESRI_NAVY)),
    height=max(CHART_HEIGHT, len(heatmap_pivot) * 22 + 150),
    width=CHART_WIDTH,
    xaxis_title="Hour of Day (Pacific)",
    yaxis=dict(autorange="reversed"),  # no longer collides with CHART_LAYOUT
    margin=dict(l=100, r=40, t=80, b=60),
)

print("  Requesting AI interpretation for Chart 4...")
hourly_discharge = (
    df.groupby("hour")["power_kw"].mean().sort_values(ascending=False)
)
top_hrs = hourly_discharge.head(3).index.tolist()
interp4 = call_claude(CHART_SYSTEM, f"""
Interpret this hourly dispatch heatmap for {project}.

Key data:
- Period: {n_days} days ({start_str} to {end_str})
- Top 3 discharge hours (Pacific): {top_hrs}
- % of time discharging: {discharging_pct:.1f}%
- % of time charging: {charging_pct:.1f}%
- % of time idle: {idle_pct:.1f}%

Focus on: Does the utility follow a consistent daily pattern or respond dynamically to prices?
Which hours dominate discharge? Does this align with CAISO evening peak periods (hours 17-21)?
""")


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating executive summary...")

exec_system = """You are an expert energy analyst at DESRI, a renewable energy developer.
Write a concise executive summary for a battery dispatch analysis report.
The audience is non-technical utility executives and investors.
Rules:
- Lead with the most important dollar figure.
- 4-5 sentences max.
- No jargon. Plain English. Active voice.
- Write in HTML using only <p> and <strong> tags."""

exec_summary = call_claude(exec_system, f"""
Write an executive summary for this battery dispatch analysis.

Project: {project}
Period: {start_str} to {end_str} ({n_days} days)
Battery: {cap_mw} MW / {cap_mwh} MWh, RTE {rte*100:.1f}%

Actual revenue:           {fmt_dollars(actual_rev)}
Best achievable (DA opt): {fmt_dollars(rev.get('arbitrage_day_ahead', 0))}
Revenue leakage:          {fmt_dollars(leakage_vs_best)} ({fmt_dollars(leakage_per_day)}/day)
Annualized leakage:       {fmt_dollars(leakage_annualized)}/year

Utility dispatch behavior:
- {discharging_pct:.1f}% of time discharging, {charging_pct:.1f}% charging, {idle_pct:.1f}% idle
- {discharge_at_high_lmp:.1f}% of discharge events occurred during top-quartile price hours
- Closest strategy match: {STRATEGY_NAMES.get(best_match, 'n/a')}
  (correlation {match_scores.get(best_match, float('nan')):.3f})
- Mean SOC during top 50 price spikes: {soc_during_peaks:.1f}%

Summarize: what the utility did, what was optimal, the gap, and one key behavioral insight.
""")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIC RECOMMENDATIONS  (closing synthesis — robustness layer)
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating strategic recommendations...")

rec_system = """You are a senior energy strategist at DESRI, a renewable energy developer.
Write a closing 'Strategic Recommendations' section for a battery dispatch report.
The audience is utility executives and DESRI deal teams.
Rules:
- Be specific and actionable; tie each recommendation to the numbers provided.
- Prioritize by dollar impact, most valuable first.
- Provide 3 to 5 recommendations.
- Write in HTML using ONLY <ul> and <li> tags (one <li> per recommendation,
  each starting with a bold action in <strong> tags). No headers, no <p>."""

recommendations = call_claude(rec_system, f"""
Write strategic recommendations for {project}.

Headline numbers ({start_str} to {end_str}, {n_days} days):
- Actual revenue: {fmt_dollars(actual_rev)}
- Best achievable strategy ({STRATEGY_NAMES[best_strategy]}): {fmt_dollars(rev[best_strategy])}
- Revenue leakage: {fmt_dollars(leakage_vs_best)} ({fmt_dollars(leakage_per_day)}/day,
  {fmt_dollars(leakage_annualized)}/year annualized)

Dispatch behavior:
- {discharging_pct:.1f}% discharging, {charging_pct:.1f}% charging, {idle_pct:.1f}% idle
- {discharge_at_high_lmp:.1f}% of discharge events landed in top-quartile price hours
- Mean SOC during top 50 price spikes: {soc_during_peaks:.1f}%
- Closest strategy match: {STRATEGY_NAMES.get(best_match, 'n/a')}
  (correlation {match_scores.get(best_match, float('nan')):.3f})

Recommend concrete operational and contractual changes that would close the leakage gap.
""")

# Robustness: never let a missing key, API error, or empty reply leave a blank
# section. call_claude wraps every failure/placeholder in <em>...</em>, so that
# marker is a reliable signal to substitute a data-driven fallback.
if not recommendations or "<em>" in recommendations:
    recommendations = (
        "<ul>"
        f"<li><strong>Capture the leakage.</strong> Closing the gap to "
        f"{STRATEGY_NAMES[best_strategy]} is worth ~{fmt_dollars(leakage_annualized)}/year; "
        f"prioritize dispatch changes that target the highest-price hours.</li>"
        f"<li><strong>Hold charge for peaks.</strong> Mean SOC during the top 50 price "
        f"spikes was {soc_during_peaks:.1f}% — raise reserved state-of-charge ahead of "
        f"forecast peak windows.</li>"
        f"<li><strong>Discipline discharge timing.</strong> Only {discharge_at_high_lmp:.1f}% "
        f"of discharge events occurred in top-quartile price hours; tighten the price "
        f"threshold that triggers discharge.</li>"
        "</ul>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("\nAssembling HTML report...")

report_date = datetime.now().strftime("%B %d, %Y at %H:%M")

# Revenue table rows
rev_rows = ""
for key, name in STRATEGY_NAMES.items():
    if key not in rev:
        continue
    diff     = rev[key] - actual_rev
    diff_str = f"+{fmt_dollars(diff)}" if diff >= 0 else fmt_dollars(diff)
    best_marker = " ◄" if key == best_strategy else ""
    score_str   = f"{match_scores[key]:.3f}" if key in match_scores else "n/a"
    diff_class  = "num-red" if diff >= 0 else "num-green"
    rev_rows += f"""
      <tr>
        <td>{name}</td>
        <td class="num">{fmt_dollars(rev[key])}</td>
        <td class="num {diff_class}">{diff_str}{best_marker}</td>
        <td class="num">{score_str}</td>
      </tr>"""

rev_rows += f"""
      <tr class="actual-row">
        <td><strong>Actual Dispatch</strong></td>
        <td class="num"><strong>{fmt_dollars(actual_rev)}</strong></td>
        <td class="num">—</td>
        <td class="num">—</td>
      </tr>"""

best_match_corr_str = (
    f"{match_scores.get(best_match, float('nan')):.3f}" if best_match else "n/a"
)

# ── Cycle-warranty flag for the metrics band ─────────────────────────────────
cycle_class = "danger" if cycle_over_warranty else "success"
warn_mark   = " ⚠" if cycle_over_warranty else ""

# ── Charging-vs-discharging behavioral read (bullets) ────────────────────────
if _is_nan(cd_spread):
    cd_read_html = ("<ul><li>Insufficient distinct charge/discharge intervals to "
                    "assess price timing on this sample.</li></ul>")
else:
    quality = ("Charges low, discharges high" if cd_spread > 0
               else "Inverted timing — sells cheaper than it buys")
    cd_read_html = (
        "<ul>"
        f"<li><strong>{quality}</strong>: {fmt_lmp(lmp_when_charging)} vs "
        f"{fmt_lmp(lmp_when_discharging)}/MWh — <strong>{fmt_lmp(cd_spread)}/MWh</strong> spread.</li>"
        f"<li><strong>{fmt_pct1(dis_above_median)}</strong> of discharges above median price; "
        f"<strong>{fmt_pct1(chg_below_median)}</strong> of charges below.</li>"
        f"<li>SOC-slope vs power-sign labels agree <strong>{cd_crosscheck_pct:.0f}%</strong>.</li>"
        "</ul>"
    )

# ── Glossary & Methodology — FIXED, hard-coded text (not AI-generated) ───────
# These definitions and methods do not change run-to-run, so they are authored
# here verbatim rather than sent to Claude.
glossary_html = f"""
  <div class="section-label">Reference</div>
  <div class="section-title">Glossary &amp; Methodology</div>
  <div class="glossary">

    <p>This report compares how the battery <em>actually</em> dispatched against
    several model-optimal dispatch strategies, and explains the behavioral and
    financial gap. Every figure on the Analysis and Graphs tabs is defined below.
    All energy is settled at 5-minute resolution, so one interval = 1/12 hour.</p>

    <h3>Headline metrics</h3>
    <dl>
      <dt>Arbitrage revenue (period &amp; daily)</dt>
      <dd>Energy revenue from buying (charging) and selling (discharging) against
      market price. Per interval: <em>net&nbsp;power&nbsp;(MW) × price&nbsp;($/MWh) ×
      1/12&nbsp;h</em>, summed over the period. Daily figures divide the period total
      by the number of days ({n_days}). Discharge earns revenue; charging is a cost.</dd>

      <dt>Revenue leakage</dt>
      <dd>Best optimal strategy's revenue minus actual revenue — the dollars left
      on the table. Shown for the period, per day, and annualized (per-day × 365).</dd>

      <dt>Energy throughput (MWh)</dt>
      <dd>Total energy moved. Discharged MWh sums the positive part of
      <em>power_kw/1000 × 1/12</em>; charged MWh sums the negative part. Throughput
      drives both revenue opportunity and battery wear.</dd>

      <dt>Average MWh per day</dt>
      <dd>Period discharged MWh ÷ number of days ({n_days}).</dd>

      <dt>Equivalent full cycles</dt>
      <dd>Total discharged MWh ÷ usable energy capacity ({cap_mwh:.0f}&nbsp;MWh). One
      full cycle = one full battery's worth of energy delivered (a
      <em>discharge-throughput</em> convention). We extrapolate to cycles/day and
      cycles/year and compare the annualized figure against the warranty limit of
      {max_cy:.0f} cycles/year.</dd>

      <dt>Round-trip efficiency (RTE)</dt>
      <dd>Fraction of charged energy recoverable on discharge. The pipeline computes
      it as the median of valid Round-Trip-Efficiency numerator/denominator cycle
      observations (clamped to a plausible 50–105% band); here it is {rte*100:.1f}%.
      A single project-level scalar is surfaced (no time distribution).</dd>

      <dt>Charge / discharge classification</dt>
      <dd>Each interval is labeled by the <strong>slope of state-of-charge</strong>:
      SOC rising by more than {SOC_SLOPE_DEADBAND:g} %-points per 5&nbsp;min = charging,
      falling by more than that = discharging, otherwise idle. The sign of
      <em>power_kw</em> (− = charging, + = discharging) is used only as a cross-check.</dd>

      <dt>Correlation &amp; driver scores</dt>
      <dd>Pearson correlation (−1…+1) measures how closely two series move together.
      "Strategy match" correlates actual dispatch against each optimal schedule.
      "Driver" scores correlate actual dispatch against external signals (price,
      irradiance, grid stress). See the model definitions below.</dd>
    </dl>

    <h3>Models &amp; methods</h3>
    <dl>
      <dt>LP dispatch optimizer</dt>
      <dd>A linear program maximizes revenue over the period subject to power limits
      ({cap_mw:.0f}&nbsp;MW), energy limits ({cap_mwh:.0f}&nbsp;MWh), SOC dynamics with
      RTE, and an annual cycle budget pro-rated to the period. It outputs the
      revenue-maximizing charge/discharge schedule for a given price signal and
      discharge-eligibility rule.</dd>

      <dt>The four strategies</dt>
      <dd>
        <strong>RT Arbitrage (ceiling)</strong> — full-foresight solve on real-time
        prices; an unrealistic upper bound.<br>
        <strong>DA Arbitrage (realistic optimal)</strong> — solved in rolling 48-hour
        windows on day-ahead prices with no foresight past the window edge; the
        achievable benchmark.<br>
        <strong>Scarcity / Grid Stress</strong> — may discharge only during
        high grid-stress intervals; a reliability floor.<br>
        <strong>Hybrid</strong> — scarcity reserve plus opportunistic discharge in the
        highest-price intervals; lands between scarcity and arbitrage.
      </dd>

      <dt>Correlation analyses (Models 1 &amp; 2)</dt>
      <dd><strong>Model 1</strong> is the plain Pearson correlation of actual dispatch
      against each driver signal. <strong>Model 2</strong> repeats that correlation
      within regimes — high-price (RT LMP above the 75th percentile) and high-SOC
      (above 70%) — to expose relationships that only hold conditionally.</dd>

      <dt>Driver / feature-importance model (Model 3)</dt>
      <dd>A logistic regression predicts the binary discharge decision
      (|power| above {int(1000)}&nbsp;kW) from standardized market and state features —
      including <strong>irradiance</strong>, price, grid stress, SOC and hour-of-day.
      Standardized coefficients (and SHAP values where available) rank which signals
      most influence the utility's behavior, with an out-of-sample AUC reported for
      honesty. This is the most informative view when it predicts well, because it
      weighs all drivers jointly rather than one at a time.</dd>
    </dl>

    <h3>Data sources</h3>
    <div class="src"><b>PowerFactors</b><span>Battery telemetry — state of charge,
      active power, effective irradiance, and the RTE numerator/denominator signals,
      at 5-minute resolution.</span></div>
    <div class="src"><b>CAISO</b><span>Day-ahead (hourly, forward-filled to 5&nbsp;min)
      and real-time 5-minute locational marginal prices (LMP) at the project node,
      pulled via gridstatus.</span></div>
    <div class="src"><b>EIA-930</b><span>Hourly system demand and demand forecast for
      the balancing authority; their difference defines the grid-stress signal.</span></div>
  </div>
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DESRI Dispatch Report — {project} — {start_str} to {end_str}</title>
<style>
  :root {{
    --navy:     #1B3A6B;
    --mid:      #2A5298;
    --light:    #E8EEF8;
    --border:   #D0DAF0;
    --bg:       #ffffff;
    --surface:  #F7F9FC;
    --surface2: #EEF2FA;
    --text:     #1B3A6B;
    --subtext:  #5A6E8C;
    --red:      #C0392B;
    --green:    #1A7A4A;
    --orange:   #E05A2B;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}

  /* ── Header ── */
  .header {{
    background: var(--navy);
    padding: 32px 64px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 32px;
  }}
  .header-left {{ flex: 1; }}
  .header-eyebrow {{
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6);
    margin-bottom: 8px;
  }}
  .header h1 {{
    font-size: 24px;
    font-weight: 700;
    color: white;
    margin-bottom: 6px;
    letter-spacing: -0.3px;
  }}
  .header-meta {{
    color: rgba(255,255,255,0.65);
    font-size: 12px;
  }}
  .header-logo {{
    flex-shrink: 0;
    opacity: 0.95;
  }}

  /* ── Blue rule under header ── */
  .header-rule {{
    height: 4px;
    background: linear-gradient(90deg, #2A9D8F 0%, #2A5298 50%, #1B3A6B 100%);
  }}

  /* ── KPI strip ── */
  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}
  .kpi {{
    padding: 24px 32px;
    border-right: 1px solid var(--border);
  }}
  .kpi:last-child {{ border-right: none; }}
  .kpi-label {{
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 6px;
    font-weight: 600;
  }}
  .kpi-value {{
    font-size: 26px;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.5px;
  }}
  .kpi-sub {{
    font-size: 11px;
    color: var(--subtext);
    margin-top: 3px;
  }}
  .kpi.danger  .kpi-value {{ color: var(--red);    }}
  .kpi.success .kpi-value {{ color: var(--green);  }}
  .kpi.warn    .kpi-value {{ color: var(--orange); }}

  /* ── Content ── */
  .content {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 48px 64px;
  }}

  /* ── Section labels ── */
  .section-label {{
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--subtext);
    font-weight: 600;
    margin-bottom: 4px;
  }}
  .section-title {{
    font-size: 20px;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--light);
  }}

  /* ── Executive summary ── */
  .exec-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--navy);
    border-radius: 4px;
    padding: 24px 28px;
    margin-bottom: 48px;
    font-size: 15px;
    line-height: 1.8;
    color: var(--text);
  }}
  .exec-box p {{ margin-bottom: 10px; }}
  .exec-box p:last-child {{ margin-bottom: 0; }}
  .exec-box strong {{ color: var(--navy); font-weight: 700; }}

  /* ── Strategic recommendations ── */
  .rec-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--navy);
    border-radius: 4px;
    padding: 24px 28px;
    margin-bottom: 48px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }}
  .rec-text {{ flex: 1; font-size: 14.5px; line-height: 1.75; color: var(--text); }}
  .rec-text ul {{ list-style: none; }}
  .rec-text li {{
    position: relative;
    padding: 8px 0 8px 22px;
    border-bottom: 1px solid var(--light);
  }}
  .rec-text li:last-child {{ border-bottom: none; }}
  .rec-text li::before {{
    content: "";
    position: absolute;
    left: 0; top: 16px;
    width: 7px; height: 7px;
    background: var(--mid);
    border-radius: 2px;
  }}
  .rec-text strong {{ color: var(--navy); font-weight: 700; }}

  /* ── Revenue table ── */
  .rev-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 16px 0 40px;
  }}
  .rev-table th {{
    background: var(--navy);
    color: rgba(255,255,255,0.85);
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 10px 16px;
    text-align: left;
    font-weight: 600;
  }}
  .rev-table td {{
    padding: 11px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  .rev-table tr:hover td {{ background: var(--surface); }}
  .rev-table .num      {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .rev-table .num-red  {{ text-align: right; color: var(--red);   font-weight: 600; }}
  .rev-table .num-green{{ text-align: right; color: var(--green); font-weight: 600; }}
  .rev-table .actual-row td {{
    background: var(--surface2);
    font-weight: 600;
    border-top: 1px solid var(--border);
  }}

  /* ── Chart card ── */
  .chart-card {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 32px;
    box-shadow: 0 1px 4px rgba(27,58,107,0.06);
  }}
  .chart-body {{ padding: 4px; }}
  .chart-body > div {{ width: 100% !important; }}

  /* ── AI interpretation ── */
  .ai-interp {{
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 18px 24px;
    font-size: 13.5px;
    line-height: 1.75;
    display: flex;
    gap: 14px;
    align-items: flex-start;
  }}
  .ai-badge {{
    flex-shrink: 0;
    background: var(--navy);
    color: white;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 8px;
    border-radius: 3px;
    margin-top: 2px;
  }}
  .ai-interp-text p {{ margin-bottom: 8px; color: var(--text); }}
  .ai-interp-text p:last-child {{ margin-bottom: 0; }}
  .ai-interp-text strong {{ color: var(--navy); font-weight: 700; }}

  /* ── Footer ── */
  .footer {{
    background: var(--navy);
    padding: 16px 64px;
    color: rgba(255,255,255,0.55);
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .footer-logo {{ opacity: 0.5; }}

  .section-gap {{ margin-bottom: 48px; }}
</style>
</head>
<body>

<!-- ── HEADER ── -->
<div class="header">
  <div class="header-left">
    <div class="header-eyebrow">Battery Dispatch Analysis · Confidential</div>
    <h1>{project} — Dispatch Performance Report</h1>
    <div class="header-meta">
      {start_str} to {end_str} &nbsp;·&nbsp;
      {n_days} days &nbsp;·&nbsp;
      {cap_mw:.0f} MW / {cap_mwh:.0f} MWh &nbsp;·&nbsp;
      RTE {rte*100:.1f}% &nbsp;·&nbsp;
      Generated {report_date}
    </div>
  </div>
  <div class="header-logo">
    {DESRI_LOGO_WHITE}
  </div>
</div>
<div class="header-rule"></div>

<!-- ── KPI STRIP ── -->
<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Actual Revenue</div>
    <div class="kpi-value">{fmt_dollars(actual_rev)}</div>
    <div class="kpi-sub">{n_days}-day period</div>
  </div>
  <div class="kpi success">
    <div class="kpi-label">DA Optimal (Benchmark)</div>
    <div class="kpi-value">{fmt_dollars(rev.get('arbitrage_day_ahead', 0))}</div>
    <div class="kpi-sub">Realistic achievable ceiling</div>
  </div>
  <div class="kpi danger">
    <div class="kpi-label">Revenue Leakage</div>
    <div class="kpi-value">{fmt_dollars(leakage_vs_best)}</div>
    <div class="kpi-sub">{fmt_dollars(leakage_per_day)}/day · {fmt_dollars(leakage_annualized)}/yr est.</div>
  </div>
  <div class="kpi warn">
    <div class="kpi-label">Closest Strategy Match</div>
    <div class="kpi-value" style="font-size:16px">{STRATEGY_NAMES.get(best_match,'n/a').split('(')[0].strip()}</div>
    <div class="kpi-sub">Corr: {best_match_corr_str}</div>
  </div>
</div>

<!-- ── TOP METRICS BAND (throughput · MWh/day · RTE · cycles) ── -->
<div class="kpi-strip metrics-row">
  <div class="kpi metric">
    <div class="kpi-label">Energy Throughput</div>
    <div class="kpi-value">{discharged_mwh:,.0f} MWh</div>
    <div class="kpi-sub">discharged &nbsp;·&nbsp; {charged_mwh:,.0f} MWh charged</div>
  </div>
  <div class="kpi metric">
    <div class="kpi-label">Avg Throughput / Day</div>
    <div class="kpi-value">{mwh_per_day:,.0f} MWh</div>
    <div class="kpi-sub">discharged per day over {n_days} days</div>
  </div>
  <div class="kpi metric">
    <div class="kpi-label">Round-Trip Efficiency</div>
    <div class="kpi-value">{rte*100:.1f}%</div>
    <div class="kpi-sub">project-level measured RTE</div>
  </div>
  <div class="kpi {cycle_class}">
    <div class="kpi-label">Equivalent Full Cycles</div>
    <div class="kpi-value">{eq_cycles:.1f}</div>
    <div class="kpi-sub">{cycles_per_day:.2f}/day · {cycles_per_year:.0f}/yr proj. vs {max_cy:.0f} warranty{warn_mark}</div>
  </div>
</div>

<!-- ── TAB BAR ── -->
<div class="tabbar" role="tablist">
  <button class="tab-btn active" data-tab="tab-analysis">Analysis</button>
  <button class="tab-btn" data-tab="tab-graphs">Graphs &amp; Interpretation</button>
  <button class="tab-btn" data-tab="tab-glossary">Glossary &amp; Methodology</button>
</div>

<!-- ── CONTENT ── -->
<div class="content">

  <!-- ===================== TAB 1 — ANALYSIS ===================== -->
  <div class="tab-pane active" id="tab-analysis">

    <div class="section-label">Overview</div>
    <div class="section-title">Executive Summary</div>
    <div class="exec-box">{exec_summary}</div>

    <div class="section-label">Financials</div>
    <div class="section-title">Revenue by Strategy</div>
    <table class="rev-table">
      <thead>
        <tr>
          <th>Strategy</th>
          <th style="text-align:right">Total Revenue</th>
          <th style="text-align:right">vs Actual</th>
          <th style="text-align:right">Correlation</th>
        </tr>
      </thead>
      <tbody>{rev_rows}</tbody>
    </table>

    <div class="section-gap"></div>

    <div class="section-label">Price Timing</div>
    <div class="section-title">Charging vs Discharging Prices</div>
    <div class="cd-grid">
      <div class="cd-cell">
        <div class="cd-label">Avg RT LMP while charging</div>
        <div class="cd-value">{fmt_lmp(lmp_when_charging)}/MWh</div>
      </div>
      <div class="cd-cell">
        <div class="cd-label">Avg RT LMP while discharging</div>
        <div class="cd-value">{fmt_lmp(lmp_when_discharging)}/MWh</div>
      </div>
      <div class="cd-cell spread">
        <div class="cd-label">Charge → discharge spread</div>
        <div class="cd-value">{fmt_lmp(cd_spread)}/MWh</div>
      </div>
    </div>
    <div class="insight-box">{cd_read_html}</div>

    <div class="section-gap"></div>

    <div class="section-label">Behavioral Drivers</div>
    <div class="section-title">What Best Explains the Dispatch?</div>
    <div class="insight-box">{drivers_verdict_html}{drivers_table_html}</div>

    <div class="section-gap"></div>

    <div class="section-label">Closing</div>
    <div class="section-title">Strategic Recommendations</div>
    <div class="rec-box">
      <div class="ai-badge">AI</div>
      <div class="rec-text">{recommendations}</div>
    </div>
  </div>

  <!-- ===================== TAB 2 — GRAPHS ===================== -->
  <div class="tab-pane" id="tab-graphs">

    <div class="daterange">
      <label for="dr-start">From</label>
      <input type="date" id="dr-start" value="{start_str}" min="{start_str}" max="{end_str}">
      <label for="dr-end">To</label>
      <input type="date" id="dr-end" value="{end_str}" min="{start_str}" max="{end_str}">
      <span style="font-size:12px;color:var(--subtext)">filters the time-series charts (Dispatch &amp; SOC)</span>
      <div class="dr-presets">
        <button class="dr-btn" data-days="1">1d</button>
        <button class="dr-btn" data-days="3">3d</button>
        <button class="dr-btn" data-days="7">7d</button>
        <button class="dr-btn active" data-days="all">All</button>
      </div>
    </div>

    <div class="section-label">Chart 01</div>
    <div class="section-title">Dispatch Timeline</div>
    <div class="chart-card" id="card-chart1">
      <div class="chart-head">
        <span class="ct-title">Actual vs Optimal Dispatch</span>
        <button class="fs-btn" data-card="card-chart1" data-chart="chart1">Full screen</button>
      </div>
      <div class="chart-body">{html_chart1}</div>
      <div class="ai-interp">
        <div class="ai-badge">AI</div>
        <div class="ai-interp-text">{interp1}</div>
      </div>
    </div>

    <div class="section-label">Chart 02</div>
    <div class="section-title">Revenue Comparison</div>
    <div class="chart-card" id="card-chart2">
      <div class="chart-head">
        <span class="ct-title">Revenue by Strategy</span>
        <button class="fs-btn" data-card="card-chart2" data-chart="chart2">Full screen</button>
      </div>
      <div class="chart-body">{html_chart2}</div>
      <div class="ai-interp">
        <div class="ai-badge">AI</div>
        <div class="ai-interp-text">{interp2}</div>
      </div>
    </div>

    <div class="section-label">Chart 03</div>
    <div class="section-title">State of Charge vs Price</div>
    <div class="chart-card" id="card-chart3">
      <div class="chart-head">
        <span class="ct-title">SOC vs RT LMP</span>
        <button class="fs-btn" data-card="card-chart3" data-chart="chart3">Full screen</button>
      </div>
      <div class="chart-body">{html_chart3}</div>
      <div class="ai-interp">
        <div class="ai-badge">AI</div>
        <div class="ai-interp-text">{interp3}</div>
      </div>
    </div>

    <div class="section-label">Chart 04</div>
    <div class="section-title">Hourly Dispatch Behavior</div>
    <div class="chart-card" id="card-chart4">
      <div class="chart-head">
        <span class="ct-title">Dispatch by Hour of Day</span>
        <button class="fs-btn" data-card="card-chart4" data-chart="chart4">Full screen</button>
      </div>
      <div class="chart-body">{html_chart4}</div>
      <div class="ai-interp">
        <div class="ai-badge">AI</div>
        <div class="ai-interp-text">{interp4}</div>
      </div>
    </div>
  </div>

  <!-- ===================== TAB 3 — GLOSSARY ===================== -->
  <div class="tab-pane" id="tab-glossary">
    {glossary_html}
  </div>

</div>

<!-- ── FOOTER ── -->
<div class="footer">
  <span>DESRI · Battery Dispatch Analysis · Confidential</span>
  <span>Generated {report_date} · Model: {MODEL}</span>
</div>

</body>
</html>"""

# ── Inject new CSS + JS ──────────────────────────────────────────────────────
# Authored as plain strings (single braces) and spliced in, so the large CSS/JS
# blocks don't need every brace doubled for the f-string above. The DESRI theme
# variables (--navy, --border, --surface, ...) are reused — no new colors.
EXTRA_CSS = """
  /* ── Top metrics band (second KPI row) ── */
  .metrics-row { border-top: 1px solid var(--border); }
  .kpi.metric .kpi-value { color: var(--navy); }

  /* ── Tabs ── */
  .tabbar {
    display: flex; gap: 2px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 64px;
  }
  .tab-btn {
    background: transparent; border: none;
    border-bottom: 3px solid transparent;
    padding: 15px 24px;
    font-family: inherit; font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 600;
    color: var(--subtext); cursor: pointer;
  }
  .tab-btn:hover { color: var(--navy); }
  .tab-btn.active { color: var(--navy); border-bottom-color: var(--navy); }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  /* ── Insight cards (drivers summary, charge/discharge read) ── */
  .insight-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--mid);
    border-radius: 4px;
    padding: 22px 26px; margin-bottom: 24px;
  }
  .insight-box > p { margin-bottom: 10px; font-size: 14.5px; color: var(--text); }
  .insight-box > p:last-child { margin-bottom: 0; }
  .insight-box strong { color: var(--navy); font-weight: 700; }
  .insight-box ul { list-style: none; }
  .insight-box li {
    position: relative; padding: 6px 0 6px 20px; color: var(--text);
    border-bottom: 1px solid var(--light); font-size: 14px;
  }
  .insight-box li:last-child { border-bottom: none; }
  .insight-box li::before {
    content: ""; position: absolute; left: 0; top: 13px;
    width: 7px; height: 7px; background: var(--mid); border-radius: 2px;
  }

  /* ── Model comparison table ── */
  .model-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 14px; }
  .model-table th {
    background: var(--navy); color: rgba(255,255,255,0.85);
    text-align: left; padding: 9px 14px;
    font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase; font-weight: 600;
  }
  .model-table td { padding: 10px 14px; border-bottom: 1px solid var(--border); vertical-align: top; color: var(--text); }
  .model-table tr:last-child td { border-bottom: none; }
  .model-pick { color: var(--green); font-weight: 700; font-size: 11px; white-space: nowrap; }

  /* ── Charge/discharge readout cells ── */
  .cd-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }
  .cd-cell { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 18px 20px; }
  .cd-cell .cd-label {
    font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--subtext); font-weight: 600; margin-bottom: 6px;
  }
  .cd-cell .cd-value { font-size: 22px; font-weight: 700; color: var(--navy); letter-spacing: -0.5px; }
  .cd-cell.spread .cd-value { color: var(--orange); }

  /* ── Chart card head + full screen ── */
  .chart-head { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--border); }
  .chart-head .ct-title { font-size: 13px; font-weight: 700; color: var(--navy); letter-spacing: 0.3px; }
  .fs-btn {
    background: var(--navy); color: #fff; border: none; border-radius: 4px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px; padding: 6px 12px; cursor: pointer;
  }
  .fs-btn:hover { background: var(--mid); }
  .chart-card.fullscreen { position: fixed; inset: 0; z-index: 9999; margin: 0; border-radius: 0; overflow: auto; background: #fff; }
  .chart-card.fullscreen .chart-body { padding: 8px; display: flex; justify-content: center; }
  /* Plotly drives the plot size via relayout in fullscreen — don't force it in CSS. */
  /* Hide the bullet interpretation in fullscreen so it doesn't clutter the plot. */
  .chart-card.fullscreen .ai-interp { display: none; }

  /* ── Date-range picker ── */
  .daterange {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 14px 18px; margin-bottom: 28px;
  }
  .daterange label { font-size: 11px; letter-spacing: 1px; text-transform: uppercase; color: var(--subtext); font-weight: 600; }
  .daterange input[type=date] { font-family: inherit; padding: 6px 8px; border: 1px solid var(--border); border-radius: 4px; color: var(--navy); background: #fff; }
  .dr-presets { display: flex; gap: 6px; margin-left: auto; }
  .dr-btn { background: #fff; border: 1px solid var(--border); color: var(--navy); border-radius: 4px; padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer; }
  .dr-btn:hover, .dr-btn.active { background: var(--navy); color: #fff; }

  /* ── Bullet interpretation lists (CHART_SYSTEM now returns <ul>) ── */
  .ai-interp-text ul { list-style: none; margin: 0; padding: 0; }
  .ai-interp-text li { position: relative; padding: 5px 0 5px 18px; color: var(--text); }
  .ai-interp-text li::before { content: ""; position: absolute; left: 0; top: 11px; width: 6px; height: 6px; background: var(--mid); border-radius: 2px; }
  .ai-interp-text li strong { color: var(--navy); font-weight: 700; }
  .ai-interp-text em { color: var(--subtext); }

  /* ── Glossary ── */
  .glossary { font-size: 14px; }
  .glossary h3 { font-size: 15px; color: var(--navy); margin: 26px 0 8px; padding-bottom: 6px; border-bottom: 1px solid var(--light); }
  .glossary p { margin-bottom: 12px; color: var(--text); }
  .glossary dl { margin: 8px 0 18px; }
  .glossary dt { font-weight: 700; color: var(--navy); margin-top: 14px; }
  .glossary dd { margin: 3px 0 0 0; color: var(--text); }
  .glossary .src { display: flex; gap: 10px; margin: 8px 0; }
  .glossary .src b { color: var(--navy); min-width: 110px; display: inline-block; }
"""

TAB_JS = """
<script>
(function(){
  var CHART_IDS = ['chart1','chart2','chart3','chart4'];
  var TIME_CHARTS = ['chart1','chart3'];
  var DATA_START = '__DATA_START__', DATA_END = '__DATA_END__';

  function resizeChart(id){
    var el = document.getElementById(id);
    if(el && window.Plotly){ try { Plotly.Plots.resize(el); } catch(e){} }
  }
  function resizeAll(){ CHART_IDS.forEach(resizeChart); }

  // ----- Tabs -----
  var tabs  = document.querySelectorAll('.tab-btn');
  var panes = document.querySelectorAll('.tab-pane');
  function activate(id){
    tabs.forEach(function(t){ t.classList.toggle('active', t.getAttribute('data-tab')===id); });
    panes.forEach(function(p){ p.classList.toggle('active', p.id===id); });
    // Plotly mis-sizes charts rendered while hidden — resize on reveal.
    if(id==='tab-graphs'){ setTimeout(resizeAll, 30); }
  }
  tabs.forEach(function(t){
    t.addEventListener('click', function(){ activate(t.getAttribute('data-tab')); });
  });

  window.addEventListener('resize', resizeAll);

  // ----- Full screen per chart -----
  // Each figure is built with a FIXED layout width (~1300px); Plotly.resize
  // alone keeps that width, so the plot just sits left-aligned in the wide
  // overlay. To actually fill the screen we relayout to explicit viewport
  // dimensions on enter, and restore the original width/height on exit.
  function fsDims(){
    return { width: Math.max(320, window.innerWidth - 32),
             height: Math.max(320, Math.round(window.innerHeight * 0.84)) };
  }
  function enterFs(chartId){
    var el = document.getElementById(chartId);
    if(!el || !window.Plotly) return;
    if(el._origW === undefined){
      el._origW = (el.layout && el.layout.width)  || null;
      el._origH = (el.layout && el.layout.height) || null;
    }
    var d = fsDims();
    try { Plotly.relayout(el, {width: d.width, height: d.height, autosize: false}); } catch(e){}
  }
  function exitFs(chartId){
    var el = document.getElementById(chartId);
    if(!el || !window.Plotly) return;
    try { Plotly.relayout(el, {width: el._origW, height: el._origH}); } catch(e){}
  }
  document.querySelectorAll('.fs-btn').forEach(function(btn){
    btn.addEventListener('click', function(){
      var card = document.getElementById(btn.getAttribute('data-card'));
      var chartId = btn.getAttribute('data-chart');
      var on = card.classList.toggle('fullscreen');
      btn.textContent = on ? 'Exit full screen' : 'Full screen';
      if(on){ enterFs(chartId); } else { exitFs(chartId); }
    });
  });
  document.addEventListener('keydown', function(e){
    if(e.key==='Escape'){
      document.querySelectorAll('.chart-card.fullscreen').forEach(function(card){
        card.classList.remove('fullscreen');
        var b = card.querySelector('.fs-btn');
        if(b){ b.textContent='Full screen'; exitFs(b.getAttribute('data-chart')); }
      });
    }
  });
  // Keep a fullscreen chart filling the viewport if the window is resized.
  window.addEventListener('resize', function(){
    var card = document.querySelector('.chart-card.fullscreen');
    if(card){ var b = card.querySelector('.fs-btn'); if(b){ enterFs(b.getAttribute('data-chart')); } }
  });

  // ----- Shared date-range picker -----
  var startInput = document.getElementById('dr-start');
  var endInput   = document.getElementById('dr-end');
  function applyRange(s, e){
    if(!s || !e) return;
    var range = [s + ' 00:00:00', e + ' 23:59:59'];
    TIME_CHARTS.forEach(function(id){
      var el = document.getElementById(id);
      if(!el || !window.Plotly) return;
      var upd = {'xaxis.range': range};
      if(id==='chart1'){ upd['xaxis2.range'] = range; }   // shared LMP sub-panel
      try { Plotly.relayout(el, upd); } catch(err){}
    });
  }
  function resetRange(){
    TIME_CHARTS.forEach(function(id){
      var el = document.getElementById(id);
      if(!el || !window.Plotly) return;
      var upd = {'xaxis.autorange': true};
      if(id==='chart1'){ upd['xaxis2.autorange'] = true; }
      try { Plotly.relayout(el, upd); } catch(err){}
    });
  }
  function markPreset(btn){
    document.querySelectorAll('.dr-btn').forEach(function(b){ b.classList.toggle('active', b===btn); });
  }
  if(startInput && endInput){
    startInput.addEventListener('change', function(){ applyRange(startInput.value, endInput.value); markPreset(null); });
    endInput.addEventListener('change',   function(){ applyRange(startInput.value, endInput.value); markPreset(null); });
  }
  function isoDate(d){ return d.toISOString().slice(0,10); }
  document.querySelectorAll('.dr-btn').forEach(function(btn){
    btn.addEventListener('click', function(){
      markPreset(btn);
      var days = btn.getAttribute('data-days');
      if(days==='all'){ startInput.value=DATA_START; endInput.value=DATA_END; resetRange(); return; }
      var end = new Date(DATA_END + 'T00:00:00');
      var start = new Date(end);
      start.setDate(start.getDate() - (parseInt(days,10) - 1));
      var startStr = isoDate(start);
      if(startStr < DATA_START){ startStr = DATA_START; }
      startInput.value = startStr; endInput.value = DATA_END;
      applyRange(startInput.value, endInput.value);
    });
  });
})();
</script>
"""
TAB_JS = TAB_JS.replace("__DATA_START__", start_str).replace("__DATA_END__", end_str)

html = html.replace("</style>", EXTRA_CSS + "\n</style>")
html = html.replace("</body>", TAB_JS + "\n</body>")

# ── Save ───────────────────────────────────────────────────────────────────────
out_file = os.path.join(
    OUTPUT_DIR,
    f"{project}_report_{start_str}_to_{end_str}.html"
)
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n  Report saved: {out_file}")
print("\n" + "="*60)
print("Report complete. Open the HTML file in any browser.")
print("="*60)
