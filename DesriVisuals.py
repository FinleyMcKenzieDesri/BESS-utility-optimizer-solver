"""
DESRI Battery Dispatch Visualizer
===================================
Reads the optimized CSV produced by DesriOptimizer.py and generates
four charts that tell the complete dispatch story visually.

Charts produced:
    1. Dispatch comparison    — actual vs all optimal strategies over time
    2. Revenue waterfall      — actual vs each strategy, dollar gap visible
    3. SOC + LMP overlay      — was the battery available during price spikes?
    4. Hourly behavior heatmap — utility's dispatch pattern across days/hours

All charts saved as interactive HTML (open in any browser) and static PNG.

USAGE:
    1. Run DesriPipeline.py then DesriOptimizer.py first
    2. Set OPTIMIZED_CSV below to point at the optimized output file
    3. Run this script
    4. Find charts in a new folder named {project}_charts/

REQUIREMENTS:
    pip install pandas plotly kaleido
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OPTIMIZED_CSV = r"C:\VS code\Arroyo Hybrid_optimized_2025-09-01_to_2026-06-10.csv"

# Chart appearance
TEMPLATE      = "plotly_dark"    # options: plotly_dark, plotly_white, plotly
CHART_WIDTH   = 1400
CHART_HEIGHT  = 600
HEATMAP_HEIGHT = 500

# Strategy display names and colors
STRATEGY_CONFIG = {
    "actual": {
        "label": "Actual Dispatch",
        "color": "#00D4FF",
        "dash":  "solid",
        "width": 2.5,
    },
    "arbitrage_real_time": {
        "label": "RT Arbitrage (optimal ceiling)",
        "color": "#FF6B6B",
        "dash":  "dot",
        "width": 1.5,
    },
    "arbitrage_day_ahead": {
        "label": "DA Arbitrage (realistic optimal)",
        "color": "#FFD93D",
        "dash":  "dash",
        "width": 1.5,
    },
    "scarcity": {
        "label": "Scarcity / Grid Stress",
        "color": "#6BCB77",
        "dash":  "dashdot",
        "width": 1.5,
    },
    "hybrid": {
        "label": "Hybrid",
        "color": "#C77DFF",
        "dash":  "longdash",
        "width": 1.5,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("DESRI Battery Dispatch Visualizer")
print("="*60)

df = pd.read_csv(OPTIMIZED_CSV)
df["time"] = pd.to_datetime(df["time"], utc=True)
df["time_local"] = df["time"].dt.tz_convert("America/Los_Angeles")

project    = df["project"].iloc[0]
start_str  = str(df["time_local"].min().date())
end_str    = str(df["time_local"].max().date())
n_days     = (df["time_local"].max() - df["time_local"].min()).days + 1
cap_mw     = float(df["battery_mw"].iloc[0])
cap_mwh    = float(df["battery_mwh"].iloc[0])

# Output folder
out_dir = f"{project}_charts"
os.makedirs(out_dir, exist_ok=True)

print(f"\n  Project:  {project}")
print(f"  Period:   {start_str} to {end_str} ({n_days} days)")
print(f"  Battery:  {cap_mw} MW / {cap_mwh} MWh")
print(f"  Output:   {out_dir}/")

# Detect which strategy columns are present
available_strategies = []
for name in STRATEGY_CONFIG:
    if name == "actual":
        if "power_kw" in df.columns:
            available_strategies.append("actual")
    else:
        if f"optimal_{name}_kw" in df.columns:
            available_strategies.append(name)

print(f"  Strategies found: {available_strategies}")

# Revenue summary for waterfall
rev = {}
if "actual_revenue_usd" in df.columns:
    rev["actual"] = df["actual_revenue_usd"].sum()
for name in [s for s in available_strategies if s != "actual"]:
    col = f"optimal_{name}_revenue_usd"
    if col in df.columns:
        rev[name] = df[col].sum()

best_strategy = max((k for k in rev if k != "actual"), key=lambda k: rev[k])


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — DISPATCH COMPARISON (actual vs all optimal strategies)
# ══════════════════════════════════════════════════════════════════════════════

print("\n  Building Chart 1: Dispatch comparison...")

fig1 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.06,
    subplot_titles=["Battery Dispatch: Actual vs Optimal Strategies (MW)",
                    "RT LMP ($/MWh)"],
)

# Convert kW to MW for readability
mw_scale = 1 / 1000

# Plot actual dispatch
fig1.add_trace(
    go.Scatter(
        x=df["time_local"],
        y=df["power_kw"] * mw_scale,
        name=STRATEGY_CONFIG["actual"]["label"],
        line=dict(
            color=STRATEGY_CONFIG["actual"]["color"],
            width=STRATEGY_CONFIG["actual"]["width"],
            dash=STRATEGY_CONFIG["actual"]["dash"],
        ),
        hovertemplate="%{x}<br>Actual: %{y:.1f} MW<extra></extra>",
    ),
    row=1, col=1,
)

# Plot each optimal strategy
for name in [s for s in available_strategies if s != "actual"]:
    cfg = STRATEGY_CONFIG[name]
    col = f"optimal_{name}_kw"
    fig1.add_trace(
        go.Scatter(
            x=df["time_local"],
            y=df[col] * mw_scale,
            name=cfg["label"],
            line=dict(color=cfg["color"], width=cfg["width"], dash=cfg["dash"]),
            hovertemplate=f"%{{x}}<br>{cfg['label']}: %{{y:.1f}} MW<extra></extra>",
            opacity=0.85,
        ),
        row=1, col=1,
    )

# Zero line
fig1.add_hline(y=0, line_width=0.5, line_color="gray", row=1, col=1)

# RT LMP in lower panel
fig1.add_trace(
    go.Scatter(
        x=df["time_local"],
        y=df["lmp_rt"],
        name="RT LMP",
        line=dict(color="#FF9500", width=1),
        fill="tozeroy",
        fillcolor="rgba(255,149,0,0.15)",
        hovertemplate="%{x}<br>RT LMP: $%{y:.2f}/MWh<extra></extra>",
    ),
    row=2, col=1,
)

fig1.update_layout(
    template=TEMPLATE,
    title=dict(
        text=f"{project} — Battery Dispatch Comparison<br>"
             f"<sup>{start_str} to {end_str}  |  "
             f"Best optimal: {STRATEGY_CONFIG[best_strategy]['label']}  |  "
             f"Revenue leakage vs best: ${rev[best_strategy]-rev.get('actual',0):,.0f}</sup>",
        font=dict(size=16),
    ),
    height=CHART_HEIGHT + 200,
    width=CHART_WIDTH,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis_title="Power (MW)  [+ = discharge, − = charge]",
    yaxis2_title="$/MWh",
)

fig1.write_html(f"{out_dir}/01_dispatch_comparison.html")
try:
    fig1.write_image(f"{out_dir}/01_dispatch_comparison.png", scale=2)
except Exception:
    pass   # kaleido not installed — HTML still saved
print("    Saved: 01_dispatch_comparison.html")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — REVENUE WATERFALL
# ══════════════════════════════════════════════════════════════════════════════

print("  Building Chart 2: Revenue waterfall...")

# Build ordered bar data: actual first, then strategies low to high
revenue_items = [("actual", rev.get("actual", 0))]
for name in sorted([k for k in rev if k != "actual"], key=lambda k: rev[k]):
    revenue_items.append((name, rev[name]))

labels  = [STRATEGY_CONFIG[n]["label"] for n, _ in revenue_items]
values  = [v for _, v in revenue_items]
colors  = [STRATEGY_CONFIG[n]["color"] for n, _ in revenue_items]
actual_rev = rev.get("actual", 0)
gap_labels = []
for name, val in revenue_items:
    if name == "actual":
        gap_labels.append("Actual")
    else:
        gap = val - actual_rev
        gap_labels.append(f"+${gap:,.0f} vs actual")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=labels,
    y=values,
    marker_color=colors,
    text=[f"${v:,.0f}" for v in values],
    textposition="outside",
    customdata=gap_labels,
    hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<br>%{customdata}<extra></extra>",
))

# Horizontal line at actual revenue
fig2.add_hline(
    y=actual_rev,
    line_dash="dash",
    line_color=STRATEGY_CONFIG["actual"]["color"],
    annotation_text=f"Actual: ${actual_rev:,.0f}",
    annotation_position="top right",
)

fig2.update_layout(
    template=TEMPLATE,
    title=dict(
        text=f"{project} — Revenue by Dispatch Strategy<br>"
             f"<sup>{start_str} to {end_str}  |  {n_days} days  |  "
             f"{cap_mw} MW / {cap_mwh} MWh battery</sup>",
        font=dict(size=16),
    ),
    height=CHART_HEIGHT,
    width=900,
    yaxis_title="Total Revenue (USD)",
    xaxis_title="",
    showlegend=False,
)

fig2.write_html(f"{out_dir}/02_revenue_waterfall.html")
try:
    fig2.write_image(f"{out_dir}/02_revenue_waterfall.png", scale=2)
except Exception:
    pass
print("    Saved: 02_revenue_waterfall.html")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — SOC + LMP OVERLAY
# Was the battery charged and available during price spikes?
# ══════════════════════════════════════════════════════════════════════════════

print("  Building Chart 3: SOC + LMP overlay...")

fig3 = make_subplots(specs=[[{"secondary_y": True}]])

# SOC as filled area
fig3.add_trace(
    go.Scatter(
        x=df["time_local"],
        y=df["soc_pct"],
        name="State of Charge (%)",
        line=dict(color="#00D4FF", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.15)",
        hovertemplate="%{x}<br>SOC: %{y:.1f}%<extra></extra>",
    ),
    secondary_y=False,
)

# RT LMP as line on second axis
fig3.add_trace(
    go.Scatter(
        x=df["time_local"],
        y=df["lmp_rt"],
        name="RT LMP ($/MWh)",
        line=dict(color="#FF9500", width=1.5),
        hovertemplate="%{x}<br>RT LMP: $%{y:.2f}/MWh<extra></extra>",
    ),
    secondary_y=True,
)

# Shade negative LMP periods (grid oversupply — ideal charging windows)
neg_lmp = df[df["lmp_rt"] < 0]
if len(neg_lmp) > 0:
    fig3.add_trace(
        go.Scatter(
            x=pd.concat([neg_lmp["time_local"], neg_lmp["time_local"].iloc[::-1]]),
            y=pd.concat([neg_lmp["lmp_rt"], pd.Series([0]*len(neg_lmp))]),
            fill="toself",
            fillcolor="rgba(107,203,119,0.2)",
            line=dict(width=0),
            name="Negative LMP (free charging)",
            showlegend=True,
            hoverinfo="skip",
        ),
        secondary_y=True,
    )

fig3.update_layout(
    template=TEMPLATE,
    title=dict(
        text=f"{project} — State of Charge vs RT LMP<br>"
             f"<sup>Green shading = negative LMP (grid oversupply, ideal charge window). "
             f"Battery should be LOW during these periods and HIGH before evening peaks.</sup>",
        font=dict(size=16),
    ),
    height=CHART_HEIGHT,
    width=CHART_WIDTH,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig3.update_yaxes(title_text="State of Charge (%)", secondary_y=False, range=[0, 110])
fig3.update_yaxes(title_text="RT LMP ($/MWh)", secondary_y=True)

fig3.write_html(f"{out_dir}/03_soc_lmp_overlay.html")
try:
    fig3.write_image(f"{out_dir}/03_soc_lmp_overlay.png", scale=2)
except Exception:
    pass
print("    Saved: 03_soc_lmp_overlay.html")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — HOURLY BEHAVIOR HEATMAP
# Day-of-week × hour-of-day colored by average actual dispatch power
# Reveals the utility's behavioral pattern independent of price signals
# ══════════════════════════════════════════════════════════════════════════════

print("  Building Chart 4: Hourly behavior heatmap...")

df["date"]    = df["time_local"].dt.date
df["hour"]    = df["time_local"].dt.hour
df["weekday"] = df["time_local"].dt.day_name()

# Pivot: rows = dates, columns = hours, values = mean power (MW)
heatmap_data = (
    df.groupby(["date", "hour"])["power_kw"]
    .mean()
    .reset_index()
)
heatmap_pivot = heatmap_data.pivot(index="date", columns="hour", values="power_kw")
heatmap_pivot = heatmap_pivot * mw_scale  # convert to MW

# Row labels: "Mon Jun 1" style
row_labels = [
pd.Timestamp(str(d)).strftime("%a %b %#d")
    if hasattr(pd.Timestamp(str(d)), 'strftime') else str(d)
    for d in heatmap_pivot.index
]

# Color scale: blue = charging, white = idle, red = discharging
# Symmetric around 0 so zero is always white
max_abs = max(abs(heatmap_pivot.values[~np.isnan(heatmap_pivot.values)]).max(), 1)

fig4 = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=[f"{h:02d}:00" for h in heatmap_pivot.columns],
    y=row_labels,
    colorscale=[
        [0.0,  "#1E90FF"],   # full charge → blue
        [0.5,  "#1a1a2e"],   # zero / idle → dark (matches dark template)
        [1.0,  "#FF4444"],   # full discharge → red
    ],
    zmid=0,
    zmin=-max_abs,
    zmax=max_abs,
    colorbar=dict(
        title="MW",
        tickvals=[-max_abs, -max_abs/2, 0, max_abs/2, max_abs],
        ticktext=[
            f"−{max_abs:.0f} (charge)",
            f"−{max_abs/2:.0f}",
            "0",
            f"+{max_abs/2:.0f}",
            f"+{max_abs:.0f} (discharge)",
        ],
    ),
    hovertemplate="<b>%{y}  %{x}</b><br>Avg power: %{z:.1f} MW<extra></extra>",
))

fig4.update_layout(
    template=TEMPLATE,
    title=dict(
        text=f"{project} — Dispatch Behavior by Hour of Day<br>"
             f"<sup>Blue = charging  |  Red = discharging  |  "
             f"Consistent columns reveal time-of-day dispatch rules independent of price</sup>",
        font=dict(size=16),
    ),
    height=max(HEATMAP_HEIGHT, len(heatmap_pivot) * 35 + 150),
    width=CHART_WIDTH,
    xaxis_title="Hour of Day (Pacific)",
    yaxis_title="Date",
    yaxis=dict(autorange="reversed"),
)

fig4.write_html(f"{out_dir}/04_hourly_heatmap.html")
try:
    fig4.write_image(f"{out_dir}/04_hourly_heatmap.png", scale=2)
except Exception:
    pass
print("    Saved: 04_hourly_heatmap.html")


# ══════════════════════════════════════════════════════════════════════════════
# BONUS — CHART 5: CUMULATIVE REVENUE OVER TIME
# Shows how actual vs optimal revenue diverges across the period
# ══════════════════════════════════════════════════════════════════════════════

print("  Building Chart 5: Cumulative revenue...")

fig5 = go.Figure()

if "actual_revenue_usd" in df.columns:
    fig5.add_trace(go.Scatter(
        x=df["time_local"],
        y=df["actual_revenue_usd"].cumsum(),
        name=STRATEGY_CONFIG["actual"]["label"],
        line=dict(color=STRATEGY_CONFIG["actual"]["color"],
                  width=STRATEGY_CONFIG["actual"]["width"]),
        hovertemplate="%{x}<br>Cumulative actual: $%{y:,.0f}<extra></extra>",
    ))

for name in [s for s in available_strategies if s != "actual"]:
    rev_col = f"optimal_{name}_revenue_usd"
    if rev_col not in df.columns:
        continue
    cfg = STRATEGY_CONFIG[name]
    fig5.add_trace(go.Scatter(
        x=df["time_local"],
        y=df[rev_col].cumsum(),
        name=cfg["label"],
        line=dict(color=cfg["color"], width=cfg["width"], dash=cfg["dash"]),
        hovertemplate=f"%{{x}}<br>Cumulative {cfg['label']}: $%{{y:,.0f}}<extra></extra>",
        opacity=0.85,
    ))

fig5.add_hline(y=0, line_width=0.5, line_color="gray")

fig5.update_layout(
    template=TEMPLATE,
    title=dict(
        text=f"{project} — Cumulative Revenue Over Time<br>"
             f"<sup>Gap between lines is revenue leakage accumulating in real time</sup>",
        font=dict(size=16),
    ),
    height=CHART_HEIGHT,
    width=CHART_WIDTH,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis_title="Cumulative Revenue (USD)",
)

fig5.write_html(f"{out_dir}/05_cumulative_revenue.html")
try:
    fig5.write_image(f"{out_dir}/05_cumulative_revenue.png", scale=2)
except Exception:
    pass
print("    Saved: 05_cumulative_revenue.html")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(f"All charts saved to: {out_dir}/")
print()
print("  01_dispatch_comparison.html  — actual vs all strategies over time")
print("  02_revenue_waterfall.html    — revenue by strategy, gap visible")
print("  03_soc_lmp_overlay.html      — was battery available during price spikes?")
print("  04_hourly_heatmap.html       — behavioral pattern by hour of day")
print("  05_cumulative_revenue.html   — revenue leakage accumulating over time")
print()
print("Open any .html file in a browser for interactive charts.")
print("PNG files saved alongside each HTML (requires kaleido).")
print()
print("Next step: run DesriNarrative.py for the LLM interpretation.")
print("="*60)
