"""
DESRI Battery Dispatch Narrative
==================================
Reads the optimized CSV produced by DesriOptimizer.py, assembles a
structured data summary, calls the Claude API to generate a plain-English
analysis, and saves a styled HTML report.

The report contains:
    - Executive summary (non-technical, 1 page)
    - Per-strategy breakdown (what each optimal looked like and why)
    - Behavioral analysis (what the utility appears to be optimizing for)
    - Revenue leakage quantification
    - Anomalies and flags
    - Recommendations

USAGE:
    1. Run DesriPipeline.py → DesriOptimizer.py first
    2. Set OPTIMIZED_CSV below
    3. Run this script
    4. Open the HTML report in any browser

REQUIREMENTS:
    pip install pandas numpy requests
    (No anthropic SDK needed — uses the API directly)
"""

import json
import warnings
import numpy  as np
import pandas as pd
import requests
from datetime import datetime

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OPTIMIZED_CSV = r"C:\VS code\Arroyo_optimized_2026-06-01_to_2026-06-08.csv"
OUTPUT_DIR    = r"C:\VS code"

# Claude API — no key needed, handled by claude.ai environment
# If running outside claude.ai, add your key:
#   ANTHROPIC_API_KEY = "sk-ant-..."
# and uncomment the Authorization header in call_claude() below
ANTHROPIC_API_KEY = ""   # leave blank when running inside claude.ai artifacts

MODEL = "claude-sonnet-4-20250514"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def call_claude(system_prompt, user_prompt):
    """Call Claude API and return the text response."""
    headers = {
        "Content-Type": "application/json",
        # Uncomment if running outside claude.ai:
        # "x-api-key": ANTHROPIC_API_KEY,
    }
    body = {
        "model": MODEL,
        "max_tokens": 4000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def fmt_dollars(val):
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    elif abs(val) >= 1_000:
        return f"${val:,.0f}"
    return f"${val:.2f}"


def corr_label(score):
    if np.isnan(score):
        return "n/a"
    if score > 0.7:   return f"{score:.3f} (strong)"
    if score > 0.4:   return f"{score:.3f} (moderate)"
    if score > 0.1:   return f"{score:.3f} (weak)"
    if score > -0.1:  return f"{score:.3f} (none)"
    return f"{score:.3f} (inverse)"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND SUMMARIZE THE OPTIMIZED CSV
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("DESRI Battery Dispatch Narrative")
print("="*60)

print("\nLoading optimized CSV...")
df = pd.read_csv(OPTIMIZED_CSV)
df["time"] = pd.to_datetime(df["time"], utc=True)
df["time_local"] = df["time"].dt.tz_convert("America/Los_Angeles")
df["hour"] = df["time_local"].dt.hour

project   = df["project"].iloc[0]
start_str = str(df["time_local"].min().date())
end_str   = str(df["time_local"].max().date())
n_days    = (df["time_local"].max() - df["time_local"].min()).days + 1
cap_mw    = float(df["battery_mw"].iloc[0])
cap_mwh   = float(df["battery_mwh"].iloc[0])
rte       = float(df["rte"].iloc[0])
max_cy    = float(df["max_cycles_year"].iloc[0])

print(f"  Project: {project}  |  {start_str} to {end_str}  |  {n_days} days")

# ── Revenue figures ────────────────────────────────────────────────────────────
strategy_names = {
    "arbitrage_real_time": "RT Arbitrage (perfect foresight ceiling)",
    "arbitrage_day_ahead": "DA Arbitrage (realistic optimal)",
    "scarcity":            "Scarcity / Grid Stress",
    "hybrid":              "Hybrid (stress + price)",
}

rev = {}
if "actual_revenue_usd" in df.columns:
    rev["actual"] = df["actual_revenue_usd"].sum()

for key in strategy_names:
    col = f"optimal_{key}_revenue_usd"
    if col in df.columns:
        rev[key] = df[col].sum()

best_strategy     = max((k for k in rev if k != "actual"), key=lambda k: rev[k])
actual_rev        = rev.get("actual", 0)
leakage_vs_best   = rev[best_strategy] - actual_rev
leakage_per_day   = leakage_vs_best / n_days
leakage_annualized = leakage_per_day * 365

# ── Dispatch behavior ─────────────────────────────────────────────────────────
actual_kw = df["power_kw"].fillna(0)
charging_pct    = (actual_kw < -100).sum() / len(df) * 100
discharging_pct = (actual_kw >  100).sum() / len(df) * 100
idle_pct        = (actual_kw.abs() <= 100).sum() / len(df) * 100

soc_mean = df["soc_pct"].mean()
soc_min  = df["soc_pct"].min()
soc_max  = df["soc_pct"].max()

lmp_rt_mean = df["lmp_rt"].mean()
lmp_da_mean = df["lmp_da"].mean()
lmp_rt_max  = df["lmp_rt"].max()

# ── When does utility discharge vs LMP ────────────────────────────────────────
high_lmp_threshold = df["lmp_rt"].quantile(0.75)
discharge_intervals = df[df["power_kw"] > 1000]
discharge_at_high_lmp = (
    discharge_intervals["lmp_rt"] > high_lmp_threshold
).mean() * 100 if len(discharge_intervals) > 0 else 0

# ── Correlation scores ────────────────────────────────────────────────────────
match_scores = {}
for key in strategy_names:
    opt_col = f"optimal_{key}_kw"
    if opt_col in df.columns:
        mask = actual_kw.notna() & df[opt_col].notna()
        if mask.sum() > 10:
            match_scores[key] = float(
                np.corrcoef(actual_kw[mask], df[opt_col][mask])[0, 1]
            )
        else:
            match_scores[key] = np.nan

best_match = max(match_scores, key=lambda k: match_scores[k]
                 if not np.isnan(match_scores[k]) else -999)

# ── Grid stress context ───────────────────────────────────────────────────────
grid_stress_mean     = df["grid_stress_mwh"].mean()
grid_stress_max      = df["grid_stress_mwh"].max()
high_stress_pct      = (df["grid_stress_mwh"] > 2000).sum() / len(df) * 100

# ── Hourly discharge pattern ──────────────────────────────────────────────────
hourly_discharge = (
    df.groupby("hour")["power_kw"]
    .mean()
    .sort_values(ascending=False)
)
top_discharge_hours   = hourly_discharge.head(3).index.tolist()
top_discharge_vals    = hourly_discharge.head(3).values.tolist()

# ── SOC during highest LMP ────────────────────────────────────────────────────
top_lmp_intervals = df.nlargest(50, "lmp_rt")
soc_during_peaks  = top_lmp_intervals["soc_pct"].mean()

print("  Summary statistics computed.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD THE PROMPT AND CALL CLAUDE
# ══════════════════════════════════════════════════════════════════════════════

print("\nCalling Claude API to generate narrative...")

SYSTEM_PROMPT = """You are an expert energy analyst specializing in battery energy storage 
systems (BESS) and electricity market optimization for utility-scale solar+storage projects. 

You work for DESRI, a renewable energy developer that sells solar+BESS projects to utilities. 
Once sold, the utilities control battery dispatch — DESRI does not. Your job is to analyze 
how a utility is using their battery compared to various optimal strategies, quantify the 
revenue gap, and provide clear insights about what the utility appears to be optimizing for.

Write in a professional but accessible tone. The report has two audiences:
1. Executive summary: non-technical utility or investor audience — no jargon, dollar figures front and center
2. Technical detail: DESRI internal team — strategy mechanics, correlation analysis, behavioral patterns

Structure your response as clean HTML sections using <h2>, <h3>, <p>, <ul>, <table> tags.
Do not include <html>, <head>, or <body> tags — just the inner content sections.
Use inline styles sparingly for emphasis. Be direct and specific — use the exact numbers provided."""

USER_PROMPT = f"""
Generate a battery dispatch analysis report for the following project data.
Use the exact numbers provided — do not estimate or approximate.

═══════════════════════════════════════════
PROJECT DATA
═══════════════════════════════════════════

Project:          {project}
Analysis period:  {start_str} to {end_str} ({n_days} days)
Battery specs:    {cap_mw} MW / {cap_mwh} MWh
Round-trip RTE:   {rte*100:.1f}%
Max cycles/year:  {max_cy:.0f}

═══════════════════════════════════════════
REVENUE RESULTS
═══════════════════════════════════════════

Actual revenue:                  {fmt_dollars(actual_rev)}
Best optimal strategy:           {strategy_names[best_strategy]} → {fmt_dollars(rev[best_strategy])}
Revenue leakage (vs best):       {fmt_dollars(leakage_vs_best)} over {n_days} days
Revenue leakage per day:         {fmt_dollars(leakage_per_day)}/day
Annualized leakage estimate:     {fmt_dollars(leakage_annualized)}/year

All strategy revenues:
{chr(10).join(f"  {strategy_names[k]:<45} {fmt_dollars(rev[k])}" for k in strategy_names if k in rev)}
  Actual dispatch:                                              {fmt_dollars(actual_rev)}

Note on strategy hierarchy:
- RT Arbitrage is the theoretical ceiling (perfect price foresight — not achievable in practice)
- DA Arbitrage is the realistic benchmark (uses prices known the night before — what utility should achieve)
- Scarcity reserves capacity for grid stress events, dispatching only when grid is under pressure
- Hybrid balances price arbitrage with grid stress response

═══════════════════════════════════════════
DISPATCH BEHAVIOR
═══════════════════════════════════════════

Time distribution:
  Charging:     {charging_pct:.1f}% of intervals
  Discharging:  {discharging_pct:.1f}% of intervals
  Idle:         {idle_pct:.1f}% of intervals

SOC utilization:
  Mean SOC:     {soc_mean:.1f}%
  Min SOC:      {soc_min:.1f}%
  Max SOC:      {soc_max:.1f}%

Average SOC during top 50 highest-price intervals: {soc_during_peaks:.1f}%
  (Interpretation: if SOC is HIGH during price spikes, the battery was available but not dispatched.
   If SOC is LOW during price spikes, the battery had already been depleted before prices peaked.)

Discharge timing:
  Top hours for discharge (Pacific): {top_discharge_hours}
  % of discharge events occurring during top-quartile LMP hours: {discharge_at_high_lmp:.1f}%
  (>50% = good price alignment. <50% = discharging without regard to price)

LMP context:
  Mean RT LMP:  ${lmp_rt_mean:.2f}/MWh
  Mean DA LMP:  ${lmp_da_mean:.2f}/MWh
  Peak RT LMP:  ${lmp_rt_max:.2f}/MWh

═══════════════════════════════════════════
STRATEGY MATCH ANALYSIS
═══════════════════════════════════════════

Correlation of actual dispatch with each optimal strategy:
{chr(10).join(f"  {strategy_names[k]:<45} {corr_label(match_scores.get(k, float('nan')))}" for k in strategy_names)}

Closest behavioral match: {strategy_names[best_match]}
  (Correlation measures dispatch direction alignment — positive = utility tends to 
   charge/discharge in the same direction as the strategy, negative = opposite behavior)

═══════════════════════════════════════════
GRID STRESS CONTEXT
═══════════════════════════════════════════

Mean grid stress (actual - forecast demand): {grid_stress_mean:,.0f} MWh
Max grid stress:                             {grid_stress_max:,.0f} MWh
% of intervals with stress > 2,000 MWh:     {high_stress_pct:.1f}%
  (High stress = actual demand significantly exceeded day-ahead forecast,
   suggesting the grid needed additional resources beyond what was planned)

═══════════════════════════════════════════
REPORT STRUCTURE REQUIRED
═══════════════════════════════════════════

Please write the report with these exact sections:

1. EXECUTIVE SUMMARY
   - 3-4 sentences max. Dollar figures front and center. No jargon.
   - What the utility did, what was optimal, and what the gap cost.

2. WHAT THE UTILITY DID
   - Plain-English description of actual dispatch behavior
   - When they charged, when they discharged, and whether it aligned with price signals
   - What the SOC pattern reveals about their strategy

3. WHAT OPTIMAL LOOKED LIKE
   - Per-strategy breakdown: what each strategy would have done and why
   - Focus especially on the DA Arbitrage (realistic benchmark) vs actual
   - Keep RT Arbitrage in context as a theoretical ceiling only

4. THE REVENUE GAP
   - Quantify leakage clearly with a summary table
   - Per-day and annualized figures
   - Which strategy the utility most resembles and what that implies

5. LIKELY DISPATCH MOTIVATION
   - Based on the behavioral data, what appears to be driving the utility's decisions?
   - Options: capacity obligation, conservative SOC management, DA price response, 
     grid reliability mandate, other
   - Be specific about which signals the data supports

6. FLAGS AND ANOMALIES
   - Anything in the data that warrants closer investigation
   - Any behavior patterns that don't fit any known strategy

7. RECOMMENDATIONS FOR DESRI
   - 2-3 actionable recommendations based on this analysis
   - Could be about future contract structuring, dispatch monitoring, or utility engagement
"""

narrative_html = call_claude(SYSTEM_PROMPT, USER_PROMPT)
print("  Claude response received.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — WRAP IN STYLED HTML AND SAVE
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding HTML report...")

report_date = datetime.now().strftime("%B %d, %Y at %H:%M")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DESRI Dispatch Analysis — {project} — {start_str} to {end_str}</title>
<style>
  :root {{
    --bg:       #0f1117;
    --surface:  #1a1d2e;
    --border:   #2a2d3e;
    --accent:   #00d4ff;
    --green:    #6bcb77;
    --red:      #ff6b6b;
    --yellow:   #ffd93d;
    --purple:   #c77dff;
    --text:     #e8eaf0;
    --subtext:  #9096a8;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    padding: 0;
  }}

  /* Header */
  .report-header {{
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1d2e 100%);
    border-bottom: 2px solid var(--accent);
    padding: 40px 60px;
  }}
  .report-header .label {{
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .report-header h1 {{
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .report-header .meta {{
    color: var(--subtext);
    font-size: 14px;
  }}

  /* KPI strip */
  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border-bottom: 1px solid var(--border);
  }}
  .kpi {{
    padding: 24px 32px;
    border-right: 1px solid var(--border);
  }}
  .kpi:last-child {{ border-right: none; }}
  .kpi .kpi-label {{
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 6px;
  }}
  .kpi .kpi-value {{
    font-size: 26px;
    font-weight: 700;
    color: var(--accent);
  }}
  .kpi .kpi-sub {{
    font-size: 12px;
    color: var(--subtext);
    margin-top: 2px;
  }}
  .kpi.danger  .kpi-value {{ color: var(--red);    }}
  .kpi.success .kpi-value {{ color: var(--green);  }}
  .kpi.warn    .kpi-value {{ color: var(--yellow); }}

  /* Content */
  .content {{
    max-width: 1000px;
    margin: 0 auto;
    padding: 48px 60px;
  }}

  h2 {{
    font-size: 20px;
    font-weight: 700;
    color: var(--accent);
    margin: 40px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  h2:first-child {{ margin-top: 0; }}

  h3 {{
    font-size: 15px;
    font-weight: 600;
    color: var(--yellow);
    margin: 24px 0 8px;
  }}

  p {{ margin: 10px 0; color: var(--text); }}

  ul, ol {{
    margin: 10px 0 10px 24px;
    color: var(--text);
  }}
  li {{ margin: 6px 0; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 14px;
  }}
  th {{
    background: var(--surface);
    color: var(--subtext);
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}

  /* Footer */
  .report-footer {{
    border-top: 1px solid var(--border);
    padding: 24px 60px;
    color: var(--subtext);
    font-size: 12px;
    display: flex;
    justify-content: space-between;
  }}
</style>
</head>
<body>

<div class="report-header">
  <div class="label">DESRI · Battery Dispatch Analysis</div>
  <h1>{project} — Dispatch Performance Report</h1>
  <div class="meta">
    Analysis period: {start_str} to {end_str} &nbsp;·&nbsp;
    {n_days} days &nbsp;·&nbsp;
    {cap_mw:.0f} MW / {cap_mwh:.0f} MWh &nbsp;·&nbsp;
    RTE: {rte*100:.1f}% &nbsp;·&nbsp;
    Generated: {report_date}
  </div>
</div>

<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Actual Revenue</div>
    <div class="kpi-value">{fmt_dollars(actual_rev)}</div>
    <div class="kpi-sub">{n_days}-day period</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">DA Optimal Revenue</div>
    <div class="kpi-value success">{fmt_dollars(rev.get('arbitrage_day_ahead', 0))}</div>
    <div class="kpi-sub">Realistic benchmark</div>
  </div>
  <div class="kpi danger">
    <div class="kpi-label">Revenue Leakage</div>
    <div class="kpi-value">{fmt_dollars(leakage_vs_best)}</div>
    <div class="kpi-sub">{fmt_dollars(leakage_per_day)}/day · {fmt_dollars(leakage_annualized)}/yr est.</div>
  </div>
  <div class="kpi warn">
    <div class="kpi-label">Closest Strategy Match</div>
    <div class="kpi-value" style="font-size:18px">{strategy_names[best_match].split('(')[0].strip()}</div>
    <div class="kpi-sub">Corr: {match_scores.get(best_match, float('nan')):.3f}</div>
  </div>
</div>

<div class="content">
{narrative_html}
</div>

<div class="report-footer">
  <span>DESRI · Confidential · Battery Dispatch Analysis</span>
  <span>Generated {report_date} · Model: {MODEL}</span>
</div>

</body>
</html>"""

# ── Save ───────────────────────────────────────────────────────────────────────
import os
out_file = os.path.join(
    OUTPUT_DIR,
    f"{project}_dispatch_report_{start_str}_to_{end_str}.html"
)
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n  Report saved: {out_file}")
print("\n" + "="*60)
print("Narrative complete. Open the HTML file in any browser.")
print("="*60)
