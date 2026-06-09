"""
DESRI Battery Dispatch Optimizer
==================================
Reads a unified CSV produced by DesriPipeline.py and runs LP optimization
for four dispatch strategies. Compares each to actual dispatch behavior,
calculates revenue leakage, and identifies which strategy the utility's
actual behavior most resembles.

USAGE:
    1. Run DesriPipeline.py first to produce the unified CSV
    2. Set UNIFIED_CSV below to point at that file
    3. Run this script
    4. Find the results CSV and terminal summary in the same folder

REQUIREMENTS:
    pip install pandas numpy pulp
"""

import warnings
import numpy  as np
import pandas as pd
import pulp

# Optional ML stack for the behavioral decision model (Route 3). The core
# optimizer runs fine without these; if scikit-learn is absent we just skip
# that one section instead of crashing.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

UNIFIED_CSV = r"C:\VS code\Arroyo Hybrid_unified_2025-09-01_to_2026-06-10.csv"

# Override battery specs here if needed — otherwise pulled from unified CSV
# Set to None to use values embedded in the unified CSV by the pipeline
BATTERY_CAPACITY_MW_OVERRIDE  = None   # e.g. 150
BATTERY_CAPACITY_MWH_OVERRIDE = None   # e.g. 600
RTE_OVERRIDE                  = None   # e.g. 0.90
MAX_CYCLES_YEAR_OVERRIDE      = None   # e.g. 365

# Hybrid strategy tuning. The hybrid battery reserves capacity for grid-stress
# events (like scarcity) but ALSO discharges opportunistically during the most
# expensive price intervals. HYBRID_ARBITRAGE_WEIGHT sets that opportunistic
# share: 0.5 = also permit discharge during the top 50% highest-price intervals.
# Higher → more arbitrage-like; lower → more scarcity-like. (This is no longer a
# price blend, so the two weights need not sum to 1.)
HYBRID_ARBITRAGE_WEIGHT  = 0.5   # discharge also allowed in top 50% price intervals
HYBRID_SCARCITY_WEIGHT   = 0.5   # retained for reference; stress reserve is always on

# Grid stress threshold — intervals above this (MWh delta) are flagged
# as high-stress and prioritized in the scarcity strategy
GRID_STRESS_HIGH_THRESHOLD = 2000   # MWh above forecast

# ── Behavioral analysis — characterize HOW actual dispatch tracks each strategy
DISCHARGE_THRESHOLD_KW = 1000    # |power| above this counts as a dispatch event
HIGH_PRICE_QUANTILE    = 0.75    # "high price" regime = RT LMP above this quantile
HIGH_SOC_PCT           = 70      # "high SOC" regime cutoff (%)
MAX_LAG_INTERVALS      = 36      # +/- lag window for timing search (36 x 5min = +/-3h)
# hour is encoded cyclically (hour_sin/hour_cos) so midnight wraps to 11pm and
# the model can fit a daily "bump" rather than a forced linear slope.
DECISION_FEATURES      = ["lmp_rt", "lmp_da", "soc_pct", "grid_stress_mwh",
                          "hour_sin", "hour_cos", "irradiance_wm2"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_unified(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def get_spec(df, col, override):
    """Use override if set, otherwise read from first row of unified CSV."""
    if override is not None:
        return override
    return float(df[col].iloc[0])


def intervals_to_hours(n=1):
    """Each interval is 5 minutes = 1/12 hour."""
    return n / 12.0


def pro_rate_cycles(max_cycles_year, n_days):
    """Pro-rate annual cycle limit to the number of days in the dataset."""
    return max_cycles_year * (n_days / 365.0)


def run_lp(price_signal, capacity_mw, capacity_mwh, rte,
           max_cycles, intervals_per_hour=12, discharge_mask=None):
    """
    Core LP optimizer. Maximizes revenue from battery dispatch given
    a price signal ($/MWh at each 5-min interval).

    Decision variables:
        charge[t]    — MW charged at interval t  (>= 0)
        discharge[t] — MW discharged at interval t (>= 0)
        soc[t]       — MWh stored at interval t

    Constraints:
        - Charge and discharge bounded by capacity_mw
        - SOC bounded by [0, capacity_mwh]
        - SOC dynamics: soc[t] = soc[t-1] + charge[t]*rte*dt - discharge[t]*dt
        - Cannot charge and discharge simultaneously (linearized via bounds)
        - Total full cycles <= max_cycles (cumulative throughput / capacity_mwh)
        - discharge[t] forced to 0 where discharge_mask[t] is False

    discharge_mask: optional boolean sequence (length T). Where False, the
    battery may NOT discharge at that interval (it can still charge). None
    means discharge is allowed everywhere — i.e. unconstrained arbitrage.
    Charging is never constrained, so the battery can always refill in
    readiness for the next permitted discharge window.

    Returns a DataFrame with columns: optimal_charge_kw, optimal_discharge_kw,
    optimal_power_kw (net, discharge positive), optimal_soc_pct, optimal_revenue
    """
    T   = len(price_signal)
    dt  = 1.0 / intervals_per_hour   # hours per interval

    prob = pulp.LpProblem("battery_dispatch", pulp.LpMaximize)

    # Decision variables
    charge    = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=capacity_mw)
                 for t in range(T)]
    # Where discharge_mask[t] is False, cap discharge at 0 — the battery is
    # held in reserve and may not dispatch at that interval.
    discharge = [
        pulp.LpVariable(
            f"d_{t}", lowBound=0,
            upBound=(capacity_mw
                     if (discharge_mask is None or discharge_mask[t])
                     else 0.0),
        )
        for t in range(T)
    ]
    soc       = [pulp.LpVariable(f"s_{t}", lowBound=0, upBound=capacity_mwh)
                 for t in range(T)]

    # Objective: maximize revenue from discharging, minus cost of charging
    # Price signal is $/MWh; power is MW; dt is hours → revenue in $
    prob += pulp.lpSum(
        (discharge[t] - charge[t]) * price_signal[t] * dt
        for t in range(T)
    )

    # SOC dynamics
    soc_initial = capacity_mwh * 0.5   # start at 50% SOC
    for t in range(T):
        if t == 0:
            prob += soc[t] == soc_initial + charge[t] * rte * dt - discharge[t] * dt
        else:
            prob += soc[t] == soc[t-1]   + charge[t] * rte * dt - discharge[t] * dt

    # Cycle limit — total energy throughput / capacity = number of cycles
    prob += (
        pulp.lpSum(discharge[t] * dt for t in range(T)) <= max_cycles * capacity_mwh
    )

    # Solve (suppress solver output)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract results
    results = pd.DataFrame({
        "optimal_charge_kw":    [pulp.value(charge[t])    * 1000 for t in range(T)],
        "optimal_discharge_kw": [pulp.value(discharge[t]) * 1000 for t in range(T)],
    })
    results["optimal_power_kw"] = (
        results["optimal_discharge_kw"] - results["optimal_charge_kw"]
    )

    # Rebuild SOC from solution
    soc_vals = [soc_initial]
    for t in range(T):
        new_soc = (soc_vals[-1]
                   + pulp.value(charge[t])    * rte * dt
                   - pulp.value(discharge[t]) * dt)
        soc_vals.append(max(0, min(capacity_mwh, new_soc)))
    results["optimal_soc_pct"] = [s / capacity_mwh * 100 for s in soc_vals[1:]]

    # Revenue per interval
    results["optimal_revenue_usd"] = (
        results["optimal_power_kw"] / 1000 * price_signal.values * dt
    )

    return results


def strategy_match_score(actual_kw, optimal_kw):
    """
    Pearson correlation between actual and optimal dispatch.
    Returns a score from -1 to 1; higher = more similar behavior.
    """
    mask = actual_kw.notna() & optimal_kw.notna()
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(actual_kw[mask], optimal_kw[mask])[0, 1])


def actual_revenue(actual_kw, price_mwh, dt=1/12):
    """Estimate revenue from actual dispatch using RT LMP."""
    return float((actual_kw / 1000 * price_mwh * dt).sum())


def _safe_corr(a, b):
    """Pearson correlation with guards for tiny samples / zero variance."""
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return np.nan
    aa, bb = a[m], b[m]
    if aa.std() == 0 or bb.std() == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def conditioned_correlations(df, actual_col, optimal_col, price_hi, soc_hi):
    """
    Route 1 — correlation of actual vs optimal dispatch WITHIN regimes.

    A single global correlation hides *when* behavior aligns. Slicing by price
    and SOC regime surfaces conditional logic — e.g. tight tracking of a
    strategy only above a price floor, and none below it.

    Returns {regime: corr}.
    """
    a, o = df[actual_col], df[optimal_col]
    hp = df["lmp_rt"] > price_hi
    hs = df["soc_pct"] > soc_hi
    return {
        "all":      _safe_corr(a, o),
        "hi_price": _safe_corr(a[hp], o[hp]),
        "lo_price": _safe_corr(a[~hp], o[~hp]),
        "hi_soc":   _safe_corr(a[hs], o[hs]),
        "lo_soc":   _safe_corr(a[~hs], o[~hs]),
    }


def lagged_correlation(actual, optimal, max_lag):
    """
    Route 2 — normalized cross-correlation over a +/- lag window.

    For each lag L, correlates actual[t] with optimal[t-L]; the lag with peak
    correlation reveals whether the utility tracks the signal but on a delay
    (L > 0 => actual responds L intervals AFTER the optimal signal, e.g. an
    internal scheduling cycle).

    Uses Pearson-per-lag rather than scipy.signal.correlate, whose raw argmax
    is dominated by signal magnitude rather than timing alignment.
    """
    a = actual.reset_index(drop=True)
    o = optimal.reset_index(drop=True)
    rows = [(lag, _safe_corr(a, o.shift(lag)))
            for lag in range(-max_lag, max_lag + 1)]
    tbl = pd.DataFrame(rows, columns=["lag", "corr"]).dropna()
    if tbl.empty:
        return None
    best = tbl.loc[tbl["corr"].idxmax()]
    zero = tbl.loc[tbl["lag"] == 0, "corr"]
    return {
        "best_lag":  int(best["lag"]),
        "best_corr": float(best["corr"]),
        "zero_corr": float(zero.iloc[0]) if len(zero) else np.nan,
    }


def dispatch_decision_model(df, features, discharge_threshold):
    """
    Route 3 — logistic model of the utility's binary discharge decision.

    Models P(discharge) from market/state features and reports STANDARDIZED
    coefficients (comparable across features) — i.e. which signals the utility
    actually responds to, and in which direction. This is a behavioral model of
    their implicit decision function, not just a correlation.

    Returns a dict of results, or {"error": ...} if it can't be built.
    """
    if not HAS_SKLEARN:
        return {"error": "scikit-learn not installed"}

    work = df.copy()
    work["discharged"] = (work["power_kw"] > discharge_threshold).astype(int)
    feats = [f for f in features if f in work.columns]
    X = work[feats].apply(lambda c: c.fillna(c.median()))
    y = work["discharged"]

    if y.nunique() < 2:
        return {"error": f"only one dispatch class present (all={int(y.iloc[0])})"}

    # Standardize so coefficient magnitudes are directly comparable.
    Xs = StandardScaler().fit_transform(X)

    # Time-ordered holdout for an honest AUC, then refit on all data for coefs.
    split = int(len(Xs) * 0.75)
    auc = np.nan
    if y.iloc[:split].nunique() == 2 and y.iloc[split:].nunique() == 2:
        m = LogisticRegression(max_iter=1000)
        m.fit(Xs[:split], y.iloc[:split])
        auc = float(roc_auc_score(y.iloc[split:], m.predict_proba(Xs[split:])[:, 1]))

    final = LogisticRegression(max_iter=1000)
    final.fit(Xs, y)
    coefs = sorted(zip(feats, final.coef_[0]), key=lambda x: abs(x[1]), reverse=True)

    out = {"coefs": coefs, "auc": auc, "discharge_rate": float(y.mean()),
           "n": int(len(y)), "shap": None}

    if HAS_SHAP:
        try:
            sv = shap.LinearExplainer(final, Xs).shap_values(Xs)
            mean_abs = np.abs(sv).mean(axis=0)
            out["shap"] = sorted(zip(feats, mean_abs),
                                 key=lambda x: x[1], reverse=True)
        except Exception:
            out["shap"] = None
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("DESRI Battery Dispatch Optimizer")
print("="*60)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading unified CSV...")
df = load_unified(UNIFIED_CSV)
print(f"  {len(df)} intervals, {df['project'].iloc[0]}, "
      f"{df['time'].min().date()} to {df['time'].max().date()}")

# ── Battery specs ──────────────────────────────────────────────────────────────
cap_mw  = get_spec(df, "battery_mw",       BATTERY_CAPACITY_MW_OVERRIDE)
cap_mwh = get_spec(df, "battery_mwh",      BATTERY_CAPACITY_MWH_OVERRIDE)
rte     = get_spec(df, "rte",              RTE_OVERRIDE)
max_cy  = get_spec(df, "max_cycles_year",  MAX_CYCLES_YEAR_OVERRIDE)

n_days      = (df["time"].max() - df["time"].min()).days + 1
max_cycles  = pro_rate_cycles(max_cy, n_days)

print(f"\n  Battery specs:")
print(f"    Capacity:       {cap_mw} MW / {cap_mwh} MWh")
print(f"    RTE:            {rte*100:.1f}%")
print(f"    Cycle limit:    {max_cy}/yr → {max_cycles:.1f} for {n_days} days")

# ── Fill missing LMP with forward fill then backfill ──────────────────────────
df["lmp_rt"] = df["lmp_rt"].ffill().bfill()
df["lmp_da"] = df["lmp_da"].ffill().bfill()
df["grid_stress_mwh"] = df["grid_stress_mwh"].ffill().bfill()

# ── Build price signals for each strategy ─────────────────────────────────────

# Strategy 1: Pure RT arbitrage — use real-time LMP directly
price_rt = df["lmp_rt"].copy()

# Strategy 2: DA arbitrage — use day-ahead LMP (what was known night before)
price_da = df["lmp_da"].copy()

# Strategy 3: Scarcity — reliability dispatch. Settles against the REAL RT LMP
# (no fictional premium), but the battery may only DISCHARGE during high
# grid-stress intervals; otherwise it holds charge in reserve for the next
# stress event. This is the most constrained strategy, so it is the floor.
price_scarcity        = df["lmp_rt"].copy()
high_stress           = df["grid_stress_mwh"] > GRID_STRESS_HIGH_THRESHOLD
scarcity_discharge_ok = high_stress

# Strategy 4: Hybrid — also reserves for stress events, but additionally permits
# opportunistic discharge during the most expensive price intervals (top
# HYBRID_ARBITRAGE_WEIGHT share). Its allowed-discharge set is a SUPERSET of
# scarcity's and a SUBSET of unconstrained arbitrage's, so — using the same RT
# price for all three — its revenue lands between them by construction.
price_hybrid        = df["lmp_rt"].copy()
price_cutoff        = df["lmp_rt"].quantile(1 - HYBRID_ARBITRAGE_WEIGHT)
hybrid_discharge_ok = high_stress | (df["lmp_rt"] >= price_cutoff)

# ── Run LP for each strategy ───────────────────────────────────────────────────
# Each entry is (price_signal, discharge_mask). None mask = discharge allowed
# everywhere (unconstrained arbitrage).
strategies = {
    "arbitrage_real_time": (price_rt,       None),
    "arbitrage_day_ahead": (price_da,       None),
    "scarcity":     (price_scarcity, scarcity_discharge_ok),
    "hybrid":       (price_hybrid,   hybrid_discharge_ok),
}

print("\nRunning LP optimizations...")
results = {}
for name, (signal, discharge_ok) in strategies.items():
    print(f"  [{name}] solving...", end=" ", flush=True)
    mask = None if discharge_ok is None else discharge_ok.reset_index(drop=True).values
    res = run_lp(
        price_signal=signal.reset_index(drop=True),
        capacity_mw=cap_mw,
        capacity_mwh=cap_mwh,
        rte=rte,
        max_cycles=max_cycles,
        discharge_mask=mask,
    )
    results[name] = res
    total_rev = res["optimal_revenue_usd"].sum()
    print(f"done  →  ${total_rev:,.0f} optimal revenue")

# ── Add all optimal columns to main dataframe ─────────────────────────────────
for name, res in results.items():
    df[f"optimal_{name}_kw"]          = res["optimal_power_kw"].values
    df[f"optimal_{name}_soc_pct"]     = res["optimal_soc_pct"].values
    df[f"optimal_{name}_revenue_usd"] = res["optimal_revenue_usd"].values

# ── Actual revenue (using RT LMP as settlement price) ─────────────────────────
actual_kw       = df["power_kw"].fillna(0)
rev_actual      = actual_revenue(actual_kw, df["lmp_rt"])
df["actual_revenue_usd"] = actual_kw / 1000 * df["lmp_rt"] * (1/12)

# ── Revenue summary ────────────────────────────────────────────────────────────
rev_summary = {}
for name, res in results.items():
    rev_summary[name] = res["optimal_revenue_usd"].sum()

best_strategy = max(rev_summary, key=rev_summary.get)

# ── Strategy match scores ──────────────────────────────────────────────────────
match_scores = {}
for name, res in results.items():
    match_scores[name] = strategy_match_score(
        actual_kw, pd.Series(res["optimal_power_kw"].values)
    )
best_match = max(match_scores, key=lambda k: match_scores[k]
                 if not np.isnan(match_scores[k]) else -999)

# ── Revenue leakage ────────────────────────────────────────────────────────────
leakage_vs_best = rev_summary[best_strategy] - rev_actual


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"\n  Project:    {df['project'].iloc[0]}")
print(f"  Period:     {df['time'].min().date()} to {df['time'].max().date()} "
      f"({n_days} days)")

print(f"\n  Revenue by strategy (${n_days}-day period):")
print(f"    {'Strategy':<25} {'Optimal Revenue':>16}  {'vs Actual':>12}")
print(f"    {'-'*25} {'-'*16}  {'-'*12}")
for name in strategies:
    diff = rev_summary[name] - rev_actual
    marker = " ◄ BEST" if name == best_strategy else ""
    print(f"    {name:<25} ${rev_summary[name]:>14,.0f}  "
          f"${diff:>+11,.0f}{marker}")
print(f"    {'ACTUAL':<25} ${rev_actual:>14,.0f}")

print(f"\n  Revenue leakage vs best strategy: ${leakage_vs_best:,.0f}")
print(f"  Leakage per day:                  ${leakage_vs_best/n_days:,.0f}/day")

print(f"\n  Strategy match scores (correlation with actual dispatch):")
for name, score in sorted(match_scores.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(abs(score) * 20) if not np.isnan(score) else ""
    marker = " ◄ CLOSEST MATCH" if name == best_match else ""
    print(f"    {name:<25} {score:>+.3f}  {bar}{marker}")

print(f"\n  Interpretation:")
print(f"    The utility's actual dispatch most closely resembles the")
print(f"    '{best_match}' strategy (correlation: {match_scores[best_match]:.3f}).")
if leakage_vs_best > 0:
    print(f"    Compared to the optimal '{best_strategy}' strategy, the utility")
    print(f"    left ${leakage_vs_best:,.0f} on the table over {n_days} days "
          f"(${leakage_vs_best/n_days:,.0f}/day).")
else:
    print(f"    Actual dispatch captured revenue within range of all strategies.")


# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL ANALYSIS — how, when, and to which signal the utility responds
# ══════════════════════════════════════════════════════════════════════════════
# The single match score above answers "which strategy does actual dispatch
# resemble overall?" These three views go deeper: WHEN it aligns (Route 1),
# WHETHER it lags the signal (Route 2), and WHAT drives each decision (Route 3).

print("\n" + "="*60)
print("BEHAVIORAL ANALYSIS")
print("="*60)

# Derived fields (also written to the export): local hour-of-day + dispatch flag.
df["hour"]       = df["time"].dt.tz_convert("America/Los_Angeles").dt.hour
# Cyclic encoding of hour-of-day: maps 0..23 onto a circle so 11pm and midnight
# are neighbors (not 23 apart) and the decision model can fit a daily bump.
df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
df["discharged"] = (df["power_kw"] > DISCHARGE_THRESHOLD_KW).astype(int)
price_hi         = df["lmp_rt"].quantile(HIGH_PRICE_QUANTILE)

# ── Route 1: feature-conditioned correlation ─────────────────────────────────
print(f"\n  Route 1 - Correlation of actual vs optimal dispatch, by regime")
print(f"    (hi_price = RT LMP > ${price_hi:,.0f};  hi_soc = SOC > {HIGH_SOC_PCT}%)")
hdr = (f"    {'Strategy':<22}{'all':>8}{'hi_price':>10}"
       f"{'lo_price':>10}{'hi_soc':>9}{'lo_soc':>9}")
print(hdr)
print("    " + "-"*(len(hdr)-4))
cfmt = lambda x: f"{x:+.3f}" if not np.isnan(x) else "n/a"
for name in strategies:
    c = conditioned_correlations(df, "power_kw", f"optimal_{name}_kw",
                                 price_hi, HIGH_SOC_PCT)
    print(f"    {name:<22}{cfmt(c['all']):>8}{cfmt(c['hi_price']):>10}"
          f"{cfmt(c['lo_price']):>10}{cfmt(c['hi_soc']):>9}{cfmt(c['lo_soc']):>9}")

# ── Route 2: lag cross-correlation (vs the closest-match strategy) ───────────
print(f"\n  Route 2 - Timing: does actual dispatch lag the '{best_match}' signal?")
lag = lagged_correlation(df["power_kw"], df[f"optimal_{best_match}_kw"],
                         MAX_LAG_INTERVALS)
if lag is None:
    print("    Insufficient data for lag analysis.")
else:
    mins = lag["best_lag"] * 5
    direction = ("actual FOLLOWS optimal" if lag["best_lag"] > 0
                 else "actual LEADS optimal" if lag["best_lag"] < 0
                 else "in phase")
    print(f"    Peak corr {lag['best_corr']:+.3f} at lag {lag['best_lag']:+d} "
          f"intervals ({mins:+d} min)  ->  {direction}")
    print(f"    Corr at zero lag: {lag['zero_corr']:+.3f}   "
          f"(timing-alignment uplift: {lag['best_corr']-lag['zero_corr']:+.3f})")

# ── Route 3: decision-threshold model ────────────────────────────────────────
print(f"\n  Route 3 - What drives the discharge decision? (logistic regression)")
dm = dispatch_decision_model(df, DECISION_FEATURES, DISCHARGE_THRESHOLD_KW)
if "error" in dm:
    print(f"    Skipped: {dm['error']}")
    if dm["error"].startswith("scikit"):
        print("    -> install with:  python -m pip install scikit-learn")
else:
    auc_txt = f"   |   holdout AUC: {dm['auc']:.3f}" if not np.isnan(dm["auc"]) else ""
    print(f"    Discharge events: {dm['discharge_rate']*100:.1f}% of "
          f"{dm['n']:,} intervals{auc_txt}")
    print(f"    Standardized coefficients (sign = direction, |size| = influence):")
    for feat, coef in dm["coefs"]:
        bar = "#" * int(min(abs(coef), 3) / 3 * 20)
        print(f"      {feat:<18}{coef:>+7.3f}  {bar}")
    if dm["shap"]:
        print(f"    Mean |SHAP| (overall feature impact):")
        for feat, val in dm["shap"]:
            print(f"      {feat:<18}{val:>7.3f}")
    else:
        print("    (install 'shap' for per-decision explanations)")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

project    = df["project"].iloc[0]
start_str  = str(df["time"].min().date())
end_str    = str(df["time"].max().date())
out_file   = f"{project}_optimized_{start_str}_to_{end_str}.csv"

df.to_csv(out_file, index=False)

print(f"\n  Output saved: {out_file}")
print(f"  Columns added: " + ", ".join(
    [f"optimal_{n}_kw" for n in strategies] +
    [f"optimal_{n}_revenue_usd" for n in strategies] +
    ["actual_revenue_usd"]
))
print("\n" + "="*60)
print("Optimizer complete. Ready for LLM narrative layer.")
print("="*60)
