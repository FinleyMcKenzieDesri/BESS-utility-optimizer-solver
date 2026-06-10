"""
DESRI Battery Dispatch Data Pipeline
=====================================
Ingests a PowerFactors CSV export, pulls matching LMP and EIA-930 data
for the same date range, aligns everything to 5-minute resolution,
computes RTE from numerator/denominator columns, and exports one clean
unified CSV per project — ready for the optimizer or ML input.

Handles single-project and multi-project CSV exports automatically.

USAGE:
    1. Export from PowerFactors with these 7 signals per project:
           Effective Irradiance, SOC, Active Power BESS,
           Active Export Energy BESS, Active Import Energy BESS,
           Round Trip Efficiency Denominator, Round Trip Efficiency Numerator
    2. Fill in the CONFIGURATION block below
    3. Run the script
    4. Find one unified CSV per project in the same folder

REQUIREMENTS:
    pip install gridstatus pandas requests pytz
"""

import re
import warnings
import requests
import pandas as pd
import pytz
import gridstatus

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit all values in this block before running
# ══════════════════════════════════════════════════════════════════════════════

# API + file
EIA_API_KEY       = "Enter_your_EIA_API_key_here"  # or set EIA_API_KEY env variable
POWERFACTORS_FILE = r"Enter_path_to_your_PowerFactors_export_here.csv"  # e.g. r"C:\VS code\PowerFactors_export.csv"

# ── Region selection ────────────────────────────────────────────────────────
# Each entry bundles everything that varies by ISO. To add a region later,
# add one dictionary entry — no logic changes needed.
REGION_CONFIG = {
    "CAISO": {
        "iso_class":      "CAISO",          # gridstatus class name
        "eia_respondent": "CISO",           # EIA-930 respondent code
        "timezone":       "America/Los_Angeles",
        # NP15 = Northern CA, SP15 = Southern CA, ZP26 = Central CA
        "default_node":   "TH_NP15_GEN-APND",
    },
    "ERCOT": {
        "iso_class":      "Ercot",
        "eia_respondent": "ERCO",
        "timezone":       "America/Chicago",
        "default_node":   "HB_HOUSTON",     # VERIFY node string before relying on it
    },
    "PJM": {
        "iso_class":      "PJM",
        "eia_respondent": "PJM",
        "timezone":       "America/New_York",
        "default_node":   "PJM-RTO",        # VERIFY node string before relying on it
    },
    "MISO": {
        "iso_class":      "MISO",
        "eia_respondent": "MISO",
        "timezone":       "America/New_York",
        "default_node":   "ARKANSAS.HUB",   # VERIFY node string before relying on it
    },
    "SPP": {
        "iso_class":      "SPP",
        "eia_respondent": "SWPP",
        "timezone":       "America/Chicago",
        "default_node":   "SPPNORTH_HUB",   # VERIFY node string before relying on it
    },
    "NYISO": {
        "iso_class":      "NYISO",
        "eia_respondent": "NYIS",
        "timezone":       "America/New_York",
        "default_node":   "N.Y.C.",         # VERIFY node string before relying on it
    },
    "ISONE": {
        "iso_class":      "ISONE",
        "eia_respondent": "ISNE",
        "timezone":       "America/New_York",
        "default_node":   "NEMASSBOST",     # VERIFY node string before relying on it
    },
}

REGION = "CAISO"            # pick one key from REGION_CONFIG above
NODE   = None               # None → use the region's default_node, or paste a specific node

# Battery physical specs — used by the optimizer script
BATTERY_CAPACITY_MW   = 150     # max charge/discharge rate (MW)
BATTERY_CAPACITY_MWH  = 600     # total energy storage capacity (MWh)
                                # if unknown, use: BATTERY_CAPACITY_MW * 4
MAX_CYCLES_PER_YEAR   = 365     # warranty/contractual full-cycle limit per year
RTE_FALLBACK          = 0.90    # assumed RTE if insufficient data to calculate
                                # (used by optimizer when measured RTE unavailable)

# ── Internal ──────────────────────────────────────────────────────────────────
cfg     = REGION_CONFIG[REGION]
NODE    = NODE or cfg["default_node"]
PACIFIC = pytz.timezone(cfg["timezone"])   # local timezone for the selected region

# Signal keywords used to auto-detect and map PowerFactors columns
# Edit only if PowerFactors changes its column naming convention
SIGNAL_PATTERNS = {
    "irradiance_wm2":       "Effective Irradiance",
    "soc_pct":              "SOC",
    "power_kw":             "Active Power BESS",
    "export_kwh":           "Active Export Energy BESS",
    "import_kwh":           "Active Import Energy BESS",
    "rte_denominator_kwh":  "Round Trip Efficiency.Denominator",
    "rte_numerator_kwh":    "Round Trip Efficiency.Numerator",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def detect_projects(columns):
    """Extract unique project names from PowerFactors column headers.
    Headers look like: '- / Arroyo - Active Power BESS.Final (kW) (5m)'
    """
    projects = []
    for col in columns[1:]:  # skip Time column
        m = re.search(r"/\s+([^-]+?)\s+-", col)
        if m:
            name = m.group(1).strip()
            if name not in projects:
                projects.append(name)
    return projects


def map_project_columns(columns, project_name, signal_patterns):
    """Return {clean_name: original_column} for one project."""
    mapping = {}
    for clean_name, keyword in signal_patterns.items():
        for col in columns:
            if project_name in col and keyword in col:
                mapping[clean_name] = col
                break
    return mapping


def compute_rte(df, window=12):
    """
    Compute rolling cycle RTE from numerator/denominator columns.
    PowerFactors accumulates these across a rolling window rather than
    per-interval, so raw interval values are noisy. We take the median
    of valid (both nonzero, ratio between 0.5 and 1.05) cycle observations
    and return a single scalar representing the project's effective RTE.
    Falls back to RTE_FALLBACK if insufficient data.
    """
    num = pd.to_numeric(df["rte_numerator_kwh"],   errors="coerce")
    den = pd.to_numeric(df["rte_denominator_kwh"],  errors="coerce")

    # Only compute where both are positive
    mask  = (num > 0) & (den > 0)
    ratio = (num[mask] / den[mask])

    # Filter to plausible RTE range (50%–105%)
    valid = ratio[(ratio >= 0.50) & (ratio <= 1.05)]

    if len(valid) >= 10:
        rte = float(valid.median())
        print(f"    Measured RTE: {rte:.3f} ({rte*100:.1f}%)  "
              f"[from {len(valid)} valid cycle observations]")
        return rte
    else:
        print(f"    Insufficient RTE data ({len(valid)} valid obs) — "
              f"using fallback: {RTE_FALLBACK}")
        return RTE_FALLBACK


def get_lmp_chunked(iso, start_date, end_date, market, node, chunk_days=7):
    """
    Pull LMP in chunked date windows and concatenate the results.

    CAISO OASIS rejects (or returns empty for) a single REAL_TIME_5_MIN request
    spanning many days — the 5-min dataset is simply too large for one SingleZip
    call — which surfaces downstream as gridstatus' "No objects to concatenate".
    Day-ahead hourly is small enough to pull in one shot, but RT 5-min is ~288
    intervals/day, so a month-long pull must be split. Each window is at most
    `chunk_days` days; `end` is exclusive, matching iso.get_lmp(date=..., end=...).
    Windows are non-overlapping, so concatenation introduces no duplicate rows.
    """
    frames  = []
    w_start = pd.Timestamp(start_date)
    end_ts  = pd.Timestamp(end_date)
    while w_start < end_ts:
        w_end = min(w_start + pd.Timedelta(days=chunk_days), end_ts)
        try:
            part = iso.get_lmp(
                date=str(w_start.date()),
                end=str(w_end.date()),
                market=market,
                locations=[node],
            )
        except Exception as e:
            print(f"    WARNING: {market} pull failed for "
                  f"{w_start.date()}→{w_end.date()}: {e}")
            part = None
        if part is not None and len(part):
            frames.append(part)
        w_start = w_end

    if not frames:
        raise RuntimeError(
            f"No {market} data returned for {node} from {start_date} to "
            f"{end_date}. CAISO OASIS may be unavailable, or the range includes "
            f"dates not yet published. Try a smaller/older date range."
        )
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD POWERFACTORS CSV + DETECT PROJECTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 1: Loading PowerFactors CSV")
print("="*60)

raw = pd.read_csv(POWERFACTORS_FILE, encoding="utf-8-sig")
print(f"  Loaded {len(raw)} rows, {len(raw.columns)} columns")

projects = detect_projects(raw.columns)
print(f"  Detected {len(projects)} project(s): {projects}")

# Parse timestamps
raw["Time"] = pd.to_datetime(raw["Time"])
# ambiguous=False resolves the DST "fall-back" hour (e.g. 2025-11-02 01:00,
# which occurs twice in Pacific) deterministically to standard time (PST),
# rather than "infer", which crashes when the export doesn't contain both
# copies of the repeated hour. nonexistent handles the spring-forward gap.
raw["Time"] = raw["Time"].dt.tz_localize(
    PACIFIC, ambiguous=False, nonexistent="shift_forward"
)
raw = raw.set_index("Time").sort_index()

start_date = raw.index.min().date()
end_date   = raw.index.max().date()
print(f"  Date range: {start_date} to {end_date}")
print(f"  Total 5-min intervals: {len(raw)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PULL ISO LMP (Day-Ahead + Real-Time)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(f"STEP 2: Pulling {REGION} LMP from gridstatus (node: {NODE})")
print("="*60)

caiso = getattr(gridstatus, cfg["iso_class"])()

print(f"  Pulling Day-Ahead LMP for {start_date} to {end_date}...")
lmp_da_raw = caiso.get_lmp(
    date=str(start_date),
    end=str(end_date),
    market="DAY_AHEAD_HOURLY",
    locations=[NODE],
)
print(f"  Got {len(lmp_da_raw)} hourly DA rows")

print(f"  Pulling Real-Time 5-min LMP for {start_date} to {end_date}...")
# Chunked pull: a single month-long RT 5-min request overruns CAISO OASIS and
# comes back empty ("No objects to concatenate"). Weekly windows stay within
# limits. `end` stays exclusive and identical to the original single-shot call
# (and to the DA pull below), so coverage is unchanged — only the request is split.
lmp_rt_raw = get_lmp_chunked(
    caiso, start_date, end_date,
    market="REAL_TIME_5_MIN", node=NODE, chunk_days=7,
)
print(f"  Got {len(lmp_rt_raw)} real-time rows")

# Clean DA — upsample hourly to 5-min by forward fill
lmp_da = (
    lmp_da_raw[["Interval Start", "LMP"]]
    .rename(columns={"Interval Start": "time", "LMP": "lmp_da"})
    .copy()
)
lmp_da["time"] = pd.to_datetime(lmp_da["time"]).dt.tz_convert(PACIFIC)
lmp_da = lmp_da.set_index("time").sort_index()
lmp_da_5min = lmp_da.resample("5min").ffill()

# Clean RT — already 5-min
lmp_rt = (
    lmp_rt_raw[["Interval Start", "LMP"]]
    .rename(columns={"Interval Start": "time", "LMP": "lmp_rt"})
    .copy()
)
lmp_rt["time"] = pd.to_datetime(lmp_rt["time"]).dt.tz_convert(PACIFIC)
lmp_rt = lmp_rt.set_index("time").sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PULL EIA-930 DEMAND + FORECAST
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3: Pulling EIA-930 Demand and Forecast")
print("="*60)

start_utc = PACIFIC.localize(
    pd.Timestamp(start_date)
).astimezone(pytz.utc)
end_utc = PACIFIC.localize(
    pd.Timestamp(end_date) + pd.Timedelta(days=1)
).astimezone(pytz.utc)

eia_params = {
    "api_key": EIA_API_KEY,
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": cfg["eia_respondent"],
    "start": start_utc.strftime("%Y-%m-%dT%H"),
    "end":   end_utc.strftime("%Y-%m-%dT%H"),
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "length": 5000,
}
eia_params["facets[type][0]"] = "D"
eia_params["facets[type][1]"] = "DF"

print(f"  Pulling EIA-930 for {cfg['eia_respondent']} {start_date} to {end_date}...")
eia_resp = requests.get(
    "https://api.eia.gov/v2/electricity/rto/region-data/data/",
    params=eia_params,
)
eia_raw = pd.DataFrame(eia_resp.json()["response"]["data"])
print(f"  Got {len(eia_raw)} EIA rows")

eia_pivot = eia_raw.pivot_table(
    index="period", columns="type", values="value", aggfunc="first"
).reset_index()
eia_pivot.columns.name = None
eia_pivot = eia_pivot.rename(columns={
    "period": "time",
    "D":  "eia_demand_mwh",
    "DF": "eia_forecast_mwh",
})
eia_pivot["time"] = (
    pd.to_datetime(eia_pivot["time"])
    .dt.tz_localize("UTC")
    .dt.tz_convert(PACIFIC)
)
for col in ["eia_demand_mwh", "eia_forecast_mwh"]:
    eia_pivot[col] = pd.to_numeric(eia_pivot[col], errors="coerce")

eia_pivot = eia_pivot.set_index("time").sort_index()
eia_5min  = eia_pivot.resample("5min").ffill()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD ONE UNIFIED CSV PER PROJECT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 4: Building unified dataframes")
print("="*60)

output_files = []

for project in projects:
    print(f"\n  ── Project: {project} ──────────────────────────────")

    # Map columns for this project
    col_map = map_project_columns(raw.columns, project, SIGNAL_PATTERNS)
    missing_signals = [s for s in SIGNAL_PATTERNS if s not in col_map]
    if missing_signals:
        print(f"    WARNING: Missing signals: {missing_signals}")

    # Extract project columns and rename
    pf = raw[[v for v in col_map.values() if v in raw.columns]].copy()
    pf = pf.rename(columns={v: k for k, v in col_map.items()})

    # Convert all to numeric
    for col in pf.columns:
        pf[col] = pd.to_numeric(pf[col], errors="coerce")

    # Compute RTE
    print(f"    Computing RTE...")
    measured_rte = compute_rte(pf)

    # Drop raw RTE columns — keep only the computed scalar (stored in metadata row)
    # The scalar gets written to the config summary at the bottom
    pf = pf.drop(columns=["rte_denominator_kwh", "rte_numerator_kwh"], errors="ignore")

    # Merge market data
    unified = pf.copy()
    unified = unified.join(lmp_da_5min, how="left")
    unified = unified.join(lmp_rt,      how="left")
    unified = unified.join(eia_5min,    how="left")

    # Derived columns
    unified["grid_stress_mwh"] = (
        unified["eia_demand_mwh"] - unified["eia_forecast_mwh"]
    )

    # Add metadata columns
    unified.insert(0, "project",          project)
    unified.insert(1, "battery_mw",       BATTERY_CAPACITY_MW)
    unified.insert(2, "battery_mwh",      BATTERY_CAPACITY_MWH)
    unified.insert(3, "rte",              round(measured_rte, 4))
    unified.insert(4, "max_cycles_year",  MAX_CYCLES_PER_YEAR)

    unified = unified.reset_index().rename(columns={"Time": "time"})

    print(f"    Unified shape: {unified.shape[0]} rows × {unified.shape[1]} columns")
    print(f"    Columns: {list(unified.columns)}")

    # ── Data quality summary ───────────────────────────────────────────────
    print(f"\n    Data Quality:")
    total = len(unified)
    for col in ["soc_pct", "power_kw", "export_kwh", "import_kwh",
                "lmp_da", "lmp_rt", "eia_demand_mwh", "grid_stress_mwh"]:
        n_missing = unified[col].isnull().sum()
        pct = n_missing / total * 100
        status = "clean" if n_missing == 0 else f"{n_missing} missing ({pct:.1f}%)"
        print(f"      {col:<25} {status}")

    # Timestamp gap check
    expected = pd.date_range(
        start=unified["time"].min(),
        end=unified["time"].max(),
        freq="5min", tz=PACIFIC,
    )
    gap_count = len(set(expected) - set(unified["time"]))
    print(f"      Timestamp gaps:           {gap_count}")

    # Power spike check
    spikes = unified[unified["power_kw"].abs() > BATTERY_CAPACITY_MW * 1000 * 1.1]
    print(f"      Power spikes >110% cap:   {len(spikes)}")

    # SOC range check
    soc_bad = unified[(unified["soc_pct"] < 0) | (unified["soc_pct"] > 100)]
    print(f"      SOC out of range:         {len(soc_bad)}")

    # Key stats
    print(f"\n    Key statistics:")
    print(unified[["power_kw", "soc_pct", "lmp_da", "lmp_rt", "grid_stress_mwh"]]
          .describe().round(2).to_string())

    # Export
    fname = f"{project}_unified_{start_date}_to_{end_date}.csv"
    unified.to_csv(fname, index=False)
    output_files.append(fname)
    print(f"\n    Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("Pipeline complete.")
print(f"Output files ({len(output_files)}):")
for f in output_files:
    print(f"  {f}")
print("="*60)
print()
print("Next step: run DesriOptimizer.py pointing at any of these files.")
