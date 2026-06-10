"""
DESRI Pipeline Launcher
========================
One script to rule them all. Point it at a PowerFactors CSV export and it
runs the full pipeline automatically:

    DesriPipeline.py  →  unified CSV
    DesriOptimizer.py →  optimized CSV
    DesriReport.py    →  single HTML report (charts + AI interpretation)

USAGE:
    1. Set POWERFACTORS_FILE below to your CSV export path
    2. Set REGION and any battery specs if different from defaults
    3. Run:  python DesriRun.py
    4. Open the HTML report that appears in OUTPUT_DIR

REQUIREMENTS:
    All requirements from DesriPipeline, DesriOptimizer, and DesriReport:
    pip install pandas numpy pulp gridstatus requests pytz plotly
    Optional: pip install scikit-learn shap kaleido
"""

import os
import sys
import subprocess
import re
import time

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this block only
# ══════════════════════════════════════════════════════════════════════════════

POWERFACTORS_FILE = r"C:\VS code\2026-05-12 - 2026-06-11 - 7 Devices - 7 Signals (2).csv"
OUTPUT_DIR        = r"C:\VS code"

# Where the four scripts live (default: same folder as this launcher)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Region + battery specs — passed through to DesriPipeline.py
REGION                = "CAISO"
NODE                  = None       # None = use region default
BATTERY_CAPACITY_MW   = 150
BATTERY_CAPACITY_MWH  = 600
MAX_CYCLES_PER_YEAR   = 365
RTE_FALLBACK          = 0.90

# EIA API key — or set EIA_API_KEY environment variable
EIA_API_KEY = os.environ.get("EIA_API_KEY", "Insert_your_EIA_API_key_here")

# Anthropic API key — or set ANTHROPIC_API_KEY environment variable
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "Insert_your_Anthropic_API_key_here")

# ══════════════════════════════════════════════════════════════════════════════
# LAUNCHER
# ══════════════════════════════════════════════════════════════════════════════

def banner(text):
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


def patch_config(script_path, replacements: dict) -> str:
    """
    Read a script, apply key=value substitutions in the CONFIGURATION block,
    write a temporary patched copy, and return its path.

    replacements: {variable_name: new_python_literal_string}
    e.g. {"POWERFACTORS_FILE": 'r"C:\\path\\file.csv"'}
    """
    with open(script_path, "r", encoding="utf-8") as f:
        source = f.read()

    for var, new_val in replacements.items():
        # Match the FIRST top-level assignment of VAR at the start of a line:
        #   VAR = <value>   (raw strings, plain strings, os.environ.get(...),
        #   trailing inline comments, and irregular spacing around = all OK,
        #   since (.+) greedily captures the entire remainder of the line and
        #   we replace it wholesale with new_val).
        pattern = rf'^([ \t]*{re.escape(var)}[ \t]*=[ \t]*)(.+)$'
        # Use a function as the replacement so backslashes in new_val
        # (e.g. Windows paths) are inserted literally and NOT treated
        # as regex escape sequences like \g, \1, \V, etc.
        patched, n = re.subn(
            pattern,
            lambda m: m.group(1) + new_val,
            source,
            count=1,
            flags=re.MULTILINE,
        )
        # Only warn on a genuine no-match. (Previously this compared strings,
        # which falsely warned whenever new_val already equaled the existing
        # value — e.g. OUTPUT_DIR already set to the same path.)
        if n == 0:
            print(f"    WARNING: Could not patch '{var}' in {os.path.basename(script_path)}")
        source = patched

    tmp_path = script_path.replace(".py", "_run_tmp.py")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(source)
    return tmp_path


def run_script(script_path, label):
    """Run a Python script in a subprocess and stream its output."""
    print(f"\n  Running {label}...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # stream to console in real time
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ {label} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(1)
    print(f"\n  ✓ {label} completed in {elapsed:.1f}s")
    return elapsed


def infer_pipeline_output(powerfactors_file: str, output_dir: str) -> list[str]:
    """
    DesriPipeline.py saves files named:
        {project}_unified_{start_date}_to_{end_date}.csv
    We can't know the project names before running, so we scan for new
    *_unified_*.csv files that appear in OUTPUT_DIR after the pipeline runs.
    Returns a list of matching file paths.
    """
    pattern = re.compile(r"^.+_unified_.+_to_.+\.csv$")
    before  = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
    return before, pattern


def find_new_files(output_dir: str, before: set, pattern, min_mtime: float = 0.0) -> list[str]:
    after = set(os.listdir(output_dir))
    new   = after - before
    found = [
        os.path.join(output_dir, f)
        for f in new
        if pattern.match(f)
        and os.path.getmtime(os.path.join(output_dir, f)) >= min_mtime
    ]
    return found


# Wall-clock snapshot at the very start of the run. Only files written at or
# after this moment count as "produced by this run" — this filters out stale
# unified/optimized CSVs left in OUTPUT_DIR by previous runs.
RUN_START_TIME = time.time()

# ── Validate inputs ────────────────────────────────────────────────────────────
banner("DESRI Pipeline Launcher")
print(f"\n  Input:  {POWERFACTORS_FILE}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Region: {REGION}")

if not os.path.exists(POWERFACTORS_FILE):
    print(f"\n  ERROR: PowerFactors file not found:\n  {POWERFACTORS_FILE}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Script paths ──────────────────────────────────────────────────────────────
pipeline_script  = os.path.join(SCRIPTS_DIR, "DesriPipeline.py")
optimizer_script = os.path.join(SCRIPTS_DIR, "DesriOptimizer.py")
report_script    = os.path.join(SCRIPTS_DIR, "DesriReport.py")

for path, name in [(pipeline_script,  "DesriPipeline.py"),
                   (optimizer_script, "DesriOptimizer.py"),
                   (report_script,    "DesriReport.py")]:
    if not os.path.exists(path):
        print(f"\n  ERROR: {name} not found at:\n  {path}")
        sys.exit(1)

print(f"\n  Scripts found in: {SCRIPTS_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PATCH AND RUN DESRIPIPELINE
# ══════════════════════════════════════════════════════════════════════════════

banner("STEP 1 — DesriPipeline.py")

node_val = f'"{NODE}"' if NODE else "None"
pipeline_patches = {
    "POWERFACTORS_FILE":    f'r"{POWERFACTORS_FILE}"',
    "REGION":               f'"{REGION}"',
    "NODE":                 node_val,
    "BATTERY_CAPACITY_MW":  str(BATTERY_CAPACITY_MW),
    "BATTERY_CAPACITY_MWH": str(BATTERY_CAPACITY_MWH),
    "MAX_CYCLES_PER_YEAR":  str(MAX_CYCLES_PER_YEAR),
    "RTE_FALLBACK":         str(RTE_FALLBACK),
}
if EIA_API_KEY:
    pipeline_patches["EIA_API_KEY"] = f'"{EIA_API_KEY}"'

# Snapshot OUTPUT_DIR before running so we can detect new files
before_snap, unified_pattern = infer_pipeline_output(POWERFACTORS_FILE, OUTPUT_DIR)
# Pipeline saves to the working directory (cwd) by default — we'll run it from OUTPUT_DIR
pipeline_patches["POWERFACTORS_FILE"] = f'r"{POWERFACTORS_FILE}"'

pipeline_tmp = patch_config(pipeline_script, pipeline_patches)
try:
    # Run from OUTPUT_DIR so unified CSVs land there
    result = subprocess.run(
        [sys.executable, pipeline_tmp],
        cwd=OUTPUT_DIR,
        text=True,
    )
    if result.returncode != 0:
        print("\n  ✗ DesriPipeline FAILED")
        sys.exit(1)
    print("\n  ✓ DesriPipeline completed")
finally:
    if os.path.exists(pipeline_tmp):
        os.remove(pipeline_tmp)

# Detect unified CSVs produced — only files written during THIS run
unified_csvs = find_new_files(OUTPUT_DIR, before_snap, unified_pattern, RUN_START_TIME)
if not unified_csvs:
    # Fallback: scan unified CSVs in output dir, still filtered to this run
    unified_csvs = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if unified_pattern.match(f)
        and os.path.getmtime(os.path.join(OUTPUT_DIR, f)) >= RUN_START_TIME
    ]

if not unified_csvs:
    print("\n  ERROR: No unified CSV found after pipeline run.")
    print(f"  Expected a file matching *_unified_*_to_*.csv in {OUTPUT_DIR}")
    sys.exit(1)

print(f"\n  Unified CSV(s) produced:")
for f in unified_csvs:
    print(f"    {f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 + 3 — OPTIMIZER → REPORT for each project
# ══════════════════════════════════════════════════════════════════════════════

report_files = []

for unified_csv in unified_csvs:
    project_name = os.path.basename(unified_csv).replace("_unified_", "_").split("_")[0]
    banner(f"STEP 2 — DesriOptimizer  ({os.path.basename(unified_csv)})")

    # Patch and run optimizer
    optimizer_patches = {
        "UNIFIED_CSV": f'r"{unified_csv}"',
    }
    optimizer_tmp = patch_config(optimizer_script, optimizer_patches)
    try:
        result = subprocess.run(
            [sys.executable, optimizer_tmp],
            cwd=OUTPUT_DIR, text=True,
        )
        if result.returncode != 0:
            print("\n  ✗ DesriOptimizer FAILED")
            sys.exit(1)
        print("\n  ✓ DesriOptimizer completed")
    finally:
        if os.path.exists(optimizer_tmp):
            os.remove(optimizer_tmp)

    # Detect optimized CSV — named {project}_optimized_{dates}.csv.
    # Only consider files written during this run, newest first.
    opt_pattern   = re.compile(r"^.+_optimized_.+_to_.+\.csv$")
    optimized_csvs = sorted(
        [os.path.join(OUTPUT_DIR, f)
         for f in os.listdir(OUTPUT_DIR)
         if opt_pattern.match(f)
         and os.path.getmtime(os.path.join(OUTPUT_DIR, f)) >= RUN_START_TIME],
        key=os.path.getmtime, reverse=True,
    )
    if not optimized_csvs:
        print("\n  ERROR: No optimized CSV found after optimizer run.")
        sys.exit(1)

    optimized_csv = optimized_csvs[0]
    print(f"\n  Optimized CSV: {optimized_csv}")

    # ── STEP 3 — Report ───────────────────────────────────────────────────────
    banner(f"STEP 3 — DesriReport  ({os.path.basename(optimized_csv)})")

    report_patches = {
        "OPTIMIZED_CSV": f'r"{optimized_csv}"',
        "OUTPUT_DIR":    f'r"{OUTPUT_DIR}"',
    }
    if ANTHROPIC_API_KEY:
        # Inject the plain key string, not another os.environ.get(...) wrapper.
        report_patches["ANTHROPIC_API_KEY"] = f'"{ANTHROPIC_API_KEY}"'

    report_tmp = patch_config(report_script, report_patches)
    try:
        result = subprocess.run(
            [sys.executable, report_tmp],
            cwd=OUTPUT_DIR, text=True,
        )
        if result.returncode != 0:
            print("\n  ✗ DesriReport FAILED")
            sys.exit(1)
        print("\n  ✓ DesriReport completed")
    finally:
        if os.path.exists(report_tmp):
            os.remove(report_tmp)

    # Find the HTML report produced during this run, newest first
    html_pattern = re.compile(r"^.+_report_.+_to_.+\.html$")
    html_files   = sorted(
        [os.path.join(OUTPUT_DIR, f)
         for f in os.listdir(OUTPUT_DIR)
         if html_pattern.match(f)
         and os.path.getmtime(os.path.join(OUTPUT_DIR, f)) >= RUN_START_TIME],
        key=os.path.getmtime, reverse=True,
    )
    if html_files:
        report_files.append(html_files[0])


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

banner("COMPLETE")
print(f"\n  Pipeline finished. Report(s) ready:")
for r in report_files:
    print(f"\n    {r}")
print(f"\n  Open any .html file in a browser to view the full report.")
print()
