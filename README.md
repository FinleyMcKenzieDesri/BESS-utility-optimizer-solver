# BESS-utility-optimizer-solver
Correlate Power Factors BESS data with market data to understand utilities behavior regarding dispatch of BESS systems
Currently set to the CAISO region

You may have to pip install various plugins

In the DesriPipeline.py document replace the following lines of code with your own data

Line 38:  EIA_API_KEY = "your API key from https://www.eia.gov/opendata/"

Line 39: POWERFACTORS_FILE = r"your system route to Power Factors Spreadsheet"
 
  spreadsheet must contain  Effective Irradiance, SOC, Active Power BESS,
  Active Export Energy BESS, Active Import Energy BESS,
  Round Trip Efficiency Denominator, Round Trip Efficiency Numerator

Line 48: BATTERY_CAPACITY_MW   = # max charge/discharge rate (MW)

Line 49: BATTERY_CAPACITY_MWH  = # total energy storage capacity (MWh)
                                # if unknown, use: BATTERY_CAPACITY_MW * 4
Line 51: MAX_CYCLES_PER_YEAR   = # warranty/contractual full-cycle limit per year

Line 52: RTE_FALLBACK          = # assumed RTE if insufficient data to calculate

DesriPipelineTemplate.py considerations
Regional Configuration — Known Limitations
This pipeline supports multiple ISOs (CAISO, ERCOT, PJM, MISO, SPP, NYISO, ISONE) via the REGION_CONFIG selector. Be aware of the following before trusting output for any non-CAISO region:

Node names are not validated. Each ISO uses its own LMP node naming convention. The default_node for each region is a placeholder hub, not a project-specific settlement point. Always replace it with the actual settlement node from the project's interconnection agreement. A wrong or stale node string will either error out or silently return the wrong location's prices.
Zonal vs. nodal pricing differ. Hub/zone prices can diverge significantly from a project's actual settlement node, especially in congested areas. Default hubs give a regional approximation only.
EIA-930 respondent codes are region-specific and must match the selected ISO. A mismatch between the ISO price data and the EIA demand region will misalign the grid-stress signal without raising an error.
Timezones are configured per region but not cross-checked against PowerFactors timestamps. PowerFactors exports carry no timezone; the pipeline assumes local time for the selected region. If a project's data is exported in a different timezone than its market region, all time alignment (and therefore the optimizer results) will be silently offset.
Non-ISO regions are unsupported. Much of the Southeast and Mountain West does not operate organized markets with public nodal LMPs. Projects in these areas cannot be analyzed with this pipeline as-is.
Only CAISO has been validated end-to-end. All other regions share the same code path but have not been tested against known-good output. Treat first runs in a new region as unverified until spot-checked against ISO portal data.



DesriOptimizer.py
Methods Used to Infer Utility Dispatch Strategy
The optimizer infers which strategy a utility's behavior aligns with using four layered methods, in increasing depth:
1. Linear Program (dispatch optimization) — PuLP / CBC solver
Generates the optimal dispatch for each strategy (RT arbitrage, DA arbitrage, scarcity, hybrid) under physical constraints (power cap, energy cap, RTE, cycle limit). This produces the benchmark each strategy is measured against. Note: assumes perfect foresight; outputs are theoretical ceilings.
2. Pearson correlation (global match score)
Correlates actual dispatch power against each optimal strategy's dispatch power across all intervals. Highest correlation = closest behavioral match. Measures directional co-movement only; insensitive to magnitude and timing.
3. Regime-conditioned correlation (Route 1)
Repeats the correlation within data subsets — high vs. low price, high vs. low SOC. Reveals conditional behavior (e.g. a utility that tracks arbitrage only above a price floor). Surfaces when alignment holds rather than assuming it's constant.
4. Lagged cross-correlation (Route 2)
Correlates actual dispatch against the best-match strategy shifted across a ±3-hour window. Identifies whether the utility tracks a signal on a consistent time delay (e.g. an internal scheduling cycle). Uses per-lag Pearson rather than raw cross-correlation to avoid magnitude dominance.
5. Logistic regression (Route 3 — decision model) — scikit-learn
Models the binary discharge decision P(discharge) from market and state features (RT/DA LMP, SOC, grid stress, hour-of-day cyclically encoded, irradiance). Reports standardized coefficients (comparable across features) showing which signals drive dispatch and in which direction. Validated with a time-ordered holdout AUC. This is the only method that models the decision function directly rather than comparing curves.
6. SHAP values (Route 3 extension) — shap, optional
Decomposes the logistic model's per-decision feature contributions, giving an explainable breakdown of what drove each dispatch decision. Skipped silently if shap is not installed.
Interpretation caveat for the repo: None of these establish intent or causation. They establish statistical association between observed dispatch and each candidate strategy. The correlation/regression outputs are evidence to inform a human conclusion, not an automated verdict.

**** note only working for solar currently
