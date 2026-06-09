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
