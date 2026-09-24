GARCH Monte Carlo Simulation & Risk Analysis

This project implements an advanced quantitative finance pipeline in Python to simulate future price paths and assess market risk for the S&P 500 index (^GSPC) using a GARCH(1,1) volatility model combined with Monte Carlo simulation. 

Overview
The script automatically fetches historical closing prices from Yahoo Finance starting from January 2021, computes log returns, and fits a GARCH(1,1) model via the arch library to capture time-varying volatility and clustering effects. By extracting the fitted parameters and the latest conditional volatility, it runs 2,000 parallel path simulations over a 365-step horizon. The model dynamically updates variance step-by-step using random normal shocks, projecting future prices and evaluating key risk metrics such as the 95% Value at Risk. Finally, it generates a comprehensive visualization featuring individual simulation paths, the expected mean trajectory, key price levels, and a statistical annotation box.

Requirements & Installation
To execute the script, ensure you have Python installed along with the required financial and data science libraries. You can install them via terminal by running the following command:

```bash
pip install arch yfinance pandas numpy matplotlib
