S&P 500 Advanced GARCH(1,1) Monte Carlo Simulation & Risk Analysis

This repository contains a professional quantitative finance script designed to forecast future price trajectories and evaluate market risk for the S&P 500 index (^GSPC). By blending econometric volatility modeling with stochastic simulation techniques, the project moves beyond standard constant-variance assumptions to capture the complex, time-varying nature of real-world financial markets.

Methodology & Architecture
The script begins by automatically downloading historical daily closing prices from Yahoo Finance starting from January 2021. It computes percentage log returns and fits a GARCH(1,1) model using the specialized arch library. This econometric modeling step is fundamental because financial time series frequently exhibit volatility clustering, a phenomenon where periods of high market turbulence tend to cluster together. Once the model is fitted, the script extracts key parameters including the constant mean, the omega baseline variance, the ARCH alpha parameter measuring the reaction to recent market shocks, and the GARCH beta parameter representing the persistence of volatility over time.

Using the latest conditional volatility and the most recent closing price as the starting baseline, the architecture executes two thousand parallel Monte Carlo paths. Unlike traditional geometric Brownian motion that assumes constant variance, this simulation dynamically updates the conditional variance at each step using past residuals and random shocks drawn from a normal distribution. This recursive mechanism allows the simulation to reflect realistic market stress, calm periods, and sudden volatility spikes rather than relying on flat statistical assumptions.

Risk Evaluation & Output
At the final step of the simulation horizon, the script aggregates all terminal prices to compute the expected future price path and evaluate downside risk. It calculates the return distribution relative to the initial entry price and extracts the 95% Value at Risk, providing a rigorous statistical metric for potential capital loss under normal adverse market conditions. The output is rendered through a publication-quality Matplotlib visualization that displays every simulated path with low opacity, highlights the crimson expected mean trajectory, sets horizontal boundaries for current and expected prices, and features an integrated statistics box summarizing the core financial findings directly on the chart.

Installation & Execution
To run this analysis locally, ensure you have Python installed along with the required econometric and data visualization packages. You can install all dependencies directly from your terminal by executing the following command:

```bash
pip install arch yfinance pandas numpy matplotlib
