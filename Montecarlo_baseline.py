import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ticker = "^GSPC"
data = yf.download(ticker, start="2021-01-01", end="2026-01-01")

if isinstance(data.columns, pd.MultiIndex):
  prices = data["Close"].iloc[:, 0]
else:
  prices = data["Close"]

log_returns = np.log(prices / prices.shift(1)).dropna()

mu = log_returns.mean() * 252
sigma = log_returns.std() * np.sqrt(252)
S_0 = prices.iloc[-1]

num_days = 365
num_simulations = 500

dt = 1 / 252

simulation_matrix = np.zeros((num_days, num_simulations))
simulation_matrix[0] = S_0

np.random.seed(42)
z = np.random.normal(loc=0.0, scale=1.0, size=(num_days - 1, num_simulations))

for t in range(1, num_days):
  drift_term = (mu - 0.5 * sigma**2) * dt
  diffusion_term = sigma * np.sqrt(dt) * z[t - 1]
  simulation_matrix[t] = simulation_matrix[t - 1] * np.exp(
      drift_term + diffusion_term
  )

plt.figure(figsize=(12, 6))

plt.plot(simulation_matrix, color="dodgerblue", alpha=0.08)

plt.plot(
    simulation_matrix.mean(axis=1),
    color="crimson",
    linewidth=2.5,
    label="Prezzo Medio Atteso",
)

plt.title(
    f"Simulazione di Monte Carlo - Indice {ticker} (365 Giorni)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Giorni di Simulazione", fontsize=11)
plt.ylabel("Prezzo Stimato", fontsize=11)
plt.legend(frameon=True, facecolor="white", shadow=True)
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()