import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ticker = "^GSPC"
data = yf.download(ticker, start="2021-01-01", end="2026-01-01")
prices = (
    data["Close"].iloc[:, 0]
    if isinstance(data.columns, pd.MultiIndex)
    else data["Close"]
)

log_returns = np.log(prices / prices.shift(1)).dropna()

mu = log_returns.mean() * 252
sigma = log_returns.std() * np.sqrt(252)
S_0 = prices.iloc[-1]

num_days = 252
num_simulations = 2000
dt = 1 / 252

simulated_prices = np.zeros((num_days, num_simulations))
simulated_prices[0] = S_0

np.random.seed(42)
z = np.random.normal(0, 1, size=(num_days - 1, num_simulations))

for t in range(1, num_days):
  drift_term = (mu - 0.5 * sigma**2) * dt
  diffusion_term = sigma * np.sqrt(dt) * z[t - 1]
  simulated_prices[t] = simulated_prices[t - 1] * np.exp(
      drift_term + diffusion_term
  )

final_prices = simulated_prices[-1, :]
expected_final_price = final_prices.mean()
returns_dist = (final_prices - S_0) / S_0
var_95 = np.percentile(returns_dist, 5)

print(f"--- RISULTATI NUMERICI PREVISTI ---")
print(f"Prezzo di Entrata (S_0): {S_0:.2f}")
print(f"Prezzo Medio Previsto (252 Giorni di Borsa): {expected_final_price:.2f}")
print(f"Value at Risk (VaR 95%): {var_95*100:.2f}%")

plt.figure(figsize=(12, 6))

plt.plot(simulated_prices, color="dodgerblue", alpha=0.04)

mean_path = simulated_prices.mean(axis=1)
plt.plot(
    mean_path,
    color="darkorange",
    linewidth=2.5,
    label="Percorso Medio Atteso",
)

plt.axhline(
    y=S_0,
    color="royalblue",
    linestyle="--",
    linewidth=2,
    label=f"Prezzo di Entrata (Oggi): {S_0:.2f}",
)

plt.axhline(
    y=expected_final_price,
    color="crimson",
    linestyle="--",
    linewidth=2,
    label=f"Prezzo di Uscita Previsto (1 Anno): {expected_final_price:.2f}",
)

stats_box = (
    f"Sintesi Previsionale (GBM):\n"
    f"• Entrata (S_0): {S_0:.2f}\n"
    f"• Uscita Attesa: {expected_final_price:.2f}\n"
    f"• Variazione % Attesa: {((expected_final_price - S_0)/S_0)*100:+.2f}%\n"
    f"• Rischio VaR (95%): {var_95*100:.2f}%"
)
props = dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9, edgecolor="gray")
plt.gca().text(
    0.03,
    0.95,
    stats_box,
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=props,
)

plt.title(
    "Simulazione Monte Carlo (GBM) - Proiezioni e Analisi di Rischio (252"
    " Giorni)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Giorni di Borsa Aperta (Orizzonte 1 Anno)", fontsize=11)
plt.ylabel("Prezzo Stimato dell'Indice", fontsize=11)
plt.legend(loc="lower right", frameon=True, facecolor="white")
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
