from arch import arch_model
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

log_returns = np.log(prices / prices.shift(1)).dropna() * 100

garch_model = arch_model(log_returns, vol="Garch", p=1, q=1, mean="Constant")
garch_results = garch_model.fit(disp="off")

omega = garch_results.params["omega"]
alpha = garch_results.params["alpha[1]"]
beta = garch_results.params["beta[1]"]
mu = garch_results.params["mu"]
last_vol = garch_results.conditional_volatility.iloc[-1]
S_0 = prices.iloc[-1]

num_days = 365
num_simulations = 2000
dt = 1 / 255

simulated_prices = np.zeros((num_days, num_simulations))
simulated_vol = np.zeros((num_days, num_simulations))

simulated_prices[0] = S_0
simulated_vol[0] = last_vol

np.random.seed(42)

for t in range(1, num_days):
  z = np.random.normal(0, 1, num_simulations)

  if t == 1:
    prev_var = last_vol**2
    prev_resid_sq = (log_returns.iloc[-1] - mu) ** 2
  else:
    prev_var = simulated_vol[t - 1] ** 2
    prev_resid_sq = prev_var * (z_prev**2)

  next_var = omega + alpha * prev_resid_sq + beta * prev_var
  next_vol = np.sqrt(next_var)
  simulated_vol[t] = next_vol

  daily_return = (mu / 100) * dt + (next_vol / 100) * np.sqrt(dt) * z
  simulated_prices[t] = simulated_prices[t - 1] * np.exp(daily_return)

  z_prev = z

final_prices = simulated_prices[-1, :]
expected_final_price = final_prices.mean()
returns_dist = (final_prices - S_0) / S_0
var_95 = np.percentile(returns_dist, 5)

print(f"--- RISULTATI NUMERICI PREVISTI ---")
print(f"Prezzo di Entrata (S_0): {S_0:.2f}")
print(f"Prezzo Medio Previsto: {expected_final_price:.2f}")
print(f"Value at Risk (VaR 95%): {var_95*100:.2f}%")

plt.figure(figsize=(12, 6))

plt.plot(simulated_prices, color="mediumpurple", alpha=0.04)

mean_path = simulated_prices.mean(axis=1)
plt.plot(
    mean_path,
    color="darkorange",
    linewidth=2.5,
    label="Percorso Medio Atteso",
)

plt.axhline(
    y=S_0,
    color="dodgerblue",
    linestyle="--",
    linewidth=2,
    label=f"Prezzo di Entrata (Oggi): {S_0:.2f}",
)

plt.axhline(
    y=expected_final_price,
    color="crimson",
    linestyle="--",
    linewidth=2,
    label=f"Prezzo di Uscita Previsto: {expected_final_price:.2f}",
)

stats_box = (
    f"Sintesi Previsionale (GARCH):\n"
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
    "Simulazione Monte Carlo GARCH - Proiezioni e Analisi di Rischio",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Passi di Simulazione (365 Giorni)", fontsize=11)
plt.ylabel("Prezzo Stimato dell'Indice", fontsize=11)
plt.legend(loc="lower right", frameon=True, facecolor="white")
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()