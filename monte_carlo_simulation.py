import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo simulation for stock price prediction using Geometric Brownian Motion
# S_t = S_0 * exp((μ - (1/2)σ²)t + σ√t * Z_t)
# S_t represents the stock price at the end of the simulation
# S_0 is the initial stock price
# μ is the expected return
# σ is the volatility
# t is the time step
# Z_t is a standard normal random variable

# The drift (μ - (1/2)σ²)t is the predictable component of the return
# The shock σ√t * Z_t is the random component taken from a bell curve and is scaled by volatility

ticker = "AAPL"
stock = yf.Ticker(ticker)
history = stock.history(period="2y")
closing_prices = history['Close']
daily_returns = closing_prices.pct_change().dropna()
mean_return = daily_returns.mean()
volatility = daily_returns.std()
drift = mean_return - (0.5 * volatility ** 2)

# Simulation
num_simulations = 1000
num_days = 252  
initial_price = closing_prices.iloc[-1]
shocks = np.random.normal(0, 1, (num_days, num_simulations))
daily_time_step = 1
daily_returns_matrix = np.exp(drift * daily_time_step + volatility * np.sqrt(daily_time_step) * shocks)
price_paths = initial_price * np.cumprod(daily_returns_matrix, axis=0)
day_zero = np.full((1, num_simulations), initial_price)
full_price_paths = np.vstack([day_zero, price_paths])

# Calculate statistics
final_prices = full_price_paths[-1]
mean_final = np.mean(final_prices)
median_final = np.median(final_prices)
std_final = np.std(final_prices)
percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])

print(f"Initial price: ${initial_price:.2f}")
print(f"Mean final price: ${mean_final:.2f}")
print(f"Median final price: ${median_final:.2f}")
print(f"Standard deviation of final prices: ${std_final:.2f}")
print("Percentiles:")
for p, val in zip([5, 25, 50, 75, 95], percentiles):
    print(f"  {p}th: ${val:.2f}")

# Plot results
colors = plt.cm.jet(np.linspace(0, 1, num_simulations))

plt.figure(figsize=(12, 6))
for i in range(num_simulations):
    plt.plot(full_price_paths[:, i], color=colors[i], alpha=0.3)
plt.xlabel('Days')
plt.ylabel('Price')
plt.title(f'Monte Carlo Simulation for {ticker}')
plt.grid(True)
plt.show()