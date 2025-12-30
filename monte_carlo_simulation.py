import yfinance as yf
import numpy as np

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

# Simulaiton parameters
num_simulations = 1000
num_days = 252  
initial_price = closing_prices[-1]
shocks = np.random.normal(0, 1, (num_days, num_simulations))
daily_time_step = 1
daily_returns_matrix = np.exp(drift * daily_time_step + volatility * shocks)
price_paths = initial_price * np.cumprod(daily_returns_matrix, axis=0)