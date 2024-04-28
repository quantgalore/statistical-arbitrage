# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

def check_cointegration(x, y):
    # Step 1: Regress one on another
    x = sm.add_constant(x)  # Adding a constant for the OLS regression
    result = sm.OLS(y, x).fit()
    x = x[:, 1]  # Remove the constant after fitting
    residual = result.resid

    # Step 2: Check for stationarity of residuals
    adf_result = adfuller(residual)
    return adf_result[1] 
    
polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

trading_dates = calendar.schedule(start_date = "2015-01-01", end_date = (datetime.today())).index.strftime("%Y-%m-%d").values

# =============================================================================
# Enter your desired tickers
# =============================================================================

portfolio_1_tickers = np.array(["FDX"])
portfolio_2_tickers = np.array(["UPS"])

portfolio_1_data_list = []
portfolio_2_data_list = []

for portfolio_1_stock in portfolio_1_tickers:
    
    stock_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{portfolio_1_stock}/range/1/day/{trading_dates[0]}/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
    stock_data.index = pd.to_datetime(stock_data.index, unit="ms", utc=True).tz_convert("America/New_York")
    stock_data["pct_change"] = stock_data["c"].pct_change()
    stock_data["ticker"] = portfolio_1_stock
    stock_data = stock_data[["c", "pct_change", "ticker"]].dropna()
    
    portfolio_1_data_list.append(stock_data)
    
for portfolio_2_stock in portfolio_2_tickers:
    
    stock_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{portfolio_2_stock}/range/1/day/{trading_dates[0]}/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
    stock_data.index = pd.to_datetime(stock_data.index, unit="ms", utc=True).tz_convert("America/New_York")
    stock_data["pct_change"] = stock_data["c"].pct_change()
    stock_data["ticker"] = portfolio_1_stock
    stock_data = stock_data[["c", "pct_change", "ticker"]].dropna()
    
    portfolio_2_data_list.append(stock_data)

portfolio_1_data_original = pd.concat(portfolio_1_data_list)
portfolio_1_data = portfolio_1_data_original.groupby(level=0).mean(numeric_only = True)
portfolio_1_data["portfolio_performance"] = portfolio_1_data["pct_change"].cumsum()*100

portfolio_2_data_original = pd.concat(portfolio_2_data_list)
portfolio_2_data = portfolio_2_data_original.groupby(level=0).mean(numeric_only = True)
portfolio_2_data["portfolio_performance"] = portfolio_2_data["pct_change"].cumsum()*100

# =============================================================================
# Stationary Test
# =============================================================================

combined_portfolio_data = pd.concat([portfolio_1_data.add_prefix("1_"), portfolio_2_data.add_prefix("2_")], axis = 1).dropna()
combined_portfolio_data["spread"] = abs(combined_portfolio_data["1_portfolio_performance"] - combined_portfolio_data["2_portfolio_performance"])

# Test p-value to ensure stationarity | the closer to < 0.05, the better
p_value = check_cointegration(combined_portfolio_data['1_portfolio_performance'].values, combined_portfolio_data['2_portfolio_performance'].values)
print(f"P-Value: {p_value}")

# =============================================================================
# Backtesting
# =============================================================================

combined_portfolio_data['spread_ma'] = combined_portfolio_data['spread'].rolling(window=200).mean()
combined_portfolio_data['std_dev'] = combined_portfolio_data['spread'].rolling(window=200).std()

# Calculate the upper and lower bollinger bands
combined_portfolio_data['upper_band'] = combined_portfolio_data['spread_ma'] + (combined_portfolio_data['std_dev'] * 1)
combined_portfolio_data['lower_band'] = combined_portfolio_data['spread_ma'] - (combined_portfolio_data['std_dev'] * 1)

# Trading signal of when spread widens n-deviations above average
combined_portfolio_data["trading_signal"] = combined_portfolio_data.apply(lambda row: 1 if (row['spread'] > row['upper_band']) else 0, axis=1)

combined_portfolio_data['entry_signal'] = (combined_portfolio_data['trading_signal'] == 1) & (combined_portfolio_data['trading_signal'].shift(1) != 1)
combined_portfolio_data['exit_signal'] = (combined_portfolio_data['trading_signal'] == 0) & (combined_portfolio_data['trading_signal'].shift(1) != 0)

combined_portfolio_data["underperformer_price"] = combined_portfolio_data.apply(lambda row: row['1_c'] if (row['2_portfolio_performance'] > row['1_portfolio_performance']) else row['2_c'], axis=1)
combined_portfolio_data["overperformer_price"] = combined_portfolio_data.apply(lambda row: row['1_c'] if (row['2_portfolio_performance'] < row['1_portfolio_performance']) else row['2_c'], axis=1)

trades = []
day = 0

while day < len(combined_portfolio_data):
    day_data = combined_portfolio_data.iloc[day]
    if day_data['entry_signal']:
        long_entry = day_data['underperformer_price']
        short_entry = day_data['overperformer_price']

        long_side = 'underperformer_price'
        short_side = 'overperformer_price'

        for day_after in range(day + 1, len(combined_portfolio_data)):
            day_after_data = combined_portfolio_data.iloc[day_after]
            if day_after_data['exit_signal']:
                long_exit = day_after_data[long_side]
                short_exit = day_after_data[short_side]

                long_pnl = long_exit - long_entry
                short_pnl = short_entry - short_exit
                gross_pnl = long_pnl + short_pnl

                trade_data = {
                    "open_date": combined_portfolio_data.index[day],
                    "close_date": combined_portfolio_data.index[day_after],
                    "long_pnl": long_pnl,
                    "short_pnl": short_pnl,
                    "gross_pnl": gross_pnl
                }
                trades.append(trade_data)

                day = day_after  # Move the day index to the day after the trade close
                break

    day += 1  # Increment the day if no trade was closed

trades_df = pd.DataFrame(trades)

win_rate = round((len(trades_df[trades_df["gross_pnl"] > 0]) / len(trades_df)) * 100, 2)
print(f"Win Rate: {win_rate}%")

# =============================================================================
# Strategy PnL
# =============================================================================

plt.figure(dpi=200)
plt.xticks(rotation=45)
plt.title(f"FDX-UPS Cointegration Strategy")
plt.plot(trades_df["open_date"], 100 + trades_df["gross_pnl"].cumsum())
plt.plot(trades_df["open_date"], 100 + trades_df["long_pnl"].cumsum())
plt.plot(trades_df["open_date"], 100 + trades_df["short_pnl"].cumsum())
plt.legend(["gross_pnl", "long_only_pnl", "short_only_pnl"])
plt.ylabel("Capital")
plt.xlabel("Date")
plt.show()

# =============================================================================
# Underlying Spread Visualizations
# =============================================================================


plt.figure(dpi=200)
plt.xticks(rotation=45)
plt.title("FDX-UPS Return Spread")
plt.ylabel("% Difference in Returns")
plt.xlabel("Date")
plt.plot(combined_portfolio_data.index, combined_portfolio_data["spread"])
plt.show()

plt.figure(dpi=200)
plt.xticks(rotation=45)
plt.title(f"FDX-UPS Spread")
plt.plot(combined_portfolio_data.index, combined_portfolio_data["spread"])
plt.plot(combined_portfolio_data.index, combined_portfolio_data["upper_band"])
plt.plot(combined_portfolio_data.index, combined_portfolio_data["lower_band"])
plt.legend(["spread", "upper_bound",  "lower_bound"])
plt.show()

plt.figure(dpi=600)
plt.xticks(rotation=45)
plt.title(f"FDX-UPS Return Performance")
plt.plot(combined_portfolio_data.index, combined_portfolio_data["1_portfolio_performance"])
plt.plot(combined_portfolio_data.index, combined_portfolio_data["2_portfolio_performance"])
plt.legend(["FDX", "UPS"])
plt.show()