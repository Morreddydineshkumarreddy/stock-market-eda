"""
Stock Market EDA - Exploratory Data Analysis
Author: Dinesh Kumar Reddy M
Description: Comprehensive EDA on stock market data using yfinance.
             Includes trend analysis, volume, moving averages, and correlation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
TICKERS   = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START     = "2022-01-01"
END       = datetime.today().strftime("%Y-%m-%d")
MAIN_TICK = "AAPL"   # Primary stock for detailed analysis

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

# ── 1. Download Data ───────────────────────────────────────────────────────────
def download_data(tickers, start, end):
    print(f"📥 Downloading data for: {', '.join(tickers)}")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    print(f"✅ Data loaded — {len(data)} trading days\n")
    return data

# ── 2. Basic Summary ───────────────────────────────────────────────────────────
def print_summary(data, ticker):
    close = data["Close"][ticker].dropna()
    print(f"{'='*50}")
    print(f"  {ticker} — Summary Statistics")
    print(f"{'='*50}")
    print(f"  Period     : {close.index[0].date()} → {close.index[-1].date()}")
    print(f"  Start Price: ${close.iloc[0]:.2f}")
    print(f"  End Price  : ${close.iloc[-1]:.2f}")
    total_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  52-wk High : ${close.tail(252).max():.2f}")
    print(f"  52-wk Low  : ${close.tail(252).min():.2f}")
    print(f"  Volatility : {close.pct_change().std() * np.sqrt(252) * 100:.2f}% (annualised)\n")

# ── 3. Price + Moving Averages ─────────────────────────────────────────────────
def plot_price_ma(data, ticker):
    close = data["Close"][ticker].dropna()
    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(close.index, close,  color="#2196F3", lw=1.5, label="Close Price", alpha=0.9)
    ax.plot(ma20.index,  ma20,   color="#FF9800", lw=1.2, label="MA 20",  linestyle="--")
    ax.plot(ma50.index,  ma50,   color="#4CAF50", lw=1.2, label="MA 50",  linestyle="--")
    ax.plot(ma200.index, ma200,  color="#E91E63", lw=1.2, label="MA 200", linestyle="--")

    ax.set_title(f"{ticker} — Close Price & Moving Averages", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{ticker}_price_ma.png", dpi=150)
    plt.show()
    print(f"  📊 Saved: {ticker}_price_ma.png")

# ── 4. Daily Returns Distribution ─────────────────────────────────────────────
def plot_returns(data, ticker):
    returns = data["Close"][ticker].pct_change().dropna() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Histogram
    axes[0].hist(returns, bins=60, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].axvline(returns.mean(), color="#E91E63", lw=2, linestyle="--", label=f"Mean: {returns.mean():.3f}%")
    axes[0].set_title(f"{ticker} — Daily Returns Distribution", fontweight="bold")
    axes[0].set_xlabel("Daily Return (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Rolling 30-day volatility
    vol = returns.rolling(30).std()
    axes[1].fill_between(vol.index, vol, color="#FF9800", alpha=0.5)
    axes[1].plot(vol.index, vol, color="#FF9800", lw=1)
    axes[1].set_title(f"{ticker} — 30-Day Rolling Volatility", fontweight="bold")
    axes[1].set_ylabel("Std Dev of Returns (%)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.tight_layout()
    plt.savefig(f"{ticker}_returns.png", dpi=150)
    plt.show()
    print(f"  📊 Saved: {ticker}_returns.png")

# ── 5. Volume Analysis ─────────────────────────────────────────────────────────
def plot_volume(data, ticker):
    close  = data["Close"][ticker].dropna()
    volume = data["Volume"][ticker].dropna()
    colors = ["#4CAF50" if close.iloc[i] >= close.iloc[i-1] else "#F44336"
              for i in range(1, len(close))]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(close.index, close, color="#2196F3", lw=1.5)
    axes[0].set_title(f"{ticker} — Price & Volume", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Price (USD)")

    axes[1].bar(volume.index[1:], volume.iloc[1:], color=colors, width=1, alpha=0.8)
    axes[1].set_ylabel("Volume")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.tight_layout()
    plt.savefig(f"{ticker}_volume.png", dpi=150)
    plt.show()
    print(f"  📊 Saved: {ticker}_volume.png")

# ── 6. Multi-Stock Comparison ──────────────────────────────────────────────────
def plot_normalized_comparison(data, tickers):
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, ticker in enumerate(tickers):
        close = data["Close"][ticker].dropna()
        norm  = (close / close.iloc[0]) * 100   # Normalise to 100 at start
        ax.plot(norm.index, norm, label=ticker, color=COLORS[i], lw=1.5)

    ax.axhline(100, color="grey", lw=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Normalised Price Comparison (Base = 100)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalised Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("multi_stock_comparison.png", dpi=150)
    plt.show()
    print("  📊 Saved: multi_stock_comparison.png")

# ── 7. Correlation Heatmap ─────────────────────────────────────────────────────
def plot_correlation(data, tickers):
    returns = data["Close"][tickers].pct_change().dropna()
    corr    = returns.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Return Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("  📊 Saved: correlation_heatmap.png")

# ── 8. RSI Indicator ───────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def plot_rsi(data, ticker):
    close = data["Close"][ticker].dropna()
    rsi   = compute_rsi(close)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(close.index, close, color="#2196F3", lw=1.5)
    axes[0].set_title(f"{ticker} — Price & RSI (14)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Price (USD)")

    axes[1].plot(rsi.index, rsi, color="#9C27B0", lw=1.2)
    axes[1].axhline(70, color="#F44336", lw=1, linestyle="--", label="Overbought (70)")
    axes[1].axhline(30, color="#4CAF50", lw=1, linestyle="--", label="Oversold (30)")
    axes[1].fill_between(rsi.index, rsi, 70, where=(rsi >= 70), alpha=0.2, color="#F44336")
    axes[1].fill_between(rsi.index, rsi, 30, where=(rsi <= 30), alpha=0.2, color="#4CAF50")
    axes[1].set_ylabel("RSI")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.tight_layout()
    plt.savefig(f"{ticker}_rsi.png", dpi=150)
    plt.show()
    print(f"  📊 Saved: {ticker}_rsi.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Stock Market EDA — Starting Analysis\n")

    data = download_data(TICKERS, START, END)

    print_summary(data, MAIN_TICK)

    print("📈 Plotting Price & Moving Averages...")
    plot_price_ma(data, MAIN_TICK)

    print("📉 Plotting Daily Returns...")
    plot_returns(data, MAIN_TICK)

    print("📊 Plotting Volume Analysis...")
    plot_volume(data, MAIN_TICK)

    print("🔗 Plotting Multi-Stock Comparison...")
    plot_normalized_comparison(data, TICKERS)

    print("🌡️  Plotting Correlation Heatmap...")
    plot_correlation(data, TICKERS)

    print("📐 Plotting RSI Indicator...")
    plot_rsi(data, MAIN_TICK)

    print("\n✅ EDA Complete! All charts saved as PNG files.")
    print("   Tip: Change MAIN_TICK or TICKERS at the top to analyse any stock.\n")
