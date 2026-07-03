# 📈 Stock Market EDA

A comprehensive Exploratory Data Analysis (EDA) of stock market data using Python. This project pulls real-time historical data and produces clean, publication-quality visualisations to uncover price trends, volatility patterns, and inter-stock correlations.

---

## 📊 What This Project Does

| Analysis | Description |
|----------|-------------|
| **Price & Moving Averages** | Close price with MA-20, MA-50, MA-200 overlays |
| **Returns Distribution** | Daily return histogram + 30-day rolling volatility |
| **Volume Analysis** | Price-volume chart with green/red volume bars |
| **Multi-Stock Comparison** | Normalised performance of 5 tech stocks |
| **Correlation Heatmap** | Return correlations to spot co-movement |
| **RSI Indicator** | 14-day Relative Strength Index with overbought/oversold zones |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- `yfinance` — real-time stock data
- `pandas` / `numpy` — data manipulation
- `matplotlib` / `seaborn` — visualisation

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Morreddydineshkumarreddy/stock-market-eda.git
cd stock-market-eda

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python eda_analysis.py
```

**Output:** 6 chart PNG files saved in the working directory, plus a printed summary.

---

## ⚙️ Configuration

Edit the top of `eda_analysis.py` to customise:

```python
TICKERS   = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Stocks to compare
START     = "2022-01-01"                                  # Start date
MAIN_TICK = "AAPL"                                        # Primary stock for deep-dive
```

---

## 📁 Project Structure

```
stock-market-eda/
├── eda_analysis.py      # Main script
├── requirements.txt     # Dependencies
└── README.md
```

---

## 💡 Key Insights (AAPL, 2022–2024)

- MA-200 acted as strong support through the 2023 recovery
- Daily returns are approximately normally distributed with slight negative skew
- MSFT and GOOGL show the highest correlation (>0.85), suggesting similar macro sensitivity
- RSI flagged 3 clear overbought zones that preceded short-term pullbacks

---

## 📚 What I Learned

- How to fetch and clean financial time-series data with `yfinance`
- Computing and interpreting technical indicators (MA, RSI)
- Building multi-panel matplotlib figures for financial dashboards
- Correlation analysis to understand portfolio diversification

---

*Built by [Dinesh Kumar Reddy M](https://github.com/Morreddydineshkumarreddy)*
