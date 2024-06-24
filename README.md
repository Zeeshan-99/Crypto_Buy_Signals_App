Auto Crypto Trading Bot
This project is an advanced auto crypto trading bot built using Python and Streamlit, designed to trade on the Binance exchange. It employs various technical indicators and trading strategies to automate cryptocurrency trading efficiently.

Table of Contents
(1) Features
(2) Installation
(3) Usage
(3) Trading Strategies
(4) Simulation
(5) Visualization
Contributing
License
Features
EMA Convergence: Detects precise moments when EMAs converge, signaling potential trade opportunities.
RSI Analysis: Enhances signal accuracy by incorporating Relative Strength Index values.
Flexible Strategy Selection: Choose from multiple algorithms to match your trading style.
Trade Simulation: Simulates trades with stop-loss and take-profit mechanisms to evaluate strategy performance.
Detailed Visualization: Interactive candlestick charts with EMA and RSI overlays, complete with trade signal annotations.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/Zeeshan-99/auto-crypto-trading-bot.git
cd auto-crypto-trading-bot
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Configure the bot via the sidebar:

Select the symbol (e.g., BTC/USDT, ETH/USDT).
Choose the timeframe (e.g., 1m, 5m, 15m, 30m, 1h, 4h, 1d).
Set your initial investment in USDT.
Adjust the EMA convergence tolerance percentage.
Set the percentage of funds to be used per trade.
Select the desired trading algorithm.
View the results:

The main panel displays the candlestick chart with EMA and RSI overlays.
Trade signals (Buy/Sell) are annotated on the chart.
Trade history and performance metrics are shown in the sidebar.
Trading Strategies
Algorithm 1
This strategy identifies buy signals when EMAs converge and the RSI is below 30, and sell signals when the RSI is above 70.

Algorithm 2
This strategy places trades based on the trend of EMAs and a convergence tolerance, taking into account recent price movements.

Algorithm 3
This strategy uses a lookback period to confirm trends in price movements and EMA alignment before signaling trades.

Simulation
The bot simulates trades with the following rules:

Stop Loss: 2% below the buy price for long trades, 2% above the sell price for short trades.
Take Profit: 4% above the buy price for long trades, 4% below the sell price for short trades.
Visualization
The app visualizes data using Plotly, offering interactive candlestick charts with:

EMA overlays (20, 50, 100 periods)
RSI line
Buy/Sell signal annotations
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for details.
