import streamlit as st
import pandas as pd
import ccxt
import plotly.graph_objects as go

# Function to fetch data from Binance
@st.cache_data
def fetch_data(symbol, timeframe, limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    return data

# Function to calculate EMAs
def calculate_ema(data, periods):
    for period in periods:
        data[f'EMA_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
    return data

# Trading strategy function
def apply_trading_strategy(data, tolerance):
    data['signal'] = 0
    
    # Find points where EMAs converge within the tolerance
    ema_converge = (
        (data['EMA_20'] / data['EMA_50'] >= (1 - tolerance / 100)) &
        (data['EMA_20'] / data['EMA_50'] <= (1 + tolerance / 100)) &
        (data['EMA_50'] / data['EMA_100'] >= (1 - tolerance / 100)) &
        (data['EMA_50'] / data['EMA_100'] <= (1 + tolerance / 100))
    )
    
    # Identify buy and sell short signals
    data.loc[ema_converge & (data['close'] > data['EMA_20'].shift(1)), 'signal'] = 1  # Buy signal
    data.loc[ema_converge & (data['close'] < data['EMA_20'].shift(1)), 'signal'] = -1  # Sell short signal

    data['position'] = data['signal'].diff()

    buy_signals = data[data['position'] == 1]
    sell_signals = data[data['position'] == -1]

    return data, buy_signals, sell_signals

# Simulate trades with updated conditions
def simulate_trades(data, symbol, tolerance, initial_balance=10000, lot_size=1):
    balance = initial_balance
    initial_invested_balance = initial_balance
    trades = []
    active_trade = None
    trade_type = None

    for index, row in data.iterrows():
        if row['position'] == 1 and active_trade is None:
            # Open new buy trade
            buy_price = row['close']
            active_trade = {
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': buy_price,
                'buy_time': row['timestamp']
            }
            trade_type = 'Buy'
        elif row['position'] == -1 and active_trade is None:
            # Open new sell short trade
            sell_short_price = row['close']
            active_trade = {
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': sell_short_price,
                'sell_short_time': row['timestamp']
            }
            trade_type = 'Sell Short'
        elif row['position'] == -1 and active_trade is not None and trade_type == 'Buy':
            # Close buy trade and take sell short trade
            current_price = row['close']
            balance += lot_size * (current_price - active_trade['buy_price'])
            trades.append({
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': active_trade['buy_price'],
                'sell_price': current_price,
                'buy_time': active_trade['buy_time'],
                'sell_time': row['timestamp'],
                'profit_loss': lot_size * (current_price - active_trade['buy_price'])
            })
            active_trade = None
        elif row['position'] == 1 and active_trade is not None and trade_type == 'Sell Short':
            # Close sell short trade and take buy trade
            current_price = row['close']
            balance += lot_size * (active_trade['sell_short_price'] - current_price)
            trades.append({
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': active_trade['sell_short_price'],
                'buy_cover_price': current_price,
                'sell_short_time': active_trade['sell_short_time'],
                'buy_cover_time': row['timestamp'],
                'profit_loss': lot_size * (active_trade['sell_short_price'] - current_price)
            })
            active_trade = None

    # Close any remaining active trade at the last timestamp
    if active_trade is not None:
        if trade_type == 'Buy':
            current_price = data['close'].iloc[-1]
            trades.append({
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': active_trade['buy_price'],
                'sell_price': current_price,
                'buy_time': active_trade['buy_time'],
                'sell_time': data['timestamp'].iloc[-1],
                'profit_loss': lot_size * (current_price - active_trade['buy_price'])
            })
            balance += lot_size * (current_price - active_trade['buy_price'])
        elif trade_type == 'Sell Short':
            current_price = data['close'].iloc[-1]
            trades.append({
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': active_trade['sell_short_price'],
                'buy_cover_price': current_price,
                'sell_short_time': active_trade['sell_short_time'],
                'buy_cover_time': data['timestamp'].iloc[-1],
                'profit_loss': lot_size * (active_trade['sell_short_price'] - current_price)
            })
            balance += lot_size * (active_trade['sell_short_price'] - current_price)

    remaining_balance = balance
    return initial_invested_balance, remaining_balance, trades

# Function to compute winning rate
def compute_winning_rate(trades):
    if len(trades) == 0:
        return 0
    wins = [trade for trade in trades if trade['profit_loss'] > 0]
    win_rate = len(wins) / len(trades)
    return win_rate

# List of popular cryptocurrencies
popular_coins = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT',
    'DOGE/USDT', 'DOT/USDT', 'UNI/USDT', 'BCH/USDT', 'LTC/USDT'
]
### Wide page config
st.set_page_config(layout="wide")

# Main Streamlit app
st.title("Crypto Trading App with EMA Strategy")

# Automatically refresh the page every 60 seconds using JavaScript
st.markdown("""
    <script>
    function refresh() {
        setTimeout(function() {
            location.reload();
        }, 60000);
    }
    refresh();
    </script>
    """, unsafe_allow_html=True)

symbol = st.selectbox("Select a cryptocurrency:", popular_coins)
timeframe = st.selectbox("Select timeframe:", ['1m', '5m', '15m', '30m', '1h', '4h', '1d'], index=0)
tolerance = st.slider("Select tolerance level (%)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)

data = fetch_data(symbol, timeframe)
data = calculate_ema(data, periods=[20, 50, 100])
data, buy_signals, sell_signals = apply_trading_strategy(data, tolerance)
initial_invested_balance, final_balance, trades = simulate_trades(data, symbol, tolerance)

win_rate = compute_winning_rate(trades)

st.write(f"Initial Invested Balance: {initial_invested_balance:.2f} USDT")
st.write(f"Final Balance: {final_balance:.2f} USDT")
st.write(f"Winning rate: {win_rate:.2%}")

fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'], name='market data'))

# Add EMAs
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_20'], line=dict(color='blue', width=1), name='EMA 20'))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_50'], line=dict(color='red', width=1), name='EMA 50'))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_100'], line=dict(color='green', width=1), name='EMA 100'))

# Buy signals
fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals['close'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'))

# Sell short signals
fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals['close'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Short Signal'))

fig.update_layout(title=f'{symbol} Price Data with EMA Trading Strategy ({timeframe})', 
                xaxis_title='Time', 
                yaxis_title='Price',
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type='date'
                ),
                yaxis=dict(
                    title='Price',
                    side='left',
                    showgrid=True,
                    zeroline=False,
                    tickformat="$.2f"
                ),
                template='plotly_dark',
                height=800,
                dragmode='zoom',  # Enables zooming
                hovermode='x unified'  # Unified hover mode for better interactivity
                )

# Enable date range buttons and rangeslider
fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    ),
    rangeslider=dict(visible=True),
    type='date'
)

# # Enable y-axis zooming
fig.update_yaxes(
    range=[data['close'].min() * 0.95, data['close'].max() * 1.05],  # Adjust the initial visible range as desired
    fixedrange=False  # Allow zooming
)
st.plotly_chart(fig, use_container_width=True)
# Display all trades in a table
if trades:
    trades_df = pd.DataFrame(trades)
    trades_df.index += 1
    trades_df.index.name = "S.N"
    trades_df['Profit/Loss'] = trades_df['profit_loss'].map(lambda x: f"{x:.2f} USDT")

    buy_trades = trades_df[trades_df['type'] == 'Buy']
    sell_short_trades = trades_df[trades_df['type'] == 'Sell Short']

    if not buy_trades.empty:
        buy_trades['Entry Time'] = buy_trades['buy_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        buy_trades['Exit Time'] = buy_trades['sell_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        buy_trades = buy_trades[['symbol', 'type', 'buy_price', 'sell_price', 'Entry Time', 'Exit Time', 'Profit/Loss']]
        buy_trades.columns = ['Symbol', 'Type', 'Entry Price', 'Exit Price', 'Entry Time', 'Exit Time', 'Profit/Loss']

    if not sell_short_trades.empty:
        sell_short_trades['Entry Time'] = sell_short_trades['sell_short_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        sell_short_trades['Exit Time'] = sell_short_trades['buy_cover_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        sell_short_trades = sell_short_trades[['symbol', 'type', 'sell_short_price', 'buy_cover_price', 'Entry Time', 'Exit Time', 'Profit/Loss']]
        sell_short_trades.columns = ['Symbol', 'Type', 'Entry Price', 'Exit Price', 'Entry Time', 'Exit Time', 'Profit/Loss']

    if not buy_trades.empty or not sell_short_trades.empty:
        st.write("All Trade History")
        if not buy_trades.empty:
            st.dataframe(buy_trades)
        if not sell_short_trades.empty:
            st.dataframe(sell_short_trades)
else:
    st.write("No trades executed.")
