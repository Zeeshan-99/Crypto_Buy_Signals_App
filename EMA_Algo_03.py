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

# Function to calculate technical indicators
def calculate_indicators(data):
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['EMA_100'] = data['close'].ewm(span=100, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['close'])
    return data

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Improved trading strategy function
def apply_trading_strategy(data, tolerance):
    data['Signal'] = ''
    data['EMAs_Converge'] = (
        (abs(data['EMA_20'] / data['EMA_50'] - 1) <= tolerance / 100) &
        (abs(data['EMA_20'] / data['EMA_100'] - 1) <= tolerance / 100) &
        (abs(data['EMA_50'] / data['EMA_100'] - 1) <= tolerance / 100)
    )
    
    for i in range(2, len(data) - 1):  # Check from the third row to the second last row
        if data['EMAs_Converge'].iloc[i]:
            # Additional criteria using RSI
            if data['RSI'].iloc[i] < 30:
                # Buy signal criteria
                data.at[i + 1, 'Signal'] = 'Buy'
            elif data['RSI'].iloc[i] > 70:
                # Sell signal criteria
                data.at[i + 1, 'Signal'] = 'Sell'

    signals = data[data['Signal'].isin(['Buy', 'Sell'])].reset_index(drop=True)
    return signals

# Simulate trades with dynamic lot size based on funds usage percentage
def simulate_trades(data, signals, symbol, initial_balance=10000, funds_usage_percentage=100):
    balance = initial_balance
    trades = []
    active_trade = None
    trade_type = None

    for index, row in signals.iterrows():
        close_price = data.loc[data['timestamp'] == row['timestamp'], 'close'].values[0]
        available_funds = balance * (funds_usage_percentage / 100)
        max_possible_lot_size = available_funds / close_price
        
        if max_possible_lot_size < 1:
            lot_size = max_possible_lot_size  # If funds are insufficient, use the maximum possible lot size
        else:
            lot_size = int(max_possible_lot_size)  # Use integer lot size for trading

        if row['Signal'] == 'Buy' and (active_trade is None or trade_type == 'Sell'):
            if active_trade is not None and trade_type == 'Sell':
                profit_loss = active_trade['lot_size'] * (active_trade['sell_short_price'] - close_price)
                balance += profit_loss
                trades.append({
                    'symbol': symbol,
                    'type': 'Sell Short',
                    'sell_short_price': active_trade['sell_short_price'],
                    'buy_cover_price': close_price,
                    'sell_short_time': active_trade['sell_short_time'],
                    'buy_cover_time': row['timestamp'],
                    'lot_size': active_trade['lot_size'],
                    'profit_loss': profit_loss
                })
            active_trade = {
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': close_price,
                'buy_time': row['timestamp'],
                'lot_size': lot_size
            }
            trade_type = 'Buy'
        elif row['Signal'] == 'Sell' and (active_trade is None or trade_type == 'Buy'):
            if active_trade is not None and trade_type == 'Buy':
                profit_loss = active_trade['lot_size'] * (close_price - active_trade['buy_price'])
                balance += profit_loss
                trades.append({
                    'symbol': symbol,
                    'type': 'Buy',
                    'buy_price': active_trade['buy_price'],
                    'sell_price': close_price,
                    'buy_time': active_trade['buy_time'],
                    'sell_time': row['timestamp'],
                    'lot_size': active_trade['lot_size'],
                    'profit_loss': profit_loss
                })
            active_trade = {
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': close_price,
                'sell_short_time': row['timestamp'],
                'lot_size': lot_size
            }
            trade_type = 'Sell'

    # Close any remaining active trade at the last timestamp
    if active_trade is not None:
        if trade_type == 'Buy':
            current_price = data['close'].iloc[-1]
            profit_loss = active_trade['lot_size'] * (current_price - active_trade['buy_price'])
            trades.append({
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': active_trade['buy_price'],
                'sell_price': current_price,
                'buy_time': active_trade['buy_time'],
                'sell_time': data['timestamp'].iloc[-1],
                'lot_size': active_trade['lot_size'],
                'profit_loss': profit_loss
            })
        elif trade_type == 'Sell':
            current_price = data['close'].iloc[-1]
            profit_loss = active_trade['lot_size'] * (active_trade['sell_short_price'] - current_price)
            trades.append({
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': active_trade['sell_short_price'],
                'buy_cover_price': current_price,
                'sell_short_time': active_trade['sell_short_time'],
                'buy_cover_time': data['timestamp'].iloc[-1],
                'lot_size': active_trade['lot_size'],
                'profit_loss': profit_loss
            })

    final_balance = balance
    return initial_balance, final_balance, trades

# Function to compute winning rate
def compute_winning_rate(trades):
    if not trades:
        return 0
    winning_trades = [trade for trade in trades if trade['profit_loss'] > 0]
    win_rate = len(winning_trades) / len(trades)
    return win_rate

### for full screen
st.set_page_config(layout="wide")

# Streamlit app
st.title("Enhanced Crypto Trading Bot")

# User inputs
symbol = st.sidebar.text_input("Symbol", value="BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "4h", "1d"], index=3)
tolerance = st.sidebar.slider("EMA Convergence Tolerance (%)", min_value=0.01, max_value=0.3, value=0.02, step=0.01)
initial_investment = st.sidebar.number_input("Initial Investment (USDT)", min_value=100.0, value=10000.0)
funds_usage_percentage = st.sidebar.slider("Funds Usage Percentage", min_value=10, max_value=100, value=100, step=5)

# Fetch and process data
data = fetch_data(symbol, timeframe)
data = calculate_indicators(data)

# Apply trading strategy
signals = apply_trading_strategy(data, tolerance)

# Simulate trades
initial_invested_balance, final_balance, trades = simulate_trades(data, signals, symbol, initial_balance=initial_investment, funds_usage_percentage=funds_usage_percentage)

# Compute winning rate
win_rate = compute_winning_rate(trades)

# Display results
st.sidebar.subheader("Trading Signals and Performance Metrics")
st.sidebar.markdown(f"**Timeframe:** {timeframe}")
st.sidebar.markdown(f"**Initial Investment:** {initial_investment} USDT")
st.sidebar.markdown(f"**Funds Usage Percentage:** {funds_usage_percentage}%")
st.sidebar.markdown(f"**Final Balance:** {final_balance:.2f} USDT")
st.sidebar.markdown(f"**Winning Rate:** {win_rate * 100:.2f}%")

# Plot candlestick chart with EMAs
st.subheader("Candlestick Chart with EMAs and Trade Signals")
fig = go.Figure(data=[go.Candlestick(x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])])

fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_50'], mode='lines', name='EMA 50', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_100'], mode='lines', name='EMA 100', line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['RSI'], mode='lines', name='RSI', line=dict(color='orange', width=2)))

# Add annotations for buy and sell signals
for index, row in signals.iterrows():
    if row['Signal'] == 'Buy':
        fig.add_annotation(
            x=row['timestamp'],
            y=row['close'],
            text="Buy",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    elif row['Signal'] == 'Sell':
        fig.add_annotation(
            x=row['timestamp'],
            y=row['close'],
            text="Sell",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=40
        )

fig.update_layout(title=f'{symbol} Candlestick Chart with EMAs and Trade Signals', 
                xaxis_title='Date', 
                yaxis_title='Price (USDT)', 
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
                width=1300,
                dragmode='zoom',  # Enables zooming
                hovermode='x unified'  # Unified hover mode for better interactivity
)
fig.update_xaxes(rangeselector=dict(
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
                type='date')

# Enable y-axis zooming
fig.update_yaxes(
    range=[data['close'].min() * 0.95, data['close'].max() * 1.05],  # Adjust the initial visible range as desired
    fixedrange=False  # Allow zooming
)
st.plotly_chart(fig)

# Display trade history
# List of all possible columns
all_columns = [
    'symbol', 'type', 'buy_price', 'sell_price', 'sell_short_price', 'buy_cover_price',
    'buy_time', 'sell_time', 'sell_short_time', 'buy_cover_time', 'lot_size', 'profit_loss'
]

# Display trade history
if trades:
    st.subheader("Trade History")
    trades_df = pd.DataFrame(trades)
    
    # Filter columns to only those present in the DataFrame
    existing_columns = [col for col in all_columns if col in trades_df.columns]
    st.table(trades_df[existing_columns])
else:
    st.subheader("No trades executed.")
    
    # Create an empty DataFrame with the required columns
    empty_df = pd.DataFrame(columns=all_columns)
    st.table(empty_df)
