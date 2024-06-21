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

# Simulate trades with dynamic lot size based on funds usage percentage
def simulate_trades(data, symbol, tolerance, initial_balance=10000, funds_usage_percentage=100):
    balance = initial_balance
    trades = []
    active_trade = None
    trade_type = None

    for index, row in data.iterrows():
        # Calculate the maximum possible lot size based on the funds usage percentage
        available_funds = balance * (funds_usage_percentage / 100)
        max_possible_lot_size = available_funds / row['close']
        
        if max_possible_lot_size < 1:
            lot_size = max_possible_lot_size  # If funds are insufficient, use the maximum possible lot size
        else:
            lot_size = int(max_possible_lot_size)  # Use integer lot size for trading

        if row['position'] == 1 and active_trade is None:
            # Open new buy trade
            buy_price = row['close']
            active_trade = {
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': buy_price,
                'buy_time': row['timestamp'],
                'lot_size': lot_size
            }
            trade_type = 'Buy'
        elif row['position'] == -1 and active_trade is None:
            # Open new sell short trade
            sell_short_price = row['close']
            active_trade = {
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': sell_short_price,
                'sell_short_time': row['timestamp'],
                'lot_size': lot_size
            }
            trade_type = 'Sell Short'
        elif row['position'] == -1 and active_trade is not None and trade_type == 'Buy':
            # Close buy trade and take sell short trade
            current_price = row['close']
            profit_loss = lot_size * (current_price - active_trade['buy_price'])
            balance += profit_loss
            trades.append({
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': active_trade['buy_price'],
                'sell_price': current_price,
                'buy_time': active_trade['buy_time'],
                'sell_time': row['timestamp'],
                'lot_size': lot_size,
                'profit_loss': profit_loss
            })
            active_trade = {
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': current_price,
                'sell_short_time': row['timestamp'],
                'lot_size': lot_size
            }
            trade_type = 'Sell Short'
            
        elif row['position'] == 1 and active_trade is not None and trade_type == 'Sell Short':
            # Close sell short trade and take buy trade
            current_price = row['close']
            profit_loss = lot_size * (active_trade['sell_short_price'] - current_price)
            balance += profit_loss
            trades.append({
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': active_trade['sell_short_price'],
                'buy_cover_price': current_price,
                'sell_short_time': active_trade['sell_short_time'],
                'buy_cover_time': row['timestamp'],
                'lot_size': lot_size,
                'profit_loss': profit_loss
            })
            active_trade = {
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': current_price,
                'buy_time': row['timestamp'],
                'lot_size': lot_size
            }
            trade_type = 'Buy'

    # Close any remaining active trade at the last timestamp
    if active_trade is not None:
        if trade_type == 'Buy':
            current_price = data['close'].iloc[-1]
            profit_loss = lot_size * (current_price - active_trade['buy_price'])
            trades.append({
                'symbol': symbol,
                'type': 'Buy',
                'buy_price': active_trade['buy_price'],
                'sell_price': current_price,
                'buy_time': active_trade['buy_time'],
                'sell_time': data['timestamp'].iloc[-1],
                'lot_size': lot_size,
                'profit_loss': profit_loss
            })
            balance += profit_loss
        elif trade_type == 'Sell Short':
            current_price = data['close'].iloc[-1]
            profit_loss = lot_size * (active_trade['sell_short_price'] - current_price)
            trades.append({
                'symbol': symbol,
                'type': 'Sell Short',
                'sell_short_price': active_trade['sell_short_price'],
                'buy_cover_price': current_price,
                'sell_short_time': active_trade['sell_short_time'],
                'buy_cover_time': data['timestamp'].iloc[-1],
                'lot_size': lot_size,
                'profit_loss': profit_loss
            })
            balance += profit_loss

    remaining_balance = balance
    return initial_balance, remaining_balance, trades

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

# Wide page config
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

# Add sliders for initial investment and funds usage percentage
initial_investment = st.slider("Initial Investment (USDT)", min_value=100, max_value=100000, value=10000, step=100)
funds_usage_percentage = st.slider("Funds Usage Percentage (%)", min_value=10, max_value=100, value=100, step=10)

# Fetch data and apply trading strategy
data = fetch_data(symbol, timeframe)
data = calculate_ema(data, periods=[20, 50, 100])
data, buy_signals, sell_signals = apply_trading_strategy(data, tolerance)

# Simulate trades with dynamic lot size based on funds usage percentage
initial_invested_balance, final_balance, trades = simulate_trades(data, symbol, tolerance, initial_balance=initial_investment, funds_usage_percentage=funds_usage_percentage)

win_rate = compute_winning_rate(trades)

# Display results
st.subheader("Trading Signals and Performance Metrics")
st.markdown(f"**Timeframe:** {timeframe}")
st.markdown(f"**Initial Investment:** {initial_investment} USDT")
st.markdown(f"**Funds Usage Percentage:** {funds_usage_percentage}%")
st.markdown(f"**Final Balance:** {final_balance:.2f} USDT")
st.markdown(f"**Winning Rate:** {win_rate * 100:.2f}%")


# Plot candlestick chart with EMAs
st.subheader("Candlestick Chart with EMAs and Trade Signals")
fig = go.Figure(data=[go.Candlestick(x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])])

fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_50'], mode='lines', name='EMA 50', line=dict(color='red', width=1)))
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_100'], mode='lines', name='EMA 100', line=dict(color='green', width=1)))

# Add annotations for buy and sell signals
for index, row in buy_signals.iterrows():
    fig.add_annotation(
        x=row['timestamp'],
        y=row['close'],
        text="Buy",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

for index, row in sell_signals.iterrows():
    fig.add_annotation(
        x=row['timestamp'],
        y=row['close'],
        text="Sell Short",
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
                width=1500,
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


