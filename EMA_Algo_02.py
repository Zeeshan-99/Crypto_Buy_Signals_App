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
def apply_trading_strategy_01(data, tolerance):
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

def apply_trading_strategy_02(data, tolerance):
    data['Signal'] = ''
    data['EMAs_Converge'] = ((abs(data['EMA_20'] / data['EMA_50'] - 1) <= tolerance) &
                            (abs(data['EMA_20'] / data['EMA_100'] - 1) <= tolerance) &
                            (abs(data['EMA_50'] / data['EMA_100'] - 1) <= tolerance))
    
    in_signal = False
    last_signal = None

    for i in range(3, len(data)-2):  # Check at least 3 candles for trends
        if data['EMAs_Converge'].iloc[i]:
            # Buy signal criteria
            if (data['EMA_20'].iloc[i-1] > data['EMA_20'].iloc[i-2]) and \
                (data['EMA_50'].iloc[i-1] > data['EMA_50'].iloc[i-2]) and \
                (data['EMA_100'].iloc[i-1] > data['EMA_100'].iloc[i-2]):
                data.at[i + 1, 'Signal'] = 'Buy'
                in_signal = True
                last_signal = 'Buy'
            # Sell signal criteria
            elif (data['EMA_20'].iloc[i-1] < data['EMA_20'].iloc[i-2]) and \
                (data['EMA_50'].iloc[i-1] < data['EMA_50'].iloc[i-2]) and \
                (data['EMA_100'].iloc[i-1] < data['EMA_100'].iloc[i-2]):
                data.at[i + 1, 'Signal'] = 'Sell'
                in_signal = True
                last_signal = 'Sell'
            else:
                if in_signal:
                    data.at[i + 1, 'Signal'] = last_signal
        else:
            in_signal = False

    signals = data[data['Signal'].isin(['Buy', 'Sell'])].reset_index(drop=True)
    return signals

def apply_trading_strategy_03(data, ema_periods=[20, 50, 100], tolerance=0.01, lookback=3):
    ema_cols = [f'EMA_{period}' for period in ema_periods]
    
    for col in ema_cols:
        if col not in data.columns:
            raise ValueError(f"Data does not contain required EMA column: {col}")
    
    data['Signal'] = ''
    data['EMAs_Converge'] = data.apply(
        lambda row: all(
            abs(row[ema_cols[i]] / row[ema_cols[j]] - 1) <= tolerance 
            for i in range(len(ema_cols)) for j in range(i+1, len(ema_cols))
        ), axis=1
    )
    
    in_signal = False
    last_signal = None

    for i in range(lookback, len(data) - lookback):
        if data['EMAs_Converge'].iloc[i]:
            if all(data['close'].iloc[i-j] < data['close'].iloc[i-j+1] for j in range(lookback, 0, -1)) and \
               all(data[ema_cols[k]].iloc[i-1] > data[ema_cols[k]].iloc[i-2] for k in range(len(ema_cols))):
                data.at[i + 1, 'Signal'] = 'Buy'
                in_signal = True
                last_signal = 'Buy'
            elif all(data['close'].iloc[i-j] > data['close'].iloc[i-j+1] for j in range(lookback, 0, -1)) and \
                 all(data[ema_cols[k]].iloc[i-1] < data[ema_cols[k]].iloc[i-2] for k in range(len(ema_cols))):
                data.at[i + 1, 'Signal'] = 'Sell'
                in_signal = True
                last_signal = 'Sell'
            else:
                if in_signal:
                    data.at[i + 1, 'Signal'] = last_signal
        else:
            in_signal = False

    signals = data[data['Signal'].isin(['Buy', 'Sell'])].reset_index(drop=True)
    return signals

# Simulate trades with SL and TP
def simulate_trades(data, signals, symbol, initial_balance=10000, funds_usage_percentage=100):
    balance = initial_balance
    trades = []
    active_trade = None
    trade_type = None
    stop_loss = 0.02
    take_profit = 0.04

    for i, row in data.iterrows():
        if active_trade is not None:
            if active_trade['type'] == 'Buy':
                if row['close'] >= active_trade['buy_price'] * (1 + take_profit):
                    # Take profit for buy trade
                    profit_loss = active_trade['lot_size'] * (row['close'] - active_trade['buy_price'])
                    percentage_profit_loss = ((row['close'] - active_trade['buy_price']) / active_trade['buy_price']) * 100
                    trades.append({
                        'symbol': symbol,
                        'type': 'Buy',
                        'buy_price': active_trade['buy_price'],
                        'sell_price': row['close'],
                        'buy_time': active_trade['buy_time'],
                        'sell_time': row['timestamp'],
                        'lot_size': active_trade['lot_size'],
                        'profit_loss': profit_loss,
                        '%Profit_Loss': str(round(percentage_profit_loss, 2)) + "%",
                        'signal': 'Sold bought stock Profit'
                    })
                    balance += profit_loss
                    active_trade = None
                elif row['close'] <= active_trade['buy_price'] * (1 - stop_loss):
                    # Stop loss for buy trade
                    profit_loss = active_trade['lot_size'] * (row['close'] - active_trade['buy_price'])
                    percentage_profit_loss = ((row['close'] - active_trade['buy_price']) / active_trade['buy_price']) * 100
                    trades.append({
                        'symbol': symbol,
                        'type': 'Buy',
                        'buy_price': active_trade['buy_price'],
                        'sell_price': row['close'],
                        'buy_time': active_trade['buy_time'],
                        'sell_time': row['timestamp'],
                        'lot_size': active_trade['lot_size'],
                        'profit_loss': profit_loss,
                        '%Profit_Loss': str(round(percentage_profit_loss, 2)) + "%",
                        'signal': 'Sold bought stock Loss'
                    })
                    balance += profit_loss
                    active_trade = None

            elif active_trade['type'] == 'Sell Short':
                if row['close'] <= active_trade['sell_price'] * (1 - take_profit):
                    # Take profit for sell trade
                    profit_loss = active_trade['lot_size'] * (active_trade['sell_price'] - row['close'])
                    percentage_profit_loss = ((active_trade['sell_price'] - row['close']) / active_trade['sell_price']) * 100
                    trades.append({
                        'symbol': symbol,
                        'type': 'Sell Short',
                        'sell_price': active_trade['sell_price'],
                        'buy_to_cover_price': row['close'],
                        'sell_time': active_trade['sell_time'],
                        'buy_to_cover_time': row['timestamp'],
                        'lot_size': active_trade['lot_size'],
                        'profit_loss': profit_loss,
                        '%Profit_Loss': str(round(percentage_profit_loss, 2)) + "%",
                        'signal': 'Sold short stock Profit'
                    })
                    balance += profit_loss
                    active_trade = None
                elif row['close'] >= active_trade['sell_price'] * (1 + stop_loss):
                    # Stop loss for sell trade
                    profit_loss = active_trade['lot_size'] * (active_trade['sell_price'] - row['close'])
                    percentage_profit_loss = ((active_trade['sell_price'] - row['close']) / active_trade['sell_price']) * 100
                    trades.append({
                        'symbol': symbol,
                        'type': 'Sell Short',
                        'sell_price': active_trade['sell_price'],
                        'buy_to_cover_price': row['close'],
                        'sell_time': active_trade['sell_time'],
                        'buy_to_cover_time': row['timestamp'],
                        'lot_size': active_trade['lot_size'],
                        'profit_loss': profit_loss,
                        '%Profit_Loss': str(round(percentage_profit_loss, 2)) + "%",
                        'signal': 'Sold short stock Loss'
                    })
                    balance += profit_loss
                    active_trade = None

        signal = signals[signals['timestamp'] == row['timestamp']]
        if not signal.empty:
            if signal.iloc[0]['Signal'] == 'Buy' and active_trade is None:
                lot_size = (balance * (funds_usage_percentage / 100)) / row['close']
                active_trade = {
                    'type': 'Buy',
                    'buy_price': row['close'],
                    'buy_time': row['timestamp'],
                    'lot_size': lot_size
                }
            elif signal.iloc[0]['Signal'] == 'Sell' and active_trade is None:
                lot_size = (balance * (funds_usage_percentage / 100)) / row['close']
                active_trade = {
                    'type': 'Sell Short',
                    'sell_price': row['close'],
                    'sell_time': row['timestamp'],
                    'lot_size': lot_size
                }

    return trades, balance

# Streamlit UI
st.title('Trading Strategy Backtesting')

# Parameters input
symbol = st.text_input('Enter Symbol (e.g., BTC/USDT)', 'BTC/USDT')
timeframe = st.selectbox('Select Timeframe', ['1m', '5m', '15m', '1h', '4h', '1d'], index=5)
limit = st.number_input('Enter Number of Data Points', min_value=100, max_value=1000, value=500)
strategy = st.selectbox('Select Trading Strategy', ['Strategy 1', 'Strategy 2', 'Strategy 3'])
tolerance = st.slider('Set Tolerance for EMA Convergence (%)', min_value=0.0, max_value=5.0, value=0.5)

if st.button('Fetch Data and Apply Strategy'):
    data = fetch_data(symbol, timeframe, limit)
    data = calculate_indicators(data)

    if strategy == 'Strategy 1':
        signals = apply_trading_strategy_01(data, tolerance)
    elif strategy == 'Strategy 2':
        signals = apply_trading_strategy_02(data, tolerance)
    elif strategy == 'Strategy 3':
        signals = apply_trading_strategy_03(data, tolerance=tolerance/100)

    st.write('Signals:', signals)

    trades, final_balance = simulate_trades(data, signals, symbol)
    st.write('Trades:', trades)
    st.write(f'Final Balance: {final_balance}')

    # Plotting the data with signals
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['timestamp'],
                                 open=data['open'],
                                 high=data['high'],
                                 low=data['low'],
                                 close=data['close'],
                                 name='Candlesticks'))

    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_20'], mode='lines', name='EMA 20'))
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_50'], mode='lines', name='EMA 50'))
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['EMA_100'], mode='lines', name='EMA 100'))

    buy_signals = signals[signals['Signal'] == 'Buy']
    sell_signals = signals[signals['Signal'] == 'Sell']

    fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals['close'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals['close'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'))

    fig.update_layout(title=f'{symbol} Price Data with Trading Signals', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
