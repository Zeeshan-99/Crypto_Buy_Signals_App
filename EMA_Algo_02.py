import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    return data

def calculate_emas(data):
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
    return data

def generate_signals_algo_01(data, tolerance):
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

def generate_signals_algo_02(data, tolerance):
    data['Signal'] = ''
    data['EMAs_Converge'] = ((abs(data['EMA_20'] / data['EMA_50'] - 1) <= tolerance) &
                             (abs(data['EMA_20'] / data['EMA_100'] - 1) <= tolerance) &
                             (abs(data['EMA_50'] / data['EMA_100'] - 1) <= tolerance))
    
    in_signal = False
    last_signal = None

    for i in range(3, len(data)-2):  # Check at least 3 candles for trends
        if data['EMAs_Converge'].iloc[i]:
            # Buy signal criteria
            if (data['Close'].iloc[i-2] < data['Close'].iloc[i-1] < data['Close'].iloc[i]) and \
               (data['EMA_20'].iloc[i-1] > data['EMA_20'].iloc[i-2]) and \
               (data['EMA_50'].iloc[i-1] > data['EMA_50'].iloc[i-2]) and \
               (data['EMA_100'].iloc[i-1] > data['EMA_100'].iloc[i-2]):
                data.at[i + 1, 'Signal'] = 'Buy'
                in_signal = True
                last_signal = 'Buy'
            # Sell signal criteria
            elif (data['Close'].iloc[i-2] > data['Close'].iloc[i-1] > data['Close'].iloc[i]) and \
                 (data['EMA_20'].iloc[i-1] < data['EMA_20'].iloc[i-2]) and \
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

def generate_signals_algo_03(data):
    data['Signal'] = ''
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']
    
    in_position = False

    for i in range(1, len(data)-1):
        if data['MACD_Diff'].iloc[i-1] < 0 and data['MACD_Diff'].iloc[i] > 0:
            data.at[i, 'Signal'] = 'Buy'
            in_position = True
        elif data['MACD_Diff'].iloc[i-1] > 0 and data['MACD_Diff'].iloc[i] < 0:
            data.at[i, 'Signal'] = 'Sell'
            in_position = False
        elif in_position:
            data.at[i, 'Signal'] = 'Hold'

    signals = data[data['Signal'].isin(['Buy', 'Sell'])].reset_index(drop=True)
    return signals

def generate_signals_algo_04(data):
    data['Signal'] = ''
    data['RSI'] = calculate_rsi(data['Close'], 14)
    
    in_position = False

    for i in range(1, len(data)-1):
        if data['RSI'].iloc[i-1] < 30 and data['RSI'].iloc[i] > 30:
            data.at[i, 'Signal'] = 'Buy'
            in_position = True
        elif data['RSI'].iloc[i-1] > 70 and data['RSI'].iloc[i] < 70:
            data.at[i, 'Signal'] = 'Sell'
            in_position = False
        elif in_position:
            data.at[i, 'Signal'] = 'Hold'

    signals = data[data['Signal'].isin(['Buy', 'Sell'])].reset_index(drop=True)
    return signals

def calculate_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_close_price_diff(df):
    start_idx = None
    current_signal = None
    
    for idx, row in df.iterrows():
        signal = row['Signal']
        
        if signal != current_signal:
            if start_idx is not None:
                end_idx = idx - 1
                df.loc[end_idx, 'Close_Price_Diff'] = df.loc[start_idx, 'Close'] - df.loc[end_idx, 'Close']
            start_idx = idx
            current_signal = signal
    
    if start_idx is not None and current_signal is not None:
        end_idx = df.index[-1]
        df.loc[end_idx, 'Close_Price_Diff'] = df.loc[start_idx, 'Close'] - df.loc[end_idx, 'Close']

# Assume all your previous functions and imports are here

def plot_data(data, signals, tolerance, option):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'))

    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_20'], mode='lines', name='EMA 20'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_50'], mode='lines', name='EMA 50'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_100'], mode='lines', name='EMA 100'))

    if option == 'Algorithm 4':  # RSI Algorithm
        fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
        fig.add_trace(go.Scatter(x=data['Date'], y=[30] * len(data), mode='lines', name='RSI Oversold', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=data['Date'], y=[70] * len(data), mode='lines', name='RSI Overbought', line=dict(dash='dot')))

    fig.add_trace(go.Scatter(x=signals[signals['Signal'] == 'Buy']['Date'], y=signals[signals['Signal'] == 'Buy']['Low'],
                             mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                             name='Buy Signal'))
    fig.add_trace(go.Scatter(x=signals[signals['Signal'] == 'Sell']['Date'], y=signals[signals['Signal'] == 'Sell']['High'],
                             mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                             name='Sell Signal'))

    fig.update_layout(
        title=f'Crypto/Stock Prices with Persistent Buy/Sell Signals ({option})',
        xaxis_title='Date',
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
        range=[data['Close'].min() * 0.95, data['Close'].max() * 1.05],  # Adjust the initial visible range as desired
        fixedrange=False  # Allow zooming
    )
    return fig

def plot_seaborn(signals):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    sns.lineplot(x=range(len(signals[signals['Signal']=='Sell'])), 
                 y="Close_Price_Diff",  
                 data=signals[signals['Signal']=='Sell'],
                 ax=ax[0])
    
    ax[0].set_title('Sell')
    ax[0].grid()
    ax[0].axhline(0, color='red', linewidth=2)
    
    sns.lineplot(x=range(len(signals[signals['Signal']=='Buy'])),  
                 y="Close_Price_Diff",  
                 data=signals[signals['Signal']=='Buy'],
                 ax=ax[1])
    ax[1].set_title('Buy')
    ax[1].grid()
    ax[1].axhline(0, color='red', linewidth=2)
    
    return fig

st.set_page_config(layout="wide")
st.title('Crypto/Stock Analysis with Buy/Sell Signals')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = calculate_emas(data)
    
    st.sidebar.subheader('Tolerance Parameters')
    tolerance = st.sidebar.slider('Tolerance', min_value=0.01, max_value=0.2, value=0.05, step=0.01, key='tolerance')
    
    st.sidebar.subheader('Choose Algorithm')
    option = st.sidebar.selectbox(
        'Choose an algorithm:',
        ('Algorithm 1', 'Algorithm 2', 'Algorithm 3', 'Algorithm 4')
    )
    
    if option == 'Algorithm 1':
        signals = generate_signals_algo_01(data, tolerance)
    elif option == 'Algorithm 2':
        signals = generate_signals_algo_02(data, tolerance)
    elif option == 'Algorithm 3':
        signals = generate_signals_algo_03(data)
    else:
        signals = generate_signals_algo_04(data)
    
    calculate_close_price_diff(signals)

    st.subheader('Original Dataset')
    with st.expander("Expand to view the original dataset"):
        st.dataframe(data, height=500)

    st.subheader('Resultant Dataset with Signals')
    with st.expander("Expand to view the resultant dataset with signals"):
        st.dataframe(signals, height=500)

    fig = plot_data(data, signals, tolerance, option)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Seaborn Line Plots for Close Price Differences')
    fig_seaborn = plot_seaborn(signals)
    st.pyplot(fig_seaborn)

else:
    st.write("Please upload a CSV file to proceed.")
