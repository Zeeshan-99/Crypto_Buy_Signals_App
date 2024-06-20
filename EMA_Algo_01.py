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

def generate_signals(data, tolerance=0.02):
    data['Signal'] = ''
    data['EMAs_Converge'] = ((abs(data['EMA_20'] / data['EMA_50'] - 1) <= tolerance) &
                             (abs(data['EMA_20'] / data['EMA_100'] - 1) <= tolerance) &
                             (abs(data['EMA_50'] / data['EMA_100'] - 1) <= tolerance))
    
    in_signal = False
    last_signal = None

    for i in range(2, len(data)-2):
        if data['EMAs_Converge'][i]:
            if (data['Close'][i-2] < data['Close'][i-1] < data['Close'][i]) and (data['EMA_20'][i-1] > data['EMA_20'][i-2]) and (data['EMA_50'][i-1] > data['EMA_50'][i-2]) and (data['EMA_100'][i-1] > data['EMA_100'][i-2]):
                data.loc[i, 'Signal'] = 'Buy'
                in_signal = True
                last_signal = 'Buy'
            elif (data['Close'][i-2] > data['Close'][i-1] > data['Close'][i]) and (data['EMA_20'][i-1] < data['EMA_20'][i-2]) and (data['EMA_50'][i-1] < data['EMA_50'][i-2]) and (data['EMA_100'][i-1] < data['EMA_100'][i-2]):
                data.loc[i, 'Signal'] = 'Sell'
                in_signal = True
                last_signal = 'Sell'
            else:
                if in_signal:
                    data.loc[i, 'Signal'] = last_signal
        else:
            in_signal = False

    signals = data[data['Signal'].isin(['Buy', 'Sell'])]
    signals.reset_index(drop=True, inplace=True)
    return signals

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

st.set_page_config(layout="wide")
st.title('AAPL Stock Analysis with Buy/Sell Signals')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    data = calculate_emas(data)
    
    signals = generate_signals(data)
    
    calculate_close_price_diff(signals)

    st.subheader('Original Dataset')
    with st.expander("Expand to view the original dataset"):
        st.dataframe(data, height=500)

    st.subheader('Resultant Dataset with Signals')
    with st.expander("Expand to view the resultant dataset with signals"):
        st.dataframe(data, height=500)

    st.subheader('Filtered Dataset with Buy/Sell Signals')
    with st.expander("Expand to view the filtered dataset with buy/sell signals"):
        st.dataframe(signals, height=500)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick',
                                 yaxis='y'))

    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_20'], mode='lines', name='EMA 20', yaxis='y'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_50'], mode='lines', name='EMA 50', yaxis='y'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_100'], mode='lines', name='EMA 100', yaxis='y'))

    fig.add_trace(go.Scatter(x=signals[signals['Signal'] == 'Buy']['Date'], y=signals[signals['Signal'] == 'Buy']['Low'],
                             mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                             name='Buy Signal', yaxis='y'))
    fig.add_trace(go.Scatter(x=signals[signals['Signal'] == 'Sell']['Date'], y=signals[signals['Signal'] == 'Sell']['High'],
                             mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                             name='Sell Signal', yaxis='y'))

    fig.update_layout(
        title='AAPL Stock Prices with Persistent Buy/Sell Signals at EMA Convergence (5% Tolerance)',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date',
            fixedrange=False
        ),
        yaxis=dict(
            title='Price',
            side='left',
            showgrid=True,
            zeroline=False,
            fixedrange=False,
            tickformat="$.2f"  # format y-axis ticks to two decimal places
        ),
        yaxis2=dict(
            title='Price',
            overlaying='y',
            side='right',
            showgrid=True,
            zeroline=False,
            tickformat="$.2f"  # format y-axis ticks to two decimal places
        ),
        template='plotly_dark',
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Seaborn Line Plots for Close Price Differences')

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
    
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")
