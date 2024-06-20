import streamlit as st
import pandas as pd
import cryptocompare
from datetime import datetime
import plotly.graph_objs as go
import time

# Initialize an empty DataFrame with column names
columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(columns=columns)

# Create a Plotly FigureWidget for real-time updates
fig = go.FigureWidget(
    go.Candlestick(x=[], open=[], high=[], low=[], close=[],
                   increasing_line_color='green', decreasing_line_color='red')
)

# Update layout with gridlines and y-axis formatting
fig.update_layout(
    title='Real-Time Candlestick Chart',
    xaxis_title='Timestamp',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    yaxis=dict(
        tickformat='.2f',  # Format y-axis ticks to 2 decimal places
    ),
    yaxis_showgrid=True,  # Show gridlines on y-axis
)

# Display the figure widget in full screen mode
chart = st.plotly_chart(fig, use_container_width=True)

# Global variable to control fetching process
fetching_data = False

# Function to fetch real-time data
def fetch_real_time_data(symbol, interval=60):
    global df, fetching_data

    while fetching_data:
        try:
            # Fetch the current price data
            price = cryptocompare.get_price(symbol, currency='USD', full=True)
            
            if price and 'RAW' in price and symbol in price['RAW']:
                data = price['RAW'][symbol]['USD']
                timestamp = datetime.now()
                new_row = pd.DataFrame([[timestamp, symbol, data['OPEN24HOUR'], 
                                         data['HIGH24HOUR'], data['LOW24HOUR'], 
                                         data['PRICE'], data['VOLUME24HOUR']]], 
                                       columns=columns)
                
                # Update the DataFrame with the new row
                df = pd.concat([new_row, df], ignore_index=True)
                df = df[columns]  # Ensure columns are in correct order
                
                # Update the candlestick chart and display
                update_candlestick_chart(df)

                # Display the updated table
                update_table(df)
            
            # Sleep for the specified interval
            time.sleep(interval)
        
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            time.sleep(interval)

# Function to update the candlestick chart using Plotly
def update_candlestick_chart(data):
    try:
        if data.empty:
            st.warning("DataFrame is empty, cannot plot.")
            return

        fig.data[0].x = data['timestamp']
        fig.data[0].open = data['open']
        fig.data[0].high = data['high']
        fig.data[0].low = data['low']
        fig.data[0].close = data['close']
        
        # Display the candlestick chart in full screen
        chart.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error updating candlestick chart: {e}")

# Function to update the table display
def update_table(data):
    st.subheader('Last Updated Data')
    if not data.empty:
        # Display the latest 10 rows in reverse chronological order
        latest_data = data.head(10).iloc[::-1].reset_index(drop=True)
        st.table(latest_data)
    else:
        st.write("DataFrame is empty, no data to display.")

# Main function to run the Streamlit app
def main():
    global fetching_data

    st.title('Real-Time Candlestick Chart')

    # Selection widgets
    start_button = st.button('Start')
    stop_button = st.button('Stop')
    interval = st.selectbox('Select interval (seconds)', [30, 60, 120], index=1)
    symbol = st.selectbox('Select cryptocurrency', ['BTC', 'ETH', 'XRP'], index=0)

    if start_button:
        fetching_data = True
        fetch_real_time_data(symbol, interval)

    if stop_button:
        fetching_data = False

if __name__ == "__main__":
    main()
