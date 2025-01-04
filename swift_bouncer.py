# src/screeners/swift_bouncer.py

import os
import pandas as pd
import plotly.graph_objs as go
import requests
import plotly.colors as colors
import json


def find_start_end_time_swift_bouncer(df_btc, lookback=144, window=5):
    """
    Find start and end times for the swift bouncer screener based on BTCUSDT data.

    Parameters:
    - df_btc (DataFrame): BTCUSDT DataFrame with datetime index.
    - lookback (int): Number of candles to look back in BTCUSDT data (e.g., 288).
    - window (int): Number of candles before and after to consider for local extrema.

    Returns:
    - tuple: (start_time, end_time, current_time) based on detected peaks and valleys.
    """

    # Parameters to control thresholds
    rate_of_change_thres = 0.025  # Minimum rate of change per bar (% per bar)
    abs_drop_amount_thres = 0.9   # Minimum absolute drop amount (%)

    # Get the last 'lookback' periods
    df = df_btc.iloc[-lookback:].copy()

    # Current time
    cur_time = df.index[-1]

    # Identify peaks and valleys using partial windows at edges
    df['Peak'] = False
    df['Valley'] = False

    n = len(df)
    for i in range(n):
        left_i = max(0, i - window)
        right_i = min(n - 1, i + window)

        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]

        local_highs = df['High'].iloc[left_i:right_i + 1]
        local_lows = df['Low'].iloc[left_i:right_i + 1]

        # If current candle's high is the max in the local window, it's a peak
        if current_high == local_highs.max():
            df.at[df.index[i], 'Peak'] = True

        # If current candle's low is the min in the local window, it's a valley
        if current_low == local_lows.min():
            df.at[df.index[i], 'Valley'] = True

    # Get all valley times (sorted chronologically)
    valley_times = df.index[df['Valley']]

    start_time = None
    end_time = None

    # We'll try from the most recent valley backward
    for candidate_valley_time in valley_times[::-1]:
        candidate_valley_low = df.loc[candidate_valley_time, 'Low']
        end_time = candidate_valley_time

        # Find peaks before this valley
        peaks_before_valley = df.loc[:end_time][df['Peak']]

        valid_peak_time = None
        max_abs_drop_amount = 0

        # Iterate over peaks in reverse chronological order (from most recent to oldest)
        for peak_time in peaks_before_valley.index[::-1]:
            peak_high = df.loc[peak_time, 'High']

            # Calculate absolute drop amount in percentage
            abs_drop_amount = ((peak_high - candidate_valley_low) / peak_high) * 100

            peak_loc = df.index.get_loc(peak_time)
            valley_loc = df.index.get_loc(end_time)
            num_bars = valley_loc - peak_loc

            if num_bars <= 0:
                continue  # Skip if no candles between peak and valley

            # Calculate rate of change (% change per bar)
            rate_of_change = abs_drop_amount / num_bars

            # Check if peak qualifies
            if abs_drop_amount > abs_drop_amount_thres and rate_of_change > rate_of_change_thres:
                # If this peak has a higher abs drop amount, update valid_peak_time
                if abs_drop_amount > max_abs_drop_amount:
                    max_abs_drop_amount = abs_drop_amount
                    valid_peak_time = peak_time

        # If we found a valid peak from this valley, break out and consider this done
        if valid_peak_time is not None:
            start_time = valid_peak_time
            break
        else:
            # Otherwise, try the next (earlier) valley
            continue

    # If no valley worked out
    if start_time is None:
        end_time = None  # No valid end_time either

    print(f"Start Time: {start_time}, End Time: {end_time}")

    # make a plot of the OHLC and the peak and valley using plotly
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='OHLC'))

    # Add all peaks and valleys for visualization
    all_peaks = df[df['Peak']]
    all_valleys = df[df['Valley']]
    fig.add_trace(go.Scatter(x=all_peaks.index, y=all_peaks['High'],
                             mode='markers', marker=dict(color='red'), name='Peaks'))
    fig.add_trace(go.Scatter(x=all_valleys.index, y=all_valleys['Low'],
                             mode='markers', marker=dict(color='green'), name='Valleys'))

    # Add the end_time marker if available
    if end_time is not None:
        fig.add_trace(go.Scatter(x=[end_time], y=[df.loc[end_time, 'Low']],
                                 mode='markers', marker=dict(color='blue'), name='End Time'))

    # Add the start_time marker if available
    if start_time is not None:
        fig.add_trace(go.Scatter(x=[start_time], y=[df.loc[start_time, 'High']],
                                 mode='markers', marker=dict(color='blue'), name='Start Time'))

    # fig.show()

    return start_time, end_time, cur_time


class SwiftBouncer():
    def __init__(self, data_dir):
        self.output_dir = os.path.join(data_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.df_results = None
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t1_str = None
        self.top_n = 10
        self.bottom_n = 10
        self.top_n_plot = 10
        self.bottom_n_plot = 10
        self.interval = '5m'
        self.top_symbols_swift = None
        self.top_symbols_drop = None

    def process(self, symbol_data_filtered):
        """
        Processes symbols to identify swift bouncer based on BTCUSDT's downtrend and rebound.

        Saves top 20 bullish and bottom 20 bearish symbols with normalized values.
        """

        # Step 1: Identify t1, t2, t3 in BTCUSDT's 5m data
        df_btc = symbol_data_filtered['BTCUSDT'][self.interval]
        t1, t2, t3 = find_start_end_time_swift_bouncer(df_btc, lookback=144, window=5)
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        # if t1 is None, send a message to discord and return
        if t1 is None:
            response = requests.post(
                webhook_jiehouyusheng,
                headers={'Content-Type': 'application/json'},
                data=json.dumps({'content': f"🙈暂无行情 耐心等待🙈\n"})
            )
            return

        self.t1_str = pd.Timestamp(self.t1).strftime('%Y%m%d_%H%M%S')
        self.t3_str = pd.Timestamp(self.t3).strftime('%Y%m%d_%H%M%S')

        # Step 2: Calculate drop_score, bounce_score, swift_score for each symbol
        results = []
        for symbol, data in symbol_data_filtered.items():

            df = data.get(self.interval)
            if df is None or df.empty:
                continue
            elif df['Close'].min() == df['Close'].max():
                # print(symbol)
                continue

            # Calculate price_t1, price_t3, and lowest price between t1 and t3
            try:
                price_t1 = df.loc[t1]['Close']
                price_t3 = df.loc[t3]['Close']
                df_t1_t3 = df.loc[t1:t3]
                lowest_price = df_t1_t3['Close'].min()
            except KeyError:
                # If any of the times are not present, skip this symbol
                continue

            # Normalize close prices
            df['Normalized_Close'] = df['Close'] / price_t1

            # Calculate drop_score, bounce_score, swift_score
            drop_score = ((lowest_price - price_t1) / price_t1) * 100  # Refined drop_score
            bounce_score = ((price_t3 - lowest_price) / lowest_price) * 100
            swift_score = ((price_t3 - price_t1) / price_t1) * 100

            results.append({
                'Symbol': symbol,
                'drop_score': drop_score,
                'bounce_score': bounce_score,
                'swift_score': swift_score
            })

        # Create a DataFrame from results
        df_results = pd.DataFrame(results)

        # Rank the symbols according to swift_score
        df_results.sort_values('swift_score', ascending=False, inplace=True)
        df_results.reset_index(drop=True, inplace=True)
        self.df_results = df_results

        # Save watchlists
        self.save_watchlist(df_results)

        # Step 3: Plotting
        self.plot_results(df_results, symbol_data_filtered, df_btc)

        # Plot watchlist progression
        # self.plot_watchlist_progression()

    def save_watchlist(self, df_results):
        """
        Save the top symbols to watchlist CSV files, tracking their progression over time.

        Parameters:
        - df_results (DataFrame): DataFrame containing ranking of symbols.
        """
        t1_str = self.t1_str
        t3_str = self.t3_str

        # Prepare watchlists
        # Swift Score Watchlist
        swift_watchlist_file = os.path.join(self.output_dir, f'watchlist_swift_score_{t1_str}.csv')

        top_symbols_swift = df_results.sort_values('swift_score', ascending=False).head(self.top_n_plot)
        symbols_swift = top_symbols_swift['Symbol'].tolist()
        rankings_swift = list(range(1, self.top_n_plot + 1))  # 1 to 20

        # Drop Score Watchlist
        drop_watchlist_file = os.path.join(self.output_dir, f'watchlist_drop_score_{t1_str}.csv')

        top_symbols_drop = df_results.sort_values('drop_score', ascending=False).head(self.top_n_plot)
        symbols_drop = top_symbols_drop['Symbol'].tolist()
        rankings_drop = list(range(1, self.top_n_plot + 1))  # 1 to 20

        # Update or create Swift Score Watchlist DataFrame
        if os.path.exists(swift_watchlist_file):
            df_swift = pd.read_csv(swift_watchlist_file, index_col=0)
        else:
            df_swift = pd.DataFrame(index=rankings_swift)

        df_swift[t3_str] = symbols_swift

        # Update or create Drop Score Watchlist DataFrame
        if os.path.exists(drop_watchlist_file):
            df_drop = pd.read_csv(drop_watchlist_file, index_col=0)
        else:
            df_drop = pd.DataFrame(index=rankings_drop)

        df_drop[t3_str] = symbols_drop

        # Save the DataFrames
        df_swift.to_csv(swift_watchlist_file)
        df_drop.to_csv(drop_watchlist_file)

        self.top_symbols_swift = top_symbols_swift
        self.top_symbols_drop = top_symbols_drop

    def plot_results(self, df_results, symbol_data_filtered, df_btc):
        """
        Plot the close prices for all symbols with specified coloring.

        Parameters:
        - df_results (DataFrame): DataFrame containing ranking of symbols.
        - symbol_data_filtered (dict): Dictionary containing OHLCV data for symbols.
        - df_btc (DataFrame): OHLCV data for BTCUSDT.
        """
        # Get the top and bottom symbols
        top_n = self.top_n_plot  # Number of top symbols to plot

        # top_symbols = df_results.head(self.top_n)['Symbol'].tolist()[:top_n]
        top_symbols_swift = self.top_symbols_swift.head(self.top_n)['Symbol'].tolist()[:top_n]
        top_symbols_drop = self.top_symbols_drop.head(self.top_n)['Symbol'].tolist()[:top_n]

        # Prepare gradient colors
        green_colors = colors.n_colors('rgb(0, 250, 0)', 'rgb(0, 100, 0)', top_n, colortype='rgb')
        yellow_colors = colors.n_colors('rgb(255, 255, 0)', 'rgb(255, 140, 0)', top_n, colortype='rgb')

        # Plotting - top symbols, swift
        fig = go.Figure()

        # Initialize min and max y for auto-adjust
        min_y = float('inf')
        max_y = float('-inf')

        # Plot all symbols in grey
        for symbol, data in symbol_data_filtered.items():
            if symbol in top_symbols_swift or symbol in top_symbols_drop or symbol == 'BTCUSDT':
                continue  # Skip top/bottom symbols and BTCUSDT for now
            df = data.get(self.interval)
            if df is None or df.empty:
                continue

            # if the max and min is the same, skip the symbol
            if df['Close'].max() == df['Close'].min():
                continue

            # Calculate normalized close prices
            try:
                df['Normalized_Close'] = df['Close'] / df.loc[self.t1, 'Close']
            except KeyError:
                print(f"Skipping {symbol} as 't1' is not present in its index.")
                continue

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Normalized_Close'],
                mode='lines',
                line=dict(color='grey', width=1),
                name=symbol,
                showlegend=False,
            ))
            min_y = min(min_y, df.loc[self.t1: self.t3, 'Normalized_Close'].min())
            max_y = max(max_y, df.loc[self.t1: self.t3, 'Normalized_Close'].max())

        # Plot top symbols (drop) with gradient yellow
        # first filter out the ones that are already in the top_symbols_swift
        top_symbols_drop_unique = [symbol for symbol in top_symbols_drop if symbol not in top_symbols_swift]


        for i, symbol in enumerate(top_symbols_drop_unique):
            data = symbol_data_filtered[symbol][self.interval]
            df = data.copy()
            df['Normalized_Close'] = df['Close'] / df.loc[self.t1, 'Close']  # Normalize
            color = yellow_colors[i]
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Normalized_Close'],
                mode='lines',
                line=dict(color=color, width=2),
                name=symbol,
                showlegend=False,
            ))
            min_y = min(min_y, df.loc[self.t1: self.t3, 'Normalized_Close'].min())
            max_y = max(max_y, df.loc[self.t1: self.t3, 'Normalized_Close'].max())

        # Plot top symbols with gradient green
        for i, symbol in enumerate(top_symbols_swift):
            data = symbol_data_filtered[symbol][self.interval]
            df = data.copy()
            df['Normalized_Close'] = df['Close'] / df.loc[self.t1, 'Close']  # Normalize
            color = green_colors[i]
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Normalized_Close'],
                mode='lines',
                line=dict(color=color, width=2),
                name=symbol,
                showlegend=False,
            ))
            min_y = min(min_y, df.loc[self.t1: self.t3, 'Normalized_Close'].min())
            max_y = max(max_y, df.loc[self.t1: self.t3, 'Normalized_Close'].max())


        # Plot BTCUSDT in orange
        df_btc['Normalized_Close'] = df_btc['Close'] / df_btc.loc[self.t1, 'Close']
        fig.add_trace(go.Scatter(
            x=df_btc.index,
            y=df_btc['Normalized_Close'],
            mode='lines',
            line=dict(color='orange', width=3),
            name='BTCUSDT',
            showlegend=False,
        ))

        # Add annotations at the end of each curve
        for symbol in top_symbols_drop_unique + top_symbols_swift + ['BTCUSDT']:
            if symbol == 'BTCUSDT':
                df = df_btc
                color = 'orange'
            else:
                data = symbol_data_filtered[symbol][self.interval]
                df = data.copy()
                df['Normalized_Close'] = df['Close'] / df.loc[self.t1, 'Close']
                if symbol in top_symbols_swift:
                    idx = top_symbols_swift.index(symbol)
                    color = green_colors[idx]
                else:
                    idx = top_symbols_drop_unique.index(symbol)
                    color = yellow_colors[idx]

            last_time = df.index[-1]
            last_value = df['Normalized_Close'].iloc[-1]
            fig.add_annotation(
                x=last_time,
                y=last_value,
                xref='x',
                yref='y',
                text=' ' + symbol[:-4],  # Remove 'USDT' for cleaner labels
                showarrow=False,
                font=dict(size=12, color=color),
                xanchor='left'
            )

        # Update layout
        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[self.t1, self.t3],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min_y, max_y],
            ),
            plot_bgcolor='black',
            paper_bgcolor='black',
            margin=dict(l=5, r=80, t=5, b=5),
            showlegend=False
        )

        # # show the fig
        # fig.show()

        # Save the interactive plot as HTML
        html_file = os.path.join(self.output_dir, 'swift_bouncer_plot.html')
        fig.write_html(html_file)

        # Save the plot and send to discord
        plot_file = os.path.join(self.output_dir, 'swift_bouncer_plot.png')
        fig.write_image(plot_file)

        self.send_plot_to_discord(plot_file)

    # def plot_watchlist_progression(self):
    #     """
    #     Plot the progression of the watchlist over time and send to Discord.
    #     """
    #     t1_str = self.t1_str
    #
    #     # Plot for swift_score watchlist
    #     swift_watchlist_file = os.path.join(self.output_dir, f'watchlist_swift_score_{t1_str}.csv')
    #
    #     if os.path.exists(swift_watchlist_file):
    #         df_swift = pd.read_csv(swift_watchlist_file, index_col=0)
    #         self.plot_rankings_over_time(df_swift, 'Swift Score Watchlist Progression', 'swift_watchlist_progression.png')
    #     else:
    #         print(f"Swift Score Watchlist file not found: {swift_watchlist_file}")
    #
    #     # Plot for drop_score watchlist
    #     drop_watchlist_file = os.path.join(self.output_dir, f'watchlist_drop_score_{t1_str}.csv')
    #
    #     if os.path.exists(drop_watchlist_file):
    #         df_drop = pd.read_csv(drop_watchlist_file, index_col=0)
    #         self.plot_rankings_over_time(df_drop, 'Drop Score Watchlist Progression', 'drop_watchlist_progression.png')
    #     else:
    #         print(f"Drop Score Watchlist file not found: {drop_watchlist_file}")

    def plot_rankings_over_time(self, df_watchlist, title, output_filename):
        """
        Plot the ranking progression of symbols over time.

        Parameters:
        - df_watchlist (DataFrame): DataFrame containing symbols and their rankings over time.
        - title (str): Title of the plot.
        - output_filename (str): Name of the output image file.
        """
        # Transpose and stack the DataFrame
        df_transposed = df_watchlist.transpose()  # Rows: timestamps, Columns: rankings
        df_stacked = df_transposed.stack().reset_index()
        df_stacked.columns = ['timestamp', 'ranking', 'symbol']
        df_stacked['ranking'] = df_stacked['ranking'].astype(int)

        # Pivot to get DataFrame with symbols as columns, rankings as values
        df_plot = df_stacked.pivot(index='timestamp', columns='symbol', values='ranking')

        # Sort symbols based on latest rankings
        latest_rankings = df_plot.iloc[-1].dropna().sort_values()
        symbols_sorted = latest_rankings.index.tolist()

        # Assign colors to symbols based on their latest rankings
        colorscale = colors.n_colors('rgb(0, 250, 0)', 'rgb(255, 255, 0)', self.top_n, colortype='rgb')
        color_mapping = {}
        for i, symbol in enumerate(symbols_sorted):
            color_mapping[symbol] = colorscale[i]

        # Plot lines for each symbol
        fig = go.Figure()

        for symbol in symbols_sorted:
            y_values = df_plot[symbol]
            x_values = pd.to_datetime(df_plot.index)

            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                line=dict(color=color_mapping[symbol], width=2),
                name=symbol,
                showlegend=False
            ))

        # Add annotations at the last point
        for symbol in symbols_sorted:
            y_value = df_plot[symbol].iloc[-1]
            x_value = pd.to_datetime(df_plot.index[-1])
            fig.add_annotation(
                x=x_value,
                y=y_value,
                xref='x',
                yref='y',
                text=symbol[:-4],  # Remove 'USDT' for cleaner labels
                showarrow=False,
                font=dict(size=12, color=color_mapping[symbol]),
                xanchor='left',
                yanchor='middle'
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Ranking',
            yaxis=dict(
                autorange='reversed',
                dtick=1,
                range=[self.top_n + 0.5, 0.5]
            ),
            plot_bgcolor='white',
            margin=dict(l=40, r=200, t=40, b=40),
        )

        # Save the plot
        output_path = os.path.join(self.output_dir, output_filename)
        fig.write_image(output_path)

        # Send the plot to Discord
        self.send_plot_to_discord(output_path)

    def send_plot_to_discord(self, image_path):
        """
        Send the plot image to the Discord channel with a beautifully formatted table.

        Parameters:
        - image_path (str): Path to the image file.
        """

        # Get the top and bottom symbols
        top_symbols_swift = self.top_symbols_swift.head(self.top_n)['Symbol'].tolist()
        top_symbols_drop = self.top_symbols_drop.tail(self.bottom_n)['Symbol'].tolist()

        # Determine the maximum number of rows needed
        max_rows = max(len(top_symbols_swift), len(top_symbols_drop))

        # Pad the shorter list with empty strings to align the table
        top_symbols_swift += [''] * (max_rows - len(top_symbols_swift))
        top_symbols_drop += [''] * (max_rows - len(top_symbols_drop))

        # Combine the bullish and bearish symbols into a list of tuples
        combined_symbols = list(zip(top_symbols_swift, top_symbols_drop))

        # Create the markdown table with two columns
        def create_combined_markdown_table(combined_symbols):
            table = "```\n"
            table += f"{'综合战力⚔️':<16}| {'最强防御🛡️':<18}\n"
            table += f"{'-'*18}-|-{'-'*18}\n"
            for bullish, bearish in combined_symbols:
                # bullish = bullish + '.P'
                # bearish = bearish + '.P'
                table += f"{bullish:<18} | {bearish:<18}\n"
            table += f"{self.t1_str}\n"
            table += "```\n"
            return table

        combined_table = create_combined_markdown_table(combined_symbols)


        # Construct the embed with the combined table
        embed = {
            "title": "狗庄战力指数⚔️🛡️",
            "color": 0x00FF00,  # Green color
            "fields": [
                {
                    "name": "",
                    "value": combined_table,
                    "inline": False
                }
            ],
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "image": {
                "url": "attachment://swift_bouncer_plot.png"
            }
        }

        # Prepare the payload
        payload = {
            "embeds": [embed]
        }

        # Send the embed with the image as attachment
        with open(image_path, 'rb') as f:
            files = {
                'file': ('swift_bouncer_plot.png', f, 'image/png')
            }
            # To correctly send both embed and files, use 'payload_json' in 'data'
            response = requests.post(
                webhook_jiehouyusheng,
                data={'payload_json': json.dumps(payload)},
                files=files
            )

        # Check the response for errors
        if response.status_code != 200:
            print(f"Failed to send message. Status code: {response.status_code}")
            try:
                error_response = response.json()
                print(json.dumps(error_response, indent=4, ensure_ascii=False))
            except ValueError:
                print(response.text)






