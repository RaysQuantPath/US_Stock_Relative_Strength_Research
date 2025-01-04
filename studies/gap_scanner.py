# params
N_LONG_TREND_CONSECUTIVE_CANDLES = 20
THRES_FAST_SLOW_EMA_SLOPE_DIFF = 0.1
DIST_GAP_FROM_PRICE = 0.9

# output
SEND_PLOTS = False


def run_gap_scanner(trading_symbol_processor, send_plots=SEND_PLOTS):
    if trading_symbol_processor.interval != '1d':
        return None

    if trading_symbol_processor.df_price is None:
        return None

    try:
        df_price_daily = trading_symbol_processor.df_price.copy()
        # keep only the last 120 candles
        df_price_daily = df_price_daily.iloc[-120:]

        # Condition #1 - if 55MEA is not above 144EMA for the last N_LONG_TREND_CONSECUTIVE_CANDLES, return None
        if not (df_price_daily['EMA_55'].iloc[-N_LONG_TREND_CONSECUTIVE_CANDLES:] >
                df_price_daily['EMA_144'].iloc[-N_LONG_TREND_CONSECUTIVE_CANDLES:]
        ).all():
            return None

        # Condition #2 - with N_LONG_TREND_CONSECUTIVE_CANDLES, if the 55EMA and 144EMA do not have a positive slope
        slope_55EMA = np.polyfit(
            range(N_LONG_TREND_CONSECUTIVE_CANDLES),
            df_price_daily['EMA_55'].iloc[-N_LONG_TREND_CONSECUTIVE_CANDLES:],
            1
        )[0]
        slope_144EMA = np.polyfit(
            range(N_LONG_TREND_CONSECUTIVE_CANDLES),
            df_price_daily['EMA_144'].iloc[-N_LONG_TREND_CONSECUTIVE_CANDLES:],
            1
        )[0]

        if slope_55EMA < 0 or slope_144EMA < 0:
            return None

        # Condition #3 - 55EMA slope must be at least THRES_FAST_SLOW_EMA_SLOPE_DIFF greater than 144EMA slope
        if not slope_55EMA > slope_144EMA * (1 + THRES_FAST_SLOW_EMA_SLOPE_DIFF):
            return None

        # Find and update gaps
        gaps = detect_and_update_gaps(df_price_daily)
        open_short_gaps = [gap for gap in gaps if gap['status'] != 'filled' and gap['gap_type'] == 'SHORT']
        open_long_gaps = [gap for gap in gaps if gap['status'] != 'filled' and gap['gap_type'] == 'LONG']

        # if there is no open long gap
        if len(open_long_gaps) == 0:
            return None

        # if there is a short gap above the current price
        current_close_price = df_price_daily['Close'].iloc[-1]
        short_gaps_above_price = [
            gap for gap in open_short_gaps
            if gap['value_gap_higher'] > current_close_price
        ]
        if len(short_gaps_above_price) > 0:
            return None

        # if the highest open long gap is less than DIST_GAP_FROM_PRICE of the current price
        highest_open_long_gap = max(open_long_gaps, key=lambda x: x['value_gap_higher'])
        ref_recent_gap_price = highest_open_long_gap['value_gap_higher']
        ref_recent_gap_price = round(ref_recent_gap_price, 3)
        if ref_recent_gap_price < current_close_price * DIST_GAP_FROM_PRICE:
            return None

        if SEND_PLOTS:
            # Plot the gaps with EMAs and RSI
            fig = plot_gaps_with_ohlcv(
                df_price_daily=df_price_daily,
                gaps=gaps,
                symbol=trading_symbol_processor.symbol,
                interval=trading_symbol_processor.interval
            )

            # Post the plot to the alert channel for daily watchlist preview
            fig.write_image(f"fig_gap_scanner_{trading_symbol_processor.symbol}.png")
            webhook_gap_scanner.post(
                content=f"{trading_symbol_processor.symbol} {trading_symbol_processor.interval}",
                file={
                    "file1": open(f"fig_gap_scanner_{trading_symbol_processor.symbol}.png", "rb"),
                },
            )

            # remove the plot
            os.remove(f"fig_gap_scanner_{trading_symbol_processor.symbol}.png")

        gap_scanner_results = {
            'symbol': trading_symbol_processor.symbol,
            'gaps': gaps,
            'open_long_gaps': open_long_gaps,
            'ref_value': ref_recent_gap_price,
        }
        return gap_scanner_results

    except Exception as e:
        print(f"Error running gap scanner for {trading_symbol_processor.symbol}: {str(e)}")
        traceback.print_exc()
        return None
