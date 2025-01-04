from utils.all_imports import *


class TradingSymbolProcessor:
    def __init__(self, interval):
        self.interval = interval
        self.dict_watchlist = {}
        self._load_watchlist()
        self._setup_webhooks()
        self._setup_current_timestamp()

    def setup_for_new_symbol(self, symbol):
        # This method resets and initializes everything necessary for a new symbol
        self.symbol = symbol
        self.df_price = None
        self.cur_rvol = None
        self.cur_mvol = None
        self.cur_vol = None
        self.cur_rsi = None
        self.cur_rvol_multiplier = None
        self.cur_mvol_multiplier = None
        self.cur_EMA9 = None
        self.cur_EMA21 = None
        self.cur_EMA55 = None
        self.cur_EMA144 = None
        self.cur_EMA233 = None
        self.pre_rvol = None
        self.pre_mvol = None
        self.pre_vol = None
        self.pre_rsi = None
        self.pre_rvol_multiplier = None
        self.pre_mvol_multiplier = None
        self.pre_EMA21 = None
        self.pre_EMA55 = None
        self.pre_EMA144 = None
        self.pre_EMA233 = None
        self.price_ath = None
        self.price_atl = None
        self.price_ath_52w = None
        self.price_atl_52w = None

    def _load_watchlist(self):

        if self.interval == '1d':
            return

        ### Load symbols from the gap scanner watchlist
        path_watchlist_gap = 'watchlist\\watchlist_gap_scanner.csv'
        if not os.path.exists(path_watchlist_gap):
            print(f"Error: {path_watchlist_gap} does not exist.")
        else:
            df_watchlist_gap = pd.read_csv(path_watchlist_gap)
            self.list_wl_symbols_gap_scanner = df_watchlist_gap['symbol'].tolist()
            df_watchlist_gap.set_index('symbol', inplace=True)
            self.dict_watchlist['gap_scanner'] = df_watchlist_gap

        ### load symbols from the black horse watchlist
        path_watchlist_black_horse = 'watchlist\\watchlist_black_horse.csv'
        if not os.path.exists(path_watchlist_black_horse):
            print(f"Error: {path_watchlist_black_horse} does not exist.")
        else:
            df_watchlist_black_horse = pd.read_csv(path_watchlist_black_horse)
            self.list_wl_symbols_black_horse = df_watchlist_black_horse['symbol'].tolist()
            df_watchlist_black_horse.set_index('symbol', inplace=True)
            self.dict_watchlist['black_horse'] = df_watchlist_black_horse

        # return self.dict_watchlist

    def _setup_webhooks(self):
        # Setup Discord webhooks for notifications

        # - gap trading alerts
        self.webhook_discord_gap_scanner = Discord(url=dict_dc_webhook_trading_signal['gap_scanner'])

        # - black horse alerts
        self.webhook_discord_black_horse = Discord(url=dict_dc_webhook_trading_signal['black_horse'])

    def _setup_current_timestamp(self):
        current_time = int(time.time() * 1000)
        self.interval_duration_ms = dict_interval_duration_ms[self.interval]
        self.current_time_recent_close = current_time - (current_time % self.interval_duration_ms)
        self.current_time = current_time
        self.start_time_price = (self.current_time_recent_close
                                 - NUM_CANDLE_HIST_PRICE_DAILY * dict_interval_duration_ms[self.interval])

    def _get_price_data_daily(self):
        '''Use Yahoo Finance API to get the price data for the symbol'''
        # For daily, the start time is 6 months ago
        try:
            # Fetch data from Yahoo Finance
            stock_data = yf.Ticker(self.symbol)

            # Attempt to fetch 5 years of daily data
            df_price = stock_data.history(period="5y", interval="1d")
            if df_price.empty:
                df_price = stock_data.history(period="max", interval="1d")

            # report an error if the data is still empty
            if df_price.empty:
                raise Exception("No price data available")

            # Reset index to make date a column and rename columns
            df_price.reset_index(inplace=True)
            df_price.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

            # Remove unnecessary columns
            df_price = df_price[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # round OHLC columns to 3 decimal places
            df_price = df_price.round(3)

            # Calculate the close time by setting the hour to 16:00 EST
            df_price['Close Time'] = df_price['Time'] + pd.DateOffset(hours=16)

            # get rid of the last row if its "Close time" is in future
            now = pd.Timestamp.now(tz='America/New_York')  # Match the timezone
            if df_price['Close Time'].iloc[-1] > now:
                df_price = df_price.iloc[:-1]

            self.df_price = df_price

            return df_price

        except Exception as e:
            error_msg = f"Error getting price data for {self.symbol}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.df_price = None

    def _get_price_data_intraday(self, interval='30m'):
        try:
            # Fetch data from Yahoo Finance
            stock_data = yf.Ticker(self.symbol)

            # get the price data for the last 3 months or max
            for period in ['1mo', '3mo', 'max']:
                df_price = stock_data.history(period=period, interval=interval)
                if not df_price.empty:
                    break

            df_price.reset_index(inplace=True)
            df_price.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            df_price = df_price[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

            if self.interval == '30m':
                df_price['Close Time'] = df_price['Time'] + pd.DateOffset(minutes=30)
            elif self.interval == '15m':
                df_price['Close Time'] = df_price['Time'] + pd.DateOffset(minutes=15)

            # get rid of the last row if its "Close time" is in future
            if df_price['Close Time'].iloc[-1] > pd.Timestamp.now(tz='America/New_York'):
                df_price = df_price.iloc[:-1]

            self.df_price = df_price

        except Exception as e:
            error_msg = f"Error getting price data for {self.symbol}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.df_price = None

    def _calc_technical_indicators(self):
        try:
            df_price = self.df_price.copy()

            # get the max and min for all time
            self.price_ath = df_price['High'].max()
            self.price_atl = df_price['Low'].min()

            # get the 52w high and low
            self.price_ath_52w = df_price['High'].iloc[-255:].max()
            self.price_atl_52w = df_price['Low'].iloc[-255:].min()

            # calculate technical indicators
            df_price['EMA_9'] = talib.SMA(df_price['Close'], timeperiod=9)
            df_price['EMA_21'] = talib.SMA(df_price['Close'], timeperiod=21)
            df_price['EMA_55'] = talib.SMA(df_price['Close'], timeperiod=55)
            df_price['EMA_144'] = talib.SMA(df_price['Close'], timeperiod=144)
            df_price['RSI'] = talib.RSI(df_price['Close'], timeperiod=14)
            df_price['ATR'] = talib.ATR(df_price['High'], df_price['Low'], df_price['Close'], timeperiod=14)
            df_price['Volume_MA'] = talib.SMA(df_price['Volume'], timeperiod=20)

            # body high and lows
            df_price['body_high'] = np.maximum(df_price['Open'], df_price['Close'])
            df_price['body_low'] = np.minimum(df_price['Open'], df_price['Close'])

            # status params for the current candle
            self.cur_rsi = df_price['RSI'].iloc[-1]
            self.cur_vol = df_price['Volume'].iloc[-1]
            self.cur_EMA9 = df_price['EMA_9'].iloc[-1]
            self.cur_EMA21 = df_price['EMA_21'].iloc[-1]
            self.cur_EMA55 = df_price['EMA_55'].iloc[-1]
            self.cur_EMA144 = df_price['EMA_144'].iloc[-1]
            self.cur_rvol = df_price['Volume_MA'].iloc[-1]
            self.cur_mvol = df_price.iloc[-200:]['Volume'].median()
            self.cur_rvol_multiplier = self.cur_vol / self.cur_rvol
            self.cur_mvol_multiplier = self.cur_vol / self.cur_mvol

            # status params for the previous candle
            self.pre_rsi = df_price['RSI'].iloc[-2]
            self.pre_vol = df_price['Volume'].iloc[-2]
            self.pre_EMA21 = df_price['EMA_21'].iloc[-2]
            self.pre_EMA55 = df_price['EMA_55'].iloc[-2]
            self.pre_EMA144 = df_price['EMA_144'].iloc[-2]
            self.pre_rvol_multiplier = self.pre_vol / self.cur_rvol
            self.pre_mvol_multiplier = self.pre_vol / self.cur_mvol

            if self.interval == '1d':
                df_price['EMA_233'] = talib.SMA(df_price['Close'], timeperiod=233)
                self.cur_EMA233 = df_price['EMA_233'].iloc[-1]
                self.pre_EMA233 = df_price['EMA_233'].iloc[-2]

            # calculate PA indicators
            df_price['Direction'] = np.where(df_price['Close'] > df_price['Open'], 'U', 'D')
            df_price['lower_pinbar_length'] = np.minimum(df_price['Open'], df_price['Close']) - df_price['Low']
            df_price['upper_pinbar_length'] = df_price['High'] - np.maximum(df_price['Open'], df_price['Close'])

            # update the dataframe
            df_price.dropna(inplace=True)
            self.df_price = df_price

        except Exception as e:
            error_msg = f"Error calculating technical indicators for {self.symbol}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            # self.df

    def run_gap_scanner(self):
        return run_gap_scanner(self)

    def run_black_horse(self):
        return run_black_horse(self)

    def run_rsihl_alerts(self):
        return run_rsihl_alerts(self)


    def run(self):

        print(f"Processing {self.symbol} {self.interval}...")

        ### for 1d interval, run all the scanners
        if self.interval in ['1d']:
            self._get_price_data_daily()
            self._calc_technical_indicators()
            dict_results = {'gap_scanner': self.run_gap_scanner(),
                            'black_horse': self.run_black_horse(),
                            }

        ### for 30m, run the intraday alerts
        elif self.interval in ['30m']:
            # get the price data
            self._get_price_data_intraday()
            self._calc_technical_indicators()
            dict_results = {'rsihl_alerts': self.run_rsihl_alerts()}


        return dict_results

# make a local __main__ function that just print outs the wathclist info
# if __name__ == '__main__':
#     t = TradingSymbolProcessor('1d')
#     t._load_watchlist()
#     print(t.dict_watchlist)
#     print("Watchlist loaded successfully!")
