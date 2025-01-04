def load_symbols():
    # Load symbols from a list_symbols.txt file
    path_symbols = 'symbols\\list_us_stocks.csv'
    if not os.path.exists(path_symbols):
        print(f"Error: {path_symbols} does not exist.")
        return None
    else:
        df_symbols = pd.read_csv(path_symbols)
        all_symbols = df_symbols['Symbol'].tolist()

    # reorder the symbols in alphabetical order
    all_symbols.sort()

    return all_symbols

def load_symbols_nasdaq_all():
    path_symbols = 'symbols\\list_all_nasdaq_symbols.csv'
    if not os.path.exists(path_symbols):
        print(f"Error: {path_symbols} does not exist.")
        return None
    else:
        df_symbols = pd.read_csv(path_symbols)
        all_symbols = df_symbols['symbol'].tolist()

    return all_symbols

def load_symbols_watchlist():

    # create the master watch list dataframe, with three columns
    # symbol, ref_value, strategy
    df_watchlist = pd.DataFrame(columns=['symbol', 'ref_value', 'strategy'])

    ### Load symbols from the gap scanner watchlist
    path_watchlist_gap = 'watchlist\\watchlist_gap_scanner.csv'
    df_watchlist_gap = pd.read_csv(path_watchlist_gap)
    df_watchlist_gap['strategy'] = 'gap_scanner'
    df_watchlist = pd.concat([df_watchlist, df_watchlist_gap], ignore_index=True)

    ### load symbols from the black horse watchlist
    path_watchlist_black_horse = 'watchlist\\watchlist_black_horse.csv'
    df_watchlist_black_horse = pd.read_csv(path_watchlist_black_horse)
    df_watchlist_black_horse['strategy'] = 'black_horse'
    df_watchlist = pd.concat([df_watchlist, df_watchlist_black_horse], ignore_index=True)

    ### remove duplicates and create a list of symbols
    df_watchlist = df_watchlist.drop_duplicates(subset=['symbol'])
    all_symbols = df_watchlist['symbol'].tolist()

    return all_symbols

def worker_init(interval):
    # This will create a global processor object that can be reused within the worker process
    global processor
    processor = TradingSymbolProcessor(interval)

def process_symbol(symbol):
    # print(f"Processing {symbol}")
    processor.setup_for_new_symbol(symbol)
    results = processor.run()

    # pause for 2 second
    time.sleep(2)

    return results

def post_analysis_gap_scanner(results_gap_scanner):
    webhook_gap_scanner = Discord(url=dict_dc_webhook_trading_signal['gap_scanner'])
    num_results_ar = len(results_gap_scanner)
    if num_results_ar == 0:
        msg_ar_results = f"--缺口策略 | No Results.\n"

    else:
        msg_ar_results = f"--缺口策略 | {num_results_ar} results.\n"

        # save the list to a csv file
        path_wl_gap_scanner = f"watchlist\\watchlist_gap_scanner.csv"
        df_results_gap_scanner = pd.DataFrame.from_dict(results_gap_scanner)
        df_results_gap_scanner = df_results_gap_scanner[['symbol', 'ref_value']]
        df_results_gap_scanner.to_csv(path_wl_gap_scanner, index=False)

    return msg_ar_results




def main(interval):

    # set the number of processes
    # num_processes = multiprocessing.cpu_count()
    num_processes = 4

    # load symbols
    if interval == '1d':
        all_symbols = load_symbols_nasdaq_all()
        all_symbols = all_symbols[::-1]
    else:
        all_symbols = load_symbols_watchlist()

    with multiprocessing.Pool(num_processes, initializer=worker_init, initargs=(interval,)) as pool:
        results_all = pool.map(process_symbol, all_symbols)  # Collects all results into a list

    # filter out the None results, and show which ones are None
    results_all = [result for result in results_all if result is not None]

    ### -- for 1d interval, we run the daily scanner
    # process the gap_scanner results
    if interval in dict_study_intervals['gap_scanner']:
        results_gap_scanner = [result['gap_scanner'] for result in results_all if result['gap_scanner'] is not None]
        msg_results_gap_scanner = post_analysis_gap_scanner(results_gap_scanner)
        msg_master_log += msg_results_gap_scanner


if __name__ == '__main__':

    # use argparse to get the interval
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", help="Interval to run the scanner", default='1d')
    args = parser.parse_args()
    interval = args.interval

    # debug override
    # interval = '30m'

    # check the current date, if it is Staurday or Sunday, then skip the job
    current_date = datetime.datetime.now(pytz.timezone('US/Eastern'))
    if current_date.weekday() in [5, 6]:
        print(f"Today is {current_date.strftime('%Y-%m-%d')}, which is a weekend. Skip the job.")
        exit()

    if interval not in ['30m', '1d']:
        print(f"Error: {interval} is not a valid interval.")
        exit()

    if interval in ['30m']:
        current_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
        time_market_open = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
        time_market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        if current_time < time_market_open or current_time > time_market_close:
            print(f"Current time is outside of the trading hours. Skip the job.")
            exit()
    else:
        # run the job
        main(interval)



