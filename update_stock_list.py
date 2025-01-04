from utils.all_imports import *

# Create a rate-limited session
session = LimiterSession(per_second=1)

def get_spy_sector(symbol):
    ticker = yf.Ticker(symbol, session=session)
    sector = ticker.info.get('sector')  # Get the sector from yfinance
    print(sector)

    # Create a mapping from sectors to SPY sectors (example, needs refinement)
    sector_mapping = {
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Technology': 'XLK',
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Healthcare': 'XLV',
        'Financial Services': 'XLF',
        'Basic Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }
    return sector_mapping.get(sector, 'NA')

def process_symbol(symbol):
    spy_sector = get_spy_sector(symbol)
    return {'Symbol': symbol, 'IS_SPY': 'Y', 'SPY_SECTOR_ETF': spy_sector}

def get_spy_holdings():
    import re

    # Ref: https://stackoverflow.com/a/75845569/
    # Source: https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-sp-500-etf-trust-spy
    url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
    df_spy_holdings = pd.read_excel(url, engine='openpyxl', index_col='Ticker', skiprows=4).dropna()
    df_spy_holdings = df_spy_holdings.reset_index()
    df_spy_holdings = df_spy_holdings.rename(columns={'Ticker': 'symbol'})

    # initial list
    list_spy_symbols = df_spy_holdings['symbol'].tolist()
    list_spy_symbols.sort()

    # only keep symbols that are strictly alphabetic (A-Z)
    # first, uppercase everything to ensure uniformity
    list_spy_symbols = [x.upper() for x in list_spy_symbols if x]

    # filter out any symbol that isn't purely alphabetical
    list_spy_symbols = [x for x in list_spy_symbols if re.match(r'^[A-Z]+$', x)]

    return list_spy_symbols

def get_qqq_holdings():
    import requests
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
    res = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=headers)
    main_data = res.json()['data']['data']['rows']

    list_qqq_symbols = []
    for i in range(len(main_data)):
        list_qqq_symbols.append(main_data[i]['symbol'])

    # sort the symbols alphabetically
    list_qqq_symbols.sort()

    return list_qqq_symbols

def get_iwm_holdings():

    list_iwm_symbols = []
    response_tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_1000_Index")
    for df in response_tables:
        # Check if the DataFrame has more than 900 rows and a column named "Symbol", "symbols", or "symbol" (case-insensitive)
        if len(df) > 900 and any(col.lower() == 'symbol' for col in df.columns):
            # If found, extract the symbols and return them
            list_iwm_symbols = df['Symbol'].tolist()  # Assuming the column is named 'Symbol'

    return list_iwm_symbols

def get_custom_symbols():
    # This function reads symbols from a local Python file.
    from symbols.custom_symbols import custom_symbols_list  # Assuming the file is named 'custom_symbols.py'
    return custom_symbols_list

if __name__ == '__main__':  # Important for multiprocessing on Windows

    # set a timer
    start_time = time.time()

    # get the S&P, QQQ, and IWM symbols
    list_spy_symbols = get_spy_holdings()
    list_qqq_symbols = get_qqq_holdings()
    list_iwm_symbols = get_iwm_holdings()  # Add this line

    with Pool(processes=2) as pool:  # Set the number of processes to 4
        # Use imap instead of map for lazy evaluation
        sp500_symbols = list(pool.imap(process_symbol, list_spy_symbols))
        # Add a delay to avoid hitting rate limits
        time.sleep(1)

    # --- Process NASDAQ 100 ---
    nasdaq100_symbols = []
    for symbol in list_qqq_symbols:  # Add tqdm
        nasdaq100_symbols.append({'Symbol': symbol, 'IS_QQQ': 'Y'})

    # --- Process Russell 2000 ---
    russell2000_symbols = []
    for symbol in list_iwm_symbols:
        russell2000_symbols.append({'Symbol': symbol, 'IS_IWM': 'Y'})

    # --- Combine the data ---
    combined_symbols = []
    for sp500_symbol in sp500_symbols:
        # Check if this S&P 500 symbol is also in NASDAQ 100 or Russell 2000
        if any(s['Symbol'] == sp500_symbol['Symbol'] for s in nasdaq100_symbols):
            # If it is in NASDAQ 100, update the existing entry with IS_QQQ = 'Y'
            sp500_symbol['IS_QQQ'] = 'Y'
        else:
            sp500_symbol['IS_QQQ'] = 'N'
        if any(s['Symbol'] == sp500_symbol['Symbol'] for s in russell2000_symbols):
            # If it is in Russell 2000, update the existing entry with IS_IWM = 'Y'
            sp500_symbol['IS_IWM'] = 'Y'
        else:
            sp500_symbol['IS_IWM'] = 'N'
        combined_symbols.append(sp500_symbol)

    for nasdaq100_symbol in nasdaq100_symbols:
        # Check if this NASDAQ 100 symbol is NOT in S&P 500
        if not any(s['Symbol'] == nasdaq100_symbol['Symbol'] for s in combined_symbols):
            # If not, add it with IS_SPY = 'N' and sector info as NULL
            nasdaq100_symbol['IS_SPY'] = 'N'
            nasdaq100_symbol['SPY_SECTOR_ETF'] = None
            nasdaq100_symbol['IS_IWM'] = 'N'  # Add IS_IWM as 'N'
            combined_symbols.append(nasdaq100_symbol)

    for russell2000_symbol in russell2000_symbols:
        # Check if this Russell 2000 symbol is NOT in S&P 500 or NASDAQ 100
        if not any(s['Symbol'] == russell2000_symbol['Symbol'] for s in combined_symbols):
            # If not, add it with IS_SPY = 'N', IS_QQQ = 'N' and sector info as NULL
            russell2000_symbol['IS_SPY'] = 'N'
            russell2000_symbol['IS_QQQ'] = 'N'
            russell2000_symbol['SPY_SECTOR_ETF'] = None
            combined_symbols.append(russell2000_symbol)

    # --- Add custom symbols ---
    custom_symbols = get_custom_symbols()
    for symbol in custom_symbols:
        if not any(s['Symbol'] == symbol for s in combined_symbols):
            combined_symbols.append({
                'Symbol': symbol,
                'IS_SPY': 'N',
                'IS_QQQ': 'N',
                'IS_IWM': 'N',
                'SPY_SECTOR_ETF': None
            })

    # Write combined data to CSV
    with open('symbols\\list_us_stocks.csv', 'w', newline='') as csvfile:
        fieldnames = ['Symbol', 'IS_SPY', 'IS_QQQ', 'IS_IWM',
                      'SPY_SECTOR_ETF']  # Add IS_IWM to fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for symbol_data in combined_symbols:
            writer.writerow(symbol_data)

    # send an message to the log channel
    webhook_discord_master = Discord(url=dict_dc_webhook_trading_signal['master_log'])
    # report the duration
    duration = round(time.time() - start_time, 1)
    num_symbols = len(combined_symbols)
    content = f"Gap Scanner | 运行时间 | {duration}秒\n"
    content += f"Symbols processed: {num_symbols}"
    webhook_discord_master.post(
        content=content
    )

    # Print the time taken
    print(f"Time taken: {time.time() - start_time} seconds")