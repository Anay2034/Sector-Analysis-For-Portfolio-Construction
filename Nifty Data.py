import yfinance as yf
import pandas as pd

# Define a dictionary of Nifty sector indices and their ticker symbols
nifty_sectors = {
    "Nifty_50": "^NSEI",
    "Nifty_Bank": "^NSEBANK",
    "Nifty_IT": "^CNXIT",
    "Nifty_FMCG": "^CNXFMCG",
    "Nifty_Pharma": "^CNXPHARMA",
    "Nifty_Auto": "^CNXAUTO",
    "Nifty_Metal": "^CNXMETAL",
    "Nifty_Realty": "^CNXREALTY",
    "Nifty_Energy": "^CNXENERGY",
    "Nifty_Infra": "^CNXINFRA",
    "Nifty_Services_Sector": "^CNXSERVICE",
    "Nifty_Financial_Services": "^CNXFIN",
    "Nifty_Media": "^CNXMEDIA",
    "Nifty_Commodities": "^CNXCOMMOD",
    "Nifty_PSE": "^CNXPSE",
    "Nifty_Private_Bank": "^CNXPRBANK",
    "Nifty_PSU_Bank": "^CNXPSUBANK",
    "Nifty_Consumption": "^CNXCONSUM",
    "Nifty_CPSE": "^CNXCPSE",
    "Nifty_MNC": "^CNXMNC",
    "Nifty_Growth_Sectors_15": "^CNXGRSECT15",
    "Nifty_Midcap_50": "^NSEMDCP50",
    "Nifty_Smallcap_100": "^NSESMCP100"
}

# Define the start and end dates
start_date = "2011-01-01"
end_date = "2021-12-31"

# Initialize an empty DataFrame to hold all the data
all_data = pd.DataFrame()

# Download data for each sector
for sector, ticker in nifty_sectors.items():
    print(f"Downloading data for {sector}...")
    data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")
    
    # Extract only the 'Close' price and rename the column to the sector name
    data = data[['Close']].rename(columns={'Close': sector})
    
    # Merge the data into the all_data DataFrame
    if all_data.empty:
        all_data = data
    else:
        all_data = all_data.join(data, how='outer')

# Save the combined data to a CSV file
file_name = "Nifty_Sectors_2011_2021_monthly.csv"
all_data.to_csv(file_name)
print(f"All data saved to {file_name}")

# Display the first few rows of the combined dataframe
print(all_data.head())
