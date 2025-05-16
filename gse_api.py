import requests
import pandas as pd
from datetime import datetime, time
import os
import time as time_module  # Avoid conflict with 'time' object


# Function to fetch GSE data and return it in a structured format
def fetch_gse_data():
    url = "https://dev.kwayisi.org/apis/gse/equities"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        records = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for stock in data:
            records.append({
                "Date Time": current_time,
                "Stock Name": stock['name'],
                "Price": stock['price']
            })

        # Print the fetched data to the console
        print("Fetched Data:")
        for record in records:
            print(f"Date Time: {record['Date Time']}, Stock Name: {record['Stock Name']}, Price: {record['Price']}")

        return records
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
        return []


# Function to append data to an existing Excel file or create a new one
def append_to_excel(file_name):
    records = fetch_gse_data()
    if records:
        df = pd.DataFrame(records)

        # Check if file exists
        if not os.path.exists(file_name):
            # Create new Excel file if it doesn't exist
            df.to_excel(file_name, index=False)
            print(f"New report created and saved as {file_name}")
        else:
            # Append data to existing Excel file
            with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
            print(f"Data appended to existing report: {file_name}")


# Function to check if the current time is after market close
def is_end_of_trading_day():
    current_time = datetime.now().time()
    market_close_time = time(15, 0)  # GSE closes at 3:00 PM
    return current_time >= market_close_time


# Function to execute only at end of the trading day
def run_at_end_of_trade_day():
    while True:
        if is_end_of_trading_day():
            print("End of trading day detected. Running report generation...")
            append_to_excel("GSE_Stock_Report.xlsx")
            break
        else:
            print("Trading day is not over yet. Waiting until the end of trading day...")
            time_module.sleep(3600)  # Sleep for 1 hour and then check again


# Run the report generation
if __name__ == "__main__":
    run_at_end_of_trade_day()
