import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

def scrape_exchange_rates():
    url = "https://www.bog.gov.gh/treasury-and-the-markets/daily-interbank-fx-rates/"
    response = requests.get(url)

    # Check if request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

    # Parse the page content
    soup = BeautifulSoup(response.content, "html.parser")

    # Attempt to find the first table
    table = soup.find("table")  # Assuming there's only one main table

    # Check if a table was found
    if table is None:
        print("No table found on the page.")
        return None

    # Extract all rows of data, ignoring headers
    rows = []
    for row in table.find_all("tr"):
        row_data = [cell.text.strip() for cell in row.find_all("td")]
        if row_data:  # Only append non-empty rows
            rows.append(row_data)

    # Create a DataFrame without specific column names
    df = pd.DataFrame(rows)
    return df

def save_to_excel(df):
    # Generate a file name based on today's date
    file_name = f"exchange_rates_{datetime.now().strftime('%Y%m%d')}.xlsx"
    df.to_excel(file_name, index=False)
    print(f"Data saved to {file_name}")

def main():
    while True:
        df = scrape_exchange_rates()
        if df is not None:
            save_to_excel(df)
        time.sleep(5)  # Wait for 5 seconds before the next iteration

if __name__ == "__main__":
    main()
