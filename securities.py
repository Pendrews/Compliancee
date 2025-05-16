import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_generic_table(url):
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the first table on the page
    table = soup.find('table')
    if not table:
        print("No table found on the page.")
        return

    rows = table.find_all('tr')
    data = []

    # Extract header
    headers = [header.text.strip() for header in rows[0].find_all('th')]

    # Extract each row's data
    for row in rows[1:]:  # Skip the header row
        columns = row.find_all('td')
        row_data = [col.text.strip() for col in columns]

        # Append the row data as a dictionary
        if row_data:  # Only add non-empty rows
            data.append(dict(zip(headers, row_data)))

    return data

# Run the scraper on the new link and print results
url = 'https://www.doobia.com/mutual-funds/gh/top-ranked/money-market-funds'
csd_data = scrape_generic_table(url)
if csd_data:
    # Create a DataFrame from the scraped data
    df = pd.DataFrame(csd_data)

    # Save the DataFrame to an Excel file
    output_file = 'gog_notes_and_bonds_data.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")
else:
    print("No data was scraped.")
