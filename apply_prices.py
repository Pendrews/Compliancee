import pandas as pd

# Load the Comp_inv.xlsx file
comp_inv_df = pd.read_excel('Comp_inv.xlsx')

# Load the Stock_CIS_PRICES.xlsx file again (in case you mod2ified the column name)
stock_prices_df = pd.read_excel('Stock_CIS_PRICES.xlsx')

# Assuming the correct column name is found and updated below
correct_column_name = 'Actual_Column_Name_From_Stock_CIS_PRICES' # Update this with the correct name

# Create a dictionary to map 'Description' to its corresponding correct value
price_mapping = dict(zip(stock_prices_df['Description'], stock_prices_df[correct_column_name]))

# Update the 'PRICE_PER_UNIT_SHARE_AT_VALUE_DATE' column in comp_inv_df based on matching 'INVESTMENT_ID' with 'Description'
comp_inv_df['PRICE_PER_UNIT_SHARE_AT_VALUE_DATE'] = comp_inv_df['INVESTMENT_ID'].map(price_mapping).fillna(comp_inv_df['PRICE_PER_UNIT_SHARE_AT_VALUE_DATE'])

# Save the updated DataFrame back to Comp_inv.xlsx
comp_inv_df.to_excel('Comp_inv_updated.xlsx', index=False)

print("Update completed successfully and saved to 'Comp_inv_updated.xlsx'.")
