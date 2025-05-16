import pandas as pd
import requests

API_ENDPOINT = "http://192.168.100.131:8080/ords/ultra_inv/ad_ons_api/send_coupons"

def send_data_to_endpoint(excel_file):
    # Load the Excel file
    try:
        report_df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"File '{excel_file}' not found.")
        return

    # Debug: Print the column names to verify the presence of 'ID' column
    print("Columns in the Excel file:", report_df.columns)

    # Convert all columns with datetime types to string, handling NaT
    for col in report_df.columns:
        if pd.api.types.is_datetime64_any_dtype(report_df[col]):
            report_df[col] = report_df[col].apply(
                lambda x: f"Paid on {x.strftime('%m/%d/%Y')}" if pd.notna(x) else "Paid on N/A"
            )

    # Iterate through rows and convert each row to JSON-friendly format
    for index, row in report_df.iterrows():
        # Extract the 'ID' from the row
        row_id = row.get('ID', None)  # None will be used if 'ID' is missing

        # Extract other required data
        scheme_id = row['Scheme_ID']
        fund_manager = row['fund_manager_who_adviced']
        investment_id = row['Investment_ID']
        instrument = row['Instrument']
        last_coupon_date = row['Last_Coupon_Date']
        coupon_amount = row['coupon_amount']
        interest = row['interest']
        maturity_date = row['Maturity_Date']
        maturity_value = row['Maturity_Value']
        total_value = row['Total_Value']

        # Debug: Print if 'Investment_ID' is missing
        if pd.isna(investment_id):
            print(f"Investment_ID is empty for row {index + 1}")

        # Create the data dictionary for the row, with 'ID' from the Excel file
        data = {
            "ID": row_id,  # Take 'ID' from the Excel file
            "Scheme_ID": scheme_id,
            "fund_manager_who_adviced": fund_manager,
            "Investment_ID": investment_id,
            "Instrument": instrument,
            "Last_Coupon_Date": last_coupon_date,
            "coupon_amount": coupon_amount,
            "interest": interest,
            "Maturity_Date": maturity_date,
            "Maturity_Value": maturity_value,
            "Total_Value": total_value
        }

        # Debug: Print the data dictionary
        print(f"Row {index + 1} data to be sent:", data)

        # Handle NaN, NaT, or infinite values by converting to None
        for key, value in data.items():
            if pd.isna(value) or value == float('inf') or value == float('-inf'):
                data[key] = None

        try:
            # Send POST request to the API endpoint with the row data
            response = requests.post(API_ENDPOINT, json=data)

            # Check the response status
            if response.status_code == 200:
                print(f"Row {index + 1} sent successfully.")
            else:
                print(f"Failed to send row {index + 1}. Status code: {response.status_code}")
                print("Response:", response.text)
        except requests.RequestException as e:
            print(f"An error occurred while sending row {index + 1}: {e}")

# Example usage
excel_report_file = "Oct-2024 Coupons & Maturity.xlsx"  # Replace with the actual file name
send_data_to_endpoint(excel_report_file)
