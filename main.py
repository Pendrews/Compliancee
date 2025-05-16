import pandas as pd
from datetime import datetime, timedelta
import calendar
import get_data

# get_data

# Read the existing Excel file
input_file = 'Comp_inv.xlsx'
output_file = 'new_comp_inv.xlsx'

# Load the data into a DataFrame
df = pd.read_excel(input_file)

# Create a new DataFrame with the required columns
new_columns = {
    'ID': 'ID',
    'Scheme_ID': 'SCHEME_ID',
    'Instrument': 'INSTRUMENT',
    'Asset Classification': 'ASSET_CLASSIFICATION',
    'Issuer Name': 'ISSUER_NAME',
    'Fund Manager who adviced': 'FUND_MANAGER_WHO_ADVICED',
    'Investment ID': 'INVESTMENT_ID',
    'Asset Tenor': 'ASSET_TENOR',
    'Issue Date': 'ISSUE_DATE',
    'Asset Class': 'ASSET_CLASS',
    'Asset Allocation per percent': None,  # Add logic if needed
    'Date of Investment': 'DATE_OF_INVESTMENT',
    'Maturity Date': None,  # Add logic to calculate maturity date
    'Reporting Date': None,  # Set to current date
    'Remaining Days to Maturity': None,  # Add logic to calculate
    'Days Run': None,  # Add logic to calculate
    'Interest Rate percent': 'INTEREST_RATE_PERCENT',
    'Discount Rate percent': 'DISCOUNT_RATE_PERCENT',
    'Coupon Rate percent': 'COUPON_RATE_PERCENT',
    'Currency Conversion Rate': 'CURRENCY_CONVERSION_RATE',
    'Currency': 'CURRENCY',
    'Face Value': 'FACE_VALUE',
    'Amount Invested (GHS)': 'AMOUNT_INVESTED',
    'Amount Invested in Foreign Currency (Eurobond / External Investment)': 'AMOUNT_INVESTED__IN_FOREIGN_CURRENCY',
    'Investment Charge Rate percent': None,  # Add logic if needed
    'Investment Charge Amount (GHS)': 'INVESTMENT_CHARGE_AMOUNT',
    'Coupon Paid': None,  # Add logic if needed
    'Accrued Interest since purchase or last payment(for Bonds)': None,  # Add logic if needed
    'Accrued Interest / Coupon for the month': None,  # Add logic if needed
    'Outstanding Interest to Maturity': None,  # Add logic if needed
    'Accumulated Impairement Provision': None,  # Add logic if needed
    'Price Per Unit / Share at Purchase': 'PRICE_PER_UNIT_SHARE_AT_PURCHASE',
    'Price Per Unit / Share at Value Date': 'PRICE_PER_UNIT_SHARE_AT_VALUE_DATE',
    'Price Per Unit / Share at Last Reporting Date': None,  # Add logic if needed
    'Capital Gains': None,  # Add logic if needed
    'Dividend Received': None,  # Add logic if needed
    'Number of Units / Shares': 'NUMBER_OF_UNITS_OR_SHARES',
    'Market Value (GHS)': None,  # Add logic if needed
    'Holding Period Return Per an Investment (percent)': None,  # Add logic if needed
    'Holding Period Return Per an Investment Weighted percent': None,  # Add logic if needed
    'Disposal Proceeds (GHS)': None,  # Add logic if needed
    'Disposal Instructions': None,  # Add logic if needed
    'Yield on Disposal (GHS)': None,  # Add logic if needed
}

# Initialize the new DataFrame
new_df = pd.DataFrame()

# Populate new DataFrame with values from old DataFrame
for new_col, old_col in new_columns.items():
    if old_col in df.columns:
        new_df[new_col] = df[old_col]
    else:
        new_df[new_col] = None  # Or handle as needed, e.g., set to NaN


# Add logic to calculate fields like Maturity Date, Remaining Days, etc.

# Example logic for Maturity Date based on Asset Tenor (if it's in years)
def calculate_maturity_date(row):
    if (pd.notna(row['Issue Date']) and
            pd.notna(row['Asset Tenor']) and
            row['Asset Classification'] == 'Amortised Cost' and
            row['Asset Class'] != 'Bank Balance'):
        return row['Issue Date'] + pd.DateOffset(days=int(row['Asset Tenor']))
    return None


# Apply the function to calculate Maturity Date
new_df['Maturity Date'] = new_df.apply(calculate_maturity_date, axis=1)

# Convert date-related columns to datetime and format to MM/DD/YYYY
date_columns = ['Maturity Date', 'Date of Investment', 'Issue Date', 'Reporting Date']

for col in date_columns:
    new_df[col] = pd.to_datetime(new_df[col], errors='coerce').dt.strftime('%m/%d/%Y')


def calculate_reporting_date(row):
    reporting_date_field = 'Reporting Date'

    # Get today's date
    today = datetime.today()
    # Find the first day of the current month
    first_day_of_current_month = today.replace(day=1)
    # Subtract one day to get the last day of the previous month
    last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)

    # Format the date as MM/DD/YYYY
    formatted_reporting_date = last_day_of_previous_month.strftime('%m/%d/%Y')

    # Assign this value to the Reporting Date field in the row
    row[reporting_date_field] = formatted_reporting_date
    return row[reporting_date_field]


new_df['Reporting Date'] = new_df.apply(calculate_reporting_date, axis=1)


def calculate_days_run(row):
    date_of_investment_field = 'Date of Investment'
    reporting_date_field = 'Reporting Date'

    # Ensure 'Reporting Date' is in datetime format
    reporting_date = pd.to_datetime(row[reporting_date_field], format='%m/%d/%Y', errors='coerce')
    # Ensure 'Date of Investment' is in datetime format
    date_of_investment = pd.to_datetime(row[date_of_investment_field], errors='coerce')

    if pd.notna(reporting_date) and pd.notna(date_of_investment):
        # Calculate the difference in days
        days_run = (reporting_date - date_of_investment).days
        return days_run
    return None  # Return None if either date is missing


# Apply the function to calculate Days Run
new_df['Reporting Date'] = new_df.apply(calculate_reporting_date, axis=1)  # Calculate Reporting Date
new_df['Days Run'] = new_df.apply(calculate_days_run, axis=1)


def calculate_remaining_days_to_maturity(row):
    asset_tenor_field = 'Asset Tenor'
    days_run_field = 'Days Run'
    if pd.notna(row[asset_tenor_field]) and pd.notna(row[days_run_field]):
        asset_tenor = int(row[asset_tenor_field])
        days_run = int(row[days_run_field])
        remaining_days_to_maturity = asset_tenor - days_run
        return remaining_days_to_maturity if remaining_days_to_maturity >= 0 else 0
    return None


# Apply the calculations to the DataFrame
new_df['Reporting Date'] = new_df.apply(calculate_reporting_date, axis=1)  # Calculate Reporting Date
new_df['Days Run'] = new_df.apply(calculate_days_run, axis=1)  # Calculate Days Run
new_df['Remaining Days to Maturity'] = new_df.apply(calculate_remaining_days_to_maturity, axis=1)  # Calculate Remaining Days to Maturity


def calculate_investment_charge_rate(row):
    investment_charge_amount_field = 'Investment Charge Amount (GHS)'
    amount_invested_field = 'Amount Invested (GHS)'

    # Ensure both 'Investment Charge Amount (GHS)' and 'Amount Invested (GHS)' are available and valid
    if pd.notna(row[investment_charge_amount_field]) and pd.notna(row[amount_invested_field]) and row[amount_invested_field] != 0:
        # Calculate the investment charge rate percent
        investment_charge_rate = (row[investment_charge_amount_field] / row[amount_invested_field]) * 100
        # Return the value formatted as a string with '%' sign and rounded to 2 decimal places
        return f"{investment_charge_rate:.5f}%"
    return None  # Return None if any value is missing or invalid


# Apply the Investment Charge Rate calculation
new_df['Investment Charge Rate percent'] = new_df.apply(calculate_investment_charge_rate, axis=1)


def calculate_coupon_paid(row):
    instrument_field = 'Instrument'
    issue_date_field = 'Issue Date'
    maturity_date_field = 'Maturity Date'
    investment_date_field = 'Date of Investment'
    reporting_date_field = 'Reporting Date'
    coupon_rate_field = 'Coupon Rate percent'
    face_value_field = 'Face Value'
    currency_conversion_rate_field = 'Currency Conversion Rate'

    # Ensure the instrument is a bond (check if 'BONDS' or 'NOTES' is in the instrument field)
    if 'BONDS' in str(row[instrument_field]).upper() or 'NOTES' in str(row[instrument_field]).upper():
        # Convert relevant fields to datetime format
        issue_date = pd.to_datetime(row[issue_date_field], errors='coerce')
        maturity_date = pd.to_datetime(row[maturity_date_field], format='%m/%d/%Y', errors='coerce')
        investment_date = pd.to_datetime(row[investment_date_field], errors='coerce')
        reporting_date = pd.to_datetime(row[reporting_date_field], format='%m/%d/%Y', errors='coerce')

        if pd.notna(issue_date) and pd.notna(maturity_date) and pd.notna(investment_date) and pd.notna(reporting_date):
            # Step 1: Calculate the total number of coupons between Issue Date and Maturity Date (bi-annual = 2 coupons/year)
            total_coupons = ((maturity_date.year - issue_date.year) * 2) + ((maturity_date.month - issue_date.month) // 6)

            # Step 2: Calculate the number of coupons between Investment Date and Reporting Date (bi-annual)
            eligible_coupons = ((reporting_date.year - investment_date.year) * 2) + ((reporting_date.month - investment_date.month) // 6)

            # Make sure that eligible_coupons doesn't exceed total_coupons
            eligible_coupons = min(eligible_coupons, total_coupons)

            # Step 3: Calculate the daily interest
            if pd.notna(row[coupon_rate_field]) and pd.notna(row[face_value_field]):
                coupon_rate_percent = row[coupon_rate_field] / 100
                face_value = row[face_value_field]
                daily_interest = (coupon_rate_percent * face_value) / 364

                # Step 4: Calculate the value of each coupon (182 days for bi-annual payment)
                coupon_value = daily_interest * 182

                # Step 5: Calculate the total coupon paid
                total_coupon_paid = coupon_value * eligible_coupons

                # Step 6: Check if Currency Conversion Rate is available and apply it
                if pd.notna(row[currency_conversion_rate_field]):
                    currency_conversion_rate = row[currency_conversion_rate_field]
                    total_coupon_paid *= currency_conversion_rate  # Multiply by the conversion rate

                return total_coupon_paid

    return None  # Return None if it's not a bond or data is missing


new_df['Coupon Paid'] = new_df.apply(calculate_coupon_paid, axis=1)


def calculate_accrued_interest(row):
    global days_in_year, days_since_last_payment
    instrument_field = 'Instrument'
    issue_date_field = 'Issue Date'
    investment_date_field = 'Date of Investment'
    reporting_date_field = 'Reporting Date'
    coupon_rate_field = 'Coupon Rate percent'
    face_value_field = 'Face Value'
    accrued_interest_field = 'Accrued Interest since purchase or last payment(for Bonds)'
    currency_conversion_rate_field = 'Currency Conversion Rate'

    # Convert relevant fields to string and uppercase for comparison
    instrument = str(row[instrument_field]).upper()

    # Convert relevant fields to datetime format
    issue_date = pd.to_datetime(row[issue_date_field], errors='coerce')
    investment_date = pd.to_datetime(row[investment_date_field], errors='coerce')
    reporting_date = pd.to_datetime(row[reporting_date_field], format='%m/%d/%Y', errors='coerce')

    # Ensure the instrument is a bond, bill, or bank security and dates are available
    if pd.notna(issue_date) and pd.notna(investment_date) and pd.notna(reporting_date):

        # Step 1: Check if the instrument is "BONDS"
        if 'BONDS' in instrument:
            # Calculate the total number of days the security has run from the issue date to the reporting date
            days_run = (reporting_date - issue_date).days

            # MOD operation to find days since the last coupon payment (semi-annual payments = 182 days)
            days_since_last_coupon = days_run % 182

            # Subtract this from the reporting date to get the exact last coupon payment date
            last_coupon_payment_date = reporting_date - pd.Timedelta(days=days_since_last_coupon)

            # If the last coupon payment date is greater than the reporting date, use the issue date
            if last_coupon_payment_date > reporting_date:
                last_coupon_payment_date = issue_date

            # Calculate the number of days since the last coupon payment
            days_since_last_payment = (reporting_date - last_coupon_payment_date).days

            # Use 364 days for "BONDS"
            days_in_year = 364
            print(f'Last Coupon Payment Date: {last_coupon_payment_date}')

        # Step 2: Check if the instrument is "BILLS"
        elif 'BILLS' in instrument:
            # Accrued interest is calculated from the issue date
            days_since_last_payment = (reporting_date - issue_date).days

            # Use 364 days for "BILLS"
            days_in_year = 364

        # Step 3: Check if the instrument is "BANK SECURITIES"
        elif 'BANK SECURITIES' in instrument:
            # Accrued interest is calculated from the issue date
            days_since_last_payment = (reporting_date - issue_date).days

            # Use 365 days for "BANK SECURITIES"
            days_in_year = 365

        # Step 4: Calculate the daily interest and accrued interest
        if pd.notna(row[coupon_rate_field]) and pd.notna(row[face_value_field]):
            coupon_rate_percent = row[coupon_rate_field] / 100
            face_value = row[face_value_field]

            # Calculate the daily interest based on the instrument type
            daily_interest = (coupon_rate_percent * face_value) / days_in_year

            # Calculate the accrued interest
            accrued_interest = daily_interest * days_since_last_payment

            # Step 5: Check if Currency Conversion Rate is available and apply it
            if pd.notna(row[currency_conversion_rate_field]):
                currency_conversion_rate = row[currency_conversion_rate_field]
                accrued_interest *= currency_conversion_rate  # Multiply by the conversion rate

            # Update the existing field with the calculated accrued interest
            row[accrued_interest_field] = accrued_interest
            print(f'Accrued Interest: {accrued_interest} for {days_since_last_payment} days with daily interest of {daily_interest}')

            return row[accrued_interest_field]

    return None  # Return None if data is missing or invalid



new_df['Accrued Interest since purchase or last payment(for Bonds)'] = new_df.apply(calculate_accrued_interest, axis=1)


def calculate_accrued_interest_for_month(row):
    instrument_field = 'Instrument'
    investment_date_field = 'Date of Investment'
    reporting_date_field = 'Reporting Date'
    coupon_rate_field = 'Coupon Rate percent'
    face_value_field = 'Face Value'
    accrued_interest_field = 'Accrued Interest / Coupon for the month'
    currency_conversion_rate_field = 'Currency Conversion Rate'

    # Ensure the instrument is a bond (check if 'BONDS', 'BILLS', 'NOTES', or 'BANK SECURITIES' is in the instrument field)
    if 'BONDS' in str(row[instrument_field]).upper() or 'BILLS' in str(row[instrument_field]).upper() or 'NOTES' in str(row[instrument_field]).upper() or 'BANK SECURITIES' in str(row[instrument_field]).upper():
        # Convert relevant fields to datetime format
        investment_date = pd.to_datetime(row[investment_date_field], errors='coerce')
        reporting_date = pd.to_datetime(row[reporting_date_field], format='%m/%d/%Y', errors='coerce')

        if pd.notna(investment_date) and pd.notna(reporting_date):
            # Step 1: Calculate the daily interest
            if pd.notna(row[coupon_rate_field]) and pd.notna(row[face_value_field]):
                coupon_rate_percent = row[coupon_rate_field] / 100
                face_value = row[face_value_field]
                daily_interest = (coupon_rate_percent * face_value) / 364

                # Step 2: Determine the number of days in the Reporting Month
                if investment_date.month == reporting_date.month and investment_date.year == reporting_date.year:
                    # If Investment Date is within the same month as the Reporting Date, calculate days between Investment Date and Reporting Date
                    days_to_calculate = (reporting_date - investment_date).days + 1  # Including the day of investment
                    print(f"Investment and Reporting date are in the same month. Days to calculate: {days_to_calculate}")
                else:
                    # Calculate the number of days in the Reporting Month
                    days_in_reporting_month = calendar.monthrange(reporting_date.year, reporting_date.month)[1]
                    days_to_calculate = days_in_reporting_month
                    print(f"Investment and Reporting date are in different months. Days in the reporting month: {days_to_calculate}")

                # Step 3: Calculate the accrued interest for the reporting month
                accrued_interest_for_month = daily_interest * days_to_calculate
                print(f"Daily interest: {daily_interest}, Accrued interest for the reporting month: {accrued_interest_for_month}")

                # Step 4: Check if Currency Conversion Rate is available and apply it
                if pd.notna(row[currency_conversion_rate_field]):
                    currency_conversion_rate = row[currency_conversion_rate_field]
                    accrued_interest_for_month *= currency_conversion_rate  # Multiply by the conversion rate

                # Update the existing field with the calculated accrued interest for the reporting month
                row[accrued_interest_field] = accrued_interest_for_month

                return row[accrued_interest_field]

    return None  # Return None if it's not a bond or data is missing


def calculate_accrued_interest_for_month(row):
    instrument_field = 'Instrument'
    investment_date_field = 'Date of Investment'
    reporting_date_field = 'Reporting Date'
    coupon_rate_field = 'Coupon Rate percent'
    face_value_field = 'Face Value'
    accrued_interest_field = 'Accrued Interest / Coupon for the month'
    currency_conversion_rate_field = 'Currency Conversion Rate'

    # Ensure the instrument is a bond (check if 'BONDS', 'BILLS', 'NOTES', or 'BANK SECURITIES' is in the instrument field)
    if 'BONDS' in str(row[instrument_field]).upper() or 'BILLS' in str(row[instrument_field]).upper() or 'NOTES' in str(row[instrument_field]).upper() or 'BANK SECURITIES' in str(row[instrument_field]).upper():
        # Convert relevant fields to datetime format
        investment_date = pd.to_datetime(row[investment_date_field], errors='coerce')
        reporting_date = pd.to_datetime(row[reporting_date_field], format='%m/%d/%Y', errors='coerce')

        if pd.notna(investment_date) and pd.notna(reporting_date):
            # Step 1: Calculate the daily interest
            if pd.notna(row[coupon_rate_field]) and pd.notna(row[face_value_field]):
                coupon_rate_percent = row[coupon_rate_field] / 100
                face_value = row[face_value_field]
                daily_interest = (coupon_rate_percent * face_value) / 364

                # Step 2: Determine the number of days in the Reporting Month
                if investment_date.month == reporting_date.month and investment_date.year == reporting_date.year:
                    # If Investment Date is within the same month as the Reporting Date, calculate days between Investment Date and Reporting Date
                    days_to_calculate = (reporting_date - investment_date).days + 1  # Including the day of investment
                    print(f"Investment and Reporting date are in the same month. Days to calculate: {days_to_calculate}")
                else:
                    # Calculate the number of days in the Reporting Month
                    days_in_reporting_month = calendar.monthrange(reporting_date.year, reporting_date.month)[1]
                    days_to_calculate = days_in_reporting_month
                    print(f"Investment and Reporting date are in different months. Days in the reporting month: {days_to_calculate}")

                # Step 3: Calculate the accrued interest for the reporting month
                accrued_interest_for_month = daily_interest * days_to_calculate
                print(f"Daily interest: {daily_interest}, Accrued interest for the reporting month: {accrued_interest_for_month}")

                # Step 4: Check if Currency Conversion Rate is available and apply it
                if pd.notna(row[currency_conversion_rate_field]):
                    currency_conversion_rate = row[currency_conversion_rate_field]
                    accrued_interest_for_month *= currency_conversion_rate  # Multiply by the conversion rate

                # Update the existing field with the calculated accrued interest for the reporting month
                row[accrued_interest_field] = accrued_interest_for_month

                return row[accrued_interest_field]

    return None  # Return None if it's not a bond or data is missing


new_df['Accrued Interest / Coupon for the month'] = new_df.apply(calculate_accrued_interest_for_month, axis=1)

# Apply the Accrued Interest calculation


def calculate_outstanding_interest_to_maturity(row):
    instrument_field = 'Instrument'
    maturity_date_field = 'Maturity Date'
    reporting_date_field = 'Reporting Date'
    coupon_rate_field = 'Coupon Rate percent'
    face_value_field = 'Face Value'
    outstanding_interest_field = 'Outstanding Interest to Maturity'
    currency_conversion_rate_field = 'Currency Conversion Rate'

    # Ensure the instrument is either a bond, bill, or bank security
    if 'BONDS' in str(row[instrument_field]).upper() or 'BILLS' in str(row[instrument_field]).upper() or 'BANK SECURITIES' in str(row[instrument_field]).upper():
        # Convert relevant fields to datetime format
        maturity_date = pd.to_datetime(row[maturity_date_field], format='%m/%d/%Y', errors='coerce')
        reporting_date = pd.to_datetime(row[reporting_date_field], format='%m/%d/%Y', errors='coerce')

        if pd.notna(maturity_date) and pd.notna(reporting_date):
            # Step 1: Calculate the number of days remaining to maturity
            days_remaining_to_maturity = (maturity_date - reporting_date).days
            print(f"Days remaining to maturity: {days_remaining_to_maturity}")

            # Step 2: Calculate the daily interest
            if pd.notna(row[coupon_rate_field]) and pd.notna(row[face_value_field]):
                coupon_rate_percent = row[coupon_rate_field] / 100
                face_value = row[face_value_field]
                daily_interest = (coupon_rate_percent * face_value) / 364
                print(f"Daily interest: {daily_interest}")

                # Step 3: Calculate the outstanding interest to maturity
                outstanding_interest = daily_interest * days_remaining_to_maturity
                print(f"Outstanding interest to maturity: {outstanding_interest}")

                # Step 4: Check if Currency Conversion Rate is available and apply it
                if pd.notna(row[currency_conversion_rate_field]):
                    currency_conversion_rate = row[currency_conversion_rate_field]
                    outstanding_interest *= currency_conversion_rate  # Multiply by the conversion rate

                # Update the existing field with the calculated outstanding interest
                row[outstanding_interest_field] = outstanding_interest

                return row[outstanding_interest_field]

    return None  # Return None if it's not a bond, bill, or bank security, or if data is missing


def calculate_capital_gains(row):
    asset_classification_field = 'Asset Classification'
    price_per_unit_value_field = 'Price Per Unit / Share at Value Date'
    num_units_shares_field = 'Number of Units / Shares'
    amount_invested_field = 'Amount Invested (GHS)'
    capital_gains_field = 'Capital Gains'

    # Check if Asset Classification is not "Amortised Cost"
    if row[asset_classification_field] != "Amortised Cost":
        # Ensure relevant fields are available for calculation
        if pd.notna(row[price_per_unit_value_field]) and pd.notna(row[num_units_shares_field]) and pd.notna(row[amount_invested_field]):
            # Step 1: Calculate the capital gains
            capital_gains = (row[price_per_unit_value_field] * row[num_units_shares_field]) - row[amount_invested_field]
            print(f"Capital Gains: {capital_gains}")

            # Step 2: Update the field with calculated capital gains
            row[capital_gains_field] = capital_gains

            return row[capital_gains_field]

    return None  # Return None if Asset Classification is "Amortised Cost" or if data is missing


# Apply the Capital Gains calculation
new_df['Capital Gains'] = new_df.apply(calculate_capital_gains, axis=1)

# Apply the Outstanding Interest to Maturity calculation
new_df['Outstanding Interest to Maturity'] = new_df.apply(calculate_outstanding_interest_to_maturity, axis=1)


def calculate_market_value(row):
    instrument_field = 'Instrument'
    investment_id_field = 'Investment ID'
    asset_class_field = 'Asset Class'
    asset_classification_field = 'Asset Classification'
    face_value_field = 'Face Value'
    accrued_interest_field = 'Accrued Interest since purchase or last payment(for Bonds)'
    capital_gains_field = 'Capital Gains'
    market_value_field = 'Market Value (GHS)'
    currency_conversion_rate_field = 'Currency Conversion Rate'

    # Substring patterns to look for in Investment ID
    specific_investment_id_patterns = [
        "GOG-BD-16/02/27-A6308-1838-10",
        "GOG-BD-15/02/28-A6309-1838-10"
    ]

    # Convert instrument and asset class to uppercase for comparison
    instrument = str(row[instrument_field]).upper()
    asset_class = str(row[asset_class_field]).upper()
    investment_id = str(row[investment_id_field]).upper()

    # Default the currency conversion rate to 1 if not available
    currency_conversion_rate = row.get(currency_conversion_rate_field, 1) or 1

    # Step 1: Check if the Investment ID contains "Receivable"
    if "RECEIVABLE" in investment_id:
        if pd.notna(row[face_value_field]):
            market_value = row[face_value_field] * currency_conversion_rate
            print(f"Market Value (Receivable Investment ID): {market_value}")
            row[market_value_field] = market_value
            return row[market_value_field]

    # Step 2: Check if the Asset Class contains "BANK BALANCE"
    if "BANK BALANCE" in asset_class:
        if pd.notna(row[face_value_field]):
            market_value = row[face_value_field] * currency_conversion_rate
            print(f"Market Value (BANK BALANCE): {market_value}")
            row[market_value_field] = market_value
            return row[market_value_field]

    # Step 3: Check if the Investment ID contains any of the specific patterns
    if any(pattern in investment_id for pattern in specific_investment_id_patterns):
        # Only use the accrued interest without adding the face value
        if pd.notna(row[accrued_interest_field]):
            market_value = row[accrued_interest_field] * currency_conversion_rate
            print(f"Market Value (special Investment ID): {market_value}")
            row[market_value_field] = market_value
            return row[market_value_field]

    # Step 4: Check if the instrument is "BONDS", "NOTES", "BILLS", or "BANK SECURITIES"
    if 'BONDS' in instrument or 'NOTES' in instrument or 'BILLS' in instrument or 'BANK SECURITIES' in instrument:
        # Ensure the relevant fields are available
        if pd.notna(row[accrued_interest_field]) and pd.notna(row[face_value_field]):
            # Calculate the market value as Face Value + Accrued Interest
            market_value = (row[face_value_field] * currency_conversion_rate) + row[accrued_interest_field]
            print(f"Market Value (BONDS/NOTES/BILLS): {market_value}")
            row[market_value_field] = market_value
            return row[market_value_field]

    # Step 5: If the instrument is not "BONDS", "NOTES", "BILLS" and Asset Classification is not "Amortised Cost"
    elif row[asset_classification_field] != "Amortised Cost":
        # Ensure the relevant fields are available
        if pd.notna(row[capital_gains_field]) and pd.notna(row[face_value_field]):
            # Calculate the market value as Face Value + Capital Gains
            market_value = (row[face_value_field] + row[capital_gains_field]) * currency_conversion_rate
            print(f"Market Value (Non-BONDS/NOTES/BILLS): {market_value}")
            row[market_value_field] = market_value
            return row[market_value_field]

    return None  # Return None if conditions are not met or data is missing


new_df['Accrued Interest since purchase or last payment(for Bonds)'] = new_df.apply(calculate_accrued_interest, axis=1)
# Apply the Market Value (GHS) calculation
new_df['Market Value (GHS)'] = new_df.apply(calculate_market_value, axis=1)


def calculate_asset_allocation_percent(df):
    market_value_field = 'Market Value (GHS)'
    asset_allocation_field = 'Asset Allocation per percent'

    # Step 1: Calculate the total sum of all "Market Value (GHS)" in the DataFrame
    total_market_value = df[market_value_field].sum()
    print(f"Total Market Value (GHS): {total_market_value}")

    # Step 2: Define a function to calculate Asset Allocation per percent for each row
    def calculate_row_asset_allocation(row):
        if pd.notna(row[market_value_field]) and total_market_value > 0:
            # Calculate Asset Allocation per percent
            asset_allocation = (row[market_value_field] / total_market_value) * 100
            print(f"Asset Allocation for row: {asset_allocation}%")
            return asset_allocation
        return None  # Return None if the market value is missing or invalid

    # Step 3: Apply the function to each row
    df[asset_allocation_field] = df.apply(calculate_row_asset_allocation, axis=1)

    return df


new_df = calculate_asset_allocation_percent(new_df)


def calculate_holding_period_return(row):
    market_value_field = 'Market Value (GHS)'  # Pre-calculated field
    amount_invested_field = 'Amount Invested (GHS)'
    hpr_field = 'Holding Period Return Per an Investment (percent)'
    currency_conversion_rate_field = 'Currency Conversion Rate'
    investment_id_field = 'Investment ID'

    # List of specific Investment IDs to skip
    specific_investment_id_patterns = [
        "GOG-BD-16/02/27-A6308-1838-10",
        "GOG-BD-15/02/28-A6309-1838-10"
    ]

    # Convert Investment ID to string and uppercase for comparison
    investment_id = str(row[investment_id_field]).upper()

    # Check if the Investment ID matches any of the specific patterns, if so, leave blank
    if any(pattern in investment_id for pattern in specific_investment_id_patterns):
        row[hpr_field] = None
        return None

    # Ensure relevant fields are available and Amount Invested is not 0
    if pd.notna(row[market_value_field]) and pd.notna(row[amount_invested_field]) and row[amount_invested_field] != 0:
        # Step 1: Apply currency conversion to Amount Invested if a conversion rate is provided
        amount_invested = row[amount_invested_field]
        if pd.notna(row[currency_conversion_rate_field]):
            currency_conversion_rate = row[currency_conversion_rate_field]
            amount_invested *= currency_conversion_rate

        # Step 2: Calculate the holding period return
        holding_period_return = ((row[market_value_field] - amount_invested) / amount_invested) * 100
        print(f"Holding Period Return: {holding_period_return}%")

        # Step 3: Update the field with the calculated holding period return
        row[hpr_field] = holding_period_return

        return row[hpr_field]

    return None  # Return None if conditions are not met



# Ensure Market Value (GHS) is calculated before applying Holding Period Return calculation
new_df['Market Value (GHS)'] = new_df.apply(calculate_market_value, axis=1)

# Apply the Holding Period Return Per an Investment calculation
new_df['Holding Period Return Per an Investment (percent)'] = new_df.apply(calculate_holding_period_return, axis=1)


def calculate_weighted_holding_period_return(row):
    hpr_field = 'Holding Period Return Per an Investment (percent)'
    asset_allocation_field = 'Asset Allocation per percent'
    weighted_hpr_field = 'Holding Period Return Per an Investment Weighted percent'

    # Ensure relevant fields are available
    if pd.notna(row[hpr_field]) and pd.notna(row[asset_allocation_field]):
        # Step 1: Calculate the weighted holding period return
        weighted_hpr = (row[hpr_field] * row[asset_allocation_field]) / 100

        print(f"Weighted Holding Period Return: {weighted_hpr}%")

        # Step 2: Update the field with the calculated weighted holding period return
        row[weighted_hpr_field] = weighted_hpr

        return row[weighted_hpr_field]

    return None  # Return None if data is missing


# Ensure Holding Period Return and Asset Allocation are calculated
new_df['Holding Period Return Per an Investment (percent)'] = new_df.apply(calculate_holding_period_return, axis=1)
new_df = calculate_asset_allocation_percent(new_df)

# Apply the Weighted Holding Period Return Per an Investment calculation
new_df['Holding Period Return Per an Investment Weighted percent'] = new_df.apply(calculate_weighted_holding_period_return, axis=1)


new_df.to_excel(output_file, index=False)

print(f"New Excel file created: {output_file}")

print('preparing coupon, interest and maturity payment schedule')

# import received_payments
# import send_coupons
