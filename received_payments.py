import pandas as pd
from datetime import timedelta


def calculate_last_coupon_date():
    # Load the Excel file
    df = pd.read_excel('new_comp_inv.xlsx')

    # Strip any leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Print columns to verify the exact column names
    print("Columns in the Excel file:", df.columns)

    # Filter rows where 'Instrument' contains 'BOND', 'BILL', or 'BANK SECURITIES'
    bond_bill_df = df[
        ((df['Instrument'].str.contains('BOND', case=False, na=False)) |
         (df['Instrument'].str.contains('BILL', case=False, na=False)) |
         (df['Instrument'].str.contains('BANK SECURITIES', case=False, na=False))) &
        ~df['Investment ID'].str.contains('Receivable', case=False, na=False)
    ]

    # Define coupon interval (182 days for semi-annual payments)
    coupon_interval = timedelta(days=182)

    # Prepare a list to store report data
    report_data = []

    # Iterate over each row in filtered DataFrame
    for index, row in bond_bill_df.iterrows():
        date_of_investment = pd.to_datetime(row['Date of Investment'])
        reporting_date = pd.to_datetime(row['Reporting Date'])
        maturity_date = pd.to_datetime(row['Maturity Date'])

        # Calculate the days passed and intervals for bonds
        days_passed = (reporting_date - date_of_investment).days
        intervals_passed = days_passed // 182  # Integer division to find full 182-day periods

        # Calculate the last coupon payment date
        last_coupon_date = date_of_investment + intervals_passed * coupon_interval

        # Calculate coupon amount for bonds
        face_value = row['Face Value']
        coupon_rate = row['Coupon Rate percent'] / 100
        coupon_amount = face_value * coupon_rate / 2  # Semi-annual coupon

        # Apply currency conversion rate
        currency_conversion_rate = row['Currency Conversion Rate']
        coupon_amount_converted = coupon_amount * currency_conversion_rate

        # Initialize total payment and coupon interest paid
        total_payment = 0
        coupon_interest_paid = 0

        # Handle 'BOND' instruments
        if 'BOND' in row['Instrument'].upper():
            if maturity_date < reporting_date:
                # Investment matured; add face value
                total_payment = coupon_amount_converted + face_value
            else:
                # Not matured; only include the coupon payment
                total_payment = coupon_amount_converted
                coupon_interest_paid = coupon_amount_converted

        # Handle 'BILL' and 'BANK SECURITIES' instruments similarly for maturity value
        elif 'BILL' in row['Instrument'].upper() or 'BANK SECURITIES' in row['Instrument'].upper():
            if maturity_date < reporting_date:
                # Calculate maturity value as face value + interest (simple interest)
                interest_rate = row['Coupon Rate percent'] / 100
                bill_interest = face_value * interest_rate * (days_passed / 365)
                maturity_value = face_value + bill_interest
                total_payment = maturity_value
                coupon_interest_paid = bill_interest

        # Include records where a coupon was paid within the reporting month for bonds
        if 'BOND' in row['Instrument'].upper():
            if (last_coupon_date.month == reporting_date.month and last_coupon_date.year == reporting_date.year):
                report_data.append({
                    'ID': index + 1,
                    'Scheme_ID': row['Scheme_ID'],
                    'fund_manager_who_adviced': row['Fund Manager who adviced'],
                    'Investment_ID': row['Investment ID'],
                    'Instrument': row['Instrument'],
                    'Face Value': face_value,  # Adding Face Value column here
                    'Last_Coupon_Date': last_coupon_date.date(),
                    'coupon_amount': round(total_payment, 2),
                    'interest': round(coupon_interest_paid, 2)
                })

        # Include records for matured 'BILL' or 'BANK SECURITIES' within the reporting month
        elif ('BILL' in row['Instrument'].upper() or 'BANK SECURITIES' in row['Instrument'].upper()) and maturity_date < reporting_date:
            if maturity_date.month == reporting_date.month and maturity_date.year == reporting_date.year:
                report_data.append({
                    'ID': row['ID'],
                    'Scheme_ID': row['Scheme_ID'],
                    'fund_manager_who_adviced': row['Fund Manager who adviced'],
                    'Investment_ID': row['Investment ID'],
                    'Instrument': row['Instrument'],
                    'Face Value': face_value,  # Adding Face Value column here
                    'Maturity_Date': maturity_date.date(),
                    'Maturity_Value': round(total_payment, 2),
                    'interest': round(coupon_interest_paid, 2)
                })

    # Create DataFrame for report
    report_df = pd.DataFrame(report_data)

    # Calculate 'Total_Value' column
    if 'coupon_amount' in report_df.columns and 'Maturity_Value' in report_df.columns:
        report_df['Total_Value'] = report_df['coupon_amount'].fillna(0) + report_df['Maturity_Value'].fillna(0)
    else:
        report_df['Total_Value'] = report_df['coupon_amount'].fillna(0)

    # Reorder columns to make 'ID' the first column
    report_df = report_df[['ID', 'Scheme_ID', 'fund_manager_who_adviced', 'Face Value'] +
                          [col for col in report_df.columns if col not in ['ID', 'Scheme_ID', 'fund_manager_who_adviced', 'Face Value']]]

    # Save the report if there are entries with coupons paid or matured bills
    if not report_df.empty:
        report_month_year = reporting_date.strftime("%b-%Y")
        output_filename = f"{report_month_year} Coupons & Maturity.xlsx"
        report_df.to_excel(output_filename, index=False)
        print(f"Report saved as '{output_filename}'")
    else:
        print("No coupons or matured bills to report; no report generated.")

# Example usage
calculate_last_coupon_date()
