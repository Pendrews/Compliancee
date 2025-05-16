import pandas as pd

# Load the input Excel file
input_file = "Nominated.xlsx"
df = pd.read_excel(input_file)

# Group the data by ACCOUNTNO
grouped = df.groupby(['ACCOUNTNO', 'CONTRIBUTOR_ID_TYPE', 'CONTRIBUTOR_ID_NO'])

# Initialize an empty list to store the transformed rows
transformed_data = []

# Iterate over each group (each account)
for (account_no, contributor_id_type_1, contributor_id_no_2), group in grouped:
    # Initialize a dictionary to store the row data
    row_data = {
        'ACCOUNTNO': account_no,
        'CONTRIBUTOR_ID_TYPE': contributor_id_type_1,
        'CONTRIBUTOR_ID_NO': contributor_id_no_2
    }

    # Iterate over each beneficiary in the group
    for i, (_, beneficiary) in enumerate(group.iterrows(), start=1):
        row_data[f'BENEFICIARY_NAME_{i}'] = beneficiary['BENEFICIARY_NAME']
        row_data[f'BENEFICIARY_ID_{i}'] = beneficiary.get('BENEFICIARY_ID', '')  # Assuming BENEFICIARY_ID exists
        row_data[f'BENEFICIARY_GHANA_CARD_NO_{i}'] = beneficiary['BENEFICIARY_GHANA_CARD_NO']
        row_data[f'BENEFICIARY_PHONE_{i}'] = beneficiary['BENEFICIARY_PHONE']
        row_data[f'ALLOCATION_{i}'] = beneficiary['ALLOCATION']

    # Append the row to the transformed data
    transformed_data.append(row_data)

# Convert the transformed data into a DataFrame
transformed_df = pd.DataFrame(transformed_data)

# Save the transformed data to a new Excel file
output_file = "Transformed_Nominated.xlsx"
transformed_df.to_excel(output_file, index=False)

print(f"Transformation complete. Output saved to {output_file}")
