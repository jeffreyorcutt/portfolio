import pandas as pd
import numpy as np
import pickle


def load_pickles(output_files):
    combined_data = pd.DataFrame()
    for file in output_files:
        data = pickle.load(open(file, 'rb'))
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

def remove_duplicates_by_year(df):
    # Ensure 'adate' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['adate']):
        df['adate'] = pd.to_datetime(df['adate'], errors='coerce')
    
    df = df.copy()
    # Extract year from 'adate'
    df['year'] = df['adate'].dt.year
    
    total_removed = 0
    removed_by_year = {}
    
    # Work on a copy of the DataFrame so that the original is preserved
    df_clean = df.copy()
    
    # Iterate through each unique year (ignoring missing years)
    for year in sorted(df_clean['year'].dropna().unique()):
        # Get records for the current year
        subset = df_clean[df_clean['year'] == year]
        
        # Identify duplicate records (all columns compared; keep the first occurrence)
        duplicates_mask = subset.duplicated(keep='first')
        duplicates_count = duplicates_mask.sum()
        
        removed_by_year[year] = duplicates_count
        total_removed += duplicates_count
        
        # Remove duplicate rows for the current year from the overall DataFrame
        df_clean = df_clean.drop(subset[duplicates_mask].index)
    
    # Print the counts of duplicates removed per year
    print('Duplicates removed per year:')
    for year, count in removed_by_year.items():
        print(f'Year {int(year)}: {count}')
    print(f'Total records removed: {total_removed}')
    print(f'Total records remaining: {len(df_clean)}')
    print(f'Total records in original DataFrame: {len(df)}')
    
    # Optionally, drop the temporary 'year' column if not needed further
    df_clean = df_clean.drop(columns=['year'])
    
    return df_clean

def transform_data(combined_data):
    # Count and remove rows with sex 0
    if 'sex' in combined_data.columns:
        zero_sex_count = (combined_data['sex'] == '0').sum()
        combined_data = combined_data[combined_data['sex'] != '0']
        print(f'Number of rows with sex 0: {zero_sex_count}')
        
    else:
        print("The 'sex' column is not present in the data.")

    # # Average age by injt and sex
    # # Assuming there is an 'age' column and a 'sex' column in the data
    # average_age_by_injt_sex = combined_data.groupby(['injt', 'sex'])['age'].mean()
    # print('Average age by injt and sex:')
    # print(average_age_by_injt_sex)

    combined_data = remove_duplicates_by_year(combined_data)
    

    # Overall injured count
    overall_injured_count = len(combined_data)
    print(f'Overall injured count: {overall_injured_count}')

    # Ensure the adate column is in datetime format and extract the year
    if 'adate' in combined_data.columns:
        combined_data['adate'] = pd.to_datetime(combined_data['adate'], errors='coerce')
        combined_data = combined_data.copy()
        combined_data.loc[:, 'year'] = combined_data['adate'].dt.year

    # Sort the DataFrame by the 'injt' column
    combined_data = combined_data.sort_values(by='injt')

    # Load the lookup table
    lookup_table = pd.read_excel('./Data/Data Lookup Table.xlsx')

    # Subset the lookup table by 'column' = 'injt'
    lookup_table_injt = lookup_table[lookup_table['column'] == 'injt']

    # Ensure the 'injt' column in both DataFrames is a string for proper matching
    combined_data['injt'] = combined_data['injt'].astype(str)
    lookup_table_injt.loc[:, 'value'] = lookup_table_injt['value'].astype(str)

    # Create a mapping dictionary from injt to description
    mapping = lookup_table_injt.set_index('value')['description'].to_dict()

    # Create the new 'veh_desc' column using the mapping
    combined_data['veh_desc'] = combined_data['injt'].map(mapping)

    # Replace any missing values with 'Other than Specified'
    combined_data['veh_desc'] = combined_data['veh_desc'].fillna('Other than Specified')

    # Ensure the age column is numeric (if not already) and coerce errors to NaN
    combined_data['age'] = pd.to_numeric(combined_data['age'], errors='coerce')

    # Count the total number of rows before filtering
    total_rows = combined_data.shape[0]

    # Define a valid age mask: ages between 0 and 100 (inclusive)
    valid_age_mask = (combined_data['age'] >= 0) & (combined_data['age'] <= 100)

    # Filter the DataFrame to only include rows with valid ages
    combined_data = combined_data[valid_age_mask].copy()

    # Count the number of valid rows and calculate how many were filtered out
    valid_rows = combined_data.shape[0]
    rows_filtered_out = total_rows - valid_rows

    print(f'Total rows before filtering: {total_rows}')
    print(f'Rows after filtering: {valid_rows}')
    print(f'Rows filtered out by age: {rows_filtered_out}')

    # Ensure the adate column is a datetime object
    combined_data['adate'] = pd.to_datetime(combined_data['adate'], errors='coerce')

    # Define the six-year period (adjust these as needed)
    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2023-12-31 23:59:59')

    # Count total rows before filtering
    total_rows = combined_data.shape[0]

    # Create a boolean mask for dates within the specified period
    valid_date_mask = (combined_data['adate'] >= start_date) & (combined_data['adate'] <= end_date)

    # Get the records that are being filtered out (i.e. not in the valid date range)
    filtered_out = combined_data[~valid_date_mask].copy()

    # Display a sample of the filtered out records
    if filtered_out.empty:
        print('No records were filtered out by the date criteria.')
    else:
        sample_filtered = filtered_out.sample(n=5, random_state=42)
        print(sample_filtered[['adate']])

    # Filter the DataFrame
    combined_data = combined_data[valid_date_mask].copy()

    # Count rows that remain and how many were filtered out
    valid_rows = combined_data.shape[0]
    rows_filtered_out = total_rows - valid_rows

    print(f'Total rows before date filtering: {total_rows}')
    print(f'Rows with dates between {start_date.date()} and {end_date.date()}: {valid_rows}')
    print(f'Rows filtered out: {rows_filtered_out}')

    # Create a new column for the weekday name (e.g., Monday, Tuesday, etc.)
    combined_data['weekday'] = combined_data['adate'].dt.day_name()

    # Create conditions for our weekend definition:
    # - Friday: hour must be 19 or later.
    # - Saturday: any time qualifies.
    # - Sunday: hour must be less than 2.
    friday_mask = (combined_data['adate'].dt.weekday == 4) & (combined_data['adate'].dt.hour >= 19)
    saturday_mask = combined_data['adate'].dt.weekday == 5
    sunday_mask = combined_data['adate'].dt.weekday == 6
    monday_mask = (combined_data['adate'].dt.weekday == 0) & (combined_data['adate'].dt.hour < 2)

    # Combine conditions: if any of these conditions is True, mark as 'Weekend'
    weekend_mask = friday_mask | saturday_mask | sunday_mask | monday_mask

    # Create a new column that indicates if the incident is on the weekend or weekday.
    combined_data['weekend'] = np.where(weekend_mask, 'Weekend', 'Weekday')

    # # Verify by displaying a sample of relevant columns:
    # print(combined_data[['adate', 'weekday', 'weekend']].head(10))

    # # Check the results
    # print(combined_data[['injt', 'veh_desc']].head())

    # Extract the hour from the accident date.
    hours = combined_data['adate'].dt.hour

    # Define the conditions for each time category.
    conditions = [
        ((hours >= 22) | (hours < 2)),    # 22:00 to 01:59
        ((hours >= 2) & (hours < 6)),       # 02:00 to 05:59
        ((hours >= 6) & (hours < 10)),      # 06:00 to 09:59
        ((hours >= 10) & (hours < 14)),     # 10:00 to 13:59
        ((hours >= 14) & (hours < 18)),     # 14:00 to 17:59
        ((hours >= 18) & (hours < 22))      # 18:00 to 21:59
    ]

    # Define labels corresponding to each time category.
    labels = [
        'Night (22:00–01:59)',
        'Early Morning (02:00–05:59)',
        'Morning (06:00–09:59)',
        'Late Morning (10:00–13:59)',
        'Afternoon (14:00–17:59)',
        'Evening (18:00–21:59)'
    ]

    # Use np.select to create the new column based on the conditions.
    combined_data['time_category'] = np.select(conditions, labels, default='Unknown')

    # Optionally convert the time_category column to an ordered categorical type.
    time_cat_type = pd.CategoricalDtype(categories=labels, ordered=True)
    combined_data['time_category'] = combined_data['time_category'].astype(time_cat_type)

    # Check a sample of the new column along with the accident date.
    #print(combined_data[['adate', 'time_category']].head())
    # Define mapping dictionaries for each risk variable
    risk_mappings = {
        'risk1': {1: 'Alcohol', 0: 'No Alcohol', 9999: 'Unknown', np.nan: 'Unknown'},
        'risk2': {1: 'Drug Impairment', 0: 'No Drug Impairment', 9999: 'Unknown', np.nan: 'Unknown'},
        'risk3': {1: 'Seatbelt', 0: 'No Seatbelt', 9999: 'Unknown', np.nan: 'Unknown'},
        'risk4': {1: 'Helmet', 0: 'No Helmet', 9999: 'Unknown', np.nan: 'Unknown'}, 
        'risk5': {1: 'Cellphone', 0: 'No Cellphone', 9999: 'Unknown', np.nan: 'Unknown'}
    }

    # For each risk column, convert it to numeric (if necessary) and map it to a new categorical column.
    for risk, mapping in risk_mappings.items():
        # Ensure the risk column is numeric; invalid parsing will become NaN.
        combined_data[risk] = pd.to_numeric(combined_data[risk], errors='coerce')
        
        # Define new column name based on the risk variable
        if risk == 'risk1':
            new_column = 'alcohol_status'
        elif risk == 'risk2':
            new_column = 'drug_impairment_status'
        elif risk == 'risk3':
            new_column = 'seatbelt_status'
        elif risk == 'risk4':
            new_column = 'helmet_status'
        elif risk == 'risk5':
            new_column = 'cellphone_status'
        
        # Map the risk column values to their corresponding text labels.
        combined_data[new_column] = combined_data[risk].map(mapping)
        
        # Convert the new column to a categorical data type.
        combined_data[new_column] = combined_data[new_column].astype('category')

    # Verify that the new columns have been added correctly.
    # print(combined_data[['risk1', 'alcohol_status', 
    #                     'risk2', 'drug_impairment_status',
    #                     'risk3', 'seatbelt_status', 
    #                     'risk4', 'helmet_status', 
    #                     'risk5', 'cellphone_status']].head())

    # List the raw risk columns that are now redundant
    redundant_risk_columns = ['risk1', 'risk2', 'risk3', 'risk4', 'risk5']

    # Drop these columns from combined_data
    combined_data.drop(columns=redundant_risk_columns, inplace=True, errors='ignore')

    # Verify they have been removed
    #print(combined_data.columns)




    # Merge the combined data with the lookup table on the 'injt' column
    merged_data = pd.merge(combined_data, lookup_table_injt, left_on='injt', right_on='new_value', how='left')

    # Overall injured count
    overall_injured_count = len(merged_data)
    print(f'Overall injured count: {overall_injured_count}')

    return merged_data