import os
import pickle
import pandas as pd
import numpy as np
from utils.prov_names import ProvNames
from utils.regex_patterns import RegexPatterns
import datetime

def make_time_24hr(time, am_pm):
    if am_pm.lower() == 'pm' and time.hour < 12:
        time = (datetime.datetime.combine(datetime.date.today(), time) + datetime.timedelta(hours=12)).time()
    elif am_pm.lower() == 'am' and time.hour == 12:
        time = time.replace(hour=0)
    return time
    
def parse_time(time_str):
    try:
        return datetime.datetime.strptime(time_str, '%H:%M').time()
    except ValueError:
        return datetime.datetime.strptime(time_str, '%H:%M:%S').time()
def parse_good_date(date_str):
    try:
        return datetime.datetime.strptime(date_str, '%m/%d/%Y')
    except ValueError:
        return datetime.datetime.strptime(date_str, '%d/%m/%Y') 

#the thai calendar year is 543 years advanced than the gregorian year, 
#this function handles the conversion
def parse_thai_date(date_str):
    try:
        # Split the date string into day, month, and year components
        day, month, year = date_str.split('/')
        
        # Convert the Thai Buddhist year to the Gregorian year
        gregorian_year = int(year) - 543
        
        # Reconstruct the date string with the adjusted year
        adjusted_date_str = f'{day}/{month}/{gregorian_year}'
        
        # Parse the reconstructed date string
        date = datetime.datetime.strptime(adjusted_date_str, '%d/%m/%Y')
    except ValueError:
        try:
            # Split the date string into month, day, and year components
            month, day, year = date_str.split('/')
            
            # Convert the Thai Buddhist year to the Gregorian year
            gregorian_year = int(year) - 543
            
            # Reconstruct the date string with the adjusted year
            adjusted_date_str = f'{month}/{day}/{gregorian_year}'
            
            # Parse the reconstructed date string
            date = datetime.datetime.strptime(adjusted_date_str, '%m/%d/%Y')
        except ValueError:
            raise ValueError(f"Date {date_str} does not match format '%d/%m/%Y' or '%m/%d/%Y'")
    return date

def standardize_row(row, regex_patterns, prov_names_obj):
    
    #parse_date_cols = ['adate','hdate','rdate','timer','er','diser']
    parse_date_cols = ['adate']

    #need to check %H if it is 12 or 24 hour format %I is 12 hour format
    #will do this by checking group 3 for either AM or PM
    for col_name, col_value in row.items():

        #there are pairs of columns that are related, dates and times that are sometimes split, and not necessarily
        #in the same format. This works through those columns, and passes standardized dates and times to the parsing functions
        #as needed. 
        if str(col_name) in parse_date_cols:
            target_col = ''
            if str(col_name) == 'adate':
                target_col = 'atime'
            elif str(col_name) == 'hdate':
                target_col = 'htime'
            elif str(col_name) == 'rdate':
                target_col = 'timer'
            elif str(col_name) == 'er':
                target_col = 'er_t'
            elif str(col_name) == 'diser':           
                target_col = 'diser_t'

            #print(str(col_name) + ' '  + str(col_value))
            
            # base_zero_date = regex_patterns.pattern_zero_date.match(str(col_value))
            # target_zero_date = regex_patterns.pattern_zero_date.match(str(row[target_col]))
            # if base_zero_date:
            #     print('case 0')
            #     date = np.nan
            # if base_zero_date.group(2) == '00:00:00':  
            #     time = np.nan
                
            base_match_1899_date = regex_patterns.pattern_1899_date.match(str(col_value))
            base_match_yr_first_date = regex_patterns.pattern_yr_first_date.match(str(col_value))
            base_match_thai_date = regex_patterns.pattern_thai_date.match(str(col_value))
            base_match_every_date = regex_patterns.pattern_every_date.match(str(col_value))

            target_match_1899 = regex_patterns.pattern_1899_date.match(str(row[target_col]))
            target_match_yr_first = regex_patterns.pattern_yr_first_date.match(str(row[target_col]))
            target_match_thai = regex_patterns.pattern_thai_date.match(str(row[target_col]))
            target_match_every = regex_patterns.pattern_every_date.match(str(row[target_col]))

            if base_match_1899_date:
                #print('case 1')
                date = datetime.datetime.strptime(base_match_1899_date.group(1), '%m/%d/%Y')
                time = parse_time(base_match_1899_date.group(2))
                am_pm = base_match_1899_date.group(3)
                time = make_time_24hr(time, am_pm)
            elif base_match_yr_first_date:
                #print('case 2')
                if base_match_yr_first_date.group(1) == '0000/00/00' or base_match_yr_first_date.group(1) == '0000-00-00':
                    date = datetime.datetime(2000, 1, 1)  # Default date
                else:
                    date = datetime.datetime.strptime(base_match_yr_first_date.group(1), '%Y-%m-%d')
                time = parse_time(base_match_yr_first_date.group(2))
                am_pm = base_match_yr_first_date.group(3)
                time = make_time_24hr(time, am_pm)
            elif base_match_thai_date:
                #print('case 3')
                date = parse_thai_date(base_match_thai_date.group(1))
                time = parse_time(base_match_thai_date.group(2))
                am_pm = base_match_thai_date.group(3)
                time = make_time_24hr(time, am_pm)
            elif base_match_every_date:

                date = parse_good_date(base_match_every_date.group(1))
                time = parse_time(base_match_every_date.group(2))
                am_pm = base_match_every_date.group(3)
                
                time = make_time_24hr(time, am_pm)
            
            if target_match_1899:
                #print('case 1b')
                if target_match_1899.group(1) == '00/00/0000' or target_match_1899.group(1) == '00-00-0000':
                    target_date = datetime.datetime(2000, 1, 1)  # Default date
                else:
                    target_date = datetime.datetime.strptime(target_match_1899.group(1), '%m/%d/%Y')
                target_time = parse_time(target_match_1899.group(2))
                target_am_pm = target_match_1899.group(3)
                target_time = make_time_24hr(target_time, target_am_pm)
            elif target_match_yr_first:
                #print('case 2b')
                if target_match_yr_first.group(1) == '0000/00/00' or target_match_yr_first.group(1) == '0000-00-00':
                    target_date = datetime.datetime(2000, 1, 1)  # Default date
                else:
                    target_date = datetime.datetime.strptime(target_match_yr_first.group(1), '%Y-%m-%d')
                target_time = parse_time(target_match_yr_first.group(2))
                target_am_pm = target_match_yr_first.group(3)
                target_time = make_time_24hr(target_time, target_am_pm)
            elif target_match_thai:
                #print('case 3b')
                target_date = parse_thai_date(target_match_thai.group(1))
                target_time = parse_time(target_match_thai.group(2))
                target_am_pm = target_match_thai.group(3)
                target_time = make_time_24hr(target_time, target_am_pm)
            elif target_match_every:
                #print('case 4b')
                if target_match_every.group(1) == '00/00/0000' or target_match_every.group(1) == '00-00-0000':
                    target_date = datetime.datetime(2000, 1, 1)  # Default date
                else:
                    target_date = parse_good_date(target_match_every.group(1)) 
                target_time = parse_time(target_match_every.group(2))
                target_am_pm = target_match_every.group(3)
                target_time = make_time_24hr(target_time, target_am_pm)
            else:
                target_date = datetime.datetime(2000, 1, 1)
                target_time = datetime.time(0,0)

            combined_target_datetime = datetime.datetime.combine(target_date, target_time)
            combined_datetime = datetime.datetime.combine(date, time)
            if combined_target_datetime.time() != combined_datetime.time() and combined_datetime.time() > datetime.time(0,0):
                combined_datetime = combined_datetime.replace(
                    hour=combined_target_datetime.hour, 
                    minute=combined_target_datetime.minute, 
                    second=combined_target_datetime.second)
            
            formatted_datetime = combined_datetime.strftime('%Y-%m-%d %H:%M')
            
            row[col_name] = formatted_datetime
        if str(col_name) == 'prov':
            #print(f'Original value of row['prov']: {row[col_name]}')
            
            if prov_names_obj.contains_key(row[col_name]):
                row[col_name] = prov_names_obj.get_value(row[col_name])
                #print(f'Updated value of row['prov']: {str(row[col_name])}')
            elif prov_names_obj.contains_id(row['aplace']):
                row[col_name] = prov_names_obj.get_value(prov_names_obj.get_prov_id_value(row['aplace']))
                #print(f'Updated value of row['prov']: {str(row[col_name])}')
            else:   
                #drop the row if the province and aplace are not in the dictionary
                return None
    
    return row

def process_file(input_file, output_file, row_counts, prov_name_obj, regex_patterns):
    print(f'Processing file: {input_file}')
    
    parse_date_cols = ['adate', 'hdate', 'diser', 'rdate', 'timer']
    delete_cols = ['atime', 'htime', 'diag1', 'diag2', 'diag3', 'diag4', 'diag5', 'diag6', 
                   'br1', 'br2', 'br3', 'br4', 'br5', 'br6', 'occu_t', 'er', 'er_t', 'diser_t', 'timer_t']
    raw_data = pd.read_csv(filepath_or_buffer=input_file, dtype=str, parse_dates=False)

    # Standardize column names: lowercase and replace spaces with underscores
    raw_data.columns = raw_data.columns.str.lower().str.replace(' ', '_')
    

    
    prov_name_obj.create_aplace_prov_dict(raw_data, './Data/prov_ids.csv')
    #print(raw_data.columns)

    raw_data = raw_data.apply(standardize_row, axis=1, regex_patterns=regex_patterns, prov_names_obj=prov_name_obj)

    # Drop rows where 'prov' is blank
    raw_data = raw_data.dropna(subset=['prov'])

    columns_to_replace_blanks = ['risk1', 'risk2', 'risk3', 'risk4', 'risk5']

    # Replace 'N' and blanks with 99 in the specified columns
    replacement_dict = {'N': 9999, np.nan: 9999, '': 9999, '2': 9999, '9': 9999, '999': 9999, '902':9999}
    raw_data[columns_to_replace_blanks] = raw_data[columns_to_replace_blanks].replace(replacement_dict)
    # Convert the specified columns to categorical (factors)
    raw_data[columns_to_replace_blanks] = raw_data[columns_to_replace_blanks].astype('category')
    
    
    # some of the files fail to have a head_injury column, but have the br1-br5 columns
    # which can be used to infer the head_injury column if any of the br columns are 1
    if 'head_injury' not in raw_data.columns:
        columns_to_check = ['br1', 'br2', 'br3', 'br4', 'br5']
        raw_data[columns_to_check] = raw_data[columns_to_check].fillna(0).astype(int)
        if (raw_data[columns_to_check] == 1).any().any():
            raw_data['head_injury'] = 1
        else:
            raw_data['head_injury'] = 0
    else:
        # Explicitly infer objects to retain old behavior for downcasting
        #raw_data['head_injury'] = raw_data['head_injury'].infer_objects(copy=False)

        # Replace blanks with 99, 'HI' with 1, and 'Non HI' with 0 in the Head_Injury column
        raw_data['head_injury'] = raw_data['head_injury'].map({'': 99, 'HI': 1, 'Non HI': 0}).fillna(raw_data['head_injury'])

        # Convert the Head_Injury column to categorical (factor)
        raw_data['head_injury'] = raw_data['head_injury'].astype('category')
    if 'mass_casualty' in raw_data.columns:
        raw_data = raw_data.rename(columns={'mass_casualty': 'mass_casualty'})



    

    columns_to_standardize = [ 'injp', 'injt' ]
    raw_data[columns_to_standardize] = raw_data[columns_to_standardize].replace([np.nan, 'N', 'n'], 9999)
    raw_data[columns_to_standardize] = raw_data[columns_to_standardize].astype('category')
    
    raw_data['mass_casualty'] = raw_data['mass_casualty'].replace([np.nan, 'N'], 0)
    raw_data[parse_date_cols] = raw_data[parse_date_cols].apply(pd.to_datetime, format='mixed', errors='coerce')



    # Cleanse the injt column
    raw_data['injt'] = raw_data['injt'].str.lstrip('0')  # Remove leading zeros
    raw_data['injt'] = raw_data['injt'].replace(['', '0901', '00', '0', '0999', '901','902', '999'], '9999')  # Replace specific values with 9999
    raw_data['injt'] = pd.to_numeric(raw_data['injt'], errors='coerce').fillna(9999).astype(int)  # Convert to numeric and handle errors

    # Convert the age column to numeric
    if 'age' in raw_data.columns:
        raw_data['age'] = pd.to_numeric(raw_data['age'], errors='coerce')

    # # Set staer to '7' if it is np.nan and staward has any value
    # raw_data['staer'] = raw_data.apply(lambda row: '7' if pd.isna(row['staer']) and pd.notna(row['staward']) else row['staer'], axis=1)

    # Add deceased column
    raw_data['deceased'] = raw_data.apply(lambda row: 1 if row['staer'] in ['1', '6'] or row['staward'] == '6' else 0, axis=1)

    #delete the columns that are not needed
    #time columns that have been converted: atime, htime, diser_t, er_t, timer_t
    #diagnosis related columns, diag1-6, br1-6
    #occupation related columns: occu_t, occu
    #treatment related: ems, athosp, staer, staward ** staer and staward are used to create the deceased column
    #activity related: activity, injby, injfrom
    delete_cols = ['atime', 'htime', 'diag1', 'diag2', 'diag3', 'diag4', 'diag5', 'diag6', 
                   'br1', 'br2', 'br3', 'br4', 'br5', 'br6', 'occu_t', 'er', 'er_t', 'diser_t', 'timer_t',
                   'injby', 'injfrom', 'occu', 'ems', 'activity', 'ems', 'atohosp','staer', 'staward']
    raw_data = raw_data.drop(columns=delete_cols, errors='ignore')

    # Count the rows before saving
    row_count = len(raw_data)


    # Write the DataFrame to a pickle file
    pickle.dump(raw_data, open(output_file, 'wb'))
    return row_count


def load_and_process_files(input_files, output_files):
    row_count = 0
    row_counts = {}
    prov_name_obj = ProvNames()
    regex_patterns = RegexPatterns()
    for input_file, output_file in zip(input_files, output_files):
        if os.path.exists(output_file):
            print(f'Output file {output_file} already exists. Skipping processing for {input_file}.')
            continue
        row_count = process_file(input_file, output_file, row_counts, prov_name_obj, regex_patterns)
        row_counts[input_file] = row_count
    print( row_counts)
    