import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import OrdinalEncoder

#this function gets the EDA numbers needed for early in the paper
def exploratory_analysis_numbers(combined_data):
    motorcycle_data = combined_data[combined_data['veh_desc'] == 'Motorcycle'].copy()
    print('Motorcycle Accidents by Year')
    motorcycle_year_counts = motorcycle_data['year'].value_counts()
    print(motorcycle_year_counts)
    print('Total Motorcycle Accidents:', motorcycle_year_counts.sum())
    motorcycle_helmet_counts = motorcycle_data['helmet_status'].value_counts()
    print('Helmet Status Counts:')
    print(motorcycle_helmet_counts)

    total_injuries = len(combined_data)
    print('\nTotal Injuries:', total_injuries)

    # Group by veh_desc to count injuries and calculate each as a percentage of the total
    injuries_by_vehicle = (combined_data.groupby('veh_desc')
                                         .size()
                                         .reset_index(name='injury_count'))
    injuries_by_vehicle['percentage'] = injuries_by_vehicle['injury_count'] / total_injuries * 100
    print('\nInjuries by Vehicle Type:')
    print(injuries_by_vehicle)
    
    # define the baseline and target years
    baseline_year = 2019
    target_year = 2023
    # Extract the year from 'adate'
    combined_data['year'] = combined_data['adate'].dt.year
    
    # Group data by province and year to count accidents
    province_year_counts = combined_data.groupby(['prov', 'year']).size().reset_index(name='accidents')
    
    # Get accident counts for the baseline year (e.g., 2022)
    baseline = province_year_counts[province_year_counts['year'] == baseline_year][['prov', 'accidents']]
    baseline = baseline.rename(columns={'accidents': 'baseline_accidents'})
    
    # Get accident counts for the target year (2023)
    target = province_year_counts[province_year_counts['year'] == target_year][['prov', 'accidents']]
    target = target.rename(columns={'accidents': 'target_accidents'})
    
    # Merge the two DataFrames on province, using outer join to include all provinces
    merged = pd.merge(baseline, target, on='prov', how='outer').fillna(0)
    
    # Calculate the raw increase and percentage increase
    merged['increase'] = merged['target_accidents'] - merged['baseline_accidents']
    merged['pct_increase'] = merged.apply(
        lambda row: (row['increase'] / row['baseline_accidents'] * 100) if row['baseline_accidents'] != 0 else None, 
        axis=1
    )
    
    # Sort provinces by the raw increase in descending order
    merged = merged.sort_values(by='increase', ascending=False)
    print(merged)
    provinces_gdf = gpd.read_file('./Data/tha_admbnda_adm1_rtsd_20220121.shp')
    plot_accident_increase_map(merged, provinces_gdf, value_column='increase')
    
        # Count injuries by gender
    gender_counts = motorcycle_data['sex'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    
    # Calculate mean and median age by gender
    age_stats = motorcycle_data.groupby('sex')['age'].agg(['mean', 'median']).reset_index()
    age_stats.columns = ['Gender', 'Mean Age', 'Median Age']
    
    # Print results
    print('Injured Motorcyclists by Gender:')
    print(gender_counts)
    print('\nMean and Median Age by Gender:')
    print(age_stats)
    
# Plot the distribution of accidents by years
def plot_accidents_by_year(df):
    # Ensure the 'adate' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['adate']):
        df['adate'] = pd.to_datetime(df['adate'], errors='coerce')
    
    # Drop rows where the date conversion failed
    df = df.dropna(subset=['adate'])
    
    # Extract the year from the accident date
    df['year'] = df['adate'].dt.year
    
    # Define the range of years for the histogram
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    bins = range(min_year, max_year + 2)  # +2 ensures the last year is included
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['year'], bins=bins, color='steelblue', edgecolor='black', align='left')
    plt.xlabel('Year')
    plt.ylabel('Number of Injuries')
    #plt.title('Histogram of Injuries from Vehicle Accidents by Year')
    plt.xticks(range(min_year, max_year + 1))
    
    # Save the plot to a PNG file using the file_name variable
    file_name =  './data/plot_output/eda_accident_year.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

# This creates a geopandas map of the increase in accidents by province from 2019 to 2023, not the entire dataset
def plot_accident_increase_map(province_changes, provinces_gdf, value_column='pct_increase'):

    # Merge the province change data with the geospatial data based on the 'prov' column
    merged_gdf = provinces_gdf.merge(province_changes, left_on='ADM1_EN', right_on='prov', how='left')
    
    # Set up the plot with a specific figure size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the merged GeoDataFrame using a reversed RdYlGn colormap
    # This ensures that the highest accident increases (the highest values) are in red,
    # middle values appear yellow, and the lowest in green.
    merged_gdf.plot(column=value_column, cmap='RdYlGn_r', linewidth=0.8, 
                    ax=ax, edgecolor='0.8', legend=True)
    
    # Add a title and remove the axis for a cleaner map view
    #ax.set_title('Motorcycle Accident Increase by Province (2019 to 2023)', fontsize=15)
    ax.axis('off')
    
    file_name = './data/plot_output/eda_accident_increase_2019_2023_map.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()    
    
# This function plots the number of accidents by time category and helmet status
def plot_helmet_status_by_time(df):
    
    # Group the data by 'time_category' and 'helmet_status' and count occurrences
    grouped = df.groupby(['time_category', 'helmet_status'], observed=False).size().reset_index(name='count')

    # Pivot the data so that each row is a time_category and columns are helmet statuses.
    pivot = grouped.pivot(index='time_category', columns='helmet_status', values='count').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind='bar', ax=ax)

    ax.set_xlabel('Time Category')
    ax.set_ylabel('Number of Accidents')
    #ax.set_title('Accidents by Time and Helmet Status (2018-2023)')

    plt.xticks(rotation=45)

    plt.tight_layout()

    file_name = './data/plot_output/eda_helmet_status.png'

    plt.savefig(file_name, bbox_inches='tight')
    plt.close()



def plot_motorcycle_by_age(motorcycle_accidents): 
    # Group the data by 'age' and 'sex' and count occurrences
    age_sex_counts = motorcycle_accidents.groupby(['age', 'sex']).size().unstack(fill_value=0)
    
    # Extract counts for males and females. If a category is missing, default to zeros.
    males = age_sex_counts.get('1', pd.Series(0, index=age_sex_counts.index))
    females = age_sex_counts.get('2', pd.Series(0, index=age_sex_counts.index))
    
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(age_sex_counts.index, males, label='Male')
    ax.bar(age_sex_counts.index, females, bottom=males, label='Female')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Count of Accidents')
    #ax.set_title('Motorcycle Accidents by Age and Gender (2018-2023)')
    ax.legend()
    
    # Set x-axis ticks every 5 years to reduce clutter
    max_age = int(motorcycle_accidents['age'].max())
    ax.set_xticks(range(0, max_age + 1, 5))
    
    file_name = './data/plot_output/eda_hist_motorcycle_age.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_motorcycle_by_year(motorcycle_accidents):
    # Ensure the 'adate' column is datetime; convert if necessary
    # had some data consistency issues initially, left the check in
    if not pd.api.types.is_datetime64_any_dtype(motorcycle_accidents['adate']):
        motorcycle_accidents['adate'] = pd.to_datetime(motorcycle_accidents['adate'], errors='coerce')
    
    # Drop rows with missing dates
    motorcycle_accidents = motorcycle_accidents.dropna(subset=['adate'])
    
    # Extract the year from the accident date
    motorcycle_accidents['year'] = motorcycle_accidents['adate'].dt.year
    
    # Group the data by year and gender (sex) and count occurrences
    year_sex_counts = motorcycle_accidents.groupby(['year', 'sex']).size().unstack(fill_value=0)
    
    # Extract counts for males and females (if a group is missing, it defaults to zeros)
    males = year_sex_counts.get('1', pd.Series(0, index=year_sex_counts.index))
    females = year_sex_counts.get('2', pd.Series(0, index=year_sex_counts.index))
    
    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(year_sex_counts.index, males, label='Male')
    ax.bar(year_sex_counts.index, females, bottom=males, label='Female')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Count of Accidents')
    #ax.set_title('Injuries from Motorcycle Accidents by Year and Gender (2018-2023)')
    ax.legend()
    
    # Set x-axis ticks to each year and rotate labels if needed
    ax.set_xticks(year_sex_counts.index)
    ax.set_xticklabels(year_sex_counts.index, rotation=45)
    
    file_name = './data/plot_output/eda_hist_motorcycle_year.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    
def plot_motorcycle_by_role_and_year(motorcycle_accidents):
    # Ensure the 'adate' column is datetime; convert if necessary
    if not pd.api.types.is_datetime64_any_dtype(motorcycle_accidents['adate']):
        motorcycle_accidents['adate'] = pd.to_datetime(motorcycle_accidents['adate'], errors='coerce')
    
    # Drop rows with missing dates
    motorcycle_accidents = motorcycle_accidents.dropna(subset=['adate'])
    
    # Extract the year from the accident date
    motorcycle_accidents['year'] = motorcycle_accidents['adate'].dt.year
    
    # Group the data by year and roles and count occurrences
    year_role_counts = motorcycle_accidents.groupby(['year', 'injp']).size().unstack(fill_value=0)
    
    # Extract counts for males and females (if a group is missing, it defaults to zeros)
    drivers = year_role_counts.get('2', pd.Series(0, index=year_role_counts.index))
    passengers = year_role_counts.get('3', pd.Series(0, index=year_role_counts.index))
    others = year_role_counts.get('1', pd.Series(0, index=year_role_counts.index))

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(year_role_counts.index, drivers, label='Driver')
    ax.bar(year_role_counts.index, passengers, bottom=drivers, label='Passengers')
    ax.bar(year_role_counts.index, others, bottom=drivers, label='Other(Pedestrian, etc.)')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Count of Accidents')
    #ax.set_title('Injuries from Motorcycle Accidents by Year and Role (2018-2023)')
    ax.legend()
    
    # Set x-axis ticks to each year and rotate labels if needed
    ax.set_xticks(year_role_counts.index)
    ax.set_xticklabels(year_role_counts.index, rotation=45)
    
    file_name = './data/plot_output/eda_role_year.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()    

# This function plots the helmet compliance by age band
def plot_helmet_compliance_by_age_band(motorcycle_accidents):
    max_age = int(motorcycle_accidents['age'].max())

    labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    banded_bins = [0, 18, 30, 45, 60, max_age]
    motorcycle_accidents['age_band'] = pd.cut(motorcycle_accidents['age'], 
        bins=banded_bins, labels=labels, right=True, include_lowest=True)

    # Group by age_band and calculate helmet compliance metrics
    compliance_by_age = motorcycle_accidents.groupby('age_band', as_index=False, observed=False).agg(
        helmet_compliant_cases=('helmet_status', lambda x: (x == 'Helmet').sum()),
        total_cases=('helmet_status', 'count')
    )
    compliance_by_age['helmet_compliance_rate'] = (
        compliance_by_age['helmet_compliant_cases'] / compliance_by_age['total_cases']
    )

    # Define a custom color mapping for each age band
    # Used ChatGPT to generate the color mapping
    color_map_age = {
        '0-18': '#fee8c8',    # light
        '19-30': '#fdbb84',   # light-medium
        '31-45': '#fc8d59',   # medium
        '46-60': '#e34a33',   # medium-dark
        '60+': '#b30000'      # dark
    }

    # Create a list of colors corresponding to the age bands in compliance_by_age
    colors = [color_map_age[str(age_band)] for age_band in compliance_by_age['age_band']]

    # Plot the helmet compliance rate by age band
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(compliance_by_age['age_band'].astype(str), compliance_by_age['helmet_compliance_rate'], color=colors)
    ax.set_xlabel('Age Band')
    ax.set_ylabel('Helmet Compliance Rate')
    #ax.set_title('Helmet Compliance by Age Bands for Motorcycle Accidents')

    # Adjust the layout and optionally add a note at the bottom
    plt.subplots_adjust(bottom=0.2)
    note_text = ('Note: Data based on hospitalization records of motorcycle accident victims.\n'
                'Source: https://data.go.th/dataset/injury-surveillance')
    plt.figtext(0.5, 0.05, note_text, ha='center', fontsize=10, wrap=True)

    file_name = './data/plot_output/eda_helmet_comp_age_band.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    
def plot_alcohol_status_by_time(df):
    # Group by time_category and the new helmet_status column.
    grouped_helmet = df.groupby(['time_category', 'alcohol_status'], observed=False).size().reset_index(name='count')
    print(grouped_helmet.head())
    # Pivot the data so that each row is a time_category and columns are helmet statuses.
    pivot_helmet = grouped_helmet.pivot(index='time_category', columns='alcohol_status', values='count').fillna(0)

    # Plot the grouped bar chart.
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_helmet.plot(kind='bar', ax=ax)
    ax.set_xlabel('Time Category')
    ax.set_ylabel('Number of Motorcycle Accidents')
    #ax.set_title('Accidents by Time and Alcohol Status (2018-2023)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = './data/plot_output/eda_alcohol_time.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    
 # This function plots the number of injuries by vehicle type   
def plot_injuries_by_vehicle(df):
    # Group by 'injt' and count the number of injuries
    injuries_count_by_injt = df.groupby('veh_desc').size().reset_index(name='count').sort_values(by='count', ascending=False)  

    # Create a bar chart of the top 10 counts of injuries
    top_10_injuries = injuries_count_by_injt.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_10_injuries['veh_desc'], top_10_injuries['count'], color='skyblue')
    plt.xlabel('Count of Injuries (millions of hospital treatments)')
    plt.ylabel('Vehicle Type')
    #plt.title('Top 10 Hospitalizations by Vehicle Type Involved in Injury (2018-2023)')
    note_text = ('Note: The data presented is based on hospitalization records of vehicle accident victims.\n'
                 'Source: https://data.go.th/dataset/injury-surveillance')
    plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=10, wrap=True)

    plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    file_name = './data/plot_output/eda_veh_type.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

# This function plots helmet compliance by gender
def plot_helmet_compliance_by_gender(df):
    motorcycle_accidents = df[df['veh_desc' == 'Motorcycle']]
    # Group by sex and calculate helmet compliance metrics
    compliance_by_sex = motorcycle_accidents.groupby('sex', as_index=False).agg(
        helmet_compliant_cases=('helmet_status', lambda x: (x == 'Helmet').sum()),
        total_cases=('helmet_status', 'count')
    )
    compliance_by_sex['helmet_compliance_rate'] = (
        compliance_by_sex['helmet_compliant_cases'] / compliance_by_sex['total_cases']
    )
    # Map the sex codes in the aggregated DataFrame
    compliance_by_sex['sex'] = compliance_by_sex['sex'].replace({'1': 'Male', '2': 'Female'})

    # Define a color mapping for the sex categories
    # Used ChatGPT to generate the color mapping
    color_map = {'Male': '#1f77b4',    # A deep blue
                 'Female': '#ff7f0e'}  # A warm orange

    fig, ax = plt.subplots(figsize=(8, 6))
    # Use list comprehension to assign colors based on sex
    bars = ax.bar(compliance_by_sex['sex'], compliance_by_sex['helmet_compliance_rate'],
                  color=[color_map[sex] for sex in compliance_by_sex['sex']])
    ax.set_xlabel('Sex')
    ax.set_ylabel('Helmet Compliance Rate')
    #ax.set_title('Helmet Compliance by Sex for Motorcycle Accidents')
    plt.xticks(rotation=0)

    # Adjust the bottom margin to make room for the note
    plt.subplots_adjust(bottom=0.25)
    note_text = ('Note: The data presented is based on hospitalization records of vehicle accident victims.\n'
                 'Source: https://data.go.th/dataset/injury-surveillance')
    plt.figtext(0.5, 0.1, note_text, ha='center', fontsize=10, wrap=True)

    file_name = './data/plot_output/eda_helmet_comp_gender.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

# This function plots the helmet compliance by province using a geopandas map
def plot_province_map(df):
    # Load the shapefile
    shapefile_path = './Data/tha_admbnda_adm1_rtsd_20220121.shp'
    gdf = gpd.read_file(shapefile_path)
    
    # Filter down to motorcycle accidents (injt == 2)
    motorcycle_accidents =df[df['veh_desc'] == 'Motorcycle'].copy()
    # print(f'Number of motorcycle accidents: {motorcycle_accidents.shape[0]}')

    # Remove rows where helmet_status is 'Unknown' to ensure valid comparisons
    motorcycle_accidents = motorcycle_accidents[motorcycle_accidents['helmet_status'] != 'Unknown'].copy()

    # Here, helmet compliance is defined as the number of records with risk4 == 1 
    # divided by the total number of records (after filtering).
    helmet_compliant_count = (motorcycle_accidents['helmet_status'] == 'Helmet').sum()
    total_count = motorcycle_accidents.shape[0]
    helmet_compliance_rate = helmet_compliant_count / total_count if total_count else 0

    # Create a DataFrame summarizing the helmet compliance
    helmet_compliance_df = pd.DataFrame({
        'helmet_compliance_rate': [helmet_compliance_rate],
        'total_motorcycle_accidents': [total_count],
        'helmet_compliant_cases': [helmet_compliant_count]
    })

    # print(helmet_compliance_df)
    # Group by province and compute helmet compliance
    helmet_compliance_by_prov = motorcycle_accidents.groupby('prov', as_index=False).agg(
        helmet_compliant_cases=('helmet_status', lambda x: (x == 'Helmet').sum()),
        total_accidents=('helmet_status', 'count')
    )
    helmet_compliance_by_prov['helmet_compliance_rate'] = (
        helmet_compliance_by_prov['helmet_compliant_cases'] / helmet_compliance_by_prov['total_accidents']
    )

    # print(helmet_compliance_by_prov)

    # Calculate the helmet compliance rate for each province
    helmet_compliance_by_prov['helmet_compliance_rate'] = (
        helmet_compliance_by_prov['helmet_compliant_cases'] / helmet_compliance_by_prov['total_accidents']
    )
    gdf_merged = gdf.merge(helmet_compliance_by_prov, left_on='ADM1_EN', right_on='prov', how='left')

    # Plot the choropleth map based on helmet compliance rate
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_merged.plot(column='helmet_compliance_rate', ax=ax, legend=True,
                    cmap='RdYlGn', edgecolor='black')
    #ax.set_title('Hospitalized Helmet Compliance by Province (2018-2023)')
    # Add a note at the bottom center of the figure
    note_text = ('Note: The data presented is based on hospitalization records of motorcycle accident victims.\n'
                'Source: https://data.go.th/dataset/injury-surveillance')
    plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=10, wrap=True)
    ax.set_axis_off()
    file_name = './data/plot_output/eda_helmet_comp_province_map.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

# Needed some additional EDA, specifically looking at fatalities and head injuries
# creates two plots - fatalities by helmet status and head injuries by helmet status
def exploratory_death_analysis(df):
    file_name = './data/plot_output/eda_motorcycle_death_analysis.png'
    # Filter for motorcycle accidents
    motorcycle_accidents = df[df['veh_desc'] == 'Motorcycle'].copy()

    # Further filter for motorcyclists that died
    died_motorcyclists = motorcycle_accidents[motorcycle_accidents['deceased'] == 1].copy()

    # Optional: Count records with missing helmet info (where helmet_status is 'Unknown')
    missing_helmet_info = died_motorcyclists[died_motorcyclists['helmet_status'] == 'Unknown']
    num_missing = missing_helmet_info.shape[0]
    print(f"Records with missing helmet info (helmet_status=='Unknown'): {num_missing}")

    # Remove rows where helmet_status is 'Unknown' to ensure valid comparisons
    died_motorcyclists = died_motorcyclists[died_motorcyclists['helmet_status'] != 'Unknown'].copy()

    # Aggregate counts by outcome and helmet_status
    # Here, assume 'deceased' is 1 for death and 0 for survival.
    grouped_counts = motorcycle_accidents.groupby(['helmet_status', 'deceased']).size().reset_index(name='count')

    # Pivot the table so that each row corresponds to a helmet_status and columns correspond to outcomes
    pivot_table = grouped_counts.pivot(index='helmet_status', columns='deceased', values='count').fillna(0)

    # Rename the outcome columns for clarity: 0 -> 'Survived', 1 -> 'Died'
    pivot_table.columns = ['Survived' if col == 0 else 'Died' for col in pivot_table.columns]

    # Optionally, sort the index if desired
    pivot_table = pivot_table.sort_index()

    # --- Plotting ---

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create a grouped bar plot using the pivot_table index (helmet_status) and the outcome columns.
    bar_plot = pivot_table.plot(kind='bar', ax=ax)

    #ax.set_title('Motorcyclist Outcomes by Helmet Status (2018-2023)')
    ax.set_xlabel('Helmet Status')
    ax.set_ylabel('Count (in millions)')

    # Format y-axis ticks to show numbers in millions
    def millions(x, pos):
        return f'{x/1e6:.1f}M'
    ax.yaxis.set_major_formatter(FuncFormatter(millions))

    # --- Annotate Each Bar with the Proportion within its Helmet Status Group ---
    # Each group (row in pivot_table) represents a helmet_status category.
    # For each helmet status, we want to annotate the percentage for each outcome relative to the row total.
    for row_idx, helmet_status in enumerate(pivot_table.index):
        # Total count for this helmet status category
        row_total = pivot_table.loc[helmet_status].sum()
        for col_idx, outcome in enumerate(pivot_table.columns):
            count = pivot_table.loc[helmet_status, outcome]
            # Compute the proportion for the outcome within its helmet status group (percentage)
            proportion = (count / row_total) * 100 if row_total > 0 else 0
            
            # Get the bar object corresponding to the current outcome and helmet_status.
            # Note: ax.containers is a list of BarContainer objects in order of the outcome columns.
            bar = ax.containers[col_idx][row_idx]
            
            # Determine the center x-coordinate and the top y-coordinate for annotation.
            x_coord = bar.get_x() + bar.get_width() / 2
            y_coord = bar.get_height()
            
            # Annotate the bar with the computed percentage.
            ax.text(x_coord, y_coord, f'{proportion:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)

    # # Close the figure to free up resources.
    plt.close()

    file_name = './data/plot_output/eda_head_injury.png'
    # Aggregate counts by helmet_status and head_injury outcome.
    # Assume head_injury is 1 for head injuries and 0 for no head injuries.
    grouped_counts = motorcycle_accidents.groupby(['helmet_status', 'head_injury']).size().reset_index(name='count')

    # Pivot the table so that rows correspond to helmet_status and columns correspond to head injury outcomes.
    pivot_table = grouped_counts.pivot(index='helmet_status', columns='head_injury', values='count').fillna(0)

    # Rename the outcome columns for clarity (0 = 'No Head Injury', 1 = 'Head Injury').
    pivot_table.columns = ['No Head Injury' if col == 0 else 'Head Injury' for col in pivot_table.columns]

    # (Optional) sort the helmet_status categories
    pivot_table = pivot_table.sort_index()
    print(pivot_table)
    # Create a grouped bar plot.
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_plot = pivot_table.plot(kind='bar', ax=ax)
    #ax.set_title('Motorcyclist Head Injury Outcomes by Helmet Status (2018-2023)')
    ax.set_xlabel('Helmet Status')
    ax.set_ylabel('Count (in millions)')

    # Format the y-axis to display numbers in millions.
    def millions(x, pos):
        return f'{x/1e6:.1f}M'
    ax.yaxis.set_major_formatter(FuncFormatter(millions))

    # Annotate each bar with the proportion (percentage) within the helmet_status group.
    # The percentage is computed as the count for the outcome divided by the total count for that helmet_status.
    # ChatGPT helped with getting the labels on the bars
    for row_idx, helmet_status in enumerate(pivot_table.index):
        # Total count for the current helmet status group
        row_total = pivot_table.loc[helmet_status].sum()
        for col_idx, outcome in enumerate(pivot_table.columns):
            count = pivot_table.loc[helmet_status, outcome]
            proportion = (count / row_total) * 100 if row_total > 0 else 0
            
            # Get the bar for this helmet_status and outcome.
            bar = ax.containers[col_idx][row_idx]
            x_coord = bar.get_x() + bar.get_width() / 2
            y_coord = bar.get_height()
            ax.text(x_coord, y_coord, f'{proportion:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()



# This function creates a partial dependence plot for a given continuous feature using the model and data
def make_partial_plot(model, X_full, feature_name, file_name='./data/plot_output/partial_plot.png'):

    # PD plot for the 'age' feature
    fig, ax = plt.subplots(figsize=(8, 6))

    PartialDependenceDisplay.from_estimator(model, X_full, [feature_name], ax=ax)

    # Set the title and labels.
    #ax.set_title(f'Partial Dependence Plot for {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Partial Dependence')
    # Save the plot to a file (e.g., PNG format).
    plt.savefig(file_name)

    # Close the figure to free up resources.
    plt.close()

# This plotting function is a ChatGPT creation. Modified it with the ability to save the plot 
# and added the probability labels, also the ability to change the mapping of x-axis is my work
def make_cat_partial_plot(model, X_full, feature_name, file_name='./data/plot_output/cat_partial_plot.png', mapping_dict=None):
    # Get the unique categories in the order they appear (or sort as desired)
    categories = X_full[feature_name].dropna().unique()
    
    # Initialize a list to store the average predicted probabilities for each category.
    avg_predictions = []
    
    # For each unique category, set the entire column to that category and compute the average prediction.
    for cat in categories:
        X_temp = X_full.copy()
        X_temp[feature_name] = cat  # force the entire column to a given category
        # The model's pipeline should handle the raw categorical values.
        preds = model.predict_proba(X_temp)[:, 1]  # probability of the positive class
        avg_predictions.append(np.mean(preds))
    
    # If a mapping dictionary is provided, generate new labels; otherwise, use categories as-is.
    if mapping_dict is not None:
        # For each category, use the mapped label if it exists; otherwise, keep the original.
        new_labels = [mapping_dict.get(cat, cat) for cat in categories]
    else:
        new_labels = categories

    # Create the plot using bar positions based on an index.
    x_positions = np.arange(len(new_labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw the bars with the new x-axis tick labels.
    bars = ax.bar(x_positions, avg_predictions, color='skyblue', tick_label=new_labels)
    
    #ax.set_title(f'Partial Dependence of {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Average Predicted Probability')
    ax.set_ylim(0, 1)

    # Annotate each bar with its value.
    for bar, avg in zip(bars, avg_predictions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x-coordinate: center of the bar
            height + 0.02,                       # y-coordinate: slightly above the bar
            f'{avg:.2f}',                        # annotation text (formatted to 2 decimals)
            ha='center', va='bottom', fontsize=10, color='black'
        )

    # Rotate x-axis labels for better readability.
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot if a file name is provided.
    if file_name:
        plt.savefig(file_name)
        print(f'Plot saved to {file_name}')
    
    # plt.show()

#modified the ChatGPT model above to add a subset of categories to plot
def make_cat_partial_plot_subset(model, X, feature, categories_to_include, file_name=None):

    # Filter the categories based on user input.
    # This ensures that only the desired subset is used; if a specified category is not in the data,
    # it will be ignored.
    unique_categories = X[feature].dropna().unique()
    # Keep only those categories that are in both the data and the user-defined list.
    selected_categories = [cat for cat in unique_categories if cat in categories_to_include]
    
    if not selected_categories:
        raise ValueError('None of the specified categories are present in the data.')

    # List to store average predicted probability for each selected category.
    avg_predictions = []

    for cat in selected_categories:
        X_temp = X.copy()
        # Force the entire column to the category value.
        X_temp[feature] = cat
        # Calculate prediction probabilities (assuming binary classification).
        preds = model.predict_proba(X_temp)[:, 1]  # use probability for positive class
        avg_predictions.append(np.mean(preds))
    
    # Create the bar plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(selected_categories, avg_predictions, color='skyblue')
    #ax.set_title(f'Partial Dependence of {feature} (Subset)')
    ax.set_xlabel(feature)
    ax.set_ylabel('Average Predicted Probability')
    ax.set_ylim(0, 1)

    # Annotate each bar with its predicted probability.
    for bar, avg in zip(bars, avg_predictions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x-coordinate, center of the bar
            height + 0.02,                       # y-coordinate, slightly above bar
            f'{avg:.2f}',                        # annotation text
            ha='center', va='bottom', fontsize=10, color='black'
        )

    # Rotate x-axis labels for clarity.
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot if a file name is provided.
    if file_name:
        plt.savefig(file_name)
        print(f'Plot saved to {file_name}')

    #plt.show()