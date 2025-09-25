import pandas as pd
import numpy as np
import pickle
import datetime

import os
import concurrent.futures
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
import geopandas as gpd

from utils.regex_patterns import RegexPatterns
from utils.prov_names import ProvNames


from sklearn.inspection import PartialDependenceDisplay


from data_loader import load_and_process_files, process_file
from data_processor import load_pickles, transform_data
from data_visualization import exploratory_analysis_numbers, plot_accidents_by_year, exploratory_death_analysis
from data_visualization import plot_helmet_status_by_time, plot_alcohol_status_by_time, plot_injuries_by_vehicle, plot_province_map
from data_visualization import plot_motorcycle_by_age, plot_motorcycle_by_year, plot_motorcycle_by_role_and_year
from data_visualization import plot_helmet_compliance_by_age_band, make_partial_plot, make_cat_partial_plot
from data_visualization import make_cat_partial_plot_subset
from helmet_regressions import logistic_reg_model, grid_search_model, random_grid_model, model_w_bayesian
from helmet_regressions import compare_models, model_2023_w_bayesian, random_grid_2023_model

input_files = [
    './Data/is2018.csv',
    './Data/is2019.csv',
    './Data/is2020.csv',
    './Data/is2021.csv', 
    './Data/is2022.csv',
    './Data/is2023.csv'
]

output_files = [
    './Data/processed_is2018.pcl',
    './Data/processed_is2019.pcl',
    './Data/processed_is2020.pcl',
    './Data/processed_is2021.pcl',
    './Data/processed_is2022.pcl',
    './Data/processed_is2023.pcl'
]
   

  

row_counts = {}

# Process files (if needed)
load_and_process_files(input_files, output_files)

# Load combined data from pickles
combined_data = load_pickles(output_files)

# Transform combined data
combined_data = transform_data(combined_data)

# Generate plots
#plot_helmet_status_by_time(combined_data)




# print(combined_data.columns)
###
### Exploratory Data Analysis (EDA)
###

exploratory_analysis_numbers(combined_data)
exploratory_death_analysis(combined_data)
plot_accidents_by_year(combined_data)
plot_injuries_by_vehicle(combined_data)
motorcycle_data = combined_data[combined_data['veh_desc'] == 'Motorcycle']
#remove rows where people not in vehicle
motorcycle_data = motorcycle_data[motorcycle_data['injp'].isin(['1', '2'])]
motorcycle_data['injp'] = motorcycle_data['injp'].map({'1': 'Driver', '2': 'Passenger'})
print(motorcycle_data['injp'].unique())
# Filter to include only rows with valid helmet statuses.
valid_statuses = ['Helmet', 'No Helmet']
motorcycle_data = motorcycle_data[motorcycle_data['helmet_status'].isin(valid_statuses)].copy()
mapping = {'Helmet': 1, 'No Helmet': 0}
motorcycle_data['helmet_used'] = motorcycle_data['helmet_status'].map(mapping)
print('Motorcycle accidents with known helmet status:', len(motorcycle_data))

### EDA Plots

plot_alcohol_status_by_time(motorcycle_data)
plot_helmet_status_by_time(motorcycle_data)
plot_helmet_compliance_by_age_band(motorcycle_data)
plot_motorcycle_by_age(motorcycle_data)
plot_motorcycle_by_year(motorcycle_data)
plot_motorcycle_by_role_and_year(motorcycle_data)

###
### Overall Dataset Models
###

### Definitely consider commenting out anything you're not wanting to run, especially the ____ grid search ___ if you
### are not using stored models. The other models take from a few minutes to 7-8 hours to run. Grid search takes 6 days

# Decides to run the models using stored models or not
using_stored_models = True
rf_random_model =  random_grid_model(motorcycle_data, use_stored_model=using_stored_models, model_size=0.20)

# rf_bayesian_model = model_w_bayesian(motorcycle_data, use_stored_model=using_stored_models, model_size=0.20)
# logistic_reg_model = logistic_reg_model(motorcycle_data, use_stored_model=using_stored_models, model_size=0.20)
# ### grid search
# rf_grid_model = grid_search_model(motorcycle_data, use_stored_model=using_stored_models, model_size=0.20)


rf_feature_names = rf_random_model.named_steps['preprocessor'].get_feature_names_out()
rf_importances = rf_random_model.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({'feature': rf_feature_names, 'importance': rf_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)


# Get the feature names from the preprocessor and the feature importances from the classifier.
rf_feature_names = rf_random_model.named_steps['preprocessor'].get_feature_names_out()
rf_importances = rf_random_model.named_steps['classifier'].feature_importances_

# Build the DataFrame of feature importances.
importance_df = pd.DataFrame({'feature': rf_feature_names, 'importance': rf_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Select the top 10 features.
top10_features = importance_df.head(10)
# Display the top 10 features.
print(top10_features)

# Filter for province features (assuming province features contain 'cat__prov_' in their names).
prov_importance_df = importance_df[importance_df['feature'].str.contains('cat__prov_')].copy()

# Sort the province features by importance (already sorted overall, but ensure in case).
prov_importance_df = prov_importance_df.sort_values(by='importance', ascending=False)

# Get the top ten province features.
top_ten_prov = prov_importance_df.head(10)

print('Top Ten Province Features by Importance:')
print(top_ten_prov)

gender_importance_df = importance_df[importance_df['feature'].str.contains('cat__sex_')].copy()
gender_importance_df = gender_importance_df.sort_values(by='importance', ascending=False)
alcohol_importance_df = importance_df[importance_df['feature'].str.contains('cat__alcohol_status_')].copy()
alcohol_importance_df = alcohol_importance_df.sort_values(by='importance', ascending=False)

print('Gender Features by Importance:')
print(gender_importance_df)
print('Alcohol Features by Importance:')
print(alcohol_importance_df)

X_full = motorcycle_data[['age', 'sex', 'time_category', 'prov', 
                    'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]


### Partial Plots for Entire Dataset using best model - continuous PDP takes a long time to run... 
make_partial_plot(rf_random_model, X_full, 'age', file_name='./data/plot_output/age_partial_plot.png')
make_cat_partial_plot(rf_random_model, X_full, 'alcohol_status', file_name='./data/plot_output/alcohol_partial_plot.png')
make_cat_partial_plot(rf_random_model, X_full, 'sex', file_name='./data/plot_output/sex_partial_plot.png', mapping_dict={'1':'Male', '2':'Female'})
make_cat_partial_plot(rf_random_model, X_full, 'time_category', file_name='./data/plot_output/time_partial_plot.png')
prov_list = top_ten_prov['feature'].str.replace('cat__prov_', '', regex=False).tolist()
make_cat_partial_plot_subset(rf_random_model, X_full, 'prov', prov_list, file_name='./data/plot_output/prov_partial_plot.png')
# summary = compare_models(motorcycle_data, use_stored_model=using_stored_models, model_size=1.0)
# print(summary)


###
### 2023 Analysis
###

### Same situation here, if you are not using stored models, expect a couple hours to run.

# Decides to run the models using stored models or not
using_stored_models = True

# Random Grid Search and importances.
rf_2023_random_model =  random_grid_2023_model(motorcycle_data, use_stored_model=using_stored_models, model_size=0.20)
rf_2023_feature_names = rf_2023_random_model.named_steps['preprocessor'].get_feature_names_out()
rf_2023_importances = rf_2023_random_model.named_steps['classifier'].feature_importances_
importance_2023_df = pd.DataFrame({'feature': rf_2023_feature_names, 'importance': rf_2023_importances})
importance_2023_df = importance_2023_df.sort_values(by='importance', ascending=False)

# Bayesian Search and importances
# rf_bayesian_2023_model = model_2023_w_bayesian(motorcycle_data, use_stored_model=using_stored_models, model_size=0.20)
# rf_2023_feature_names = rf_bayesian_2023_model.named_steps['preprocessor'].get_feature_names_out()
# rf_2023_importances = rf_bayesian_2023_model.named_steps['classifier'].feature_importances_
# importance_2023_df = pd.DataFrame({'feature': rf_2023_feature_names, 'importance': rf_2023_importances})
# importance_2023_df = importance_2023_df.sort_values(by='importance', ascending=False)
# print(importance_2023_df)

# Get the feature names from the preprocessor and the feature importances from the classifier.
importance_df = importance_2023_df.copy()

# Select the top 10 features.
top10_features = importance_2023_df.head(10)
print(top10_features)
# Filter for province features (assuming province features contain 'cat__prov_' in their names).
prov_importance_df = importance_2023_df[importance_2023_df['feature'].str.contains('cat__prov_')].copy()

# Sort the province features by importance (already sorted overall, but ensure in case).
prov_importance_df = prov_importance_df.sort_values(by='importance', ascending=False)

# Get the top ten province features.
top_ten_prov = prov_importance_df.head(10)

print('Top Ten Province Features by Importance:')
print(top_ten_prov)

gender_importance_df = importance_df[importance_df['feature'].str.contains('cat__sex_')].copy()
gender_importance_df = gender_importance_df.sort_values(by='importance', ascending=False)
alcohol_importance_df = importance_df[importance_df['feature'].str.contains('cat__alcohol_status_')].copy()
alcohol_importance_df = alcohol_importance_df.sort_values(by='importance', ascending=False)


# Partial Plots for 2023 using best model
X_full = motorcycle_data[['age', 'sex', 'time_category', 'prov', 
                    'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
   
make_partial_plot(rf_2023_random_model, X_full, 'age', file_name='./data/plot_output/age_2023_partial_plot.png')
make_cat_partial_plot(rf_2023_random_model, X_full, 'alcohol_status', file_name='./data/plot_output/alcohol_2023_partial_plot.png')
make_cat_partial_plot(rf_2023_random_model, X_full, 'sex', file_name='./data/plot_output/sex_2023_partial_plot.png', mapping_dict={'1':'Male', '2':'Female'})
make_cat_partial_plot(rf_2023_random_model, X_full, 'time_category', file_name='./data/plot_output/time_2023_partial_plot.png')
prov_list = top_ten_prov['feature'].str.replace('cat__prov_', '', regex=False).tolist()
make_cat_partial_plot_subset(rf_2023_random_model, X_full, 'prov', prov_list, file_name='./data/plot_output/prov_2023_partial_plot.png')


