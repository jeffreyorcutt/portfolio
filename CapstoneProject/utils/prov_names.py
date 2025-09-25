import pandas as pd
import numpy as np


#this class handles the translation of province names from Thai to English
class ProvNames:
    def __init__(self):
        self.prov_names = pd.read_csv('./Data/prov_names.csv', skiprows=0,
            index_col=0)
        self.aplace_prov = pd.read_csv('./Data/prov_ids.csv', skiprows=0,
            index_col=0)

        self.prov_names_dict = self.flatten_dict(self.prov_names)
        self.aplace_prov_dict = self.flatten_dict(self.aplace_prov)

    def flatten_dict(self, df):
        flat_dict = {}
        for index, row in df.iterrows():
            for col in df.columns:
                flat_dict[index] = row[col]
        return flat_dict

    def print_dict(self):
        print(self.prov_names_dict)
    def contains_key(self, key):
        key_exists = key in self.prov_names_dict
        #print(f'Checking key: {key}, Exists: {key_exists}')
        return key_exists
    def get_value(self, key):
        #print('returning value ' + str(self.prov_names_dict[key]))
        return str(self.prov_names_dict[key])
    def contains_id(self, key):
        key_exists = key in self.aplace_prov_dict
        return key_exists
    def get_prov_id_value(self, key):
        return str(self.aplace_prov_dict[key])
    
    def create_aplace_prov_dict(self, df, output_csv):
        # Filter out rows where 'aplace' or 'prov' is blank or NaN
        filtered_df = df.dropna(subset=['aplace', 'prov'])
        filtered_df = filtered_df[(filtered_df['aplace'] != '') & (filtered_df['prov'] != '')]

        # Create dictionary with 'aplace' as keys and 'prov' as values
        aplace_prov_dict = dict(zip(filtered_df['aplace'], filtered_df['prov']))

        # Write the dictionary to the output CSV file
        output_df = pd.DataFrame(list(aplace_prov_dict.items()), columns=['aplace', 'prov'])
        output_df.to_csv(output_csv, index=False)

        return