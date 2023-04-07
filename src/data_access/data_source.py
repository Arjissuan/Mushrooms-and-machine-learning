import numpy as np
import pandas as pd
import os

class DataSource:
    def __init__(self):
        self.primary_data_path="../files/PrimaryData/primary_data_generated.csv"
        self.secondary_data_path="../files/SecondaryData/secondary_data_generated.csv"

    def get_primary_data_frame(self):
        return pd.read_csv(self.primary_data_path, delimiter=";")

    def get_secondary_data_frame(self):
        return pd.read_csv(self.secondary_data_path, delimiter=";")

    def exchange_nones_to_false(self, df):
        new_df = pd.DataFrame()
        for col in df.columns:
            vect = np.array(df[col]).astype(str)
            vect[np.isnan(vect)] = 0
            new_df = pd.concat([new_df, {col:vect}], axis=1)
        return new_df


