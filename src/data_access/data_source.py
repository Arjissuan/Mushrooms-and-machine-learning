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

    def exchange_nones_to_value(self, data_frame, new_value='f'):
        funk = lambda x: data_frame[x].replace(np.nan, new_value)
        new_df = list(map(funk, data_frame.columns))
        return pd.DataFrame(new_df).T

