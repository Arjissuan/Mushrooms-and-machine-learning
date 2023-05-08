import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataSource:
    def __init__(self):
        self.primary_data_path = "../files/PrimaryData/primary_data_generated.csv"
        self.secondary_data_path = "../files/SecondaryData/secondary_data_generated.csv"

    def get_primary_data_frame(self):
        return pd.read_csv(self.primary_data_path, delimiter=";")

    def get_secondary_data_frame(self):
        return pd.read_csv(self.secondary_data_path, delimiter=";")

    def exchange_nones_to_value(self, data_frame, new_value='v'):
        funk = lambda x: data_frame[x].replace(np.nan, new_value)
        new_df = list(map(funk, data_frame.columns))
        return pd.DataFrame(new_df).T

    def train_test_data(self, data_frame, size=0.8):
        edible = data_frame.query('clas=="e"',inplace=False)
        poisonous = data_frame.query('clas=="p"',inplace=False)
        edible = pd.DataFrame(data=edible.values, columns=edible.columns)
        poisonous = pd.DataFrame(data=poisonous.values, columns=poisonous.columns)
        tot_poi_le = int(len(poisonous)*size)
        tot_ed_le = int(len(edible)*size)

        training = pd.concat([poisonous.loc[0:tot_poi_le-1,:],
                              edible.loc[0:tot_ed_le-1, :]
                              ], ignore_index=True)

        testing = pd.concat([poisonous.loc[tot_poi_le:len(poisonous),:],
                            edible.loc[tot_ed_le:len(edible), :]
                             ], ignore_index=True)

        return {"Test_y":testing['clas'], "Test_x":testing.drop(columns=['clas']),
                "Train_y":training['clas'], "Train_x":training.drop(columns=['clas'])}

    def data_for_test_train_fromsci(self, dataframe, size):
        x_train, x_test, y_train, y_test = train_test_split(dataframe.drop(columns=['clas']),
                                                            dataframe['clas'],
                                                            test_size=size,
                                                            random_state=42,
                                                            )

        return x_train, y_train, x_test, y_test



    def edible_percent(self, df, row='t'): #should be done data exploring class? Or it will redundant
        perc = lambda x: (x, (df[x][row]*100)/np.sum(df[x]))
        df_perc = dict(map(perc, df.columns))
        return pd.DataFrame(index=['t%_of_sum'], data=df_perc)

    def exchange_str_to_ints(self, df):
        cols = df.drop(['clas',"cap-diameter", 'stem-width', 'stem-height'], axis=1 ).columns
        sets = list(map(lambda x:tuple(set(df[x])), cols))
        new_sets = []
        for s in sets:
            lil_set = []
            for i in range(len(s)):
                lil_set.append(i)
            new_sets.append(lil_set)
        func = lambda x: df[cols[x]].replace(to_replace=sets[x], value=new_sets[x])
        #print(list(zip(sets,new_sets,cols)))
        new_df = list(map(func, range(len(cols))))
        new_df = pd.DataFrame(data=new_df.copy()).T
        fresh_df = pd.concat([df.loc[:, ['clas',"cap-diameter", 'stem-width', 'stem-height']], new_df], axis=1)
        return fresh_df

    def exchange_str_to_vect(self, df):
        cols = df.drop(['clas',"cap-diameter", 'stem-width', 'stem-height'], axis=1).columns
        sets = list(map(lambda x: tuple(set(df[x])), cols))
        new_sets = []






