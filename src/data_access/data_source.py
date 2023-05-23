import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


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
        edible = data_frame.query('clas=="e"', inplace=False)
        poisonous = data_frame.query('clas=="p"', inplace=False)
        edible = pd.DataFrame(data=edible.values, columns=edible.columns)
        poisonous = pd.DataFrame(data=poisonous.values, columns=poisonous.columns)
        tot_poi_le = int(len(poisonous) * size)
        tot_ed_le = int(len(edible) * size)

        training = pd.concat([poisonous.loc[0:tot_poi_le - 1, :],
                              edible.loc[0:tot_ed_le - 1, :]
                              ], ignore_index=True)

        testing = pd.concat([poisonous.loc[tot_poi_le:len(poisonous), :],
                             edible.loc[tot_ed_le:len(edible), :]
                             ], ignore_index=True)

        return {"Test_y": testing['clas'], "Test_x": testing.drop(columns=['clas']),
                "Train_y": training['clas'], "Train_x": training.drop(columns=['clas'])}

    def data_for_test_train_fromsci(self, dataframe, size):
        x_train, x_test, y_train, y_test = train_test_split(dataframe.drop(columns=['clas']),
                                                            dataframe['clas'],
                                                            test_size=size,
                                                            random_state=42,
                                                            )

        return x_train, y_train, x_test, y_test

    # finding percentage of mushroms labeled as edible
    # should be done data exploring class? Or it will be redundant
    def edible_percent(self, index, columns, row='e'):
        cross = pd.crosstab(index=index, columns=columns)
        funk = lambda x: (x, (cross[x][row] * 100 / np.sum(cross[x])))
        percentage = dict(map(funk, cross.columns))
        return pd.DataFrame(index=["e%"], data=percentage)

    # changing non-numerical values from dataframe ine
    def exchange_str_to_ints(self, df, cols_to_pass=('clas', "cap-diameter", 'stem-width', 'stem-height')):
        cols = df.drop(cols_to_pass, axis=1).columns
        sets = list(map(lambda x: tuple(set(df[x])), cols))
        new_sets = []
        for s in sets:
            lil_set = []
            for i in range(len(s)):
                lil_set.append(i)
            new_sets.append(lil_set)
        func = lambda x: df[cols[x]].replace(to_replace=sets[x], value=new_sets[x])
        # print(list(zip(sets,new_sets,cols)))
        new_df = list(map(func, range(len(cols))))
        new_df = pd.DataFrame(data=new_df.copy()).T
        fresh_df = pd.concat([df.loc[:, ['clas', "cap-diameter", 'stem-width', 'stem-height']], new_df], axis=1)
        return fresh_df

    # binarization of non-numerical values into 'binary' vectors
    def exchange_str_to_vect(self, df, cols_to_pass=('clas', "cap-diameter", 'stem-width', 'stem-height')):
        cols = df.drop(cols_to_pass, axis=1).columns
        sets = list(map(lambda x: tuple(set(df[x])), cols))
        new_set = []
        for tank in sets:
            vect = []
            for i in range(len(tank)):
                value = list(np.zeros(len(tank), dtype=int))
                value[i] = 1
                vect.append(''.join(str(i) for i in value))
            new_set.append(vect)

        gener = lambda x: df[cols[x]].replace(to_replace=sets[x], value=new_set[x])
        new_df = list(map(gener, range(len(cols))))
        new_df = pd.DataFrame(data=new_df.copy()).T
        new_df = new_df.applymap(lambda x: list(pd.Series(x.split("0")).replace(to_replace="", value=0)))
        fresh_df = pd.concat([df.loc[:, ['clas', "cap-diameter", 'stem-width', 'stem-height']], new_df], axis=1)
        return fresh_df

    def aply_one_hot_encoder(self, df):
        cols = df.drop(['clas', "cap-diameter", 'stem-width', 'stem-height'], axis=1).columns
        ohe = OneHotEncoder()
        for column in cols:
            # df[column] = ohe.fit_transform(X=df[column].to_numpy().reshape(-1,1))
            wartosc = ohe.fit_transform(X=df[[column]])
        print(wartosc)
        return df
