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

    def exchange_nones_to_value(self, *df, new_value='v'):
        if df == ():
            df = self.get_secondary_data_frame()
        else:
            df = df[0]
        funk = lambda x: df[x].replace(np.nan, new_value)
        new_df = list(map(funk, df.columns))
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

    def data_for_test_train_fromsci(self, dataframe, size, r_state=42):
        x_train, x_test, y_train, y_test = train_test_split(dataframe.drop(columns=['clas']),
                                                            dataframe['clas'],
                                                            test_size=size,
                                                            random_state=r_state,
                                                            ) #42
        return x_train, y_train, x_test, y_test

    # finding percentage of mushroms labeled as edible
    # should be done data exploring class? Or it will be redundant
    def edible_percent(self, index, columns, row='e'):
        cross = pd.crosstab(index=index, columns=columns)
        funk = lambda x: (x, (cross[x][row] * 100 / np.sum(cross[x])))
        percentage = dict(map(funk, cross.columns))
        return pd.DataFrame(index=["e%"], data=percentage)

    # changing categorical values from dataframe ine
    def exchange_str_to_ints(self, *df, cols_to_pass=('clas', "cap-diameter", 'stem-width', 'stem-height')):
        if df == ():
            df = self.get_secondary_data_frame()
        else:
            df = df[0]

        cols = df.drop([col for col in cols_to_pass], axis=1).columns
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

    # binarization of ceterogical values into 'binary' vectors
    def exchange_str_to_vect(self, *df, cols_to_pass=('clas', "cap-diameter", 'stem-width', 'stem-height')):
        if df == ():
            df = self.get_secondary_data_frame()
        else:
            df = df[0]
        cols = df.drop([col for col in cols_to_pass], axis=1).columns
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

    # Function returns same result as one hot encoder, each category(column with multiple values) is converted into
    # column. One value from category into one column. They are asigned 1 or 0.
    def aply_one_hot_encoder(self, *df, cols_to_pass=('clas', "cap-diameter", 'stem-width', 'stem-height')):
        if df == ():
            df = self.get_secondary_data_frame()
        else:
            df = df[0]

        cols = df.drop([col for col in cols_to_pass], axis=1).columns
        values = list(map(lambda x: tuple(set(df[x])), cols))
        #print(1*np.logical_not(np.array(df[cols[0]]) != values[0][0]))
        new_df = pd.DataFrame()
        for index, col in enumerate(cols):
            data_funk = lambda x: 1*np.logical_not(np.array(df[col]) != x)
            data_vects = list(map(data_funk, values[index]))
            bufor_df = pd.DataFrame(data=data_vects, index=["{}_{}".format(col, str(j)) for j in values[index]]).T
            new_df = pd.concat([new_df, bufor_df], axis=1)
        new_df = pd.concat([df.loc[:, [col for col in cols_to_pass]], new_df], axis=1)
        #print(new_df)
        return new_df

    def cross_validation(self, *df, number):
        if df == ():
            df = self.get_secondary_data_frame()
        else:
            df = df[0]
        size = 0
        cross = []
        while len(df) > size:
            last_size = size
            size += int(len(df) / number)
            if size == len(df) - 1:
                size += 1
            part_df = df.iloc[last_size:size, :]
            cross.append(part_df)
            print(len(part_df))
        indexes = np.array(range(number))
        cross_dict = dict(zip(indexes, cross))
        zestawy = []
        for i in cross_dict.keys():
            xtrain = pd.concat(list(map(lambda x: cross_dict.get(x), indexes[indexes != i]))).drop(columns=['clas'])
            ytrain = pd.concat(list(map(lambda x: cross_dict.get(x), indexes[indexes != i])))['clas']
            xtest = cross_dict.get(i).drop(columns=['clas'])
            ytest = cross_dict.get(i)['clas']
            zestawy.append((xtest, ytest, xtrain, ytrain))
        return zestawy
