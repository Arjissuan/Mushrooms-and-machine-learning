import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit


class DataSource:
    def __init__(self):
        self.primary_data_path = "../files/PrimaryData/primary_data_edited.csv"
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

    def train_test_data(self, data_frame, size=0.8, clas="clas"):
        edible = data_frame.query(f'{clas}=="e"', inplace=False)
        poisonous = data_frame.query(f'{clas}=="p"', inplace=False)
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

    def data_for_test_train_fromsci(self, dataframe, size=0.8, r_state=42, clas='clas'):
        x_train, x_test, y_train, y_test = train_test_split(dataframe.drop(columns=[clas]),
                                                            dataframe[clas],
                                                            test_size=size,
                                                            random_state=r_state,
                                                            ) #42
        return x_train, y_train, x_test, y_test

    # finding percentage of mushroms labeled as edible
    # should be done data exploring class? Or it will be redundant
    def edible_percent(self, index, columns, row='e'):
        if row=="e":
            cross = pd.crosstab(index=index, columns=columns)
            funk = lambda x: (x, (cross[x][row] * 100 / np.sum(cross[x])))
            percentage = dict(map(funk, cross.columns))
            return pd.DataFrame(index=["edible"], data=percentage)
        elif row=="p":
            cross = pd.crosstab(index=index, columns=columns)
            funk = lambda x: (x, (cross[x][row] * 100 / np.sum(cross[x])))
            percentage = dict(map(funk, cross.columns))
            return pd.DataFrame(index=["poisonous"], data=percentage)
        else:
            cross = pd.crosstab(index=index, columns=columns)
            funk = lambda x: (x, (cross[x][row] * 100 / np.sum(cross[x])))
            percentage = dict(map(funk, cross.columns))
            return pd.DataFrame(index=[row], data=percentage)

    # changing categorical values from dataframe ine
    def exchange_str_to_ints(self, *df, cols_to_pass=('clas', "cap-diameter", 'stem-width', 'stem-height')):
        if df == ():
            df = self.exchange_nones_to_value()
        else:
            df = df[0]
        print(type(df))
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
            df = self.exchange_str_to_ints()
        else:
            df = df[0]

        size = 0
        cross = []
        while len(df) > size:
            last_size = size
            size += np.round(len(df) / number, decimals=0).astype(int)
            if size == len(df) - 1:
                size += 1
            part_df = df.iloc[last_size:size, :]
            cross.append(part_df)
            #print(last_size, size, len(part_df))
        indexes = np.array(range(number))
        cross_dict = dict(zip(indexes, cross))
        data_kits = []
        for i in cross_dict.keys():
            xtrain = pd.concat(list(map(lambda x: cross_dict.get(x), indexes[indexes != i]))).drop(columns=['clas'])
            ytrain = pd.concat(list(map(lambda x: cross_dict.get(x), indexes[indexes != i])))['clas']
            xtest = cross_dict.get(i).drop(columns=['clas'])
            ytest = cross_dict.get(i)['clas']
            data_kits.append((xtest, ytest, xtrain, ytrain))
        return data_kits

    def cross_vali_Kfold(self, *df, number):
        if df == ():
            df = self.exchange_str_to_ints()
        else:
            df = df[0]

        kf = KFold(n_splits=number)
        new_df = df.clas
        data_kits = []
        for train, test in kf.split(new_df):
            print(f"{train} {test}")
            xtrain = df.iloc[train,:].drop(columns=['clas'])
            ytrain = df.iloc[train,:]["clas"]
            ytest = df.iloc[test,:]["clas"]
            xtest = df.iloc[test,:].drop(columns=["clas"])
            bufor = (xtest, ytest, xtrain, ytrain)
            data_kits.append(bufor)
        return data_kits

    def cross_vali_shuffle(self, *df, number, r_state, test_size):
        if df == ():
            df = self.exchange_str_to_ints()
            df = self.aply_one_hot_encoder(df)
        else:
            df = df[0]

        cf = StratifiedShuffleSplit(n_splits=number, random_state=r_state, test_size=test_size)
        data_kits = []
        for train, test in cf.split(df.drop(["clas"], axis=1), df.clas):
            print(f"  Train index={train}, and lengths = {len(train)}")
            print(f"  Test index={test}, and lengths= {len(test)}")
            xtest = df.iloc[test,:].drop(columns=['clas'])
            ytest = df.iloc[test,:]['clas']
            xtrain = df.iloc[train, :].drop(columns=['clas'])
            ytrain = df.iloc[train, :]['clas']
            bufor = (xtest, ytest, xtrain, ytrain)
            data_kits.append(bufor)
        return data_kits

    def data_merge(self, *df):
        if df == ():
            df1 = self.get_primary_data_frame()
            df2 = self.get_secondary_data_frame()
            numeric_cols = df2.select_dtypes(include=np.number).columns.tolist()

            df1 = self.exchange_nones_to_value(df1)
            df2 = self.exchange_nones_to_value(df2)
        else:
            df1 = df[0]
            df2 = df[1]
            numeric_cols = df2.select_dtypes(include=np.number).columns.tolist()

        numeric_min = lambda x: list(map(float, x.strip('[]][').split(",")))[0]
        numeric_max = lambda x: list(map(float, x.strip("[]][").split(",") ))[-1]

        numeric_is_in = lambda y, z: True if y <= df2[col][indx] <= z else False
        is_in = lambda y: True if df2[col][indx] in y else False

        family = []
        name = []
        for indx in range(len(df2)):
            bufor = []
            for col in df2.columns:
                vector = []
                if col in numeric_cols:
                    min = list(map(lambda x: numeric_min(x), df1[col]))
                    max = list(map(lambda x: numeric_max(x), df1[col]))
                    vector.append(list(map(numeric_is_in, min, max)))
                elif col not in numeric_cols:
                    vector.append(list(map(is_in, df1[col])))
                bufor.append(vector)
            indx_array = np.array(bufor).T
            try:
                which_one = np.where(np.sum(indx_array, axis=2)[:, 0] == 21)[0][0]
                family.append(df1["family"][which_one])
                name.append(df1["name"][which_one])
            except IndexError:
                family.append(np.nan)
                name.append(np.nan)
            print(indx)

        df2["family"] = family
        df2["name"] = name
        return df2



