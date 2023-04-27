import numpy as np
from sklearn.model_selection import train_test_split
from src.data_access.data_source import DataSource
from src.machine_learning_algorithms.estimators import Linear_SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ds = DataSource()
    df = ds.get_secondary_data_frame()

    est = Linear_SVC()

    a = ds.train_test_data(data_frame=df)
    x_train, y_train, x_test, y_test = ds.data_for_test_train_fromsci(ds.exchange_nones_to_value(df,new_value='g'), 0.2)
    an = x_test.loc[:,['cap-diameter', "stem-height"]]
    b = x_train.loc[:,['cap-diameter',"stem-height"]]

    prediction = est.estimate(an, y_test,b, y_train)

    print(prediction, y_test)







