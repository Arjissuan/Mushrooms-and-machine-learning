import numpy as np
from sklearn.model_selection import train_test_split
from src.data_access.data_source import DataSource
from src.machine_learning_algorithms.estimators import Learning_method
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ds = DataSource()
    df = ds.get_secondary_data_frame()

    # ready_df = ds.exchange_str_to_ints(ds.exchange_nones_to_value(df, new_value='v'))
    # #ready_df = ds.exchange_str_to_vect(df)
    ready_df = ds.aply_one_hot_encoder(df=df)

    x_train, y_train, x_test, y_test = ds.data_for_test_train_fromsci(ready_df, 0.2)
    # x_test_copy = x_test.loc[:,['habitat', 'season', 'cap-color', 'cap-shape', 'stem-color']]
    # x_train_copy = x_train.loc[:,['habitat', 'season', 'cap-color', 'cap-shape','stem-color']]

    est = Learning_method(x_test, y_test, x_train, y_train)
    prediction = est.Bayas()
    prediction2 = est.DecisionTree()
    prediction3 = est.rand_forest()
    prediction4 = est.support_vector_machines()
    prediction5 = est.network()
    print(est.generalization(prediction), ' Naive Bayas')
    print(est.generalization(prediction2), ' Decision Tree')
    print(est.generalization(prediction3), ' Random Forest')
    print(est.generalization(prediction4), ' SVM')
    print(est.generalization(prediction5), ' Neural Network')










