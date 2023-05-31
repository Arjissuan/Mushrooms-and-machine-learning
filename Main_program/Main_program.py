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

    #ready_df = ds.exchange_str_to_ints(ds.exchange_nones_to_value(df, new_value='v'))
    #ready_df = ds.exchange_str_to_vect(df=df)
    #ready_df = ds.aply_one_hot_encoder(df)
    #
    # x_train, y_train, x_test, y_test = ds.data_for_test_train_fromsci(ready_df, 0.2)
    # # x_test_copy = x_test.loc[:,['habitat', 'season', 'cap-color', 'cap-shape', 'stem-color']]
    # # x_train_copy = x_train.loc[:,['habitat', 'season', 'cap-color', 'cap-shape','stem-color']]
    #
    # est = Learning_method(x_test, y_test, x_train, y_train)
    # prediction = est.Bayas()
    # prediction2 = est.DecisionTree()
    # prediction3 = est.rand_forest()
    # prediction4 = est.support_vector_machines()
    # prediction5 = est.network()
    # print(est.generalization(prediction), ' Naive Bayas')
    # print(est.generalization(prediction2), ' Decision Tree')
    # print(est.generalization(prediction3), ' Random Forest')
    # print(est.generalization(prediction4), ' SVM')
    # print(est.generalization(prediction5), ' Neural Network')

    ready_df = ds.exchange_str_to_ints()
    ready_df = ds.cross_validation(ready_df, number=12)
    arrey = np.ndarray(shape=(1,6))
    for i, data in enumerate(ready_df):
        estim = Learning_method(data[0], data[1], data[2], data[3])
        prediction = estim.Bayas()
        prediction2 = estim.DecisionTree()
        prediction3 = estim.rand_forest()
        prediction4 = estim.support_vector_machines()
        prediction5 = estim.network()
        bufor = np.concatenate(
            (estim.generalization(prediction, "Bayas"),
             estim.generalization(prediction2, "Decision_Tree"),
             estim.generalization(prediction3, "Random_Forest"),
             estim.generalization(prediction4, "SVM"),
             estim.generalization(prediction5, "Neural_Network")),
            axis=0)
        arrey = np.concatenate((arrey, bufor))
    print(arrey)
    result_df = pd.DataFrame(data=arrey, columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
    print(result_df)
    means = lambda x: (np.mean(result_df.query("used_estimator == @x")["Accuracy"]), x)

