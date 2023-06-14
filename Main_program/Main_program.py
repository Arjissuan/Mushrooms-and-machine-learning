import numpy as np
from sklearn.model_selection import train_test_split
from src.data_access.data_source import DataSource
from src.machine_learning_algorithms.estimators import Learning_method
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.ds = DataSource()

    def analasis_one(self):
        df = self.ds.get_secondary_data_frame()
        ready_df = self.ds.exchange_str_to_ints(self.ds.exchange_nones_to_value(df, new_value='v'))
        #ready_df = ds.exchange_str_to_vect(df=df)
        ready_df = self.ds.aply_one_hot_encoder(df)

        x_train, y_train, x_test, y_test = self.ds.data_for_test_train_fromsci(ready_df, 0.2, r_state=45)
        # x_test_copy = x_test.loc[:,['habitat', 'season', 'cap-color', 'cap-shape', 'stem-color']]
        # x_train_copy = x_train.loc[:,['habitat', 'season', 'cap-color', 'cap-shape','stem-color']]

        est = Learning_method(x_test, y_test, x_train, y_train)
        prediction = est.Bayas()
        prediction2 = est.DecisionTree()
        prediction3 = est.rand_forest()
        prediction4 = est.support_vector_machines()
        prediction5 = est.network()
        data_generalization = pd.DataFrame(data=[
            est.generalization(prediction, ' Naive Bayas'),
            est.generalization(prediction2, ' Decision Tree'),
            est.generalization(prediction3, ' Random Forest'),
            est.generalization(prediction4, ' SVM'),
            est.generalization(prediction5, ' Neural Network')
        ], columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
        data_for_clas_efi = est.clasification_eficiency(data_generalization)
        whole_data = pd.concat([data_generalization, data_for_clas_efi], axis=1)
        print(whole_data)

    def analasis_two(self):
        ds = DataSource()
        #df = ds.get_secondary_data_frame()
        #ready_df = ds.exchange_str_to_ints()
        ready_df = ds.aply_one_hot_encoder()
        ready_df = ds.cross_validation(ready_df, number=5)
        result_df = pd.DataFrame(columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
        for i, data in enumerate(ready_df):
            print(i)
            estim = Learning_method(data[0], data[1], data[2], data[3])
            # prediction = estim.Bayas()
            prediction2 = estim.DecisionTree()
            prediction3 = estim.rand_forest(n=50)
            # prediction4 = estim.support_vector_machines()
            # prediction5 = estim.network()
            bufor = pd.DataFrame(data=[
                estim.generalization(prediction2, "Decision_Tree"),
                estim.generalization(prediction3, "Random_Forest"),
                # estim.generalization(prediction, "Bayas"),
                # estim.generalization(prediction4, "SVM"),
                # estim.generalization(prediction5, "Neural_Network")
            ], columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
            result_df = pd.concat([result_df, bufor], ignore_index=True)
        print(result_df)
        print(estim.cross_validation_means(result_df))


if __name__ == '__main__':
    # Main().analasis_one()
    ds = DataSource()
    ready_df = ds.aply_one_hot_encoder()
    ready_df = ds.cross_vali_Kfold(ready_df,number=5)
    result_df = pd.DataFrame(columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
    for i, data in enumerate(ready_df):
        etimator = Learning_method(data[0], data[1], data[2], data[3])
        pred1 = etimator.DecisionTree()
        pred2 = etimator.rand_forest()
        bufor = pd.DataFrame(data=[
            etimator.generalization(pred1, "Decision_Tree"),
            etimator.generalization(pred2, "Rand_forest")
        ],columns= ["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
        result_df = pd.concat([result_df, bufor], ignore_index=True)
    print(result_df)
    print(etimator.cross_validation_means(result_df))

    Main().analasis_two()

