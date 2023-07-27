from src.data_access.data_source import DataSource
from src.machine_learning_algorithms.estimators import Learning_method
import pandas as pd
from matplotlib import pyplot


class Main:
    def __init__(self):
        self.ds = DataSource()

    def analasis_one(self):
        df = self.ds.get_secondary_data_frame()
        ready_df = self.ds.exchange_str_to_ints(self.ds.exchange_nones_to_value(df, new_value='v'))
        #ready_df = ds.exchange_str_to_vect(df=df)
        ready_df = self.ds.aply_one_hot_encoder(df)

        x_train, y_train, x_test, y_test = self.ds.data_for_test_train_fromsci(ready_df, 0.2, r_state=42)
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
        return whole_data

    def analasis_two(self):
        #df = ds.get_secondary_data_frame()
        #ready_df = ds.exchange_str_to_ints()
        ready_df = self.ds.aply_one_hot_encoder()
        ready_df = self.ds.cross_validation(ready_df, number=5)
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

    def analasis_three(self):
        ready_df = self.ds.aply_one_hot_encoder()
        ready_df = self.ds.cross_vali_Kfold(ready_df, number=5)
        result_df = pd.DataFrame(columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
        for i, data in enumerate(ready_df):
            etimator = Learning_method(data[0], data[1], data[2], data[3])
            pred1 = etimator.DecisionTree()
            pred2 = etimator.rand_forest()
            bufor = pd.DataFrame(data=[
                etimator.generalization(pred1, "Decision_Tree"),
                etimator.generalization(pred2, "Rand_forest")
            ], columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
            result_df = pd.concat([result_df, bufor], ignore_index=True)
        print(result_df)
        print(etimator.cross_validation_means(result_df))

    def analasis_four(self):
        ready_df = self.ds.cross_vali_shuffle(number=5, r_state=1, test_size=0.2)
        result_df = pd.DataFrame(columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
        for i, data in enumerate(ready_df):
            estimator = Learning_method(data[0], data[1], data[2], data[3])
            pred1 = estimator.DecisionTree()
            pred2 = estimator.rand_forest()
            pred3 = estimator.support_vector_machines()
            bufor = pd.DataFrame(data=[
                estimator.generalization(pred1, "Decision_Tree"),
                estimator.generalization(pred2, "Rand_forest"),
                estimator.generalization(pred3, "SVM")
            ], columns=["Accuracy", "TN", "TP", "FN", "FP", "Estimator"])
            result_df = pd.concat([result_df, bufor], ignore_index=True)
        #print(result_df)
        # print(estimator.cross_validation_means(result_df))
        general = estimator.cross_validation_means(result_df)
        clas_eficien = estimator.clasification_eficiency(general)
        whole_data = pd.concat([general, clas_eficien], axis=1)
        print(whole_data)
        return whole_data

    def analisis_five(self):
        ds = DataSource()
        df = ds.exchange_nones_to_value()
        df = df.drop(columns=["cap-diameter", "stem-width", "stem-height"], axis=1)
        print(df)
        ready_df = ds.exchange_str_to_ints(df)
        x_train, y_train, x_test, y_test = ds.data_for_test_train_fromsci(ready_df, 0.2, r_state=45)
        estimator = Learning_method(x_test, y_test, x_train, y_train)
        xtrain_fs, xtest_fs, fs = estimator.Feature_selection()
        names = fs.get_feature_names_out()
        for item in range(len(fs.scores_)):
            print(f"Feature {names[item]}: {fs.scores_[item]}")

        pyplot.figure()
        pyplot.bar([ix for ix in range(len(fs.scores_))], fs.scores_)
        pyplot.show()


if __name__ == '__main__':
    #Main().analasis_one().to_csv(sep="\t", path_or_buf="./Analiza.csv")
    #Main().analasis_four().to_csv(sep="\t", path_or_buf="./Analiza_cross_validacji.csv")


    #ds.data_merge()