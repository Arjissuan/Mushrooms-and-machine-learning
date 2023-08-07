import numpy
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2

class Learning_method:
    def __init__(self, test_x, test_y, train_x, train_y):
        self.test_x = test_x
        self.test_y = test_y
        self.train_x = train_x
        self.train_y = train_y

    def support_vector_machines(self):
        clf = svm.SVC()
        clf.fit(X=self.train_x, y=self.train_y)
        return clf.predict(X=self.test_x)

    def rand_forest(self, n=10):
        lnm = RandomForestClassifier(n_estimators=n)
        lnm.fit(self.train_x,self.train_y)
        return lnm.predict(self.test_x)

    def DecisionTree(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(X=self.train_x, y=self.train_y)
        return clf.predict(X=self.test_x)

    def Bayas(self):
        gnb = GaussianNB()
        gnb.fit(X=self.train_x,y=self.train_y)
        return gnb.predict(X=self.test_x)

    def network(self, layers=(4,2), activ_func="relu"):
        est = MLPClassifier(hidden_layer_sizes=layers, activation=activ_func)
        est.fit(X=self.train_x, y=self.train_y)
        return est.predict(self.test_x)

    def generalization(self, estimated_y, used_estimator, pred_val="e", Het_pred_val="p"):
        est = np.asarray(estimated_y)
        test = pd.Series.to_numpy(self.test_y)
        numb_of_good = np.count_nonzero(est == test)
        accuracy = np.divide(np.multiply(numb_of_good,100), len(test))

        TP = np.sum(np.logical_and(np.isin(test, pred_val), np.isin(est, pred_val)))
        TN = np.sum(np.logical_and(np.isin(test, Het_pred_val), np.isin(est, Het_pred_val)))
        FN = np.sum(np.logical_and(np.isin(test, pred_val), np.isin(est, Het_pred_val)))
        FP = np.sum(np.logical_and(np.isin(test, Het_pred_val),np.isin(est, pred_val))) #want as low as possible

        return [accuracy, TN, TP, FN, FP, used_estimator]

    def cross_validation_means(self, df):
        column = df.drop(columns=["Estimator"], axis=1).columns
        estimators = tuple(set(df.Estimator))
        means = lambda x: (x, np.mean(df[df.Estimator == x][column[i]].astype(float)))
        values = [[] for item in estimators]
        for i in range(len(column)):
            bufo = dict(map(means, estimators))
            for j in range(len(values)):
                values[j].append(bufo[estimators[j]])
        for indx in range(len(estimators)):
            values[indx].append(estimators[indx])
        new_df = pd.DataFrame(data=np.array(values), columns=df.columns)
        return new_df

    def clasification_eficiency(self, array):
        #print(array)
        sensitivity = array["TP"].astype(np.float64)/(array["TP"].astype(np.float64)+array["FN"].astype(numpy.float64))
        specificity = array["TN"].astype(np.float64)/(array["TN"].astype(np.float64)+array["FP"].astype(np.float64))
        PPV = array["TP"].astype(np.float64)/(array["TP"].astype(np.float64)+array["FP"].astype(np.float64))
        NPV = array["TN"].astype(np.float64)/(array["TN"].astype(np.float64)+array["FN"].astype(np.float64))
        return pd.DataFrame({"sensitivity":sensitivity, "specificity":specificity, "PPV":PPV, "NPV":NPV})

    def Feature_selection(self):
        fs = SelectKBest(score_func=chi2, k='all')
        fs.fit(self.train_x, self.train_y)
        Xtrain_fs = fs.transform(self.train_x)
        Xtest_fs = fs.transform(self.test_x)
        return Xtest_fs, Xtrain_fs, fs
