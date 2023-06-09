import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

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

    def generalization(self, estim_clas, used_estimator):
        est = np.asarray(estim_clas)
        test = pd.Series.to_numpy(self.test_y)
        numb_of_good = np.count_nonzero(est == test)
        accuracy = np.divide(np.multiply(numb_of_good,100), len(test))
        TN = np.sum(np.logical_and(self.test_y == "e", estim_clas == "e"))
        TP = np.sum(np.logical_and(self.test_y == "p", estim_clas == "p"))
        FN = np.sum(np.logical_and(self.test_y == "e", estim_clas == "p")) #want as low as possible
        FP = np.sum(np.logical_and(self.test_y == "p", estim_clas == "e"))
        return [accuracy, TN, TP, FN, FP, used_estimator]

    def cross_validation_means(self, df):
        estimators = tuple(set(df.Estimator))
        print(estimators)
        means = lambda x: (x, np.mean(df[df.Estimator == x].Accuracy.astype(float)))
        values = dict(map(means, estimators))
        return values

    def clasification_eficiency(self, array):
        sensitivity = array["TP"]/(array["TP"]+array["FN"])
        specificity = array["TN"]/(array["TN"]+array["FP"])
        PPV = array["TP"]/(array["TP"]+array["FP"])
        NPV = array["TN"]/(array["TN"]+array["FN"])
        return pd.DataFrame({"sensitivity":sensitivity, "specificity":specificity, "PPV":PPV, "NPV":NPV})


