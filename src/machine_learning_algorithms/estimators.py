import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
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

    def perceptron(self, ):
        lnm = Perceptron()
        lnm.fit(self.train_x, self.train_y)
        return lnm.predict(self.test_x)

    def is_same(self, estim_clas):
        est = np.array(estim_clas)
        test = pd.Series.to_numpy(self.test_y)
        numb_of_good = np.count_nonzero(est == test)

        return np.divide(np.multiply(numb_of_good,100), len(test))




