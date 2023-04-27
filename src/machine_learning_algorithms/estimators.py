from sklearn import svm

class Linear_SVC:
    def estimate(self, test_x, test_y, train_x, train_y):
        clf = svm.SVC()
        clf.fit(X=train_x, y=train_y)
        estimation = clf.predict(X=test_x)
        return estimation

