import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import preprocessingData as pr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline


class treeBoosting:

    def __init__(self):
        self.preproc = pr.preprocessingData()
        self.preproc.getkernelacpReduction()
        self.dataPCA = self.preproc.dataPCA
        self.labels = self.preproc.labels
        self.n = 30
        self.accuracy = np.zeros(self.n)
        self.training = np.zeros(self.n)


    def trainingbytreeClassifier(self):
        self.score_acc = 0
        self.nlayer = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.preproc.data.values[0:768, 0:8], self.labels, test_size=0.3)
        for i in range(self.n):
            self.treeboosting = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1), n_estimators=(i+1)*10, learning_rate=0.1)
            self.treeboosting.fit(self.X_train, self.y_train)
            self.training[i] = self.treeboosting.score(self.X_train, self.y_train)
            self.accuracy[i] = self.treeboosting.score(self.X_test, self.y_test)
            if (self.accuracy[i] > self.training[i] and self.accuracy[i] > self.score_acc):
                self.score_acc = self.accuracy[i]

        plt.plot([10 * (i + 1) for i in range(30)], self.training, 'b')
        plt.plot([10 * (i + 1) for i in range(30)], self.accuracy, 'r')
        plt.title("score Training vs score Accuracy")
        plt.ylabel("score")
        plt.xlabel("Nombre Echantillon")
        plt.grid(True)
        plt.show()

    def trainingbyrandomForest(self):
        self.score_acc = 0
        self.nlayer = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.preproc.data.values[0:768, 0:8],self.labels, test_size=0.3)
        for i in range(self.n):

            self.forestboosting = AdaBoostClassifier(base_estimator = RandomForestClassifier(max_depth=1),n_estimators=(i + 1) * 10, learning_rate=0.1)
            self.forestboosting.fit(self.X_train, self.y_train)
            self.training[i] = self.forestboosting.score(self.X_train, self.y_train)
            self.accuracy[i] = self.forestboosting.score(self.X_test, self.y_test)
            if (self.accuracy[i] > self.training[i] and self.accuracy[i] > self.score_acc):
                self.score_acc = self.accuracy[i]


        plt.plot([10 * (i + 1) for i in range(30)], self.training, 'b')
        plt.plot([10 * (i + 1) for i in range(30)], self.accuracy, 'r')
        plt.title("score Training vs score Accuracy")
        plt.ylabel("score")
        plt.xlabel("Nombre Echantillon")
        plt.grid(True)
        plt.show()


    def frontieredeDecisionTree(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataPCA[:, [0, 1]], self.labels,test_size=0.3)
        self.treeboosting = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1),n_estimators=200, learning_rate=0.1)
        self.treeboosting.fit(self.X_train, self.y_train)
        self.treeboosting.score(self.X_train, self.y_train)
        self.treeboosting.score(self.X_test, self.y_test)
        cmp = np.array(['r', 'g'])
        #plt.scatter(self.X_train[:,0],self.X_train[:,1],c=cmp[self.y_train],s=5,edgecolors='none')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='none', s=5, edgecolors=cmp[self.y_test])
        nx, ny = 400, 400
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = self.treeboosting.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        plt.show()


    def frontieredeDecisionForest(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataPCA[:, [0, 1]], self.labels,test_size=0.3)
        self.forestboosting = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=1),n_estimators=300, learning_rate=0.1)
        self.forestboosting.fit(self.X_train, self.y_train)
        self.forestboosting.score(self.X_train, self.y_train)
        self.forestboosting.score(self.X_test, self.y_test)

        cmp = np.array(['r', 'g'])
        #plt.scatter(self.X_train[:,0],self.X_train[:,1],c=cmp[self.y_train],s=5,edgecolors='none')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='none', s=5, edgecolors=cmp[self.y_test])
        nx, ny = 400, 400
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = self.forestboosting.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        plt.show()




    def crossingValidationtree(self):
        pipeline = make_pipeline(AdaBoostClassifier(RandomForestClassifier()))
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        cv_results = cross_val_score(pipeline, self.preproc.data.values[0:768, 0:8], self.labels, cv=kf)
        return cv_results.mean()

    def crossingValidationforest(self):
        pipeline = make_pipeline(AdaBoostClassifier(RandomForestClassifier()))
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        cv_results = cross_val_score(pipeline, self.preproc.data.values[0:768, 0:8], self.labels, cv=kf)
        return cv_results.mean()