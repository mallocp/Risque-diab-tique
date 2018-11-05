import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import preprocessingData as pr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score


class treeForest:
    def __init__(self):
        self.preproc = pr.preprocessingData()
        self.preproc.getkernelacpReduction()
        self.dataPCA = self.preproc.dataPCA
        self.labels = self.preproc.labels
        self.n = 30
        self.accuracy = np.zeros(self.n)
        self.training = np.zeros(self.n)

    def trainingbytreeClassifier(self):
        self.score_acc =0
        self.nlayer =0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.preproc.data.values[0:768, 0:8], self.labels, test_size=0.3)
        for i in range(self.n):
            self.treeforest = make_pipeline(RandomForestClassifier(max_depth=3, n_estimators=(i + 1) * 10))
            self. treeforest.fit(self.X_train, self.y_train)
            self.training[i] = self.treeforest.score(self.X_train, self.y_train)
            self.accuracy[i] = self.treeforest.score(self.X_test, self.y_test)
            if (self.accuracy[i] > self.training[i] and self.accuracy[i] > self.score_acc):
                self.score_acc = self.accuracy[i]
               # self.nlayer = self.treeforest.n_estimators

        plt.plot([10 * (i + 1) for i in range(30)], self.training, 'b')
        plt.plot([10 * (i + 1) for i in range(30)], self.accuracy, 'r')
        plt.title("score Training vs score Accuracy")
        plt.ylabel("score")
        plt.xlabel("Nombre Echantillon")
        plt.grid(True)
        plt.show()



    def frontieredeDecision(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataPCA[:, [0, 1]], self.labels,test_size=0.3)
        self.treeforest = RandomForestClassifier(max_depth=3, n_estimators=200)
        self.treeforest.fit(self.X_train, self.y_train)
        self.treeforest.score(self.X_train, self.y_train)
        self.treeforest.score(self.X_test, self.y_test)
        cmp = np.array(['r', 'g'])
        #plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=cmp[self.y_train], s=5, edgecolors='none')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='none', s=5, edgecolors=cmp[self.y_test])
        nx, ny = 400, 400
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z =  self.treeforest.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        plt.show()


    def crossingValidationforest(self):
            pipeline = make_pipeline(RandomForestClassifier(max_depth=3, n_estimators=200))
            kf = KFold(n_splits=10, shuffle=True, random_state=1)
            self.cv_results = cross_val_score(pipeline, self.preproc.data.values[0:768, 0:8], self.labels, cv=kf, scoring="accuracy")
            self.cv_results2 = self.cv_results.mean()

            """target_probabilities = pipeline.fit(self.X_train, self.y_train).predict_proba(self.X_test)[:, 1]

            false_positive_rate, true_positive_rate, threshold = roc_curve(self.y_test, target_probabilities)
            plt.title("Receiver Operating Characteristic")
            plt.plot(false_positive_rate, true_positive_rate)
            plt.plot([0, 1], ls="--")
            plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.show()
            print(roc_auc_score(self.y_test, target_probabilities))"""

