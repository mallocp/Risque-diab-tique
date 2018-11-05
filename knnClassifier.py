import numpy as np
import matplotlib.pyplot as plt
import preprocessingData as pr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

class knnClassifier:
    def __init__(self):
        self.preproc = pr.preprocessingData()
        self.preproc.getkernelacpReduction()
        self.dataPCA = self.preproc.dataPCA
        self.labels = self.preproc.labels


    def trainingAnalyse(self):
        self.score_acc = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataPCA[:, [0, 1]], self.labels, test_size=0.3)
        self.neigh = KNeighborsClassifier(n_neighbors=8)
        self.neigh.fit(self.X_train, self.y_train)
        self.trainingScore = self.neigh.score(self.X_train, self.y_train)
        self.accuracyScore = self.neigh.score(self.X_test, self.y_test)

    def frontieredeDecision(self):
        cmp = np.array(['r', 'g'])
        #plt.scatter(self.X_train[:,0],self.X_train[:,1],c=cmp[self.y_train],s=5,edgecolors='none')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='none', s=5, edgecolors=cmp[self.y_test])
        nx, ny = 400, 400
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = self.neigh.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        plt.show()

    def crossingValidationknn(self):
        tuned_parameters = {'n_neighbors': [1,3,5,8,10,13,15]}
        self.clf2 = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=10, verbose=0)
        self.clf2.fit(self.preproc.data.values[0:768, 0:8], self.labels)
        self.param =  self.clf2.best_params_
        pipeline = make_pipeline(KNeighborsClassifier(n_neighbors = self.param['n_neighbors']))
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        self.cv_results = cross_val_score(pipeline, self.preproc.dataPCA, self.labels,cv=kf, scoring="accuracy")
        self.cv_results2 = self.cv_results.mean()
        """false_positive_rate, true_positive_rate, threshold = roc_curve(self.y_test, target_probabilities)
        plt.title("Receiver Operating Characteristic")
        plt.plot(false_positive_rate, true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()

        print(roc_auc_score(self.y_test,target_probabilities))"""