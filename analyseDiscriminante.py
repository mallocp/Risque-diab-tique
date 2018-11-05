import numpy as np
import matplotlib.pyplot as plt
import preprocessingData as pr
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold,cross_val_score

class analyseDiscriminante:
    def __init__(self):
        self.preproc = pr.preprocessingData()
        self.preproc.standardizeRobData()
        self.preproc.getkernelacpReduction()
        self.dataPCA = self.preproc.dataPCA
        self.labels = self.preproc.labels

    def trainingAnalyse(self):
        self.score_acc = 0.80
        self.nlayer = 0
        for i in range(10000):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.preproc.data.values[0:768, 0:8], self.labels, test_size=0.3)
            self.analyse  = LinearDiscriminantAnalysis()
            self.analyse.fit(self.X_train, self.y_train)
            self.trainingScore = self.analyse.score(self.X_train, self.y_train)
            self.accuracyScore = self.analyse.score(self.X_test, self.y_test)
            if (self.accuracyScore > self.trainingScore and self.accuracyScore >= self.score_acc):
                self.score_acc = self.accuracyScore
                print(self.score_acc)

    def frontieredeDecision(self):
        cmp = np.array(['r', 'g'])
        #plt.scatter(self.X_train[:,0],self.X_train[:,1],c=cmp[self.y_train],s=5,edgecolors='none')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='none', s=5, edgecolors=cmp[self.y_test])
        nx, ny = 400, 400
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = self.analyse.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        plt.show()

    def frontieredeDecision2(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        cmp = np.array(['r', 'g'])
        ax.scatter(self.X_test[:, 0], self.X_test[:, 1], self.X_test[:, 2], c=cmp[self.y_test], s=5, cmap=plt.cm.coolwarm)
        nx, ny = 400, 400
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = self.analyse.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.show()

    def crossingValidation(self):
        pipeline = make_pipeline(LinearDiscriminantAnalysis())
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        cv_results = cross_val_score(pipeline, self.dataPCA[:, [0, 1]], self.labels, cv=kf)
        return cv_results.mean()
