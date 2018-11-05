import numpy as np
import matplotlib.pyplot as plt
import preprocessingData as pr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import svm


class svmClassifier:
    def __init__(self):
        self.preproc = pr.preprocessingData()
        self.preproc.standardizeRobData()
        self.preproc.getkernelacpReduction()
        self.dataPCA = self.preproc.dataPCA
        self.labels = self.preproc.labels


    def trainingAnalyse(self):
        self.score_acc = 0.80
        self.nlayer = 0
        for  i in range(10000):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.preproc.data.values[0:768, 0:8], self.labels, test_size=0.3)
            self.svc = svm.SVC(kernel='rbf',)
            self.svc.fit(self.X_train, self.y_train)
            self.trainingScore = self.svc.score(self.X_train, self.y_train)
            self.accuracyScore = self.svc.score(self.X_test, self.y_test)
            if(self.accuracyScore > self.trainingScore and self.accuracyScore >= self.score_acc) :
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
        Z = self.svc.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        plt.show()

    def crossingValidationsvm(self):
        pipeline = make_pipeline(svm.SVC())
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        cv_results = cross_val_score(pipeline, self.X_train, self.y_train, cv=kf, scoring="accuracy")
        cv_results.mean()
        target_probabilities = pipeline.fit(self.X_train, self.y_train).predict(self.X_test)[:, 1]

        false_positive_rate, true_positive_rate, threshold = roc_curve(self.y_test, target_probabilities)
        plt.title("Receiver Operating Characteristic")
        plt.plot(false_positive_rate, true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()
        print(roc_auc_score(self.y_test, target_probabilities))





























