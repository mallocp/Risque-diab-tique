import numpy as np
import matplotlib.pyplot as plt
import preprocessingData as pr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from  sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

class mplClassifier:
    def __init__(self):
        self.preproc = pr.preprocessingData()
        self.preproc.standardizeRobData()
        self.preproc.getkernelacpReduction()
        self.dataPCA = self.preproc.dataPCA
        self.labels = self.preproc.labels
        self.n = 30
        self.accuracy = np.zeros(self.n)
        self.training = np.zeros(self.n)


    def trainingMPL(self):
        self.score_acc = 0
        self.nlayer = 0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.preproc.data.values[0:768, 0:8], self.labels, test_size=0.3)
        for i in range(self.n):
            self.mpl = MLPClassifier(solver='lbfgs',hidden_layer_sizes=((i+1), ))
            self.mpl.fit(self.X_train, self.y_train)
            self.training[i] = self.mpl.score(self.X_train, self.y_train)
            self.accuracy[i] = self.mpl.score(self.X_test, self.y_test)
            if(self.accuracy[i] > self.training[i] and self.accuracy[i] > self.score_acc ) :
                self.score_acc = self.accuracy[i]
                self.nlayer = self.mpl.hidden_layer_sizes

        plt.plot([(i + 1) for i in range(30)], self.training, 'b')
        plt.plot([(i + 1) for i in range(30)], self.accuracy, 'r')
        plt.title("score Training vs score Accuracy")
        plt.ylabel("score")
        plt.xlabel("n_layer_hidden")
        plt.grid(True)
        plt.show()



    def frontieredeDecision(self):
        tuned_parameters = {'hidden_layer_sizes': [(5,), (20,), (50,), (100,), (150,), (200,)], 'alpha': [0.001, 0.01, 1, 2]}
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataPCA[:, [0, 1]], self.labels,test_size=0.3)
        self.mpl = GridSearchCV(MLPClassifier(solver='lbfgs', hidden_layer_sizes=13), tuned_parameters, cv=5, verbose=0)
        self.mpl.fit(self.X_train, self.y_train)
        self.training = self.mpl.score(self.X_train, self.y_train)
        self.accuracy = self.mpl.score(self.X_test, self.y_test)
        cmp = np.array(['r', 'g'])
        #plt.scatter(self.X_train[:,0],self.X_train[:,1],c=cmp[self.y_train],s=5,edgecolors='none')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='none', s=5, edgecolors=cmp[self.y_test])
        nx, ny = 400, 400
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = self.mpl.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        plt.show()

    def crossingValidationmpl(self):
        tuned_parameters = {'hidden_layer_sizes': [(5,), (20,), (50,), (100,), (150,), (200,)], 'alpha': [0.001, 0.01, 1, 2]}
        self.clf2 = GridSearchCV(MLPClassifier(solver='lbfgs'), tuned_parameters, cv=10, verbose=0)
        self.clf2.fit(self.preproc.data.values[0:768, 0:8], self.labels)
        self.param = self.clf2.best_params_
        pipeline = make_pipeline(MLPClassifier(solver='lbfgs', hidden_layer_sizes=self.param['hidden_layer_sizes'], alpha=self.param['alpha']))
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        self.cv_results = cross_val_score(pipeline, self.preproc.data.values[0:768, 0:8], self.labels, cv=kf, scoring="accuracy")
        self.cv_results2 = self.cv_results.mean()



        """target_probabilities=pipeline.fit(self.X_train, self.y_train).predict_proba(self.X_test)[:,1]

        false_positive_rate, true_positive_rate, threshold = roc_curve(self.y_test, target_probabilities)
        plt.title("Receiver Operating Characteristic")
        plt.plot(false_positive_rate, true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()
        print(roc_auc_score(self.y_test, target_probabilities))"""


    def gridSearch(self):
        tuned_parameters = {'hidden_layer_sizes': [(5,), (20,), (50,), (100,), (150,), (200,)],'alpha': [0.001, 0.01, 1, 2]}
        self.clf2 = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, verbose=1)

        self.clf2.fit(self.X_train, self.y_train)
        self.param =  self.clf2.best_params_
        return  self.param, self.clf2