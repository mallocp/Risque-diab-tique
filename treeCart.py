import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import preprocessingData as pr
import matplotlib.pyplot as plt
import pydotplus
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline


class treeCart:

    def __init__(self):
        self.preproc = pr.preprocessingData()
        self.preproc.getkernelacpReduction()
        self.dataPCA = self.preproc.dataPCA
        self.labels = self.preproc.labels
        self.n = 30
        self.accuracy = np.zeros(self.n)
        self.training = np.zeros(self.n)


    def trainingtreeCart(self):
        self.score_acc = 0
        for i in range(self.n):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataPCA[:, [0, 1]], self.labels, test_size=0.3)
            self.treeCart = tree.DecisionTreeClassifier(max_depth = i+1, min_samples_leaf = 5)
            self.treeCart.fit(self.X_train, self.y_train)
            self.training[i] = self.treeCart.score(self.X_train, self.y_train)
            self.accuracy[i] = self.treeCart.score(self.X_test, self.y_test)
            if (self.accuracy[i] > self.training[i] and self.accuracy[i] > self.score_acc):
                self.score_acc = self.accuracy[i]


        plt.plot([(i + 1) for i in range(30)], self.training, 'b')
        plt.plot([(i + 1) for i in range(30)], self.accuracy, 'r')
        plt.title("score Training vs score Accuracy")
        plt.ylabel("score")
        plt.xlabel("Nombre Profondeur Noeud (Max dept)")
        plt.grid(True)
        plt.show()


    def frontieredeDecision(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataPCA[:, [0, 1]], self.labels,test_size=0.3)
        self.treeCart = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
        self.treeCart.fit(self.X_train, self.y_train)
        self.treeCart.score(self.X_train, self.y_train)
        self.treeCart.score(self.X_test, self.y_test)
        cmp = np.array(['r', 'g'])
        #plt.scatter(self.X_train[:,0],self.X_train[:,1],c=cmp[self.y_train],s=5,edgecolors='none')
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='none', s=5, edgecolors=cmp[self.y_test])
        nx, ny = 400, 400
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        Z = self.treeCart.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, Z, [0.5])
        plt.show()


    def affichagetreeCart(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataPCA[:, [0, 1]], self.labels,test_size=0.3)
        self.treeCart = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
        clf2 = self.treeCart.fit(self.X_test, self.y_test)
        dot_data = tree.export_graphviz(clf2, out_file=None,
                                        feature_names=self.preproc.data.columns[0:8],
                                        class_names=True,
                                        filled=True, rounded=True, proportion=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_gif("pimaTree.gif")

    def crossingValidationtree(self):
        pipeline = make_pipeline(tree.DecisionTreeClassifier())
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        cv_results = cross_val_score(pipeline, self.dataPCA[:, [0, 1]], self.labels, cv=kf)
        return cv_results.mean()