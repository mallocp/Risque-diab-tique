import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,KernelPCA
import prince
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import LocallyLinearEmbedding

class preprocessingData:

    def __init__(self):
        self.data = pd.read_csv("pima.csv")

    def getData(self):
       return self.data
   
    def descriptionData(self):
        return self.data.describe()
    

    def visualizeData(self):
        pd.tools.plotting.radviz(self.data, self.data.columns[8])

    def plotboxData(self):
        dfp = pd.DataFrame(self.data.values[0:8])
        dfp.plot.box()



    def outliersData(self):

        listOutilers = []
        columnsList = list(self.data.columns[0:8])
        for item in columnsList:
            q1,q3 = np.percentile(self.data[item],[25,75])
            iqr = q3 - q1
            lower_bound = q1 - (iqr * 1.5)
            upper_bound = q3 + (iqr * 1.5)
            listOutilers.append(self.data[item][(self.data[item] > upper_bound) | (self.data[item] < lower_bound)].values)
        for lists in listOutilers:
             print(lists)

    def outliersCount(self):

        OutilersCount = []
        columnsList = list(self.data.columns[0:8])
        for item in columnsList:
            q1, q3 = np.percentile(self.data[item],[25,75])
            iqr = q3 - q1
            lower_bound = q1 - (iqr * 1.5)
            upper_bound = q3 + (iqr * 1.5)
            OutilersCount.append(self.data[item][(self.data[item] > upper_bound) | (self.data[item] < lower_bound)].size)
        print(np.sum(OutilersCount))


    def outliersIndex(self):
        indexOutilers = []
        listIndex = []
        columnsList = list(self.data.columns[0:8])
        for item in columnsList: # matrices des index
            q1,q3 = np.percentile(self.data[item],[25,75])
            iqr = q3 - q1
            lower_bound = q1 - (iqr * 1.5)
            upper_bound = q3 + (iqr * 1.5)
            indexOutilers.append(self.data[item][(self.data[item] > upper_bound) | (self.data[item] < lower_bound)].index)

        for row in indexOutilers:
            for i in row:
                listIndex.append(i)
        listIndex = list(set(listIndex))
        return listIndex

    def outliersExtract(self):
        pd.set_option('mode.chained_assignment', None)
        self.data.is_copy = None
        listIndex = self.outliersIndex()
        columnsList = list(self.data.columns[0:8])
        for item in columnsList:
            for i in self.data[item].index:
                if i in listIndex:
                    self.data[item][i]!= -1  # on remplace les valeurs outliers suivant leur index par -1, pour une selection sans outliers plus facile
        return self.data[self.data > -1].dropna(axis=0,how='any')

    def standardizeData(self):
        scaler = preprocessing.StandardScaler()
        for item in list(self.data.columns[0:8]):
            self.data[item]= scaler.fit_transform(self.data[item])
        return self.data

    def standardizeRobData(self):
        robust_scaler = preprocessing.RobustScaler()
        for item in list(self.data.columns[0:8]):
            self.data[item]= robust_scaler.fit_transform(self.data[item])
        return self.data



    def acpReduction(self):


        pca = prince.PCA(self.data[list(self.data.columns[0:8])], n_components=-1)
        pca.plot_rows(ellipse_fill=True)
        pca.plot_correlation_circle()


        """pca = PCA(n_components=2)

        pca.fit(self.data.values[0:768, 0:8])

        self.dataPCA = pca.transform(self.data.values[0:768, 0:8])

        labels = np.array(self.data.values[:, 8], int)
        cmp = np.array(['r', 'g'])
        #print(pca.explained_variance_ratio_)
        plt.scatter(self.dataPCA[:, 0],self.dataPCA[:, 1], c=cmp[labels], s=5, edgecolors='none')
        plt.xlabel("Premier composant")
        plt.ylabel("Deuxieme composant")
        plt.title("Graphique ACP")"""


    def getacpReduction(self):
        pca = PCA(n_components=8)
        pca.fit(self.data.values[0:768, 0:8])
        self.dataPCA = pca.transform(self.data.values[0:768, 0:8])
        self.labels = np.array(self.data.values[:, 8], int)

    def kernelacpReduction2(self):

        kpca = KernelPCA(kernel="rbf")
        self.dataPCA = kpca.fit_transform(self.data.values[0:768, 0:8])
        labels = np.array(self.data.values[:, 8], int)
        cmp = np.array(['r', 'g'])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.dataPCA[:, 0], self.dataPCA[:, 1], self.dataPCA[:, 2], c=cmp[labels], cmap=plt.cm.coolwarm)

        # Label the axes
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.title("Graphique KernelACP")

    def kernelacpReduction(self):

        kpca = KernelPCA(kernel="rbf")
        self.dataPCA = kpca.fit_transform(self.data.values[0:768, 0:8])
        labels = np.array(self.data.values[:, 8], int)
        cmp = np.array(['r', 'g'])

        plt.scatter(self.dataPCA[:, 0], self.dataPCA[:, 1], c=cmp[labels], cmap=plt.cm.coolwarm)

        # Label the axes
        plt.xlabel("Premier composant")
        plt.ylabel("Deuxieme composant")
        plt.title("Graphique KernelACP")




    def getkernelacpReduction(self):
        kpca = KernelPCA(kernel="rbf")
        self.dataPCA = kpca.fit_transform(self.data.values[0:768, 0:8])
        self.labels = np.array(self.data.values[:, 8], int)


    def getLLE(self):
        lle = LocallyLinearEmbedding(n_neighbors=4)
        self.dataPCA = lle.fit_transform(self.data.values[0:768, 0:8])
        self.labels = np.array(self.data.values[:, 8], int)


    def lleReduction(self):
        lle = LocallyLinearEmbedding(n_neighbors=4)
        self.dataPCA = lle.fit_transform(self.data.values[0:768, 0:8])
        labels = np.array(self.data.values[:, 8], int)
        cmp = np.array(['r', 'g'])
        plt.scatter( self.dataPCA[:, 0], self.dataPCA[:, 1], c=cmp[labels], s=10, edgecolors='none')
        plt.xlabel("Premier composant")
        plt.ylabel("Deuxieme composant")
        plt.title("Graphique KernelACP")




  