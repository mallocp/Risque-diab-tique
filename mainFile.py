import preprocessingData as pr
import treeCart as cart
import treeBagging as bag
import treeForest as ft
import treeBoosting as bt
import knnClassifier as kc
import analyseDiscriminante as an 
import mplClassifier as pl
import svmClassifier as sv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)   
preproc = pr.preprocessingData()
treecart = cart.treeCart()
treebagging = bag.treeBagging()
treeforest = ft.treeForest()
treeboosting = bt.treeBoosting()
knnclassifier = kc.knnClassifier() 
analysediscriminante = an.analyseDiscriminante()
mplclassifier = pl.mplClassifier()
svmclassifier = sv.svmClassifier()



