#Author: Bo Bekkouche
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score
from sklearn.neighbors import KNeighborsTransformer
import community as community_louvain
import networkx as nx
import random

def prepareData(X):
    X = X.reshape((X.shape[0], -1))
    return X

def reduceDimensions_PCA(X):
    pca = PCA(n_components=32)
    components = pca.fit_transform(X)
    return components

def convert2graph(components):
    knn = KNeighborsTransformer(n_neighbors=10, n_jobs=-1)
    graph = knn.fit_transform(components)
    G = nx.Graph(graph)
    return G

def clusterGraphData(G):
    labels = community_louvain.best_partition(G, resolution=2.0,random_state=8)
    clusterLabels=np.array(list(labels.items()))[0:60000,1]
    return clusterLabels

def evaluateClusters(clusterLabels,correctLabels):
    vscore=v_measure_score(clusterLabels, correctLabels)
    return vscore

def mapClusters2labels(clusterLabels,trainLabels):
    nClasses=10
    classes=np.arange(0,nClasses)
    labelMap=np.full([nClasses, 4],-1,dtype=int)
    numberOfUsedLabeledDatapoints=0
    for i in np.arange(0,len(clusterLabels)):
        if trainLabels[i] in classes:
            classes=classes[classes != trainLabels[i]]
            labelMap[trainLabels[i],0:3]=[trainLabels[i], clusterLabels[i],i]
            numberOfUsedLabeledDatapoints+=1
        if numberOfUsedLabeledDatapoints>=10:
            break
    return labelMap

def countCorrectLabels(clusterLabels,labelMap,trainLabels):
    correct=0
    inferredLabels=np.array([],dtype=int)
    for i in np.arange(0,len(clusterLabels)):
        inferedLabel=labelMap[labelMap[0:10, 1] == clusterLabels[i], 0]
        if len(inferedLabel)==0:
            inferedLabel=random.randint(0,9)
            print('Warning! A class was assigned randomly due to duplicate inference.')
        else:
            inferedLabel=inferedLabel[0]
        if inferedLabel==trainLabels[i]:
            correct+=1
        inferredLabels = np.append(inferredLabels, inferedLabel)
    correctPerc=100*correct/len(clusterLabels)
    return inferredLabels,correctPerc