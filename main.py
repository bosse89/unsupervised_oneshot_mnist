#Author: Bo Bekkouche

#Acknolwdgements
#Code from the following websites were used to inspire and accelerate this code project:
#https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
#https://github.com/nathandelara/MNIST-unsupervised
from timeit import default_timer as timer
start = timer()
import numpy as np
import data
import clustering as cl
import neural_network as nn


print('Loading data.')
(trainData, trainLabels), (testData, testLabels)=data.loadDataset_MnistKeras()
print('Preparing data.')
trainingData_cl=cl.prepareData(trainData)
print('Reducing dimensions.')
components=cl.reduceDimensions_PCA(trainingData_cl)
print('Converting data to graph.')
graph=cl.convert2graph(components)
print('Clustering data.')
clusterLabels=cl.clusterGraphData(graph)
print('Evaluating clusters.')
vscore=cl.evaluateClusters(clusterLabels,trainLabels)
print('Clusters have V-measure(homogeneity and completeness) = '+str(vscore))
print('Inferring clusters to labels using the correct labels of the first 10 data samples.')
labelMap=cl.mapClusters2labels(clusterLabels,trainLabels)
print('Inferring cluster labels using map and counting correctly labeled data samples.')
inferredLabels,corrPerc=cl.countCorrectLabels(clusterLabels,labelMap,trainLabels)
print('Correctly labeled data samples: ' + str(corrPerc))
print('Preparing data.')
trainData,inferredLabels=nn.prepareData(trainData,inferredLabels)
testData,testLabels=nn.prepareData(testData,testLabels)
print('Training neural network data.')
history = nn.trainModel(trainData,inferredLabels,testData,testLabels)
end = timer()
compTimeMin=np.floor((end - start)/60)
compTimeSec=(((end - start)/60)-compTimeMin)*60
print('Elapsed time: ' + str(compTimeMin)+ ' minutes and ' + str(compTimeSec)+ ' seconds.')
nn.summarizeDiagnostics(history)

print('Completed.')