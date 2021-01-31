#Author: Bo Bekkouche
from keras.datasets import mnist

def loadDataset_MnistLocal():
    imagefile = 'data/MNIST_ubyte/train-images.idx3-ubyte'
    trainX = idx2numpy.convert_from_file(imagefile)
    trainX2 = trainX.reshape((trainX.shape[0], -1))
    # trainX2=np.transpose(trainX2)
    imagefile = 'data/MNIST_ubyte/train-labels.idx1-ubyte'
    trainY = idx2numpy.convert_from_file(imagefile)
    imagefile = 'data/MNIST_ubyte/t10k-images.idx3-ubyte'
    testX = idx2numpy.convert_from_file(imagefile)
    imagefile = 'data/MNIST_ubyte/t10k-labels.idx1-ubyte'
    testY = idx2numpy.convert_from_file(imagefile)

def loadDataset_MnistKeras():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    return (trainX, trainY), (testX, testY)
def prepPixels(X):
    X_norm = X.astype('float32')
    X_norm = X_norm / 255.0
    return X_norm


