#Author: Bo Bekkouche
import numpy as np
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import data

def prepareData(X,Y):
    X = X.reshape((X.shape[0], 28, 28, 1))
    Y = to_categorical(Y,num_classes=10)
    X= data.prepPixels(X)
    return X, Y

def defineModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def trainModel(trainX, trainY,testX,testY):
    model = defineModel()
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
    model.save('models/model1')
    return history



def summarizeDiagnostics(history):
    print('Final accuracy on training data: ' + str(history.history['accuracy'][-1])+'.')
    print('Final accuracy on test data: ' + str(history.history['val_accuracy'][-1])+'.')
    pyplot.figure(figsize=(7, 7))
    pyplot.subplot(2, 1, 1)
    pyplot.title('Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train',marker='.')
    pyplot.plot(history.history['val_loss'], color='orange', label='test',marker='.')
    pyplot.legend()

    pyplot.subplot(2, 1, 2)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train',marker='.')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test',marker='.')
    pyplot.legend()
    pyplot.show()
