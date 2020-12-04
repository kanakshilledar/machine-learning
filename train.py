# this script uses mnist dataset for training the hand written digit
# recognizer model
# importing libraries
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# splitting data set as train and test
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# preprocessing input data
numOfTrainImages = xTrain.shape[0]
numOfTestImages = xTest.shape[0]
imgWidth = 28
imgHeight = 28

xTrain = xTrain.reshape(xTrain.shape[0], imgHeight, imgWidth, 1)
xTest = xTest.reshape(xTest.shape[0], imgHeight, imgWidth, 1)
inputShape = (imgHeight, imgWidth, 1)

xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255

# converting class of vectors to binary class

numClasses = 10
yTrain = keras.utils.to_categorical(yTrain, numClasses)
yTest = keras.utils.to_categorical(yTest, numClasses)

# defining model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = inputShape))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation = 'softmax'))

# compiling the model! :) 
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

# fitting the model on training data
model.fit(xTrain, yTrain, batch_size = 128, epochs = 80, verbose = 1, validation_data = (xTest, yTest))

# evaluating model on test data
score = model.evaluate(xTest, yTest, verbose = 1)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

# saving model
model.save('handModel.h5')