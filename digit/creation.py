from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense, Flatten, AveragePooling2D
from keras.models import Sequential, load_model



def load_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print("loading data done")
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    print("prep pixels done")
    return train_norm, test_norm


def define_model():
    model = Sequential()

    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def harness():
    trainX, trainY, testX, testY = load_data()

    trainX, testX = prep_pixels(trainX, testX)

    model = define_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)

    model.save('model.h5')

    # model = load_model('model.h5')

    # model.summary()

    # _, acc = model.evaluate(testX, testY, verbose= 0)

    # print('> %.3f' % (acc * 100.0))


harness()

