import numpy as np

# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="oda8KKpxlDgWmJG5KsYrrhmIV")

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling1D, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.datasets import mnist

from sklearn.model_selection import train_test_split


def main():
    X, y = np.load('pysam-dataset-n3-X.npy'), np.load('pysam-dataset-n3-y.npy')

    new_X = list()
    for xi in X:
        new_xi = np.dstack((xi[:, 0].reshape(7, 1), xi[:, 1].reshape(7, 1), xi[:, 2].reshape(7, 1), xi[:, 3].reshape(7, 1)))
        new_X.append(new_xi)

    new_X = np.array(new_X)
    X = new_X

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.10)

    input_layer = Input(shape=(7, 1, 4))
    conv_1 = Conv2D(filters=5, kernel_size=3, padding='same', activation='relu')(input_layer)
    # pool_1 = MaxPooling1D(pool_size=2)(conv_1)

    flatten = Flatten()(conv_1)
    predictions = Dense(4, activation='softmax')(flatten)

    # model.add(Conv1D(filters=5, kernel_size=3, padding='same', activation='relu', input_shape=(7, 1)))
    # model.add(MaxPooling1D())
    # # model.add(Dense(40, activation='relu'))
    # # model.add(BatchNormalization())
    # # model.add(Dense(40, activation='relu'))

    # # model.add(Dropout(0.25))
    # model.add(Dense(4, activation='softmax'))

    model = Model(input_layer, predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    batch_size = 10000
    epochs = 200

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validate, y_validate))

if __name__ == '__main__':
    main()
