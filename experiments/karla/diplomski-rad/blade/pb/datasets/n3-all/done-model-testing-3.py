from comet_ml import Experiment

experiment = Experiment(api_key="oda8KKpxlDgWmJG5KsYrrhmIV", project_name="consensusnet")

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPool2D

import sys
module_path = '/home/diplomski-rad/consensus-net/src/python/dataset/'
if module_path not in sys.path:
    print('Adding dataset module.')
    sys.path.append(module_path)

import dataset

X_train = np.load('./pysam-all-dataset-n3-X-reshaped-train.npy')
X_validate = np.load('./pysam-all-dataset-n3-X-reshaped-validate.npy')
y_train = np.load('./pysam-all-dataset-n3-y-reshaped-train.npy')
y_validate = np.load('./pysam-all-dataset-n3-y-reshaped-validate.npy')

input_layer = Input(shape=(7, 1, 4))
conv_1 = Conv2D(filters=40, kernel_size=3, padding='same', activation='relu')(input_layer)
pool_1 = MaxPool2D(pool_size=(2, 1))(conv_1)
conv_2 = Conv2D(filters=20, kernel_size=3, padding='same', activation='relu')(pool_1)

flatten = Flatten()(conv_2)
predictions = Dense(4, activation='softmax')(flatten)

model = Model(input_layer, predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

batch_size = 10000
epochs = 50

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validate, y_validate))
