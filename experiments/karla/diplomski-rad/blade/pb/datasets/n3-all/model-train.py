from comet_ml import Experiment

experiment = Experiment(api_key="oda8KKpxlDgWmJG5KsYrrhmIV", project_name="consensusnet")

import numpy as np

import sys
module_path = '/home/diplomski-rad/consensus-net/src/python/dataset/'
if module_path not in sys.path:
    print('Adding dataset module.')
    sys.path.append(module_path)

import dataset

X, y, X_train, X_validate, y_train, y_validate = dataset.read_dataset_and_reshape_for_conv(
    './pysam-all-dataset-n3-X.npy', './pysam-all-dataset-n3-y.npy', 0.1)

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv1D, MaxPooling1D, Conv2D

input_layer = Input(shape=(7, 1, 4))
conv_1 = Conv2D(filters=20, kernel_size=3, padding='same', activation='relu')(input_layer)

flatten = Flatten()(conv_1)
predictions = Dense(4, activation='softmax')(flatten)

model = Model(input_layer, predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

batch_size = 10000
epochs = 25

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validate, y_validate))

model.save('./model.h5')