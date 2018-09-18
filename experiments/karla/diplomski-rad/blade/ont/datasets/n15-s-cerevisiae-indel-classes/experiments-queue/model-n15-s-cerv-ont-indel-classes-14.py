from comet_ml import Experiment

experiment = Experiment(api_key="oda8KKpxlDgWmJG5KsYrrhmIV", project_name="consensusnet")

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard

import sys

module_path = '/home/diplomski-rad/consensus-net/src/python/utils/'
if module_path not in sys.path:
    print('Adding utils module.')
    sys.path.append(module_path)
from args_parsers import parse_train_args

def main(args):
    args = parse_train_args(args)
    
    X_train = np.load(args.X_train)
    X_validate = np.load(args.X_validate)
    y_train = np.load(args.y_train)
    y_validate = np.load(args.y_validate)
    
    model_save_path = args.model_save_path

    example_shape = X_train.shape[1:]
    input_layer = Input(shape=example_shape)

    conv_1 = Conv2D(filters=16, kernel_size=3, padding='same', activation='selu')(input_layer)
    pool_1 = MaxPooling2D(pool_size=(2, 1))(conv_1)
    conv_2 = Conv2D(filters=32, kernel_size=3, padding='same', activation='selu')(pool_1)

    flatten = Flatten()(conv_2)
    dense_1 = Dense(1042, activation='selu')(flatten)
    dropout_1 = Dropout(0.25)(dense_1)
    predictions = Dense(6, activation='softmax')(dropout_1)

    model = Model(input_layer, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    batch_size = 10000
    epochs = 50
    callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                 TensorBoard(log_dir='./logs', write_images=True, histogram_freq=0)]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validate, y_validate), callbacks=callbacks)
    model.save(model_save_path)

if __name__ == '__main__':
    main(sys.argv[1:])