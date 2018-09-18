from comet_ml import Experiment

experiment = Experiment(api_key="oda8KKpxlDgWmJG5KsYrrhmIV", project_name="consensusnet")

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPool2D
from keras.callbacks import LearningRateScheduler, EarlyStopping

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


    def lr_schedule(epoch, lr):
        if epoch > 50:
            if epoch % 10 == 0:
                return lr * 0.95
        return lr

    lr_callback = LearningRateScheduler(lr_schedule)
    callbacks = [lr_callback, EarlyStopping(monitor='val_loss', patience=3)]

    example_shape = X_train.shape[1:]
    input_layer = Input(shape=example_shape)
    conv_1 = Conv2D(filters=40, kernel_size=3, padding='same', activation='relu')(input_layer)
    pool_1 = MaxPool2D(pool_size=(2, 1))(conv_1)
    conv_2 = Conv2D(filters=40, kernel_size=3, padding='same', activation='relu')(pool_1)
    bn_1 = BatchNormalization()(conv_2)

    flatten = Flatten()(bn_1)
    predictions = Dense(6, activation='softmax')(flatten)

    model = Model(input_layer, predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    batch_size = 10000
    epochs = 150

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validate, y_validate), callbacks=callbacks)
    model.save(model_save_path)

if __name__ == '__main__':
    main(sys.argv[1:])