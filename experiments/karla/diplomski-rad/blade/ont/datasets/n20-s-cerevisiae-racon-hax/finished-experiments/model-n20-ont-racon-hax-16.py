from comet_ml import Experiment

experiment = Experiment(api_key="oda8KKpxlDgWmJG5KsYrrhmIV", project_name="consensusnet")

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv1D, MaxPooling1D, SeparableConv1D
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.regularizers import l1, l2, l1_l2

import sys

module_path = '/home/diplomski-rad/consensus-net/src/python/utils/'
if module_path not in sys.path:
    print('Adding utils module.')
    sys.path.append(module_path)
from args_parsers import parse_train_args

def main(args):
    args = parse_train_args(args)
    
    X_train = np.load(args.X_train)
    X_valid = np.load(args.X_validate)
    y_train = np.load(args.y_train)
    y_valid = np.load(args.y_validate)
    
    model_save_path = args.model_save_path

    def lr_schedule(epoch, lr):
        if epoch > 50:
            if epoch % 5 == 0:
                return lr * 0.95
        return lr

    lr_callback = LearningRateScheduler(lr_schedule)
    callbacks = [lr_callback, EarlyStopping(monitor='val_loss', patience=3)]

    input_shape = X_train.shape[1:]
    num_output_classes = y_train.shape[1]

    input_layer = Input(shape=input_shape)
    conv_1 = Conv1D(filters=40, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_layer)
    pool_1 = MaxPooling1D(pool_size=(2))(conv_1)
    conv_2 = SeparableConv1D(filters=40, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(pool_1)
    bn_1 = BatchNormalization()(conv_2)

    flatten = Flatten()(bn_1)
    predictions = Dense(num_output_classes, activation='softmax')(flatten)

    model = Model(input_layer, predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    batch_size = 10000
    epochs = 150

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=callbacks)
    model.save(model_save_path)

if __name__ == '__main__':
    main(sys.argv[1:])