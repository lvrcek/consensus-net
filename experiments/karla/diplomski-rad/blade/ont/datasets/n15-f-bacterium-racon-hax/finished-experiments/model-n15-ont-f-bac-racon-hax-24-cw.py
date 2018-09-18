from comet_ml import Experiment

experiment = Experiment(api_key="oda8KKpxlDgWmJG5KsYrrhmIV", project_name="consensusnet")

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, Dropout
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPool2D
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard
from keras.regularizers import l2

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
    tensorboard_output_dir = args.tensorboard_output_dir
    class_weights = args.class_weights
    if class_weights is not None:
        class_weights = np.load(class_weights).item()


    def lr_schedule(epoch, lr):
        if epoch > 50:
            if epoch % 10 == 0:
                return lr * 0.95
        return lr

    lr_callback = LearningRateScheduler(lr_schedule)
    callbacks = [lr_callback, 
                 EarlyStopping(monitor='val_loss', patience=3),
                 TensorBoard(log_dir=tensorboard_output_dir, write_images=True, write_grads=True, histogram_freq=5, batch_size=10000)]

    input_shape = X_train.shape[1:]
    num_output_classes = y_train.shape[1]
    
    input_layer = Input(shape=input_shape)

    conv_1 = Conv1D(filters=16, kernel_size=4, padding='same', activation='selu', kernel_regularizer=l2(1e-3))(input_layer)
    pool_1 = MaxPooling1D(pool_size=(5), strides=1)(conv_1)

    conv_2 = Conv1D(filters=32, kernel_size=4, padding='same', activation='selu', kernel_regularizer=l2(1e-3))(pool_1)
    pool_2 = MaxPooling1D(pool_size=(4), strides=1)(conv_2)

    conv_3 = Conv1D(filters=48, kernel_size=4, padding='same', activation='selu', kernel_regularizer=l2(1e-3))(pool_2)
    pool_3 = MaxPooling1D(pool_size=(3), strides=1)(conv_3)

    flatten = Flatten()(pool_3)

    dn_1 = Dense(336, activation='selu')(flatten)
    drop = Dropout(0.5)(dn_1)

    predictions = Dense(num_output_classes, activation='softmax')(drop)

    model = Model(input_layer, predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    batch_size = 10000
    epochs = 150

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validate, y_validate), callbacks=callbacks, class_weight=class_weights)
    model.save(model_save_path)

if __name__ == '__main__':
    main(sys.argv[1:])