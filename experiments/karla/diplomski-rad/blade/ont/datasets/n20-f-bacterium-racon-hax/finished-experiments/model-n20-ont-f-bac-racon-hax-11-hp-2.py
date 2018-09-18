from comet_ml import Experiment
from comet_ml import Optimizer

API_KEY = 'oda8KKpxlDgWmJG5KsYrrhmIV'

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPool2D
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard

import sys

module_path = '/home/diplomski-rad/consensus-net/src/python/utils/'
if module_path not in sys.path:
    print('Adding utils module.')
    sys.path.append(module_path)
from args_parsers import parse_train_args

X_train, X_validate = None, None
y_train, y_validate = None, None
tensorboard_output_dir = None

def build_model_and_callbacks(suggestion):
    
    def lr_schedule(epoch, lr):
        if epoch > 50:
            if epoch % 10 == 0:
                return lr * 0.95
        return lr

    lr_callback = LearningRateScheduler(lr_schedule)
    callbacks = [lr_callback, 
                 EarlyStopping(monitor='val_loss', patience=3),
                 TensorBoard(log_dir=tensorboard_output_dir, write_images=True, histogram_freq=0)]

    input_shape = X_train.shape[1:]
    num_output_classes = y_train.shape[1]
    
    input_layer = Input(shape=input_shape)
    conv_1 = Conv1D(filters=suggestion['filters_1'], kernel_size=suggestion['kernel_size_1'], padding='same', activation='relu')(input_layer)
    pool_1 = MaxPooling1D(pool_size=(2))(conv_1)
    conv_2 = Conv1D(filters=suggestion['filters_2'], kernel_size=suggestion['kernel_size_1'], padding='same', activation='relu')(pool_1)
    bn_1 = BatchNormalization()(conv_2)

    flatten = Flatten()(bn_1)
    predictions = Dense(num_output_classes, activation='softmax')(flatten)

    model = Model(input_layer, predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    
    return model, callbacks

def train_with_optimizer(suggestion, experiment):
    experiment.log_multiple_params(suggestion)
    
    model, callbacks = build_model_and_callbacks(suggestion)
    
    model.fit(
        X_train,
        y_train,
        batch_size=suggestion['batch_size'],
        epochs=suggestion['epochs'],
        validation_data=(X_validate, y_validate),
        callbacks=callbacks
    )
    
    evaluation = model.evaluate(X_validate, y_validate, verbose=0)
    
    metrics = {
        "evaluation_loss": evaluation[0],
        "evaluation_accuracy": evaluation[1],
    }
    
    experiment.log_multiple_metrics(metrics)
    return metrics["evaluation_accuracy"]

def run_optimizer():
    optimizer = Optimizer(API_KEY)

    params = """
    epochs integer [15, 100] [20]
    batch_size integer [5000, 15000] [10000]
    filters_1 integer [30, 50] [40]
    filters_2 integer [30, 50] [40]
    kernel_size_1 integer [3, 10] [3]
    kernel_size_2 integer [3, 10] [3]
    """

    optimizer.set_params(params)
    # get_suggestion will raise when no new suggestion is available
    i = 0
    while True:
        i += 1

        # Get a suggestion
        suggestion = optimizer.get_suggestion()
        print()
        print('----------------------------------')
        print(i)
        print('batch_size', suggestion['batch_size'])
        print('epochs', suggestion['epochs'])
        print('filters_1', suggestion['filters_1'])
        print('filters_2', suggestion['filters_2'])
        print('kernel_size_1', suggestion['kernel_size_1'])
        print('kernel_size_2', suggestion['kernel_size_2'])
        print('----------------------------------')
        print()

        # Create a new experiment associated with the Optimizer
        experiment = Experiment(
            api_key=API_KEY, project_name="consensusnet")
        
        score = train_with_optimizer(suggestion, experiment)

        # Report the score back
        suggestion.report_score("accuracy", score)

def main(args):
    args = parse_train_args(args)
    
    global X_train, X_validate, y_train, y_validate
    global tensorboard_output_dir
    
    X_train = np.load(args.X_train)
    X_validate = np.load(args.X_validate)
    y_train = np.load(args.y_train)
    y_validate = np.load(args.y_validate)
    
    model_save_path = args.model_save_path
    tensorboard_output_dir = args.tensorboard_output_dir
    
    run_optimizer()


#     model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validate, y_validate), callbacks=callbacks)
#     model.save(model_save_path)

if __name__ == '__main__':
    main(sys.argv[1:])