from comet_ml import Experiment
from comet_ml import Optimizer

API_KEY = 'oda8KKpxlDgWmJG5KsYrrhmIV'
# experiment = Experiment(api_key="oda8KKpxlDgWmJG5KsYrrhmIV", project_name="consensusnet")

import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, Dropout
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPool2D
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard
from keras.regularizers import l2

X_validate, y_validate = np.load('./dataset-n20-X-validate.npy'), np.load('./dataset-n20-y-validate.npy')
print(X_validate.shape)
print(y_validate.shape)

def create_model(weights_regularizer):
    def lr_schedule(epoch, lr):
        if epoch > 50:
            if epoch % 10 == 0:
                return lr * 0.95
        return lr

    lr_callback = LearningRateScheduler(lr_schedule)
    callbacks = [lr_callback, 
                 EarlyStopping(monitor='loss', patience=3),]
    #              TensorBoard(log_dir=tensorboard_output_dir, write_images=True, write_grads=True, histogram_freq=5, batch_size=10000)]

    input_shape = X_validate.shape[1:]
    num_output_classes = y_validate.shape[1]

    input_layer = Input(shape=input_shape)

    conv_1 = Conv1D(filters=16, kernel_size=4, padding='same', activation='selu', kernel_regularizer=l2(weights_regularizer))(input_layer)
    pool_1 = MaxPooling1D(pool_size=(5), strides=1)(conv_1)

    conv_2 = Conv1D(filters=32, kernel_size=4, padding='same', activation='selu', kernel_regularizer=l2(weights_regularizer))(pool_1)
    pool_2 = MaxPooling1D(pool_size=(4), strides=1)(conv_2)

    conv_3 = Conv1D(filters=48, kernel_size=4, padding='same', activation='selu', kernel_regularizer=l2(weights_regularizer))(pool_2)
    pool_3 = MaxPooling1D(pool_size=(3), strides=1)(conv_3)

    flatten = Flatten()(pool_3)

    dn_1 = Dense(336, activation='selu')(flatten)
    drop = Dropout(0.5)(dn_1)

    predictions = Dense(num_output_classes, activation='softmax')(drop)

    model = Model(input_layer, predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    
    return model

def train_with_optimizer(suggestion, experiment):
    experiment.log_multiple_params(suggestion)
    
    model = create_model(suggestion['weight_regularizer'])
    
    model.fit(
        X_validate,
        y_validate,
        batch_size=suggestion['batch_size'],
        epochs=suggestion['epochs'],
        validation_data=(X_validate, y_validate)
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
    epochs integer [5, 10] [5]
    batch_size integer [5000, 15000] [10000]
    weight_regularizer real [1e-5, 1e-2] [1e-3]
    """

    optimizer.set_params(params)
    # get_suggestion will raise when no new suggestion is available
    i = 0
    while True:

        # Get a suggestion
        suggestion = optimizer.get_suggestion()
        print()
        print('----------------------------------')
        print(i)
        print(suggestion)
        print('batch_size', suggestion['batch_size'])
        print('epochs', suggestion['epochs'])
        print('weight_regularizer', suggestion['weight_regularizer'])
        print('----------------------------------')
        print()

        # Create a new experiment associated with the Optimizer
        experiment = Experiment(
            api_key=API_KEY, project_name="consensusnet")
        
        score = train_with_optimizer(suggestion, experiment)

        # Report the score back
        suggestion.report_score("accuracy", score)
        
if __name__ == '__main__':
    run_optimizer()