import numpy as np
import progressbar
from sklearn.model_selection import train_test_split


def read_dataset_and_reshape_for_conv(path_X, path_y, validation_size=None):
    """
    Reads X and y from given paths and reshapes them for applying in
    convolutional networks.

    Reshaping is done by splitting different letters in separate channels,
    eg. letter 'A' has it's own channel, letter 'C' has it's own channel, etc.

    :param path_X: path to X data
    :type path_X: str
    :param path_y: path to y data
    :type path_y: str
    :param validation_size: specifies percentage of dataset used for validation
    :type validation_size: float
    :return: If validation_size is None, returns just X and y reshaped. If
    validation_size is float, returns a tuple in following order: (X, y,
    X_train, X_validate, y_train, y_validate).
    :rtype: tuple of np.ndarray
    """
    if not validation_size == None:
        if validation_size < 0 or validation_size > 1.0:
            raise ValueError('Validation size must be float from [0, 1], but {} '
                             'given.'.format(validation_size))

    X, y = np.load(path_X), np.load(path_y)
    print('X shape before reshaping:', X.shape)
    print('y shape before reshaping:', y.shape)

    new_X = list()
    neighbourhood_size = X[0].shape[0]
    # Number of columns is equal to the number of letters in dataset (A, C,
    # G, T, I, D, ...).
    num_columns = X[0].shape[1]
    num_data = X.shape[0]
    with progressbar.ProgressBar(max_value=num_data) as progress_bar:
        for i, xi in enumerate(X):
            new_xi = np.dstack(
                [xi[:, col_index].reshape(neighbourhood_size, 1)
                 for col_index in range(num_columns)]
            )
            new_X.append(new_xi)
            progress_bar.update(i)

    new_X = np.array(new_X)
    X = new_X
    print('X shape after reshaping:', X.shape)
    print('y shape after reshaping:', y.shape)

    if validation_size == None:
        return X, y
    else:
        print('Splitting to train and validation set.')
        X_train, X_validate, y_train, y_validate = train_test_split(
            X, y, test_size=validation_size)
        print('X_train shape:', X_train.shape)
        print('X_validate shape:', X_validate.shape)
        print('y_train:', y_train.shape)
        print('y_validate:', y_validate.shape)
        return X, y, X_train, X_validate, y_train, y_validate

