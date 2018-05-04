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
    if validation_size is not None:
        if validation_size < 0 or validation_size > 1.0:
            raise ValueError('Validation size must be float from [0, 1], but {}'
                             ' given.'.format(validation_size))

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

    if validation_size is not None:
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


def _calc_empty_rows(X):
    """
    Calculates which rows in X are empty rows (i.e. all numbers in that row
    are equal to 0).

    :param X: 2-D data
    :type X: np.ndarray
    :return: 1-D array with 1s on positions which correspond to empty rows.
    :rtype: np.ndarray
    """
    empty_row = np.zeros((1, 4))
    empty_rows = [int(v) for v in np.all(empty_row == X, axis=1)]
    return empty_rows


def create_dataset_with_neighbourhood(X_paths, y_paths, neighbourhood_size):
    """
    Creates datasets by mixing all pileups with given neighbourhood_size.

    Datasets are concatenated after extracting neighbourhood_size positions in
    given datasets separately.

    Dataset at i-th positino in X_paths should match given labels at i-th
    positino in y_paths.

    :param X_paths: list of paths to X pileup dataset
    :type X_paths: list of str
    :param y_paths: list of paths to y pileup dataset
    :type y_paths: list of str
    :param neighbourhood_size: number of neighbours to use from one size (eg.
        if you set this parameter to 3, it will take 3 neighbours from both
        sides so total number of positions in one example will be 7 -
        counting the middle position)
    :type neighbourhood_size: float
    :return:
    :rtype tuple of np.ndarray
    """
    if not len(X_paths) == len(y_paths):
        raise ValueError('Number of X_paths and y_paths should be the same!')

    new_X, new_y = list(), list()
    for X_path, y_path in zip(X_paths, y_paths):
        print('Parsing ', X_path, ' and ', y_path)

        X, y = np.load(X_path), np.load(y_path)
        # Removing last column which everything which was not 'A' nor 'C' nor
        #  'G' nor 'T'.
        y = y[:, :4]

        empty_rows = _calc_empty_rows(X)

        print('Creating dataset with neighbrouhood ...')
        with progressbar.ProgressBar(max_value=X.shape[0]) as progress_bar:
            # TODO(ajuric): Check if this can be speed up.
            for i in range(X.shape[0]):
                progress_bar.update(i)
                if empty_rows[i] == 1:
                    continue  # current row is empty row
                if i < neighbourhood_size or \
                   i >= X.shape[0] - neighbourhood_size:
                    # Current position is not suitable to build an example.
                    continue
                zeros_to_left = np.sum(empty_rows[i - neighbourhood_size:i])
                zeros_to_right = np.sum(
                    empty_rows[i + 1:i + neighbourhood_size + 1])
                if zeros_to_left == 0 and zeros_to_right == 0:
                    new_X.append(
                        X[i - neighbourhood_size:i + neighbourhood_size + 1])
                    new_y.append(y[i])

    return new_X, new_y
