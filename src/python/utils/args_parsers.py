import argparse


def parse_train_args(args):
    """
    Parses arguments for model training.

    Arguments should be passed just like 'sys.argv[1:]'.

    Needed arguments are X_train_path, y_train_path, X_validate_path,
    y_validate_path and model_save_path.

    :param args: list of args from sys.argv[1:]
    :type args: list of str
    :return: parsed arguments for training
    :rtype: Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('X_train', type=str, help='X_train path.')
    parser.add_argument('y_train', type=str, help='y_train path.')
    parser.add_argument('X_validate', type=str, help='X_validate path.')
    parser.add_argument('y_validate', type=str, help='y_validate path.')
    parser.add_argument('model_save_path', type=str,
                        help='Path for trained model saving.')

    return parser.parse_args(args)
