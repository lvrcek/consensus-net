import os

X_TRAIN_PATH = 'X_train_path'
Y_TRAIN_PATH = 'y_train_path'
X_VALIDATE_PATH = 'X_validate_path'
Y_VALIDATE_PATH = 'y_validate_path'

EXPERIMENT_START_CMD = 'python3 {} {} {} {} {} {} {}'
EXPERIMENT_MOVE_CMD = 'mv {} {}'


def experiment_queue(experiment_paths, dataset_paths, model_save_paths,
                     tensorboard_output_dirs, finished_experiments_dir_path):
    """
    Starts list of experiments (models) for training models.

    All models are trained one after another on same GPU, because currently
    only one GPU is supported.

    Argument dataset_paths should be dict with keys 'X_train_path',
    'y_train_path', 'X_validate_path' and 'y_validate_path' with
    corresponding paths.

    Every experiment file should be formatted in a way that it receives two
    arguments:
        - X_train_path: path to X_train
        - y_train_path: path to y_train
        - X_validate_path: path to X_validate
        - y_validate_path: path to y_validate
        - model_save_path: path where saved model will be saved
        - tensorboard_output_dir: path where data for Tensorboard will be saved

    After experiment is finished, it is moved to
    finished_experiments_dir_path directory to separate it from unfinished
    experiments.

    :param experiment_paths: list of paths to Python files
        containing models for training
    :type experiment_paths: str
    :param dataset_paths: map of paths to X_train, y_train, X_validate,
        y_validate
    :type dataset_paths: dict
    :param model_save_paths: path for saving trained model
    :type model_save_paths: list of str
    :param tensorboard_output_dirs: Path to directory where Tensorboard data
    will be saved.
    :type tensorboard_output_dirs: list of str
    :param finished_experiments_dir_path: directory where to move finished
        experiments to mark them as finished
    :type finished_experiments_dir_path: str
    """

    for experiment_path in experiment_paths:
        model_name = os.path.split(experiment_path)[-1]
        if os.path.exists(
                os.path.join(finished_experiments_dir_path, model_name)):
            raise ValueError('Given model({}) already exists in '
                             'finished_experiments_dir. It may be already '
                             'executed.'.format(experiment_path))

    X_train_path = dataset_paths[X_TRAIN_PATH]
    y_train_path = dataset_paths[Y_TRAIN_PATH]
    X_validate_path = dataset_paths[X_VALIDATE_PATH]
    y_validate_path = dataset_paths[Y_VALIDATE_PATH]

    for experiment_path, model_save_path, tensorboard_output_dir in zip(
            experiment_paths, model_save_paths, tensorboard_output_dirs):
        print('----> Starting experiment {}. <----'.format(experiment_path))
        os.system(EXPERIMENT_START_CMD.format(
            experiment_path, X_train_path, y_train_path, X_validate_path,
            y_validate_path, model_save_path, tensorboard_output_dir))

        print('----> Marking experiment {} as finished... <----'.format(
            experiment_path))
        os.system(EXPERIMENT_MOVE_CMD.format(
            experiment_path, finished_experiments_dir_path))
