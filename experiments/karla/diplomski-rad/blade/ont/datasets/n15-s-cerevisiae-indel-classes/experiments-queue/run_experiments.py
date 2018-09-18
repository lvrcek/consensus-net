import sys

module_path = '/home/diplomski-rad/consensus-net/src/python/training/'
if module_path not in sys.path:
    print('Adding training module.')
    sys.path.append(module_path)
from training import experiment_queue
from training import X_TRAIN_PATH, X_VALIDATE_PATH, Y_TRAIN_PATH, Y_VALIDATE_PATH

experiment_paths = ['./model-n15-s-cerv-ont-indel-classes-14.py']
dataset_paths = {
    X_TRAIN_PATH: '../dataset-n15-X-reshaped-train.npy',
    Y_TRAIN_PATH: '../dataset-n15-y-reshaped-train.npy',
    X_VALIDATE_PATH: '../dataset-n15-X-reshaped-validate.npy',
    Y_VALIDATE_PATH: '../dataset-n15-y-reshaped-validate.npy'
}
model_save_paths = ['../model-14.h5']
finished_experiments_dir_path = '../finished-experiments/'

experiment_queue(experiment_paths, dataset_paths, model_save_paths, finished_experiments_dir_path)