import sys

module_path = '/home/diplomski-rad/consensus-net/src/python/training/'
if module_path not in sys.path:
    print('Adding training module.')
    sys.path.append(module_path)
from training import experiment_queue
from training import X_TRAIN_PATH, X_VALIDATE_PATH, Y_TRAIN_PATH, Y_VALIDATE_PATH

experiment_paths = [
#     './model-n20-ont-f-bac-racon-hax-11.py',
#     './model-n20-ont-f-bac-racon-hax-23.py',
#     './model-n20-ont-f-bac-racon-hax-24.py',
#     './model-n20-ont-f-bac-racon-hax-11-repeat.py',
#     './model-n20-ont-f-bac-racon-hax-23-repeat.py',
#     './model-n20-ont-f-bac-racon-hax-24-repeat.py',
#     './model-n20-ont-f-bac-racon-hax-11-cw.py',
#     './model-n20-ont-f-bac-racon-hax-23-cw.py',
#     './model-n20-ont-f-bac-racon-hax-24-cw.py',
#     './model-n20-ont-f-bac-racon-hax-11-hp.py',
#     './model-n20-ont-f-bac-racon-hax-23-hp.py',
#     './model-n20-ont-f-bac-racon-hax-24-hp.py',
    './model-n20-ont-f-bac-racon-hax-11-hp-2.py',
    './model-n20-ont-f-bac-racon-hax-23-hp-2.py',
]
dataset_paths = {
    X_TRAIN_PATH: '../dataset-n20-X-train.npy',
    Y_TRAIN_PATH: '../dataset-n20-y-train.npy',
    X_VALIDATE_PATH: '../dataset-n20-X-validate.npy',
    Y_VALIDATE_PATH: '../dataset-n20-y-validate.npy',
}
model_save_paths = [
#     '../model-11.h5',
#     '../model-23.h5',
#     '../model-24.h5',
#     '../model-11-repeat.h5',
#     '../model-23-repeat.h5',
#     '../model-24-repeat.h5',
#     '../model-11-cw.h5',
#     '../model-23-cw.h5',
#     '../model-24-cw.h5',
#     '../model-11-hp.h5',
#     '../model-23-hp.h5',
#     '../model-24-hp.h5',
    '../model-11-hp-2.h5',
    '../model-23-hp-2.h5',
]
tensorboard_output_dirs = [
#     '../model-11-tb',
#     '../model-23-tb',
#     '../model-24-tb',
#     '../model-11-tb-repeat',
#     '../model-23-tb-repeat',
#     '../model-24-tb-repeat',
#     '../model-11-tb-cw',
#     '../model-23-tb-cw',
#     '../model-24-tb-cw',
#     '../model-11-tb-hp',
#     '../model-23-tb-hp',
#     '../model-24-tb-hp',
    '../model-11-tb-hp-2',
    '../model-23-tb-hp-2',
]
finished_experiments_dir_path = '../finished-experiments/'

# class_weights_train = '../class_weight_train.npy'
class_weights_train = None

experiment_queue(experiment_paths, dataset_paths, model_save_paths, tensorboard_output_dirs, finished_experiments_dir_path, class_weights_train)