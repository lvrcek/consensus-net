import sys

module_path = '/home/diplomski-rad/consensus-net/src/python/training/'
if module_path not in sys.path:
    print('Adding training module.')
    sys.path.append(module_path)
from training import experiment_queue
from training import X_TRAIN_PATH, X_VALIDATE_PATH, Y_TRAIN_PATH, Y_VALIDATE_PATH

experiment_paths = [
#     './model-n20-ont-racon-hax-1.py',
#     './model-n20-ont-racon-hax-2.py',
#     './model-n20-ont-racon-hax-3.py',
#     './model-n20-ont-racon-hax-4.py',
#     './model-n20-ont-racon-hax-5.py',
#     './model-n20-ont-racon-hax-6.py',
#     './model-n20-ont-racon-hax-7.py',
#     './model-n20-ont-racon-hax-8.py',
#     './model-n20-ont-racon-hax-9.py',
#     './model-n20-ont-racon-hax-10.py',
#     './model-n20-ont-racon-hax-11.py',
#     './model-n20-ont-racon-hax-12.py',
#     './model-n20-ont-racon-hax-13.py',
#     './model-n20-ont-racon-hax-14.py',
#     './model-n20-ont-racon-hax-15.py',
#     './model-n20-ont-racon-hax-16.py',
#     './model-n20-ont-racon-hax-17.py',
#     './model-n20-ont-racon-hax-18.py',
#     './model-n20-ont-racon-hax-19.py',
#     './model-n20-ont-racon-hax-20.py',
#     './model-n20-ont-racon-hax-21.py',
#     './model-n20-ont-racon-hax-22.py',
#     './model-n20-ont-racon-hax-11-TB.py',
#     './model-n20-ont-racon-hax-15-TB.py',
#     './model-n20-ont-racon-hax-23.py',
#     './model-n20-ont-racon-hax-24.py',
    './model-n20-ont-racon-hax-25.py',
    './model-n20-ont-racon-hax-26.py',
    './model-n20-ont-racon-hax-27.py',
    './model-n20-ont-racon-hax-28.py',
]
dataset_paths = {
    X_TRAIN_PATH: '../dataset-n20-X-train.npy',
    Y_TRAIN_PATH: '../dataset-n20-y-train.npy',
    X_VALIDATE_PATH: '../dataset-n20-X-validate.npy',
    Y_VALIDATE_PATH: '../dataset-n20-y-validate.npy'
}
model_save_paths = [
#     '../model-1.h5',
#     '../model-2.h5',
#     '../model-3.h5',
#     '../model-4.h5',
#     '../model-5.h5',
#     '../model-6.h5',
#     '../model-7.h5',
#     '../model-8.h5',
#     '../model-9.h5',
#     '../model-10.h5', 
#     '../model-11.h5',
#     '../model-12.h5', 
#     '../model-13.h5',
#     '../model-14.h5',
#     '../model-15.h5',
#     '../model-16.h5',
#     '../model-17.h5',
#     '../model-18.h5',
#     '../model-19.h5',
#     '../model-20.h5',
#     '../model-21.h5',
#     '../model-22.h5',
#     '../model-11-TB.h5',
#     '../model-15-TB.h5'
#     '../model-23.h5',
#     '../model-24.h5',
    '../model-25.h5',
    '../model-26.h5',
    '../model-27.h5',
    '../model-28.h5',
]
tensorboard_output_dirs = [
#     '../model-23-tb',
#     '../model-24-tb',
    '../model-25-tb',
    '../model-26-tb',
    '../model-27-tb',
    '../model-28-tb',
]
finished_experiments_dir_path = '../finished-experiments/'

experiment_queue(experiment_paths, dataset_paths, model_save_paths, tensorboard_output_dirs, finished_experiments_dir_path)