import sys

module_path = '/home/diplomski-rad/consensus-net/src/python/training/'
if module_path not in sys.path:
    print('Adding training module.')
    sys.path.append(module_path)
from training import experiment_queue
from training import X_TRAIN_PATH, X_VALIDATE_PATH, Y_TRAIN_PATH, Y_VALIDATE_PATH

experiment_paths = ['./model-n15-s-cerv-ont-indel-classes-1.py',
                    './model-n15-s-cerv-ont-indel-classes-2.py',
                    './model-n15-s-cerv-ont-indel-classes-3.py',
                    './model-n15-s-cerv-ont-indel-classes-4.py',
                    './model-n15-s-cerv-ont-indel-classes-5.py',
                    './model-n15-s-cerv-ont-indel-classes-5.py',
                    './model-n15-s-cerv-ont-indel-classes-6.py',
                    './model-n15-s-cerv-ont-indel-classes-7.py',
                    './model-n15-s-cerv-ont-indel-classes-8.py',
                    './model-n15-s-cerv-ont-indel-classes-9.py',
                    './model-n15-s-cerv-ont-indel-classes-10.py',
                    './model-n15-s-cerv-ont-indel-classes-11.py',
                    './model-n15-s-cerv-ont-indel-classes-12.py',
                    './model-n15-s-cerv-ont-indel-classes-13.py',
                    './model-n15-s-cerv-ont-indel-classes-14.py']
dataset_paths = {
    X_TRAIN_PATH: '../dataset-n20-X-reshaped-train.npy',
    Y_TRAIN_PATH: '../dataset-n20-y-reshaped-train.npy',
    X_VALIDATE_PATH: '../dataset-n20-X-reshaped-validate.npy',
    Y_VALIDATE_PATH: '../dataset-n20-y-reshaped-validate.npy'
}
model_save_paths = ['../model-1.h5',
                    '../model-2.h5',
                    '../model-3.h5',
                    '../model-4.h5',
                    '../model-5.h5',
                    '../model-6.h5',
                    '../model-7.h5',
                    '../model-8.h5',
                    '../model-9.h5',
                    '../model-10.h5', 
                    '../model-11.h5',
                    '../model-12.h5', 
                    '../model-13.h5',
                    '../model-14.h5',]
finished_experiments_dir_path = '../finished-experiments/'

experiment_queue(experiment_paths, dataset_paths, model_save_paths, finished_experiments_dir_path)