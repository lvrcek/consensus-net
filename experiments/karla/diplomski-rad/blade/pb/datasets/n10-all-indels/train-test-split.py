import numpy as np

import sys
module_path = '/home/diplomski-rad/consensus-net/src/python/dataset/'
if module_path not in sys.path:
    print('Adding dataset module.')
    sys.path.append(module_path)

import dataset

X, y, X_train, X_validate, y_train, y_validate = dataset.read_dataset_and_reshape_for_conv(
    './dataset-all-n10-X-indels.npy', './dataset-all-n10-y-indels.npy', 0.1)


np.save('./dataset-n10-X-reshaped-script', X)
np.save('./dataset-n10-y-reshaped-scipt', y)
np.save('./dataset-n10-X-reshaped-train', X_train)
np.save('./dataset-n10-X-reshaped-validate', X_validate)
np.save('./dataset-n10-y-reshaped-train', y_train)
np.save('./dataset-n10-y-reshaped-validate', y_validate)
