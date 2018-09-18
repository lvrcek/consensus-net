import numpy as np

import sys
module_path = '/home/diplomski-rad/consensus-net/src/python/dataset/'
if module_path not in sys.path:
    print('Adding dataset module.')
    sys.path.append(module_path)

import dataset

X_paths = [
    '/home/diplomski-rad/blade/ont/racon-hax-pileups/s-cerevisiae-r7-racon-MSA/pileups-X-0.npy',
    '/home/diplomski-rad/blade/ont/racon-hax-pileups/s-cerevisiae-r9-racon-MSA/pileups-X-0.npy'
]
y_paths = [
    '/home/diplomski-rad/blade/ont/racon-hax-pileups/s-cerevisiae-r7-racon-MSA/pileups-y-0.npy',
    '/home/diplomski-rad/blade/ont/racon-hax-pileups/s-cerevisiae-r9-racon-MSA/pileups-y-0.npy'
]
neighbourhood_size = 20
save_directory_path = './'

X, y = dataset.create_dataset_with_neighbourhood(neighbourhood_size, mode='training',
    X_paths=X_paths, y_paths=y_paths)

X, y = X[0], y[0]


from sklearn.model_selection import train_test_split

validation_size = 0.1

X_train, X_validate, y_train, y_validate = train_test_split(
            X, y, test_size=validation_size)

np.save('./dataset-n20-X-train', X_train)
np.save('./dataset-n20-X-validate', X_validate)
np.save('./dataset-n20-y-train', y_train)
np.save('./dataset-n20-y-validate', y_validate)