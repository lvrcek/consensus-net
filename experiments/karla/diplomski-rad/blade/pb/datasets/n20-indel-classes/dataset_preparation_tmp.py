import numpy as np

import sys
module_path = '/home/diplomski-rad/consensus-net/src/python/dataset/'
if module_path not in sys.path:
    print('Adding dataset module.')
    sys.path.append(module_path)

import dataset

X_paths = ['/home/diplomski-rad/blade/pb/pileups/e-coli-NCTC86-all-contigs-indel-classes/pileups-X-0-indels.npy',
           '/home/diplomski-rad/blade/pb/pileups/m-morgani-NCTC235-all-contigs-indel-classes/pileups-X-0-indels.npy',
           '/home/diplomski-rad/blade/pb/pileups/s-enterica-NCTC92-all-contigs-indel-classes/pileups-X-0-indels.npy',
           '/home/diplomski-rad/blade/pb/pileups/s-enterica-NCTC129-all-contigs-indel-classes/pileups-X-0-indels.npy']
y_paths = ['/home/diplomski-rad/blade/pb/pileups/e-coli-NCTC86-all-contigs-indel-classes/pileups-y-0.npy',
           '/home/diplomski-rad/blade/pb/pileups/m-morgani-NCTC235-all-contigs-indel-classes/pileups-y-0.npy',
           '/home/diplomski-rad/blade/pb/pileups/s-enterica-NCTC92-all-contigs-indel-classes/pileups-y-0.npy',
           '/home/diplomski-rad/blade/pb/pileups/s-enterica-NCTC129-all-contigs-indel-classes/pileups-y-0.npy']
neighbourhood_size = 20
save_directory_path = './'

X, y, X_save_paths, y_save_paths = dataset.create_dataset_with_neighbourhood(
    X_paths, y_paths, neighbourhood_size, mode='training', save_directory_path=save_directory_path)

X, y = X[0], y[0]

X, y, X_train, X_validate, y_train, y_validate = dataset.read_dataset_and_reshape_for_conv(
    X_save_paths, y_save_paths, 0.1)

np.save('./dataset-n20-X-reshaped', X)
np.save('./dataset-n20-y-reshaped', y)
np.save('./dataset-n20-X-reshaped-train', X_train)
np.save('./dataset-n20-X-reshaped-validate', X_validate)
np.save('./dataset-n20-y-reshaped-train', y_train)
np.save('./dataset-n20-y-reshaped-validate', y_validate)