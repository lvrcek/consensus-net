import numpy as np

X, y = np.load('./dataset-n10-X-reshaped.npy'), np.load('./dataset-n10-X-reshaped.npy')

import progressbar

with progressbar.ProgressBar(max_value=X.shape[0]) as progress_bar:
    for i, xi in enumerate(X):
        progress_bar.update(i)
        np.save('./X/{}'.format(i), xi)
