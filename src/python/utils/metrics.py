import numpy as np

from abc import abstractmethod
from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score


class Metric(Callback):
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y_true = np.argmax(y, axis=1)

    def on_train_end(self, logs=None):
        self._print_start_info()
        y_pred = np.asarray(self.model.predict(self.X))
        y_pred = np.argmax(y_pred, axis=1)
        score = self._calc_metric(y_pred)
        self._print_metric_score(score)

    @abstractmethod
    def _print_start_info(self):
        pass

    @abstractmethod
    def _calc_metric(self, y_pred):
        pass

    @abstractmethod
    def _print_metric_score(self, score):
        pass


class Precision(Metric):
    def __init__(self, model, X, y, mode):
        Metric.__init__(self, model, X, y)
        self.mode = mode

    def _print_start_info(self):
        print('Calculating Precision-{} ...'.format(self.mode))

    def _calc_metric(self, y_pred):
        return precision_score(self.y_true, y_pred, average=self.mode)

    def _print_metric_score(self, score):
        print('Precision: {:.4f}, mode: {}'.format(score, self.mode))


class Recall(Metric):
    def __init__(self, model, X, y, mode):
        Metric.__init__(self, model, X, y)
        self.mode = mode

    def _print_start_info(self):
        print('Calculating Recall-{} ...'.format(self.mode))

    def _calc_metric(self, y_pred):
        return recall_score(self.y_true, y_pred, average=self.mode)

    def _print_metric_score(self, score):
        print('Recal: {:.4f}, mode: {}'.format(score, self.mode))


class F1(Metric):
    def __init__(self, model, X, y, mode):
        Metric.__init__(self, model, X, y)
        self.mode = mode

    def _print_start_info(self):
        print('Calculating F1-{} ...'.format(self.mode))

    def _calc_metric(self, y_pred):
        return f1_score(self.y_true, y_pred, average=self.mode)

    def _print_metric_score(self, score):
        print('F1: {:.4f}, mode: {}'.format(score, self.mode))
