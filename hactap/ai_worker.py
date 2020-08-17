import torch
from sklearn.base import BaseEstimator
from hactap.logging import get_logger

logger = get_logger()


class AIWorker(BaseEstimator):
    def __init__(self, model, skip_update=False):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        print('device', self.device, self.use_cuda)
        self.model = model

        # if self.use_cuda:
        #     self.model.estimator.to(self.device)

        self.skip_update = skip_update
        self.trained = False

    def fit(self, X, y):
        print('device', self.device, self.use_cuda)
        if self.skip_update and self.trained:
            logger.warning(
                'The training was skipped ({}).'.format(
                    self.model.estimator.__class__.__name__
                )
            )
            return self.model.estimator

        if self.use_cuda:
            X.to(self.device)
            y.to(self.device)

        self.model.fit(X, y)

        self.trained = True
        return self.model.estimator

    def predict(self, x):
        x.to(self.device)
        return self.model.predict(x)

    def query(self, x, n_instances):
        return self.model.query(x, n_instances=n_instances)
