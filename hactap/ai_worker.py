from sklearn.base import BaseEstimator
from hactap.logging import get_logger

logger = get_logger()


class AIWorker(BaseEstimator):
    def __init__(self, model, skip_update=False):
        self.model = model
        self.skip_update = skip_update
        self.trained = False

    def fit(self, X, y):
        if self.skip_update and self.trained:
            logger.warning(
                'The training was skipped ({}).'.format(
                    self.model.estimator.__class__.__name__
                )
            )
            return self.model.estimator

        self.model.fit(X, y)
        self.trained = True
        return self.model.estimator

    def predict(self, x):
        return self.model.predict(x)

    def query(self, x, n_instances):
        return self.model.query(x, n_instances=n_instances)
