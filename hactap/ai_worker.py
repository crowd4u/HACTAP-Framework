import torch

from hactap.logging import get_logger


logger = get_logger()


class AIWorker:
    def __init__(self, model):
        self.model = model

    def fit(self, train_dataset):
        logger.debug("Start training {}.".format(self.get_worker_name()))

        length_dataset = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))

        self.model.fit(x_train, y_train)
        return

    def predict(self, x_test):
        logger.debug(
            "AI worker {} predicts {} tasks.".format(
                self.get_worker_name(), len(x_test)
            )
        )

        return self.model.predict(x_test)

    def query(self, x_test, n_instances=None):
        query_indexes, samples = self.model.query(
            x_test, n_instances=n_instances
        )

        return query_indexes

    def get_worker_name(self):
        return self.model.__class__.__name__


class ComitteeAIWorker(AIWorker):
    def __init__(self, model):
        super().__init__(
            model
        )

    def fit(self, train_dataset):
        logger.debug("Start training {}.".format(self.get_worker_name()))

        length_dataset = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))
        self.model.teach(x_train, y_train, only_new=True)
        return
