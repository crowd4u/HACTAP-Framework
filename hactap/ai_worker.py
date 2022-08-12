import abc
from typing import List
# from typing import Union

import torch
from torch.utils.data import TensorDataset

from hactap.logging import get_logger

logger = get_logger()


class BaseAIWorker(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        train_dataset: TensorDataset
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        x_test: List
    ) -> List:
        raise NotImplementedError

    @abc.abstractmethod
    def get_worker_name(
        self,
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def query(
        self,
        x: List,
        n_instances: int
    ) -> List:
        raise NotImplementedError


class BaseModel(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        x: List,
        y: List
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        x: List
    ) -> List:
        raise NotImplementedError


class BaseProbaModel(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        x: List,
        y: List
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        x: List
    ) -> List:
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(
        self,
        x: List,
        y: List,
        index: List
    ) -> List:
        raise NotImplementedError


class BaseActiveModel(BaseAIWorker):
    @abc.abstractmethod
    def teach(
        self,
        x: List,
        y: List,
        only_new: bool
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        x: List
    ) -> List:
        raise NotImplementedError

    @abc.abstractmethod
    def query(
        self,
        x: List,
        n_instances: int
    ) -> List:
        raise NotImplementedError


class AIWorker(BaseAIWorker):
    def __init__(self, model: BaseModel):
        self.model = model

    def fit(self, train_dataset: TensorDataset) -> None:
        logger.debug("Start training {}.".format(self.get_worker_name()))

        length_dataset = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))

        self.model.fit(x_train, y_train)
        return

    def predict(self, x_test: List) -> List:
        logger.debug(
            "AI worker ({}) predicts {} tasks.".format(
                self.get_worker_name(), len(x_test)
            )
        )

        return self.model.predict(x_test)

    def query(
        self,
        x: List,
        n_instances: int
    ) -> List:
        raise NotImplementedError

    def get_worker_name(self) -> str:
        return self.model.__class__.__name__


class ComitteeAIWorker(BaseAIWorker):
    def __init__(self, model: BaseActiveModel):
        self.model = model

    def fit(self, train_dataset: TensorDataset) -> None:
        logger.debug("Start training {}.".format(self.get_worker_name()))

        length_dataset = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))
        self.model.teach(x_train, y_train, only_new=True)
        return

    def predict(self, x_test: List) -> List:
        logger.debug(
            "AI worker {} predicts {} tasks.".format(
                self.get_worker_name(), len(x_test)
            )
        )
        return self.model.predict(x_test)

    def query(
        self,
        x_test: List,
        n_instances: int = 0
    ) -> List:
        query_indexes, samples = self.model.query(
            x_test, n_instances=n_instances
        )

        return query_indexes

    def get_worker_name(self) -> str:
        return self.model.__class__.__name__


class ProbaAIWorker(BaseAIWorker):
    def __init__(self, model: BaseProbaModel, threshold: float):
        self.model = model
        self._threshold = threshold

    def fit(self, train_dataset: TensorDataset) -> None:
        logger.debug("Start training {}.".format(self.get_worker_name()))

        length_dataset = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))

        self.model.fit(x_train, y_train)
        return

    def predict(self, x_test: List) -> List:
        logger.debug(
            "AI worker {} predicts {} tasks.".format(
                self.get_worker_name(), len(x_test)
            )
        )
        return self.model.predict(x_test)

    @property
    def get_threshold(self) -> float:
        return self._threshold

    def set_threshold(self, threshold: float):
        self._threshold = threshold
        return

    def predict_proba(
        self, x_test: List, y_test: List, index: List
    ) -> List:
        logger.debug(
            "AI worker ({}) predicts {} tasks.".format(
                self.get_worker_name(), len(x_test)
            )
        )

        proba = self.model.predict_proba(x_test)
        y_list = []
        pred = []
        indexes = []
        for y, p, i in zip(y_test.tolist(), proba, index):
            proba_max = max(p)
            if proba_max > self._threshold:
                y_list.append(y)
                pred.append(list(p).index(proba_max))
                indexes.append(i)

        return y_list, pred, indexes

    def query(
        self,
        x: List,
        n_instances: int
    ) -> List:
        raise NotImplementedError

    def get_worker_name(self) -> str:
        return self.model.__class__.__name__
