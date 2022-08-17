import abc
from typing import Any, List, Optional

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
    ) -> List[Optional[Any]]:
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
        return self.__class__.__name__+" with "+self.model.__class__.__name__


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
    def __init__(self, model: BaseModel, threshold: float):
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

    @property
    def get_threshold(self) -> float:
        return self._threshold

    def set_threshold(self, threshold: float):
        self._threshold = threshold
        return

    def predict(self, x_test: List) -> List[Optional[Any]]:
        logger.debug(
            "AI worker ({}) predicts {} tasks.".format(
                self.get_worker_name(), len(x_test)
            )
        )

        proba = self.model.predict_proba(x_test)
        pred = []
        for p in proba:
            proba_max = max(p)
            if proba_max > self._threshold:
                pred.append(list(p).index(proba_max))
            else:
                pred.append(None)
        return pred

    def query(
        self,
        x: List,
        n_instances: int
    ) -> List:
        raise NotImplementedError

    def get_worker_name(self) -> str:
        return self.__class__.__name__+" with "+self.model.__class__.__name__


class ActiveProbaAIWorker(ProbaAIWorker):
    def __init__(
        self,
        model: BaseModel,
        inital_threshold: float = 0.0,
        final_threshold: float = 0.99,
        threshold_diff: float = 0.1
    ):
        super().__init__(model, inital_threshold)
        self._inital_th = inital_threshold
        self._final_th = final_threshold
        self._th_diff = threshold_diff

    def predict(self, x_test: List) -> List[Optional[Any]]:
        logger.debug(
            "AI worker ({}) predicts {} tasks.".format(
                self.get_worker_name(), len(x_test)
            )
        )

        proba = self.model.predict_proba(x_test)
        pred = []
        for p in proba:
            proba_max = max(p)
            if proba_max > self._threshold:
                pred.append(list(p).index(proba_max))
            else:
                pred.append(None)

        next_th = self.get_threshold + self._th_diff
        if (self._th_diff < 0 and next_th < self._final_th) or (self._th_diff > 0 and next_th > self._final_th):  # NOQA
            next_th = self._final_th
        self.set_threshold(next_th)

        return pred
