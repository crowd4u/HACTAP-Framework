import abc
from statistics import median
from typing import Any, Dict, List, Optional

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


class BaseAIWorkerWithFeedback(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        train_dataset: TensorDataset,
        feedback: Dict
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

    @property
    def get_threshold(self) -> float:
        return self._threshold

    def set_threshold(self, threshold: float):
        self._threshold = threshold
        return

    def predict(self, x_test: List) -> List[Optional[Any]]:
        logger.debug(
            "AI worker ({}) predicts {} tasks. With threshold: {}".format(
                self.get_worker_name(), len(x_test), self._threshold
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
        model: BaseProbaModel,
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
            "AI worker ({}) predicts {} tasks. With threshold: {}".format(
                self.get_worker_name(), len(x_test), self._threshold
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


class ProbaMedianAIWorker(ProbaAIWorker):
    def __init__(
        self,
        model: BaseProbaModel,
        min_threshold: float = 0.0,
        offset: float = 0.0
    ):
        super().__init__(model, min_threshold)
        self._offset = offset

    def predict(self, x_test: List) -> List[Optional[Any]]:
        logger.debug(
            "AI worker ({}) predicts {} tasks. With threshold: {}".format(
                self.get_worker_name(), len(x_test), self._threshold
            )
        )

        proba = self.model.predict_proba(x_test)

        proba_med = [median(distro) for distro in proba.T]
        pred = []
        for p in proba:
            proba_max = max(p)
            idx_max = list(p).index(proba_max)
            if proba_max > self._threshold and proba_max > (proba_med[idx_max] + self._offset):
                pred.append(idx_max)
            else:
                pred.append(None)

        return pred


class AIWorkerWithFeedback(BaseAIWorkerWithFeedback, ProbaAIWorker):
    def __init__(
        self,
        model: BaseProbaModel,
        inital_threshold: float = 0.5,
        lr: float = 0.3,
        offset: float = 0.05
    ) -> None:
        self.model = model
        self._feedback = {}
        self._inital_th = inital_threshold
        self._ths = []
        self._lr = lr
        self._offset = offset
        self._iter_n = 0

    def fit(self, train_dataset: TensorDataset, feedback: Dict = None) -> None:
        logger.debug("Start training {}.".format(self.get_worker_name()))

        if feedback is not None and feedback != {}:
            self._feedback = feedback
            logger.debug("AIW {}, (id:{}) recieves feedback: {}".format(
                self.get_worker_name(), id(self), feedback
            ))

        self._get_new_feedback = True
        self._iter_n += 1

        length_dataset = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))

        self.model.fit(x_train, y_train)
        return

    def predict(self, x_test: List) -> List:
        proba = self.model.predict_proba(x_test)

        if len(self._ths) == 0:
            self._ths = [self._inital_th for _ in range(len(proba[0]))]

        if self._get_new_feedback:
            self._get_new_feedback = False
            for idx in self._feedback.keys():
                th = self._ths[idx]
                fb = self._feedback[idx]
                # acc_sample = fb["n_of_match"] / (fb["n_of_match"] + fb["n_of_conflict"])
                if fb["is_accepted"]:
                    continue
                diff = abs(self._lr * 1 / self._iter_n + self._offset)\
                    * (-1 if th > self._inital_th else 1)
                new_th = diff + th
                new_th = 0.95 if new_th >= 1 else 0 if new_th <= 0 else new_th
                self._ths[idx] = new_th

        logger.debug(
            "Iter_n:{} AI worker ({}) id:{} predicts {} tasks. with thresholds: {}".format(
                self._iter_n, self.get_worker_name(), id(self), len(x_test), self._ths
            )
        )

        pred = []
        for p in proba:
            proba_max = max(p)
            idx_max = list(p).index(proba_max)
            if proba_max > self._ths[idx_max]:
                pred.append(idx_max)
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
