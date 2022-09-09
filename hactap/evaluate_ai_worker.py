import abc
from typing import List

from hactap.ai_worker import BaseAIWorker
from hactap.task_cluster import TaskCluster


class BaseEvalClass(object, metaclass=abc.ABCMeta):
    def __init__(
        self,
        list_ai_workers: List[BaseAIWorker]
    ) -> None:
        raise NotImplementedError

    def eval_ai_worker(
        self,
        ai_worker_index: int,
        task_cluster: List[TaskCluster]
    ) -> bool:
        raise NotImplementedError


class EvalAIWByLearningCurve(BaseEvalClass):
    pass
