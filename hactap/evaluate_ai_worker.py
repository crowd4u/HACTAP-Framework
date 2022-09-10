import abc
from typing import List
from scipy import stats
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

    def get_evaluateble_aiw_ids(self):
        raise NotImplementedError

    def increment_n_iter(self):
        raise NotImplementedError


class EvalAIWByBinTest(BaseEvalClass):
    def __init__(
        self,
        list_ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        significance_level: float
    ) -> None:
        self._list_ai_workers = list_ai_workers
        self._next_iter = [1 for _ in range(len(list_ai_workers))]
        self._n_skip = [1 for _ in range(len(list_ai_workers))]
        self._iter = 1
        self.accuracy_requirement = accuracy_requirement
        self.significance_level = significance_level

    def eval_ai_worker(
        self,
        ai_worker_index: int,
        task_cluster: List[TaskCluster]
    ) -> bool:
        if self._next_iter[ai_worker_index] > self._iter:
            return False
        else:
            for tc in task_cluster:
                if self._bin_test(tc):
                    self._next_iter[ai_worker_index] = self._iter + 1
                    return True
            self._update_n_skip(ai_worker_index)
            self._next_iter[ai_worker_index] += self._n_skip[ai_worker_index]
            return False

    def increment_n_iter(self):
        self._iter += 1
        return self._iter

    def get_evaluateble_aiw_ids(self):
        return [idx for idx, next in enumerate(self._next_iter) if next <= self._iter]

    def _update_n_skip(self, aiw_index: int) -> None:
        self._n_skip[aiw_index] *= 2

    def _bin_test(self, task_cluster: TaskCluster) -> bool:
        p_value = stats.binom_test(
            task_cluster.match_rate_with_human,
            task_cluster.match_rate_with_human + task_cluster.conflict_rate_with_human, # NOQA
            p=self.accuracy_requirement,
            alternative='greater'
        )
        return p_value < self.significance_level


class EvalAIWByLearningCurve(BaseEvalClass):
    pass
