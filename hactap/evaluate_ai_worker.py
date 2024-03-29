import abc
from typing import Callable, List, Dict
from scipy import stats, optimize
from numpy import inf
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

    def increment_n_iter(self):
        raise NotImplementedError

    def report(self) -> Dict:
        raise NotImplementedError

    @property
    def n_iter(self):
        raise NotImplementedError


class NonEvalClass(BaseEvalClass):
    def __init__(self, list_ai_workers: List[BaseAIWorker]):
        self._list_ai_workers = list_ai_workers
        self._iter = 1

    def increment_n_iter(self):
        self._iter += 1
        return self._iter

    def eval_ai_worker(
        self,
        ai_worker_index: int,
        task_cluster: List[TaskCluster]
    ) -> bool:
        return True

    def report(self) -> Dict:
        eval_log = []
        for idx, aiw in enumerate(self._list_ai_workers):
            eval_report = {
                "id": idx,
                "name": aiw.get_worker_name()
            }
            eval_log.append(eval_report)
        return {
            "EvalType": self.__class__.__name__,
            "iter": self._iter,
            "evals": eval_log
        }

    @property
    def n_iter(self):
        return self._iter


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
        self.accuracy_aiw = [0.0 for _ in range(len(list_ai_workers))]

    def eval_ai_worker(
        self,
        ai_worker_index: int,
        task_cluster: List[TaskCluster]
    ) -> bool:
        self.accuracy_aiw[ai_worker_index] = self.get_accuracy_of_ai_worker(task_cluster)
        if self._next_iter[ai_worker_index] > self._iter:
            return False
        else:
            for tc in task_cluster:
                if self._bin_test(tc):
                    self._next_iter[ai_worker_index] = self._iter + 1
                    return True
            self._update_n_skip(ai_worker_index)
            return False

    def get_accuracy_of_ai_worker(self, task_clusters: List[TaskCluster]) -> float:
        n_match = 0
        n_conflict = 0
        for task_cluster in task_clusters:
            n_match += task_cluster.match_rate_with_human
            n_conflict += task_cluster.conflict_rate_with_human
        return n_match / (n_match + n_conflict) if n_conflict + n_match != 0 else 0

    def increment_n_iter(self):
        self._iter += 1
        return self._iter

    def report(self) -> Dict:
        eval_log = []
        for idx, aiw in enumerate(self._list_ai_workers):
            eval_report = {
                "id": idx,
                "name": aiw.get_worker_name(),
                "next_iter": self._next_iter[idx],
                "n_skip": self._n_skip[idx],
                "accuracy_aiw": self.accuracy_aiw[idx]
            }
            eval_log.append(eval_report)
        return {
            "EvalType": self.__class__.__name__,
            "acc_req": self.accuracy_requirement,
            "iter": self._iter,
            "evals": eval_log
        }

    @property
    def n_iter(self):
        return self._iter

    def _update_n_skip(self, aiw_index: int) -> None:
        self._n_skip[aiw_index] *= 2
        self._next_iter[aiw_index] += self._n_skip[aiw_index]

    def _bin_test(self, task_cluster: TaskCluster) -> bool:
        p_value = stats.binom_test(
            task_cluster.match_rate_with_human,
            task_cluster.match_rate_with_human + task_cluster.conflict_rate_with_human, # NOQA
            p=self.accuracy_requirement,
            alternative='greater'
        )
        return p_value < self.significance_level


class EvalAIWByLearningCurve(BaseEvalClass):
    def __init__(
        self,
        list_ai_workers: List[BaseAIWorker],
        error_rate_requirement: float,
        significance_level: float,
        max_iter_n: int = 1000,
        model: Callable = None,
        maxfev: int = 5000,
        n_skip_init: int = 8
    ) -> None:
        self._list_ai_workers = list_ai_workers
        self._iter = 1
        self._error_rate_requirement = error_rate_requirement
        self.significance_level = significance_level
        self._learning_curve = [[error_rate_requirement for _ in range(max_iter_n)] for _ in range(len(list_ai_workers))]
        self._err_aiw = [[1.0 for _ in range(max_iter_n)] for _ in range(len(list_ai_workers))]
        self._MAX_LEN_OF_ITER = max_iter_n
        self._model_curve = model if model is not None else self._model_pow3
        self._maxfev = maxfev
        self._next_iter = [n_skip_init for _ in range(len(list_ai_workers))]
        self._n_skip_init = n_skip_init
        self._is_accepted = [False for _ in range(len(list_ai_workers))]

    def eval_ai_worker(
        self,
        ai_worker_index: int,
        task_cluster: List[TaskCluster]
    ) -> bool:
        err = self._get_err_of_clusters(task_cluster)
        self._err_aiw[ai_worker_index][self._iter-1] = err
        x = list(range(1, self._iter+1))
        y = self._err_aiw[ai_worker_index][:self._iter]
        if self.n_iter < self._n_skip_init:
            return self.eval_ai_worker_by_bin(ai_worker_index, task_cluster)
        if self._next_iter[ai_worker_index] > self.n_iter:
            return self._is_accepted[ai_worker_index]
        else:
            try:
                opt, _ = optimize.curve_fit(
                    self._model_curve, x, y, maxfev=self._maxfev, bounds=([0,0,0], [inf, inf, inf])
                )
                self._update_learning_curve(ai_worker_index, self._model_curve, opt)
                accepted = self._learning_curve[ai_worker_index][self._iter] < self._error_rate_requirement
            except RuntimeError:
                accepted = self.eval_ai_worker_by_bin(ai_worker_index, task_cluster)
            self._is_accepted[ai_worker_index] = accepted
            if accepted:
                self._next_iter[ai_worker_index] = self._iter + 1
            else:
                pos = self._WhereLearningCurveSatisfyRequirement(ai_worker_index)
                self._next_iter[ai_worker_index] = (pos - self._iter) // 2 + self._iter
            return accepted

    def _WhereLearningCurveSatisfyRequirement(self, aiw_idx: int):
        pos = self._iter
        while pos < self._MAX_LEN_OF_ITER:
            if self._learning_curve[aiw_idx][pos-1] < self._error_rate_requirement:
                break
            pos += 1
        return pos

    def increment_n_iter(self):
        self._iter += 1
        return self._iter

    def report(self) -> Dict:
        eval_log = []
        for idx, aiw in enumerate(self._list_ai_workers):
            eval_report = {
                "id": idx,
                "name": aiw.get_worker_name(),
                "err": self._err_aiw[idx],
                "learning_curve": self._learning_curve[idx]
            }
            eval_log.append(eval_report)
        return {
            "EvalType": self.__class__.__name__,
            "acc_req": 1 - self._error_rate_requirement,
            "iter": self._iter,
            "evals": eval_log
        }

    def eval_ai_worker_by_bin(
        self,
        ai_worker_index: int,
        task_cluster: List[TaskCluster]
    ) -> bool:
        for tc in task_cluster:
            if self._bin_test(tc):
                return True
        return False

    def _bin_test(self, task_cluster: TaskCluster) -> bool:
        p_value = stats.binom_test(
            task_cluster.match_rate_with_human,
            task_cluster.match_rate_with_human + task_cluster.conflict_rate_with_human, # NOQA
            p=1-self._error_rate_requirement,
            alternative='greater'
        )
        return p_value < self.significance_level

    def _get_err_of_clusters(self, task_clusters: List[TaskCluster]):
        conflicts = sum([tc.conflict_rate_with_human for tc in task_clusters])
        matches = sum([tc.match_rate_with_human for tc in task_clusters])
        return conflicts / (conflicts + matches) if conflicts + matches != 0 else 0

    def _model_pow3(self, x, a, b, c):
        return a * x**(-1 * b) + c

    def _update_learning_curve(
        self,
        aiw_index: int,
        func: Callable,
        params: List
    ) -> None:
        self._learning_curve[aiw_index] = [func(x, *params) for x in range(self._MAX_LEN_OF_ITER)]

    @property
    def n_iter(self):
        return self._iter
