import abc
from typing import Callable, List, Dict
from scipy import stats, optimize
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
    def __init__(
        self,
        list_ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        max_iter_n: int = 1000,
        model: Callable = None,
        param_init: List = [],
        maxfev: int = 5000
    ) -> None:
        self._list_ai_workers = list_ai_workers
        self._learning_curve = [[accuracy_requirement for _ in range(max_iter_n)] for _ in range(len(list_ai_workers))]
        self._err_aiw = [[0 for _ in range(max_iter_n)] for _ in range(len(list_ai_workers))]
        self._iter = 1
        self.acc_req = accuracy_requirement
        self._MAX_LEN_OF_ITER = max_iter_n
        self._model_curve = model if model is not None else self._model_pow3
        self._params_init = param_init if param_init != [] else [2, 3, 0]
        self._maxfev = maxfev

    def eval_ai_worker(
        self,
        ai_worker_index: int,
        task_cluster: List[TaskCluster]
    ) -> bool:
        # update learning curve here
        err = self._get_err_of_clusters(task_cluster)
        self._err_aiw[ai_worker_index][self._iter-1] = err
        x = list(range(1, self._iter+1))
        y = self._err_aiw[ai_worker_index][:self._iter]
        if len(x) < len(self._params_init):
            return err < 1 - self.acc_req
        opt, _ = optimize.curve_fit(
            self._model_curve, x, y, self._params_init, maxfev=self._maxfev
        )
        self._update_learning_curve(ai_worker_index, self._model_curve, opt)
        return self._learning_curve[ai_worker_index][self._iter] < 1 - self.acc_req

    def increment_n_iter(self):
        self._iter += 1
        return self._iter

    def get_evaluateble_aiw_ids(self):
        # return [idx for idx, curve in enumerate(self._learning_curve) if curve[self._iter] < 1 - self.acc_req]
        return list(range(len(self._list_ai_workers)))

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
