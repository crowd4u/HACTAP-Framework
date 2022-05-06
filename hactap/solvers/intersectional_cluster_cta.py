from decimal import Clamped
from typing import List
from typing import Callable
from typing import Tuple

import random
from collections import Counter

import itertools
from grpc import Call
from sqlalchemy import func
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset

from hactap import solvers
from hactap.logging import get_logger
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import BaseAIWorker
from hactap.reporter import Reporter
from hactap.task_cluster import TaskCluster

logger = get_logger()

PREDICT_BATCH_SIZE = 10_000


def key_of_task_cluster_k(x: Tuple[int, int, int]) -> int:
    return x[0]


def group_by_task_cluster(
    ai_worker: BaseAIWorker,
    dataset: Dataset,
    indexes: List[int]
) -> List:
    test_set_loader = DataLoader(
        dataset, batch_size=PREDICT_BATCH_SIZE
    )

    test_set_predict = []
    test_set_y = []

    for index, (sub_test_x, sub_test_y) in enumerate(test_set_loader):
        sub_test_predict = ai_worker.predict(sub_test_x)

        # print('sub_test_predict', sub_test_predict)
        # print('sub_test_y', sub_test_y.tolist())

        test_set_predict.extend(sub_test_predict)
        test_set_y.extend(sub_test_y.tolist())

    # print('dataset_predict', len(test_set_predict))
    # print('dataset_y', len(test_set_y))
    # print('dataset_indexes', len(indexes))

    tcs = itertools.groupby(
        sorted(
            list(zip(test_set_predict, test_set_y, indexes)),
            key=key_of_task_cluster_k
        ),
        key_of_task_cluster_k
    )

    return list(map(lambda x: (x[0], list(x[1])), tcs))


class intersectional_cluster_CTA(solvers.CTA):
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        human_crowd_batch_size: int,
        ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        n_of_classes: int,
        significance_level: float,
        reporter: Reporter,
        retire_used_test_data: bool = False,
        n_of_majority_vote: int = 1,
        clustering_function: Callable = None,
    ) -> None:
        super().__init__(
            tasks,
            human_crowd,
            human_crowd_batch_size,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            significance_level,
            reporter,
            retire_used_test_data,
            n_of_majority_vote
        )
        self.__clustering_funcion = clustering_function

    @property
    def clustering_function(self) -> Callable:
        return self.__clustering_funcion

    def run(self) -> Tasks:
        self.initialize()
        self.report_log()

        self.assign_to_human_workers(
            n_of_majority_vote=1
        )
        self.report_log()

        while not self.check_n_of_class():
            self.assign_to_human_workers(
                n_of_majority_vote=1
            )
            self.report_log()

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters_by_any()
            random.shuffle(task_cluster_candidates)

            # assign tasks to accepted task clusters
            for task_cluster_k in task_cluster_candidates:
                if self.tasks.is_completed:
                    break

                task_cluster_k.update_status(self.tasks)
                accepted = self._evalate_task_cluster_by_bin_test(
                    task_cluster_k
                )

                if accepted:
                    self.assign_tasks_to_task_cluster(task_cluster_k)

            self.assign_to_human_workers()
            self.report_log()

        self.finalize()

        return self.tasks

    def list_task_clusters_by_any(self) -> List[TaskCluster]:
        task_clusters_by_ai_worker = []

        for index, _ in enumerate(self.ai_workers):
            task_clusters_by_ai_worker.extend(
                self.create_task_cluster_from_ai_worker(index)
            )

        task_clusters_by_any_function = self.create_task_cluster_from_any_function() # NOQA

        task_clusters = self.intersection_of_task_clusters(
                task_clusters_by_ai_worker,
                task_clusters_by_any_function
            )
        return task_clusters

    def create_task_cluster_from_any_function(
        self,
        function: Callable = None
    ) -> List[TaskCluster]:
        task_clusters: List[TaskCluster] = []
        cluster_function = function if function is not None else self.clustering_funcion # NOQA
        ai_worker = None

        tc_train = group_by_task_cluster(
            ai_worker,
            self.tasks.train_set,
            self.tasks.train_indexes
        )

        tc_test = group_by_task_cluster(
            ai_worker,
            self.tasks.test_set,
            self.tasks.test_indexes
        )

        tc_remain = list(group_by_task_cluster(
            ai_worker,
            self.tasks.X_assignable,
            self.tasks.assignable_indexes
        ))

        for key, items_of_tc_test in tc_test:
            # print('key', key)
            # print('tc_train', tc_train)
            # print(key, items_of_tc)
            human_labels = list(map(lambda x: x[1], items_of_tc_test))
            occurence_count = Counter(human_labels)
            max_human_label = occurence_count.most_common(1)[0][0]
            # print(human_labels)
            # print(max_human_label)

            # print('items_of_tc_test', items_of_tc_test)
            items_of_tc_train = []
            _items_of_tc_train = list(filter(lambda x: x[0] == key, tc_train)) # NOQA
            # print('items_of_tc_train', items_of_tc_train)

            if len(_items_of_tc_train) == 1:
                items_of_tc_train = _items_of_tc_train[0][1]

            items_of_tc_remain = []
            _items_of_tc_remain = list(filter(lambda x: x[0] == key, tc_remain)) # NOQA

            if len(_items_of_tc_remain) == 1:
                # print(_items_of_tc_remain[0][1])
                items_of_tc_remain = _items_of_tc_remain[0][1]
            # print('items_of_tc_remain', items_of_tc_remain)

            rule = {
                "rule": {
                    "from": key,
                    "to": max_human_label
                },
                "stat": {
                    # "y_pred": list(map(lambda x: x[0], items_of_tc_test)),
                    # "answerable_tasks_ids": list(map(lambda x: x[2], items_of_tc_remain)), # NOQA

                    "y_pred_test": list(map(lambda x: x[0], items_of_tc_test)),
                    "y_pred_train": list(map(lambda x: x[0], items_of_tc_train)), # NOQA
                    "y_pred_remain": list(map(lambda x: x[0], items_of_tc_remain)), # NOQA

                    "y_pred_test_human": list(map(lambda x: x[1], items_of_tc_test)), # NOQA
                    "y_pred_train_human": list(map(lambda x: x[1], items_of_tc_train)), # NOQA
                    "y_pred_remain_human": list(map(lambda x: x[1], items_of_tc_remain)), # NOQA

                    "y_pred_test_ids": list(map(lambda x: x[2], items_of_tc_test)), # NOQA
                    "y_pred_train_ids": list(map(lambda x: x[2], items_of_tc_train)), # NOQA
                    "y_pred_remain_ids": list(map(lambda x: x[2], items_of_tc_remain)) # NOQA
                }
            }

            task_clusters.append(
                TaskCluster(None, rule)
            )

        return task_clusters

    def intersection_of_task_clusters(
        task_clusters_with_ai_worker: List(TaskCluster),
        task_clusters_without_ai_worker: List(TaskCluster)
    ) -> List(TaskCluster):
        pass
