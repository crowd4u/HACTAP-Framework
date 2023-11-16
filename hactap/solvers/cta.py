from scipy import stats
import random
from typing import List
from typing import Tuple

import itertools
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset

from hactap import solver
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
        dataset, batch_size=PREDICT_BATCH_SIZE, shuffle=False
    )

    test_set_predict = []
    test_set_y = []
    for _, (sub_test_x, sub_test_y) in enumerate(test_set_loader):
        sub_test_predict = ai_worker.predict(sub_test_x)
        test_set_predict.extend(sub_test_predict)
        test_set_y.extend(sub_test_y.tolist())

    items = []
    for pred, y, idx in zip(test_set_predict, test_set_y, indexes):
        if pred is not None:
            items.append([pred, y, idx])

    tcs = itertools.groupby(
        sorted(items, key=key_of_task_cluster_k),
        key_of_task_cluster_k
    )

    return list(map(lambda x: (x[0], list(x[1])), tcs))


class CTA(solver.Solver):
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
        n_of_majority_vote: int = 1
    ) -> None:
        super().__init__(
            tasks,
            human_crowd,
            human_crowd_batch_size,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            reporter,
            n_of_majority_vote=n_of_majority_vote
        )
        self.significance_level = significance_level
        self.retire_used_test_data = retire_used_test_data

    def run(self) -> Tasks:
        self.initialize()
        self.report_log()

        self.assign_to_human_workers()
        self.report_log()

        while not self.check_n_of_class():
            self.assign_to_human_workers()
            self.report_log()

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters()
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

    def list_task_clusters(self) -> List[TaskCluster]:
        task_clusters = []

        for index, _ in enumerate(self.ai_workers):

            task_clusters.extend(
                self.create_task_cluster_from_ai_worker(index)
            )

        return task_clusters

    def create_task_cluster_from_ai_worker(
        self,
        ai_worker_index: int
    ) -> List[TaskCluster]:
        task_clusters: List[TaskCluster] = []
        ai_worker = self.ai_workers[ai_worker_index]

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
                TaskCluster(ai_worker, ai_worker_index, rule)
            )

        return task_clusters

    def assign_tasks_to_task_cluster(self, tack_cluster: TaskCluster) -> None:
        self.tasks.bulk_update_labels_by_ai(
            tack_cluster.assignable_task_indexes,
            tack_cluster.y_pred
        )

        for ati, ypd in zip(
            tack_cluster.assignable_task_indexes,
            tack_cluster.y_pred
        ):
            self.reporter.log_task_assignment(
                tack_cluster.model.get_worker_name(),
                ati, ypd
            )

        if self.retire_used_test_data:
            self.tasks.retire_human_label(
                tack_cluster.assignable_task_idx_test
            )

        self.report_assignment((
            tack_cluster.model.get_worker_name(),
            tack_cluster.rule["rule"],
            'a={}, b={}'.format(
                tack_cluster.match_rate_with_human,
                tack_cluster.conflict_rate_with_human
            ),
            'assigned_task={}'.format(
                tack_cluster.n_answerable_tasks
            )

        ))
        self.report_log()

    def _evalate_task_cluster_by_bin_test(
        self,
        task_cluster_k: TaskCluster
    ) -> bool:
        if task_cluster_k.n_answerable_tasks == 0:
            return False

        p_value = stats.binom_test(
            task_cluster_k.match_rate_with_human,
            task_cluster_k.match_rate_with_human + task_cluster_k.conflict_rate_with_human, # NOQA
            p=self.accuracy_requirement,
            alternative='greater'
        )

        return p_value < self.significance_level

        # y_pred = torch.tensor(aiw.predict(dataset.x_test))

        # task_clusters = {}
        # candidates = []

        # for y_human_i, y_pred_i in zip(dataset.y_test, y_pred):
        #     # print(y_human_i, y_pred_i)
        #     if int(y_pred_i) not in task_clusters:
        #         task_clusters[int(y_pred_i)] = []
        #     task_clusters[int(y_pred_i)].append(int(y_human_i))

        # for cluster_i, items in task_clusters.items():
        #     most_common_label = collections.Counter(items).most_common(1)

        #     # クラスタに含まれるデータがある場合に、そのクラスタの評価が行える
        #     # このif本当に要る？？？
        #     if len(most_common_label) == 1:
        #         label_type, label_count = collections.Counter(
        #             items
        #         ).most_common(1)[0]
        #         p_value = stats.binom_test(
        #             label_count,
        #             n=len(items),
        #             p=self.accuracy_requirement,
        #             alternative='greater'
        #         )
        #         # print(collections.Counter(items), p_value)

        #         log = {
        #             'ai_worker': aiw,
        #             'ai_worker_id': worker_id,
        #             'accepted_rule': {
        #                 "from": cluster_i,
        #                 "to": label_type
        #             },
        #             'was_accepted': p_value < self.significance_level
        #         }

        #         candidates.append(log)

        # return candidates
