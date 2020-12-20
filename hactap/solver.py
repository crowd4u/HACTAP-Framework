from typing import List
from typing import Union
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from hactap.logging import get_logger
from hactap.utils import report_metrics
from hactap.tasks import Tasks
from hactap.ai_worker import BaseAIWorker
from hactap.human_crowd import IdealHumanCrowd
from hactap.reporter import Reporter


logger = get_logger()


class Solver():
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        ai_workers: List[Union[BaseAIWorker]],
        accuracy_requirement: float,
        n_of_classes: int,
        reporter: Reporter = None,
        n_of_majority_vote: int = 1
    ) -> None:
        self.tasks = tasks
        self.human_crowd = human_crowd
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement
        self.n_of_classes = n_of_classes
        self.n_of_majority_vote = n_of_majority_vote

        self.logs: List[dict] = []
        self.assignment_log: List[Tuple] = []
        self.reporter = reporter

    def run(self) -> Tasks:
        pass

    def initialize(self) -> None:
        if self.reporter:
            self.reporter.initialize()

    def finalize(self) -> None:
        if self.reporter:
            self.reporter.finalize(self.assignment_log)

    def report_log(self) -> None:
        if self.reporter:
            self.reporter.log_metrics(report_metrics(self.tasks))

    def report_assignment(
        self,
        assignment_log: Tuple
    ) -> None:
        self.assignment_log.append(assignment_log)
        logger.debug('new assignment: %s', self.assignment_log[-1])

    def check_n_of_class(self) -> bool:
        n_of_classes = self.n_of_classes
        train_set = self.tasks.train_set
        test_set = self.tasks.test_set

        train_data = DataLoader(
            train_set, batch_size=len(train_set)
        )

        test_data = DataLoader(
            test_set, batch_size=len(test_set)
        )

        _, y_train = next(iter(train_data))
        _, y_test = next(iter(test_data))
        cond_a = len(torch.unique(y_train)) == n_of_classes
        cond_b = len(torch.unique(y_test)) == n_of_classes
        return cond_a and cond_b

    def assign_to_human_workers(
        self,
        target_indexes: List[int] = [],
        n_of_majority_vote: Union[int, None] = None
    ) -> List[int]:
        if n_of_majority_vote is None:
            n_of_majority_vote = self.n_of_majority_vote

        if not self.tasks.is_completed:
            assigned_indexes = self.human_crowd.assign(
                self.tasks,
                target_indexes
            )
            logger.debug('new assignment: huamn %s', len(assigned_indexes))

            for n in range(n_of_majority_vote - 1):
                self.human_crowd.assign(self.tasks, assigned_indexes)
                logger.debug('majority_vote: huamn %s', len(assigned_indexes))

            return assigned_indexes
        else:
            return []

    # def create_task_cluster_from_ai_worker(
    #     self,
    #     ai_worker_index: int
    # ) -> List[TaskCluster]:
    #     task_clusters: dict = {}
    #     task_clusters_for_remaining_y: dict = {}
    #     task_clusters_for_remaining_ids: dict = {}
    #     candidates = []
    #     batch_size = 10000

    #     # y_test = np.array([y for x, y in iter(self.tasks.test_set)])
    #     test_set = self.tasks.test_set
    #     test_set_loader = torch.utils.data.DataLoader(
    #         test_set, batch_size=len(test_set)
    #     )
    #     _, y_test = next(iter(test_set_loader))

    #     logger.debug('predict - test')
    #     y_pred = []

    #     test_data = DataLoader(
    #         self.tasks.test_set, batch_size=batch_size
    #     )

    #     for index, (pd_y_i, _) in enumerate(test_data):
    #         result = self.ai_workers[ai_worker_index].predict(pd_y_i)
    #         for _, result_i in enumerate(result):
    #             y_pred.append(result_i)

    #     for y_human_i, y_pred_i in zip(y_test, y_pred):
    #         # print(y_human_i, y_pred_i)
    #         if int(y_pred_i) not in task_clusters:
    #             task_clusters[int(y_pred_i)] = []

    #         task_clusters[int(y_pred_i)].append(int(y_human_i))

    #     _z_i = 0

    #     predict_data = DataLoader(
    #         self.tasks.X_assignable, batch_size=batch_size
    #     )

    #     assignable_indexes = self.tasks.assignable_indexes

    #     logger.debug('predict - remaining')
    #     # print('size of x', len(self.tasks.X_assignable))
    #     # print('size of assignable_indexes', len(assignable_indexes))

    #     for index, (pd_i, _) in enumerate(predict_data):
    #         print('_calc_assignable_tasks', index)
    #         y_pred = self.ai_workers[ai_worker_index].predict(pd_i)

    #         for yp in y_pred:
    #             yp = int(yp)
    #             if yp not in task_clusters_for_remaining_y:
    #                 task_clusters_for_remaining_y[yp] = []
    #                 task_clusters_for_remaining_ids[yp] = []

    #             task_clusters_for_remaining_y[yp].append(yp)
    #             task_clusters_for_remaining_ids[yp].append(
    #                 assignable_indexes[_z_i]
    #             )

    #             _z_i += 1

    #     for cluster_i, items in task_clusters.items():
    #         most_common_label: List[Tuple[Any, int]] = collections.Counter(
    #             items
    #         ).most_common(1)

    #         # クラスタに含まれるデータがある場合に、そのクラスタの評価が行える
    #         # このif本当に要る？？？
    #         if len(most_common_label) == 1:
    #             # label_type, label_count = collections.Counter(
    #             #     items
    #             # ).most_common(1)[0]

    #             label_type = most_common_label[0][0]
    #             # label_count = most_common_label[0][1]

    #             # print('label_type', label_type)
    #             # print('label_count', label_count)

    #             if cluster_i in task_clusters_for_remaining_y:
    #                 stat_y_pred = task_clusters_for_remaining_y[cluster_i]
    #             else:
    #                 stat_y_pred = []

    #             if cluster_i in task_clusters_for_remaining_ids:
    #                 stat_answerable_tasks_ids = task_clusters_for_remaining_ids[cluster_i] # NOQA
    #             else:
    #                 stat_answerable_tasks_ids = []

    #             # stat_y_pred = task_clusters_for_remaining_y[clust
    #             # er_i] if cluster_
    #             # i in task_clusters_for_remaining_y else []
    #             # stat_answerable_tasks_ids = task_clusters_
    #             # for_remaining_ids[clu
    #             # ster_i] if cluster_i in task_clusters_for_r
    #             # emaining_ids else []

    #             log = {
    #                 "rule": {
    #                     "from": cluster_i,
    #                     "to": label_type
    #                 },
    #                 "stat": {
    #                     "y_pred": stat_y_pred,
    #                     "answerable_tasks_ids": stat_answerable_tasks_ids,
    #                     "y_pred_test": [],
    #                     "y_pred_train": [],
    #                     "y_pred_test_ids": [],
    #                     "y_pred_train_ids": [],
    #                 }
    #             }

    #             candidates.append(
    #                 TaskCluster(self.ai_workers[ai_worker_index], log)
    #             )
    #     return candidates
