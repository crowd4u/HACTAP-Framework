import torch
from hactap.logging import get_logger
from hactap.utils import report_metrics
from hactap.task_cluster import TaskCluster
from hactap.human_crowd import get_labels_from_humans_by_random
import collections
import numpy as np
from torch.utils.data import DataLoader

logger = get_logger()


class Solver():
    def __init__(self, tasks, ai_workers, accuracy_requirement, n_of_classes, reporter=None, human_crowd=None): # NOQA
        self.tasks = tasks
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement
        self.n_of_classes = n_of_classes

        self.logs = []
        self.assignment_log = []
        self.reporter = reporter

        if human_crowd:
            self.get_labels_from_humans = human_crowd
        else:
            self.get_labels_from_humans = get_labels_from_humans_by_random

    def run(self):
        pass

    def initialize(self):
        if self.reporter:
            self.reporter.initialize()

    def finalize(self):
        if self.reporter:
            self.reporter.finalize(self.assignment_log)

    def report_log(self):
        if self.reporter:
            self.reporter.log_metrics(report_metrics(self.tasks))

    def report_assignment(self, assignment_log):
        self.assignment_log.append(assignment_log)
        logger.debug('new assignment: %s', self.assignment_log[-1])

    def check_n_of_class(self):
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

    def assign_to_human_workers(self):
        if not self.tasks.is_completed:
            labels = self.get_labels_from_humans(
                self.tasks,
                self.human_crowd_batch_size
            )
            logger.debug('new assignment: huamn %s', len(labels))

    def list_task_clusters(self):
        task_clusters = []

        for index, _ in enumerate(self.ai_workers):

            task_clusters.extend(
                self.create_task_cluster_from_ai_worker(index)
            )

        return task_clusters

    def create_task_cluster_from_ai_worker(self, ai_worker_index):
        task_clusters = {}
        task_clusters_for_remaining_y = {}
        task_clusters_for_remaining_ids = {}
        candidates = []
        batch_size = 10000

        y_test = np.array([y for x, y in iter(self.tasks.test_set)])

        logger.debug('predict - test')
        y_pred = []

        test_data = DataLoader(
            self.tasks.test_set, batch_size=batch_size
        )

        for index, (pd_y_i, _) in enumerate(test_data):
            result = self.ai_workers[ai_worker_index].predict(pd_y_i)
            for _, result_i in enumerate(result):
                y_pred.append(result_i)

        for y_human_i, y_pred_i in zip(y_test, y_pred):
            # print(y_human_i, y_pred_i)
            if int(y_pred_i) not in task_clusters:
                task_clusters[int(y_pred_i)] = []

            task_clusters[int(y_pred_i)].append(int(y_human_i))

        _z_i = 0

        predict_data = DataLoader(
            self.tasks.X_assignable, batch_size=batch_size
        )

        assignable_indexes = self.tasks.assignable_indexes

        logger.debug('predict - remaining')
        print('size of x', len(self.tasks.X_assignable))
        print('size of assignable_indexes', len(assignable_indexes))

        for index, (pd_i, _) in enumerate(predict_data):
            print('_calc_assignable_tasks', index)
            y_pred = self.ai_workers[ai_worker_index].predict(pd_i)

            for yp in y_pred:
                yp = int(yp)
                if yp not in task_clusters_for_remaining_y:
                    task_clusters_for_remaining_y[yp] = []
                    task_clusters_for_remaining_ids[yp] = []

                task_clusters_for_remaining_y[yp].append(yp)
                task_clusters_for_remaining_ids[yp].append(
                    assignable_indexes[_z_i]
                )

                _z_i += 1

        for cluster_i, items in task_clusters.items():
            most_common_label = collections.Counter(items).most_common(1)

            # クラスタに含まれるデータがある場合に、そのクラスタの評価が行える
            # このif本当に要る？？？
            if len(most_common_label) == 1:
                label_type, label_count = collections.Counter(
                    items
                ).most_common(1)[0]

                # print('label_type', label_type)
                # print('label_count', label_count)

                if cluster_i in task_clusters_for_remaining_y:
                    stat_y_pred = task_clusters_for_remaining_y[cluster_i]
                else:
                    stat_y_pred = []

                if cluster_i in task_clusters_for_remaining_ids:
                    stat_answerable_tasks_ids = task_clusters_for_remaining_ids[cluster_i] # NOQA
                else:
                    stat_answerable_tasks_ids = []

                # stat_y_pred = task_clusters_for_remaining_y[clust
                # er_i] if cluster_
                # i in task_clusters_for_remaining_y else []
                # stat_answerable_tasks_ids = task_clusters_
                # for_remaining_ids[clu
                # ster_i] if cluster_i in task_clusters_for_r
                # emaining_ids else []

                log = {
                    "rule": {
                        "from": cluster_i,
                        "to": label_type
                    },
                    "stat": {
                        "y_pred": stat_y_pred,
                        "answerable_tasks_ids": stat_answerable_tasks_ids
                    }
                }

                candidates.append(
                    TaskCluster(self.ai_workers[ai_worker_index], log)
                )
        return candidates
