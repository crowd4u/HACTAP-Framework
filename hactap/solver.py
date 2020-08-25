from hactap.logging import get_logger
from hactap.utils import report_metrics
from hactap.task_cluster import TaskCluster
from hactap.human_crowd import get_labels_from_humans
import collections
import numpy as np
from torch.utils.data import DataLoader

logger = get_logger()


class Solver():
    def __init__(self, tasks, ai_workers, accuracy_requirement, reporter=None): # NOQA
        self.tasks = tasks
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement

        self.logs = []
        self.assignment_log = []
        self.reporter = reporter

    def run(self):
        pass

    def initialize(self):
        self.reporter.initialize()

    def finalize(self):
        self.reporter.finalize(self.assignment_log)

    def report_log(self):
        self.reporter.log_metrics(report_metrics(self.tasks))

    def report_assignment(self, assignment_log):
        self.assignment_log.append(assignment_log)
        logger.debug('new assignment: %s', self.assignment_log[-1])

    def assign_to_human_workers(self):
        if not self.tasks.is_completed:
            labels = get_labels_from_humans(
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

                stat_y_pred = task_clusters_for_remaining_y[cluster_i] if cluster_i in task_clusters_for_remaining_y else []
                stat_answerable_tasks_ids = task_clusters_for_remaining_ids[cluster_i] if cluster_i in task_clusters_for_remaining_ids else []

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
