from hactap.logging import get_logger
from hactap.utils import report_metrics
from hactap.task_cluster import TaskCluster
import torch
import collections
import numpy as np

class Solver():
    def __init__(self, tasks, ai_workers, accuracy_requirement):
        self.tasks = tasks
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement

        self.logs = []
        self.assignment_log = []
        self.logger = get_logger()

    def run(self):
        pass

    def report_log(self):
        self.logs.append(report_metrics(self.tasks))
        self.logger.debug('log: %s', self.logs[-1])

    def report_assignment(self, assignment_log):
        self.assignment_log.append(assignment_log)
        self.logger.debug('new assignment: %s', self.assignment_log[-1])

    def assign_to_human_workers(self):
        if not self.tasks.is_completed:
            if len(self.tasks.assignable_indexes) < self.human_crowd_batch_size:
                n_instances = len(self.tasks.assignable_indexes)
            else:
                n_instances = self.human_crowd_batch_size

            query_idx = np.random.choice(
                self.tasks.assignable_indexes,
                size=n_instances,
                replace=False
            )

            initial_labels = self.tasks.get_ground_truth(query_idx)

            self.tasks.bulk_update_labels_by_human(query_idx, initial_labels)

    def list_task_clusters(self):
        task_clusters = []

        for index, _ in enumerate(self.ai_workers):

            task_clusters.extend(
                self.create_task_cluster_from_ai_worker(index)
            )

        return task_clusters

    def create_task_cluster_from_ai_worker(self, ai_worker_index):
        task_clusters = {}
        candidates = []

        X_test, y_test = self.tasks.test_set

        y_pred = torch.tensor(self.ai_workers[ai_worker_index].predict(X_test))

        for y_human_i, y_pred_i in zip(y_test, y_pred):
            # print(y_human_i, y_pred_i)
            if int(y_pred_i) not in task_clusters:
                task_clusters[int(y_pred_i)] = []

            task_clusters[int(y_pred_i)].append(int(y_human_i))

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

                log = {
                    "rule": {
                        "from": cluster_i,
                        "to": label_type
                    },
                }

                candidates.append(TaskCluster(self.ai_workers[ai_worker_index], log))
        return candidates

    # def calc_assignable_tasks(self, task_cluster_k):
    #     accepted_rule = task_cluster_k.rule["rule"]

    #     assigned_idx = range(len(self.tasks.x_remaining))
    #     y_pred = torch.tensor(task_cluster_k.model.predict(self.tasks.x_remaining))
    #     mask = y_pred == accepted_rule['from']

    #     _assigned_idx = list(compress(assigned_idx, mask.numpy()))
    #     _y_pred = y_pred.masked_select(mask)
    #     print(_y_pred)
    #     _y_pred[_y_pred == accepted_rule['from']] = accepted_rule['to']
    #     _y_pred.type(torch.LongTensor)
    #     print(_y_pred)
    #     print('filter', len(_assigned_idx), len(_y_pred))

    #     return _assigned_idx, _y_pred