from hactap.logging import get_logger
from hactap.utils import report_metrics
from hactap.task_cluster import TaskCluster
import torch
import collections

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
        if len(self.tasks.x_remaining) != 0:
            if len(self.tasks.x_remaining) < self.human_crowd_batch_size:
                n_instances = len(self.tasks.x_remaining)
            else:
                n_instances = self.human_crowd_batch_size
            query_idx, _ = self.ai_workers[0].query(
                self.tasks.x_remaining,
                n_instances=n_instances
            )
            self.tasks.assign_tasks_to_human(query_idx)

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

        y_pred = torch.tensor(self.ai_workers[ai_worker_index].predict(self.tasks.x_test))

        for y_human_i, y_pred_i in zip(self.tasks.y_test, y_pred):
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

                print('label_type', label_type)
                print('label_count', label_count)

                log = {
                    "rule": {
                        "from": cluster_i,
                        "to": label_type
                    },
                }

                candidates.append(TaskCluster(self.ai_workers[ai_worker_index], log))
        return candidates

