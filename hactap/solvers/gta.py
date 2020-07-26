import random
import torch
import collections
from itertools import compress

from hactap import solver
from hactap.task_cluster import TaskCluster

NUMBER_OF_MONTE_CARLO_TRIAL = 100_000


class GTA(solver.Solver):
    def __init__(
        self,
        tasks,
        ai_workers,
        accuracy_requirement,
        human_crowd_batch_size,
        significance_level
    ):
        super().__init__(tasks, ai_workers, accuracy_requirement)
        self.human_crowd_batch_size = human_crowd_batch_size
        self.significance_level = significance_level

    def run(self):
        human_task_cluster = TaskCluster(0, 0)
        accepted_task_clusters = [human_task_cluster]

        while self.tasks.is_not_completed:
            task_cluster_candidates = self.list_task_clusters()
            random.shuffle(task_cluster_candidates)

            for task_cluster_k in task_cluster_candidates:
                if len(self.tasks.x_remaining) == 0:
                    break

                task_cluster_k.update_status(self.tasks)
                accepted_task_clusters[0].update_status_human(self.tasks)

                accepted = self._evalate_task_cluster_by_beta_dist(
                    accepted_task_clusters,
                    task_cluster_k
                )

                if accepted:
                    accepted_task_clusters.append(task_cluster_k)

                    assignable_task_indexes, y_pred = self.calc_assignable_tasks(task_cluster_k)
                    self.tasks.assign_tasks_to_ai(assignable_task_indexes, y_pred)
                    self.report_assignment((
                        task_cluster_k.model,
                        task_cluster_k.rule,
                        len(y_pred)
                    ))
                    self.report_log()

            self.assign_to_human_workers()
            self.report_log()

        return self.logs, self.assignment_log



    def calc_assignable_tasks(self, task_cluster_k):
        accepted_rule = task_cluster_k.rule["rule"]

        assigned_idx = range(len(self.tasks.x_remaining))
        y_pred = torch.tensor(task_cluster_k.model.predict(self.tasks.x_remaining))
        mask = y_pred == accepted_rule['from']

        _assigned_idx = list(compress(assigned_idx, mask.numpy()))
        _y_pred = y_pred.masked_select(mask)
        print(_y_pred)
        _y_pred[_y_pred == accepted_rule['from']] = accepted_rule['to']
        _y_pred.type(torch.LongTensor)
        print(_y_pred)
        print('filter', len(_assigned_idx), len(_y_pred))

        return _assigned_idx, _y_pred


    def _evalate_al_worker_by_task_cluster(self, worker_id, aiw, dataset):
        y_pred = torch.tensor(aiw.predict(dataset.x_test))

        task_clusters = {}
        candidates = []

        for y_human_i, y_pred_i in zip(dataset.y_test, y_pred):
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

                log = {
                    'ai_worker': aiw,
                    'ai_worker_id': worker_id,
                    'accepted_rule': {
                        "from": cluster_i,
                        "to": label_type
                    },
                    'was_accepted': None
                }

                candidates.append(log)

        return candidates

    def _evalate_task_cluster_by_beta_dist(
        self,
        accepted_task_clusters,
        task_cluster_i
    ):

        if task_cluster_i.n_answerable_tasks == 0:
            return False

        # TODO: 最小タスク数を考慮する必要があるか確認
        # if task_cluster_i.n_answerable_tasks < 10 * trial_id:
        #     return False

        target_list = accepted_task_clusters + [task_cluster_i]

        count_success = 0.0

        for i in range(NUMBER_OF_MONTE_CARLO_TRIAL):
            numer = 0.0
            denom = 0.0
            for task_cluster in target_list:
                numer += (
                    task_cluster.bata_dist[i] * task_cluster.n_answerable_tasks
                )
                denom += task_cluster.n_answerable_tasks

            overall_accuracy = numer / denom

            # print(overall_accuracy, task_cluster.n_answerable_tasks)

            if overall_accuracy >= self.accuracy_requirement:
                count_success += 1.0

        p_value = 1.0 - (count_success / NUMBER_OF_MONTE_CARLO_TRIAL)
        # print(NUMBER_OF_MONTE_CARLO_TRIAL, count_success, p_value)
        # print(target_list)

        return p_value < self.significance_level
