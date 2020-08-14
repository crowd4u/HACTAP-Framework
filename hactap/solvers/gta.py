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
        self.report_log()

        human_task_cluster = TaskCluster(0, 0)
        remain_cluster = TaskCluster(0, 0)
        accepted_task_clusters = [human_task_cluster, remain_cluster]

        while not self.tasks.is_completed:

            for w_i, ai_worker in enumerate(self.ai_workers):
                X_train, y_train = self.tasks.train_set
                ai_worker.fit(X_train, y_train)

            task_cluster_candidates = self.list_task_clusters()
            random.shuffle(task_cluster_candidates)

            for task_cluster_k in task_cluster_candidates:
                if self.tasks.is_completed:
                    break

                task_cluster_k.update_status(self.tasks)
                assignable_task_indexes, y_pred = task_cluster_k._calc_assignable_tasks(self.tasks.X_assignable, self.tasks.assignable_indexes)

                accepted_task_clusters[0].update_status_human(self.tasks)
                accepted_task_clusters[1].update_status_remain(self.tasks, assignable_task_indexes)

                accepted = self._evalate_task_cluster_by_beta_dist(
                    accepted_task_clusters,
                    task_cluster_k
                )

                if accepted:
                    accepted_task_clusters.append(task_cluster_k)

                    print("hogeee", len(self.tasks.assignable_indexes), len(assignable_task_indexes))

                    self.tasks.bulk_update_labels_by_ai(assignable_task_indexes, y_pred)
                    # TODO: 復活する
                    # self.tasks.lock_human_test(task_cluster_k.assignable_task_idx_test)

                    self.report_assignment((
                        task_cluster_k.model.model.estimator.__class__.__name__,
                        task_cluster_k.rule,
                        'a={}, b={}'.format(task_cluster_k.match_rate_with_human, task_cluster_k.conflict_rate_with_human),
                        'assigned_task={}'.format(len(y_pred))

                    ))
                    self.report_log()

            self.assign_to_human_workers()
            self.report_log()

        return self.logs, self.assignment_log


    def _evalate_task_cluster_by_beta_dist(
        self,
        accepted_task_clusters,
        task_cluster_i
    ):
        if task_cluster_i.n_answerable_tasks == 0:
            return False

        # タスククラスタのサイズで評価するかどうかを決める
        # if task_cluster_i.n_answerable_tasks < 10:
        #     return False

        # if task_cluster_i.n_answerable_tasks * (1 - self.accuracy_requirement) < 5:
        #     return False

        target_list = accepted_task_clusters + [task_cluster_i]

        count_success = 0.0

        overall_accuracies = []

        for i in range(NUMBER_OF_MONTE_CARLO_TRIAL):
            numer = 0.0
            denom = 0.0
            for task_cluster in target_list:
                numer += (
                    task_cluster.bata_dist[i] * task_cluster.n_answerable_tasks
                )
                denom += task_cluster.n_answerable_tasks

            overall_accuracy = numer / denom
            overall_accuracies.append(overall_accuracy)

            if overall_accuracy >= self.accuracy_requirement:
                count_success += 1.0

        p_value = 1.0 - (count_success / NUMBER_OF_MONTE_CARLO_TRIAL)
        # print(NUMBER_OF_MONTE_CARLO_TRIAL, count_success, p_value)
        # print(target_list)

        # print("denom", denom)
        # print("===== {} ===== {}".format(task_cluster_i.n_answerable_tasks, p_value))
        print("overall_accuracies", random.sample(overall_accuracies, 3))
        print("p_value", p_value, "1-p", (count_success / NUMBER_OF_MONTE_CARLO_TRIAL), p_value < self.significance_level)

        return p_value < self.significance_level
