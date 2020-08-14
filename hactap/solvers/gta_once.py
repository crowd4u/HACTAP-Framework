import random
import torch
import collections
from itertools import compress

from hactap.solvers import GTA
from hactap.task_cluster import TaskCluster

NUMBER_OF_MONTE_CARLO_TRIAL = 100_000


class GTAOnce(GTA):
    def __init__(
        self,
        tasks,
        ai_workers,
        accuracy_requirement,
        human_crowd_batch_size,
        significance_level
    ):
        super().__init__(tasks, ai_workers, accuracy_requirement, human_crowd_batch_size, significance_level)

    def run(self):
        self.report_log()

        human_task_cluster = TaskCluster(0, 0)
        accepted_task_clusters = [human_task_cluster]

        assign_memo = {}

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

                    assignable_task_indexes, y_pred = task_cluster_k._calc_assignable_tasks(self.tasks.x_remaining)

                    memo_key = task_cluster_k.rule['rule']['to']

                    if memo_key not in assign_memo:

                        accepted_task_clusters.append(task_cluster_k)
                        assign_memo[memo_key] = True

                        self.tasks.assign_tasks_to_ai(assignable_task_indexes, y_pred)

                        self.report_assignment((
                            task_cluster_k.model.model.estimator.__class__.__name__,
                            task_cluster_k.rule,
                            len(y_pred)
                        ))
                        self.report_log()

            self.assign_to_human_workers()
            self.report_log()

        return self.logs, self.assignment_log
