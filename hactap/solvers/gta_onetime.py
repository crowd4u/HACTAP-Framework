import random

from hactap.solvers import GTA
from hactap.task_cluster import TaskCluster


class GTAOneTime(GTA):
    def __init__(
        self,
        tasks,
        human_crowd,
        ai_workers,
        accuracy_requirement,
        n_of_classes,
        significance_level,
        reporter,
        n_monte_carlo_trial=100000,
        minimum_sample_size=0
    ):
        super().__init__(
            tasks,
            human_crowd,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            significance_level,
            reporter,
            n_monte_carlo_trial=100000,
            minimum_sample_size=0
        )

    def run(self):
        self.initialize()
        self.report_log()

        self.assign_to_human_workers()
        self.report_log()

        # print('self.check_n_of_class()', self.check_n_of_class())

        while not self.check_n_of_class():
            self.assign_to_human_workers()
            self.report_log()
            # print('self.check_n_of_class()', self.check_n_of_class())

        human_task_cluster = TaskCluster(0, 0)
        # remain_cluster = TaskCluster(0, 0)
        accepted_task_clusters = [human_task_cluster]

        assign_memo = {}

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters()
            random.shuffle(task_cluster_candidates)

            for task_cluster_k in task_cluster_candidates:
                if self.tasks.is_completed:
                    break

                task_cluster_k.update_status(self.tasks, n_monte_carlo_trial=self.n_monte_carlo_trial) # NOQA
                accepted_task_clusters[0].update_status_human(self.tasks, n_monte_carlo_trial=self.n_monte_carlo_trial) # NOQA
                # accepted_task_clusters[1].update_status_remain(
                #     self.tasks,
                #     task_cluster_k.n_answerable_tasks,
                #     self.accuracy_requirement
                # )

                accepted = self._evalate_task_cluster_by_beta_dist(
                    accepted_task_clusters,
                    task_cluster_k
                )

                memo_key = task_cluster_k.rule['rule']['to']

                if accepted and (memo_key not in assign_memo):

                    assign_memo[memo_key] = True

                    accepted_task_clusters.append(task_cluster_k)

                    self.tasks.bulk_update_labels_by_ai(
                        task_cluster_k.assignable_task_indexes,
                        task_cluster_k.y_pred
                    )
                    self.tasks.retire_human_label(
                        task_cluster_k.assignable_task_idx_test
                    )

                    self.report_assignment((
                        task_cluster_k.model.model.__class__.__name__, # NOQA
                        task_cluster_k.rule["rule"],
                        'a={}, b={}'.format(
                            task_cluster_k.match_rate_with_human,
                            task_cluster_k.conflict_rate_with_human
                        ),
                        'assigned_task={}'.format(
                            task_cluster_k.n_answerable_tasks
                        )

                    ))
                    self.report_log()

            self.assign_to_human_workers()
            self.report_log()

        self.finalize()
        return self.tasks
