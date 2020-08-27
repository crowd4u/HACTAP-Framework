import random

from hactap.solvers import GTA
from hactap.task_cluster import TaskCluster

NUMBER_OF_MONTE_CARLO_TRIAL = 100_000


class GTAOneTime(GTA):
    def __init__(
        self,
        tasks,
        ai_workers,
        accuracy_requirement,
        human_crowd_batch_size,
        significance_level,
        reporter
    ):
        super().__init__(
            tasks,
            ai_workers,
            accuracy_requirement,
            human_crowd_batch_size,
            significance_level,
            reporter
        )

    def run(self):
        self.initialize()
        self.report_log()

        human_task_cluster = TaskCluster(0, 0)
        remain_cluster = TaskCluster(0, 0)
        accepted_task_clusters = [human_task_cluster, remain_cluster]

        assign_memo = {}

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
                assignable_task_indexes, y_pred = task_cluster_k._calc_assignable_tasks( # NOQA
                    self.tasks.X_assignable, self.tasks.assignable_indexes
                )

                accepted_task_clusters[0].update_status_human(self.tasks)
                accepted_task_clusters[1].update_status_remain(
                    self.tasks,
                    assignable_task_indexes,
                    self.accuracy_requirement
                )

                accepted = self._evalate_task_cluster_by_beta_dist(
                    accepted_task_clusters,
                    task_cluster_k
                )

                memo_key = task_cluster_k.rule['rule']['to']

                if accepted and (memo_key not in assign_memo):

                    assign_memo[memo_key] = True

                    accepted_task_clusters.append(task_cluster_k)

                    self.tasks.bulk_update_labels_by_ai(
                        assignable_task_indexes, y_pred
                    )
                    self.tasks.retire_human_label(
                        task_cluster_k.assignable_task_idx_test
                    )

                    self.report_assignment((
                        task_cluster_k.model.model.estimator.__class__.__name__, # NOQA
                        task_cluster_k.rule,
                        'a={}, b={}'.format(
                            task_cluster_k.match_rate_with_human,
                            task_cluster_k.conflict_rate_with_human
                        ),
                        'assigned_task={}'.format(len(y_pred))

                    ))
                    self.report_log()

            self.assign_to_human_workers()
            self.report_log()

        self.finalize()
        print(assign_memo)

        return self.logs, self.assignment_log

        # while not self.tasks.is_completed:
        #     task_cluster_candidates = self.list_task_clusters()
        #     random.shuffle(task_cluster_candidates)

        #     for task_cluster_k in task_cluster_candidates:
        #         if len(self.tasks.x_remaining) == 0:
        #             break

        #         task_cluster_k.update_status(self.tasks)
        #         accepted_task_clusters[0].update_status_human(self.tasks)

        #         accepted = self._evalate_task_cluster_by_beta_dist(
        #             accepted_task_clusters,
        #             task_cluster_k
        #         )

        #         if accepted:

        #             assignable_task_indexes, y_pred =
        #  task_cluster_k._calc_assignable_tasks(self.tasks.x_remaining)

        #             memo_key = task_cluster_k.rule['rule']['to']

        #             if memo_key not in assign_memo:

        #                 accepted_task_clusters.append(task_cluster_k)
        #                 assign_memo[memo_key] = True

        #                 self.tasks.assign_tasks_to_ai(ass
        # ignable_task_indexes, y_pred)

        #                 self.report_assignment((
        #                     task_cluster_k.model.model.estimator.__class__.__name__,
        #                     task_cluster_k.rule,
        #                     len(y_pred)
        #                 ))
        #                 self.report_log()

        #     self.assign_to_human_workers()
        #     self.report_log()

        # return self.logs, self.assignment_log
