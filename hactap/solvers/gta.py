import random

from hactap.logging import get_logger
from hactap import solver
from hactap.task_cluster import TaskCluster

logger = get_logger()


class GTA(solver.Solver):
    def __init__(
        self,
        tasks,
        ai_workers,
        accuracy_requirement,
        n_of_classes,
        human_crowd_batch_size,
        significance_level,
        reporter,
        human_crowd,
        retire_used_test_data=True,
        n_monte_carlo_trial=100000,
        minimum_sample_size=-1,
        prior_distribution=[1, 1]
    ):
        super().__init__(
            tasks, ai_workers, accuracy_requirement, n_of_classes, reporter,
            human_crowd
        )
        self.human_crowd_batch_size = human_crowd_batch_size
        self.significance_level = significance_level
        self.retire_used_test_data = retire_used_test_data
        self.n_monte_carlo_trial = n_monte_carlo_trial
        self.minimum_sample_size = minimum_sample_size
        self.prior_distribution = prior_distribution

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
        # accepted_task_clusters = [human_task_cluster, remain_cluster]
        accepted_task_clusters = [human_task_cluster]

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
                    self.accuracy_requirement,
                    accepted_task_clusters,
                    task_cluster_k
                )

                if accepted:
                    accepted_task_clusters.append(task_cluster_k)

                    self.tasks.bulk_update_labels_by_ai(
                        task_cluster_k.assignable_task_indexes,
                        task_cluster_k.y_pred
                    )

                    if self.retire_used_test_data:
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

    def _evalate_task_cluster_by_beta_dist(
        self,
        accuracy_requirement,
        accepted_task_clusters,
        task_cluster_i
    ):
        logger.debug('evalate_task_cluster_by_beta_dist')

        if task_cluster_i.n_answerable_tasks == 0:
            logger.debug("  rejected by minimum_sample_size")
            return False

        if self.minimum_sample_size == -1:
            n_of_human_labels = task_cluster_i.match_rate_with_human + task_cluster_i.conflict_rate_with_human # NOQA
            cond_a = n_of_human_labels * accuracy_requirement >= 5
            cond_b = n_of_human_labels * (1 - accuracy_requirement) >= 5
            if not (cond_a and cond_b):
                logger.debug("  rejected by minimum_sample_size")
                return False
        else:
            if (task_cluster_i.match_rate_with_human + task_cluster_i.conflict_rate_with_human) <= self.minimum_sample_size:  # NOQA
                logger.debug("  rejected by minimum_sample_size")
                return False

        target_list = accepted_task_clusters + [task_cluster_i]
        logger.debug("n_of_tcs: {}".format(len(target_list)))

        count_success = 0.0
        # overall_accuracies = []

        denom = 0.0
        for task_cluster in target_list:
            denom += task_cluster.n_answerable_tasks

        for i in range(self.n_monte_carlo_trial):
            numer = 0.0
            for task_cluster in target_list:
                numer += (
                    task_cluster.bata_dist[i] * task_cluster.n_answerable_tasks
                )
            overall_accuracy = numer / denom
            # overall_accuracies.append(overall_accuracy)

            if overall_accuracy >= self.accuracy_requirement:
                count_success += 1.0

        # overall_accuracies = np.asarray(overall_accuracies)

        p_value = 1.0 - (count_success / self.n_monte_carlo_trial)
        is_accepted = p_value < self.significance_level
        logger.debug("count_success: {}".format(count_success))
        # logger.debug("ave: {}".format(np.average(overall_accuracies)))
        # logger.debug("var: {}".format(np.var(overall_accuracies)))
        logger.debug("p_value: {}".format(p_value))
        logger.debug("->is_accepted: {}".format(is_accepted))
        print("denom", denom)

        return is_accepted
