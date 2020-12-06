from typing import List

import random
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from hactap.logging import get_logger
from hactap import solver
from hactap.task_cluster import TaskCluster
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import AIWorker, BaseAIWorker
from hactap.reporter import Reporter

logger = get_logger()


class GTA_AA(solver.Solver):
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        n_of_classes: int,
        significance_level: float,
        reporter: Reporter,
        retire_used_test_data: bool = True,
        n_monte_carlo_trial: int = 100000,
        minimum_sample_size: int = -1,
        prior_distribution: List[int] = [1, 1],
        n_of_majority_vote: int = 1,
        additional_assignment_strategy: str = 'all'
    ) -> None:
        super().__init__(
            tasks,
            human_crowd,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            reporter,
        )
        self.significance_level = significance_level
        self.retire_used_test_data = retire_used_test_data
        self.n_monte_carlo_trial = n_monte_carlo_trial
        self.minimum_sample_size = minimum_sample_size
        self.prior_distribution = prior_distribution
        self.n_of_majority_vote = n_of_majority_vote
        self.additional_assignment_strategy = additional_assignment_strategy

    def run(self) -> Tasks:
        self.initialize()
        self.report_log()

        assigned_indexes = self.assign_to_human_workers()
        if self.additional_assignment_strategy == 'all':
            for n in range(self.n_of_majority_vote - 1):
                print('do majority_vote')
                self.assign_to_human_workers(assigned_indexes)
        self.report_log()

        # print('self.check_n_of_class()', self.check_n_of_class())

        while not self.check_n_of_class():
            assigned_indexes = self.assign_to_human_workers()
            if self.additional_assignment_strategy == 'all':
                for n in range(self.n_of_majority_vote - 1):
                    print('do majority_vote')
                    self.assign_to_human_workers(assigned_indexes)
            self.report_log()
            # print('self.check_n_of_class()', self.check_n_of_class())

        human_task_cluster = TaskCluster(AIWorker(MLPClassifier()), {})
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

                print("confusion matrix:")
                cm_ai = []
                cm_human = []
                cm_ti = []
                test_y_predict = task_cluster_k.test_y_predict
                test_y_human = task_cluster_k.test_y_human
                assignable_task_idx_test = task_cluster_k.assignable_task_idx_test # NOQA
                print(assignable_task_idx_test)

                for _p, _h, _ti in zip(
                    test_y_predict,
                    test_y_human,
                    assignable_task_idx_test
                ):
                    if int(_p) == int(_h):
                        cm_ai.append(1)
                    else:
                        cm_ai.append(0)
                        cm_ti.append(_ti)

                    cm_human.append(1)

                # TODO: fpのタスクを再割り当てするのを試す
                print(confusion_matrix(test_y_human, test_y_predict))
                print(confusion_matrix(cm_human, cm_ai).ravel())
                print(len(cm_ti), cm_ti)

                if self.additional_assignment_strategy == 'conflict':
                    if random.random() > 0.5:
                        for n in range(self.n_of_majority_vote - 1):
                            self.assign_to_human_workers(cm_ti)
                        accepted = False

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
                        task_cluster_k.model.get_worker_name(),
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

            assigned_indexes = self.assign_to_human_workers()
            if self.additional_assignment_strategy == 'all':
                for n in range(self.n_of_majority_vote - 1):
                    print('do majority_vote')
                    self.assign_to_human_workers(assigned_indexes)
            self.report_log()

        self.finalize()

        return self.tasks

    def _evalate_task_cluster_by_beta_dist(
        self,
        accuracy_requirement: float,
        accepted_task_clusters: List[TaskCluster],
        task_cluster_i: TaskCluster
    ) -> bool:
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
