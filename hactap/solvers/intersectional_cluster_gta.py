from typing import List

import random
from hactap import solvers
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import BaseAIWorker
from hactap.reporter import Reporter
from hactap.task_cluster import TaskCluster
from hactap.intersectional_model import IntersectionalModel


class IntersectionalClusterGTA(solvers.IntersectionalClusterCTA, solvers.GTA):
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        human_crowd_batch_size: int,
        ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        n_of_classes: int,
        significance_level: float,
        reporter: Reporter,
        clustering_function: IntersectionalModel,
        retire_used_test_data: bool = True,
        n_monte_carlo_trial: int = 100000,
        minimum_sample_size: int = -1,
        prior_distribution: List[int] = [1, 1],
        n_of_majority_vote: int = 1,
        report_all_task_clusters: bool = False
    ) -> None:
        super().__init__(
            tasks,
            human_crowd,
            human_crowd_batch_size,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            significance_level,
            reporter,
            clustering_function,
            retire_used_test_data,
            n_of_majority_vote,
            report_all_task_clusters
        )
        self.n_monte_carlo_trial = n_monte_carlo_trial
        self.minimum_sample_size = minimum_sample_size
        self.prior_distribution = prior_distribution

    def run(self) -> Tasks:
        self.initialize()
        self.report_log()

        self.assign_to_human_workers()
        self.report_log()

        # print('self.check_n_of_class()', self.check_n_of_class())

        while not self.check_n_of_class():
            self.assign_to_human_workers()
            self.report_log()
            # print('self.check_n_of_class()', self.check_n_of_class())

        human_task_cluster = TaskCluster(None, -1, {})
        # remain_cluster = TaskCluster(0, 0)
        # accepted_task_clusters = [human_task_cluster, remain_cluster]
        accepted_task_clusters = [human_task_cluster]

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters_by_any()
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
                    self.assign_tasks_to_task_cluster(task_cluster_k)

            self.assign_to_human_workers()
            self.report_log()

        self.finalize()

        return self.tasks
