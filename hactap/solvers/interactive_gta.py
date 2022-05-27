from typing import List

import random
from sklearn.neural_network import MLPClassifier

from hactap.logging import get_logger
from hactap import solvers
from hactap.task_cluster import TaskCluster
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import AIWorker, BaseAIWorker
from hactap.reporter import Reporter

logger = get_logger()


class InteractiveGTA(solvers.GTA, solvers.InteractiveCTA):
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
        retire_used_test_data: bool = True,
        n_monte_carlo_trial: int = 100000,
        minimum_sample_size: int = -1,
        prior_distribution: List[int] = [1, 1],
        n_of_majority_vote: int = 1,
        interaction_strategy: str = 'conflict'
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
            retire_used_test_data=retire_used_test_data,
            n_monte_carlo_trial=n_monte_carlo_trial,
            minimum_sample_size=minimum_sample_size,
            prior_distribution=prior_distribution,
            n_of_majority_vote=n_of_majority_vote
        )
        self.interaction_strategy = interaction_strategy

    def run(self) -> Tasks:
        self.initialize()
        self.report_log()

        self.assign_to_human_workers(
            n_of_majority_vote=1
        )
        self.report_log()

        while not self.check_n_of_class():
            self.assign_to_human_workers(
                n_of_majority_vote=1
            )
            self.report_log()

        human_task_cluster = TaskCluster(AIWorker(MLPClassifier()), -1, {})
        accepted_task_clusters = [human_task_cluster]

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters()
            random.shuffle(task_cluster_candidates)

            if random.random() > 0.5:
                additional_assiguments = self.create_additional_task_assignment(task_cluster_candidates, self.interaction_strategy) # NOQA

                self.assign_to_human_workers(
                    additional_assiguments
                )
                self.report_log()

            else:
                for task_cluster_k in task_cluster_candidates:
                    if self.tasks.is_completed:
                        break

                    task_cluster_k.update_status(self.tasks, n_monte_carlo_trial=self.n_monte_carlo_trial) # NOQA
                    accepted_task_clusters[0].update_status_human(self.tasks, n_monte_carlo_trial=self.n_monte_carlo_trial) # NOQA

                    accepted = self._evalate_task_cluster_by_beta_dist(
                        self.accuracy_requirement,
                        accepted_task_clusters,
                        task_cluster_k
                    )

                    if accepted:
                        accepted_task_clusters.append(task_cluster_k)
                        self.assign_tasks_to_task_cluster(task_cluster_k)

                self.assign_to_human_workers(
                    n_of_majority_vote=1
                )
                self.report_log()

        self.finalize()

        return self.tasks
