from typing import Dict, List

import random
from sklearn.neural_network import MLPClassifier

from hactap.logging import get_logger
from hactap import solvers
from hactap.task_cluster import TaskCluster
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import AIWorker, BaseAIWorker
from hactap.reporter import EvalAIReporter, Reporter
from hactap.evaluate_ai_worker import BaseEvalClass


logger = get_logger()

PREDICT_BATCH_SIZE = 10_000


class GTALimit(solvers.GTA):
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
        EvaluateAIClass: BaseEvalClass = None,
        evaluate_ai_class_params: Dict = {},
        aiw_reporter: EvalAIReporter = None
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
            n_of_majority_vote=n_of_majority_vote,
            n_monte_carlo_trial=n_monte_carlo_trial,
            minimum_sample_size=minimum_sample_size,
            prior_distribution=prior_distribution
        )
        self.EvalAIClass: BaseEvalClass = EvaluateAIClass(
            self.ai_workers,
            **evaluate_ai_class_params
        ) if EvaluateAIClass is not None else None
        self.eval_reporter = aiw_reporter if aiw_reporter is not None else None
        self.aiw_assignment_log = [[] for _ in range(len(ai_workers))]

    def initialize(self) -> None:
        if self.reporter:
            self.reporter.initialize()
        if self.eval_reporter:
            self.eval_reporter.initialize()

    def finalize(self) -> None:
        if self.reporter:
            self.reporter.finalize(self.assignment_log)
        if self.eval_reporter:
            self.eval_reporter.finalize(self.aiw_assignment_log)

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

        human_task_cluster = TaskCluster(AIWorker(MLPClassifier()), -1, {})
        # remain_cluster = TaskCluster(0, 0)
        # accepted_task_clusters = [human_task_cluster, remain_cluster]
        accepted_task_clusters = [human_task_cluster]

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters()

            random.shuffle(task_cluster_candidates)

            n_accepted_cluster_before = len(accepted_task_clusters)
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

            self.assign_to_human_workers()
            self.report_log()
            n_accepted_cluster = len(accepted_task_clusters) - n_accepted_cluster_before
            if self.EvalAIClass is not None and self.eval_reporter is not None:
                self.eval_reporter.log_metrics(
                    self.EvalAIClass.report().update(
                        {
                            "n_evaluated_cluster": len(task_cluster_candidates),
                            "n_accepted_cluster": n_accepted_cluster
                        }
                    )
                )

        self.finalize()

        return self.tasks

    def list_task_clusters(self) -> List[TaskCluster]:
        task_clusters = []

        for index, _ in enumerate(self.ai_workers):
            clusters = self.create_task_cluster_from_ai_worker(index)
            if self.EvalAIClass:
                for tc in clusters:
                    tc.update_status(self.tasks, n_monte_carlo_trial=self.n_monte_carlo_trial)
                acceptable = self.EvalAIClass.eval_ai_worker(index, clusters)
                if acceptable:
                    logger.debug(
                        "AI Worker ({}) Accepted".format(
                            self.ai_workers[index].get_worker_name()
                        )
                    )
                    task_clusters.extend(clusters)
                else:
                    logger.debug(
                        "AI Worker ({}) Rejected".format(
                            self.ai_workers[index].get_worker_name()
                        )
                    )
            else:
                task_clusters.extend(clusters)

        if self.EvalAIClass:
            self.EvalAIClass.increment_n_iter()

        return task_clusters

    def assign_tasks_to_task_cluster(self, task_cluster: TaskCluster) -> None:
        self.tasks.bulk_update_labels_by_ai(
            task_cluster.assignable_task_indexes,
            task_cluster.y_pred
        )

        for ati, ypd in zip(
            task_cluster.assignable_task_indexes,
            task_cluster.y_pred
        ):
            self.reporter.log_task_assignment(
                task_cluster.model.get_worker_name(),
                ati, ypd
            )

        if self.retire_used_test_data:
            self.tasks.retire_human_label(
                task_cluster.assignable_task_idx_test
            )

        self.report_assignment((
            task_cluster.model.get_worker_name(),
            task_cluster.rule["rule"],
            'a={}, b={}'.format(
                task_cluster.match_rate_with_human,
                task_cluster.conflict_rate_with_human
            ),
            'assigned_task={}'.format(
                task_cluster.n_answerable_tasks
            )
        ))
        if self.EvalAIClass:
            self.aiw_assignment_log[task_cluster.aiw_id].append(
                {
                    "n_iter": self.EvalAIClass.n_iter,
                    "rule": task_cluster.rule["rule"],
                    "match_rate": task_cluster.match_rate_with_human,
                    "conflict_rate": task_cluster.conflict_rate_with_human,
                    "assigned_task": task_cluster.n_answerable_tasks
                }
            )

        self.report_log()
