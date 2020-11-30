from scipy import stats
import random
from typing import List

from hactap import solver
from hactap.logging import get_logger
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import BaseAIWorker
from hactap.reporter import Reporter
from hactap.task_cluster import TaskCluster

logger = get_logger()


class CTA(solver.Solver):
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        n_of_classes: int,
        significance_level: float,
        reporter: Reporter,
        retire_used_test_data: bool = False
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

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters()
            random.shuffle(task_cluster_candidates)

            # assign tasks to accepted task clusters
            for task_cluster_k in task_cluster_candidates:
                if self.tasks.is_completed:
                    break

                task_cluster_k.update_status(self.tasks)

                accepted = self._evalate_task_cluster_by_bin_test(
                    task_cluster_k
                )

                if accepted:
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

            self.assign_to_human_workers()
            self.report_log()

        self.finalize()

        return self.tasks

    def _evalate_task_cluster_by_bin_test(
        self,
        task_cluster_k: TaskCluster
    ) -> bool:
        p_value = stats.binom_test(
            task_cluster_k.match_rate_with_human,
            task_cluster_k.match_rate_with_human + task_cluster_k.conflict_rate_with_human, # NOQA
            p=self.accuracy_requirement,
            alternative='greater'
        )

        return p_value < self.significance_level

        # y_pred = torch.tensor(aiw.predict(dataset.x_test))

        # task_clusters = {}
        # candidates = []

        # for y_human_i, y_pred_i in zip(dataset.y_test, y_pred):
        #     # print(y_human_i, y_pred_i)
        #     if int(y_pred_i) not in task_clusters:
        #         task_clusters[int(y_pred_i)] = []
        #     task_clusters[int(y_pred_i)].append(int(y_human_i))

        # for cluster_i, items in task_clusters.items():
        #     most_common_label = collections.Counter(items).most_common(1)

        #     # クラスタに含まれるデータがある場合に、そのクラスタの評価が行える
        #     # このif本当に要る？？？
        #     if len(most_common_label) == 1:
        #         label_type, label_count = collections.Counter(
        #             items
        #         ).most_common(1)[0]
        #         p_value = stats.binom_test(
        #             label_count,
        #             n=len(items),
        #             p=self.accuracy_requirement,
        #             alternative='greater'
        #         )
        #         # print(collections.Counter(items), p_value)

        #         log = {
        #             'ai_worker': aiw,
        #             'ai_worker_id': worker_id,
        #             'accepted_rule': {
        #                 "from": cluster_i,
        #                 "to": label_type
        #             },
        #             'was_accepted': p_value < self.significance_level
        #         }

        #         candidates.append(log)

        # return candidates
