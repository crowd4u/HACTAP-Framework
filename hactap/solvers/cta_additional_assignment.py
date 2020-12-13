from typing import List

import random
from scipy import stats
from sklearn.metrics import confusion_matrix
from collections import Counter

from hactap import solver
from hactap.logging import get_logger
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import BaseAIWorker
from hactap.reporter import Reporter
from hactap.task_cluster import TaskCluster

logger = get_logger()


class CTA_AA(solver.Solver):
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        n_of_classes: int,
        significance_level: float,
        reporter: Reporter,
        retire_used_test_data: bool = False,
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

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters()
            random.shuffle(task_cluster_candidates)

            if self.additional_assignment_strategy == 'conflict' and random.random() > 0.5: # NOQA
                global_cm_ti_train = []
                global_cm_ti_test = []

                for task_cluster_k in task_cluster_candidates:
                    task_cluster_k.update_status(self.tasks)

                    print("confusion matrix:")
                    cm_ai = []
                    cm_human = []
                    cm_ti_train = []
                    cm_ti_test = []
                    test_y_predict = task_cluster_k.test_y_predict
                    test_y_human = task_cluster_k.test_y_human
                    assignable_task_idx_test = task_cluster_k.assignable_task_idx_test # NOQA

                    train_y_predict = task_cluster_k.train_y_predict
                    train_y_human = task_cluster_k.train_y_human
                    assignable_task_idx_train = task_cluster_k.assignable_task_idx_train # NOQA
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
                            cm_ti_test.append(_ti)

                        cm_human.append(1)

                    for _p, _h, _ti in zip(
                        train_y_predict,
                        train_y_human,
                        assignable_task_idx_train
                    ):
                        if int(_p) == int(_h):
                            cm_ai.append(1)
                        else:
                            cm_ai.append(0)
                            cm_ti_train.append(_ti)

                        cm_human.append(1)

                    # TODO: fpのタスクを再割り当てするのを試す
                    print(confusion_matrix(test_y_human, test_y_predict))
                    print(confusion_matrix(cm_human, cm_ai).ravel())
                    print(len(cm_ti_test), cm_ti_test)

                    global_cm_ti_test.extend(cm_ti_test)
                    global_cm_ti_train.extend(cm_ti_train)

                counts_global_cm_ti_test = Counter(global_cm_ti_test)
                new_list_test = sorted(
                    global_cm_ti_test,
                    key=lambda x: (counts_global_cm_ti_test[x], x),
                    reverse=True
                )

                counts_global_cm_ti_train = Counter(global_cm_ti_train)
                new_list_train = sorted(
                    global_cm_ti_train,
                    key=lambda x: (counts_global_cm_ti_train[x], x),
                    reverse=True
                )

                # TODO: n_of_majority_vote を超えているやつは捨てる
                raw_y_human_original = self.tasks.raw_y_human_original

                print('before target_global_cm_ti', len(list(set(new_list_test)))) # NOQA

                target_global_cm_ti_test = list(filter(
                    lambda ti: (
                        len(raw_y_human_original[ti]) < self.n_of_majority_vote
                    ),
                    list(set(new_list_test))
                ))

                target_global_cm_ti_train = list(filter(
                    lambda ti: (
                        len(raw_y_human_original[ti]) < self.n_of_majority_vote
                    ),
                    list(set(new_list_train))
                ))

                print('after target_global_cm_ti', len(target_global_cm_ti_test)) # NOQA
                print()

                batch_size = self.human_crowd.n_of_batch_size
                for n in range(self.n_of_majority_vote - 1):
                    self.assign_to_human_workers(
                        target_global_cm_ti_test[:batch_size]
                    )
                    self.assign_to_human_workers(
                        target_global_cm_ti_train[:batch_size]
                    )

                self.report_log()

            else:
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

            assigned_indexes = self.assign_to_human_workers()
            if self.additional_assignment_strategy == 'all':
                for n in range(self.n_of_majority_vote - 1):
                    print('do majority_vote')
                    self.assign_to_human_workers(assigned_indexes)

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
