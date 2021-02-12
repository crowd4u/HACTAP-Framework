from typing import List
from typing import Callable

import random
from collections import Counter

from hactap import solvers
from hactap.logging import get_logger
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import BaseAIWorker
from hactap.reporter import Reporter
from hactap.task_cluster import TaskCluster

logger = get_logger()


def epsilon_handler_static(thre: float) -> Callable[[Tasks], float]:
    def epsilon_handler(tasks: Tasks) -> float:
        return thre
    return epsilon_handler


class InteractiveCTA(solvers.CTA):
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
        interaction_strategy: str = 'conflict',
        epsilon_handler: Callable[[Tasks], float] = epsilon_handler_static(0.5), # NOQA
    ) -> None:
        super().__init__(
            tasks,
            human_crowd,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            significance_level,
            reporter,
            retire_used_test_data,
            n_of_majority_vote
        )
        self.interaction_strategy = interaction_strategy
        self.epsilon_handler = epsilon_handler

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

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters()
            random.shuffle(task_cluster_candidates)

            if random.random() < self.epsilon_handler(self.tasks):
                print("Exploration")
                additional_assiguments = self.create_additional_task_assignment(task_cluster_candidates, self.interaction_strategy) # NOQA

                self.assign_to_human_workers(
                    additional_assiguments,
                )
                self.report_log()

            else:
                print("Exploitation")
                # assign tasks to accepted task clusters
                for task_cluster_k in task_cluster_candidates:
                    if self.tasks.is_completed:
                        break

                    task_cluster_k.update_status(self.tasks)
                    accepted = self._evalate_task_cluster_by_bin_test(
                        task_cluster_k
                    )

                    if accepted:
                        self.assign_tasks_to_task_cluster(task_cluster_k)

                self.assign_to_human_workers(
                    n_of_majority_vote=1
                )
                self.report_log()

        self.finalize()

        return self.tasks

    def create_additional_task_assignment(
        self,
        task_clusters: List[TaskCluster],
        comparison_method: str,
    ) -> List[int]:
        global_cm_ti_train = []
        global_cm_ti_test = []

        for task_cluster_k in task_clusters:
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
            print('assignable_task_idx_test', len(assignable_task_idx_test))

            # TODO: 不一致、一致、ランダム、多数決の偏りで追加割り当て戦略を決めると比較するかも

            for _p, _h, _ti in zip(
                test_y_predict,
                test_y_human,
                assignable_task_idx_test
            ):
                if comparison_method == 'random':
                    cm_ti_test.append(_ti)

                if int(_p) == int(_h):
                    cm_ai.append(1)

                    if comparison_method == 'matching':
                        cm_ti_test.append(_ti)
                else:
                    cm_ai.append(0)
                    if comparison_method == 'conflict':
                        cm_ti_test.append(_ti)

                cm_human.append(1)

            for _p, _h, _ti in zip(
                train_y_predict,
                train_y_human,
                assignable_task_idx_train
            ):
                if comparison_method == 'random':
                    cm_ti_train.append(_ti)

                if int(_p) == int(_h):
                    cm_ai.append(1)
                    if comparison_method == 'matching':
                        cm_ti_train.append(_ti)
                else:
                    cm_ai.append(0)
                    if comparison_method == 'conflict':
                        cm_ti_train.append(_ti)

                cm_human.append(1)

            # TODO: fpのタスクを再割り当てするのを試す
            # print(confusion_matrix(test_y_human, test_y_predict))
            # print(confusion_matrix(cm_human, cm_ai).ravel())
            # print(len(cm_ti_test), cm_ti_test)

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

        # n_of_majority_vote を超えているやつは捨てる
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

        batch_size = int(self.human_crowd.n_of_batch_size / 2)

        if comparison_method == 'random':
            random.shuffle(target_global_cm_ti_test)
            random.shuffle(target_global_cm_ti_train)

        output_list = target_global_cm_ti_test[:batch_size] # NOQA
        output_list.extend(target_global_cm_ti_train[:batch_size]) # NOQA

        return output_list
