from typing import List
from typing import Optional
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from hactap.tasks import Tasks
from hactap.ai_worker import BaseAIWorker
from hactap.human_crowd import IdealHumanCrowd
from hactap import solver
from hactap.logging import get_logger
from hactap.human_crowd import get_labels_from_humans_by_random
from hactap.reporter import Reporter

logger = get_logger()


class ALA(solver.Solver):
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        n_of_classes: int,
        reporter: Reporter,
        test_with_random: bool = True
    ) -> None:
        super().__init__(
            tasks,
            human_crowd,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            reporter
        )
        self.test_with_random = test_with_random

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
            self.ai_workers[0].fit(train_set)

            score = self.__evalate_al_worker_by_test_accuracy(
                self.ai_workers[0]
            )

            if score > self.accuracy_requirement:
                x_assignable = DataLoader(
                    self.tasks.X_assignable, batch_size=10_000
                )
                assignable_indexes = self.tasks.assignable_indexes
                y_pred = []

                for index, (sub_x_assignable) in enumerate(x_assignable):
                    # print(sub_x_assignable)
                    sub_y_pred = self.ai_workers[0].predict(
                        sub_x_assignable[0]
                    )
                    y_pred.extend(sub_y_pred)

                self.tasks.bulk_update_labels_by_ai(
                    assignable_indexes,
                    y_pred
                )

                self.report_log()

            if not self.tasks.is_completed:
                if self.test_with_random:
                    human_label_size = int(
                        self.human_crowd.n_of_batch_size / 2
                    )
                    get_labels_from_humans_by_random(
                        self.tasks, human_label_size, label_target='test'
                    )
                    self.get_labels_from_humans_by_query(
                        human_label_size,
                        label_target='train'
                    )
                    self.report_log()
                else:
                    self.get_labels_from_humans_by_query(
                        self.human_crowd.n_of_batch_size,
                        label_target=None
                    )
                    self.report_log()

        self.finalize()
        return self.tasks

    def get_labels_from_humans_by_query(
        self,
        human_label_size: int,
        label_target: Optional[str]
    ) -> None:
        zx_assignable = self.tasks.X_assignable_human()
        assignable_indexes = self.tasks.human_assignable_indexes()
        yx_assignable = DataLoader(
            zx_assignable, batch_size=len(zx_assignable)
        )
        x_assignable = next(iter(yx_assignable))[0]

        if len(x_assignable) < human_label_size:
            human_label_size = len(x_assignable)
        related_query_indexes = self.ai_workers[0].query(
            x_assignable, n_instances=human_label_size
        )
        query_indexes = []
        for related_query_indexes_i in related_query_indexes:
            query_indexes.append(
                assignable_indexes[related_query_indexes_i]
            )

        query_labels = self.tasks.get_ground_truth(query_indexes)
        self.tasks.bulk_update_labels_by_human(
            query_indexes, query_labels, label_target=label_target
        )
        return

    def __evalate_al_worker_by_test_accuracy(
        self,
        aiw: BaseAIWorker
    ) -> float:
        test_set = self.tasks.test_set
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=len(test_set)
        )
        x, y = next(iter(loader))

        return accuracy_score(y, aiw.predict(x))

    # def __evalate_al_worker_by_cv(self, aiw: float) -> float:
    #     logger.debug("cross validation -")
    #     n_splits = 5
    #     test_set = self.tasks.test_set
    #     length_dataset = len(test_set)
    #     loader = torch.utils.data.DataLoader(
    #         test_set, batch_size=length_dataset
    #     )
    #     x, y = next(iter(loader))

    #     cross_validation_scores = []
    #     kf = KFold(n_splits=n_splits)
    #     for train_indexes, test_indexes in kf.split(test_set):
    #         x_test, y_test = x[test_indexes], y[test_indexes]

    #         try:
    #             aiw.fit(Subset(test_set, train_indexes))
    #             cross_validation_scores.append(
    #                 accuracy_score(y_test, aiw.predict(x_test))
    #             )
    #         except IndexError as err:
    #             cross_validation_scores.append(
    #                 0
    #             )
    #             logger.error(
    #                 "train failed. return 0 as the score. {}".format(err)
    #             )

    #         logger.debug("cross validation - {}".format(
    #             cross_validation_scores
    #         ))

    #     return np.mean(cross_validation_scores)
