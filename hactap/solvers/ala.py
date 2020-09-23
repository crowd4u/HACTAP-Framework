import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.data import Subset

from hactap import solver
from hactap.logging import get_logger
from hactap.human_crowd import get_labels_from_humans_by_random

logger = get_logger()


class ALA(solver.Solver):
    def __init__(
        self,
        tasks,
        ai_workers,
        accuracy_requirement,
        n_of_classes,
        human_crowd_batch_size,
        reporter,
        human_crowd
    ):
        super().__init__(
            tasks, ai_workers, accuracy_requirement, n_of_classes, reporter,
            human_crowd
        )
        self.human_crowd_batch_size = human_crowd_batch_size

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

        while not self.tasks.is_completed:
            score = self.__evalate_al_worker_by_cv(
                self.ai_workers[0]
            )

            train_set = self.tasks.train_set
            self.ai_workers[0].fit(train_set)

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
                human_label_size = int(self.human_crowd_batch_size / 2)
                x_assignable = self.tasks.X_assignable_human()
                assignable_indexes = self.tasks.human_assignable_indexes()
                x_assignable = DataLoader(
                    x_assignable, batch_size=len(x_assignable)
                )
                x_assignable = next(iter(x_assignable))[0]
                # print('x_assignable', x_assignable)
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
                # print('query_indexes', query_indexes)
                query_labels = self.tasks.get_ground_truth(query_indexes)
                self.tasks.bulk_update_labels_by_human(
                    query_indexes, query_labels, label_target='train'
                )
                self.report_log()

                get_labels_from_humans_by_random(
                    self.tasks, human_label_size, label_target='test'
                )
                self.report_log()

        self.finalize()
        return self.tasks

    def __evalate_al_worker_by_cv(self, aiw):
        logger.debug("cross validation -")
        n_splits = 5
        test_set = self.tasks.test_set
        length_dataset = len(test_set)
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=length_dataset
        )
        x, y = next(iter(loader))

        cross_validation_scores = []
        kf = KFold(n_splits=n_splits)
        for train_indexes, test_indexes in kf.split(test_set):
            x_test, y_test = x[test_indexes], y[test_indexes]

            try:
                aiw.fit(Subset(test_set, train_indexes))
                cross_validation_scores.append(
                    accuracy_score(y_test, aiw.predict(x_test))
                )
            except IndexError as err:
                cross_validation_scores.append(
                    0
                )
                logger.error(
                    "train failed. return 0 as the score. {}".format(err)
                )

            logger.debug("cross validation - {}".format(
                cross_validation_scores
            ))

        return np.mean(cross_validation_scores)
