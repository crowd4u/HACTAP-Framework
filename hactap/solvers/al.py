import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.data import Subset

from hactap import solver


class AL(solver.Solver):
    def __init__(
        self,
        tasks,
        ai_workers,
        accuracy_requirement,
        human_crowd_batch_size,
        reporter
    ):
        super().__init__(tasks, ai_workers, accuracy_requirement, reporter)
        self.human_crowd_batch_size = human_crowd_batch_size

    def run(self):
        self.initialize()
        self.report_log()

        while not self.tasks.is_completed:
            score = self.__evalate_al_worker_by_cv(self.ai_workers[0])

            if score > self.accuracy_requirement:
                train_set = self.tasks.train_set
                self.ai_workers[0].fit(train_set)

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
                # TODO: 半分はランダムにし、それをテストに使う
                x_assignable = self.tasks.X_assignable
                assignable_indexes = self.tasks.assignable_indexes
                x_assignable = DataLoader(
                    x_assignable, batch_size=len(x_assignable)
                )
                x_assignable = next(iter(x_assignable))[0]
                # print('x_assignable', x_assignable)
                related_query_indexes = self.ai_workers[0].query(
                    x_assignable, n_instances=self.human_crowd_batch_size
                )
                query_indexes = []
                for related_query_indexes_i in related_query_indexes:
                    query_indexes.append(
                        assignable_indexes[related_query_indexes_i]
                    )
                # print('query_indexes', query_indexes)
                query_labels = self.tasks.get_ground_truth(query_indexes)
                self.tasks.bulk_update_labels_by_human(
                    query_indexes, query_labels
                )
                self.report_log()

        self.finalize()
        return self.tasks

    def __evalate_al_worker_by_cv(self, aiw):
        test_set = self.tasks.test_set
        length_dataset = len(test_set)
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=length_dataset
        )
        x, y = next(iter(loader))

        cross_validation_scores = []
        kf = KFold(n_splits=5)
        for train_indexes, test_indexes in kf.split(test_set):
            x_test, y_test = x[test_indexes], y[test_indexes]
            # print(x_test, y_test)

            aiw.fit(Subset(test_set, train_indexes))
            cross_validation_scores.append(
                accuracy_score(y_test, aiw.predict(x_test))
            )

        # print(cross_validation_scores)
        return np.mean(cross_validation_scores)
