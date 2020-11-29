import torch
from sklearn.metrics import accuracy_score
from hactap.logging import get_logger
from torch.utils.data import DataLoader
from hactap import solver
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.reporter import Reporter
from hactap.ai_worker import AIWorker

logger = get_logger()


class Baseline(solver.Solver):
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        ai_workers: list,
        accuracy_requirement: float,
        n_of_classes: int,
        reporter: Reporter,
    ) -> None:
        super().__init__(
            tasks,
            human_crowd,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            reporter,
        )

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

            self.assign_to_human_workers()
            # if not self.tasks.is_completed:
            #     get_labels_from_humans_by_random(
            #     self.tasks, self.human_crowd_batch_size, label_target=None
            #     )
            #     self.report_log()

        self.finalize()
        return self.tasks

    def __evalate_al_worker_by_test_accuracy(self, aiw: AIWorker) -> float:
        test_set = self.tasks.test_set
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=len(test_set)
        )
        x, y = next(iter(loader))

        return accuracy_score(y, aiw.predict(x))
