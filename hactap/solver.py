from typing import List
from typing import Union
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from hactap.logging import get_logger
from hactap.utils import report_metrics
from hactap.tasks import Tasks
from hactap.ai_worker import BaseAIWorker
from hactap.human_crowd import IdealHumanCrowd
from hactap.reporter import Reporter


logger = get_logger()


class Solver():
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        ai_workers: List[Union[BaseAIWorker]],
        accuracy_requirement: float,
        n_of_classes: int,
        reporter: Reporter = None,
        n_of_majority_vote: int = 1
    ) -> None:
        self.tasks = tasks
        self.human_crowd = human_crowd
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement
        self.n_of_classes = n_of_classes
        self.n_of_majority_vote = n_of_majority_vote

        self.logs: List[dict] = []
        self.assignment_log: List[Tuple] = []
        self.reporter = reporter

    def run(self) -> Tasks:
        pass

    def initialize(self) -> None:
        if self.reporter:
            self.reporter.initialize()

    def finalize(self) -> None:
        if self.reporter:
            self.reporter.finalize(self.assignment_log)

    def report_log(self) -> None:
        if self.reporter:
            self.reporter.log_metrics(report_metrics(self.tasks))

    def report_assignment(
        self,
        assignment_log: Tuple
    ) -> None:
        self.assignment_log.append(assignment_log)
        logger.debug('new assignment: %s', self.assignment_log[-1])

    def check_n_of_class(self) -> bool:
        n_of_classes = self.n_of_classes
        train_set = self.tasks.train_set
        test_set = self.tasks.test_set

        train_data = DataLoader(
            train_set, batch_size=len(train_set)
        )

        test_data = DataLoader(
            test_set, batch_size=len(test_set)
        )

        _, y_train = next(iter(train_data))
        _, y_test = next(iter(test_data))
        cond_a = len(torch.unique(y_train)) == n_of_classes
        cond_b = len(torch.unique(y_test)) == n_of_classes
        return cond_a and cond_b

    def assign_to_human_workers(
        self,
        target_indexes: List[int] = [],
        n_of_majority_vote: Union[int, None] = None
    ) -> List[int]:
        if n_of_majority_vote is None:
            n_of_majority_vote = self.n_of_majority_vote

        if not self.tasks.is_completed:
            assigned_indexes = self.human_crowd.assign(
                self.tasks,
                target_indexes
            )
            logger.debug('new assignment: huamn %s', len(assigned_indexes))

            for n in range(n_of_majority_vote - 1):
                self.human_crowd.assign(self.tasks, assigned_indexes)
                logger.debug('majority_vote: huamn %s', len(assigned_indexes))

            return assigned_indexes
        else:
            return []
