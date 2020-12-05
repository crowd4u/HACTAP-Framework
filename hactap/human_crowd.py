from typing import List
from typing import Union

import numpy as np
import random

from hactap.tasks import Tasks


class IdealHumanCrowd:
    def __init__(
        self,
        assignment_order: str,
        n_of_batch_size: int,
        correct_prob: float
    ):
        self.assignment_order = assignment_order
        self.n_of_batch = n_of_batch_size
        self.correct_prob = correct_prob

    @property
    def n_of_batch_size(self) -> int:
        return self.n_of_batch

    def assign(
        self,
        tasks: Tasks,
        target_indexes: List[int] = []
    ) -> list:
        n_of_samples = 1000
        additional = False

        if len(target_indexes) != 0:
            query_idx = target_indexes
            additional = True
        else:
            if len(tasks.human_assignable_indexes()) < self.n_of_batch: # NOQA
                n_instances = len(tasks.human_assignable_indexes())
            else:
                n_instances = self.n_of_batch

            query_idx = np.random.choice(
                tasks.human_assignable_indexes(),
                size=n_instances,
                replace=False
            )
        ground_truth_labels = tasks.get_ground_truth(query_idx)

        human_labels = []

        for gtl in ground_truth_labels:
            label_candidates_copy = tasks.class_candidates.copy()
            is_success = random.choices(
                [True, False],
                [
                    n_of_samples * self.correct_prob,
                    n_of_samples * (1 - self.correct_prob)
                ]
            )[0]
            if is_success:
                human_labels.append(gtl)
            else:
                label_candidates_copy.remove(gtl)
                human_labels.append(random.choice(label_candidates_copy))

        tasks.bulk_update_labels_by_human(
            query_idx, human_labels, label_target=None, additional=additional
        )
        return query_idx


def get_labels_from_humans_by_random(
    tasks: Tasks,
    human_crowd_batch_size: int,
    label_target: Union[str, None] = None
) -> list:
    if len(tasks.human_assignable_indexes()) < human_crowd_batch_size: # NOQA
        n_instances = len(tasks.human_assignable_indexes())
    else:
        n_instances = human_crowd_batch_size

    query_idx = np.random.choice(
        tasks.human_assignable_indexes(),
        size=n_instances,
        replace=False
    )

    # query_idx = tasks.human_assignable_indexes()[:n_instances]

    # tasks.human_assignable_indexes()

    # print(query_idx, len(query_idx))

    # print('query_idx', query_idx)

    initial_labels = tasks.get_ground_truth(query_idx)
    # print('initial_labels', initial_labels)

    # print('initial_labels', initial_labels)
    tasks.bulk_update_labels_by_human(query_idx, initial_labels, label_target)
    return initial_labels


def get_labels_from_humans_by_original_order(
    tasks: Tasks,
    human_crowd_batch_size: int,
    label_target: bool = None
) -> List:
    if len(tasks.human_assignable_indexes()) < human_crowd_batch_size: # NOQA
        n_instances = len(tasks.human_assignable_indexes())
    else:
        n_instances = human_crowd_batch_size

    query_idx = tasks.human_assignable_indexes()[:n_instances]

    initial_labels = tasks.get_ground_truth(query_idx)
    tasks.bulk_update_labels_by_human(
        query_idx, initial_labels, label_target=None
    )
    return initial_labels
