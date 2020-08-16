import numpy as np


def get_labels_from_humans(tasks, human_crowd_batch_size):
    if len(tasks.assignable_indexes) < human_crowd_batch_size:
        n_instances = len(tasks.assignable_indexes)
    else:
        n_instances = human_crowd_batch_size

    query_idx = np.random.choice(
        tasks.assignable_indexes,
        size=n_instances,
        replace=False
    )

    initial_labels = tasks.get_ground_truth(query_idx)
    tasks.bulk_update_labels_by_human(query_idx, initial_labels)

    return initial_labels
