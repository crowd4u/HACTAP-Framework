import numpy as np


def get_labels_from_humans(tasks, human_crowd_batch_size):
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
    tasks.bulk_update_labels_by_human(query_idx, initial_labels)
    return initial_labels
