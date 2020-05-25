from sklearn.metrics import accuracy_score
import hashlib
import numpy as np
import time


def report_metrics(dataset):
    accuracy_all = accuracy_score(
        dataset.y_assigned_ground_truth,
        dataset.y_assigned
    )

    accuracy_ai = accuracy_score(dataset.y_ai_ground_truth, dataset.y_ai)

    return {
        "n_human_tasks": len(dataset.x_human),
        "n_ai_tasks": len(dataset.x_ai),
        "n_all_tasks": len(dataset.x_human) + len(dataset.x_ai),
        "accuracy_all": accuracy_all if accuracy_all == accuracy_all else 0,
        "accuracy_ai": accuracy_ai if accuracy_ai == accuracy_ai else 0
    }


def get_experiment_id(args):
    return hashlib.md5(str(args).encode()).hexdigest()


def get_timestamp():
    return str(time.time()).split('.')[0]


def random_strategy(_classifier, x_current, n_instances):
    query_idx = np.random.choice(
        range(len(x_current)),
        size=n_instances,
        replace=False
    )
    return query_idx, x_current[query_idx]
