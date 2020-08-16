from sklearn.metrics import accuracy_score
import hashlib
import numpy as np
import time


def report_metrics(tasks):
    accuracy_all = accuracy_score(
        tasks.y_all_labeled_ground_truth,
        tasks.y_all_labeled
    )

    accuracy_ai = accuracy_score(
        tasks.y_ai_labeled_ground_truth,
        tasks.y_ai_labeled
    )

    return {
        "n_human_tasks": len(tasks.human_labeled_indexes),
        "n_ai_tasks": len(tasks.ai_labeled_indexes),
        "n_all_tasks": len(tasks.all_labeled_indexes),
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
