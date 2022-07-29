from typing import Dict
from hactap.task_cluster import TaskCluster
from hactap.tasks import Tasks

from sklearn.metrics import accuracy_score, f1_score
import hashlib
# import numpy as np
import time
import torchvision


def report_metrics(tasks: Tasks) -> dict:
    y_all_labeled_ground_truth_for_metric = tasks.y_all_labeled_ground_truth_for_metric # NOQA
    y_all_labeled_for_metric = tasks.y_all_labeled_for_metric

    accuracy_all = accuracy_score(
        y_all_labeled_ground_truth_for_metric,
        y_all_labeled_for_metric
    )

    f1_all = f1_score(
        y_all_labeled_ground_truth_for_metric,
        y_all_labeled_for_metric,
        average='macro'
    )

    y_ai_labeled_ground_truth_for_metric = tasks.y_ai_labeled_ground_truth_for_metric # NOQA
    y_ai_labeled_for_metric = tasks.y_ai_labeled_for_metric
    accuracy_ai = accuracy_score(
        y_ai_labeled_ground_truth_for_metric,
        y_ai_labeled_for_metric
    )

    f1_ai = f1_score(
        y_ai_labeled_ground_truth_for_metric,
        y_ai_labeled_for_metric,
        average='macro'
    )

    return {
        "n_human_tasks": len(tasks.human_labeled_indexes),
        "n_human_tasks_all": tasks.human_labeled_mv(),
        "n_ai_tasks": len(tasks.ai_labeled_indexes),
        "n_all_tasks": len(tasks.all_labeled_indexes),
        "accuracy_all": accuracy_all if accuracy_all == accuracy_all else 0,
        "accuracy_ai": accuracy_ai if accuracy_ai == accuracy_ai else 0,
        "f1_all": f1_all if f1_all == f1_all else 0,
        "f1_ai": f1_ai if f1_ai == f1_ai else 0
    }


def report_task_cluster(task_cluster: TaskCluster, accepted: bool):
    return {
        "id": task_cluster.id,
        "sub_id": task_cluster.sub_id,
        "ai_worker": task_cluster.model.get_worker_name(),
        "ai_worker_id": task_cluster.aiw_id,
        "accepted": accepted,
        "match_rate_with_human": task_cluster.match_rate_with_human,
        "conflict_rate_with_human": task_cluster.conflict_rate_with_human,
        "rule": task_cluster.rule["rule"]
    }


def report_run_iter(iter_n: int, next_tc_id: int):
    return {
        "iter_n": iter_n,
        "next_task_cluster_id": next_tc_id
    }


def get_experiment_id(args: Dict) -> str:
    return hashlib.md5(str(args).encode()).hexdigest()


def get_timestamp() -> str:
    return str(time.time()).split('.')[0]


# def random_strategy(_classifier, x_current, n_instances):
#     query_idx = np.random.choice(
#         range(len(x_current)),
#         size=n_instances,
#         replace=False
#     )
#     return query_idx, x_current[query_idx]


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def get_label(self, index: int) -> tuple:
        path, label = self.imgs[index]
        return (path, label)
