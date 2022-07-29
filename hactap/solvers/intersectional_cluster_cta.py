from typing import List
from typing import Tuple

import random
from collections import Counter

import itertools
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from hactap import solvers
from hactap.logging import get_logger
from hactap.tasks import Tasks
from hactap.human_crowd import IdealHumanCrowd
from hactap.ai_worker import BaseAIWorker
from hactap.reporter import Reporter
from hactap.task_cluster import TaskCluster
from hactap.intersectional_model import IntersectionalModel

logger = get_logger()


def key_of_task_cluster_k(x: Tuple[int, int, int]) -> int:
    return x[0]


def group_by_task_cluster(
    clustering_function: IntersectionalModel,
    dataset: Dataset,
    indexes: List[int]
) -> List:
    length_dataset = len(dataset)
    predict_loader = DataLoader(
        dataset, batch_size=length_dataset
    )
    test_set_loader = DataLoader(
        dataset, batch_size=length_dataset
    )

    clustering_function.fit(dataset)
    sub_text_x, _ = next(iter(predict_loader))
    test_set_predict = clustering_function.predict(sub_text_x)
    test_set_y = []
    for _, (_, sub_test_y) in enumerate(test_set_loader):
        test_set_y.extend(sub_test_y.tolist())

    # print('dataset_predict', len(test_set_predict))
    # print('dataset_y', len(test_set_y))
    # print('dataset_indexes', len(indexes))

    tcs = itertools.groupby(
        sorted(
            list(zip(test_set_predict, test_set_y, indexes)),
            key=key_of_task_cluster_k
        ),
        key_of_task_cluster_k
    )

    return list(map(lambda x: (x[0], list(x[1])), tcs))


def get_all_of_intersection(A: List, B: List) -> List:
    result = []
    for a in A:
        for b in B:
            c = set(a).intersection(set(b))
            if len(c) > 0:
                result.append(c)
    return result


class IntersectionalClusterCTA(solvers.CTA):
    def __init__(
        self,
        tasks: Tasks,
        human_crowd: IdealHumanCrowd,
        human_crowd_batch_size: int,
        ai_workers: List[BaseAIWorker],
        accuracy_requirement: float,
        n_of_classes: int,
        significance_level: float,
        reporter: Reporter,
        clustering_function: IntersectionalModel,
        retire_used_test_data: bool = False,
        n_of_majority_vote: int = 1,
        report_all_task_clusters: bool = False
    ) -> None:
        super().__init__(
            tasks,
            human_crowd,
            human_crowd_batch_size,
            ai_workers,
            accuracy_requirement,
            n_of_classes,
            significance_level,
            reporter,
            retire_used_test_data,
            n_of_majority_vote,
            report_all_task_clusters
        )
        self.clustering_funcion = clustering_function
        self.task_cluster_id = 0

    def run(self) -> Tasks:
        self.initialize()
        self.report_log()

        self.assign_to_human_workers()
        self.report_log()

        while not self.check_n_of_class():
            self.assign_to_human_workers()
            self.report_log()

        while not self.tasks.is_completed:
            train_set = self.tasks.train_set
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(train_set)

            task_cluster_candidates = self.list_task_clusters_by_any()
            random.shuffle(task_cluster_candidates)

            # assign tasks to accepted task clusters
            for task_cluster_k in task_cluster_candidates:
                if self.tasks.is_completed:
                    break

                task_cluster_k.update_status(self.tasks)
                accepted = self._evalate_task_cluster_by_bin_test(
                    task_cluster_k
                )
                if self.report_all_task_clusters:
                    self.report_task_cluster(task_cluster_k, accepted)

                if accepted:
                    self.assign_tasks_to_task_cluster(task_cluster_k)

            self.assign_to_human_workers()
            self.report_log()

        self.finalize()

        return self.tasks

    def list_task_clusters_by_any(self) -> List[TaskCluster]:
        task_clusters = []

        task_clusters_by_any_function = self.create_task_cluster_from_any_function() # NOQA
        logger.debug(f"n_of_tcs by user {len(task_clusters_by_any_function)}")

        for index, _ in enumerate(self.ai_workers):
            task_cluster_by_ai_worker = self.create_task_cluster_from_ai_worker(index)  # NOQA
            logger.debug(f"n_of_tcs by ai {len(task_cluster_by_ai_worker)}")
            intersectional_task_cluster = self.intersection_of_task_clusters(
                task_cluster_by_ai_worker,
                task_clusters_by_any_function
            )
            task_clusters.extend(intersectional_task_cluster)
            tc_len = len(intersectional_task_cluster)
            logger.debug(f"n_of_ic_tcs {tc_len}")

        logger.debug(f"n_of_ic_tcs_all {len(task_clusters)}")
        return task_clusters

    def create_task_cluster_from_any_function(
        self,
        function: IntersectionalModel = None
    ) -> List[TaskCluster]:
        task_clusters: List[TaskCluster] = []
        cluster_function = function if function is not None else self.clustering_funcion # NOQA

        tc_train = group_by_task_cluster(
            cluster_function,
            self.tasks.train_set,
            self.tasks.train_indexes
        )

        tc_test = group_by_task_cluster(
            cluster_function,
            self.tasks.test_set,
            self.tasks.test_indexes
        )

        tc_remain = list(group_by_task_cluster(
            cluster_function,
            self.tasks.X_assignable,
            self.tasks.assignable_indexes
        ))

        for key, items_of_tc_test in tc_test:
            # print('key', key)
            # print('tc_train', tc_train)
            # print(key, items_of_tc)
            human_labels = list(map(lambda x: x[1], items_of_tc_test))
            occurence_count = Counter(human_labels)
            max_human_label = occurence_count.most_common(1)[0][0]
            # print(human_labels)
            # print(max_human_label)

            # print('items_of_tc_test', items_of_tc_test)
            items_of_tc_train = []
            _items_of_tc_train = list(filter(lambda x: x[0] == key, tc_train)) # NOQA
            # print('items_of_tc_train', items_of_tc_train)

            if len(_items_of_tc_train) == 1:
                items_of_tc_train = _items_of_tc_train[0][1]

            items_of_tc_remain = []
            _items_of_tc_remain = list(filter(lambda x: x[0] == key, tc_remain)) # NOQA

            if len(_items_of_tc_remain) == 1:
                # print(_items_of_tc_remain[0][1])
                items_of_tc_remain = _items_of_tc_remain[0][1]
            # print('items_of_tc_remain', items_of_tc_remain)

            rule = {
                "rule": {
                    "from": key,
                    "to": max_human_label
                },
                "stat": {
                    # "y_pred": list(map(lambda x: x[0], items_of_tc_test)),
                    # "answerable_tasks_ids": list(map(lambda x: x[2], items_of_tc_remain)), # NOQA

                    "y_pred_test": list(map(lambda x: x[0], items_of_tc_test)),
                    "y_pred_train": list(map(lambda x: x[0], items_of_tc_train)), # NOQA
                    "y_pred_remain": list(map(lambda x: x[0], items_of_tc_remain)), # NOQA

                    "y_pred_test_human": list(map(lambda x: x[1], items_of_tc_test)), # NOQA
                    "y_pred_train_human": list(map(lambda x: x[1], items_of_tc_train)), # NOQA
                    "y_pred_remain_human": list(map(lambda x: x[1], items_of_tc_remain)), # NOQA

                    "y_pred_test_ids": list(map(lambda x: x[2], items_of_tc_test)), # NOQA
                    "y_pred_train_ids": list(map(lambda x: x[2], items_of_tc_train)), # NOQA
                    "y_pred_remain_ids": list(map(lambda x: x[2], items_of_tc_remain)) # NOQA
                }
            }

            task_clusters.append(
                TaskCluster(None, -1, rule)
            )

        return task_clusters

    def intersection_of_task_clusters(
        self,
        ai_task_clusters: List[TaskCluster],
        user_task_clusters: List[TaskCluster]
    ) -> List[TaskCluster]:
        ai_info = None
        ic_task_cluster = []
        ai_tcs_id = []
        for atc in ai_task_clusters:
            atc.update_status(self.tasks)
            ai_tc_ids = set()
            atc_all_indexes = atc.assignable_task_indexes + atc.assignable_task_idx_test + atc.assignable_task_idx_train  # NOQA
            for x in atc_all_indexes:
                ai_tc_ids.add(x)
            ai_tcs_id.append(ai_tc_ids)

        user_tcs_id = []
        for utc in user_task_clusters:
            utc.update_status(self.tasks)
            user_tc_ids = set()
            utc_all_indexes = utc.assignable_task_indexes + utc.assignable_task_idx_test + utc.assignable_task_idx_train  # NOQA
            for x in utc_all_indexes:
                user_tc_ids.add(x)
            user_tcs_id.append(user_tc_ids)
        # print("len ai_tcs_id[0]:", len(ai_tcs_id[0]))
        # print("len user_tcs_id[0]:", len(user_tcs_id[0]))
        tcs_ids = get_all_of_intersection(ai_tcs_id, user_tcs_id)
        # print("len tcs_ids:", len(tcs_ids))
        # print("all tasks of tcs_ids:", sum(map(lambda x: len(x), tcs_ids)))
        # print("len tcs_ids[0]:", len(tcs_ids[0]))

        ai_cluster_rule = {}
        stats = {}
        aic_index = 0
        for cluster in ai_task_clusters:
            serialized_cluster = {}
            for y, h, id in zip(cluster.rule["stat"]["y_pred_test"], cluster.rule["stat"]["y_pred_test_human"], cluster.rule["stat"]["y_pred_test_ids"]):  # NOQA
                serialized_cluster[id] = {
                    "status": "test",
                    "y_pred_test": y,
                    "y_pred_test_human": h,
                    "y_pred_test_id": id,
                    "aic_index": aic_index
                }
            for y, h, id in zip(cluster.rule["stat"]["y_pred_train"], cluster.rule["stat"]["y_pred_train_human"], cluster.rule["stat"]["y_pred_train_ids"]):  # NOQA
                serialized_cluster[id] = {
                    "status": "train",
                    "y_pred_train": y,
                    "y_pred_train_human": h,
                    "y_pred_train_id": id,
                    "aic_index": aic_index
                }
            for y, h, id in zip(cluster.rule["stat"]["y_pred_remain"], cluster.rule["stat"]["y_pred_remain_human"], cluster.rule["stat"]["y_pred_remain_ids"]):  # NOQA
                serialized_cluster[id] = {
                    "status": "remain",
                    "y_pred_remain": y,
                    "y_pred_remain_human": h,
                    "y_pred_remain_id": id,
                    "aic_index": aic_index
                }

            ai_cluster_rule[aic_index] = cluster.rule["rule"]
            if ai_info is None:
                ai_info = {
                    "model": cluster.model,
                    "id": cluster.aiw_id
                }
            stats.update(serialized_cluster)
            aic_index += 1

        for ids in tcs_ids:
            items_tc_test = []
            items_tc_train = []
            items_tc_remain = []
            items_tc_test_human = []
            items_tc_train_human = []
            items_tc_remain_human = []
            items_tc_test_ids = []
            items_tc_train_ids = []
            items_tc_remain_ids = []
            rule_from = None
            aic_id = None
            for id in ids:
                if aic_id is None:
                    aic_id = stats[id]["aic_index"]
                    rule_from = ai_cluster_rule[aic_id]["from"]
                status = stats[id]["status"]
                if status == "test":
                    items_tc_test.append(stats[id]["y_pred_test"])
                    items_tc_test_human.append(stats[id]["y_pred_test_human"])
                    items_tc_test_ids.append(id)
                elif status == "train":
                    items_tc_train.append(stats[id]["y_pred_train"])
                    items_tc_train_human.append(stats[id]["y_pred_train_human"])  # NOQA
                    items_tc_train_ids.append(id)
                elif status == "remain":
                    items_tc_remain.append(stats[id]["y_pred_remain"])
                    items_tc_remain_human.append(stats[id]["y_pred_remain_human"])  # NOQA
                    items_tc_remain_ids.append(id)
                else:
                    raise Exception(f"there is no item whose id is {id}")
            # print("len items_tc_test_human:", len(items_tc_test_human))
            # print("num items_tc_test_human:", len(set(items_tc_test_human)))

            if len(items_tc_test) == 0:
                # print("rejected by the number of human labels")
                continue
            occurence_count = Counter(items_tc_test_human)
            max_human_label = occurence_count.most_common(1)[0][0]
            # print("IC_CTA: max_human_label", max_human_label)
            # print('IC_CTA: rule_from', rule_from)

            rule = {
                "rule": {
                    "from": rule_from,
                    "to": max_human_label
                },
                "stat": {
                    "y_pred_test": items_tc_test,
                    "y_pred_train": items_tc_train,
                    "y_pred_remain": items_tc_remain,

                    "y_pred_test_human": items_tc_test_human,
                    "y_pred_train_human": items_tc_train_human,
                    "y_pred_remain_human": items_tc_remain_human,

                    "y_pred_test_ids": items_tc_test_ids,
                    "y_pred_train_ids": items_tc_train_ids,
                    "y_pred_remain_ids": items_tc_remain_ids
                }
            }

            ic_task_cluster.append(
                TaskCluster(
                    ai_info["model"],
                    ai_info["id"],
                    rule,
                    id=self.task_cluster_id
                )
            )
            self.task_cluster_id += 1

        return ic_task_cluster
