import random
import torch
import collections
from scipy import stats
from itertools import compress

from hactap.logging import get_logger
from hactap.utils import report_metrics
from hactap.task_cluster import TaskCluster

NUMBER_OF_MONTE_CARLO_TRIAL = 100000

logger = get_logger()


def evalate_al_worker_by_task_cluster(worker_id, aiw, dataset, quality_req):
    # aiw.fit(dataset.x_train, dataset.y_train)
    y_pred = torch.tensor(aiw.predict(dataset.x_test))

    task_clusters = {}
    candidates = []

    for y_human_i, y_pred_i in zip(dataset.y_test, y_pred):
        # print(y_human_i, y_pred_i)
        if int(y_pred_i) not in task_clusters:
            task_clusters[int(y_pred_i)] = []
        task_clusters[int(y_pred_i)].append(int(y_human_i))

    for cluster_i, items in task_clusters.items():
        most_common_label = collections.Counter(items).most_common(1)

        # クラスタに含まれるデータがある場合に、そのクラスタの評価が行える
        # このif本当に要る？？？
        if len(most_common_label) == 1:
            label_type, label_count = collections.Counter(items).most_common(1)[0]
            p_value = stats.binom_test(
                label_count,
                n=len(items),
                p=quality_req,
                alternative='greater'
            )
            # print(collections.Counter(items), p_value)

            log = {
                'ai_worker_id': worker_id,
                'accepted_rule': {
                    "from": cluster_i,
                    "to": label_type
                },
                'was_accepted': p_value < 0.05
            }

            candidates.append(log)

    # print(task_clusters.keys())
    return candidates


def evalate_task_cluster_by_beta_dist(
        accepted_task_clusters, task_cluster_i, quality_req, significance_level, trial_id
    ):

    if task_cluster_i.n_answerable_tasks < 10 * trial_id:
        return False

    target_list = accepted_task_clusters + [task_cluster_i]

    count_success = 0.0

    for i in range(NUMBER_OF_MONTE_CARLO_TRIAL):
        numer = 0.0
        denom = 0.0
        for task_cluster in target_list:
            numer += (task_cluster.bata_dist[i] * task_cluster.n_answerable_tasks)
            denom += task_cluster.n_answerable_tasks

        overall_accuracy = numer / denom

        # print(overall_accuracy, task_cluster.n_answerable_tasks)

        if overall_accuracy > quality_req:
            count_success += 1.0

    p_value = 1.0 - (count_success / NUMBER_OF_MONTE_CARLO_TRIAL)
    # print(NUMBER_OF_MONTE_CARLO_TRIAL, count_success, p_value)
    # print(target_list)

    return p_value < significance_level


class DOBA():
    def __init__(self, dataset, ai_workers, accuracy_requirement, human_crowd_batch_size):
        self.dataset = dataset
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement
        self.human_crowd_batch_size = human_crowd_batch_size

    def run(self):
        logs = []

        logs.append(report_metrics(self.dataset))
        logger.info('log: %s', logs[-1])

        human_task_cluster = TaskCluster(0, 0)

        accepted_task_clusters = [human_task_cluster]

        while self.dataset.is_not_completed:
            ai_worker_list = []

            self.ai_workers[0].fit(self.dataset.x_train, self.dataset.y_train)
            self.ai_workers[1].fit(self.dataset.x_train, self.dataset.y_train)
            # learner3.fit(dataset.x_train, dataset.y_train)

            ai_worker_list.extend(
                evalate_al_worker_by_task_cluster(1, self.ai_workers[0], self.dataset, self.accuracy_requirement)
            )
            ai_worker_list.extend(
                evalate_al_worker_by_task_cluster(2, self.ai_workers[1], self.dataset, self.accuracy_requirement)
            )
            ai_worker_list.extend(
                evalate_al_worker_by_task_cluster(3, self.ai_workers[2], self.dataset, self.accuracy_requirement)
            )

            print('ai_workers {}'.format(ai_worker_list))

            # if args.enable_quality_guarantee != 1:
                # learner.fit(dataset.x_human, dataset.y_human)

            # logger.info('Task Clusters %s', ai_worker_list)
            print(ai_worker_list)
            random.shuffle(ai_worker_list)
            for ai_worker in ai_worker_list:
                # 残タスク数が0だと推論できないのでこれが必要
                if len(self.dataset.x_remaining) == 0:
                    break

                if ai_worker['ai_worker_id'] == 1:
                    learner = self.ai_workers[0]

                if ai_worker['ai_worker_id'] == 2:
                    learner = self.ai_workers[1]

                if ai_worker['ai_worker_id'] == 3:
                    learner = self.ai_workers[2]

                # ここで候補タスククラスタの状態を確定する
                # learner.fit(dataset.x_train, dataset.y_train)
                task_cluster_i = TaskCluster(learner, ai_worker)
                task_cluster_i.update_status(self.dataset)
                accepted_task_clusters[0].update_status_human(self.dataset)

                ai_worker['was_accepted'] = evalate_task_cluster_by_beta_dist(
                    accepted_task_clusters,
                    task_cluster_i,
                    self.accuracy_requirement,
                    0.05,
                    10
                )
                if ai_worker['was_accepted']:
                    accepted_task_clusters.append(task_cluster_i)
                    # learner.fit(dataset.x_human, dataset.y_human)

                if not ai_worker['was_accepted']:
                    continue

                logger.info('Accepted task clusters: %s', accepted_task_clusters)

                accepted_rule = ai_worker['accepted_rule']

                if accepted_rule['from'] == '*' and accepted_rule['to'] == '*':
                    assigned_idx = range(len(self.dataset.x_remaining))
                    y_pred = torch.tensor(learner.predict(self.dataset.x_remaining))
                    self.dataset.assign_tasks_to_ai(assigned_idx, y_pred)
                else:
                    assigned_idx = range(len(self.dataset.x_remaining))
                    y_pred = torch.tensor(learner.predict(self.dataset.x_remaining))
                    mask = y_pred == accepted_rule['from']

                    _assigned_idx = list(compress(assigned_idx, mask.numpy()))
                    _y_pred = y_pred.masked_select(mask)
                    # print(_y_pred)
                    _y_pred[_y_pred == accepted_rule['from']] = accepted_rule['to']
                    _y_pred.type(torch.LongTensor)
                    # print(_y_pred)
                    # print('filter', len(_assigned_idx), len(_y_pred))
                    self.dataset.assign_tasks_to_ai(_assigned_idx, _y_pred)

            # result['logs'].append(make_new_log(logger, dataset))
            # logger.info('log: %s', report_metrics(self.dataset))
            logs.append(report_metrics(self.dataset))
            logger.info('log: %s', logs[-1])

            if len(self.dataset.x_remaining) != 0:
                if len(self.dataset.x_remaining) < self.human_crowd_batch_size:
                    n_instances = len(self.dataset.x_remaining)
                else:
                    n_instances = self.human_crowd_batch_size
                query_idx, _ = learner.query(
                    self.dataset.x_remaining,
                    n_instances=n_instances
                )
                self.dataset.assign_tasks_to_human(query_idx)

            # result['logs'].append(make_new_log(logger, self.dataset))
            # logger.info('log: %s', report_metrics(self.dataset))
            logs.append(report_metrics(self.dataset))
            logger.info('log: %s', logs[-1])

        return logs
