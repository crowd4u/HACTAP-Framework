import torch
import collections
from scipy import stats
import random
from itertools import compress

from hactap.utils import report_metrics
from hactap.logging import get_logger

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


class OBA():
    def __init__(self, tasks, ai_workers, accuracy_requirement, human_crowd_batch_size):
        pass
        self.tasks = tasks
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement
        self.human_crowd_batch_size = human_crowd_batch_size

    def run(self):

        logs = []

        # logger = logging.getLogger('HACTAP')
        logs.append(report_metrics(self.tasks))
        logger.info('log: %s', logs[-1])

        # HACTAP Main Loop
        while self.tasks.is_not_completed:
            ai_worker_list = []

            self.ai_workers[0].fit(self.tasks.x_train, self.tasks.y_train)
            self.ai_workers[1].fit(self.tasks.x_train, self.tasks.y_train)
            # learner3.fit(self.tasks.x_train, self.tasks.y_train)
            ai_worker_list.extend(
                evalate_al_worker_by_task_cluster(1, self.ai_workers[0], self.tasks, self.accuracy_requirement)
            )
            ai_worker_list.extend(
                evalate_al_worker_by_task_cluster(2, self.ai_workers[1], self.tasks, self.accuracy_requirement)
            )
            ai_worker_list.extend(
                evalate_al_worker_by_task_cluster(3, self.ai_workers[2], self.tasks, self.accuracy_requirement)
            )

            # if args.enable_quality_guarantee != 1:
            # learner.fit(self.tasks.x_human, self.tasks.y_human)

            logger.info('Task Clusters %s', ai_worker_list)
            random.shuffle(ai_worker_list)
            for ai_worker in ai_worker_list:
                # 残タスク数が0だと推論できないのでこれが必要
                if len(self.tasks.x_remaining) == 0:
                    break

                if ai_worker['ai_worker_id'] == 1:
                    learner = self.ai_workers[0]

                if ai_worker['ai_worker_id'] == 2:
                    learner = self.ai_workers[1]

                if ai_worker['ai_worker_id'] == 3:
                    learner = self.ai_workers[2]

                if not ai_worker['was_accepted']:
                    continue

                # logger.info('Accepted task clusters: %s', accepted_task_clusters)

                accepted_rule = ai_worker['accepted_rule']

                if accepted_rule['from'] == '*' and accepted_rule['to'] == '*':
                    assigned_idx = range(len(self.tasks.x_remaining))
                    y_pred = torch.tensor(learner.predict(self.tasks.x_remaining))
                    self.tasks.assign_tasks_to_ai(assigned_idx, y_pred)
                else:
                    assigned_idx = range(len(self.tasks.x_remaining))
                    y_pred = torch.tensor(learner.predict(self.tasks.x_remaining))
                    mask = y_pred == accepted_rule['from']

                    _assigned_idx = list(compress(assigned_idx, mask.numpy()))
                    _y_pred = y_pred.masked_select(mask)
                    # print(_y_pred)
                    _y_pred[_y_pred == accepted_rule['from']] = accepted_rule['to']
                    _y_pred.type(torch.LongTensor)
                    # print(_y_pred)
                    # print('filter', len(_assigned_idx), len(_y_pred))
                    self.tasks.assign_tasks_to_ai(_assigned_idx, _y_pred)

            # result['logs'].append(make_new_log(logger, self.tasks))
            # logger.info('log: %s', report_metrics(self.tasks))
            logs.append(report_metrics(self.tasks))
            logger.info('log: %s', logs[-1])

            if len(self.tasks.x_remaining) != 0:
                if len(self.tasks.x_remaining) < self.human_crowd_batch_size:
                    n_instances = len(self.tasks.x_remaining)
                else:
                    n_instances = self.human_crowd_batch_size
                query_idx, _ = learner.query(
                    self.tasks.x_remaining,
                    n_instances
                )
                self.tasks.assign_tasks_to_human(query_idx)

            # result['logs'].append(make_new_log(logger, self.tasks))
            logs.append(report_metrics(self.tasks))
            logger.info('log: %s', logs[-1])

        return logs
