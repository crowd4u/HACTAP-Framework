import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import compress
from sklearn.model_selection import KFold

from hactap.utils import report_metrics
from hactap.logging import get_logger

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='[%(levelname)s - %(asctime)s] %(message)s'
# )

logger = get_logger()


def cross_validation(learner, _x, _y):
    kfold = KFold(n_splits=5)
    scores = np.array([])

    for train_index, test_index in kfold.split(_x):
        # print("kf_train:", len(train_index), "kf_test:", len(test_index))
        kf_x_train, kf_x_test = _x[train_index], _x[test_index]
        kf_y_train, kf_y_test = _y[train_index], _y[test_index]

        learner.fit(kf_x_train, kf_y_train)
        kf_y_pred = learner.predict(kf_x_test)

        scores = np.append(
            scores,
            accuracy_score(kf_y_test, kf_y_pred)
        )

    learner.fit(kf_x_train, kf_y_train)
    kf_y_pred = learner.predict(_x)

    scores = np.append(
        scores,
        accuracy_score(_y, kf_y_pred)
    )
    return scores


def evalate_al_worker_by_cv(worker_id, aiw, dataset, quality_requirements):
    cross_validation_scores = cross_validation(aiw, dataset.x_test, dataset.y_test)
    score_cv_mean = np.mean(cross_validation_scores)
    log = {
        'ai_worker_id': worker_id,
        'accepted_rule': {
            "from": "*",
            "to": "*"
        },
        'score_cv': cross_validation_scores,
        'score_cv_mean': score_cv_mean,
        'was_accepted': score_cv_mean > quality_requirements
    }
    return log


class AL():
    def __init__(self, dataset, ai_workers, accuracy_requirement, human_crowd_batch_size):
        self.dataset = dataset
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement
        self.human_crowd_batch_size = human_crowd_batch_size

    def run(self):
        logs = []

        logs.append(report_metrics(self.dataset))
        logger.info('log: %s', logs[-1])

        learner1 = self.ai_workers[0]

        while self.dataset.is_not_completed:
            ai_worker_list = []

            learner1.fit(self.dataset.x_train, self.dataset.y_train)
            ai_worker_list.append(
                evalate_al_worker_by_cv(1, learner1, self.dataset, self.accuracy_requirement)
            )

            # if args.enable_quality_guarantee != 1:
            # learner.fit(self.dataset.x_human, self.dataset.y_human)

            logger.info('Task Clusters %s', ai_worker_list)
            random.shuffle(ai_worker_list)
            for ai_worker in ai_worker_list:
                # 残タスク数が0だと推論できないのでこれが必要
                if len(self.dataset.x_remaining) == 0:
                    break

                learner = learner1

                if not ai_worker['was_accepted']:
                    continue

                # logger.info('Accepted task clusters: %s', accepted_task_clusters)

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

            # result['logs'].append(make_new_log(logger, self.dataset))

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
            logs.append(report_metrics(self.dataset))
            logger.info('log: %s', logs[-1])

        return logs
