import hashlib
import time
import argparse
import warnings
import pickle
import collections
from itertools import compress
import os
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from skorch import NeuralNetClassifier
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import beta


warnings.simplefilter('ignore')

NUMBER_OF_MONTE_CARLO_TRIAL = 10000

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s - %(asctime)s] %(message)s'
)

#////////// library
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def custom_query_strategy(_classifier, x_current, n_instances):
    query_idx = np.random.choice(range(len(x_current)), size=n_instances, replace=False)
    return query_idx, x_current[query_idx]

QUERY_STRATEGIES = {
    'uncertainty_sampling': uncertainty_sampling,
    'random': custom_query_strategy
}

def now():
    return str(time.time()).split('.')[0]

DATASETS = {
    'mnist': {
        'data': MNIST('.', download=True, transform=ToTensor()),
    }
}

class AIWorker:
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

class TaskCluster:
    def __init__(self, model, info):
        self.__model = model
        self.__info = info

        self.__n_answerable_tasks = 0
        self.__match_rate_with_human = 0
        self.__conflict_rate_with_human = 0
        self.__bata_dist = []


    def update_status(self, dataset):
        # calc for remaining tasks
        assigned_idx = range(len(dataset.x_remaining))
        y_pred = torch.tensor(self.__model.predict(dataset.x_remaining))
        mask = y_pred == self.__info['accepted_rule']['to']

        _assigned_idx = list(compress(assigned_idx, mask.numpy()))
        _y_pred = y_pred.masked_select(mask)

        print('remaining: ', len(y_pred), 'answerable: ', len(_y_pred))
        self.__n_answerable_tasks = len(_y_pred)

        # calc for human labeled tasks
        assigned_idx = range(len(dataset.x_test))
        y_pred = torch.tensor(self.__model.predict(dataset.x_test))
        mask = y_pred == self.__info['accepted_rule']['to']

        _assigned_idx = list(compress(assigned_idx, mask.numpy()))
        _y_pred = y_pred.masked_select(mask)
        _y_human = dataset.y_test.masked_select(mask)
        # print(len(_y_pred), len(_y_human))

        for _p, _h in zip(_y_pred, _y_human):
            if _p == _h:
                self.__match_rate_with_human += 1

        self.__conflict_rate_with_human = len(_y_pred) - self.__match_rate_with_human

        print(len(_y_pred), self.__match_rate_with_human, self.__conflict_rate_with_human)

        self.__bata_dist = beta.rvs(
            (1 + self.__match_rate_with_human),
            (1 + self.__conflict_rate_with_human),
            size=NUMBER_OF_MONTE_CARLO_TRIAL
        )

        print(self.__bata_dist)

        return

    @property
    def n_answerable_tasks(self):
        return self.__n_answerable_tasks

    @property
    def bata_dist(self):
        return self.__bata_dist

class Dataset:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

        self.__x_remaining = x
        self.__y_remaining = y

        self.__x_human = torch.tensor([], dtype=torch.long)
        self.__y_human = torch.tensor([], dtype=torch.long)

        self.__x_ai = torch.tensor([], dtype=torch.long)
        self.__y_ai = torch.tensor([], dtype=torch.long)
        self.__y_ai_ground_truth = torch.tensor([], dtype=torch.long)

        self.__x_train = []
        self.__y_train = []
        self.__x_test = []
        self.__y_test = []

    @property
    def x_remaining(self):
        return self.__x_remaining

    @property
    def y_remaining(self):
        return self.__y_remaining

    @property
    def x_human(self):
        return self.__x_human

    @property
    def y_human(self):
        return self.__y_human

    @property
    def x_ai(self):
        return self.__x_ai

    @property
    def y_ai(self):
        return self.__y_ai

    @property
    def y_ai_ground_truth(self):
        return self.__y_ai_ground_truth

    @property
    def y_assigned(self):
        return torch.cat([self.__y_human, self.__y_ai], dim=0)

    @property
    def y_assigned_ground_truth(self):
        return torch.cat([self.__y_human, self.__y_ai_ground_truth], dim=0)

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def x_test(self):
        return self.__x_test

    @property
    def y_test(self):
        return self.__y_test

    @property
    def is_not_completed(self):
        return self.__x_remaining.nelement() != 0

    def assign_tasks_to_human(self, task_ids):
        print('> assign_tasks_to_human')
        # print(task_ids)
        print(len(self.__x_human), len(self.__x_remaining))
        # print(self.__ .type(), self.__x_remaining.type())

        if len(self.__x_human) == 0:
            self.__x_human = self.__x_remaining[task_ids]
            self.__y_human = self.__y_remaining[task_ids]
        else:
            self.__x_human = torch.cat([self.__x_human, self.__x_remaining[task_ids]], dim=0)
            self.__y_human = torch.cat([self.__y_human, self.__y_remaining[task_ids]], dim=0)

        remaining_idx = np.isin(range(len(self.__x_remaining)), task_ids, invert=True)
        self.__x_remaining = self.__x_remaining[remaining_idx]
        self.__y_remaining = self.__y_remaining[remaining_idx]

        print(len(self.__x_human), len(self.__y_remaining))

        split = int(
            len(self.__x_human) / 2
        )
        self.__x_train = self.__x_human[split:]
        self.__y_train = self.__y_human[split:]
        self.__x_test = self.__x_human[:split]
        self.__y_test = self.__y_human[:split]


    def assign_tasks_to_ai(self, task_ids, y_pred):
        print('> assign_tasks_to_ai')
        print(len(self.__x_ai), len(self.__y_remaining))
        # print(type(y_pred))

        if len(self.__x_ai) == 0:
            self.__x_ai = self.__x_remaining[task_ids]
            self.__y_ai = y_pred
            self.__y_ai_ground_truth = self.__y_remaining[task_ids]
        else:
            self.__x_ai = torch.cat([self.__x_ai, self.__x_remaining[task_ids]], dim=0)
            self.__y_ai = torch.cat([self.__y_ai, y_pred], dim=0)
            self.__y_ai_ground_truth = torch.cat([self.__y_ai_ground_truth, self.__y_remaining[task_ids]], dim=0)

        remaining_idx = np.isin(range(len(self.__x_remaining)), task_ids, invert=True)
        self.__x_remaining = self.__x_remaining[remaining_idx]
        self.__y_remaining = self.__y_remaining[remaining_idx]
        print(len(self.__x_ai), len(self.__y_remaining))


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
    return scores

def make_new_log(_logger, dataset):
    log = {
        "n_human_tasks": len(dataset.x_human),
        "n_ai_tasks": len(dataset.x_ai),
        "n_all_tasks": len(dataset.x_human) + len(dataset.x_ai),
        "accuracy_all": accuracy_score(dataset.y_assigned_ground_truth, dataset.y_assigned),
        "accuracy_ai": accuracy_score(dataset.y_ai_ground_truth, dataset.y_ai),
    }
    _logger.info('log %s', log)
    return log

    # if np.mean(cross_validation_scores) > args.quality_requirements:
    #     n_all_tasks = 10000
        # n_human_tasks -= 200

    # log = {
    #     "n_human_tasks": n_human_tasks,
    #     "n_ai_tasks": n_ai_tasks,
    #     "n_all_tasks": n_all_tasks,
    #     # "accuracy_all": accuracy_score(y_ground_truth, y_finished),
    #     # "accuracy_ai": np.mean(cross_validation_scores),
    #     # "accuracy_ai_cv": cross_validation_scores,
    #     # "accuracy_ai_test": accuracy_score(y_test, learner.predict(x_test))
    # }

def evalate_al_worker_by_cv(aiw, dataset, quality_requirements):
    cross_validation_scores = cross_validation(aiw, dataset.x_human, dataset.y_human)
    score_cv_mean = np.mean(cross_validation_scores)
    log = {
        'ai_worker_id': 1,
        'accepted_rule': {
            "from": "*",
            "to": "*"
        },
        'score_cv': cross_validation_scores,
        'score_cv_mean': score_cv_mean,
        'was_accepted': score_cv_mean > quality_requirements
    }
    return log

def evalate_al_worker_by_task_cluster(aiw, dataset, quality_req):
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
                'ai_worker_id': 1,
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
        accepted_task_clusters, task_cluster_i, quality_req, significance_level
    ):
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


#//////////

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = logging.getLogger('HACTAP')

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_size', default=10000, type=int)
    parser.add_argument('--external_test_size', default=1000, type=int)
    parser.add_argument('--quality_requirements', default=0.8, type=float)
    parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
    parser.add_argument('--human_query_strategy', default='random')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--enable_task_cluster', default=0, type=int)
    parser.add_argument('--enable_quality_guarantee', default=0, type=int)
    parser.add_argument('--group_id', default='default')
    parser.add_argument('--trial_id', default=1, type=int)
    # TODO: parser.add_argument('--ai_workers', default="[MLP, LR, KM]")

    args = parser.parse_args()
    experiment_id = hashlib.md5(str(args).encode()).hexdigest()

    result = args.__dict__
    result['experiment_id'] = experiment_id
    result['started_at'] = now()
    result['logs'] = []

    logger.info('Experiment settings %s', result)

    classifier = NeuralNetClassifier(
        MLP,
        criterion=nn.CrossEntropyLoss, # => what is this?
        train_split=None,
        verbose=0,
        device=device
    )

    query_strategy = QUERY_STRATEGIES[args.human_query_strategy]

    dataset_length = args.task_size + args.external_test_size
    dataloader = DataLoader(DATASETS[args.dataset]['data'], shuffle=True, batch_size=dataset_length)
    x_root, y_root = next(iter(dataloader))
    x_root = x_root.reshape(dataset_length, 1, 28, 28)

    x_train, y_train = x_root[:(args.task_size)], y_root[:(args.task_size)]
    # x_test, y_test = x_root[(args.task_size):], y_root[(args.task_size):]

    dataset = Dataset(x_train, y_train)

    result['logs'].append(make_new_log(logger, dataset))

    # take the initial data
    initial_idx = np.random.choice(
        range(len(x_train)),
        size=args.human_crowd_batch_size,
        replace=False
    )
    dataset.assign_tasks_to_human(initial_idx)


    learner = ActiveLearner(
        estimator=classifier,
        X_training=dataset.x_train, y_training=dataset.y_train,
        query_strategy=query_strategy
    )


    result['logs'].append(make_new_log(logger, dataset))

    accepted_task_clusters = []

    # HACTAP Main Loop
    while dataset.is_not_completed:
        ai_worker_list = []

        if args.enable_task_cluster:
            learner.fit(dataset.x_train, dataset.y_train)
            ai_worker_list.extend(
                evalate_al_worker_by_task_cluster(learner, dataset, args.quality_requirements)
            )
        else:
            ai_worker_list.append(
                evalate_al_worker_by_cv(learner, dataset, args.quality_requirements)
            )
            learner.fit(dataset.x_human, dataset.y_human)

        # if args.enable_quality_guarantee != 1:
            # learner.fit(dataset.x_human, dataset.y_human)

        logger.info('Task Clusters %s', ai_worker_list)

        for ai_worker in ai_worker_list:
            # 残タスク数が0だと推論できないのでこれが必要
            if len(dataset.x_remaining) == 0:
                break

            if args.enable_quality_guarantee == 1:
                # ここで候補タスククラスタの状態を確定する
                # learner.fit(dataset.x_train, dataset.y_train)
                task_cluster_i = TaskCluster(learner, ai_worker)
                task_cluster_i.update_status(dataset)

                ai_worker['was_accepted'] = evalate_task_cluster_by_beta_dist(
                    accepted_task_clusters,
                    task_cluster_i,
                    args.quality_requirements,
                    0.05
                )
                if ai_worker['was_accepted']:
                    accepted_task_clusters.append(task_cluster_i)
                    # learner.fit(dataset.x_human, dataset.y_human)

            if not ai_worker['was_accepted']:
                continue

            logger.info('Accepted task clusters: %s', accepted_task_clusters)

            accepted_rule = ai_worker['accepted_rule']

            if accepted_rule['from'] == '*' and accepted_rule['to'] == '*':
                assigned_idx = range(len(dataset.x_remaining))
                y_pred = torch.tensor(learner.predict(dataset.x_remaining))
                dataset.assign_tasks_to_ai(assigned_idx, y_pred)
            else:
                assigned_idx = range(len(dataset.x_remaining))
                y_pred = torch.tensor(learner.predict(dataset.x_remaining))
                mask = y_pred == accepted_rule['to']

                _assigned_idx = list(compress(assigned_idx, mask.numpy()))
                _y_pred = y_pred.masked_select(mask)
                print('filter', len(_assigned_idx), len(_y_pred))
                dataset.assign_tasks_to_ai(_assigned_idx, _y_pred)

        result['logs'].append(make_new_log(logger, dataset))

        if len(dataset.x_remaining) != 0:
            if len(dataset.x_remaining) < args.human_crowd_batch_size:
                n_instances = len(dataset.x_remaining)
            else:
                n_instances = args.human_crowd_batch_size
            query_idx, _ = learner.query(
                dataset.x_remaining,
                n_instances=n_instances
            )
            dataset.assign_tasks_to_human(query_idx)

        result['logs'].append(make_new_log(logger, dataset))

    group_dir = './results/{}/'.format(args.group_id)
    os.makedirs(group_dir, exist_ok=True)
    with open('{}/{}.pickle'.format(group_dir, experiment_id), 'wb') as file:
        result['finished_at'] = now()
        logger.info('result %s', result)
        pickle.dump(result, file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
