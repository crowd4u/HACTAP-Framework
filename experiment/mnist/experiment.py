import argparse
import warnings
import pickle
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from hactap import solvers
from hactap.utils import report_metrics
from hactap.utils import get_experiment_id
from hactap.utils import get_timestamp
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.utils import random_strategy
from hactap.logging import get_logger

warnings.simplefilter('ignore')

logger = get_logger()


def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', default='al')
    parser.add_argument('--task_size', default=10000, type=int)
    parser.add_argument('--quality_requirements', default=0.8, type=float)
    parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
    parser.add_argument('--group_id', default='default')
    parser.add_argument('--trial_id', default=1, type=int)
    parser.add_argument('--external_test_size', default=1000, type=int)
    parser.add_argument('--significance_level', default=0.05, type=float)
    args = parser.parse_args()

    # prepare the result store
    experiment_id = get_experiment_id(args)
    result = args.__dict__
    result['experiment_id'] = experiment_id
    result['started_at'] = get_timestamp()
    result['logs'] = []
    logger.info('Experiment settings %s', result)

    # parepare the tasks
    dataset_length = args.task_size  # + args.external_test_size
    dataloader = DataLoader(
        MNIST('.', download=True, transform=ToTensor()),
        shuffle=True,
        batch_size=dataset_length
    )
    x_root, y_root = next(iter(dataloader))
    x_root = x_root.reshape(dataset_length, 28*28)
    x_train, y_train = x_root[:(args.task_size)], y_root[:(args.task_size)]
    tasks = Tasks(x_train, y_train)

    result['logs'].append(report_metrics(tasks))
    logger.debug('log: %s', result['logs'][-1])

    # take the initial data
    initial_idx = np.random.choice(
        tasks.assignable_indexes,
        size=args.human_crowd_batch_size,
        replace=False
    )

    initial_labels = tasks.get_ground_truth(initial_idx)

    tasks.bulk_update_labels_by_human(initial_idx, initial_labels)

    # select query strategy
    if args.solver == 'al':
        query_strategy = uncertainty_sampling
    else:
        query_strategy = random_strategy

    X_train, y_train = tasks.train_set

    # build AI workers
    aiw_1 = AIWorker(
        ActiveLearner(
            estimator=MLPClassifier(),
            X_training=X_train, y_training=y_train,
            query_strategy=query_strategy
        ),
        skip_update=False
    )

    aiw_2 = AIWorker(
        ActiveLearner(
            estimator=LogisticRegression(),
            X_training=X_train, y_training=y_train,
            query_strategy=query_strategy
        ),
        skip_update=False
    )

    aiw_3 = AIWorker(
        ActiveLearner(
            estimator=KMeans(n_clusters=20),
            X_training=X_train, y_training=y_train,
            query_strategy=query_strategy
        ),
        skip_update=True
    )

    # start task assignment
    if args.solver == 'al':
        solver = solvers.AL(
            tasks,
            [aiw_1],
            args.quality_requirements,
            args.human_crowd_batch_size
        )
    elif args.solver == 'oba':
        solver = solvers.OBA(
            tasks,
            [aiw_1, aiw_2, aiw_3],
            args.quality_requirements,
            args.human_crowd_batch_size,
            args.significance_level
        )
    elif args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            [aiw_1, aiw_2, aiw_3],
            args.quality_requirements,
            args.human_crowd_batch_size,
            args.significance_level
        )
    elif args.solver == 'gtaonce':
        solver = solvers.GTAOnce(
            tasks,
            [aiw_1, aiw_2, aiw_3],
            args.quality_requirements,
            args.human_crowd_batch_size,
            args.significance_level
        )

    logs, _ = solver.run()

    result['logs'].extend(logs)
    # result['assignment_logs'] = assignment_logs

    group_dir = './results/{}/'.format(args.group_id)
    os.makedirs(group_dir, exist_ok=True)
    with open('{}/{}.pickle'.format(group_dir, experiment_id), 'wb') as file:
        result['finished_at'] = get_timestamp()
        logger.info('result %s', result)
        pickle.dump(result, file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
