import numpy as np
import argparse
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from modAL.models import ActiveLearner

from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.utils import get_experiment_id
from hactap.utils import get_timestamp
from hactap.utils import random_strategy
from hactap.logging import get_logger


from mind_ai_worker import MindAIWorker


DATASET_PATH = '~/Google Drive/snippets/mind_dataset/dataset_mind'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', default='gta')
    parser.add_argument('--task_size', default=1000, type=int)
    parser.add_argument('--quality_requirements', default=0.8, type=float)
    parser.add_argument('--human_crowd_batch_size', default=200, type=int)
    parser.add_argument('--group_id', default='default')
    parser.add_argument('--trial_id', default=1, type=int)
    parser.add_argument('--significance_level', default=0.05, type=float)
    args = parser.parse_args()

    # Prepare result dict
    experiment_id = get_experiment_id(args)
    result = args.__dict__
    result['experiment_id'] = experiment_id
    result['started_at'] = get_timestamp()
    result['logs'] = []

    # Prepare logger
    logger = get_logger()
    logger.info('Experiment settings %s', result)

    # Prepare task
    height = 122 #int(500*0.2)
    width = 110 #int(455*0.2)

    mind_dataset = torchvision.datasets.ImageFolder(
        DATASET_PATH,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((height, width)),
            ToTensor()
        ])
    )

    dataloader = DataLoader(dataset=mind_dataset, batch_size=args.task_size, shuffle=True)
    X_root, y_root = next(iter(dataloader))
    tasks = Tasks(X_root, y_root)

    initial_idx = np.random.choice(
        tasks.assignable_indexes,
        size=args.human_crowd_batch_size,
        replace=False
    )
    initial_labels = tasks.get_ground_truth(initial_idx)
    tasks.bulk_update_labels_by_human(initial_idx, initial_labels)

    # Prepare AI worker
    X_train, y_train = tasks.train_set
    aiw_0 = AIWorker(
        ActiveLearner(
            estimator=MindAIWorker(),
            X_training=X_train, y_training=y_train,
            query_strategy=random_strategy
        )
    )
    ai_workers = [aiw_0]

    if args.solver == 'al':
        solver = solvers.AL(
            tasks,
            ai_workers,
            args.quality_requirements,
            args.human_crowd_batch_size
        )
    elif args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            ai_workers,
            args.quality_requirements,
            args.human_crowd_batch_size,
            args.significance_level
        )

    logs, _ = solver.run()

if __name__ == "__main__":
    main()
