import argparse
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.utils import random_strategy
from hactap.reporter import Reporter
from hactap.human_crowd import get_labels_from_humans

from mind_ai_worker import MindAIWorker


# DATASET_PATH = '~/Google Drive/snippets/mind_dataset/dataset_mind'
DATASET_PATH = '~/dataset_mind'

height = 122  # int(500*0.2)
width = 110  # int(455*0.2)

parser = argparse.ArgumentParser()
parser.add_argument('--solver', default='gta')
parser.add_argument('--task_size', default=500, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=200, type=int)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)


def main():
    args = parser.parse_args()
    reporter = Reporter(args)

    # Prepare task
    mind_dataset = torchvision.datasets.ImageFolder(
        DATASET_PATH,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((height, width)),
            ToTensor()
        ])
    )
    dataloader = DataLoader(
        dataset=mind_dataset,
        batch_size=args.task_size,
        shuffle=True
    )
    X_root, y_root = next(iter(dataloader))
    tasks = Tasks(X_root, y_root)

    get_labels_from_humans(tasks, args.human_crowd_batch_size)

    # Prepare AI workers
    if args.solver == 'al':
        query_strategy = uncertainty_sampling
    else:
        query_strategy = random_strategy

    X_train, y_train = tasks.train_set
    ai_workers = [
        AIWorker(
            ActiveLearner(
                estimator=MindAIWorker(),
                X_training=X_train, y_training=y_train,
                query_strategy=query_strategy
            )
        )
    ]

    if args.solver == 'al':
        solver = solvers.AL(
            tasks,
            ai_workers,
            args.quality_requirements,
            args.human_crowd_batch_size,
            reporter=reporter
        )
    elif args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            ai_workers,
            args.quality_requirements,
            args.human_crowd_batch_size,
            args.significance_level,
            reporter=reporter
        )

    solver.run()


if __name__ == "__main__":
    main()
