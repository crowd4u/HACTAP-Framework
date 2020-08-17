import torch
import pandas as pd
import argparse
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import torchvision.models as models
from skorch import NeuralNetClassifier

from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.utils import random_strategy, ImageFolderWithPaths
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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args = parser.parse_args()
    reporter = Reporter(args)

    # Prepare task
    mind_dataset = ImageFolderWithPaths(
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
    X_root, y_root, idx_root = next(iter(dataloader))
    tasks = Tasks(X_root, y_root, indexes=idx_root)

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
                query_strategy=query_strategy
            )
        ),
        AIWorker(
            ActiveLearner(
                estimator=NeuralNetClassifier(
                    models.resnet18(), device=device
                ),
                query_strategy=query_strategy
            )
        ),
        AIWorker(
            ActiveLearner(
                estimator=NeuralNetClassifier(
                    models.mobilenet_v2(), device=device
                ),
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

    output = solver.run()

    output_df = pd.DataFrame(
        {
            'index': output.raw_indexes,
            'y_human': output.raw_y_human,
            'y_ai': output.raw_y_ai,
            'ground_truth': output.raw_ground_truth
        }
    )

    output_name = './results/{}/{}_output.csv'.format(
        reporter.group_id, reporter.experiment_id
    )
    output_df.to_csv(output_name)


if __name__ == "__main__":
    main()