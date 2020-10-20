import argparse
import warnings
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.logging import get_logger
from hactap.reporter import Reporter
from hactap.human_crowd import get_labels_from_humans_by_random

warnings.simplefilter('ignore')
logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--solver',
    default='gta',
    choices=['gta', 'gta_retire', 'gta_onetime']
)
parser.add_argument('--task_size', default=10000, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)
parser.add_argument('--n_monte_carlo_trial', default=100000, type=int)
parser.add_argument('--minimum_sample_size', default=-1, type=int)


def main():
    args = parser.parse_args()
    reporter = Reporter(args)

    # parepare the tasks
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.reshape(28*28))
    ])
    mnist_dataset = MNIST('.', download=True, transform=transform)
    mnist_dataset = random_split(
        mnist_dataset,
        [args.task_size, len(mnist_dataset) - args.task_size]
    )[0]
    data_index = range(len(mnist_dataset))
    tasks = Tasks(mnist_dataset, data_index)

    # Build AI workers
    ai_workers = [
        AIWorker(MLPClassifier()),
        AIWorker(LogisticRegression()),
        AIWorker(KMeans()),
    ]

    # Start task assignment
    if args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            ai_workers,
            args.quality_requirements,
            10,
            args.human_crowd_batch_size,
            args.significance_level,
            reporter=reporter,
            human_crowd=get_labels_from_humans_by_random,
            n_monte_carlo_trial=args.n_monte_carlo_trial,
            minimum_sample_size=args.minimum_sample_size
        )
    elif args.solver == 'gta_retire':
        solver = solvers.GTA(
            tasks,
            ai_workers,
            args.quality_requirements,
            10,
            args.human_crowd_batch_size,
            args.significance_level,
            reporter=reporter,
            human_crowd=get_labels_from_humans_by_random,
            retire_used_test_data=True,
            n_monte_carlo_trial=args.n_monte_carlo_trial,
            minimum_sample_size=args.minimum_sample_size
        )
    elif args.solver == 'gta_onetime':
        solver = solvers.GTAOneTime(
            tasks,
            ai_workers,
            args.quality_requirements,
            10,
            args.human_crowd_batch_size,
            args.significance_level,
            reporter=reporter,
            human_crowd=get_labels_from_humans_by_random,
            n_monte_carlo_trial=args.n_monte_carlo_trial,
            minimum_sample_size=args.minimum_sample_size
        )

    solver.run()


if __name__ == "__main__":
    main()
