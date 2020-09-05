import argparse
import warnings
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from modAL.models import ActiveLearner
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier

from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.logging import get_logger
from hactap.reporter import Reporter
from hactap.human_crowd import get_labels_from_humans_by_random

warnings.simplefilter('ignore')
logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--solver', default='al')
parser.add_argument('--task_size', default=10000, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)


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
        AIWorker(MLPClassifier(activation='logistic', solver='sgd')),
        AIWorker(LogisticRegression()),
        AIWorker(KMeans(n_clusters=10)),
        AIWorker(KMeans(n_clusters=20)),
        AIWorker(DecisionTreeClassifier()),
        AIWorker(SVC()),
        AIWorker(KNeighborsClassifier()),
        AIWorker(GaussianProcessClassifier()),
        AIWorker(MultinomialNB())
    ]

    al_ai_worker = [
        AIWorker(ActiveLearner(
            estimator=MLPClassifier()
        ))
    ]

    # Start task assignment
    if args.solver == 'al':
        solver = solvers.AL(
            tasks,
            al_ai_worker,
            args.quality_requirements,
            10,
            args.human_crowd_batch_size,
            reporter=reporter,
            human_crowd=get_labels_from_humans_by_random
        )
    elif args.solver == 'cta':
        solver = solvers.CTA(
            tasks,
            ai_workers,
            args.quality_requirements,
            10,
            args.human_crowd_batch_size,
            args.significance_level,
            reporter=reporter,
            human_crowd=get_labels_from_humans_by_random
        )
    elif args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            ai_workers,
            args.quality_requirements,
            10,
            args.human_crowd_batch_size,
            args.significance_level,
            reporter=reporter,
            human_crowd=get_labels_from_humans_by_random
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
            human_crowd=get_labels_from_humans_by_random
        )

    solver.run()


if __name__ == "__main__":
    main()
