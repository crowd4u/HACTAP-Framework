import argparse
import warnings
from torch.utils.data import random_split
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from torchvision import transforms

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid # NOQA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB # NOQA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV # NOQA

from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.logging import get_logger
from hactap.reporter import Reporter
from hactap.human_crowd import IdealHumanCrowd

warnings.simplefilter('ignore')
logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    default='mnist',
    choices=['mnist', 'fmnist', 'kmnist']
)
parser.add_argument(
    '--solver',
    default='cta_mv',
    choices=['cta_mv', 'gta_mv', 'interactive_cta', 'interactive_gta']
)
parser.add_argument('--task_size', default=10000, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
parser.add_argument('--human_crowd_correct_proba', default=1.0, type=float)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)
parser.add_argument('--n_of_majority_vote', default=1, type=int)
parser.add_argument(
    '--interaction_strategy',
    default='conflict',
    choices=['match', 'conflict', 'random'],
    type=str
)
parser.add_argument(
    '--epsilon_handler',
    default='static',
    choices=['static', 'ntasks']
)
parser.add_argument(
    '--epsilon_handler_static',
    default=0.5,
    type=float
)


def epsilon_handler_static(thre):
    def epsilon_handler(tasks):
        return thre
    return epsilon_handler


def epsilon_handler_ntasks():
    def epsilon_handler(tasks):
        return 1.0 - len(tasks.all_labeled_indexes) / len(tasks.raw_y_human)

    return epsilon_handler


def main():
    args = parser.parse_args()
    reporter = Reporter(args)

    # parepare the tasks
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.reshape(28*28))
    ])

    if args.dataset == 'fmnist':
        dataset = FashionMNIST('.', download=True, transform=transform)
    elif args.dataset == 'kmnist':
        dataset = KMNIST('.', download=True, transform=transform)
    else:
        dataset = MNIST('.', download=True, transform=transform)

    dataset = random_split(
        dataset,
        [args.task_size, len(dataset) - args.task_size]
    )[0]
    data_index = range(len(dataset))
    tasks = Tasks(dataset, data_index)

    ai_workers = [
        AIWorker(MLPClassifier()),
        AIWorker(ExtraTreeClassifier()),
        AIWorker(LogisticRegression()),
        AIWorker(KMeans()),
        AIWorker(DecisionTreeClassifier()),
        AIWorker(SVC()),
        AIWorker(KNeighborsClassifier()),
        AIWorker(MultinomialNB()),
        AIWorker(AdaBoostClassifier()),
        AIWorker(PassiveAggressiveClassifier()),
        AIWorker(RidgeClassifier()),
        AIWorker(RidgeClassifierCV()),
        AIWorker(ComplementNB()),
        AIWorker(NearestCentroid())
    ]

    human_crowd = IdealHumanCrowd(
        'random',
        args.human_crowd_batch_size,
        args.human_crowd_correct_proba
    )

    if args.epsilon_handler == 'ntasks':
        epsilon_handler = epsilon_handler_ntasks()
    else:
        epsilon_handler = epsilon_handler_static(args.epsilon_handler_static)

    # Start task assignment
    args_base = (
        tasks,
        human_crowd,
        ai_workers,
        args.quality_requirements,
        10,
        args.significance_level,
    )

    if args.solver == 'cta_mv':
        solver = solvers.CTA(
            *args_base,
            reporter=reporter,
            n_of_majority_vote=args.n_of_majority_vote,
        )
    elif args.solver == 'gta_mv':
        solver = solvers.GTA(
            *args_base,
            reporter=reporter,
            n_of_majority_vote=args.n_of_majority_vote,
        )
    elif args.solver == 'interactive_cta':
        solver = solvers.InteractiveCTA(
            *args_base,
            reporter=reporter,
            n_of_majority_vote=args.n_of_majority_vote,
            interaction_strategy=args.interaction_strategy,
            epsilon_handler=epsilon_handler
        )
    elif args.solver == 'interactive_gta':
        solver = solvers.InteractiveGTA(
            *args_base,
            reporter=reporter,
            n_of_majority_vote=args.n_of_majority_vote,
            interaction_strategy=args.interaction_strategy
        )

    solver.run()


if __name__ == "__main__":
    main()
