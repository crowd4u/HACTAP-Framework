import argparse
import warnings
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from modAL.models import ActiveLearner, Committee
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
from hactap.ai_worker import AIWorker, ComitteeAIWorker
from hactap.logging import get_logger
from hactap.reporter import Reporter
from hactap.human_crowd import get_labels_from_humans_by_random

warnings.simplefilter('ignore')
logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--solver',
    default='cta',
    choices=['ala_us', 'ala_qbc']
)
parser.add_argument('--task_size', default=10000, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)
parser.add_argument('--test_with_random', default=False, type=bool)


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
    al_ai_worker = [
        AIWorker(ActiveLearner(estimator=MLPClassifier())),
    ]

    al_ai_workers_comittee = [
        ComitteeAIWorker(
            Committee(
                learner_list=[
                    ActiveLearner(estimator=MLPClassifier()),
                    ActiveLearner(estimator=ExtraTreeClassifier()),
                    ActiveLearner(estimator=LogisticRegression()),
                    # ActiveLearner(estimator=KMeans()),
                    # -> can not use kmeans here
                    ActiveLearner(estimator=DecisionTreeClassifier()),
                    ActiveLearner(estimator=SVC(probability=True)),
                    # -> need this option to access probability
                    ActiveLearner(estimator=KNeighborsClassifier()),
                    ActiveLearner(estimator=GaussianProcessClassifier()),
                    ActiveLearner(estimator=MultinomialNB()),
                    ActiveLearner(estimator=AdaBoostClassifier()),
                    # ActiveLearner(estimator=PassiveAggressiveClassifier()),
                    # ActiveLearner(estimator=RidgeClassifier()),
                    # ActiveLearner(estimator=RidgeClassifierCV()),
                    # -> no predict_proba method
                    ActiveLearner(estimator=ComplementNB()),
                    # ActiveLearner(estimator=NearestCentroid())
                    # -> faced some errors
                ]
            )
        )
    ]

    # Start task assignment
    if args.solver == 'ala_us':
        solver = solvers.ALA(
            tasks,
            al_ai_worker,
            args.quality_requirements,
            10,
            args.human_crowd_batch_size,
            reporter=reporter,
            human_crowd=get_labels_from_humans_by_random,
            test_with_random=args.test_with_random
        )
    elif args.solver == 'ala_qbc':
        solver = solvers.ALA(
            tasks,
            al_ai_workers_comittee,
            args.quality_requirements,
            10,
            args.human_crowd_batch_size,
            reporter=reporter,
            human_crowd=get_labels_from_humans_by_random,
            test_with_random=args.test_with_random
        )

    solver.run()


if __name__ == "__main__":
    main()
