import argparse
import warnings
import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from torchvision import transforms
from modAL.models import ActiveLearner, Committee
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
from hactap.ai_worker import AIWorker, ComitteeAIWorker
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
    default='cta',
    choices=['baseline', 'ala', 'cta', 'gta', 'ic_cta']
)
parser.add_argument('--task_size', default=10000, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
parser.add_argument('--human_crowd_correct_proba', default=1.0, type=float)
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

    # Build AI workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ai_workers = [
        AIWorker(MLPClassifier(device=device)),
        AIWorker(ExtraTreeClassifier(device=device)),
        AIWorker(LogisticRegression(device=device)),
        AIWorker(KMeans(device=device)),
        AIWorker(DecisionTreeClassifier(device=device)),
        AIWorker(SVC(device=device)),
        AIWorker(KNeighborsClassifier(device=device)),
        AIWorker(GaussianProcessClassifier(n_jobs=-2, device=device)),
        AIWorker(MultinomialNB(device=device)),
        AIWorker(AdaBoostClassifier(device=device)),
        AIWorker(PassiveAggressiveClassifier(device=device)),
        AIWorker(RidgeClassifier(device=device)),
        AIWorker(RidgeClassifierCV(device=device)),
        AIWorker(ComplementNB(device=device)),
        AIWorker(NearestCentroid(device=device))
    ]

    al_ai_workers_comittee = [
        ComitteeAIWorker(
            Committee(
                learner_list=[
                    ActiveLearner(estimator=MLPClassifier(device=device)),
                    ActiveLearner(estimator=ExtraTreeClassifier(device=device)),
                    ActiveLearner(estimator=LogisticRegression(device=device)),
                    # ActiveLearner(estimator=KMeans(device=device)),
                    # -> can not use kmeans here
                    ActiveLearner(estimator=DecisionTreeClassifier(device=device)), # NOQA
                    ActiveLearner(estimator=SVC(probability=True, device=device)), # NOQA
                    # -> need this option to access probability
                    ActiveLearner(estimator=KNeighborsClassifier(device=device)), # NOQA
                    ActiveLearner(estimator=GaussianProcessClassifier(n_jobs=-2, device=device)), # NOQA
                    ActiveLearner(estimator=MultinomialNB(device=device)),
                    ActiveLearner(estimator=AdaBoostClassifier(device=device)),
                    # ActiveLearner(estimator=PassiveAggressiveClassifier(device=device)),
                    # ActiveLearner(estimator=RidgeClassifier(device=device)),
                    # ActiveLearner(estimator=RidgeClassifierCV(device=device)),
                    # -> no predict_proba method
                    ActiveLearner(estimator=ComplementNB(device=device)),
                    # ActiveLearner(estimator=NearestCentroid(device=device))
                    # -> faced some errors
                ]
            )
        )
    ]

    human_crowd = IdealHumanCrowd(
        args.human_crowd_correct_proba
    )

    # Start task assignment
    if args.solver == 'baseline':
        solver = solvers.Baseline(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            al_ai_workers_comittee,
            args.quality_requirements,
            10,
            reporter=reporter
        )
    elif args.solver == 'ala':
        solver = solvers.ALA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            al_ai_workers_comittee,
            args.quality_requirements,
            10,
            reporter=reporter
        )
    elif args.solver == 'cta':
        solver = solvers.CTA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter,
        )
    elif args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter
        )
    elif args.solver == "ic_cta":
        kmeans = KMeans(n_clusters=4)
        solver = solvers.IntersectionalClusterCTA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter,
            clustering_function=kmeans.fit_predict
        )

    solver.run()


if __name__ == "__main__":
    main()
