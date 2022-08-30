import argparse
import warnings
from torch.utils.data import random_split
from torchvision.datasets import MNIST
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
from hactap.ai_worker import AIWorker, ProbaAIWorker, ProbaMedianAIWorker, AIWorkerWithFeedback
from hactap.logging import get_logger
from hactap.reporter import Reporter
from hactap.human_crowd import IdealHumanCrowd

warnings.simplefilter('ignore')
logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--solver',
    default='gta',
    choices=['gta', 'gta_retire', 'gta_onetime', 'gta_fb']
)
parser.add_argument(
    '--ai_worker_type',
    default='default',
    choices=['default', 'proba', 'proba_median']
)
parser.add_argument('--task_size', default=10000, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_correct_proba', default=1.0, type=float)
parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)
parser.add_argument('--n_monte_carlo_trial', default=100000, type=int)
parser.add_argument('--minimum_sample_size', default=-1, type=int)
parser.add_argument('--prior_distribution', nargs=2, default=[1, 1], type=int)
parser.add_argument('--ai_worker_proba_threshold', default=0.7, type=float)


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
    if args.solver == "gta_fb":
        threshold = args.ai_worker_proba_threshold
        ai_workers = [
            AIWorkerWithFeedback(MLPClassifier(), threshold),
            AIWorkerWithFeedback(LogisticRegression(), threshold),
            AIWorkerWithFeedback(SVC(probability=True), threshold),
            AIWorkerWithFeedback(KNeighborsClassifier(), threshold),
            AIWorkerWithFeedback(GaussianProcessClassifier(n_jobs=-2), threshold),
            AIWorkerWithFeedback(MultinomialNB(), threshold),
            AIWorkerWithFeedback(AdaBoostClassifier(), threshold),
            AIWorkerWithFeedback(ComplementNB(), threshold)
        ]
    elif args.ai_worker_type == 'default':
        ai_workers = [
            AIWorker(MLPClassifier()),
            AIWorker(ExtraTreeClassifier()),
            AIWorker(LogisticRegression()),
            AIWorker(KMeans()),
            AIWorker(DecisionTreeClassifier()),
            AIWorker(SVC()),
            AIWorker(KNeighborsClassifier()),
            AIWorker(GaussianProcessClassifier(n_jobs=-2)),
            AIWorker(MultinomialNB()),
            AIWorker(AdaBoostClassifier()),
            AIWorker(PassiveAggressiveClassifier()),
            AIWorker(RidgeClassifier()),
            AIWorker(RidgeClassifierCV()),
            AIWorker(ComplementNB()),
            AIWorker(NearestCentroid())
        ]
    elif args.ai_worker_type == 'proba':
        threshold = args.ai_worker_proba_threshold
        ai_workers = [
            ProbaAIWorker(MLPClassifier(), threshold),
            ProbaAIWorker(LogisticRegression(), threshold),
            ProbaAIWorker(SVC(probability=True), threshold),
            ProbaAIWorker(KNeighborsClassifier(), threshold),
            ProbaAIWorker(GaussianProcessClassifier(n_jobs=-2), threshold),
            ProbaAIWorker(MultinomialNB(), threshold),
            ProbaAIWorker(AdaBoostClassifier(), threshold),
            ProbaAIWorker(ComplementNB(), threshold)
        ]
    elif args.ai_worker_type == 'proba_median':
        threshold = args.ai_worker_proba_threshold
        ai_workers = [
            ProbaMedianAIWorker(MLPClassifier(), threshold),
            ProbaMedianAIWorker(LogisticRegression(), threshold),
            ProbaMedianAIWorker(SVC(probability=True), threshold),
            ProbaMedianAIWorker(KNeighborsClassifier(), threshold),
            ProbaMedianAIWorker(GaussianProcessClassifier(n_jobs=-2), threshold),
            ProbaMedianAIWorker(MultinomialNB(), threshold),
            ProbaMedianAIWorker(AdaBoostClassifier(), threshold),
            ProbaMedianAIWorker(ComplementNB(), threshold)
        ]

    human_crowd = IdealHumanCrowd(
        args.human_crowd_correct_proba
    )

    # Start task assignment
    if args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter,
            retire_used_test_data=False,
            n_monte_carlo_trial=args.n_monte_carlo_trial,
            minimum_sample_size=args.minimum_sample_size,
            prior_distribution=args.prior_distribution
        )
    elif args.solver == 'gta_retire':
        solver = solvers.GTA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter,
            retire_used_test_data=True,
            n_monte_carlo_trial=args.n_monte_carlo_trial,
            minimum_sample_size=args.minimum_sample_size,
            prior_distribution=args.prior_distribution
        )
    elif args.solver == 'gta_onetime':
        solver = solvers.GTAOneTime(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter,
            n_monte_carlo_trial=args.n_monte_carlo_trial,
            minimum_sample_size=args.minimum_sample_size
        )
    elif args.solver == 'gta_fb':
        solver = solvers.GTA_FB(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter,
            n_monte_carlo_trial=args.n_monte_carlo_trial,
            minimum_sample_size=args.minimum_sample_size
        )

    solver.run()


if __name__ == "__main__":
    main()
