import argparse
from math import ceil
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
from hactap.ai_worker import AIWorker
from hactap.logging import get_logger
from hactap.reporter import Reporter
from hactap.human_crowd import IdealHumanCrowd
from hactap.evaluate_ai_worker import EvalAIWByBinTest, EvalAIWByLearningCurve

warnings.simplefilter('ignore')
logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--task_size', default=10000, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
parser.add_argument('--human_crowd_correct_proba', default=1.0, type=float)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)
parser.add_argument('--n_monte_carlo_trial', default=100000, type=int)
parser.add_argument('--minimum_sample_size', default=-1, type=int)
parser.add_argument('--prior_distribution', nargs=2, default=[1, 1], type=int)
parser.add_argument(
    '--eval_ai_worker',
    default='bin_test',
    choices=['none', 'bin_test', 'learning_curve']
)
parser.add_argument('--ai_quality_requirements', default=0, type=float)


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
        AIWorker(ExtraTreeClassifier()),
        AIWorker(LogisticRegression()),
        AIWorker(KMeans()),
        AIWorker(DecisionTreeClassifier()),
        AIWorker(SVC()),
        # AIWorker(KNeighborsClassifier()),
        # AIWorker(GaussianProcessClassifier(n_jobs=-2)),
        # AIWorker(MultinomialNB()),
        # AIWorker(AdaBoostClassifier()),
        # AIWorker(PassiveAggressiveClassifier()),
        # AIWorker(RidgeClassifier()),
        # AIWorker(RidgeClassifierCV()),
        # AIWorker(ComplementNB()),
        # AIWorker(NearestCentroid())
    ]
    human_crowd = IdealHumanCrowd(
        args.human_crowd_correct_proba
    )

    if args.ai_quality_requirements == 0:
        ai_quality_req = args.quality_requirements - 0.05
    else:
        ai_quality_req = args.ai_quality_requirements

    if args.eval_ai_worker == "bin_test":
        EvalAIClass = EvalAIWByBinTest
        eval_ai_params = {
            "accuracy_requirement": ai_quality_req,
            "significance_level": args.significance_level
        }
    elif args.eval_ai_worker == "learning_curve":
        EvalAIClass = EvalAIWByLearningCurve
        eval_ai_params = {
            "accuracy_requirement": ai_quality_req,
            "max_iter_n": ceil(args.task_size / args.human_crowd_batch_size)
        }
    elif args.eval_ai_worker == "none":
        EvalAIClass = None
        eval_ai_params = {}

    # Start task assignment
    solver = solvers.GTALimit(
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
        prior_distribution=args.prior_distribution,
        EvaluateAIClass=EvalAIClass,
        evaluate_ai_class_params=eval_ai_params
    )

    solver.run()


if __name__ == "__main__":
    main()
