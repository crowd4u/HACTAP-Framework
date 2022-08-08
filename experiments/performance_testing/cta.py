import argparse
import warnings
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker, ProbaAIWorker
from hactap.logging import get_logger
from hactap.reporter import Reporter
from hactap.human_crowd import IdealHumanCrowd

warnings.simplefilter('ignore')
logger = get_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--solver',
    default='cta',
    choices=['cta', 'cta_retire', 'cta_proba']
)
parser.add_argument('--task_size', default=10000, type=int)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=2000, type=int)
parser.add_argument('--human_crowd_correct_proba', default=1.0, type=float)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)
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
    ai_workers = [
        AIWorker(MLPClassifier()),
        AIWorker(ExtraTreeClassifier()),
        AIWorker(LogisticRegression()),
        AIWorker(KMeans()),
        AIWorker(DecisionTreeClassifier())
    ]

    proba_ai_workers = [
        ProbaAIWorker(MLPClassifier()),
        ProbaAIWorker(ExtraTreeClassifier()),
        ProbaAIWorker(LogisticRegression())
    ]

    human_crowd = IdealHumanCrowd(
        args.human_crowd_correct_proba
    )

    # Start task assignment
    if args.solver == 'cta':
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
    elif args.solver == 'cta_retire':
        solver = solvers.CTA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter,
            retire_used_test_data=True
        )
    elif args.solver == 'cta_proba':
        solver = solvers.CTA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            proba_ai_workers,
            args.quality_requirements,
            10,
            args.significance_level,
            reporter=reporter,
            ai_worker_proba_threshold=args.ai_worker_proba_threshold
        )

    solver.run()


if __name__ == "__main__":
    main()
