import torch
import pandas as pd
import argparse
import torchvision
from torchvision.transforms import ToTensor
from modAL.models import ActiveLearner
import torchvision.models as models
from skorch import NeuralNetClassifier

from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.utils import ImageFolderWithPaths
from hactap.reporter import Reporter
from hactap.human_crowd import get_labels_from_humans_by_original_order
from hactap.human_crowd import get_labels_from_humans_by_random

from mind_ai_worker import MindAIWorker


DATASET_PATH = './dataset'
height = 122
width = 110

parser = argparse.ArgumentParser()
parser.add_argument('--solver', default='gta')
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=200, type=int)
parser.add_argument('--human_crowd_mode', default='random')
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)


def main():
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

    print("dataset size", len(mind_dataset))
    data_index = range(len(mind_dataset))
    human_labelable_index = []
    human_labelable_timestamp = []
    image_paths = []
    label_order = pd.read_csv('label_order.csv')

    for index in range(len(mind_dataset)):
        path, label = mind_dataset.get_label(index)
        image_paths.append(path)
        if label != 3:
            human_labelable_index.append(index)
            human_labelable_timestamp.append(
                int(label_order[label_order['path'] == path]['created_at'])
            )

    # print(human_labelable_timestamp)
    human_labelable_timestamp, human_labelable_index = zip(*sorted(
        zip(human_labelable_timestamp, human_labelable_index)
    ))

    print('human_labelable_index', len(human_labelable_index))
    tasks = Tasks(mind_dataset, data_index, human_labelable_index)
    # get_labels_from_humans_by_random(tasks, args.human_crowd_batch_size)

    # Prepare AI workers
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ai_workers = [
        AIWorker(MindAIWorker()),
        AIWorker(NeuralNetClassifier(
            models.resnet18(),
            device=device,
            train_split=None
        ))
    ]

    al_ai_workers = [
        AIWorker(ActiveLearner(
            estimator=MindAIWorker()
        ))
    ]

    if args.human_crowd_mode == 'order':
        human_crowd = get_labels_from_humans_by_original_order
    else:
        human_crowd = get_labels_from_humans_by_random

    if args.solver == 'al':
        solver = solvers.AL(
            tasks,
            al_ai_workers,
            args.quality_requirements,
            args.human_crowd_batch_size,
            reporter=reporter,
            human_crowd=human_crowd
        )
    elif args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            ai_workers,
            args.quality_requirements,
            args.human_crowd_batch_size,
            args.significance_level,
            reporter=reporter,
            human_crowd=human_crowd
        )

    output = solver.run()

    output_df = pd.DataFrame(
        {
            'index': image_paths,
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
