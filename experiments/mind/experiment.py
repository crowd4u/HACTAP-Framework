import torch
import pandas as pd
import argparse
import torchvision
from torchvision.transforms import ToTensor
from modAL.models import ActiveLearner, Committee
import torchvision.models as models
from skorch import NeuralNetClassifier
from math import ceil

from hactap import solvers
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker, ComitteeAIWorker, ProbaAIWorker
from hactap.utils import ImageFolderWithPaths
from hactap.reporter import Reporter, EvalAIReporter
# from hactap.human_crowd import get_labels_from_humans_by_original_order
# from hactap.human_crowd import get_labels_from_humans_by_random
from hactap.human_crowd import IdealHumanCrowd
from hactap.evaluate_ai_worker import EvalAIWByBinTest, EvalAIWByLearningCurve

from mind_ai_worker import MindAIWorker


DATASET_PATH = './dataset'
height = 122
width = 110

parser = argparse.ArgumentParser()
parser.add_argument('--solver', default='gta', choices=['gta', 'ala', 'gta_limit'])
parser.add_argument(
    '--ai_worker_type',
    default='default',
    choices=['default', 'proba', 'mix']
)
parser.add_argument('--quality_requirements', default=0.8, type=float)
parser.add_argument('--human_crowd_batch_size', default=200, type=int)
parser.add_argument(
    '--human_crowd_mode',
    default='order',
    choices=['random', 'order']
)
parser.add_argument('--group_id', default='default')
parser.add_argument('--trial_id', default=1, type=int)
parser.add_argument('--significance_level', default=0.05, type=float)
parser.add_argument(
    '--dataset',
    default='mind-10',
    choices=['mind-106', 'mind-10', 'mind-106-amt', 'mind-10-amt']
)
parser.add_argument('--ai_worker_proba_threshold', default=0.7, type=float)
parser.add_argument(
    '--eval_ai_worker',
    default='bin_test',
    choices=['none', 'bin_test', 'learning_curve']
)
parser.add_argument('--ai_quality_requirements', default=0, type=float)


def main():
    args = parser.parse_args()
    reporter = Reporter(args)

    # Prepare task
    if args.dataset == 'mind-106':
        dataset_path = './dataset/mind_106'
        label_order_path = './dataset/mind_106-label_order.csv'
    elif args.dataset == 'mind-106-amt':
        dataset_path = './dataset/mind_106_amt'
        label_order_path = './dataset/mind_106_amt-label_order.csv'
    elif args.dataset == 'mind-10-amt':
        dataset_path = './dataset/mind_10_amt'
        label_order_path = './dataset/mind_10_amt-label_order.csv'
    else:
        dataset_path = './dataset/mind_10'
        label_order_path = './dataset/mind_10-label_order.csv'

    mind_dataset = ImageFolderWithPaths(
        dataset_path,
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
    label_order = pd.read_csv(label_order_path)

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
    if args.ai_worker_type == 'default':
        ai_workers = [
            AIWorker(NeuralNetClassifier(
                MindAIWorker,
                device=device,
                train_split=None,
                max_epochs=50,
                optimizer=torch.optim.SGD,
            )),
            AIWorker(NeuralNetClassifier(
                models.resnet18(),
                device=device,
                train_split=None
            )),
            AIWorker(NeuralNetClassifier(
                models.mobilenet_v2(),
                device=device,
                train_split=None
            ))
        ]
    elif args.ai_worker_type == 'proba':
        threshold = args.ai_worker_proba_threshold
        ai_workers = [
            ProbaAIWorker(
                NeuralNetClassifier(
                    MindAIWorker,
                    device=device,
                    train_split=None,
                    max_epochs=50,
                    optimizer=torch.optim.SGD,
                ),
                threshold
            ),
            ProbaAIWorker(
                NeuralNetClassifier(
                    models.resnet18(),
                    device=device,
                    train_split=None
                ),
                threshold
            ),
            ProbaAIWorker(
                NeuralNetClassifier(
                    models.mobilenet_v2(),
                    device=device,
                    train_split=None
                ),
                threshold
            )
        ]
    elif args.ai_worker_type == 'mix':
        threshold = args.ai_worker_proba_threshold
        ai_workers = [
            ProbaAIWorker(
                NeuralNetClassifier(
                    MindAIWorker,
                    device=device,
                    train_split=None,
                    max_epochs=50,
                    optimizer=torch.optim.SGD,
                ),
                threshold
            ),
            ProbaAIWorker(
                NeuralNetClassifier(
                    models.resnet18(),
                    device=device,
                    train_split=None
                ),
                threshold
            ),
            ProbaAIWorker(
                NeuralNetClassifier(
                    models.mobilenet_v2(),
                    device=device,
                    train_split=None
                ),
                threshold
            ),
            AIWorker(NeuralNetClassifier(
                MindAIWorker,
                device=device,
                train_split=None,
                max_epochs=50,
                optimizer=torch.optim.SGD,
            )),
            AIWorker(NeuralNetClassifier(
                models.resnet18(),
                device=device,
                train_split=None
            )),
            AIWorker(NeuralNetClassifier(
                models.mobilenet_v2(),
                device=device,
                train_split=None
            ))
        ]

    al_ai_workers_comittee = [
        ComitteeAIWorker(
            Committee(
                learner_list=[
                    ActiveLearner(estimator=NeuralNetClassifier(
                        MindAIWorker,
                        device=device,
                        train_split=None,
                        max_epochs=50,
                        optimizer=torch.optim.SGD,
                    )),
                    # ActiveLearner(estimator=NeuralNetClassifier(
                    #     models.resnet18(),
                    #     device=device,
                    #     train_split=None
                    # )),
                    # ActiveLearner(estimator=NeuralNetClassifier(
                    #     models.mobilenet_v2(),
                    #     device=device,
                    #     train_split=None
                    # ))
                ]
            )
        )
    ]

    human_crowd = IdealHumanCrowd(
        1.0
    )

    # if args.human_crowd_mode == 'order':
    #     human_crowd = get_labels_from_humans_by_original_order
    # else:
    #     human_crowd = get_labels_from_humans_by_random

    if args.solver == 'ala':
        solver = solvers.ALA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            al_ai_workers_comittee,
            args.quality_requirements,
            3,
            reporter=reporter,
        )
    elif args.solver == 'gta':
        solver = solvers.GTA(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            3,
            args.significance_level,
            reporter=reporter,
        )
    elif args.solver == 'gta_limit':
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
                "max_iter_n": ceil(args.task_size / args.human_crowd_batch_size),
                "significance_level": args.significance_level
            }
        elif args.eval_ai_worker == "none":
            EvalAIClass = None
            eval_ai_params = {}

        aiw_reporter = EvalAIReporter(args)

        solver = solvers.GTALimit(
            tasks,
            human_crowd,
            args.human_crowd_batch_size,
            ai_workers,
            args.quality_requirements,
            3,
            args.significance_level,
            reporter=reporter,
            EvaluateAIClass=EvalAIClass,
            evaluate_ai_class_params=eval_ai_params,
            aiw_reporter=aiw_reporter
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
