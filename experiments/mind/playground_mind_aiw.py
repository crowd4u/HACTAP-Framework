import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # NOQA

from hactap.ai_worker import AIWorker
from hactap.human_crowd import get_labels_from_humans_by_random
from hactap.utils import ImageFolderWithPaths
from hactap.tasks import Tasks

from mind_ai_worker import MindAIWorker

DATASET_PATH = './dataset/mind_10_amt'
height = 122
width = 110


def main():
    # prepare dataset
    mind_dataset = ImageFolderWithPaths(
        DATASET_PATH,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((height, width)),
            ToTensor()
        ])
    )

    data_index = range(len(mind_dataset))
    human_labelable_index = []
    image_paths = []

    for index in range(len(mind_dataset)):
        path, label = mind_dataset.get_label(index)
        image_paths.append(path)
        if label != 3:
            human_labelable_index.append(index)

    print('human_labelable_index', len(human_labelable_index))
    tasks = Tasks(mind_dataset, data_index, human_labelable_index)
    get_labels_from_humans_by_random(tasks, 5000)
    # get_labels_from_humans(tasks, 10000)

    # init ai worker
    ai_worekr = AIWorker(MindAIWorker())

    # train
    train_set = tasks.train_set
    ai_worekr.fit(train_set)

    # test
    test_set = tasks.test_set
    test_data = DataLoader(test_set, batch_size=len(test_set))
    x_test, y_test = next(iter(test_data))
    y_pred = ai_worekr.predict(x_test)

    # output summary
    print('accuracy', accuracy_score(y_test, y_pred))
    print('confusion_matrix', confusion_matrix(y_test, y_pred))
    print('classification_report', classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
