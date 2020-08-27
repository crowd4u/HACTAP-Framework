import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch.utils.data import random_split
from torchvision import transforms

from hactap.ai_worker import AIWorker
# from hactap.dataset import Dataset
from hactap.tasks import Tasks
from hactap.human_crowd import get_labels_from_humans_by_random


def build_dataset():
    # task_size = 2000
    # dataloader = DataLoader(
    #     MNIST('.', download=True, transform=ToTensor()),
    #     shuffle=True,
    #     batch_size=task_size
    # )
    # x_root, y_root = next(iter(dataloader))
    # x_root = x_root.reshape(task_size, 28*28)
    # x_train, y_train = x_root[:task_size], y_root[:task_size]
    # tasks = Tasks(x_train, y_train)

    # # take the initial data
    # initial_idx = np.random.choice(
    #     range(len(x_train)),
    #     size=1000,
    #     replace=False
    # )
    # # dataset.assign_tasks_to_human(initial_idx)

    # initial_labels = tasks.get_ground_truth(initial_idx)

    # tasks.bulk_update_labels_by_human(initial_idx, initial_labels)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.reshape(28*28))
    ])
    mnist_dataset = MNIST('.', download=True, transform=transform)
    mnist_dataset = random_split(
        mnist_dataset,
        [2000, len(mnist_dataset) - 2000]
    )[0]
    data_index = range(len(mnist_dataset))
    tasks = Tasks(mnist_dataset, data_index)

    get_labels_from_humans_by_random(tasks, 1000)

    return tasks


def build_ai_worker(tasks):
    return AIWorker(MLPClassifier())
