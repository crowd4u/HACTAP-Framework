from torchvision.datasets import MNIST
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch.utils.data import random_split
from torchvision import transforms

from hactap.ai_worker import AIWorker
from hactap.tasks import Tasks
from hactap.human_crowd import get_labels_from_humans_by_random


def build_dataset():
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
