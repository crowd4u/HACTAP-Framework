import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner

from hactap.ai_worker import AIWorker
from hactap.dataset import Dataset
from hactap.utils import random_strategy

def build_dataset():
    task_size = 2000
    dataloader = DataLoader(
        MNIST('.', download=True, transform=ToTensor()),
        shuffle=True,
        batch_size=task_size
    )
    x_root, y_root = next(iter(dataloader))
    x_root = x_root.reshape(task_size, 28*28)
    x_train, y_train = x_root[:task_size], y_root[:task_size]
    dataset =  Dataset(x_train, y_train, [])

    # take the initial data
    initial_idx = np.random.choice(
        range(len(x_train)),
        size=1000,
        replace=False
    )
    dataset.assign_tasks_to_human(initial_idx)

    return dataset

def build_ai_worker(dataset):
    aiw = AIWorker(
        ActiveLearner(
            estimator=LogisticRegression(),
            X_training=dataset.x_train, y_training=dataset.y_train,
            query_strategy=random_strategy
        ),
    )

    return aiw