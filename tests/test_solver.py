import unittest
from sklearn.linear_model import LogisticRegression
import torch
import collections
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
from modAL.models import ActiveLearner

from hactap.dataset import Dataset
from hactap.ai_worker import AIWorker
from hactap.utils import random_strategy
from hactap.task_cluster import TaskCluster
from hactap.solver import Solver

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

class TestSolver(unittest.TestCase):

    def test_build_task_clusters(self):
        dataset = build_dataset()
        ai_worker = build_ai_worker(dataset)

        solver = Solver(
            dataset,
            [ai_worker],
            0.9,
        )

        task_clusters = solver.list_task_clusters()

        self.assertIsInstance(task_clusters, list)
        self.assertEqual(len(task_clusters), 10)
        self.assertIsInstance(task_clusters[0], TaskCluster)

if __name__ == '__main__':
    unittest.main()
