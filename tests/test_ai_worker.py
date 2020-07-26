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

class TestAIWorker(unittest.TestCase):

    def test_ai_worker_create(self):
        aiw = AIWorker(
            LogisticRegression()
        )

        self.assertIsInstance(aiw, AIWorker)

if __name__ == '__main__':
    unittest.main()
