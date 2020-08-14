import unittest
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from hactap.dataset import Dataset

class TestDataset(unittest.TestCase):

    def test_dataset_create(self):
        task_size = 100
        dataloader = DataLoader(
            MNIST('.', download=True, transform=ToTensor()),
            shuffle=True,
            batch_size=task_size
        )
        x_root, y_root = next(iter(dataloader))
        x_root = x_root.reshape(task_size, 28*28)
        x_train, y_train = x_root[:task_size], y_root[:task_size]
        dataset = Dataset(x_train, y_train, [])

        self.assertIsInstance(dataset, Dataset)

if __name__ == '__main__':
    unittest.main()
