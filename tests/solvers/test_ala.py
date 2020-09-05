import unittest
from torchvision.datasets import FakeData
from torchvision import transforms
from sklearn.neural_network import MLPClassifier
from modAL.models import ActiveLearner

from hactap.solvers import AL
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker


class TestALA(unittest.TestCase):
    def test_run(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.reshape(28*28))
        ])
        dataset = FakeData(
            size=2000, num_classes=5,
            image_size=(1, 28, 28), transform=transform
        )
        dataset_index = range(len(dataset))
        tasks = Tasks(dataset, dataset_index)

        ai_worker = AIWorker(ActiveLearner(MLPClassifier(max_iter=500)))

        solver = AL(
            tasks,
            [ai_worker],
            0.9,
            5,
            500,
            None,
            None
        )

        self.assertIsInstance(solver.run(), Tasks)


if __name__ == '__main__':
    unittest.main()
