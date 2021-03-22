import unittest
from torchvision.datasets import FakeData
from torchvision import transforms
from sklearn.neural_network import MLPClassifier

from hactap.solvers import Baseline
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.reporter import Reporter
from hactap.human_crowd import IdealHumanCrowd


class TestBaseline(unittest.TestCase):
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

        ai_worker = AIWorker(MLPClassifier(max_iter=500))

        human_crowd = IdealHumanCrowd(
            0.9
        )

        solver = Baseline(
            tasks,
            human_crowd,
            500,
            [ai_worker],
            0.9,
            5,
            Reporter()
        )

        self.assertIsInstance(solver.run(), Tasks)


if __name__ == '__main__':
    unittest.main()
