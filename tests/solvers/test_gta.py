import unittest
from torchvision.datasets import FakeData
from torchvision import transforms
from sklearn.linear_model import LogisticRegression

from hactap.solvers import GTA
from hactap.tasks import Tasks
from hactap.ai_worker import AIWorker
from hactap.reporter import Reporter
from hactap.human_crowd import IdealHumanCrowd


class TestGTA(unittest.TestCase):
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

        ai_worker = AIWorker(LogisticRegression(max_iter=500))

        human_crowd = IdealHumanCrowd(
            'random',
            500,
            0.9
        )

        solver = GTA(
            tasks,
            human_crowd,
            [ai_worker],
            0.9,
            5,
            0.05,
            Reporter(),
            None
        )

        self.assertIsInstance(solver.run(), Tasks)


if __name__ == '__main__':
    unittest.main()
