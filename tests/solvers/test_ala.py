import unittest

from torchvision.datasets import FakeData
from torchvision import transforms
from sklearn.neural_network import MLPClassifier
from modAL.models import Committee, ActiveLearner

from hactap.solvers import ALA
from hactap.tasks import Tasks
from hactap.ai_worker import ComitteeAIWorker
from hactap.human_crowd import IdealHumanCrowd
from hactap.reporter import Reporter


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

        ai_worker = ComitteeAIWorker(
            Committee(
                learner_list=[
                    ActiveLearner(MLPClassifier(max_iter=500))
                ]
            )
        )

        human_crowd = IdealHumanCrowd(
            'random',
            500,
            0.9
        )

        solver = ALA(
            tasks,
            human_crowd,
            [ai_worker],
            0.9,
            5,
            Reporter()
        )

        self.assertIsInstance(solver.run(), Tasks)


if __name__ == '__main__':
    unittest.main()
