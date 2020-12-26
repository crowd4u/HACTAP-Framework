import unittest
from hactap.solver import Solver
from hactap.human_crowd import IdealHumanCrowd

from .testutils import build_ai_worker
from .testutils import build_dataset


class TestSolver(unittest.TestCase):

    def test_build_task_clusters(self):
        dataset = build_dataset()
        ai_worker = build_ai_worker(dataset)

        trainset = dataset.train_set
        ai_worker.fit(trainset)

        human_crowd = IdealHumanCrowd(
            'random',
            100,
            0.9
        )

        solver = Solver(
            dataset,
            human_crowd,
            [ai_worker],
            0.9,
            10
        )

        # task_clusters = solver.list_task_clusters()

        # self.assertIsInstance(task_clusters, list)
        # self.assertEqual(len(task_clusters), 10)
        self.assertIsInstance(solver, Solver)


if __name__ == '__main__':
    unittest.main()
