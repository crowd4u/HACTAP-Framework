import unittest
from hactap.task_cluster import TaskCluster
from hactap.solver import Solver

from .testutils import build_ai_worker
from .testutils import build_dataset


class TestSolver(unittest.TestCase):

    def test_build_task_clusters(self):
        dataset = build_dataset()
        ai_worker = build_ai_worker(dataset)

        trainset = dataset.train_set
        ai_worker.fit(trainset)

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
