import unittest
from sklearn.linear_model import LogisticRegression

from hactap.ai_worker import AIWorker
from hactap.task_cluster import TaskCluster

class TestTaskCluster(unittest.TestCase):
    def test_create_task_cluster_for_humans(self):
        self.assertIsInstance(TaskCluster(0, 0), TaskCluster)

    def test_create_task_cluster_for_the_ai(self):
        aiw = AIWorker(
            LogisticRegression()
        )
        self.assertIsInstance(TaskCluster(aiw, []), TaskCluster)


if __name__ == '__main__':
    unittest.main()
