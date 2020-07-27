import unittest
from sklearn.linear_model import LogisticRegression

from hactap.ai_worker import AIWorker
from hactap.task_cluster import TaskCluster

from .testutils import build_dataset
from .testutils import build_ai_worker

class TestTaskCluster(unittest.TestCase):
    def test_create_task_cluster_for_humans(self):
        self.assertIsInstance(TaskCluster(0, 0), TaskCluster)

    def test_create_task_cluster_for_the_ai(self):
        aiw = AIWorker(
            LogisticRegression()
        )
        self.assertIsInstance(TaskCluster(aiw, []), TaskCluster)

    def test_update_status_human(self):
        dataset = build_dataset()

        tc = TaskCluster(None, None)
        tc.update_status_human(dataset)

        self.assertEqual(tc.n_answerable_tasks, 0)
        self.assertEqual(len(tc.bata_dist), 100_000)

    def test_update_status_ai(self):
        dataset = build_dataset()
        aiw = build_ai_worker(dataset)
        
        tc = TaskCluster(aiw, {
            "rule": {
                "from": 0,
                "to": 0
            }
        })

        tc.update_status(dataset)

        self.assertIsInstance(tc.n_answerable_tasks, int)
        self.assertEqual(len(tc.bata_dist), 100_000)

    def test___calc_assignable_tasks(self):
        dataset = build_dataset()
        aiw = build_ai_worker(dataset)
        
        tc = TaskCluster(aiw, {
            "rule": {
                "from": 0,
                "to": 0
            }
        })
        _assigned_idx, _y_pred = tc._calc_assignable_tasks(dataset.x_test)
        self.assertEqual(len(_assigned_idx), len(_y_pred))



if __name__ == '__main__':
    unittest.main()
