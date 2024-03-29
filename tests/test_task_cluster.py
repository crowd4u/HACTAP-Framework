import unittest
from sklearn.linear_model import LogisticRegression

from hactap.ai_worker import AIWorker
from hactap.task_cluster import TaskCluster
from hactap.solvers import CTA
from hactap.human_crowd import IdealHumanCrowd
from hactap.reporter import Reporter

from .testutils import build_dataset
from .testutils import build_ai_worker


class TestTaskCluster(unittest.TestCase):
    def test_create_task_cluster_for_humans(self):
        self.assertIsInstance(TaskCluster(AIWorker(LogisticRegression()), 0, info={}), TaskCluster)

    def test_create_task_cluster_for_the_ai(self):
        aiw = AIWorker(
            LogisticRegression()
        )
        self.assertIsInstance(TaskCluster(aiw, 0, {}), TaskCluster)

    def test_update_status_human(self):
        dataset = build_dataset()

        tc = TaskCluster(AIWorker(LogisticRegression()), 0, {
            "rule": {
                "from": 0,
                "to": 0
            },
            "stat": {
                "answerable_tasks_ids": [],
                "y_pred": []
            }
        })
        tc.update_status_human(dataset)

        self.assertEqual(tc.n_answerable_tasks, 1000)
        self.assertEqual(len(tc.bata_dist), 1)

    def test_update_status_ai(self):
        dataset = build_dataset()
        aiw = build_ai_worker(dataset)

        trainset = dataset.train_set
        aiw.fit(trainset)

        tc = TaskCluster(aiw, 0, {
            "rule": {
                "from": 0,
                "to": 0
            },
            "stat": {
                    "y_pred_test": [],
                    "y_pred_train": [],
                    "y_pred_remain": [],

                    "y_pred_test_human": [],
                    "y_pred_train_human": [],
                    "y_pred_remain_human": [],

                    "y_pred_test_ids": [],
                    "y_pred_train_ids": [],
                    "y_pred_remain_ids": []
            }
        })

        tc.update_status(dataset)

        self.assertIsInstance(tc.n_answerable_tasks, int)
        self.assertEqual(len(tc.bata_dist), 0)

    # def test___calc_assignable_tasks(self):
    #     dataset = build_dataset()
    #     aiw = build_ai_worker(dataset)

    #     trainset = dataset.train_set
    #     aiw.fit(trainset)

    #     test_set = dataset.test_set

    #     tc = TaskCluster(aiw, {
    #         "rule": {
    #             "from": 0,
    #             "to": 0
    #         }
    #     })
    #     _assigned_idx, _y_pred = tc._calc_assignable_tasks(
    #         test_set, np.array(range(len(dataset.test_indexes)))
    #     )
    #     self.assertEqual(len(_assigned_idx), len(_y_pred))

    def test_update_status(self):
        dataset = build_dataset()
        ai_worker = build_ai_worker(dataset)
        trainset = dataset.train_set
        ai_worker.fit(trainset)
        # test_set = dataset.test_set
        # test_indexes = dataset.test_indexes

        human_crowd = IdealHumanCrowd(
            0.9
        )

        solver = CTA(
            dataset,
            human_crowd,
            500,
            [ai_worker],
            0.9,
            10,
            0.05,
            Reporter()
        )

        # print(solver.list_task_clusters())

        tc_k = solver.list_task_clusters()[0]
        print(tc_k.rule)
        print(tc_k.match_rate_with_human, tc_k.conflict_rate_with_human)
        tc_k.update_status(dataset)
        print(tc_k.match_rate_with_human, tc_k.conflict_rate_with_human)

        # tc_k = TaskCluster(aiw, {
        #     "rule": {
        #         "from": 0,
        #         "to": 0
        #     },
        #     "stat": {
        #         "answerable_tasks_ids": [],
        #         "y_pred": []
        #     }
        # })

        # tc_k.update_status(dataset)
        # self.assertEqual(tc_k.match_rate_with_human, 1)


if __name__ == '__main__':
    unittest.main()
