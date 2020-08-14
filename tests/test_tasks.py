import torch
import unittest
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from hactap.tasks import Tasks

class TestTasks(unittest.TestCase):

    def test_Tasks(self):
        dataset = Tasks([], [])
        self.assertIsInstance(dataset, Tasks)

    def test_tasks_of_mnist(self):
        task_size = 100
        dataloader = DataLoader(
            MNIST('.', download=True, transform=ToTensor()),
            shuffle=True,
            batch_size=task_size
        )
        x_root, y_root = next(iter(dataloader))
        x_root = x_root.reshape(task_size, 28*28)
        x_train, y_train = x_root[:task_size], y_root[:task_size]

        tasks = Tasks(x_train, y_train)
        self.assertIsInstance(tasks, Tasks)

    # def test_the_length_of_getters(self):
    #     X = [1, 2, 3, 4, 5]
    #     y = [1, 0, 1, 0, 6]
    #     dataset = Tasks(X, y)
    #     self.assertEqual(dataset.X, X)
    #     self.assertEqual(dataset.y, y)
    #     self.assertEqual(dataset.y_human, [None] * len(y))
    #     self.assertEqual(dataset.y_ai, [None] * len(y))

    def test_len(self):
        X = [1, 2, 3, 4, 5]
        y = [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)
        self.assertEqual(len(tasks), 0)

        tasks.update_label_by_human(0, 3)
        self.assertEqual(len(tasks), 1)

    def test_getitem(self):
        X = [1, 2, 3, 4, 5]
        y = [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.update_label_by_human(0, 3)
        self.assertEqual(tasks[0], (1, 3))


    def test_update_label_by_human(self):
        X = [1, 2, 3, 4, 5]
        y = [1, 0, 1, 0, 6]
        dataset = Tasks(X, y)
        self.assertEqual(dataset.update_label_by_human(0, 3), 3)

    def test_bulk_update_labels_by_human(self):
        X = [1, 2, 3, 4, 5]
        y = [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)
        tasks.bulk_update_labels_by_human([1, 2, 3, 4], [0, 1, 0, 1])
        self.assertEqual(tasks.y_human_labeled, [0, 1, 0, 1])
        # self.assertEqual(dataset[0][1], 3)

    def test_update_label_by_ai(self):
        X = [1, 2, 3, 4, 5]
        y = [1, 0, 1, 0, 6]
        dataset = Tasks(X, y)
        self.assertEqual(dataset.update_label_by_ai(0, 3), 3)

    # def test_humans_labeled_indexes(self):
    #     X = [1, 2, 3, 4, 5]
    #     y = [1, 0, 1, 0, 6]
    #     tasks = Tasks(X, y)
    #     tasks.update_label_by_human(0, 3)
    #     tasks.update_label_by_human(1, 3)

    #     self.assertEqual(tasks.humans_labeled_indexes, [0, 1])

    def test_tasks_is_completed(self):
        X = [1, 2, 3]
        y = [1, 0, 1]
        tasks = Tasks(X, y)
        tasks.update_label_by_human(0, 0)
        tasks.update_label_by_ai(1, 0)
        self.assertFalse(tasks.is_completed)
        tasks.update_label_by_human(2, 0)
        self.assertTrue(tasks.is_completed)

    # def test_y_assigned_ground_truth(self):
    #     X = [1, 2, 3]
    #     y = [1, 0, 1]
    #     tasks = Tasks(X, y)
    #     tasks.update_human_label(0, 1)
    #     tasks.update_ai_label(1, 2)
    #     # tasks.update_ai_label(2, 3)

    #     self.assertEqual(tasks.y_assigned_ground_truth, [1, 0])

    # def test_y_assigned(self):
    #     Xtasks = [1, 2, 3]
    #     y = [1, 0, 1]
    #     tasks = Tasks(X, y)
    #     tasks.update_human_label(0, 1)
    #     tasks.update_ai_label(1, 2)
    #     # tasks.update_ai_label(2, 3)

    #     self.assertEqual(tasks.y_assigned, [1, 2])

    # def test_y_ai_ground_truth(self):
    #     X = [1, 2, 3]
    #     y = [1, 0, 1]
    #     tasks = Tasks(X, y)
    #     tasks.update_human_label(0, 1)
    #     tasks.update_ai_label(1, 2)
    #     tasks.update_ai_label(2, 3)

    #     self.assertCountEqual(tasks.y_ai_ground_truth, [0, 1])

    # def test_y_ai_assigned(self):
    #     X = [1, 2, 3]
    #     y = [1, 0, 1]
    #     tasks = Tasks(X, y)
    #     tasks.update_human_label(0, 1)
    #     tasks.update_ai_label(1, 2)
    #     tasks.update_ai_label(2, 3)

    #     self.assertCountEqual(tasks.y_ai_assigned, [2, 3])

    # def test_the_length_at_the_init_state(self):
    #     X = [1, 2, 3, 4, 5]
    #     y = [1, 0, 1, 0, 6]
    #     dataset = Tasks(X, y)
    #     self.assertEqual(len(dataset), 0)

    # def test_unlabeled_indexes(self):
    #     X = [1, 2, 3, 4, 5]
    #     y = [1, 0, 1, 0, 6]
    #     dataset = Tasks(X, y)
    #     self.assertEqual(len(dataset.unlabeled_indexes()), len(y))

    # def test_labeled_indexes(self):
    #     X = [1, 2, 3, 4, 5]
    #     y= [1, 0, 1, 0, 6]
    #     dataset = Tasks(X, y)
    #     self.assertEqual(len(dataset.labeled_indexes()), 0)

    def test_all_labeled_indexes(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.update_label_by_ai(0, 0)
        tasks.update_label_by_human(1, 0)

        self.assertEqual(tasks.all_labeled_indexes, [0, 1])

    def test_y_all_labeled_ground_truth(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.update_label_by_ai(0, 0)
        tasks.update_label_by_human(1, 0)

        self.assertEqual(tasks.y_all_labeled_ground_truth, [1, 0])

    def test_y_all_labeled(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.update_label_by_ai(0, 0)
        tasks.update_label_by_human(1, 0)

        self.assertEqual(tasks.y_all_labeled, [0, 0])

    def test_ai_labeled_indexes(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.update_label_by_ai(0, 0)
        tasks.update_label_by_human(1, 0)

        self.assertEqual(tasks.ai_labeled_indexes, [0])

    def test_y_ai_labeled_ground_truth(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.update_label_by_ai(0, 0)
        tasks.update_label_by_human(1, 0)

        self.assertEqual(tasks.y_ai_labeled_ground_truth, [1])

    def test_y_ai_labeled(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.update_label_by_ai(0, 0)
        tasks.update_label_by_human(1, 0)

        self.assertEqual(tasks.y_ai_labeled, [0])

    def test_human_labeled_indexes(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.update_label_by_ai(0, 0)
        tasks.update_label_by_human(1, 0)

        self.assertEqual(tasks.human_labeled_indexes, [1])

    def test_get_ground_truth(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        self.assertEqual(tasks.get_ground_truth([0, 2, 4]), [1, 1, 6])

    def test_assignable_indexes(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)
        self.assertEqual(tasks.assignable_indexes, [0, 1, 2, 3, 4])

    def test_X_assignable(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)
        self.assertEqual(list(tasks.X_assignable), X)
        self.assertEqual(len(tasks.X_assignable), len(tasks.assignable_indexes))

    def test_get_train_set(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.bulk_update_labels_by_human([0, 1, 2, 3], [0, 1, 0, 1])

        X_train, y_train = tasks.train_set

        # print(X_train)

        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(y_train), 2)

    def test_get_test_set(self):
        X = [1, 2, 3, 4, 5]
        y= [1, 0, 1, 0, 6]
        tasks = Tasks(X, y)

        tasks.bulk_update_labels_by_human([0, 1, 2, 3], [0, 1, 0, 1])

        X_train, y_train = tasks.test_set

        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(y_train), 2)

if __name__ == '__main__':
    unittest.main()
