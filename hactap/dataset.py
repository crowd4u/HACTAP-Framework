import torch
import numpy as np
from sklearn.model_selection import train_test_split

from hactap.logging import get_logger

logger = get_logger()


class Dataset:
    def __init__(self, x, y, path):
        self.__x = x
        self.__y = y
        self.__path = np.asarray(path)
        self.__final_label = []

        self.__x_remaining = x
        self.__y_remaining = y
        self.__path_remaining = np.asarray(path)

        self.__x_human = torch.tensor([], dtype=torch.long)
        self.__y_human = torch.tensor([], dtype=torch.long)
        self.__path_human = []

        self.__x_ai = torch.tensor([], dtype=torch.long)
        self.__y_ai = torch.tensor([], dtype=torch.long)
        # self.__path_ai = torch.tensor([], dtype=torch.long)
        self.__y_ai_ground_truth = torch.tensor([], dtype=torch.long)

        self.__x_train = []
        self.__y_train = []
        self.__x_test = []
        self.__y_test = []

    @property
    def x_all(self):
        return self.__x

    @property
    def y_all(self):
        return self.__y

    @property
    def x_remaining(self):
        return self.__x_remaining

    @property
    def y_remaining(self):
        return self.__y_remaining

    @property
    def x_human(self):
        # print(
        #     self.__x_human.shape
        # )
        # print(
        #     self.__x_human.numpy().shape
        # )
        return self.__x_human

    @property
    def y_human(self):
        return self.__y_human

    @property
    def x_ai(self):
        return self.__x_ai

    @property
    def y_ai(self):
        return self.__y_ai

    @property
    def y_ai_ground_truth(self):
        return self.__y_ai_ground_truth

    @property
    def y_assigned(self):
        return torch.cat([self.__y_human.type(torch.LongTensor), self.__y_ai.type(torch.LongTensor)], dim=0)

    @property
    def y_assigned_ground_truth(self):
        return torch.cat([self.__y_human, self.__y_ai_ground_truth], dim=0)

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def x_test(self):
        return self.__x_test

    @property
    def y_test(self):
        return self.__y_test

    @property
    def is_not_completed(self):
        return self.__x_remaining.nelement() != 0
    
    @property
    def final_label(self):
        return self.__final_label

    def assign_tasks_to_human(self, task_ids):
        # logger.debug('> assign_tasks_to_human')
        # print(task_ids)
        # print(len(self.__x_human), len(self.__x_remaining))
        # print(self.__ .type(), self.__x_remaining.type())
        # print('__path_remaining')
        # print(len(self.__path_remaining))


            # self.__path_human = torch.cat([self.__path_human, self.__path_remaining[task_ids]], dim=0)

            # self.__path_human.extend(self.__path_remaining[task_ids])

        # update train / test set

        _X_train, _X_test, _y_train, _y_test = train_test_split(self.__x_remaining[task_ids], self.__y_remaining[task_ids], test_size=0.5)

        if len(self.__x_train) == 0:
            self.__x_train = _X_train
            self.__y_train = _y_train
            self.__x_test = _X_test
            self.__y_test = _y_test
        else:
            self.__x_train = torch.cat([self.__x_train, _X_train], dim=0)
            self.__y_train = torch.cat([self.__y_train, _y_train], dim=0)
            self.__x_test = torch.cat([self.__x_test, _X_test], dim=0)
            self.__y_test = torch.cat([self.__y_test, _y_test], dim=0)

        # update human set

        if len(self.__x_human) == 0:
            self.__x_human = self.__x_remaining[task_ids]
            self.__y_human = self.__y_remaining[task_ids]
            # self.__path_human = self.__path_remaining[task_ids]
        else:
            self.__x_human = torch.cat([self.__x_human, self.__x_remaining[task_ids]], dim=0)
            self.__y_human = torch.cat([self.__y_human, self.__y_remaining[task_ids]], dim=0)

        remaining_idx = np.isin(range(len(self.__x_remaining)), task_ids, invert=True)
        self.__x_remaining = self.__x_remaining[remaining_idx]
        self.__y_remaining = self.__y_remaining[remaining_idx]
        if len(self.__path) is not 0:
            self.__path_remaining = self.__path_remaining[remaining_idx]

        # print(len(self.__x_human), len(self.__y_remaining))

        # split = int(
        #     len(self.__x_human) / 2
        # )

        # split = np.random.choice(range(len(self.__x_human)), size=int(len(self.__x_human) / 2), replace=False)
        # split_test = np.isin(range(len(self.__x_human)), split, invert=True)
        # self.__x_train = self.__x_human[split]
        # self.__y_train = self.__y_human[split]
        # self.__x_test = self.__x_human[split_test]
        # self.__y_test = self.__y_human[split_test]

        # self.__x_human = torch.cat([self.__x_human, self.__x_remaining[task_ids]], dim=0)

        

        # self.__x_train = X_train
        # self.__y_train = y_train

        # self.__x_test = X_test
        # self.__y_test = y_test



    def lock_human_test(self, task_ids):
        # self.used_human_task_ids 

        original = len(self.__x_test)

        remaining_idx = np.isin(range(len(self.__x_test)), task_ids, invert=True)
        self.__x_test = self.__x_test[remaining_idx]
        self.__y_test = self.__y_test[remaining_idx]

        print('test size', original, len(self.__x_test), 'diff', len(task_ids))
        return 

    def assign_tasks_to_ai(self, task_ids, y_pred):
        # logger.debug('> assign_tasks_to_ai')
        # print(len(self.__x_ai), len(self.__y_remaining))
        # print(type(y_pred))

        if len(self.__x_ai) == 0:
            self.__x_ai = self.__x_remaining[task_ids]
            self.__y_ai = y_pred
            self.__y_ai_ground_truth = self.__y_remaining[task_ids]
        else:
            self.__x_ai = torch.cat([self.__x_ai, self.__x_remaining[task_ids]], dim=0)
            self.__y_ai = torch.cat([self.__y_ai.type(torch.LongTensor), y_pred.type(torch.LongTensor)], dim=0)
            self.__y_ai_ground_truth = torch.cat([self.__y_ai_ground_truth, self.__y_remaining[task_ids]], dim=0)

        if len(self.__path) is not 0:
            self.__final_label.append((
                self.__path_remaining[task_ids],
                y_pred
            ))

        count_remaining = len(self.__x_remaining)
        remaining_idx = np.isin(range(len(self.__x_remaining)), task_ids, invert=True)
        self.__x_remaining = self.__x_remaining[remaining_idx]
        self.__y_remaining = self.__y_remaining[remaining_idx]

        print(count_remaining, "->", len(self.__x_remaining), ':', len(task_ids))

        if len(self.__path) is not 0:
            self.__path_remaining = self.__path_remaining[remaining_idx]
        # print(len(self.__x_ai), len(self.__y_remaining))
