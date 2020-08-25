import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np

# ref:
# - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class Tasks(Dataset):
    def __init__(self, dataset, data_index, human_labelable_index=None):
        self.__dataset = dataset
        if human_labelable_index:
            self.__human_labelable_index = human_labelable_index
        else:
            self.__human_labelable_index = data_index

        self.__y_ground_truth = np.array(
            [dataset[i][1] for i in range(len(dataset))]
        )
        self.__indexes = data_index
        self.__y_human = [None] * len(data_index)
        self.__y_ai = [None] * len(data_index)

        self.__X_train = []
        self.__X_test = []

        self.__y_train = []
        self.__y_test = []

        self.train_indexes = []
        self.test_indexes = []

        self.retired_human_label = []

    @property
    def raw_indexes(self):
        return self.__indexes

    @property
    def raw_y_human(self):
        return self.__y_human

    @property
    def raw_y_ai(self):
        return self.__y_ai

    @property
    def raw_ground_truth(self):
        return self.__y_ground_truth

    @property
    def is_completed(self):
        return len(self.all_labeled_indexes_for_metric) == len(self.__human_labelable_index)

    @property
    def train_set(self):
        return self.__trainset

    @property
    def test_set(self):
        return self.__testset

    @property
    def all_labeled_indexes(self):
        indexes = []

        for index, (y_human_i, y_ai_i) in enumerate(zip(self.__y_human, self.__y_ai)): # NOQA
            if y_human_i is not None or y_ai_i is not None:
                indexes.append(index)

        return indexes

    @property
    def all_labeled_indexes_for_metric(self):
        indexes = []

        for index, (y_human_i, y_ai_i) in enumerate(zip(self.__y_human, self.__y_ai)): # NOQA
            if y_human_i is not None or y_ai_i is not None and (index in self.__human_labelable_index):
                indexes.append(index)

        return indexes

    @property
    def y_all_labeled_ground_truth(self):
        y = []

        for index in self.all_labeled_indexes:
            y.append(self.__y_ground_truth[index])

        return y

    @property
    def y_all_labeled(self):
        y = []

        for index in self.all_labeled_indexes:
            if self.__y_human[index] is not None:
                y.append(self.__y_human[index])
            else:
                y.append(self.__y_ai[index])

        return y

    @property
    def y_all_labeled_ground_truth_for_metric(self):
        y = []

        for index in self.all_labeled_indexes_for_metric:
            y.append(self.__y_ground_truth[index])

        return y

    @property
    def y_all_labeled_for_metric(self):
        y = []

        for index in self.all_labeled_indexes_for_metric:
            if self.__y_human[index] is not None:
                y.append(self.__y_human[index])
            else:
                y.append(self.__y_ai[index])

        return y

    @property
    def ai_labeled_indexes(self):
        indexes = []

        for index, y_ai_i in enumerate(self.__y_ai):
            if y_ai_i is not None:
                indexes.append(index)

        return indexes

    @property
    def ai_labeled_indexes_for_metric(self):
        indexes = []

        for index, y_ai_i in enumerate(self.__y_ai):
            if y_ai_i is not None and (index in self.__human_labelable_index):
                indexes.append(index)

        return indexes

    @property
    def y_ai_labeled_ground_truth(self):
        y = []

        for index in self.ai_labeled_indexes:
            y.append(self.__y_ground_truth[index])

        return y

    @property
    def y_ai_labeled(self):
        y = []

        for index in self.ai_labeled_indexes:
            y.append(self.__y_ai[index])

        return y

    @property
    def y_ai_labeled_ground_truth_for_metric(self):
        y = []

        for index in self.ai_labeled_indexes_for_metric:
            y.append(self.__y_ground_truth[index])

        return y

    @property
    def y_ai_labeled_for_metric(self):
        y = []

        for index in self.ai_labeled_indexes_for_metric:
            y.append(self.__y_ai[index])

        return y

    @property
    def human_labeled_indexes(self):
        indexes = []

        for index, y_human_i in enumerate(self.__y_human):
            if y_human_i is not None:
                indexes.append(index)

        return indexes

    @property
    def y_human_labeled(self):
        y = []

        for index in self.human_labeled_indexes:
            y.append(self.__y_human[index])

        return y

    @property
    def x_human_labeled(self):
        x = []

        for index in self.human_labeled_indexes:
            x.append(self.__X[index])

        return x

    # for pytorch API
    def __len__(self):
        return len(self.human_labeled_indexes)

    # for pytorch API
    def __getitem__(self, index):
        labeled_indexes = self.human_labeled_indexes
        return self.__dataset[labeled_indexes[index]], self.__y_human[labeled_indexes[index]] # NOQA

    def update_label_by_human(self, index, label):
        if self.__y_human[index] or self.__y_ai[index]:
            raise Exception("duplicate assignment (HM)")

        self.__y_human[index] = label
        # print('self.__y_human[index]', self.__y_human[index])
        return self.__y_human[index]

    def update_label_by_ai(self, index, label):
        if self.__y_human[index] or self.__y_ai[index]:
            raise Exception("duplicate assignment (AI)")

        self.__y_ai[index] = label
        return self.__y_ai[index]

    def bulk_update_labels_by_ai(self, indexes, y_pred):
        for _, (index, y_pred_i) in enumerate(zip(indexes, y_pred)):
            self.update_label_by_ai(index, y_pred_i)

    def bulk_update_labels_by_human(self, indexes, labels):
        for index, label in zip(indexes, labels):
            self.update_label_by_human(index, label)

        self.__update_train_test_set()

        return

    def get_ground_truth(self, indexes):
        y = []

        for index in indexes:
            y.append(self.__dataset[index][1])

        return y

    def retire_human_label(self, indexes):
        self.retired_human_label.extend(indexes)

        self.__update_train_test_set()

    def __update_train_test_set(self):
        train_indexes, test_indexes = train_test_split(
            self.human_labeled_indexes, test_size=0.5
        )

        # print('human_labeled_indexes', self.human_labeled_indexes)
        # print('self.retired_human_label', self.retired_human_label)

        retired_human_label = self.retired_human_label

        masked_test_indexes = []
        for ti in test_indexes:
            if ti not in retired_human_label:
                masked_test_indexes.append(ti)

        print('test_indexes', len(test_indexes))
        print('masked_test_indexes', len(masked_test_indexes))

        self.train_indexes = train_indexes
        self.test_indexes = test_indexes

        self.__trainset = Subset(self.__dataset, train_indexes)
        self.__testset = Subset(self.__dataset, test_indexes)
        return

    def human_assignable_indexes(self):

        indexes = []

        # for index, (y_human_i, y_ai_i) in enumerate(zip(self.__y_human, self.__y_ai)): # NOQA
        #     if y_human_i is None and y_ai_i is None and (index in self.__human_labelable_index): # NOQA
        #         indexes.append(index)

        for i, hli in enumerate(self.__human_labelable_index):
            if self.__y_human[hli] is None and self.__y_ai[hli] is None:
                indexes.append(hli)

        return indexes

    @property
    def assignable_indexes(self):
        indexes = []

        for index, (y_human_i, y_ai_i) in enumerate(zip(self.__y_human, self.__y_ai)): # NOQA
            if y_human_i is None and y_ai_i is None:
                indexes.append(index)

        return indexes

    @property
    def X_assignable(self):
        return Subset(self.__dataset, self.assignable_indexes)
