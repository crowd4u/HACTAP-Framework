import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np

# ref:
# - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class Tasks(Dataset):
    def __init__(self, X, y_ground_truth):
        self.__X = X
        if type(X) == list:
            self.__X = torch.tensor(self.__X)

        self.__y_ground_truth = y_ground_truth
        self.__y_human = [None] * len(y_ground_truth)
        self.__y_ai =  [None] * len(y_ground_truth)

        self.__X_train = []
        self.__X_test = []

        self.__y_train = []
        self.__y_test = []

    @property
    def is_completed(self):
        return len(self.all_labeled_indexes) == len(self.__y_ground_truth)

    @property
    def train_set(self):
        return self.__X_train, self.__y_train

    @property
    def test_set(self):
        return self.__X_test, self.__y_test

    @property
    def all_labeled_indexes(self):
        indexes = []

        for index, (y_human_i, y_ai_i) in enumerate(zip(self.__y_human, self.__y_ai)):
            if y_human_i is not None or y_ai_i is not None:
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
    def ai_labeled_indexes(self):
        indexes = []

        for index, y_ai_i in enumerate(self.__y_ai):
            if y_ai_i is not None:
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
        return self.__X[labeled_indexes[index]], self.__y_human[labeled_indexes[index]]


    def update_label_by_human(self, index, label):
        if self.__y_human[index] is not None or self.__y_ai[index] is not None:
            raise Exception("duplicate assignment (HM)")
        # raise Exception("duplicate assignment (HM)")

        self.__y_human[index] = label
        return self.__y_human[index]

    def update_label_by_ai(self, index, label):
        if self.__y_human[index] is not None or self.__y_ai[index] is not None:
            raise Exception("duplicate assignment (AI)")

        self.__y_ai[index] = label
        return self.__y_ai[index]

    def bulk_update_labels_by_ai(self, indexes, labels):
        for index, label in zip(indexes, labels):
            self.update_label_by_ai(index, label)

    def bulk_update_labels_by_human(self, indexes, labels):
        for index, label in zip(indexes, labels):
            self.update_label_by_human(index, label)

        train_indexes, test_indexes = train_test_split(range(len(self)), test_size=0.5)

        self.train_indexes = train_indexes
        self.test_indexes = test_indexes

        train_loader = DataLoader(Subset(self, train_indexes), batch_size=len(train_indexes))
        X_train, y_train = next(iter(train_loader))

        X_train, y_train = next(iter(train_loader))

        test_loader = DataLoader(Subset(self, test_indexes), batch_size=len(test_indexes))
        X_test, y_test = next(iter(test_loader))


        self.__X_train = X_train
        self.__y_train = y_train

        self.__X_test = X_test
        self.__y_test = y_test

        return

    def get_ground_truth(self, indexes):
        y = []

        for index in indexes:
            y.append(self.__y_ground_truth[index])

        return y

    @property
    def assignable_indexes(self):
        indexes = []

        for index, (y_human_i, y_ai_i) in enumerate(zip(self.__y_human, self.__y_ai)):
            if y_human_i is None and y_ai_i is None:
                indexes.append(index)

        return indexes

    @property
    def X_assignable(self):
        return torch.index_select(self.__X, 0, torch.tensor(self.assignable_indexes))
