import torch
from itertools import compress
from scipy.stats import beta

NUMBER_OF_MONTE_CARLO_TRIAL = 100_000


class TaskCluster:
    def __init__(self, model, info):
        self.__model = model
        self.__info = info

        self.__n_answerable_tasks = 0
        self.__match_rate_with_human = 0
        self.__conflict_rate_with_human = 0
        self.__bata_dist = []

    def update_status_human(self, dataset):
        # self.__n_answerable_tasks = len(dataset.x_remaining)
        self.__n_answerable_tasks = 0
        self.__bata_dist = beta.rvs(
            (1 + len(dataset.x_human)),
            1,
            size=NUMBER_OF_MONTE_CARLO_TRIAL
        )

    def update_status(self, dataset):
        # calc for remaining tasks
        assigned_idx = range(len(dataset.x_remaining))
        y_pred = torch.tensor(self.__model.predict(dataset.x_remaining))
        mask = y_pred == self.__info['accepted_rule']['from']

        _assigned_idx = list(compress(assigned_idx, mask.numpy()))
        _y_pred = y_pred.masked_select(mask)
        _y_pred[_y_pred == self.__info['accepted_rule']['from']] = self.__info['accepted_rule']['to']
        _y_pred.type(torch.LongTensor)

        print('remaining: ', len(y_pred), 'answerable: ', len(_y_pred))
        self.__n_answerable_tasks = len(_y_pred)

        # calc for human labeled tasks
        assigned_idx = range(len(dataset.x_test))
        y_pred = torch.tensor(self.__model.predict(dataset.x_test))
        mask = y_pred == self.__info['accepted_rule']['from']

        _assigned_idx = list(compress(assigned_idx, mask.numpy()))
        _y_pred = y_pred.masked_select(mask)
        _y_pred[_y_pred == self.__info['accepted_rule']['from']] = self.__info['accepted_rule']['to']
        _y_pred.type(torch.LongTensor)
        _y_human = dataset.y_test.masked_select(mask)
        # print(len(_y_pred), len(_y_human))

        for _p, _h in zip(_y_pred, _y_human):
            if _p == int(_h):
                self.__match_rate_with_human += 1

        self.__conflict_rate_with_human = len(_y_pred) - self.__match_rate_with_human

        # print(len(_y_pred), self.__match_rate_with_human, self.__conflict_rate_with_human)

        self.__bata_dist = beta.rvs(
            (1 + self.__match_rate_with_human),
            (1 + self.__conflict_rate_with_human),
            size=NUMBER_OF_MONTE_CARLO_TRIAL
        )

        # print(self.__bata_dist)

        return

    @property
    def n_answerable_tasks(self):
        return self.__n_answerable_tasks

    @property
    def bata_dist(self):
        return self.__bata_dist
