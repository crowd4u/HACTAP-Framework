import torch
import numpy as np
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

        self.__assignable_task_idx_test = []

    @property
    def rule(self):
        return self.__info

    @property
    def model(self):
        return self.__model

    @property
    def assignable_task_idx_test(self):
        return self.__assignable_task_idx_test

    @property
    def match_rate_with_human(self):
        return self.__match_rate_with_human

    @property
    def conflict_rate_with_human(self):
        return self.__conflict_rate_with_human

    def update_status_human(self, dataset):
        self.__n_answerable_tasks = len(dataset.human_labeled_indexes)

        X_train, _ = dataset.train_set
        self.__bata_dist = beta.rvs(
            1 + len(X_train),
            1,
            size=NUMBER_OF_MONTE_CARLO_TRIAL
        )

    def update_status_remain(self, dataset, diff_ids):
        self.__n_answerable_tasks = len(dataset.X_assignable) - len(diff_ids)
        X_test, _ = dataset.test_set
        self.__bata_dist = beta.rvs(
            1 + len(X_test),
            1,
            size=NUMBER_OF_MONTE_CARLO_TRIAL
        )

    def update_status(self, dataset):

        X_test, y_test = dataset.test_set

        if len(X_test) == 0:
            self.__n_answerable_tasks = 0
            self.__assignable_task_idx_test = []
            self.__match_rate_with_human = 0
            self.__conflict_rate_with_human = 0
        else:
            assignable_task_idx_rem, _ = self._calc_assignable_tasks(dataset.X_assignable, dataset.assignable_indexes)
            self.__n_answerable_tasks = len(assignable_task_idx_rem)

            assignable_task_idx_test, y_preds_test = self._calc_assignable_tasks(X_test, np.array(range(len(dataset.test_indexes))))
            self.__assignable_task_idx_test = assignable_task_idx_test

            # TODO: test
            y_human = torch.index_select(y_test, 0, torch.tensor(assignable_task_idx_test, dtype=torch.long))
            # print(type(dataset.y_test), type(y_humans))
            # print(len(dataset.y_test), len(y_humans), len(assignable_task_idx_test))

            self.__match_rate_with_human = 0

            for _p, _h in zip(y_preds_test, y_human):
                if _p == int(_h):
                    self.__match_rate_with_human += 1

            self.__conflict_rate_with_human = len(y_preds_test) - self.__match_rate_with_human
            # print(self.__match_rate_with_human, self.__conflict_rate_with_human, len(y_preds_test))

        self.__bata_dist = beta.rvs(
            (1 + self.__match_rate_with_human),
            (1 + self.__conflict_rate_with_human),
            size=NUMBER_OF_MONTE_CARLO_TRIAL
        )

        return

        
    # def update_status1(self, dataset):
    #     # calc for remaining tasks
    #     assigned_idx = range(len(dataset.x_remaining))
    #     y_pred = torch.tensor(self.__model.predict(dataset.x_remaining))
    #     mask = y_pred == self.__info['rule']['from']

    #     _assigned_idx = list(compress(assigned_idx, mask.numpy()))
    #     _y_pred = y_pred.masked_select(mask)
    #     _y_pred[_y_pred == self.__info['rule']['from']] = self.__info['rule']['to']
    #     _y_pred.type(torch.LongTensor)

    #     print('remaining: ', len(y_pred), 'answerable: ', len(_y_pred))
    #     self.__n_answerable_tasks = len(_y_pred)

    #     # calc for human labeled tasks
    #     assigned_idx = range(len(dataset.x_test))
    #     y_pred = torch.tensor(self.__model.predict(dataset.x_test))
    #     mask = y_pred == self.__info['rule']['from']

    #     _assigned_idx = list(compress(assigned_idx, mask.numpy()))
    #     _y_pred = y_pred.masked_select(mask)
    #     _y_pred[_y_pred == self.__info['rule']['from']] = self.__info['rule']['to']
    #     _y_pred.type(torch.LongTensor)
    #     _y_human = dataset.y_test.masked_select(mask)
    #     # print(len(_y_pred), len(_y_human))

    #     for _p, _h in zip(_y_pred, _y_human):
    #         if _p == int(_h):
    #             self.__match_rate_with_human += 1

    #     self.__conflict_rate_with_human = len(_y_pred) - self.__match_rate_with_human

    #     # print(len(_y_pred), self.__match_rate_with_human, self.__conflict_rate_with_human)

    #     self.__bata_dist = beta.rvs(
    #         (1 + self.__match_rate_with_human),
    #         (1 + self.__conflict_rate_with_human),
    #         size=NUMBER_OF_MONTE_CARLO_TRIAL
    #     )

    #     # print(self.__bata_dist)

    #     return

    def _calc_assignable_tasks(self, x, assignable_indexes):
        rule = self.rule["rule"]

        assigned_idx = np.array(assignable_indexes)
        
        # print(x)
        # print(type(x))
        y_pred = torch.tensor(self.model.predict(x))

        _y_pred = []
        _assigned_idx = []

        for ypi, yp in enumerate(y_pred):
            if yp == rule['from']:
                _y_pred.append(rule['to'])
                _assigned_idx.append(assignable_indexes[ypi])

        return _assigned_idx, _y_pred

    @property
    def n_answerable_tasks(self):
        return self.__n_answerable_tasks

    @property
    def bata_dist(self):
        return self.__bata_dist
