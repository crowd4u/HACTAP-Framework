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

    @property
    def rule(self):
        return self.__info

    @property
    def model(self):
        return self.__model

    def update_status_human(self, dataset):
        self.__n_answerable_tasks = 0 #len(dataset.x_remaining)?
        self.__bata_dist = beta.rvs(
            (1 + len(dataset.x_human)),
            1,
            size=NUMBER_OF_MONTE_CARLO_TRIAL
        )

    def update_status(self, dataset):
        assignable_task_idx_rem, _ = self._calc_assignable_tasks(dataset.x_remaining)
        self.__n_answerable_tasks = len(assignable_task_idx_rem)

        assignable_task_idx_test, y_preds_test = self._calc_assignable_tasks(dataset.x_test)

        # TODO: test
        y_human = torch.index_select(dataset.y_test, 0, torch.tensor(assignable_task_idx_test, dtype=torch.long))
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

    def _calc_assignable_tasks(self, x):
        rule = self.rule["rule"]

        assigned_idx = range(len(x))
        y_pred = torch.tensor(self.model.predict(x))
        mask = y_pred == rule['from']
        # for mm, yy in zip(mask, y_pred):
            # print(mm, yy, rule['from'], rule["to"])
            # print(yy)
        _assigned_idx = list(compress(assigned_idx, mask.numpy()))
        _y_pred = y_pred.masked_select(mask)
        _y_pred[_y_pred == rule['from']] = rule['to']
        # _y_pred.type(torch.LongTensor)
        # print(_y_pred)
        # print('filter', len(_assigned_idx), len(_y_pred))
        # print(mask)
        # print(torch.unique(torch.tensor(mask), return_counts=True, sorted=True))
        # _, count = torch.unique(torch.tensor(mask), return_counts=True, sorted=True)
        # print(count[1])
        # print(len(_assigned_idx), count[1])

        return _assigned_idx, _y_pred

    @property
    def n_answerable_tasks(self):
        return self.__n_answerable_tasks

    @property
    def bata_dist(self):
        return self.__bata_dist
