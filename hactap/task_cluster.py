import torch
import numpy as np
from scipy.stats import beta
from torch.utils.data import Subset
from torch.utils.data import DataLoader

NUMBER_OF_MONTE_CARLO_TRIAL = 500_000


class TaskCluster:
    def __init__(self, model, info, prior_distribution=[1, 1]):
        self.__model = model
        self.__info = info
        self.__prior_distribution = prior_distribution

        self.__n_answerable_tasks = 0
        self.__match_rate_with_human = 0
        self.__conflict_rate_with_human = 0
        self.__bata_dist = []

        self.__assignable_task_idx_test = []

    @property
    def match_rate_with_human(self):
        return self.__match_rate_with_human

    @property
    def conflict_rate_with_human(self):
        return self.__conflict_rate_with_human

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
    def assignable_task_indexes(self):
        return self.__assignable_task_indexes

    @property
    def y_pred(self):
        return self.__y_pred

    def update_status_human(self, dataset, n_monte_carlo_trial=1):
        self.__n_answerable_tasks = len(dataset.human_labeled_indexes)

        self.__bata_dist = beta.rvs(
            self.__prior_distribution[0] + self.__n_answerable_tasks,
            self.__prior_distribution[1],
            size=n_monte_carlo_trial
        )

    # def update_status_remain(self, dataset, diff_ids, quality_req, n_monte_carlo_trial=1): # NOQA
    #     self.__n_answerable_tasks = len(dataset.human_assignable_indexes()) - diff_ids # NOQA

    #     self.__bata_dist = beta.rvs(
    #         1 + int(self.__n_answerable_tasks * quality_req),
    #         1 + int(self.__n_answerable_tasks * (1 - quality_req)),
    #         size=n_monte_carlo_trial
    #     )

    def update_status(self, dataset, n_monte_carlo_trial=1):
        self.__n_answerable_tasks = 0
        self.__assignable_task_indexes = []
        self.__y_pred = []

        test_set = dataset.test_set
        test_indexes = dataset.test_indexes

        if len(test_set) == 0:
            self.__assignable_task_idx_test = []
            self.__match_rate_with_human = 0
            self.__conflict_rate_with_human = 0
            return

        # 現在の dataset.assignable_indexes に基づいて、そのタスククラスタに割り当て可能なindexesを更新する
        assignable_indexes = dataset.assignable_indexes
        human_assignable_indexes = dataset.human_assignable_indexes()

        for answerable_tasks_id, y_pred_i in zip(self.__info["stat"]["answerable_tasks_ids"], self.__info["stat"]["y_pred"]): # NOQA
            if answerable_tasks_id in assignable_indexes:
                if answerable_tasks_id in human_assignable_indexes:
                    self.__n_answerable_tasks += 1

                self.__assignable_task_indexes.append(answerable_tasks_id)
                self.__y_pred.append(self.rule['rule']['to'])

        # dataset,test_set を対象に、割り当て可能なindexesを計算する
        assignable_task_idx_test2, y_preds_test = self._calc_assignable_tasks( # NOQA
            test_set, np.array(range(len(test_indexes)))
        )

        # 直前で入手したやつは相対的なindexesなので、絶対値を計算する
        assignable_task_idx_test = []
        for hoge_i in assignable_task_idx_test2:
            assignable_task_idx_test.append(test_indexes[hoge_i])
        self.__assignable_task_idx_test = assignable_task_idx_test

        # test_setのsubsetを作る
        human_ds = Subset(test_set, assignable_task_idx_test2)

        # 人間のラベルを参照する
        y_human = np.array([y for x, y in iter(human_ds)])

        # 一致回数と不一致回数を計算する
        self.__match_rate_with_human = 0
        for _p, _h in zip(y_preds_test, y_human):
            if int(_p) == int(_h):
                self.__match_rate_with_human += 1

        self.__conflict_rate_with_human = len(y_preds_test) - self.__match_rate_with_human # NOQA

        # ベータ分布に従う乱数を生成する
        self.__bata_dist = []

        for x in range(int(n_monte_carlo_trial / 100_000)):
            self.__bata_dist.extend(beta.rvs(
                (self.__prior_distribution[0] + self.__match_rate_with_human),
                (self.__prior_distribution[1] + self.__conflict_rate_with_human), # NOQA
                size=100_000
            ))
        return

    def _calc_assignable_tasks(self, x, assignable_indexes):
        rule = self.rule["rule"]

        batch_size = 10000
        _y_pred = []
        _assigned_idx = []

        _z_i = 0

        predict_data = DataLoader(x, batch_size=batch_size)

        print('size of x', len(x))
        print('size of assignable_indexes', len(assignable_indexes))

        for index, (pd_i, _) in enumerate(predict_data):
            print('_calc_assignable_tasks', index)
            y_pred = torch.tensor(self.model.predict(pd_i))

            for ypi, yp in enumerate(y_pred):

                if yp == rule['from']:
                    _y_pred.append(rule['to'])
                    _assigned_idx.append(
                        assignable_indexes[_z_i]
                    )

                _z_i += 1

        return _assigned_idx, _y_pred

    @property
    def n_answerable_tasks(self):
        return self.__n_answerable_tasks

    @property
    def bata_dist(self):
        return self.__bata_dist
