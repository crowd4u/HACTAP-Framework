# from typing import Union
from typing import List
from typing import Dict

from scipy.stats import beta

from hactap.logging import get_logger
from hactap.ai_worker import BaseAIWorker
from hactap.tasks import Tasks

logger = get_logger()
NUMBER_OF_MONTE_CARLO_TRIAL = 500_000


class TaskCluster:
    def __init__(
        self,
        model: BaseAIWorker,
        aiw_id: int,
        info: Dict[str, Dict],
        prior_distribution: List[int] = [1, 1]
    ):
        self.__model = model
        self.__aiw_id = aiw_id
        self.__info = info
        self.__prior_distribution = prior_distribution

        self.__n_answerable_tasks = 0
        self.__match_rate_with_human = 0
        self.__conflict_rate_with_human = 0
        self.__bata_dist: List = []

        self.__assignable_task_idx_test: List = []
        self.__test_y_human: List = []
        self.__test_y_predict: List = []

        self.__assignable_task_idx_train: List = []
        self.__train_y_human: List = []
        self.__train_y_predict: List = []

    @property
    def match_rate_with_human(self) -> int:
        return self.__match_rate_with_human

    @property
    def conflict_rate_with_human(self) -> int:
        return self.__conflict_rate_with_human

    @property
    def aiw_id(self) -> int:
        return self.__aiw_id

    @property
    def rule(self) -> Dict:
        return self.__info

    @property
    def model(self) -> BaseAIWorker:
        return self.__model

    @property
    def assignable_task_idx_test(self) -> List:
        return self.__assignable_task_idx_test

    @property
    def assignable_task_idx_train(self) -> List:
        return self.__assignable_task_idx_train

    @property
    def assignable_task_indexes(self) -> List:
        return self.__assignable_task_indexes

    @property
    def y_pred(self) -> List:
        return self.__y_pred

    @property
    def test_y_human(self) -> List:
        return self.__test_y_human

    @property
    def test_y_predict(self) -> List:
        return self.__test_y_predict

    @property
    def train_y_human(self) -> List:
        return self.__train_y_human

    @property
    def train_y_predict(self) -> List:
        return self.__train_y_predict

    def update_status_human(
        self,
        dataset: Tasks,
        n_monte_carlo_trial: int = 1
    ) -> None:
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

    def update_status(
        self,
        dataset: Tasks,
        n_monte_carlo_trial: int = 1
    ) -> None:
        self.__n_answerable_tasks = 0
        self.__assignable_task_indexes = []
        self.__y_pred = []

        test_indexes = dataset.test_indexes

        if len(test_indexes) == 0:
            self.__assignable_task_idx_test = []
            self.__match_rate_with_human = 0
            self.__conflict_rate_with_human = 0
            return

        # 現在の dataset.assignable_indexes に基づいて、そのタスククラスタに割り当て可能なindexesを更新する
        assignable_indexes = dataset.assignable_indexes
        human_assignable_indexes = dataset.human_assignable_indexes()

        for answerable_tasks_id, y_pred_i in zip(self.__info["stat"]["y_pred_remain_ids"], self.__info["stat"]["y_pred_remain"]): # NOQA
            if answerable_tasks_id in set(assignable_indexes):
                if answerable_tasks_id in set(human_assignable_indexes):
                    self.__n_answerable_tasks += 1

                self.__assignable_task_indexes.append(answerable_tasks_id)
                self.__y_pred.append(self.rule['rule']['to'])

        y_pred_test = []
        y_pred_test_human = []
        y_pred_test_ids = []
        for answerable_tasks_id, y_pred_i, y_pred_i_human in zip(self.__info["stat"]["y_pred_test_ids"], self.__info["stat"]["y_pred_test"], self.__info["stat"]["y_pred_test_human"]): # NOQA
            if answerable_tasks_id in set(test_indexes):
                y_pred_test.append(self.rule['rule']['to'])
                y_pred_test_human.append(y_pred_i_human)
                y_pred_test_ids.append(answerable_tasks_id)

        self.__assignable_task_idx_test = y_pred_test_ids
        self.__assignable_task_idx_train = self.__info["stat"]["y_pred_train_ids"] # NOQA

        # # dataset,test_set を対象に、割り当て可能なindexesを計算する
        # assignable_task_idx_test2, y_preds_test = self._calc_assignable_tasks( # NOQA
        #     test_set, np.array(range(len(test_indexes)))
        # )

        # assignable_task_idx_train2, y_preds_train = self._calc_assignable_tasks( # NOQA
        #     train_set, np.array(range(len(train_indexes)))
        # )

        # if len(assignable_task_idx_test2) == 0:
        #     self.__assignable_task_idx_test = []
        #     self.__match_rate_with_human = 0
        #     self.__conflict_rate_with_human = 0
        #     return

        # 直前で入手したやつは相対的なindexesなので、絶対値を計算する
        # assignable_task_idx_test = []
        # for hoge_i in assignable_task_idx_test2:
        #     assignable_task_idx_test.append(test_indexes[hoge_i])
        # self.__assignable_task_idx_test = assignable_task_idx_test

        # assignable_task_idx_train = []
        # for hoge_i in assignable_task_idx_train2:
        #     assignable_task_idx_train.append(train_indexes[hoge_i])
        # self.__assignable_task_idx_train = assignable_task_idx_train

        # print("!!! ===========")
        # print("test set size {}".format(len(test_set)))
        # print(
        # "assignable_task_idx_test {}".format(len(assignable_task_idx_test))
        # )
        # print(
        # "assignable_task_idx_test2 {}".format(len(assignable_task_idx_test2))
        # )

        if len(y_pred_test_ids) != 0:
            # 人間のラベルを参照する
            # y_human = np.array([y for x, y in iter(human_ds)])

            self.__test_y_human = y_pred_test_human
            self.__test_y_predict = y_pred_test

            self.__train_y_human = self.__info["stat"]["y_pred_train_human"]
            self.__train_y_predict = [self.rule['rule']['to']] * len(self.__info["stat"]["y_pred_train"]) # NOQA

            # 一致回数と不一致回数を計算する
            self.__match_rate_with_human = 0

            for _p, _h in zip(
                y_pred_test,
                y_pred_test_human
            ):
                # for _p, _h in zip(y_preds_test, y_human_test):
                if int(_p) == int(_h):
                    self.__match_rate_with_human += 1

            self.__conflict_rate_with_human = len(y_pred_test) - self.__match_rate_with_human # NOQA

            # ベータ分布に従う乱数を生成する
            self.__bata_dist = []

            for x in range(int(n_monte_carlo_trial / 100_000)):
                self.__bata_dist.extend(beta.rvs(
                    (self.__prior_distribution[0] + self.__match_rate_with_human), # NOQA
                    (self.__prior_distribution[1] + self.__conflict_rate_with_human), # NOQA
                    size=100_000
                ))
            return

    # def _calc_assignable_tasks(
    #     self,
    #     x: Dataset,
    #     assignable_indexes: List
    # ) -> Tuple[List, List]:
    #     rule = self.rule["rule"]

    #     batch_size = 10000
    #     _y_pred = []
    #     _assigned_idx = []

    #     _z_i = 0

    #     predict_data = DataLoader(x, batch_size=batch_size)

    #     # print('size of x', len(x))
    #     # print('size of assignable_indexes', len(assignable_indexes))

    #     for index, (pd_i, _) in enumerate(predict_data):
    #         print('_calc_assignable_tasks', index)
    #         y_pred = torch.tensor(self.model.predict(pd_i))

    #         for ypi, yp in enumerate(y_pred):

    #             if yp == rule['from']:
    #                 _y_pred.append(rule['to'])
    #                 _assigned_idx.append(
    #                     assignable_indexes[_z_i]
    #                 )

    #             _z_i += 1

    #     return _assigned_idx, _y_pred

    @property
    def n_answerable_tasks(self) -> int:
        return self.__n_answerable_tasks

    @property
    def bata_dist(self) -> List:
        return self.__bata_dist

    def update_status2(  # type: ignore
        self,
        test_indexes,
        assignable_indexes,
        n_monte_carlo_trial: int = 1
    ) -> None:
        self.__n_answerable_tasks = 0
        self.__assignable_task_indexes = []
        self.__y_pred = []

        if len(test_indexes) == 0:
            self.__assignable_task_idx_test = []
            self.__match_rate_with_human = 0
            self.__conflict_rate_with_human = 0
            return

        # 現在の dataset.assignable_indexes に基づいて、そのタスククラスタに割り当て可能なindexesを更新する

        for answerable_tasks_id, y_pred_i in zip(self.__info["stat"]["y_pred_remain_ids"], self.__info["stat"]["y_pred_remain"]): # NOQA
            if answerable_tasks_id in assignable_indexes:
                self.__n_answerable_tasks += 1

                self.__assignable_task_indexes.append(answerable_tasks_id)
                self.__y_pred.append(self.rule['rule']['to'])

        y_pred_test = []
        y_pred_test_human = []
        y_pred_test_ids = []
        for answerable_tasks_id, y_pred_i, y_pred_i_human in zip(self.__info["stat"]["y_pred_test_ids"], self.__info["stat"]["y_pred_test"], self.__info["stat"]["y_pred_test_human"]): # NOQA
            if answerable_tasks_id in test_indexes:
                y_pred_test.append(self.rule['rule']['to'])
                y_pred_test_human.append(y_pred_i_human)
                y_pred_test_ids.append(answerable_tasks_id)

        self.__assignable_task_idx_test = y_pred_test_ids
        self.__assignable_task_idx_train = self.__info["stat"]["y_pred_train_ids"] # NOQA

        if len(y_pred_test_ids) != 0:
            # 人間のラベルを参照する
            # y_human = np.array([y for x, y in iter(human_ds)])

            self.__test_y_human = y_pred_test_human
            self.__test_y_predict = y_pred_test

            self.__train_y_human = self.__info["stat"]["y_pred_train_human"]
            self.__train_y_predict = [self.rule['rule']['to']] * len(self.__info["stat"]["y_pred_train"]) # NOQA

            # 一致回数と不一致回数を計算する
            self.__match_rate_with_human = 0

            for _p, _h in zip(
                y_pred_test,
                y_pred_test_human
            ):
                # for _p, _h in zip(y_preds_test, y_human_test):
                if int(_p) == int(_h):
                    self.__match_rate_with_human += 1

            self.__conflict_rate_with_human = len(y_pred_test) - self.__match_rate_with_human # NOQA

            # ベータ分布に従う乱数を生成する
            self.__bata_dist = []

            for x in range(int(n_monte_carlo_trial / 100_000)):
                self.__bata_dist.extend(beta.rvs(
                    (self.__prior_distribution[0] + self.__match_rate_with_human), # NOQA
                    (self.__prior_distribution[1] + self.__conflict_rate_with_human), # NOQA
                    size=100_000
                ))
            return
