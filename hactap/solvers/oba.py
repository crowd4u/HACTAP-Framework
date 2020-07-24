import torch
import collections
from scipy import stats
import random
from itertools import compress

from hactap import solver


class OBA(solver.Solver):
    def __init__(
        self,
        tasks,
        ai_workers,
        accuracy_requirement,
        human_crowd_batch_size,
        significance_level
    ):
        super().__init__(tasks, ai_workers, accuracy_requirement)
        self.human_crowd_batch_size = human_crowd_batch_size
        self.significance_level = significance_level

    def run(self):
        self.report_log()

        while self.tasks.is_not_completed:
            task_cluster_candidates = []

            # prepare task cluster candidates
            for w_i, ai_worker in enumerate(self.ai_workers):
                ai_worker.fit(self.tasks.x_train, self.tasks.y_train)

                task_cluster_candidates.extend(
                    self._evalate_al_worker_by_task_cluster(
                        w_i,
                        ai_worker,
                        self.tasks
                    )
                )

            self.logger.info(
                '#task cluster candidates %s',
                len(task_cluster_candidates)
            )
            random.shuffle(task_cluster_candidates)

            # assign tasks to accepted task clusters
            for ai_worker in task_cluster_candidates:

                if not ai_worker['was_accepted']:
                    continue

                # 残りのタスクすうが0だと推論できなくてエラーになる
                if len(self.tasks.x_remaining) == 0:
                    break

                aiw = ai_worker['ai_worker']
                accepted_rule = ai_worker['accepted_rule']

                assigned_idx = range(len(self.tasks.x_remaining))
                y_pred = torch.tensor(aiw.predict(self.tasks.x_remaining))
                mask = y_pred == accepted_rule['from']

                _assigned_idx = list(compress(assigned_idx, mask.numpy()))
                _y_pred = y_pred.masked_select(mask)
                # print(_y_pred)
                _y_pred[_y_pred == accepted_rule['from']] = accepted_rule['to']
                _y_pred.type(torch.LongTensor)
                # print(_y_pred)
                # print('filter', len(_assigned_idx), len(_y_pred))
                self.tasks.assign_tasks_to_ai(_assigned_idx, _y_pred)
                self.report_log()

            self.assign_to_human_workers()
            self.report_log()

        return self.logs, self.assignment_log

    def _evalate_al_worker_by_task_cluster(self, worker_id, aiw, dataset):
        y_pred = torch.tensor(aiw.predict(dataset.x_test))

        task_clusters = {}
        candidates = []

        for y_human_i, y_pred_i in zip(dataset.y_test, y_pred):
            # print(y_human_i, y_pred_i)
            if int(y_pred_i) not in task_clusters:
                task_clusters[int(y_pred_i)] = []
            task_clusters[int(y_pred_i)].append(int(y_human_i))

        for cluster_i, items in task_clusters.items():
            most_common_label = collections.Counter(items).most_common(1)

            # クラスタに含まれるデータがある場合に、そのクラスタの評価が行える
            # このif本当に要る？？？
            if len(most_common_label) == 1:
                label_type, label_count = collections.Counter(
                    items
                ).most_common(1)[0]
                p_value = stats.binom_test(
                    label_count,
                    n=len(items),
                    p=self.accuracy_requirement,
                    alternative='greater'
                )
                # print(collections.Counter(items), p_value)

                log = {
                    'ai_worker': aiw,
                    'ai_worker_id': worker_id,
                    'accepted_rule': {
                        "from": cluster_i,
                        "to": label_type
                    },
                    'was_accepted': p_value < self.significance_level
                }

                candidates.append(log)

        return candidates
