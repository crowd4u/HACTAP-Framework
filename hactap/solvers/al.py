import torch
import numpy as np
from sklearn.model_selection import cross_val_score

from hactap import solver


class AL(solver.Solver):
    def __init__(
        self,
        tasks,
        ai_workers,
        accuracy_requirement,
        human_crowd_batch_size
    ):
        super().__init__(tasks, ai_workers, accuracy_requirement)
        self.human_crowd_batch_size = human_crowd_batch_size

    def run(self):
        self.report_log()

        while self.tasks.is_not_completed:
            ai_worker_list = []

            ai_worker_list.append(
                self._evalate_al_worker_by_cv(
                    1,
                    self.ai_workers[0],
                    self.tasks,
                    self.accuracy_requirement
                )
            )
            self.ai_workers[0].fit(self.tasks.x_train, self.tasks.y_train)

            self.logger.debug('Task Clusters %s', ai_worker_list)
            for ai_worker in ai_worker_list:
                # 残タスク数が0だと推論できないのでこれが必要
                if len(self.tasks.x_remaining) == 0:
                    break

                if not ai_worker['was_accepted']:
                    continue

                assigned_idx = range(len(self.tasks.x_remaining))
                y_pred = torch.tensor(
                    self.ai_workers[0].predict(self.tasks.x_remaining)
                )
                self.tasks.assign_tasks_to_ai(assigned_idx, y_pred)
                self.report_log()

            self.assign_to_human_workers()
            self.report_log()

        return self.logs, self.assignment_log

    def _evalate_al_worker_by_cv(
        self,
        worker_id,
        aiw,
        dataset,
        quality_requirements
    ):
        cross_validation_scores = cross_val_score(
            aiw,
            dataset.x_test,
            dataset.y_test,
            scoring='accuracy',
            cv=5,
        )
        score_cv_mean = np.mean(cross_validation_scores)
        log = {
            'ai_worker_id': worker_id,
            'accepted_rule': {
                "from": "*",
                "to": "*"
            },
            'score_cv': cross_validation_scores,
            'score_cv_mean': score_cv_mean,
            'was_accepted': score_cv_mean > quality_requirements
        }
        return log
