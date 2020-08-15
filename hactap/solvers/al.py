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
        human_crowd_batch_size,
        reporter
    ):
        super().__init__(tasks, ai_workers, accuracy_requirement, reporter)
        self.human_crowd_batch_size = human_crowd_batch_size

    def run(self):
        self.initialize()
        self.report_log()

        while not self.tasks.is_completed:
            ai_worker_list = []

            ai_worker_list.append(
                self._evalate_al_worker_by_cv(
                    1,
                    self.ai_workers[0],
                    self.tasks,
                    self.accuracy_requirement
                )
            )

            X_train, y_train = self.tasks.train_set
            self.ai_workers[0].fit(X_train, y_train)

            for ai_worker in ai_worker_list:
                # 残タスク数が0だと推論できない
                if self.tasks.is_completed:
                    break

                if not ai_worker['was_accepted']:
                    continue

                # assigned_idx = range(len(self.tasks.X_assignable))
                y_pred = torch.tensor(
                    self.ai_workers[0].predict(self.tasks.X_assignable)
                )
                self.tasks.bulk_update_labels_by_ai(
                    self.tasks.assignable_indexes, y_pred
                )
                self.report_log()

            self.assign_to_human_workers()
            self.report_log()

        self.finalize()

        return self.logs, self.assignment_log

    def _evalate_al_worker_by_cv(
        self,
        worker_id,
        aiw,
        dataset,
        quality_requirements
    ):
        X_test, y_test = self.tasks.test_set
        cross_validation_scores = cross_val_score(
            aiw,
            X_test,
            y_test,
            scoring='accuracy',
            cv=5,
            # n_jobs=5
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
