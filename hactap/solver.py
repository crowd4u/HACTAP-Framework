from hactap.logging import get_logger
from hactap.utils import report_metrics

class Solver():
    def __init__(self, tasks, ai_workers, accuracy_requirement):
        self.tasks = tasks
        self.ai_workers = ai_workers
        self.accuracy_requirement = accuracy_requirement

        self.logs = []
        self.assignment_log = []
        self.logger = get_logger()

    def run(self):
        pass

    def report_log(self):
        self.logs.append(report_metrics(self.tasks))
        self.logger.debug('log: %s', self.logs[-1])

    def report_assignment(self, assignment_log):
        self.assignment_log.append(assignment_log)
        self.logger.debug('new assignment: %s', self.assignment_log[-1])

    def assign_to_human_workers(self):
        if len(self.tasks.x_remaining) != 0:
            if len(self.tasks.x_remaining) < self.human_crowd_batch_size:
                n_instances = len(self.tasks.x_remaining)
            else:
                n_instances = self.human_crowd_batch_size
            query_idx, _ = self.ai_workers[0].query(
                self.tasks.x_remaining,
                n_instances=n_instances
            )
            self.tasks.assign_tasks_to_human(query_idx)
