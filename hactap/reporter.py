import hashlib
import time
import os
import pickle

from hactap.logging import get_logger

logger = get_logger()


class Reporter:
    def __init__(self, params):
        self.__params = params

        self.report = params.__dict__
        self.report['experiment_id'] = self.__get_experiment_id(params)
        self.report['started_at'] = self.__get_timestamp()
        self.logs = []

        logger.info('Experiment settings %s', self.report)

    @property
    def group_id(self):
        return self.__params.group_id

    @property
    def experiment_id(self):
        return self.report['experiment_id']

    def initialize(self):
        # logger.info('Experiment settings %s', self.report)
        pass

    def log_metrics(self, log):
        self.logs.append(log)
        logger.info('log %s', self.logs[-1])

    def finalize(self):
        group_dir = './results/{}/'.format(self.__params.group_id)
        os.makedirs(group_dir, exist_ok=True)

        filename = '{}/{}.pickle'.format(
            group_dir,
            self.report['experiment_id']
        )
        with open(filename, 'wb') as file:
            self.report['finished_at'] = self.__get_timestamp()
            logger.info(
                'Experiment Finished %s',
                [self.report, self.logs[-3:]]
            )

            self.report['logs'] = self.logs
            pickle.dump(self.report, file, pickle.HIGHEST_PROTOCOL)

    def __get_experiment_id(self, args):
        return hashlib.md5(str(args).encode()).hexdigest()

    def __get_timestamp(self):
        return str(time.time()).split('.')[0]
