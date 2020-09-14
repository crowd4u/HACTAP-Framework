import torch


class AIWorker:
    def __init__(self, model):
        self.model = model

    def fit(self, train_dataset):
        length_dataset = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))

        self.model.fit(x_train, y_train)
        return

    def predict(self, x_test):
        return self.model.predict(x_test)

    def query(self, x_test, n_instances=None):
        query_indexes, samples = self.model.query(
            x_test, n_instances=n_instances
        )
        # print('query_indexes', query_indexes, samples)
        return query_indexes

    def get_worker_name(self):
        return self.model.__class__.__name__


class ComitteeAIWorker(AIWorker):
    def __init__(self, model):
        super().__init__(
            model
        )

    def fit(self, train_dataset):
        length_dataset = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))
        self.model.teach(x_train, y_train, only_new=True)
        return


class AIWorker2:
    def __init__(self, model, skip_update=False):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        print('device', self.device, self.use_cuda)
        self.model = model

        self.skip_update = skip_update
        self.trained = False

    def fit(self, train_dataset):
        print('device', self.device, self.use_cuda)
        print(self.model.__class__.__name__)

        if self.model.__class__.__name__ == 'MindAIWorker':
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=len(train_dataset)
            )
            x, y = next(iter(train_dataloader))
            print('y', y)
            self.model.fit(x, y)
            return

        if self.skip_update and self.trained:
            # logger.warning(
            #     'The training was skipped ({}).'.format(
            #         self.model.estimator.__class__.__name__
            #     )
            # )
            return self.model.estimator

        self.model.fit(train_dataset, y=None)

        self.trained = True
        return

    def predict(self, x):
        return self.model.predict(x)

    def query(self, x, n_instances):
        return self.model.query(x, n_instances=n_instances)
