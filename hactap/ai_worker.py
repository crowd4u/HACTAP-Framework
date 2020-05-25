class AIWorker:
    def __init__(self, model, skip_update):
        self.model = model
        self.skip_update = skip_update

    def re_train(self, dataset):
        if self.skip_update:
            return

        self.train(dataset)
        return

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def query(self, x, n_instances):
        return self.model.query(x, n_instances=n_instances)
