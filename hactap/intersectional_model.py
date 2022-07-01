from typing import Callable, List

from sklearn.cluster import KMeans, AgglomerativeClustering

from torch.utils.data import TensorDataset, DataLoader


class IntersectionalModel():
    def __init__(
        self,
        method="kmeans",
        n_clusters=4,
        transform: Callable = None
    ) -> None:
        self._method = method
        if method == "kmeans":
            self._model = KMeans(n_clusters=n_clusters)
        else:
            self._model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=method
            )
        self._transform = transform

    def fit(self, train_dataset: TensorDataset) -> None:
        length_dataset = len(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))

        if self._transform is not None:
            x_train = self._transform(x_train)
        self._model.fit_predict(x_train, y_train)
        return

    def predict(self, x_test: List) -> List:
        if self._transform is not None:
            x_test = self._transform(x_test)
        return self._model.labels_.tolist()
