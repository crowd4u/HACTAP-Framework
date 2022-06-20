from typing import Callable, List

from hactap.ai_worker import BaseModel

from torch.utils.data import TensorDataset, DataLoader


class IntersectionalModel():
    def __init__(
        self,
        model: BaseModel,
        transform: Callable = None
    ) -> None:
        self._model = model
        self._transform = transform

    def fit(self, train_dataset: TensorDataset) -> None:
        length_dataset = len(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=length_dataset
        )
        x_train, y_train = next(iter(train_loader))

        if self._transform is not None:
            print("transform data")
            x_train = self._transform(x_train)
        print("x_train\n", type(x_train))
        self._model.fit(x_train, y_train)
        return

    def predict(self, x_test: List) -> List:
        return self._model.predict(x_test)
