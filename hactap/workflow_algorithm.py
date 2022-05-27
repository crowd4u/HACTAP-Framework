import abc


class WorkflowAlgorithm(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate( # type: ignore
        self,
        event_type,
        dataset,
        task_assignments,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def assign( # type: ignore
        self,
        task_assignments,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(
        self,
    ) -> str:
        raise NotImplementedError
