import abc


class WorkflowAlgorithm(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate(
        self,
        event_type,
        dataset,
        task_assignments,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def assign(
        self,
        task_assignments,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(
        self,
    ) -> str:
        raise NotImplementedError
