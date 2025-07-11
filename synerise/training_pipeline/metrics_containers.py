from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricContainer(ABC):
    """
    Abstract class for storing metrics about the model's performance.
    Instances of this class are returned by `MetricCalculator`s.
    """

    @abstractmethod
    def compute_weighted_metric(self) -> float:
        """
        Method for computing the final score of a task, based on the metrics stored in
        the `MetricContainer`
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ChurnMetricContainer(MetricContainer):
    """
    Instance of the class `MetricContainer` for storing metrics reported from
    Churn tasks.
    """

    val_auroc: float

    def compute_weighted_metric(self) -> float:
        return self.val_auroc


@dataclass(frozen=True)
class PropensityMetricContainer(MetricContainer):
    """
    Instance of the class `MetricContainer` for storing metrics reported from
    Propensity tasks.
    """

    val_auroc: float
    val_diversity: float
    val_novelty: float

    def compute_weighted_metric(self) -> float:
        return 0.8 * self.val_auroc + 0.1 * self.val_diversity + 0.1 * self.val_novelty
