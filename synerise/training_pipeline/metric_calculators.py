import torch
import numpy as np

from torchmetrics import (
    AUROC,
)
from abc import ABC, abstractmethod

from synerise.training_pipeline.metrics_containers import (
    MetricContainer,
    ChurnMetricContainer,
    PropensityMetricContainer,
)
from synerise.training_pipeline.metrics import (
    Novelty,
    Diversity,
)


class MetricCalculator(ABC):
    """
    Class that computes all metrics related to the evaluation of a task. This class
    behaves like a metric in that it takes the predictions and targets from
    the model, and handles computing all the related metrics.
    """

    @abstractmethod
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Method for updating metrics managed by the metric tracker.

        Args:
            self
            predictions (torch.Tensor): predictions of the model
            targets (torch.Tensor): targets that the model is trying to predict.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
    ) -> MetricContainer:
        """
        Method for computing metrics managed by the metric tracker.

        Args:
            self

        Returns:
            MetricContainer : dataclass storing metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def to(
        self,
        device: torch.device,
    ):
        """
        Move evaluation objects to the same device as computation.

        Args:
            self
            device (torch.device): Device which to move the evaluation objects.
        """
        raise NotImplementedError


class ChurnMetricCalculator(MetricCalculator):
    """
    Instance of the abstract `MetricCalculator` class for computing metrics for
    chrun type tasks.
    """

    def __init__(self):
        self.val_auroc = AUROC(task="binary")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.val_auroc.update(predictions, targets)

    def compute(self) -> ChurnMetricContainer:
        auroc = self.val_auroc.compute()
        self.val_auroc.reset()

        return ChurnMetricContainer(val_auroc=auroc.item())

    def to(self, device: torch.device):
        self.val_auroc = self.val_auroc.to(device)


class PropensityMetricCalculator(MetricCalculator):
    """
    Instance of the abstract `MetricCalculator` class for computing metrics for
    propensity type tasks.
    """

    def __init__(self, output_dim: int, popularity_data: np.ndarray):
        self.val_auroc = AUROC(
            task="multilabel",
            num_labels=output_dim,
        )
        self.val_novelty = Novelty(popularity_data=popularity_data)
        self.val_diversity = Diversity(output_dim=output_dim)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.val_auroc.update(predictions, targets)
        self.val_diversity.update(predictions=predictions)
        self.val_novelty.update(predictions=predictions)

    def compute(self) -> PropensityMetricContainer:
        auroc = self.val_auroc.compute()
        self.val_auroc.reset()
        diversity = self.val_diversity.compute()
        self.val_diversity.reset()
        novelty = self.val_novelty.compute()
        self.val_novelty.reset()

        return PropensityMetricContainer(
            val_auroc=auroc.item(),
            val_diversity=diversity.item(),
            val_novelty=novelty.item(),
        )

    def to(self, device: torch.device):
        self.auroc = self.val_auroc.to(device)
        self.val_diversity = self.val_diversity.to(device=device)
        self.val_novelty = self.val_novelty.to(device=device)
