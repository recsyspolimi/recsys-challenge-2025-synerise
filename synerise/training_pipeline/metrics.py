import numpy as np
import torch
from torchmetrics import Metric as PLMetric


class Diversity(PLMetric):
    """
    Diversity metric

    Metric for calculating average entropy of the probability
    distributions describing the models predictions for the user.

    A high score corresponds to the model recommending all the
    possible targets with similar probability, while a low score
    corresponds to the model recommending specific items.
    """

    def __init__(self, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "output_dim", default=torch.tensor(output_dim, dtype=torch.float32)
        )
        self.add_state(
            "sum_of_entropies",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_of_entropies",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        predictions: torch.Tensor,
    ):
        assert predictions.shape[1] == self.output_dim

        predictions = torch.nn.functional.sigmoid(predictions)
        predictions /= torch.unsqueeze(predictions.sum(axis=-1), dim=-1)
        self.sum_of_entropies -= torch.sum(
            predictions * torch.log2(predictions + 10e-10)
        )
        self.num_of_entropies += predictions.size()[0]

    def compute(self) -> torch.Tensor:
        return self.sum_of_entropies / (
            (self.num_of_entropies + 10e-10) * torch.log2(self.output_dim)
        )


class Novelty(PLMetric):
    """
    Novelty metric

    Metric for calculating the average popularity score of the
    predictions of the model. This is done by taking the k top
    prediction of the model for a given user, and summing the
    popularities of these items in the training data.

    The popularity of an item is its ranking based on the number
    of times that item has been bought in the training data.
    """

    def __init__(self, popularity_data: np.ndarray, k=10, **kwargs):
        super().__init__(**kwargs)
        self.k = min(k, len(popularity_data))

        """
            Note that min value of popularity is the sum of the k smallest
            popularities, all with weight 1. This way, we pre-normalize so
            that the popualrity of this case is exactly 1.
            Upon taking reciprocal, this will yield a maximal novelty score
            of 1.
        """
        max_popularity = np.sum(np.sort(popularity_data)[-self.k :])
        self.add_state(
            "popularity_data", default=torch.tensor(popularity_data / max_popularity)
        )

        self.add_state(
            "sum_of_popularities",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_of_datapoints",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        predictions: torch.Tensor,
    ):
        assert predictions.shape[1] <= len(self.popularity_data)

        predictions = torch.nn.functional.sigmoid(predictions)

        values, indices = torch.topk(predictions, self.k, dim=1)

        self.sum_of_popularities += torch.sum(values * self.popularity_data[indices])
        self.num_of_datapoints += len(predictions)

    def compute(self) -> torch.Tensor:
        """
        Returns one over the average popularity.
        """
        """
            Note that this is already normalized by the normalization of poplularity_data
            at the beginning.
        """

        return (1 - self.sum_of_popularities / (self.num_of_datapoints + 10e-10)) ** 100
