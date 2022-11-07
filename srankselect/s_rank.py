import math
from enum import Enum, auto
import numpy as np


class RankType(Enum):
    HAMMING = auto()
    EUCLIDEAN = auto()
    MIXED = auto()


class SRank:
    """SRank.
    A class for selecting features for Unsupervised Learning using the
    method described in the following paper:
    https://www.public.asu.edu/~huanliu/papers/pakdd00clu.pdf
    """

    def __init__(
        self,
        data: np.ndarray,
        drop_outliers: bool = True,
        categorical_values: np.ndarray = np.array([]),
        scale_values: bool = True,
        n_bins: int = 10,
        labels: list = [],
    ) -> None:
        """__init__.

        Parameters
        ----------
        data : npt.ArrayLike
            Data of shape {n_samples, n_features}. Categorical data should be encoded ordinally.
        drop_outliers : bool
            Flag for dropping outliers while preprocessing
        categorical_values : npt.ArrayLike
            Array of categorical values indexes (columns).
            If defined the Hamming method will be used to measure similarity,
            otherwise the euclidian method will be used.
        scale_values : bool
            Flag for scaling values using the MinMaxScaler sklearn function.
        n_bins : int
            Number of bins between [3, 50] for discretization of continuous values,
            only used if there are categical values.
        random_state : int
            seed for sampling.
        """

        # Data Validation.
        if not isinstance(data, np.ndarray) or not np.any(data):
            raise ValueError(
                "data must be of numpy ndarray type and must not be empty."
            )

        if data.ndim != 2:
            raise ValueError(
                f"data must be of shape [n_samples, n_features]. Got {data.ndim} dimensions instead."
            )

        # Categorical values array validation
        if not isinstance(categorical_values, np.ndarray):
            raise ValueError(
                "categorical_values must be of numpy ndarray type. Got {type(categorical_values)} instead."
            )

        if (
            categorical_values.ndim > 1
            or categorical_values.shape[0] > data.shape[1]
            or np.any(categorical_values > data.shape[1] - 1)
        ):
            raise ValueError(
                "categorical_values must be of shape {< n_features} and must contain postive integers lower than data n_features."
            )

        # Other attribute validation
        if not isinstance(scale_values, bool):
            raise ValueError(
                f"scale_values must be of boolean type. Got {type(scale_values)} instead."
            )

        if not isinstance(n_bins, int) or n_bins < 3 or n_bins > 50:
            raise ValueError("n_bins must be of integer type between (0, 50].")

        if not isinstance(drop_outliers, bool):
            raise ValueError("value_error must be of boolean type.")

        if not isinstance(labels, list):
            raise ValueError("labels must be a list of shape {n_features}")

        if len(labels) == 0:
            self.labels = [i for i in range(data.shape[1])]

        if len(labels) != data.shape[1]:
            raise ValueError("labels must be a list of shape {n_features}")

        self.data = data
        self.categorical_values = categorical_values
        self.continuous_values = [
            i for i in range(self.data.shape[1]) if i not in categorical_values
        ]
        self.scale_values = scale_values
        self.n_bins = n_bins
        self.drop_outliers = drop_outliers
        self.labels = labels
        self.ranking = {}

        if len(self.categorical_values) == 0:
            self.process_type = RankType.EUCLIDEAN
        elif len(self.continuous_values) == 0:
            self.process_type = RankType.HAMMING
        else:
            self.process_type = RankType.MIXED

        self._preprocess()

    def _preprocess(self) -> None:
        if self.process_type == RankType.HAMMING:
            return
        if self.drop_outliers:
            self._clean_outliers()
        if self.scale_values and self.process_type == RankType.EUCLIDEAN:
            self._scale_values()
        if self.process_type == RankType.MIXED:
            self._discretize()
        return

    def _clean_outliers(self) -> None:
        continuous_data = self.data[:, self.continuous_values].copy()
        median = np.median(continuous_data, axis=0)
        idx = np.where(np.isnan(continuous_data))
        continuous_data[idx] = np.take(median, idx[1])
        q1 = np.quantile(continuous_data, 0.25, axis=0)
        q3 = np.quantile(continuous_data, 0.75, axis=0)
        iqr = q3 - q1
        lower_limit = q1 - (1.5 * iqr)
        upper_limit = q3 + (1.5 * iqr)
        idx = np.where(
            np.logical_or(continuous_data < lower_limit, continuous_data > upper_limit)
        )
        continuous_data[idx] = np.take(median, idx[1])
        self.data[:, self.continuous_values] = continuous_data
        return

    def _scale_values(self) -> None:
        continuous_data = self.data[:, self.continuous_values].copy()
        continuous_data = (continuous_data - continuous_data.min(axis=0)) / (
            continuous_data.max(axis=0) - continuous_data.min(axis=0)
        )
        self.data[:, self.continuous_values] = continuous_data
        return

    def _discretize(self) -> None:
        continuous_data = self.data[:, self.continuous_values].copy()
        bin_edges = np.zeros(continuous_data.shape[1], dtype=object)
        for i in range(continuous_data.shape[1]):
            col_min, col_max = continuous_data[:, i].min(), continuous_data[:, i].max()
            bin_edges[i] = np.linspace(col_min, col_max, self.n_bins + 1)
        for j in range(continuous_data.shape[1]):
            continuous_data[:, j] = np.searchsorted(
                bin_edges[j][1:-1], continuous_data[:, j], side="right"
            )
        self.data[:, self.continuous_values] = continuous_data
        pass

    def fit(self, rank_sample: float = 1.0, rank_iterations: int = 1) -> None:
        if not isinstance(rank_sample, float) or rank_sample > 1 or rank_sample <= 0:
            raise ValueError("rank_sample must be of type float between (0, 1]")
        if (
            not isinstance(rank_iterations, int)
            or rank_iterations < 1
            or rank_iterations > 100
        ):
            raise ValueError("rank iterations must be of type integer between (1, 100]")
        ranking = self._s_rank(rank_sample=rank_sample, rank_iterations=rank_iterations)
        for idx, entropy in enumerate(ranking):
            self.ranking[idx] = {"entropy": entropy, "label": self.labels[idx]}
        return

    def _s_rank(self, rank_sample: float, rank_iterations: int) -> np.ndarray:
        ranking = np.zeros(self.data.shape[1])
        sample_size = np.round(rank_sample * self.data.shape[0], 0)
        for _ in range(rank_iterations):
            selection_idx = np.random.choice(
                self.data.shape[0], size=int(sample_size), replace=False
            )
            sample_array = self.data[selection_idx, :].copy()
            ranking = +self._rank(sample_array)
        return ranking

    def _rank(self, array: np.ndarray) -> np.ndarray:
        ranking = np.zeros(array.shape[1])
        for idx in range(array.shape[1]):
            array_f = np.delete(array, idx, axis=1)
            entropy = self._get_entropy(array_f)
            ranking[idx] += entropy
        return ranking

    def _get_entropy(self, array: np.ndarray) -> float:
        if self.process_type == RankType.EUCLIDEAN:
            dist_matrix = self.matrix_euclidean_distance(array)
            alpha = -math.log(0.5) / dist_matrix.mean()
            sim_matrix = np.exp(-alpha * dist_matrix)
        else:
            sim_matrix = self.matrix_hamming_distance(array)
        eps = 1e-10
        entropy = -1 * np.sum(
            sim_matrix * np.log(sim_matrix + eps)
            + (1 - sim_matrix + eps) * np.log(1 - sim_matrix + eps)
        )
        return entropy

    @staticmethod
    def matrix_euclidean_distance(x1: np.ndarray) -> np.ndarray:
        x2 = x1.reshape(x1.shape[0], 1, x1.shape[1])
        return np.sqrt(
            np.einsum("ijk, ijk -> ij", np.subtract(x1, x2), np.subtract(x1, x2))
        )

    @staticmethod
    def matrix_hamming_distance(x1: np.ndarray) -> np.ndarray:
        x2 = x1.reshape(x1.shape[0], 1, x1.shape[1])
        return np.average((x1 == x2), axis=2)
