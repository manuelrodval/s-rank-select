import numbers
import functools
from enum import Enum, auto
import sys
from typing import Union, List, Callable
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype, is_object_dtype
import numpy as np
from numpy.random import RandomState
from scipy.spatial.distance import hamming
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn import datasets


##############################################################################

ComposableFunction = Callable[[DataFrame], None]


def compose(*functions: ComposableFunction) -> ComposableFunction:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


class RankType(Enum):
    HAMMING = auto()
    EUCLIDEAN = auto()


class SRank:
    """A class for selecting features for Unsupervised Learning using the
    method described in the following paper:
    https://www.public.asu.edu/~huanliu/papers/pakdd00clu.pdf
    """

    def __init__(
        self,
        n_samples: int = 35,
        sample_size: float = 0.05,
        n_bins: int = 10,
        drop_non_num: bool = False,
        random_state: Union[int, RandomState] = None,
    ) -> None:
        """__init__ Class initializer
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to take from the data.
        sample_size: float, optional
            Sample size for each run.
        n_bins: int, optional
            Number of bin bucket for discretization process of numerical values
        random_state : Union[int, RandomState], optional
            Seed to generate random selection of data, by default None
        """

        # n_samples
        if n_samples < 35:
            raise ValueError(
                f"Number of samples should be an integer >= 35, got {n_samples} instead."
            )
        self.n_samples = n_samples

        # Sample Size
        if not isinstance(sample_size, float) or sample_size > 1 or sample_size <= 0:
            raise ValueError(
                f"Sample size should be a float between (0, 1.0]. Got {sample_size} instead."
            )
        self.sample_size = sample_size

        # Bins for discretization
        if not isinstance(n_bins, int) or n_bins < 0 or n_bins >= 50:
            raise ValueError(
                f"Bin value should be an integer between [1, 50]. Got {sample_size} instead."
            )
        self.n_bins = n_bins

        # Drop non numerical values
        if not isinstance(drop_non_num, bool):
            raise ValueError(
                f"Drop non numerical values flag should be boolean. \
                Got {drop_non_num} instead."
            )
        self.drop_non_num = drop_non_num

        # random_state => Extracted from sklearn utils
        if random_state is None or random_state is np.random:
            self.random_state = np.random.mtrand._rand
        elif isinstance(random_state, numbers.Integral):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            raise ValueError(
                f"{random_state} cannot be used to random_state a \
                numpy.random.RandomState instance."
            )

        # Construct Pipelines
        self._euclidean_pipeline = compose(
            self._validate_dataframe, self._scale_values, self._sample_rank
        )
        self._hamming_pipeline = compose(
            self._validate_dataframe,
            self._scale_values,
            self._discretize_values,
            self._sample_rank,
        )

    def _validate_dataframe(self, df: DataFrame) -> DataFrame:

        # Validate if input is instance of pd.DataFrame
        if not isinstance(df, DataFrame):
            raise ValueError(
                f"Input data must be pandas.DataFrame instance. Got {type(data)} instead."
            )

        # Validate DataFrame dtypes
        non_numeric = [i for i in df if not is_numeric_dtype(df[i])]
        if non_numeric and not self.drop_non_num:
            raise ValueError(
                f"DataFrame dtypes must be numeric if drop_non_num is False. \
                Got the following non numeric columns: {non_numeric}."
            )
        elif non_numeric and drop_non_num:
            df.drop(labels=non_numeric, axis=1, inplace=True)

        # Clean DataFrame
        df.reset_index(drop=True, inplace=True)
        df.dropna(inplace=True)

        return df

    def _scale_values(self, df: DataFrame) -> DataFrame:
        scaler = MinMaxScaler()
        df_numeric = df.drop(self.cat_cols).copy()
        df_numeric = DataFrame(
            scaler.fit_transform(df_numeric), columns=df_numeric.columns
        )
        return df[self.cat_cols].join(df_numeric)

    def _discretize_values(self, df: DataFrame):
        est = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="uniform")
        df_numeric = df.drop(self.cat_cols).copy()
        df_discrete = DataFrame(
            est.fit_transform(df_numeric), columns=df_numeric.columns
        )
        return df[self.cat_cols].join(df_numeric)

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    @staticmethod
    def hamming_distance(x1, x2):
        return hamming(x1, x2)

    def _sample_rank(self, df: DataFrame) -> None:

        distance_function = (
            self.euclidean_distance
            if self.process_type is RankType.EUCLIDEAN
            else self.hamming_distance
        )

        entropy_rank = {feature: 0 for feature in df.columns}

        # Sample n_sample times from the DataFrame
        for sample in range(self.n_samples):
            sample_df = df.sample(
                frac=self.sample_size,
                ignore_index=True,
                # random_state=self.random_state
            )
            # Iterate over each feature removing it to asses entropy
            for feature in entropy_rank.keys():
                feature_df = sample_df.drop(feature, axis=1).copy()
                # List comprehension to compute cross distance for each row
                distance_matrix = np.array(
                    [
                        [distance_function(r1, r2) for _, r1 in feature_df.iterrows()]
                        for _, r2 in feature_df.iterrows()
                    ]
                )
                if self.process_type is RankType.EUCLIDEAN:
                    alpha = -np.log(0.5) / distance_matrix.mean()
                    sim_matrix = np.exp(-alpha * distance_matrix)
                else:
                    sim_matrix = distance_matrix
                # np.nan_to_num used to avoid inf * 0
                entropy = -np.nansum(
                    (sim_matrix * np.log2(sim_matrix))
                    + ((1 - sim_matrix) * np.log2(1 - sim_matrix))
                )
                # Add entropy to dictionary
                entropy_rank[feature] += entropy
        # Get entropy ranking
        self._rank = Series(entropy_rank).sort_values(ascending=False)

    def fit(self, df: DataFrame, cat_cols: List[str] = None) -> None:
        """fit [summary]

        Parameters
        ----------
        df : DataFrame
            [description]
        cat_cols : Iterable[str], optional
            [description], by default None
        """
        # Validate cat_cols input
        if cat_cols:
            if not isinstance(cat_cols, list):
                raise ValueError(
                    f"Categorical columns should be a list. Got {cat_cols} instead."
                )
            if not all(item in cat_cols for item in df.columns):
                raise ValueError(
                    f"Categorical columns should be part of DataFrame.columns."
                )
            self.cat_cols = cat_cols
        else:
            self.cat_cols = []

        # Assign Process Type:
        ## Hamming for nominal and mixed data.
        ## Euclidean for Numerical only data.
        self.process_type = RankType.HAMMING if cat_cols else RankType.EUCLIDEAN

        # Run Pipelines
        if self.process_type is RankType.EUCLIDEAN:
            self._euclidean_pipeline(df)
        elif self.process_type is RankType.HAMMING:
            self._hamming_pipeline(df)
        else:
            raise ValueError(f"Process type is not defined")

    @property
    def rank(self, slice: int = None) -> Series:
        return self._rank

    def select_best(
        self,
        n_clusters=2,
    ) -> None:
        pass

    @property
    def best_labels(self) -> DataFrame:
        return self._best


def main():
    # import some data to play with
    iris = datasets.load_iris()
    df = DataFrame(iris.data)

    # Ranker
    ranker = SRank()
    ranker.fit(df)


if __name__ == "__main__":
    main()
