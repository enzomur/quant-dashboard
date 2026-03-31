"""
Base classes for factor calculations.

All factors inherit from BaseFactor and return FactorResult.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class FactorResult:
    """Result from a factor calculation."""
    name: str
    raw_value: float
    normalized_score: float  # Z-score normalized
    percentile_rank: float   # 0-100 percentile
    confidence: str          # "High", "Medium", "Low"

    def __repr__(self):
        return f"FactorResult({self.name}: raw={self.raw_value:.4f}, z={self.normalized_score:.2f}, pct={self.percentile_rank:.1f}%)"


class BaseFactor(ABC):
    """
    Abstract base class for all factors.

    Provides common normalization and winsorization methods.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, *args, **kwargs) -> FactorResult:
        """Calculate the factor value. Must be implemented by subclasses."""
        pass

    @staticmethod
    def normalize_zscore(
        value: float,
        values: Union[pd.Series, np.ndarray, List[float]],
        winsorize_std: float = 3.0
    ) -> float:
        """
        Normalize a value to z-score relative to a distribution.

        Args:
            value: The value to normalize
            values: Distribution of values for computing mean/std
            winsorize_std: Cap z-scores at this many standard deviations

        Returns:
            Z-score (capped at +/- winsorize_std)
        """
        if isinstance(values, list):
            values = np.array(values)
        elif isinstance(values, pd.Series):
            values = values.values

        values = values[~np.isnan(values)]

        if len(values) == 0:
            return 0.0

        mean = np.mean(values)
        std = np.std(values)

        if std == 0 or np.isnan(std):
            return 0.0

        z = (value - mean) / std
        return np.clip(z, -winsorize_std, winsorize_std)

    @staticmethod
    def normalize_percentile(
        value: float,
        values: Union[pd.Series, np.ndarray, List[float]]
    ) -> float:
        """
        Calculate percentile rank of a value within a distribution.

        Args:
            value: The value to rank
            values: Distribution of values

        Returns:
            Percentile rank (0-100)
        """
        if isinstance(values, list):
            values = np.array(values)
        elif isinstance(values, pd.Series):
            values = values.values

        values = values[~np.isnan(values)]

        if len(values) == 0:
            return 50.0

        # Percentile rank
        rank = np.sum(values <= value) / len(values) * 100
        return rank

    @staticmethod
    def winsorize(
        values: Union[pd.Series, np.ndarray],
        lower_pct: float = 1.0,
        upper_pct: float = 99.0
    ) -> np.ndarray:
        """
        Winsorize values at specified percentiles.

        Args:
            values: Array of values
            lower_pct: Lower percentile cutoff
            upper_pct: Upper percentile cutoff

        Returns:
            Winsorized array
        """
        if isinstance(values, pd.Series):
            values = values.values

        lower = np.nanpercentile(values, lower_pct)
        upper = np.nanpercentile(values, upper_pct)

        return np.clip(values, lower, upper)

    @staticmethod
    def minmax_normalize(
        value: float,
        values: Union[pd.Series, np.ndarray],
        invert: bool = False
    ) -> float:
        """
        Min-max normalize a value to 0-1 range.

        Args:
            value: Value to normalize
            values: Distribution
            invert: If True, lower values get higher scores

        Returns:
            Normalized value (0-1)
        """
        if isinstance(values, pd.Series):
            values = values.values

        values = values[~np.isnan(values)]

        if len(values) == 0:
            return 0.5

        min_val = np.min(values)
        max_val = np.max(values)

        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)

        if invert:
            normalized = 1 - normalized

        return np.clip(normalized, 0, 1)

    def _determine_confidence(
        self,
        data_completeness: float,
        sample_size: Optional[int] = None,
        min_sample: int = 60
    ) -> str:
        """
        Determine confidence level based on data quality.

        Args:
            data_completeness: Fraction of required data available (0-1)
            sample_size: Number of observations
            min_sample: Minimum sample for high confidence

        Returns:
            "High", "Medium", or "Low"
        """
        if data_completeness < 0.5:
            return "Low"

        if sample_size is not None:
            if sample_size < min_sample // 2:
                return "Low"
            elif sample_size < min_sample:
                return "Medium"

        if data_completeness >= 0.9:
            return "High"
        elif data_completeness >= 0.7:
            return "Medium"
        else:
            return "Low"
