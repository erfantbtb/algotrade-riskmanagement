import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.tsa.stattools import adfuller
import warnings
from typing import Union
import portfolio_management 
from portfolio_management import _distribution_test
warnings.filterwarnings("ignore")


class CheckData:
    def __init__(self, rets: pd.DataFrame) -> None:
        self.rets = rets

    def check_num_samples(self,
                          rets: Union[pd.DataFrame, pd.Series],
                          samples: int = 100) -> bool:
        """Checks amount of data! 

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            samples (int): Number of samples!

        Raises:
            TypeError: _description_

        Returns:
            bool: _description_
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("Returns should be pd.Series or pd.DataFrame!")

        else:
            cond = len(rets) >= samples
            return cond

    def check_stationary(self, rets: Union[pd.DataFrame, pd.Series], level: float = 0.05) -> pd.DataFrame:
        """Check if the series is Stationary or not

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            level (float, optional): _description_. Defaults to 0.05.

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: bool value for each asset in rets DatFrame
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.check_stationary, level=level)

        elif isinstance(rets, pd.Series):
            result = adfuller(rets)
            p_val = result[1]
            cond = p_val < level
            return cond

        else:
            raise TypeError("Returns should be pd.Series or pd.DataFrame!")

    def check_normality(self, rets: Union[pd.DataFrame, pd.Series], level: float = 0.05) -> pd.DataFrame:
        """_summary_

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            level (float, optional): _description_. Defaults to 0.05.

        Returns:
            pd.DataFrame: _description_
        """
        cond = _distribution_test.DistributionTest(rets, level=level).normal_test(rets)
        return cond

    def check_consistency(self, rets: Union[pd.DataFrame, pd.Series], threshold: float = 0.2) -> pd.DataFrame:
        """Checks consistency of data! 

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            threshold (float, optional): _description_. Defaults to 0.2.

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.check_consistency, threshold=threshold)

        elif isinstance(rets, pd.Series):
            cond = np.std(rets) <= threshold
            return cond

        else:
            raise TypeError("Returns should be pd.Series or pd.DataFrame!")

    def check_high_dimension(self, rets: Union[pd.DataFrame, pd.Series], threshold: float = 20) -> bool:
        """Checks if we have more than 20 assets to trade or not! 
           We are checking dimension for this one!

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            threshold (float, optional): _description_. Defaults to 20.

        Raises:
            TypeError: _description_

        Returns:
            bool: _description_
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("Returns should be pd.Series or pd.DataFrame!")

        else:
            return len(rets.columns) >= threshold

    def check_outliers(self, rets: Union[pd.DataFrame, pd.Series], threshold: float = 2) -> pd.DataFrame:
        """Checks if more than 5% of data is considered outlier or not! 

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            threshold (float, optional): _description_. Defaults to 2.

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.check_outliers, threshold=threshold)

        elif isinstance(rets, pd.Series):
            z = np.abs(st.zscore(rets, ddof=1))
            cond = np.sum(z >= threshold)
            cond = cond / len(rets)

            if cond >= 0.05:
                return True
            else:
                return False

        else:
            raise TypeError("Returns should be pd.Series or pd.DataFrame!")

    def __call__(self) -> str:

        if self.check_num_samples(self.rets) >= 100 and self.check_normality(self.rets):
            selected_method = 'historical'

        elif self.check_num_samples(self.rets) >= 100 and not self.check_stationary(self.rets):
            return 'EWMA'
