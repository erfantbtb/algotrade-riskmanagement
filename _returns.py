import numpy as np
import pandas as pd
from typing import Union
from sklearn.covariance import ShrunkCovariance
from sklearn.covariance import MinCovDet
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS


class ExpectedReturns:
    def __init__(self, rets: Union[pd.Series, pd.DataFrame]) -> None:
        if isinstance(rets, np.ndarray):
            raise TypeError("The argument rets cannot be numpy array")

        elif isinstance(rets, pd.DataFrame):
            self.rets = rets
            self.num_assets = len(rets.columns)

        elif isinstance(rets, pd.Series):
            self.rets = rets
            self.num_assets = 1

    def historical_return(self, rets: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.DataFrame]:
        """Estimate expected return using sample mean of an asset or a portfolio

        Args:
            rets (Union[pd.Series, pd.DataFrame]): returns of an asset or a portfolio

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.DataFrame]: Expected Return
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.historical_return)

        elif isinstance(rets, pd.Series):
            expected_return = rets.mean()
            return expected_return

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def exp_weighted_average(self,
                             rets: Union[pd.Series, pd.DataFrame],
                             decay_rate: float = 0.90,
                             adjust: bool = True) -> Union[float, pd.DataFrame]:
        """calculating expected return with EWMA method (give recent values more weight)

        Args:
            rets (pd.Series | pd.DataFrame): returns of that stock or portfolio
            decay_rate (float, optional): how much weight should vary with time. Defaults to 0.90.

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.DataFrame]: EWMA expected return 
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.exp_weighted_average, decay_rate=decay_rate, adjust=adjust)

        elif isinstance(rets, pd.Series):
            ewma_mean_matrix = rets.ewm(alpha=decay_rate, adjust=adjust).std()
            expected_return = ewma_std_matrix
            expected_return = expected_return.iloc[-1]
            return expected_return

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def james_stein_return(self,
                           rets: Union[pd.DataFrame, pd.Series]):
        """Estimate expected return using james stein estimator

        Args:
            rets (Union[pd.DataFrame, pd.Series]): returns of that stock or portfolio

        Raises:
            TypeError: _description_

        Returns:
            _type_: JS expected return
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.james_stein_return)

        elif isinstance(rets, pd.Series):
            n = len(rets)
            max_likelihood_mean = np.mean(rets)
            shrinkage_factor = np.max(
                0, 1 - ((n - 2) * np.var(rets)) / (np.sum((rets - max_likelihood_mean)**2)))
            expected_return = (1 - shrinkage_factor) * max_likelihood_mean
            return expected_return

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def sklearn_estimation_shrunk(self, rets: pd.DataFrame, weight: float = 0.1) -> pd.DataFrame:
        """Uses Shrunk expected return estimation for estimating expected return

        Args:
            rets (pd.DataFrame): returns of each stock in portfolio
            weight (float): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: expected return of portfolio
        """
        if isinstance(rets, pd.DataFrame):
            cov = ShrunkCovariance(shrinkage=weight).fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            expected_return = cov.location_
            return expected_return

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def sklearn_estimation_ledoit(self, rets: pd.DataFrame, weight: float = 0.1) -> pd.DataFrame:
        """Uses ledoit Covariance estimation for estimating Expected return

        Args:
            rets (pd.DataFrame): returns of each stock in portfolio
            weight (float): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Expected return of portfolio
        """
        if isinstance(rets, pd.DataFrame):
            cov = LedoitWolf(shrinkage=weight).fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            expected_return = cov.location_
            return expected_return

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def sklearn_estimation_mincov(self, rets: pd.DataFrame, weight: float = 0.1) -> pd.DataFrame:
        """Uses robust min Covariance estimation for estimating Expected return

        Args:
            rets (pd.DataFrame): returns of each stock in portfolio
            weight (float): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Expected return of portfolio
        """
        if isinstance(rets, pd.DataFrame):
            cov = MinCovDet(shrinkage=weight).fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            expected_return = cov.location_
            return expected_return

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def sklearn_estimation_oracle(self, rets: pd.DataFrame, weight: float = 0.1) -> pd.DataFrame:
        """Uses oracle Covariance estimation for estimating Expected return
           assumes returns has gaussian distribution

        Args:
            rets (pd.DataFrame): returns of each stock in portfolio
            weight (float): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Expected return of portfolio
        """
        if isinstance(rets, pd.DataFrame):
            cov = OAS(shrinkage=weight).fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            expected_return = cov.location_
            return expected_return

        else:
            raise TypeError("The argument rets cannot be numpy array")
