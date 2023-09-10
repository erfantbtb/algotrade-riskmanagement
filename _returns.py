import numpy as np
import pandas as pd
from typing import Union

from sklearn.covariance import ShrunkCovariance
from sklearn.covariance import MinCovDet
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_validate


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
            ewma_mean_matrix = rets.ewm(alpha=decay_rate, adjust=adjust).mean()
            expected_return = ewma_mean_matrix
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
            expected_return = pd.DataFrame(data=[expected_return],
                                           columns=rets.columns).squeeze()
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
            cov = LedoitWolf().fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            expected_return = cov.location_
            expected_return = pd.DataFrame(data=[expected_return],
                                           columns=rets.columns).squeeze()
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
            cov = MinCovDet().fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            expected_return = cov.location_
            expected_return = pd.DataFrame(data=[expected_return],
                                           columns=rets.columns).squeeze()
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
            cov = OAS().fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            expected_return = cov.location_
            expected_return = pd.DataFrame(data=[expected_return],
                                           columns=rets.columns).squeeze()
            return expected_return

        else:
            raise TypeError("The argument rets cannot be numpy array")
        
    def CAPM(self, 
             rets: Union[pd.DataFrame, pd.Series],
             market_returns: pd.DataFrame, 
             rf: float = 0.0,
             fit_intercept: bool = False) -> pd.DataFrame:
        """Single factor model CAPM

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            market_returns (pd.DataFrame): _description_
            rf (float, optional): _description_. Defaults to 0.0.
            fit_intercept (bool, optional): _description_. Defaults to False.

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Expected return! 
        """
        if isinstance(rets, pd.DataFrame):
            rets = rets - rf
            return rets.aggregate(self.CAPM, market_returns, rf, fit_intercept)
        
        elif isinstance(rets, pd.Series):
            lr = LinearRegression(fit_intercept=fit_intercept)
            lr.fit(market_returns, rets)
            beta = lr.coef_ 
            expected_returns = (beta * market_returns).mean()
            return expected_returns 
        
        else: 
            raise TypeError("rets should be either pd.DataFrame or pd.Series")
        
    def multi_factor(self, 
                     rets: Union[pd.DataFrame, pd.Series],
                     factors: pd.DataFrame, 
                     fit_intercept: bool = False,
                     rf: float = 0.0) -> pd.DataFrame:
        """Multi factor models using simple linear regression

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            factors (pd.DataFrame): _description_
            fit_intercept (bool, optional): _description_. Defaults to False.
            rf (float, optional): _description_. Defaults to 0.0.

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Expected return! 
        """
        if isinstance(rets, pd.DataFrame):
            rets = rets - rf
            return rets.aggregate(self.CAPM, factors, fit_intercept, rf)
        
        elif isinstance(rets, pd.Series):
            lr = LinearRegression(fit_intercept=fit_intercept)
            lr.fit(factors, rets)
            beta = lr.coef_ 
            expected_returns = (beta * factors).mean()
            return expected_returns 
        
        else: 
            raise TypeError("rets should be either pd.DataFrame or pd.Series")
        
    def multi_factor_regularize(self, 
                     rets: Union[pd.DataFrame, pd.Series],
                     factors: pd.DataFrame, 
                     fit_intercept: bool = False,
                     rf: float = 0.0, 
                     method: str = "lasso") -> pd.DataFrame:
        """Multi factor models using lasso and ridge!

        Args:
            rets (Union[pd.DataFrame, pd.Series]): _description_
            factors (pd.DataFrame): _description_
            fit_intercept (bool, optional): _description_. Defaults to False.
            rf (float, optional): _description_. Defaults to 0.0.
            method (str, optional): _description_. Defaults to "lasso".

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Expected return! 
        """
        if isinstance(rets, pd.DataFrame):
            rets = rets - rf
            return rets.aggregate(self.CAPM, factors, fit_intercept, rf, method)
        
        elif isinstance(rets, pd.Series):
            if method == "lasso":
                lr = LassoCV(cv=5, fit_intercept=fit_intercept)
                
            elif method == "ridge":
                lr = RidgeCV(cv=5, fit_intercept=fit_intercept)
                
            else:
                raise ValueError("Only possible methods are lasso and ridge")
                
            lr.fit(factors, rets)
            beta = lr.coef_ 
            expected_returns = (beta * factors).mean()
            return expected_returns 
        
        else: 
            raise TypeError("rets should be either pd.DataFrame or pd.Series")
        
         
            
