import numpy as np
import pandas as pd
import cvxpy as cv
from scipy.optimize import minimize
from _risk import *
from _returns import *


class Portfolio:
    def __init__(self,
                 rets: pd.DataFrame,
                 risk_free_rate: float = 0.0,
                 ) -> None:
        self.rets = rets
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(rets.columns)
        self.bounds = [(0, 1) for _ in range(self.num_assets)]
        self.constraints = (
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1})
        self.initial_weights = np.ones(self.num_assets) / self.num_assets

        self.ret_obj = ExpectedReturns(self.rets)
        self.risk_obj = Risk(self.rets)
        self.methods_mu_dict = {"historical": self.ret_obj.historical_return,
                                "EWMA": self.ret_obj.exp_weighted_average,
                                "JS": self.ret_obj.james_stein_return,
                                "shrunk": self.ret_obj.sklearn_estimation_shrunk,
                                "lediot": self.ret_obj.sklearn_estimation_ledoit,
                                "mincovdet": self.ret_obj.sklearn_estimation_mincov,
                                "oracle": self.ret_obj.sklearn_estimation_oracle, }

        self.methods_cov_dict = {"historical": self.risk_obj.historical_risk,
                                 "EWMA": self.risk_obj.exp_weighted_average,
                                 "semi": self.risk_obj.historical_semi_risk,
                                 "shrunk": self.risk_obj.sklearn_estimation_shrunk,
                                 "lediot": self.risk_obj.sklearn_estimation_ledoit,
                                 "mincovdet": self.risk_obj.sklearn_estimation_mincov,
                                 "oracle": self.risk_obj.sklearn_estimation_oracle, }

    def portfolio_stats(self,
                        method_mu: str = "historical",
                        method_cov: str = "historical") -> None:
        """In this function covariance matrix and expected returns or 
           mean vectors are calculated to use in optimization

        Args:
            method_mu (str, optional): _description_. Defaults to "historical".
            method_cov (str, optional): _description_. Defaults to "historical".

        Raises:
            NotImplemented: _description_

        Returns:
            _type_: _description_
        """
        try:
            self.expected_returns = self.methods_mu_dict[method_mu](self.rets)
            self.cov_matrix = self.methods_cov_dict[method_cov](self.rets)

        except:
            raise ValueError("Method that you want is not available!")

    def _negative_sharpe_ratio(self, weights: Union[pd.DataFrame, np.ndarray]) -> float:
        """negative of Sharpe ratio as objective function

        Args:
            weights (Union[pd.DataFrame, np.ndarray]): _description_

        Returns:
            float: -sharpe_ratio
        """
        mean = np.sum(weights * self.expected_returns)
        volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (mean - self.risk_free_rate) / volatility
        return -sharpe_ratio

    def _negative_return(self, weights) -> float:
        """negative of expected return as objective function

        Args:
            weights (_type_): _description_

        Returns:
            float: _description_
        """
        return -np.sum(weights * self.expected_returns)

    def _portfolio_volatility(self, weights) -> float:
        """Volatility of portfolio as objective function

        Args:
            weights (_type_): _description_

        Returns:
            float: _description_
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def _portfolio_utility(self, weights, gamma: float = 0.01) -> float:
        mean = np.sum(weights * self.expected_returns)
        volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        utility = (mean - self.risk_free_rate) - gamma * volatility
        return -utility

    def optimize_portfolio(self, objective: str):
        """Optimize portfolio based on objective that you want

        Args:
            objective (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        if objective == 'min_risk':
            result = minimize(self._portfolio_volatility,
                              self.initial_weights,
                              method='SLSQP',
                              bounds=self.bounds,
                              constraints=self.constraints)

        elif objective == 'max_sharpe':
            result = minimize(self._negative_sharpe_ratio,
                              self.initial_weights,
                              method='SLSQP',
                              bounds=self.bounds,
                              constraints=self.constraints)

        elif objective == 'max_return':
            result = minimize(self._negative_return,
                              self.initial_weights,
                              method='SLSQP',
                              bounds=self.bounds,
                              constraints=self.constraints)

        elif objective == 'max_utility':
            result = minimize(self._portfolio_utility,
                              self.initial_weights,
                              method='SLSQP',
                              bounds=self.bounds,
                              constraints=self.constraints)
        else:
            raise ValueError(
                "Invalid objective. Supported objectives are 'min_risk', 'max_sharpe', and 'max_return'.")

        optimized_weights = pd.Series(
            result.x, index=self.expected_returns.index, name='Weights')
        optimized_return = np.sum(result.x * self.expected_returns)
        optimized_volatility = np.sqrt(
            np.dot(result.x.T, np.dot(self.cov_matrix, result.x)))
        self.result = result

        return np.round(optimized_weights, 3), optimized_return, optimized_volatility
