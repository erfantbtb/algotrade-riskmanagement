import numpy as np
import pandas as pd
import cvxpy as cv
from scipy.optimize import minimize
from _risk import *
from _returns import *
from typing import List


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
                                "ledoit": self.ret_obj.sklearn_estimation_ledoit,
                                "mincovdet": self.ret_obj.sklearn_estimation_mincov,
                                "oracle": self.ret_obj.sklearn_estimation_oracle, }

        self.methods_cov_dict = {"historical": self.risk_obj.historical_risk,
                                 "EWMA": self.risk_obj.exp_weighted_average,
                                 "semi": self.risk_obj.historical_semi_risk,
                                 "shrunk": self.risk_obj.sklearn_estimation_shrunk,
                                 "ledoit": self.risk_obj.sklearn_estimation_ledoit,
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
            
            return self.expected_returns, self.cov_matrix

        except:
            raise NotImplementedError("Method that you want is not available yet!")
        
    def scenario_based_stats(self,
                             mu_bullish: pd.DataFrame, 
                             mu_bearish: pd.DataFrame, 
                             cov_bullish: pd.DataFrame, 
                             cov_bearish: pd.DataFrame, 
                             transition_matrix: np.ndarray):
        """This function is used when we want to have scenario based portfolio optimization
            and for now it uses 2 scenarios 1 is bullish and other one is bearish.

        Args:
            mu_bullish (pd.DataFrame): _description_
            mu_bearish (pd.DataFrame): _description_
            cov_bullish (pd.DataFrame): _description_
            cov_bearish (pd.DataFrame): _description_
            transition_matrix (np.ndarray): _description_

        Raises:
            TypeError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        try: 
            self.expected_returns = transition_matrix[0] * mu_bullish + transition_matrix[1] * mu_bearish
            self.cov_matrix = transition_matrix[0] * cov_bullish + transition_matrix[1] * cov_bearish
            
            return self.expected_returns, self.cov_matrix
        
        except:
            if not isinstance(transition_matrix, np.ndarray):
                raise TypeError("Transition matrix should be numpy array")
            
            elif transition_matrix.shape != (1, 2):
                raise ValueError("Shape of transition matrix should be (1, 2)") 
                

    def black_litterman_stats(self, ):
        pass

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

    def _negative_return(self, weights: Union[pd.DataFrame, np.ndarray]) -> float:
        """negative of expected return as objective function

        Args:
            weights (: Union[pd.DataFrame, np.ndarray]): _description_

        Returns:
            float: _description_
        """
        return -np.sum(weights * self.expected_returns)

    def _portfolio_volatility(self, 
                              weights: Union[pd.DataFrame, np.ndarray],
                              r_bar=None) -> float:
        """Volatility of portfolio as objective function

        Args:
            weights (: Union[pd.DataFrame, np.ndarray]): _description_

        Returns:
            float: _description_
        """
        if r_bar == None:
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def _portfolio_utility(self, 
                           weights: Union[pd.DataFrame, np.ndarray], 
                           gamma: float = 0.01) -> float:
        """ This objective is simillar to sharpe ratio but instead of dividing it substracts 
            expected return and risk with coefficient of gamma

        Args:
            weights (Union[pd.DataFrame, np.ndarray]): _description_
            gamma (float, optional): _description_. Defaults to 0.01.

        Returns:
            float: _description_
        """
        mean = np.sum(weights * self.expected_returns)
        volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        utility = (mean - self.risk_free_rate) - gamma * volatility
        return -utility

    def _portfolio_parity(self, weights: Union[pd.DataFrame, np.ndarray]) -> float:
        """ This objective function is simillar to GMV objective but instead of minimizing risk 
            based on money it minimizes based on the risk itself and risk of each asset will be 
            same as other one! 

        Args:
            weights (Union[pd.DataFrame, np.ndarray]): _description_

        Returns:
            float: _description_
        """
        portfolio_variance = np.dot(
            weights.T, np.dot(self.cov_matrix, weights))
        asset_contributions = weights * \
            np.dot(self.cov_matrix, weights) / portfolio_variance
        target_risk_contributions = np.ones(len(weights)) / len(weights)
        rc_diff = asset_contributions - target_risk_contributions
        return np.sum(rc_diff**2)

    def optimize_portfolio(self, 
                           objective: str, 
                           custom_constraints: List[dict] = None) -> pd.DataFrame:
        """Optimize portfolio based on objective that you want

        Args:
            objective (str): objective of your porfolio
            custom_constraints (List[dict], optional): constraints that you want to put on weights. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: optimized weight of each asset in dataframe.
        """
        if custom_constraints != None:
            for constraints in custom_constraints:
                self.constraints.append(constraints)
            
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

        elif objective == "risk_parity":
            result = minimize(self._portfolio_parity,
                              self.initial_weights,
                              method='SLSQP',
                              bounds=self.bounds,
                              constraints=self.constraints)
        else:
            objective_list = ['min_risk', 'max_sharpe',  'max_return', 'risk_parity', 'max_utility']
            raise ValueError(f"Invalid objective. Supported objectives are: {objective_list} .")

        optimized_weights = pd.Series(result.x,
                                      index=self.expected_returns.index,
                                      name='Weights')
        optimized_return = np.sum(result.x * self.expected_returns)
        optimized_volatility = np.sqrt(
            np.dot(result.x.T, np.dot(self.cov_matrix, result.x)))
        self.result = result

        return np.round(optimized_weights, 3)
