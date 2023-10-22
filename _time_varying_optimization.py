import numpy as np
import pandas as pd
from _method_test import *
from _measurements import *
from _weight_optimization import *
from _regime_detection import *
from typing import Any, Union
<<<<<<< HEAD
from tqdm.notebook import tqdm, trange
=======
from tqdm import tqdm, trange
>>>>>>> ecd4e6580a83c60a84e84bc794b7e34bb3ac99ab


class RollingTimeOptimization:
    def __init__(self, rets: pd.DataFrame, params: dict) -> None:
        self.rets = rets
        self.train_low = params["train_low"]
        self.train_high = params["train_high"]
        self.test_low = params["test_low"]
        self.test_high = params["test_high"]
        self.step = params["step"]
        self.method_mu = params["method_mu"]
        self.method_cov = params["method_cov"]
        self.objective = params["objective"]
        self.rets_opt = pd.DataFrame()
        self.rets_eq = pd.DataFrame()
        self.results = []

    def one_epoch(self,
                  train: pd.DataFrame,
                  test: pd.DataFrame,
                  method_mu: str,
                  method_cov: str,
                  objective: str) -> pd.DataFrame:
        """Define one epoch of portfolio management loop for finding the best interval! 

        Args:
            train (pd.DataFrame): _description_
            test (pd.DataFrame): _description_
            method_mu (str): _description_
            method_cov (str): _description_
            objective (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        portfo = Portfolio(train)
        portfo.portfolio_stats(method_mu=method_mu, method_cov=method_cov)

        w = portfo.optimize_portfolio(objective=objective)
        w_eq = pd.Series(np.ones(len(test.columns)) /
                         len(test.columns), index=test.columns)

        rets_opt = Measurements(test).portfolio_returns(test, w)
        rets_eq = Measurements(test).portfolio_returns(test, w_eq)

        self.rets_opt = pd.concat([self.rets_opt, rets_opt], ignore_index=True)
        self.rets_eq = pd.concat([self.rets_eq, rets_eq], ignore_index=True)

    def train_loop(self,
                   rets: pd.DataFrame,
                   method_mu: str,
                   method_cov: str,
                   objective: str,
                   train_int: int,
                   test_int: int):
        """Defining one loop for 1 train interval and 1 test interval

        Args:
            rets (pd.DataFrame): _description_
            method_mu (str): _description_
            method_cov (str): _description_
            objective (str): _description_
            train_int (int): _description_
            test_int (int): _description_
        """
        for n in range(int(len(rets) / (test_int))):
            train_start = n * (test_int)
            train_end = train_start + train_int
            test_start = n * (test_int) + train_int
            test_end = test_start + test_int

            train_df = rets[train_start: train_end]
            test_df = rets[test_start: test_end]
            self.one_epoch(train_df,
                           test_df,
                           method_mu=method_mu,
                           method_cov=method_cov,
                           objective=objective)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
<<<<<<< HEAD
        pbar = tqdm(total=len(range(self.test_low, self.test_high + 1, self.step)),
                    desc=f"Test Search Space for {self.method_mu} and {self.method_cov}",
                    leave=False)

        for train_int in tqdm(range(self.train_low, self.train_high + 1, self.step), desc=f"Test Search Space for {self.method_mu} and {self.method_cov}"):
            for test_int in range(self.test_low, self.test_high + 1, self.step):
                pbar.update(1)

=======
        train_range = range(self.train_low, self.train_high + 1, self.step)
        test_range = range(self.test_low, self.test_high + 1, self.step)
        
        pbar_train = tqdm(total=len(train_range), leave=False)
        
        for train_int in train_range:
            pbar_train.set_description(desc=f"Train Search Space: train interval = {train_int}")
            pbar_train.update()
            
            for test_int in test_range:             
>>>>>>> ecd4e6580a83c60a84e84bc794b7e34bb3ac99ab
                self.train_loop(self.rets,
                                self.method_mu,
                                self.method_cov,
                                self.objective,
                                train_int,
                                test_int)
<<<<<<< HEAD

=======
>>>>>>> ecd4e6580a83c60a84e84bc794b7e34bb3ac99ab
                st = pd.DataFrame(Measurements(
                    self.rets_opt).analyze(self.rets_opt, 1))
                st_eq = pd.DataFrame(Measurements(
                    self.rets_eq).analyze(self.rets_eq, 2))

                df = pd.concat([st, st_eq], axis=1).T
                df.index = ["optimzied_portfolio", "equal_portfolio"]
                df = np.round(df.T, 3)
                self.results.append(df)

                self.rets_opt = pd.DataFrame()
                self.rets_eq = pd.DataFrame()
<<<<<<< HEAD
            pbar.reset()
            # pbar.close()

=======
            
        pbar_train.close()
        
>>>>>>> ecd4e6580a83c60a84e84bc794b7e34bb3ac99ab
        return self.results


class ExpandingTimeOptimization:
    def __init__(self, rets: pd.DataFrame, params: dict) -> None:
        self.rets = rets
        self.start = params["start"]
        self.test_low = params["test_low"]
        self.test_high = params["test_high"]
        self.step = params["step"]
        self.method_mu = params["method_mu"]
        self.method_cov = params["method_cov"]
        self.objective = params["objective"]
        self.rets_opt = pd.DataFrame()
        self.rets_eq = pd.DataFrame()
        self.results = []

    def one_epoch(self,
                  train: pd.DataFrame,
                  test: pd.DataFrame,
                  method_mu: str,
                  method_cov: str,
                  objective: str) -> pd.DataFrame:
        """Define one epoch of portfolio management loop for finding the best interval! 

        Args:
            train (pd.DataFrame): _description_
            test (pd.DataFrame): _description_
            method_mu (str): _description_
            method_cov (str): _description_
            objective (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        portfo = Portfolio(train)
        portfo.portfolio_stats(method_mu=method_mu, method_cov=method_cov)

        w = portfo.optimize_portfolio(objective=objective)
        w_eq = pd.Series(np.ones(len(test.columns)) /
                         len(test.columns), index=test.columns)

        rets_opt = Measurements(test).portfolio_returns(test, w)
        rets_eq = Measurements(test).portfolio_returns(test, w_eq)

        self.rets_opt = pd.concat([self.rets_opt, rets_opt], ignore_index=True)
        self.rets_eq = pd.concat([self.rets_eq, rets_eq], ignore_index=True)

    def train_loop(self,
                   rets: pd.DataFrame,
                   method_mu: str,
                   method_cov: str,
                   objective: str,
                   train_int: int,
                   test_int: int):
        """Defining one loop for 1 train interval and 1 test interval

        Args:
            rets (pd.DataFrame): _description_
            method_mu (str): _description_
            method_cov (str): _description_
            objective (str): _description_
            train_int (int): _description_
            test_int (int): _description_
        """
        for n in range(int(len(rets) / (test_int))):
            train_start = n * (test_int)
            train_end = train_start + train_int
            test_start = n * (test_int) + train_int
            test_end = test_start + test_int

            train_df = rets[: train_end]
            test_df = rets[test_start: test_end]
            self.one_epoch(train_df,
                           test_df,
                           method_mu=method_mu,
                           method_cov=method_cov,
                           objective=objective)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        train_int = self.start
        for test_int in range(self.test_low, self.test_high + 1, self.step):
            self.train_loop(self.rets,
                            self.method_mu,
                            self.method_cov,
                            self.objective,
                            train_int,
                            test_int)

            st = pd.DataFrame(Measurements(
                self.rets_opt).analyze(self.rets_opt, 1))
            st_eq = pd.DataFrame(Measurements(
                self.rets_eq).analyze(self.rets_eq, 2))
            df = pd.concat([st, st_eq])
            df.index = ["optimzied_portfolio", "equal_portfolio"]
            df = np.round(df.T, 4)
            self.results.append(df)

        return self.results


# class RegimeTimeOptimization:
#     def __init__(self, rets: pd.DataFrame, params: dict) -> None:
#         self.rets = rets
#         self.train_low = params["train_low"]
#         self.train_high = params["train_high"]
#         self.test_low = params["test_low"]
#         self.test_high = params["test_high"]
#         self.step = params["step"]
#         self.method_mu = params["method_mu"]
#         self.method_cov = params["method_cov"]
#         self.objective = params["objective"]
#         self.rets_opt = pd.DataFrame()
#         self.rets_eq = pd.DataFrame()
#         self.results = []

#     def one_epoch(self,
#                   rets: pd.DataFrame,
#                   bullish_rets: pd.DataFrame,
#                   bearish_rets: pd.DataFrame,
#                   method_mu: str,
#                   method_cov: str,
#                   objective: str,
#                   transition_matrix: np.ndarray,
#                   num_samples: int = 5000) -> pd.DataFrame:
#         # simulate bullish returns
#         portfo_bull = Portfolio(bullish_rets)
#         mu_bull, cov_bull = portfo_bull.portfolio_stats(method_mu, method_cov)
#         bullish_rets_simulated = np.random.multivariate_normal(
#             mu_bull, cov_bull, num_samples)

#         # simulate bearish returns
#         portfo_bear = Portfolio(bearish_rets)
#         mu_bear, cov_bear = portfo_bear.portfolio_stats(method_mu, method_cov)
#         bearish_rets_simulated = np.random.multivariate_normal(
#             mu_bear, cov_bear, num_samples)

#         # calculate mean and covariance matrix of simulated returns
#         mu1, cov1 = bullish_rets_simulated.mean(), bullish_rets_simulated.cov()
#         mu2, cov2 = bearish_rets_simulated.mean(), bearish_rets_simulated.cov()

#         # creat portfolio using simulated means and covariances
#         org_portfolio = Portfolio(rets)
#         org_portfolio.scenario_based_stats(
#             mu1, mu2, cov1, cov2, transition_matrix)

#         w = org_portfolio.optimize_portfolio(objective=objective)
#         w_eq = pd.Series(np.ones(len(rets.columns)) /
#                          len(rets.columns), index=rets.columns)

#         rets_opt = Measurements(rets).portfolio_returns(rets, w)
#         rets_eq = Measurements(rets).portfolio_returns(rets, w_eq)

#         self.rets_opt = pd.concat([rets_opt, rets_opt], ignore_index=True)
#         self.rets_eq = pd.concat([rets_eq, rets_eq], ignore_index=True)
