import numpy as np
import pandas as pd
from _method_test import *
from _measurements import *
from _weight_optimization import *
from _regime_detection import *
from typing import Any, Union


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
        for train_int in range(self.train_low, self.train_high + 1, self.step):
            for test_int in range(self.test_low, self.test_high + 1, self.step):
                print(
                    f"Starting portfolio management system for {train_int} and {test_int}")
                self.train_loop(self.rets,
                                self.method_mu,
                                self.method_cov,
                                self.objective,
                                train_int,
                                test_int)
                print("Calculating and analyzing perforemance of system")
                st = pd.DataFrame(Measurements(
                    self.rets_opt).analyze(self.rets_opt, 1))
                st_eq = pd.DataFrame(Measurements(
                    self.rets_eq).analyze(self.rets_eq, 2))

                df = pd.concat([st, st_eq], axis=1).T
                df.index = ["optimzied_portfolio", "equal_portfolio"]
                df = np.round(df.T, 3)
                self.results.append(df)
                print("--------------------------------")

                self.rets_opt = pd.DataFrame()
                self.rets_eq = pd.DataFrame()

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


class RegimeTimeOptimization:
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
                  rets: pd.DataFrame,
                  bullish_rets: pd.DataFrame,
                  bearish_rets: pd.DataFrame,
                  method_mu: str,
                  method_cov: str,
                  objective: str,
                  transition_matrix: np.ndarray,
                  num_samples: int = 5000) -> pd.DataFrame:
        # simulate bullish returns
        portfo_bull = Portfolio(bullish_rets)
        mu_bull, cov_bull = portfo_bull.portfolio_stats(method_mu, method_cov)
        bullish_rets_simulated = np.random.multivariate_normal(
            mu_bull, cov_bull, num_samples)

        # simulate bearish returns
        portfo_bear = Portfolio(bearish_rets)
        mu_bear, cov_bear = portfo_bear.portfolio_stats(method_mu, method_cov)
        bearish_rets_simulated = np.random.multivariate_normal(
            mu_bear, cov_bear, num_samples)

        # calculate mean and covariance matrix of simulated returns
        mu1, cov1 = bullish_rets_simulated.mean(), bullish_rets_simulated.cov()
        mu2, cov2 = bearish_rets_simulated.mean(), bearish_rets_simulated.cov()

        # creat portfolio using simulated means and covariances
        org_portfolio = Portfolio(rets)
        org_portfolio.scenario_based_stats(
            mu1, mu2, cov1, cov2, transition_matrix)

        w = org_portfolio.optimize_portfolio(objective=objective)
        w_eq = pd.Series(np.ones(len(rets.columns)) /
                         len(rets.columns), index=rets.columns)

        rets_opt = Measurements(rets).portfolio_returns(rets, w)
        rets_eq = Measurements(rets).portfolio_returns(rets, w_eq)

        self.rets_opt = pd.concat([rets_opt, rets_opt], ignore_index=True)
        self.rets_eq = pd.concat([rets_eq, rets_eq], ignore_index=True)
