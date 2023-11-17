import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from typing import Union

sns.set_theme()
warnings.filterwarnings("ignore")


class Measurements:
    def __init__(self, returns: Union[pd.DataFrame, pd.Series]):
        self.returns = returns

    def average_return(self,
                       rets: Union[pd.Series, pd.DataFrame],
                       freq: float) -> Union[pd.Series, float]:
        """ This function calculates average return for time freq that is given.
            for example if we want annualized data, if our return timeframe is monthly 
            freq should be 12. 

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series 
            freq (float): timeframe that we want average of it

        Returns:
            Union[pd.Series, pd.DataFrame]: average return for timeframe that is given
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        else:
            cum_return = ((1 + rets).cumprod())
            tot_rets = cum_return.iloc[-1] ** (freq / len(rets))

        return tot_rets - 1

    def cumulative_return(self,
                          rets: Union[pd.Series, pd.DataFrame],
                          timeframe: str = "Y") -> Union[pd.Series, pd.DataFrame]:
        """ cumulative return for given rets dataframe or series

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series
            timeframe (str, optional): timeframe for output cumulative returns. Defaults to "Y".

        Returns:
            Union[pd.Series, pd.DataFrame]: cumulative return for given rets 
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        else:
            annual_returns = rets.resample(timeframe).mean()
            cumulative_returns = (annual_returns + 1).cumprod() - 1

        return cumulative_returns

    def last_return(self,
                    rets: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, float]:
        """ This function gives last cumulative return value for given series or 
            dataframe. 

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series

        Returns:
            Union[pd.Series, float]: last cumulative return for given rets 
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        else:
            cum_return = (1 + rets).cumprod() - 1

        return cum_return.iloc[-1]

    def volatility(self,
                   rets: Union[pd.Series, pd.DataFrame],
                   freq: float = 1) -> Union[pd.Series, float]:
        """ This function calculates volatility or standard deviation of stock returns 

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series
            freq (float, optional): TimeFrame that we want to have std based on. Defaults to 1.

        Raises:
            TypeError: _description_

        Returns:
            Union[pd.Series, float]: Standard deviation or volatility of return input
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        else:
            vol = rets.std(ddof=1)
        return vol * np.sqrt(freq)

    def sharpe_ratio(self,
                     rets: Union[pd.Series, pd.DataFrame],
                     rf: float = 0.0,
                     freq: float = 1) -> Union[float, pd.Series]:
        """ This function calculates sharpe ratio of given returns

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series
            rf (float, optional): risk free rate. Defaults to 0.0.
            freq (float, optional): frequency or freq for average and std. Defaults to 1.

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.Series]: Sharpe Ratio of given returns
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        else:
            Rp = self.average_return(rets, freq=freq)
            vol = self.volatility(rets, freq)

        return (Rp - rf) / vol

    def sortino_ratio(self,
                      rets: Union[pd.Series, pd.DataFrame],
                      rf: float = 0.0,
                      freq: float = 1) -> Union[float, pd.Series]:
        """ This function calculates sortino ratio of given returns

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series
            rf (float, optional): risk free rate. Defaults to 0.0.
            freq (float, optional): frequency or freq for average and std. Defaults to 1.

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.Series]: Sortino Ratio of given returns
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        else:
            Rp = self.average_return(rets, freq=freq)
            vol = self.volatility(rets[rets < 0], freq)

        return (Rp - rf) / vol

    def max_drawdown(self,
                     rets: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
        """ This function calculates maximum drawdown of given return

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.Series]: maximum drawdown value
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        else:
            cum_returns = (1 + rets).cumprod()
            pre_peaks = cum_returns.cummax()
            drawdowns = (cum_returns - pre_peaks) / pre_peaks

        return abs(drawdowns.min())

    def kurtosis(self,
                 rets: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
        """ This function calculates kurtosis for given returns

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.Series]: kurtosis for given returns
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        return st.kurtosis(rets)

    def skewness(self,
                 rets: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
        """ This function calculates skewness for given returns

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.Series]: skewness for given returns
        """
        if isinstance(rets, np.ndarray):
            raise TypeError("rets cannot be numpy ndarray.")

        return st.skew(rets)

    def VaR_historical(self,
                       rets: Union[pd.Series, pd.DataFrame],
                       level: float = 0.05) -> Union[float, pd.Series]:
        """ This function calculates value at risk for given confidence level

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series
            level (float, optional): confidence level. Defaults to 0.05.

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.Series]: Value at Risk estimation (historical)
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.VaR_historical)

        elif isinstance(rets, pd.Series):
            return -np.percentile(rets, level*100, axis=0)

        else:
            raise TypeError("rets should be pandas dataframe or series!")

    def gaussian_VaR(self,
                     rets: Union[pd.Series, pd.DataFrame],
                     level: float = 0.05) -> Union[float, pd.Series]:
        """ This function calculates value at risk for given confidence level

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series
            level (float, optional): confidence level. Defaults to 0.05.

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.Series]: Value at Risk estimation (gaussian distribution)
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.gaussian_VaR)

        elif isinstance(rets, pd.Series):
            z = st.norm.ppf(level)
            return -(rets.mean() + z*rets.std(ddof=0))

        else:
            raise TypeError("rets should be pandas dataframe!")

    def nongaussian_VaR(self,
                        rets: Union[pd.Series, pd.DataFrame],
                        level: float = 0.05):
        """ This function calculates value at risk for given confidence level

        Args:
            rets (Union[pd.Series, pd.DataFrame]): return dataframe or series
            level (float, optional): confidence level. Defaults to 0.05.

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.Series]: Value at Risk estimation (nongaussian distribution)
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.nongaussian_VaR)

        elif isinstance(rets, pd.Series):
            s = self.skewness(rets)
            k = self.kurtosis(rets)
            z = st.norm.ppf(level)
            z = z + (z**2 - 1) / 6 * s + 1/24 * (z**3 - 3*z) * \
                (k) - 1/36 * s**2 * (2*z**3 - 5*z)
            return -(rets.mean() + z*rets.std(ddof=0))

        else:
            raise TypeError("rets should be pandas dataframe or series!")

    def CVaR(self,
             rets: Union[pd.Series, pd.DataFrame],
             VaR_value: float) -> Union[pd.Series, float]:
        """ This function calculates conditional value at risk 

        Args:
            rets (Union[pd.Series, pd.DataFrame]): _description_
            VaR_value (float): Value at risk for given returns

        Raises:
            TypeError: _description_

        Returns:
            _type_: conditional value at risk for given return
        """
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.CVaR)

        elif isinstance(rets, pd.Series):
            cvar_val = - rets[rets < VaR_value].mean()
            return cvar_val

        else:
            raise TypeError("rets should be pandas dataframe!")

    def ENC(self, weights: pd.DataFrame) -> float:
        """ Calculate Effective number of assets or diversification

        Args:
            weights (pd.DataFrame): weights of each asset

        Returns:
            float: ENC value
        """
        if isinstance(weights, np.ndarray):
            raise TypeError("weights cannot be numpy ndarray!")

        enc = np.sum((weights)**2)
        return 1 / enc

    def portfolio_returns(self,
                          rets: Union[pd.DataFrame, pd.Series],
                          weights: pd.DataFrame = None) -> pd.Series:
        if isinstance(rets, pd.Series):
            return rets

        elif isinstance(rets, pd.DataFrame):
            # creat portfolio return using weights and each stock returns
            idx = rets.index
            rets = np.dot(rets, weights)
            rets = pd.DataFrame(rets)
            rets.columns = ["portfolio_returns"]
            rets.index = idx
            return rets

        else:
            raise TypeError(
                "rets should be either pandas dataframe or pandas series")

    def analyze(self,
                rets: Union[pd.DataFrame, pd.Series],
                weights: pd.DataFrame) -> pd.DataFrame:
        if isinstance(rets, pd.DataFrame):
            # creat portfolio return using weights and each stock returns
            idx = rets.index
            rets = np.dot(rets, weights)
            rets = pd.DataFrame(rets)
            rets.columns = ["portfolio_returns"]
            rets.index = idx
            return self.analyze(rets.squeeze(), weights)

        elif isinstance(rets, pd.Series):
            statistics = {
                "start_date": [rets.index[0]], 
                "end_date": [rets.index[-1]], 
                "risk_free_rate": [0.0], 
                "Cumulative Return": [self.last_return(rets)],
                "Annualized Return": [self.average_return(rets, freq=252)],
                "Maximum Drawdown": [self.max_drawdown(rets)],
                "Annualized Volatility": [self.volatility(rets, freq=252)],
                "Sharpe Ratio": [self.sharpe_ratio(rets, freq=252)],
                "Sortino Ratio": [self.sortino_ratio(rets, freq=252)],
                "Kurtosis": [self.kurtosis(rets)],
                "Skewness": [self.skewness(rets)],
                "VaR Historical": [self.VaR_historical(rets)],
                "CVaR Historical": [self.CVaR(rets, self.VaR_historical(rets))],
                "VaR Gaussian": [self.gaussian_VaR(rets)],
                "CVaR Gaussian": [self.CVaR(rets, self.gaussian_VaR(rets))],
                "VaR Nongaussian": [self.nongaussian_VaR(rets)],
                "CVaR Nongaussian": [self.CVaR(rets, self.nongaussian_VaR(rets))],
                # "ENC": [self.ENC(weights=weights)]
            }
            statistics_df = pd.DataFrame(statistics)
            statistics_df.index = ["portfolio_stats"]
            return statistics_df.T

        else:
            raise TypeError("self.rets should be pandas dataframe or dataseries!")
