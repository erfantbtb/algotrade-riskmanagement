import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

sns.set_theme()
warnings.filterwarnings("ignore")


class Measurements:
    def __init__(self, returns):
        self.returns = returns

    def cumulative_return(self, rets, timeframe: str = "Y"):
        annual_returns = (1 + rets).resample(timeframe).prod() - 1
        cumulative_returns = (annual_returns + 1).cumprod() - 1
        return cumulative_returns

    def last_return(self, rets):
        cum_return = (1 + rets).cumprod() - 1
        return cum_return[-1]

    def convert_timeframe(self, rets: pd.DataFrame, timeframe: str):
        desired_returns = (1 + rets).resample(timeframe).prod() - 1
        return desired_returns

    def volatility(self, rets, freq: float = 1):
        vol = rets.std(ddof=1)
        return vol * np.sqrt(freq)

    def sharpe_ratio(self, rets, rf: float = 0.0, freq: float = 1):
        Rp = rets.mean()
        vol = self.volatility(rets, freq)
        return (Rp - rf) / vol

    def sortino_ratio(self, rets, rf: float = 0.0):
        Rp = rets.mean()
        vol = self.volatility(rets.loc[rets < 0])
        return (Rp - rf) / vol

    def max_drawdown(self, rets):
        cum_returns = (1 + rets).cumprod()
        pre_peaks = cum_returns.cummax()
        drawdowns = (cum_returns - pre_peaks) / pre_peaks
        return abs(drawdowns.min())

    def kurtosis(self, rets):
        return st.kurtosis(rets)

    def skewness(self, rets):
        return st.skew(rets)

    def VaR_historical(self, rets, level: float = 0.05):
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.VaR_historical)

        elif isinstance(rets, pd.Series):
            return -np.percentile(rets, level*100, axis=0)

        else:
            raise TypeError("self.rets should be pandas dataframe!")

    def gaussian_VaR(self, rets, level: float = 0.05):
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.gaussian_VaR)

        elif isinstance(rets, pd.Series):
            z = st.norm.ppf(level)
            return -(rets.mean() + z*rets.std(ddof=0))

        else:
            raise TypeError("self.rets should be pandas dataframe!")

    def nongaussian_VaR(self, rets, level: float = 0.05):
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
            raise TypeError("self.rets should be pandas dataframe!")

    def CVaR(self, rets, VaR_value: float):
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.CVaR)

        elif isinstance(rets, pd.Series):
            cvar_val = - rets[rets < VaR_value].mean()
            return cvar_val

        else:
            raise TypeError("self.rets should be pandas dataframe!")

    def analyze(self, rets, weights: pd.DataFrame) -> pd.DataFrame:
        if isinstance(rets, pd.DataFrame):
            idx = rets.index
            rets = np.dot(rets, weights)
            rets = pd.DataFrame(rets)
            rets.columns = ["Portfolio_returns"]
            rets.index = idx
            return self.analyze(rets.squeeze(), weights)

        elif isinstance(rets, pd.Series):
            statistics = {
                "cum_return": [self.last_return(rets)],
                "max_drawdown": [self.max_drawdown(rets)],
                "volatility": [self.volatility(rets, 1)],
                "sharpe_ratio": [self.sharpe_ratio(rets, freq=1)],
                "sortino_ratio": [self.sortino_ratio(rets)],
                "kurtosis": [self.kurtosis(rets)],
                "skewness": [self.skewness(rets)],
                "var_hist": [self.VaR_historical(rets)],
                "cvar_hist": [self.CVaR(rets, self.VaR_historical(rets))],
                "var_gauss": [self.gaussian_VaR(rets)],
                "cvar_gauss": [self.CVaR(rets, self.gaussian_VaR(rets))],
                "var_nongauss": [self.nongaussian_VaR(rets)],
                "cvar_nongauss": [self.CVaR(rets, self.nongaussian_VaR(rets))]}

            return statistics

        else:
            raise TypeError("self.rets should be pandas dataframe!")
