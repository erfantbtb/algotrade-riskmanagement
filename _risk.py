import numpy as np
import pandas as pd
from typing import Union
from arch import arch_model
from sklearn.covariance import ShrunkCovariance
from sklearn.covariance import MinCovDet
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS


class Risk:
    def __init__(self, rets: Union[pd.Series, pd.DataFrame]) -> None:
        if isinstance(rets, np.ndarray):
            raise TypeError("The argument rets cannot be numpy array")

        elif isinstance(rets, pd.DataFrame):
            self.rets = rets
            self.num_assets = len(rets.columns)

        elif isinstance(rets, pd.Series):
            self.rets = rets
            self.num_assets = 1

    def historical_risk(self, rets: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.DataFrame]:
        """This function calculate historical or sample risk of an asset or a portfolio of assets 

        Args:
            rets (pd.Series | pd.DataFrame): returns of that stock or portfolio

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.DataFrame]: Sample risk of that stock or portfolio
        """
        if isinstance(rets, pd.DataFrame):
            risk = rets.cov()

        elif isinstance(rets, pd.Series):
            risk = rets.std(ddof=1)

        else:
            raise TypeError("The argument rets cannot be numpy array")

        return risk

    def historical_semi_risk(self,
                             rets: Union[pd.Series, pd.DataFrame],
                             threshold: float = 0) -> Union[float, pd.DataFrame]:
        """Estimating semi covariance matrix using historical semi covariance method

        Args:
            rets (Union[pd.Series, pd.DataFrame]): returns of that stock or portfolio
            threshold (float, optional): Values that are under this threshold. Defaults to 0.

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.DataFrame]: Sample semi risk of that stock or portfolio
        """
        semi_rets = rets.loc[rets <= threshold]

        if isinstance(semi_rets, pd.DataFrame):
            risk = semi_rets.cov()

        elif isinstance(semi_rets, pd.Series):
            risk = semi_rets.std(ddof=1)

        else:
            raise TypeError("The argument rets cannot be numpy array")

        return risk

    def time_varying_risk(self,
                          rets: Union[pd.Series, pd.DataFrame],
                          p: int,
                          q: int,
                          forecast_timestep: int,
                          vol: str = "Garch") -> Union[float, pd.DataFrame]:
        """estimating time varying risk of an stock or portfolio using arch and garch models! 

        Args:
            rets (pd.Series | pd.DataFrame): returns of that stock or portfolio
            p (int): The number of lagged conditional variances to include in the model.
            q (int): The order of lagged conditional variances in the model.
            forecast_timestep (int): future interval for prediction
            vol (str, optional): model for estimation. Defaults to "Garch".

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.DataFrame]: Time varying risk of that stock or portfolio
        """
        if isinstance(rets, pd.DataFrame):
            garch_models = {}
            for asset in rets.columns:
                garch_model = arch_model(rets[asset], vol=vol, p=p, q=q)
                garch_models[asset] = garch_model.fit()

            conditional_variances = {
                asset: model.conditional_volatility for asset, model in garch_models.items()}

            risk = np.zeroes((self.num_assets, self.num_assets))
            for i in range(len(garch_models.keys())):
                model = garch_models.keys[i]
                risk[i, i] = garch_models[model].forecast(
                    start=len(rets), horizon=forecast_timestep).variance.values

        elif isinstance(rets, pd.Series):
            model = arch_model(rets, vol=vol, p=p, q=q)
            model_result = model.fit()
            risk = model_result.forecast(
                start=len(rets), horizon=forecast_timestep).variance.values

        else:
            raise TypeError("The argument rets cannot be numpy array")

        return risk

    def exp_weighted_average(self,
                             rets: Union[pd.Series, pd.DataFrame],
                             decay_rate: float = 0.50,
                             adjust: bool = True) -> Union[float, pd.DataFrame]:
        """calculating risk with EWMA method (give recent values more weight)

        Args:
            rets (pd.Series | pd.DataFrame): returns of that stock or portfolio
            decay_rate (float, optional): how much weight should vary with time. Defaults to 0.90.

        Raises:
            TypeError: _description_

        Returns:
            Union[float, pd.DataFrame]: EWMA risk 
        """
        if isinstance(rets, pd.DataFrame):
            ewma_covariance_matrix = rets.ewm(
                alpha=decay_rate, adjust=adjust).cov()
            risk = ewma_covariance_matrix
            risk.dropna(inplace=True)
            risk = risk.xs(risk.index.levels[0][-1], level='Date')

        elif isinstance(rets, pd.Series):
            ewma_std_matrix = rets.ewm(alpha=decay_rate, adjust=adjust).std()
            risk = ewma_std_matrix
            risk = risk.iloc[-1]

        else:
            raise TypeError("The argument rets cannot be numpy array")

        return risk

    def sklearn_estimation_shrunk(self, rets: pd.DataFrame, weight: float = 0.1) -> pd.DataFrame:
        """Uses Shrunk Covariance estimation for estimating covariance matrix

        Args:
            rets (pd.DataFrame): returns of each stock in portfolio
            weight (float): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Covariance matrix or risk of portfolio
        """
        if isinstance(rets, pd.DataFrame):
            cov = ShrunkCovariance(shrinkage=weight).fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            return risk

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def sklearn_estimation_ledoit(self, rets: pd.DataFrame, weight: float = 0.1) -> pd.DataFrame:
        """Uses ledoit Covariance estimation for estimating covariance matrix

        Args:
            rets (pd.DataFrame): returns of each stock in portfolio
            weight (float): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Covariance matrix or risk of portfolio
        """
        if isinstance(rets, pd.DataFrame):
            cov = LedoitWolf(shrinkage=weight).fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            return risk

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def sklearn_estimation_mincov(self, rets: pd.DataFrame, weight: float = 0.1) -> pd.DataFrame:
        """Uses robust min Covariance estimation for estimating covariance matrix

        Args:
            rets (pd.DataFrame): returns of each stock in portfolio
            weight (float): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Covariance matrix or risk of portfolio
        """
        if isinstance(rets, pd.DataFrame):
            cov = MinCovDet(shrinkage=weight).fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            return risk

        else:
            raise TypeError("The argument rets cannot be numpy array")

    def sklearn_estimation_oracle(self, rets: pd.DataFrame, weight: float = 0.1) -> pd.DataFrame:
        """Uses oracle Covariance estimation for estimating covariance matrix
           assumes returns has gaussian distribution

        Args:
            rets (pd.DataFrame): returns of each stock in portfolio
            weight (float): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: Covariance matrix or risk of portfolio
        """
        if isinstance(rets, pd.DataFrame):
            cov = OAS(shrinkage=weight).fit(rets)
            risk = cov.covariance_
            risk = pd.DataFrame(data=risk, index=rets.columns,
                                columns=rets.columns)
            return risk

        else:
            raise TypeError("The argument rets cannot be numpy array")
