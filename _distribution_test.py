import numpy as np
import pandas as pd
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")


class DistributionTest:
    def __init__(self, returns: pd.DataFrame, level: float = 0.01) -> None:
        self.rets = returns
        self.level = level

    def normal_test(self, rets: pd.DataFrame) -> bool:
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.normal_test)

        elif isinstance(rets, pd.Series):
            _, p_val = st.jarque_bera(rets)

        else:
            raise TypeError("rets are not pandas dataframe")

        return p_val > self.level

    def lognormal_test(self, rets: pd.DataFrame) -> bool:
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.lognormal_test)

        elif isinstance(rets, pd.Series):
            log_normal_data = np.random.lognormal(mean=0, sigma=1, size=1000)
            log_normal_fit_params = st.lognorm.fit(log_normal_data)
            log_normal_dist = st.lognorm(*log_normal_fit_params)
            _, p_val = st.kstest(rets, log_normal_dist.cdf)

        else:
            raise TypeError("rets are not pandas dataframe")

        return p_val > self.level

    def student_test(self, rets: pd.DataFrame) -> bool:
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.student_test)

        elif isinstance(rets, pd.Series):
            _, p_val = st.kstest(rets, 't', args=(len(rets)-1,))

        else:
            raise TypeError("rets are not pandas dataframe")

        return p_val > self.level

    def pareto_test(self, rets: pd.DataFrame) -> bool:
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.pareto_test)

        elif isinstance(rets, pd.Series):
            fit_params = st.pareto.fit(rets)
            pareto_dist = st.pareto(*fit_params)
            _, p_val = st.kstest(rets, pareto_dist.cdf)

        else:
            raise TypeError("rets is not pandas dataframe")

        return p_val > self.level

    def loggamma_test(self, rets: pd.DataFrame) -> bool:
        if isinstance(rets, pd.DataFrame):
            return rets.aggregate(self.pareto_test)

        elif isinstance(rets, pd.Series):
            fit_params = st.loggamma.fit(data)
            loggamma_dist = st.loggamma(*fit_params)
            _, p_val = st.kstest(data, loggamma_dist.cdf)

        else:
            raise TypeError("rets is not pandas dataframe")

        return p_val > self.level
