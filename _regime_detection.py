import numpy as np 
import pandas as pd 
import cvxpy as cp 
from typing import Union   
from scipy.optimize import minimize
import matplotlib.pyplot as plt 


class RegimeDetection:
    def __init__(self, rets: Union[pd.DataFrame, pd.Series]) -> None:
        self.rets = rets  
        
    def trend_filtering(self,
                        rets: Union[pd.DataFrame, pd.Series], 
                        lambda_value: float) -> Union[pd.DataFrame, pd.Series]:
        if isinstance(rets, pd.DataFrame):
            return rets.agg(self.trend_filtering, lambda_value=lambda_value)
        
        elif isinstance(rets, pd.Series):
            rets = rets.to_numpy()
            n = np.size(rets)
            x_ret = rets.reshape(n)

            Dfull = np.diag([1]*n) - np.diag([1]*(n-1),1)
            D = Dfull[0:(n-1),]

            beta = cp.Variable(n)
            lambd = cp.Parameter(nonneg=True)

            def tf_obj(x,beta,lambd):
                return cp.norm(x-beta,2)**2 + lambd*cp.norm(cp.matmul(D, beta),1)

            problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))

            lambd.value = lambda_value
            problem.solve()

            return beta.value
        
        else: 
            raise TypeError("rets should be either pd.DataFrame or pd.Series!")
    
    def regime_switch(self, 
                      betas: Union[pd.DataFrame, pd.Series],
                      threshold: float = 1e-5):
        if isinstance(betas, pd.DataFrame):
            return betas.agg(self.regime_switch, threshold=threshold)
        
        elif isinstance(betas, pd.Series):
            n = len(betas)
            init_points = [0]
            curr_reg = (betas[0]>threshold)
            for i in range(n):
                if (betas[i]>threshold) == (not curr_reg):
                    curr_reg = not curr_reg
                    init_points.append(i)
            init_points.append(n)
            return init_points
        
        else: 
            raise TypeError("rets should be either pd.DataFrame or pd.Series!")
        
    def regime_switch_series(self, betas, threshold=1e-5):
        if isinstance(betas, pd.DataFrame):
            return betas.agg(self.regime_switch_series, threshold=threshold)
        
        elif isinstance(betas, pd.Series):
            n = len(betas)
            save = np.zeros(n)
            for i in range(n):
                if (betas[i]>threshold):
                    save[i] = 1
                else:
                    save[i] = -1
            return save
        
        else: 
            raise TypeError("rets should be either pd.DataFrame or pd.Series!")
        
    def simulate_returns(self,
                         mu1: pd.DataFrame, 
                         mu2: pd.DataFrame,
                         cov1: pd.DataFrame, 
                         cov2: pd.DataFrame, 
                         p11: pd.DataFrame,
                         p22: pd.DataFrame,
                         n: int) -> pd.DataFrame:
        s_1 = np.random.multivariate_normal(mu1, cov1, n).T
        s_2 = np.random.multivariate_normal(mu2, cov2, n).T
        regime = np.ones(n)
        for i in range(n-1):
            if regime[i] == 1:
                if np.random.rand() <= p11:
                    regime[i+1] = 1
                else:
                    regime[i+1] = 2
            else:
                if np.random.rand() <= p22:
                    regime[i+1] = 2
                else:
                    regime[i+1] = 1
        return (regime*s_1 + (1-regime)*s_2).T
            
        
        
