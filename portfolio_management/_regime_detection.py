import numpy as np 
import pandas as pd 
import cvxpy as cp 
from typing import Union   
import matplotlib.pyplot as plt 
import keras as ks 
from sklearn.base import BaseEstimator 


class RegimeDetection:
    def __init__(self, rets: Union[pd.DataFrame, pd.Series]) -> None:
        self.rets = rets  
        
    def trend_filtering(self,
                        rets: Union[pd.DataFrame, pd.Series], 
                        lambda_value: float) -> Union[pd.DataFrame, pd.Series]:
        """ This module filters trends of a series or dataframe. what it really do is that it 
            smooths serie or dataframe that is given to it.  

        Args:
            rets (Union[pd.DataFrame, pd.Series]): Serie or dataframe that needs to be smoothed. 
            lambda_value (float): penalty value or value for smoothing. 

        Raises:
            TypeError: _description_

        Returns:
            Union[pd.DataFrame, pd.Series]: Smooth value serie or dataframe 
        """
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
                      threshold: float = 1e-5) -> list:
        """ This function calculates indexes of where trend is switched.

        Args:
            betas (Union[pd.DataFrame, pd.Series]): Smooth values for returns! (not price)
            threshold (float, optional): Threshold that checks if trend is changes or not. Defaults to 1e-5.

        Raises:
            TypeError: _description_

        Returns:
            list: indexes that trends are changed
        """
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
        
    def regime_switch_series(self, 
                             betas: Union[pd.Series, pd.DataFrame], 
                             threshold: float = 1e-5) -> Union[pd.Series, pd.DataFrame]:
        """ This function calculates trends by giving them 1 and -1 in order to define 
            bullish trend and bearish trend

        Args:
            betas (Union[pd.Series, pd.DataFrame]): Smooth value of returns
            threshold (float, optional): _description_. Defaults to 1e-5.

        Raises:
            TypeError: _description_

        Returns:
            Union[pd.Series, pd.DataFrame]: return dataframe or series of trends 
        """
        if isinstance(betas, pd.DataFrame):
            return betas.agg(self.regime_switch_series, threshold=threshold)
        
        elif isinstance(betas, pd.Series):
            n = len(betas)
            save = np.zeros(n)
            for i in range(n):
                if (betas.iloc[i]>threshold):
                    save[i] = 1
                else:
                    save[i] = -1
            return save
        
        else: 
            raise TypeError("rets should be either pd.DataFrame or pd.Series!")
        
