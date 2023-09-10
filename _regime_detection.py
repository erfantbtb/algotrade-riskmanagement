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
        
    def plot_regime_color(self, dataset, regime_num=0, TR_num=1, lambda_value=16, log_TR = True):
        returns = dataset.iloc[:,regime_num]
        TR = dataset.iloc[:,TR_num]
        betas = self.trend_filtering(returns.values,lambda_value)
        regimelist = self.regime_switch(betas)
        curr_reg = np.sign(betas[0]-1e-5)
        y_max = np.max(TR) + 500
        
        if log_TR:
            fig, ax = plt.subplots()
            for i in range(len(regimelist)-1):
                if curr_reg == 1:
                    ax.axhspan(0, y_max+500, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                        facecolor='green', alpha=0.3)
                else:
                    ax.axhspan(0, y_max+500, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                        facecolor='red', alpha=0.5)
                curr_reg = -1 * curr_reg
            
            fig.set_size_inches(12,9)   
            plt.plot(TR, label='Total Return')
            plt.ylabel('SP500 Log-scale')
            plt.xlabel('Year')
            plt.yscale('log')
            plt.xlim([dataset.index[0], dataset.index[-1]])
            plt.ylim([80, 3000])
            plt.yticks([100, 500, 1000, 2000, 3000],[100, 500, 1000, 2000, 3000])
            plt.title('Regime Plot of SP 500', fontsize=24)
            plt.show()
            
        
        
