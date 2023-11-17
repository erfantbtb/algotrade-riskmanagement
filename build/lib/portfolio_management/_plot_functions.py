import numpy as np 
import pandas as pd 
from portfolio_management._measurements import * 
from portfolio_management._weight_optimization import * 
import matplotlib.pyplot as plt 
import seaborn as sns 
from typing import Union
sns.set_theme()


class Plotting:
    def __init__(self, rets: pd.DataFrame) -> None:
        if isinstance(rets, pd.DataFrame):
            self.rets = rets 
        else:
            raise TypeError("rets should be pandas dataframe only.")
        
    def plot_efficient_frontier(self,
                                num_simulations: int = 100000, 
                                show_optimum_portfolios: bool = True) -> None:
        num_stocks = len(self.rets.columns)
        mean_returns = self.rets.mean()
        cov_matrix = self.rets.cov()
        results = np.zeros((3, num_simulations))

        for i in range(num_simulations):
            weights = np.random.random(num_stocks)
            weights /= np.sum(weights)

            # Calculate portfolio statistics
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            portfolio_sharpe_ratio = portfolio_return / portfolio_std_dev

            # Store portfolio results
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std_dev
            results[2, i] = portfolio_sharpe_ratio
            
        # Find the portfolio with minimum risk
        min_risk_idx = np.argmin(results[1])
        min_risk_return = results[0, min_risk_idx]
        min_risk_std_dev = results[1, min_risk_idx]

        # Find the portfolio with maximum Sharpe ratio
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_return = results[0, max_sharpe_idx]
        max_sharpe_std_dev = results[1, max_sharpe_idx]
        
        # Equally weighted portfolio
        equal_weights = np.ones(num_stocks) / num_stocks
        equal_return = np.dot(equal_weights, mean_returns)
        equal_std_dev = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
        
        # Plot individual stocks
        plt.figure(figsize=(11, 7))
        for i in range(num_stocks):
            plt.scatter(np.sqrt(cov_matrix.iloc[i, i]), 
                        mean_returns[i], 
                        marker='o',
                        color='b')
            plt.annotate(self.rets.columns[i], 
                         (np.sqrt(cov_matrix.iloc[i, i]), 
                          mean_returns[i]),
                         xytext=(10, -10),
                         textcoords='offset points', 
                         ha='center')

        # Plot efficient frontier
        plt.scatter(results[1], results[0], marker='x', c=results[0]/results[1], label='Portfolios')
        
        plt.scatter(min_risk_std_dev, 
                    min_risk_return,
                    marker='o', 
                    s=70, 
                    color='g',
                    label='Minimum Risk')
        plt.annotate("GMV", 
                     (min_risk_std_dev,
                      min_risk_return), 
                     xytext=(10, -10),
                     textcoords='offset points', 
                     ha='center')
        
        plt.scatter(max_sharpe_std_dev,
                    max_sharpe_return,
                    s=70,
                    marker='o',
                    color='g',
                    label='Maximum Sharpe Ratio')
        plt.annotate("MSR", 
                     (max_sharpe_std_dev,
                      max_sharpe_return), 
                     xytext=(10, -10),
                     textcoords='offset points', 
                     ha='center')
        
        plt.scatter(equal_std_dev,
                    equal_return,
                    s=70,
                    marker='o', 
                    color='black',
                    label='Equally Weighted')
        plt.annotate("EW", 
                     (equal_std_dev,
                      equal_return), 
                     xytext=(10, -10),
                     textcoords='offset points', 
                     ha='center')
        
        plt.xlabel('Standard Deviation')
        plt.ylabel('Returns')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.show()
        
    def plot_weights(self, weights: Union[pd.DataFrame, pd.Series]) -> None:
        labels = self.rets.columns 
        
        if isinstance(weights, pd.DataFrame) or isinstance(weights, pd.Series):
            weights = weights.to_numpy()
            weights = weights.reshape(-1, )
            
        else: 
            weights = weights.reshape(-1, )
        
        plt.pie(weights,
                labels=labels,
                autopct='%1.1f%%',
                pctdistance=1.13, 
                labeldistance=.4)
        plt.title("Weights Pie Chart")
        plt.show()
        
    def plot_cum_return(self, 
                        rets: Union[pd.DataFrame, pd.Series], 
                        ):
       pass
            
        
