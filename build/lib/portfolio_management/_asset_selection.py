import numpy as np 
import pandas as pd 
from typing import Union    

import matplotlib.pyplot as plt 
import matplotlib.pylab as pl 

from sklearn.covariance import GraphicalLasso
from sklearn.covariance import GraphicalLassoCV 
from sklearn.cluster import AffinityPropagation 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
from sklearn import manifold 


class GraphicalAnalysis:
    def __init__(self, rets: pd.DataFrame) -> None:
        if isinstance(rets, pd.DataFrame) or isinstance(rets, pd.Series):
            self.rets = rets 
        else:
            raise TypeError("rets should be pd.DataFrame")
        
    def graphical_lasso_estimation(self, 
                                   rets: pd.DataFrame,
                                   max_iter: int = 1000, 
                                   cv: int = 5, 
                                   alphas: list = [1e-2]) -> pd.DataFrame:
        if isinstance(rets, pd.DataFrame) and len(alphas) == 1:
            edge_model = GraphicalLassoCV(max_iter=max_iter, 
                                          cv=cv, 
                                          alphas=alphas)
            rets_sc = (rets - rets.mean(axis=0)) / rets.std(axis=0)
            edge_model.fit(rets_sc)
            cov_matrix = edge_model.covariance_ 
            prec_matrix = edge_model.precision_ 
        
        else:
            raise ValueError("alpha argument should have only 1 value!")
        
            
