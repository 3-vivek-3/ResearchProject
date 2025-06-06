from abc import ABC, abstractmethod
from src.Environments import AbstractEnvironment
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
import numpy as np

class AbstractLearner(ABC):
    """
    Attributes:
        T: Horizon
        d: ambient dimension
        history: A list of (action, reward) tuples, updated per round
        p: the number of warmup rounds before selecting features
        k: number of features to be selected after warmup rounds
    """
    def __init__(self, T : int, d: int, params : dict):
        self.T = T
        self.history = []

        if "featureSelector" not in params.keys():
            self.featureSelector = "anovaF"
        else:
            self.featureSelector = params["featureSelector"]

        self.p: int = params.get("p")
        assert 0 < self.p <= T

        self.k: int = params.get("k")
        assert 0 < self.k <= d

    @abstractmethod
    def run(self, env : AbstractEnvironment, logger):
        pass

    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def total_reward(self):
        pass

    @abstractmethod
    def cum_reward(self):
        pass
    
    '''
    Method that performs feature selection and returns the indices of the
    selected features. The feature selection technique that is chosen 
    depends on the featureSelector field in params.
    '''
    def selectKFeatures(self, X, y, k):
        if self.featureSelector == "anovaF":
            return self.anovaF(X, y, k)
        elif self.featureSelector == "rfe":
            return self.rfe(X, y, k)
        else:
            raise ValueError("Unknown feature selector. Use 'anovaF' or 'rfe'")
        
    # score_func=f_classif for classification problem.    
    # score_func=f_regression for regression problem.
    def anovaF(self, X, y, k, score_func=f_regression):
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        return selector.get_support(indices=True)
    
    def rfe(self, X, y, k):
        estimator = Lasso(alpha=0.01, max_iter=10000)

        selector  = RFE(estimator = estimator, 
                        n_features_to_select=k, 
                        step = 1,
                        verbose = 0)
        selector.fit(X, y)

        return np.where(selector.support_)[0]