from abc import ABC, abstractmethod
from src.Environments import AbstractEnvironment
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np

class AbstractLearner(ABC):
    def __init__(self, T : int, params : dict):
        '''
        :param actions: Array of the actions of the agent
        :param T: Horizon
        '''
        self.T = T
        self.t = 0
        self.history = []

        if "featureSelector" not in params.keys():
            self.featureSelector = "anovaF"
        else:
            self.featureSelector = params["featureSelector"]

    @abstractmethod
    def run(self, env : AbstractEnvironment, logger):
        pass

    @abstractmethod
    def select_action(self, context):
        pass

    @abstractmethod
    def total_reward(self):
        pass

    @abstractmethod
    def cum_reward(self):
        pass

    def feature_map(self, action, context):
        return action
    
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

    # score_func=f_regression for regression problem.
    # Isn't this a regression problem?
    def anovaF(self, X, y, k, score_func=f_classif):
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        return selector.get_support(indices=True)
    
    def rfe(self, X, y, k):
        estimator = LogisticRegression(solver='liblinear', max_iter=1000)

        selector  = RFE(estimator = estimator, 
                        n_features_to_select=k, 
                        step = 1,
                        verbose = 0)
        selector.fit(X, y)

        return np.where(selector.support_)[0]