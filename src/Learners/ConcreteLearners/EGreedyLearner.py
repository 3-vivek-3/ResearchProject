from src.Environments import SparseLinearEnvironment
from src.Learners import AbstractLearner
from sklearn.linear_model import SGDRegressor
import numpy as np

class EGreedyLearner(AbstractLearner):
    """
    Epsilon-greedy learner for SCLB with feature selection: in each 
    round, it either explores by picking a random action with probability 
    ε, or exploits by selecting the action that maximizes reward under a 
    sparse linear model estimated via online Lasso (SGD with L1 penalty). 
    After a specified number of warm-up rounds p, it performs feature 
    selection to restrict the model to the top k features, then continues 
    learning in the reduced feature space.

    Attributes:
        (super)
        T: horizon
        d: ambient dimension
        history: a list of (action, reward) tuples, updated per round
        p: the number of warmup rounds before selecting features
        k: number of features to be selected after warmup rounds

        ()
        epsilon: exploration rate in (0, 1)
        action_set:
        regressor:
        selected_features: 
    """
    def __init__(self, T: int, d: int, params: dict):
        super().__init__(T, d, params)

        self.epsilon: float = params["epsilon"]
        assert 0 <= self.epsilon <= 1

        self.action_set = []

        # Lasso regressor to compute estimated theta (parameter vector)
        self.regressor = SGDRegressor(penalty="l1", alpha=0.01, fit_intercept=True, random_state=0)

        # Elastic net regressor
        # By default, l1_ratio = 0.15, meaning l1=0.0015 and l2=0.0085
        ## self.regressor = SGDRegressor(penalty="elasticnet", alpha=0.01)

        self.selected_features = None

    def run(self, env: SparseLinearEnvironment, logger = None):

        for t in range(1, self.T + 1):

            # Generate a new set of actions (feature vectors)
            self.action_set = env.observe_actions()

            # Select an action (feature vector) through exploration or exploitation
            action = self.select_action(t)

            if self.selected_features is None:
                action_for_model = action
            else:
                action_for_model = action[self.selected_features]

            # Compute the reward corresponding to the selected action (feature vector)
            reward = env.reveal_reward(action)

            x = action_for_model.reshape(1, -1)
            y = np.array([reward])

            # Update regressor
            self.regressor.partial_fit(x, y)

            # Append (action, reward) at round t to history
            self.history.append((action, reward))

            env.record_regret(reward, self.action_set)

            if logger is not None:
                logger.log(t, reward, env.regret[-1])

            if t == self.p:
                self.do_feature_selection()

    def do_feature_selection(self):
        X_full = np.vstack([entry[0] for entry in self.history])
        y_full = np.array([entry[1] for entry in self.history])

        self.selected_features = np.array(self.selectKFeatures(X_full, y_full, self.k))

        X_reduced = X_full[:, self.selected_features]

        new_regressor = SGDRegressor(penalty="l1", alpha=0.01, fit_intercept=True, random_state=0)
        new_regressor.partial_fit(X_reduced, y_full)
        self.regressor = new_regressor

    '''
    Adapts the learner to the reduced feature space.
    - selects k features
    - builds a new regressor from the old one to work for the reduced
        feature space.
    - 
    '''
    def do_feature_selection1(self):
        # X_full = the feature vectors from history for the first p rounds
        # y_full = the rewards from history for the first p rounds

        X_full = np.vstack([entry[0] for entry in self.history])
        y_full = np.array([entry[1] for entry in self.history])

        self.selected_features = np.array(self.selectKFeatures(X_full, y_full, self.k))

        new_regressor = SGDRegressor(penalty="l1", alpha=0.01, fit_intercept=True, random_state=0)

        dummy_x = np.zeros((1, self.k))
        dummy_y = np.zeros(1)

        new_regressor.partial_fit(dummy_x, dummy_y)

        old_coef = self.regressor.coef_
        old_intercept = self.regressor.intercept_

        new_regressor.coef_ = old_coef[self.selected_features].copy()
        new_regressor.intercept_ = old_intercept.copy() 

        self.regressor = new_regressor

    '''
    Selects the best action.
    - if p > t, then use a reduced action set, else use the normal action set
    '''
    def select_action(self, t):

        # If regressor hasn't been run yet, then the parameter 'coef_' doesn't exist.
        if not hasattr(self.regressor, 'coef_') or np.random.rand() < self.epsilon:
            i = np.random.randint(len(self.action_set))
            return self.action_set[i]

        else:
            if t > self.p:
                model_action_set = np.vstack([self.reduce_action(a) for a in self.action_set])
            else:
                model_action_set = self.action_set

            # Compute estimated rewards using estimated theta and feature vectors
            estimated_rewards = self.regressor.predict(model_action_set)
            
            # Select action index whose corresponding estimated reward is maximum. 
            best_reward_id = np.argmax(estimated_rewards)
            
            # Return feature vector corresponding to selected action. 
            return self.action_set[best_reward_id]
    
    
    '''
    Reduces the dimensionality of an action using the selected features.
    '''
    def reduce_action(self, action):
        if self.selected_features is None:
            return action
        return action[self.selected_features]
    
    def total_reward(self):
        total = 0
        for (_, reward) in self.history:
            total += reward
        return total
    
    def cum_reward(self):
        cumulative = []
        for (_, reward) in self.history:
             cumulative.append(reward)
        return cumulative
    
    def get_selected_features(self):
        return self.selected_features