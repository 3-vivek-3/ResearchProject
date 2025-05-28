from src.Environments import SparseLinearEnvironment
from src.Learners import AbstractLearner
from sklearn.linear_model import SGDRegressor
import numpy as np

class ETCLearner(AbstractLearner):
    """
    ETC learner for a sparse linear bandit: ETC is characterised by the 
    number of times it explores each arm. This is denoted by m. Given that 
    there are N actions, the algorithm will explore for mN rounds before 
    choosing a fixed single action for the remaining rounds.

    Attributes:
        (super)
        T: Horizon
        t: Current round
        history: A list of (action, reward) tuples, updated per round

        ()
        p: the number of warmup rounds before selecting features
        k: number of features to be selected after warmup rounds
        m: the number of exploration cycles per action
        N: the number of actions


        (internal state)
        action_set:
        regressor:
        optimal_action: the chosen action that will be used throughout the
            exploitation phase.
        selected_features: 
    """
    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        self.p: int = params.get("p", 10)
        assert 0 <= self.p <= T

        self.k: int = params.get("k", 0)
        assert self.k > 0

        self.m = params.get("m")
        self.N = 0

        self.action_set = None

        # Lasso regressor to compute estimated theta (parameter vector)
        self.regressor = SGDRegressor(penalty="l1", alpha=0.01, fit_intercept=True, random_state=0)

        self.optimal_action_id = None

        self.selected_features = None

    def run(self, env : SparseLinearEnvironment, logger = None):
        self.N = env.actions

        for t in range(1, self.T + 1):

            # Generate a new set of actions (feature vectors)
            self.action_set = env.observe_actions()

            # Select an action (feature vector) through exploration or exploitation
            context = env.generate_context() # why?
            action = self.select_action(t, context)

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

            self.history.append((action, reward))

            env.record_regret(reward, self.action_set)

            # Log the actions
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

    def select_action(self, t, context):
        if t < self.N * self.m:
            return self.action_set[t % self.N]
        
        if t == self.N * self.m:
            if t > self.p:
                model_action_set = np.vstack([self.reduce_action(a) for a in self.action_set])
            else:
                model_action_set = self.action_set
            
            # Compute estimated rewards using estimated theta and feature vectors
            estimated_rewards = self.regressor.predict(model_action_set)

            # Select action index whose corresponding estimated reward is maximum. 
            self.optimal_action_id = np.argmax(estimated_rewards)

        return self.action_set[self.optimal_action_id]
    
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