from src.Environments import SparseLinearEnvironment
from src.Learners import AbstractLearner
from sklearn.linear_model import SGDRegressor
import numpy as np

class ETCLearner(AbstractLearner):
    """
    ETC SCLB learner with feature selection: the learner explores the 
    actions for the first mN rounds, visiting each action m times. At 
    round mN, it locks in on the estimated param vector 𝜃^ by fitting
    the Lasso (SGD with L1 penalty). In subsequent rounds, the learner 
    exploits, by selecting the action that maximises reward under the 
    Lasso's sparse linear model. After a number of rounds mN <= p <= T, 
    it performs feature selection to select the top k features. In the 
    remaining rounds, the learner exploits on the reduced feature space.

    Attributes:
        (super)
        T: Horizon
        d: ambient dimension
        history: A list of (action, reward) tuples, updated per round
        p: the number of warmup rounds before selecting features
        k: number of features to be selected after warmup rounds

        ()
        m: the number of exploration cycles per action
        N: the number of actions

        action_set:
        regressor:
        X_exploration:
        y_exploration:
        selected_features: 
    """
    def __init__(self, T : int, d:int, params : dict):
        super().__init__(T, d, params)

        self.m = params.get("m")
        self.N = 0

        self.action_set = None

        # Lasso regressor to compute estimated theta (parameter vector)
        self.regressor = SGDRegressor(penalty="l1", alpha=0.01, fit_intercept=True, random_state=0)

        self.X_exploration = None
        self.y_exploration = None

        self.selected_features = None

    def run(self, env : SparseLinearEnvironment, logger = None):
        self.N = env.actions

        assert self.m * self.N <= self.T

        # Feature selection only in the exploitation stage.
        assert self.p >= self.m * self.N

        for t in range(1, self.T + 1):

            # Generate new action set (feature vectors).
            self.action_set = env.observe_actions()

            # Select an action (feature vector) through exploration or exploitation
            action = self.select_action(t)

            # Compute the reward corresponding to the selected action (feature vector)
            reward = env.reveal_reward(action)

            self.history.append((action, reward))

            env.record_regret(reward, self.action_set)

            # Log the actions
            if logger is not None:
                logger.log(t, self.p, self.k, reward, env.regret[-1])

            # Fix estimated param vector for exploitation after mN rounds
            if t == self.m * self.N:
                self.X_exploration = np.vstack([entry[0] for entry in self.history])
                self.y_exploration = np.array([entry[1] for entry in self.history])
                self.regressor.partial_fit(self.X_exploration, self.y_exploration)

            if t == self.p:
                self.do_feature_selection()

            # Fix estimated param vector for exploitation after mN rounds
            # Feature selection is done immediately.
            #if t == self.m * self.N:
            #    self.X_exploration = np.vstack([entry[0] for entry in self.history])
            #    self.y_exploration = np.array([entry[1] for entry in self.history])
            #    self.do_feature_selection()

    '''
    Adapts the learner to the reduced feature space.
    - selects k features
    - builds a new regressor from the old one to work for the reduced
        feature space.
    - 
    '''
    def do_feature_selection(self):
        self.selected_features = np.array(self.selectKFeatures(self.X_exploration, self.y_exploration, self.k))

        X_reduced = self.X_exploration[:, self.selected_features]

        new_regressor = SGDRegressor(penalty="l1", alpha=0.01, fit_intercept=True, random_state=0)
        new_regressor.partial_fit(X_reduced, self.y_exploration)
        self.regressor = new_regressor

    def select_action(self, t):
        # Exploration
        if t <= self.m * self.N:
            return self.action_set[t % self.N]
        
        # Exploitation
        if t > self.m * self.N:
            '''
            '''
            #if self.selected_features is None:
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