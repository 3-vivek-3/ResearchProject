from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner
import numpy as np

class TSLearner(AbstractLearner):
    """
    Thompson Sampling

    Attributes:
        (super)
        T: Horizon
        history: a list of (action, reward) tuples, updated per round
        p: number of warmup rounds before selecting features
        k: number of features to be selected after warmup rounds

        ()
        regularisation:

        action_set:
        V: regularized design matrix
        b: response vector
        theta: 
        selected_features:
    """
    def __init__(self, T : int, d: int, params : dict):
        super().__init__(T, d, params)

        self.regularization = params["regularization"]
        self.nu = None

        self.action_set = []
        self.d = None
        self.V_inv = None
        self.b = None
        self.mu = None

        self.selected_features = None
        self.rng = np.random.default_rng()


    def run(self, env: AbstractEnvironment, logger=None):
        print("\nTSLearner")
        self.d = env.get_ambient_dim()
        self.nu = env.get_sigma()
        self.V_inv = np.eye(self.d) / self.regularization # (d, d) -> (k, k)
        self.b = np.zeros(self.d).reshape(-1, 1) # (d, 1) -> (k, 1)
        self.mu = np.zeros(self.d).reshape(-1, 1) # (d, 1) -> (k, 1)

        for t in range(1, self.T + 1):
            
            # Generate a new set of actions (feature vectors)
            self.action_set = env.observe_actions() # (actions, d)

            # Select an action (feature vector) through exploration or exploitation
            action = self.select_action2(t) # (1, d)

            if self.selected_features is None:
                action_for_model = action # (1, d)
            else:
                action_for_model = action[self.selected_features] # (1, k)

            x = action_for_model.reshape(-1, 1) # (d, 1) -> (k, 1)

            # Compute the reward corresponding to the selected action (feature vector)
            reward = env.reveal_reward(action)
            
            self.update4(x, reward)

            # Append (action, reward) at round t to history
            self.history.append((action, reward))

            env.record_regret(reward, self.action_set)

            if logger is not None:
                logger.log(t, self.p, self.k, reward, env.regret[-1])

            if t == self.p:
                self.do_feature_selection()

    def do_feature_selection(self):
        X_full = np.vstack([entry[0] for entry in self.history])
        y_full = np.array([entry[1] for entry in self.history])

        self.selected_features = np.array(self.selectKFeatures(X_full, y_full, self.k))

        idx = self.selected_features
        V_full = np.linalg.inv(self.V_inv)
        V_reduced = V_full[np.ix_(idx, idx)]
        self.V_inv = np.linalg.inv(V_reduced)

        self.mu = self.mu[idx]
        self.b = self.b[idx]
    
    def select_action(self, t):
        if self.selected_features is None:
            X = np.vstack(self.action_set) # (actions, d)
        else:
            X = np.vstack([self.reduce_action(a) for a in self.action_set]) # (actions, k)

        sample_theta = np.random.multivariate_normal(self.mu.ravel(), (self.nu * self.nu) * self.V_inv).reshape(-1, 1)

        scores = X.dot(sample_theta).ravel() # (actions, )

        best_action_id = np.argmax(scores)

        return self.action_set[best_action_id]
    
    def select_action2(self, t):
        L = np.linalg.cholesky((self.nu**2) * self.V_inv)
        z = self.rng.standard_normal(self.mu.shape)
        sample_theta = self.mu + L @ z

        if self.selected_features is not None:
            X = self.action_set[:, self.selected_features]
        else:
            X = self.action_set

        scores = X.dot(sample_theta).ravel()
        best_action_id = np.argmax(scores)
        return self.action_set[best_action_id]

    """
    Sherman-Morrison formula
    # Let V ∈ R^{dxd}, x ∈ R^d
    # V' = V + x xᵀ
    # V'⁻¹ = V⁻¹ - (V⁻¹ x xᵀ V⁻¹) / (1 + xᵀ V⁻¹ x)
    
    """
    def update3(self, x, reward):
        V_inv_x = self.V_inv @ x
        x_T_V_inv = x.T @ self.V_inv

        x_T_V_inv_x = float(x.T @ V_inv_x)

        # Sherman-Morrison rank-1 update
        self.V_inv -= (V_inv_x @ x_T_V_inv) / (1.0 + x_T_V_inv_x)

        self.b += reward * x
        self.mu = self.V_inv @ self.b

    def update4(self, x, reward):
        v = self.V_inv.dot(x)
        denom = 1.0 + float(x.T.dot(v))

        self.V_inv -= (v @ v.T) / denom

        self.b += reward * x
        self.mu = self.V_inv.dot(self.b)

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