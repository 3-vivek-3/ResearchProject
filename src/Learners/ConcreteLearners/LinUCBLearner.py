from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner
import numpy as np

class LinUCBLearner(AbstractLearner):
    """
    UCB for linear bandit: based on the principal of optimism in the 
    face of uncertainty. For a linear bandit, this is the ucb value.

    Attributes:
        (super)
        T: Horizon
        history: a list of (action, reward) tuples, updated per round
        p: number of warmup rounds before selecting features
        k: number of features to be selected after warmup rounds

        ()
        delta:
        regularisation:

        action_set:
        V: regularized design matrix
        b: response vector
        theta: 
        selected_features:
    """
    def __init__(self, T : int, d: int, params : dict):
        super().__init__(T, d, params)

        self.delta = params["delta"]

        self.regularization = params["regularization"]

        self.alpha = params.get("alpha", -1)

        self.action_set = []
        self.d = None
        self.V = None
        self.V_inv = None
        self.b = None
        self.theta = None

        self.selected_features = None


    def run(self, env: AbstractEnvironment, logger=None):
        self.d = env.get_ambient_dim()
        self.V = self.regularization * np.eye(self.d)
        self.V_inv = np.eye(self.d) / self.regularization # (d, d) -> (k, k)
        self.b = np.zeros(self.d).reshape(-1, 1) # (d, 1) -> (k, 1)
        self.theta = np.zeros(self.d).reshape(-1, 1) # (d, 1) -> (k, 1)

        for t in range(1, self.T + 1):
            
            # Generate a new set of actions (feature vectors)
            self.action_set = env.observe_actions() # (actions, d)

            # Select an action (feature vector) through exploration or exploitation
            action = self.select_action2(t) # (1, d)

            if self.selected_features is None:
                action_for_model = action # (1, d)
            else:
                action_for_model = action[self.selected_features] # (1, k)

            x = np.array(action_for_model).reshape(-1, 1) # (d, 1) -> (k, 1)

            # Compute the reward corresponding to the selected action (feature vector)
            reward = env.reveal_reward(action)
            
            self.update3(x, reward)

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

        self.theta = self.theta[idx]
        self.b = self.b[idx]

    def select_action(self, t):
        beta = np.sqrt(self.regularization)
        beta += np.sqrt(2 * np.log(1/self.delta) + self.d * (np.log(1 + (t - 1)/(self.regularization * self.d))))

        V_inv = np.linalg.inv(self.V)
        max_ucb = -float('inf')
        best_action = self.action_set[0]
        for a in self.action_set:
            feat = np.array(a)
            ucb = feat.T @ self.theta + beta * np.sqrt(feat.T @ V_inv @ feat)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = a

        return best_action
    
    def select_action2(self, t):
        if self.alpha == -1:
            beta = np.sqrt(self.regularization)
            beta += np.sqrt(
                2 * np.log(1/self.delta) 
                + self.d * np.log(1 + (t - 1)/(self.regularization * self.d)))
        else:
            beta = self.alpha

        if self.selected_features is None:
            X = np.vstack(self.action_set) # (actions, d)
        else:
            X = np.vstack([self.reduce_action(a) for a in self.action_set]) # (actions, k)

        X_T_theta = X.dot(self.theta).ravel() # (actions, )
        X_T_V_inv = X.dot(self.V_inv) # (actions, d)
        X_T_V_inv_X = np.einsum("ij, ij->i", X_T_V_inv, X) # (actions, )

        # ucb_i = (x_i)^T.theta + beta * sqrt((x_i)^T.(V)^-1.(x_i))
        ucbs = X_T_theta + beta * np.sqrt(X_T_V_inv_X)

        best_action_id = np.argmax(ucbs)

        return self.action_set[best_action_id]

    def update(self, feat, reward):
        self.V += feat @ feat.T
        self.b += reward * feat
        self.theta = np.linalg.inv(self.V) @ self.b

    def update2(self, x, reward):
        V_inv_x = self.V_inv @ x
        x_T_V_inv = x.T @ self.V_inv

        x_T_V_inv_x = float(x.T @ V_inv_x)

        # Sherman-Morrison rank-1 update
        self.V_inv -= (V_inv_x @ x_T_V_inv) / (1.0 + x_T_V_inv_x)

        #self.theta += (reward - self.theta.dot(x.flatten())) * V_inv_x.flatten()
        self.theta += (reward - float(x.T @ self.theta)) * V_inv_x

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
        self.theta = self.V_inv @ self.b
    
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