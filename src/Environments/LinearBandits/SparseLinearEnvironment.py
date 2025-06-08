from ..AbstractEnvironment import AbstractEnvironment
import numpy as np
import scipy.sparse as sp

class SparseLinearEnvironment(AbstractEnvironment):
    
    """
    Attributes:
        AbstractEnvironment:

        d:              the ambient dimension, full dimension of the feature space
        action_set:     stores the currently generated action set.
        regret:
        cum_regret:

        SparseLinearEnvironment:

        actions:        number of actions
        sparsity:       sparsity parameter
        true_theta:     should be auto-generated
        sigma:          parameter for randomness
    """

    def __init__(self, params):

        super().__init__(params)
        self.actions = params["actions"]
        self.sparsity = 0.9
        self.sigma = 1

        '''
        During the first trial, when an environment is created for the first
        time, it will record all the action sets generated over T rounds.
        Settings simulator will store the list of action sets and the unknown
        parameter vector theta. In the subsequent trials, the environment 

        For each trial, the environment will create T action_sets and one 
        parameter vector. This is done for multiple trials. All of this is
        one simulation.

        We store list(env_setup), where env_setup = (list(action_set),theta*)
        '''
        self.preloaded_action_sets = params.get("action_sets", None)
        if self.preloaded_action_sets is None:
            self.recorded_action_sets = []
        else:
            self.recorded_action_sets = None
        
        self.current_round = 0

        self.true_theta = params.get("true_theta", None)
        if self.true_theta is None:
            self.true_theta = self.generate_theta()

    def reveal_reward(self, action):
        return self.true_theta.T.dot(action.reshape((self.d, 1)))[0][0] + np.random.normal(loc=0.0, scale=self.sigma)
    
    '''
    Generate the sparse, TRUE parameter vector. 
    '''
    def generate_theta(self):
        return sp.random(self.d, 1, (1 - self.sparsity))

    def record_regret(self, reward, feature_set):
        best_reward = -float('inf')
        for a in feature_set:
            best_reward = max(self.true_theta.T.dot(a.reshape((self.d, 1)))[0][0], best_reward)

        empirical_regret = best_reward - reward
        self.cum_regret += empirical_regret
        self.regret.append(empirical_regret)

    """
    Returns a (actions x d) array. In generation mode, draws and stores;
    in replay mode, returns the pre-recorded set for this round.
    """
    def observe_actions(self):
        if self.preloaded_action_sets is not None:
            action_set = self.preloaded_action_sets[self.current_round]
        else:
            action_set = np.random.normal(0, 1, size=(self.actions, self.d))
            self.recorded_action_sets.append(action_set)
        
        self.action_set = action_set
        self.current_round += 1
        return action_set
    
    '''
    Retrieve the recorded action sets after generation mode
    '''
    def get_recorded_action_sets(self):
        if self.get_recorded_action_sets is None:
            raise RuntimeError("No action sets recorded: environment is in replay mode")
        return self.recorded_action_sets
    
    def get_theta(self):
        return self.true_theta