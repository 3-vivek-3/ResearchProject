from src.Environments import LinearEnvironment
from src.Learners import ETCLearner
from src.utility.Logger import ResultLogger
from src.utility.SettingsSimulator import SettingsSimulator
from src.utility.SettingsSimulator2 import SettingsSimulator2

import matplotlib
matplotlib.use("tkagg")
import numpy as np
import matplotlib.pyplot as plt

def main():

    # simple_example()
    complex_simulation()

def complex_simulation():

    settings_dir = "configurations"
    data_dir = "data"

    #file_name = "toy.json"

    #file_name = "EGreedy_config.json"
    #file_name = "EGreedy-FS_config.json"
    #file_name = "EGreedy-FS_config_playground.json"
    #file_name = "EGreedy-FS_config2.json"

    #file_name = "ETC_config.json"
    #file_name = "ETC-FS_config.json"
    #file_name = "ETC-FS_config2.json"

    #file_name = "LinUCB_config.json"
    #file_name = "LinUCB-FS_config.json"
    file_name = "LinUCB-FS_config2.json"

    #data_file_name = "env_10x2x5_90%_sparsity_5_trials.pkl.gz"

    #data_file_name = "env_500x50x200_0%_sparsity.pkl.gz"
    #data_file_name = "env_500x50x200_10%_sparsity.pkl.gz"
    #data_file_name = "env_500x50x200_50%_sparsity.pkl.gz"
    #data_file_name = "env_500x50x200_90%_sparsity.pkl.gz"
    #data_file_name = "env_500x50x200_90%_sparsity_50_trials.pkl.gz"
    #data_file_name = "env2_500x50x200_90%_sparsity_50_trials.pkl.gz"

    #data_file_name = "env_500x10x100_50%_sparsity_20_trials.pkl.gz"
    #data_file_name = "env_500x10x100_90%_sparsity_20_trials.pkl.gz"
    data_file_name = "env_500x10x100_90%_sparsity_30_trials.pkl.gz"
    
    #simulator = SettingsSimulator(settings_dir, file_name, data_dir, data_file_name)
    simulator = SettingsSimulator2(settings_dir, file_name, data_dir, data_file_name)
    
    simulator.simulate_all()

def simple_example():
    print("Beginning a Bandit Simulation")

    simulation_name = "Example_Simulation"

    env = LinearEnvironment({
        "d" : 2,
        "action_set" : list([ [x, y] for x in range(3) for y in range(4)]),
        "true_theta" : [5, -3],
        "sigma" : 1,
        "k" : 12
    })

    learner = ETCLearner(24, {})

    logger = ResultLogger(simulation_name)
    logger.new_log()

    learner.run(env, logger)
    rewards = learner.cum_reward()
    cum_regret = env.get_cumulative_regret()

    print("Terminating the Bandit Simulation")

    plt.title("Rewards over time")
    plt.plot(np.cumsum(rewards), label="ETC")
    plt.legend()
    plt.show()

    plt.title("Regret")
    plt.plot(cum_regret)
    plt.show()

if __name__ == "__main__":
    main()
