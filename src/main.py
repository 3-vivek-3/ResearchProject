from src.Environments import LinearEnvironment
from src.Learners import ETCLearner
from src.utility.Logger import ResultLogger
from src.utility.SettingsSimulator import SettingsSimulator
import matplotlib
matplotlib.use("tkagg")
import numpy as np
import matplotlib.pyplot as plt

def main():

    # simple_example()
    complex_simulation()

def complex_simulation():

    settings_dir = "configurations"
    #simulator = SettingsSimulator(settings_dir, "Egreedy_config.json")
    #simulator = SettingsSimulator(settings_dir, "Egreedy_config1.json")
    #simulator = SettingsSimulator(settings_dir, "Egreedy_config_fs.json")
    #simulator = SettingsSimulator(settings_dir, "Egreedy_config_playground.json")
    #simulator = SettingsSimulator(settings_dir, "Egreedy_config_playground1.json")

    simulator = SettingsSimulator(settings_dir, "ETC_config.json")
    #simulator = SettingsSimulator(settings_dir, "LinUCB_sparse_config.json")
    #simulator = SettingsSimulator(settings_dir, "ETC_sparse_config_test.json")
    simulator.simulate_all()

if __name__ == "__main__":
    main()
