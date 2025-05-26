from src.utility.Logger import ResultLogger
from src.utility.Visualizer import Visualizer
from src import Environments, Learners
from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner

import os.path
import json
from datetime import datetime
from tqdm import trange

import gzip     
import pickle
from pathlib import Path

class SettingsSimulator:

    def __init__(self, settings_dir, filename="simulation_config.json"):

        self.settings_dir = settings_dir
        self.filename = filename
        self.settings_path = os.path.join(settings_dir, self.filename)

        self._read_settings()

        #data_file_name = "env_500x50x200_0%_sparsity.pkl.gz"
        #data_file_name = "env_500x50x200_10%_sparsity.pkl.gz"
        #data_file_name = "env_500x50x200_50%_sparsity.pkl.gz"
        #data_file_name = "env_500x50x200_90%_sparsity.pkl.gz"
        data_file_name = "env_500x50x200_90%_sparsity_50_trials.pkl.gz"

        #data_file_name = "env_100x10x20_10%_sparsity.pkl.gz"
        #data_file_name = "env_100x10x20_50%_sparsity.pkl.gz"
        #data_file_name = "env_100x10x20_90%_sparsity.pkl.gz"

        data_file = Path("data") / data_file_name
        if data_file.exists():
            with gzip.open(data_file, "rb") as f:
                loaded = pickle.load(f)
            self.trials_action_sets = loaded["action_sets"]
            self.trials_theta       = loaded["thetas"]
            self.trials_action_sets_recorded = True
        else:
            raise RuntimeError("Data file not loaded")
            #self.trials_action_sets_recorded = False
            #self.trials_action_sets = []   # list of lists of action matrices per trial
            #self.trials_theta       = []   # list of theta vectors per trial

        self.logger = ResultLogger(self.name)
        self.logger.new_log()

        self._replicate_settings(self.logger.get_results_dir(self.filename))
        self.visualizer : Visualizer = Visualizer(self.logger.log_dir, self.do_export, self.do_show)

        self.curr_simulation = 0

    def _read_settings(self):
        
        # the data loaded from a config file.
        data = json.load(open(self.settings_path, mode = "r", encoding="utf-8"))

        assert data["simulations"] is not None
        assert data["name"] is not None

        self.name = data["name"]
        self.do_export = data["export_figures"]
        self.do_show = data["show_figures"]

        # contains the actual data for all simulations
        self.settings = data["simulations"]
        self.num_simulations = len(self.settings)

        # a mapping of simulation name to simulation settings
        simulation_names = list(map(lambda sim : sim["name"], self.settings))

        # Make sure that simulation names are unique
        if len(simulation_names) != len(set(simulation_names)):
            raise RuntimeError("Simulation names are not unique")

        self.run_names = simulation_names

    def _replicate_settings(self, file_path : str):

        # Determine the total number of trials
        total_trials = sum(map(lambda x : x["trials"], self.settings))

        data = {
            "name" : self.name,
            "date" : datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),

            "number of simulations" : self.num_simulations,
            "number of trials" : total_trials,
            "simulation names" : self.run_names,

            "simulations" : self.settings
        }

        with open(file_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent = 4)


    def simulate_next(self):

        if self.curr_simulation >= self.num_simulations:
            return

        curr_settings = self.settings[self.curr_simulation]

        # Extract the parameters
        name = curr_settings["name"]
        trials = curr_settings["trials"]
        horizon = curr_settings["horizon"]

        env_cls = getattr(Environments, curr_settings["env"])
        learner_cls = getattr(Learners, curr_settings["learner"])


        for trial in trange(trials, desc=f"Running {name}"):
            print("\nTrial: ", trial + 1)
            # Set up the logger
            self.logger.set_simulation(name, trial + 1)

            # Build environment parameters, always copy base config
            env_params = dict(curr_settings["env_config"])

            if self.trials_action_sets_recorded:
                env_params["action_sets"] = self.trials_action_sets[trial]
                env_params["true_theta"]  = self.trials_theta[trial]

            
            #if self.curr_simulation > 0:
            #    # Replay recorded action sets and theta after 1st simulation
            #    env_params["action_sets"] = self.trials_action_sets[trial]
            #    env_params["true_theta"] = self.trials_theta[trial]

            
            # Instantiate a new copy of the environment and learner
            env : AbstractEnvironment = env_cls(env_params)
            learner : AbstractLearner = learner_cls(horizon, curr_settings["learner_config"])

            learner.run(env, self.logger)

            print("\n")
            print(env.get_theta())
            print("\n")
            print(learner.get_selected_features())

            #if self.curr_simulation == 0:
            #    self.trials_action_sets.append(env.get_recorded_action_sets())
            #    self.trials_theta.append(env.get_theta())

        #self.trials_action_sets_recorded = True
        self.curr_simulation += 1

    def simulate_all(self):

        while self.curr_simulation < self.num_simulations:
            print("\nsimulation: ", self.curr_simulation)
            self.simulate_next()

        self.visualizer.generate_graphs()