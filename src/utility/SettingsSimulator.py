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

    def __init__(self, settings_dir, file_name, data_dir, data_file_name):

        # Config file path
        self.settings_path = os.path.join(settings_dir, file_name)
        self._read_settings()

        # Data file path
        self.data_file = Path(data_dir) / data_file_name
        self._read_data()

        self.logger = ResultLogger(self.name)
        self.logger.new_log()

        self.trials = len(self.trials_action_sets)
        self.horizon, self.actions, self.d = self.trials_action_sets[0].shape

        print(f"\ntrials: {self.trials}, horizon: {self.horizon}, actions: {self.actions}, amb_dim: {self.d}")
        
        self.visualizer : Visualizer = Visualizer(self.logger.log_dir, self.do_export, self.do_show)

        self.curr_simulation = 0
        self.num_simulations = len(self.settings)

        # Create a replica json file in the log folder.
        self._replicate_settings(self.logger.get_results_dir(file_name))

    def _read_data(self):
        if self.data_file.exists():
            with gzip.open(self.data_file, "rb") as f:
                loaded = pickle.load(f)
            self.trials_action_sets = loaded["action_sets"]
            self.trials_theta       = loaded["thetas"]
            self.trials_action_sets_recorded = True
        else:
            raise RuntimeError("Data file not loaded")

    def _read_settings(self):
        
        # Data loaded from the config file.
        data = json.load(open(self.settings_path, mode = "r", encoding="utf-8"))

        assert data["simulations"] is not None
        assert data["name"] is not None

        self.name = data["name"]

        self.do_export = data["export_figures"]
        self.do_show = data["show_figures"]

        # contains the actual data for all simulations
        self.settings = data["simulations"]

    def _replicate_settings(self, file_path : str):

        # Determine the total number of trials
        #total_trials = sum(map(lambda x : x["trials"], self.settings))

        data = {
            "name" : self.name,
            "date" : datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
            "data file" : str(self.data_file),

            "number of simulations" : self.num_simulations,
            "trials" : self.trials,
            "horizon" : self.horizon,
            "actions" : self.actions,
            "ambient dimension" : self.d, 

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

        env_cls = getattr(Environments, curr_settings["env"])
        learner_cls = getattr(Learners, curr_settings["learner"])

        for trial in trange(self.trials, desc=f"Running {name}"):
            print("\nTrial: ", trial + 1)
            # Set up the logger
            self.logger.set_simulation(name, trial + 1)

            # Build environment parameters, always copy base config
            #env_params = dict(curr_settings["env_config"])
            env_params = {
                "d" : self.d,
                "actions" : self.actions
            }

            if self.trials_action_sets_recorded:
                env_params["action_sets"] = self.trials_action_sets[trial]
                env_params["true_theta"]  = self.trials_theta[trial]
            
            # Instantiate a new copy of the environment and learner
            env : AbstractEnvironment = env_cls(env_params)
            learner : AbstractLearner = learner_cls(self.horizon, self.d, curr_settings["learner_config"])

            learner.run(env, self.logger)

            #print("\n")
            #print(env.get_theta())
            #print("\n")
            #print(learner.get_selected_features())

        self.curr_simulation += 1

    def simulate_all(self):

        while self.curr_simulation < self.num_simulations:
            print("\nsimulation: ", self.curr_simulation)
            self.simulate_next()

        self.visualizer.generate_graphs()