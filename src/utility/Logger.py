import os
import csv
from datetime import datetime
from typing import Optional


class ResultLogger:
    '''
    A logger class that handles creation of run-specific directories and 
    CSV log files, and provides methods to record the results for various 
    simulations in a run.
    '''

    def __init__(self, name):
        self.base_dir = "../results"
        self.fields = ["Name", "Trial", "Round", "p", "k", "Reward", "Regret", "Time"]
        self.name = name
        self.log_dir: Optional[str] = None
        self.log_file_path: Optional[str] = None

        self.curr_sim_name = ""
        self.curr_sim_trial = 1

    def new_log(self):
        '''
        Create a new timestamped directory under the base results folder,
        initialize a CSV file named 'log.csv' within it, and write the 
        header row. 
        Raises an error if the directory already exists to prevent 
        overwriting.
        '''
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        
        # New folder path: <base_dir>/run_<name>_<timestamp>
        self.log_dir = os.path.join(self.base_dir, f"run_{self.name}_{timestamp}")
        
        # Create new folder. Throw error if folder already exists.
        os.makedirs(self.log_dir, exist_ok=False)

        # CSV file path: <base_dir>/run_<name>_<timestamp>/log.csv
        self.log_file_path = os.path.join(self.log_dir, "log.csv")

        # Open the CSV file and write the headers.
        with open(self.log_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.fields)

    def log(self, t, p, k, reward, regret):
        '''
        Append a new row to the CSV log file with simulation name, trial 
        number, round number, reward, regret, and the current timestamp.
        Raises a RuntimeError if new_log() has not been called first.
        '''
        row = [self.curr_sim_name, self.curr_sim_trial, t, p, k, reward, regret,
               datetime.now().strftime("%d/%m/%Y-%H:%M:%S")]

        if not self.log_file_path:
            raise RuntimeError("Log file not initialized. Call new_log() first.")

        with open(self.log_file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def set_simulation(self, name, trial):
        self.curr_sim_name = name
        self.curr_sim_trial = trial

    def get_results_dir(self, file_name : str = "") -> str:
        return os.path.join(self.log_dir, file_name)