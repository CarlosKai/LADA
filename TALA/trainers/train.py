import sys

import torch
import os
import pandas as pd
import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from algorithms import *
from trainers.abstract_trainer import AbstractTrainer
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()
       


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        super().__init__(args)
        self.results_columns = ["scenario", "run", "acc", "f1_score"]


    def fit(self):

        # table with metrics
        table_results = pd.DataFrame(columns=self.results_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                src_id, trg_id, run_id)
                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)
                
                # initiate the domain adaptation algorithm
                self.initialize_algorithm()

                # Train the domain adaptation algorithm
                self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.src_test_dl, self.trg_train_dl, self.trg_test_dl,
                                                                         self.loss_avg_meters, self.logger)

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, self.last_model, self.best_model)

                # Calculate risks and metrics
                metrics = self.algorithm.test_process("Target Domain Test", self.trg_test_dl, self.logger)
                metrics_list = [metric.cpu().item() if isinstance(metric, torch.Tensor) else metric for metric in
                                metrics]

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics_list)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, self.results_columns)

        # Save tables to file if needed
        self.save_tables_to_file(table_results, 'results')

    def test(self):
        # Results dataframes
        last_results = pd.DataFrame(columns=self.results_columns)
        best_results = pd.DataFrame(columns=self.results_columns)

        # Cross-domain scenarios
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)

                # Logging
                self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))

                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Build model
                self.initialize_algorithm()

                # Load chechpoint 
                last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)

                # Testing the last model
                self.algorithm.network.load_state_dict(last_chk)

                metrics = self.algorithm.test_process("-", self.trg_test_dl, self.logger)
                last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", run_id,
                                                             metrics)
                

                # Testing the best model
                self.algorithm.network.load_state_dict(best_chk)
                best_metrics = self.algorithm.test_process("-", self.trg_test_dl, self.logger)
                # Append results to tables
                best_results = self.append_results_to_tables(best_results, f"{src_id}_to_{trg_id}", run_id,
                                                             best_metrics)

        last_scenario_mean_std = last_results.groupby('scenario')[['acc', 'f1_score']].agg(['mean', 'std'])
        best_scenario_mean_std = best_results.groupby('scenario')[['acc', 'f1_score']].agg(['mean', 'std'])

        # Save tables to file if needed
        self.save_tables_to_file(last_scenario_mean_std, 'last_results')
        self.save_tables_to_file(best_scenario_mean_std, 'best_results')




