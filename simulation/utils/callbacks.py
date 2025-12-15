"""
Copyright 2025 AUMOVIO. All rights reserved.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import ot
from typing import List
from collections import defaultdict
import time
import json




from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.common.utils.io_utils import (
    delete_file,
    list_files_in_directory,
    path_exists,
    read_pickle,
    safe_path_to_string,
)

from common_utils.time_tracking import timeit

logger = logging.getLogger(__name__)


class FindOutliersCallback(AbstractMainCallback):

    def __init__(self, aggregator_save_path: str, threshold=0.1):
        self._aggregator_save_path = Path(aggregator_save_path)
        self._threshold = threshold

    @timeit
    def on_run_simulation_end(self) -> None:
        aggregate_metric_file = [p for p in self._aggregator_save_path.glob("*.parquet")]
        if len(aggregate_metric_file)>0:
            aggregate_metric_file = aggregate_metric_file[0]
            aggregate_df = pd.read_parquet(aggregate_metric_file)
            df_scenarios = aggregate_df[aggregate_df["log_name"].notna()]
            # list of low percentile scenarios
            # low = df_scenarios["score"].quantile(self._quantile)
            outliers = df_scenarios[df_scenarios["score"] < self._threshold][["log_name", "scenario"]]
            output_json = outliers.to_json(orient="records", lines=False, indent=2)
            with open(self._aggregator_save_path / f"outliers.json", "w") as f:
                json.dump(output_json, f)


class PredictorMetricFileCallback(AbstractMainCallback):

    def __init__(self, metric_file_output_path: str, scenario_metric_root: str):
        """
        Constructor of PredictorMetricFileCallback.
        Output path must be local
        :param metric_file_output_path: Path to save integrated metric files.
        :param scenario_metric_paths: A list of paths with scenario metric files.
        """
        self._metric_file_output_path = Path(metric_file_output_path)

        self._metric_file_output_path.mkdir(exist_ok=True, parents=True)
        self._scenario_metric_root = Path(scenario_metric_root)

    @timeit
    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        # Integrate scenario metric files into metric statistic files
        metrics = defaultdict(list)
        list_of_scenario_metric_files = [p for p in self._scenario_metric_root.rglob("*.parquet") if p.is_file()]
        
        scenario_scores = defaultdict(dict)

        for scenario_metric_file in list_of_scenario_metric_files:
            df = pd.read_parquet(scenario_metric_file)
            metric_score = df.mean(skipna=True).mean(skipna=True)
            metric_name = df.attrs["metric_name"]
            scenario_name = df.attrs["scenario_name"]
            scenario_type = df.attrs["scenario_type"]

            data = {
                "log_name": df.attrs["scenario_log"],
                "scenario": scenario_name,
                "scenario_type": scenario_type,
                "planner_name": df.attrs["planner_name"],
                "predictor_name": df.attrs["predictor_name"],
                "metric_statistics_name": df.attrs["metric_name"],
                "metric_score": metric_score,
                "metric_computator": "",
                "metric_category": "low-level",
                "metric_score_unit": "",
                "time_series_unit": "",
                "time_series_timestamps": [],
                "time_series_values": [], #df.mean(axis=1, skipna=True),
                "time_series_selected_frames": [],
                }
            metrics[df.attrs["metric_name"]].append(pd.DataFrame([data]))

            scenario_scores[scenario_name]["scenario_type"] = scenario_type
            scenario_scores[scenario_name][metric_name] = metric_score

        global_scores = {}
        scenario_type_scores = defaultdict(dict)

        for metric_name, dataframes in metrics.items():
            save_path = self._metric_file_output_path / (metric_name + '.parquet')
            concat_pandas = pd.concat([*dataframes], ignore_index=True)
            # for final parquet
            global_mean = concat_pandas["metric_score"].mean(skipna=True)
            global_scores[metric_name] = global_mean
            # compute scenario type averages
            avg = concat_pandas.groupby("scenario_type", dropna=False)["metric_score"].mean().reset_index(name="metric_score")
            cnt = concat_pandas.groupby("scenario_type", dropna=False).size().reset_index(name="num_scenarios")
            summ = avg.merge(cnt, on="scenario_type", how="left")
            concat_pandas["num_scenarios"] = np.nan
            concat_pandas["aggregator_type"] = "average"

            # for final parquet
            for _, r in avg.iterrows():
                stype = r["scenario_type"]
                scenario_type_scores[stype][metric_name] = r["metric_score"]

            rows = []
            for _, r in summ.iterrows():
                row = {c: None for c in concat_pandas.columns}
                row["scenario_name"] = r["scenario_type"] 
                row["scenario_type"] = np.nan
                row["aggregator_type"] = "average"
                row["metric_statistics_name"] = metric_name 
                row["metric_score"] = r["metric_score"]  
                row["num_scenarios"] = float(r["num_scenarios"]) 
                rows.append(row)
            agg_rows = pd.DataFrame(rows, columns=concat_pandas.columns)
            concat_pandas = pd.concat([concat_pandas, agg_rows], ignore_index=True)
            # save results
            concat_pandas.to_parquet(safe_path_to_string(save_path))

        if global_scores:
            rows = []
            # Row for global average across all scenarios
            global_row = {"scenario_type": "final_score"}
            global_row.update(global_scores)
            rows.append(global_row)
            # One row per scenario_type
            for stype, metric_dict in scenario_type_scores.items():
                row = {"scenario_type": stype}
                row.update(metric_dict)
                rows.append(row)

            # One row per scenario
            for scen, metric_dict in scenario_scores.items():
                stype = metric_dict.get("scenario_type")
                metric_values = {
                    k: v for k, v in metric_dict.items() if k != "scenario_type"
                }
                row = {
                    "level": "scenario",
                    "scenario_type": stype,
                    "scenario_name": scen,
                }
                row.update(metric_values)
                rows.append(row)

            summary_df = pd.DataFrame(rows)
            # Optional: ensure column order: scenario_type first
            meta_cols = ["level", "scenario_type", "scenario_name"]
            metric_cols = [c for c in summary_df.columns if c not in meta_cols]
            summary_df = summary_df[meta_cols + metric_cols]

            summary_save_path = (
                self._metric_file_output_path / "open_loop_averaged_prediction_metrics.parquet"
            )
            summary_df.to_parquet(safe_path_to_string(summary_save_path))

class PredictorMetricCallback(AbstractCallback):
    """
    Writes into the same files
    """

    def __init__(self, output_directory, metric_dir):
        self._output_directory = Path(output_directory) # root directory for all logs and results
        self._metric_dir = metric_dir # where computed metrics are stored in parquet files

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        pass

    @timeit
    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        scenario_dir = self._get_scenario_folder(planner.name(), setup.scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        timestep_logs = planner.timestep_logs
        # compute temporal consistency
        logs = self.compute_temporal_consistency(timestep_logs)
        # generate metric dataframes
        metrics = list(planner.predictor.config.metrics.predict.keys()) + ["predicted_track_difficulty", "w2_dirac", "w2_dmm", "w2_gmm", "w2_gaussian"]
        dfs = self.logs_to_dataframes(timestep_logs, metrics)
        # save into scenario folder
        scenario_dir = self._get_scenario_folder(planner.name(), setup.scenario)
        for metric in dfs.keys():
            dfs[metric].to_parquet(Path(scenario_dir, metric).with_suffix(".parquet"))
        # generate planner costs dataframe and save to folder
        df_data = {}
        for t in range(logs["metadata"]["iterations"]):
            df_data[t] = logs[t]["planner_costs"]
            df_planner_costs = pd.DataFrame.from_dict(df_data, orient="index")
            df_planner_costs.index.name = "timestep"
            df_planner_costs.columns.name = "costs"
            df_planner_costs.attrs.update(logs["metadata"])
            df_planner_costs.attrs["metric_name"] = "planner_costs"
        df_planner_costs.to_parquet(Path(scenario_dir, "planner_costs").with_suffix(".parquet"))

        # # aggregate into per-simulation means
        # # timeseries = 
        # for metric in dfs.keys():
        #     dfs[metric].mean(skipna=True).mean()

    def _get_scenario_folder(self, planner_name: str, scenario: AbstractScenario) -> Path:
        """
        Compute scenario folder directory where all files will be stored.
        :param planner_name: planner name.
        :param scenario: for which to compute directory name.
        :return directory path.
        """
        return self._output_directory / self._metric_dir / f"{scenario.scenario_type}_{scenario.log_name}_{planner_name}"  # type: ignore
    

    def logs_to_dataframes(self, logs, metrics):
        """
        Convert logs dictionary to dictionary of dataframes with obstacles as columns and timesteps as index.
        
        Parameters:
        logs: dict with structure logs[timestep][metric_string][value_array] 
            and logs[timestep]['obstacle_id_string_list']
        
        Returns:
        dict of DataFrames, one for each metric
        """
        # Get all unique timesteps, metrics, and obstacles
        timesteps = range(logs["metadata"]["iterations"])
        # Get all metrics (excluding obstacle_id_string_list)
        metrics = set(metrics)
        # Get all unique obstacles across all timesteps
        unique_obstacle_ids = set()
        for t in timesteps:
            unique_obstacle_ids.update(logs[t]["predicted_object_ids"])
        # Create a DataFrame for each metric
        dataframes = {}
        for metric in metrics:
            df_data = {}
            for t in timesteps:
                # Fill in actual values if they exist
                obstacle_ids = logs[t]["predicted_object_ids"]
                values = logs[t].get(metric) #np.full(len(obstacle_ids), np.nan)

                if values is None:
                    values = np.full(len(obstacle_ids), np.nan)

                if len(obstacle_ids) > 0 and len(values) != len(obstacle_ids):
                    raise ValueError(
                        f"Length mismatch at t={t} for metric '{metric}': "
                        f"{len(values)=}, {len(obstacle_ids)=}"
                    )

                # Map only values we know about at this timestep
                metrics_by_id = dict(zip(obstacle_ids, values))

                # Now fill *all* unique obstacle IDs, NaN if missing
                timestep_data = {
                    oid: metrics_by_id.get(oid, np.nan)
                    for oid in unique_obstacle_ids
                }

                df_data[t] = timestep_data
            # Create DataFrame (transpose so timesteps are rows, obstacles are columns)
            df = pd.DataFrame.from_dict(df_data, orient='index')
            df.index.name = 'timestep'
            df.columns.name = 'obstacle_id'
            df.attrs.update(logs["metadata"])
            df.attrs["metric_name"] = metric
            dataframes[metric] = df
        return dataframes
    
    def compute_temporal_consistency(self, logs):
        log_std_range=(-1.609, 5.0)
        def construct_covariance(sigma_x, sigma_y, rho):
            C = np.stack([
                np.stack([sigma_x**2, rho * sigma_x * sigma_y], axis=-1),
                np.stack([rho * sigma_x * sigma_y, sigma_y**2], axis=-1)
            ], axis=-2)
            return C

        fillers = [np.nan for oid in logs[0]["predicted_object_ids"]]
        logs[0].update({"w2_dirac": fillers, "w2_gaussian": fillers, "w2_dmm": fillers, "w2_gmm": fillers})

        for t in range(1, logs["metadata"]["iterations"]):
            obstacle_ids = logs[t]["predicted_object_ids"]
            consistency_metrics = {"w2_dirac": [], "w2_gaussian": [], "w2_dmm": [], "w2_gmm": []}
            for oidx_curr, oid in enumerate(obstacle_ids):
                if oid not in logs[t-1]["predicted_object_ids"]:
                    # if obstacle hasnt been predicted in previous timestep, cant compute consistency
                    consistency_metrics["w2_dirac"].append(np.nan)
                    consistency_metrics["w2_dmm"].append(np.nan)
                    consistency_metrics["w2_gaussian"].append(np.nan)
                    consistency_metrics["w2_gmm"].append(np.nan)
                else: #otherwise compute consistency metrics
                    # sigma_x = torch.exp(torch.clip(predicted_traj[..., 2], min=log_std_range[0], max=log_std_range[1]))  # shape (B, m, T)
                    # sigma_y = torch.exp(torch.clip(predicted_traj[..., 3], min=log_std_range[0], max=log_std_range[1]))  # shape (B, m, T)
                    # rho     = torch.clip(predicted_traj[..., 4], -0.5, 0.5)  # shape (B, m, T)
                    # previous prediction
                    oidx_prev = logs[t-1]["predicted_object_ids"].index(oid)
                    xy_prev = logs[t-1]["predicted_trajectory"][oidx_prev,:,-1,:2]
                    pi_prev = logs[t-1]["predicted_probability"][oidx_prev]
                    sigma_x_prev = np.exp(np.clip(logs[t-1]["predicted_trajectory"][oidx_prev,:,-1,2], a_min=log_std_range[0], a_max=log_std_range[1]))
                    sigma_y_prev = np.exp(np.clip(logs[t-1]["predicted_trajectory"][oidx_prev,:,-1,3], a_min=log_std_range[0], a_max=log_std_range[1]))
                    rho_prev = np.clip(logs[t-1]["predicted_trajectory"][oidx_prev,:,-1,4], a_min=-0.5, a_max= 0.5)
                    C_prev = construct_covariance(sigma_x_prev, sigma_y_prev, rho_prev)
                    top1_prev = np.argmax(pi_prev)
                    # current prediction
                    xy_curr = logs[t]["predicted_trajectory"][oidx_curr,:,-2,:2]
                    pi_curr = logs[t]["predicted_probability"][oidx_curr]
                    sigma_x_curr = np.exp(np.clip(logs[t]["predicted_trajectory"][oidx_curr,:,-2,2], a_min=log_std_range[0], a_max=log_std_range[1]))
                    sigma_y_curr = np.exp(np.clip(logs[t]["predicted_trajectory"][oidx_curr,:,-2,3], a_min=log_std_range[0], a_max=log_std_range[1]))
                    rho_curr = np.clip(logs[t]["predicted_trajectory"][oidx_curr,:,-2,4], a_min=-0.5, a_max=0.5)
                    C_curr = construct_covariance(sigma_x_curr, sigma_y_curr, rho_curr)
                    top1_curr = np.argmax(pi_curr)
                    # uncertainty-unaware optimal transport
                    distances = ot.dist(xy_prev, xy_curr, metric="euclidean")**2
                    consistency_metrics["w2_dirac"].append(np.sqrt(distances[top1_prev,top1_curr]))
                    consistency_metrics["w2_dmm"].append(np.sqrt(ot.emd2(pi_prev, pi_curr, distances)))
                    # uncertainty-aware optimal transport
                    consistency_metrics["w2_gaussian"].append(ot.gaussian.bures_wasserstein_distance(xy_prev[top1_prev], xy_curr[top1_curr], C_prev[top1_prev], C_curr[top1_curr]))
                    consistency_metrics["w2_gmm"].append(ot.gmm.gmm_ot_loss(xy_prev, xy_curr, C_prev, C_curr, pi_prev, pi_curr))
            logs[t].update(consistency_metrics)
        return logs



class ResetPlannerCallback(AbstractCallback):
    """
    Writes into the same files
    """

    def __init__(self):
        pass

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        del planner.predictor # free cuda memory after simulation
        