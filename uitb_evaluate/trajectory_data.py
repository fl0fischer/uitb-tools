import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy import stats

import os
import pickle
import itertools
import warnings
import xmltodict
import logging

logging.getLogger().setLevel(logging.INFO)

PLOTS_DIR_DEFAULT = os.path.abspath("../_plots/")

INDEPENDENT_JOINTS = [  # 'eye-rx',
    # 'eye-ry',
    'elv_angle',
    'shoulder_elv',
    'shoulder_rot',
    'elbow_flexion',
    'pro_sup',
    # 'deviation',
    # 'flexion'
]

ACTUATOR_NAMES = ['DELT1',
                  'DELT2',
                  'DELT3',
                  'SUPSP',
                  'INFSP',
                  'SUBSC',
                  'TMIN',
                  'TMAJ',
                  'PECM1',
                  'PECM2',
                  'PECM3',
                  'LAT1',
                  'LAT2',
                  'LAT3',
                  'CORB',
                  'TRIlong',
                  'TRIlat',
                  'TRImed',
                  'ANC',
                  'SUP',
                  'BIClong',
                  'BICshort',
                  'BRA',
                  'BRD',
                  # 'ECRL',
                  # 'ECRB',
                  # 'ECU',
                  # 'FCR',
                  # 'FCU',
                  # 'PL',
                  'PT',
                  'PQ',
                  # 'FDSL',
                  # 'FDSR',
                  # 'FDSM',
                  # 'FDSI',
                  # 'FDPL',
                  # 'FDPR',
                  # 'FDPM',
                  # 'FDPI',
                  # 'EDCL',
                  # 'EDCR',
                  # 'EDCM',
                  # 'EDCI',
                  # 'EDM',
                  # 'EIP',
                  # 'EPL',
                  # 'EPB',
                  # 'FPL',
                  # 'APL'
                  ]

ACTUATOR_NAMES_DICT = {'DELT1': 'deltoid1_r',
                       'DELT2': 'deltoid2_r',
                       'DELT3': 'deltoid3_r',
                       'SUPSP': 'supraspinatus_r',
                       'INFSP': 'infraspinatus_r',
                       'SUBSC': 'subscapularis_r',
                       'TMIN': 'teres_minor_r',
                       'TMAJ': 'teres_major_r',
                       'PECM1': 'pectoralis_major1_r',
                       'PECM2': 'pectoralis_major2_r',
                       'PECM3': 'pectoralis_major3_r',
                       'LAT1': 'latissimus_dorsi1_r',
                       'LAT2': 'latissimus_dorsi2_r',
                       'LAT3': 'latissimus_dorsi3_r',
                       'CORB': 'coracobrachialis_r',
                       'TRIlong': 'triceps_longhead_r',
                       'TRIlat': 'triceps_lateralis_r',
                       'TRImed': 'triceps_medialis_r',
                       'ANC': 'anconeus_r',
                       'SUP': 'supinator_brevis_r',
                       'BIClong': 'biceps_longhead_r',
                       'BICshort': 'biceps_shorthead_r',
                       'BRA': 'brachialis_r',
                       'BRD': 'brachioradialis_r',
                       'PT': 'pronator_teres_r',
                       'PQ': 'pron_quad_r'}


### BASE CLASS FOR TRAJECTORY DATA
class TrajectoryData(object):
    SHOW_MINJERK = False  # if this is set to True, end-effector methods yield MinJerk trajectories instead of actual trajectories!

    def __init__(self):
        self.initialized = True
        self.preprocessed = False
        self.trials_defined = False

    # methods to override:
    # ----------------------------

    def preprocess(self):
        raise NotImplementedError

    def compute_indices(self):
        raise NotImplementedError

    # read-only properties [the respective internal attributes must be set in the subclass!]:
    # -----------------------------

    @property
    def position_series(self):
        if self.SHOW_MINJERK:
            return self._minjerk_position_series
        return self._position_series

    @property
    def velocity_series(self):
        if self.SHOW_MINJERK:
            return self._minjerk_velocity_series
        return self._velocity_series

    @property
    def acceleration_series(self):
        if self.SHOW_MINJERK:
            return self._minjerk_acceleration_series
        return self._acceleration_series

    @property
    def qpos_series(self):
        return self._qpos_series

    @property
    def qvel_series(self):
        return self._qvel_series

    @property
    def qacc_series(self):
        return self._qacc_series

    @property
    def act_series(self):
        return self._act_series

    @property
    def target_position_series(self):
        return self._target_position_series

    @property
    def target_radius_series(self):
        return self._target_radius_series

    @property
    def target_idx_series(self):
        return self._target_idx_series

    @property
    def time_series(self):
        return self._time_series

    @property
    def time_per_step(self):
        return self._time_per_step

    @property
    def indices(self):
        return self._indices

    @property
    def distance_to_target_series(self):
        return self._distance_to_target_series

    @property
    def action_series(self):
        return self._action_series

    @property
    def control_series(self):
        return self._control_series

    @property
    def reward_series(self):
        return self._reward_series

    # -----------------------------

    def compute_minjerk(self, MINJERK_USER_CONSTRAINTS=True, targetbound_as_target=False):
        self.minjerk_targetbound_as_target = targetbound_as_target

        assert self.initialized and self.preprocessed and self.trials_defined, "ERROR: Need to call preprocess() and compute_indices() first! "

        ## REFERENCE TRAJECTORY: MinJerk
        self._minjerk_position_series = np.zeros_like(np.squeeze(self._position_series)) * np.nan
        self._minjerk_velocity_series = np.zeros_like(np.squeeze(self._velocity_series)) * np.nan
        self._minjerk_acceleration_series = np.zeros_like(np.squeeze(self._acceleration_series)) * np.nan

        for episode_index_current, (last_idx, current_idx, next_idx) in enumerate(self.selected_movements_indices):
            if isinstance(last_idx, list):
                assert len(
                    last_idx) == 1, "Indices are prepared for computing summary statistics. Use compute_statistics() instead."  # e.g., if len(AGGREGATION_VARS) > 0
                last_idx = last_idx[0]
                current_idx = current_idx[0]
                next_idx = next_idx[0]

            if self.minjerk_targetbound_as_target:
                next_idx_copy = next_idx
                try:  # use "target_radius" column
                    assert len(np.unique(self.target_radius_series[
                                         current_idx + 1: next_idx])) == 1, "ERROR: Target radius changes during movement! Cannot find reliable information about when target has been entered.\nFix dataset or use target center as distance (set 'targetbound_as_target=False')."
                    relindices_insidetarget_trial = np.where(np.linalg.norm(
                        np.array(self.position_series[current_idx + 1: next_idx]) - np.array(
                            self.target_position_series[current_idx + 1: next_idx]), axis=1) <
                                                             self.target_radius_series[current_idx + 1])[0]
                except:  # use "inside_target" column
                    relindices_insidetarget_trial = \
                    np.where(self.data[self.data_key]["inside_target"][current_idx + 1:next_idx])[0]
                assert len(
                    relindices_insidetarget_trial) > 0, "Using target boundary instead of target center as desired target position failed."
                targetbound_idx = (current_idx + 1 + relindices_insidetarget_trial[
                    0]) + 1  # adding 1: -> targetbound_idx corresponds to first index that is not included anymore in considered trial series
                next_idx = targetbound_idx
                # input((next_idx_copy, next_idx))

            if MINJERK_USER_CONSTRAINTS:
                T = [np.concatenate((self._position_series[current_idx], self._position_series[next_idx - 1]))]
            else:
                T = [np.squeeze([self._target_position_series[last_idx] if last_idx >= 0 else self._position_series[0],
                                 self._target_position_series[current_idx]]).reshape(-1, )]
            x0 = np.concatenate((np.squeeze([self._position_series[current_idx], self._velocity_series[current_idx],
                                             self._acceleration_series[current_idx]]).reshape(-1, ), T[0]))
            dim = 3
            x_minjerk, u_minjerk = minimumjerk_deterministic(next_idx - current_idx - 1, x0=x0, T=T,
                                                             final_vel=self._velocity_series[next_idx - 1],
                                                             final_acc=self._acceleration_series[next_idx - 1], P=2,
                                                             dim=dim, dt=self._time_per_step, initialuservalues=None)
            self._minjerk_position_series[current_idx: next_idx] = x_minjerk[:, :dim]
            self._minjerk_velocity_series[current_idx: next_idx] = x_minjerk[:, dim:2 * dim]
            self._minjerk_acceleration_series[current_idx: next_idx] = x_minjerk[:, 2 * dim:3 * dim]

    def get_statistics_info(self):
        if hasattr(self, "stats_episode_index_current"):
            return {"episode_index_current": self.stats_episode_index_current,
                    "compute_deviation": self.stats_compute_deviation, "normalize_time": self.stats_normalize_time}

    def compute_statistics(self, episode_index_current, effective_projection_path=False, targetbound_as_target=False,
                           compute_deviation=False, normalize_time=False, use_joint_data_only=False):
        self.stats_episode_index_current = episode_index_current
        self.stats_effective_projection_path = effective_projection_path
        self.stats_targetbound_as_target = targetbound_as_target
        self.stats_compute_deviation = compute_deviation
        self.stats_normalize_time = normalize_time

        self.stats_use_joint_data_only = use_joint_data_only

        if normalize_time:
            logging.warning(
                f"NORMALIZE_TIME was set to True, but is has no effect on the computation of distributions (mean, variability, etc.)!")

        last_idx_hlp, current_idx_hlp, next_idx_hlp = self.selected_movements_indices[episode_index_current]

        assert isinstance(last_idx_hlp,
                          list), "No data to aggregate. Use compute_trial() instead."  # e.g., if len(AGGREGATION_VARS) > 0

        next_idx_hlp_copy = next_idx_hlp.copy()
        if self.stats_targetbound_as_target:
            # compute (relative) indices at which end-effector is inside target ("relindices_insidetarget_trial") ["+ 1" is used to avoid errors if target switches one step too late...] #TODO: ensure that target switch indices are exact!
            targetbound_idx_hlp = []
            for current_idx, next_idx in zip(current_idx_hlp, next_idx_hlp):
                try:  # use "target_radius" column
                    assert len(np.unique(self.target_radius_series[
                                         current_idx + 1: next_idx])) == 1, "ERROR: Target radius changes during movement! Cannot find reliable information about when target has been entered.\nFix dataset or use target center as distance (set 'targetbound_as_target=False')."
                    relindices_insidetarget_trial = np.where(np.linalg.norm(
                        np.array(self.position_series[current_idx + 1: next_idx]) - np.array(
                            self.target_position_series[current_idx + 1: next_idx]), axis=1) <
                                                             self.target_radius_series[current_idx + 1])[0]
                except:  # use "inside_target" column
                    relindices_insidetarget_trial = \
                    np.where(self.data[self.data_key]["inside_target"][current_idx + 1:next_idx])[0]

                assert len(
                    relindices_insidetarget_trial) > 0, "Using target boundary instead of target center as desired target position failed."
                targetbound_idx = (current_idx + 1 + relindices_insidetarget_trial[
                    0]) + 1  # adding 1: -> targetbound_idx corresponds to first index that is not included anymore in considered trial series
                targetbound_idx_hlp.append(targetbound_idx)
            next_idx_hlp = targetbound_idx_hlp

        if not self.stats_use_joint_data_only:
            if self.stats_effective_projection_path:  # ensures that first and last value of projected_trajectories_pos_trial equal 0 and 1, respectively
                init_val = self.position_series[current_idx_hlp]
                final_val = self.position_series[[i - 1 for i in next_idx_hlp]]
            else:
                init_val = np.unique(self.target_position_series[[i + 1 for i in last_idx_hlp if i >= 0]],
                                     axis=0).reshape(-1, )
                final_val = np.unique(self.target_position_series[[i + 1 for i in current_idx_hlp if i >= 0]],
                                      axis=0).reshape(-1, )
                assert init_val.shape == (3,), "ERROR: Cannot reliably determine (nominal) initial position."
                assert final_val.shape == (3,), "ERROR: Cannot reliably determine (nominal) target position."
                assert init_val is not final_val, "ERROR: Initial and target position do not differ!"

            self.init_val = init_val
            self.final_val = final_val

            # for current_idx, next_idx in zip(current_idx_hlp, next_idx_hlp):
            #     input((self.init_val, self.position_series[current_idx], self.final_val, self.position_series[next_idx]))

            self.projected_trajectories_pos_mean, self.projected_trajectories_pos_cov, self.projected_trajectories_pos_min, self.projected_trajectories_pos_max = compute_trajectory_statistics(
                self.position_series, current_idx_hlp, next_idx_hlp, project=True, init_val=init_val,
                final_val=final_val, use_rel_vals=True, output_deviation=compute_deviation)
            self.projected_trajectories_vel_mean, self.projected_trajectories_vel_cov, self.projected_trajectories_vel_min, self.projected_trajectories_vel_max = compute_trajectory_statistics(
                self.velocity_series, current_idx_hlp, next_idx_hlp, project=True, init_val=init_val,
                final_val=final_val, use_rel_vals=False, output_deviation=compute_deviation)
            self.projected_trajectories_acc_mean, self.projected_trajectories_acc_cov, self.projected_trajectories_acc_min, self.projected_trajectories_acc_max = compute_trajectory_statistics(
                self.acceleration_series, current_idx_hlp, next_idx_hlp, project=True, init_val=init_val,
                final_val=final_val, use_rel_vals=False, output_deviation=compute_deviation)

        self.qpos_series_mean, self.qpos_series_cov, self.qpos_series_min, self.qpos_series_max = compute_trajectory_statistics(
            self.qpos_series, current_idx_hlp, next_idx_hlp)
        self.qvel_series_mean, self.qvel_series_cov, self.qvel_series_min, self.qvel_series_max = compute_trajectory_statistics(
            self.qvel_series, current_idx_hlp, next_idx_hlp)
        self.qacc_series_mean, self.qacc_series_cov, self.qacc_series_min, self.qacc_series_max = compute_trajectory_statistics(
            self.qacc_series, current_idx_hlp, next_idx_hlp)
        self.target_pos_mean, self.target_pos_cov, _, _ = compute_trajectory_statistics(self.target_position_series,
                                                                                        current_idx_hlp, next_idx_hlp)
        self.target_radius_mean, self.target_radius_cov, _, _ = compute_trajectory_statistics(self.target_radius_series,
                                                                                              current_idx_hlp,
                                                                                              next_idx_hlp)
        self.target_idx_mean, self.target_idx_cov, _, _ = compute_trajectory_statistics(self.target_idx_series,
                                                                                        current_idx_hlp, next_idx_hlp)
        if not np.isnan(self.target_idx_mean).all():
            self.target_idx_mean = self.target_idx_mean.astype(int)
        _, self.time_series_cov, self.time_series_extended, _ = compute_trajectory_statistics(self.time_series,
                                                                                              current_idx_hlp,
                                                                                              next_idx_hlp,
                                                                                              rel_to_init=True,
                                                                                              normalize=normalize_time)
        if not normalize_time and not (
                np.isclose(self.time_series_cov, 0).all() or np.isnan(self.time_series_cov).all()):
            raise ValueError(f"ERROR: Ensure that time series are correctly aligned.")
        self.distance_to_target_mean, self.distance_to_target_cov, self.distance_to_target_min, self.distance_to_target_max = compute_trajectory_statistics(
            self.distance_to_target_series, current_idx_hlp, next_idx_hlp)
        if hasattr(self, "_distance_to_joystick_series"):
            self.distance_to_joystick_mean, self.distance_to_joystick_cov, self.distance_to_joystick_min, self.distance_to_joystick_max = compute_trajectory_statistics(
                self._distance_to_joystick_series, current_idx_hlp, next_idx_hlp)

    def compute_action_statistics(self, episode_index_current, targetbound_as_target=False, normalize_time=False):
        self.action_stats_episode_index_current = episode_index_current
        self.action_stats_targetbound_as_target = targetbound_as_target
        self.action_stats_normalize_time = normalize_time

        if normalize_time:
            logging.warning(
                f"NORMALIZE_TIME was set to True, but is has no effect on the computation of distributions (mean, variability, etc.)!")

        last_idx_hlp, current_idx_hlp, next_idx_hlp = self.selected_movements_indices[episode_index_current]

        assert isinstance(last_idx_hlp,
                          list), "No data to aggregate. Use compute_trial() instead."  # e.g., if len(AGGREGATION_VARS) > 0

        next_idx_hlp_copy = next_idx_hlp.copy()
        if self.action_stats_targetbound_as_target:
            # compute (relative) indices at which end-effector is inside target ("relindices_insidetarget_trial") ["+ 1" is used to avoid errors if target switches one step too late...] #TODO: ensure that target switch indices are exact!
            targetbound_idx_hlp = []
            for current_idx, next_idx in zip(current_idx_hlp, next_idx_hlp):
                try:  # use "target_radius" column
                    assert len(np.unique(self.target_radius_series[
                                         current_idx + 1: next_idx])) == 1, "ERROR: Target radius changes during movement! Cannot find reliable information about when target has been entered.\nFix dataset or use target center as distance (set 'targetbound_as_target=False')."
                    relindices_insidetarget_trial = np.where(np.linalg.norm(
                        np.array(self.position_series[current_idx + 1: next_idx]) - np.array(
                            self.target_position_series[current_idx + 1: next_idx]), axis=1) <
                                                             self.target_radius_series[current_idx + 1])[0]
                except:  # use "inside_target" column
                    relindices_insidetarget_trial = \
                    np.where(self.data[self.data_key]["inside_target"][current_idx + 1:next_idx])[0]

                assert len(
                    relindices_insidetarget_trial) > 0, "Using target boundary instead of target center as desired target position failed."
                targetbound_idx = (current_idx + 1 + relindices_insidetarget_trial[
                    0]) + 1  # adding 1: -> targetbound_idx corresponds to first index that is not included anymore in considered trial series
                targetbound_idx_hlp.append(targetbound_idx)
            next_idx_hlp = targetbound_idx_hlp

        if hasattr(self, "_action_series"):
            self.action_series_mean, self.action_series_cov, self.action_series_min, self.action_series_max = compute_trajectory_statistics(
                self.action_series, current_idx_hlp, next_idx_hlp)
        if hasattr(self, "_control_series"):
            self.control_series_mean, self.control_series_cov, self.control_series_min, self.control_series_max = compute_trajectory_statistics(
                self.control_series, current_idx_hlp, next_idx_hlp)
        if hasattr(self, "_reward_series"):
            self.reward_series_mean, self.reward_series_cov, self.reward_series_min, self.reward_series_max = compute_trajectory_statistics(
                self.reward_series, current_idx_hlp, next_idx_hlp)
        _, self.action_stats_time_series_cov, self.action_stats_time_series_extended, _ = compute_trajectory_statistics(
            self.time_series, current_idx_hlp, next_idx_hlp, rel_to_init=True, normalize=normalize_time)
        if not normalize_time and not (np.isclose(self.action_stats_time_series_cov, 0).all() or np.isnan(
                self.action_stats_time_series_cov).all()):
            raise ValueError(f"ERROR: Ensure that time series are correctly aligned.")

    def get_trial_info(self):
        if hasattr(self, "trial_index_current"):
            return {"trial_index_current": self.trial_index_current, "compute_deviation": self.trial_compute_deviation,
                    "normalize_time": self.trial_normalize_time, "joint_id": self.trial_joint_id}

    def compute_trial(self, trial_index_current, effective_projection_path=False, targetbound_as_target=False,
                      dwell_time=0, compute_deviation=False, normalize_time=False):
        self.trial_index_current = trial_index_current
        self.trial_effective_projection_path = effective_projection_path
        self.trial_targetbound_as_target = targetbound_as_target
        self.trial_dwell_time = dwell_time  # in seconds; used for movement time computation (e.g., for Fitts' Law)
        self.trial_compute_deviation = compute_deviation
        self.trial_normalize_time = normalize_time

        last_idx, current_idx, next_idx = self.selected_movements_indices[trial_index_current]

        if isinstance(last_idx, list):
            assert len(
                last_idx) == 1, "Indices are prepared for computing summary statistics. Use compute_statistics() instead."  # e.g., if len(AGGREGATION_VARS) > 0
            last_idx = last_idx[0]
            current_idx = current_idx[0]
            next_idx = next_idx[0]

        # compute (relative) indices at which end-effector is inside target ("relindices_insidetarget_trial") ["+ 1" is used to avoid errors if target switches one step too late...] #TODO: ensure that target switch indices are exact!
        try:  # use "target_radius" column
            assert len(np.unique(self.target_radius_series[
                                 current_idx + 1: next_idx])) == 1, "ERROR: Target radius changes during movement! Cannot find reliable information about when target has been entered.\nFix dataset or use target center as distance (set 'targetbound_as_target=False')."
            self.relindices_insidetarget_trial = np.where(np.linalg.norm(
                np.array(self.position_series[current_idx + 1: next_idx]) - np.array(
                    self.target_position_series[current_idx + 1: next_idx]), axis=1) < self.target_radius_series[
                                                              current_idx + 1])[0]
        except:  # use "inside_target" column
            self.relindices_insidetarget_trial = \
            np.where(self.data[self.data_key]["inside_target"][current_idx + 1:next_idx])[0]

        next_idx_copy = next_idx
        if self.trial_targetbound_as_target:
            assert len(
                self.relindices_insidetarget_trial) > 0, "Using target boundary instead of target center as desired target position failed."
            targetbound_idx = (current_idx + 1 + self.relindices_insidetarget_trial[
                0]) + 1  # adding 1: -> targetbound_idx corresponds to first index that is not included anymore in considered trial series
            next_idx = targetbound_idx

        if effective_projection_path:  # ensures that first and last value of projected_trajectories_pos_trial equal 0 and 1, respectively
            init_val = self.position_series[current_idx]
            final_val = self.position_series[next_idx - 1]
        else:
            init_val = self.target_position_series[last_idx + 1] if last_idx >= 0 else self.position_series[0]
            final_val = self.target_position_series[current_idx + 1]
            # TODO: use target boundary position as final_val if self.trial_targetbound_as_target?

        self.init_val = init_val
        self.final_val = final_val

        time_series_shifted = np.array(self.time_series[current_idx: next_idx]) - self.time_series[current_idx]
        if normalize_time:
            time_series_shifted = (time_series_shifted - time_series_shifted[0]) / (
                        time_series_shifted[-1] - time_series_shifted[0])
        self.time_series_trial = time_series_shifted

        self.target_position_series_trial = self.target_position_series[current_idx: next_idx]
        self.target_radius_series_trial = self.target_radius_series[current_idx: next_idx]
        self.target_idx_series_trial = self.target_idx_series[current_idx: next_idx]

        self.position_series_trial = self.position_series[current_idx: next_idx]
        self.velocity_series_trial = self.velocity_series[current_idx: next_idx]
        self.acceleration_series_trial = self.acceleration_series[current_idx: next_idx]
        self.projected_trajectories_pos_trial = project_trajectory(self.position_series_trial, init_val=init_val,
                                                                   final_val=final_val, use_rel_vals=True,
                                                                   normalize_quantity=True,
                                                                   output_deviation=compute_deviation)
        self.projected_trajectories_vel_trial = project_trajectory(self.velocity_series_trial, init_val=init_val,
                                                                   final_val=final_val, use_rel_vals=False,
                                                                   normalize_quantity=False,
                                                                   output_deviation=compute_deviation)
        self.projected_trajectories_acc_trial = project_trajectory(self.acceleration_series_trial, init_val=init_val,
                                                                   final_val=final_val, use_rel_vals=False,
                                                                   normalize_quantity=False,
                                                                   output_deviation=compute_deviation)
        self.qpos_series_trial = self.qpos_series[current_idx: next_idx, :]
        self.qvel_series_trial = self.qvel_series[current_idx: next_idx, :]
        self.qacc_series_trial = self.qacc_series[current_idx: next_idx, :]

        self.distance_to_target_trial = self.distance_to_target_series[current_idx: next_idx]

        # Optional attributes
        for attr in ["act_series", "action_series", "control_series", "reward_series"]:
            if hasattr(self, f"_{attr}") and getattr(self, f"_{attr}") is not None:
                setattr(self, f"{attr}_trial", getattr(self, f"_{attr}")[current_idx: next_idx, ...])

        # Statistics
        self.target_width_trial = 2 * self.target_radius_series[current_idx]
        self.target_distance_trial = np.linalg.norm(self.final_val - self.init_val)
        # self.fitts_ID_trial = np.log2(2*(self.target_distance_trial/self.target_width_trial))
        self.fitts_ID_trial = np.log2((self.target_distance_trial / self.target_width_trial) + 1)
        if self.trial_targetbound_as_target:
            self.effective_MT_trial = (next_idx - current_idx) * self.time_per_step  # here: next_idx <- targetbound_idx
        else:
            self.effective_MT_trial = (next_idx - current_idx) * self.time_per_step - self.trial_dwell_time

        # compute number of target (re-)entries and covariance of end-effector position when inside target (i.e., only for "relindices_insidetarget_trial")
        target_re_entries_meta_indices = np.where(np.diff(self.relindices_insidetarget_trial) != 1)[0] + 1
        self.num_target_entries_trial = len(
            target_re_entries_meta_indices) + 1 if target_re_entries_meta_indices.size > 0 else 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            position_series_trial_insidetarget = np.array(self.position_series[current_idx + 1: next_idx_copy])[
                self.relindices_insidetarget_trial]
            self.endeffector_insidetarget_covariance = np.cov(position_series_trial_insidetarget, rowvar=False) if \
            position_series_trial_insidetarget.shape[0] != 1 else np.zeros(
                (position_series_trial_insidetarget.shape[1], position_series_trial_insidetarget.shape[1]))


class TrajectoryData_MultipleInstances(TrajectoryData):
    def __init__(self, trajectories):
        super(TrajectoryData_MultipleInstances, self).__init__()

        assert isinstance(trajectories, list) or isinstance(trajectories,
                                                            tuple), "Trajectory list has wrong type (only lists or tuples are valid)."
        assert all([isinstance(trajectory, TrajectoryData) for trajectory in
                    trajectories]), "Invalid element in trajectory list (all elements need to be 'TrajectoryData' instances)."
        self.trajectories = trajectories

        assert all([trajectory.preprocessed for trajectory in
                    trajectories]), "ERROR: All TrajectoryData elements need to be preprocessed!"
        assert all([not trajectory.trials_defined for trajectory in
                    trajectories]), "ERROR: Trials were already selected for at least one of the used TrajectoryData elements. Ensure that this MultipleInheritance instance is generated directly after preprocessing the individual TrajectoryData instances."

        # Combine indices (makes use of "combine_indices()" function from used TrajectoryData class)
        assert all([type(i) == type(j) for i, j in itertools.combinations(self.trajectories,
                                                                          2)]), "ERROR: Cannot combine indices, since instances of different classes are used."
        indices_list = [i._indices for i in self.trajectories]
        n_samples_list = [len(i.position_series) for i in self.trajectories]
        self.trajectories[0].__class__.combine_indices(self, indices_list, n_samples_list)
        self.preprocessed = True

    def __getattr__(self, attr):
        # print(f"__getattr__ is used.")
        if not all([hasattr(trajectory, attr) for trajectory in self.trajectories]):
            raise AttributeError(f"'{attr}' is not available for all chosen TrajectoryData instances.")
        attr_values = [getattr(trajectory, attr) for trajectory in self.trajectories]
        if all([isinstance(value, str) for value in attr_values]) and all(
                [i == j for i, j in itertools.combinations(attr_values, 2)]):
            value = attr_values[0]
        elif all([isinstance(value, np.ndarray) for value in attr_values]) or all(
                [isinstance(value, list) for value in attr_values]):
            value = np.concatenate(attr_values)
        elif all([isinstance(value, pd.core.arrays.integer.IntegerArray) for value in attr_values]):
            value = pd.array(np.concatenate(attr_values), dtype="Int64")
        elif all([isinstance(value, int) or isinstance(value, float) for value in attr_values]):
            value = np.array([attr_values])
        else:
            raise TypeError(
                f"Attr '{attr}': One of the Trajectory Data types ({[type(value) for value in attr_values]}) cannot be used for concatenation.")
        setattr(self, attr, value)
        return value

    def compute_indices(self, *args, **kwargs):
        """
        Computes self.selected_movements_indices based on self.indices, which in turn was computed in this class' __init__() using combine_indices() from respective TrajectoryData class
        """
        assert all([type(i) == type(j) for i, j in itertools.combinations(self.trajectories,
                                                                          2)]), "ERROR: Cannot compute selected indices, since instances of different classes are used."
        self.trajectories[0].__class__.compute_indices(self, *args, **kwargs)

        self.trials_defined = True


class TrajectoryData_RL(TrajectoryData):

    def __init__(self, DIRNAME_SIMULATION, filename, REPEATED_MOVEMENTS=False,
                 independent_joints=None):

        self.DIRNAME_SIMULATION = DIRNAME_SIMULATION
        self.filepath = os.path.join(self.DIRNAME_SIMULATION, filename)  # warning: here, "self.filepath" is directory!
        self.REPEATED_MOVEMENTS = REPEATED_MOVEMENTS  # if True, combine individual log files into one data structure

        if independent_joints is None:
            self.independent_joints = [  # 'eye-rx',
                # 'eye-ry',
                'elv_angle',
                'shoulder_elv',
                'shoulder_rot',
                'elbow_flexion',
                'pro_sup',
                'deviation',
                'flexion'
            ]
        else:
            self.independent_joints = independent_joints

        super().__init__()

        if self.REPEATED_MOVEMENTS:
            rep_movs_data = {}
            rep_movs_data_action = {}
            for subdir in [i for i in os.listdir((os.path.expanduser(self.filepath))) if
                           os.path.isdir(os.path.join(os.path.expanduser(self.filepath), i))]:
                subdir_abs = os.path.join(os.path.expanduser(self.filepath), subdir)
                for subsubdir in [i for i in os.listdir(subdir_abs) if os.path.isdir(os.path.join(subdir_abs, i))]:
                    subsubdir_abs = os.path.join(os.path.expanduser(self.filepath), subdir, subsubdir)
                    rep_movs_filepath = os.path.join(subsubdir_abs, "state_log.pickle")
                    rep_movs_filepath_action = os.path.join(subsubdir_abs, "action_log.pickle")
                    with open(os.path.expanduser(rep_movs_filepath), "rb") as f:
                        helper = pickle.load(f)
                        for k, v in helper.items():
                            rep_movs_data[f"{subdir}__{subsubdir}__{k}"] = v
                    with open(os.path.expanduser(rep_movs_filepath_action), "rb") as f:
                        helper = pickle.load(f)
                        for k, v in helper.items():
                            rep_movs_data_action[f"{subdir}__{subsubdir}__{k}"] = v
            self.data = rep_movs_data
            self.data_action = rep_movs_data_action
        else:
            with open(os.path.join(self.filepath, "state_log.pickle"), "rb") as f:
                self.data = pickle.load(f)
            with open(os.path.join(self.filepath, "action_log.pickle"), "rb") as f:
                self.data_action = pickle.load(f)

        self.EPISODE_ID_NUMS = len(
            np.unique(list(map(lambda x: x.split('episode_')[-1].split('_')[0], self.data.keys()))))
        print(f"{self.EPISODE_ID_NUMS} episodes identified.")

        self.data_copy = self.data.copy()
        self.data_action_copy = self.data_action.copy()

    def preprocess(self, MOVEMENT_IDS=None, RADIUS_IDS=None, EPISODE_IDS=None, split_trials=True,
                   endeffector_name="fingertip", target_name="target"):
        # INFO: "MOVEMENT_IDS", "RADIUS_IDS", and "EPISODE_IDS" should be array-like (list, np.array, range-object, etc.)
        self.endeffector_name = endeffector_name
        self.target_name = target_name

        self.EPISODE_IDS = [str(EPISODE_ID).zfill(len(list(self.data.keys())[0].split("episode_")[-1].split("__")[0]))
                            for EPISODE_ID in EPISODE_IDS or np.unique(
                [int(i.split("episode_")[-1].split("__")[0]) for i in list(self.data.keys())])]

        if self.REPEATED_MOVEMENTS:
            self.MOVEMENT_IDS = [
                str(MOVEMENT_ID).zfill(len(list(self.data.keys())[0].split("movement_")[-1].split("__")[0])) for
                MOVEMENT_ID in MOVEMENT_IDS or np.unique(
                    [int(i.split("movement_")[-1].split("__")[0]) for i in list(self.data.keys())])]
            self.RADIUS_IDS = [str(RADIUS_ID).zfill(len(list(self.data.keys())[0].split("radius_")[-1].split("__")[0]))
                               for RADIUS_ID in RADIUS_IDS or np.unique(
                    [int(i.split("radius_")[-1].split("__")[0]) for i in list(self.data.keys())])]
        else:
            assert MOVEMENT_IDS is None
            assert RADIUS_IDS is None

        #         if REPEATED_MOVEMENTS is not None:
        #             self.REPEATED_MOVEMENTS = REPEATED_MOVEMENTS  #can be overwritten here (useful, if one ones to include several log files as individual episodes in self.data (e.g., when computing summary statistics))

        # self.AGGREGATE_TRIALS = AGGREGATE_TRIALS or (EPISODE_ID == "VARIABLE")

        # reset to data/data_action resulting from __init__() [necessary to call preprocess() multiple times in a row]:
        self.data = self.data_copy.copy()
        self.data_action = self.data_action_copy.copy()

        EPISODE_ID_NUMS = len(self.EPISODE_IDS)  # if EPISODE_IDS is not None else self.EPISODE_ID_NUMS

        #         if self.AGGREGATE_TRIALS:
        #             if EPISODE_ID == "VARIABLE":  #self.REPEATED_MOVEMENTS:
        #                 EPISODE_ID_LIST = [f"{EPISODE_ID_CURRENT}".zfill(len(list(self.data.keys())[0].split("episode_")[-1])) for EPISODE_ID_CURRENT in range(EPISODE_ID_NUMS)]
        #                 assert self."radius" not in AGGREGATION_VARS
        #             else:
        #                 EPISODE_ID_LIST = [EPISODE_ID]

        #         if not self.AGGREGATE_TRIALS:
        #             if EPISODE_ID.isdigit():
        #                 self.EPISODE_ID = EPISODE_ID.zfill(len(list(self.data.keys())[0].split("episode_")[-1]))  #.zfill(3 + (("100episodes" not in filepath) and ("state_log" not in filepath) and ("TwoLevel" not in filepath)) - ("TwoLevel" in filepath))
        #                 data_key = f"episode_{self.EPISODE_ID}"
        #             else:
        #                 data_key = EPISODE_ID
        #             self.data_key = data_key
        #         else:  #i.e., EPISODE_ID == "VARIABLE"
        #             #assert self.REPEATED_MOVEMENTS, "ERROR: Check code dependencies..."

        #             if "radius" not in AGGREGATION_VARS and self.REPEATED_MOVEMENTS:
        #                 self.EPISODE_ID = EPISODE_ID

        #                 self.data_copy = self.data.copy()
        #                 self.data_action_copy = self.data_action.copy()

        #                 for RADIUS_ID_CURRENT in range(self.RADIUS_ID_NUMS):
        #                     self.data[f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"] = {}
        #                     self.data_action[f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"] = {}
        #                     for EPISODE_ID_CURRENT in EPISODE_ID_LIST:
        #                         data_key = f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{EPISODE_ID_CURRENT}"
        #                         for k, v in self.data[data_key].items():
        #                             if k != f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}":
        #                                 self.data[f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"][k] = v if k not in self.data[f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"] else self.data[f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"][k] + v
        #                         for k, v in self.data_action[data_key].items():
        #                             if k != f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}":
        #                                 self.data_action[f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"][k] = v if k not in self.data_action[f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"] else self.data_action[f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"][k] + v
        #             else:
        #                 self.EPISODE_ID = EPISODE_ID

        #                 assert not self.REPEATED_MOVEMENTS, "In this case, data keys should consist of episode IDs only."
        #                 assert self.EPISODE_ID == "VARIABLE"

        #                 self.data[f"episode_{self.EPISODE_ID}"] = {}
        #                 self.data_action[f"episode_{self.EPISODE_ID}"] = {}
        #                 for EPISODE_ID_CURRENT in EPISODE_ID_LIST:
        #                     data_key = f"episode_{EPISODE_ID_CURRENT}"
        #                     for k, v in self.data[data_key].items():
        #                         if k != f"episode_{self.EPISODE_ID}":
        #                             self.data[f"episode_{self.EPISODE_ID}"][k] = v if k not in self.data[f"episode_{self.EPISODE_ID}"] else self.data[f"episode_{self.EPISODE_ID}"][k] + v
        #                     for k, v in self.data_action[data_key].items():
        #                         if k != f"episode_{self.EPISODE_ID}":
        #                             self.data_action[f"episode_{self.EPISODE_ID}"][k] = v if k not in self.data_action[f"episode_{self.EPISODE_ID}"] else self.data_action[f"episode_{self.EPISODE_ID}"][k] + v

        #                 data_key = f"episode_{self.EPISODE_ID}"

        self._position_series = []
        self._velocity_series = []
        self._acceleration_series = []
        self._qpos_series = []
        self._qvel_series = []
        self._qacc_series = []
        self._target_position_series = []
        self._target_radius_series = []
        self._target_idx_series = []
        self._time_series = []
        self._time_per_step = []
        self._indices = []

        self._distance_to_target_series = []

        self._action_series = []
        self._control_series = []
        self._reward_series = []

        if self.REPEATED_MOVEMENTS:
            # -> Data keys consist of movement, radius, and episode index
            self._movement_idx_trials = []
            self._radius_idx_trials = []
            self._episode_idx_trials = []
            self._target_idx_trials = []

            total_steps = 0

            for MOVEMENT_ID_CURRENT in self.MOVEMENT_IDS:
                for RADIUS_ID_CURRENT in self.RADIUS_IDS:
                    for EPISODE_ID_CURRENT in self.EPISODE_IDS:
                        data_key = f"movement_{MOVEMENT_ID_CURRENT}__radius_{RADIUS_ID_CURRENT}__episode_{EPISODE_ID_CURRENT}"
                        self._movement_idx_trials.append(int(MOVEMENT_ID_CURRENT))
                        self._radius_idx_trials.append(int(RADIUS_ID_CURRENT))
                        self._episode_idx_trials.append(int(EPISODE_ID_CURRENT))

                        self._position_series.append(np.squeeze(self.data[data_key][f"{endeffector_name}_xpos"]))
                        self._velocity_series.append(np.squeeze(self.data[data_key][f"{endeffector_name}_xvelp"]))
                        self._qpos_series.append(np.squeeze(
                            self.data[data_key]["qpos"] if "qpos" in self.data[data_key] else np.zeros(
                                (len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan))
                        self._qvel_series.append(np.squeeze(
                            self.data[data_key]["qvel"] if "qvel" in self.data[data_key] else np.zeros(
                                (len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan))
                        self._qacc_series.append(np.squeeze(
                            self.data[data_key]["qacc"] if "qacc" in self.data[data_key] else np.zeros(
                                (len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan))
                        if target_name == "target":
                            self._target_position_series.append(np.squeeze(self.data[data_key][f"target_position"]))
                        else:
                            self._target_position_series.append(np.squeeze(self.data[data_key][f"{target_name}_xpos"]))
                        self._target_radius_series.append(np.squeeze(
                            self.data[data_key][f"{target_name}_radius"] if f"{target_name}_radius" in self.data[
                                data_key] else np.ones(
                                shape=(len(self.data[data_key][f"{target_name}_xpos"]),)) * np.nan))
                        self._target_idx_series.append(np.squeeze(
                            self.data[data_key][f"{target_name}_idx"] if f"{target_name}_idx" in self.data[
                                data_key] else np.array(self._target_radius_series[-1]) * np.nan))
                        self._time_series.append(np.squeeze(self.data[data_key]["timestep"]))

                        self._distance_to_target_series.append(
                            np.linalg.norm(self._target_position_series[-1] - self._position_series[-1], axis=1))

                        self._action_series.append(np.squeeze(self.data_action[data_key]["action"]))
                        self._control_series.append(np.squeeze(self.data_action[data_key]["ctrl"]))
                        self._reward_series.append(np.squeeze(self.data_action[data_key]["reward"]))
                        if len(self._action_series[-1]) == len(self._position_series[
                                                                   -1]) - 1:  # duplicate last action if this allows to match length of position and action series
                            self._action_series[-1] = np.vstack((self._action_series[-1], self._action_series[-1][-1]))
                        if len(self._control_series[-1]) == len(self._position_series[
                                                                    -1]) - 1:  # duplicate last control if this allows to match length of position and control series
                            self._control_series[-1] = np.vstack(
                                (self._control_series[-1], self._control_series[-1][-1]))
                        if len(self._reward_series[-1]) == len(self._position_series[
                                                                   -1]) - 1:  # add 0 to reward series if this allows to match length of position and reward series
                            self._reward_series[-1] = np.hstack((self._reward_series[-1], [0]))  # 1D

                        self._time_per_step.append(np.diff(self._time_series[
                                                               -1]).mean())  # 0.01 if self.REPEATED_MOVEMENTS else np.diff(self._time_series[-1]).mean())  #0.01

                        if split_trials:
                            # current_indices = np.append(np.insert(np.where(self.data[data_key]["target_hit"])[0], 0, 0), len(self.data[data_key]["target_hit"]))
                            try:
                                current_indices = np.insert(np.where(self.data[data_key][f"{target_name}_spawned"])[0],
                                                            0, 0)
                            except KeyError:
                                current_indices = []
                            if len(current_indices) <= 2:  # recompute self._indices based on switches in target position
                                current_indices = \
                                np.where(np.diff(np.squeeze(self._target_position_series[-1]), axis=0).sum(axis=1))[
                                    0] + 1
                            assert len(
                                current_indices) > 2, f"Current_indices ({current_indices}) should have more than 2 entries."
                        else:
                            current_indices = np.array([0, len(self._position_series[-1]) - 1])
                        self._target_idx_trials.append(np.squeeze(self.data[data_key][f"{target_name}_idx"])[
                                                           current_indices] if f"{target_name}_idx" in self.data[
                            data_key] else [-1] * len(current_indices))
                        current_indices += total_steps  # transform to "global" indices of concatenated arrays
                        self._indices.append(current_indices)

                        ## WARNING: deprecated (e.g., use current_indices instead of self._indices...)!
                        # # recover end-effector position time series [only for corrupted pickle file]
                        # if self.filepath in ["log.pickle", "state_log.pickle"] or self.filepath.endswith("log_one_policy_100episodes_100Hz.pickle"):
                        #     current_position_series = (np.cumsum(self._velocity_series[-1], axis=0) * self._time_per_step[-1])
                        #     current_position_series = pd.DataFrame(current_position_series).apply(lambda x: savgol_filter(x, 15, 3, deriv=0, delta = self._time_per_step[-1])).values
                        #     current_position_series += self._target_position_series[-1][self._indices[1] - 1] - current_position_series[self._indices[1] - 1] #ensure that target is reached at target hit/switch time
                        #     self._position_series[-1] = current_position_series

                        # self._velocity_series.append(pd.DataFrame(self._position_series[-1]).apply(lambda x: savgol_filter(x, 15, 3, deriv=1, delta = self._time_per_step[-1])).values)
                        self._acceleration_series.append(pd.DataFrame(self._position_series[-1]).apply(
                            lambda x: savgol_filter(x, 15, 3, deriv=2, delta=self._time_per_step[-1])).values)

                        total_steps += len(self.data[data_key]["timestep"])

            self._movement_idx_trials = np.hstack(self._movement_idx_trials)  # 1D
            self._radius_idx_trials = np.hstack(self._radius_idx_trials)  # 1D
            self._episode_idx_trials = np.hstack(self._episode_idx_trials)  # 1D

        else:
            # -> Data keys only consist of episode index
            self._episode_idx_trials = []
            self._target_idx_trials = []

            total_steps = 0

            for EPISODE_ID_CURRENT in self.EPISODE_IDS:
                data_key = f"episode_{EPISODE_ID_CURRENT}"
                self._episode_idx_trials.append(int(EPISODE_ID_CURRENT))

                self._position_series.append(np.squeeze(self.data[data_key][f"{endeffector_name}_xpos"]))
                self._velocity_series.append(np.squeeze(self.data[data_key][f"{endeffector_name}_xvelp"]))
                self._qpos_series.append(np.squeeze(
                    self.data[data_key]["qpos"] if "qpos" in self.data[data_key] else np.zeros(
                        (len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan))
                self._qvel_series.append(np.squeeze(
                    self.data[data_key]["qvel"] if "qvel" in self.data[data_key] else np.zeros(
                        (len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan))
                self._qacc_series.append(np.squeeze(
                    self.data[data_key]["qacc"] if "qacc" in self.data[data_key] else np.zeros(
                        (len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan))
                if target_name == "target":
                    self._target_position_series.append(np.squeeze(self.data[data_key][f"target_position"]))
                else:
                    self._target_position_series.append(np.squeeze(self.data[data_key][f"{target_name}_xpos"]))
                self._target_radius_series.append(np.squeeze(
                    self.data[data_key][f"{target_name}_radius"] if f"{target_name}_radius" in self.data[
                        data_key] else np.ones(shape=(len(self.data[data_key][f"{target_name}_xpos"]),)) * np.nan))
                self._target_idx_series.append(np.squeeze(
                    self.data[data_key][f"{target_name}_idx"] if f"{target_name}_idx" in self.data[
                        data_key] else np.array(self._target_radius_series[-1]) * np.nan))
                self._time_series.append(np.squeeze(self.data[data_key]["timestep"]))

                self._distance_to_target_series.append(
                    np.linalg.norm(self._target_position_series[-1] - self._position_series[-1], axis=1))

                self._action_series.append(np.squeeze(self.data_action[data_key]["action"]))
                self._control_series.append(np.squeeze(self.data_action[data_key]["ctrl"]))
                self._reward_series.append(np.squeeze(self.data_action[data_key]["reward"]))
                if len(self._action_series[-1]) == len(self._position_series[
                                                           -1]) - 1:  # duplicate last action if this allows to match length of position and action series
                    self._action_series[-1] = np.vstack((self._action_series[-1], self._action_series[-1][-1]))
                if len(self._control_series[-1]) == len(self._position_series[
                                                            -1]) - 1:  # duplicate last control if this allows to match length of position and control series
                    self._control_series[-1] = np.vstack((self._control_series[-1], self._control_series[-1][-1]))
                if len(self._reward_series[-1]) == len(self._position_series[
                                                           -1]) - 1:  # add 0 to reward series if this allows to match length of position and reward series
                    self._reward_series[-1] = np.hstack((self._reward_series[-1], [0]))  # 1D

                self._time_per_step.append(np.diff(self._time_series[-1]).mean())  # 0.01
                if split_trials:
                    # current_indices = np.append(np.insert(np.where(self.data[data_key]["target_hit"])[0], 0, 0), len(self.data[data_key]["target_hit"]))
                    try:
                        current_indices = np.insert(np.where(self.data[data_key][f"{target_name}_spawned"])[0], 0, 0)
                    except KeyError:
                        current_indices = []
                    if len(current_indices) <= 2:  # recompute self._indices based on switches in target position
                        current_indices = \
                        np.where(np.diff(np.squeeze(self._target_position_series[-1]), axis=0).sum(axis=1))[0] + 1
                    assert len(
                        current_indices) > 2, f"Current_indices ({current_indices}) should have more than 2 entries."
                else:
                    current_indices = np.array([0, len(self._position_series[-1]) - 1])
                self._target_idx_trials.append(
                    np.squeeze(self.data[data_key][f"{target_name}_idx"])[current_indices] if f"{target_name}_idx" in
                                                                                              self.data[data_key] else [
                                                                                                                           -1] * len(
                        current_indices))
                current_indices += total_steps  # transform to "global" indices of concatenated arrays
                self._indices.append(current_indices)

                ## WARNING: deprecated (e.g., use current_indices instead of self._indices...)!
                # # recover end-effector position time series [only for corrupted pickle file]
                # if self.filepath in ["log.pickle", "state_log.pickle"] or self.filepath.endswith("log_one_policy_100episodes_100Hz.pickle"):
                #     current_position_series = (np.cumsum(self._velocity_series[-1], axis=0) * self._time_per_step[-1])
                #     current_position_series = pd.DataFrame(current_position_series).apply(lambda x: savgol_filter(x, 15, 3, deriv=0, delta = self._time_per_step[-1])).values
                #     current_position_series += self._target_position_series[-1][self._indices[1] - 1] - current_position_series[self._indices[1] - 1] #ensure that target is reached at target hit/switch time
                #     self._position_series[-1] = current_position_series

                # self._velocity_series.append(pd.DataFrame(self._position_series[-1]).apply(lambda x: savgol_filter(x, 15, 3, deriv=1, delta = self._time_per_step[-1])).values)
                self._acceleration_series.append(pd.DataFrame(self._position_series[-1]).apply(
                    lambda x: savgol_filter(x, 15, 3, deriv=2, delta=self._time_per_step[-1])).values)

                total_steps += len(self.data[data_key]["timestep"])

            self._episode_idx_trials = np.hstack(self._episode_idx_trials)  # 1D

        assert len(self._position_series) == len(self._acceleration_series)
        assert len(self._position_series) == len(self._velocity_series)
        assert len(self._position_series) == len(self._target_position_series)
        assert len(self._position_series) == len(self._target_radius_series)
        assert len(self._position_series) == len(self._target_idx_series)
        assert len(self._position_series) == len(self._distance_to_target_series)

        assert len(self._position_series) == len(self._qpos_series)
        assert len(self._position_series) == len(self._qvel_series)
        assert len(self._position_series) == len(self._qacc_series)

        assert len(self._position_series) == len(self._action_series)
        assert len(self._position_series) == len(self._control_series)
        assert len(self._position_series) == len(self._reward_series)

        ## reset data_key for storing to correct file:
        # RADIUS_ID_CURRENT = "ALL"
        # data_key = f"movement_{self.MOVEMENT_ID}__radius_{RADIUS_ID_CURRENT}__episode_{self.EPISODE_ID}"
        # self.data_key = data_key

        self._position_series = np.vstack(self._position_series)
        self._velocity_series = np.vstack(self._velocity_series)
        self._acceleration_series = np.vstack(self._acceleration_series)
        self._qpos_series = np.vstack(self._qpos_series)
        self._qvel_series = np.vstack(self._qvel_series)
        self._qacc_series = np.vstack(self._qacc_series)
        self._target_position_series = np.vstack(self._target_position_series)
        self._target_radius_series = np.hstack(self._target_radius_series)  # 1D
        self._target_idx_series = np.hstack(self._target_idx_series)  # 1D
        self._time_series = np.hstack(self._time_series)  # 1D
        self._time_per_step = np.mean(self._time_per_step)  # scalar

        self._distance_to_target_series = np.hstack(self._distance_to_target_series)  # 1D

        self._indices = np.vstack(self._indices)
        self._target_idx_trials = np.vstack(self._target_idx_trials)

        self._target_idx_trials_copy = self._target_idx_trials.copy()  # self._target_idx_trials might be overwritten in compute_indices()
        self._episode_idx_trials_copy = self._episode_idx_trials.copy()  # self._episode_idx_trials might be overwritten in compute_indices()
        if self.REPEATED_MOVEMENTS:
            self._movement_idx_trials_copy = self._movement_idx_trials.copy()  # self._movement_idx_trials might be overwritten in compute_indices()
            self._radius_idx_trials_copy = self._radius_idx_trials.copy()  # self._radius_idx_trials might be overwritten in compute_indices()

        self._action_series = np.vstack(self._action_series)
        self._control_series = np.vstack(self._control_series)
        self._reward_series = np.hstack(self._reward_series)  # 1D

        assert len(self._position_series) == total_steps
        assert len(self._episode_idx_trials) == len(self._indices)
        assert self._target_idx_trials.shape == self._indices.shape

        print(f"{self._indices.shape[0]} movement sequences identified.")
        print(f"{self._indices.shape[0] * (self._indices.shape[1] - 1)} trials identified.")

        #         else:

        #             self._position_series = self.data[data_key]["fingertip_xpos"]
        #             self._velocity_series = self.data[data_key]["fingertip_xvelp"]
        #             self._qpos_series = np.squeeze(self.data[data_key]["qpos"] if "qpos" in self.data[data_key] else np.zeros((len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan)
        #             self._qvel_series = np.squeeze(self.data[data_key]["qvel"] if "qvel" in self.data[data_key] else np.zeros((len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan)
        #             self._qacc_series = np.squeeze(self.data[data_key]["qacc"] if "qacc" in self.data[data_key] else np.zeros((len(self.data[data_key]["timestep"]), len(self.independent_joints))) * np.nan)
        #             self._target_position_series = self.data[data_key]["target_position"]
        #             self._target_radius_series = self.data[data_key]["target_radius"]
        #             self._target_idx_series = self.data[data_key]["target_idx"] if "target_idx" in self.data[data_key] else np.array(self.data[data_key]["target_radius"]) * np.nan
        #             self._time_series = self.data[data_key]["timestep"]

        #             self._action_series = np.squeeze(self.data_action[data_key]["action"])
        #             self._control_series = np.squeeze(self.data_action[data_key]["ctrl"])
        #             self._reward_series = np.squeeze(self.data_action[data_key]["reward"])

        #             self._time_per_step = np.diff(self._time_series).mean()  #0.01
        #             #self._indices = np.append(np.insert(np.where(self.data[data_key]["target_hit"])[0], 0, 0), len(self.data[data_key]["target_hit"]))
        #             self._indices = np.insert(np.where(self.data[data_key]["target_spawned"])[0], 0, 0)
        #             if len(self._indices) <= 2:  #recompute self._indices based on switches in target position
        #                 self._indices = np.where(np.diff(np.squeeze(self._target_position_series), axis=0).sum(axis=1))[0] + 1
        #             assert len(self._indices) > 2

        #             # Ensure that first trial corresponds to movement towards target 1, as it is for TrajectoryData_STUDY below
        #             if not (np.isnan(trajectories_SIMULATION.target_idx_series).all() or trajectories_SIMULATION.target_idx_series[self._indices[0]] == 1):
        #                 self._indices = self._indices[1:]
        #             assert (np.isnan(trajectories_SIMULATION.target_idx_series).all() or trajectories_SIMULATION.target_idx_series[self._indices[0]] == 1), "Cannot align indices of simulation and study data."

        #             # recover end-effector position time series [only for corrupted pickle file]
        #             if self.filepath in ["log.pickle", "state_log.pickle"] or self.filepath.endswith("log_one_policy_100episodes_100Hz.pickle"):
        #                 self._position_series = (np.cumsum(self._velocity_series, axis=0) * self._time_per_step)
        #                 self._position_series = pd.DataFrame(self._position_series).apply(lambda x: savgol_filter(x, 15, 3, deriv=0, delta = self._time_per_step)).values
        #                 self._position_series += self._target_position_series[self._indices[1] - 1] - self._position_series[self._indices[1] - 1] #ensure that target is reached at target hit/switch time

        #             #self._velocity_series = pd.DataFrame(self._position_series).apply(lambda x: savgol_filter(x, 15, 3, deriv=1, delta = self._time_per_step)).values
        #             self._acceleration_series = pd.DataFrame(self._position_series).apply(lambda x: savgol_filter(x, 15, 3, deriv=2, delta = self._time_per_step)).values

        self.preprocessed = True

    def combine_indices(self, indices_list, n_samples_list):
        assert len(indices_list) == len(
            n_samples_list), "Number of indices lists/arrays to be combined does not match given number of samples per TrajectoryData instance."

        if not isinstance(self, TrajectoryData_MultipleInstances):
            logging.warning(
                "WARNING: This function should only be called from a 'TrajectoryData_MultipleInstances' instance!")

        self._indices = np.concatenate(
            [indices + sum(n_samples_list[:meta_idx]) for (meta_idx, indices) in enumerate(indices_list)]).astype(int)

        return self._indices

    def compute_indices(self, TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None, N_MOVS=None, AGGREGATION_VARS=[]):
        # INFO: AGGREGATION_VARS includes variables, which are neglected when aggregating different trials to some "movement distribution" (e.g., "episode" aggregates trials independent of the episode, "targetoccurrence" aggregates trials with the same(!) target ID within an episode (more accurately, within the respective row of self._indices))

        if self.REPEATED_MOVEMENTS:
            assert set(AGGREGATION_VARS).issubset({"episode", "radius", "movement", "target",
                                                   "targetoccurrence"}), f'Invalid member(s) of AGGREGATION_VARS: {[i for i in AGGREGATION_VARS if i not in {"episode", "radius", "movement", "target", "targetoccurrence"}]}.'
            if TRIAL_IDS is None:
                TRIAL_IDS = [1]  # use second movement only be default
        else:
            assert set(AGGREGATION_VARS).issubset({"episode", "target",
                                                   "targetoccurrence"}), f'Invalid member(s) of AGGREGATION_VARS: {[i for i in AGGREGATION_VARS if i not in {"episode", "target", "targetoccurrence"}]}.'
        self.AGGREGATION_VARS = AGGREGATION_VARS

        self.TARGET_IDS = TARGET_IDS  # list of target indices or None; if None: allow for all targets

        self.TRIAL_IDS = TRIAL_IDS  # list of (meta) indices, None, or "different_target_sizes"; if None or "different_target_sizes": use META_IDS
        self.META_IDS = META_IDS  # list of (meta) indices or None; if None: use N_MOVS
        self.N_MOVS = N_MOVS  # only used if TRIAL_IDS is None or "different_target_sizes"; if None: use all trials

        # INFO: TRIAL_IDS and META_IDS can be used interchangeably here (unless for "different_target_sizes", which is only a valid param value for TRIAL_IDS)
        # INFO: In order to select specific episode, movement, or radius IDs, call preprocess() with respective arguments.

        # Reset trial indices to default
        self._target_idx_trials = self._target_idx_trials_copy.copy()
        self._episode_idx_trials = self._episode_idx_trials_copy.copy()
        if self.REPEATED_MOVEMENTS:
            self._movement_idx_trials = self._movement_idx_trials_copy.copy()
            self._radius_idx_trials = self._radius_idx_trials_copy.copy()

        #         if self.AGGREGATE_TRIALS:
        #             self.selected_movements_indices = [list(zip(np.concatenate(([-1], indices_radius)), indices_radius, indices_radius[1:]))[1::2] for indices_radius in self._indices]
        #             assert len(self.selected_movements_indices) == self.RADIUS_ID_NUMS if self."radius" not in AGGREGATION_VARS else 1

        #             # concatenate last_idx, current_idx, and next_idx for all trials:
        #             self.selected_movements_indices = [tuple([self.selected_movements_indices_radius_trial[j] for self.selected_movements_indices_radius_trial in self.selected_movements_indices_radius] for j in range(3)) for self.selected_movements_indices_radius in self.selected_movements_indices]

        # Create tuples of (last_idx, current_idx, and next_idx) for all trials)
        self.selected_movements_indices = np.squeeze(
            np.vstack([list(zip(np.concatenate(([-1], idx)), idx, idx[1:])) for idx in self._indices]))

        # if self.REPEATED_MOVEMENTS:
        #    assert len(self.selected_movements_indices) == len(self.MOVEMENT_IDS) * len(self.RADIUS_IDS) * len(self.EPISODE_IDS)
        # self.selected_movements_indices = np.squeeze(np.vstack([list(zip(np.concatenate(([-1], idx)), idx, idx[1:]))[1::2] for idx in self._indices]))
        # self._target_idx_trials = self._target_idx_trials[:, 1::2]
        #         elif self.TRIAL_IDS == "different_target_sizes":
        #             self.selected_movements_indices = list(zip(np.concatenate(([-1], self._indices)), self._indices, self._indices[1:]))

        #             episode_target_radii = np.sort(np.unique(self._target_radius_series[:-1]))
        #             assert len(episode_target_radii) > 1, "ERROR: Could not find different target sizes in used data set. Set TRIAL_IDS to a list of indices or None."
        #             target_radii = episode_target_radii[np.round(np.linspace(0, len(episode_target_radii) - 1, min(self.N_MOVS, len(episode_target_radii)))).astype(int)]

        #             # for each of the self.N_MOVS values in target_radii, choose first movement with this target radius:
        #             self.selected_movements_indices_target_sizes = [(i, target_radius) for target_radius in target_radii for i in [j for j in self.selected_movements_indices if (self.target_radius_series[j[1]] == target_radius)][:1]]
        #             self.selected_movements_indices, self.target_sizes = list(map((lambda x: x[0]), self.selected_movements_indices_target_sizes)), list(map((lambda x: x[1]), self.selected_movements_indices_target_sizes))
        #         else:
        #             #self.selected_movements_indices = list(zip(np.concatenate(([-1], self._indices)), self._indices, self._indices[1:]))
        #             self.selected_movements_indices = np.squeeze(np.vstack([list(zip(np.concatenate(([-1], idx)), idx, idx[1:])) for idx in self._indices]))

        # assert self._target_idx_trials.shape == np.array(self.selected_movements_indices).shape

        #         if self.TARGET_IDS is not None:
        #             mask_selected_target_idx = np.isin(self._target_idx_trials, self.TARGET_IDS)
        #             assert self._target_idx_trials.shape == self._indices.shape

        #             self._target_idx_trials = [[self._target_idx_trials[i, j] for j in range(len(self._target_idx_trials[i])) if mask_selected_target_idx[i,j]] for i in range(len(self._target_idx_trials))]
        #             self._indices = [[self._indices[i, j] for j in range(len(self._indices[i])) if mask_selected_target_idx[i,j]] for i in range(len(self._indices))]

        # Flatten trials attributes (to match shape of self.selected_movements_indices, which consists of individual "trials" instead of trial sequences (as self._indices)):
        assert self._target_idx_trials.ndim in (1, 2)
        if self._target_idx_trials.ndim == 2:
            self._episode_idx_trials = self._episode_idx_trials.repeat([len(x) - 1 for x in self._target_idx_trials])
            if self.REPEATED_MOVEMENTS:
                self._movement_idx_trials = self._movement_idx_trials.repeat(
                    [len(x) - 1 for x in self._target_idx_trials])
                self._radius_idx_trials = self._radius_idx_trials.repeat([len(x) - 1 for x in self._target_idx_trials])
            self._target_idx_trials_meta_indices = np.hstack(
                [list(range(len(i))) for i in self._target_idx_trials[:, :-1]])  # trial ID relative to row
            self._targetoccurrence_idx_trials = [len(np.where(x[:i] == y)[0]) for x in
                                                 self._target_idx_trials_copy[:, :-1] for i, y in
                                                 enumerate(x)]
            self._target_idx_trials = np.hstack(self._target_idx_trials[:,
                                                :-1])  # remove last target index in every row, since this is not used as "current_idx" (center element) in selected_movements_indices computed above
        else:
            self._target_idx_trials_meta_indices = np.zeros(len(self._target_idx_trials)).astype(
                np.int64)  # np.arange(len(self._target_idx_trials)) #trial ID relative to row
            self._targetoccurrence_idx_trials = np.zeros(len(self._target_idx_trials)).astype(np.int64)

        if self.selected_movements_indices.ndim == 1:
            self.selected_movements_indices = self.selected_movements_indices.reshape(1, -1)
        assert len(self.selected_movements_indices) == len(self._target_idx_trials) == len(
            self._episode_idx_trials) == len(self._target_idx_trials_meta_indices)

        # Aggregate selected_movements_indices according to AGGREGATION_VARS, using only trials preselected via TARGET_IDS and TRIALS_IDS/META_IDS/N_MOVS:
        preselection_include_target_ids = lambda \
            target_id: target_id in self.TARGET_IDS if self.TARGET_IDS is not None else True
        preselection_include_trials = lambda k: (k in self.TRIAL_IDS) if self.TRIAL_IDS is not None else (
                    k in self.META_IDS) if self.META_IDS is not None else (
                    k in range(self.N_MOVS)) if self.N_MOVS is not None else True
        self.selected_movements_indices = [self.selected_movements_indices[np.where([all([getattr(self,
                                                                                                  f"_{agg_var}_idx_trials")[
                                                                                              k] == i[j] for j, agg_var
                                                                                          in enumerate(
                self.AGGREGATION_VARS)] + [preselection_include_target_ids(self._target_idx_trials[k])] + [
                                                                                             preselection_include_trials(
                                                                                                 self._target_idx_trials_meta_indices[
                                                                                                     k])]) for k in
                                                                                     range(
                                                                                         len(self.selected_movements_indices))])[
            0]].tolist() for i in itertools.product(
            *[np.unique(getattr(self, f"_{agg_var}_idx_trials")) for agg_var in self.AGGREGATION_VARS])]
        # self.selected_movements_indices = [i for i in self.selected_movements_indices if len(i) > 0]
        self.selected_movements_indices_swapped = [
            [selected_movements_indices_row[j] for selected_movements_indices_row in self.selected_movements_indices if
             j < len(selected_movements_indices_row)] for j in range(max([len(i) for i in
                                                                          self.selected_movements_indices]))]  # np.swapaxes(self.selected_movements_indices, 1, 0)
        self.selected_movements_indices = [tuple(
            [selected_movements_indices_selection_trial[j] for selected_movements_indices_selection_trial in
             selected_movements_indices_selection] for j in range(3)) for selected_movements_indices_selection in
                                           self.selected_movements_indices_swapped]

        # if self.TRIAL_IDS is not None:
        #     self.META_IDS = self.TRIAL_IDS
        #     self.selected_movements_indices = [self.selected_movements_indices[i] for i in self.META_IDS]
        # elif self.META_IDS is not None:
        #     self.selected_movements_indices = [self.selected_movements_indices[i] for i in self.META_IDS]
        # elif self.N_MOVS is not None:
        #     self.selected_movements_indices = self.selected_movements_indices[:self.N_MOVS]

        self.trials_defined = True

        return self.selected_movements_indices


### PRE-PROCESS EXPERIMENTALLY OBSERVED USER DATA
class TrajectoryData_STUDY(TrajectoryData):
    STUDY_DIRECTION_NUMS = 13  # number of targets in experimental ISO task

    # trial IDs of movements used to identify the optimal cost weights in MPC (should thus be neglected when comparing study data to MPC data)
    trainingset_indices = {'U1, Standing_ID_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U1, Standing_Cursor_Handtuned_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U1, Standing_Pad_ID_ISO_15_plane': [30, 29, 28, 27, 26],
                           'U1, Standing_Pad_Handtuned_vertical_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U2, Standing_ID_ISO_15_plane': [24, 23, 22, 21, 20],
                           'U2, Standing_Cursor_Handtuned_ISO_15_plane': [46, 45, 44, 43, 42],
                           'U2, Standing_Pad_ID_ISO_15_plane': [30, 29, 28, 27, 26],
                           'U2, Standing_Pad_Handtuned_vertical_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U3, Standing_ID_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U3, Standing_Cursor_Handtuned_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U3, Standing_Pad_ID_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U3, Standing_Pad_Handtuned_vertical_ISO_15_plane': [33, 32, 31, 30, 29],
                           'U4, Standing_ID_ISO_15_plane': [33, 32, 31, 30, 29],
                           'U4, Standing_Cursor_Handtuned_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U4, Standing_Pad_ID_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U4, Standing_Pad_Handtuned_vertical_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U5, Standing_ID_ISO_15_plane': [30, 29, 28, 27, 26],
                           'U5, Standing_Cursor_Handtuned_ISO_15_plane': [27, 26, 25, 24, 22],
                           'U5, Standing_Pad_ID_ISO_15_plane': [27, 26, 25, 24, 23],
                           'U5, Standing_Pad_Handtuned_vertical_ISO_15_plane': [26, 25, 24, 23, 20],
                           'U6, Standing_ID_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U6, Standing_Cursor_Handtuned_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U6, Standing_Pad_ID_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U6, Standing_Pad_Handtuned_vertical_ISO_15_plane': [31, 30, 29, 28, 27]}

    def __init__(self, DIRNAME_STUDY, USER_ID="U1", TASK_CONDITION="Standing_ID_ISO_15_plane",
                 FILENAME_STUDY_TARGETPOSITIONS=None,
                 independent_joints=None, actuator_names=None):
        self.USER_ID = USER_ID
        self.TASK_CONDITION = TASK_CONDITION
        
        self.DIRNAME_STUDY = DIRNAME_STUDY
        self.FILENAME_STUDY_TARGETPOSITIONS = FILENAME_STUDY_TARGETPOSITIONS

        if independent_joints is None:
            self.independent_joints = [  # 'eye-rx',
                # 'eye-ry',
                'elv_angle',
                'shoulder_elv',
                'shoulder_rot',
                'elbow_flexion',
                'pro_sup',
                'deviation',
                'flexion'
            ]
        else:
            self.independent_joints = independent_joints

        if actuator_names is None:
            self.actuator_names = ['DELT1',
                                   'DELT2',
                                   'DELT3',
                                   'SUPSP',
                                   'INFSP',
                                   'SUBSC',
                                   'TMIN',
                                   'TMAJ',
                                   'PECM1',
                                   'PECM2',
                                   'PECM3',
                                   'LAT1',
                                   'LAT2',
                                   'LAT3',
                                   'CORB',
                                   'TRIlong',
                                   'TRIlat',
                                   'TRImed',
                                   'ANC',
                                   'SUP',
                                   'BIClong',
                                   'BICshort',
                                   'BRA',
                                   'BRD',
                                   # 'ECRL',
                                   # 'ECRB',
                                   # 'ECU',
                                   # 'FCR',
                                   # 'FCU',
                                   # 'PL',
                                   'PT',
                                   'PQ',
                                   # 'FDSL',
                                   # 'FDSR',
                                   # 'FDSM',
                                   # 'FDSI',
                                   # 'FDPL',
                                   # 'FDPR',
                                   # 'FDPM',
                                   # 'FDPI',
                                   # 'EDCL',
                                   # 'EDCR',
                                   # 'EDCM',
                                   # 'EDCI',
                                   # 'EDM',
                                   # 'EIP',
                                   # 'EPL',
                                   # 'EPB',
                                   # 'FPL',
                                   # 'APL'
                                   ]
        else:
            self.actuator_names = actuator_names

        self.actuator_names_dict = {'DELT1': 'deltoid1_r',
                                    'DELT2': 'deltoid2_r',
                                    'DELT3': 'deltoid3_r',
                                    'SUPSP': 'supraspinatus_r',
                                    'INFSP': 'infraspinatus_r',
                                    'SUBSC': 'subscapularis_r',
                                    'TMIN': 'teres_minor_r',
                                    'TMAJ': 'teres_major_r',
                                    'PECM1': 'pectoralis_major1_r',
                                    'PECM2': 'pectoralis_major2_r',
                                    'PECM3': 'pectoralis_major3_r',
                                    'LAT1': 'latissimus_dorsi1_r',
                                    'LAT2': 'latissimus_dorsi2_r',
                                    'LAT3': 'latissimus_dorsi3_r',
                                    'CORB': 'coracobrachialis_r',
                                    'TRIlong': 'triceps_longhead_r',
                                    'TRIlat': 'triceps_lateralis_r',
                                    'TRImed': 'triceps_medialis_r',
                                    'ANC': 'anconeus_r',
                                    'SUP': 'supinator_brevis_r',
                                    'BIClong': 'biceps_longhead_r',
                                    'BICshort': 'biceps_shorthead_r',
                                    'BRA': 'brachialis_r',
                                    'BRD': 'brachioradialis_r',
                                    'PT': 'pronator_teres_r',
                                    'PQ': 'pron_quad_r'}

        super().__init__()

        try:
            data_markers_STUDY = pd.read_csv(os.path.join(self.DIRNAME_STUDY,
                f"{USER_ID}/P1_filtered/PhaseSpace_{USER_ID}_{TASK_CONDITION}.csv"))
            data_IK_STUDY = pd.read_csv(os.path.join(self.DIRNAME_STUDY,
                f"{USER_ID}/P3_filtered/PhaseSpace_{USER_ID}_{TASK_CONDITION}.csv"))
            data_ID_STUDY = pd.read_csv(os.path.join(self.DIRNAME_STUDY,
                f"{USER_ID}/P4_filtered/PhaseSpace_{USER_ID}_{TASK_CONDITION}.csv"))
        except FileNotFoundError:
            data_markers_STUDY = pd.read_csv(os.path.join(self.DIRNAME_STUDY,
                f"{USER_ID}/P1_filtered/PhaseSpace_{USER_ID}_{TASK_CONDITION}.csv"))
            data_IK_STUDY = pd.read_csv(os.path.join(self.DIRNAME_STUDY,
                f"{USER_ID}/P3_filtered/PhaseSpace_{USER_ID}_{TASK_CONDITION}.csv"))
            data_ID_STUDY = pd.read_csv(os.path.join(self.DIRNAME_STUDY,
                f"{USER_ID}/P4_filtered/PhaseSpace_{USER_ID}_{TASK_CONDITION}.csv"))

        # interpolate data_ID_STUDY at time steps of data_IK_STUDY
        data_ID_STUDY_interpolated = \
        pd.concat((data_IK_STUDY.set_index("time"), data_ID_STUDY.set_index("time")), axis=1).loc[:,
        data_ID_STUDY.set_index("time").columns].sort_index().interpolate(method="index").loc[
            data_IK_STUDY["time"]].reset_index()

        try:
            # data_StaticOptimization_STUDY = pd.read_csv(os.path.join(self.DIRNAME_STUDY, f"{USER_ID}/P5/PhaseSpace_{USER_ID}_{TASK_CONDITION}_Cropped_Free_StaticOptimization_activation.sto"), skiprows=8, delimiter="\t")
            data_StaticOptimization_STUDY = self._control_xml_to_DataFrame(os.path.join(self.DIRNAME_STUDY,
                f"{USER_ID}/P5/PhaseSpace_{USER_ID}_{TASK_CONDITION}_Cropped_Free_StaticOptimization_controls.xml"))

            # interpolate data_StaticOptimization_STUDY at time steps of data_IK_STUDY
            data_StaticOptimization_STUDY_interpolated = \
            pd.concat((data_IK_STUDY.set_index("time"), data_StaticOptimization_STUDY.set_index("time")), axis=1).loc[:,
            data_StaticOptimization_STUDY.set_index("time").columns].sort_index().interpolate(method="index").loc[
                data_IK_STUDY["time"]].reset_index()
            # TODO: are StaticOpimization files time-aligned with other files??
            self.static_optimization_loaded = True
        except FileNotFoundError:
            logging.error(f"Cannot load StaticOptimization Data ({self.USER_ID}, {self.TASK_CONDITION}).")
            data_StaticOptimization_STUDY_interpolated = pd.DataFrame()
            self.static_optimization_loaded = False

        # combine end-effector, joint angle/velocity/acceleration, and joint torque data
        self.data = pd.concat(
            (data_markers_STUDY, data_IK_STUDY, data_ID_STUDY_interpolated, data_StaticOptimization_STUDY_interpolated),
            axis=1)
        self.data_action = data_StaticOptimization_STUDY_interpolated

    def preprocess(self):

        # load indices of individual trials (IMPORTANT INFO: depending on the preprocessing, the initial indices might exclude reaction times, i.e., they might correspond to the first time step after the new target occured at which some initial position/velocity/acceleration constraints were satisfied))
        self._indices = np.load(os.path.join(self.DIRNAME_STUDY,
            f'{self.USER_ID}/PhaseSpace_{self.USER_ID}_{self.TASK_CONDITION}_SubMovIndices.npy'),
            allow_pickle=True)

        self._position_series = self.data.loc[:, [f"end_effector_pos_{xyz}" for xyz in ("X", "Y", "Z")]].values
        self._velocity_series = self.data.loc[:, [f"end_effector_vel_{xyz}" for xyz in ("X", "Y", "Z")]].values
        self._acceleration_series = self.data.loc[:, [f"end_effector_acc_{xyz}" for xyz in ("X", "Y", "Z")]].values
        self._qpos_series = self.data.loc[:, [f"{joint_name}_pos" for joint_name in self.independent_joints]].values
        self._qvel_series = self.data.loc[:, [f"{joint_name}_vel" for joint_name in self.independent_joints]].values
        self._qacc_series = self.data.loc[:, [f"{joint_name}_acc" for joint_name in self.independent_joints]].values
        if self.static_optimization_loaded:
            self._act_series = self.data.loc[:, [self.actuator_names_dict[i] for i in self.actuator_names]].values
            self._action_series = self._act_series.copy()
            self._control_series = self._act_series.copy()
        self._target_position_series = self.data.loc[:, [f"target_{xyz}" for xyz in ("x", "y", "z")]].values
        self._target_radius_series = 0.025 * np.ones((self._target_position_series.shape[0],))
        self._target_idx_series = pd.cut(self.data.index, bins=pd.IntervalIndex(
            list(map(lambda x: pd.Interval(*x, closed="left"), self._indices[:, :2])))).map(
            lambda x: self._indices[np.where(self._indices[:, 0] == x.left)[0], 3]).astype("Int64").values
        self._time_series = self.data["time"].values
        # ensures equality of the three time columns resulting from the concatenation above
        assert all(self._time_series[:, 0] == self._time_series[:, 1]) and all(
            self._time_series[:, 0] == self._time_series[:, 2])
        self._time_series = self._time_series[:, 0]
        self._time_per_step = np.mean(np.diff(self._time_series))

        self._distance_to_target_series = np.linalg.norm(self._target_position_series - self._position_series, axis=1)

        # Transform end-effector values to coordinate system used for MuJoCo simulation (x -> front, y -> left, z -> up)
        self._position_series = self._position_series[:, [2, 0, 1]]
        self._velocity_series = self._velocity_series[:, [2, 0, 1]]
        self._acceleration_series = self._acceleration_series[:, [2, 0, 1]]
        self._target_position_series = self._target_position_series[:, [2, 0, 1]]

        ## Store mapping from target ID to target coordinates as fallback solution, if coordinates of (desired) initial position cannot be derived from some other movement towards that initial position
        # experiment_metadata = pd.read_csv(os.path.join(self.DIRNAME_STUDY, f"{USER_ID}/Experiment_{TASK_CONDITION}.csv")
        # self._target_positions_by_ID = {k: experiment_metadata.loc[experiment_metadata["Target.Id"] == k, ["Target.Position.x", "Target Position.y", "Target Position.z"]].iloc[0].values() for k in np.unique(experiment_metadata["Target.Id"])}
        if self.FILENAME_STUDY_TARGETPOSITIONS is not None:
            experiment_goals = pd.read_csv(self.FILENAME_STUDY_TARGETPOSITIONS)
            self._target_positions_by_ID = {(k + 1) % 13: np.array([v[2], -v[0], v[1]]) for k, v in
                                            experiment_goals.iterrows()}

        self._indices_copy = self._indices.copy()  # might be overwritten by self.compute_indices()

        self.preprocessed = True

    def compute_indices(self, TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None, N_MOVS=None, AGGREGATION_VARS=[], ignore_trainingset_trials=False):
        self.TARGET_IDS = TARGET_IDS  # Target ID [second-last column of self.indices]; if None: use TRIAL_IDS
        self.TRIAL_IDS = TRIAL_IDS  # Trial ID [last column of self.indices]; can be combined with self.TARGET_IDS; if None: use META_IDS
        self.META_IDS = META_IDS  # index positions (i.e., sequential numbering of trials in indices, without counting removed outliers); if None: use N_MOVS
        self.N_MOVS = N_MOVS  # only used if TRIAL_IDS and META_IDS are both None; if None: use all trials

        assert set(AGGREGATION_VARS).issubset({"targetoccurrence",
                                               "all"}), f'Invalid member(s) of AGGREGATION_VARS: {[i for i in AGGREGATION_VARS if i not in {"targetoccurrence", "all"}]}.'
        self.AGGREGATION_VARS = AGGREGATION_VARS

        if self.TARGET_IDS is not None:
            assert self.META_IDS is None, "Cannot use both TARGET_IDS and META_IDS. Use TARGET_IDS and TRIAL_IDS instead."
            assert self.N_MOVS is None, "Cannot use both TARGET_IDS and N_MOVS. Use TARGET_IDS and TRIAL_IDS instead."

        self._indices = self._indices_copy.copy()  # reset to indices computed by self.preprocess()

        # group indices of trials with same movement direction (i.e., same target position)
        # WARNING: first group contains movements to target with target_idx 1, last group contains movements to target with target_idx 0!
        # WARNING: last_idx corresponds to first index of a trial with target corresponding to inital (target) position of current trial, although this trial does not have to be executed earlier
        self.trials_to_current_init_pos = np.where(self._indices[1:, 3] == self._indices[0, 2])[0] + 1
        assert len(
            self.trials_to_current_init_pos) > 0, f"Cannot determine target position of target {self._indices[0, 2]}, since no trial to this target was found."
        if "all" in self.AGGREGATION_VARS:
            direction_meta_indices = list(range(len(self._indices)))
            direction_meta_indices_before = [np.where(self._indices[:, 3] == i)[0][0] for i in self._indices[:, 2]]
            self.selected_movements_indices = list(
                zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                    self._indices[direction_meta_indices, 1]))

            self.selected_movements_indices = [tuple(
                [selected_movements_indices_trial[j] for selected_movements_indices_trial in
                 self.selected_movements_indices] for j in range(3))]
        elif "targetoccurrence" in self.AGGREGATION_VARS:
            # TODO: simplify this code (reuse code of TrajectoryData_RL, which allows for arbitrary AGGREGATION_VARS?)
            if self.TARGET_IDS is not None:  # self.TARGET_IDS consists of Target IDs (0, ..., self.STUDY_DIRECTION_NUMS - 1)
                assert set(self.TARGET_IDS).issubset(set(list(range(1, self.STUDY_DIRECTION_NUMS)) + [
                    0])), f"ERROR: Invalid entry in TARGET_IDS (only integers between 0 and {self.STUDY_DIRECTION_NUMS - 1} are allowed)!"
                if self.TRIAL_IDS is not None:
                    self.selected_movements_indices = [list(
                        zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                            self._indices[direction_meta_indices, 1])) for target_idx in self.TARGET_IDS if
                                                       len(direction_meta_indices := np.where(
                                                           (self._indices[:, 3] == target_idx) & np.isin(
                                                               self.indices[:, 4], self.TRIAL_IDS))[0]) > 0 if
                                                       len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                      3] == (
                                                                                                                  target_idx - 1) % self.STUDY_DIRECTION_NUMS) & np.isin(
                                                           self.indices[:, 4], self.TRIAL_IDS))[0]) > 0]
                else:
                    self.selected_movements_indices = [list(
                        zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                            self._indices[direction_meta_indices, 1])) for target_idx in self.TARGET_IDS if
                                                       len(direction_meta_indices :=
                                                           np.where((self._indices[:, 3] == target_idx))[0]) > 0 if
                                                       len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                      3] == (
                                                                                                                  target_idx - 1) % self.STUDY_DIRECTION_NUMS))[
                                                           0]) > 0]
                # pass #this if-condition can be removed, as it was only added to ensure consistency with TrajectoriesData_RL
            elif self.TRIAL_IDS is not None:
                self.selected_movements_indices = [list(
                    zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                        self._indices[direction_meta_indices, 1])) for target_idx in
                                                   list(range(1, self.STUDY_DIRECTION_NUMS)) + [0] if
                                                   len(direction_meta_indices := np.where(
                                                       (self._indices[:, 3] == target_idx) & (
                                                           np.isin(self.indices[:, 4], self.TRIAL_IDS)))[0]) > 0 if
                                                   len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                  3] == (
                                                                                                              target_idx - 1) % self.STUDY_DIRECTION_NUMS))[
                                                       0]) > 0]
            elif self.META_IDS is not None:
                self.selected_movements_indices = [list(
                    zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                        self._indices[direction_meta_indices, 1])) for target_idx in
                                                   list(range(1, self.STUDY_DIRECTION_NUMS)) + [0] if
                                                   len(direction_meta_indices := np.where(
                                                       (self._indices[:, 3] == target_idx) & (
                                                           np.isin(np.arange(len(self._indices)), self.META_IDS)))[
                                                       0]) > 0 if len(direction_meta_indices_before := np.where(
                        (self._indices[:, 3] == (target_idx - 1) % self.STUDY_DIRECTION_NUMS))[0]) > 0]
            elif self.N_MOVS is not None:
                self.selected_movements_indices = [list(
                    zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                        self._indices[direction_meta_indices, 1])) for target_idx in
                                                   list(range(1, self.STUDY_DIRECTION_NUMS)) + [0] if
                                                   len(direction_meta_indices := np.where(
                                                       (self._indices[:, 3] == target_idx) & (
                                                           np.isin(np.arange(len(self._indices)),
                                                                   np.arange(self.N_MOVS))))[0]) > 0 if
                                                   len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                  3] == (
                                                                                                              target_idx - 1) % self.STUDY_DIRECTION_NUMS))[
                                                       0]) > 0]
            else:
                self.selected_movements_indices = [list(
                    zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                        self._indices[direction_meta_indices, 1])) for target_idx in
                                                   list(range(1, self.STUDY_DIRECTION_NUMS)) + [0] if
                                                   len(direction_meta_indices :=
                                                       np.where((self._indices[:, 3] == target_idx))[0]) > 0 if
                                                   len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                  3] == (
                                                                                                              target_idx - 1) % self.STUDY_DIRECTION_NUMS))[
                                                       0]) > 0]
                assert len(self.selected_movements_indices) == self.STUDY_DIRECTION_NUMS

                # concatenate last_idx, current_idx, and next_idx for all selected trials (TARGET_IDS is used afterwards):
            self.selected_movements_indices = [tuple(
                [selected_movements_indices_direction_trial[j] for selected_movements_indices_direction_trial in
                 selected_movements_indices_direction] for j in range(3)) for selected_movements_indices_direction in
                                               self.selected_movements_indices]

        else:
            ##self.selected_movements_indices = list(zisince no mop(np.hstack(([self._indices[self.trials_to_current_init_pos[0], 0]], self._indices[:-1, 0])), self._indices[:, 0], self._indices[:, 1]))
            get_meta_indices_for_target_idx = lambda x: np.where(self._indices[:, 3] == x)[0]
            get_closest_meta_index = lambda index_list, fixed_index: index_list[
                np.argmin([np.abs(x - fixed_index) for x in index_list])] if len(index_list) > 0 else np.nan
            # input(([(get_meta_indices_for_target_idx(idx_row[2]), get_closest_meta_index(get_meta_indices_for_target_idx(idx_row[2]), idx_row[0])) for idx_row in self._indices]))

            # indices_of_trials_to_current_init_position = np.hstack(([self.indices[get_closest_meta_index(get_meta_indices_for_target_idx(idx_row[2]), meta_idx), 0] for meta_idx, idx_row in enumerate(self._indices)]))
            closest_meta_indices = [get_closest_meta_index(get_meta_indices_for_target_idx(idx_row[2]), meta_idx) for
                                    meta_idx, idx_row in enumerate(self._indices)]
            ### VARIANT 1:  #FLAG123
            if self.FILENAME_STUDY_TARGETPOSITIONS is None:
                # Remove trials with initial position never reached (NOTE: if it is ensured, that effective_projection_path==True is used later, these trials do not need to be removed and the first column ("last_idx")) can be set arbitrarily...):
                closest_meta_indices_processed = [i for i in closest_meta_indices if not np.isnan(i)]
                if len(closest_meta_indices_processed) < len(closest_meta_indices):
                    logging.error(f"STUDY ({self.USER_ID}, {self.TASK_CONDITION}) - Removed trials with Trial ID {self.indices[np.where(np.isnan(closest_meta_indices))[0], 4]}, since no movement to initial position (Target(s) {np.unique(self.indices[np.where(np.isnan(closest_meta_indices))[0], 2])}) could be identified.")
                indices_of_trials_to_current_init_position = np.hstack(([self.indices[closest_meta_indices_processed, 0]]))
                assert len(closest_meta_indices) == len(self._indices)
                self._indices = self.indices[np.where(~np.isnan(closest_meta_indices))[0], :]
            ### VARIANT 2 (dirty hack (TODO: this definitely needs code clean up!) - directly store target ID with negative sign (and shifted by -100) instead of frame ID in indices_of_trials_to_current_init_position and thus in first column of self.selected_movements_indices, if no movement to the respective init position exists)
            else:
                closest_meta_indices_processed = closest_meta_indices
                indices_of_trials_to_current_init_position = np.hstack(([
                    self.indices[i, 0] if not np.isnan(i) else -100 - self.indices[idx, 2] for idx, i in enumerate(
                        closest_meta_indices_processed)]))  # closest_meta_indices_processed should have same length as self._indices here by definition
            #####################
            assert len(closest_meta_indices_processed) == len(self._indices)
            self.trials_to_current_init_pos = np.where(self._indices[1:, 3] == self._indices[0, 2])[0] + 1

            self.selected_movements_indices = list(
                zip(indices_of_trials_to_current_init_position, self._indices[:, 0], self._indices[:, 1]))

            if ignore_trainingset_trials:
                trainingset_TRIAL_IDS = self.trainingset_indices[f'{self.USER_ID}, {self.TASK_CONDITION}']
                trainingset_META_IDS = np.where(np.isin(self.indices[:, 4], trainingset_TRIAL_IDS))[0]
                assert len(self.selected_movements_indices) == len(
                    self.indices), f"ERROR: 'indices' and 'selected_movements_indices' do not have the same length!"
                self._indices = np.array([self._indices[i] for i in range(len(self.selected_movements_indices)) if
                                          i not in trainingset_META_IDS])
                self.selected_movements_indices = [self.selected_movements_indices[i] for i in
                                                   range(len(self.selected_movements_indices)) if
                                                   i not in trainingset_META_IDS]
                logging.warning(
                    f"STUDY ({self.USER_ID}, {self.TASK_CONDITION}) - Ignore training set indices {trainingset_TRIAL_IDS}.")

            if self.TARGET_IDS is not None:  # self.TARGET_IDS consists of Target IDs (0, ..., self.STUDY_DIRECTION_NUMS - 1)
                if self.TRIAL_IDS is not None:
                    self.META_IDS = np.where(
                        np.isin(self.indices[:, 3], self.TARGET_IDS) & np.isin(self.indices[:, 4], self.TRIAL_IDS))[0]
                else:
                    self.META_IDS = np.where(np.isin(self.indices[:, 3], self.TARGET_IDS))[0]
                self.selected_movements_indices = [self.selected_movements_indices[i] for i in self.META_IDS]
                # self.target_positions_per_trial = [self.target_positions_per_trial[i] for i in self.META_IDS]
                # self.target_vecs = [self.target_vecs[i] for i in self.META_IDS]
            elif self.TRIAL_IDS is not None:  # self.TRIAL_IDS consists of Trial IDs (0, 1, ...)
                assert len(self.selected_movements_indices) == len(
                    self.indices), f"ERROR: 'indices' and 'selected_movements_indices' do not have the same length!"
                self.META_IDS = np.where(np.isin(self.indices[:, 4], self.TRIAL_IDS))[0]
                self.selected_movements_indices = [self.selected_movements_indices[i] for i in self.META_IDS]
            elif self.META_IDS is not None:
                self.selected_movements_indices = [self.selected_movements_indices[i] for i in self.META_IDS]
            elif self.N_MOVS is not None:
                # Only keep first N_MOVS trials:
                ##self._indices = self._indices[:self.N_MOVS]
                # no need to start to use "1:self.N_MOVS+1" here, since order of trials should have already changed in self._indices and self.selected_movements_indices...
                self.selected_movements_indices = self.selected_movements_indices[:self.N_MOVS]
                # self.target_positions_per_trial = self.target_positions_per_trial[:self.N_MOVS]
                # self.target_vecs = self.target_vecs[:self.N_MOVS]

            if self.TARGET_IDS is None and self.TRIAL_IDS is not None:
                assert set(self.TRIAL_IDS).issubset(
                    set(self.indices[:, 4])), f"ERROR: Invalid Trial ID(s)! Valid Trial IDs:\n{self.indices[:, 4]}"

            # Compute target vector for each trial to ensure that trial data is consistent and correct
            target_position_series = self._target_position_series  # [0] if self.AGGREGATE_TRIALS else self._target_position_series
            if not ignore_trainingset_trials:
                assert len(np.unique(np.round(target_position_series[self._indices[self.trials_to_current_init_pos, 0]],
                                          decimals=12))) == 3, f"ERROR: Target positions do not match for trials with same init/target id ({target_position_series[self._indices[self.trials_to_current_init_pos, 0]]})!"
            # self.selected_movements_indices_per_trial = list(zip(np.hstack(([self._indices[self.trials_to_current_init_pos[0], 0]], self._indices[:-1, 0])), self._indices[:, 0], self._indices[:, 1]))
            ### VARIANT 1:  #FLAG123
            if self.FILENAME_STUDY_TARGETPOSITIONS is None:
                self.target_positions_per_trial = target_position_series[np.squeeze(self.selected_movements_indices)[..., :2], :]
            ### VARIANT 2:
            else:
                self.target_positions_per_trial = np.array([target_position_series[i, :] if i[0]>=0 else [self._target_positions_by_ID[-(i[0]+100)], target_position_series[i[1], :]] for i in np.array(self.selected_movements_indices)[..., :2]])
            #####################
            self.target_vecs = self.target_positions_per_trial[..., 1, :] - self.target_positions_per_trial[..., 0, :]
            # WARNING: self.selected_movements_indices_per_trial is NOT updated to account for TRIAL_IDS/META_IDS/N_MOVS
            # WARNING: self.target_positions_per_trial and target_vecs are ONLY updated to account for TRIAL_IDS/META_IDS/N_MOVS if self.AGGREGATE_TRIALS==False

        self.trials_defined = True

        return self.selected_movements_indices

    def combine_indices(self, indices_list, n_samples_list):
        assert len(indices_list) == len(
            n_samples_list), "Number of indices lists/arrays to be combined does not match given number of samples per TrajectoryData instance."

        if not isinstance(self, TrajectoryData_MultipleInstances):
            logging.warning(
                "WARNING: This function should only be called from a 'TrajectoryData_MultipleInstances' instance!")

        self._indices = np.vstack([indices + (sum(n_samples_list[:meta_idx]) * np.hstack(
            (np.ones((indices.shape[0], 2)), np.zeros((indices.shape[0], 3))))) for (meta_idx, indices) in
                                   enumerate(indices_list)]).astype(int)
        self._indices_copy = self._indices.copy()

        return self._indices

    def _control_xml_to_DataFrame(self, filename):
        with open(filename, 'rb') as f:
            data_StaticOptimization_STUDY_xml = xmltodict.parse(f.read())

        data_StaticOptimization_STUDY_times = {
            cl['@name']: [float(cl_el['t']) for cl_el in cl['x_nodes']['ControlLinearNode']] for cl in
            data_StaticOptimization_STUDY_xml['OpenSimDocument']['ControlSet']['objects']['ControlLinear']}
        data_StaticOptimization_STUDY_values = {
            cl['@name']: [float(cl_el['value']) for cl_el in cl['x_nodes']['ControlLinearNode']] for cl in
            data_StaticOptimization_STUDY_xml['OpenSimDocument']['ControlSet']['objects']['ControlLinear']}
        data_StaticOptimization_STUDY_time = {'time': list(data_StaticOptimization_STUDY_times.values())[0]}
        assert all([data_StaticOptimization_STUDY_time['time'] == data_StaticOptimization_STUDY_times[i] for i in
                    data_StaticOptimization_STUDY_times])

        data_StaticOptimization_STUDY = pd.concat(
            (pd.DataFrame(data_StaticOptimization_STUDY_time), pd.DataFrame(data_StaticOptimization_STUDY_values)),
            axis=1)

        return data_StaticOptimization_STUDY


class TrajectoryData_MPC(TrajectoryData):
    MPC_DIRECTION_NUMS = 13  # number of targets in experimental ISO task

    # trial IDs of movements used to identify the optimal cost weights in MPC (should thus be neglected when comparing study data to MPC data -> set ignore_trainingset_trials=True when calling compute_indices() in BOTH TrajectoryData_MPC and TrajectoryData_STUDY!)
    trainingset_indices = {'U1, Standing_ID_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U1, Standing_Cursor_Handtuned_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U1, Standing_Pad_ID_ISO_15_plane': [30, 29, 28, 27, 26],
                           'U1, Standing_Pad_Handtuned_vertical_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U2, Standing_ID_ISO_15_plane': [24, 23, 22, 21, 20],
                           'U2, Standing_Cursor_Handtuned_ISO_15_plane': [46, 45, 44, 43, 42],
                           'U2, Standing_Pad_ID_ISO_15_plane': [30, 29, 28, 27, 26],
                           'U2, Standing_Pad_Handtuned_vertical_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U3, Standing_ID_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U3, Standing_Cursor_Handtuned_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U3, Standing_Pad_ID_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U3, Standing_Pad_Handtuned_vertical_ISO_15_plane': [33, 32, 31, 30, 29],
                           'U4, Standing_ID_ISO_15_plane': [33, 32, 31, 30, 29],
                           'U4, Standing_Cursor_Handtuned_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U4, Standing_Pad_ID_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U4, Standing_Pad_Handtuned_vertical_ISO_15_plane': [31, 30, 29, 28, 27],
                           'U5, Standing_ID_ISO_15_plane': [30, 29, 28, 27, 26],
                           'U5, Standing_Cursor_Handtuned_ISO_15_plane': [27, 26, 25, 24, 22],
                           'U5, Standing_Pad_ID_ISO_15_plane': [27, 26, 25, 24, 23],
                           'U5, Standing_Pad_Handtuned_vertical_ISO_15_plane': [26, 25, 24, 23, 20],
                           'U6, Standing_ID_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U6, Standing_Cursor_Handtuned_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U6, Standing_Pad_ID_ISO_15_plane': [32, 31, 30, 29, 28],
                           'U6, Standing_Pad_Handtuned_vertical_ISO_15_plane': [31, 30, 29, 28, 27]}

    def __init__(self, DIRNAME_SIMULATION, SIMULATION_SUBDIR, USER_ID="U1", TASK_CONDITION="Standing_ID_ISO_15_plane",
                 FILENAME_STUDY_TARGETPOSITIONS=None,
                 independent_joints=None, actuator_names=None):
        self.USER_ID = USER_ID
        self.TASK_CONDITION = TASK_CONDITION

        self.DIRNAME_SIMULATION = DIRNAME_SIMULATION

        self.FILENAME_STUDY_TARGETPOSITIONS = FILENAME_STUDY_TARGETPOSITIONS
        self.SIMULATION_SUBDIR = SIMULATION_SUBDIR

        if independent_joints is None:
            self.independent_joints = [  # 'eye-rx',
                # 'eye-ry',
                'elv_angle',
                'shoulder_elv',
                'shoulder_rot',
                'elbow_flexion',
                'pro_sup',
                'deviation',
                'flexion'
            ]
        else:
            self.independent_joints = independent_joints

        if actuator_names is None:
            self.actuator_names = self.independent_joints  # INFO: torque-actuated model is used here!
        else:
            self.actuator_names = actuator_names

        self.actuator_names_dict = {'DELT1': 'deltoid1_r',
                                    'DELT2': 'deltoid2_r',
                                    'DELT3': 'deltoid3_r',
                                    'SUPSP': 'supraspinatus_r',
                                    'INFSP': 'infraspinatus_r',
                                    'SUBSC': 'subscapularis_r',
                                    'TMIN': 'teres_minor_r',
                                    'TMAJ': 'teres_major_r',
                                    'PECM1': 'pectoralis_major1_r',
                                    'PECM2': 'pectoralis_major2_r',
                                    'PECM3': 'pectoralis_major3_r',
                                    'LAT1': 'latissimus_dorsi1_r',
                                    'LAT2': 'latissimus_dorsi2_r',
                                    'LAT3': 'latissimus_dorsi3_r',
                                    'CORB': 'coracobrachialis_r',
                                    'TRIlong': 'triceps_longhead_r',
                                    'TRIlat': 'triceps_lateralis_r',
                                    'TRImed': 'triceps_medialis_r',
                                    'ANC': 'anconeus_r',
                                    'SUP': 'supinator_brevis_r',
                                    'BIClong': 'biceps_longhead_r',
                                    'BICshort': 'biceps_shorthead_r',
                                    'BRA': 'brachialis_r',
                                    'BRD': 'brachioradialis_r',
                                    'PT': 'pronator_teres_r',
                                    'PQ': 'pron_quad_r'}

        super().__init__()

        self.data = pd.read_csv(os.path.join(self.DIRNAME_SIMULATION,
            f'{USER_ID}/{SIMULATION_SUBDIR}/{TASK_CONDITION}/complete.csv'))
        # TODO: self.data_action = ?

    def preprocess(self):

        # load indices of individual trials (timesteps at which new target was visible for the first time)
        self._indices = np.load(os.path.join(self.DIRNAME_SIMULATION,
            f'{self.USER_ID}/{self.SIMULATION_SUBDIR}/{self.TASK_CONDITION}/SubMovIndices.npy'),
            allow_pickle=True)

        self._position_series = self.data.loc[:, [f"end-effector_xpos_{xyz}" for xyz in ("x", "y", "z")]].values
        self._velocity_series = self.data.loc[:, [f"end-effector_xvel_{xyz}" for xyz in ("x", "y", "z")]].values
        self._acceleration_series = self.data.loc[:, [f"end-effector_xacc_{xyz}" for xyz in ("x", "y", "z")]].values
        self._qpos_series = self.data.loc[:, [f"{joint_name}_pos" for joint_name in self.independent_joints]].values
        self._qvel_series = self.data.loc[:, [f"{joint_name}_vel" for joint_name in self.independent_joints]].values
        self._qacc_series = self.data.loc[:, [f"{joint_name}_acc" for joint_name in self.independent_joints]].values
        self._act_series = self.data.loc[:, [f"ACT_{joint_name}" for joint_name in self.independent_joints]].values
        self._control_series = self.data.loc[:, [f"A_{joint_name}" for joint_name in self.independent_joints]].values
        self._action_series = self._control_series.copy()
        self._target_position_series = self.data.loc[:, [f"target_{xyz}" for xyz in ("x", "y", "z")]].values
        self._target_radius_series = 0.025 * np.ones((self._target_position_series.shape[0],))
        self._target_idx_series = pd.cut(self.data.index, bins=pd.IntervalIndex(
            list(map(lambda x: pd.Interval(*x, closed="left"), self._indices[:, :2])))).map(
            lambda x: self._indices[np.where(self._indices[:, 0] == x.left)[0], 3]).astype("Int64").values
        self._time_series = self.data["time"].values
        self._time_per_step = np.mean(np.diff(self._time_series)[np.abs(stats.zscore(np.diff(self._time_series))) <= 3])

        self._distance_to_target_series = np.linalg.norm(self._target_position_series - self._position_series, axis=1)

        # Transform end-effector values to coordinate system used for MuJoCo simulation (x -> front, y -> left, z -> up)
        self._position_series = self._position_series[:, [2, 0, 1]]
        self._velocity_series = self._velocity_series[:, [2, 0, 1]]
        self._acceleration_series = self._acceleration_series[:, [2, 0, 1]]
        self._target_position_series = self._target_position_series[:, [2, 0, 1]]

        ## Store mapping from target ID to target coordinates as fallback solution, if coordinates of (desired) initial position cannot be derived from some other movement towards that initial position
        # experiment_metadata = pd.read_csv(os.path.join(self.DIRNAME_SIMULATION, f"{USER_ID}/Experiment_{TASK_CONDITION}.csv"))
        # self._target_positions_by_ID = {k: experiment_metadata.loc[experiment_metadata["Target.Id"] == k, ["Target.Position.x", "Target Position.y", "Target Position.z"]].iloc[0].values() for k in np.unique(experiment_metadata["Target.Id"])}
        if self.FILENAME_STUDY_TARGETPOSITIONS is not None:
            experiment_goals = pd.read_csv(self.FILENAME_STUDY_TARGETPOSITIONS)
            self._target_positions_by_ID = {(k + 1) % 13: np.array([v[2], -v[0], v[1]]) for k, v in
                                            experiment_goals.iterrows()}

        self._indices_copy = self._indices.copy()  # might be overwritten by self.compute_indices()

        self.preprocessed = True

    def compute_indices(self, TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None, N_MOVS=None, AGGREGATION_VARS=[], ignore_trainingset_trials=False):
        self.TARGET_IDS = TARGET_IDS  # Target ID [second-last column of self.indices]; if None: use TRIAL_IDS
        self.TRIAL_IDS = TRIAL_IDS  # Trial ID [last column of self.indices]; if None: use META_IDS
        self.META_IDS = META_IDS  # index positions (i.e., sequential numbering of trials in indices, without counting removed outliers); if None: use N_MOVS
        self.N_MOVS = N_MOVS  # only used if TRIAL_IDS and META_IDS are both None; if None: use all trials

        assert set(AGGREGATION_VARS).issubset({"targetoccurrence",
                                               "all"}), f'Invalid member(s) of AGGREGATION_VARS: {[i for i in AGGREGATION_VARS if i not in {"targetoccurrence", "all"}]}.'
        self.AGGREGATION_VARS = AGGREGATION_VARS

        if self.TARGET_IDS is not None:
            assert self.META_IDS is None, "Cannot use both TARGET_IDS and META_IDS. Use TARGET_IDS and TRIAL_IDS instead."
            assert self.N_MOVS is None, "Cannot use both TARGET_IDS and N_MOVS. Use TARGET_IDS and TRIAL_IDS instead."

        self._indices = self._indices_copy.copy()  # reset to indices computed by self.preprocess()

        # group indices of trials with same movement direction (i.e., same target position)
        # WARNING: first group contains movements to target with target_idx 1, last group contains movements to target with target_idx 0!
        # WARNING: last_idx corresponds to first index of a trial with target corresponding to inital (target) position of current trial, although this trial does not have to be executed earlier
        self.trials_to_current_init_pos = np.where(self._indices[1:, 3] == self._indices[0, 2])[0] + 1
        assert len(
            self.trials_to_current_init_pos) > 0, f"Cannot determine target position of target {self._indices[0, 2]}, since no trial preceding a movement to the current target was found."
        if "all" in self.AGGREGATION_VARS:
            direction_meta_indices = list(range(len(self._indices)))
            direction_meta_indices_before = [np.where(self._indices[:, 3] == i)[0][0] for i in self._indices[:, 2]]
            self.selected_movements_indices = list(
                zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                    self._indices[direction_meta_indices, 1]))

            self.selected_movements_indices = [tuple(
                [selected_movements_indices_trial[j] for selected_movements_indices_trial in
                 self.selected_movements_indices] for j in range(3))]
        elif "targetoccurrence" in self.AGGREGATION_VARS:
            # TODO: simplify this code (reuse code of TrajectoryData_RL, which allows for arbitrary AGGREGATION_VARS?)
            if self.TARGET_IDS is not None:  # self.TARGET_IDS consists of Target IDs (0, ..., self.MPC_DIRECTION_NUMS - 1)
                assert set(self.TARGET_IDS).issubset(set(list(range(1, self.MPC_DIRECTION_NUMS)) + [
                    0])), f"ERROR: Invalid entry in TARGET_IDS (only integers between 0 and {self.MPC_DIRECTION_NUMS - 1} are allowed)!"
                if self.TRIAL_IDS is not None:
                    self.selected_movements_indices = [list(
                        zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                            self._indices[direction_meta_indices, 1])) for target_idx in self.TARGET_IDS if
                                                       len(direction_meta_indices := np.where(
                                                           (self._indices[:, 3] == target_idx) & np.isin(
                                                               self.indices[:, 4], self.TRIAL_IDS))[0]) > 0 if
                                                       len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                      3] == (
                                                                                                                  target_idx - 1) % self.STUDY_DIRECTION_NUMS) & np.isin(
                                                           self.indices[:, 4], self.TRIAL_IDS))[0]) > 0]
                else:
                    self.selected_movements_indices = [list(
                        zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                            self._indices[direction_meta_indices, 1])) for target_idx in self.TARGET_IDS if
                                                       len(direction_meta_indices :=
                                                           np.where((self._indices[:, 3] == target_idx))[0]) > 0 if
                                                       len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                      3] == (
                                                                                                                  target_idx - 1) % self.MPC_DIRECTION_NUMS))[
                                                           0]) > 0]
                # pass #this if-condition can be removed, as it was only added to ensure consistency with TrajectoriesData_RL
            elif self.TRIAL_IDS is not None:
                self.selected_movements_indices = [list(
                    zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                        self._indices[direction_meta_indices, 1])) for target_idx in
                                                   list(range(1, self.MPC_DIRECTION_NUMS)) + [0] if
                                                   len(direction_meta_indices := np.where(
                                                       (self._indices[:, 3] == target_idx) & (
                                                           np.isin(self.indices[:, 4], self.TRIAL_IDS)))[0]) > 0 if
                                                   len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                  3] == (
                                                                                                              target_idx - 1) % self.MPC_DIRECTION_NUMS))[
                                                       0]) > 0]
            elif self.META_IDS is not None:
                self.selected_movements_indices = [list(
                    zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                        self._indices[direction_meta_indices, 1])) for target_idx in
                                                   list(range(1, self.MPC_DIRECTION_NUMS)) + [0] if
                                                   len(direction_meta_indices := np.where(
                                                       (self._indices[:, 3] == target_idx) & (
                                                           np.isin(np.arange(len(self._indices)), self.META_IDS)))[
                                                       0]) > 0 if len(direction_meta_indices_before := np.where(
                        (self._indices[:, 3] == (target_idx - 1) % self.MPC_DIRECTION_NUMS))[0]) > 0]
            elif self.N_MOVS is not None:
                self.selected_movements_indices = [list(
                    zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                        self._indices[direction_meta_indices, 1])) for target_idx in
                                                   list(range(1, self.MPC_DIRECTION_NUMS)) + [0] if
                                                   len(direction_meta_indices := np.where(
                                                       (self._indices[:, 3] == target_idx) & (
                                                           np.isin(np.arange(len(self._indices)),
                                                                   np.arange(self.N_MOVS))))[0]) > 0 if
                                                   len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                  3] == (
                                                                                                              target_idx - 1) % self.MPC_DIRECTION_NUMS))[
                                                       0]) > 0]
            else:
                self.selected_movements_indices = [list(
                    zip(self._indices[direction_meta_indices_before, 0], self._indices[direction_meta_indices, 0],
                        self._indices[direction_meta_indices, 1])) for target_idx in
                                                   list(range(1, self.MPC_DIRECTION_NUMS)) + [0] if
                                                   len(direction_meta_indices :=
                                                       np.where((self._indices[:, 3] == target_idx))[0]) > 0 if
                                                   len(direction_meta_indices_before := np.where((self._indices[:,
                                                                                                  3] == (
                                                                                                              target_idx - 1) % self.MPC_DIRECTION_NUMS))[
                                                       0]) > 0]
                assert len(self.selected_movements_indices) == self.MPC_DIRECTION_NUMS

                # concatenate last_idx, current_idx, and next_idx for all selected trials (TARGET_IDS is used afterwards):
            self.selected_movements_indices = [tuple(
                [selected_movements_indices_direction_trial[j] for selected_movements_indices_direction_trial in
                 selected_movements_indices_direction] for j in range(3)) for selected_movements_indices_direction in
                                               self.selected_movements_indices]

        else:
            ##self.selected_movements_indices = list(zip(np.hstack(([self._indices[self.trials_to_current_init_pos[0], 0]], self._indices[:-1, 0])), self._indices[:, 0], self._indices[:, 1]))
            get_meta_indices_for_target_idx = lambda x: np.where(self._indices[:, 3] == x)[0]
            get_closest_meta_index = lambda index_list, fixed_index: index_list[
                np.argmin([np.abs(x - fixed_index) for x in index_list])] if len(index_list) > 0 else np.nan
            # indices_of_trials_to_current_init_position = np.hstack(([self.indices[get_closest_meta_index(get_meta_indices_for_target_idx(idx_row[2]), meta_idx), 0] for meta_idx, idx_row in enumerate(self._indices)]))
            # input(([(get_meta_indices_for_target_idx(idx_row[2]), get_closest_meta_index(get_meta_indices_for_target_idx(idx_row[2]), idx_row[0])) for idx_row in self._indices]))

            closest_meta_indices = [get_closest_meta_index(get_meta_indices_for_target_idx(idx_row[2]), meta_idx) for
                                    meta_idx, idx_row in enumerate(self._indices)]
            ### VARIANT 1:  #FLAG123
            if self.FILENAME_STUDY_TARGETPOSITIONS is None:
                # Remove trials with initial position never reached (NOTE: if it is ensured, that effective_projection_path==True is used later, these trials do not need to be removed and the first column ("last_idx")) can be set arbitrarily...):
                closest_meta_indices_processed = [i for i in closest_meta_indices if not np.isnan(i)]
                if len(closest_meta_indices_processed) < len(closest_meta_indices):
                    logging.error(f"MPC ({self.USER_ID}, {self.TASK_CONDITION}, {self.SIMULATION_SUBDIR}) - Removed trials with Trial ID {self.indices[np.where(np.isnan(closest_meta_indices))[0], 4]}, since no movement to initial position (Target(s) {np.unique(self.indices[np.where(np.isnan(closest_meta_indices))[0], 2])}) could be identified.")
                indices_of_trials_to_current_init_position = np.hstack(([self.indices[closest_meta_indices_processed, 0]]))
                assert len(closest_meta_indices) == len(self._indices)
                self._indices = self.indices[np.where(~np.isnan(closest_meta_indices))[0], :]
            else:
                ### VARIANT 2 (dirty hack (TODO: this definitely needs code clean up!) - directly store target ID with negative sign (and shifted by -100) instead of frame ID in indices_of_trials_to_current_init_position and thus in first column of self.selected_movements_indices, if no movement to the respective init position exists)
                closest_meta_indices_processed = closest_meta_indices
                indices_of_trials_to_current_init_position = np.hstack(([self.indices[i, 0] if not np.isnan(i) else -100-self.indices[idx, 2] for idx, i in enumerate(closest_meta_indices_processed)]))  #closest_meta_indices_processed should have same length as self._indices here by definition
            #####################
            assert len(closest_meta_indices_processed) == len(self._indices)
            self.trials_to_current_init_pos = np.where(self._indices[1:, 3] == self._indices[0, 2])[0] + 1

            self.selected_movements_indices = list(
                zip(indices_of_trials_to_current_init_position, self._indices[:, 0], self._indices[:, 1]))

            if ignore_trainingset_trials:
                trainingset_TRIAL_IDS = self.trainingset_indices[f'{self.USER_ID}, {self.TASK_CONDITION}']
                trainingset_META_IDS = np.where(np.isin(self.indices[:, 4], trainingset_TRIAL_IDS))[0]
                assert len(self.selected_movements_indices) == len(
                    self.indices), f"ERROR: 'indices' and 'selected_movements_indices' do not have the same length!"
                self._indices = np.array([self._indices[i] for i in range(len(self.selected_movements_indices)) if
                                          i not in trainingset_META_IDS])
                self.selected_movements_indices = [self.selected_movements_indices[i] for i in
                                                   range(len(self.selected_movements_indices)) if
                                                   i not in trainingset_META_IDS]
                logging.warning(
                    f"MPC ({self.USER_ID}, {self.TASK_CONDITION}, {self.SIMULATION_SUBDIR}) - Ignore training set indices {trainingset_TRIAL_IDS}.")

            if self.TARGET_IDS is not None:  # self.TARGET_IDS consists of Target IDs (0, ..., self.MPC_DIRECTION_NUMS - 1)
                if self.TRIAL_IDS is not None:
                    self.META_IDS = np.where(
                        np.isin(self.indices[:, 3], self.TARGET_IDS) & np.isin(self.indices[:, 4], self.TRIAL_IDS))[0]
                else:
                    self.META_IDS = np.where(np.isin(self.indices[:, 3], self.TARGET_IDS))[0]
                self.selected_movements_indices = [self.selected_movements_indices[i] for i in self.META_IDS]
                # self.target_positions_per_trial = [self.target_positions_per_trial[i] for i in self.META_IDS]
                # self.target_vecs = [self.target_vecs[i] for i in self.META_IDS]
            elif self.TRIAL_IDS is not None:  # self.TRIAL_IDS consists of Trial IDs (0, 1, ...)
                assert len(self.selected_movements_indices) == len(
                    self.indices), f"ERROR: 'indices' and 'selected_movements_indices' do not have the same length!"
                self.META_IDS = np.where(np.isin(self.indices[:, 4], self.TRIAL_IDS))[0]
                self.selected_movements_indices = [self.selected_movements_indices[i] for i in self.META_IDS]
            elif self.META_IDS is not None:
                self.selected_movements_indices = [self.selected_movements_indices[i] for i in self.META_IDS]
            elif self.N_MOVS is not None:
                # Only keep first N_MOVS trials:
                ##self._indices = self._indices[:self.N_MOVS]
                # no need to start to use "1:self.N_MOVS+1" here, since order of trials should have already changed in self._indices and self.selected_movements_indices...
                self.selected_movements_indices = self.selected_movements_indices[:self.N_MOVS]
                # self.target_positions_per_trial = self.target_positions_per_trial[:self.N_MOVS]
                # self.target_vecs = self.target_vecs[:self.N_MOVS]

            if self.TARGET_IDS is None and self.TRIAL_IDS is not None:
                assert set(self.TRIAL_IDS).issubset(
                    set(self.indices[:, 4])), f"ERROR: Invalid Trial ID(s)! Valid Trial IDs:\n{self.indices[:, 4]}"

            # Compute target vector for each trial to ensure that trial data is consistent and correct
            # TODO: check if these lines are deprecated
            target_position_series = self._target_position_series  # [0] if self.AGGREGATE_TRIALS else self._target_position_series
            if not ignore_trainingset_trials:
                assert len(np.unique(np.round(target_position_series[self._indices[self.trials_to_current_init_pos, 0]],
                                          decimals=12))) == 3, f"ERROR: Target positions do not match for trials with same init/target id!"
            # self.selected_movements_indices_per_trial = list(zip(np.hstack(([self._indices[self.trials_to_current_init_pos[0], 0]], self._indices[:-1, 0])), self._indices[:, 0], self._indices[:, 1]))
            ### VARIANT 1:  #FLAG123
            if self.FILENAME_STUDY_TARGETPOSITIONS is None:
                self.target_positions_per_trial = target_position_series[np.squeeze(self.selected_movements_indices)[..., :2], :]
            ### VARIANT 2:
            else:
                self.target_positions_per_trial = np.array([target_position_series[i, :] if i[0] >= 0 else [
                    self._target_positions_by_ID[-(i[0] + 100)], target_position_series[i[1], :]] for i in
                                                            np.array(self.selected_movements_indices)[..., :2]])
            #####################
            self.target_vecs = self.target_positions_per_trial[..., 1, :] - self.target_positions_per_trial[..., 0, :]

        self.trials_defined = True

        return self.selected_movements_indices

    def combine_indices(self, indices_list, n_samples_list):
        assert len(indices_list) == len(
            n_samples_list), "Number of indices lists/arrays to be combined does not match given number of samples per TrajectoryData instance."

        if not isinstance(self, TrajectoryData_MultipleInstances):
            logging.warning(
                "WARNING: This function should only be called from a 'TrajectoryData_MultipleInstances' instance!")

        self._indices = np.vstack([indices + (sum(n_samples_list[:meta_idx]) * np.hstack(
            (np.ones((indices.shape[0], 2)), np.zeros((indices.shape[0], 3))))) for (meta_idx, indices) in
                                   enumerate(indices_list)]).astype(int)
        self._indices_copy = self._indices.copy()

        return self._indices

    def _control_xml_to_DataFrame(self, filename):
        with open(filename, 'rb') as f:
            data_StaticOptimization_STUDY_xml = xmltodict.parse(f.read())

        data_StaticOptimization_STUDY_times = {
            cl['@name']: [float(cl_el['t']) for cl_el in cl['x_nodes']['ControlLinearNode']] for cl in
            data_StaticOptimization_STUDY_xml['OpenSimDocument']['ControlSet']['objects']['ControlLinear']}
        data_StaticOptimization_STUDY_values = {
            cl['@name']: [float(cl_el['value']) for cl_el in cl['x_nodes']['ControlLinearNode']] for cl in
            data_StaticOptimization_STUDY_xml['OpenSimDocument']['ControlSet']['objects']['ControlLinear']}
        data_StaticOptimization_STUDY_time = {'time': list(data_StaticOptimization_STUDY_times.values())[0]}
        assert all([data_StaticOptimization_STUDY_time['time'] == data_StaticOptimization_STUDY_times[i] for i in
                    data_StaticOptimization_STUDY_times])

        data_StaticOptimization_STUDY = pd.concat(
            (pd.DataFrame(data_StaticOptimization_STUDY_time), pd.DataFrame(data_StaticOptimization_STUDY_values)),
            axis=1)

        return data_StaticOptimization_STUDY


def project_trajectory(trajectory, init_val=None, final_val=None, use_rel_vals=True, normalize_quantity=True,
                       output_deviation=False):
    # INFO: use_rel_vals should be True when projecting position, but not when projecting velocity, acceleration, etc.!
    # INFO: if output_deviation is True, deviation from direct path between initial and target position
    if init_val is None:
        init_val = trajectory[0]
    if final_val is None:
        final_val = trajectory[-1]
    init_val = np.array(init_val)
    final_val = np.array(final_val)
    assert np.any(init_val != final_val)

    proj_vector = final_val - init_val
    if normalize_quantity:  # results in normalized projected position, velocity, etc.
        proj_vector_normalized = proj_vector / np.linalg.norm(proj_vector) ** 2  # normalize vector to project on
    else:
        proj_vector_normalized = proj_vector / np.linalg.norm(proj_vector)  # normalize vector to project on
    proj_vector_normalized = proj_vector_normalized.flatten()

    projected_trajectory = np.array(
        [np.dot(state_vec - init_val * use_rel_vals, proj_vector_normalized) for state_vec in trajectory])

    if output_deviation:
        # compute deviation from direct path
        projected_trajectory_deviation = np.array(
            [np.linalg.norm((state_vec - init_val * use_rel_vals) - state_vec_projected * proj_vector) for
             state_vec, state_vec_projected in zip(trajectory, projected_trajectory)])
        return projected_trajectory_deviation

    return projected_trajectory


def project_trajectory_cov(trajectory, init_val, final_val, normalize_quantity=True):
    init_val = np.array(init_val)
    final_val = np.array(final_val)
    assert np.any(init_val != final_val)

    proj_vector = final_val - init_val
    if normalize_quantity:  # results in normalized projected position, velocity, etc.
        proj_vector_normalized = proj_vector / np.linalg.norm(proj_vector) ** 2  # normalize vector to project on
    else:
        proj_vector_normalized = proj_vector / np.linalg.norm(proj_vector)  # normalize vector to project on
    proj_vector_normalized = proj_vector_normalized.flatten()

    projected_trajectory_cov = np.array(
        [np.dot(proj_vector_normalized, np.dot(state_cov, proj_vector_normalized)) for state_cov in trajectory])
    return projected_trajectory_cov


def compute_trajectory_statistics(trajectory, current_idx_list, next_idx_list, project=False, rel_to_init=False,
                                  normalize=False, **kwargs):
    assert isinstance(current_idx_list, list)
    assert isinstance(next_idx_list, list)

    if "init_val" in kwargs and np.array(kwargs["init_val"]).ndim != 1:
        assert np.array(kwargs["init_val"]).ndim == np.array(kwargs["final_val"]).ndim == 2
        assert len(kwargs["init_val"]) == len(kwargs["final_val"]) == len(current_idx_list) == len(next_idx_list)
        different_init_and_final_vals = True
    else:
        different_init_and_final_vals = False

    trajectory_trials = []
    for meta_idx, (current_idx, next_idx) in enumerate(zip(current_idx_list, next_idx_list)):
        if project:
            # input((kwargs["init_val"], kwargs["final_val"], trajectory[current_idx: next_idx]))
            # input((kwargs["final_val"] - kwargs["init_val"]))
            if different_init_and_final_vals:
                kwargs_modified = kwargs.copy()
                kwargs_modified["init_val"] = kwargs["init_val"][meta_idx]
                kwargs_modified["final_val"] = kwargs["final_val"][meta_idx]
                trajectory_trial = project_trajectory(trajectory[current_idx: next_idx], **kwargs_modified)
            else:
                trajectory_trial = project_trajectory(trajectory[current_idx: next_idx], **kwargs)
        else:
            trajectory_trial = trajectory[current_idx: next_idx].copy()
        if rel_to_init:
            trajectory_trial -= trajectory_trial[0]
        if normalize:  # for time series, use trajectory_min generated with normalize=True
            trajectory_trial /= np.max(trajectory_trial)
        trajectory_trials.append(trajectory_trial)

    data_lengths = sorted([len(i) for i in trajectory_trials], reverse=True)
    if len(data_lengths) == 1:
        data_lengths.append(data_lengths[0])
    global test_var, test_current_idx_list, test_next_idx_list
    test_var = trajectory
    test_current_idx_list = current_idx_list
    test_next_idx_list = next_idx_list

    trajectory_mean = np.array(
        [np.mean([k[j] for k in trajectory_trials if j < k.shape[0]], axis=0) for j in range(data_lengths[0])])
    trajectory_cov = np.array(
        [np.cov([k[j] for k in trajectory_trials if j < k.shape[0]], rowvar=False) for j in range(data_lengths[1])] + [
            np.array(0.) if trajectory_trials[0].ndim == 1 else np.zeros(
                (trajectory_trials[0].shape[1], trajectory_trials[0].shape[1])) for _ in
            range(data_lengths[1], data_lengths[0])])
    trajectory_min = np.array(
        [np.min([k[j] for k in trajectory_trials if j < k.shape[0]], axis=0) for j in range(data_lengths[0])])
    trajectory_max = np.array(
        [np.max([k[j] for k in trajectory_trials if j < k.shape[0]], axis=0) for j in range(data_lengths[0])])

    assert len(trajectory_mean) == len(
        trajectory_cov)  # ensure that indices apply to both trajectory_mean and trajectory_cov

    return trajectory_mean, trajectory_cov, trajectory_min, trajectory_max


# Compute Minjerk trajectory as reference trajectory
def minimumjerk_deterministic(N, T, initialuservalues, dim, P, dt, x0, final_vel=None, final_acc=None,
                              passage_times=None):
    if x0 is None:
        x0 = np.hstack((np.mean(np.squeeze(initialuservalues), axis=0), T[0]))

    if passage_times is None:
        passage_times = np.linspace(0, N, P).astype(
            int)  # WARNING: here: equally distributed target passage times!
    # else:
    #     passage_times = np.insert(passage_times, 0, 0)  #assume that argument "passage_times" had P - 1 values, as first "target switch" (i.e., from initial position to first real target) is fixed

    assert len(passage_times) == P

    n = dim * (3 + P)  # dimension of state vector (incorporating position, velocity, acceleration and "via-points" T)

    assert P == 2
    N_minjerk = np.ceil(passage_times[1]).astype(int)

    if final_vel is None:  # only used in VARIANT 2
        final_vel = np.zeros((dim,))
    if final_acc is None:  # only used in VARIANT 2
        final_acc = np.zeros((dim,))
    assert final_vel.shape == (dim,)
    assert final_acc.shape == (dim,)

    #     ### VARIANT 1:  (this variant terminates with zero velocity and acceleration, independent of "final_vel" and "final_acc"!)
    #     ## (Online) Control Algorithm

    #     A = [None] * (N_minjerk)
    #     B = [None] * (N_minjerk)

    #     x = np.zeros((N + 1, n))
    #     u = np.zeros((N, dim))

    #     x[0] = x0
    #     for i in range(0, N_minjerk):
    #         u[i] = T[0][1*dim:]

    #         ### COMPUTE time-dependent system matrices (with time relexation due to usage of passage_times[1] instead of int(passage_times[1]); for stochastic wrapper, see minjerk_get_all_timedependent_system_matrices()):
    #         movement_time = (passage_times[1] - i) * dt
    #         A_continuous = np.vstack(
    #             (np.hstack((np.zeros(shape=(dim, dim)), np.eye(dim), np.zeros(shape=(dim, (P + 1) * dim)))),
    #              np.hstack((np.zeros(shape=(dim, 2 * dim)), np.eye(dim), np.zeros(shape=(dim, P * dim)))),
    #              np.hstack(((-60 / (movement_time ** 3)) * np.eye(dim), (-36 / (movement_time ** 2)) * np.eye(dim),
    #                         (-9 / movement_time) * np.eye(dim), np.zeros(shape=(dim, P * dim)))),
    #              np.zeros(shape=(P * dim, (P + 3) * dim))))
    #         B_continuous = np.vstack((np.zeros(shape=(2 * dim, dim)),
    #                                   (60 / (movement_time ** 3)) * np.eye(dim), np.zeros(shape=(P * dim, dim))))
    #         # using explicit solution formula:
    #         A[i] = expm(A_continuous * dt)
    #         B[i] = np.linalg.pinv(A_continuous).dot(A[i] - np.eye((3 + P) * dim)).dot(B_continuous)
    #         # B[i] = np.vstack((np.linalg.inv(A_continuous[:3*dim, :3*dim]).dot(A[i][:3*dim, :3*dim] - np.eye(3 * dim)).dot(B_continuous[:3*dim]),
    #         #                     np.zeros(shape=(P * dim, dim))))
    #         ############################################

    #         x[i + 1] = A[i].dot(x[i]) + B[i].dot(u[i])

    #     for i in range(N_minjerk, N):  # extend trajectory with constant last value
    #         u[i] = T[0][1*dim:]
    #         x[i + 1] = x[i] * np.repeat(np.array([1, 0, 0, 1, 1]), dim)

    ### VARIANT 2:
    # Explicit Solution Formula (current implementation only yields position time series!)
    t_f = passage_times[1] * dt
    # compute MinJerk trajectory per dimension

    # SHORT CODE:
    coeff_vec = np.array([[x0[0 + i], t_f * x0[1 * dim + i], 0.5 * (t_f ** 2) * x0[2 * dim + i],
                           -10 * x0[0 + i] - 6 * t_f * x0[1 * dim + i] - 1.5 * (t_f ** 2) * x0[2 * dim + i] + 10 * T[0][
                               1 * dim + i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i],
                           15 * x0[0 + i] + 8 * t_f * x0[1 * dim + i] + 1.5 * (t_f ** 2) * x0[2 * dim + i] - 15 * T[0][
                               1 * dim + i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) * final_acc[i],
                           -6 * x0[0 + i] - 3 * t_f * x0[1 * dim + i] - 0.5 * (t_f ** 2) * x0[2 * dim + i] + 6 * T[0][
                               1 * dim + i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]] for i in
                          range(dim)] +
                         [[x0[1 * dim + i], t_f * x0[2 * dim + i],
                           (3 / t_f) * (-10 * x0[0 + i] - 6 * t_f * x0[1 * dim + i] - 1.5 * (t_f ** 2) * x0[
                               2 * dim + i] + 10 * T[0][1 * dim + i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) *
                                        final_acc[i]),
                           (4 / t_f) * (15 * x0[0 + i] + 8 * t_f * x0[1 * dim + i] + 1.5 * (t_f ** 2) * x0[
                               2 * dim + i] - 15 * T[0][1 * dim + i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) *
                                        final_acc[i]),
                           (5 / t_f) * (-6 * x0[0 + i] - 3 * t_f * x0[1 * dim + i] - 0.5 * (t_f ** 2) * x0[
                               2 * dim + i] + 6 * T[0][1 * dim + i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) *
                                        final_acc[i]), 0] for i in range(dim)] +
                         [[x0[2 * dim + i],
                           (2 / t_f) * (3 / t_f) * (-10 * x0[0 + i] - 6 * t_f * x0[1 * dim + i] - 1.5 * (t_f ** 2) * x0[
                               2 * dim + i] + 10 * T[0][1 * dim + i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) *
                                                    final_acc[i]),
                           (3 / t_f) * (4 / t_f) * (15 * x0[0 + i] + 8 * t_f * x0[1 * dim + i] + 1.5 * (t_f ** 2) * x0[
                               2 * dim + i] - 15 * T[0][1 * dim + i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) *
                                                    final_acc[i]),
                           (4 / t_f) * (5 / t_f) * (-6 * x0[0 + i] - 3 * t_f * x0[1 * dim + i] - 0.5 * (t_f ** 2) * x0[
                               2 * dim + i] + 6 * T[0][1 * dim + i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) *
                                                    final_acc[i]), 0, 0] for i in range(dim)])
    x = np.squeeze(
        [coeff_vec @ np.array([(j / passage_times[1]) ** ii for ii in range(6)]) for j in range(N_minjerk + 1)])
    # LONG CODE:
    # x = []
    # for i in range(dim):  #position
    #     coeff_vec = np.array([x0[0 + i], t_f * x0[1*dim + i], 0.5 * (t_f ** 2) * x0[2*dim + i],
    #                        -10 * x0[0 + i] - 6 * t_f * x0[1*dim + i] - 1.5 * (t_f ** 2) * x0[2*dim + i] + 10 * T[0][1*dim + i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i],
    #                        15 * x0[0 + i] + 8 * t_f * x0[1*dim + i] + 1.5 * (t_f ** 2) * x0[2*dim + i] - 15 * T[0][1*dim + i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) * final_acc[i],
    #                        -6 * x0[0 + i] - 3 * t_f * x0[1*dim + i] - 0.5 * (t_f ** 2) * x0[2*dim + i] + 6 * T[0][1*dim + i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]])
    #     x_current_dimension = [coeff_vec @ np.array([(j / passage_times[1]) ** ii for ii in range(6)]) for j in range(N_minjerk + 1)]
    #     x.append(x_current_dimension)
    # for i in range(dim):  #velocity
    #     coeff_vec = np.array([x0[1*dim + i], t_f * x0[2*dim + i],
    #                        (3 / t_f) * (-10 * x0[0 + i] - 6 * t_f * x0[1*dim + i] - 1.5 * (t_f ** 2) * x0[2*dim + i] + 10 * T[0][1*dim + i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]),
    #                        (4 / t_f) * (15 * x0[0 + i] + 8 * t_f * x0[1*dim + i] + 1.5 * (t_f ** 2) * x0[2*dim + i] - 15 * T[0][1*dim + i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) * final_acc[i]),
    #                        (5 / t_f) * (-6 * x0[0 + i] - 3 * t_f * x0[1*dim + i] - 0.5 * (t_f ** 2) * x0[2*dim + i] + 6 * T[0][1*dim + i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i])])
    #     x_current_dimension = [coeff_vec @ np.array([(j / passage_times[1]) ** ii for ii in range(5)]) for j in range(N_minjerk + 1)]
    #     x.append(x_current_dimension)
    # for i in range(dim):  #acceleration
    #     coeff_vec = np.array([x0[2*dim + i],
    #                        (2 / t_f) * (3 / t_f) * (-10 * x0[0 + i] - 6 * t_f * x0[1*dim + i] - 1.5 * (t_f ** 2) * x0[2*dim + i] + 10 * T[0][1*dim + i] - 4 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i]),
    #                        (3 / t_f) * (4 / t_f) * (15 * x0[0 + i] + 8 * t_f * x0[1*dim + i] + 1.5 * (t_f ** 2) * x0[2*dim + i] - 15 * T[0][1*dim + i] + 7 * t_f * final_vel[i] - 1 * (t_f ** 2) * final_acc[i]),
    #                        (4 / t_f) * (5 / t_f) * (-6 * x0[0 + i] - 3 * t_f * x0[1*dim + i] - 0.5 * (t_f ** 2) * x0[2*dim + i] + 6 * T[0][1*dim + i] - 3 * t_f * final_vel[i] + 0.5 * (t_f ** 2) * final_acc[i])])
    #     x_current_dimension = [coeff_vec @ np.array([(j / passage_times[1]) ** ii for ii in range(4)]) for j in range(N_minjerk + 1)]
    #     x.append(x_current_dimension)
    # x = np.array(x).T

    if N > N_minjerk:
        x = np.concatenate((x, [x[-1]] * (N - N_minjerk)))
    u = None  # [T[0][1*dim:]] * (N_minjerk + 1)

    return x, u

