import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import os

# -> creates complete.csv files required for TrajectoryData_MPC instances (adapted from plots_markus.ipynb)
def preprocess_movement_data_simulation(simulation_dirname, study_dirname, participant, task_condition, user_initial_delay=0.5, start_outside_target=False, initial_acceleration_constraint=0, scale_movement_times=True, scale_to_mean_movement_times=True, simulation_subdir="sim_wt20"):

    simulation_dirname = os.path.abspath(simulation_dirname)
    study_dirname = os.path.abspath(study_dirname)

    target_switch_indices_data_users_markers = np.load(os.path.join(study_dirname, f'_trialIndices/{participant}_{task_condition}_SubMovIndices.npy'), allow_pickle=True)

    experiment_info_timestamp = list(set([f.split('_')[0] for f in os.listdir(os.path.join(study_dirname, "_trialIndices")) if (os.path.isfile(os.path.join(study_dirname, "_trialIndices", f))) & (participant in f) & (task_condition in f)]))
    assert len(experiment_info_timestamp) == 1
    experiment_info = pd.read_csv(os.path.join(study_dirname, '_trialData/Experiment_{}_{}.csv'.format(participant, task_condition)))
    
    #TRANSLATION OF TARGET POSITIONS FROM GLOBAL COORDINATE SPACE TO SHOULDER-CENTERED COORDINATE SPACE:
    target_array_user = (experiment_info.loc[:, "Target.Position.x":"Target Position.z"].iloc[0:] - experiment_info.loc[0, "Shoulder.Position.x":"Shoulder.Position.z"].tolist()).to_numpy()
    #target_array_user = (experiment_info.loc[:, "Target.Position.x":"Target Position.z"].iloc[0:]).to_numpy()
    target_array_user *= np.array([-1, 1, 1])

    joint_list = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]

    trajectory_plots_data_COMPLETE = pd.DataFrame()
    target_switch_indices_single = np.array([0])
    for movement_id in range(len(target_array_user)-1):
    #    trajectory_plots_data = trajectory_plots_data.append(pd.read_csv(os.path.join(simulation_dirname, '{}/fittstask-osim-13-09-2020-18-45-34{}_{}_/{:02d}.csv'.format(participant, task_condition, participant, movement_id))))
        try:
    #             trajectory_plots_data = trajectory_plots_data.append(pd.read_csv(os.path.join(simulation_dirname, '{:02d}.csv'.format(movement_id))))
            trajectory_plots_data = pd.read_csv(os.path.join(simulation_dirname, '{}/{}/{}/{}/complete.csv'.format(participant, simulation_subdir, task_condition, movement_id)))
            
            #TRANSLATION OF SIMULATION BODY POSITIONS FROM GLOBAL COORDINATE SPACE TO SHOULDER-CENTERED COORDINATE SPACE:
            # Shift shoulder to origin:
            rotshoulder = experiment_info.loc[0, "Shoulder.Position.x":"Shoulder.Position.z"].values * np.array([-1, 1, 1])
            for xpos_column in trajectory_plots_data.columns:
                if 'xpos_x' in xpos_column:
                    trajectory_plots_data.loc[:, xpos_column] -= rotshoulder[0]
                elif 'xpos_y' in xpos_column:
                    trajectory_plots_data.loc[:, xpos_column] -= rotshoulder[1]
                elif 'xpos_z' in xpos_column:
                    trajectory_plots_data.loc[:, xpos_column] -= rotshoulder[2]
                    
            ### REMOVE (WRONG) INITIAL VELOCITY:
            ####trajectory_plots_data.loc[trajectory_plots_data.index[0], ["end-effector_xvel_" + xyz for xyz in ['x', 'y', 'z']]] = np.nan

            # TODO: apply transfer function!
            
            # Filter trajectories using Savitzky-Golay Filter (only accelerations and jerk):
            trajectory_plots_data_filtered = pd.DataFrame(index=trajectory_plots_data.index, columns=["time"] + ["end-effector" + suffix + xyz for suffix in ["_xpos_", "_xvel_", "_xacc_", "_xjerk_"] for xyz in ['x', 'y', 'z']] + ["end_effector" + suffix + "_proj" for suffix in ["_pos", "_vel", "_acc", "_jerk"]] + [jn + suffix for suffix in ["_pos", "_vel", "_acc", "_frc"] for jn in joint_list] + ['A_' + jn for jn in joint_list])
            trajectory_plots_data_filtered["time"] = trajectory_plots_data["time"]
            trajectory_plots_data_filtered[["end-effector_xpos_" + xyz for xyz in ['x', 'y', 'z']]] = trajectory_plots_data[["end-effector_xpos_" + xyz for xyz in ['x', 'y', 'z']]]#.apply(lambda x: savgol_filter(x, 15, 3, deriv=0, delta = trajectory_plots_data.time.diff().mean(), axis=0))
            trajectory_plots_data_filtered[["end-effector_xvel_" + xyz for xyz in ['x', 'y', 'z']]] = trajectory_plots_data[["end-effector_xvel_" + xyz for xyz in ['x', 'y', 'z']]]#.apply(lambda x: savgol_filter(x, 15, 3, deriv=0, delta = trajectory_plots_data.time.diff().mean(), axis=0))
            trajectory_plots_data_filtered[["end-effector_xacc_" + xyz for xyz in ['x', 'y', 'z']]] = trajectory_plots_data[["end-effector_xvel_" + xyz for xyz in ['x', 'y', 'z']]].apply(lambda x: savgol_filter(x, 15, 3, deriv=1, delta = trajectory_plots_data.time.diff().mean(), axis=0))
            trajectory_plots_data_filtered[["end-effector_xjerk_" + xyz for xyz in ['x', 'y', 'z']]] = trajectory_plots_data[["end-effector_xvel_" + xyz for xyz in ['x', 'y', 'z']]].apply(lambda x: savgol_filter(x, 15, 3, deriv=2, delta = trajectory_plots_data.time.diff().mean(), axis=0))
            qpos_SG = trajectory_plots_data[[cn + '_pos' for cn in joint_list]]#.apply(lambda x: savgol_filter(x, 15, 3, deriv=0, delta = trajectory_plots_data.time.diff().mean(), axis=0))
            qvel_SG = trajectory_plots_data[[cn + '_vel' for cn in joint_list]]#.apply(lambda x: savgol_filter(x, 15, 3, deriv=0, delta = trajectory_plots_data.time.diff().mean(), axis=0))
            qacc_SG = trajectory_plots_data[[cn + '_vel' for cn in joint_list]].apply(lambda x: savgol_filter(x, 15, 3, deriv=1, delta = trajectory_plots_data.time.diff().mean(), axis=0))
            tau_SG = trajectory_plots_data[[cn + '_frc' for cn in joint_list]]#.apply(lambda x: savgol_filter(x, 15, 3, deriv=0, delta = trajectory_plots_data.time.diff().mean(), axis=0))
            trajectory_plots_data_filtered[[cn + '_pos' for cn in joint_list]] = qpos_SG
            trajectory_plots_data_filtered[[cn + '_vel' for cn in joint_list]] = qvel_SG
            trajectory_plots_data_filtered[[cn + '_acc' for cn in joint_list]] = qacc_SG
            trajectory_plots_data_filtered[[cn + '_frc' for cn in joint_list]] = tau_SG
            trajectory_plots_data_filtered[['A_' + cn for cn in joint_list]] = trajectory_plots_data[['A_' + cn for cn in joint_list]]
            trajectory_plots_data_filtered[['ACT_' + cn for cn in joint_list]] = trajectory_plots_data[['ACT_' + cn for cn in joint_list]]
            trajectory_plots_data_filtered[['EXT_' + cn for cn in joint_list]] = trajectory_plots_data[['EXT_' + cn for cn in joint_list]]

            # Compute end-effector projections:
    #         trajectory_plots_data_filtered = pd.concat((trajectory_plots_data_filtered, trajectory_plots_data_filtered["time"].apply(lambda x: pd.Series(target_array_user[[int(init_pos_ID) for (marker_time_raw, _, init_pos_ID, _, _) in marker_times_tuples_without_acc_constraint_raw if x >= marker_time_raw][-1]] if len([init_pos_ID for (marker_time_raw, _, init_pos_ID, _, _) in marker_times_tuples_without_acc_constraint_raw if x >= marker_time_raw]) > 0 else np.nan, index=['init_x', 'init_y', 'init_z']))), axis=1)
    #         trajectory_plots_data_filtered = pd.concat((trajectory_plots_data_filtered, trajectory_plots_data_filtered["time"].apply(lambda x: pd.Series(target_array_user[[int(target_pos_ID) for (marker_time_raw, _, _, target_pos_ID, _) in marker_times_tuples_without_acc_constraint_raw if x >= marker_time_raw][-1]] if len([target_pos_ID for (marker_time_raw, _, _, target_pos_ID, _) in marker_times_tuples_without_acc_constraint_raw if x >= marker_time_raw]) > 0 else np.nan, index=['target_x', 'target_y', 'target_z']))), axis=1)
            trajectory_plots_data_filtered[['init_x', 'init_y', 'init_z']] = target_array_user[movement_id]
            trajectory_plots_data_filtered[['target_x', 'target_y', 'target_z']] = target_array_user[movement_id + 1]
            normalizedCentroidVector = (trajectory_plots_data_filtered[["target_x", "target_y", "target_z"]].values - trajectory_plots_data_filtered[["init_x", "init_y", "init_z"]].values)/np.tile(np.linalg.norm((trajectory_plots_data_filtered[["target_x", "target_y", "target_z"]].values - trajectory_plots_data_filtered[["init_x", "init_y", "init_z"]].values), axis=1), (3, 1)).transpose()
            #input((trajectory_plots_data_filtered[["end-effector_xpos_" + xyz + '_pos' for xyz in ['x', 'y', 'z']]], trajectory_plots_data_filtered[['init_x', 'init_y', 'init_z']], trajectory_plots_data_filtered[['target_x', 'target_y', 'target_z']]))
            centroidVectorProjection = ((trajectory_plots_data_filtered[["end-effector_xpos_" + xyz for xyz in ['x', 'y', 'z']]] - trajectory_plots_data_filtered[["init_x", "init_y", "init_z"]].values) * normalizedCentroidVector).sum(axis=1)
            centroidVelProjection = ((trajectory_plots_data_filtered[["end-effector_xvel_" + xyz for xyz in ['x', 'y', 'z']]]) * normalizedCentroidVector).sum(axis=1)
            centroidAccProjection = ((trajectory_plots_data_filtered[["end-effector_xacc_" + xyz for xyz in ['x', 'y', 'z']]]) * normalizedCentroidVector).sum(axis=1)
            centroidJerkProjection = ((trajectory_plots_data_filtered[["end-effector_xjerk_" + xyz for xyz in ['x', 'y', 'z']]]) * normalizedCentroidVector).sum(axis=1)
            #input((centroidVectorProjection))
            trajectory_plots_data_filtered['end_effector_pos_proj'] = centroidVectorProjection
            trajectory_plots_data_filtered['end_effector_vel_proj'] = centroidVelProjection
            trajectory_plots_data_filtered['end_effector_acc_proj'] = centroidAccProjection
            trajectory_plots_data_filtered['end_effector_jerk_proj'] = centroidJerkProjection

            trajectory_plots_data_COMPLETE = pd.concat((trajectory_plots_data_COMPLETE, trajectory_plots_data_filtered))
            target_switch_indices_single = np.concatenate((target_switch_indices_single, np.array([trajectory_plots_data_COMPLETE.shape[0]])))
        except FileNotFoundError:
            print('{}/{}/{}/{}/{}/complete.csv not found!'.format(simulation_dirname, participant, simulation_subdir, task_condition, movement_id))
            ### WARNING: The following line ignores movements for which no simulation data is available!
            target_switch_indices_data_users_markers = np.delete(target_switch_indices_data_users_markers, np.where(target_switch_indices_data_users_markers[:, -1] == movement_id)[0], axis=0)
    target_switch_indices = np.array([(i,j-1) for (i,j) in zip(target_switch_indices_single[:-1], target_switch_indices_single[1:])])
    #trajectory_plots_data = trajectory_plots_data.drop("Unnamed: 0", axis=1).reset_index()
    trajectory_plots_data = trajectory_plots_data_COMPLETE.reset_index()
    del trajectory_plots_data_COMPLETE
    ##target_switch_indices = np.delete(target_switch_indices, movement_indices_TO_DELETE, axis=0)
    #target_switch_indices = target_switch_indices[target_switch_indices_data_users_markers[:, 4]]
    #print(len(target_switch_indices), target_switch_indices_data_users_markers.shape[0])
    assert len(target_switch_indices) == target_switch_indices_data_users_markers.shape[0]
    target_switch_indices = np.hstack((target_switch_indices, target_switch_indices_data_users_markers[:, 2:])).astype(int)
    
    np.save(os.path.join(simulation_dirname, f'{participant}/{simulation_subdir}/{task_condition}/SubMovIndices.npy'), target_switch_indices)
    
    # Store filtered (and filtered and projected) simulation trajectories (i.e., no recomputations are required during plotting):
    if not os.path.exists(os.path.expanduser(os.path.join(simulation_dirname, f'{participant}/{simulation_subdir}/{task_condition}'))):
        try:
            os.makedirs(os.path.expanduser(os.path.join(simulation_dirname, f'{participant}/{simulation_subdir}/{task_condition}')))
        except FileExistsError:  #might occur when multiple instances run in parallel (e.g., on cluster)
            pass
    trajectory_plots_data.to_csv(os.path.join(simulation_dirname, f'{participant}/{simulation_subdir}/{task_condition}/complete.csv'))
    
    return trajectory_plots_data, target_switch_indices

def preprocess_movement_data_simulation_complete(simulation_dirname, study_dirname, user_initial_delay=0.5, start_outside_target=False, initial_acceleration_constraint=0, scale_movement_times=True, scale_to_mean_movement_times=True, participant_list=['U1', 'U2', 'U3', 'U4', 'U5', 'U6'], task_condition_list=["Standing_ID_ISO_15_plane", "Standing_Cursor_Handtuned_ISO_15_plane", "Standing_Pad_ID_ISO_15_plane", "Standing_Pad_Handtuned_vertical_ISO_15_plane"], simulation_subdir='sim_wt20'):
    # participant_list = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6']  #['Armin', 'Joerg', 'Maria', 'Markus', 'Miroslav', 'Sina']
    # task_condition_list = ["Standing_ID_ISO_15_plane", "Standing_Cursor_Handtuned_ISO_15_plane", "Standing_Pad_ID_ISO_15_plane", "Standing_Pad_Handtuned_vertical_ISO_15_plane"]  #, "Sitting_ID_ISO_15_plane",
    #               #"Sitting_Cursor_Handtuned_ISO_15_plane", "Sitting_Pad_ID_ISO_15_plane",
    #               #"Sitting_Pad_Handtuned_vertical_ISO_15_plane"]
    
    # LOAD DATA:
    trajectory_plots_data, target_switch_indices = {}, {}
    for task_condition in task_condition_list:
        trajectory_plots_data[task_condition], target_switch_indices[task_condition] = {}, {}
        for current_participant in participant_list:
            trajectory_plots_data[task_condition][current_participant], target_switch_indices[task_condition][current_participant] = preprocess_movement_data_simulation(simulation_dirname, study_dirname, current_participant, task_condition, user_initial_delay=user_initial_delay, start_outside_target=start_outside_target, initial_acceleration_constraint=initial_acceleration_constraint, scale_movement_times=scale_movement_times, scale_to_mean_movement_times=scale_to_mean_movement_times, simulation_subdir=simulation_subdir)

    return trajectory_plots_data, target_switch_indices