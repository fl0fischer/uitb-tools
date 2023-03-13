import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import pickle
import logging
import io, os
from uitb_evaluate.trajectory_data import PLOTS_DIR_DEFAULT, INDEPENDENT_JOINTS, \
    TrajectoryData_MPC, TrajectoryData_MultipleInstances

def trajectoryplot(PLOTTING_ENV, USER_ID, TASK_CONDITION,
                   common_simulation_subdir, filename, trajectories_SIMULATION,
                   trajectories_STUDY=None,
                   trajectories_SUPPLEMENTARY=None,
                   independent_joints=INDEPENDENT_JOINTS,
                   REPEATED_MOVEMENTS=False,
                   USER_ID_FIXED="<unknown-user>",
                   ignore_trainingset_trials_mpc_userstudy=True,  #only used if PLOTTING_ENV.startswith("MPC-userstudy")
                   MOVEMENT_IDS=None, RADIUS_IDS=None, EPISODE_IDS=None, r1_FIXED=None, r2_FIXED=None,
                   r1list=None, r2list=None, COST_FUNCTION=None,
                   EFFECTIVE_PROJECTION_PATH=True, USE_TARGETBOUND_AS_DIST=False, MINJERK_USER_CONSTRAINTS=False,
                   TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None, N_MOVS=None, AGGREGATION_VARS=[],
                   PLOT_TRACKING_DISTANCE=False, PLOT_ENDEFFECTOR=False, JOINT_ID=0,
                   PLOT_DEVIATION=False,
                   NORMALIZE_TIME=False, #DWELL_TIME=0.3,
                   PLOT_TIME_SERIES=True,
                   PLOT_VEL_ACC=False, PLOT_RANGES=False, CONF_LEVEL="min/max",
                   SHOW_MINJERK=False, SHOW_STUDY=False, STUDY_ONLY=False,
                   ENABLE_LEGENDS_AND_COLORBARS=True, ALLOW_DUPLICATES_BETWEEN_LEGENDS=False,
                   predefined_ylim=None,
                   plot_width="thirdpage",
                   STORE_PLOT=False, STORE_AXES_SEPARATELY=True, PLOTS_DIR=PLOTS_DIR_DEFAULT):

    # # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
    # MOVEMENT_IDS = None #range(1,9) #[i for i in range(10) if i != 1]
    # RADIUS_IDS = None
    # EPISODE_IDS = [7]
    #
    # r1_FIXED = None #r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
    # r2_FIXED = None #r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"
    #
    # #TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"
    #
    # # WHAT TO COMPUTE?
    # EFFECTIVE_PROJECTION_PATH = (PLOTTING_ENV == "RL-UIB")  #if True, projection path connects effective initial and final position instead of nominal target center positions
    # USE_TARGETBOUND_AS_DIST = False  #True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
    # MINJERK_USER_CONSTRAINTS = True
    #
    # # WHICH/HOW MANY MOVS?
    # """
    # IMPORTANT INFO:
    # if isinstance(trajectories, TrajectoryData_RL):
    #     -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
    #     -> TRIAL_IDS and META_IDS are equivalent
    # if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
    #     -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
    #     -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
    # In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
    # """
    # TARGET_IDS = range(1,4)  #corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
    # TRIAL_IDS = range(1, 14)  #[i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
    # META_IDS = None  #index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
    # N_MOVS = None  #number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
    # AGGREGATION_VARS = [] #["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]
    #
    # # WHAT TO PLOT?
    # PLOT_TRACKING_DISTANCE = False  #if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
    # PLOT_ENDEFFECTOR = False  #if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
    # JOINT_ID = 2  #only used if PLOT_ENDEFFECTOR == False
    # PLOT_DEVIATION = False  #only if PLOT_ENDEFFECTOR == True
    #
    # # HOW TO PLOT?
    # NORMALIZE_TIME = False
    # #DWELL_TIME = 0.3  #tail of the trajectories that is not shown (in seconds)
    # PLOT_TIME_SERIES = True  #if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
    # PLOT_VEL_ACC = False  #if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
    # PLOT_RANGES = False
    # #CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True
    #
    # # WHICH BASELINE?
    # SHOW_MINJERK = False
    # SHOW_STUDY = True
    # STUDY_ONLY = False  #only used if PLOTTING_ENV == "MPC-taskconditions"
    #
    # # PLOT (WHICH) LEGENDS AND COLORBARS?
    # ENABLE_LEGENDS_AND_COLORBARS = True  #if False, legends (of axis 0) and colobars are removed
    # ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  #if False, legend of axis 1 only contains values not included in legend of axis 0
    #
    # # STORE PLOT?
    # STORE_PLOT = False
    # STORE_AXES_SEPARATELY = True  #if True, store left and right axis to separate figures
    #
    # ####

    if plot_width == "thirdpage":
        # for 0.33\linewidth or 0.25\linewidth plots (one-column format):
        params = {'figure.titlesize': 16,  # Fontsize of the figure title
                  'axes.titlesize': 16,  # Fontsize of the axes title
                  'axes.labelsize': 14,  # Fontsize of the x and y labels
                  'xtick.labelsize': 14,  # Fontsize of the xtick labels
                  'ytick.labelsize': 14,  # Fontsize of the ytick labels
                  'legend.fontsize': 12,  # Fontsize of the legend entries
                  'legend.title_fontsize': 12,  # Fontsize of the legend title
                  # 'font.size': 12  #Default fontsize
                  }
    elif plot_width == "halfpage":
        # for 0.5\linewidth plots (one-column format):
        params = {'figure.titlesize': 12, #Fontsize of the figure title
                  'axes.titlesize': 12, #Fontsize of the axes title
                  'axes.labelsize': 10, #Fontsize of the x and y labels
                  'xtick.labelsize': 10, #Fontsize of the xtick labels
                  'ytick.labelsize': 10, #Fontsize of the ytick labels
                  'legend.fontsize': 9, #Fontsize of the legend entries
                  'legend.title_fontsize': 9, #Fontsize of the legend title
                  #'font.size': 12  #Default fontsize
                 }
    else:
        params = {}
    plt.rcParams.update(params)

    # plt.rcParams.update(plt.rcParamsDefault)

    plt.ion()
    endeffector_fig, endeffector_ax = plt.subplots(1, 2, figsize=[9, 3])

    for axis in endeffector_ax:
        axis.clear()
    # if "cbaxes" in locals():  # inset_axes
    #     try:
    #         cbaxes.remove()
    #     except ValueError:
    #         pass
    endeffector_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.9, wspace=0.26, hspace=0.2)


    if PLOT_RANGES:
        #assert REPEATED_MOVEMENTS, "ERROR: Check code dependencies..."
        assert PLOT_TIME_SERIES, "Plotting distributions (e.g., as CI or min/max-ranges) of Phasepace/Hooke profiles is not implemented yet."

    if PLOT_DEVIATION:
        assert PLOT_TIME_SERIES, "Deviation should be plotted as Time Series, not as Phasespace/Hooke plots."
        assert PLOT_ENDEFFECTOR, "Deviation can only be plotted for end-effector projections, not for (scalar) joint angle/velocity values."

    if PLOT_TRACKING_DISTANCE:
        assert PLOT_TIME_SERIES, "Remaining distance to target can only be plotted as Time Series, not in combination with Phasespace/Hooke plots."
        assert PLOT_ENDEFFECTOR, "Remaining distance to target can only be plotted for end-effector, not for (scalar) joint angle/velocity values."
        assert not PLOT_DEVIATION, "PLOT_DEVIATION, which shows projected deviation from direct path to a (stationary) target, cannot be used in combination with PLOT_TRACKING_DISTANCE, which shows remaining distance to (stationary or non-stationary) target."
        assert not PLOT_VEL_ACC, "PLOT_VEL_ACC should be False in order to plot remaining distance to target."

        assert STORE_AXES_SEPARATELY, "Remaining distance plotted on left axis should be stored using a separate filename."

    if REPEATED_MOVEMENTS:
        assert TARGET_IDS is None, "Target IDs do not exist in 'repeated movements' datasets."
        #assert PLOT_TIME_SERIES, "Plotting distributions (e.g., as CI or min/max-ranges) of Phasepace/Hooke profiles is not implemented yet."
        assert not SHOW_STUDY, "Cannot plot data from user study together with simulation data for different radii."

    #if SHOW_MINJERK:
    #    assert not SHOW_STUDY, "Cannot currently plot both minjerk and user study trajectories as reference."

    if SHOW_STUDY:
        assert isinstance(trajectories_SIMULATION, TrajectoryData_MPC) or \
        (isinstance(trajectories_SIMULATION, list) and all([isinstance(_t, TrajectoryData_MPC) for _t in trajectories_SIMULATION])) or \
        "iso-" in filename

    if STUDY_ONLY:
        assert SHOW_STUDY

    if len(AGGREGATION_VARS) > 0:
        # This assert can be disabled.
        assert not NORMALIZE_TIME, f"NORMALIZE_TIME has no effect on the computation of distributions (mean, variability, etc.)!"

        assert not SHOW_MINJERK, "MinJerk Distributions are currently not available."

    AGGREGATION_VARS_STUDY = [agg_var for agg_var in AGGREGATION_VARS if agg_var in ["targetoccurrence", "all"]] #+ ["all"]


    # Specify plot filename
    pprint_range_or_intlist = lambda x: (f"{x.start}-{x.stop-1}" if x.start < x.stop-1 else f"{x.start}" if x.start == x.stop-1 else "ERROR") if isinstance(x, range) else ((f"{min(x)}-{max(x)}" if min(x) != max(x) else f"{min(x)}") if set(range(min(x), max(x) + 1)) == set(x) else ",".join([str(i) for i in sorted(set(x))])) if isinstance(x, list) or (isinstance(x, np.ndarray) and x.ndim == 1) else f"0-{x-1}" if isinstance(x, int) else "ERROR"

    _plot_EPISODE_ID = f"E{pprint_range_or_intlist(EPISODE_IDS)}_" if EPISODE_IDS is not None else ""
    _plot_MOVEMENT_ID = f"M{pprint_range_or_intlist(MOVEMENT_IDS)}_" if MOVEMENT_IDS is not None else ""
    _plot_RADIUS_ID = f"R{pprint_range_or_intlist(RADIUS_IDS)}_" if RADIUS_IDS is not None else ""

    _plot_r1_FIXED = f"r1_{float(r1_FIXED):.8e}_" if r1_FIXED is not None else ""  #only used if PLOTTING_ENV == "MPC-costweights"
    _plot_r2_FIXED = f"r2_{float(r2_FIXED):.8e}_" if r2_FIXED is not None else ""  #only used if PLOTTING_ENV == "MPC-costweights"

    _plot_JOINT_ID = "EE_" if PLOT_ENDEFFECTOR else f"{independent_joints[JOINT_ID]}_"
    _plot_DEVIATION = "dev_" if PLOT_DEVIATION else ""
    _plot_PHASESPACE = "PS_" if not PLOT_TIME_SERIES else ""
    _plot_VEL_ACC = "velacc_" if PLOT_VEL_ACC and PLOT_TIME_SERIES else ""

    _plot_TARGET_ID = f"T{pprint_range_or_intlist(TARGET_IDS)}_" if TARGET_IDS is not None else ""
    _plot_METATRIAL_ID = f"Tr{pprint_range_or_intlist(TRIAL_IDS)}_" if TRIAL_IDS is not None else f"MT{pprint_range_or_intlist(META_IDS)}_" if META_IDS is not None else f"MN{pprint_range_or_intlist(N_MOVS)}_" if N_MOVS is not None else ""
    _plot_AGGREGATE = f"AGG-{'-'.join(AGGREGATION_VARS)}_" if len(AGGREGATION_VARS) > 0 else ""
    _plot_RANGE = f"{CONF_LEVEL.replace('/', '') if isinstance(CONF_LEVEL, str) else str(CONF_LEVEL) + 'CI'}_" if PLOT_RANGES else ""
    _plot_EFFECTIVE_PROJECTION_PATH = "eP_" if EFFECTIVE_PROJECTION_PATH else ""
    _plot_USE_TARGETBOUND_AS_DIST = "TB_" if USE_TARGETBOUND_AS_DIST == True else ""
    if PLOTTING_ENV == "RL-UIB":
        plot_filename_ID = f"UIB/{filename}/{_plot_EPISODE_ID}{_plot_MOVEMENT_ID}{_plot_RADIUS_ID}{_plot_JOINT_ID}{_plot_DEVIATION}{_plot_PHASESPACE}{_plot_VEL_ACC}{_plot_TARGET_ID}{_plot_METATRIAL_ID}{_plot_AGGREGATE}{_plot_RANGE}{_plot_EFFECTIVE_PROJECTION_PATH}{_plot_USE_TARGETBOUND_AS_DIST}"
    elif PLOTTING_ENV == "MPC-costs":
        plot_filename_ID = f"MPC/{common_simulation_subdir}/{USER_ID}/{TASK_CONDITION}/{_plot_JOINT_ID}{_plot_DEVIATION}{_plot_PHASESPACE}{_plot_VEL_ACC}{_plot_TARGET_ID}{_plot_METATRIAL_ID}{_plot_AGGREGATE}{_plot_RANGE}{_plot_EFFECTIVE_PROJECTION_PATH}{_plot_USE_TARGETBOUND_AS_DIST}"
    elif PLOTTING_ENV == "MPC-costweights":
        plot_filename_ID = f"MPC/{common_simulation_subdir}/{USER_ID}/{TASK_CONDITION}/{_plot_r1_FIXED}{_plot_r2_FIXED}{_plot_JOINT_ID}{_plot_DEVIATION}{_plot_PHASESPACE}{_plot_VEL_ACC}{_plot_TARGET_ID}{_plot_METATRIAL_ID}{_plot_AGGREGATE}{_plot_RANGE}{_plot_EFFECTIVE_PROJECTION_PATH}{_plot_USE_TARGETBOUND_AS_DIST}"
    elif PLOTTING_ENV == "MPC-horizons":
        plot_filename_ID = f"MPC/{common_simulation_subdir}/{USER_ID}/{TASK_CONDITION}/{_plot_JOINT_ID}{_plot_DEVIATION}{_plot_PHASESPACE}{_plot_VEL_ACC}{_plot_TARGET_ID}{_plot_METATRIAL_ID}{_plot_AGGREGATE}{_plot_RANGE}{_plot_EFFECTIVE_PROJECTION_PATH}{_plot_USE_TARGETBOUND_AS_DIST}"
    elif PLOTTING_ENV == "MPC-taskconditions":
        plot_filename_ID = f"MPC/{common_simulation_subdir}/{USER_ID}/{common_taskcondition_subdir}/{'STUDY_' if STUDY_ONLY else ''}{_plot_JOINT_ID}{_plot_DEVIATION}{_plot_PHASESPACE}{_plot_VEL_ACC}{_plot_TARGET_ID}{_plot_METATRIAL_ID}{_plot_AGGREGATE}{_plot_RANGE}{_plot_EFFECTIVE_PROJECTION_PATH}{_plot_USE_TARGETBOUND_AS_DIST}"
    elif PLOTTING_ENV == "MPC-userstudy" or PLOTTING_ENV == "MPC-userstudy-baselineonly" or PLOTTING_ENV == "MPC-simvsuser-colored":
        plot_filename_ID = f"MPC/{common_simulation_subdir}/{USER_ID_FIXED}_FIXED/{TASK_CONDITION}/{_plot_JOINT_ID}{_plot_DEVIATION}{_plot_PHASESPACE}{_plot_VEL_ACC}{_plot_TARGET_ID}{_plot_METATRIAL_ID}{_plot_AGGREGATE}{_plot_RANGE}{_plot_EFFECTIVE_PROJECTION_PATH}{_plot_USE_TARGETBOUND_AS_DIST}"
    elif PLOTTING_ENV == "MPC-betweenuser":
        plot_filename_ID = f"MPC/{common_simulation_subdir}/{user_list_string}/{TASK_CONDITION}/{_plot_JOINT_ID}{_plot_DEVIATION}{_plot_PHASESPACE}{_plot_VEL_ACC}{_plot_TARGET_ID}{_plot_METATRIAL_ID}{_plot_AGGREGATE}{_plot_RANGE}{_plot_EFFECTIVE_PROJECTION_PATH}{_plot_USE_TARGETBOUND_AS_DIST}"
    else:
        raise NotImplementedError
    if plot_filename_ID[-1] in ["-", "_"]:
        plot_filename_ID = plot_filename_ID[:-1]

    # Define trajectory sets to plot:
    if PLOTTING_ENV == "RL-UIB":
        ## RL
        if filename.endswith("max-freq-"):  #speed comparison (usually used for TrackingEnv)
            trajectories_info = []
            for i, trajectories in enumerate(trajectories_SIMULATION):
                trajectories.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS, EPISODE_IDS=EPISODE_IDS, split_trials="tracking" not in filename and "driving" not in filename)
                trajectories.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS)
                trajectories.SHOW_MINJERK = False
                trajectories_info.append((trajectories, {'linestyle': '-', 'label': 'Simulation', 'color': matplotlib.colormaps['nipy_spectral'](i/len(trajectories_SIMULATION))}, {'alpha': 0.2}, ""))
        elif "driving" in filename:  #RemoteDrivingEnv
            trajectories_SIMULATION.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS, EPISODE_IDS=EPISODE_IDS, split_trials=False, endeffector_name="car")
            trajectories_SIMULATION.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS)
            trajectories_SIMULATION.SHOW_MINJERK = False

            trajectories_SUPPLEMENTARY.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS, EPISODE_IDS=EPISODE_IDS, split_trials=False, endeffector_name="fingertip", target_name="joystick")
            trajectories_SUPPLEMENTARY.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS)
            trajectories_SUPPLEMENTARY.SHOW_MINJERK = False
            trajectories_info = [(trajectories_SUPPLEMENTARY, {'linestyle': '-', 'label': 'Fingertip', 'color': matplotlib.colormaps['nipy_spectral'](0.4)}, {'alpha': 0.2}, ""),
                                (trajectories_SIMULATION, {'linestyle': '-', 'label': 'Car', 'color': matplotlib.colormaps['nipy_spectral'](0.9)}, {'alpha': 0.2}, ""),
                                ]  #contains tuples consisting of 1. a TrajectoryData instance, 2. a dict with plotting kwargs for regular (line) plots, 3. a dict with plotting kwargs for "fill_between" plots, and 4. a code string to execute at the beginning
        else:
            trajectories_SIMULATION.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS, EPISODE_IDS=EPISODE_IDS, split_trials="tracking" not in filename and "driving" not in filename)
            trajectories_SIMULATION.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS)
            trajectories_SIMULATION.SHOW_MINJERK = False
            trajectories_info = [(trajectories_SIMULATION, {'linestyle': '-', 'label': 'Simulation', 'target_cmap': 'nipy_spectral'}, {'alpha': 0.2}, "")]  #contains tuples consisting of 1. a TrajectoryData instance, 2. a dict with plotting kwargs for regular (line) plots, 3. a dict with plotting kwargs for "fill_between" plots, and 4. a code string to execute at the beginning
        if SHOW_STUDY:
            trajectories_STUDY.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY)
            trajectories_info.append((trajectories_STUDY, {'linestyle': '--', 'label': 'Human', 'target_cmap': 'nipy_spectral'}, {'alpha': 0.2}, ""))

    #         # OPTIONAL: show all users (between-user comparison)
    #         for trajectories in trajectories_SUPPLEMENTARY:
    #             trajectories.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY)
    # #            trajectories_info.append((trajectories, {'linestyle': ':', 'label': 'Other Users', 'color': matplotlib.colormaps['turbo'](0.2), 'alpha': 0.6}, {'alpha': 0.2}, ""))
    #             trajectories_info.append((trajectories, {'linestyle': ':', 'label': 'Other Users', 'target_cmap': 'nipy_spectral', 'alpha': 0.4}, {'alpha': 0.2}, ""))
    elif PLOTTING_ENV == "MPC-costs":
        ## MPC - Comparison of cost functions
        trajectories_SIMULATION1, trajectories_SIMULATION2, trajectories_SIMULATION3 = trajectories_SIMULATION
        trajectories_SIMULATION1.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS, ignore_trainingset_trials=True)  #MPC
        trajectories_SIMULATION2.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS, ignore_trainingset_trials=True)  #MPC
        trajectories_SIMULATION3.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS, ignore_trainingset_trials=True)  #MPC
        map_conditions = lambda x: "JAC" if "accjoint" in x else "CTC" if "ctc" in x else "DC" if "cso" in x else x
        trajectories_info = [(trajectories_SIMULATION1, {'linestyle': '-', 'label': map_conditions(trajectories_SIMULATION1.SIMULATION_SUBDIR), 'color': matplotlib.colormaps['turbo'](255*1//4)}, {'alpha': 0.2, }, ""), (trajectories_SIMULATION2, {'linestyle': ':', 'label': map_conditions(trajectories_SIMULATION2.SIMULATION_SUBDIR), 'color': matplotlib.colormaps['turbo'](255*2//4)}, {'alpha': 0.2}, ""), (trajectories_SIMULATION3, {'linestyle': '-.', 'label': map_conditions(trajectories_SIMULATION3.SIMULATION_SUBDIR), 'color': matplotlib.colormaps['turbo'](255*3//4)}, {'alpha': 0.2}, "")]
        if SHOW_STUDY:
            trajectories_STUDY.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY, ignore_trainingset_trials=True)
            trajectories_info.append((trajectories_STUDY, {'linestyle': '--', 'label': 'Study', 'color': 'black'}, {'alpha': 0.2}, ""))
    elif PLOTTING_ENV == "MPC-costweights":
        ## MPC - COMPARISON OF COST WEIGHTS OF A GIVEN COST FUNCTION
        assert SHOW_STUDY == False, "Did not prepare user study data for this case."
        trajectories_WEIGHTS_selected = trajectories_SIMULATION.copy()
        if r1_FIXED is not None:
            trajectories_WEIGHTS_selected = [i for i in trajectories_WEIGHTS_selected if f"r1_{float(r1_FIXED):.8e}" in i.SIMULATION_SUBDIR]
        if r2_FIXED is not None:
            trajectories_WEIGHTS_selected = [i for i in trajectories_WEIGHTS_selected if f"r2_{float(r2_FIXED):.8e}" in i.SIMULATION_SUBDIR]
        for trajectories in trajectories_WEIGHTS_selected:
            trajectories.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY, ignore_trainingset_trials=True)
        trajectories_info = [(trajectories, {'linestyle': '-', 'label': ", ".join(filter(None, [
            f"$r_1={r1_FIXED:.2g}$" if r1_FIXED is not None else "",
            f"$r_2={r2_FIXED:.2g}$" if r2_FIXED is not None else ""])), 'color': matplotlib.colormaps['turbo'](
            i / len(trajectories_WEIGHTS_selected)), 'alpha': 0.8}, {'alpha': 0.2},
                              """colorarray = r2list if r1_FIXED is not None else r1list; scalarmappaple = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=colorarray.min(), vmax=colorarray.max()), cmap=matplotlib.colormaps['turbo']);scalarmappaple.set_array(colorarray);cbaxes = inset_axes(endeffector_ax[0], width="3%", height="40%", loc=6, bbox_to_anchor=(305 if r1_FIXED is None else 290, -20 if r1_FIXED is None else -20, 200, 300)); plt.colorbar(scalarmappaple, cax=cbaxes, orientation='vertical', format="%.2g"); cbaxes.set_ylabel(r"$r_{2}$" if r1_FIXED is not None else r"$r_{1}$", rotation=0);""" if i == 0 and (
                                          (r1_FIXED is not None) or (r2_FIXED is not None)) else "") for i, trajectories
                             in enumerate(trajectories_WEIGHTS_selected)]

    elif PLOTTING_ENV == "MPC-horizons":
        ## MPC - COMPARISON OF DIFFERENT MPC HORIZONS N
        for trajectories in trajectories_SIMULATION:
            trajectories.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY, ignore_trainingset_trials=True)
        trajectories_info = [(trajectories, {'linestyle': '-', 'label': f"N={int(trajectories.SIMULATION_SUBDIR.split('_N')[-1].split('_')[0])}", 'color': matplotlib.colormaps['turbo'](i/len(trajectories_SIMULATION)), 'alpha': 0.8}, {'alpha': 0.2}, """colorarray = r2list if r1_FIXED is not None else r1list; scalarmappaple = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=colorarray.min(), vmax=colorarray.max()), cmap=matplotlib.colormaps['turbo']);scalarmappaple.set_array(colorarray);cbaxes = inset_axes(endeffector_ax[0], width="3%", height="60%", loc=6, bbox_to_anchor=(460 if r1_FIXED is None else 430, 40 if r1_FIXED is None else 30, 200, 300)); plt.colorbar(scalarmappaple, cax=cbaxes, orientation='vertical', format="%.2g"); cbaxes.set_ylabel(r"$r_{2}$" if r1_FIXED is not None else r"$r_{1}$", rotation=0);""" if i==0 and (False) else "") for i, trajectories in enumerate(trajectories_SIMULATION)]
        if SHOW_STUDY:
            trajectories_STUDY.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY, ignore_trainingset_trials=True)
            trajectories_info.append((trajectories_STUDY, {'linestyle': '--', 'label': 'Study', 'color': 'black'}, {'alpha': 0.2}, ""))
    elif PLOTTING_ENV == "MPC-taskconditions":
        ## MPC - COMPARISON OF DIFFERENT INTERACTION TECHNIQUES/TASK CONDITIONS
        if STUDY_ONLY:
            trajectories_info = []
        else:
            trajectories_CONDITIONS_selected = trajectories_CONDITIONS.copy()
            #trajectories_CONDITIONS_selected = [i for i in trajectories_CONDITIONS_selected if i.TASK_CONDITION in TASK_CONDITION_LIST_SELECTED]
            for trajectories in trajectories_CONDITIONS_selected:
                trajectories.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY, ignore_trainingset_trials=True)
            map_task_conditions = lambda x: " ".join(["Virtual Pad" if "_Pad_" in x else "Virtual Cursor", "ID" if "_ID_" in x else "Erg." if "_Ergonomic_" in x else "???"])
            trajectories_info = [(trajectories, {'linestyle': '-', 'label': map_task_conditions(trajectories.TASK_CONDITION), 'color': matplotlib.colormaps['turbo']((i+1)/(len(trajectories_CONDITIONS_selected)+3)), 'alpha': 0.8}, {'alpha': 0.2}, "") for i, trajectories in enumerate(trajectories_CONDITIONS_selected)]
        if SHOW_STUDY:
            trajectories_USERS_selected = trajectories_SUPPLEMENTARY.copy()
            #trajectories_USERS_selected = [i for i in trajectories_USERS_selected if i.TASK_CONDITION in TASK_CONDITION_LIST_SELECTED]
            for i, trajectories in enumerate(trajectories_USERS_selected):
                trajectories.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY, ignore_trainingset_trials=True)
                trajectories_info.append((trajectories, {'linestyle': '-', 'label': f'{map_task_conditions(trajectories.TASK_CONDITION)}', 'color': matplotlib.colormaps['turbo']((i+1)/(len(trajectories_USERS_selected)+1))}, {'alpha': 0.2}, ""))
    elif PLOTTING_ENV == "MPC-userstudy" or PLOTTING_ENV == "MPC-userstudy-baselineonly":
        ## MPC - Simulation vs. User comparisons of single cost function
        trajectories_SIMULATION.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS, ignore_trainingset_trials=ignore_trainingset_trials_mpc_userstudy)  #MPC
        trajectories_info = [(trajectories_SIMULATION, {'linestyle': '-', 'label': 'Simulation', 'color': matplotlib.colormaps['turbo'](0.75)}, {'alpha': 0.2, }, "")]
        if SHOW_STUDY:
            trajectories_STUDY.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY, ignore_trainingset_trials=ignore_trainingset_trials_mpc_userstudy)
            if TARGET_IDS is None:  #between-user comparison
                trajectories_info.append((trajectories_STUDY, {'linestyle': '--', 'label': f'Baseline ({USER_ID_FIXED})', 'color': 'black'}, {'alpha': 0.2}, ""))
            else:  #within-user comparison
                trajectories_info.append((trajectories_STUDY, {'linestyle': '--', 'label': f'Study', 'color': 'black'}, {'alpha': 0.2}, ""))
        if PLOTTING_ENV == "MPC-userstudy":
            for trajectories in [i for i in trajectories_SUPPLEMENTARY if i.USER_ID != USER_ID_FIXED]:
                trajectories.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS,
                                             N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY,
                                             ignore_trainingset_trials=ignore_trainingset_trials_mpc_userstudy)  # STUDY

            if TARGET_IDS is None:  #between-user comparison
                trajectories_info.extend([(trajectories, {'linestyle': ':', 'label': 'Other Users', 'color': matplotlib.colormaps['turbo'](0.2), 'alpha': 0.8}, {'alpha': 0.2}, "") for i, trajectories in enumerate(trajectories_SUPPLEMENTARY) if trajectories.USER_ID != USER_ID_FIXED])
    elif PLOTTING_ENV == "MPC-simvsuser-colored":
        ## MPC - Simulation vs. User comparisons of single cost function
        trajectories_SIMULATION.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS,
                                                N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS,
                                                ignore_trainingset_trials=ignore_trainingset_trials_mpc_userstudy)  # MPC
        trajectories_info = [(trajectories_SIMULATION,
                              {'linestyle': '-', 'label': 'Simulation', 'target_cmap': 'nipy_spectral'},
                              {'alpha': 0.2, }, "")]
        if SHOW_STUDY:
            trajectories_STUDY.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS,
                                               N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY,
                                               ignore_trainingset_trials=ignore_trainingset_trials_mpc_userstudy)
            trajectories_info.append((trajectories_STUDY,
                                      {'linestyle': '--', 'label': f'Study', 'target_cmap': 'nipy_spectral'},
                                      {'alpha': 0.2}, ""))
    elif PLOTTING_ENV == "MPC-betweenuser":
        ## MPC - Between-user comparisons: A simulation class incorporating different users vs. a study class incorporating different users
        trajectories_SIMULATION.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS, ignore_trainingset_trials=True)  #MPC
        trajectories_info = [(trajectories_SIMULATION, {'linestyle': '-', 'label': 'Simulation', 'color': matplotlib.colormaps['turbo'](0.75)}, {'alpha': 0.2, }, "")]
        if SHOW_STUDY:
            trajectories_STUDY.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS_STUDY, ignore_trainingset_trials=True)
            trajectories_info.append((trajectories_STUDY, {'linestyle': '--', 'label': f'Study', 'color': matplotlib.colormaps['turbo'](0.2)}, {'alpha': 0.2}, ""))

    else:
        raise NotImplementedError

    if SHOW_MINJERK:
        if SHOW_STUDY:  #use user data as reference trajectory for initial and final values of MinJerk
            trajectories_STUDY.compute_minjerk(MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS, targetbound_as_target=(USE_TARGETBOUND_AS_DIST == True or USE_TARGETBOUND_AS_DIST == "MinJerk-only"))
            trajectories_info.append((trajectories_STUDY, {'linestyle': ':', 'label': 'MinJerk', 'color': 'black'}, {'alpha': 0.2}, "trajectories.SHOW_MINJERK = True"))
        else:
            trajectories_SIMULATION.compute_minjerk(MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS, targetbound_as_target=(USE_TARGETBOUND_AS_DIST == True or USE_TARGETBOUND_AS_DIST == "MinJerk-only"))
            trajectories_info.append((trajectories_SIMULATION, {**{'linestyle': ':', 'label': 'MinJerk'}, **({'target_cmap': 'nipy_spectral'} if PLOTTING_ENV == "RL-UIB" else {})}, {'alpha': 0.2}, "trajectories.SHOW_MINJERK = True"))

    ## ANALYSIS OF POINTING TASKS -> Use trajectories projections onto direct path between initial and (fixed) target position
    if PLOT_DEVIATION:

        endeffector_ax[0].set_title("Position Deviation from Direct Path")
        endeffector_ax[0].set_xlabel("Time " + ('[normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
        endeffector_ax[0].set_ylabel("Position Deviation $\left(m\right)$")

        endeffector_ax[1].set_title("Velocity Deviation from Direct Path")
        endeffector_ax[1].set_xlabel("Time " + (' [normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
        endeffector_ax[1].set_ylabel("Velocity Deviation $\left(\frac{m}{s}\right)$")
    else:
        if PLOT_ENDEFFECTOR:
            if PLOT_TIME_SERIES:
                if PLOT_VEL_ACC:
                    endeffector_ax[0].set_title("Projected Velocity Time Series")
                    endeffector_ax[0].set_xlabel("Time " + ('[normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                    endeffector_ax[0].set_ylabel(r"Velocity  $\left(\frac{m}{s}\right)$")
                    #endeffector_ax[0].set_ylabel(r"Velocity  $\left(\frac{1}{s}\right)$ [normalized]")

                    endeffector_ax[1].set_title("Projected Acceleration Time Series")
                    endeffector_ax[1].set_xlabel("Time " + ('[normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                    endeffector_ax[1].set_ylabel(r"Acceleration $\left(\frac{m}{s^2}\right)$")
                    #endeffector_ax[1].set_ylabel(r"Acceleration $\left(\frac{1}{s^2}\right)$ [normalized]")
                else:
                    if PLOT_TRACKING_DISTANCE:
                        endeffector_ax[0].set_title(f"Distance to {'Joystick/' if PLOTTING_ENV == 'RL-UIB' and 'driving' in filename else ''}Target")
                        endeffector_ax[0].set_xlabel("Time " + ('[normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                        endeffector_ax[0].set_ylabel(r"Distance $\left(m\right)$")
                    else:
                        endeffector_ax[0].set_title("Projected Position Time Series")
                        endeffector_ax[0].set_xlabel("Time " + ('[normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                        endeffector_ax[0].set_ylabel("Position [normalized]")

                    endeffector_ax[1].set_title("Projected Velocity Time Series")
                    endeffector_ax[1].set_xlabel("Time " + ('[normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                    endeffector_ax[1].set_ylabel(r"Velocity $\left(\frac{m}{s}\right)$")
                    #endeffector_ax[1].set_ylabel(r"Velocity $\left(\frac{1}{s}\right)$ [normalized]")
            else:
                endeffector_ax[0].set_title("Projected Phasespace Plot")
                endeffector_ax[0].set_xlabel("Position [normalized]")
                endeffector_ax[0].set_ylabel(r"Velocity $\left(\frac{m}{s}\right)$")
                #endeffector_ax[0].set_ylabel(r"Velocity $\left(\frac{1}{s}\right)$ [normalized]")

                endeffector_ax[1].set_title("Projected Hooke Plot")
                endeffector_ax[1].set_xlabel("Position [normalized]")
                endeffector_ax[1].set_ylabel(r"Acceleration $\left(\frac{m}{s^2}\right)$")
                #endeffector_ax[1].set_ylabel(r"Acceleration $\left(\frac{1}{s^2}\right)$ [normalized]")
        else:
            if PLOT_TIME_SERIES:
                if PLOT_VEL_ACC:
                    endeffector_ax[0].set_title(f"Joint Velocity – {independent_joints[JOINT_ID]}")
                    endeffector_ax[0].set_xlabel("Time " + ('[normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                    endeffector_ax[0].set_ylabel(r"Joint Velocity $\left(\frac{rad}{s}\right)$")

                    endeffector_ax[1].set_title(f"Joint Velocity – {independent_joints[JOINT_ID]}")
                    endeffector_ax[1].set_xlabel("Time " + (' [normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                    endeffector_ax[1].set_ylabel(r"Joint Acceleration $\left(\frac{rad}{s^2}\right)$")
                else:
                    endeffector_ax[0].set_title(f"Joint Angle – {independent_joints[JOINT_ID]}")
                    endeffector_ax[0].set_xlabel("Time " + ('[normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                    endeffector_ax[0].set_ylabel(r"Joint Angle $\left(rad\right)$")

                    endeffector_ax[1].set_title(f"Joint Velocity – {independent_joints[JOINT_ID]}")
                    endeffector_ax[1].set_xlabel("Time " + (' [normalized]' if NORMALIZE_TIME else r'$\left(s\right)$'))
                    endeffector_ax[1].set_ylabel(r"Joint Velocity $\left(\frac{rad}{s}\right)$")
            else:
                endeffector_ax[0].set_title(f"{independent_joints[JOINT_ID]} – Phasespace Plot")
                endeffector_ax[0].set_xlabel(r"Joint Angle $\left(rad\right)$")
                endeffector_ax[0].set_ylabel(r"Joint Velocity $\left(\frac{rad}{s}\right)$")

                endeffector_ax[1].set_title(f"{independent_joints[JOINT_ID]} – Hooke Plot")
                endeffector_ax[1].set_xlabel(r"Joint Angle $\left(rad\right)$")
                endeffector_ax[1].set_ylabel(r"Joint Acceleration $\left(\frac{rad}{s^2}\right)$")

        if PLOTTING_ENV == "MPC-userstudy-baselineonly":
            map_task_conditions = lambda x: " ".join(["Virtual Pad" if "_Pad_" in x else "Virtual Cursor", "ID" if "_ID_" in x else "Erg." if "_Ergonomic_" in x else "???"])
            endeffector_ax[0].set_title(map_task_conditions(TASK_CONDITION))
            endeffector_ax[1].set_title(map_task_conditions(TASK_CONDITION))

    if (PLOTTING_ENV == "RL-UIB") and (filename.endswith("max-freq-") or "driving" in filename):
        endeffector_ax[0].set_yscale("log")

    methods_handles = []

    for trajectories, trajectory_plotting_kwargs, range_plotting_kwargs, code_to_exec in trajectories_info:
        exec(code_to_exec)  #necessary to get MinJerk trajectories from RL-Simulation class

        #methods_handles.append(Line2D([0], [0], color="black", **{k: v for k,v in trajectory_plotting_kwargs.items() if k != "color"}))
        methods_handles.append(Line2D([0], [0], **dict({"color": "black"}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["target_cmap"]})))

        endeffector_ax[0].set_prop_cycle(None)
        endeffector_ax[1].set_prop_cycle(None)

        for episode_index_current, (last_idx_hlp, current_idx_hlp, next_idx_hlp) in enumerate(trajectories.selected_movements_indices):
            if isinstance(last_idx_hlp, list) and len(last_idx_hlp) > 1:  #if len(AGGREGATION_VARS) > 0

                trajectories.compute_statistics(episode_index_current, effective_projection_path=EFFECTIVE_PROJECTION_PATH, targetbound_as_target=trajectories.SHOW_MINJERK if USE_TARGETBOUND_AS_DIST == "MinJerk-only" else USE_TARGETBOUND_AS_DIST, compute_deviation=PLOT_DEVIATION, normalize_time=NORMALIZE_TIME, use_joint_data_only="all" in AGGREGATION_VARS or isinstance(trajectories, TrajectoryData_MultipleInstances))

                # if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS):
                #     print(2*trajectories.target_radius_mean[0], trajectories.time_series_extended[-1], np.max(trajectories.projected_trajectories_vel_mean))

                if PLOT_TIME_SERIES:
                    ### POSITION (or VELOCITY) PLOT
                    latest_plot, = endeffector_ax[0].plot(trajectories.time_series_extended, trajectories.distance_to_target_mean if PLOT_TRACKING_DISTANCE else (trajectories.projected_trajectories_vel_mean if PLOT_VEL_ACC else trajectories.projected_trajectories_pos_mean) if PLOT_ENDEFFECTOR else (trajectories.qvel_series_mean[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qpos_series_mean[:, JOINT_ID]), label=f'W={2*trajectories.target_radius_mean[0]:.2g}' if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) else f'T{trajectories.target_idx_mean[0]}' if not np.isnan(trajectories.target_idx_mean).all() else f"{trajectories.max_frequency}Hz" if hasattr(trajectories, "max_frequency") else None, **{**{'color': matplotlib.colormaps[trajectory_plotting_kwargs["target_cmap"]](((-0.04 + 4*trajectories.target_idx_mean[0]/13)%1) + False*((trajectories.target_idx_mean[0]+1)/(2*13) + (0.5 if trajectories.target_idx_mean[0]%2 else 0))) if ("target_cmap" in trajectory_plotting_kwargs) and (not ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS)) and (not np.isnan(trajectories.target_idx_mean).all()) else None}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["label", "target_cmap"]}})

                    # if isinstance(trajectories, TrajectoryData_MultipleInstances) and isinstance(trajectories.trajectories[0], TrajectoryData_STUDY):
                    #     PLOT_RANGES = True
                    # #print(2*trajectories.target_radius_mean[0], trajectories.time_series_extended[-1], trajectories.projected_trajectories_vel_mean.max())

                    if PLOT_RANGES:
                        if CONF_LEVEL == "min/max":
                            # add min/max bounds
                            endeffector_ax[0].fill_between(trajectories.time_series_extended, trajectories.distance_to_target_min if PLOT_TRACKING_DISTANCE else (trajectories.projected_trajectories_vel_min if PLOT_VEL_ACC else trajectories.projected_trajectories_pos_min) if PLOT_ENDEFFECTOR else (trajectories.qvel_series_min[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qpos_series_min[:, JOINT_ID]), trajectories.distance_to_target_max if PLOT_TRACKING_DISTANCE else (trajectories.projected_trajectories_vel_max if PLOT_VEL_ACC else trajectories.projected_trajectories_pos_max) if PLOT_ENDEFFECTOR else (trajectories.qvel_series_max[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qpos_series_max[:, JOINT_ID]), color=latest_plot.get_color(), **{k: v for k,v in range_plotting_kwargs.items() if k != "color"})
                        else:
                            # add confidence interval bounds
                            if PLOT_DEVIATION:
                                raise NotImplementedError("ERROR: Not sure if it makes sense to plot difference between projected standard deviation and direct path...")
                            if PLOT_VEL_ACC:
                                projected_or_joint_trajectories_pos_or_vel_std =  np.sqrt(trajectories.projected_trajectories_vel_cov if PLOT_ENDEFFECTOR else trajectories.qvel_series_cov[:, JOINT_ID])
                            else:
                                projected_or_joint_trajectories_pos_or_vel_std =  np.sqrt(trajectories.distance_to_target_cov if PLOT_TRACKING_DISTANCE else trajectories.projected_trajectories_pos_cov if PLOT_ENDEFFECTOR else trajectories.qpos_series_cov[:, JOINT_ID])
                            endeffector_ax[0].fill_between(trajectories.time_series_extended, ((trajectories.distance_to_target_mean if PLOT_TRACKING_DISTANCE else trajectories.projected_trajectories_vel_mean if PLOT_VEL_ACC else trajectories.projected_trajectories_pos_mean) if PLOT_ENDEFFECTOR else (trajectories.qvel_series_mean[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qpos_series_mean[:, JOINT_ID])) - stats.norm.ppf((CONF_LEVEL + 1)/2) * projected_or_joint_trajectories_pos_or_vel_std, ((trajectories.distance_to_target_mean if PLOT_TRACKING_DISTANCE else trajectories.projected_trajectories_vel_mean if PLOT_VEL_ACC else trajectories.projected_trajectories_pos_mean) if PLOT_ENDEFFECTOR else (trajectories.qvel_series_mean[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qpos_series_mean[:, JOINT_ID])) + stats.norm.ppf((CONF_LEVEL + 1)/2) * projected_or_joint_trajectories_pos_or_vel_std, color=latest_plot.get_color(), **{k: v for k,v in range_plotting_kwargs.items() if k != "color"})

                        # display lower bound of target area (i.e., target center - target radius) in projected position plot or distance plot
                        if (not PLOT_DEVIATION) and PLOT_ENDEFFECTOR:
                            if PLOT_TRACKING_DISTANCE:
                                assert len(np.unique(trajectories.target_radius_mean)) == 1
                                if len(trajectories.selected_movements_indices) == 1 and len(trajectories_info) == 1:  #only one trajectory is plotted
                                    print("A")
                                    endeffector_ax[0].axhline(trajectories.target_radius_mean[0], linestyle='--', color="black", alpha=1, label="Target Boundary")
                                elif len(np.unique([i.target_radius_series for i, _, _, _ in trajectories_info])) == 1:  #all trajectories have same target radius
                                    endeffector_ax[0].axhline(trajectories.target_radius_mean[0], linestyle='--', color="grey", alpha=1) #, label="Target Boundary")
                                else:
                                    endeffector_ax[0].axhline(trajectories.target_radius_mean[0], linestyle='--', color=latest_plot.get_color(), alpha=1)
                            else:
                                #TODO: modify init_val, final_val!
                                target_radius_normalized = trajectories.target_radius_mean[0]/np.linalg.norm(trajectories.final_val-trajectories.init_val)   #relative (i.e., normalized) target radius
                                target_area_boundary_projected = 1 - target_radius_normalized
                                endeffector_ax[0].axhline(target_area_boundary_projected, linestyle='--', color=latest_plot.get_color(), alpha=.2)

                    ### VELOCITY (or ACCELERATION) PLOT
                    latest_plot, = endeffector_ax[1].plot(trajectories.time_series_extended, (trajectories.projected_trajectories_acc_mean if PLOT_VEL_ACC else trajectories.projected_trajectories_vel_mean) if PLOT_ENDEFFECTOR else (trajectories.qacc_series_mean[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qvel_series_mean[:, JOINT_ID]), label=f'W={2*trajectories.target_radius_mean[0]:.2g}' if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) else f'T{trajectories.target_idx_mean[0]}' if not np.isnan(trajectories.target_idx_mean).all() else None, **{**{'color': matplotlib.colormaps[trajectory_plotting_kwargs["target_cmap"]](((-0.04 + 4*trajectories.target_idx_mean[0]/13)%1) + False*((trajectories.target_idx_mean[0]+1)/(2*13) + (0.5 if trajectories.target_idx_mean[0]%2 else 0))) if ("target_cmap" in trajectory_plotting_kwargs) and (not ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS)) and (not np.isnan(trajectories.target_idx_mean).all()) else None}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["label", "target_cmap"]}})

                    if PLOT_RANGES:
                        if CONF_LEVEL == "min/max":
                            # add min/max bounds
                            endeffector_ax[1].fill_between(trajectories.time_series_extended, (trajectories.projected_trajectories_acc_min if PLOT_VEL_ACC else trajectories.projected_trajectories_vel_min) if PLOT_ENDEFFECTOR else (trajectories.qacc_series_min[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qvel_series_min[:, JOINT_ID]), (trajectories.projected_trajectories_acc_max if PLOT_VEL_ACC else trajectories.projected_trajectories_vel_max) if PLOT_ENDEFFECTOR else (trajectories.qacc_series_max[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qvel_series_max[:, JOINT_ID]), color=latest_plot.get_color(), **{k: v for k,v in range_plotting_kwargs.items() if k != "color"})
                        else:
                            # add confidence interval bounds
                            if PLOT_VEL_ACC:
                                projected_or_joint_trajectories_vel_or_acc_std =  np.sqrt(trajectories.projected_trajectories_acc_cov if PLOT_ENDEFFECTOR else trajectories.qacc_series_cov[:, JOINT_ID])
                            else:
                                projected_or_joint_trajectories_vel_or_acc_std =  np.sqrt(trajectories.projected_trajectories_vel_cov if PLOT_ENDEFFECTOR else trajectories.qvel_series_cov[:, JOINT_ID])
                            endeffector_ax[1].fill_between(trajectories.time_series_extended, ((trajectories.projected_trajectories_acc_mean if PLOT_VEL_ACC else trajectories.projected_trajectories_vel_mean) if PLOT_ENDEFFECTOR else (trajectories.qacc_series_mean[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qvel_series_mean[:, JOINT_ID])) - stats.norm.ppf((CONF_LEVEL + 1)/2) * projected_or_joint_trajectories_vel_or_acc_std, ((trajectories.projected_trajectories_acc_mean if PLOT_VEL_ACC else trajectories.projected_trajectories_vel_mean) if PLOT_ENDEFFECTOR else (trajectories.qacc_series_mean[:, JOINT_ID] if PLOT_VEL_ACC else trajectories.qvel_series_mean[:, JOINT_ID])) + stats.norm.ppf((CONF_LEVEL + 1)/2) * projected_or_joint_trajectories_vel_or_acc_std, color=latest_plot.get_color(), **{k: v for k,v in range_plotting_kwargs.items() if k != "color"})
                else:
                    ### PHASESPACE PLOT
                    latest_plot, = endeffector_ax[0].plot(trajectories.projected_trajectories_pos_mean if PLOT_ENDEFFECTOR else trajectories.qpos_series_mean[:, JOINT_ID], trajectories.projected_trajectories_vel_mean if PLOT_ENDEFFECTOR else trajectories.qvel_series_mean[:, JOINT_ID], label=f'W={2*trajectories.target_radius_mean[0]:.2g}' if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) else f'T{trajectories.target_idx_mean[0]}' if not np.isnan(trajectories.target_idx_mean).all() else None, **{**{'color': matplotlib.colormaps[trajectory_plotting_kwargs["target_cmap"]](((-0.04 + 4*trajectories.target_idx_mean[0]/13)%1) + False*((trajectories.target_idx_mean[0]+1)/(2*13) + (0.5 if trajectories.target_idx_mean[0]%2 else 0))) if ("target_cmap" in trajectory_plotting_kwargs) and (not ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS)) and (not np.isnan(trajectories.target_idx_mean).all()) else None}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["label", "target_cmap"]}})

                    if PLOT_RANGES:
                        raise NotImplementedError("""Can show distribution of aggregated trials only if PLOT_TIME_SERIES == True.""")

                    ### HOOKE PLOT
                    latest_plot, = endeffector_ax[1].plot(trajectories.projected_trajectories_pos_mean if PLOT_ENDEFFECTOR else trajectories.qpos_series_mean[:, JOINT_ID], trajectories.projected_trajectories_acc_mean if PLOT_ENDEFFECTOR else trajectories.qacc_series_mean[:, JOINT_ID], label=f'W={2*trajectories.target_radius_mean[0]:.2g}' if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) else f'T{trajectories.target_idx_mean[0]}' if not np.isnan(trajectories.target_idx_mean).all() else None, **{**{'color': matplotlib.colormaps[trajectory_plotting_kwargs["target_cmap"]](((-0.04 + 4*trajectories.target_idx_mean[0]/13)%1) + False*((trajectories.target_idx_mean[0]+1)/(2*13) + (0.5 if trajectories.target_idx_mean[0]%2 else 0))) if ("target_cmap" in trajectory_plotting_kwargs) and (not ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS)) and (not np.isnan(trajectories.target_idx_mean).all()) else None}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["label", "target_cmap"]}})

                    if PLOT_RANGES:
                        raise NotImplementedError("""Can show distribution of aggregated trials only if PLOT_TIME_SERIES == True.""")

            else:
                assert not PLOT_RANGES, "ERROR: Cannot plot ranges, as no trials were aggregated!"

                trajectories.compute_trial(episode_index_current, effective_projection_path=EFFECTIVE_PROJECTION_PATH, targetbound_as_target=trajectories.SHOW_MINJERK if USE_TARGETBOUND_AS_DIST == "MinJerk-only" else USE_TARGETBOUND_AS_DIST, compute_deviation=PLOT_DEVIATION, normalize_time=NORMALIZE_TIME)

                projected_or_joint_trajectory_pos = trajectories.projected_trajectories_pos_trial if PLOT_ENDEFFECTOR else trajectories.qpos_series_trial[:, JOINT_ID]
                projected_or_joint_trajectory_vel = trajectories.projected_trajectories_vel_trial if PLOT_ENDEFFECTOR else trajectories.qvel_series_trial[:, JOINT_ID]
                projected_or_joint_trajectory_acc = trajectories.projected_trajectories_acc_trial if PLOT_ENDEFFECTOR else trajectories.qacc_series_trial[:, JOINT_ID]

                ## for interchangeability between projection and covariance, see https://math.stackexchange.com/a/2576783
                if PLOT_TIME_SERIES:
                    ### POSITION (or VELOCITY) PLOT
                    latest_plot, = endeffector_ax[0].plot(trajectories.time_series_trial, trajectories.distance_to_target_trial if PLOT_TRACKING_DISTANCE else (projected_or_joint_trajectory_vel if PLOT_VEL_ACC else projected_or_joint_trajectory_pos), label=f'W={2*trajectories.target_radius_series_trial[0]:.2g}' if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) else f'T{trajectories.target_idx_series_trial[0]}' if not np.isnan(trajectories.target_idx_series).all() else None, **{**{'color': matplotlib.colormaps[trajectory_plotting_kwargs["target_cmap"]](((-0.04 + 4*trajectories.target_idx_series_trial[0]/13)%1) + False*((trajectories.target_idx_series_trial[0]+1)/(2*13) + (0.5 if trajectories.target_idx_series_trial[0]%2 else 0))) if ("target_cmap" in trajectory_plotting_kwargs) and (not ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS)) and (not np.isnan(trajectories.target_idx_series).all()) else None}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["label", "target_cmap"]}})

                    ### VELOCITY (or ACCELERATION) PLOT
                    latest_plot, = endeffector_ax[1].plot(trajectories.time_series_trial, projected_or_joint_trajectory_acc if PLOT_VEL_ACC else projected_or_joint_trajectory_vel, label=f'W={2*trajectories.target_radius_series_trial[0]:.2g}' if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) else f'T{trajectories.target_idx_series_trial[0]}' if not np.isnan(trajectories.target_idx_series).all() else None, **{**{'color': matplotlib.colormaps[trajectory_plotting_kwargs["target_cmap"]](((-0.04 + 4*trajectories.target_idx_series_trial[0]/13)%1) + False*((trajectories.target_idx_series_trial[0]+1)/(2*13) + (0.5 if trajectories.target_idx_series_trial[0]%2 else 0))) if ("target_cmap" in trajectory_plotting_kwargs) and (not ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS)) and (not np.isnan(trajectories.target_idx_series).all()) else None}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["label", "target_cmap"]}})

                else:
                    ### PHASESPACE PLOT
                    latest_plot, = endeffector_ax[0].plot(projected_or_joint_trajectory_pos, projected_or_joint_trajectory_vel, label=f'W={2*trajectories.target_radius_series_trial[0]:.2g}' if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) else f'T{trajectories.target_idx_series_trial[0]}' if not np.isnan(trajectories.target_idx_series).all() else None, **{**{'color': matplotlib.colormaps[trajectory_plotting_kwargs["target_cmap"]](((-0.04 + 4*trajectories.target_idx_series_trial[0]/13)%1) + False*((trajectories.target_idx_series_trial[0]+1)/(2*13) + (0.5 if trajectories.target_idx_series_trial[0]%2 else 0))) if ("target_cmap" in trajectory_plotting_kwargs) and (not ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS)) and (not np.isnan(trajectories.target_idx_series).all()) else None}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["label", "target_cmap"]}})

                    ### HOOKE PLOT
                    latest_plot, = endeffector_ax[1].plot(projected_or_joint_trajectory_pos, projected_or_joint_trajectory_acc, label=f'W={2*trajectories.target_radius_series_trial[0]:.2g}' if ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) else f'T{trajectories.target_idx_series_trial[0]}' if not np.isnan(trajectories.target_idx_series).all() else None, **{**{'color': matplotlib.colormaps[trajectory_plotting_kwargs["target_cmap"]](((-0.04 + 4*trajectories.target_idx_series_trial[0]/13)%1) + False*((trajectories.target_idx_series_trial[0]+1)/(2*13) + (0.5 if trajectories.target_idx_series_trial[0]%2 else 0))) if ("target_cmap" in trajectory_plotting_kwargs) and (not ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS)) and (not np.isnan(trajectories.target_idx_series).all()) else None}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["label", "target_cmap"]}})

    #reset to show regular (study) trajectories first
    for trajectories_info_current in trajectories_info:
        trajectories_info_current[0].SHOW_MINJERK = False

    #if SHOW_DIFFERENT_TARGET_SIZES or ("radius" not in AGGREGATION_VARS and REPEATED_MOVEMENTS) or "target_idx" in (list(trajectories_SIMULATION.data.values())[0].keys() if isinstance(trajectories_SIMULATION.data, dict) else trajectories_SIMULATION.data.columns):
        # axis 0 legend
    if len(trajectories_info) == 1 and all([len(trajectories.selected_movements_indices) == 1 for trajectories, _, _, _ in trajectories_info]):  #only one trajectory is plotted
        mean_handle = Line2D([0], [0], **dict({"color": latest_plot.get_color(), "label": "Mean"}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["target_cmap", "color", "label"]}))
        confidence_interval_handle = Patch(facecolor=latest_plot.get_color(), alpha=.2, label='Range' if CONF_LEVEL == "min/max" else f'{100*CONF_LEVEL}%-CI')
    else:
        mean_handle = Line2D([0], [0], **dict({"color": 'grey', "label": "Mean"}, **{k: v for k,v in trajectory_plotting_kwargs.items() if k not in ["target_cmap", "color", "label"]}))
        confidence_interval_handle = Patch(facecolor='grey', alpha=.2, label='Range' if CONF_LEVEL == "min/max" else f'{100*CONF_LEVEL}%-CI')

    # if SHOW_MINJERK:
    #     endeffector_ax[0].legend(handles=[solid_line, dashed_line])
    # elif SHOW_STUDY:
    #     endeffector_ax[0].legend(handles=[solid_line, dashdotted_line])
    # else:
    #     endeffector_ax[0].legend(handles=[solid_line])

    handles, labels = endeffector_ax[0].get_legend_handles_labels()

    # remove duplicate entries
    handles = [handle for idx, (handle, label) in enumerate(zip(handles, labels)) if labels.index(label) == idx]
    labels = [label for idx, label in enumerate(labels) if labels.index(label) == idx]
    methods_handles = [handle for idx, handle in enumerate(methods_handles) if [i.get_label() for i in methods_handles].index(handle.get_label()) == idx]

    # if (REPEATED_MOVEMENTS and ("radius" not in AGGREGATION_VARS)):  #ignore handle for dashed MinJerk trajectory, as this is not shown in this case
    #     endeffector_ax[0].legend(handles=handles)
    # elif SHOW_MINJERK:
    #     endeffector_ax[0].legend(handles=[solid_line, dashed_line]+handles)
    # elif SHOW_STUDY:
    #     endeffector_ax[0].legend(handles=[solid_line, dashdotted_line]+handles)
    # else:
    #     endeffector_ax[0].legend(handles=handles)
    if (len(methods_handles) == 1) and (PLOTTING_ENV != "MPC-costweights" or COST_FUNCTION == "dist"):
        methods_handles = []
    elif "driving" in filename and (PLOTTING_ENV != "MPC-costweights" or COST_FUNCTION == "dist"):
        handles.extend(methods_handles)
        methods_handles = []
    if PLOTTING_ENV == "MPC-userstudy-baselineonly":
        pass
    elif PLOT_RANGES: #if len(AGGREGATION_VARS) > 0:  # if len(last_idx_hlp) > 1
        handles.insert(0, confidence_interval_handle)
        if len(methods_handles) == 0:
            handles.insert(0, mean_handle)
    if PLOTTING_ENV == "MPC-taskconditions":
        handles = []
    if "color" in trajectory_plotting_kwargs and len(handles) == 1:
        legend_handles = methods_handles
    else:
        legend_handles = methods_handles+handles

    if len(legend_handles) > 0:
        endeffector_ax[0].legend(handles=legend_handles, ncol=(len(trajectories_SIMULATION) + 2)//2 if (PLOTTING_ENV == "RL-UIB") and filename.endswith("max-freq-") else 2 if "driving" in filename and PLOTTING_ENV == "RL-UIB" else 1)

    # remove axis 0 legend (and colorbars) completely
    if not ENABLE_LEGENDS_AND_COLORBARS:
        if endeffector_ax[0].get_legend() is not None:
            #remove legend
            endeffector_ax[0].get_legend().remove()

        #remove colorbars
        if len(endeffector_fig.axes) > len(endeffector_ax):
            for ax in endeffector_fig.axes[len(endeffector_ax):]:
                 ax.remove()

    # axis 1 legend
    handles, labels = endeffector_ax[1].get_legend_handles_labels()

    # remove duplicate entries
    handles = [handle for idx, (handle, label) in enumerate(zip(handles, labels)) if labels.index(label) == idx]
    labels = [label for idx, label in enumerate(labels) if labels.index(label) == idx]

    # remove entries from axis 1 legend
    if not ALLOW_DUPLICATES_BETWEEN_LEGENDS:
        handles_ax0, labels_ax0 = endeffector_ax[0].get_legend_handles_labels()
        handles = [handle for handle, label in zip(handles, labels) if handle not in handles_ax0 and label not in labels_ax0]
        labels = [labels for handle, label in zip(handles, labels) if handle not in handles_ax0 and label not in labels_ax0]

    if PLOT_RANGES:  #len(AGGREGATION_VARS) > 0:  # if len(last_idx_hlp) > 1
        handles.insert(0, confidence_interval_handle)
    if not (("color" in trajectory_plotting_kwargs and len(handles) == 1) or len(handles) == 0):
        endeffector_ax[1].legend(handles=handles)

    if predefined_ylim is not None:
        endeffector_ax[0].set_ylim(predefined_ylim)
        endeffector_ax[1].set_ylim(predefined_ylim)

    if STORE_PLOT:
        if STORE_AXES_SEPARATELY:
            for ax_idx_keep in range(len(endeffector_fig.axes[:len(endeffector_ax)])):
                # see https://stackoverflow.com/a/45812071
                buf = io.BytesIO()
                pickle.dump(endeffector_fig, buf)
                buf.seek(0)
                fig_hlp = pickle.load(buf)

                plot_filename_ID_extended = plot_filename_ID + ["_dist" if PLOT_TRACKING_DISTANCE else "_pos", "_vel", "_acc"][ax_idx_keep + PLOT_VEL_ACC] if PLOT_TIME_SERIES else ["_phasespace", "_hooke"][ax_idx_keep]

                # Re-position colorbar
                for idx, ax in enumerate(fig_hlp.axes[:len(endeffector_ax)]):
                    if idx != ax_idx_keep:
                        fig_hlp.delaxes(ax)
                if ax_idx_keep != 0:  #keep additional colorbar axes only for first subplot axis
                    for ax in fig_hlp.axes[1:]:
                        fig_hlp.delaxes(ax)
                fig_hlp.axes[0].set_subplotspec(matplotlib.gridspec.SubplotSpec(matplotlib.gridspec.GridSpec(1, 1), 0))
                fig_hlp.set_figwidth(4.5)
                fig_hlp.subplots_adjust(left=0.2 + 0*(0.03 if ax_idx_keep==1 else 0), right=0.95, bottom=0.2, top=0.88, wspace=0.2, hspace=0.2)
                if ax_idx_keep == 0 and len(fig_hlp.axes) > 1:
                    fig_hlp.axes[-1].set_axes_locator(InsetPosition(fig_hlp.axes[0], [0.75 if r1_FIXED is None else 0.675, 0.1 if r1_FIXED is None else 0.1, 0.03, 0.6]))  #(460 if r1_FIXED is None else 430, 40 if r1_FIXED is None else 30, 200, 300)))

                _filename = os.path.join(PLOTS_DIR, f"{plot_filename_ID_extended}.png")
                if not os.path.exists(os.path.dirname(_filename)):
                    os.makedirs(os.path.dirname(_filename))
                fig_hlp.savefig(_filename, bbox_inches='tight', dpi=300)

                plt.close(fig_hlp)
        else:
            _filename = os.path.join(PLOTS_DIR, f"{plot_filename_ID}.png")
            if not os.path.exists(os.path.dirname(_filename)):
                os.makedirs(os.path.dirname(_filename))
            endeffector_fig.savefig(_filename, bbox_inches='tight', dpi=300)
    else:
        if STORE_AXES_SEPARATELY:
            for ax_idx_keep in range(len(endeffector_ax)):
                plot_filename_ID_extended = plot_filename_ID + ["_dist" if PLOT_TRACKING_DISTANCE else "_pos", "_vel", "_acc"][ax_idx_keep + PLOT_VEL_ACC] if PLOT_TIME_SERIES else ["_phasespace", "_hooke"][ax_idx_keep]
                logging.warning(f"Plotting is disabled.\nCurrent filename: {plot_filename_ID_extended}.png")
        else:
            logging.warning(f"Plotting is disabled.\nCurrent filename: {plot_filename_ID}.png")
