import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
from uitb_evaluate.trajectory_data import PLOTS_DIR_DEFAULT


def sumstatsplot(filename, trajectories_SIMULATION,
                 REPEATED_MOVEMENTS=False,
                 MOVEMENT_IDS=None, RADIUS_IDS=None, EPISODE_IDS=None,
                 TARGET_IDS=None, TRIAL_IDS=None, META_IDS=None, N_MOVS=None, AGGREGATION_VARS=[],
                 REMOVE_FAILED=True, REMOVE_OUTLIERS=False, EFFECTIVE_PROJECTION_PATH=True,
                 USE_TARGETBOUND_AS_DIST=False,
                 DWELL_TIME=0,  # only used if USE_TARGETBOUND_AS_DIST == False
                 MAX_TIME=np.inf,  # only used if REMOVE_FAILED == True
                 PLOT_TYPE="mean_groups",  # "alldata", "boxplot", "mean_groups", "meandata", "density_ID",
                 BOXPLOT_category="ID",
                 BOXPLOT_nbins=5,
                 BOXPLOT_qbins=True,
                 # whether to use quantile-based bins (i.e., same number of samples per bin) or range-based bins (i.e., same length of each bin interval)
                 DENSITY_group_nIDbins=1,  # number of ID groups
                 DENSITY_group_IDbin_ID=0,
                 # index of ID group (between 0 and DENSITY_group_nIDbins-1) for which a density plot of movement times is created (only used if PLOT_TYPE == "density_ID")
                 DENSITY_group_nMTbins=50,
                 plot_width="sumstats",
                 STORE_PLOT=False, STORE_AXES_SEPARATELY=True, PLOTS_DIR=PLOTS_DIR_DEFAULT):
    if plot_width == "sumstats":
        params = {'legend.fontsize': 6,  # Fontsize of the legend entries
                  'legend.title_fontsize': 6,  # Fontsize of the legend title
                  # 'font.size': 12  #Default fontsize
                  }
    elif plot_width == "thirdpage":
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
        params = {'figure.titlesize': 12,  # Fontsize of the figure title
                  'axes.titlesize': 12,  # Fontsize of the axes title
                  'axes.labelsize': 10,  # Fontsize of the x and y labels
                  'xtick.labelsize': 10,  # Fontsize of the xtick labels
                  'ytick.labelsize': 10,  # Fontsize of the ytick labels
                  'legend.fontsize': 9,  # Fontsize of the legend entries
                  'legend.title_fontsize': 9,  # Fontsize of the legend title
                  # 'font.size': 12  #Default fontsize
                  }
    else:
        params = {}
    plt.rcParams.update(params)

    plt.ion()
    fittslaw_fig = plt.figure(figsize=[4, 3])
    fittslaw_ax = fittslaw_fig.gca()
    # fittslaw_fig, fittslaw_ax = plt.subplots(1, 2, figsize=[9, 3])

    fittslaw_ax.clear()
    fittslaw_fig.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.8)  # , wspace=0.26, hspace=0.2)

    _plot_DENSITY_group = f"group{DENSITY_group_IDbin_ID}" if PLOT_TYPE == "density_ID" else ""
    _plot_filename_header = "movement_time_" if PLOT_TYPE == "density_ID" else "fittslaw_"
    plot_filename_fitts = f"UIB/{filename}/{_plot_filename_header}{PLOT_TYPE}{_plot_DENSITY_group}{'_eP' if EFFECTIVE_PROJECTION_PATH else ''}{'_TB' if USE_TARGETBOUND_AS_DIST else ''}_DWELL{DWELL_TIME}{'_rF' if REMOVE_FAILED else ''}{'_rO' if REMOVE_OUTLIERS else ''}"

    if PLOT_TYPE != "boxplot":
        BOXPLOT_category = "ID (bits)"  # used for xlabel

    # Compute ID and MT pairs from dataset:
    distance_list = []
    width_list = []
    ID_list = []
    MT_list = []
    target_position_list = []
    initial_position_list = []
    failed_movements_counter = 0

    trajectories = trajectories_SIMULATION

    trajectories.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS, EPISODE_IDS=EPISODE_IDS)
    trajectories.compute_indices(TARGET_IDS=TARGET_IDS, TRIAL_IDS=TRIAL_IDS, META_IDS=META_IDS, N_MOVS=N_MOVS, AGGREGATION_VARS=AGGREGATION_VARS)

    # assert not REPEATED_MOVEMENTS
    # if REPEATED_MOVEMENTS:
    #     EPISODE_ID_LIST = list(trajectories.data_copy.keys())
    # else:
    #     EPISODE_ID_LIST = range(trajectories.EPISODE_ID_NUMS)

    # for EPISODE_ID in EPISODE_ID_LIST:
    #     trajectories.preprocess(EPISODE_ID=str(EPISODE_ID), REPEATED_MOVEMENTS=False)
    #     # Use all available trials
    #     trajectories.compute_indices(TRIAL_IDS=None, META_IDS=None, N_MOVS=None)

    for trial_index_current, (last_idx, current_idx, next_idx) in enumerate(trajectories.selected_movements_indices):
        try:
            trajectories.compute_trial(trial_index_current, effective_projection_path=EFFECTIVE_PROJECTION_PATH,
                                       targetbound_as_target=USE_TARGETBOUND_AS_DIST, dwell_time=DWELL_TIME,
                                       compute_deviation=False, normalize_time=False)
        except AssertionError:
            failed_movements_counter += 1
        else:
            MT_list.append(trajectories.effective_MT_trial)
            width_list.append(trajectories.target_width_trial)
            distance_list.append(trajectories.target_distance_trial)
            ID_list.append(trajectories.fitts_ID_trial)
            target_position_list.append(trajectories.final_val)
            initial_position_list.append(trajectories.init_val)

    # Remove outliers:
    movement_indices_TO_DELETE = []
    if REMOVE_OUTLIERS or REMOVE_FAILED:
        # Ensure that movements that did not hit the target are removed
        if REMOVE_FAILED:
            movement_indices_TO_DELETE = np.where(np.array(MT_list) >= MAX_TIME - DWELL_TIME - 1e-6)[0].tolist()

        # Ensure that too long movements are removed (all movements with duration - mean(duration) > 3 * std(duration))
        # MT_list_zscores = stats.zscore(MT_list)
        if REMOVE_OUTLIERS:
            movement_indices_TO_DELETE = np.where(np.abs(stats.zscore(MT_list)) > 3)[0].tolist()

        distance_list = [val for idx, val in enumerate(distance_list) if idx not in movement_indices_TO_DELETE]
        width_list = [val for idx, val in enumerate(width_list) if idx not in movement_indices_TO_DELETE]
        ID_list = [val for idx, val in enumerate(ID_list) if idx not in movement_indices_TO_DELETE]
        MT_list = [val for idx, val in enumerate(MT_list) if idx not in movement_indices_TO_DELETE]
        target_position_list = [val for idx, val in enumerate(target_position_list) if
                                idx not in movement_indices_TO_DELETE]
        initial_position_list = [val for idx, val in enumerate(initial_position_list) if
                                 idx not in movement_indices_TO_DELETE]

    add_info = f'{len(movement_indices_TO_DELETE)} outliers were dropped' if movement_indices_TO_DELETE else '', f'{failed_movements_counter} movements failed' if failed_movements_counter else ''
    add_info_nonempty = [i for i in add_info if len(i)]
    add_info_str = f" ({'; '.join(add_info_nonempty)})" if add_info_nonempty else ""
    print(f"INFO: Using {len(ID_list)} movements from {len(trajectories.data)} episodes{add_info_str}.")

    # Linear regression:
    poly_coef = np.polyfit(ID_list, MT_list, 1)
    p_linreg = np.poly1d(poly_coef)

    # Coefficient of determination:
    SQE = np.square(np.linalg.norm(p_linreg(ID_list) - np.mean(MT_list)))
    SQT = np.square(np.linalg.norm(MT_list - np.mean(MT_list)))
    R2 = SQE / SQT
    print("Fitts' Law - Coefficient of determination (R^2): {}".format(R2))

    if PLOT_TYPE == "boxplot":
        df = pd.DataFrame({"Distance": distance_list, "Width": width_list, "ID": ID_list, "MT": MT_list})
        if BOXPLOT_qbins:
            df_bins = df.groupby([pd.qcut(df[BOXPLOT_category], BOXPLOT_nbins)])["MT"].apply(list)
        else:
            df_bins = df.groupby([pd.cut(df[BOXPLOT_category], BOXPLOT_nbins)])["MT"].apply(list)
        df_bins_renamed = df_bins.rename(lambda x: x if type(x) == float else f"{x.left.round(2)}-{x.right.round(2)}",
                                         axis=0)
        fittslaw_ax.boxplot(df_bins_renamed, labels=df_bins_renamed.index, positions=[x.mid for x in df_bins.index],
                            widths=[0.8 * (x.right - x.left) for x in df_bins.index])
        fittslaw_ax.relim()
        if BOXPLOT_category == "ID":
            lin_regression_x = locals()[f"{BOXPLOT_category}_list"]
            lin_regression_y = p_linreg(ID_list)
            lin_regression_sorted = sorted(zip(lin_regression_x, lin_regression_y))
            lin_regression_x = [x for x, y in lin_regression_sorted]
            lin_regression_y = [y for x, y in lin_regression_sorted]
            fittslaw_ax.plot(lin_regression_x, lin_regression_y, color="red")
    elif PLOT_TYPE == "mean_groups":
        df = pd.DataFrame({"Distance": distance_list, "Width": width_list, "ID": ID_list, "MT": MT_list})
        if REPEATED_MOVEMENTS:
            assert df[
                       "Width"].nunique() == BOXPLOT_nbins, f"Set 'BOXPLOT_nbins' to the number of different target sizes included in the current dataset ({df['Width'].nunique()})!"
            if BOXPLOT_qbins:
                df_bins_groupby = df.groupby([pd.qcut(df["Distance"], BOXPLOT_nbins), df["Width"]])["MT"]
            else:
                df_bins_groupby = df.groupby([pd.cut(df["Distance"], BOXPLOT_nbins), df["Width"]])["MT"]
        else:
            if BOXPLOT_qbins:
                df_bins_groupby = \
                    df.groupby([pd.qcut(df["Distance"], BOXPLOT_nbins), pd.qcut(df["Width"], BOXPLOT_nbins)])["MT"]
            else:
                df_bins_groupby = \
                    df.groupby([pd.cut(df["Distance"], BOXPLOT_nbins), pd.cut(df["Width"], BOXPLOT_nbins)])["MT"]
        df_bins = df_bins_groupby.mean()
        df_bins_renamed = df_bins.rename(lambda x: x if type(x) == float else f"{x.left.round(2)}-{x.right.round(2)}",
                                         axis=0)
        df_bins_renamed.index = df_bins_renamed.index.map(lambda x: f"D: {x[0]}, W: {x[1]}")
        trans_dict_dist = {number: chr(ord('@') + number + 1) for number in range(BOXPLOT_nbins)}
        trans_dict_width = {number: str(number) for number in range(BOXPLOT_nbins)}
        group_identifiers_dist = [(str(trans_dict_dist[idx]), f"D: {value}") for idx, value in
                                  enumerate(df_bins.index.get_level_values(0).unique().tolist())]
        group_identifiers_width = [(str(trans_dict_width[idx]), f"W: {value}") for idx, value in
                                   enumerate(df_bins.index.get_level_values(1).unique().tolist())]
        group_identifiers = df_bins.index.map(lambda x: "".join(
            [trans_dict_dist[df_bins.index.get_level_values(0).unique().tolist().index(x[0])],
             trans_dict_width[df_bins.index.get_level_values(1).unique().tolist().index(x[1])]]))
        if REPEATED_MOVEMENTS:
            assert df[
                       "Width"].nunique() == BOXPLOT_nbins, f"Set 'BOXPLOT_nbins' to the number of different target sizes included in the current dataset ({df['Width'].nunique()})!"
            if BOXPLOT_qbins:
                df["classification"] = [df_bins.index.get_loc((x["Distance"], x["Width"])) for _, x in
                                        pd.concat((pd.qcut(df["Distance"], BOXPLOT_nbins), df["Width"]),
                                                  axis=1).iterrows()]
            else:
                df["classification"] = [df_bins.index.get_loc((x["Distance"], x["Width"])) for _, x in
                                        pd.concat((pd.cut(df["Distance"], BOXPLOT_nbins), df["Width"]),
                                                  axis=1).iterrows()]
        else:
            if BOXPLOT_qbins:
                df["classification"] = [df_bins.index.get_loc((x["Distance"], x["Width"])) for _, x in pd.concat(
                    (pd.qcut(df["Distance"], BOXPLOT_nbins), pd.qcut(df["Width"], BOXPLOT_nbins)), axis=1).iterrows()]
            else:
                df["classification"] = [df_bins.index.get_loc((x["Distance"], x["Width"])) for _, x in pd.concat(
                    (pd.cut(df["Distance"], BOXPLOT_nbins), pd.cut(df["Width"], BOXPLOT_nbins)), axis=1).iterrows()]
        df_bins = df_bins.reset_index()
        # df_bins["ID"] = df_bins.apply(lambda x: pd.Interval(np.log2(2*x["Distance"].left / x["Width"]), np.log2(2*x["Distance"].right / x["Width"])) if type(x["Width"]) == float else (pd.Interval(np.log2(2*x["Distance"] / x["Width"].right), np.log2(2*x["Distance"] / x["Width"].left)) if type(x["Distance"]) == float else pd.Interval(np.log2(2*x["Distance"].left / x["Width"].right), np.log2(2*x["Distance"].right / x["Width"].left))), axis=1)
        # df_bins["Average ID"] = df_bins.apply(lambda x: np.log2(2*x["Distance"].mid / x["Width"]) if type(x["Width"]) == float else (np.log2(2*x["Distance"] / x["Width"].mid) if type(x["Distance"]) == float else np.log2(2*x["Distance"].mid / x["Width"].mid)), axis=1)
        df_bins["ID"] = df_bins.apply(lambda x: pd.Interval(np.log2((x["Distance"].left / x["Width"]) + 1),
                                                            np.log2((x["Distance"].right / x["Width"]) + 1)) if type(
            x["Width"]) == float else (pd.Interval(np.log2((x["Distance"] / x["Width"].right) + 1),
                                                   np.log2((x["Distance"] / x["Width"].left) + 1)) if type(
            x["Distance"]) == float else pd.Interval(np.log2((x["Distance"].left / x["Width"].right) + 1),
                                                     np.log2((x["Distance"].right / x["Width"].left) + 1))), axis=1)
        df_bins["Average ID"] = df_bins.apply(
            lambda x: np.log2((x["Distance"].mid / x["Width"]) + 1) if type(x["Width"]) == float else (
                np.log2((["Distance"] / x["Width"].mid) + 1) if type(x["Distance"]) == float else np.log2(
                    (x["Distance"].mid / x["Width"].mid) + 1)), axis=1)
        df_bins["Number of Samples"] = df_bins_groupby.size().reset_index(drop=True)
        xtick_offset = 0

        ### ONLY SHOW SOME CONDITIONS:
        # df = df.loc[(df["classification"] >= 5) & (df["classification"] < 15)]
        # group_identifiers_dist = group_identifiers_dist[1:3]
        # group_identifiers = group_identifiers[5:15]
        # df_bins = df_bins.iloc[5:15]
        # df_bins_renamed = df_bins_renamed.iloc[5:15]
        # xtick_offset = 5

        fittslaw_ax.scatter(range(xtick_offset, xtick_offset + len(df_bins_renamed)), df_bins_renamed,
                            color="red")  # , positions=[x.mid for x in df_bins.index], widths=[0.8*(x.right - x.left) for x in df_bins.index])
        fittslaw_ax.scatter(df["classification"], df["MT"], color="blue", s=0.2)
        ## VARIANT 1 - detailed xlabels:
        # fittslaw_ax.set_xticks(range(len(df_bins_renamed)), df_bins_renamed.index, fontsize=4)
        ## VARIANT 2 - abbreviations as xlabels, additional legend:
        # fittslaw_ax.set_xticks(df_bins[~df_bins["MT"].isna()].index, df_bins[~df_bins["MT"].isna()].index + 1)
        fittslaw_ax.set_xticks(df_bins[~df_bins["MT"].isna()].index, group_identifiers[~df_bins["MT"].isna()])

        class MarkerHandler(matplotlib.legend_handler.HandlerBase):  # source: https://stackoverflow.com/a/47395401
            def create_artists(self, legend, integer, xdescent, ydescent,
                               width, height, fontsize, trans):
                marker_obj = matplotlib.markers.MarkerStyle(f'${integer}$')  # Here you place your integer
                path = marker_obj.get_path().transformed(marker_obj.get_transform())

                path._vertices = np.array(path._vertices) * 8  # To make it larger
                patch = matplotlib.patches.PathPatch(path, color="black", lw=0,
                                              transform=trans + matplotlib.transforms.Affine2D().translate(0, 5.5))
                return [patch]

        fittslaw_ax.legend([x for x, y in group_identifiers_dist + group_identifiers_width],
                           [y for x, y in group_identifiers_dist + group_identifiers_width],
                           handler_map={str: MarkerHandler()}, handletextpad=0,
                           handlelength=1)  # , prop={'size': 6})#, fontsize=8)
        ################################
        fittslaw_ax.relim()
    elif PLOT_TYPE == "meandata":
        df = pd.DataFrame({"Distance": distance_list, "Width": width_list, "ID": ID_list, "MT": MT_list})
        if REPEATED_MOVEMENTS:
            assert df[
                       "Width"].nunique() == BOXPLOT_nbins, f"Set 'BOXPLOT_nbins' to the number of different target sizes included in the current dataset ({df['Width'].nunique()})!"
            if BOXPLOT_qbins:
                df_bins_groupby = df.groupby([pd.qcut(df["Distance"], BOXPLOT_nbins), df["Width"]])["MT"]
            else:
                df_bins_groupby = df.groupby([pd.cut(df["Distance"], BOXPLOT_nbins), df["Width"]])["MT"]
        else:
            if BOXPLOT_qbins:
                df_bins_groupby = \
                    df.groupby([pd.qcut(df["Distance"], BOXPLOT_nbins), pd.qcut(df["Width"], BOXPLOT_nbins)])["MT"]
            else:
                df_bins_groupby = \
                    df.groupby([pd.cut(df["Distance"], BOXPLOT_nbins), pd.cut(df["Width"], BOXPLOT_nbins)])["MT"]
        df_bins = df_bins_groupby.mean()
        df_bins_renamed = df_bins.rename(lambda x: x if type(x) == float else f"{x.left.round(2)}-{x.right.round(2)}",
                                         axis=0)
        df_bins_renamed.index = df_bins_renamed.index.map(lambda x: f"D: {x[0]}, W: {x[1]}")
        df_bins = df_bins.reset_index()
        # df_bins["ID"] = df_bins.apply(lambda x: pd.Interval(np.log2(2*x["Distance"].left / x["Width"]), np.log2(2*x["Distance"].right / x["Width"])) if type(x["Width"]) == float else (pd.Interval(np.log2(2*x["Distance"] / x["Width"].right), np.log2(2*x["Distance"] / x["Width"].left)) if type(x["Distance"]) == float else pd.Interval(np.log2(2*x["Distance"].left / x["Width"].right), np.log2(2*x["Distance"].right / x["Width"].left))), axis=1)
        # df_bins["Average ID"] = df_bins.apply(lambda x: np.log2(2*x["Distance"].mid / x["Width"]) if type(x["Width"]) == float else (np.log2(2*x["Distance"] / x["Width"].mid) if type(x["Distance"]) == float else np.log2(2*x["Distance"].mid / x["Width"].mid)), axis=1)
        df_bins["ID"] = df_bins.apply(lambda x: pd.Interval(np.log2((x["Distance"].left / x["Width"]) + 1),
                                                            np.log2((x["Distance"].right / x["Width"]) + 1)) if type(
            x["Width"]) == float else (pd.Interval(np.log2((x["Distance"] / x["Width"].right) + 1),
                                                   np.log2((x["Distance"] / x["Width"].left) + 1)) if type(
            x["Distance"]) == float else pd.Interval(np.log2((x["Distance"].left / x["Width"].right) + 1),
                                                     np.log2((x["Distance"].right / x["Width"].left) + 1))), axis=1)
        df_bins["Average ID"] = df_bins.apply(
            lambda x: np.log2((x["Distance"].mid / x["Width"]) + 1) if type(x["Width"]) == float else (
                np.log2((["Distance"] / x["Width"].mid) + 1) if type(x["Distance"]) == float else np.log2(
                    (x["Distance"].mid / x["Width"].mid) + 1)), axis=1)

        df_bins["Number of Samples"] = df_bins_groupby.size().reset_index(drop=True)

        ### ONLY SHOW SOME CONDITIONS:
        #     df_bins = df_bins.iloc[5:15]
        #     df_bins_renamed = df_bins_renamed.iloc[5:15]

        # Coefficient of determination on mean groups:
        ID_list_mean = df_bins.loc[~df_bins["MT"].isna(), "Average ID"]
        MT_list_mean = df_bins.loc[~df_bins["MT"].isna(), "MT"]
        poly_coef_mean = np.polyfit(ID_list_mean, MT_list_mean, 1)
        p_linreg_mean = np.poly1d(poly_coef_mean)
        SQE_mean = np.square(np.linalg.norm(p_linreg_mean(ID_list_mean) - np.mean(MT_list_mean)))
        SQT_mean = np.square(np.linalg.norm(MT_list_mean - np.mean(MT_list_mean)))
        R2_mean = SQE_mean / SQT_mean
        print("Fitts' Law [ON MEAN PER CONDITION] - Coefficient of determination (R^2): {}".format(R2_mean))

        fittslaw_ax.scatter(df_bins["Average ID"], df_bins["MT"])
        fittslaw_ax.plot(df_bins["Average ID"], p_linreg_mean(df_bins["Average ID"]), color="red")
    elif PLOT_TYPE == "alldata":
        fittslaw_ax.scatter(ID_list, MT_list)
        fittslaw_ax.plot(ID_list, p_linreg(ID_list), color="red")
    elif PLOT_TYPE == "density_ID":
        df = pd.DataFrame({"Distance": distance_list, "Width": width_list, "ID": ID_list, "MT": MT_list})
        if BOXPLOT_qbins:
            df_bins_ID_helper = pd.qcut(df["ID"], DENSITY_group_nIDbins)
            df_bins_ID_groupby = df.groupby([df_bins_ID_helper])["MT"]
            df_bins_ID = df_bins_ID_groupby.mean()
            df["classification"] = [df_bins_ID.index.get_loc(x) for x in df_bins_ID_helper.values]
        else:
            df_bins_ID_helper = pd.cut(df["ID"], DENSITY_group_nIDbins)
            df_bins_ID_groupby = df.groupby([df_bins_ID_helper])["MT"]
            df_bins_ID = df_bins_ID_groupby.mean()
            df["classification"] = [df_bins_ID.index.get_loc(x) for x in df_bins_ID_helper.values]
        ID_groups = sorted(df_bins_ID_helper.unique())
        df_bins_ID = df_bins_ID.reset_index()
        df_bins_ID["Average ID"] = df_bins_ID.apply(lambda x: x["ID"].mid, axis=1)
        df_bins_ID["Number of Samples"] = df_bins_ID_groupby.size().reset_index(drop=True)

        fittslaw_ax.hist(df.loc[df["classification"] == DENSITY_group_IDbin_ID, "MT"], DENSITY_group_nMTbins)
    else:
        raise NotImplementedError

    # # Save the default tick positions, so we can reset them...
    # locs = fittslaw_ax.get_xticks()
    # # Reset the xtick locations.
    # fittslaw_ax.set_xticks(locs)

    if PLOT_TYPE == "density_ID":
        fittslaw_ax.set_xlabel("MT (s)")
    elif PLOT_TYPE not in ["mean_groups"]:
        fittslaw_ax.set_xlabel(BOXPLOT_category + (f" bits" if (BOXPLOT_category == "ID") else ""))  # "ID (bits)")
    if PLOT_TYPE == "density_ID":
        fittslaw_ax.set_ylabel("Frequency")
    else:
        fittslaw_ax.set_ylabel("MT (s)")
    if PLOT_TYPE == "meandata":
        fittslaw_ax.set_title(
            "Fitts' Law on Mean Data\n($R^2=${:.4g}; $a=${:.2g}, $b=${:.2g})".format(R2_mean, poly_coef_mean[1],
                                                                                     poly_coef_mean[
                                                                                         0]))  # , fontsize=14) #fontsize=18)
    elif PLOT_TYPE == "density_ID":
        fittslaw_ax.set_title(
            f"Movement Time – Density Plot (ID: {ID_groups[DENSITY_group_IDbin_ID]})" if DENSITY_group_nIDbins > 1 else "Movement Time – Density Plot")
    else:
        fittslaw_ax.set_title("Fitts' Law\n($R^2=${:.4g}; $a=${:.2g}, $b=${:.2g})".format(R2, poly_coef[1], poly_coef[
            0]))  # , fontsize=14) #fontsize=18)

    if PLOT_TYPE in ["mean_groups"]:
        for xticklabel in fittslaw_ax.get_xticklabels():
            xticklabel.set_fontsize(6)
        # for legendtext in fittslaw_ax.get_legend().get_texts():
        #     legendtext.set_fontsize(6)
        for legendpatch in fittslaw_ax.get_legend().get_patches():
            legendpatch.set_transform(matplotlib.transforms.Affine2D().scale(0.5) + legendpatch.get_transform())

    fittslaw_fig.tight_layout()

    if STORE_PLOT:
        _filename = os.path.join(PLOTS_DIR, f"{plot_filename_fitts}.png")
        if not os.path.exists(os.path.dirname(_filename)):
            os.makedirs(os.path.dirname(_filename))
        fittslaw_fig.savefig(_filename, dpi=300)
    else:
        logging.warning(f"Plotting is disabled.\nCurrent filename: {plot_filename_fitts}.png")
