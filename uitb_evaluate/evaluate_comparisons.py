import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import os, logging
import itertools
from uitb_evaluate.trajectory_data import PLOTS_DIR_DEFAULT

def quantitativecomparisonplot(PLOTTING_ENV_COMPARISON, QUANTITY, res_dict,
                               SIMULATION_SUBDIR_LIST,
                               TASK_CONDITION_LIST,
                               USER2USER_FIXED=False,
                               USER2USER=False,
                               SIM2USER_PER_USER=False,
                               USER_ID_LIST=["U1", "U2", "U3", "U4", "U5", "U6"],
                               USER_ID_FIXED="<unknown-user>",
                               COLOR_PALETTE="turbo",
                               ENABLE_LEGEND=True,
                               STORE_PLOT=False, PLOTS_DIR=PLOTS_DIR_DEFAULT):

    if len(SIMULATION_SUBDIR_LIST) > 1:
        common_simulation_subdir = "".join([i for i, j, k in zip(*SIMULATION_SUBDIR_LIST) if i == j == k])
        if len(common_simulation_subdir) == 0:
            common_simulation_subdir = "ALLCOSTS"
        elif common_simulation_subdir[-1] in ["-", "_"]:
            common_simulation_subdir = common_simulation_subdir[:-1]
        _plot_SIMULATION_SUBDIR_LIST = common_simulation_subdir
    else:
        _plot_SIMULATION_SUBDIR_LIST = "__".join(SIMULATION_SUBDIR_LIST)
    _plot_TASK_CONDITION_LIST = "__".join(TASK_CONDITION_LIST)

    quantitative_comparison_fig = plt.figure(figsize=[13.5, 3] if SIM2USER_PER_USER else [4.5, 3])
    quantitative_comparison_ax = quantitative_comparison_fig.gca()

    quantitative_comparison_ax.clear()
    if quantitative_comparison_fig.get_figwidth() > 6:
        quantitative_comparison_fig.subplots_adjust(left=0.075, right=0.975, bottom=0.1, top=0.9)#, wspace=0.26, hspace=0.2)
    else:
        quantitative_comparison_fig.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.9)#, wspace=0.26, hspace=0.2)

    # ###
    # QUANTITY = "pos"
    #
    # USER2USER_FIXED = False  #if True, compare predictability of user movements of USER_ID_FIXED between simulation and other users
    # USER2USER = False  #if True, compare predictability of user movements (of all users) between simulation and respective other users
    #
    # SIM2USER_PER_USER = False  #if True, compare cost function simulation for each user separately
    #
    # COLOR_PALETTE = "turbo"  #None
    # ENABLE_LEGEND = True  #if False, legends is removed
    # ###

    if len(SIMULATION_SUBDIR_LIST) > 1:
        map_conditions = lambda x: "JAC" if "accjoint" in x else "CTC" if "ctc" in x else "DC" if "cso" in x else x
    else:
        map_conditions = lambda x: x if x.startswith("U") else "Sim." if (PLOTTING_ENV_COMPARISON == "RL-UIB") else ("JAC" if "accjoint" in x else "CTC" if "ctc" in x else "DC" if "cso" in x else x) if USER2USER_FIXED else "Simulation\nvs. User"
    label_dict = {"pos": ("Cursor Positions", r"$(m)$"),
                 "vel": ("Cursor Velocities", r"$\left(\frac{m}{s}\right)$"),
                 "acc": ("Cursor Accelerations", r"$\left(\frac{m}{s^2}\right)$"),
                 "qpos": ("Joint Angles", r"$(rad)$"),
                 "qvel": ("Joint Velocities", r"$\left(\frac{rad}{s}\right)$"),
                 "qacc": ("Joint Accelerations", r"$\left(\frac{rad}{s^2}\right)$")}
    map_task_conditions = lambda x: " ".join(["Virtual Pad" if "_Pad_" in x else "Virtual Cursor", "ID" if "_ID_" in x else "Erg." if "_Ergonomic_" in x else "???"])

    quantitative_comparison_ax.set_title(f"{'All' if len(TASK_CONDITION_LIST) == 4 else '/'.join([map_task_conditions(x) for x in TASK_CONDITION_LIST])} Trials of {USER_ID_FIXED} – {label_dict[QUANTITY][0]}" if USER2USER_FIXED else f"{'All Trials' if len(TASK_CONDITION_LIST) == 4 else '/'.join([map_task_conditions(x) for x in TASK_CONDITION_LIST])} – {label_dict[QUANTITY][0]}", fontsize=14)
    quantitative_comparison_ax.set_ylabel(f"RMSE {label_dict[QUANTITY][1]}", fontsize=14)
    quantitative_comparison_ax.tick_params(axis='both', which='major', labelsize=14)
    quantitative_comparison_ax.tick_params(axis='both', which='minor', labelsize=12)
    quantitative_comparison_ax.set_yscale("log")

    if SIM2USER_PER_USER:
        assert not (USER2USER or USER2USER_FIXED)
        boxplot_df_melt = pd.DataFrame()
        boxplot_df_user = pd.Series(dtype=str)
        for i in range(len(USER_ID_LIST)):
            boxplot_df = pd.DataFrame({map_conditions(k): pd.Series(np.concatenate(v[i*len(TASK_CONDITION_LIST): (i+1)*len(TASK_CONDITION_LIST)]).ravel()) for k, v in res_dict.items()})

            boxplot_df_melt = pd.concat((boxplot_df_melt, boxplot_df.melt()))
            boxplot_df_user = pd.concat((boxplot_df_user, pd.Series([USER_ID_LIST[i]] * boxplot_df.melt().shape[0])))
        boxplot_df_melt["User"] = boxplot_df_user
        boxplot_df_melt = boxplot_df_melt.rename({'variable': 'Cost function', 'value': f"RMSE {label_dict[QUANTITY][1]}"}, axis=1)
        sns.boxplot(data=boxplot_df_melt, x="User", y=f"RMSE {label_dict[QUANTITY][1]}", hue="Cost function", ax=quantitative_comparison_ax, palette=COLOR_PALETTE)

        quantitative_comparison_ax.legend(loc="best", ncol=boxplot_df_melt["Cost function"].nunique())
    else:
        boxplot_df = pd.DataFrame({map_conditions(k): pd.Series(np.concatenate(v).ravel()) for k, v in res_dict.items()})

        sns.boxplot(data=boxplot_df, ax=quantitative_comparison_ax, palette=[matplotlib.cm.get_cmap(COLOR_PALETTE)(0.2 if i.startswith("U") else 0.75) for i in boxplot_df.columns] if USER2USER or USER2USER_FIXED else COLOR_PALETTE)

    # Add statistics annotations:
    if USER2USER_FIXED:
        pass #add_stat_annotation(data=boxplot_df, ax=quantitative_comparison_ax, box_pairs=[(map_conditions(i), j) for i in SIMULATION_SUBDIR_LIST for j in USER_ID_LIST if j != USER_ID_FIXED], test="Wilcoxon" if not USER2USER_FIXED and not USER2USER else "Mann-Whitney", text_format="star", loc="inside", verbose=2)
    elif USER2USER:
        pass
    else:
        if not SIM2USER_PER_USER:
            add_stat_annotation(data=boxplot_df, ax=quantitative_comparison_ax, box_pairs=itertools.combinations(boxplot_df.keys(), 2), test="Wilcoxon" if not USER2USER_FIXED and not USER2USER else "Mann-Whitney", text_format="star", loc="inside", verbose=2)

    # Print additional statistic test results
    if not USER2USER_FIXED and not USER2USER:
        kstest_results = [(cn, stats.kstest(boxplot_df[cn], 'norm')) for cn in boxplot_df.columns]
        #friedman_results = f"Friedman - $\chi^{2}$:{stats.friedmanchisquare(optparam_info_df_STATS['2OL-Eq"], optparam_info_df_STATS["MinJerk"], optparam_info_df_STATS["LQR"]).statistic}"#, p-value: {stats.friedmanchisquare(optparam_info_df_STATS).pvalue}"
        #friedman_results = stats.friedmanchisquare(optparam_info_df_STATS["2OL-Eq"], optparam_info_df_STATS["MinJerk"], optparam_info_df_STATS["LQR"])
        friedman_results = stats.friedmanchisquare(*[boxplot_df[cn] for cn in boxplot_df.columns])
        wilcoxon_results = [((cn1, cn2), stats.wilcoxon(boxplot_df[cn1], boxplot_df[cn2], alternative="two-sided"),
                             'z-statistic:', (stats.wilcoxon(boxplot_df[cn1], boxplot_df[cn2], alternative="two-sided").statistic - boxplot_df.shape[0] * (boxplot_df.shape[0] + 1) / 4) / ((boxplot_df.shape[0] * (boxplot_df.shape[0] + 1) * (2 * boxplot_df.shape[0] + 1) / 24) ** 0.5),
                             'z-statistic (cp., unsigned):', stats.norm.isf(stats.wilcoxon(boxplot_df[cn1], boxplot_df[cn2], alternative="two-sided").pvalue / 2)) for cn1, cn2 in list(itertools.combinations(boxplot_df.columns, 2))]
        #print(f"{QUANTITY}:\n\t{kstest_results}\n\t{friedman_results}\n\t{wilcoxon_results}")
        print(f"\n{QUANTITY}")
        print('Kolmogorov-Smirnov:', *kstest_results, sep='\n\t')
        print(f'Friedman ({boxplot_df.columns.values}):', friedman_results, sep='\n\t')
        print('Wilcoxon Signed Rank:', *wilcoxon_results, sep='\n\t')
    elif USER2USER and not USER2USER_FIXED:
        assert boxplot_df.shape[1] == 2
        mannwhitneyu_results = [stats.mannwhitneyu(boxplot_df.iloc[:, 0], boxplot_df.iloc[:, 1], nan_policy='omit'),
                                'z-statistic (with continuity correction):', (stats.mannwhitneyu(boxplot_df.iloc[:, 0], boxplot_df.iloc[:, 1], nan_policy='omit').statistic - ((sum(~boxplot_df.iloc[:, 0].isna()) * sum(~boxplot_df.iloc[:, 1].isna())) / 2) + 0.5) / ((sum(~boxplot_df.iloc[:, 0].isna()) * (sum(~boxplot_df.iloc[:, 1].isna())) * (sum(~boxplot_df.iloc[:, 0].isna()) + sum(~boxplot_df.iloc[:, 1].isna()) + 1) / 12) ** 0.5),
                                ]
        print(f"\n{QUANTITY}")
        print('Mann-Whitney U:', *mannwhitneyu_results, sep='\n\t')

    # Remove legend
    if not ENABLE_LEGEND:
        current_legend = quantitative_comparison_ax.get_legend()
        if current_legend:
            current_legend.remove()

    # # Manually change whisker label
    # test = quantitative_comparison_ax.get_xticklabels()
    # test[1].set_text("User\nvs. User")
    # quantitative_comparison_ax.set_xticklabels(test)

    # STORE TO FILE
    if USER2USER_FIXED:
        if PLOTTING_ENV_COMPARISON == "RL-UIB":
            plot_filename_quantitative_ID = f"UIB/{_plot_SIMULATION_SUBDIR_LIST}/{USER_ID_FIXED}_FIXED/RMSE_{QUANTITY}"
        elif PLOTTING_ENV_COMPARISON == "MPC":
            plot_filename_quantitative_ID = f"MPC/{_plot_SIMULATION_SUBDIR_LIST}/{USER_ID_FIXED}_FIXED/{_plot_TASK_CONDITION_LIST}/RMSE_{QUANTITY}"
        else:
            raise NotImplementedError
    elif USER2USER:
        if PLOTTING_ENV_COMPARISON == "MPC":
            plot_filename_quantitative_ID = f"MPC/{_plot_SIMULATION_SUBDIR_LIST}/ALLUSERS/{_plot_TASK_CONDITION_LIST}/RMSE_{QUANTITY}"
        else:
            raise NotImplementedError
    else:
        if SIM2USER_PER_USER:
            if PLOTTING_ENV_COMPARISON == "MPC":
                plot_filename_quantitative_ID = f"MPC/{_plot_SIMULATION_SUBDIR_LIST}/ALLUSERS/{_plot_TASK_CONDITION_LIST}/RMSE_simonly_peruser_{QUANTITY}"
            else:
                raise NotImplementedError
        else:
            if PLOTTING_ENV_COMPARISON == "MPC":
                plot_filename_quantitative_ID = f"MPC/{_plot_SIMULATION_SUBDIR_LIST}/ALLUSERS/{_plot_TASK_CONDITION_LIST}/RMSE_simonly_{QUANTITY}"
            else:
                raise NotImplementedError
    if STORE_PLOT:
        _filename = os.path.join(PLOTS_DIR, f"{plot_filename_quantitative_ID}.png")
        if not os.path.exists(os.path.dirname(_filename)):
            os.makedirs(os.path.dirname(_filename))
        quantitative_comparison_fig.savefig(_filename, dpi=300)
    else:
        logging.warning(f"Plotting is disabled.\nCurrent filename: {plot_filename_quantitative_ID}.png")