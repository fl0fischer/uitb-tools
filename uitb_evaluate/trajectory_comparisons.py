import logging
import numpy as np
from scipy.interpolate import interp1d

### RUN QUANTITATIVE COMPARISONS BETWEEN TWO DATASETS
class QuantitativeComparison(object):
    
    def __init__(self, trajectories1, trajectories2):
        self.trajectories1 = trajectories1  #TrajectoryData instance
        self.trajectories2 = trajectories2  #TrajectoryData instance
    
    def compare(self, data_name1, data_name2=None, cols=None, metric="RMSE", mean_axis=None):
        #WARNING: "data_name1" and "data_name2" need to correspond to some "_trial" (or "_mean") attribute!
        #INFO: "cols" should be array-like, consisting of column indices to take into account; if "None", all data columns are used.
        #INFO: "mean_axis" is used as "axis" argument of np.mean(), i.e., "None" corresponds to mean of flattened array, and "0" corresponds to mean along time axis for each data column.
        
        if data_name2 is None:
            data_name2 = data_name1
        assert isinstance(data_name1, str) and isinstance(data_name2, str)  #data_name1 and data_name2 should be strings corresponding to the attributes of trajectories1 and trajectories2, respectively, that should be compared
            
        data1 = getattr(self.trajectories1, data_name1)
        data2 = getattr(self.trajectories2, data_name2)
        
        if cols is not None:
            data1 = data1[..., cols]
            data2 = data2[..., cols]
        
        if len(data1) != len(data2):
            assert hasattr(self.trajectories1, "time_series_trial") and hasattr(self.trajectories2, "time_series_trial")
            #input((self.trajectories1.time_series_trial, self.trajectories2.time_series_trial))
            if any([i != j for i,j in zip(self.trajectories1.time_series_trial, self.trajectories2.time_series_trial)]):
                logging.warning(f"Time series do not match. Interpolate second dataset at time series of first dataset.")
                #data2 = np.interp(self.trajectories1.time_series_trial, self.trajectories2.time_series_trial, data2)
                data2_f = interp1d(self.trajectories2.time_series_trial, data2, axis=0, fill_value="extrapolate")
                data2 = data2_f(self.trajectories1.time_series_trial)
            else:
                logging.warning(f"Ignore overhanging last {max(len(data1), len(data2)) - min(len(data1), len(data2))} timesteps.")
        
        if metric == "RMSE":
            #WARNING: if data1 and data2 are multi-dimensional, np.mean() takes mean along all dimensions!
            res = np.sqrt(np.mean([(j-i)**2 for i,j in zip(data1, data2)], axis=mean_axis))
        elif metric == "initial_distance":
            res = np.mean([np.abs(data1[0] - data2[0])], axis=mean_axis)
        else:
            raise NotImplementedError
            
        return res
    
    def compare_all_trials(self, data_name1, data_name2=None, ignore_unpaired_trials=False, effective_projection_path=False, targetbound_as_target=False, **kwargs):
        
        # INFO: "ignore_unpaired_trials" can be used even if self.trajectories1.indices and self.trajectories2.indices are not of the same form (e.g., for comparison between RL and STUDY data)
        if ignore_unpaired_trials and (len(self.trajectories1.selected_movements_indices) != len(self.trajectories2.selected_movements_indices)):  #remove indices only available in data_name1
            logging.warning(f"Indices do not match. Remove redundant trials from first dataset.")
            assert len(self.trajectories1.selected_movements_indices) >= len(self.trajectories2.selected_movements_indices), f"{len(self.trajectories2.selected_movements_indices) - len(self.trajectories1.selected_movements_indices)} trials are missing in simulation!."
            assert len(self.trajectories1.indices) == len(self.trajectories1.selected_movements_indices), f"ERROR: 'indices' and 'selected_movements_indices' do not have the same length!"
            self.trajectories1._indices = self.trajectories1.indices[[i in self.trajectories2.indices[:, 4] for i in self.trajectories1.indices[:, 4]]]
            self.trajectories1.selected_movements_indices = [trajectories_SIMULATION.selected_movements_indices[meta_idx] for meta_idx, i in enumerate(trajectories_SIMULATION.indices[:, 4]) if i in trajectories_STUDY.indices[:, 4]]#.shape
                
        assert len(self.trajectories1.selected_movements_indices) == len(self.trajectories2.selected_movements_indices), f"Number of considered trials does not match between the two datasets ({len(self.trajectories1.selected_movements_indices)} vs. {len(self.trajectories2.selected_movements_indices)})."
        #assert np.all(self.trajectories1.indices[:, 2:] == self.trajectories2.indices[:, 2:]), "Indices do not match between the two datasets."
        
        res_all_trials = []
        for trial_index_current in range(len(self.trajectories1.selected_movements_indices)):
            self.trajectories1.compute_trial(trial_index_current, effective_projection_path=effective_projection_path, targetbound_as_target=targetbound_as_target)
            self.trajectories2.compute_trial(trial_index_current, effective_projection_path=effective_projection_path, targetbound_as_target=targetbound_as_target)
            res_all_trials.append(self.compare(data_name1, data_name2=data_name2, **kwargs))
        
        return res_all_trials
    
