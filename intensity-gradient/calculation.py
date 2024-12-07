import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats as st

per_min_vars = [
    'total_distance',
    'velocity_band6_total_distance',
    'total_hi_distance',
    'gen2_acceleration_band2plus_total_effort_count',
    'gen2_acceleration_band7plus_total_effort_count',
    'gen2_acceleration_band3plus_total_effort_count',
    'gen2_acceleration_band6plus_total_effort_count'
]

int_vars = []

for i in per_min_vars:
    col_name = i + ' per min'
    df[col_name] = df[i]/df['field_time']
    int_vars.append(col_name)

vel_zones = {
            'velocity_band2_total_distance':1.1,
            'velocity_band3_total_distance':3,
            'velocity_band4_total_distance':4.75,
            'velocity_band5_total_distance':6.25,
            'velocity_band6_total_distance':11
}



accel_zones = {
            'acceleration_band5_total_distance':0.5,
            'acceleration_band6_total_distance':1.5,
            'acceleration_band7_total_distance':2.5,
            'acceleration_band8_total_distance':11.5
}

decel_zones = {
            'acceleration_band4_total_distance':0.5,
            'acceleration_band3_total_distance':1.5,
            'acceleration_band2_total_distance':2.5,
            'acceleration_band1_total_distance':11.5
}


for i, zones in enumerate([vel_zones, accel_zones, decel_zones]):
    
    if i == 0:
        comparison = 'total_distance'
    else:
        comparison = 'field_time'
    
    for x in zones:
        df[x + ' per min'] = df[x]/df['field_time']
        df[x + ' relative'] = df[x]/df[comparison]

name_dict = {
    'Velocity Slope':vel_zones,
    'Accel Slope': accel_zones,
    'Decel Slope': decel_zones
}


def calc_slope(row, bin_mids_logs):
    a = stats.linregress(y=row, x=bin_mids_logs)

    return a.slope 


def calc_slope(row, bin_mids_logs):
    
    bin_mids_logs = np.array(bin_mids_logs)
    
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=bin_mids_logs, y=row)

    # Calculate predicted values
    y_pred = slope * bin_mids_logs + intercept

    # Calculate R-squared
    ss_total = np.sum((row - np.mean(row))**2)
    ss_residual = np.sum((row - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Calculate residual errors
    residuals = row - y_pred

    # Calculate RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    

    return [slope, intercept, r_squared, rmse, residuals]


for i in name_dict:
    
    zone_dict =  name_dict.get(i)
    zone_keys = list(zone_dict.keys())
    bin_mids = list(zone_dict.values())
    
    bin_mid_logs = []
    
    for x in bin_mids:
        bin_mid_logs.append(np.log(x))
    
    
    log_zones = []

    for x in zone_keys:
        df['Log ' + x] = np.log(df[x])
        log_zones.append('Log ' + x)

        df['Log ' + x].fillna(0, inplace=True)
        df['Log ' + x].replace([np.inf, -np.inf], 0, inplace=True)

    zone_cols = [df.columns.get_loc(i) for i in log_zones]
    
    # Apply calc_slope function to each row of the DataFrame
    result = df.iloc[:, zone_cols].apply(calc_slope, bin_mids_logs=bin_mid_logs, axis=1)

    # Create new columns from the unpacked lists
    df[[i, i +' intercept', i +' r_squared', i + ' rmse', i + ' residuals']] = pd.DataFrame(result.tolist(), index=df.index)
    
    
    df = df.loc[df[i] != 0].reset_index(drop=True)

    
    int_vars.append(i)
    


residuals_cols = []

for i in ['Velocity Slope residuals', 'Accel Slope residuals', 'Decel Slope residuals']:
    unpacked_df = pd.DataFrame(df[i].tolist()).reset_index(drop=True)
    unpacked_df = unpacked_df.add_suffix(' residual error')
    
    residuals_cols.append(list(unpacked_df.columns))

    df = pd.concat([df, unpacked_df], axis=1)

    

