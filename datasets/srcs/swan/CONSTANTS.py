from features import feature_collection as fc

'''
    List of 43 statistical features which return scalars
'''
CANDIDATE_STAT_FEATURES = [fc.get_min, fc.get_max, fc.get_median, fc.get_mean, fc.get_stddev, fc.get_var
    # , fc.get_skewness, fc.get_kurtosis, fc.get_no_local_maxima, fc.get_no_local_minima
    # , fc.get_no_local_extrema, fc.get_no_zero_crossings, fc.get_mean_local_maxima_value
    # , fc.get_mean_local_minima_value, fc.get_no_mean_local_maxima_upsurges
    # , fc.get_no_mean_local_minima_downslides, fc.get_difference_of_mins
    # , fc.get_difference_of_maxs, fc.get_difference_of_means, fc.get_difference_of_stds
    # , fc.get_difference_of_vars, fc.get_difference_of_medians, fc.get_dderivative_mean
    # , fc.get_gderivative_mean, fc.get_dderivative_stddev, fc.get_gderivative_stddev
    # , fc.get_dderivative_skewness, fc.get_gderivative_skewness, fc.get_dderivative_kurtosis
    # , fc.get_gderivative_kurtosis, fc.get_linear_weighted_average, fc.get_quadratic_weighted_average
    # , fc.get_average_absolute_change, fc.get_average_absolute_derivative_change
    # , fc.get_positive_fraction, fc.get_negative_fraction, fc.get_last_value
    # , fc.get_sum_of_last_K, fc.get_mean_last_K, fc.get_slope_of_longest_mono_increase
    # , fc.get_slope_of_longest_mono_decrease, fc.get_avg_mono_increase_slope
    # , fc.get_avg_mono_decrease_slope
                           ]

'''
    List of first 24 physical parameters with numerical values
'''
CANDIDATE_PHYS_PARAMETERS = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ'
    , 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH'
    , 'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE'
                             ]

'''
    Path to input and output
'''

IN_PATH_TO_MVTS_FL = 'data/input/time_segmented_partition3_subset/FL'
IN_PATH_TO_MVTS_NF = 'data/input/time_segmented_partition3_subset/NF'
IN_PATH_TO_MVTS_FL1 = '../partition1/FL'
IN_PATH_TO_MVTS_NF1 = '../partition1/NF'
IN_PATH_TO_MVTS_FL2 = '../partition2/FL'
IN_PATH_TO_MVTS_NF2 = '../partition2/NF'
IN_PATH_TO_MVTS_FL3 = '../partition3/FL'
IN_PATH_TO_MVTS_NF3 = '../partition3/NF'
IN_PATH_TO_MVTS_FL4 = '../partition4/FL'
IN_PATH_TO_MVTS_NF4 = '../partition4/NF'
IN_PATH_TO_MVTS_FL5 = '../partition5/FL'
IN_PATH_TO_MVTS_NF5 = '../partition5/NF'
OUT_PATH_TO_RAW_FEATURES = 'data/output/'
