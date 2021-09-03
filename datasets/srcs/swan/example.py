import os
import CONSTANTS as CONST
import features.feature_collection as fc
from features.feature_extractor import FeatureExtractor

# Choose statistical features and physical parameters
features_list = [fc.get_min, fc.get_max, fc.get_median]
params_name_list = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ']

# Input and output path
#path_to_root = os.path.join('..', CONST.IN_PATH_TO_MVTS_FL)
#path_to_dest = os.path.join('..', CONST.OUT_PATH_TO_RAW_FEATURES)
#path_to_root =  CONST.IN_PATH_TO_MVTS_FL

path_to_root =  CONST.IN_PATH_TO_MVTS_FL2
path_to_dest =  CONST.OUT_PATH_TO_RAW_FEATURES
output_filename = 'raw_features_p2_FL.csv'

# Extract features
pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.calculate_all(features_list, params_name_list=params_name_list)

path_to_root =  CONST.IN_PATH_TO_MVTS_NF2
path_to_dest =  CONST.OUT_PATH_TO_RAW_FEATURES
output_filename = 'raw_features_p2_NF.csv'

# Extract features
pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.calculate_all(features_list, params_name_list=params_name_list)

path_to_root =  CONST.IN_PATH_TO_MVTS_FL3
path_to_dest =  CONST.OUT_PATH_TO_RAW_FEATURES
output_filename = 'raw_features_p3_FL.csv'

# Extract features
pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.calculate_all(features_list, params_name_list=params_name_list)

path_to_root =  CONST.IN_PATH_TO_MVTS_NF3
path_to_dest =  CONST.OUT_PATH_TO_RAW_FEATURES
output_filename = 'raw_features_p3_NF.csv'

# Extract features
pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.calculate_all(features_list, params_name_list=params_name_list)

path_to_root =  CONST.IN_PATH_TO_MVTS_FL4
path_to_dest =  CONST.OUT_PATH_TO_RAW_FEATURES
output_filename = 'raw_features_p4_FL.csv'

# Extract features
pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.calculate_all(features_list, params_name_list=params_name_list)

path_to_root =  CONST.IN_PATH_TO_MVTS_NF4
path_to_dest =  CONST.OUT_PATH_TO_RAW_FEATURES
output_filename = 'raw_features_p4_NF.csv'

# Extract features
pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.calculate_all(features_list, params_name_list=params_name_list)

path_to_root =  CONST.IN_PATH_TO_MVTS_FL5
path_to_dest =  CONST.OUT_PATH_TO_RAW_FEATURES
output_filename = 'raw_features_p5_FL.csv'

# Extract features
pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.calculate_all(features_list, params_name_list=params_name_list)

path_to_root =  CONST.IN_PATH_TO_MVTS_NF5
path_to_dest =  CONST.OUT_PATH_TO_RAW_FEATURES
output_filename = 'raw_features_p5_NF.csv'

# Extract features
pc = FeatureExtractor(path_to_root, path_to_dest, output_filename)
pc.calculate_all(features_list, params_name_list=params_name_list)
