#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Data_2d import SubGroupData, KFold_Profiles
from Parser import Parser
from run_training import run_training
from run_testing import run_testing
import os
import sys
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)

#%% Read input yaml file
parser = Parser()

#%% Filter Data
outfolder = os.path.join(parser.params['database']['outfolder'],
                         parser.params['database']['biome_type'],
                         "soil_type_"+str(parser.params['database']['soil_type']))
soc = SubGroupData(parser.params['database']['infile'], outfolder)
_ = soc.filter_by("biome_type", parser.params['database']['biome_type'])
data = soc.filter_by("Soil_Type", int(parser.params['database']['soil_type']))

#% Standardize data to WoSiS depths
data_standard = soc.standardize_depths(std_bins = None, 
                                       outlier_cutoff = parser.params['database']['outlier_cutoff'])

#%% Kfold class
kf = KFold_Profiles(k       = int(parser.params['validation']['folds'][1]), 
                    test    = float(parser.params['validation']['test_frac']), 
                    shuffle = bool(parser.params['validation']['shuffle']), 
                    seed    = int(parser.params['validation']['seed']))
split_test_data, split_train_data, split_validation_data = kf.split(data_standard)

#%% Run training or testing
if (str(parser.params['mode']) == 'training'):
    
    #% Save parameters to yaml file inside folder
    parser.save_parameters(outfolder = outfolder)
    
    run_training(split_test_data, 
                 split_train_data, 
                 split_validation_data, 
                 parser)
    
elif (str(parser.params['mode']) == 'testing'):
    
    run_testing(split_test_data, 
                split_train_data, 
                split_validation_data, 
                parser)
    
else:
    print("Only modes allowed are 'training' and 'testing'")
    sys.exit()
