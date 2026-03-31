#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Data_2d import ProfilesData
from cWGAN import GenerativeAI
import os

#%% Function to train WGAN
def run_training(split_test_data, 
                 split_train_data, 
                 split_validation_data, 
                 parser):
    
    outfolder = os.path.join(parser.params['database']['outfolder'],
                             parser.params['database']['biome_type'],
                             "soil_type_"+str(parser.params['database']['soil_type']))
    
    folds_start = int(parser.params['validation']['folds'][0])
    folds_end = int(parser.params['validation']['folds'][1])
    for fold in range(folds_start, folds_end):
    
        # Get train and test data for current fold
        train_data = split_train_data[fold]
        validation_data = split_validation_data[fold]
        test_data = split_test_data

        profiles_name = parser.params['database']['biome_type'] \
                      + "_soil_type_" + str(parser.params['database']['soil_type']) \
                      + "_fold_" + str(fold)
           
        profiles = ProfilesData(profiles_name, 
                                train_data, 
                                validation_data, 
                                test_data, 
                                scaling = parser.params['scaling']
                                )
        
        # Instantiate GenAI
        outdir = os.path.join(outfolder,"fold_" + str(fold))
        save_folder = outdir
        gan_model = GenerativeAI(profiles               = profiles, # This is just a name now
                                 latent_dim             = int(parser.params['wgan']['latent_dim']), 
                                 gen_hidden_units       = parser.params['wgan']['gen_hidden_units'], 
                                 disc_hidden_units      = parser.params['wgan']['disc_hidden_units'], 
                                 cond_loc_hidden_units  = parser.params['wgan']['cond_loc_hidden_units'],
                                 cond_env_hidden_units  = parser.params['wgan']['cond_env_hidden_units'],
                                 cond_alt_hidden_units  = parser.params['wgan']['cond_alt_hidden_units'],
                                 cond_fuse_hidden_units = parser.params['wgan']['cond_fuse_hidden_units'],
                                 learning_rate          = float(parser.params['training']['learning_rate']), 
                                 min_learning_rate      = float(parser.params['training']['min_learning_rate']), 
                                 decrease_factor_lr     = float(parser.params['training']['decrease_factor_lr']), 
                                 patience_lr            = int(parser.params['training']['patience_lr']), 
                                 patience_es            = int(parser.params['training']['patience_es']), 
                                 batch_size             = int(parser.params['training']['batch_size']),
                                 nCritic                = int(parser.params['training']['nCritic']),
                                 beta                   = float(parser.params['training']['beta']),
                                 output_dir             = outdir, 
                                 data_dir               = save_folder
                                 )
        # Train
        gan_model.train(nEpochs  = int(parser.params['training']['nEpochs']), 
                        nEval    = int(parser.params['training']['nEval'])
                        )    

