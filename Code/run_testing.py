#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Data_2d import ProfilesData
from cWGAN import GenerativeAI
import os
import numpy as np

#%%
def create_path(fpath):
    if os.path.exists(fpath):
        return
    os.makedirs(fpath)
    
#%% Function to train WGAN
def run_testing(split_test_data, 
                 split_train_data, 
                 split_validation_data, 
                 parser):
    
    outfolder = os.path.join(parser.params['database']['outfolder'],
                             parser.params['database']['biome_type'],
                             "soil_type_"+str(parser.params['database']['soil_type']))
    
    # Create testing folder
    outfolder_test = os.path.join(outfolder,'testing')
    create_path(outfolder_test)
    
    #%% Run over all folds
    real_test_data = []
    fake_test_data = []
    realizations = int(parser.params['testing']['realizations'])
    
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
                                 beta                   = float(parser.params['training']['beta']),
                                 output_dir             = outdir, 
                                 data_dir               = save_folder
                                 )
        # test
        real_s, fake_s, epoch = gan_model.test(mode  = parser.params['testing']['mode'], 
                                               epoch = parser.params['testing']['epochs'][fold],
                                               realizations  = parser.params['testing']['realizations']
                                               )
        real_test_data.append(real_s)
        fake_test_data.append(fake_s)
        
        # Stack realizations in current fold to plot summary
        fake_s_stacked = np.empty((0, gan_model.max_len))
        for r in range(realizations):
            fake_s_stacked = np.vstack([fake_s_stacked, fake_s[r]])
        
        # Plot fold summary over all realizations
        outname = 'testing_mode_' \
                + str(parser.params['testing']['mode']) \
                + '_fold_' + str(fold)
        gan_model.plot_summary(real_s, fake_s_stacked, 
                               outname = outname, 
                               outfolder = outfolder_test, 
                               epoch = epoch,
                               density = True
                               )
        
    #%% Stack realizations and folds to plot global summary
    fake_s_stacked_global = np.empty((0, gan_model.max_len))
    for fold in range(folds_start, folds_end):
        fake_test_data_fold = fake_test_data[fold]
        fake_s_stacked = np.empty((0, gan_model.max_len))
        for r in range(realizations):
            fake_s_stacked = np.vstack([fake_s_stacked, fake_test_data_fold[r]]) 
        fake_s_stacked_global = np.vstack([fake_s_stacked_global, fake_s_stacked])
        
    #% Plot summary over all folds and realizations
    outname = 'testing_global_mode_' \
            + str(parser.params['testing']['mode'])
    gan_model.plot_summary(real_test_data[0], fake_s_stacked_global, 
                           outname = outname, 
                           outfolder = outfolder_test, 
                           epoch = 0,
                           density = True
                           )
    
    #%% Compute uppper and lower percentiles for each test site
    # fake_s_stacked_global = np.empty((gan_model.N_test, gan_model.max_len,0))
    lower = []
    upper = []
    width = []
    width_per_depth = []
    median = []
    coverage_list = []
    coverage_per_depth = []
    fake_s_stacked_list = []
    for fold in range(folds_start, folds_end):
        
        fake_s_stacked = np.stack(fake_test_data[fold], axis=2)
        
        # Compute lower and upper limits of CI for each fold
        lower.append(np.percentile(fake_s_stacked, 2.5,  axis=2))
        upper.append(np.percentile(fake_s_stacked, 97.5, axis=2))
        width.append(np.percentile(fake_s_stacked, 97.5, axis=2) - 
                     np.percentile(fake_s_stacked, 2.5,  axis=2) 
                    )
        median.append(np.percentile(fake_s_stacked, 50.0, axis=2))
        
        # Compute averge width per depth per fold
        width_per_depth.append(np.mean(width[fold], axis = 0))
        
        # Compute coverage
        # Note that all real_test_data on each fold in the list are the same
        coverage = (real_test_data[0] >= lower[fold]) & (real_test_data[0] <= upper[fold])   # shape (N_test, max_len)
        coverage_list.append(coverage)
        coverage_per_depth.append(np.mean(coverage, axis=0))
        print("\n")
        print(f"Coverage in fold {fold}:",' '.join(f'{x:.3f}' for x in coverage_per_depth[fold]))
        print(f"CI Width in fold {fold}:",' '.join(f'{x:.3f}' for x in width_per_depth[fold]))
    
        # Create a list to then stack over all folds
        fake_s_stacked_list.append(fake_s_stacked)

    # Stack all folds and compute lower and upper limits of CI
    fake_s_stacked_global = np.concatenate(fake_s_stacked_list, axis=2)
    lower_global = np.percentile(fake_s_stacked_global, 2.5, axis=2)
    upper_global = np.percentile(fake_s_stacked_global, 97.5, axis=2)
    width_global = upper_global - lower_global
    
    median_global = np.percentile(fake_s_stacked_global, 50.0 , axis=2)
    mean_global = np.mean(fake_s_stacked_global, axis=2)
    
    median_per_depth_global = np.percentile(median_global, 50.0 , axis = 0)
    mean_per_depth_global = np.mean(mean_global, axis = 0)
    
    coverage_global = (real_test_data[0] >= lower_global) & (real_test_data[0] <= upper_global)   # shape (N_test, max_len)
    coverage_per_depth_global = np.mean(coverage_global, axis=0)
    
    # Compute average width per depth (over all folds)
    width_per_depth_global = np.mean(width_global, axis = 0)
    width_percent_per_depth_global = 100 * width_per_depth_global / median_per_depth_global
    # width_percent_per_depth_global = 100 * width_per_depth_global / mean_per_depth_global
    upper_per_depth_global = np.mean(upper_global, axis = 0)
    lower_per_depth_global = np.mean(lower_global, axis = 0)
    
    # Compute RMSE per depth
    rmse_per_depth_global = np.sqrt(np.mean((real_test_data[0] - median_global)**2,axis=0)) # shape (max_len)
    # rmse_per_depth_global = np.sqrt(np.mean((real_test_data[0] - mean_global)**2,axis=0))
    
    # Compute RMSE%
    # real_test_data_safe = np.where(real_test_data[0]==0, 1e-6, real_test_data[0])
    # median_global_safe = np.where(median_global==0, 1e-6, median_global)
    # mean_global_safe = np.where(mean_global==0, 1e-6, mean_global)
    
    median_real_test_data = np.percentile(real_test_data[0], 50.0 , axis = 0)
    mean_real_test_data = np.mean(real_test_data[0], axis = 0)
    
    rmse_percent_per_depth_global = 100 * rmse_per_depth_global / median_real_test_data
    # rmse_percent_per_depth_global = 100 * rmse_per_depth_global / mean_real_test_data
    
    # Output summary statistics
    print("\n")
    print("Coverage:",' '.join(f'{x:.3f}' for x in coverage_per_depth_global))
    print("CI Width:",' '.join(f'{x:.3f}' for x in width_per_depth_global))
    print("CI Width %:",' '.join(f'{x:.3f}' for x in width_percent_per_depth_global))
    print("CI upper bound:",' '.join(f'{x:.3f}' for x in upper_per_depth_global))
    print("CI lower bound:",' '.join(f'{x:.3f}' for x in lower_per_depth_global))
    
    print("\n")
    print("rmse per depth:",' '.join(f'{x:.2f}' for x in rmse_per_depth_global))
    print("rmse% per depth:",' '.join(f'{x:.2f} %' for x in rmse_percent_per_depth_global))

    print("\n")
    print("real mean per depth:",' '.join(f'{x:.2f}' for x in mean_real_test_data))
    print("fake mean per depth:",' '.join(f'{x:.2f}' for x in mean_per_depth_global))

    print("\n")
    print("real median per depth:",' '.join(f'{x:.2f}' for x in median_real_test_data))
    print("fake median per depth:",' '.join(f'{x:.2f}' for x in median_per_depth_global))
    
    # Output testing summary to file in testing folder
    names = np.array(["Coverage:", 
                      "CI Width:", 
                      "CI Width %:",
                      "CI upper bound:",
                      "CI lower bound:",
                      "rmse per depth:",
                      "rmse% per depth:",
                      "real mean per depth:",
                      "fake mean per depth:",
                      "real median per depth:",
                      "fake median per depth:"])[:, None]
    
    data  = np.vstack((coverage_per_depth_global, 
                       width_per_depth_global, 
                       width_percent_per_depth_global,
                       upper_per_depth_global,
                       lower_per_depth_global,
                       rmse_per_depth_global,
                       rmse_percent_per_depth_global,
                       mean_real_test_data,
                       mean_per_depth_global,
                       median_real_test_data,
                       median_per_depth_global
                       ))
    
    out = np.empty((data.shape[0], data.shape[1] + 1), dtype=object)
    out[:, 0]  = names.ravel()
    out[:, 1:] = data
    fmt = ["%s"] + ["%.3f"] * data.shape[1]      # change 3 to desired decimals
    
    np.savetxt(os.path.join(outfolder_test,
              f"Summary_mode_{parser.params['testing']['mode']}_realizations_{parser.params['testing']['realizations']}.csv"), 
               out, 
               delimiter=" ", 
               fmt=fmt)
    
    #%%
    return real_test_data, fake_test_data
