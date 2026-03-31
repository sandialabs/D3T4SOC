# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy as sp
from time import time
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# pio.renderers.default='browser'

import sys
from collections import deque

#%%
@tf.function
def gaussian_kernel(x, y, sigma=0.5):
    x = tf.expand_dims(x,1)   # (m,1)
    y = tf.expand_dims(y,0)   # (1,n)
    return tf.exp(-tf.square(x-y)/(2*sigma**2))

@tf.function
def mmd_penalty(real_vals, fake_vals):
    K_rr = gaussian_kernel(real_vals, real_vals)
    K_ff = gaussian_kernel(fake_vals, fake_vals)
    K_rf = gaussian_kernel(real_vals, fake_vals)
    return tf.reduce_mean(K_rr) + tf.reduce_mean(K_ff) - 2*tf.reduce_mean(K_rf)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# The generator-discriminator model
class GenerativeAI:
    def __init__(self, 
                 profiles, 
                 latent_dim=100, 
                 gen_hidden_units=[32,32,32], 
                 disc_hidden_units=[32,32,32], 
                 cond_loc_hidden_units = [32,16],
                 cond_env_hidden_units = [32,16],
                 cond_alt_hidden_units = [32,16],
                 cond_fuse_hidden_units = [32,16],
                 weight_clip=0.01, 
                 learning_rate=1e-4, 
                 min_learning_rate = 1e-4, 
                 decrease_factor_lr = 0.5, 
                 patience_lr = 100,
                 patience_es = 200,
                 batch_size = 256,
                 nCritic = 5,
                 beta = 1,
                 output_dir='./', 
                 data_dir=None):
        
        self.profiles = profiles
        # Set some variables from profile data
        self.N = profiles.N 
        self.N_val = profiles.N_val 
        self.N_test = profiles.N_test 
        self.max_len = profiles.max_len
        
        self.latent_dim = latent_dim
        
        if batch_size == 0:
            self.batch_size = self.N
        else:
            self.batch_size = batch_size
        
        self.nCritic = nCritic
                
        self.learning_rate_list = []
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decrease_factor_lr = decrease_factor_lr
        self.patience_lr = patience_lr
        self.patience_es = patience_es
        
        self.weight_clip = weight_clip
        self.beta = beta
        
        self.generator_opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
             
        self.cond_embed_gen = self.__make_cond_embed__(cond_loc_hidden_units,
                                                       cond_env_hidden_units,
                                                       cond_alt_hidden_units,
                                                       cond_fuse_hidden_units)
        self.cond_embed_disc = self.__make_cond_embed__(cond_loc_hidden_units,
                                                       cond_env_hidden_units,
                                                       cond_alt_hidden_units,
                                                       cond_fuse_hidden_units)
        
        self.generator = self.__generator__(gen_hidden_units)
        self.discriminator = self.__discriminator__(disc_hidden_units)

        self.time = []
        self.epoch = []
        self.dloss=[]
        self.gloss=[]
        self.gloss_global=[]
        self.gloss_val=[]
        self.W = []
        self.W_val = []
        self.rmse = []
        self.rmse_val = []
        self.rmse_smooth_val=[]
        self.rmse_percent = []
        self.rmse_percent_val = []
        self.rmse_per_depth = np.empty((0, self.max_len))
        self.rmse_per_depth_val = np.empty((0, self.max_len))
        self.rmse_percent_per_depth = np.empty((0, self.max_len))
        self.rmse_percent_per_depth_val = np.empty((0, self.max_len))
        self.mmds = []
        self.mmds_val = []
        
        self.output_dir = output_dir
        self.img_dir = os.path.join(self.output_dir, 'img')
        self.__create_path__(self.img_dir)
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.__create_path__(self.model_dir)
        self.summary_dir = os.path.join(self.output_dir, 'summary')
        self.__create_path__(self.summary_dir)
        
        if data_dir is None:
            self.data_dir = os.path.join(self.output_dir, 'data')
        else:
            self.data_dir = os.path.join(output_dir, "Data")
            
        self.__create_path__(self.data_dir)
        
        self.__wgan_summary__()
        
    # Save WGAN summary
    def __wgan_summary__(self):
        outfile = os.path.join(self.summary_dir, 'generator_summary.txt')
        with open(outfile, 'w') as f:
            self.generator.summary(print_fn=lambda x: f.write(x + '\n'))
        
        outfile = os.path.join(self.summary_dir, 'discriminator_summary.txt')
        with open(outfile, 'w') as f:
            self.discriminator.summary(print_fn=lambda x: f.write(x + '\n'))
            
        outfile = os.path.join(self.summary_dir, 'cond_embed_gen_summary.txt')
        with open(outfile, 'w') as f:
            self.cond_embed_gen.summary(print_fn=lambda x: f.write(x + '\n'))
            
        outfile = os.path.join(self.summary_dir, 'cond_embed_disc_summary.txt')
        with open(outfile, 'w') as f:
            self.cond_embed_disc.summary(print_fn=lambda x: f.write(x + '\n'))
                        
    # Create folder along the path if does not exists
    def __create_path__(self, fpath):
        if os.path.exists(fpath):
            return
        os.makedirs(fpath)

    # Condition-Embedding MLP
    def __make_cond_embed__(self, cond_loc_hidden_units, 
                                  cond_env_hidden_units, 
                                  cond_alt_hidden_units,
                                  cond_fuse_hidden_units): 
        inputs  = []
        emb_list = []
        # 1) Location sub-embed
        loc_in = tf.keras.Input((2,), name='loc_in')
        x = tf.keras.layers.Dense(cond_loc_hidden_units[0], activation='relu')(loc_in)
        if (len(cond_loc_hidden_units) > 1):
            for i in range(1,len(cond_loc_hidden_units)):
                x = tf.keras.layers.Dense(cond_loc_hidden_units[i], activation='relu')(x)
        inputs.append(loc_in)
        emb_list.append(x)
            
        # 2) Environment sub-embed
        env_in = tf.keras.Input((2,), name='env_in')
        y = tf.keras.layers.Dense(cond_env_hidden_units[0], activation='relu')(env_in)
        if (len(cond_env_hidden_units) > 1):
            for i in range(1,len(cond_env_hidden_units)):
                y = tf.keras.layers.Dense(cond_env_hidden_units[i], activation='relu')(y)
        inputs.append(env_in)
        emb_list.append(y)
            
        # 3) Altitude sub-embed
        alt_in = tf.keras.Input((1,), name='alt_in')
        z = tf.keras.layers.Dense(cond_alt_hidden_units[0], activation='relu')(alt_in)
        if (len(cond_alt_hidden_units) > 1):
            for i in range(1,len(cond_alt_hidden_units)):
                z = tf.keras.layers.Dense(cond_alt_hidden_units[i], activation='relu')(z)
        inputs.append(alt_in)
        emb_list.append(z)
    
        m = tf.keras.layers.Concatenate()(emb_list)     
        for i in range(len(cond_fuse_hidden_units)):
            m = tf.keras.layers.Dense(cond_fuse_hidden_units[i], activation='relu')(m)
    
        return tf.keras.Model(inputs, m, name='cond_embed')

    # The generator model
    def __generator__(self, gen_hidden_units=[32,32,32]):
        noise_in = tf.keras.Input(shape=(self.latent_dim,), name='noise')
        cond_loc_in  = tf.keras.Input(shape=(2,), name='cond_loc')
        cond_env_in  = tf.keras.Input(shape=(2,), name='cond_env')
        cond_alt_in  = tf.keras.Input(shape=(1,), name='cond_alt')
        cond_gen_inputs = [cond_loc_in,cond_env_in,cond_alt_in]
        geo_gen_emb    = self.cond_embed_gen(cond_gen_inputs)
        
        x = tf.keras.layers.Concatenate()([noise_in, geo_gen_emb])
        for i in range(len(gen_hidden_units)):
            x = tf.keras.layers.Dense(gen_hidden_units[i], activation='relu')(x)
        soc_out   = tf.keras.layers.Dense(self.max_len, activation='tanh', name='soc_out')(x)

        return tf.keras.Model([noise_in] + cond_gen_inputs, soc_out, name='generator')

    # The discriminator model
    def __discriminator__(self, disc_hidden_units=[32,32,32]): 
        soc_in    = tf.keras.Input(shape=(self.max_len,), name='soc_in')
        cond_loc_in2  = tf.keras.Input(shape=(2,), name='cond_loc2')
        cond_env_in2  = tf.keras.Input(shape=(2,), name='cond_env2')
        cond_alt_in2  = tf.keras.Input(shape=(1,), name='cond_alt2')
        cond_disc_inputs = [cond_loc_in2,cond_env_in2,cond_alt_in2]
        geo_disc_emb    = self.cond_embed_disc(cond_disc_inputs)
        
        d = tf.keras.layers.Concatenate()([soc_in,geo_disc_emb])
        for i in range(len(disc_hidden_units)):
            d = tf.keras.layers.Dense(disc_hidden_units[i])(d)
            d = tf.keras.layers.LeakyReLU(0.2)(d)
        logit = tf.keras.layers.Dense(1, name='wgan_logit')(d)
        
        return tf.keras.Model(inputs= [soc_in] + cond_disc_inputs, outputs=logit, name='critic')
    
    @tf.function
    def __discriminator_loss__(self, real, fake):
        return tf.reduce_mean(fake) - tf.reduce_mean(real)
    
    @tf.function
    def __generator_loss__(self, fake):
        return -tf.reduce_mean(fake)
    
    #%% train step
    @tf.function
    def train_step(self, real_s, cond_loc, cond_env, cond_alt):
        
        cond_inputs = [cond_loc, cond_env, cond_alt]
        
        # multiple critic updates
        for _ in range(self.nCritic):
            noise = tf.random.normal([self.batch_size, self.latent_dim])
            with tf.GradientTape() as tape:
                fake_s    = self.generator([noise] + cond_inputs, training=True)
                real_log  = self.discriminator([real_s] + cond_inputs, training=True)
                fake_log  = self.discriminator([fake_s] + cond_inputs, training=True)
                d_loss = self.__discriminator_loss__(real_log, fake_log)

            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
            # weight clipping
            for w in self.discriminator.trainable_variables:
                w.assign(tf.clip_by_value(w, -self.weight_clip, self.weight_clip))
        
        # generator update MMD PENALTY
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            fake_s    = self.generator([noise] + cond_inputs, training=True)   # (batch, max_len)
            fake_log  = self.discriminator([fake_s] + cond_inputs, training=True)
            wgan_loss = self.__generator_loss__(fake_log)

            if (self.beta == 0.0):
                g_loss = wgan_loss
            else:
                # use tf.map_fn to compute MMD at each depth index
                idxs = tf.range(self.max_len)
                # map_fn will call mmd_penalty for each self.max_len
                mmds = tf.map_fn(
                    lambda k: mmd_penalty(real_s[:, k], fake_s[:, k]),
                    idxs,
                    fn_output_signature=tf.float32
                )  # shape (self.max_len,)
        
                # average MMD across all depths
                mmd_all = -self.__generator_loss__(mmds)
    
                # total generator loss
                g_loss = wgan_loss + self.beta * mmd_all
    
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
    
        return d_loss, g_loss

    #%%
    def __save_genertor_model__(self, epoch):
        self.generator.save(os.path.join(self.model_dir, f'{self.profiles.profile_ID}_{epoch}'))
        
    #%%
    @tf.function
    def __get_synthetic_data__(self, N, conds_loc, conds_env, conds_alt):
        cond_inputs = [conds_loc, conds_env, conds_alt]
        noise = tf.random.normal([N, self.latent_dim])
        return self.generator([noise] + cond_inputs, training=False)
    
    #%%
    @tf.function
    def __get_synthetic_gloss__(self, fake_s, conds_loc, conds_env, conds_alt):
        cond_inputs = [conds_loc, conds_env, conds_alt]
        fake_log  = self.discriminator([fake_s] + cond_inputs, training=False)
        return self.__generator_loss__(fake_log)


    #%%
    # @tf.function
    # def __median_along_axis_0__(tensor):
    #     # Sort along axis=0 (rows)
    #     sorted_tensor = tf.sort(tensor, axis=0)
    #     n = tf.shape(tensor)[0]
    #     mid = n // 2
    #     is_odd = tf.math.floormod(n, 2) == 1
    
    #     def get_odd_median():
    #         return sorted_tensor[mid, :]
    
    #     def get_even_median():
    #         return (sorted_tensor[mid-1, :] + sorted_tensor[mid, :]) / 2.0
    
    #     median = tf.where(is_odd, get_odd_median(), get_even_median())
    #     return median
    
    #%%
    # @tf.function
    # def __compute_rmse__(self, real_s, fake_s):
        
    #     # median = tf.math.reduce_median(real_s, axis=0)
    #     # median = self.__median_along_axis_0__(real_s)
    #     mean = tf.reduce_mean(real_s, axis=0)

    #     rmse_per_depth  = tf.sqrt(tf.reduce_mean((real_s - fake_s)**2, axis=0))
    #     # rmse_per_depth  = tf.sqrt(tf.reduce_mean(((real_s - fake_s)/real_s)**2, axis=0))
        
    #     # rmse_percent_per_depth = 100*rmse_per_depth/median
    #     rmse_percent_per_depth = 100*rmse_per_depth/mean
    #     # rmse_percent_per_depth = 100*rmse_per_depth

    #     rmse = tf.reduce_mean(rmse_per_depth)
    #     rmse_percent = tf.reduce_mean(rmse_percent_per_depth)

    #     return rmse, rmse_per_depth, rmse_percent, rmse_percent_per_depth

    #%%
    @tf.function
    def __compute_rmse__(self, real_s, fake_s):
        
        # median = tf.math.reduce_median(real_s, axis=0)
        # median = self.__median_along_axis_0__(real_s)
        # mean = tf.reduce_mean(real_s, axis=0)


        rmse_per_depth  = tf.sqrt(tf.reduce_mean((real_s - fake_s)**2, axis=0))
        # rmse_per_depth  = tf.sqrt(tf.reduce_mean(((real_s - fake_s)/real_s)**2, axis=0))
        
        
        
        
        
        # Create a mask to ignore where real_s == 0
        mask = tf.not_equal(real_s, 0)  # shape (N, K), dtype=bool

        # Compute squared relative error only where real_s ≠ 0
        relative_sq_error = tf.where(mask,
                                     tf.square((real_s - fake_s) / real_s),
                                     tf.zeros_like(real_s))
        
        # Count how many valid (non-zero) entries per column
        valid_counts = tf.reduce_sum(tf.cast(mask, tf.float32), axis=0)
        
        # Avoid division by zero (clip minimum count to 1)
        valid_counts = tf.maximum(valid_counts, 1.0)
        
        # Compute masked mean and take square root
        rmse_percent_per_depth = tf.sqrt(tf.reduce_sum(relative_sq_error, axis=0) / valid_counts) * 100

        # rmse_percent_per_depth  = 100*tf.sqrt(tf.reduce_mean(((real_s - fake_s)/real_s)**2, axis=0))
        # rmse_percent_per_depth = 100*rmse_per_depth/mean
        # rmse_percent_per_depth = 100*rmse_per_depth

        # rmse = tf.reduce_mean(rmse_per_depth)
        # rmse_percent = tf.reduce_mean(rmse_percent_per_depth)
        
        
        
        
        ###
        real_means = tf.reduce_mean(real_s, axis=0)
        fake_means = tf.reduce_mean(fake_s, axis=0)
        
        rmse = tf.sqrt( tf.reduce_mean( tf.square(real_means - fake_means) ) )
        # rmse_percent = rmse/tf.reduce_mean(mean)
        
        # rmse = tf.sqrt( tf.reduce_mean( tf.square(real_means - fake_means)/real_means ) )
        rmse_percent = 100*tf.sqrt( tf.reduce_mean( tf.square(real_means - fake_means)/real_means ) )

        return rmse, rmse_per_depth, rmse_percent, rmse_percent_per_depth
    
    #%%
    @tf.function
    def __compute_mmd__(self, real_s, fake_s):
        idxs = tf.range(self.max_len)
        mmds = tf.map_fn(lambda k: mmd_penalty(real_s[:,k], fake_s[:,k]),
                         idxs, fn_output_signature=tf.float32
                         ) # shape (max_len,)
        return tf.reduce_mean(mmds)
    
    #%%
    def __wasserstein_distance__(self, real, fake):
        W = 0
        for i in range(self.max_len):
            W += sp.stats.wasserstein_distance(real[:,i], fake[:,i])
        return W / self.max_len
    
    #%%
    def __save_input_data__(self):

        # Save training data
        np.savetxt(os.path.join(self.data_dir, f'train_profile_ids_{self.profiles.profile_ID}.csv')     , self.profiles.profile_ids, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'train_soc_scaled_{self.profiles.profile_ID}.csv')      , self.profiles.soc_scaled, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'train_depth_scaled_{self.profiles.profile_ID}.csv')    , self.profiles.depth_scaled, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'train_conds_loc_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_loc_scaled, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'train_conds_env_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_env_scaled, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'train_conds_alt_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_alt_scaled, delimiter=',', fmt='%.25e')

        # Save validation data
        np.savetxt(os.path.join(self.data_dir, f'validation_profile_ids_{self.profiles.profile_ID}.csv')     , self.profiles.profile_ids_val, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'validation_soc_scaled_{self.profiles.profile_ID}.csv')      , self.profiles.soc_scaled_val, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'validation_depth_scaled_{self.profiles.profile_ID}.csv')    , self.profiles.depth_scaled_val, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'validation_conds_loc_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_loc_scaled_val, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'validation_conds_env_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_env_scaled_val, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'validation_conds_alt_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_alt_scaled_val, delimiter=',', fmt='%.25e')

        # Save test data
        np.savetxt(os.path.join(self.data_dir, f'test_profile_ids_{self.profiles.profile_ID}.csv')     , self.profiles.profile_ids_test, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'test_soc_scaled_{self.profiles.profile_ID}.csv')      , self.profiles.soc_scaled_test, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'test_depth_scaled_{self.profiles.profile_ID}.csv')    , self.profiles.depth_scaled_test, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'test_conds_loc_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_loc_scaled_test, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'test_conds_env_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_env_scaled_test, delimiter=',', fmt='%.25e')
        np.savetxt(os.path.join(self.data_dir, f'test_conds_alt_scaled_{self.profiles.profile_ID}.csv'), self.profiles.conds_alt_scaled_test, delimiter=',', fmt='%.25e')
        
        # Save scaling data
        self.profiles.scaling_factors.to_csv(os.path.join(self.data_dir, f'scaling_factors_{self.profiles.profile_ID}.csv'), float_format='%.25e')
        self.profiles.centering_factors.to_csv(os.path.join(self.data_dir, f'centering_factors_{self.profiles.profile_ID}.csv'), float_format='%.25e')

    #%%
    def __plot_metrics__(self):

        # ##### Plot metrics and distances with plotly ####
        fig = go.Figure()
        
        lw = 2
        # lr
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.learning_rate_list,
            mode="lines",
            name="learning rate ",
            line=dict(color='red', width=lw+1),
        )) 
        
        # Wasserstein distances
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.W,
            mode="lines",
            name="W (train)",
            line=dict(color='red', width=lw),
        )) 

        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.W_val,
            mode="lines",
            name="W (validation)",
            line=dict(color='blue', width=lw),
        )) 
        
        # Losses
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.gloss,
            mode="lines",
            name="gloss (train)",
            line=dict(color='green', width=lw),
        )) 
        
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.gloss_global,
            mode="lines",
            name="gloss (train global)",
            line=dict(color='orange', width=lw),
        )) 
        
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.gloss_val,
            mode="lines",
            name="gloss (validation)",
            line=dict(color='black', width=lw),
        )) 
        
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.dloss,
            mode="lines",
            name="dloss (train)",
            line=dict(color='magenta', width=lw),
        )) 
        
        pos_labels = ["0-5 cm","5-15 cm", "15-30 cm", "30-60 cm", "60-100 cm", "100-200 cm"]
        
        # RMSE Training
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.rmse,
            mode="lines",
            name="rmse (train)",
            line=dict(color='magenta', width=lw+2),
        ))   
        for j in range(self.max_len):
            fig.add_trace(go.Scatter(
                x=self.epoch,
                y=self.rmse_per_depth[:,j],
                mode="lines",          
                name="rmse (train): "+str(pos_labels[j]),
                line=dict(color='magenta', width=lw),
            )) 
            
        # RMSE Validation
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.rmse_smooth_val,
            mode="lines",
            name="rmse smooth (validation)",
            line=dict(color='red', width=lw+2),
        ))   
        
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.rmse_val,
            mode="lines",
            name="rmse (validation)",
            line=dict(color='orange', width=lw+2),
        ))      
        for j in range(self.max_len):
            fig.add_trace(go.Scatter(
                x=self.epoch,
                y=self.rmse_per_depth_val[:,j],
                mode="lines",
                name="rmse (validation): "+str(pos_labels[j]),
                line=dict(color='orange', width=lw),
            ))    
            
        # RMSE% Training
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.rmse_percent,
            mode="lines",
            name="rmse% (train)",
            line=dict(color='magenta', width=lw+2),
        ))   
        for j in range(self.max_len):
            fig.add_trace(go.Scatter(
                x=self.epoch,
                y=self.rmse_percent_per_depth[:,j],
                mode="lines",          
                name="rmse% (train): "+str(pos_labels[j]),
                line=dict(color='magenta', width=lw),
            ))   
            
        # RMSE% Validation
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.rmse_percent_val,
            mode="lines",
            name="rmse% (validation)",
            line=dict(color='orange', width=lw+2),
        ))                
        for j in range(self.max_len):
            fig.add_trace(go.Scatter(
                x=self.epoch,
                y=self.rmse_percent_per_depth_val[:,j],
                mode="lines",
                name="rmse% (validation): "+str(pos_labels[j]),
                line=dict(color='orange', width=lw),
            ))  
            
        # MMDs
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.mmds,
            mode="lines",
            name="mmds (train)",
            line=dict(color='magenta', width=lw),
        ))     
        
        fig.add_trace(go.Scatter(
            x=self.epoch,
            y=self.mmds_val,
            mode="lines",
            name="mmds (validation)",
            line=dict(color='orange', width=lw),
        ))    
            
        ################
        fig.update_layout(
            xaxis=dict(
                title="Epoch",
                tickformat=",.0f"
            ))
        # fig.update_layout(yaxis_type="log")
        # fig.update_layout(yaxis=dict(title="",range=[0, 1]))
        fig.update_layout(font=dict(family="Times New Roman", size=20, color="black"))

        template='plotly_white'
        fig.update_layout(template=template)

        fig.update_layout(xaxis=dict(showline=True,tickwidth=5,linewidth=2,linecolor='black'))   
        fig.update_layout(yaxis=dict(showline=True,tickwidth=5,linewidth=2,linecolor='black'))   

        fig.update_traces(showlegend=True)
        fig.update_layout(hoverlabel=dict(namelength=-1))

        dpi = 128
        fig.update_layout(width  = 10 * dpi, height = 7 * dpi)
        html_filename = os.path.join(self.img_dir, 'stats.html')
        fig.write_html(html_filename)
             
    #%%
    def plot_summary(self, real_s, fake_s, outname, outfolder, epoch, density = False):
        
        # ###### Plot  data ##########
        # lw=2
        # ps = 30
        plt.rcParams['figure.figsize'] = (30, 30)
        plt.rcParams['figure.dpi'] = 75
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.style'] = 'normal'
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 1*plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    
        fig = plt.figure()
        # fig.suptitle(f"At Epoch={self.epoch[-1]}, dloss={self.dloss[-1]:.3e}, gloss={self.gloss[-1]:.3e}, gloss_val={self.gloss_val[-1]:.3e}, W={self.W[-1]:.3e},  W_val={self.W_val[-1]:.3e}", fontsize=20)
        fig.suptitle(f"At Epoch={epoch} -- {outname}", fontsize=20)

        gs = fig.add_gridspec(3,3)
        # ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[0,2])
        
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        ax5 = fig.add_subplot(gs[1,2])
        
        ax6 = fig.add_subplot(gs[2,0])
        ax7 = fig.add_subplot(gs[2,1])
        ax8 = fig.add_subplot(gs[2,2])
    
        # Box plots
        real_data = [real_s[:, i] for i in range(real_s.shape[1])]
        fake_data = [fake_s[:, i] for i in range(fake_s.shape[1])]
        pos = np.arange(1, real_s.shape[1] + 1)
        
        ax1.boxplot(
            real_data,
            vert=False,                             # horizontal orientation
            positions=pos,
            whis=1.5,                               # whiskers extend to 1.5×IQR
            patch_artist=True,                      # fill the boxes with color
            boxprops=dict(facecolor=((1.,0.,0.,0.2)), color=((1.,0.,0.,0.5))),  # box face & edge
            whiskerprops=dict(color=((1.,0.,0.,0.8))),            # whisker color
            capprops=dict(color=((1.,0.,0.,0.8))),                # cap color
            medianprops=dict(color=((1.,0.,0.,1.0)), linewidth=2),  # median line style
            flierprops=dict(
                            marker='o',
                            markerfacecolor=((1.,0.,0.,0.0)),
                            markeredgecolor=((1.,0.,0.,1.0)),
                            markersize=6,
                            linestyle='none'
                            ),
            showfliers=True                        # plot the outliers
        )
        ax1.boxplot(
            fake_data,
            vert=False,                             # horizontal orientation
            positions=pos,
            whis=1.5,                               # whiskers extend to 1.5×IQR
            patch_artist=True,                      # fill the boxes with color
            boxprops=dict(facecolor=((0.,0.,1.,0.2)), color=((0.,0.,1.,0.5))),  # box face & edge
            whiskerprops=dict(color=((0.,0.,1.,0.8))),            # whisker color
            capprops=dict(color=((0.,0.,1.,0.8))),                # cap color
            medianprops=dict(color=((0.,0.,1.,1.0)), linewidth=2),  # median line style
            flierprops=dict(
                            marker='o',
                            markerfacecolor=((0.,0.,1.,0.0)),
                            markeredgecolor=((0.,0.,1.,1.0)),
                            markersize=6,
                            linestyle='none'
                            ),
            showfliers=True                        # plot the outliers
        )
        
        # Set ticks
        ax1.set_yticks(pos)
        ax1.set_yticklabels(["0-5 cm","5-15 cm", "15-30 cm", "30-60 cm", "60-100 cm", "100-200 cm"])
        ax1.set_xlabel('SOC')
        ax1.invert_yaxis()
        
        ax2.boxplot(
            real_data,
            vert=False,                             # horizontal orientation
            positions=pos,
            whis=1.5,                               # whiskers extend to 1.5×IQR
            patch_artist=True,                      # fill the boxes with color
            boxprops=dict(facecolor=((1.,0.,0.,0.2)), color=((1.,0.,0.,0.5))),  # box face & edge
            whiskerprops=dict(color=((1.,0.,0.,0.8))),            # whisker color
            capprops=dict(color=((1.,0.,0.,0.8))),                # cap color
            medianprops=dict(color=((1.,0.,0.,1.0)), linewidth=2),  # median line style
            flierprops=dict(
                            marker='o',
                            markerfacecolor=((1.,0.,0.,0.0)),
                            markeredgecolor=((1.,0.,0.,1.0)),
                            markersize=6,
                            linestyle='none'
                            ),
            showfliers=False                        # plot the outliers
        )
        ax2.boxplot(
            fake_data,
            vert=False,                             # horizontal orientation
            positions=pos,
            whis=1.5,                               # whiskers extend to 1.5×IQR
            patch_artist=True,                      # fill the boxes with color
            boxprops=dict(facecolor=((0.,0.,1.,0.2)), color=((0.,0.,1.,0.5))),  # box face & edge
            whiskerprops=dict(color=((0.,0.,1.,0.8))),            # whisker color
            capprops=dict(color=((0.,0.,1.,0.8))),                # cap color
            medianprops=dict(color=((0.,0.,1.,1.0)), linewidth=2),  # median line style
            flierprops=dict(
                            marker='o',
                            markerfacecolor=((0.,0.,1.,0.0)),
                            markeredgecolor=((0.,0.,1.,1.0)),
                            markersize=6,
                            linestyle='none'
                            ),
            showfliers=False                        # plot the outliers
        )
        
        # Set ticks
        ax2.set_yticks(pos)
        ax2.set_yticklabels(["0-5 cm","5-15 cm", "15-30 cm", "30-60 cm", "60-100 cm", "100-200 cm"])
        ax2.set_xlabel('SOC')
        ax2.invert_yaxis() 
        
        # Plot soc hist
        pdf_range = [1e-3,1e-0]
        #0-5 cm
        r = 0
        
        real_s_min = np.min(real_s[:,r])
        fake_s_min = np.min(fake_s[:,r])
        s_min = min(real_s_min,fake_s_min)

        real_s_max = np.max(real_s[:,r])
        fake_s_max = np.max(fake_s[:,r])
        s_max = max(real_s_max,fake_s_max)

        real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
        fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
        W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)
        
        ax3.set_title(f"SOC Histogram (0-5cm) -- W = {W:.3e}")
        ax3.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
        d_min = real_s[:,r].min()
        d_max = real_s[:,r].max()
        ax3.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
        ax3.legend(loc="upper right")
        ax3.set_xlabel('SOC')
        ax3.set_ylabel('Count')
        ax3.set_yscale('log')
        if density: 
            ax3.set_ylabel('pdf')
            ax3.set_ylim(pdf_range)        
        #5-15 cm
        r = 1
        
        real_s_min = np.min(real_s[:,r])
        fake_s_min = np.min(fake_s[:,r])
        s_min = min(real_s_min,fake_s_min)

        real_s_max = np.max(real_s[:,r])
        fake_s_max = np.max(fake_s[:,r])
        s_max = max(real_s_max,fake_s_max)

        real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
        fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
        W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)
        
        ax4.set_title(f"SOC Histogram (5-15cm) -- W = {W:.3e}")
        ax4.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
        d_min = real_s[:,r].min()
        d_max = real_s[:,r].max()
        ax4.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
        ax4.legend(loc="upper right")
        ax4.set_xlabel('SOC')
        ax4.set_ylabel('Count')
        ax4.set_yscale('log')
        if density: 
            ax4.set_ylabel('pdf')
            ax4.set_ylim(pdf_range)        
        #15-30 cm
        r = 2
        
        real_s_min = np.min(real_s[:,r])
        fake_s_min = np.min(fake_s[:,r])
        s_min = min(real_s_min,fake_s_min)

        real_s_max = np.max(real_s[:,r])
        fake_s_max = np.max(fake_s[:,r])
        s_max = max(real_s_max,fake_s_max)

        real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
        fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
        W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)

        ax5.set_title(f"SOC Histogram (15-30cm) -- W = {W:.3e}")
        ax5.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
        d_min = real_s[:,r].min()
        d_max = real_s[:,r].max()
        ax5.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
        ax5.legend(loc="upper right")
        ax5.set_xlabel('SOC')
        ax5.set_ylabel('Count')
        ax5.set_yscale('log')
        if density: 
            ax5.set_ylabel('pdf')
            ax5.set_ylim(pdf_range)        
        #30-60 cm
        r = 3
        
        real_s_min = np.min(real_s[:,r])
        fake_s_min = np.min(fake_s[:,r])
        s_min = min(real_s_min,fake_s_min)

        real_s_max = np.max(real_s[:,r])
        fake_s_max = np.max(fake_s[:,r])
        s_max = max(real_s_max,fake_s_max)

        real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
        fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
        W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)
        
        ax6.set_title(f"SOC Histogram (30-60cm) -- W = {W:.3e}")
        ax6.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
        d_min = real_s[:,r].min()
        d_max = real_s[:,r].max()
        ax6.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
        ax6.legend(loc="upper right")
        ax6.set_xlabel('SOC')
        ax6.set_ylabel('Count')
        ax6.set_yscale('log')
        if density: 
            ax6.set_ylabel('pdf')
            ax6.set_ylim(pdf_range)        
        #60-100 cm
        r = 4
        
        real_s_min = np.min(real_s[:,r])
        fake_s_min = np.min(fake_s[:,r])
        s_min = min(real_s_min,fake_s_min)

        real_s_max = np.max(real_s[:,r])
        fake_s_max = np.max(fake_s[:,r])
        s_max = max(real_s_max,fake_s_max)

        real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
        fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
        W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)

        ax7.set_title(f"SOC Histogram (60-100cm) -- W = {W:.3e}")
        ax7.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
        d_min = real_s[:,r].min()
        d_max = real_s[:,r].max()
        ax7.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
        ax7.legend(loc="upper right")
        ax7.set_xlabel('SOC')
        ax7.set_ylabel('Count')
        ax7.set_yscale('log')
        if density: 
            ax7.set_ylabel('pdf')
            ax7.set_ylim(pdf_range)
        
        #100-200 cm
        r = 5

        real_s_min = np.min(real_s[:,r])
        fake_s_min = np.min(fake_s[:,r])
        s_min = min(real_s_min,fake_s_min)

        real_s_max = np.max(real_s[:,r])
        fake_s_max = np.max(fake_s[:,r])
        s_max = max(real_s_max,fake_s_max)

        real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
        fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
        W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)
        
        ax8.set_title(f"SOC Histogram (100-200cm) -- W = {W:.3e}")
        ax8.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
        d_min = real_s[:,r].min()
        d_max = real_s[:,r].max()
        ax8.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
        ax8.legend(loc="upper right")
        ax8.set_xlabel('SOC')
        ax8.set_ylabel('Count')
        ax8.set_yscale('log')
        if density: 
            ax8.set_ylabel('pdf')
            ax8.set_ylim(pdf_range)
        
        plt.tight_layout()
        if density: 
            plt.savefig(os.path.join(outfolder, f'{outname}_data_stats_{epoch}.pdf'))
        else:
            plt.savefig(os.path.join(outfolder, f'{outname}_data_stats_{epoch}.png'))
        plt.close()
    
    #%%
    # def plot_summary(self, real_s, fake_s, outname, outfolder, epoch, density = False):
        
    #     # ###### Plot  data ##########
    #     # lw=2
    #     # ps = 30
    #     plt.rcParams['figure.figsize'] = (30, 30)
    #     plt.rcParams['figure.dpi'] = 75
    #     plt.rcParams['font.family'] = 'Times New Roman'
    #     plt.rcParams['font.style'] = 'normal'
    #     plt.rcParams['font.size'] = 20
    #     plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    #     plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    #     plt.rcParams['legend.fontsize'] = 1*plt.rcParams['font.size']
    #     plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    #     plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    
    #     fig = plt.figure()
    #     # fig.suptitle(f"At Epoch={self.epoch[-1]}, dloss={self.dloss[-1]:.3e}, gloss={self.gloss[-1]:.3e}, gloss_val={self.gloss_val[-1]:.3e}, W={self.W[-1]:.3e},  W_val={self.W_val[-1]:.3e}", fontsize=20)
    #     fig.suptitle(f"At Epoch={epoch} -- {outname}", fontsize=20)

    #     gs = fig.add_gridspec(3,3)
    #     # ax0 = fig.add_subplot(gs[0,0])
    #     ax1 = fig.add_subplot(gs[0,1])
    #     ax2 = fig.add_subplot(gs[0,2])
        
    #     ax3 = fig.add_subplot(gs[1,0])
    #     ax4 = fig.add_subplot(gs[1,1])
    #     ax5 = fig.add_subplot(gs[1,2])
        
    #     ax6 = fig.add_subplot(gs[2,0])
    #     ax7 = fig.add_subplot(gs[2,1])
    #     ax8 = fig.add_subplot(gs[2,2])
    
    #     # Box plots
    #     real_data = [real_s[:, i] for i in range(real_s.shape[1])]
    #     fake_data = [fake_s[:, i] for i in range(fake_s.shape[1])]
    #     pos = np.arange(1, real_s.shape[1] + 1)
        
    #     ax1.boxplot(
    #         real_data,
    #         vert=False,                             # horizontal orientation
    #         positions=pos,
    #         whis=3.0,                               # whiskers extend to 3.0×IQR
    #         patch_artist=True,                      # fill the boxes with color
    #         boxprops=dict(facecolor=((1.,0.,0.,0.2)), color=((1.,0.,0.,0.5))),  # box face & edge
    #         whiskerprops=dict(color=((1.,0.,0.,0.8))),            # whisker color
    #         capprops=dict(color=((1.,0.,0.,0.8))),                # cap color
    #         medianprops=dict(color=((1.,0.,0.,1.0)), linewidth=2),  # median line style
    #         flierprops=dict(
    #                         marker='o',
    #                         markerfacecolor=((1.,0.,0.,0.0)),
    #                         markeredgecolor=((1.,0.,0.,1.0)),
    #                         markersize=6,
    #                         linestyle='none'
    #                         ),
    #         showfliers=True                        # plot the outliers
    #     )
        
    #     # Set ticks
    #     ax1.set_yticks(pos)
    #     ax1.set_yticklabels(["0-5 cm","5-15 cm", "15-30 cm", "30-60 cm", "60-100 cm", "100-200 cm"])
    #     ax1.set_xlabel('SOC')
    #     ax1.invert_yaxis()
        
    #     ax2.boxplot(
    #         real_data,
    #         vert=False,                             # horizontal orientation
    #         positions=pos,
    #         whis=3.0,                               # whiskers extend to 3.0×IQR
    #         patch_artist=True,                      # fill the boxes with color
    #         boxprops=dict(facecolor=((1.,0.,0.,0.2)), color=((1.,0.,0.,0.5))),  # box face & edge
    #         whiskerprops=dict(color=((1.,0.,0.,0.8))),            # whisker color
    #         capprops=dict(color=((1.,0.,0.,0.8))),                # cap color
    #         medianprops=dict(color=((1.,0.,0.,1.0)), linewidth=2),  # median line style
    #         flierprops=dict(
    #                         marker='o',
    #                         markerfacecolor=((1.,0.,0.,0.0)),
    #                         markeredgecolor=((1.,0.,0.,1.0)),
    #                         markersize=6,
    #                         linestyle='none'
    #                         ),
    #         showfliers=False                        # plot the outliers
    #     )
        
    #     # Set ticks
    #     ax2.set_yticks(pos)
    #     ax2.set_yticklabels(["0-5 cm","5-15 cm", "15-30 cm", "30-60 cm", "60-100 cm", "100-200 cm"])
    #     ax2.set_xlabel('SOC')
    #     ax2.invert_yaxis() 
        
    #     # Plot soc hist
    #     pdf_range = [1e-3,1e-0]
    #     #0-5 cm
    #     r = 0
        
    #     real_s_min = np.min(real_s[:,r])
    #     fake_s_min = np.min(fake_s[:,r])
    #     s_min = min(real_s_min,fake_s_min)

    #     real_s_max = np.max(real_s[:,r])
    #     fake_s_max = np.max(fake_s[:,r])
    #     s_max = max(real_s_max,fake_s_max)

    #     real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
    #     fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
    #     W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)
        
    #     ax3.set_title(f"SOC Histogram (0-5cm) -- W = {W:.3e}")
    #     ax3.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
    #     d_min = real_s[:,r].min()
    #     d_max = real_s[:,r].max()
    #     # ax3.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
    #     ax3.legend(loc="upper right")
    #     ax3.set_xlabel('SOC')
    #     ax3.set_ylabel('Count')
    #     ax3.set_yscale('log')
    #     if density: 
    #         ax3.set_ylabel('pdf')
    #         ax3.set_ylim(pdf_range)        
    #     #5-15 cm
    #     r = 1
        
    #     real_s_min = np.min(real_s[:,r])
    #     fake_s_min = np.min(fake_s[:,r])
    #     s_min = min(real_s_min,fake_s_min)

    #     real_s_max = np.max(real_s[:,r])
    #     fake_s_max = np.max(fake_s[:,r])
    #     s_max = max(real_s_max,fake_s_max)

    #     real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
    #     fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
    #     W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)
        
    #     ax4.set_title(f"SOC Histogram (5-15cm) -- W = {W:.3e}")
    #     ax4.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
    #     d_min = real_s[:,r].min()
    #     d_max = real_s[:,r].max()
    #     # ax4.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
    #     ax4.legend(loc="upper right")
    #     ax4.set_xlabel('SOC')
    #     ax4.set_ylabel('Count')
    #     ax4.set_yscale('log')
    #     if density: 
    #         ax4.set_ylabel('pdf')
    #         ax4.set_ylim(pdf_range)        
    #     #15-30 cm
    #     r = 2
        
    #     real_s_min = np.min(real_s[:,r])
    #     fake_s_min = np.min(fake_s[:,r])
    #     s_min = min(real_s_min,fake_s_min)

    #     real_s_max = np.max(real_s[:,r])
    #     fake_s_max = np.max(fake_s[:,r])
    #     s_max = max(real_s_max,fake_s_max)

    #     real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
    #     fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
    #     W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)

    #     ax5.set_title(f"SOC Histogram (15-30cm) -- W = {W:.3e}")
    #     ax5.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
    #     d_min = real_s[:,r].min()
    #     d_max = real_s[:,r].max()
    #     # ax5.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
    #     ax5.legend(loc="upper right")
    #     ax5.set_xlabel('SOC')
    #     ax5.set_ylabel('Count')
    #     ax5.set_yscale('log')
    #     if density: 
    #         ax5.set_ylabel('pdf')
    #         ax5.set_ylim(pdf_range)        
    #     #30-60 cm
    #     r = 3
        
    #     real_s_min = np.min(real_s[:,r])
    #     fake_s_min = np.min(fake_s[:,r])
    #     s_min = min(real_s_min,fake_s_min)

    #     real_s_max = np.max(real_s[:,r])
    #     fake_s_max = np.max(fake_s[:,r])
    #     s_max = max(real_s_max,fake_s_max)

    #     real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
    #     fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
    #     W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)
        
    #     ax6.set_title(f"SOC Histogram (30-60cm) -- W = {W:.3e}")
    #     ax6.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
    #     d_min = real_s[:,r].min()
    #     d_max = real_s[:,r].max()
    #     # ax6.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
    #     ax6.legend(loc="upper right")
    #     ax6.set_xlabel('SOC')
    #     ax6.set_ylabel('Count')
    #     ax6.set_yscale('log')
    #     if density: 
    #         ax6.set_ylabel('pdf')
    #         ax6.set_ylim(pdf_range)        
    #     #60-100 cm
    #     r = 4
        
    #     real_s_min = np.min(real_s[:,r])
    #     fake_s_min = np.min(fake_s[:,r])
    #     s_min = min(real_s_min,fake_s_min)

    #     real_s_max = np.max(real_s[:,r])
    #     fake_s_max = np.max(fake_s[:,r])
    #     s_max = max(real_s_max,fake_s_max)

    #     real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
    #     fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
    #     W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)

    #     ax7.set_title(f"SOC Histogram (60-100cm) -- W = {W:.3e}")
    #     ax7.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
    #     d_min = real_s[:,r].min()
    #     d_max = real_s[:,r].max()
    #     # ax7.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
    #     ax7.legend(loc="upper right")
    #     ax7.set_xlabel('SOC')
    #     ax7.set_ylabel('Count')
    #     ax7.set_yscale('log')
    #     if density: 
    #         ax7.set_ylabel('pdf')
    #         ax7.set_ylim(pdf_range)
        
    #     #100-200 cm
    #     r = 5

    #     real_s_min = np.min(real_s[:,r])
    #     fake_s_min = np.min(fake_s[:,r])
    #     s_min = min(real_s_min,fake_s_min)

    #     real_s_max = np.max(real_s[:,r])
    #     fake_s_max = np.max(fake_s[:,r])
    #     s_max = max(real_s_max,fake_s_max)

    #     real_s_scaled = (real_s[:,r] - s_min)/(s_max - s_min)
    #     fake_s_scaled = (fake_s[:,r] - s_min)/(s_max - s_min)
        
    #     W = sp.stats.wasserstein_distance(real_s_scaled, fake_s_scaled)
        
    #     ax8.set_title(f"SOC Histogram (100-200cm) -- W = {W:.3e}")
    #     ax8.hist(real_s[:,r], bins=50, color=((1.,0.,0.,0.3)), edgecolor=((1.,0.,0.,0.5)), label=f'Real {outname} data',histtype="bar", density = density)
    #     d_min = real_s[:,r].min()
    #     d_max = real_s[:,r].max()
    #     # ax8.hist(fake_s[:,r], bins=50, range=(d_min,d_max), color=((0.,0.,1.,0.3)), edgecolor=((0.,0.,1.,0.5)), label=f'Synthetic {outname} data',histtype="bar", density = density)
    #     ax8.legend(loc="upper right")
    #     ax8.set_xlabel('SOC')
    #     ax8.set_ylabel('Count')
    #     ax8.set_yscale('log')
    #     if density: 
    #         ax8.set_ylabel('pdf')
    #         ax8.set_ylim(pdf_range)
        
    #     plt.tight_layout()
    #     if density: 
    #         plt.savefig(os.path.join(outfolder, f'{outname}_data_stats_{epoch}.pdf'))
    #     else:
    #         plt.savefig(os.path.join(outfolder, f'{outname}_data_stats_{epoch}.pdf'))
    #     plt.close()
            
            
    #%%   
    def getSummaryResults(self):
        return pd.DataFrame({'epocs':self.epoch, 
                             'disc_loss':self.dloss, 
                             'gen_loss':self.gloss,
                             'gen_loss_global':self.gloss_global,
                             'gen_loss_val':self.gloss_val,
                             'W':self.W,
                             'W_val':self.W_val,
                             'rmse':self.rmse,
                             'rmse_val':self.rmse_val,
                             'rmse_val_smooth':self.rmse_smooth_val,
                             'rmse_percent':self.rmse_percent,
                             'rmse_percent_val':self.rmse_percent_val,
                             'mmds':self.mmds,
                             'mmds_val':self.mmds_val
                            })     
    #%%
    def write_metrics(self):
        loss_file = os.path.join(self.summary_dir, f'{self.profiles.profile_ID}_Summary.csv')
        df_summary = self.getSummaryResults()
        df_summary.to_csv(loss_file, index=False) 

    def read_metrics(self):
        loss_file = os.path.join(self.summary_dir, f'{self.profiles.profile_ID}_Summary.csv')
        df_summary = pd.read_csv(loss_file)
        return df_summary
    
    def find_min_metric(self, metrics, mode):
        min_index = metrics[mode].idxmin()
        return metrics["epocs"].iloc[min_index]
        
# %%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def train(self, nEpochs=20000, nEval=1000):

        start_time = time()

        # Prepare Dataset in batches
        dataset = (
            tf.data.Dataset
              .from_tensor_slices((self.profiles.soc_scaled, 
                                   self.profiles.conds_loc_scaled, 
                                   self.profiles.conds_env_scaled, 
                                   self.profiles.conds_alt_scaled))
              .shuffle(self.N)
              .batch(self.batch_size, drop_remainder=True)
        )
        
        # Save train and validation data for post-processing
        self.__save_input_data__()
        
        # Set tensors with real data for metric computation
        real_s_rescaled  = self.profiles.rescale_data(self.profiles.soc_scaled)
        real_s_rescaled_tf = tf.constant(real_s_rescaled)
        real_s_rescaled_val  = self.profiles.rescale_data(self.profiles.soc_scaled_val)
        real_s_rescaled_tf_val = tf.constant(real_s_rescaled_val)
        
        # Initialize variables and lists
        # best_W_val = np.inf
        # best_epoch = 0
        # wait = 0

        self.time = []
        self.epoch = []
        
        self.dloss=[]
        self.gloss=[]
        self.gloss_global=[]
        self.gloss_val=[]
        
        self.W = []
        self.W_val = []
        
        self.rmse = []
        self.rmse_val = []
        self.rmse_smooth_val=[]
        self.rmse_percent = []
        self.rmse_percent_val = []
        self.rmse_per_depth = np.empty((0, self.max_len))
        self.rmse_per_depth_val = np.empty((0, self.max_len))
        self.rmse_percent_per_depth = np.empty((0, self.max_len))
        self.rmse_percent_per_depth_val = np.empty((0, self.max_len))
        
        self.mmds = []
        self.mmds_val = []
        
        es_wait = 0        # es “no‐improve” counter
        lr_wait = 0        # “no‐improve” counter
        best_rmse_smooth_val = np.inf
        smooth_win = deque(maxlen=nEval)
        
        # Loop over epochs
        for epoch in range(1, nEpochs+1):
            curTime = time()
            
            self.epoch.append(epoch)    
            self.learning_rate_list.append( float(self.generator_opt.learning_rate.numpy()) )

            # Run batches
            for real_s, cond_loc, cond_env, cond_alt in dataset:
                d_loss, g_loss = self.train_step(real_s, cond_loc, cond_env, cond_alt)
            self.dloss.append(d_loss.numpy())    
            self.gloss.append(g_loss.numpy())    
             
            # Get synthetic data for entire training and validation datasets
            fake_s = self.__get_synthetic_data__(self.N, 
                                                 self.profiles.conds_loc_scaled, 
                                                 self.profiles.conds_env_scaled, 
                                                 self.profiles.conds_alt_scaled)
            fake_s_val = self.__get_synthetic_data__(self.N_val, 
                                                     self.profiles.conds_loc_scaled_val, 
                                                     self.profiles.conds_env_scaled_val, 
                                                     self.profiles.conds_alt_scaled_val)
            
            # Compute generator loss for training and validation dataset
            # Note that here, training gloss is computed in entire dataset and
            # not only in batch
            self.gloss_global.append( self.__get_synthetic_gloss__(fake_s, 
                                                               self.profiles.conds_loc_scaled, 
                                                               self.profiles.conds_env_scaled, 
                                                               self.profiles.conds_alt_scaled
                                                               ).numpy())
            self.gloss_val.append( self.__get_synthetic_gloss__(fake_s_val, 
                                                               self.profiles.conds_loc_scaled_val, 
                                                               self.profiles.conds_env_scaled_val, 
                                                               self.profiles.conds_alt_scaled_val
                                                               ).numpy())
            
            # Compute W for both training and validation dataset
            self.W.append( self.__wasserstein_distance__(self.profiles.soc_scaled, fake_s.numpy()) )
            self.W_val.append( self.__wasserstein_distance__(self.profiles.soc_scaled_val, fake_s_val.numpy()) )
            
            # Compute RMSE
            fake_s_rescaled  = self.profiles.rescale_data(fake_s.numpy())
            fake_s_rescaled_tf = tf.constant(fake_s_rescaled)
            rmse, rmse_per_depth, rmse_percent, rmse_percent_per_depth = self.__compute_rmse__(real_s_rescaled_tf, fake_s_rescaled_tf)
            self.rmse.append(rmse.numpy())
            self.rmse_percent.append(rmse_percent.numpy())
            self.rmse_per_depth = np.vstack([self.rmse_per_depth, rmse_per_depth.numpy()])
            self.rmse_percent_per_depth = np.vstack([self.rmse_percent_per_depth, rmse_percent_per_depth.numpy()])

            fake_s_rescaled_val  = self.profiles.rescale_data(fake_s_val.numpy())
            fake_s_rescaled_tf_val = tf.constant(fake_s_rescaled_val)
            rmse_val, rmse_per_depth_val, rmse_percent_val, rmse_percent_per_depth_val = self.__compute_rmse__(real_s_rescaled_tf_val, fake_s_rescaled_tf_val)
            self.rmse_val.append(rmse_val.numpy())
            self.rmse_percent_val.append(rmse_percent_val.numpy())
            self.rmse_per_depth_val = np.vstack([self.rmse_per_depth_val, rmse_per_depth_val.numpy()])
            self.rmse_percent_per_depth_val = np.vstack([self.rmse_percent_per_depth_val, rmse_percent_per_depth_val.numpy()])
            
            # FOR DEBUG !!!!!!!! TRYING TO TEST W_val early stop   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # smooth_win.append(rmse_val.numpy())
            smooth_win.append(self.W_val[-1])
            self.rmse_smooth_val.append(float(np.mean(smooth_win)))

            # Compute MMD
            mmds = self.__compute_mmd__(real_s_rescaled_tf, fake_s_rescaled_tf)
            self.mmds.append(mmds.numpy())
            mmds_val = self.__compute_mmd__(real_s_rescaled_tf_val, fake_s_rescaled_tf_val)
            self.mmds_val.append(mmds_val.numpy())
            
            # Save generator weights
            self.generator.save_weights(os.path.join(self.model_dir,f'{self.profiles.profile_ID}_{epoch}.h5'))
            # self.discriminator.save_weights(os.path.join(self.model_dir,f'disc_{self.profiles.profile_ID}_{epoch}.h5'))

            if epoch % nEval == 0:
                # Write metrics
                self.write_metrics()
                
                # Plot
                self.plot_summary(real_s_rescaled, fake_s_rescaled, 
                                  outname = 'training', 
                                  outfolder = self.img_dir, 
                                  epoch = epoch
                                  )
                self.plot_summary(real_s_rescaled_val, fake_s_rescaled_val, 
                                  outname = 'validation', 
                                  outfolder = self.img_dir, 
                                  epoch = epoch
                                  )                
                self.__plot_metrics__()                
                
            # ## Check if we need to reduce learning rate ##
            if self.rmse_smooth_val[-1] < best_rmse_smooth_val * (1 - 1e-4):  # require a tiny relative improvement
                best_rmse_smooth_val = self.rmse_smooth_val[-1]
                lr_wait = 0
                es_wait = 0                
                print(f"   New best rmse_smooth_val: {best_rmse_smooth_val:.3e} at epoch {epoch}")
            else:
                lr_wait += 1
                es_wait += 1
                
                # If no improvement for patience_lr epochs, cut LR
                if (lr_wait >= self.patience_lr):
                    old_lr = float(self.generator_opt.learning_rate.numpy())
                    new_lr = max(old_lr * self.decrease_factor_lr, self.min_learning_rate)
                    self.generator_opt.learning_rate.assign(new_lr)
                    self.discriminator_opt.learning_rate.assign(new_lr)
                    print(f"  Reducing learning rate: {old_lr:.1e} to {new_lr:.1e}")
                    lr_wait = 0  
                    
                # If no improvement stop epochs loop
                if (es_wait >= self.patience_es):
                    print(f" ####### Early stopping at epoch {epoch} #######")
                    break                

            self.time.append(time()-curTime)
            lr = float(self.generator_opt.learning_rate.numpy())
            print(f'Epoch {epoch} time = {self.time[-1]:.3e} sec, learning rate = {lr:.3e}, lr_wait = {lr_wait}, es_wait = {es_wait}', flush=True)
            
        end_time = time()
        print('\n', flush=True)
        print(f'Total time={str(timedelta(seconds=end_time-start_time))}', flush=True)
        print('\n', flush=True)
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%% TEST METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def test(self, mode, epoch, realizations):
        
        start_time = time()

        # WARNING: This should be read from fold folder, as in a truly random
        #          kFold cross-validation, we will not be able to replicate the 
        #          train-validation-test partition
        
        # Rescale test data
        real_s_rescaled_test  = self.profiles.rescale_data(self.profiles.soc_scaled_test)
        
        metrics = self.read_metrics()

        # Load weights for specific fold indicated by model_dir
        if mode == 'None':
            
            if epoch > int(metrics['epocs'].iloc[-1]):
                epoch_out = int(metrics['epocs'].iloc[-1])
            else:
                epoch_out = epoch
                
            print(f"Reading epoch {epoch_out}")
            self.generator.load_weights(os.path.join(self.model_dir,f'{self.profiles.profile_ID}_{epoch_out}.h5')) 
        else:
            if mode in list(metrics.columns):
                epoch_out = self.find_min_metric(metrics, mode) 
                print(f"Found min({mode}) at epoch={epoch_out}")
                self.generator.load_weights(os.path.join(self.model_dir,f'{self.profiles.profile_ID}_{epoch_out}.h5')) 
            else:
                print(f"Mode {mode} not in metrics summary file!")
                sys.exit()
        
        # Run model for test sites
        fake_s_rescaled_test_list = []
        for r in range(realizations):
            print(f"  Generating realization {r+1}")
            fake_s_test = self.__get_synthetic_data__(self.N_test, 
                                                      self.profiles.conds_loc_scaled_test, 
                                                      self.profiles.conds_env_scaled_test, 
                                                      self.profiles.conds_alt_scaled_test
                                                     )
            # WARNING Need to implement rescaling from reading scaling factors in fold folder
            fake_s_rescaled_test  = self.profiles.rescale_data(fake_s_test.numpy())
            # fake_s_rescaled_tf_test = tf.constant(fake_s_rescaled_test)
            fake_s_rescaled_test_list.append(fake_s_rescaled_test)

        end_time = time()
        print('\n', flush=True)
        print(f'Total time={str(timedelta(seconds=end_time-start_time))}', flush=True)
        print('\n', flush=True)        
        
        return real_s_rescaled_test, fake_s_rescaled_test_list, epoch_out
        
        