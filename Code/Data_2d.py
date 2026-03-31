# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ProfilesData:
    def __init__(self, profile_ID, train_data_all, validation_data_all, test_data_all, scaling):
        self.profile_ID = profile_ID
        
        self.train_data_all = train_data_all[["profile_id",
                                             "avg_dept",
                                             "orgc_val_1",
                                             "lon",
                                             "lat",
                                             "Annual Mean Temperature",
                                             "Annual Precipitation",
                                             "Elevation",
                                             'bin_id',
                                             'std_lo',
                                             'std_hi']].copy()
        self.validation_data_all  = validation_data_all[["profile_id",
                                             "avg_dept",
                                             "orgc_val_1",
                                             "lon",
                                             "lat",
                                             "Annual Mean Temperature",
                                             "Annual Precipitation",
                                             "Elevation",
                                             'bin_id',
                                             'std_lo',
                                             'std_hi']].copy() 
        
        self.test_data_all  = test_data_all[["profile_id",
                                             "avg_dept",
                                             "orgc_val_1",
                                             "lon",
                                             "lat",
                                             "Annual Mean Temperature",
                                             "Annual Precipitation",
                                             "Elevation",
                                             'bin_id',
                                             'std_lo',
                                             'std_hi']].copy() 
        
        self.global_coords = bool(scaling['global_coords'])
        self.scaling = pd.Series({k: v for k, v in scaling.items() if k != 'global_coords'})

        self.N = train_data_all.nunique()['profile_id']
        self.N_val = validation_data_all.nunique()['profile_id']
        self.N_test = test_data_all.nunique()['profile_id']
        self.max_len = train_data_all['profile_id'].value_counts().max()
        
        self.scaling_factors, self.centering_factors = self.__get_scales__(self.train_data_all, global_coords = self.global_coords)
        self.train_data_all_scaled = self.__scale_data__(self.train_data_all)
        self.validation_data_all_scaled = self.__scale_data__(self.validation_data_all)
        self.test_data_all_scaled = self.__scale_data__(self.test_data_all)

        self.profile_ids, self.soc_scaled, self.depth_scaled, self.conds_loc_scaled, self.conds_env_scaled, self.conds_alt_scaled = self.__preprocess_data__(self.train_data_all_scaled)
        self.profile_ids_val, self.soc_scaled_val, self.depth_scaled_val, self.conds_loc_scaled_val, self.conds_env_scaled_val, self.conds_alt_scaled_val = self.__preprocess_data__(self.validation_data_all_scaled)
        self.profile_ids_test, self.soc_scaled_test, self.depth_scaled_test, self.conds_loc_scaled_test, self.conds_env_scaled_test, self.conds_alt_scaled_test = self.__preprocess_data__(self.test_data_all_scaled)

    def __get_scales__(self, data, global_coords = True):
        mins = data.min()
        maxs = data.max()
        
        if global_coords:
            mins['lat'] = -90
            maxs['lat'] = 90
            mins['lon'] = -180
            maxs['lon'] = 180
        
        scaling_factors = pd.Series()
        centering_factors = pd.Series()
        for col in list(self.scaling.index):
            scale = self.scaling[col]
            if (scale == [-1,1]):
                scaling_factors[col] = 0.5*(maxs[col] - mins[col])
                centering_factors[col] = 0.5*(maxs[col] + mins[col])
            elif (scale == [0,1]):
                scaling_factors[col] = maxs[col] - mins[col]
                centering_factors[col] = mins[col]

        return scaling_factors, centering_factors    
            
    def __scale_data__(self, data):
        
        data_scaled = data.copy()

        for col in list(self.scaling.index):
            data_scaled[col] = (data[col]-float(self.centering_factors[col])) \
                               /float(self.scaling_factors[col])
        return data_scaled
        
    def __preprocess_data__(self, data):
            
        # Load and group DataFrame
        profiles = []
        pids = []
        for pid, grp in data.groupby('profile_id'):
            
            # profile ids
            pids.append(pid)
            
            # Latitude and longitude
            lat = grp['lat'].iloc[0]
            lon = grp['lon'].iloc[0]
            
            # MAT and MAP
            temp = grp['Annual Mean Temperature'].iloc[0]
            prec = grp['Annual Precipitation'].iloc[0]
            
            # Elevation
            alt = grp['Elevation'].iloc[0]
    
            z_arr   = grp.sort_values('avg_dept')['avg_dept'].to_numpy(dtype=np.float32)
            soc_arr = grp.sort_values('avg_dept')['orgc_val_1'].to_numpy(dtype=np.float32)
            profiles.append((lat, lon, temp, prec, alt, z_arr, soc_arr))
        
        N       = len(profiles)
        max_len = max(len(z) for _,_,_,_,_,z,_ in profiles)            

        # Reshape data
        pids_arr = np.array(pids, dtype=np.int)
        soc = np.zeros((N, max_len), dtype=np.float32)
        depth = np.zeros((N, max_len), dtype=np.float32)
        conds_loc = np.zeros((N, 2), dtype=np.float32)
        conds_env = np.zeros((N, 2), dtype=np.float32)
        conds_alt = np.zeros((N,), dtype=np.float32)
    
        for i, (lat, lon, temp, prec, alt, z_arr, soc_arr) in enumerate(profiles):
            soc[i,:] = soc_arr
            depth[i,:] = z_arr
            conds_loc[i] = [lat, lon]
            conds_env[i] = [temp, prec]
            conds_alt[i] = alt

        return pids_arr, soc, depth, conds_loc, conds_env, conds_alt
    
    def rescale_data(self, data, var = 'orgc_val_1'):
        centering_factor = float(self.centering_factors[str(var)])
        scaling_factor = float(self.scaling_factors[str(var)])
        data_rescaled = data * scaling_factor + centering_factor
        return data_rescaled
    
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class SubGroupData:
    def __init__(self, infile, outfolder):
        self.infile = infile
        self.data =  pd.read_csv(infile,index_col=0)
        self.data["avg_dept"] = 0.5*(self.data["upper_dept"] + self.data["lower_dept"])
        self.subgroupdata = self.data.copy()
        self.subgroupdata_standard = None

        self.outfolder = outfolder
        if(self.outfolder != None):
            self.__create_path__(self.outfolder)
            
    # Create folder along the path if does not exists
    def __create_path__(self, fpath):
        if os.path.exists(fpath):
            return
        os.makedirs(fpath)      
        
    # return current state of subgroupdata and saves results of consecutive calls to subgroupdata
    def filter_by(self, field, value):
        
        # Clean data
        mask = (self.subgroupdata["Soil_Type"] > 0) \
                  & (self.subgroupdata["Annual Mean Temperature"] > -100)  \
                  & (self.subgroupdata["Annual Precipitation"] >= 0) \
                  & (self.subgroupdata["Elevation"] >= 0)
        self.subgroupdata = self.subgroupdata[mask]          
        
        if (field == "biome_type") or (field == "Soil_Type"):
            mask = (self.subgroupdata[field] == value)
            self.subgroupdata = self.subgroupdata[mask]
            
        else:
            print("Error in filter_by: Only biome_type and Soil_Type are accepted!")
            sys.exit()
            
        return self.subgroupdata 
    
    
    def standardize_depths(self, std_bins = None, outlier_cutoff = 0.0):
        
        if std_bins == None:
            std_bins = pd.DataFrame({
                'bin_id':    ['0-5',  '5-15',  '15-30', '30-60', '60-100','100-200'],
                'std_lo':    [0,      5,       15,      30,      60,       100],
                'std_hi':    [5,      15,      30,      60,      100,      200]
            })
        
        # unique profiles
        profiles = self.subgroupdata[['profile_id']].drop_duplicates()
        
        # make every (profile, bin) pair
        pf_bins = profiles.merge(std_bins, how='cross')
        
        # attach each original measurement to every (profile, bin)
        df_x = (pf_bins
                .merge(self.subgroupdata, on='profile_id', how='left')
                # compute the actual overlap length
                .assign(overlap=lambda d: ((d[['std_hi','lower_dept']].min(axis=1)) 
                                        - (d[['std_lo','upper_dept']].max(axis=1))
                ).clip(lower=0))
               )
        
        # multiply soc by overlap
        df_x['w_soc'] = df_x['orgc_val_1'] * df_x['overlap']
        
        # aggregate to get weighted average within each (profile, bin)
        result = (
            df_x
            .groupby(['profile_id','lat','lon','Annual Mean Temperature', 'Annual Precipitation', 'Elevation', 'bin_id','std_lo','std_hi'], as_index=False)
            .agg(total_overlap=('overlap','sum'),
                 total_w_soc   =('w_soc','sum'))
        )
        
        # finally, weighted average
        result['soc_std'] = result['total_w_soc'] / result['total_overlap']
        result = result.sort_values(['profile_id', 'std_lo']).reset_index(drop=True)
        
        result_zero = result.copy()
        result_zero['soc_std'] = result_zero['soc_std'].fillna(0)
        result_zero.rename(columns={'soc_std':'orgc_val_1'}, inplace=True)
        result_zero['avg_dept'] = 0.5*(result_zero['std_hi'] + result_zero['std_lo'])    
        
        # Pad layers with zero values
        # This is done for layers with zeros in top layer
        # and intermediate layers surrounded by layers with non-zero values.
        # Nothing is done for deep layers with zeros.
        cnt_all = 0
        cnt_zero = 0
        result_zero_padded = pd.DataFrame()
        for pid, grp in result_zero.groupby('profile_id'):
            
            soc = grp['orgc_val_1'].values.copy()
            n = len(soc)
            i = 0       
            
            while i < n:
                if soc[i] == 0:
                    start = i
                    while i < n and soc[i] == 0:
                        i += 1
                    end = i  # soc[start:end] are zeros
            
                    # Case 1: Leading zeros (at the very beginning)
                    if start == 0 and end < n:
                        soc[start:end] = soc[end]
                    # Case 2: Internal zeros (with nonzero on both sides)
                    elif start > 0 and end < n:
                        left = soc[start-1]
                        right = soc[end]
                        avg_val = (left + right) / 2
                        soc[start:end] = avg_val
                    # Trailing zeros: do nothing (they remain zero)
                else:
                    i += 1
            
            grp['orgc_val_1'] = soc
            
            if np.mean(soc[0]) == 0.0:
                cnt_zero = cnt_zero + 1 
                print(f'Dropping profile_id {pid} due to zero soc values in entire column')
            else:
                cnt_all = cnt_all + 1 
                result_zero_padded = pd.concat([result_zero_padded,grp])
                
        result_zero_padded.reset_index(drop=True,inplace=True)
        
        # Check for Nans
        nan_flag = result_zero_padded['orgc_val_1'].isna().any()
        if nan_flag:
            print("ERROR: Nan values after standardize_depths!!")
            sys.exit()
            
        # Check for zero in top layer
        for pid, grp in result_zero_padded.groupby('profile_id'):
            grp.reset_index(drop=True,inplace=True)
            if grp['orgc_val_1'].iloc[0] == 0.0:
                print(f"Error in first layer of profile {pid}!!")
                sys.exit()
                
                
        print(f'Dropped {cnt_zero} out of {cnt_all} profiles containing all zeros')
                
        # Drop profiles with outliers
        if outlier_cutoff == 0.0:
            self.subgroupdata_standard = result_zero_padded
        else:  
            # Compute IQR per avg_dept
            q1_per_dept = (result_zero_padded.groupby('avg_dept')['orgc_val_1']
                            .apply(lambda x: x.quantile(0.25))
                            .reset_index(name='Q1')
                            )
            q3_per_dept = (result_zero_padded.groupby('avg_dept')['orgc_val_1']
                            .apply(lambda x: x.quantile(0.75))
                            .reset_index(name='Q3')
                            )
            iqr_per_dept = (result_zero_padded.groupby('avg_dept')['orgc_val_1']
                            .apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
                            .reset_index(name='IQR')
                            )
            
            cnt = 0
            result_zero_padded_cleaned = pd.DataFrame()
            for pid, grp in result_zero_padded.groupby('profile_id'):
                soc = grp['orgc_val_1'].values.copy()
                depth = grp['avg_dept'].values.copy()
                n = len(soc)
                
                flag_drop = False
                for i in range(n):
                    lower_fence = max(q1_per_dept['Q1'].iloc[i] - outlier_cutoff*iqr_per_dept['IQR'].iloc[i], 0)
                    upper_fence = q3_per_dept['Q3'].iloc[i] + outlier_cutoff*iqr_per_dept['IQR'].iloc[i]
                    
                    if (soc[i] < lower_fence) or (soc[i] > upper_fence):
                        cnt = cnt + 1
                        print(f'Dropping profile_id {pid} due to outlier soc value at depth {depth[i]}: soc={soc[i]} outside [{lower_fence},{upper_fence}] ')
                        flag_drop = True
                        break
                    
                if (not flag_drop):       
                    result_zero_padded_cleaned = pd.concat([result_zero_padded_cleaned,grp])

            print(f'Dropped {cnt} out of {cnt_all} profiles containing outliers')
            result_zero_padded_cleaned.reset_index(drop=True,inplace=True)
            self.subgroupdata_standard = result_zero_padded_cleaned
        
        return self.subgroupdata_standard 
                        
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class KFold_Profiles:
    """
        KFold cross validation at the profile level (i.e., splits profiles into folds instead of samples)
    """
    def __init__(self, k = 5, test = 0.0, shuffle = True, seed = None):
        self.k = k   
        self.test = test
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data):
        """
        data: pandas DataFrame with required column "profile_id"
        Returns: list of DataFrames split by profile_id
        """

        profile_ids = pd.unique(data["profile_id"])
        if (self.shuffle):
            rng = np.random.Generator(np.random.PCG64(seed = self.seed))
            rng.shuffle(profile_ids)
            
        if self.test != 0.0:
            # split data for testing
            split_test_index = int(self.test * len(profile_ids))
            profile_ids_test, profile_ids_train_val = np.array_split(profile_ids, [split_test_index])
            split_test_data = data[data["profile_id"].isin(profile_ids_test)]
            # Split data between folds for validation    
            split_profile_ids = np.array_split(profile_ids_train_val,indices_or_sections=self.k)
        else:
            # empty
            split_test_data = pd.DataFrame()
            # Split data between folds for validation    
            split_profile_ids = np.array_split(profile_ids,indices_or_sections=self.k)
   
        split_data = []
        for i in range(self.k):
            df = data[data["profile_id"].isin(split_profile_ids[i])]
            split_data.append(df)
            
        split_train_data = []
        split_validation_data = []
        for i in range(self.k):
            df0 = split_data[i]
            df0.reset_index(drop=True,inplace=True)
            split_validation_data.append(df0)
            
            idx = list(np.delete(np.array(range(self.k)),i))
            selected_elements = [split_data[j] for j in idx]
            for j in range(len(selected_elements)):
                if (j == 0):
                    df = selected_elements[j]
                else:
                    df1 = selected_elements[j]
                    aux = pd.concat([df,df1])
                    df = aux
                
            df.reset_index(drop=True,inplace=True)
            split_train_data.append(df)
        
        return split_test_data, split_train_data,split_validation_data