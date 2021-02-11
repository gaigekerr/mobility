#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module loads assimilated GEOSCF output for and extracts chemical (trace 
species) and meteorological output for desired grid cells (i.e., C40 cities
of interest). Model output spanning the cities of interest are placed into 
CSV files, where each column corresponds to a variable of interest.                                                           
"""
DIR = '/GWSPH/groups/anenberggrp/ghkerr/'
DIR_GEOSCF = DIR+'data/GEOSCF/'
DIR_EEA = DIR+'data/EEA/'
DIR_OUT = DIR_GEOSCF

collection = 'aqc'

def pd_read_pattern(pattern):
    """Open > 1 .csv files with a wildcard character in Pandas (code from 
    (https://stackoverflow.com/questions/ 49898742/pandas-reading-csv-files-
    with-partial-wildcard)"""
    files = glob.glob(pattern)
    df = pd.DataFrame()
    for f in files:
        df = df.append(pd.read_csv(f))
    return df.reset_index(drop=True)

import glob
import numpy as np
import xarray as xr
import pandas as pd
# # # # FOR MAJOR CITIES IN THE EUROPEAN UNION (AND ISTANBUL?)
# Open all coordinate information from EEA stations in major European cities
coords_eea = pd_read_pattern(DIR_EEA+'*_coords.csv')
# # # # FOR C40 CITIES 
# This part of the code is a little kludgey and is run locally before 
# running on Pegasus. It finds the coordinates of air quality observing 
# stations in C40 cities based on stations with observations for the 
# baseline period (note that the stations generally stay the same except 
# for Los Angeles?). The model will then be sampled at the closest grid
# cell to every station. Thus, there might be model grid cells that are 
# counted > 1 in the average, but this is at least consistent with the
# # Los Angeles
# los_coords = list(zip(obs_los_base.Longitude, obs_los_base.Latitude))
# los_coords = list(set(los_coords))
# los_coords = [t for t in los_coords if not any(isinstance(n, float) 
#     and math.isnan(n) for n in t)]
# # Mexico City
# mex_coords = list(zip(obs_mex_base.Longitude, obs_mex_base.Latitude))
# mex_coords = list(set(mex_coords))
# mex_coords = [t for t in mex_coords if not any(isinstance(n, float) 
#     and math.isnan(n) for n in t)]
# # Santiago 
# san_coords = list(zip(obs_san_base.Longitude, obs_san_base.Latitude))
# san_coords = list(set(san_coords))
# san_coords = [t for t in san_coords if not any(isinstance(n, float) 
#     and math.isnan(n) for n in t)]
# # London
# lon_coords = list(zip(obs_lon_base.Longitude, obs_lon_base.Latitude))
# lon_coords = list(set(lon_coords))
# lon_coords = [t for t in lon_coords if not any(isinstance(n, float) 
#     and math.isnan(n) for n in t)]
# # Berlin
# ber_coords = list(zip(obs_ber_base.Longitude, obs_ber_base.Latitude))
# ber_coords = list(set(ber_coords))
# ber_coords = [t for t in ber_coords if not any(isinstance(n, float) 
#     and math.isnan(n) for n in t)]
# # Milan
# mil_coords = list(zip(obs_mil_base.Longitude, obs_mil_base.Latitude))
# mil_coords = list(set(mil_coords))
# mil_coords = [t for t in mil_coords if not any(isinstance(n, float) 
#     and math.isnan(n) for n in t)]
# # Auckland 
# auc_coords = list(zip(obs_auc_base.Longitude, obs_auc_base.Latitude))
# auc_coords = list(set(auc_coords))
# auc_coords = [t for t in auc_coords if not any(isinstance(n, float) 
#     and math.isnan(n) for n in t)]
# Concatenate all coordinates and city names into one DataFrame
coords_c40 = pd.DataFrame([
    (-118.200707, 33.859662, 'Los Angeles C40'),
    (-118.4564, 34.05109, 'Los Angeles C40'),
    (-118.52839, 34.38337, 'Los Angeles C40'),
    (-118.22676, 34.06653, 'Los Angeles C40'),
    (-118.13068, 34.66959, 'Los Angeles C40'),
    (-117.85038, 34.14437, 'Los Angeles C40'),
    (-117.75138, 34.06698, 'Los Angeles C40'),
    (-118.204989, 33.901445, 'Los Angeles C40'),
    (-117.92392, 34.13648, 'Los Angeles C40'),
    (-118.5327499, 34.1992, 'Los Angeles C40'),
    (-118.0685, 34.01029, 'Los Angeles C40'),
    (-118.43049, 33.95507, 'Los Angeles C40'),
    (-118.12714, 34.13265, 'Los Angeles C40'),
    (-99.202603, 19.40405, 'Mexico City C40'),
    (-99.09659, 19.658223, 'Mexico City C40'),
    (-99.046176, 19.473692, 'Mexico City C40'),
    (-99.1761, 19.3262, 'Mexico City C40'),
    (-99.204136, 19.325146, 'Mexico City C40'),
    (-99.117641, 19.384413, 'Mexico City C40'),
    (-99.119594, 19.42461, 'Mexico City C40'),
    (-99.207658, 19.2721, 'Mexico City C40'),
    (-99.0824, 19.525995, 'Mexico City C40'),
    (-99.204597, 19.529077, 'Mexico City C40'),
    (-99.030324, 19.532968, 'Mexico City C40'),
    (-98.990189, 19.1769, 'Mexico City C40'),
    (-99.291705, 19.365313, 'Mexico City C40'),
    (-99.010564, 19.246459, 'Mexico City C40'),
    (-99.009381, 19.34561, 'Mexico City C40'),
    (-99.07388, 19.360794, 'Mexico City C40'),
    (-98.902853, 19.460415, 'Mexico City C40'),
    (-99.243524, 19.482473, 'Mexico City C40'),
    (-99.103629, 19.304441, 'Mexico City C40'),
    (-99.028212, 19.393734, 'Mexico City C40'),
    (-99.262865, 19.357357, 'Mexico City C40'),
    (-99.177173, 19.602542, 'Mexico City C40'),
    (-99.254133, 19.576963, 'Mexico City C40'),
    (-99.039644, 19.578792, 'Mexico City C40'),
    (-99.158969, 19.371612, 'Mexico City C40'),
    (-99.169794, 19.468404, 'Mexico City C40'),
    (-99.198602, 19.722186, 'Mexico City C40'),
    (-99.094517, 19.4827, 'Mexico City C40'),
    (-99.152207, 19.411617, 'Mexico City C40'),
    (-70.52325614853385, -33.37677604807965, 'Santiago C40'),
    (-70.66070229344474, -33.464176505068316, 'Santiago C40'),
    (-70.9529961584255, -33.67381931299035, 'Santiago C40'),
    (-70.59443067815702, -33.59135624038916, 'Santiago C40'),
    (-70.75014408796092, -33.437785357125556, 'Santiago C40'),
    (-70.73205509366429, -33.43307040276798, 'Santiago C40'),
    (-70.66616286078217, -33.54701601896996, 'Santiago C40'),
    (-70.65113863328291, -33.422261189967536, 'Santiago C40'),
    (-70.58816010358973, -33.5166668832046, 'Santiago C40'),
    (0.08291, 51.56948, 'London C40'),
    (-0.013630000000000001, 51.55624, 'London C40'),
    (-0.016069999999999997, 51.464690000000004, 'London C40'),
    (-0.11615, 51.55538, 'London C40'),
    (0.20143699999999998, 51.289390999999995, 'London C40'),
    (-0.17881, 51.4955, 'London C40'),
    (-0.28126, 51.37931, 'London C40'),
    (0.13285999999999998, 51.52939, 'London C40'),
    (-0.0049, 51.56238, 'London C40'),
    (-0.19086, 51.4902, 'London C40'),
    (-0.48098, 51.48799, 'London C40'),
    (-0.26561999999999997, 51.51895, 'London C40'),
    (-0.44056000000000006, 51.47917, 'London C40'),
    (-0.24779, 51.5378, 'London C40'),
    (-0.31008, 51.4894, 'London C40'),
    (-0.17661, 51.61468, 'London C40'),
    (-0.42376, 51.48113, 'London C40'),
    (-0.14165999999999998, 51.38929, 'London C40'),
    (-0.0252606, 51.47953, 'London C40'),
    (-0.12534, 51.61387, 'London C40'),
    (-0.09162999999999999, 51.5105, 'London C40'),
    (-0.25467, 51.50385, 'London C40'),
    (0.17789000000000002, 51.56375, 'London C40'),
    (-0.40278, 51.55226, 'London C40'),
    (-0.14739000000000002, 51.49323, 'London C40'),
    (-0.16652, 51.42933, 'London C40'),
    (-0.21589, 51.46372, 'London C40'),
    (-0.13516, 51.51607, 'London C40'),
    (-0.17528, 51.54422, 'London C40'),
    (-0.24877, 51.552659999999996),
    (-0.13187000000000001, 51.42821, 'London C40'),
    (-0.21586999999999998, 51.463429999999995, 'London C40'),
    (-0.12455, 51.485490000000006, 'London C40'),
    (0.17908, 51.57298, 'London C40'),
    (-0.14711475300000001, 51.49224823, 'London C40'),
    (-0.1254, 51.58398, 'London C40'),
    (-0.03742, 51.449670000000005, 'London C40'),
    (0.07400000000000001, 51.49053, 'London C40'),
    (-0.16434, 51.499140000000004, 'London C40'),
    (-0.15459, 51.52254, 'London C40'),
    (-0.01644, 51.601729999999996, 'London C40'),
    (-0.21581999999999998, 51.46503, 'London C40'),
    (-0.14179, 51.479440000000004, 'London C40'),
    (-0.13642, 51.38357, 'London C40'),
    (-0.1176, 51.36223, 'London C40'),
    (-0.48668, 51.48148, 'London C40'),
    (0.0179, 51.48688, 'London C40'),
    (-0.13194, 51.494679999999995, 'London C40'),
    (-0.29249, 51.53085, 'London C40'),
    (-0.14565999999999998, 51.5168, 'London C40'),
    (-0.10153, 51.493159999999996, 'London C40'),
    (-0.01238, 51.4725, 'London C40'),
    (-0.02027, 51.44547, 'London C40'),
    (-0.09676, 51.37395, 'London C40'),
    (-0.36299000000000003, 51.58842, 'London C40'),
    (-0.050769999999999996, 51.61486, 'London C40'),
    (-0.04216, 51.522529999999996, 'London C40'),
    (0.08561, 51.4563, 'London C40'),
    (-0.23734, 51.48019, 'London C40'),
    (-0.12163, 51.51198, 'London C40'),
    (-0.15279, 51.51393, 'London C40'),
    (-0.039639999999999995, 51.47495, 'London C40'),
    (-0.07777, 51.51385, 'London C40'),
    (-0.12311, 51.41135, 'London C40'),
    (-0.25809, 51.552479999999996, 'London C40'),
    (0.09511, 51.486959999999996, 'London C40'),
    (-0.4119, 51.48298, 'London C40'),
    (-0.03331, 51.54052, 'London C40'),
    (-0.19107000000000002, 51.456959999999995, 'London C40'),
    (-0.36476, 51.47913, 'London C40'),
    (0.030860000000000002, 51.576609999999995, 'London C40'),
    (-0.41233000000000003, 51.49817, 'London C40'),
    (-0.02201, 51.66864, 'London C40'),
    (-0.29878000000000005, 51.617329999999995, 'London C40'),
    (0.01888, 51.40555, 'London C40'),
    (-0.4557, 51.48438, 'London C40'),
    (-0.21349, 51.52105, 'London C40'),
    (-0.25725, 51.492509999999996, 'London C40'),
    (-0.14991, 51.49349, 'London C40'),
    (-0.12019, 51.51737, 'London C40'),
    (0.18488, 51.46598, 'London C40'),
    (-0.16671, 51.46369, 'London C40'),
    (-0.21771999999999997, 51.53241, 'London C40'),
    (-0.12877, 51.52798, 'London C40'),
    (0.07077, 51.45258, 'London C40'),
    (-0.20599, 51.5919, 'London C40'),
    (-0.2655, 51.52361, 'London C40'),
    (-0.28438, 51.500679999999996, 'London C40'),
    (-0.442092, 51.48107, 'London C40'),
    (-0.0782, 51.50139, 'London C40'),
    (-0.31751, 51.48986, 'London C40'),
    (0.00041, 51.483909999999995, 'London C40'),
    (0.20546, 51.520790000000005, 'London C40'),
    (-0.00214, 51.5376, 'London C40'),
    (-0.22466999999999998, 51.50456, 'London C40'),
    (0.01078, 51.49377, 'London C40'),
    (-0.12585, 51.522290000000005, 'London C40'),
    (-0.14972, 51.35866, 'London C40'),
    (0.13727999999999999, 51.49465, 'London C40'),
    (-0.46083, 51.49631, 'London C40'),
    (-0.24040999999999998, 51.37792, 'London C40'),
    (-0.09611, 51.52023, 'London C40'),
    (-0.05955, 51.4805, 'London C40'),
    (-0.34121999999999997, 51.453140000000005, 'London C40'),
    (0.06422, 51.43466, 'London C40'),
    (-0.29658, 51.41231, 'London C40'),
    (-0.10699000000000002, 51.5579, 'London C40'),
    (0.15891, 51.49061, 'London C40'),
    (0.01455, 51.51473, 'London C40'),
    (-0.06617999999999999, 51.64504, 'London C40'),
    (-0.1684, 51.48744, 'London C40'),
    (-0.23043000000000002, 51.47617, 'London C40'),
    (-0.012980000000000002, 51.489129999999996, 'London C40'),
    (-0.00842, 51.51505, 'London C40'),
    (-0.44163, 51.48878, 'London C40'),
    (-0.25703000000000004, 51.4355, 'London C40'),
    (-0.08491, 51.52645, 'London C40'),
    (-0.22479000000000002, 51.4927, 'London C40'),
    (-0.15091, 51.513000000000005, 'London C40'),
    (-0.06822, 51.5993, 'London C40'),
    (0.04073, 51.45636, 'London C40'),
    (-0.40873000000000004, 51.447390000000006, 'London C40'),
    (-0.42753, 51.4634, 'London C40'),
    (-0.19589, 51.40162, 'London C40'),
    (-0.11671, 51.51197, 'London C40'),
    (-0.11458, 51.46411, 'London C40'),
    (13.368103, 52.398406, 'Berlin C40'),
    (13.44173, 52.467535, 'Berlin C40'),
    (13.296081, 52.653269, 'Berlin C40'),
    (13.64705, 52.447697, 'Berlin C40'),
    (13.388321, 52.510178, 'Berlin C40'),
    (13.31825, 52.463611, 'Berlin C40'),
    (13.529504, 52.485296, 'Berlin C40'),
    (13.348775, 52.485814, 'Berlin C40'),
    (13.349326, 52.543041, 'Berlin C40'),
    (13.433967, 52.481709, 'Berlin C40'),
    (13.332972, 52.5066, 'Berlin C40'),
    (13.418833, 52.513606, 'Berlin C40'),
    (13.430844, 52.489451, 'Berlin C40'),
    (13.38772, 52.438115, 'Berlin C40'),
    (13.225144, 52.473192, 'Berlin C40'),
    (13.469931, 52.514072, 'Berlin C40'),
    (13.489531, 52.643525, 'Berlin C40'),
    (9.167944501742676, 45.443857653564926, 'Milan C40'),
    (9.195324807668857, 45.463346740666545, 'Milan C40'),
    (9.197460360112531, 45.470499014097, 'Milan C40'),
    (9.190933555313624, 45.49631644365102, 'Milan C40'),
    (9.235491038497502, 45.47899606168744, 'Milan C40'),
    (174.81556349, -36.90455276, 'Auckland C40'),
    (174.77116271, -36.86634888, 'Auckland C40'),
    (174.76558144, -36.84764873, 'Auckland C40'),
    (174.86430049, -37.20443705, 'Auckland C40'),
    (174.6283504, -36.86799137, 'Auckland C40'),
    (174.65200725, -36.92213583, 'Auckland C40'),
    (174.74884838, -36.78025339, 'Auckland C40')],
    columns=['Longitude', 'Latitude', 'City'])

# Open all aqc GEOSCF files 
if collection=='aqc':
    pattern = 'aqc_tavg_1hr/GEOS-CF.v01.rpl.aqc_tavg_1hr_g1440x721_v1.*.nc'
    geoscf = xr.open_mfdataset(DIR_GEOSCF+pattern)
if collection=='met':
    pattern = 'met_tavg_1hr/GEOS-CF.v01.rpl.met_tavg_1hr_g1440x721_x1.*.nc4'
    geoscf = xr.open_mfdataset(DIR_GEOSCF+pattern)    
print('GEOS-CF opened with dimensions of...', geoscf.dims)
# Combine C40/EEA stations in a single DataFrame
aqlocs = pd.concat([coords_eea[['Longitude','Latitude','City']], 
    coords_c40])

# This is the main loop to loop over the air monitoring statinos and pick 
# off the closet GEOS-CF grid cell
df = []
for index, row in aqlocs.iterrows():
    print('Handling AQC for %s...'%(row['City']))
    lat_in = row['Latitude']
    lng_in = row['Longitude']
    city = row['City']
    # Select coordinates within each city/area of interest
    geoscf_in = geoscf.sel(lat=lat_in, lon=lng_in, method="nearest")
    # Compute daily average (I wonder how the difference between the UTC time
    # of the model vs. the local observations will impact results...can 
    # XGBoost detect this difference and adjust for it? )
    geoscf_in = geoscf_in.resample(time='1D').mean()
    # Trigger loading of Dataset's data from disk
    geoscf_in = geoscf_in.compute()
    if collection=='aqc':
        # Extract coordinates and values
        time_in = geoscf_in.time.data
        lat_in = geoscf_in.lat.data
        lng_in = geoscf_in.lon.data
        co_in = geoscf_in.CO.data
        no2_in = geoscf_in.NO2.data
        o3_in = geoscf_in.O3.data
        pm25_in = geoscf_in.PM25_RH35_GCC.data
        so2_in = geoscf_in.SO2.data
        # Write data as a table with columns for each variable (time, latitude, 
        # longitude, species); note that this is similar to Method 2 from 
        # https://confluence.ecmwf.int/display/CKB/How+to+convert+NetCDF+to+CSV
        time_in_grid, lat_in_grid, lng_in_grid = [x.flatten() for x in 
            np.meshgrid(time_in, lat_in, lng_in, indexing='ij')]
        df_in = pd.DataFrame({
            'time': [t for t in time_in_grid],
            'latitude': lat_in_grid,
            'longitude': lng_in_grid,
            'CO': co_in[:].flatten(),
            'NO2':no2_in[:].flatten(),
            'O3':o3_in[:].flatten(),
            'PM25':pm25_in[:].flatten(),
            'SO2':so2_in[:].flatten(),
            'city':city})
        df.append(df_in)
    if collection=='met':
        # Extract everything but phis (surface geopotential height), troppb
        # (tropopause_pressure_based_on_blended_estimate), zl 
        # (mid_layer_heights)
        time_in = geoscf_in.time.data
        lat_in = geoscf_in.lat.data
        lng_in = geoscf_in.lon.data
        cldtt_in = geoscf_in.CLDTT.data
        ps_in = geoscf_in.PS.data
        q_in = geoscf_in.Q.data
        q10m_in = geoscf_in.Q10M.data
        q2m_in = geoscf_in.Q2M.data
        rh_in = geoscf_in.RH.data
        slp_in = geoscf_in.SLP.data
        t_in = geoscf_in.T.data
        t10m_in = geoscf_in.T10M.data
        t2m_in = geoscf_in.T2M.data
        tprec_in = geoscf_in.TPREC.data
        ts_in = geoscf_in.TS.data
        u_in = geoscf_in.U.data
        u10m_in = geoscf_in.U10M.data
        u2m_in = geoscf_in.U2M.data
        v_in = geoscf_in.V.data
        v10m_in = geoscf_in.V10M.data
        v2m_in = geoscf_in.V2M.data
        zpbl_in = geoscf_in.ZPBL.data
        time_in_grid, lat_in_grid, lng_in_grid = [x.flatten() for x in 
            np.meshgrid(time_in, lat_in, lng_in, indexing='ij')]
        df_in = pd.DataFrame({
            'time': [t for t in time_in_grid],
            'latitude': lat_in_grid,
            'longitude': lng_in_grid,
            'CLDTT': cldtt_in[:].flatten(),
            'PS':ps_in[:].flatten(),
            'Q':q_in[:].flatten(),
            'Q10M':q10m_in[:].flatten(),
            'Q2M':q2m_in[:].flatten(),
            'RH':rh_in[:].flatten(),
            'SLP':slp_in[:].flatten(),
            'T':t_in[:].flatten(),
            'T10M':t10m_in[:].flatten(),
            'T2M':t2m_in[:].flatten(),
            'TPREC':tprec_in[:].flatten(),
            'TS':ts_in[:].flatten(),
            'U':u_in[:].flatten(),
            'U10M':u10m_in[:].flatten(),
            'U2M':u2m_in[:].flatten(),
            'V':v_in[:].flatten(),
            'V10M':v10m_in[:].flatten(),
            'V2M':v2m_in[:].flatten(),
            'ZPBL':zpbl_in[:].flatten(),        
            'city':city})    
        df.append(df_in)
df = pd.concat(df)
if collection=='aqc':
    df.to_csv(DIR_OUT+'aqc_tavg_1d_cities.csv', index=False)
if collection=='met':
    df.to_csv(DIR_OUT+'met_tavg_1d_cities.csv', index=False)

# lngs = [lng_los_in, lng_mex_in, lng_san_in, lng_ber_in, lng_lon_in, 
#     lng_mil_in, lng_auc_in]
# lats = [lat_los_in, lat_mex_in, lat_san_in, lat_ber_in, lat_lon_in, 
#     lat_mil_in, lat_auc_in]
# cities = ['Los Angeles', 'Mexico City', 'Santiago', 'Berlin', 'London', 
#     'Milan', 'Auckland']
# Open all met GEOSCF files 
# df = []
# lngs = [lng_los_in, lng_mex_in, lng_san_in, lng_ber_in, lng_lon_in, 
#     lng_mil_in, lng_auc_in]
# lats = [lat_los_in, lat_mex_in, lat_san_in, lat_ber_in, lat_lon_in, 
#     lat_mil_in, lat_auc_in]
# cities = ['Los Angeles', 'Mexico City', 'Santiago', 'Berlin', 'London', 
#     'Milan', 'Auckland']
# for ci in np.arange(0, len(lngs), 1):
#     print('Handling MET for %s...'%(cities[ci]))    
#     lat_in = lats[ci]
#     lng_in = lngs[ci]
#     city = cities[ci]
#     geoscf_in = geoscf.sel(lat=lat_in, lon=lng_in, method="nearest")
#     geoscf_in = geoscf_in.resample(time='1D').mean()
#     geoscf_in = geoscf_in.compute()
#     # Extract everything but phis (surface geopotential height), troppb
#     # (tropopause_pressure_based_on_blended_estimate), zl 
#     # (mid_layer_heights)
#     time_in = geoscf_in.time.data
#     lat_in = geoscf_in.lat.data
#     lng_in = geoscf_in.lon.data
#     cldtt_in = geoscf_in.CLDTT.data
#     ps_in = geoscf_in.PS.data
#     q_in = geoscf_in.Q.data
#     q10m_in = geoscf_in.Q10M.data
#     q2m_in = geoscf_in.Q2M.data
#     rh_in = geoscf_in.RH.data
#     slp_in = geoscf_in.SLP.data
#     t_in = geoscf_in.T.data
#     t10m_in = geoscf_in.T10M.data
#     t2m_in = geoscf_in.T2M.data
#     tprec_in = geoscf_in.TPREC.data
#     ts_in = geoscf_in.TS.data
#     u_in = geoscf_in.U.data
#     u10m_in = geoscf_in.U10M.data
#     u2m_in = geoscf_in.U2M.data
#     v_in = geoscf_in.V.data
#     v10m_in = geoscf_in.V10M.data
#     v2m_in = geoscf_in.V2M.data
#     zpbl_in = geoscf_in.ZPBL.data
#     time_in_grid, lat_in_grid, lng_in_grid = [x.flatten() for x in 
#         np.meshgrid(time_in, lat_in, lng_in, indexing='ij')]
#     df_in = pd.DataFrame({
#         'time': [t for t in time_in_grid],
#         'latitude': lat_in_grid,
#         'longitude': lng_in_grid,
#         'CLDTT': cldtt_in[:].flatten(),
#         'PS':ps_in[:].flatten(),
#         'Q':q_in[:].flatten(),
#         'Q10M':q10m_in[:].flatten(),
#         'Q2M':q2m_in[:].flatten(),
#         'RH':rh_in[:].flatten(),
#         'SLP':slp_in[:].flatten(),
#         'T':t_in[:].flatten(),
#         'T10M':t10m_in[:].flatten(),
#         'T2M':t2m_in[:].flatten(),
#         'TPREC':tprec_in[:].flatten(),
#         'TS':ts_in[:].flatten(),
#         'U':u_in[:].flatten(),
#         'U10M':u10m_in[:].flatten(),
#         'U2M':u2m_in[:].flatten(),
#         'V':v_in[:].flatten(),
#         'V10M':v10m_in[:].flatten(),
#         'V2M':v2m_in[:].flatten(),
#         'ZPBL':zpbl_in[:].flatten(),        
#         'city':city})    
#     df.append(df_in)
# df = pd.concat(df)
# df.to_csv(DIR_OUT+'met_tavg_1d_cities.csv', index=False)
# del geoscf, df
# # Find nearest GEOS-CF grid cell for all EEA stations
# cities, lat_eeaatgc, lng_eeaatgc = [], [], []
# for index, row in coords_eea.iterrows():
#     lat_nearest = lat_gc[np.abs(lat_gc-row['Latitude']).argmin()]
#     lng_nearest = lng_gc[np.abs(lng_gc-row['Longitude']).argmin()]
#     city = row['City']
#     # Append values to multi-city list
#     lat_eeaatgc.append(lat_nearest)
#     lng_eeaatgc.append(lng_nearest)    
#     cities.append(city)
# # import cartopy.crs as ccrs
# # import shapely.ops as so
# # from cartopy.io import shapereader
# # from fiona.crs import from_epsg
# # from shapely.geometry import Point
# # This part of the code can be run offline (i.e., prior to parsing 
# # out all the GEOS-CF files). Essentially the shapefiles of the outlines 
# # from the MSAs are computed and any points located within the MSA are 
# # saved off. 
# # # # # # Los Angeles
# # filename = DIR_GEOGRAPHY+'losangeles/County_Boundaries-shp/County_Boundaries.shp'
# # shp = shapereader.Reader(filename)
# # losangeles = shp.geometries()
# # losangeles = list(losangeles)
# # rec = shp.records()
# # rec = list(rec)
# # losangeles = so.cascaded_union([losangeles[1],losangeles[8],losangeles[9]])
# # lat_los_in, lng_los_in = [], []
# # for ilat in lat_gc:
# #     for ilng in lng_gc: 
# #         point = Point(ilng, ilat)
# #         if losangeles.contains(point) is True:
# #             lat_los_in.append(ilat)
# #             lng_los_in.append(ilng)        
# # lat_los_in = np.unique(lat_los_in)
# # lng_los_in = np.unique(lng_los_in)
# # lat_los_in = np.array([33.75, 34., 34.25, 34.5, 34.75])
# # lng_los_in = np.array([-118.75, -118.5, -118.25, -118., -117.75])
# # # # # # Mexico City
# # filename = DIR_GEOGRAPHY+'mexicocity/'+'cdmx_transformed.shp'
# # shp = shapereader.Reader(filename)
# # cdmx = shp.geometries()
# # cdmx = list(cdmx)[566]
# # lat_mex_in, lng_mex_in = [], []
# # for ilat in lat_gc:
# #     for ilng in lng_gc: 
# #         point = Point(ilng, ilat)
# #         if cdmx.contains(point) is True:
# #             lat_mex_in.append(ilat)
# #             lng_mex_in.append(ilng)        
# # lat_mex_in = np.unique(lat_mex_in)
# # lng_mex_in = np.unique(lng_mex_in)
# lat_mex_in = np.array([19.5])
# lng_mex_in = np.array([-99.25])
# # # # # # Santiago
# # filename = DIR_GEOGRAPHY+'santiago/'+'chl_admbnda_adm1_bcn2018.shp'
# # shp = shapereader.Reader(filename)
# # santiago = shp.geometries()
# # santiago = list(santiago)[-1]
# # lat_san_in, lng_san_in = [], []
# # for ilat in lat_gc:
# #     for ilng in lng_gc: 
# #         point = Point(ilng, ilat)
# #         if santiago.contains(point) is True:
# #             lat_san_in.append(ilat)
# #             lng_san_in.append(ilng)   
# # lat_san_in = np.unique(lat_san_in)
# # lng_san_in = np.unique(lng_san_in)
# lat_san_in = np.array([-34.25, -34., -33.75, -33.5, -33.25, -33.])
# lng_san_in = np.array([-71.25, -71., -70.75, -70.5, -70.25, -70.])
# # # # # # Berlin
# # filename = DIR_GEOGRAPHY+'berlin/'+\
# #     'GISPORTAL_GISOWNER01_BERLIN_BEZIRKE_BOROUGHS01.shp'
# # shp = shapereader.Reader(filename)
# # berlin = shp.geometries()
# # berlin = list(berlin)
# # berlin = so.cascaded_union(berlin)
# # lat_ber_in, lng_ber_in = [], []
# # for ilat in lat_gc:
# #     for ilng in lng_gc: 
# #         point = Point(ilng, ilat)
# #         if berlin.contains(point) is True:
# #             lat_ber_in.append(ilat)
# #             lng_ber_in.append(ilng)   
# # lat_ber_in = np.unique(lat_ber_in)
# # lng_ber_in = np.unique(lng_ber_in)
# lat_ber_in = np.unique([52.5])
# lng_ber_in = np.unique([13.25, 13.5])
# # # # # # London
# # filename = DIR_GEOGRAPHY+'london/'+'london_transformed.shp'
# # shp = shapereader.Reader(filename)
# # london = shp.geometries()
# # london = list(london)
# # london = so.cascaded_union(london) 
# # lat_lon_in, lng_lon_in = [], []
# # for ilat in lat_gc:
# #     for ilng in lng_gc: 
# #         point = Point(ilng, ilat)
# #         if london.contains(point) is True:
# #             lat_lon_in.append(ilat)
# #             lng_lon_in.append(ilng)   
# # lat_lon_in = np.unique(lat_lon_in)
# # lng_lon_in = np.unique(lng_lon_in)
# lat_lon_in = np.array([51.5])
# lng_lon_in = np.array([-2.50000000e-01, -1.45115289e-12])
# # # # # # Milan 
# # filename = DIR_GEOGRAPHY+'milan/'+'pd101kz6162.shp'
# # shp = shapereader.Reader(filename)
# # milan = shp.geometries()
# # milan = list(milan)[0]
# # lat_mil_in, lng_mil_in = [], []
# # for ilat in lat_gc:
# #     for ilng in lng_gc: 
# #         point = Point(ilng, ilat)
# #         if milan.contains(point) is True:
# #             lat_mil_in.append(ilat)
# #             lng_mil_in.append(ilng)   
# # lat_mil_in = np.unique(lat_mil_in)
# # lng_mil_in = np.unique(lng_mil_in)
# lat_mil_in = np.array([45.5, 45.75])
# lng_mil_in = np.array([9., 9.25])
# # # # # Auckland 
# # filename = DIR_GEOGRAPHY+'auckland/'+'NZL_adm1.shp'
# # shp = shapereader.Reader(filename)
# # auckland_all = shp.geometries()
# # auckland = list(auckland_all)[0]
# # lat_auc_in, lng_auc_in = [], []
# # for ilat in lat_gc:
# #     for ilng in lng_gc: 
# #         point = Point(ilng, ilat)
# #         if auckland.contains(point) is True:
# #             lat_auc_in.append(ilat)
# #             lng_auc_in.append(ilng)
# # lat_auc_in = np.unique(lat_auc_in)
# # lng_auc_in = np.unique(lng_auc_in)
# lat_auc_in = np.array([-37.25, -37.  , -36.75, -36.5 , -36.25])
# lng_auc_in = np.array([174.25, 174.5 , 174.75, 175.  , 175.25])

# import netCDF4 as nc
# # Open a random GEOSCF file to extract coordinate information
# grid = nc.Dataset(DIR_GEOSCF+
#     'GEOS-CF.v01.rpl.aqc_tavg_1hr_g1440x721_v1.20190501.nc', 'r')
# lat_gc = grid.variables['lat'][:]
# lng_gc = grid.variables['lon'][:]
