#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module loads assimilated GEOSCF output for and extracts chemical (trace 
species) and meteorological output for desired grid cells (i.e., C40 cities
of interest). Model output spanning the cities of interest are placed into 
CSV files, where each column corresponds to a variable of interest.                                                           
"""
DIR = '/GWSPH/groups/anenberggrp/ghkerr/'
DIR_GEOSCF = DIR+'data/GEOSCF/'
DIR_OUT = DIR_GEOSCF

import numpy as np
import xarray as xr
import pandas as pd
# import cartopy.crs as ccrs
# import shapely.ops as so
# from cartopy.io import shapereader
# from fiona.crs import from_epsg
# from shapely.geometry import Point
# import netCDF4 as nc
# # # # # This part of the code can be run offline (i.e., prior to parsing 
# # out all the GEOS-CF files). Essentially the shapefiles of the outlines 
# # from the MSAs are computed and any points located within the MSA are 
# # saved off. 
# grid = nc.Dataset(DIR_GEOSCF+'GEOS-CF.v01.rpl.aqc_tavg_1hr_g1440x721_v1.20200101_0030z.nc4')
# lat_gc = grid.variables['lat'][:]
# lng_gc = grid.variables['lon'][:]
# # # # # Los Angeles
# filename = DIR_GEOGRAPHY+'losangeles/County_Boundaries-shp/County_Boundaries.shp'
# shp = shapereader.Reader(filename)
# losangeles = shp.geometries()
# losangeles = list(losangeles)
# rec = shp.records()
# rec = list(rec)
# losangeles = so.cascaded_union([losangeles[1],losangeles[8],losangeles[9]])
# lat_los_in, lng_los_in = [], []
# for ilat in lat_gc:
#     for ilng in lng_gc: 
#         point = Point(ilng, ilat)
#         if losangeles.contains(point) is True:
#             lat_los_in.append(ilat)
#             lng_los_in.append(ilng)        
# lat_los_in = np.unique(lat_los_in)
# lng_los_in = np.unique(lng_los_in)
lat_los_in = np.array([33.75, 34., 34.25, 34.5, 34.75])
lng_los_in = np.array([-118.75, -118.5, -118.25, -118., -117.75])

# # # # # Mexico City
# filename = DIR_GEOGRAPHY+'mexicocity/'+'cdmx_transformed.shp'
# shp = shapereader.Reader(filename)
# cdmx = shp.geometries()
# cdmx = list(cdmx)[566]
# lat_mex_in, lng_mex_in = [], []
# for ilat in lat_gc:
#     for ilng in lng_gc: 
#         point = Point(ilng, ilat)
#         if cdmx.contains(point) is True:
#             lat_mex_in.append(ilat)
#             lng_mex_in.append(ilng)        
# lat_mex_in = np.unique(lat_mex_in)
# lng_mex_in = np.unique(lng_mex_in)
lat_mex_in = np.array([19.5])
lng_mex_in = np.array([-99.25])

# # # # # Santiago
# filename = DIR_GEOGRAPHY+'santiago/'+'chl_admbnda_adm1_bcn2018.shp'
# shp = shapereader.Reader(filename)
# santiago = shp.geometries()
# santiago = list(santiago)[-1]
# lat_san_in, lng_san_in = [], []
# for ilat in lat_gc:
#     for ilng in lng_gc: 
#         point = Point(ilng, ilat)
#         if santiago.contains(point) is True:
#             lat_san_in.append(ilat)
#             lng_san_in.append(ilng)   
# lat_san_in = np.unique(lat_san_in)
# lng_san_in = np.unique(lng_san_in)
lat_san_in = np.array([-34.25, -34., -33.75, -33.5, -33.25, -33.])
lng_san_in = np.array([-71.25, -71., -70.75, -70.5, -70.25, -70.])

# # # # # Berlin
# filename = DIR_GEOGRAPHY+'berlin/'+\
#     'GISPORTAL_GISOWNER01_BERLIN_BEZIRKE_BOROUGHS01.shp'
# shp = shapereader.Reader(filename)
# berlin = shp.geometries()
# berlin = list(berlin)
# berlin = so.cascaded_union(berlin)
# lat_ber_in, lng_ber_in = [], []
# for ilat in lat_gc:
#     for ilng in lng_gc: 
#         point = Point(ilng, ilat)
#         if berlin.contains(point) is True:
#             lat_ber_in.append(ilat)
#             lng_ber_in.append(ilng)   
# lat_ber_in = np.unique(lat_ber_in)
# lng_ber_in = np.unique(lng_ber_in)
lat_ber_in = np.unique([52.5])
lng_ber_in = np.unique([13.25, 13.5])

# # # # # London
# filename = DIR_GEOGRAPHY+'london/'+'london_transformed.shp'
# shp = shapereader.Reader(filename)
# london = shp.geometries()
# london = list(london)
# london = so.cascaded_union(london) 
# lat_lon_in, lng_lon_in = [], []
# for ilat in lat_gc:
#     for ilng in lng_gc: 
#         point = Point(ilng, ilat)
#         if london.contains(point) is True:
#             lat_lon_in.append(ilat)
#             lng_lon_in.append(ilng)   
# lat_lon_in = np.unique(lat_lon_in)
# lng_lon_in = np.unique(lng_lon_in)
lat_lon_in = np.array([51.5])
lng_lon_in = np.array([-2.50000000e-01, -1.45115289e-12])

# # # # # Milan 
# filename = DIR_GEOGRAPHY+'milan/'+'pd101kz6162.shp'
# shp = shapereader.Reader(filename)
# milan = shp.geometries()
# milan = list(milan)[0]
# lat_mil_in, lng_mil_in = [], []
# for ilat in lat_gc:
#     for ilng in lng_gc: 
#         point = Point(ilng, ilat)
#         if milan.contains(point) is True:
#             lat_mil_in.append(ilat)
#             lng_mil_in.append(ilng)   
# lat_mil_in = np.unique(lat_mil_in)
# lng_mil_in = np.unique(lng_mil_in)
lat_mil_in = np.array([45.5, 45.75])
lng_mil_in = np.array([9., 9.25])

# # # # Auckland 
# filename = DIR_GEOGRAPHY+'auckland/'+'NZL_adm1.shp'
# shp = shapereader.Reader(filename)
# auckland_all = shp.geometries()
# auckland = list(auckland_all)[0]
# lat_auc_in, lng_auc_in = [], []
# for ilat in lat_gc:
#     for ilng in lng_gc: 
#         point = Point(ilng, ilat)
#         if auckland.contains(point) is True:
#             lat_auc_in.append(ilat)
#             lng_auc_in.append(ilng)
# lat_auc_in = np.unique(lat_auc_in)
# lng_auc_in = np.unique(lng_auc_in)
lat_auc_in = np.array([-37.25, -37.  , -36.75, -36.5 , -36.25])
lng_auc_in = np.array([174.25, 174.5 , 174.75, 175.  , 175.25])

# Open all aqc GEOSCF files 
pattern = 'aqc_tavg_1hr/GEOS-CF.v01.rpl.aqc_tavg_1hr_g1440x721_v1.*.nc4'
geoscf = xr.open_mfdataset(DIR_GEOSCF+pattern)

# This is the main loop to loop through all cities/areas of interest; note
# that additional cities/areas can be tacked on these lists 
df = []
lngs = [lng_los_in, lng_mex_in, lng_san_in, lng_ber_in, lng_lon_in, 
    lng_mil_in, lng_auc_in]
lats = [lat_los_in, lat_mex_in, lat_san_in, lat_ber_in, lat_lon_in, 
    lat_mil_in, lat_auc_in]
cities = ['Los Angeles', 'Mexico City', 'Santiago', 'Berlin', 'London', 
    'Milan', 'Auckland']
for ci in np.arange(0, len(lngs), 1):
    print('Handling AQC for %s...'%(cities[ci]))
    lat_in = lats[ci]
    lng_in = lngs[ci]
    city = cities[ci]
    # Select coordinates within each city/area of interest
    geoscf_in = geoscf.sel(lat=lat_in, lon=lng_in, method="nearest")
    # Compute daily average (I wonder how the difference between the UTC time
    # of the model vs. the local observations will impact results...can 
    # XGBoost detect this difference and adjust for it? )
    geoscf_in = geoscf_in.resample(time='1D').mean()
    # Trigger loading of Dataset's data from disk
    geoscf_in = geoscf_in.compute()
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
df = pd.concat(df)
df.to_csv(DIR_OUT+'aqc_tavg_1d_cities.csv', index=False)
del geoscf, df

# Open all met GEOSCF files 
pattern = 'met_tavg_1hr/GEOS-CF.v01.rpl.met_tavg_1hr_g1440x721_x1.*.nc4'
geoscf = xr.open_mfdataset(DIR_GEOSCF+pattern)
df = []
lngs = [lng_los_in, lng_mex_in, lng_san_in, lng_ber_in, lng_lon_in, 
    lng_mil_in, lng_auc_in]
lats = [lat_los_in, lat_mex_in, lat_san_in, lat_ber_in, lat_lon_in, 
    lat_mil_in, lat_auc_in]
cities = ['Los Angeles', 'Mexico City', 'Santiago', 'Berlin', 'London', 
    'Milan', 'Auckland']
for ci in np.arange(0, len(lngs), 1):
    print('Handling MET for %s...'%(cities[ci]))    
    lat_in = lats[ci]
    lng_in = lngs[ci]
    city = cities[ci]
    geoscf_in = geoscf.sel(lat=lat_in, lon=lng_in, method="nearest")
    geoscf_in = geoscf_in.resample(time='1D').mean()
    geoscf_in = geoscf_in.compute()
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
df.to_csv(DIR_OUT+'met_tavg_1d_cities.csv', index=False)
del geoscf, df