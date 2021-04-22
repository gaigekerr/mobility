#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and format NO2 observations from openAQ

Script uses the openAQ API to download NO2 observations for 2018-2020 for 
select large cities (generally the largest city in each country with NO2 
observations) and formats them to the same format as EEA observations. 

Created on Wed Mar 31 17:29:56 2021
"""
DIR_ROOT = '/Users/ghkerr/GW/'
DIR_OUT = DIR_ROOT+'data/aq/openaq/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# openaq package install via pip ("pip install py-openaq")
import openaq
api = openaq.OpenAQ()
# # Fetch locations of OpenAQ sites
# status, loc = api.locations(limit=100000)
# loc = loc['results']
# # The following code and what it accomplishes is a bit clunky, but
# # essentially I was interested in finding which countries have NO2 
# # monitors and if there were any in capital/largest cities. To look 
# # at this, run the following: 
# city_with_no2, where_no2, country_with_no2 = [], [], []
# for x in np.arange(len(loc)):
#     if 'no2' in loc[x]['parameters']:
#         city_with_no2.append(loc[x]['city'])
#         country_with_no2.append(loc[x]['country'])
#         where_no2.append(x)
# # Get a list of all countries with NO2 monitors
# np.unique(country_with_no2)
# As of 31 March 2021, this yields
# ['AD', 'AR', 'AT', 'AU', 'BA', 'BE', 'BG', 'BR', 'CA', 'CH', 'CL',
# 'CN', 'CO', 'CY', 'CZ', 'DE', 'DK', 'EC', 'EE', 'ES', 'FI', 'FR',
# 'GB', 'GI', 'GR', 'HK', 'HR', 'HU', 'IE', 'IL', 'IN', 'IS', 'IT',
# 'LT', 'LU', 'LV', 'MK', 'MN', 'MT', 'MX', 'NL', 'NO', 'PE', 'PL',
# 'PT', 'RO', 'RS', 'RU', 'RW', 'SE', 'SK', 'TH', 'TR', 'TW', 'US',
# 'XK']
#  I went through and fetched all the city names within each country using
# np.array(city_with_no2)[np.where(np.array(country_with_no2)=='CH')]
# then handpicked the list of biggest cities (and their English equivalent
# names). The following dataframe is used to grab data from openAQ and 
# the columns represent the following: country code, city name (from 
# openAQ site), and city name (in English)
cities = [['AR', ['Buenos Aires'], 'Buenos Aires'], 
    ['AU', ['Sydney North-west', 'Sydney East', 'Sydney North-west', 
            'Sydney South-west'], 'Sydney'], 
    #['BA', ['Sarajevo'], 'Sarajevo'], 
    ['BE', ['Brussels-Capital Region'], 'Brussels'], 
    ['BR', ['Sao Paulo', 'São Paulo'], 'Brazil'], 
    ['CH', ['Zürich'], 'Zurich'], 
    ['CA', ['N/A', 'QUEBEC'], 'Montreal'], # Deleted the final entry in 
    # Canada's coordinates info as it corresponds to Quebec City (i.e., this 
    # line: 7,-71.3697,46.7742,972,Montreal)
    ['CN', ['重庆市'], 'Chonqing'], # Beijing doesn't have good obs
    ['CO', ['Medellin'], 'Medellin'],
    #['CY', ['Λευκωσία'], 'Nicosia'],
    #['EC', ['Quito'], 'Quito'], 
    #['EE', ['Harjumaa'], 'Tallinn'], 
    #['HK', ['New Territories', 'Kowloon', 'Sai Kung', 'Central & Western',
    #   'Central', 'Causeway Bay', 'New Territories', 'Kowloon', 'Kowloon',
    #   'New Territories', 'New Territories', 'Tap Mun Police Post',
    #   'New Territories', 'New Territories', 'Eastern', 'N.T.'], 'Hong Kong'],
    ['IE', ['Dublin City'], 'Dublin'],
    ['IN', ['Mumbai','Navi Mumbai'], 'Mumbai']
    #['IL', ['ירושלים'], 'Jerusalem'],
    #['IS', ['Reykjavík'], 'Reykjavik'],
    ['LU', ['Luxemburg'], 'Luxembourg'],
    ['LV', ['Riga'], 'Riga'], 
    #['MN', ['Ulaanbaatar'], 'Ulaanbaatar'],
    ['NO', ['Oslo'], 'Oslo'],
    #['PE', ['Lima'], 'Lima'], 
    #['PT', ['Lisboa'], 'Lisbon'], 
    ['RS', ['Grad Beograd'], 'Belgrade'], 
    #['RU', ['Moscow'], 'Moscow'], 
    #['RW', ['Kigali'], 'Kigali'], 
    ['SK', ['Bratislavský kraj'], 'Bratislava'], 
    ['TH', ['Bangkok'], 'Bangkok'], 
    #['TR', ['İstanbul'], 'Istanbul'], 
    ['TW', ['臺北市'], 'Tapei']]
# After going through the output files for the first time, I found that 
# BA, CY, EC, HK, IL, IS, PE, PT, RW, TR all had an inadequate number
# of data points (and no other good cities were found in the countries), 
# so these lines were commented out in the city data DataFrame
# To check data from cities, use the following: 
# for index, row in cities.iterrows():
#     try: 
#         city = pd.read_csv(DIR_OUT+'%s_no2_2018-2020_timeseries.csv'%(row.name), 
#             engine='python')
#         city['DatetimeBegin'] = pd.to_datetime(city['DatetimeBegin'])
#         plt.plot(city.groupby(by='DatetimeBegin').mean().index, 
#             city.groupby(by='DatetimeBegin').mean()['Concentration'])
#         plt.title(row.name)
#         plt.xlim(['2019-01-01','2020-06-30'])
#         plt.show()
#     except FileNotFoundError:
#         pass
cities = pd.DataFrame(data=cities, 
    columns=['Country Code', 'Local Name', 'English Name'])
cities.set_index('Country Code', inplace=True)
limit = 100000
# Loop through each country/city 
for index, row in cities.iterrows():
    # Cities in country of interest
    cities_coi = row['Local Name']
    print('Downloading observations from %s...'%row['English Name'])
    # Grab data (but chunk it into yearly chunks to deal with the limit of 
    # 100000 entries/observations)
    # For 2018
    status, res2018a = api.measurements(city=cities_coi,  parameter='no2', 
        date_from='2018-01-01', date_to='2018-06-30', limit=limit)
    res2018a = res2018a['results']
    res2018a = pd.json_normalize(res2018a)
    # Notify in case DataFrame has > 100000 rows and therefore wasn't 
    # entirely downloaded because of the limit    
    if len(res2018a)>limit:
        print('Warning: >%d observations in 2018a for %s'%(limit, 
            row['English Name']))    
    status, res2018b = api.measurements(city=cities_coi,  parameter='no2', 
        date_from='2018-07-01', date_to='2018-12-31', limit=limit)
    res2018b = res2018b['results']
    res2018b = pd.json_normalize(res2018b)
    if len(res2018b)>limit:
        print('Warning: >%d observations in 2018b for %s'%(limit, 
            row['English Name']))
    # For 2019
    status, res2019a = api.measurements(city=cities_coi,  parameter='no2', 
        date_from='2019-01-01', date_to='2019-06-30', limit=limit)
    res2019a = res2019a['results']
    res2019a = pd.json_normalize(res2019a)
    if len(res2019a)>limit:
        print('Warning: >%d observations in 2019a for %s'%(limit, 
            row['English Name']))    
    status, res2019b = api.measurements(city=cities_coi,  parameter='no2', 
        date_from='2019-07-01', date_to='2019-12-31', limit=limit)
    res2019b = res2019b['results']
    res2019b = pd.json_normalize(res2019b)
    if len(res2019b)>limit:
        print('Warning: >%d observations in 2019b for %s'%(limit, 
            row['English Name']))
    # For 2020
    status, res2020a = api.measurements(city=cities_coi,  parameter='no2', 
        date_from='2020-01-01', date_to='2020-06-30', limit=limit)
    res2020a = res2020a['results']
    res2020a = pd.json_normalize(res2020a)
    if len(res2020a)>limit:
        print('Warning: >%d observations in 2020a for %s'%(limit, 
            row['English Name']))    
    status, res2020b = api.measurements(city=cities_coi,  parameter='no2', 
        date_from='2020-07-01', date_to='2020-12-31', limit=limit)
    res2020b = res2020b['results']
    res2020b = pd.json_normalize(res2020b)
    if len(res2020b)>limit:
        print('Warning: >%d observations in 2020b for %s'%(limit, 
            row['English Name']))    
    # Merge DataFrames
    merged = (res2018a, res2018b, res2019a, res2019b, res2020a, res2020b)
    merged = pd.concat(merged, ignore_index=True)
    if merged.empty==False:
        # Time to Datetime
        merged['date.utc'] = pd.to_datetime(merged['date.utc'])
        merged.set_index('date.utc', inplace=True)
        # Convert from ppm to ppb or µg/m³ to ppb
        if len(np.unique(merged['unit'].values))>1:
            print('Warning: mixture of units for %s'%(row['English Name']))
        if np.unique(merged['unit'].values)[0] == 'ppm':
            merged['value'] *= 1e3
        if np.unique(merged['unit'].values)[0] == 'µg/m³':
            merged['value'] *= 1/1.88 # 1 ppb = 1.88 μg/m3
        # Calculate daily mean
        merged = merged.query("value >= 0.0").groupby(
            by='location').resample('1D').mean()
        merged.reset_index(inplace=True)
        # Rename columns to be analogous to observations from EEA
        merged.rename(columns={'location': 'AirQualityStationCode', 
            'date.utc': 'DatetimeBegin', 
            'value': 'Concentration', 
            'coordinates.latitude': 'Latitude', 
            'coordinates.longitude': 'Longitude'}, inplace=True)
        # # Add column corresponding to English city name
        # merged['City'] = row['English Name']
        # Remove time from date
        merged['DatetimeBegin'] = pd.to_datetime(merged['DatetimeBegin'], 
            errors='coerce').dt.date
        # Plotting
        plt.plot(merged.groupby(by='DatetimeBegin').mean().index, 
            merged.groupby(by='DatetimeBegin').mean()['Concentration'])
        plt.xlim(['2019-01-01','2020-06-30'])
        plt.title(row.name)
        plt.show()
        # Round latitude and longitude (otherwise the unique latitude/
        # longitude coordinates aren't correct)
        merged['Longitude'] = merged['Longitude'].apply(lambda x: round(x, 4))        
        merged['Latitude'] = merged['Latitude'].apply(lambda x: round(x, 4))
        # Add city name column 
        merged['City'] = row['English Name']
        # Save off latitude and longitude coordinates for sampling the
        # GEOSCF model for machine learning (as with EEA))   
        uniquecoords = merged.groupby(['Longitude','Latitude']).size(
            ).reset_index()  
        uniquecoords.rename(columns={0: 'Count'}, inplace=True)
        uniquecoords['City'] = row['English Name']
        uniquecoords.to_csv(DIR_OUT+'%s_no2_2018-2020_coords.csv'%(
            row.name))
        # Save parsed Dataframe
        merged.to_csv(DIR_OUT+'%s_no2_2018-2020_timeseries.csv'%(row.name))

