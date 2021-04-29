#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract daily averaged AQS NO2 concentrations from 40 largest core-based
statistical areas (CBSAs) by population in the U.S. Output files containing
(1) concentrations, date, coordinate, and city information and (2) coordinate
information are returned.

Created on Thu Apr 29 10:00:13 2021
"""
DIR_AQS = '/Users/ghkerr/GW/data/aq/aqs/'

import glob
import numpy as np
import pandas as pd
# Open all observations
def pd_read_pattern(pattern):
    files = glob.glob(pattern)
    print('Reading observations from %d files...'%len(files))
    df = pd.DataFrame()
    for f in files:
        try:
            df = df.append(pd.read_csv(f))
        except UnicodeDecodeError:
            pass
    return df.reset_index(drop=True)

aqs = pd_read_pattern(DIR_AQS+'daily_42602_*.csv')
# Select top 40 MSAs by population (can be found at 
# https://en.wikipedia.org/wiki/List_of_metropolitan_statistical_areas
cbsa = {'New York-Newark-Jersey City, NY-NJ-PA':'New York',
    'Los Angeles-Long Beach-Anaheim, CA':'Los Angeles',
    'Chicago-Naperville-Elgin, IL-IN-WI':'Chicago',
    'Dallas-Fort Worth-Arlington, TX':'Dallas',
    'Houston-The Woodlands-Sugar Land, TX':'Houston',
    'Washington-Arlington-Alexandria, DC-VA-MD-WV':'Washington, DC',
    'Miami-Fort Lauderdale-West Palm Beach, FL':'Miami',
    'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD':'Philadelphia',
    'Atlanta-Sandy Springs-Roswell, GA':'Atlanta',
    'Phoenix-Mesa-Scottsdale, AZ':'Phoenix',
    'Boston-Cambridge-Newton, MA-NH':'Boston',
    'San Francisco-Oakland-Hayward, CA':'San Francisco',
    'Riverside-San Bernardino-Ontario, CA':'Riverside',
    'Detroit-Warren-Dearborn, MI':'Detroit',
    'Seattle-Tacoma-Bellevue, WA':'Seattle',
    'Minneapolis-St. Paul-Bloomington, MN-WI':'Minneapolis',
    'San Diego-Carlsbad, CA':'San Diego',
    'Tampa-St. Petersburg-Clearwater, FL':'Tampa',
    'Denver-Aurora-Lakewood, CO':'Denver',
    'St. Louis, MO-IL':'St. Louis',
    'Baltimore-Columbia-Towson, MD':'Baltimore',
    'Charlotte-Concord-Gastonia, NC-SC':'Charlotte',
    'Orlando-Kissimmee-Sanford, FL':'Orlando',
    'San Antonio-New Braunfels, TX':'San Antonio',
    'Portland-Vancouver-Hillsboro, OR-WA':'Portland',
    'Sacramento--Roseville--Arden-Arcade, CA':'Sacramento',
    'Pittsburgh, PA':'Pittsburgh',
    'Las Vegas-Henderson-Paradise, NV':'Las Vegas',
    'Austin-Round Rock, TX':'Austin',
    'Cincinnati, OH-KY-IN':'Cincinnati',
    'Kansas City, MO-KS':'Kansas City',
    'Columbus, OH':'Columbus',
    'Indianapolis-Carmel-Anderson, IN':'Indianapolis',
    'Cleveland-Elyria, OH':'Cleveland',
    'San Jose-Sunnyvale-Santa Clara, CA':'San Jose',
    'Nashville-Davidson--Murfreesboro--Franklin, TN':'Nashville',
    'Virginia Beach-Norfolk-Newport News, VA-NC':'Virginia Beach',
    'Providence-Warwick, RI-MA':'Providence',
    'Milwaukee-Waukesha-West Allis, WI':'Milwaukee',
    'Jacksonville, FL':'Jacksonville'}
aqs = aqs.loc[aqs['CBSA Name'].isin(list(cbsa.keys()))]
# Add simple city name to DataFrame
aqs['City'] = np.nan
for cbsa_i in cbsa.keys():
    short = cbsa[cbsa_i]
    aqs.loc[aqs['CBSA Name']==cbsa_i,'City'] = short
# Form station code
aqs['AirQualityStationCode'] = np.nan
aqs['AirQualityStationCode'] = (aqs['State Code'].astype(str)+
    aqs['County Code'].astype(str)+aqs['Site Num'].astype(str))
# Drop unneeded columns
keep = ['Latitude', 'Longitude','Arithmetic Mean','AirQualityStationCode',
    'Date Local','City']
aqs = aqs[keep]
# Rename columns to match conventions 
aqs.rename(columns={'Arithmetic Mean':'Concentration',
    'Date Local':'DatetimeBegin'}, inplace=True)
# Save off the relevant latitude and longitude coordinates for sampling the
# GEOSCF model for machine learning        
uniquecoords = aqs.groupby(['Longitude','Latitude','City']).size().reset_index()  
uniquecoords = pd.DataFrame(uniquecoords)
uniquecoords.rename(columns={0: 'Count'}, inplace=True)
uniquecoords.to_csv(DIR_AQS+'US_no2_2018-2020_coords.csv')
# Save parsed Dataframe
aqs.to_csv(DIR_AQS+'US_no2_2018-2020_timeseries.csv')