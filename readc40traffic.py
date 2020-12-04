#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:08:51 2020

@author: ghkerr
"""

DIR = '/Users/ghkerr/GW/'
DIR_TRAFFIC = DIR+'data/traffic/'

def read_berlin(startdate, enddate):
    """Calculate time series of traffic counts and fraction of heavy-duty 
    vehicles for traffic counting sites in Berlin. Note that traffic counts are 
    taken at five air pollution monitoring sites (MC117, MC124, MC174, MC220, 
    MC143). The naming/coordinates of these traffic count stations are the same 
    as the air quality monitoring site coordinates. 

    In the input dataset, there are absolute values for long vehicles (> 7.5 
    meters, “Lkw”) and short vehicles (< 7.5 meters, “Pkw”), subdivided per 
    direction (N-North, E – East, W – West, S – South) and per hour. 
    There are also mean hourly speed values per direction (v). Traffic data 
    might have some missing data due to instrument failures, road construction, 
    etc. These hourly data are aggregated to daily sums and by site (i.e., 
    east and west are combined to a single sum, north and south as well).

    Parameters
    ----------
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Site- and hour-summed traffic data in Berlin
    """
    import pandas as pd
    import numpy as np
    # Read pollutant-specific file
    df = pd.read_csv(DIR_TRAFFIC+'berlin/'+'traffic_counts_2019_2020_header.csv',
        delimiter=';', header=0, engine='python')
    # Change Datum column to datetime
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d.%m.%Y %H:%M')
    df.rename(columns={'Datum':'Date'}, inplace=True)
    df.set_index('Date', drop=True, inplace=True)
    # Change column types from object to float
    for column in df.columns:
        df[column] = df[column].apply(pd.to_numeric, args=('coerce',))
    # Resample from houryl to daily counts
    df = df.resample('D').sum()
    # Sum traffic counts going different directions at same sites (i.e., E and W
    # or N and S)
    df['MC117'] = (df['MC117 Dir.E Lkw']+df['MC117 Dir.W Lkw']+
        df['MC117 Dir.E Pkw']+df['MC117 Dir.W Pkw'])
    # Calculate fraction of HDV
    df['MC117_FracHDV'] = (df['MC117 Dir.E Lkw']+df['MC117 Dir.W Lkw']
        )/df['MC117']
    df['MC124'] = (df['MC124 Dir.N Lkw']+df['MC124 Dir.S Lkw']+
        df['MC124 Dir.N Pkw']+df['MC124 Dir.S Pkw'])
    df['MC124_FracHDV'] = (df['MC124 Dir.N Lkw']+df['MC124 Dir.S Lkw']
        )/df['MC124']
    df['MC174'] = (df['MC174 Dir.E Lkw']+df['MC174 Dir.W Lkw']+
        df['MC174 Dir.E Pkw']+df['MC174 Dir.W Pkw'])
    df['MC174_FracHDV'] = (df['MC174 Dir.E Lkw']+df['MC174 Dir.W Lkw']
        )/df['MC174']
    df['MC220'] = (df['MC220 Dir.N Lkw']+df['MC220 Dir.S Lkw']+
        df['MC220 Dir.N Pkw']+df['MC220 Dir.S Pkw'])
    df['MC220_FracHDV'] = (df['MC220 Dir.N Lkw']+df['MC220 Dir.S Lkw']
        )/df['MC220']
    df['MC143'] = (df['MC143 Dir.E Lkw']+df['MC143 Dir.W Lkw']+
        df['MC143 Dir.E Pkw']+df['MC143 Dir.W Pkw'])
    df['MC143_FracHDV'] = (df['MC143 Dir.E Lkw']+df['MC143 Dir.W Lkw']
        )/df['MC143']
    # Drop all summed mean hourly speed values per direction and directional 
    # columns
    df = df.drop(['MC117 Dir.E Lkw', 'MC117 Dir.W Lkw', 'MC117 Dir.E Pkw',
        'MC117 Dir.W Pkw', 'MC117 Dir.E v', 'MC117 Dir.W v', 'MC124 Dir.N Lkw',
        'MC124 Dir.S Lkw', 'MC124 Dir.N Pkw', 'MC124 Dir.S Pkw',
        'MC124 Dir.N v', 'MC124 Dir.S v', 'MC174 Dir.E Lkw', 'MC174 Dir.W Lkw',
        'MC174 Dir.E Pkw', 'MC174 Dir.W Pkw', 'MC174 Dir.E v', 'MC174 Dir.W v',
        'MC220 Dir.N Lkw', 'MC220 Dir.S Lkw', 'MC220 Dir.N Pkw',
        'MC220 Dir.S Pkw', 'MC220 Dir.N v', 'MC220 Dir.S v', 'MC143 Dir.E Lkw',
        'MC143 Dir.W Lkw', 'MC143 Dir.E Pkw', 'MC143 Dir.W Pkw',
        'MC143 Dir.E v', 'MC143 Dir.W v'], axis=1)
    # Melt such that different sites are represented in rows, not columns
    df = pd.melt(df, var_name='Site', value_name='Count', ignore_index=False)
    # 2020-07-01 has an error, so drop this!
    df = df.drop(pd.date_range('2020-07-01','2020-07-03'), errors='ignore')
    # Create column for fraction of heavy-duty vehicles
    df['Frac_HDV'] = np.nan
    # Find fraction of heavy-duty vehicles at each site/date
    for index, row in df.iterrows():
        if 'FracHDV' in row['Site']:
            frac = row['Count']
            df.loc[(df['Site']==row['Site'][:5]) & (df.index==index), 
                'Frac_HDV'] = frac
    # Drop dummy columns
    df = df[~df.Site.str.contains('FracHDV')]
    df = df.loc[startdate:enddate]
    return df

def read_auckland(startdate, enddate):
    """fuction reads the (messy) traffic data from Auckland for 2019 and 
    2020. The extracted metric in the output table is the seven day ADT. Site
    identifiers are easting and northings. "Frac_HDV" represents the fraction 
    of heavy-duty vehicles in the traffic.

    Parameters
    ----------
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Site- and hour-summed traffic data in Berlin    
    """
    import numpy as np
    import pandas as pd
    # For 2019. Note that given the "count_duration" column (=7D), it appears 
    # that the ADT column is equivalent to the "7Days ADT" in the 2020 COVID
    # data
    df_2019 = pd.read_excel(DIR_TRAFFIC+'auckland/'+
        'RAMM_Report_Summary_Final.xlsm', sheet_name='BECA Listing', header=1, 
        usecols='E,F,H,M,O,P,Q,R,S,T')
    df_2019 = df_2019.rename(columns={
        'count_date':'Count Start Date', 'ADT':'7Days ADT', 'PcCar':'Car', 
        'PcLCV':'LCV', 'PcMCV':'MCV', 'PcHCVI':'HCV-I', 'PcHCVII':'HCV-II', 
        'PcHeavy':'Heavy'})
    # Combine easting/northing column into a single column to match COVID-19
    # data
    df_2019['northing'] = 'N'+df_2019['northing'].astype(str)
    df_2019['easting'] = 'E'+df_2019['easting'].astype(str)
    df_2019['Site'] = df_2019['easting']+' '+df_2019['northing']
    # For 2020. Note that I'm only using columns AI-BG...these columns seem
    # to be in a better format (counts summed over multi-direction highways). 
    # I am considering "Road ID" to be the standard identifier for different 
    # traffic counters
    df_2020 = pd.read_excel(DIR_TRAFFIC+'auckland/'+'COVID 19 Traffic Count '+
        'Data.xlsx', sheet_name='Sheet1', header=2, 
        usecols='F,AP,AR,BB,BC,BD,BE,BF,BG')
    df_2020 = df_2020.rename(columns={'Location':'Site',
        'Unnamed: 41':'Count Start Date',
        'Unnamed: 43':'7Days ADT', 'Unnamed: 53':'Car', 'Unnamed: 55':'MCV', 
        'Unnamed: 56':'HCV-I', 'Unnamed: 57':'HCV-II',
        'Unnamed: 58':'HCV Total'})
    df_2020['Site'] = df_2020['Site'].astype(str)
    # Now for the heavy-duty vehicle contribution. For 2019, it appears that 
    # MCV + HCV-I + HCV-II = Heavy (this isn't entirely true, but the mean 
    # of this operature is 0.0008 and max = 0.01 and min = -0.01, so whatever)
    # Replace these columns with a single 'Frac_HDV' column
    df_2019['Frac_HDV'] = df_2019['Heavy']
    df_2019 = df_2019.drop(['northing', 'easting', 'LCV','MCV','HCV-I',
        'HCV-II','Car','Heavy'], axis=1)
    # For 2020, it appears that Car + LCV + HCV Total = 1. (and MCV + HCV-I + 
    # HCVII = Heavy)
    df_2020['Frac_HDV'] = df_2020['HCV Total']
    df_2020 = df_2020.drop(['Car', 'LCV', 'MCV', 'HCV-I', 'HCV-II',
        'HCV Total'], axis=1)
    # Merge datasets
    df = pd.concat([df_2019, df_2020])
    # Rename to standarized names
    df = df.rename(columns={'Count Start Date':'Date', '7Days ADT':'Count'})
    # Set time as index
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]
    return df

def read_milan(startdate, enddate):
    """Traffic data in Milan is a bit strange. Area C (which is the congestion
    charge of the historic center of the city; ~9 km^2) is the only part of the
    city that has both vehicle counts (total number of vehicles entering
    the area and LDV/HDV fraction). Thus, we only consider Area C, even though 
    it is a much smaller geographic area than the area considered for the 
    change in NO2 or the air quality observations. Note that the output values
    are not site-specific but rather represent the total number of vehicles 
    entering Area C summed over the hours of the day. 

    Parameters
    ----------
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Traffic counts for Milan (total number of vehicles entering Area C; 
        see input .xlsx file for a map of Area C
    """
    import pandas as pd
    df = pd.read_excel(DIR_TRAFFIC+'milan/'+
        'Traffic data Milan_AMAT_01_01_2019_31_10_2020.xlsx', 
        sheet_name='AreaC - % LDV_HDV', header=0)
    df = df.rename(columns={'date':'Date', 
        'Area C entering vehicles':'Count', '%HDV':'Frac_HDV'})
    df = df.drop(['%LDV', 'LDV', 'HDV'], axis=1)
    # Since these values only represent one site, make a dummy column to 
    # represent this one site (to be consistent with standardized format)
    df['Site'] = 'AreaC'    
    # Set time as index
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]    
    return df

def read_santiago(startdate, enddate):
    """Read traffic counts from Santiago. Counts are provided for a variety
    of sites for different times of the day: punta manana (6-7 am, 7-8 am, 
    8-9am, 9-10am) and punta tarde (5-6pm, 6-7pm, 7-8pm, 8-9pm). For a given 
    site and date, these observations are summed to produce a daily, site-
    specific sum. Note that there are strange notes in the Excel file for 
    specific counts that I ignored (all in Spanish).

    Parameters
    ----------
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format
        
    Returns
    -------
    df : pandas.core.frame.DataFrame
        Daily summed traffic counts for individual 

    """
    import numpy as np
    import pandas as pd
    # PM is for punta de la mañana (morning)
    df_pm = pd.read_excel(DIR_TRAFFIC+'santiago/'+
        'resumen de flujos vehiculares PM disponibles marzo a agosto 2020.xlsx', 
        sheet_name='PUNTA MAÑANA', header=3)
    # Drop information about Ubicación and Comuna - these appear to just be 
    # location information about the address and neighborhood. Instead we'll
    # use Estación for the location identifier
    df_pm = df_pm.drop(['Hora','Ubicación','Comuna'], axis=1)
    # Drop rows that are only NaN for all columns
    df_pm.dropna(inplace=True)
    # Replace missing information with NaN. As far as I can tell, there are 
    # a variety of ways that Santiago indicates missing information
    # Falla - instrument failure
    # falla - same as above, but with a lowercase f just to fuck with me
    # s/i - without information (sine información)
    # Reversible - no idea what this means in this context 
    # ERROR - no explination needed
    # desvío - detour
    df_pm.replace({'falla':np.nan, 'Falla':np.nan, 's/i':np.nan, 'Reversible':
        np.nan, 'ERROR':np.nan, 'desvío':np.nan, 'Falal':np.nan, 'reversible':
        np.nan, 'Bajo':np.nan, 'fall':np.nan, 'fala':np.nan, 'rev':np.nan, 
        'error':np.nan}, regex=True, inplace=True)
    # Set station as index and melt
    df_pm.set_index('Estación', drop=True, inplace=True)
    df_pm = df_pm.melt(ignore_index=False)
    # Some of the entries in the date column have the two letter day 
    # abbreviation in Spanish (e.g., lu for lunes). Remove these. 
    df_pm['variable'] = df_pm['variable'].astype(str)
    df_pm['variable'] = df_pm['variable'].replace(dict.fromkeys(['lu ','vi ',
        'ju ','mi ','ma ', 'Ma ', 'lu  ', 'jue ', 'ju  ', ' ju '], ''), 
        regex=True)
    df_pm.reset_index(inplace=True)
    # Dates are a mixture of Y-m-d and d-m-Y format (are you KIDDING me?!) so
    # address this
    date1 = pd.to_datetime(df_pm['variable'], errors='coerce', 
        format='%Y-%m-%d')
    date2 = pd.to_datetime(df_pm['variable'], errors='coerce', 
        format='%d-%m-%y')
    df_pm['Date'] = date1.fillna(date2)
    df_pm.rename(columns={'value':'Count','Estación':'Site'}, inplace=True)
    df_pm = df_pm.drop(['variable'], axis=1)
    df_pm['Count'] = pd.to_numeric(df_pm['Count'], errors='coerce')
    df_pm = df_pm.groupby(['Site','Date'], as_index=False)['Count'].sum()
    df_pm.set_index('Date', drop=True, inplace=True)
    # Same as above but for PT (punta de la tarde; i.e., afternoon)
    df_pt = pd.read_excel(DIR_TRAFFIC+'santiago/'+
        'resumen de flujos vehiculares PT disponibles marzo a agosto 2020.xlsx', 
        sheet_name='PUNTA TARDE', usecols='A:EA', header=3)
    # Split by different times of the day (e.g., Comparación 17:00 a 18:00 
    # horas)
    df_pt_a = df_pt.iloc[:84]
    df_pt_b = df_pt.iloc[91:175]
    df_pt_c = df_pt.iloc[182:266]
    df_pt_d = df_pt.iloc[273:357]
    df_pt = pd.concat([df_pt_a,df_pt_b,df_pt_c,df_pt_d], ignore_index=True, 
        sort=False)
    df_pt = df_pt.drop(['Espira','Ubicación','Comuna'], axis=1)
    df_pt.replace({'falla':np.nan, 'Falla':np.nan, 's/i':np.nan, 'Reversible':
        np.nan, 'ERROR':np.nan, 'desvío':np.nan, 'Falal':np.nan, 'reversible':
        np.nan, 'Bajo':np.nan, 'fall':np.nan, 'fala':np.nan, 'rev':np.nan, 
        'error':np.nan, 'no hay':np.nan, 'TROF':np.nan, 'NaTType':np.nan}, 
        regex=True, inplace=True)
    df_pt.set_index('Estación', drop=True, inplace=True)
    df_pt = df_pt.melt(ignore_index=False)
    df_pt.reset_index(inplace=True)
    df_pt['variable'] = pd.to_datetime(df_pt['variable'], format='%Y-%m-%d')
    df_pt.rename(columns={'value':'Count','Estación':'Site','variable':
        'Date'}, inplace=True)
    df_pt['Count'] = pd.to_numeric(df_pt['Count'], errors='coerce')
    df_pt = df_pt.groupby(['Site','Date'], as_index=False)['Count'].sum()
    df_pt.set_index('Date', drop=True, inplace=True)
    # Marry two datasets
    df = pd.concat([df_pm, df_pt], ignore_index=False)
    # Calculate a daily sum over morning (PM) and afternoon (PT) sites
    df = df.groupby([df.index.get_level_values(0),'Site']).sum()
    df.reset_index(inplace=True)
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]    
    return df