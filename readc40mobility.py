#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:08:51 2020

@author: ghkerr
"""
DIR = '/Users/ghkerr/GW/'
DIR_MOBILITY = DIR+'data/mobility/'

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
    df = pd.read_csv(DIR_MOBILITY+'berlin/'+
        'traffic_counts_2019_2020_header.csv', delimiter=';', header=0, 
        engine='python')
    # Change Datum column to datetime
    df['Datum'] =  pd.to_datetime(df['Datum'], format='%d.%m.%Y %H:%M')
    df.rename(columns={'Datum':'Date'}, inplace=True)
    df.set_index('Date', drop=True, inplace=True)
    # Change column types from object to float
    for column in df.columns:
        df[column] = df[column].apply(pd.to_numeric, args=('coerce',))
    # Resample from hourly to daily counts
    df = df.resample('D').sum()
    # Sum traffic counts going different directions at same sites (i.e., E and W
    # or N and S), number of passenger vehicles/freight, and the 
    # fraction of heavy-duty vehicles
    df['MC117'] = (df['MC117 Dir.E Lkw']+df['MC117 Dir.W Lkw']+
        df['MC117 Dir.E Pkw']+df['MC117 Dir.W Pkw'])
    df['MC117_passenger'] = (df['MC117 Dir.E Pkw']+df['MC117 Dir.W Pkw'])    
    df['MC117_freight'] = (df['MC117 Dir.E Lkw']+df['MC117 Dir.W Lkw'])    
    df['MC117_FracHDV'] = (df['MC117 Dir.E Lkw']+df['MC117 Dir.W Lkw']
        )/df['MC117']
    df['MC124'] = (df['MC124 Dir.N Lkw']+df['MC124 Dir.S Lkw']+
        df['MC124 Dir.N Pkw']+df['MC124 Dir.S Pkw'])
    df['MC124_passenger'] = (df['MC124 Dir.N Pkw']+df['MC124 Dir.S Pkw'])    
    df['MC124_freight'] = (df['MC124 Dir.N Lkw']+df['MC124 Dir.S Lkw'])        
    df['MC124_FracHDV'] = (df['MC124 Dir.N Lkw']+df['MC124 Dir.S Lkw']
        )/df['MC124']
    df['MC174'] = (df['MC174 Dir.E Lkw']+df['MC174 Dir.W Lkw']+
        df['MC174 Dir.E Pkw']+df['MC174 Dir.W Pkw'])
    df['MC174_passenger'] = (df['MC174 Dir.E Pkw']+df['MC174 Dir.W Pkw'])    
    df['MC174_freight'] = (df['MC174 Dir.E Lkw']+df['MC174 Dir.W Lkw'])        
    df['MC174_FracHDV'] = (df['MC174 Dir.E Lkw']+df['MC174 Dir.W Lkw']
        )/df['MC174']
    df['MC220'] = (df['MC220 Dir.N Lkw']+df['MC220 Dir.S Lkw']+
        df['MC220 Dir.N Pkw']+df['MC220 Dir.S Pkw'])
    df['MC220_passenger'] = (df['MC220 Dir.N Pkw']+df['MC220 Dir.S Pkw'])    
    df['MC220_freight'] = (df['MC220 Dir.N Lkw']+df['MC220 Dir.S Lkw'])        
    df['MC220_FracHDV'] = (df['MC220 Dir.N Lkw']+df['MC220 Dir.S Lkw']
        )/df['MC220']
    df['MC143'] = (df['MC143 Dir.E Lkw']+df['MC143 Dir.W Lkw']+
        df['MC143 Dir.E Pkw']+df['MC143 Dir.W Pkw'])
    df['MC143_passenger'] = (df['MC143 Dir.E Pkw']+df['MC143 Dir.W Pkw'])    
    df['MC143_freight'] = (df['MC143 Dir.E Lkw']+df['MC143 Dir.W Lkw'])        
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
    df['Passenger'] = np.nan
    df['Freight'] = np.nan    
    # Kludgy
    for index, row in df.iterrows():
        if 'FracHDV' in row['Site']:
            frac = row['Count']
            df.loc[(df['Site']==row['Site'][:5]) & (df.index==index), 
                'Frac_HDV'] = frac
    for index, row in df.iterrows():
        if 'passenger' in row['Site']:
            frac = row['Count']
            df.loc[(df['Site']==row['Site'][:5]) & (df.index==index), 
                'Passenger'] = frac       
    for index, row in df.iterrows():
        if 'freight' in row['Site']:
            frac = row['Count']
            df.loc[(df['Site']==row['Site'][:5]) & (df.index==index), 
                'Freight'] = frac                   
    # Drop dummy columns
    df = df[~df.Site.str.contains('FracHDV')]
    df = df[~df.Site.str.contains('passenger')]
    df = df[~df.Site.str.contains('freight')]
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
    df_2019 = pd.read_excel(DIR_MOBILITY+'auckland/'+
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
    df_2020 = pd.read_excel(DIR_MOBILITY+'auckland/'+'COVID 19 Traffic Count '+
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
    df = pd.read_excel(DIR_MOBILITY+'milan/'+
        'Traffic data Milan_AMAT_01_01_2019_31_10_2020.xlsx', 
        sheet_name='AreaC - % LDV_HDV', header=0)
    df = df.rename(columns={'date':'Date', 
        'Area C entering vehicles':'Count', '%HDV':'Frac_HDV', 
        'HDV':'Freight', 'LDV':'Passenger'})
    df = df.drop(['%LDV'], axis=1)
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
        Daily summed traffic counts for individual monitoring sites in 
        Santiago
    """
    import numpy as np
    import pandas as pd
    # PM is for punta de la mañana (morning)
    df_pm = pd.read_excel(DIR_MOBILITY+'santiago/'+
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
    df_pt = pd.read_excel(DIR_MOBILITY+'santiago/'+
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

def read_london(startdate, enddate):
    """Traffic data from London is provided as daily vehicle kilometers 
    traveled on the TLRN in March-October 2019 and 2020. Note that a percent
    change is also given for "central," "inner," and "outer" but no baseline
    values are included, so these data weren't extracted in the DataFrame. No 
    information on the paritioning of light- versus heavy-duty is given. 
    
    n.b., a map of the TLRN can be found here: https://www.cleanhighways.co.uk/
    legislation/london-roads-2

    Parameters
    ----------
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format
        
    Returns
    -------
    df : pandas.core.frame.DataFrame
        Vehicle km traveled on TLRN for March-October 2019 and 2020. 
    """
    import pandas as pd
    # Extract daily values of vehicle km travelled on the TLRN in 2020
    df_2020 = pd.read_excel(DIR_MOBILITY+'london/'+
        'COVID 19 Flow Summary Table for Andy.xlsx', sheet_name='Sheet1', 
        header=1, usecols='C,H')
    # Baseline day 2019 vehicle km travelled on the TLRN 
    df_2019 = pd.read_excel(DIR_MOBILITY+'london/'+
        'COVID 19 Flow Summary Table for Andy.xlsx', sheet_name='Sheet1', 
        header=1, usecols='C,L')
    # Drop NaN columns at the end 
    df_2019.dropna(inplace=True)
    df_2020.dropna(inplace=True)
    # Rename
    df_2019.rename(columns={'Unnamed: 2':'Date', 'TRLN':'Vehicle km'}, 
        inplace=True)
    df_2020.rename(columns={'Unnamed: 2':'Date', 'TLRN.1':'Vehicle km'}, 
        inplace=True)
    # Since the column with information about km driven uses the same 
    # dates as 2020 (MM-DD are the same, just different year), change the
    # year to 2019
    df_2019['Date'] = df_2019['Date'] - pd.DateOffset(years=1)
    # Stack DataFrames into one DataFrame
    df = pd.concat([df_2019, df_2020])
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]    
    return df

def read_mexicocity_baseline(startdate,enddate):
    """For Mexico City baseline traffic (2016-2019), there are six different 
    input files labeled C1-C6. The png image in the same folder provides a 
    legend for which file corresponds to which vehicle type with C1 
    correpsonding to the smallest type. Thus, 
    C1 = Autos 2.1-5 meters
    C2 = Microbuses and vans 5-9 meters
    C3 = Buses 9-14 meters
    C4 = Unit truck (2-6 axles) 14-18 meters
    C5 = articulated truck (5-9 axles) 18-23 meters
    C6 = biarticulated trucks (5-9 axles) > 23 meters
    
    The fraction of heavy-duty vehicles is calculated as the sum of counts from 
    C4-C6 over the total number. This function isn't the fastest, and could 
    perhaps be sped up by only reading in C4-C6 and then reading 
    'BASE_TOTAL_AFORO_2016-2019.csv' as the total number of vehicles (n.b., 
    I checked that C1+C2+C3+C4+C5+C6=TOTAL AFORO, and this is indeed the case).

    Parameters
    ----------
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format
        
    Returns
    -------
    df : pandas.core.frame.DataFrame
        Traffic counts in Mexico City for the pre-COVID period (2016-2019)
    """
    import pandas as pd
    # For light-duty vehicles (types C1-C3)
    df_light = pd.DataFrame([])
    for vtype in ['C1','C2','C3']:
        dfty = pd.read_csv(DIR_MOBILITY+'mexicocity/SEDEMA_MOVILIDAD/INFOVIAL/'+
            'BASE_%s_2016-2019.csv'%vtype, delimiter=',', header=3, 
            engine='python')
        # Drop row 
        dfty.drop(dfty.index[[0]], inplace=True)
        # Set first column as datetime 
        dfty['Unnamed: 1'] = pd.to_datetime(dfty['Unnamed: 1'])
        dfty = dfty.rename(columns={'Unnamed: 1':'Date'})
        dfty = dfty.drop(['3','NOMBRE'], axis=1)
        cols=[i for i in dfty.columns if i not in ['Date']]
        for col in cols:
            dfty[col]=pd.to_numeric(dfty[col])
        # Calculate daily sum for each site
        dfty = dfty.resample('D', on='Date').sum()
        dfty = pd.melt(dfty, var_name='Site', value_name='Count', 
            ignore_index=False)
        df_light = df_light.append(dfty, ignore_index=False)            
    # Group by index and 'Site' column 
    df_light = df_light.groupby([df_light.index,'Site'])['Count'].sum()
    df_light = df_light.reset_index()
    # Same as above but for heavy-duty vehicles (types C4-C6)
    df_heavy = pd.DataFrame([])
    for vtype in ['C4','C5','C6']:
        dfty = pd.read_csv(DIR_MOBILITY+'mexicocity/SEDEMA_MOVILIDAD/INFOVIAL/'+
            'BASE_%s_2016-2019.csv'%vtype, delimiter=',', header=3, 
            engine='python')
        dfty.drop(dfty.index[[0]], inplace=True)
        dfty['Unnamed: 1'] = pd.to_datetime(dfty['Unnamed: 1'])
        dfty = dfty.rename(columns={'Unnamed: 1':'Date'})
        dfty = dfty.drop(['3','NOMBRE'], axis=1)
        cols=[i for i in dfty.columns if i not in ['Date']]
        for col in cols:
            dfty[col]=pd.to_numeric(dfty[col])
        dfty = dfty.resample('D', on='Date').sum()
        dfty = pd.melt(dfty, var_name='Site', value_name='Count', 
            ignore_index=False)
        df_heavy = df_heavy.append(dfty, ignore_index=False)       
    # Group by index and 'Site' column 
    df_heavy = df_heavy.groupby([df_heavy.index,'Site'])['Count'].sum()
    df_heavy = df_heavy.reset_index()
    # Merge DataFrames
    df = pd.merge(df_light, df_heavy,  how='left', left_on=['Date','Site'], 
        right_on=['Date','Site'])
    # Calculate a total count category
    df['Count'] = df['Count_x']+df['Count_y']
    df['Frac_HDV'] = df['Count_y']/df['Count']
    # Drop unneeded columns
    df = df.drop(['Count_x','Count_y'], axis=1)
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]    
    return df

def read_mexicocity_lockdown(startdate, enddate):
    """Mexico City provides detailed traffic count information for six 
    different vehicle types prior to the lockdowns (see function 
    "read_mexicocity_baseline"), but for the lockdowns only Waze traffic 
    jam data for each municipality (hourly) is provided in the file
    "Conteos_lineas_muni_2019_2020" for pre-COVID and COVID months. Possible
    date ranges span 2019-05-01 to 2020-10-18.
    
    Parameters
    ----------
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format
        
    Returns
    -------
    df : pandas.core.frame.DataFrame
        Traffic counts in Mexico City for the COVID period
    """
    import pandas as pd
    df = pd.read_csv(DIR_MOBILITY+'mexicocity/'+
        'Conteo_Lineas_Muni_2019_2020.csv', delimiter=',', engine='python')
    # Combine date and hour columns
    df['Date'] = pd.to_datetime(df['date'])
    # Daily sum at each site
    df = df.groupby(['Date','city'])['Total'].sum()
    df = df.reset_index()
    df.set_index('Date', drop=True, inplace=True)
    df = df.rename(columns={'city':'Site', 'Total':'Count'})
    df = df.loc[startdate:enddate]    
    return df

def read_applemobility(start, end):
    """Read Apple mobility data for select cities (major European cities with 
    EEA monitors and C40 cities) for the time period of interest. If the 
    specified start date is before the beginning of the Apple mobility dataset
    (2020-01-13), and artificial traffic volume dataset is created by 
    averaging over pre-lockdown weekdays and constructing a weekly timeseries 
    with these data.     
    The CSV file and charts on this site show a relative volume of directions 
    requests per country/region, sub-region or city compared to a baseline 
    volume on January 13th, 2020. We define our day as midnight-to-midnight, 
    Pacific time. Cities are defined as the greater metropolitan area and their 
    geographic boundaries remain constant across the data set. In many 
    countries/regions, sub-regions, and cities, relative volume has increased 
    since January 13th, consistent with normal, seasonal usage of Apple Maps. 
    Day of week effects are important to normalize as you use this data.
    Data were downloaded on 13 Dec 2020 (final entry is 12 Dec 2020) 
    from https://covid19.apple.com/mobility. Note that data for May 11-12 are 
    not available and will appear as blank columns in the data set.    

    Parameters
    ----------
    start : str
        Start date of period of interest; YYYY-mm-dd format    
    end : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    mobility : pandas.core.frame.DataFrame
        Timeseries of relative traffic volume (with 13 Jan 2020 as the baseline
        volume) for cities for the specified time period. 
    """
    import numpy as np
    import pandas as pd
    df = pd.read_csv(DIR_MOBILITY+'applemobilitytrends-2020-12-12.csv', 
        delimiter=',', engine='python')
    station_city = ['Amsterdam', 'Athens', 'Auckland', 'Bangkok', 'Barcelona',
       'Belgrade', 'Berlin', 'Bratislava', 'Brussels', 'Bucharest',
       'Budapest', 'Cologne', 'Copenhagen', 'Dublin', 'Dusseldorf',
       'Frankfurt', 'Hamburg', 'Helsinki', 'Krakow', 'Lodz', 'London', 
       'Los Angeles', 'Luxembourg', 'Madrid', 'Marseille', 'Medellin',
       'Mexico City', 'Milan', 'Montreal', 'Munich', 'Naples',
       # Removed 'Mumbai' on 22 April
       'Oslo', 'Palermo', 'Paris', 'Prague', 'Riga', 'Rome', 'Rotterdam',
       'Santiago', 'Saragossa', 'Seville', 'Sofia', 'Stockholm',
       'Stuttgart', 'Sydney', 'Taipei City', 'Turin', 'Valencia',
       'Vienna', 'Vilnius', 'Warsaw', 'Wroclaw', 'Zagreb', 'Zurich']  
    mobility = []
    # Select region of interest and the driving timeseries; note that the if/
    # elif statements are for cities that do not have their own timeseries in 
    # the Apple mobility dataset or are classified, for example, as a sub-
    # region rather than a city   
    for cityname in station_city:
        if cityname=='Sofia':
            city = df.loc[(df['geo_type']=='country/region') & (df['region']==
                'Bulgaria') & (df['transportation_type']=='driving')]
        elif cityname=='Turin':
            city = df.loc[(df['geo_type']=='city') & (df['region']=='Torino') &
                (df['transportation_type']=='driving')]        
        elif cityname=='Zagreb':
            city = df.loc[(df['geo_type']=='country/region') & (df['region']==    
                'Croatia') & (df['transportation_type']=='driving')]
        elif cityname=='Lodz':
            city = df.loc[(df['geo_type']=='sub-region') & (df['region']==    
                'Lodz Province') & (df['transportation_type']=='driving')]
        elif cityname=='Saragossa':
            city = df.loc[(df['geo_type']=='sub-region') & (df['region']==
                'Aragon') & (df['transportation_type']=='driving')]
        elif cityname=='Wroclaw':
            city = df.loc[(df['geo_type']=='sub-region') & (df['region']==
                'Lower Silesia Province') & (df['transportation_type']==
                'driving')]
        elif cityname=='Riga':
            city = df.loc[(df['geo_type']=='country/region') & (df['region'
                ]=='Latvia') & (df['transportation_type']=='driving')]
        elif cityname=='Vilnius':
            city = df.loc[(df['geo_type']=='country/region') & (df['region'
            ]=='Lithuania') & (df['transportation_type']=='driving')]
        elif cityname=='Belgrade':
            city = df.loc[(df['geo_type']=='country/region') & (df['region'
            ]=='Serbia') & (df['transportation_type']=='driving')]            
        elif cityname=='Bratislava':
            city = df.loc[(df['geo_type']=='sub-region') & (df['region'
            ]=='Bratislava Region') & (df['transportation_type']=='driving')]
        elif cityname=='Taipei City':
            city = df.loc[(df['geo_type']=='sub-region') & (df['region'
            ]=='Taipei City') & (df['transportation_type']=='driving')]
        elif cityname=='Medellin':
            city = df.loc[(df['geo_type']=='country/region') & (df['region'
            ]=='Colombia') & (df['transportation_type']=='driving')]        
        elif cityname=='Luxembourg':
            city = df.loc[(df['geo_type']=='sub-region') & (df['region'
            ]=='Luxembourg District') & (df['transportation_type']=='driving')]        
        elif cityname=='Dublin':
            city = df.loc[(df['geo_type']=='city') & (df['region'
            ]=='Dublin') & (df['transportation_type']=='driving') &
            (df['country']=='Ireland')]                            
        elif cityname=='Los Angeles':
            # Los Angeles metropolitan area is comprised of Los Angeles and 
            # Orange counties
            city = df.loc[(df['geo_type']=='county') & (df['region'].isin([
                'Los Angeles County','Orange County'])) & 
                (df['sub-region']=='California') &
                (df['transportation_type']=='driving')]
            city = city.groupby('geo_type').mean()
        elif cityname=='Naples':
            city = df.loc[(df['geo_type']=='city') & (df['region']=='Naples') &
                (df['country']=='Italy') & (df['transportation_type'
                ]=='driving')]
        else:     
            city = df.loc[(df['geo_type']=='city') & (df['region']==cityname) &
                (df['transportation_type']=='driving')]
        todrop = ['geo_type', 'region', 'transportation_type', 'alternative_name',
            'sub-region', 'country']
        city = city.drop(todrop, axis=1, errors='ignore')
        city = pd.melt(city, var_name='Date', value_name='Volume')
        # If there are > 1 values per day (e.g., Los Angeles and Orange 
        # Counties for Los Angeles), compute a daily average
        if city.shape[0] > np.unique(city['Date'].values).shape[0]:
            city.groupby(by='Date').mean().reset_index()
        city['Date'] = pd.to_datetime(city['Date'])
        city.set_index('Date', drop=True, inplace=True)
        if (pd.to_datetime(start) < pd.to_datetime(city.index[0].strftime('%Y-%m-%d')))==True: 
            # Select quasi-pre-lockdown information
            pre = city.loc['2020-01-13':'2020-02-29'].copy()
            # Find days of week corresponding to days
            pre['day_of_week'] = pre.index.day_name()
            # Group by day of week and average
            weeklycycle = pre.groupby(by='day_of_week').mean()
            # Create dummy DataFrame extending from the specified start date 
            # to 2020-01-13
            dummy = pd.DataFrame([])
            dummy['Date'] = pd.date_range(start, '2020-01-12')
            dummy['day_of_week'] = dummy['Date'].dt.day_name()
            # Map weekly cycle from 2020 to dummy DateFrame
            dummy['Volume'] = np.nan
            dummy['Volume'] = dummy['day_of_week'].map(weeklycycle['Volume'])
            # Merge DataFrames 
            dummy = dummy.drop(['day_of_week'], axis=1)
            dummy.set_index('Date', drop=True, inplace=True)
            city = pd.concat([dummy, city])
        # Select date range
        city = city.loc[start:end]
        city['city'] = cityname
        mobility.append(city)
    mobility = pd.concat(mobility)
    # Change city names to be consistent with conventions in code    
    mobility.loc[mobility.city=='Los Angeles','city'] = 'Los Angeles C40'
    mobility.loc[mobility.city=='Mexico City','city'] = 'Mexico City C40'
    mobility.loc[mobility.city=='Santiago','city'] = 'Santiago C40'
    mobility.loc[mobility.city=='Auckland','city'] = 'Auckland C40'
    mobility.loc[mobility.city=='London','city'] = 'London C40'
    mobility.loc[mobility.city=='Berlin','city'] = 'Berlin C40'
    mobility.loc[mobility.city=='Milan','city'] = 'Milan C40'
    mobility.loc[mobility.city=='Tapei City','city'] = 'Tapei'
    return mobility

def read_losangeles(): 
    """FUNCTION NEEDS TO BE DEBUGGED
    
    Returns
    -------
    None.
    """
    import requests
    import numpy as np
    import pandas as pd
    import glob
    # # # # Open siting information (files with prefix 
    # all_text_tmg_locations)
    sitecolnames = ['Location ID', # Unique identifier generated by 
        # rc_location_seq	 
     	'Segment ID', # Pointer to route_carriageway_segments	 
     	'State Postmile', # State Postmile	 
     	'Absolute Postmile', # Absolute Postmile	 
     	'Latitude',	# Latitude	 
     	'Longitude', # Longitude	 
     	'Angle', # Used to project icons next to the freeway	 
     	'Name',	# Free form text description of the location.	 
     	'Abbrev', # Brief text description of the location.	 
     	'Freeway ID', # Freeway ID	 
     	'Freeway Direction', # A string indicating the freeway direction.	 
     	'District ID', # A string indicating the district.	 
     	'County ID', # A string indicating the county.	 
     	'City ID'] # A string indicating the city.	
    # From a **painstaking** comparison of the truck census file and the
    # location file, the two need to be matched on:
    # Absolute Postmile
    # Freeway Identifier
    # Freeway Direction
    # County Identifier
    # District Identifier  
    fname = '/Users/ghkerr/Downloads/all_text_tmg_station_configs_2019_01_01/'+\
        'all_text_tmg_locations_2019_01_01.txt'
    sites = pd.read_csv(fname, names=sitecolnames, header=None)
    
    # Field Specification (with typos fixed and weird repeated columns 
    # deleted!) from pems.dot.ca.gov for Census Trucks Day. Data from 
    # pems.dot.ca.gov -> Data Clearinghouse -> Census Trucks Day -> 
    colnames = ['Timestamp', # The date and time of the beginning 
        # of the summary interval. For example, a date of 1/1/2012 indicates 
        # that the aggregate(s) contain measurements collected between 1/1/2012
        # 00:00:00 and 1/1/2012 23:59:59. Note that hour, minute, and second 
        # values are always 0 for daily aggregations. The format is MM/DD/YYYY
        # HH24:MI:SS.	 
        'Census Station Identifier', # The unique number that identifies this 
        # census station within PeMS.	 
        'Census Substation Identifier',	# The unique number that identifies this 
        # census substation within PeMS.	 
        'Freeway Identifier', # The unique number that identifies this 
        # freeway within PeMS.	 
        'Freeway Direction', # A string indicating the freeway direction.	 
        'City Identifier',	# The unique number that identifies the city 
        # that contains this census station within PeMS.	 
        'County Identifier', # The unique number that identifies the county 
        # that contains this census station within PeMS.	 
        'District Identifier',	# The unique number that identifies the Caltrans 
        # distrcit that contains this census station within PeMS.	 
        'Absolute Postmile', # The postmile where this census station is 
        # located.	 
        'Station Type',	# A string indicating the type of station. Possible 
        # values (and their meaning are:
        # CD (Coll/Dist)
        # CH (Conventional Highway)
        # FF (Fwy-Fwy connector)
        # FR (Off Ramp)
        # HV (HOV)
        # ML (Mainline)
        # OR (On Ramp)
        'Census Station Set ID', # The unique number that identifies the 
        # census station set within PeMS.	 
        'Lane Number', # The lane number	 
        'Vehicle Class', # The vehicle class	 
        'Vehicle Count', # The vehicle count	 
        'Average Speed',# The average speed observed for this vehicle class 
        # during the summary interval.	MPH
        'Violation Count',	# The number of violations observed during his 
        # summary interval.	 
        'Violation Codes',	# A set of tuples of the form 'violation code' 
        # = 'count': ...	 
        'Single Axle Count', # The single axle count	 
        'Tandem Axle Count', # Tandem axle count	 
        'Tridem Axle Count', # The tridem axle count	   
        'Quad Axle Count', # The quad axle count	    
        'Average Gross Weight',	 # The average gross weight	unknown   
        'Gross Weight Distribution', # The gross weight distribution	    
        'Average Single Weight', # The average single weight	    
        'Average Tandem Weight', # The average tandem weight	 
        'Average Tridem Weight', # The average tridem weight	 
        'Average Quad Weight', # The average quad weight	    
        'Average Vehicle Length', # The average vehicle length	 
        'Vehicle Length Distribution', # The vehicle length distribution	 
        'Average Tandem Spacing',# The average tandem spacing	 
        'Average Tridem Spacing',# The average tridem spacing	 
        'Average Quad Spacing',	# The average quad spacing	 
        'Average Wheelbase', # The average wheelbase	 
        'Wheelbase Distribution', #The wheelbase distribution
        'Total Flex ESAL 300', # Total Flex ESAL 300	 
        'Total Flex ESAL 285', # Total Flex ESAL 285	 
        'Total Rigid ESAL 300', # Total Rigid ESAL 300	 
        'Total Rigid ESAL 285'] # Total Rigid ESAL 285
    
    # Open files for 2019-2020
    path = DIR_MOBILITY+'losangeles/'
    all_files = glob.glob(path + '/*.txt')
    all_files.sort()
    df = []
    for fname in all_files:
        li = pd.read_csv(fname, names=colnames, header=None)
        df.append(li)
    df = pd.concat(df, axis=0, ignore_index=True)    
    # Add empty rows for latitude, longitude
    df['Latitude'], df['Longitude'] = np.nan, np.nan
    
    # Information about vehicle size class comes from 
    # https://en.wikipedia.org/wiki/Vehicle_size_class:
    # 1:    Motorcycles
    # 2:	Passenger Cars 
    # 3:	Other Two-Axle Four-Tire Single-Unit Vehicles
    # 4:	Buses
    # 5:	Two-Axle, Six-Tire, Single-Unit Trucks
    # 6:	Three-Axle Single-Unit Trucks
    # 7:	Four or More Axle Single-Unit Trucks
    # 8:    Four or Fewer Axle Single-Trailer Trucks
    # 9:    Five-Axle Single-Trailer Trucks	
    # 10:	Six or More Axle Single-Trailer Trucks	
    # 11:	Five or Fewer Axle Multi-Trailer Trucks	
    # 12:	Six-Axle Multi-Trailer Trucks
    # 13:	Seven or More Axle Multi-Trailer
    # 14:	Unused	----	----
    # 15:	Unclassified Vehicle
    # Check out this infographic too: 
    # http://onlinemanuals.txdot.gov/txdotmanuals/tri/images/FHWA_Classification_Chart_FINAL.png
    
    # From a quick inspection of "Census Station Identifier" 129000 summed
    # over January 2019, here's the distribution:
    # vclass = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # counts = [0.0,12026.0,63916.0,15304.0,190176.0,40634.0,13424.0,
    #     29278.0,221582.0,782.0,9430.0,2028.0,152.0,9842.0,12122.0]
    # plt.bar(vclass,counts)
    # plt.xlim([1,16])
    # plt.xticks(np.arange(1,16,1))
    # plt.show()
    # So the largest contributors: 3 (two-axle four tire vehicles), 
    # 5 (two axle, six-tire trucks), and 9 (five-axle single-trailer trucks)
    losangeles = ['06037','06059']
    # Identify a station by the "Census Station Identifier". Note that each 
    # census station identifier has a couple substation identifiers associated 
    # with it (corresponding to a North and South Directions). Additionally, 
    # all station identifiers should have identical County Identifier,
    # District Identifier, and Absolute Postmile.
    latraf = []
    for stationid in np.unique(df['Census Station Identifier'].values):
        station = df.loc[df['Census Station Identifier']==stationid].copy()
        # Get unique match/join values 
        abs_post = np.unique(station['Absolute Postmile'].values)
        free_id = np.unique(station['Freeway Identifier'].values)
        free_direction = np.unique(station['Freeway Direction'].values)
        county_id = np.unique(station['County Identifier'].values)
        district_id = np.unique(station['District Identifier'].values)
        # Find siting information from siteinfo DataFrame using match/join
        # values
        siteinfo = sites.loc[
            (sites['Freeway ID'].isin(free_id)) &
            (sites['Freeway Direction'].isin(free_direction)) &
            (sites['County ID'].isin(county_id)) &
            (sites['District ID'].isin(district_id))]
        # Since the absolute postmile parameter is wonky and sometimes doesn't
        # match between the traffic data and site information, deal with 
        # this parameter differently 
        lattemp, lngtemp = [], []
        for ap in abs_post:
            siteinfo_ap = siteinfo.iloc[(siteinfo['Absolute Postmile']-ap
                ).abs().argsort()[:1]]
            lattemp.append(siteinfo_ap['Latitude'].values[0])
            lngtemp.append(siteinfo_ap['Longitude'].values[0])
        # Ensure that stations (if > 1 exists on N/S freeway directions) are
        # close to each other. This step is a bit kludgy
        # if ((np.ptp(lattemp) > 0.1) or (np.ptp(lngtemp) > 0.01)):
        #     print('Lat and/or lon for different highway directions at '+
        #         'Census Station Identifier %s are > 0.01 deg apart!'%(
        #         station['Census Station Identifier'].values[0]))
        # Fetch FIPS census block number based on census station 
        # latitude/longitude
        response = requests.get('https://geo.fcc.gov/api/census/block/'+
            'find?latitude=%s&longitude=%s&format=json'%(
            np.nanmean(lattemp), np.nanmean(lngtemp)))
        fips = response.json()['Block']['FIPS']
        # Lob off state + county code (first five digits) from FIPS
        fips = fips[:5]
        if fips in losangeles:
            # Group by date and vehicle class (i.e., combine over lane 
            # direction and lane number)
            station = station.groupby(by=['Timestamp','Vehicle Class']).agg(
                {'Vehicle Count':'sum'})
            station = station.reset_index()
            # Pivot table 
            station = station.pivot(index='Timestamp', columns='Vehicle Class', 
                values='Vehicle Count')
            station.columns = station.columns.map(str)
            # Sum over total vehicle count, passenger vehicles, and freight
            station['Count'] = station.loc[:,'0':'15'].sum(axis=1)
            station['Passenger'] = station.loc[:,'0':'5'].sum(axis=1)
            station['Freight'] = station.loc[:,'6':'15'].sum(axis=1)
            # Drop unneeded columns
            station = station[['Count','Passenger','Freight']]
            station['Site'] = stationid
            station['Latitude'] = np.nanmean(lattemp)
            station['Longitude'] = np.nanmean(lngtemp)
            station.index = pd.to_datetime(station.index)
            latraf.append(station)
    latraf = pd.concat(latraf, axis=0)
    df.rename(columns={'Datum':'Date'}, inplace=True)
