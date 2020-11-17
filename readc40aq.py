#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:08:37 2020

@author: ghkerr
"""

DIR = '/Users/ghkerr/GW/'
DIR_AQ = DIR+'data/aq/'

def read_auckland(pollutant, startdate, enddate):
    """Read daily air quality data for either PM2.5, O3, or NO2 sites around
    Auckland for specified time period. Data are paired with the site latitude 
    and longitude coordinates and the site name. Note that no quality control 
    is applied, and this should be done in the future with the "Status Code" 
    columns. 
    The output unit for NO2 is ppb, output units for O3 is ppb, and output
    units for PM2.5 is ug/m3. 
    
    Parameters
    ----------
    pollutant : str
        Pollutant of interest, either PM2.5, O3, or NO2
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Air quality observations in Auckland for a given pollutant and time 
        period with site information and coordinates
    """
    import pandas as pd 
    import numpy as np
    # Make dictionary with site name and coordinates. Coordinates are found
    # in 'site info.xls' in units of NZMG Easting and Northing and converted
    # to standard decimal degree lat/lon with 
    # https://www.geodesy.linz.govt.nz/concord/index.cgi
    sites = {'Glen Eden':(-36.92213583, 174.65200725),
        'Henderson':(-36.86799137, 174.62835040),
        'Khyber Pass':(-36.86634888, 174.77116271),
        'Queen Street':(-36.84764873, 174.76558144),
        'Takapuna':(-36.78025339, 174.74884838),
        'Penrose':(-36.90455276, 174.81556349),
        'Patumahoe':(-37.20443705, 174.86430049)}
    # For O3
    if pollutant=='O3':
        # Don't parse all columns, eliminate columns that repeat information 
        # (e.g., NO2 observations are repeated in ug/m3 and ppb) 
        dtype = {'Date':np.str, 
            'AC Patumahoe OZONE_MCG 24h average [µg/m³]':np.str, 
            'Status Code':np.str,
            'AC Patumahoe OZONE_PPB 24h average [ppb]':np.str, 
            'Status Code.1':np.str}
        df = pd.read_excel(DIR_AQ+'auckland/'+'daily %s.xls'%pollutant, 
            sheet_name=0, header=2, dtype=dtype)
        # Eliminate columns that repeat information 
        # (e.g., NO2 observations are repeated in ug/m3 and ppb) 
        to_drop = ['AC Patumahoe OZONE_MCG 24h average [µg/m³]', 
            'Status Code', 'Status Code.1']
        df = df.drop(to_drop, axis=1)
        # Rename station names to match sites dictionary 
        df.rename(columns={'AC Patumahoe OZONE_PPB 24h average [ppb]':
            'Patumahoe'}, inplace=True)
    # For PM2.5    
    if pollutant=='PM2.5':
        dtype = {'Date':np.str,
            'AC Patumahoe PM2.5 24h average [µg/m³]':np.str,
            'Status Code':np.str,
            'AC Penrose PM2.5 24h average [µg/m³]':np.str,
            'Status Code.1':np.str,
            'AC Queen Street PM2.5 24h average [µg/m³]':np.str,
            'Status Code.2':np.str,
            'AC Takapuna PM2.5 24h average [µg/m³]':np.str,
            'Status Code.3':np.str}
        df = pd.read_excel(DIR_AQ+'auckland/'+'daily %s.xls'%pollutant, 
            sheet_name=0, header=2, dtype=dtype)
        to_drop = ['Status Code', 'Status Code.1', 'Status Code.2', 
            'Status Code.3']
        df = df.drop(to_drop, axis=1)
        df.rename(columns={'AC Patumahoe PM2.5 24h average [µg/m³]':
            'Patumahoe', 'AC Penrose PM2.5 24h average [µg/m³]':'Penrose',
            'AC Queen Street PM2.5 24h average [µg/m³]':'Queen Street',
            'AC Takapuna PM2.5 24h average [µg/m³]':'Takapuna'}, inplace=True)
    # For NO2    
    if pollutant=='NO2':
        dtype = {'Date':np.str,     
            'AC Glen Eden NO2_MCG 24h average [µg/m³]':np.str, 
            'Status Code':np.str,
            'AC Glen Eden NO2_PPB 24h average [ppb]':np.str, 
            'Status Code.1':np.str,
            'AC Henderson NO2_MCG 24h average [µg/m³]':np.str,
            'Status Code.2':np.str,
            'AC Henderson NO2_PPB 24h average [ppb]':np.str, 
            'Status Code.3':np.str,
            'AC Khyber Pass NO2_PPB 24h average [ppb]':np.str,
            'Status Code.4':np.str,
            'AC Patumahoe NO2_MCG 24h average [µg/m³]':np.str,
            'Status Code.5':np.str,
            'AC Patumahoe NO2_PPB 24h average [ppb]':np.str,
            'Status Code.6':np.str,
            'AC Penrose NO2_MCG 24h average [µg/m³]':np.str, 
            'Status Code.7':np.str,
            'AC Penrose NO2_PPB 24h average [ppb]':np.str, 
            'Status Code.8':np.str,
            'AC Queen Street NO2_MCG 24h average [µg/m³]':np.str, 
            'Status Code.9':np.str,
            'AC Queen Street NO2_PPB 24h average [ppb]':np.str, 
            'Status Code.10':np.str,
            'AC Takapuna NO2_MCG 24h average [µg/m³]':np.str, 
            'Status Code.11':np.str,
            'AC Takapuna NO2_PPB 24h average [ppb]':np.str,
            'Status Code.12':np.str}
        df = pd.read_excel(DIR_AQ+'auckland/'+'daily %s.xls'%pollutant, 
            sheet_name=0, header=2, dtype=dtype)
        to_drop = ['AC Glen Eden NO2_MCG 24h average [µg/m³]', 'Status Code',
           'AC Henderson NO2_MCG 24h average [µg/m³]', 'Status Code.2',
           'AC Patumahoe NO2_MCG 24h average [µg/m³]', 'Status Code.5',
           'AC Penrose NO2_MCG 24h average [µg/m³]', 'Status Code.7',
           'AC Queen Street NO2_MCG 24h average [µg/m³]', 'Status Code.9',
           'AC Takapuna NO2_MCG 24h average [µg/m³]', 'Status Code.11', 
           'Status Code.1', 'Status Code.3', 'Status Code.4', 
           'Status Code.6', 'Status Code.8', 'Status Code.10', 'Status Code.12'] 
        df = df.drop(to_drop, axis=1)
        df.rename(columns={'AC Glen Eden NO2_PPB 24h average [ppb]':'Glen Eden', 
            'AC Henderson NO2_PPB 24h average [ppb]':'Henderson',
            'AC Khyber Pass NO2_PPB 24h average [ppb]':'Khyber Pass',
            'AC Patumahoe NO2_PPB 24h average [ppb]':'Patumahoe',
            'AC Penrose NO2_PPB 24h average [ppb]':'Penrose',
            'AC Queen Street NO2_PPB 24h average [ppb]':'Queen Street',
            'AC Takapuna NO2_PPB 24h average [ppb]':'Takapuna'}, inplace=True)
    # Convert observations to numeric
    df = df.apply(pd.to_numeric, args=('coerce',))
    # Replace Date column with list of dates since these are not read in 
    # correctly; n.b., that this is hard-coded in from the information 
    # at the beginning of the Auckland AQ data (header gives the first day 
    # and last day + 1, so subtract a day from the last day to get the 
    # right dimensions)
    daterange = pd.date_range('01/01/2019','12/31/2020')
    df['Date'] = daterange
    # Make observations from different stations individual rows rather than 
    # having different stations as different columns
    df = pd.melt(df, id_vars=['Date'], var_name='Site', 
        value_name='Concentration')
    df = df.astype({'Concentration':'float64', 'Site':'str'})
    # Add columns for lat/lon coordinates
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan
    for site in sites:
        site_lat = sites[site][0]
        site_lng = sites[site][1]
        df.loc[(df.Site == site), 'Latitude'] = site_lat
        df.loc[(df.Site == site), 'Longitude'] = site_lng
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]    
    return df

def read_santiago(pollutant, startdate, enddate):
    """Function reads air quality observations from stations around 
    Santiago. Hourly observations are averaged to daily concentrations. Note 
    that some stations record repeated 1.0 values - unsure what this is about.
    The output unit for NO2 is ppb, output units for O3 is ppb, and output
    units for PM2.5 is ug/m3. 

    Parameters
    ----------
    pollutant : str
        Pollutant of interest, either PM2.5, O3, or NO2
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Air quality observations in Santiago for a given pollutant and time 
        period with site information and coordinates
    """
    import pandas as pd
    import numpy as np
    # Site information is given in 'METADATA' page of air quality Excel file. 
    # Position are given in coordenadas UTM and converted to standard decimal
    # degree lat/lon with http://rcn.montana.edu/Resources/Converter.aspx
    # Note that Santiago is in UTM Zone 19
    sites = {'Independencia':(-33.422261189967536, -70.65113863328291),
        'La Florida':(-33.5166668832046, -70.58816010358973),
        'Las Condes':(-33.37677604807965, -70.52325614853385),
        'Santiago':(-33.464176505068316, -70.66070229344474),
        'Pudahuel':(-33.437785357125556, -70.75014408796092),
        'Cerrillos':(-33.49578053774785, -70.7042093293862),
        'El Bosque':(-33.54701601896996, -70.66616286078217),
        'Cerro Navia':(-33.43307040276798, -70.73205509366429),
        'Puente Alto':(-33.59135624038916, -70.59443067815702),
        'Talagante':(-33.67381931299035, -70.9529961584255),
        'Quilicura':(-33.3496312950099, -70.72384576583242)}
    df = pd.DataFrame([])
    # For PM2.5
    if pollutant=='PM2.5':
        # Format is really annoying. Columns A-L represent 2018, columns N-Y
        # represent 2019, columns AA-AL represent 2020. Read in file three
        # times and concatenate
        for year, cols in zip([2018,2019,2020],['A:L','N:Y','AA:LL']):
            dfty = pd.read_excel(DIR_AQ+'santiago/'+
                'solicitud C40-MP2.5_NO2_O3_2018_2020_2.xlsx', 
                sheet_name='MP2.5', header=0, usecols=cols)
            # Rename columns 
            if year==2018:
                dfty.rename(columns={'MP2.5 (ug/m3)':'Date'}, inplace=True) 
            if year==2019:
                dfty.rename(columns={'Estación':'Date',
                    'Independencia.1':'Independencia', 'La Florida.1':
                    'La Florida', 'Las Condes.1':'Las Condes', 'Santiago.1':
                    'Santiago', 'Pudahuel.1':'Pudahuel', 'Cerrillos.1':
                    'Cerrillos', 'El Bosque .1':'El Bosque', 'Cerro Navia.1':
                    'Cerro Navia', 'Puente Alto.1':'Puente Alto', 
                    'Talagante.1':'Talagante', 'Quilicura.1':'Quilicura'}, 
                    inplace=True)
            if year==2020:
                dfty.rename(columns={'Estación.1':'Date',
                    'Independencia.2':'Independencia', 'La Florida.2':
                    'La Florida', 'Las Condes.2':'Las Condes', 'Santiago.2':
                    'Santiago', 'Pudahuel.2':'Pudahuel', 'Cerrillos.2':
                    'Cerrillos', 'El Bosque .2':'El Bosque', 'Cerro Navia.2':
                    'Cerro Navia', 'Puente Alto.2':'Puente Alto', 
                    'Talagante.2':'Talagante', 'Quilicura.2':'Quilicura'}, 
                    inplace=True)   
            df = df.append(dfty, ignore_index=False)            
    # For O3
    if pollutant=='O3':
        for year, cols in zip([2018,2019,2020],['A:J','L:U','W:AF']):
            dfty = pd.read_excel(DIR_AQ+'santiago/'+
                'solicitud C40-MP2.5_NO2_O3_2018_2020_2.xlsx', 
                sheet_name='O3', header=0, usecols=cols)
            if year==2018:
                dfty.rename(columns={'O3 (ppb)':'Date',
                    'El Bosque ':'El Bosque'}, inplace=True) 
            if year==2019:
                dfty.rename(columns={'ppm':'Date',
                    'Independencia.1':'Independencia', 'La Florida.1':
                    'La Florida', 'Las Condes.1':'Las Condes', 'Santiago.1':
                    'Santiago', 'Pudahuel.1':'Pudahuel', 'El Bosque .1':
                    'El Bosque', 'Cerro Navia.1':'Cerro Navia', 
                    'Puente Alto.1':'Puente Alto', 'Talagante.1':
                    'Talagante'}, inplace=True)
            if year==2020:
                dfty.rename(columns={'ppm.1':'Date',
                    'Independencia.2':'Independencia', 'La Florida.2':
                    'La Florida', 'Las Condes.2':'Las Condes', 'Santiago.2':
                    'Santiago', 'Pudahuel.2':'Pudahuel', 'El Bosque .2':
                    'El Bosque', 'Cerro Navia.2':'Cerro Navia', 
                    'Puente Alto.2':'Puente Alto', 'Talagante.2':
                    'Talagante'}, inplace=True)
            df = df.append(dfty, ignore_index=False)
    # For NO2
    if pollutant=='NO2':
        for year, cols in zip([2018,2019,2020],['A:J','L:U','W:AD']):
            dfty = pd.read_excel(DIR_AQ+'santiago/'+
                'solicitud C40-MP2.5_NO2_O3_2018_2020_2.xlsx', 
                sheet_name='NO2', header=0, usecols=cols)
            if year==2018:
                dfty.rename(columns={'NO2 (ppb)':'Date', 'El Bosque ':
                    'El Bosque'}, inplace=True) 
            if year==2019:
                dfty.rename(columns={'ppb':'Date','Independencia.1':
                    'Independencia', 'La Florida.1':'La Florida', 
                    'Las Condes.1':'Las Condes', 'Santiago.1':'Santiago', 
                    'Pudahuel.1':'Pudahuel', 'El Bosque .1':'El Bosque', 
                    'Cerro Navia.1':'Cerro Navia', 'Puente Alto.1':
                    'Puente Alto', 'Talagante.1':'Talagante'}, inplace=True)
            if year==2020:
                dfty.rename(columns={'ppb.1':'Date', 'La Florida.2':
                    'La Florida', 'Las Condes.2':'Las Condes', 
                    'Santiago.2':'Santiago', 'Pudahuel.2':'Pudahuel', 
                    'El Bosque .2':'El Bosque', 'Cerro Navia.2':'Cerro Navia', 
                    'Puente Alto.2':'Puente Alto'}, inplace=True)
            df = df.append(dfty, ignore_index=False)
    # Convert observations to numeric
    df = df.apply(pd.to_numeric, args=('coerce',))
    # Based on plotting diurnal curves of O3 and NO2 for some random days, 
    # it appears that the time is in local time. Set time as index.
    df['Date'] = pd.to_datetime(df['Date'])
    # Calculate daily average 
    df = df.resample('d', on='Date').mean().dropna(how='all')
    df = pd.melt(df.reset_index(), id_vars=['Date'], var_name='Site', 
        value_name='Concentration')
    df = df.astype({'Concentration':'float64', 'Site':'str'})
    # Add columns for lat/lon coordinates
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan
    for site in sites:
        site_lat = sites[site][0]
        site_lng = sites[site][1]
        df.loc[(df.Site == site), 'Latitude'] = site_lat
        df.loc[(df.Site == site), 'Longitude'] = site_lng
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]    
    return df

def read_berlin(pollutant, startdate, enddate):
    """Read hourly O3, PM2.5, or NO2 from in-situ monitoring sites in Berlin. 
    Daily-average concentrations at each site are returned along with 
    coordinate information. 
    
    Parameters
    ----------
    pollutant : str
        Pollutant of interest, either PM2.5, O3, or NO2
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Air quality observations in Berlin for a given pollutant and time 
        period with site information and coordinates
    """
    import numpy as np
    import pandas as pd
    sites = {'MC010':('Amrumer Str.', 52.543041, 13.349326),
        'MC018':('Belziger Str.', 52.485814, 13.348775),
        'MC027':('Schichauweg', 52.398406, 13.368103),
        'MC032':('Jagen', 52.473192, 13.225144),
        'MC042':('Nansenstr.', 52.489451, 13.430844),
        'MC077':('Wiltbergstr.', 52.643525, 13.489531),
        'MC085':('Mueggleseedamm', 52.447697, 13.64705),
        'MC115':('Hardenbergplatz', 52.5066, 13.332972),
        'MC117':('Schildhornstr.', 52.463611, 13.31825),
        'MC124':('Mariendorfer Damm', 52.438115, 13.38772),
        'MC143':('Silbersteinstr.', 52.467535, 13.44173),
        'MC145':('Jaegerstieg', 52.653269, 13.296081),
        'MC171':('Brueckenstr.', 52.513606, 13.418833),
        'MC174':('Frankfurter Allee', 52.514072, 13.469931),
        'MC220':('Karl-Marx Str.', 52.481709, 13.433967),
        'MC282':('Rheingoldstr.', 52.485296, 13.529504),
        'MW088':('Leipziger Strasse', 52.510178, 13.388321)}
    # Read pollutant-specific file
    df = pd.read_csv(DIR_AQ+'berlin/'+'%s-Stundendaten_Hr.csv'%(
        pollutant), delimiter=';', header=0, engine='python')
    # Air pollution measurements are stored with Central European Time 
    # reference standard (CET). Traffic data are stored with Central European
    # Summer/Winter Time reference standard. (CET during winter and CEST 
    # during summer). So, there might be the necessity to uniform time 
    # standard in order to do a comparative analysis. All hourly data refer 
    # to the hour before the time stamp, i.e. a value assigned to the hour 
    # 05:00 hours refers to values between 04:01 – 05:00 hours.
    # Thus, change values from 01:00:00-24:00:00 to 00:00:00-23:00:00 such that 
    # they are compatible with datetime 
    hrdict = {'01:00:00':'00:00:00', '02:00:00':'01:00:00', 
        '03:00:00':'02:00:00', '04:00:00':'03:00:00', 
        '05:00:00':'04:00:00', '06:00:00':'05:00:00', 
        '07:00:00':'06:00:00', '08:00:00':'07:00:00', 
        '09:00:00':'08:00:00', '10:00:00':'09:00:00',
        '11:00:00':'10:00:00', '12:00:00':'11:00:00', 
        '13:00:00':'12:00:00', '14:00:00':'13:00:00', 
        '15:00:00':'14:00:00', '16:00:00':'15:00:00', 
        '17:00:00':'16:00:00', '18:00:00':'17:00:00', 
        '19:00:00':'18:00:00', '20:00:00':'19:00:00',
        '21:00:00':'20:00:00', '22:00:00':'21:00:00', 
        '23:00:00':'22:00:00', '24:00:00':'23:00:00'}
    df = df.replace({'Stunde':hrdict})
    df['Datum_Stunde'] = pd.to_datetime(df['Datum'] + ' ' + df['Stunde'],
        format='%d.%m.%Y %H:%M:%S')
    # Drop separate date/hour columns
    df = df.drop(['Datum','Stunde'], axis=1)
    # Daily average
    df = df.resample('d', on='Datum_Stunde').mean().dropna(how='all')
    # Melt 
    df = pd.melt(df.reset_index(), id_vars=['Datum_Stunde'], var_name='Site', 
        value_name='Concentration')
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan
    # Rename site names to street/area names and add lat/lon coordinates
    for site in sites:
        site_name = sites[site][0]
        site_lat = sites[site][1]
        site_lng = sites[site][2]
        df.loc[(df.Site == site), 'Latitude'] = site_lat
        df.loc[(df.Site == site), 'Longitude'] = site_lng
        df.loc[(df.Site == site), 'Site'] = site_name
    # Rename time information and make index
    df.rename(columns={'Datum_Stunde':'Date'}, inplace=True)
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]
    # Assuming that the units of NO2 and O3 are in μg/m3, we need to convert
    # to ppb to obtain standardized units. From https://www2.dmu.dk/
    # AtmosphericEnvironment/Expost/database/docs/PPM_conversion.pdf
    # NO2 1 ppb = 1.88 μg/m3 
    # O3 1 ppb = 2.00 μg/m3 
    # The general equation is μg/m3 = (ppb)*(12.187)*(M) / (273.15 + °C)
    # where M is the molecular weight of the gaseous pollutant. An 
    # atmospheric pressure of 1 atmosphere is assumed.    
    if pollutant=='NO2':
        df['Concentration'] = df['Concentration']/1.88
    if pollutant=='O3':
        df['Concentration'] = df['Concentration']/2.
    return df

def read_mexicocity(pollutant, startdate, enddate):
    """Read air quality observations from Mexico City for the specified 
    pollutant and time period. Observations are hourly and averaged to daily 
    mean values. I believe that the native units are ppb for O3 and ppb for
    NO2, but this should be confirmed in the future. 
    
    Parameters
    ----------
    pollutant : str
        Pollutant of interest, either PM2.5, O3, or NO2
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Air quality observations in Berlin for a given pollutant and time 
        period with site information and coordinates
    """
    import numpy as np
    import pandas as pd
    sites = {'ACO':('Acolman',19.635501,-98.912003),
        'AJU':('Ajusco',19.154674,-99.162459),
        'AJM':('Ajusco Medio',19.272100,-99.207658),
        'ATI':('Atizapan',19.576963	,-99.254133),
        'BJU':('Benito Juarez',19.371612,-99.158969),
        'CAM':('Camarones',19.468404,-99.169794),
        'CCA':('Centro de Ciencias de la Atmosfera',19.3262,-99.1761),
        'TEC':('Cerro del Tepeyac',19.487227,-99.114229),
        'CHO':('Chalco',19.266948,-98.886088),
        'COR':('CORENA',19.265346,-99.02604),
        'CUA':('Cuajimalpa',19.365313,-99.291705),
        'CUT':('Cuautitlan',19.722186,-99.198602),
        'DIC':('Diconsa',19.298819,-99.185774),
        'EAJ':('Ecoguardas Ajusco',19.271222,-99.20397),
        'EDL':('Ex Convento Desierto de los Leones',19.313357,-99.310635),
        'FAC':('FES Acatlan',19.482473,-99.243524),
        'FAR':('FES Aragon',19.473692,-99.046176),
        'GAM':('Gustavo A. Madero',19.4827,-99.094517),
        'HGM':('Hospital General de Mexico',19.411617,-99.152207),
        'INN':('Investigaciones Nucleares',19.291968,-99.38052),
        'IZT':('Iztacalco',19.384413,-99.117641),
        'LPR':('La Presa',19.534727,-99.11772),
        'LAA':('Laboratorio de Analisis Ambiental',19.483781,-99.147312),
        'IBM':('Legaria',19.443319,-99.21536),
        'LOM':('Lomas',19.403,-99.242062),
        'LLA':('Los Laureles',19.578792,-99.039644),
        'MER':('Merced',19.42461,-99.119594),
        'MGH':('Miguel Hidalgo',19.404050,-99.202603),
        'MPA':('Milpa Alta',19.176900,-98.990189),
        'MON':('Montecillo',19.460415,-98.902853),
        'MCM':('Museo de la Ciudad de Mexico',19.429071,-99.131924),
        'NEZ':('Nezahualcoyotl',19.393734,-99.028212),
        'PED':('Pedregal',19.325146,-99.204136),
        'SAG':('San Agustin',19.532968,-99.030324),
        'SNT':('San Nicolas Totolapan',19.250385,-99.256462),
        'SFE':('Santa Fe',19.357357,-99.262865),
        'SAC':('Santiago Acahualtepec',19.34561,-99.009381),
        'TAH':('Tlahuac',19.246459,-99.010564),
        'TLA':('Tlalnepantla',19.529077,-99.204597),
        'TLI':('Tultitlan',19.602542,-99.177173),
        'UIZ':('UAM Iztapalapa',19.360794,-99.07388),
        'UAX':('UAM Xochimilco',19.304441,-99.103629),
        'VIF':('Villa de las Flores',19.658223,-99.09659),
        'XAL':('Xalostoc',19.525995,-99.0824)}
    # # Check location of stations against 
    # # http://www.aire.cdmx.gob.mx/default.php?opc=%27ZaBhnmM=%27
    # # with the following
    # lat = [x[1] for x in sites.values()]
    # lng = [x[2] for x in sites.values()]
    # txts = [x for x in sites.keys()]
    # plt.scatter(lng, lat)
    # for i, txt in enumerate(txts):
    #     plt.annotate(txt, (lng[i], lat[i]))
    if pollutant == 'PM2.5':
        pollutant = 'PM25'
    df = pd.DataFrame([])
    for year in [2019, 2020]:
        dfty = pd.read_csv(DIR_AQ+'mexicocity/'+'%s_ene_jun_%d.csv'%(
            pollutant, year), delimiter=',', header=0, engine='python')
        # Replace missing data (-99) with NaNs
        dfty = dfty.replace([-99], np.nan, regex=True)
        dfty.rename(columns={'date':'Date'}, inplace=True)
        dfty['Date'] = pd.to_datetime(dfty['Date'], format='%d/%m/%Y %H:%M')
        # Calculate daily average 
        dfty = dfty.resample('d', on='Date').mean().dropna(how='all')
        df = df.append(dfty, ignore_index=False)            
    df = pd.melt(df.reset_index(), id_vars=['Date'], var_name='Site', 
        value_name='Concentration')
    # Add columns for lat/lon coordinates
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan
    for site in sites:
        site_name = sites[site][0]
        site_lat = sites[site][1]
        site_lng = sites[site][2]
        df.loc[(df.Site == site), 'Latitude'] = site_lat
        df.loc[(df.Site == site), 'Longitude'] = site_lng
        df.loc[(df.Site == site), 'Site'] = site_name
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]    
    return df 

def read_losangeles(pollutant, startdate, enddate):
    """Read air quality observations from Los Angeles for the specified 
    pollutant and time period. Observations are hourly and averaged to daily 
    mean values.
    
    Parameters
    ----------
    pollutant : str
        Pollutant of interest, either PM2.5, O3, or NO2
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Air quality observations in Los Angeles for a given pollutant and time 
        period with site information and coordinates
    """
    import glob
    import numpy as np
    import pandas as pd
    # Site information (see files in sites/ subdirectory in data directory. 
    # These values were copy and pasted from there)
    sites = {'3658':(34.669589999999999, -118.13068),
        '2484':(34.136479999999999, -117.92392),
        '3743':(33.901445000000002, -118.204989),
        '2849':(34.144370000000002, -117.85038),
        '3843':(33.793712999999997, -118.171019),
        '2899':(34.06653, -118.22676),
        '3683':(33.955069999999999,-118.43049000000001),
        '3844':(34.181977000000003, -118.36303599999999),
        '2160':(34.132649999999998, -118.12714),
        '3693':(34.010289999999998, -118.0685),
        '2898':(34.066980000000001, -117.75138),
        '2420':(34.199199999999998, -118.5327499),
        '3502':(34.383369999999999, -118.52839),
        '2494':(34.051090000000002, -118.4564),
        '3679':(33.792400000000001, -118.17525000000001),
        '3818':(33.859662, -118.20070699999999)}
    if pollutant == 'O3':
        pollutant = 'OZONE'
    if pollutant == 'PM2.5':
        pollutant = 'PM25'
    # Columns and types
    if pollutant == 'PM25':
        dtype = {'site':np.str, 'monitor':np.str, 'date':np.str,
            'start_hour':np.str, 'value':np.float64, 'variable':np.str,
            'units':np.str, 'quality':np.str, 'prelim':np.str,
            'name':np.str}
    else: 
        dtype = {'site':np.str, 'date':np.str, 'start_hour':np.str, 
            'value':np.float64, 'variable':np.str, 'units':np.str, 
            'quality':np.str, 'prelim':np.str, 'name':np.str}
    # Fetch file names for pollutant of interest
    filenames = glob.glob(DIR_AQ+'losangeles/'+'%s*.csv'%pollutant)
    filenames.sort()
    df = pd.DataFrame([])
    for filename in filenames:
        dftm = pd.read_csv(filename, delimiter=',', header=0, dtype=dtype, 
            names=list(dtype.keys()), engine='python')
        # Identify empty rows (quasi end-of-file) and remove rows after this 
        # empty row (containing download information and QA information)
        dftm = dftm.iloc[:([i for i, x in enumerate(dftm.iloc[:,1].isna()) 
            if x][0])]
        # Calculate daily average at each site
        dftm['Date'] = pd.to_datetime(dftm['date']) + pd.to_timedelta(
            dftm['start_hour'].astype(float), unit='h')
        # Convert observations to numeric
        dftm['value'] = dftm['value'].apply(pd.to_numeric, args=('coerce',))
        # Calculate daily average at each site
        dftm = dftm.groupby('site').resample('d', on='Date').mean()
        dftm.reset_index(inplace=True)
        # Change column name
        dftm.rename(columns={'site':'Site', 'value':'Concentration'}, 
            inplace=True)
        # Add columns for lat/lon coordinates
        dftm['Latitude'] = np.nan
        dftm['Longitude'] = np.nan
        for site in sites:
            site_lat = sites[site][0]
            site_lng = sites[site][1]
            dftm.loc[(dftm.Site == site), 'Latitude'] = site_lat
            dftm.loc[(dftm.Site == site), 'Longitude'] = site_lng
        df = df.append(dftm, ignore_index=False)
    df.set_index('Date', drop=True, inplace=True)
    # Convert units to standardized units 
    if (pollutant=='NO2') or (pollutant=='OZONE'):
        df['Concentration'] = df['Concentration']*1000.
    df = df.loc[startdate:enddate]
    return df
    
def read_milan(pollutant, startdate, enddate):
    """Read daily air quality data for either PM2.5, O3, or NO2 sites in Milan
    for specified time period. Data are paired with the site latitude 
    and longitude coordinates and the site name. As of 16 November 2020, I am
    unsure what units the measurements are in and am checking with C40/Milan.
    
    Parameters
    ----------
    pollutant : str
        Pollutant of interest, either PM2.5, O3, or NO2
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Air quality observations in Milan for a given pollutant and time 
        period with site information and coordinates
    """    
    import pandas as pd
    import numpy as np
    # Site information given in the "Monitoring Site Location" of the page 
    sites = {'Verziere':(45.463346740666545, 9.195324807668857),
        'via Senato':(45.470499014097, 9.197460360112531),
        'viale Liguria':(45.443857653564926, 9.167944501742676),
        'viale Marche':(45.49631644365102, 9.190933555313624),
        'Pascal':(45.47899606168744, 9.235491038497502)}
    df = pd.DataFrame([])
    # For O3
    if pollutant=='O3':
        for year, cols in zip([2019,2020],['F:I','A:D']):
            dfty = pd.read_excel(DIR_AQ+'milan/'+
                'AQ data_Milan_AMAT.xlsx', sheet_name='O3_hourly', header=3, 
                usecols=cols)
            if year==2019:
                dfty.rename(columns={'Date.1':'Date', 'Hour.1':'Hour', 
                    'MI - Verziere.1':'Verziere', 'MI - Pascal.1':'Pascal'}, 
                    inplace=True)
            if year==2020:
                dfty.rename(columns={'MI - Verziere':'Verziere', 
                    'MI - Pascal':'Pascal'}, inplace=True)
                dfty = dfty.iloc[:([i for i, x in enumerate(dfty.iloc[:,1].isna()) 
                    if x][0])]            
            df = df.append(dfty, ignore_index=False)   
    # For NO2
    if pollutant=='NO2':
        for year, cols in zip([2019,2020],['I:O','A:G']):
            dfty = pd.read_excel(DIR_AQ+'milan/'+
                'AQ data_Milan_AMAT.xlsx', sheet_name='NO2_hourly', header=3, 
                usecols=cols)
            if year==2019:
                dfty.rename(columns={'Date.1':'Date', 'Hour.1':'Hour', 
                    'MI - Verziere.1':'Verziere', 'MI - Senato.1':'Senato',
                    'MI - Liguria.1':'Liguria', 'MI - Marche.1':'Marche', 
                    'MI - Pascal.1':'Pascal'}, inplace=True)
            if year==2020:
                dfty.rename(columns={'MI - Verziere':'Verziere', 
                    'MI - Senato':'Senato', 'MI - Liguria':'Liguria', 
                    'MI - Marche':'Marche', 'MI - Pascal':'Pascal'}, inplace=True)
                dfty = dfty.iloc[:([i for i, x in enumerate(dfty.iloc[:,1].isna()) 
                    if x][0])]            
            df = df.append(dfty, ignore_index=False)   
    # For PM2.5
    if pollutant=='PM2.5':
        for year, cols in zip([2019,2020],['E:G','A:C']):
            dfty = pd.read_excel(DIR_AQ+'milan/'+
                'AQ data_Milan_AMAT.xlsx', sheet_name='PM2.5_daily', header=3, 
                usecols=cols)
            if year==2019:
                dfty.rename(columns={'Date.1':'Date', 'MI - Senato.1':'Senato', 
                    'MI - Pascal.1':'Pascal'}, inplace=True)
            if year==2020:
                dfty.rename(columns={'MI - Senato':'Senato', 
                    'MI - Pascal':'Pascal'}, inplace=True)
                dfty = dfty.iloc[:([i for i, x in enumerate(dfty.iloc[:,0].isna()) 
                    if x][0])]            
            df = df.append(dfty, ignore_index=False)
    df['Date'] = pd.to_datetime(df['Date'])
    # Calculate daily average 
    if (pollutant=='NO2') or (pollutant=='O3'):
        df = df.resample('d', on='Date').mean().dropna(how='all')
    df = pd.melt(df.reset_index(), id_vars=['Date'], var_name='Site', 
        value_name='Concentration')
    df = df.astype({'Concentration':'float64', 'Site':'str'})
    # Add columns for lat/lon coordinates
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan
    for site in sites:
        site_lat = sites[site][0]
        site_lng = sites[site][1]
        df.loc[(df.Site == site), 'Latitude'] = site_lat
        df.loc[(df.Site == site), 'Longitude'] = site_lng
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]
    return 
    
def read_london(pollutant, startdate, enddate):
    """Function as the option to read pollutant observations for London 
    (individual files for each site) with subfuction "parse" and form daily-
    averaged, parsed summary files in the standardized format. If this part of 
    the function is commented out, then the previously-formed parsed file for 
    the pollutant and time period of interested will be read. 
    
    Parameters
    ----------
    pollutant : str
        Pollutant of interest, either PM2.5, O3, or NO2
    startdate : str
        Start date of period of interest; YYYY-mm-dd format    
    enddate : str
        End date of period of interest; YYYY-mm-dd format

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Air quality observations in London for a given pollutant and time 
        period with site information and coordinates
    """
    import numpy as np
    import glob
    import pandas as pd
    sites = {'Camden - Bloomsbury': (51.52229, -0.12585), 
        'Bexley - Slade Green FDMS': (51.46598, 0.18488),
        'Croydon - Norbury Manor': (51.41004, -0.12752), 
        'City of London - Farringdon Street': (51.51453, -0.10452), 
        'City of London - Sir John Cass School': (51.51385, -0.07777), 
        'Greenwich - Falconwood FDMS': (51.4563, 0.08561), 
        'Greenwich - A206 Burrage Grove': (51.49053, 0.074), 
        'Greenwich - Plumstead High Street': (51.48696, 0.09511), 
        'Greenwich - John Harrison Way': (51.49377, 0.01078), 
        'Greenwich - Eltham': (51.45258, 0.07077), 
        'Greenwich - Westhorne Avenue': (51.45636, 0.04073), 
        'Havering - Rainham': (51.52079, 0.20546), 
        'Kensington and Chelsea - North Ken FIDAS': (51.52105, -0.21349), 
        'Lewisham - New Cross': (51.47495, -0.03964), 
        'Lewisham - Deptford': (51.47953, -0.0252606), 
        'Westminster - Marylebone Road FDMS': (51.52254, -0.15459), 
        'Redbridge - Ley Street': (51.56948, 0.08291), 
        'Southwark - Elephant and Castle': (51.49316, -0.10153), 
        'Southwark - Tower Bridge Road': (51.50139, -0.0782), 
        'Sutton - Beddington Lane north': (51.38929, -0.14166), 
        'Richmond - Teddington Bushy Park': (51.42526, -0.34561), 
        'Tower Hamlets - Blackwall': (51.51505, -0.00842), 
        'Westminster - Horseferry Road': (51.49468, -0.13194), 
        'Hammersmith and Fulham - Hammersmith Town Centre': (51.4927, -0.22479), 
        'Hillingdon - Harmondsworth Os': (51.48753, -0.47945),
        'Hillingdon Harmondsworth Osiris': (51.48753, -0.47945), 
        'London Hillingdon Harmondsworth Os': (51.48753, -0.47945),    
        'Hounslow - Chiswick': (51.49251, -0.25725), 
        'Hounslow - Brentford': (51.4894, -0.31008), 
        'Hillingdon - Heathrow': (51.47917, -0.44056),
        'Heathrow LHR2': (51.47917, -0.44056),    
        'Hillingdon - Heathrow Bath Road': (51.48107, -0.442092),
        'Heathrow Bath Road': (51.48107, -0.442092),    
        'Newham - Cam Road': (51.5376, -0.00214), 
        'Newham - Wren Close': (51.51473, 0.01455), 
        'Heathrow Green Gates': (51.48148, -0.48668), 
        'Tower Hamlets - Roadside': (51.52253, -0.04216), 
        'Bromley - Harwood Avenue': (51.40555, 0.01888), 
        'Barking and Dagenham - Rush Green':(51.56375, 0.17789),
        'Barking and Dagenham - Scrattons Farm':(51.52939, 0.13286),
        'Bexley - Belvedere West':(51.49465, 0.13728),
        'Brent - Ikea':(51.55248, -0.25809),
        'Brent - Neasden Lane':(51.55266, -0.24877),
        'Brent - John Keble Primary School':(51.5378, -0.24779),
        'Brent - ARK Franklin Primary Academy':(51.53241, -0.21772),
        'Bexley - Slade Green':(51.46598, 0.18488),
        'Bexley - Belvedere':(51.49061, 0.15891),
        'Camden - Swiss Cottage':(51.54422, -0.17528),
        'Camden - Euston Road':(51.52798, -0.12877),
        'Croydon - Norbury':(51.41135, -0.12311),
        'Croydon - Purley Way A23':(51.36223, -0.1176),
        'Croydon - Park Lane':(51.37395, -0.09676),
        'City of London - Beech Street':(51.52023, -0.09611),
        'City of London - Walbrook Wharf':(51.5105, -0.09163),
        'Ealing - Hanger Lane Gyratory':(51.53085, -0.29249),
        'Ealing - Horn Lane':(51.51895, -0.26562),
        'Ealing - Western Avenue':(51.52361, -0.2655),
        'Ealing - Acton Vale':(51.50385, -0.25467),
        'Enfield - Bush Hill Park':(51.64504, -0.06618),
        'Enfield - Derby Road':(51.61486, -0.05077),
        'Enfield - Bowes Primary School':(51.61387, -0.12534),
        'Enfield - Prince of Wales School':(51.66864, -0.02201),
        'Greenwich - Falconwood':(51.4563, 0.08561),
        'Greenwich - A206 Burrage Grove':(51.49053, 0.074),
        'Greenwich - Plumstead High Street':(51.48696, 0.09511),
        'Greenwich - Fiveways Sidcup Rd A20':(51.43466, 0.06422),
        'Greenwich - Trafalgar Road (Hoskins St)':(51.48391, 0.00041),
        'Greenwich - John Harrison Way':(51.49377, 0.01078),
        'Greenwich - Eltham':(51.45258, 0.07077),
        'Greenwich - Blackheath':(51.4725, -0.01238),
        'Greenwich - Woolwich Flyover':(51.48688, 0.0179),
        'Greenwich - Westhorne Avenue':(51.45636, 0.04073),
        'Westminster - Ebury Street (Grosvenor)':(51.49349, -0.14991),
        'Westminster - Duke Street (Grosvenor)':(51.513, -0.15091),
        'Haringey - Haringey Town Hall':(51.5993, -0.06822),
        'Haringey - Priory Park South':(51.58398, -0.1254),
        'Hillingdon - Keats Way':(51.49631, -0.46083),
        'Hackney - Old Street':(51.52645, -0.08491),
        'Lewisham - Honor Oak Park':(51.44967, -0.03742),
        'Harrow - Stanmore':(51.61733, -0.29878),
        'Harrow - Pinner Road':(51.58842, -0.36299),
        'Havering - Rainham':(51.52079, 0.20546),
        'Havering - Romford':(51.57298, 0.17908),
        'Camden - Holborn (Bee Midtown)':(51.51737, -0.12019),
        'Islington - Holloway Road':(51.55538, -0.11615),
        'Islington - Arsenal':(51.5579, -0.10699),
        'Kensington and Chelsea - North Ken':(51.52105, -0.21349),
        'Kingston - Tolworth Broadway':(51.37931, -0.28126),
        'Kingston - Cromwell Road':(51.41231, -0.29658),
        'Kingston - Kingston Vale':(51.4355, -0.25703),
        'Lambeth - Brixton Road':(51.46411, -0.11458),
        'Lambeth - Bondway Interchange':(51.48549, -0.12455),
        'Lambeth - Streatham Green':(51.42821, -0.13187),
        'Hillingdon - Harlington':(51.48878, -0.44163),
        'Lewisham - Catford':(51.44547, -0.02027),
        'Lewisham - Loampit Vale':(51.46469, -0.01607),
        'Merton - Morden Civic Centre 2':(51.40162, -0.19589),
        'Westminster - Marylebone Road':(51.52254, -0.15459),
        'Westminster - Strand (Northbank BID)':(51.51197, -0.11671),
        'Redbridge - Gardner Close':(51.57661, 0.03086),
        'Redbridge - Ley Street':(51.56948, 0.08291),
        'Richmond - Chertsey Road':(51.45314, -0.34122),
        'Richmond - Castelnau':(51.48019, -0.23734),
        'Richmond - Barnes Wetlands':(51.47617, -0.23043),
        'Southwark - A2 Old Kent Road':(51.4805, -0.05955),
        'Sutton - Wallington':(51.35866, -0.14972),
        'Sutton - Worcester Park':(51.37792, -0.24041),
        'Sutton - Beddington Lane':(51.38357, -0.13642),
        'Tower Hamlets - Mile End Road':(51.52253, -0.04216),
        'Wandsworth - Wandsworth Town Hall':(51.45696, -0.19107),
        'Wandsworth - Putney High Street':(51.46343, -0.21587),
        'Wandsworth - Putney High Street Facade':(51.46372, -0.21589),
        'Wandsworth - Putney':(51.46503, -0.21582),
        'Wandsworth - Battersea':(51.47944, -0.14179),
        'Wandsworth - Tooting High Street':(51.42933, -0.16652),
        'Wandsworth - Lavender Hill (Clapham Jct)':(51.46369, -0.16671),
        'Westminster - Covent Garden':(51.51198, -0.12163),
        'Westminster - Oxford Street':(51.51393, -0.15279),
        'Westminster - Buckingham Palace Road':(51.49323, -0.14739),
        'Westminster - Oxford Street East':(51.51607, -0.13516),
        'Westminster - Cavendish Square':(51.5168, -0.14566),
        'Barnet - Tally Ho':(51.61468, -0.17661),
        'Barnet Tally Ho':(51.61468, -0.17661),        
        'Tally Ho':(51.61468, -0.17661),        
        'Barnet - Chalgrove School':(51.5919, -0.20599),
        'Barnet Chalgrove School':(51.5919, -0.20599),        
        'London Barnet Chalgrove School':(51.5919, -0.20599),        
        'Hammersmith and Fulham - Shepherds Bush':(51.50456, -0.22467),
        "H&F Shepherd's Bush":(51.50456, -0.22467),
        "Shepherd's Bush":(51.50456, -0.22467),        
        'Hammersmith Town Centre':(51.4927, -0.22479),
        'H&F Hammersmith Town Centre':(51.4927, -0.22479),    
        'Hillingdon - South Ruislip':(51.55226, -0.40278),
        'Hillingdon South Ruislip':(51.55226, -0.40278),
        'Hillingdon 1 - South Ruislip':(51.55226, -0.40278),
        'Hillingdon - Oxford Avenue':(51.48113, -0.42376),
        'Hillingdon Oxford Avenue':(51.48113, -0.42376),
        'London Hillingdon Oxford Avenue':(51.48113, -0.42376),        
        'Hillingdon - Harmondsworth':(51.48799, -0.48098),
        'London Hillingdon Harmondsworth':(51.48799, -0.48098),
        'Hillingdon Harmondsworth':(51.48799, -0.48098),
        'Hillingdon - Hayes':(51.49817, -0.41233),
        'Hillingdon Hayes':(51.49817, -0.41233),
        'London Hillingdon Hayes':(51.49817, -0.41233),
        'Hounslow - Boston Manor Park':(51.48986, -0.31751),
        'Hounslow Boston Manor Park':(51.48986, -0.31751),
        'Hounslow - Cranford':(51.48298, -0.4119),
        'Hounslow Cranford':(51.48298, -0.4119),
        'Hounslow Chiswick':(51.49251, -0.25725),
        'Hounslow Brentford':(51.4894, -0.31008),
        'Hounslow - Heston':(51.47913, -0.36476),
        'Hounslow Heston':(51.47913, -0.36476),
        'Hounslow - Hatton Cross':(51.4634, -0.42753),
        'Hounslow Hatton Cross':(51.4634, -0.42753),
        'Hounslow - Gunnersbury':(51.50068, -0.28438),
        'Hounslow Gunnersbury':(51.50068, -0.28438),
        'Hounslow - Feltham':(51.44739, -0.40873),
        'Hounslow Feltham':(51.44739, -0.40873),
        'Kensington and Chelsea - Cromwell Road':(51.4955, -0.17881),
        'RBKC Cromwell Rd':(51.4955, -0.17881),
        'Cromwell Road':(51.4955, -0.17881),        
        'Kensington and Chelsea - Knightsbridge':(51.49914, -0.16434),
        'RBKC Knightsbridge':(51.49914, -0.16434),
        'Knightsbridge':(51.49914, -0.16434),
        'Kensington and Chelsea - Chelsea':(51.48744, -0.1684),
        'RBKC Chelsea':(51.48744, -0.1684),
        'Chelsea':(51.48744, -0.1684),
        'Kensington and Chelsea - Earls Court Road':(51.4902, -0.19086),
        'RBKC Earls Court Road':(51.4902, -0.19086),
        'Earls Court Road':(51.4902, -0.19086),        
        'Cam Road':(51.5376, -0.00214),
        'Newham Cam Road':(51.5376, -0.00214),
        'Newham Wren Close':(51.51473, 0.01455),
        'Wren Close':(51.51473, 0.01455),        
        'Hillingdon - Sipson':(51.48438, -0.4557),
        'Hillingdon Sipson':(51.48438, -0.4557),
        'Sipson':(51.48438, -0.4557),
        'Hillingdon - Heathrow Green Gates':(51.48148, -0.48668),
        'Tower Hamlets - Millwall Park':(51.48913, -0.01298),
        'Tower Hamlets - Victoria Park':(51.54052, -0.03331),
        'Waltham Forest - Dawlish Road':(51.56238, -0.0049),
        'Waltham Forest Dawlish Rd':(51.56238, -0.0049),
        'Waltham Forest - Crooked Billet':(51.60173, -0.01644),
        'Waltham Forest Crooked Billet':(51.60173, -0.01644),
        'Waltham Crooked Billet':(51.60173, -0.01644),
        'Waltham Forest - Leyton':(51.55624, -0.01363),
        'Waltham Forest Leyton':(51.55624, -0.01363),
        'Bromley - Harwood Avenue':(51.40555, 0.01888),
        'Sevenoaks - Greatness Park':(51.289391, 0.201437),
        'Westminster - Elizabeth Bridge':(51.49224823, -0.147114753)}
    def parse(pollutant):
        """The native format of the air quality observations in London is messy
        (individual CSV files for each site). This function creates an output 
        file for daily-averaged concentrations of each pollutant in the 
        standardized format. 
    
        Parameters
        ----------
        pollutant : str
            Pollutant of interest, either PM2.5, O3, or NO2
    
        Returns
        -------
        df : pandas.core.frame.DataFrame
            Air quality observations in London for a given pollutant with site 
            information and coordinates    
        """
        files = glob.glob(DIR_AQ+'london/'+ '*.csv')
        files = files
        def get_merged(files, **kwargs):
            df = pd.read_csv(files[0], **kwargs)
            for f in files[1:]:
                print(f)
                df = df.merge(pd.read_csv(f, **kwargs), how='outer')
            return df
        df = get_merged(files)
        # Rename columns unified names
        df.rename(columns={'date':'Date', 'no2':'NO2', 'o3':'O3', 'pm25':'PM25',
            'site':'Site'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        if pollutant=='NO2':
            df = df[['Date', 'Site', 'NO2']]
            df = df.pivot(index='Date', columns='Site')['NO2']
        if pollutant=='PM2.5':
            df = df[['Date', 'Site', 'PM25']]
            df = df.pivot(index='Date', columns='Site')['PM25']
        if pollutant=='O3':
            df = df[['Date', 'Site', 'O3']]
            df = df.pivot(index='Date', columns='Site')['O3']
        df.reset_index(inplace=True)
        df = df.resample('d', on='Date').mean().dropna(how='all')    
        # Drop columns that are empty
        df = df.dropna(how='all', axis=1)
        df = pd.melt(df.reset_index(), id_vars=['Date'], var_name='Site', 
            value_name='Concentration')
        df = df.astype({'Concentration':'float64', 'Site':'str'})
        # Add columns for lat/lon coordinates
        df['Latitude'] = np.nan
        df['Longitude'] = np.nan
        for site in sites:
            site_lat = sites[site][0]
            site_lng = sites[site][1]
            df.loc[(df.Site == site), 'Latitude'] = site_lat
            df.loc[(df.Site == site), 'Longitude'] = site_lng
        df.set_index('Date', drop=True, inplace=True)
        return df
    # # Parse relevant information from input files 
    # pm25 = parse('PM2.5')
    # pm25.to_csv(DIR_AQ+'london/PM25_parsed_dailyavg.csv', encoding='utf-8')
    # o3 = parse('O3')
    # o3.to_csv(DIR_AQ+'london/O3_parsed_dailyavg.csv', encoding='utf-8')
    # no2 = parse('NO2')
    # no2.to_csv(DIR_AQ+'london/NO2_parsed_dailyavg.csv', encoding='utf-8')
    # Open parsed file for pollutant of interest 
    df = pd.read_csv(DIR_AQ+'london/'+'%s_parsed_dailyavg.csv'%(pollutant),
        delimiter=',', header=0, engine='python')
    df.set_index('Date', drop=True, inplace=True)
    df = df.loc[startdate:enddate]
    return df    
