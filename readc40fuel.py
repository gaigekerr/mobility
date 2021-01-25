#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:11:14 2020

@author: ghkerr

Since the data are extremely messy, values are copy and pasted from the 
input files and then concatenated to a standard output format. For a given 
city or metropolitan region, the standard output is a DataFrame with 
the following rows
+ Passenger (includes autos particulares, camionetas SUV, taxis; e.g. Mexico
    City)
+ Light duty
+ Heavy duty (includes light-heavy duty, medium-heavy-duty, and heavy-heavy-
    duty; e.g., Los Angeles)
+ Bus
+ Other (includes motorbikes; e.g., Milan)

Fuel types include: 
+ Gasoline (includes petrol)
+ Diesel
+ Other (includes LPG and NG)
"""

DIR = '/Users/ghkerr/GW/'
DIR_FUEL = DIR+'data/fuel/'

def read_mexicocity(): 
    """Information is gathered from "ZMVM_Fleet_VKT_fuel_2018" which includes 
    fleet, VKT, and fuel information from La Zona Metropolitana del Valle de 
    México (ZMVM) as well as the fleet by states, which are part of the ZMVM. 
    This function outputs a DataFrame with the standardized output (see 
    preamble docstring). 
    
    Parameters
    ----------
    None

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Fuel mix in the Mexico City metropolitan area from 2018 for major 
        transportation and fuel sectors, units of cubic meters per year
    """
    import pandas as pd
    import numpy as np
    # The following table represents total fuel consumption in ZMVM. Note 
    # that the "fuel" tab of the Excel spreadsheet contains values of 
    # "Consumo energético" in [m3/año] and [TJ/año], and the fractions 
    # calculated with these values are roughly the same.
    dtype = {'Tipo de vehiculo':np.str, 'Gasolina':np.float64, 
        'Diesel':np.float64,'GLP':np.float64, 'GN':np.float64, 
        'Total':np.float64}
    df_raw = pd.DataFrame(np.array([
        ['Autos Particulares', 4403810.91, 8742.24, 351.09, 104.05, 4413008.29],
        ['Camionetas SUV',	1813998.32, 5933.29, 350.95, 30.93, 1820313.49],
        ['Taxis', 1292964.75, 990.54, 3852.89, 4700.04, 1302508.22],
        ['Combis y vagonetas', 567927.96, 61342.29, 1037.08, 97.78, 630405.11],
        ['Microbuses', 98581.22, 4313.14, 434552.31, 1770.18, 539216.85],
        ['Pick up',	616417.92, 12508.86, 1076.17, 941.24, 630944.19],
        ['Vehículos ≤ 3.8 t', 673444.76, 103553.96, 9819.44, 184.79, 787002.95],
        ['Tractocamiones', 0.00, 3141.24, 0.00, 0.00, 3141.24],
        ['Autobuses', 11230.92, 651808.21, 1072.08, 2093.41, 666204.62],
        ['Vehículos > 3.8 t', 348736.15, 238844.15, 91600.63, 1211.13, 680392.07],
        ['Motocicletas', 631100.64, np.nan, np.nan, np.nan, 631100.64],
        ['MB/MXB', np.nan, 34931.95, np.nan, np.nan, 34931.95],
        ['Total', 10458214., 1126110., 543713., 11134., 12139170.]]), 
        columns=list(dtype.keys()))
    for k, v in dtype.items():
        df_raw[k] = df_raw[k].astype(v)
    df_raw.set_index('Tipo de vehiculo', drop=True, inplace=True)
    # Here's how I will classify these different categories (this is roughly 
    # based on these https://afdc.energy.gov/data/10380)
    # Autos Particulares ---> Passenger
    # Camionetas SUV ---> Passenger
    # Taxis ---> Passenger
    # Combis y vagonetas (translation: passenger vans) ---> Light duty
    # Microbuses ---> Buses
    # Pick up ---> Light duty
    # Vehículos ≤ 3.8 t ---> Light duty
    # Tractocamiones (translation: tractor trailers) ---> Heavy duty
    # Autobuses ---> Buses
    # Vehículos > 3.8 t ---> Heavy duty
    # Motocicletas ---> Other
    # MB/MXB (translation: metrobus) ---> Buses
    # Form DataFrame by adding these categories together with units of 
    # volume per year (m3/year). Note that NaNs are omitted or replaced 
    # with zeros
    df = pd.DataFrame(np.array([
        # Passenger
        [(df_raw.loc['Autos Particulares']['Gasolina']+ # Gasoline
          df_raw.loc['Camionetas SUV']['Gasolina']+
          df_raw.loc['Taxis']['Gasolina']), 
         (df_raw.loc['Autos Particulares']['Diesel']+ # Diesel
          df_raw.loc['Camionetas SUV']['Diesel']+
          df_raw.loc['Taxis']['Diesel']),          
         (df_raw.loc['Autos Particulares']['GLP':'GN'].sum()+ # Other
          df_raw.loc['Camionetas SUV']['GLP':'GN'].sum()+
          df_raw.loc['Taxis']['GLP':'GN'].sum())
         ],
        [ # Light-duty
        (df_raw.loc['Combis y vagonetas']['Gasolina']+ # Gasolina
         df_raw.loc['Pick up']['Gasolina']+
         df_raw.loc['Vehículos ≤ 3.8 t']['Gasolina']), 
        (df_raw.loc['Combis y vagonetas']['Diesel']+ # Diesel
         df_raw.loc['Pick up']['Diesel']+
         df_raw.loc['Vehículos ≤ 3.8 t']['Diesel']), 
        (df_raw.loc['Combis y vagonetas']['GLP':'GN'].sum()+ # Other
         df_raw.loc['Pick up']['GLP':'GN'].sum()+
         df_raw.loc['Vehículos ≤ 3.8 t']['GLP':'GN'].sum())
        ],
        [ # Heavy-duty
        (df_raw.loc['Tractocamiones']['Gasolina']+ # Gasoline
         df_raw.loc['Vehículos > 3.8 t']['Gasolina']), 
        (df_raw.loc['Tractocamiones']['Diesel']+ # Diesel
         df_raw.loc['Vehículos > 3.8 t']['Diesel']), 
        (df_raw.loc['Tractocamiones']['GLP':'GN'].sum()+ # Other
         df_raw.loc['Vehículos > 3.8 t']['GLP':'GN'].sum())               
        ],
        [ # Buses
        (df_raw.loc['Microbuses']['Gasolina']+ # Gasoline
         df_raw.loc['Autobuses']['Gasolina']), 
        (df_raw.loc['Microbuses']['Diesel']+ # Diesel
         df_raw.loc['Autobuses']['Diesel']+
         df_raw.loc['MB/MXB']['Diesel']), 
        (df_raw.loc['Microbuses']['GLP':'GN'].sum()+ # Other
         df_raw.loc['Autobuses']['GLP':'GN'].sum()+
         df_raw.loc['MB/MXB']['GLP':'GN'].sum())
        ],
        [ # Other
        df_raw.loc['Motocicletas']['Gasolina'], # Gasoline
        0.0, # Diesel
        df_raw.loc['Motocicletas']['GLP':'GN'].sum() # Other
        ]]),
        index=['Passenger', 'Light duty', 'Heavy duty', 'Bus', 'Other'], 
        columns=['Gasoline', 'Diesel', 'Other'])
    return df

def read_milan():
    """Milan fuel data are given in "Fuel_Data.xlsx." Since we are looking 
    at the entire metropolitan region, we use the "Milan - whole city" columns
    and consider the 2020 values. Note that fuel data are derived from fuel 
    statistical data, traffic counts, and modelling evaluations. 
    
    Parameters
    ----------
    None

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Fuel mix in the Milan metropolitan area from 2020 for major 
        transportation and fuel sectors, units of percent  
    """
    import numpy as np
    import pandas as pd
    passenger_gas = 46.6
    passenger_diesel = 44.1
    passenger_other = 7.8+1.5
    light_gas = 10.7
    light_diesel = 77.2
    light_other = 5.0+7.1
    heavy_gas = 0.6
    heavy_diesel = 99.4
    heavy_other = 0
    bus_gas = 0.
    bus_diesel = 100.
    bus_other = 0.
    # Other in the Milan data is motorbikes; n.b., two stroke fuel is 
    # classified as gas
    other_gas = 2.8+97.2
    other_diesel = 0.
    other_other = 0.
    # Form DataFrame
    df = pd.DataFrame(np.array([
        [passenger_gas, passenger_diesel, passenger_other], # Passenger
        [light_gas, light_diesel, light_other], # Light-duty
        [heavy_gas, heavy_diesel, heavy_other], # Heavy-duty
        [bus_gas, bus_diesel, bus_other], # Bus
        [other_gas, other_diesel, other_other] # Other
        ]),
        index=['Passenger', 'Light duty', 'Heavy duty', 'Bus', 'Other'], 
        columns=['Gasoline', 'Diesel', 'Other'])    
    return df

def read_losangeles():
    """Standardize 2019 fuel consumption data from Los Angels from 
    "2019 Fuel Consumption.xlsx." Note since the standard output is in 
    volume/time, the "Fuel Consumption gal/yr" column was used for this. 
    On-road fuel consumption data is obtained by multiplying the vehicle miles 
    travelled (VMT) per year for each vehicle category and fuel type by the 
    corresponding average fuel efficiency. These fuel consumption estimates 
    are for our Scope 1 emissions (100% internal-internal trips + 50% of 
    internal-external/external-internal trips). Note that to be consistent with 
    the standardized format, "light duty" was considered as passenger and 
    "Light-Heavy Duty Truck" was considered as light duty.
    
    Parameters
    ----------
    None

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Fuel mix in the Los Angeles metropolitan area from 2020 for major 
        transportation and fuel sectors, units of gallons/yr
    """
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(np.array([
        [455743503.90, 1829166.14, 0.], # Passenger
        [4565663.66, 993770.26, 0.], # Light-duty
        [3859835.34+70067.00, 8414190.95+42868490.84, 1098768.27], # Heavy-duty
        [589645., 67801., 14642707.], # Bus
        [0., 0., 0.] # Other
        ]),
        index=['Passenger', 'Light duty', 'Heavy duty', 'Bus', 'Other'], 
        columns=['Gasoline', 'Diesel', 'Other'])
    return df

def read_auckland():
    """Data are from "fuel use model output_2016. It is unclear what the 
    differences are between the first and second tables, since both are labeled
    2016 and have identical row and column headers (their values are similar 
    as well, and differences between the two tables are very nuanced). The 
    "FC [L/day]" from the second table is what is compiled into this table. 
    Note that I considered "light commericial" as light-duty. 
     
    Parameters
    ----------
    None

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Fuel mix in the Auckland metropolitan area from 2016 for major 
        transportation and fuel sectors, units of L/day
    """
    import numpy as np
    import pandas as pd
    passenger_gas = 2233115.
    passenger_diesel = 203925.
    passenger_other = 10741. # Note that the second value is hybrid 
    # and electric
    light_gas = 134114.
    light_diesel = 477715.
    heavy_gas = 0.
    heavy_diesel = 58718.+50863+14571.+27736.+104553.+137565.+210317.
    bus_gas = 0.
    bus_diesel = 21917.
    df = pd.DataFrame(np.array([
        [passenger_gas, passenger_diesel, passenger_other], # Passenger
        [light_gas, light_diesel, 0.], # Light-duty
        [heavy_gas, heavy_diesel, 0.], # Heavy-duty
        [bus_gas, bus_diesel, 0.], # Bus
        [0., 0., 0.] # Other
        ]),
        index=['Passenger', 'Light duty', 'Heavy duty', 'Bus', 'Other'], 
        columns=['Gasoline', 'Diesel', 'Other'])    
    return df

def read_berlin():
    """Information is taken from the "fleet_composition_HBEFA_Berlin_2018" 
    and notes compiled by Andreas Kerschbaumer. The sheet contains Berlin’s
    mean vehicle fleet for the year 2018. The classification is taken from the 
    HBEFA-handbook (www.hbefa.net). 

    Note from a 23 November 2020 email from Andreas explains that column C
    (Antriebsart_B_D_HBEFA) denotes the engine and therefore fuel type. 
    D --> diesel
    B --> gasoline
    CNG --> compressed natural gas
    LPG --> liquified petroleum gas
    E85 --> up to 85%v/v ethanol blending in gasoline for so-called flexi-fuel
        vehicles
    Elektro --> electric
    Hybrid --> either hybrid gasoline, hybrid LGB, or hybrid diesel
    Wasserstoff --> hydrogen
    
    Pkw means passanger cars, LNF stands for light duty vehicles, LBus for 
    urban busses, RBus for coaches, SNF for heavy duty vehicles.
    
    Parameters
    ----------
    None

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Fuel mix in the Berlin metropolitan area from 2018 for major 
        transportation and fuel sectors, units of %.    
    """
    import numpy as np
    import pandas as pd
    # For passenger vehicles using gas, find intersection of PKW (column 2) and
    # B (column 3)
    passenger_gas = np.sum([0.0004,0.0004,0.0017,0.0021,0.0284,0.0752,0.0507,
        0.0017,0.0034,0.0030,0.0030,0.1499,0.0949,0.2371,0.3983,0.3626,0.8768,
        1.2622,0.3437,1.9156,9.5859,1.7451,9.3819,9.4669,0.5155,4.1114,10.9401,
        0.7789,6.1641,0.2333,0.0258,0.2328])
    # Intersection of PKW and D
    passenger_diesel = np.sum([0.0073,0.0043,0.0030,0.0009,0.0047,0.0013,
        0.0378,0.0335,0.0516,0.0090,0.0223,0.2384,0.2741,0.0709,0.6212,0.6865,
        0.1645,0.8876,1.7631,0.0335,1.5427,1.9608,0.1856,3.3226,7.6845,0.0494,
        4.3709,11.0273,0.1559,0.4464])
    # Intersection of PKW and E85, Elektro, Hybrid, LPG, Wasserstoff
    passenger_other = np.sum([0.0073,0.0172,0.2750,0.0004,0.0133,0.0352,0.1022,
        0.0228,0.0391,1.0702,0.0021,0.0735,0.2982,1.4744,0.0047,0.0146,0.2964,
        0.0580,0.0189,0.0176,0.0009,0.0133,0.0666,0.1143,0.7089,0.1383,0.0473,
        0.0021])
    # For light duty vehicles (LNF) 
    light_gas = np.sum([0.0055,0.0305,0.0360,0.0277,0.1081,0.0277,0.0222,
        0.1081,0.0139,0.0333,0.6928,0.2217,0.0388,0.2965,0.5043,0.0277,0.7953,
        0.8147,0.0720,0.0055])
    light_diesel = np.sum([0.0028,0.0055,0.0194,0.0222,0.0942,0.0277,0.0693,
        0.2023,0.1081,0.1607,0.0055,0.0028,0.0554,0.5681,0.2328,0.7371,0.6789,
        0.5542,1.8539,3.8961,2.8625,7.0302,0.9560,0.4849,3.9377,12.6417,6.3901,
        24.4548,10.3444,3.9848,13.7113,0.0055,0.0055,0.0111])
    light_other = 0.
    # SNF stands for heavy study vehicles
    heavy_gas = 0.
    heavy_diesel = np.sum([0.0400,0.0667,0.0133,0.0667,0.0133,0.1867,0.1200,
        0.1334,0.6536,0.1067,0.0534,0.0133,0.0934,0.3468,0.0934,0.2001,0.7069,
        0.1200,0.0133,0.2534,1.0137,0.0534,0.3068,0.4935,0.0534,0.0133,0.5069,
        1.4006,0.7870,1.6673,0.0400,0.0534,0.0934,1.3205,4.1750,0.0267,0.2668,
        0.1734,4.5485,7.6697,0.7736,0.5736,0.4135,6.4159,11.4979,1.6940,
        0.5335,13.2987,12.4983,2.2276,2.2142,0.7603,5.3355,9.9373,0.0267,
        0.0267,0.0400,0.2801,2.5877,0.8003,0.0934,0.0133])
    heavy_other = 0.
    # LBus and RBus for bus. (Note that all buses are diesel. Since the 
    # buses are divided into two categories, adding these values would give
    # 200%. So just indicate 100% for this)
    bus_gas = 0.
    bus_diesel = 100.
    bus_other = 0.
    df = pd.DataFrame(np.array([
        [passenger_gas, passenger_diesel, passenger_other], # Passenger
        [light_gas, light_diesel, light_other], # Light-duty
        [heavy_gas, heavy_diesel, heavy_other], # Heavy-duty
        [bus_gas, bus_diesel, bus_other], # Bus
        [0., 0., 0.] # Other
        ]),
        index=['Passenger', 'Light duty', 'Heavy duty', 'Bus', 'Other'], 
        columns=['Gasoline', 'Diesel', 'Other'])    
    return df

def read_london():
    """Road traffic fuel consumption comes from the 
    "LEGGI_2018_Transport_Update_TfL" and, specifically, from the 03c Data
    Transport sheet (Table 3.3.6). These data represent a broad scaling of the 
    2016 fuel consumption using 2016-2018 road traffic growth in Central/Inner/
    Outer London. The data was provided at borough level for all main vehicle 
    types.

    Parameters
    ----------
    None

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Fuel mix for the City of London and its 32 bouroughs from 2016 (scaled
        to 2018), units of liters.  
    """
    import numpy as np
    import pandas as pd
    df = pd.read_excel(DIR_FUEL+'london/'+'LEGGI_2018_Transport_Update_TfL'+
        '.xlsx', sheet_name='03c Data_Transport', header=0)
    # Strip off only Table 3.3.6
    df = df.iloc[255:288]
    # Drop unneeded columns
    df = df.loc[:,'Unnamed: 1':'Unnamed: 17']
    # Rename columns
    df.rename(columns={'Unnamed: 1':'LA name', 'Unnamed: 2':'LA code', 
        'Unnamed: 4':'Motorcycle (petrol)', 'Unnamed: 5':'Taxi (diesel)', 
        'Unnamed: 6':'Car (petrol)', 'Unnamed: 7':'Car (diesel)', 
        'Unnamed: 8':'LGV (petrol)', 'Unnamed: 9':'LGV (diesel)', 
        'Unnamed: 10':'Bus (diesel)', 'Unnamed: 11':'Coach (diesel)',
        'Unnamed: 12':'Rigid (diesel)', 'Unnamed: 13':'Artic (diesel)', 
        'Unnamed: 14':'Total diesel', 'Unnamed: 15':'Total petrol',
        'Unnamed: 16':'Total litres', 'Unnamed: 17':'kWh'}, inplace=True)
    # Passenger; sum of cars and taxis
    passenger_gas = df['Car (petrol)'].sum()
    passenger_diesel = df['Car (diesel)'].sum()+df['Taxi (diesel)'].sum()
    passenger_other = 0.
    # Light duty; no "Light duty" contribution is given.
    light_gas = 0.
    light_diesel = 0.
    light_other = 0.
    # Treat LGV + Rigid + Artic as "Heavy duty"; The UK government refers to 
    # large goods vehicles as LGV. The term articulated lorry ("artic" in Excel 
    # file) refers to the combination of a tractor and a trailer (in the U.S., 
    # this is called a semi-trailer truck, "tractor-trailer" or "semi-truck"). 
    heavy_gas = df['LGV (petrol)'].sum()
    heavy_diesel = (df['LGV (diesel)'].sum()+df['Rigid (diesel)'].sum()+
        df['Artic (diesel)'].sum())
    heavy_other = 0.
    # Bus + Coach as "Bus" in standardized output DataFrame. 
    # Update 25 Jan 2021: As per an email with C40 contacts in London, the data 
    # from the LEGGI provide the fuel and CO2 emissions for TFL buses and 
    # includes any electric buses – they are already taken into account 
    # via the calculations as they are part of the fleet assumptions. There are
    # LPG buses in the TFL fleet.
    # The proportion of VKM for TFL buses is provided below for 2019 and 2020 – 
    # you can see change in Diesel as TfL prepare for LEZ expansion. All 
    # information that TfL hold for buses has already been shared.
    # 2019
    # ++++++++++++++
    # Central London 
    # --------------
    # Diesel = 12%
    # Petrol = 0%
    # Hybrid = 78.0%
    # ZEV = 10%
    #
    # Inner London 
    # --------------
    # Diesel = 46%
    # Petrol = 0%
    # Hybrid = 50%
    # ZEV = 4%
    #
    # Outer London 
    # --------------
    # Diesel = 86%
    # Petrol = 0%
    # Hybrid = 13%
    # ZEV = 1%
    #        
    # 2020
    # ++++++++++++++
    # Central London 
    # --------------
    # Diesel = 0%
    # Petrol = 0%
    # Hybrid = 94%
    # ZEV = 6%
    #
    # Inner London 
    # --------------
    # Diesel = 37%
    # Petrol = 0%
    # Hybrid = 57%
    # ZEV = 6%
    #
    # Outer London 
    # --------------
    # Diesel = 83%
    # Petrol = 0%
    # Hybrid = 16%
    # ZEV = 1%
    # Replace bus information from LEGGI inventory with an average of
    # Inner + Outer + Central London for 2020:
    # Diesel = (83% + 37% + 0%)/3
    # Petrol = 0%
    # Hybrid = (94% + 57% + 16%)/3
    # ZEV = (6% + 6% + 1%)/3
    bus_gas = 0.
    bus_diesel = 40.
    bus_other = (94+57+16)/3. + (6+6+1)/3.
    # # Old way
    #bus_gas = 0.
    #bus_diesel = df['Bus (diesel)'].sum()+df['Coach (diesel)'].sum()
    #bus_other = 0.
    # Only motorcycles are given for the "Other" category 
    other_gas = df['Motorcycle (petrol)'].sum()
    other_diesel = 0.
    other_other = 0.
    df = pd.DataFrame(np.array([
        [passenger_gas, passenger_diesel, passenger_other], # Passenger
        [light_gas, light_diesel, light_other], # Light-duty
        [heavy_gas, heavy_diesel, heavy_other], # Heavy-duty
        [bus_gas, bus_diesel, bus_other], # Bus
        [other_gas, other_diesel, other_other] # Other
        ]),
        index=['Passenger', 'Light duty', 'Heavy duty', 'Bus', 'Other'], 
        columns=['Gasoline', 'Diesel', 'Other'])
    return df

def read_santiago():
    """Santiago provided a greenhouse gas inventory that is very hard to 
    interpret ("CIRIS_RM_2016_20190701_v5.3_upload.xlsx") but provided a 
    more condensed version with fuel classifications for 2016-2019 
    ("Distribución Vehículos RM_INE"). We use the most recent values from 2019
    for the following vehicle types: 
    Buses --> buses (CLASSIFY AS BUS)
    Camiones --> trucks (CLASSIFY AS HEAVY-DUTY)
    Motocicletas --> motorcycles (CLASSIFY AS OTHER)
    Otros --> other (CLASSIFY AS OTHER)
    Taxis y alquiler --> taxis and rental (CLASSIFY AS PASSENGER)
    Vehículos comerciales --> commericial vehicles (CLASSIFY AS LIGHT-DUTY; 
                              google "Vehiculos comerciales Santiago" for 
                              infomation)
    Vehículos particulares --> private vehicles (CLASSIFY AS PASSENGER)
    
    The fuel types are interesting, mainly becuase there is a category for 
    "gas" and one for "gasolineros." The "gas" category, however, is very 
    minimal compared to the "gasolineros" column (sums for 2016-2019 indiciate
    that gas is 0.2% of gasolineros). 
    
    As of 11 December 2020, I am unclear what the units of the Santiago data
    are.
    
    Parameters
    ----------
    None

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Fuel mix for Santiago; unknown units
    """
    import numpy as np
    import pandas as pd
    # Passenger (sum of Vehículos particulares and taxis y alquiler)
    passenger_gas = 1.+1414011+3742+31781
    passenger_diesel = 4992.+106574
    passenger_other = 544.+87.
    # Light-duty (vehiculos comerciales)
    light_gas = 27.+173300
    light_diesel = 272814.
    light_other = 94.
    # Heavy-duty
    heavy_gas = 849.
    heavy_diesel = 61149.
    heavy_other = 10.
    # Bus
    bus_gas = 5.+341
    bus_diesel = 17088.
    bus_other = 426.
    # Other (sum of motocicletas and otros)
    other_gas = 102968.+249+232
    other_diesel = 19.+5603
    other_other = 693.+84
    df = pd.DataFrame(np.array([
        [passenger_gas, passenger_diesel, passenger_other], # Passenger
        [light_gas, light_diesel, light_other], # Light-duty
        [heavy_gas, heavy_diesel, heavy_other], # Heavy-duty
        [bus_gas, bus_diesel, bus_other], # Bus
        [other_gas, other_diesel, other_other] # Other
        ]),
        index=['Passenger', 'Light duty', 'Heavy duty', 'Bus', 'Other'], 
        columns=['Gasoline', 'Diesel', 'Other'])    
    return df 
    