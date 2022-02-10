#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:45:30 2021

@author: ghkerr
"""
DIR = '/Users/ghkerr/GW/'
DIR_MODEL = DIR+'data/GEOSCF/'
DIR_MOBILITY = DIR+'data/mobility/'
DIR_FIG = DIR+'mobility/figs/'
DIR_AQ = DIR+'data/aq/'
DIR_TYPEFACE = '/Users/ghkerr/Library/Fonts/'
DIR_EMISSIONS = DIR+'data/emissions/'

agorange = '#CD5733'
agtan = '#F4E7C5'
agnavy = '#678096'
agblue = '#ACC2CF'
agpuke = '#979461'
agred = '#A12A19'

# Load custom font
import numpy as np
import sys
sys.path.append('/Users/ghkerr/GW/mobility/')
import readc40aq
import readc40mobility
if 'mpl' not in sys.modules:
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(
            fname=DIR_TYPEFACE+'cmunbmr.ttf')
    matplotlib.rcParams['font.family'] = prop.get_name()
    prop = matplotlib.font_manager.FontProperties(
        fname=DIR_TYPEFACE+'cmunbbx.ttf')
    matplotlib.rcParams['mathtext.bf'] = prop.get_name()
    prop = matplotlib.font_manager.FontProperties(
        fname=DIR_TYPEFACE+'cmunbmr.ttf')
    matplotlib.rcParams['mathtext.it'] = prop.get_name()
    matplotlib.rcParams['mathtext.rm'] = prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False

# Functions
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def draw_brace(ax, yspan, xx, color):
    """Draws an annotated brace on the axes."""
    ymin, ymax = yspan
    yspan = ymax - ymin
    ax_ymin, ax_ymax = ax.get_ylim()
    yax_span = ax_ymax - ax_ymin
    xmin, xmax = ax.get_xlim()
    xspan = xmax - xmin
    resolution = int(yspan/yax_span*100)*2+1 # guaranteed uneven
    beta = 300./yax_span # the higher this is, the smaller the radius
    y = np.linspace(ymin, ymax, resolution)
    y_half = y[:int(resolution/2)+1]
    x_half_brace = (1/(1.+np.exp(-beta*(y_half-y_half[0])))
                    + 1/(1.+np.exp(-beta*(y_half-y_half[-1]))))
    x = np.concatenate((x_half_brace, x_half_brace[-2::-1]))
    x = xx + (.05*x - .01)*xspan # adjust vertical position
    ax.plot(-x, y, color=color, lw=1, clip_on=False)
    return

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, 
        b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def fit_func(p, t):
    """First order linear regression for calculating total least squares
    """
    return p[0] * t + p[1]

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z

def geo_idx(dd, dd_array):
    """Function searches for nearest decimal degree in an array of decimal 
    degrees and returns the index. np.argmin returns the indices of minimum 
    value along an axis. 
    
    Parameters
    ----------
    dd : int
        Latitude or longitude whose index in dd_array is being sought
    dd_array : numpy.ndarray 
        1D array of latitude or longitude 
    
    Returns
    -------
    geo_idx : int
        Index of latitude or longitude in dd_array that is closest in value to 
        dd
    """
    import numpy as np   
    from scipy import stats
    geo_idx = (np.abs(dd_array - dd)).argmin()
    # If distance from closest cell to intended value is "far", raise error 
    if np.abs(dd_array[geo_idx] - dd) > 1.:
        return 
    return geo_idx

# From https://stackoverflow.com/questions/16107884/
# power-law-with-a-constant-factor-using-curve-fitting; note that the
# initial conditions for a and b came from the first y and x values 
# (assuming values are in order), c can be estimated as in the accepted 
# answer, and the estimate for d came from the final y values which are ~0. 
# If you're having trouble with initial conditions, this can be a good 
# starting point (see https://stackoverflow.com/questions/21420792/
# exponential-curve-fitting-in-scipy for more information)
def func(x, a, c, d):
    return a*np.exp(-c*x)+d

def build_focuscities(noC40):
    """Build table of focus cities for this study.

    Parameters
    ----------
    noc40 : bool
        If True, C40 cities outside of the European Union will be dropped from
        the DataFrame.

    Returns
    -------
    focuscities : pandas.core.frame.DataFrame
        Table containing city names, countries, population, share of passenger 
        vehicles using diesel fuel, and lockdown start and end dates.
    """    
    import numpy as np
    import pandas as pd
    from itertools import groupby
    def ranges(lst):
        pos = (j - i for i, j in enumerate(lst))
        t = 0
        for i, els in groupby(pos):
            l = len(list(els))
            el = lst[t]
            t += l
            yield range(el, el+l)
    # City | Country | Passenger diesel %
    # focuscities = [['Amsterdam', 'Netherlands', 14.0], # ACEA 
    #     ['Athens', 'Greece', 8.1], # ACEA 
    #     ['Auckland C40', 'New Zealand', 8.3], # C40 partnership
    #     ['Barcelona', 'Spain', 	58.7], # ACEA
    #     ['Berlin C40', 'Germany', 31.7], # ACEA
    #     ['Bucharest', 'Romania', 43.3], # ACEA
    #     ['Budapest', 'Hungary', 31.5], # ACEA
    #     ['Cologne', 'Germany', 	31.7], # ACEA
    #     ['Copenhagen', 'Denmark', 30.9], # ACEA 
    #     ['Dusseldorf', 'Germany', 31.7], # ACEA
    #     ['Frankfurt', 'Germany', 31.7], # ACEA
    #     ['Hamburg', 'Germany', 31.7], # ACEA 
    #     ['Helsinki', 'Finland', 27.9], # ACEA 
    #     ['Krakow', 'Poland', 31.6], # ACEA 
    #     ['Lodz', 'Poland', 31.6], # ACEA
    #     ['London C40', 'United Kingdom', 39.0], # ACEA 
    #     ['Los Angeles C40', 'United States', 0.4], # C40 partnership
    #     ['Madrid', 'Spain', 58.7], # ACEA
    #     ['Marseille', 'France', 58.9], # ACEA
    #     ['Mexico City C40', 'Mexico', 0.2], # C40 partnership
    #     ['Milan C40', 'Italy', 44.2], # ACEA 
    #     ['Munich', 'Germany', 31.7], # ACEA
    #     ['Naples', 'Italy', 44.2], # ACEA 
    #     ['Palermo', 'Italy', 44.2], # ACEA
    #     ['Paris', 'France', 58.9], # ACEA
    #     ['Prague', 'Czechia', 35.9], # ACEA
    #     ['Rome', 'Italy', 44.2], # ACEA
    #     ['Rotterdam', 'Netherlands', 14.0], # ACEA
    #     ['Santiago C40', 'Chile', 7.1], # C40 partnership
    #     ['Saragossa', 'Spain', 58.7], # ACEA
    #     ['Seville', 'Spain', 58.7], # ACEA
    #     ['Sofia', 'Bulgaria',  43.1], # ICCT partnership
    #     ['Stockholm', 'Sweden', 35.5], # ACEA
    #     ['Stuttgart', 'Germany', 31.7], # ACEA
    #     ['Turin', 'Italy', 44.2], # ACEA
    #     ['Valencia', 'Spain', 58.7], # ACEA
    #     ['Vienna', 'Austria', 55.0], # ACEA
    #     ['Vilnius', 'Lithuania', 69.2], # ACEA
    #     ['Warsaw', 'Poland', 31.6], # ACEA
    #     ['Wroclaw', 'Poland', 31.6], # ACEA
    #     ['Zagreb', 'Croatia', 52.4], # ACEA 
    #     ]
    focuscities = [
        ['Athens', 'Greece', 8.1, 16.0], # ACEA 
        ['Auckland C40', 'New Zealand', 8.3, 999], # C40 partnership
        ['Barcelona', 'Spain', 	58.7, 12.7], # ACEA
        ['Berlin', 'Germany', 31.7, 9.6], # ACEA
        ['Budapest', 'Hungary', 31.5, 13.5], # ACEA
        ['Copenhagen', 'Denmark', 30.9, 8.8], # ACEA 
        ['Helsinki', 'Finland', 27.9, 12.2], # ACEA 
        ['Krakow', 'Poland', 31.6, 41.1], # ACEA 
        ['London C40', 'United Kingdom', 39.0, 8.0], # ACEA 
        ['Los Angeles C40', 'United States', 0.4, 999], # C40 partnership
        ['Madrid', 'Spain', 58.7, 12.7], # ACEA
        ['Marseille', 'France', 58.9, 10.2], # ACEA
        ['Mexico City C40', 'Mexico', 0.2, 999], # C40 partnership
        ['Milan', 'Italy', 44.2, 11.4], # ACEA 
        ['Munich', 'Germany', 31.7, 9.6], # ACEA
        ['Paris', 'France', 58.9, 10.2], # ACEA
        ['Prague', 'Czechia', 35.9, 14.9], # ACEA
        ['Rome', 'Italy', 44.2, 11.4], # ACEA
        ['Rotterdam', 'Netherlands', 14.0, 11.0], # ACEA
        ['Santiago C40', 'Chile', 7.1, 999], # C40 partnership
        ['Sofia', 'Bulgaria',  43.1, 22], # ICCT partnership
        ['Stockholm', 'Sweden', 35.5, 10.], # ACEA
        ['Vienna', 'Austria', 55.0, 8.3], # ACEA
        ['Vilnius', 'Lithuania', 69.2, 16.8], # ACEA
        ['Warsaw', 'Poland', 31.6, 41.1], # ACEA
        ['Zagreb', 'Croatia', 52.4, 14.6], # ACEA 
        ]    
    focuscities = pd.DataFrame(focuscities, columns=['City', 'Country', 
        'Diesel share', 'Age'])    
    # Open stay-at-home data 
    sah = pd.read_csv(DIR_MOBILITY+'stay-at-home-covid.csv')
    # Meaning of column stay_home_requirements
    # 0 = No measures
    # 1 = Recommended not to leave the house
    # 2 = Required to not leave the house with exceptions for daily 
    #     exercise, grocery shopping, and ‘essential’ trips
    # 3 = Required to not leave the house with minimal exceptions (e.g. 
    #     allowed to leave only once every few days, or only one person 
    #     can leave at a time, etc.)
    # Add stay at home requirements to city information DataFrame
    focuscities['start'] = np.nan
    focuscities['end'] = np.nan    
    focuscities['startreq'] = np.nan
    focuscities['endreq'] = np.nan
    # Loop through cities and determine the dates of lockdowns
    for index, row in focuscities.iterrows():
        country = row['Country']
        sah_country = sah.loc[sah['Entity']==country]
        # Restrict to measuring period
        sah_country = sah_country.loc[sah_country['Day']<'2020-07-01']
        # Occurrences of ALL recommended or required stay-at-home measures
        where1 = np.where(sah_country['stay_home_requirements']==1.)[0]
        where2 = np.where(sah_country['stay_home_requirements']==2.)[0]
        where3 = np.where(sah_country['stay_home_requirements']==3.)[0]
        start = min(np.hstack([where1,where2,where3]))
        startdate = sah_country['Day'].values[start]
        end = max(np.hstack([where1,where2,where3]))
        enddate = sah_country['Day'].values[end]
        if pd.to_datetime(enddate) > pd.to_datetime('2020-06-30'):
            enddate = '2020-06-30'
        focuscities.loc[index, 'start'] = startdate
        focuscities.loc[index, 'end'] = enddate
        # Occurrences of REQUIRED stay-at-home measures
        if (np.shape(where2)[0]!=0) or (np.shape(where3)[0]!=0):
            start = min(np.hstack([where2,where3]))
            startdate = sah_country['Day'].values[start]
            end = max(np.hstack([where2,where3]))
            enddate = sah_country['Day'].values[end]
            if pd.to_datetime(enddate) > pd.to_datetime('2020-06-30'):
                enddate = '2020-06-30'
            focuscities.loc[index, 'startreq'] = startdate
            focuscities.loc[index, 'endreq'] = enddate
        else: 
            focuscities.loc[index, 'startreq'] = np.nan
            focuscities.loc[index, 'endreq'] = np.nan            
    if noC40==True: 
        focuscities.drop(focuscities.loc[focuscities['City'].isin([
            'Auckland C40', 'Los Angeles C40', 'Mexico City C40',
            'Santiago C40'])].index, inplace=True)
    return focuscities

def fig2(): 
    """
    """
    import numpy as np
    import pandas as pd
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    # Number of features to plot
    nft = 10
    # Reduce data to first NFEATURES number of features, and sort by median
    medians = shaps_concat.median()
    medians = pd.DataFrame(medians).sort_values(by=0,ascending=False)
    features = list(medians.index[:nft])
    medians_lon = shaps_lon.median()
    medians_lon = pd.DataFrame(medians_lon).sort_values(by=0,ascending=False)
    features_lon = list(medians_lon.index[:nft])
    # Plotting
    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot2grid((1,3),(0,0), colspan=2)
    ax2 = plt.subplot2grid((1,3),(0,2), colspan=1)
    # Pluck of observations and GEOS-CF for London
    bcm_lon = bcm.loc[bcm['City']=='London C40'].set_index('Date')
    raw_lon = raw.loc[raw['City']=='London C40'].set_index('Date')
    # Print performance metrics for paper
    idx = np.isfinite(raw_lon['NO2'][:'2019-12-31'].values) & \
        np.isfinite(bcm_lon['observed'][:'2019-12-31'].values)
    print('r for London (GEOS-CF, observed), 2019')
    print(np.corrcoef(raw_lon['NO2'][:'2019-12-31'].values[idx],
        bcm_lon['observed'][:'2019-12-31'].values[idx])[0,1])
    print('MFB for London (GEOS-CF, observed), 2019')
    print((2*(np.nansum(raw_lon['NO2'][:'2019-12-31'].values-
        bcm_lon['observed'][:'2019-12-31'].values)/np.nansum(
        raw_lon['NO2'][:'2019-12-31'].values+
        bcm_lon['observed'][:'2019-12-31'].values))))
    print('r for London (GEOS-CF, BAU), 2019')
    print(np.corrcoef(bcm_lon['predicted'][:'2019-12-31'].values[idx],
        bcm_lon['observed'][:'2019-12-31'].values[idx])[0,1])
    print('MFB for London (GEOS-CF, BAU), 2019')
    print((2*(np.nansum(bcm_lon['predicted'][:'2019-12-31'].values-
        bcm_lon['observed'][:'2019-12-31'].values)/np.nansum(
        bcm_lon['predicted'][:'2019-12-31'].values+
        bcm_lon['observed'][:'2019-12-31'].values))))
    bcm_lon = bcm_lon.resample('1D').mean().rolling(window=7,
        min_periods=1).mean()
    raw_lon = raw_lon.resample('1D').mean().rolling(
          window=7,min_periods=1).mean()
    ax1.plot(raw_lon['NO2'], ls='--', color='darkgrey', label='GEOS-CF')
    ax1.plot(bcm_lon['predicted'], '--k', label='Business-as-usual')
    ax1.plot(bcm_lon['observed'], '-k', label='Observed')
    # Fill red for positive difference between , blue for negative difference
    y1positive=(bcm_lon['observed']-bcm_lon['predicted'])>0
    y1negative=(bcm_lon['observed']-bcm_lon['predicted'])<=0
    # ax.fill_between(dat.index, dat['predicted'],
    #     dat['observed'], where=y1positive, color='red', alpha=0.5)
    ax1.fill_between(bcm_lon.index, bcm_lon['predicted'], 
        bcm_lon['observed'], where=y1negative, color=agnavy, 
        interpolate=True)
    # Draw shaded gradient region for lockdown 
    ld_lon = focuscities.loc[focuscities['City']=='London C40']
    ldstart = pd.to_datetime(ld_lon['start'].values[0])
    ldend = pd.to_datetime(ld_lon['end'].values[0])
    x = pd.date_range(ldstart,ldend)
    y = range(37)
    z = [[z] * len(x) for z in range(len(y))]
    num_bars = 100 # More bars = smoother gradient
    cmap = plt.get_cmap('Reds')
    new_cmap = truncate_colormap(cmap, 0.0, 0.35)
    ax1.contourf(x, y, z, num_bars, cmap=new_cmap, zorder=0)
    ax1.text(x[int(x.shape[0]/2.)-2], 4, 'LOCKDOWN', ha='center', 
        rotation=0, va='center', fontsize=12) 
    # Aesthetics
    ax1.set_ylim([0,36])
    ax1.set_yticks(np.linspace(0,36,7))    
    # Hide the right and top spines
    for side in ['right', 'top']:
        ax1.spines[side].set_visible(False)
    ax1.set_ylabel('NO$_{2}$ [ppbv]')
    ax1.set_xlim(['2019-01-01','2020-06-30'])
    ax1.set_xticks(['2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01',
        '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01', 
        '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', 
        '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', 
        '2020-05-01', '2020-06-01']) 
    ax1.set_xticklabels(['Jan\n2019', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan\n2020', 'Feb', 
        'Mar', 'Apr', 'May', 'Jun'], fontsize=9) 
    # Legend
    ax1.legend(frameon=False, ncol=3, loc=3, fontsize=10)
    # Replace variables names with  something more 
    # publication-worthy
    vardict = {'O3':'O$_{\mathregular{3}}$', 
        'NO2':'NO$_{\mathregular{2}}$',
        'Volume':'Traffic',
        'ZPBL':'Boundary layer height',
        'V':'Northward wind',
        'U':'Eastward wind',
        'RH':'Relative humidity',
        'Q':'Specific humidity',
        'T':'Temperature', 
        'CO':'CO',
        'PM25':'PM$_{\mathregular{2.5}}$',
        'PS':'Surface pressure'}
    colorlon = '#E69F00'
    colorall = '#56B4E9'
    # Plot boxplot and labels
    for i,var in enumerate(features):
        bplot = ax2.boxplot(shaps_concat[var].values, positions=[i], 
            widths=[0.5], patch_artist=True, whis=(10,90), vert=False, 
            showfliers=False)
        ax2.text(np.percentile(shaps_concat[var].values, 90)+0.03, i, 
            vardict[var], ha='left', va='center', fontsize=9)
        for item in ['boxes', 'whiskers', 'fliers']:
            plt.setp(bplot[item], color=agorange)  
        for item in ['medians', 'caps']:
            plt.setp(bplot[item], color='w')  
    vardict = {'O3':'O$_{\mathregular{3}}$', 
        'NO2':'NO$_{\mathregular{2}}$',
        'Volume':'Traffic',
        'ZPBL':'Boundary layer height',
        'V':'Northward\nwind',
        'U':'Eastward wind',
        'RH':'Relative humidity',
        'Q':'Specific humidity',
        'T':'Temperature', 
        'CO':'CO',
        'PM25':'PM$_{\mathregular{2.5}}$',
        'PS':'Surface pressure'}      
    for i, var in enumerate(features_lon):
        bplot_lon = ax2.boxplot(shaps_lon[var], positions=[i+12], 
            widths=[0.5], patch_artist=True, whis=(10,90), vert=False, 
            showfliers=False)
        ax2.text(np.percentile(shaps_lon[var].values, 90)+0.03, i+12, 
            vardict[var], ha='left', va='center', fontsize=9, clip_on=False)    
        for item in ['boxes', 'whiskers', 'fliers']:
            plt.setp(bplot_lon[item], color=agnavy)  
        for item in ['medians', 'caps']:
            plt.setp(bplot_lon[item], color='w')      
    draw_brace(ax2, (0, 9), 0., agorange)
    draw_brace(ax2, (12, 21), 0., agnavy)
    ax2.text(-0.3, 16.5, 'London', rotation=90, ha='center', va='center', 
        color=agnavy)
    ax2.text(-0.3, 4.5, 'All', rotation=90, ha='center', va='center', 
        color=agorange)
    ax2.set_xlim([0,2.5])
    ax2.set_xticks([0,0.5,1.,1.5,2.,2.5])
    # ax2.set_xticklabels(['0.0','','0.5','','1.0','','1.5'], fontsize=9)
    ax2.set_ylim([-0.5,22.])
    ax2.set_yticks([])
    ax2.invert_yaxis()
    for side in ['left', 'right', 'top']:
        ax2.spines[side].set_visible(False)
    plt.subplots_adjust(left=0.05, right=0.92)
    ax1.set_title('(a) London', x=0.1, y=1.02, fontsize=12)
    ax2.set_title('(b) Absolute SHAP values', y=1.02, loc='left', fontsize=12)
    plt.savefig(DIR_FIG+'fig2_eu.png', dpi=1000)
    return

def fig3(focuscities, bcm):
    """

    Parameters
    ----------
    focuscities : pandas.core.frame.DataFrame
        Table containing city names, countries, population, share of passenger 
        vehicles using diesel fuel, and lockdown start and end dates.
    bcm : pandas.core.frame.DataFrame
        XGBoost-predicted concentrations and the observed concentrations 
        (and the bias) for focus cities

    Returns
    -------
    None
    """
    from sklearn.metrics import mean_squared_error
    import math
    from scipy.optimize import curve_fit
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from dateutil.relativedelta import relativedelta
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import scipy.odr    
    from scipy import stats
    dno2, no2, diesel, cities = [], [], [], []
    for index, row in focuscities.iterrows():
        city = row['City']
        print(city)
        bcm_city = bcm.loc[bcm['City']==city]
        bcm_city.set_index('Date', inplace=True)
        # Figure out lockdown dates
        ldstart = focuscities.loc[focuscities['City']==city]['start'].values[0]
        ldstart = pd.to_datetime(ldstart)
        ldend = focuscities.loc[focuscities['City']==city]['end'].values[0]
        ldend = pd.to_datetime(ldend)
        # Calculate percentage change in NO2 during lockdown periods
        before = np.nanmean(bcm_city.loc[ldstart-relativedelta(years=1):
            ldend-relativedelta(years=1)]['predicted'])
        after = np.abs(np.nanmean(bcm_city.loc[ldstart:ldend]['anomaly']))
        pchange = -after/before*100
        # Save output
        dno2.append(pchange)
        no2.append(bcm_city['observed']['2019-01-01':'2019-12-31'].mean())
        diesel.append(focuscities.loc[focuscities['City']==city]['Diesel share'].values[0])
        cities.append(focuscities.loc[focuscities['City']==city]['City'].values[0])
    diesel = np.array(diesel)
    cities = np.array(cities)
    no2 = np.array(no2)
    dno2 = np.array(dno2)
    # Create custom colormap
    cmap = plt.get_cmap("pink_r")
    cmap = truncate_colormap(cmap, 0.4, 0.9)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
        cmaplist, cmap.N)
    cmap.set_over(color='k')
    bounds = np.linspace(8, 20, 7)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # Plotting
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,1),(0,0))
    mb = ax1.scatter(diesel, dno2, c=no2, s=18, cmap=cmap, norm=norm, 
        clip_on=False)
    ax1.set_xlabel(r'Diesel-powered passenger vehicle share [%]')
    ax1.set_ylabel(r'$\mathregular{\Delta}$ NO$_{\mathregular{2}}$ [%]')
    # Calculate slope with total least squares (ODR)
    lincoeff = np.poly1d(np.polyfit(diesel, dno2, 1))
    ax1.plot(np.unique(diesel), np.poly1d(np.polyfit(diesel, dno2, 1)
        )(np.unique(diesel)), 'black', ls='dashed', lw=1, zorder=0, 
        label='Linear fit (y=ax+b)\na=-0.53, b=-2.21')
    ax1.legend(frameon=False, bbox_to_anchor=(0.4, 0.42))
    axins1 = inset_axes(ax1, width='40%', height='5%', loc='lower left', 
        bbox_to_anchor=(0.02, 0.04, 1, 1), bbox_transform=ax1.transAxes,
        borderpad=0)
    fig.colorbar(mb, cax=axins1, orientation="horizontal", extend='both', 
        label='NO$_{\mathregular{2}}$ [ppbv]')
    axins1.xaxis.set_ticks_position('top')
    axins1.xaxis.set_label_position('top')
    ax1.tick_params(labelsize=9)
    ax1.set_xlim([-1,71])
    ax1.set_ylim([-65,5])
    # Calculate r, RMSE for linear vs. power fit
    dno2_sorted = sort_list(dno2, diesel)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        dno2, diesel)
    print('Equation coefficients should match the following:')
    print('Linear: ', lincoeff)
    print('Linear correlation between diesel and dNO2=', r_value)
    print('p-value=',p_value)
    print('RMSE for linear fit...', math.sqrt(mean_squared_error(dno2_sorted, 
        np.poly1d(np.polyfit(diesel, dno2, 1)
        )(diesel))))
    # for i, txt in enumerate(cities):
    #     if txt == 'Santiago C40':
    #         txt = 'Santiago'
    #     elif txt == 'Mexico City C40':
    #         txt = 'Mexico City'
    #     elif txt == 'Los Angeles C40':
    #         txt = 'Los Angeles'
    #     elif txt == 'Berlin C40':
    #         txt = 'Berlin'
    #     elif txt == 'Milan C40':
    #         txt = 'Milan'
    #     elif txt == 'London C40':
    #         txt = 'London'
    #     elif txt == 'Auckland C40':
    #         txt = 'Auckland'        
    #     ax1.annotate(txt, (diesel[i]+1, dno2[i]+1), fontsize=9)
    # plt.savefig(DIR_FIG+'fig3_citynames_eu.png', dpi=1000)    
    # plt.savefig(DIR_FIG+'fig3_eu.png', dpi=1000)
    return  

def fig4(bcm):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from dateutil.relativedelta import relativedelta
    from scipy import stats
    dno2, no2, diesel, cities = [], [], [], []
    for index, row in focuscities.iterrows():
        city = row['City']
        print(city)
        bcm_city = bcm.loc[bcm['City']==city]
        bcm_city.set_index('Date', inplace=True)
        # Figure out lockdown dates
        ldstart = focuscities.loc[focuscities['City']==city]['start'].values[0]
        ldstart = pd.to_datetime(ldstart)
        ldend = focuscities.loc[focuscities['City']==city]['end'].values[0]
        ldend = pd.to_datetime(ldend)
        # Calculate percentage change in NO2 during lockdown periods
        before = np.nanmean(bcm_city.loc[ldstart-relativedelta(years=1):
            ldend-relativedelta(years=1)]['predicted'])
        after = np.abs(np.nanmean(bcm_city.loc[ldstart:ldend]['anomaly']))
        pchange = -after/before*100
        # Save output
        dno2.append(pchange)
        no2.append(bcm_city['observed']['2019-01-01':'2019-12-31'].mean())
        diesel.append(focuscities.loc[focuscities['City']==city]['Diesel share'].values[0])
        cities.append(focuscities.loc[focuscities['City']==city]['City'].values[0])
    diesel = np.array(diesel)
    cities = np.array(cities)
    no2 = np.array(no2)
    dno2 = np.array(dno2)
    # Open GAINS ECLIPSE
    eclipse, coa2 = [], []
    for city in cities:
        # Find corresponding country to city
        ccountry = focuscities.loc[focuscities['City']==city]['Country'].values[0]
        if ccountry == 'United Kingdom':
            ccountry = 'United-Kingdom'
        eclipse_country = pd.read_csv(DIR_EMISSIONS+'gains/ECLIPSE_%s.csv'
            %ccountry, sep=',', skiprows=7, engine='python')
        coa2_country = pd.read_csv(DIR_EMISSIONS+'gains/COA2_%s.csv'%ccountry,  
            sep=',', skiprows=7, engine='python')    
        # Sample ratio of NOx from light-duty vehicles to total NOx emissions
        # for 2020
        eclipse_light = eclipse_country.loc[eclipse_country['[kt/yr]']==
            'Light duty vehicles']['2020']
        eclipse_total = eclipse_country.loc[eclipse_country['[kt/yr]']=='Sum']['2020']
        eclipse.append(float(eclipse_light.values[0])/
            float(eclipse_total.values[0]))
        coa2_light = coa2_country.loc[coa2_country['[kt/yr]']==
            'Light duty vehicles']['2020']
        coa2_total = coa2_country.loc[coa2_country['[kt/yr]']=='Sum']['2020']
        coa2.append(float(coa2_light.values[0])/float(coa2_total.values[0]))
    eclipse = np.array(eclipse)
    coa2 = np.array(coa2)
    # Plotting
    fig = plt.figure(figsize=(6,6))
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax3 = plt.subplot2grid((2,2),(1,0))
    ax4 = plt.subplot2grid((2,2),(1,1))
    # Hypothesis plot
    # Define numbers of generated data points and bins per axis.
    np.random.seed(7)
    x = np.linspace(0, 100, 18) # 1000 values between 0 and 100
    delta = np.random.uniform(0, 15, x.size)
    y = 0.4*x + delta -2
    y[y<13.] = np.nan
    ymask = np.isfinite(y)
    ax1.hist2d(x[ymask], y[ymask], bins=4, cmap='Greys', alpha=0.3)
    ax1.scatter(x, y, color='k', clip_on=False)
    ax1.text(0.05, 0.05, 'Smaller $\mathregular{\Delta}$NO$_{\mathregular{2}}$,'+ 
        '\nSmaller light-duty\n vehicle contribution ', 
        transform=ax1.transAxes, fontsize=7)
    # ax1.text(0.05, 0.95, 'Larger $\mathregular{\Delta}$NO$_{\mathregular{2}}$,'+ 
    #     '\nSmaller light-duty\n vehicle contribution', 
    #     transform=ax1.transAxes, fontsize=7, va='top')
    ax1.text(0.95, 0.97, 'Larger $\mathregular{\Delta}$NO$_{\mathregular{2}}$,'+ 
        '\nLarger light-duty\n vehicle contribution', clip_on=False,
        transform=ax1.transAxes, fontsize=7, va='top', ha='right')
    # ax1.text(0.95, 0.05, 'Smaller $\mathregular{\Delta}$NO$_{\mathregular{2}}$,'+ 
    #     '\nLarger light-duty\n vehicle contribution ', 
    #     transform=ax1.transAxes, fontsize=7, va='bottom', ha='right')
    ax1.spines['left'].set_position('zero')
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xlim([0,110])
    ax1.set_ylim([0,60])
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Make arrows
    ax1.plot((1), (0), ls="", marker=">", ms=7, color="k",
        transform=ax1.get_yaxis_transform(), clip_on=False)
    ax1.plot((0), (0.975), ls="", marker="^", ms=7, color="k",
        transform=ax1.get_xaxis_transform(), clip_on=False)
    ax1.set_xlabel('Increasing importance of light-\nduty vehicle NO$_{x}$', 
        loc='left')
    ax1.set_ylabel('Increasing NO$_{\mathregular{2}}$ reduction', loc='bottom')
    
    # Define bins for diesel shares
    p0 = np.nanpercentile(diesel, 0)
    p33 = np.nanpercentile(diesel, 33.3)
    p66 = np.nanpercentile(diesel, 66.6)
    p100 = np.nanpercentile(diesel, 100)
    # Small diesel shares plot
    wherebin = np.where((diesel>p0)&(diesel<=p33))[0]
    dno2bin = dno2[wherebin]
    eclipsebin = eclipse[wherebin]
    coa2bin = coa2[wherebin]
    ax2.plot(eclipsebin, -1*dno2bin, color=agred, marker='o', clip_on=False, 
        ls='none', zorder=100)
    ax2.plot(coa2bin, -1*dno2bin, color=agnavy, marker='o', clip_on=False, 
        ls='none', zorder=100)
    ax2.set_ylim([0, 25])
    ax2.set_xlim([0.05, 0.25])
    ax2.set_xticks(np.linspace(0.05, 0.25, 5))
    ax2.set_xticklabels(['5', '', '15', '', '25'])
    # Add statistical information 
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        -1*np.hstack([dno2bin, dno2bin]), 100*np.hstack([eclipsebin, coa2bin]))
    txtstr = 'r = %.2f\np-value = %.2f'%(r_value, p_value)
    ax2.text(0.02, 0.98, txtstr, color='darkgrey', 
        transform=ax2.transAxes, va='top', ha='left')
    # Medium diesel shares plot
    wherebin = np.where((diesel>p33)&(diesel<=p66))[0]
    dno2bin = dno2[wherebin]
    eclipsebin = eclipse[wherebin]
    coa2bin = coa2[wherebin]
    ax3.plot(eclipsebin, -1*dno2bin, color=agred, marker='o', ls='none')
    ax3.plot(coa2bin, -1*dno2bin, color=agnavy, marker='o', ls='none')
    ax3.set_ylim([0, 40])
    ax3.set_xticks([0,10,20,30,40])
    ax3.set_xlim([0.06, 0.34])
    ax3.set_xticks(np.linspace(0.06, 0.34, 5))
    ax3.set_xticklabels(['6','','20','','40'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        -1*np.hstack([dno2bin, dno2bin]), 100*np.hstack([eclipsebin, coa2bin]))
    txtstr = 'r = %.2f\np-value = %.2f'%(r_value, p_value)
    ax3.text(0.02, 0.98, txtstr, color='darkgrey', 
        transform=ax3.transAxes, va='top', ha='left')
    # Large diesel shares plot
    wherebin = np.where((diesel>p66)&(diesel<=p100))[0]
    dno2bin = dno2[wherebin]
    eclipsebin = eclipse[wherebin]
    coa2bin = coa2[wherebin]
    ax4.plot(eclipsebin, -1*dno2bin, color=agred, marker='o', ls='none', 
        label='ECLIPSE')
    ax4.plot(coa2bin, -1*dno2bin, color=agnavy, marker='o', ls='none',
        label='Clean Air Outlook')
    ax4.set_ylim([0, 64])
    ax4.set_yticks(np.linspace(0,64,5))
    ax4.set_xlim([0.1, 0.4])
    ax4.set_xticks(np.linspace(0.1, 0.4, 5))
    ax4.set_xticklabels(['10','','25','','40'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        -1*np.hstack([dno2bin, dno2bin]), 100*np.hstack([eclipsebin, coa2bin]))
    txtstr = 'r = %.2f\np-value = %.2f'%(r_value, p_value)
    ax4.text(0.02, 0.98, txtstr, color='darkgrey', 
        transform=ax4.transAxes, va='top', ha='left')
    ax4.legend(loc=(-0.95, -0.43), frameon=False, ncol=2)                                                                  
    for ax in [ax2, ax3, ax4]:
        ax.set_ylabel('NO$_{\mathregular{2}}$ reduction [%]', loc='bottom')
        ax.set_xlabel('NO$_{x,\:\mathregular{Light\u2212duty}}$ : '+\
            'NO$_{x,\:\mathregular{Total}}$  [%]', loc='left')    
    # Aesthetics
    ax1.set_title('(a) Hypothesis', loc='left')
    ax2.set_title('(b) Small diesel shares', loc='left')
    ax3.set_title('(c) Medium diesel shares', loc='left')
    ax4.set_title('(d) Large diesel shares', loc='left')
    plt.subplots_adjust(wspace=0.45, hspace=0.45, bottom=0.15, top=0.95)
    plt.savefig(DIR_FIG+'fig4_eu.pdf', dpi=1000)
    return

def figS1(bcm):
    """    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    # Options that work and look halfway decent are: 'GoogleTiles',
    # 'GoogleWTS', 'QuadtreeTiles', 'Stamen'
    request = cimgt.Stamen()    
    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=5, ncols=3, 
        subplot_kw={'projection':request.crs}) 
    axes = np.hstack(axes)
    # Loop through cities for which we've built BCM/BAU concentrations
    citiesunique = np.unique(bcm.City)
    citiesunique = np.sort(citiesunique)
    for i, city in enumerate(citiesunique[:15]):
        citycoords = stationcoords.loc[stationcoords['City']==city]
        # Plot station coordinates   
        axes[i].plot(citycoords['Longitude'], citycoords['Latitude'], 
            marker='o', lw=0, markersize=3, color=agred, 
            transform=ccrs.PlateCarree())
        # Aesthetics
        if ' C40' in city:
            city = city[:-4]
            axes[i].set_title(city, loc='left')
        else:
            axes[i].set_title(city, loc='left') 
        # Syntax is (x0, x1, y0, y1)
        extent = [citycoords['Longitude'].min()-0.2, 
            citycoords['Longitude'].max()+0.2, 
            citycoords['Latitude'].min()-0.2, 
            citycoords['Latitude'].max()+0.2]
        axes[i].set_extent(extent)
        axes[i].add_image(request, 11)
        axes[i].set_adjustable('datalim')
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)    
    plt.savefig(DIR_FIG+'figS1a_eu.pdf', dpi=600)
    plt.show()
    # For the rest of the focus cities
    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=5, ncols=3, 
        subplot_kw={'projection':request.crs}) 
    axes = np.hstack(axes)
    for i, city in enumerate(citiesunique[15:]):
        citycoords = stationcoords.loc[stationcoords['City']==city]
        axes[i].plot(citycoords['Longitude'], citycoords['Latitude'], 
            marker='o', lw=0, markersize=3, color=agred, 
            transform=ccrs.PlateCarree())
        # Add marker for Vilnius heating plant
        if city=='Vilnius':
            axes[i].plot(25.157202, 54.667939, marker='s', lw=0, 
                markersize=5, color=agnavy, 
                transform=ccrs.PlateCarree())        
        if ' C40' in city:
            city = city[:-4]
            axes[i].set_title(city, loc='left')
        else:
            axes[i].set_title(city, loc='left') 
        extent = [citycoords['Longitude'].min()-0.2, 
            citycoords['Longitude'].max()+0.2, 
            citycoords['Latitude'].min()-0.2, 
            citycoords['Latitude'].max()+0.2]
        if city=='Vilnius':        
            extent = [citycoords['Longitude'].min()-0.1, 
                citycoords['Longitude'].max()+0.1, 
                citycoords['Latitude'].min()-0.05, 
                citycoords['Latitude'].max()+0.05]    
        request = cimgt.Stamen()
        axes[i].set_extent(extent)
        axes[i].add_image(request, 11)
        axes[i].set_adjustable('datalim')
    # Remove blank axes
    for i in np.arange(7,15,1):
        axes[i].axis('off')     
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    plt.savefig(DIR_FIG+'figS1b_eu.pdf', dpi=600)
    plt.show()
    return

def figS2():
    """
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    # Open lockdown information 
    sah = pd.read_csv(DIR_MOBILITY+'stay-at-home-covid.csv')
    ysax1, ysax2 = [], []
    pcolorax1, pcolorax2 = [], []
    citiesax1, citiesax2 = [], []
    cmap = plt.get_cmap('Reds')
    # Plot mobility curves 
    fig = plt.figure(figsize=(8.5,6))
    ax1 = plt.subplot2grid((2,2),(0,0), rowspan=2)
    ax2 = plt.subplot2grid((2,2),(0,1), rowspan=2)
    cityloc = 0 
    cities = []
    # Loop through city DataFrame and find mobility dataset corresponding
    # to each city and plot 
    for index, row in focuscities.iterrows():
        country = row['Country']
        city = row['City']
        cities.append(city)
        # SELECT APPLE MOBILITY
        mobility_city = mobility[mobility['city'].str.contains(city)]
        mobility_city.set_index('time', inplace=True)    
        # Select 15 January - 30 June 2020
        mobility_city = mobility_city['2020-01-15':'2020-06-30']
        filler = np.empty(shape=len(mobility_city))
        filler[:] = cityloc+2
        # SELECT STAY_HOME_REQUIREMENTS 
        sah_country = sah.loc[sah['Entity']==country]
        sah_country.set_index('Day', inplace=True)
        sah_country.index = pd.to_datetime(sah_country.index)
        sah_country = sah_country['2020-01-15':'2020-06-30']
        sah_country = sah_country['stay_home_requirements']
        # Fill missing values
        idx = pd.date_range('2020-01-15','2020-06-30')
        sah_country = sah_country.reindex(idx, fill_value=np.nan)
        if cityloc < 11.*2:
            ax1.plot(mobility_city.index, (filler+
                mobility_city['Volume'].values/100.*-1), color='k')
            pcolorax1.append(sah_country.values)
            ysax1.append(cityloc+1)
            if ' C40' in city:
                city = city[:-4]
            citiesax1.append(city)
        else: 
            ax2.plot(mobility_city.index, (filler+
                mobility_city['Volume'].values/100.*-1), color='k')
            pcolorax2.append(sah_country.values)    
            ysax2.append(cityloc+1) 
            if city=='Milan C40':
                city='Milan'
            elif city=='Santiago C40':
                city='Santiago'
            citiesax2.append(city)        
        cityloc = cityloc+2
    mb = ax1.pcolormesh(sah_country.index, ysax1, np.stack(pcolorax1), 
        cmap=cmap, shading='auto', vmin=0, vmax=3)
    mb = ax2.pcolormesh(sah_country.index, ysax2, np.stack(pcolorax2), 
        cmap=cmap, shading='auto', vmin=0, vmax=3)
    # Aesthetics
    for ax in [ax1, ax2]:
        ax.set_xlim(['2020-01-15','2020-06-30'])
        ax.set_xticks(['2020-01-15', '2020-02-01', '2020-02-15', '2020-03-01', 
            '2020-03-15', '2020-04-01', '2020-04-15', '2020-05-01', 
            '2020-05-15', '2020-06-01', '2020-06-15']) 
        ax.set_xticklabels(['', 'Feb\n2020', '', 'Mar', '', 'Apr', '', 'May', 
            '', 'Jun', ''], fontsize=8) 
    plt.subplots_adjust(wspace=0.3, top=0.95, right=0.95)
    ax1.set_ylim([0,ysax1[-1]+1])
    ax1.set_yticks([x for x in ysax1])
    ax1.set_yticklabels(citiesax1)
    ax2.set_ylim([ysax2[0]-1,ysax2[-1]+1])
    ax2.set_yticks([x for x in ysax2])
    ax2.set_yticklabels(citiesax2)
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    legend_elements = [
        Patch(facecolor=cmap(0.), edgecolor='k', label='No measures'),
        Patch(facecolor=cmap(0.33), edgecolor='k', label='Recommended not '+\
            'to leave the house'),
        Patch(facecolor=cmap(0.66), edgecolor='k', label='Required to not '+\
             'leave the house with exceptions for daily exercise, grocery '+\
             'shopping, and essential trips'),
        Patch(facecolor=cmap(1.), edgecolor='k', label='Required to not '+\
             'leave the house with minimal exceptions (e.g., allowed to '+\
             'leave only once every few days,\nor only one person can '+
             'leave at a time, etc.)')]
    plt.subplots_adjust(bottom=0.23, top=0.98)
    ax1.legend(handles=legend_elements, loc='center', ncol=1, 
        bbox_to_anchor=(1.15,-0.2), frameon=False, fontsize=10)
    plt.savefig(DIR_FIG+'figS2_eu.png', dpi=1000)
    return

def figS3():
    """
    Adapted from https://matplotlib.org/2.0.2/examples/statistics/
    customized_violin_demo.html
    """
    fig = plt.figure(figsize=(8,4))
    ax1 = plt.subplot2grid((1,3),(0,0))
    ax2 = plt.subplot2grid((1,3),(0,1))
    ax3 = plt.subplot2grid((1,3),(0,2))
    # For MFB
    ax1.hlines(0, xmin=0, xmax=4, color='darkgrey', ls='--', lw=2, zorder=0)
    parts1 = ax1.violinplot([mfborig, mfbtrain, mfbvalid], showmeans=False, 
        showmedians=False, showextrema=False)
    # Add medians
    ax1.hlines(np.median(mfborig), 0.95, 1.05, color='w', linestyle='-', lw=2,
        zorder=10)
    ax1.hlines(np.median(mfbvalid), 2.95, 3.05, color='w', linestyle='-', lw=2,
        zorder=10)   
    for i, mfbi in enumerate([mfborig, mfbtrain, mfbvalid]):
        q1, medians, q3 = np.percentile(mfbi, [25, 50, 75])
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, np.sort(mfbi)[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, np.sort(mfbi)[0], q1)
        # if i!=1:    
        #     ax1.scatter(i+1, medians, marker='o', color='white', s=30, zorder=3)
        ax1.vlines(i+1, q1, q3, color='k', linestyle='-', lw=5)    
        ax1.vlines(i+1, lower_adjacent_value, upper_adjacent_value, color='k', 
            linestyle='-', lw=1)
    # For r
    ax2.hlines(1, xmin=0, xmax=4, color='darkgrey', ls='--', lw=2, zorder=0)
    parts2 = ax2.violinplot([rorig, rtrain, rvalid], showmeans=False, 
        showmedians=False, showextrema=False)
    ax2.hlines(np.median(rorig), 0.95, 1.05, color='w', linestyle='-', lw=2,
        zorder=10)
    ax2.hlines(np.median(rtrain), 1.95, 2.05, color='w', linestyle='-', lw=1,
        zorder=10)    
    ax2.hlines(np.median(rvalid), 2.95, 3.05, color='w', linestyle='-', lw=2,
        zorder=10)    
    for i, r in enumerate([rorig, rtrain, rvalid]):
        q1, medians, q3 = np.percentile(r, [25, 50, 75])
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, np.sort(r)[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, np.sort(r)[0], q1)
        ax2.vlines(i+1, q1, q3, color='k', linestyle='-', lw=5)    
        ax2.vlines(i+1, lower_adjacent_value, upper_adjacent_value, color='k', linestyle='-', lw=1)
    # F2F
    ax3.hlines(1, xmin=0, xmax=4, color='darkgrey', ls='--', lw=2, zorder=0)
    parts3 = ax3.violinplot([fac2orig, fac2train, fac2valid], showmeans=False, 
        showmedians=False, showextrema=False)
    ax3.hlines(np.median(fac2orig), 0.95, 1.05, color='w', linestyle='-', lw=2,
        zorder=10)    
    ax3.hlines(np.median(fac2valid), 2.95, 3.05, color='w', linestyle='-', lw=2,
        zorder=10)    
    for i, fac2 in enumerate([fac2orig, fac2train, fac2valid]):
        q1, medians, q3 = np.percentile(fac2, [25, 50, 75])
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, np.sort(fac2)[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, np.sort(fac2)[0], q1)
        ax3.vlines(i+1, q1, q3, color='k', linestyle='-', lw=5)    
        ax3.vlines(i+1, lower_adjacent_value, upper_adjacent_value, color='k', linestyle='-', lw=1)
    for parts in [parts1, parts2, parts3]:
        for i, pc in enumerate(parts['bodies']):
            if i==0:
                pc.set_facecolor(agnavy)
            elif i==1:
                pc.set_facecolor(agorange)
            else:
                pc.set_facecolor(agpuke)
            pc.set_edgecolor('black')
            pc.set_alpha(1)
    # Aesthetics
    ax1.set_title('(a) Mean fraction bias', loc='left')
    ax2.set_title('(b) Correlation coefficient', loc='left')
    ax3.set_title('(c) Factor-of-2 fraction', loc='left')
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(['GEOS-CF', '\nTraining', '\nTesting'], fontsize=9)
        ax.text(0.43, -0.055, 'Business-as-usual', transform=ax.transAxes, 
            fontsize=9)
        ax.tick_params(labelsize=9)
        ax.set_xlim([0.5, 3.5])    
    ax1.set_ylim([-1.5, 0.5])
    ax2.set_ylim([-0.2, 1.01])
    ax3.set_ylim([0., 1.01])
    plt.subplots_adjust(left=0.07, right=0.95, wspace=0.26)    
    plt.savefig(DIR_FIG+'figS3_eu.png', dpi=1000)
    return 

def figS4(): 
    """
    """
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=8, ncols=2) 
    axes = np.hstack(axes)
    i = 0
    for city in focuscities['City'][:17]:
        if city=='London C40':
            pass   
        else:
            ax = axes[i]
            # Pluck of observations and GEOS-CF for city
            bcmc = bcm.loc[bcm['City']==city].set_index('Date')
            rawc = raw.loc[raw['City']==city].set_index('Date')
            # Print performance metrics for paper
            idx = np.isfinite(rawc['NO2'][:'2019-12-31'].values) & \
                np.isfinite(bcmc['observed'][:'2019-12-31'].values)
            # print('r for %s (GEOS-CF, observed), 2019'%city)
            # print(np.corrcoef(rawc['NO2'][:'2019-12-31'].values[idx],
            #     bcmc['observed'][:'2019-12-31'].values[idx])[0,1])
            # print('MFB for %s (GEOS-CF, observed), 2019'%city)
            # print((2*(np.nansum(rawc['NO2'][:'2019-12-31'].values-
            #     bcmc['observed'][:'2019-12-31'].values)/np.nansum(
            #     rawc['NO2'][:'2019-12-31'].values+
            #     bcmc['observed'][:'2019-12-31'].values))))
            # print('r for %s (GEOS-CF, BAU), 2019' %city)
            # print(np.corrcoef(bcmc['predicted'][:'2019-12-31'].values[idx],
            #     bcmc['observed'][:'2019-12-31'].values[idx])[0,1])
            # print('MFB for %s (GEOS-CF, BAU), 2019'%city)
            # print((2*(np.nansum(bcmc['predicted'][:'2019-12-31'].values-
            #     bcmc['observed'][:'2019-12-31'].values)/np.nansum(
            #     bcmc['predicted'][:'2019-12-31'].values+
            #     bcmc['observed'][:'2019-12-31'].values))))
            bcmc = bcmc.resample('1D').mean().rolling(window=7,
                min_periods=1).mean()
            rawc = rawc.resample('1D').mean().rolling(
                  window=7,min_periods=1).mean()
            ax.plot(rawc['NO2'], ls='--', color='darkgrey', label='GEOS-CF')
            ax.plot(bcmc['predicted'], '--k', label='Business-as-usual')
            ax.plot(bcmc['observed'], '-k', label='Observed')
            # Fill blue for negative difference between timeseries (generally in 
            # spring 2020)
            y1positive=(bcmc['observed']-bcmc['predicted'])>0
            y1negative=(bcmc['observed']-bcmc['predicted'])<=0
            ax.fill_between(bcmc.index, bcmc['predicted'], 
                bcmc['observed'], where=y1negative, color=agnavy, 
                interpolate=True)
            # Determine the maximum of the observed, predicted, and BAU 
            # concentrations
            maxlim = np.nanmax([rawc['NO2'], bcmc['predicted'],bcmc['observed']])
            maxlim = int(math.ceil(maxlim/5))*5
            # Draw shaded gradient region for lockdown 
            ldc = focuscities.loc[focuscities['City']==city]
            ldstart = pd.to_datetime(ldc['start'].values[0])
            ldend = pd.to_datetime(ldc['end'].values[0])
            x = pd.date_range(ldstart,ldend)
            y = range(maxlim+2)
            z = [[z] * len(x) for z in range(len(y))]
            num_bars = 100 # More bars = smoother gradient
            cmap = plt.get_cmap('Reds')
            new_cmap = truncate_colormap(cmap, 0.0, 0.35)
            ax.contourf(x, y, z, num_bars, cmap=new_cmap, zorder=0)
            # Aesthetics
            # Hide the right and top spines
            for side in ['right', 'top']:
                ax.spines[side].set_visible(False)
            if ' C40' in city:
                city = city[:-4]
                ax.set_title(city, loc='left')
            else:
                ax.set_title(city, loc='left')    
            if (i % 2) == 0:
                ax.set_ylabel('NO$_{2}$ [ppbv]')
            ax.set_xlim(['2019-01-01','2020-06-30'])
            ax.set_xticks(['2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01',
                '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01', 
                '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', 
                '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', 
                '2020-05-01', '2020-06-01']) 
            ax.set_xticklabels([])
            ax.set_ylim([0, maxlim])
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            if (i==14) or (i==15): 
                ax.set_xticklabels(['Jan\n2019', '', 'Mar', '', 'May', '', 
                    'Jul', '', 'Sep', '', 'Nov', '', 'Jan\n2020', '', 
                    'Mar', '', 'May', ''], fontsize=9)
            # Legend
            if i==15:
                ax.legend(frameon=False, ncol=3, loc=3, fontsize=14, 
                    bbox_to_anchor=(-1.2,-0.95))
            i=i+1
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.97)
    plt.savefig(DIR_FIG+'figS4a_eu.png', dpi=1000)
    plt.show()
    # Rest of cities
    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=8, ncols=2) 
    axes = np.hstack(axes)
    for i, city in enumerate(focuscities['City'][17:]):
        ax = axes[i]
        bcmc = bcm.loc[bcm['City']==city].set_index('Date')
        rawc = raw.loc[raw['City']==city].set_index('Date')
        idx = np.isfinite(rawc['NO2'][:'2019-12-31'].values) & \
            np.isfinite(bcmc['observed'][:'2019-12-31'].values)
        # print('r for %s (GEOS-CF, observed), 2019'%city)
        # print(np.corrcoef(rawc['NO2'][:'2019-12-31'].values[idx],
        #     bcmc['observed'][:'2019-12-31'].values[idx])[0,1])
        # print('MFB for %s (GEOS-CF, observed), 2019'%city)
        # print((2*(np.nansum(rawc['NO2'][:'2019-12-31'].values-
        #     bcmc['observed'][:'2019-12-31'].values)/np.nansum(
        #     rawc['NO2'][:'2019-12-31'].values+
        #     bcmc['observed'][:'2019-12-31'].values))))
        # print('r for %s (GEOS-CF, BAU), 2019' %city)
        # print(np.corrcoef(bcmc['predicted'][:'2019-12-31'].values[idx],
        #     bcmc['observed'][:'2019-12-31'].values[idx])[0,1])
        # print('MFB for %s (GEOS-CF, BAU), 2019'%city)
        # print((2*(np.nansum(bcmc['predicted'][:'2019-12-31'].values-
        #     bcmc['observed'][:'2019-12-31'].values)/np.nansum(
        #     bcmc['predicted'][:'2019-12-31'].values+
        #     bcmc['observed'][:'2019-12-31'].values))))
        bcmc = bcmc.resample('1D').mean().rolling(window=7,
            min_periods=1).mean()
        rawc = rawc.resample('1D').mean().rolling(
              window=7,min_periods=1).mean()
        ax.plot(rawc['NO2'], ls='--', color='darkgrey', label='GEOS-CF')
        ax.plot(bcmc['predicted'], '--k', label='Business-as-usual')
        ax.plot(bcmc['observed'], '-k', label='Observed')
        y1positive=(bcmc['observed']-bcmc['predicted'])>0
        y1negative=(bcmc['observed']-bcmc['predicted'])<=0
        ax.fill_between(bcmc.index, bcmc['predicted'], 
            bcmc['observed'], where=y1negative, color=agnavy, 
            interpolate=True)
        maxlim = np.nanmax([rawc['NO2'], bcmc['predicted'],bcmc['observed']])
        maxlim = int(math.ceil(maxlim/5))*5
        ldc = focuscities.loc[focuscities['City']==city]
        ldstart = pd.to_datetime(ldc['start'].values[0])
        ldend = pd.to_datetime(ldc['end'].values[0])
        x = pd.date_range(ldstart,ldend)
        y = range(maxlim+2)
        z = [[z] * len(x) for z in range(len(y))]
        num_bars = 100 # More bars = smoother gradient
        cmap = plt.get_cmap('Reds')
        new_cmap = truncate_colormap(cmap, 0.0, 0.35)
        ax.contourf(x, y, z, num_bars, cmap=new_cmap, zorder=0)
        for side in ['right', 'top']:
            ax.spines[side].set_visible(False)
        if ' C40' in city:
            city = city[:-4]
            ax.set_title(city, loc='left')
        else:
            ax.set_title(city, loc='left')    
        if (i % 2) == 0:
            ax.set_ylabel('NO$_{2}$ [ppbv]')
        ax.set_xlim(['2019-01-01','2020-06-30'])
        ax.set_xticks(['2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01',
            '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01', 
            '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', 
            '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', 
            '2020-05-01', '2020-06-01']) 
        ax.set_xticklabels([])
        ax.set_ylim([0, maxlim])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if (i==3) or (i==4): 
            ax.set_xticklabels(['Jan\n2019', '', 'Mar', '', 'May', '', 
                'Jul', '', 'Sep', '', 'Nov', '', 'Jan\n2020', '', 
                'Mar', '', 'May', ''], fontsize=9)
        # Legend
        if i==8:
            ax.legend(frameon=False, ncol=1, loc=3, fontsize=14, 
                bbox_to_anchor=(1.25,-0.15))
    for i in np.arange(5,16,1):
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.97)
    plt.savefig(DIR_FIG+'figS4b_eu.png', dpi=1000)
    plt.show()
    return 

def figS5(): 
    from dateutil.relativedelta import relativedelta
    import scipy.odr
    dno2, no2, diesel, cities = [], [], [], []
    dno2dow = []
    shapsdow_all, featuresdow_all, featuresdow_lon, shapsdow_lon = [], [], [], []
    bcmdow = pd.DataFrame()
    rawdow = pd.DataFrame()
    # # Loop through cities and build bias-corrected model
    for index, row in focuscities.iterrows():
        city = row['City']    
        gcf_city = modeldow.loc[modeldow['city']==city]
        obs_city = obs.loc[obs['City']==city].copy(deep=True)
        obs_city.loc[obs_city['Concentration']==0,'Concentration']= np.nan
        std = np.nanstd(obs_city['Concentration'])
        mean = np.nanmean(obs_city['Concentration'])
        obs_city.loc[obs_city['Concentration']>mean+(3*std), 'Concentration'] = np.nan
        qaqc = obs_city.loc[(obs_city['Date']>='2019-01-01') & 
            (obs_city['Date']<='2019-12-31')]
        if qaqc.shape[0] >= (365*0.65):
            del gcf_city['city'], obs_city['City']
            merged_train, bias_train, obs_conc_train = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', '2019-12-31')
            merged_full, bias_full, obs_conc_full = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', '2020-06-30')
            (no2diff, shaps, features, ro, fac2o, mfbo, rt, fac2t, mfbt, rv, fac2v, 
                mfbv) = run_xgboost(args, merged_train, bias_train, merged_full, 
                obs_conc_full)
            shapsdow_all.append(shaps)
            if city=='London C40':
                shapsdow_lon.append(shaps)
            bcm_city = no2diff.groupby(['Date']).mean().reset_index()
            bcm_city['City'] = city
            bcmdow = bcmdow.append(bcm_city, ignore_index=True)
            raw_city = merged_full[['Date','NO2']].copy(deep=True)
            raw_city['City'] = city
            rawdow = rawdow.append(raw_city, ignore_index=True)
    for index, row in focuscities.iterrows():
        city = row['City']
        bcm_city = bcm.loc[bcm['City']==city]
        bcm_city.set_index('Date', inplace=True)
        ldstart = focuscities.loc[focuscities['City']==city]['start'].values[0]
        ldstart = pd.to_datetime(ldstart)
        ldend = focuscities.loc[focuscities['City']==city]['end'].values[0]
        ldend = pd.to_datetime(ldend)
        before = np.nanmean(bcm_city.loc[ldstart-relativedelta(years=1):
            ldend-relativedelta(years=1)]['predicted'])
        after = np.abs(np.nanmean(bcm_city.loc[ldstart:ldend]['anomaly']))
        pchange = -after/before*100
        dno2.append(pchange)
        no2.append(bcm_city['observed']['2019-01-01':'2019-12-31'].mean())
        diesel.append(focuscities.loc[focuscities['City']==city]['Diesel share'].values[0])
        cities.append(focuscities.loc[focuscities['City']==city]['City'].values[0])
        bcm_city = bcmdow.loc[bcmdow['City']==city]
        bcm_city.set_index('Date', inplace=True)
        ldstart = focuscities.loc[focuscities['City']==city]['start'].values[0]
        ldstart = pd.to_datetime(ldstart)
        ldend = focuscities.loc[focuscities['City']==city]['end'].values[0]
        ldend = pd.to_datetime(ldend)
        before = np.nanmean(bcm_city.loc[ldstart-relativedelta(years=1):
            ldend-relativedelta(years=1)]['predicted'])
        after = np.abs(np.nanmean(bcm_city.loc[ldstart:ldend]['anomaly']))
        pchange = -after/before*100
        dno2dow.append(pchange)
    dno2 = np.array(dno2)
    dno2dow = np.array(dno2dow)
    # Plotting 
    fig, ax1 = plt.subplots(1)
    ax1.plot(dno2, dno2dow, 'ko', clip_on=False)
    # 1:1 line
    ax1.plot(np.linspace(-70,1,1000), np.linspace(-70,1,1000), color='k', 
        ls='-', lw=0.25, label='Identity line')
    # Line of best fit 
    idx = np.isfinite(dno2) & np.isfinite(dno2dow)
    Model = scipy.odr.Model(fit_func)
    odr = scipy.odr.RealData(dno2[idx], dno2dow[idx])
    odr = scipy.odr.ODR(odr, Model,[np.polyfit(dno2[idx], dno2dow[idx], 1)[1],
        np.polyfit(dno2[idx], dno2dow[idx], 1)[0]], maxit=10000)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        dno2[idx], dno2dow[idx])
    output = odr.run() 
    beta = output.beta
    print(beta)
    ax1.plot(np.linspace(-70, 0, 100), ((slope*np.linspace(-70, 0, 100))+intercept),
        color='black', ls='dashed', lw=1, zorder=0, 
        label='Linear fit (y=ax+b)\na=0.92, b=-6.30')    
    txtstr = 'r = %.2f\np-value = %.2f'%(r_value, p_value)
    ax1.text(-67, -8, txtstr, color='darkgrey', ha='left')
    ax1.set_xlim([-70, 0])
    ax1.set_ylim([-70, 0])
    ax1.set_xticks(np.arange(-70,10,10))
    ax1.set_yticks(np.arange(-70,10,10))
    ax1.legend(frameon=False, loc=9, bbox_to_anchor=(0.5, 1.16), ncol=2)
    ax1.set_xlabel('$\mathregular{\Delta}$ NO$_{\mathregular{2}}$ [%]\n'+
        '(traffic characterized by Apple Mobility Trends\nReports)', loc='left')
    ax1.set_ylabel('$\mathregular{\Delta}$ NO$_{\mathregular{2}}$ [%]\n'+
        '(traffic characterized by day of week)', loc='bottom')
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig(DIR_FIG+'figS5_eu.png', dpi=1000)
    return 

def figS6(obs):
    """
    """    
    from sklearn.metrics import mean_squared_error
    import math
    from scipy.optimize import curve_fit
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from dateutil.relativedelta import relativedelta
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import scipy.odr    
    from scipy import stats
    
    # Add additional C40 cities and calculate city-wide averages
    obs_los = readc40aq.read_losangeles('NO2', '2018-01-01', '2020-12-31')
    coordstemp = obs_los.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Los Angeles C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_los = obs_los.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    obs_mex = readc40aq.read_mexicocity('NO2', '2018-01-01', '2020-12-31')
    coordstemp = obs_mex.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Mexico City C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_mex = obs_mex.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    obs_san = readc40aq.read_santiago('NO2', '2018-01-01', '2020-12-31')
    coordstemp = obs_san.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Santiago C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_san = obs_san.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    obs_ber = readc40aq.read_berlin('NO2', '2019-01-01', '2020-12-31')
    coordstemp = obs_ber.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Berlin C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_ber = obs_ber.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    obs_auc = readc40aq.read_auckland('NO2', '2019-01-01', '2020-12-31')
    coordstemp = obs_auc.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Auckland C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_auc = obs_auc.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    # Combine observations
    obs_withc40 = obs.copy(deep=True)
    obs_withc40 = obs_withc40.append(obs_los, ignore_index=False)
    obs_withc40 = obs_withc40.append(obs_mex, ignore_index=False)
    obs_withc40 = obs_withc40.append(obs_san, ignore_index=False)
    obs_withc40 = obs_withc40.append(obs_auc, ignore_index=False)
    focuscities_withc40 = build_focuscities(False)
    # Recalculate bias-corrected model 
    bcm_withc40 = pd.DataFrame()
    raw_withc40 = pd.DataFrame()
    # # Loop through cities and build bias-corrected model
    for index, row in focuscities_withc40.iterrows():
        city = row['City']    
        print(city)    
        # Select city in model/observational dataset
        gcf_city = model.loc[model['city']==city]
        obs_city = obs_withc40.loc[obs_withc40['City']==city].copy(deep=True)
        # There are some cities (e.g., Brussels) with observations equal to 0 ppb
        # that appear to just be missing data. Change these to NaN!
        obs_city.loc[obs_city['Concentration']==0,'Concentration']= np.nan
        # QA/QC: Require that year 2019 has at least 65% of observations for 
        # a given city; remove observations +/- 3 standard deviations 
        std = np.nanstd(obs_city['Concentration'])
        mean = np.nanmean(obs_city['Concentration'])
        obs_city.loc[obs_city['Concentration']>mean+(3*std), 'Concentration'] = np.nan
        qaqc = obs_city.loc[(obs_city['Date']>='2019-01-01') & 
            (obs_city['Date']<='2019-12-31')]
        if qaqc.shape[0] >= (365*0.65):
            # Remove city column otherwise XGBoost will throw a ValueError (i.e., 
            # DataFrame.dtypes for data must be int, float, bool or categorical.  
            # When categorical type is supplied, DMatrix parameter 
            # `enable_categorical` must be set to `True`.city)    
            del gcf_city['city'], obs_city['City']
            # Run XGBoost for site
            merged_train, bias_train, obs_conc_train = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', '2019-12-31')
            merged_full, bias_full, obs_conc_full = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', '2020-06-30')
            (no2diff, shaps, features, ro, fac2o, mfbo, rt, fac2t, mfbt, rv, fac2v, 
                mfbv) = run_xgboost(args, merged_train, bias_train, merged_full, 
                obs_conc_full)
            # Group data by date and average over all k-fold predictions
            bcm_city = no2diff.groupby(['Date']).mean().reset_index()
            # Add column corresponding to city name for each ID
            bcm_city['City'] = city
            bcm_withc40 = bcm_withc40.append(bcm_city, ignore_index=True)
            # Save off raw (non bias-corrected) observations
            raw_city = merged_full[['Date','NO2']].copy(deep=True)
            raw_city['City'] = city
            raw_withc40 = raw_withc40.append(raw_city, ignore_index=True)
        else:
            print('Skipping %s...'%city)
    dno2, no2, diesel, cities = [], [], [], []
    for index, row in focuscities_withc40.iterrows():
        city = row['City']
        print(city)
        bcm_city = bcm_withc40.loc[bcm_withc40['City']==city]
        bcm_city.set_index('Date', inplace=True)
        before = np.nanmean(bcm_city.loc[(
            pd.to_datetime('2020-03-15')-relativedelta(years=1)):
            (pd.to_datetime('2020-06-15')-relativedelta(years=1))]['predicted'])
        after = np.abs(np.nanmean(bcm_city.loc['2020-03-15':'2020-06-15']['anomaly']))
        pchange = -after/before*100
        # Save output
        dno2.append(pchange)
        no2.append(bcm_city['observed']['2019-01-01':'2019-12-31'].mean())
        diesel.append(focuscities_withc40.loc[focuscities_withc40['City']==city]['Diesel share'].values[0])
        cities.append(focuscities_withc40.loc[focuscities_withc40['City']==city]['City'].values[0])
    diesel = np.array(diesel)
    cities = np.array(cities)
    no2 = np.array(no2)
    dno2 = np.array(dno2)
    # Create custom colormap
    cmap = plt.get_cmap("pink_r")
    cmap = truncate_colormap(cmap, 0.4, 0.9)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
        cmaplist, cmap.N)
    cmap.set_over(color='k')
    bounds = np.linspace(8, 20, 7)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # Plotting
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,1),(0,0))
    mb = ax1.scatter(diesel, dno2, c=no2, s=18, cmap=cmap, norm=norm, 
        clip_on=False)
    ax1.set_xlabel(r'Diesel-powered passenger vehicle share [%]')
    ax1.set_ylabel(r'$\mathregular{\Delta}$ NO$_{\mathregular{2}}$ [%]')
    # Calculate slope with total least squares (ODR)
    idx = np.isfinite(diesel) & np.isfinite(dno2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        diesel[idx], dno2[idx])
    lincoeff = np.poly1d(np.polyfit(diesel[idx], dno2[idx], 1))
    print('Equation coefficients should match the following:')
    print(lincoeff)
    ax1.plot(np.unique(diesel[idx]), np.poly1d(np.polyfit(diesel[idx], 
        dno2[idx], 1))(np.unique(diesel[idx])), 'black', ls='dashed', lw=1, 
        zorder=0, label='Linear fit (y=ax+b)\na=-0.40, b=-8.91')
    # ax1.legend(frameon=False, loc=9, bbox_to_anchor=(0.5, 1.16), ncol=2)
    ax1.legend(frameon=False, bbox_to_anchor=(0.4, 0.42))
    axins1 = inset_axes(ax1, width='40%', height='5%', loc='lower left', 
        bbox_to_anchor=(0.02, 0.04, 1, 1), bbox_transform=ax1.transAxes,
        borderpad=0)
    fig.colorbar(mb, cax=axins1, orientation="horizontal", extend='both', 
        label='NO$_{\mathregular{2}}$ [ppbv]')
    axins1.xaxis.set_ticks_position('top')
    axins1.xaxis.set_label_position('top')
    ax1.tick_params(labelsize=9)
    ax1.set_xlim([-1,71])
    ax1.set_ylim([-65,5])
    # Calculate r, RMSE for linear vs. power fit
    # dno2_sorted = sort_list(dno2, diesel)
    # linfit = fit_func(beta, np.sort(diesel))
    # powerfit = func(np.sort(diesel), *popt)
    # print('Equation coefficients should match the following:')
    # print('Linear: ', beta)
    # print('Exponential: ', popt)
    print('Linear correlation between diesel and dNO2=', r_value)
    print('p-value=',p_value)
    # print('RMSE for linear fit...', math.sqrt(mean_squared_error(dno2_sorted, 
    #     linfit)))
    # print('Correlation for linear fit...', np.corrcoef(dno2_sorted, 
    #     linfit)[0,1])
    # print('RMSE for exponential fit...', math.sqrt(mean_squared_error(dno2_sorted, 
    #     powerfit)))
    for i, txt in enumerate(cities):
        if txt == 'Santiago C40':
            txt = 'Santiago'
        elif txt == 'Mexico City C40':
            txt = 'Mexico City'
        elif txt == 'Los Angeles C40':
            txt = 'Los Angeles'
        elif txt == 'Berlin C40':
            txt = 'Berlin'
        elif txt == 'Milan C40':
            txt = 'Milan'
        elif txt == 'London C40':
            txt = 'London'
        elif txt == 'Auckland C40':
            txt = 'Auckland'        
        ax1.annotate(txt, (diesel[i]+1, dno2[i]+1), fontsize=9)
    plt.savefig(DIR_FIG+'figS6_citynames_eu.png', dpi=1000)    
    # plt.savefig(DIR_FIG+'figS6_eu', dpi=1000)
    return

def figS7(obs):
    """
    """
    from sklearn.metrics import mean_squared_error
    import math
    from scipy.optimize import curve_fit
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from dateutil.relativedelta import relativedelta
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import scipy.odr    
    from scipy import stats
    
    # Add additional C40 cities and calculate city-wide averages
    obs_los = readc40aq.read_losangeles('NO2', '2018-01-01', '2020-12-31')
    coordstemp = obs_los.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Los Angeles C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_los = obs_los.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    obs_mex = readc40aq.read_mexicocity('NO2', '2018-01-01', '2020-12-31')
    coordstemp = obs_mex.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Mexico City C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_mex = obs_mex.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    obs_san = readc40aq.read_santiago('NO2', '2018-01-01', '2020-12-31')
    coordstemp = obs_san.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Santiago C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_san = obs_san.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    obs_ber = readc40aq.read_berlin('NO2', '2019-01-01', '2020-12-31')
    coordstemp = obs_ber.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Berlin C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_ber = obs_ber.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    obs_auc = readc40aq.read_auckland('NO2', '2019-01-01', '2020-12-31')
    coordstemp = obs_auc.groupby(['Longitude','Latitude']).size().reset_index()
    coordstemp['City'] = 'Auckland C40'
    coordstemp.rename({0:'Count'}, axis=1, inplace=True)
    stationcoords.append(coordstemp)
    obs_auc = obs_auc.groupby(by=['Date']).agg({
        'Concentration':'mean', 'Latitude':'mean',
        'Longitude':'mean', 'City':'first'}).reset_index()
    # Combine observations
    obs_withc40 = obs.copy(deep=True)
    obs_withc40 = obs_withc40.append(obs_los, ignore_index=False)
    obs_withc40 = obs_withc40.append(obs_mex, ignore_index=False)
    obs_withc40 = obs_withc40.append(obs_san, ignore_index=False)
    obs_withc40 = obs_withc40.append(obs_auc, ignore_index=False)
    focuscities_withc40 = build_focuscities(False)
    # Recalculate bias-corrected model 
    bcm_withc40 = pd.DataFrame()
    raw_withc40 = pd.DataFrame()
    # # Loop through cities and build bias-corrected model
    for index, row in focuscities_withc40.iterrows():
        city = row['City']    
        print(city)    
        # Select city in model/observational dataset
        gcf_city = model.loc[model['city']==city]
        obs_city = obs_withc40.loc[obs_withc40['City']==city].copy(deep=True)
        # There are some cities (e.g., Brussels) with observations equal to 0 ppb
        # that appear to just be missing data. Change these to NaN!
        obs_city.loc[obs_city['Concentration']==0,'Concentration']= np.nan
        # QA/QC: Require that year 2019 has at least 65% of observations for 
        # a given city; remove observations +/- 3 standard deviations 
        std = np.nanstd(obs_city['Concentration'])
        mean = np.nanmean(obs_city['Concentration'])
        obs_city.loc[obs_city['Concentration']>mean+(3*std), 'Concentration'] = np.nan
        qaqc = obs_city.loc[(obs_city['Date']>='2019-01-01') & 
            (obs_city['Date']<='2019-12-31')]
        if qaqc.shape[0] >= (365*0.65):
            # Remove city column otherwise XGBoost will throw a ValueError (i.e., 
            # DataFrame.dtypes for data must be int, float, bool or categorical.  
            # When categorical type is supplied, DMatrix parameter 
            # `enable_categorical` must be set to `True`.city)    
            del gcf_city['city'], obs_city['City']
            # Run XGBoost for site
            merged_train, bias_train, obs_conc_train = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', '2019-12-31')
            merged_full, bias_full, obs_conc_full = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', '2020-06-30')
            (no2diff, shaps, features, ro, fac2o, mfbo, rt, fac2t, mfbt, rv, fac2v, 
                mfbv) = run_xgboost(args, merged_train, bias_train, merged_full, 
                obs_conc_full)
            # Group data by date and average over all k-fold predictions
            bcm_city = no2diff.groupby(['Date']).mean().reset_index()
            # Add column corresponding to city name for each ID
            bcm_city['City'] = city
            bcm_withc40 = bcm_withc40.append(bcm_city, ignore_index=True)
            # Save off raw (non bias-corrected) observations
            raw_city = merged_full[['Date','NO2']].copy(deep=True)
            raw_city['City'] = city
            raw_withc40 = raw_withc40.append(raw_city, ignore_index=True)
        else:
            print('Skipping %s...'%city)
    dno2, no2, diesel, cities = [], [], [], []
    for index, row in focuscities_withc40.iterrows():
        city = row['City']
        print(city)
        bcm_city = bcm_withc40.loc[bcm_withc40['City']==city]
        bcm_city.set_index('Date', inplace=True)
        # Figure out REQUIRED lockdown dates
        ldstart = focuscities_withc40.loc[focuscities_withc40['City']==
            city]['startreq'].values[0]
        ldstart = pd.to_datetime(ldstart)
        ldend = focuscities_withc40.loc[focuscities_withc40['City']==city][
            'endreq'].values[0]
        ldend = pd.to_datetime(ldend)
        if pd.isnull(ldstart)==False:
            before = np.nanmean(bcm_city.loc[ldstart-relativedelta(years=1):
                ldend-relativedelta(years=1)]['predicted'])
            after = np.abs(np.nanmean(bcm_city.loc[ldstart:ldend]['anomaly']))
            pchange = -after/before*100
            dno2.append(pchange)
        else: 
            dno2.append(np.nan)
        no2.append(bcm_city['observed']['2019-01-01':'2019-12-31'].mean())
        diesel.append(focuscities_withc40.loc[focuscities_withc40['City']==city]['Diesel share'].values[0])
        cities.append(focuscities_withc40.loc[focuscities_withc40['City']==city]['City'].values[0])
    diesel = np.array(diesel)
    cities = np.array(cities)
    no2 = np.array(no2)
    dno2 = np.array(dno2)
    cmap = plt.get_cmap("pink_r")
    cmap = truncate_colormap(cmap, 0.4, 0.9)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', 
        cmaplist, cmap.N)
    cmap.set_over(color='k')
    bounds = np.linspace(8, 20, 7)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # Plotting
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,1),(0,0))
    mb = ax1.scatter(diesel, dno2, c=no2, s=18, cmap=cmap, norm=norm, 
        clip_on=False)
    # Cities exceeding WHO guidelines
    ax1.set_xlabel(r'Diesel-powered passenger vehicle share [%]')
    ax1.set_ylabel(r'$\mathregular{\Delta}$ NO$_{\mathregular{2}}$ [%]')
    # Calculate slope with least squares regression
    idx = np.isfinite(dno2) & np.isfinite(diesel)
    lincoeff = np.poly1d(np.polyfit(diesel[idx], dno2[idx], 1))
    ax1.plot(np.unique(diesel), np.poly1d(np.polyfit(diesel[idx], dno2[idx], 1)
        )(np.unique(diesel)), 'black', ls='dashed', lw=1, zorder=0, 
        label='Linear fit (y=ax+b)\na=-0.52, b=-8.81')    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        diesel[idx], dno2[idx])
    # ax1.legend(frameon=False, loc=9, bbox_to_anchor=(0.5, 1.16), ncol=2)
    ax1.legend(frameon=False, bbox_to_anchor=(0.4, 0.42))
    axins1 = inset_axes(ax1, width='40%', height='5%', loc='lower left', 
        bbox_to_anchor=(0.02, 0.04, 1, 1), bbox_transform=ax1.transAxes,
        borderpad=0)
    fig.colorbar(mb, cax=axins1, orientation="horizontal", extend='both', 
        label='NO$_{\mathregular{2}}$ [ppbv]')
    axins1.xaxis.set_ticks_position('top')
    axins1.xaxis.set_label_position('top')
    ax1.tick_params(labelsize=9)
    ax1.set_xlim([-1,71])
    ax1.set_ylim([-70,5])
    # Calculate r, RMSE for linear vs. power fit
    dno2_sorted = sort_list(dno2, diesel)
    dno2_sorted = np.array(dno2_sorted)
    print('Equation coefficients should match the following:')
    print('Linear: ', lincoeff)
    print('Linear correlation between diesel and dNO2=', r_value)
    print('p-value=',p_value)
    # print('RMSE for linear fit...', math.sqrt(mean_squared_error(
    #     dno2_sorted[idx2], linfit)))
    # for i, txt in enumerate(cities):
    #     if txt == 'Santiago C40':
    #         txt = 'Santiago'
    #     elif txt == 'Mexico City C40':
    #         txt = 'Mexico City'
    #     elif txt == 'Los Angeles C40':
    #         txt = 'Los Angeles'
    #     elif txt == 'Berlin C40':
    #         txt = 'Berlin'
    #     elif txt == 'Milan C40':
    #         txt = 'Milan'
    #     elif txt == 'London C40':
    #         txt = 'London'
    #     elif txt == 'Auckland C40':
    #         txt = 'Auckland'
    #     ax1.annotate(txt, (diesel[i]+1, dno2[i]+1), fontsize=9)
    #  plt.savefig(DIR_FIG+'figS7_citynames_eu.png', dpi=1000)    
    # plt.savefig(DIR_FIG+'figS7_eu.png', dpi=1000)
    return

def figS8(focuscities, bcm, model, mobility): 
    """
    Parameters
    ----------
    focuscities : pandas.core.frame.DataFrame
        Table containing city names, countries, population, share of passenger 
        vehicles using diesel fuel, and lockdown start and end dates.
    bcm : pandas.core.frame.DataFrame
        Observed and bias-corrected, business-as-usual NO2 for focus cities
        averaged over k-folds
    model : pandas.core.frame.DataFrame
        Modeled NO2, meteorology, emissions, and control information (e.g., 
        day, latitude, longitude, etc.)
    mobility : pandas.core.frame.DataFrame
        Timeseries of relative traffic volume (with 13 Jan 2020 as the baseline
        volume) for cities for the specified time period. 

    Returns
    -------
    None.

    """
    import scipy.odr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from dateutil.relativedelta import relativedelta
    dno2, dtraf, diesel, cities = [], [], [], []
    for index, row in focuscities.iterrows():
        city = row['City']
        bcm_city = bcm.loc[bcm['City']==city]
        bcm_city.set_index('Date', inplace=True)
        # Figure out lockdown dates
        ldstart = focuscities.loc[focuscities['City']==city]['start'].values[0]
        ldstart = pd.to_datetime(ldstart)
        ldend = focuscities.loc[focuscities['City']==city]['end'].values[0]
        ldend = pd.to_datetime(ldend)
        # Calculate percentage change in NO2 during lockdown periods
        before = np.nanmean(bcm_city.loc[ldstart-relativedelta(years=1):
            ldend-relativedelta(years=1)]['predicted'])
        after = np.abs(np.nanmean(bcm_city.loc[ldstart:ldend]['anomaly']))
        pchange = -after/before*100
        dno2.append(pchange)    
        # Calculate percentage change in traffic 
        mobility_city = mobility.loc[mobility.city==city]
        mobility_city = mobility_city.set_index('time')
        before = np.nanmean(mobility_city.loc['2020-01-13':ldstart]['Volume'])
        after = np.nanmean(mobility_city.loc[ldstart:ldend]['Volume'])
        # pchange = -after/before*100
        pchange = after-before
        dtraf.append(pchange)
        diesel.append(focuscities.loc[focuscities['City']==city]['Diesel share'].values[0])
        cities.append(focuscities.loc[focuscities['City']==city]['City'].values[0])
    dno2 = np.array(dno2)
    dtraf = np.array(dtraf)
    diesel = np.array(diesel)
    cities = np.array(cities)
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.scatter(dtraf, dno2, c='k', s=18, clip_on=False)
    # Calculate slope linear with regression 
    lincoeff = np.poly1d(np.polyfit(dtraf, dno2, 1))
    ax1.plot(np.unique(dtraf), np.poly1d(np.polyfit(dtraf, dno2, 1)
        )(np.unique(dtraf)), 'black', ls='dashed', lw=1, zorder=0, 
        label='Linear fit (y=ax+b)\na=-0.11, b=-18.86')
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        dtraf, dno2)
    txtstr = 'r = %.2f\np-value = %.2f'%(r_value, p_value)
    ax1.text(-82, -3, txtstr, color='darkgrey', ha='left')
    print('Equation coefficients should match the following:')
    print(lincoeff)
    print('Correlation coefficient is:')
    print(r_value)
    print('p-value is:')
    print(p_value)
    ax1.set_xlabel(r'$\mathregular{\Delta}$ Traffic [%]')
    ax1.set_ylabel(r'$\mathregular{\Delta}$ NO$_{\mathregular{2}}$ [%]')
    # ax1.legend(frameon=False, loc=9, bbox_to_anchor=(0.5, 1.16), ncol=2)
    # ax1.legend(frameon=False, bbox_to_anchor=(0.4, 0.42))
    ax1.set_ylim([-65,5])
    plt.savefig(DIR_FIG+'figS8_eu.png', dpi=1000)
    return 

def figS9(focuscities, model, obs, mobility):
    """

    Parameters
    ----------
    focuscities : pandas.core.frame.DataFrame
        Table containing city names, countries, population, share of passenger 
        vehicles using diesel fuel, and lockdown start and end dates.
    model : pandas.core.frame.DataFrame
        Modeled NO2, meteorology, emissions, and control information (e.g., 
        day, latitude, longitude, etc.)
    obs : pandas.core.frame.DataFrame
        Observed NO2 concentrations
    mobility : pandas.core.frame.DataFrame
        Timeseries of relative traffic volume (with 13 Jan 2020 as the baseline
        volume) for cities for the specified time period. 

    Returns
    -------
    None
    """
    import numpy as np
    import pandas as pd
    from dateutil.relativedelta import relativedelta
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('/Users/ghkerr/GW/mobility/')
    import readc40mobility
    
    def apple_vs_ism(model, obs, city, traffic_city):
        """Calculate bias-corrected, business-as-usual NO2 concentrations using
        Apple Mobility Trends Reports versus in-situ traffic counts. 
        
        Parameters
        ----------
        model : pandas.core.frame.DataFrame
            Modeled NO2, meteorology, emissions, and control information (e.g., 
            day, latitude, longitude, etc.)
        obs : pandas.core.frame.DataFrame
            Observed NO2 concentrations        
        city : str
            City of interest
        traffic_city : pandas.core.frame.DataFrame
            In-situ traffic counts for city of interest
    
        Returns
        -------
        bcm_city : pandas.core.frame.DataFrame
            DataFrame containing the XGBoost predicted concentrations, the 
            original observed concentrations, and the bias using traffic 
            information from the Apple Mobility Trends Reports.
        bcm_cityism : pandas.core.frame.DataFrame
            Same as no2diff but using traffic information from in-situ traffic
            counters. 
        shaps : pandas.core.frame.DataFrame
            SHAPley values for each training.
        shapsism : pandas.core.frame.DataFrame
            Same as shaps but using traffic information from in-situ traffic 
            counters. 
        raw_city : pandas.core.frame.DataFrame
            Observed NO2 concentrations 
        """
        gcf_city = model.loc[model['city']==city]
        obs_city = obs.loc[obs['City']==city].copy(deep=True)
        obs_city.loc[obs_city['Concentration']==0,'Concentration']= np.nan
        std = np.nanstd(obs_city['Concentration'])
        mean = np.nanmean(obs_city['Concentration'])
        obs_city.loc[obs_city['Concentration']>mean+(3*std), 
            'Concentration'] = np.nan
        qaqc = obs_city.loc[(obs_city['Date']>='2019-01-01') & 
            (obs_city['Date']<='2019-12-31')]
        if qaqc.shape[0] >= (365*0.65):
            del gcf_city['city'], obs_city['City']
            # Run XGBoost for site
            merged_train, bias_train, obs_conc_train = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01',
                '2019-12-31')
            merged_full, bias_full, obs_conc_full = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', 
                '2020-06-30')
            (no2diff, shaps, features, ro, fac2o, mfbo, rt, fac2t, mfbt, rv, 
                fac2v, mfbv) = run_xgboost(args, merged_train, bias_train, 
                merged_full, obs_conc_full)
            del gcf_city['Volume']
            gcf_city = pd.merge(gcf_city, traffic_city['Count'], left_on='Date', 
                right_index=True)
            gcf_city = gcf_city.rename(columns={'Count':'Volume'})
            merged_train, bias_train, obs_conc_train = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', 
                '2019-12-31')
            merged_full, bias_full, obs_conc_full = \
                prepare_model_obs(obs_city, gcf_city, '2019-01-01', 
                '2020-06-30')
            (no2diffism, shapsism, featuresism, roism, fac2oism, mfboism, rtism, 
              fac2tism, mfbtism, rvism, fac2vism, mfbvism) = run_xgboost(args, 
              merged_train, bias_train, merged_full, obs_conc_full)
            # Group data by date and average over all k-fold predictions
            bcm_city = no2diff.groupby(['Date']).mean().reset_index()
            bcm_cityism = no2diffism.groupby(['Date']).mean().reset_index()
            # Add column corresponding to city name for each ID
            bcm_city['City'] = city
            bcm_cityism['City'] = city
            # Save off raw (non bias-corrected) observations
            raw_city = merged_full[['Date','NO2']].copy(deep=True)
            raw_city['City'] = city
        return bcm_city, bcm_cityism, shaps, shapsism, raw_city
    startdate, enddate = '2019-01-01','2020-12-31'
    # # Fetch traffic data
    traffic_mil = readc40mobility.read_milan(startdate, enddate)
    traffic_ber = readc40mobility.read_berlin(startdate, enddate)
    traffic_ber = traffic_ber.groupby(traffic_ber.index).mean()
    traffic_mil = traffic_mil.groupby(traffic_mil.index).mean()
    mobility_ber = mobility.loc[mobility['city']=='Berlin']
    mobility_mil = mobility.loc[mobility['city']=='Milan']
    # Calculate bias-corrected observations
    bcmber, bcmberism, shapsber, shapsberism, rawber = apple_vs_ism(model, 
        obs, 'Berlin', traffic_ber)
    bcmmil, bcmmilism, shapsmil, shapsmilism, rawmil = apple_vs_ism(model, 
        obs, 'Milan', traffic_ber)
    # Rolling average
    bcmber = bcmber.set_index('Date')
    bcmberism = bcmberism.set_index('Date')
    rawber = rawber.set_index('Date')
    bcmber = bcmber.resample('1D').mean().rolling(window=7, 
        min_periods=1).mean()
    bcmberism = bcmberism.resample('1D').mean().rolling(window=7, 
        min_periods=1).mean()
    rawber = rawber.resample('1D').mean().rolling(window=7, 
        min_periods=1).mean()
    bcmmil = bcmmil.set_index('Date')
    bcmmilism = bcmmilism.set_index('Date')
    rawmil = rawmil.set_index('Date')
    bcmmil = bcmmil.resample('1D').mean().rolling(window=7, 
        min_periods=1).mean()
    bcmmilism = bcmmilism.resample('1D').mean().rolling(window=7, 
        min_periods=1).mean()
    rawmil = rawmil.resample('1D').mean().rolling(window=7, 
        min_periods=1).mean()
    # Plotting
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot2grid((2,2),(0,0))
    ax1b = ax1.twinx()
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax2b = ax2.twinx()
    ax3 = plt.subplot2grid((2,2),(1,0))
    ax4 = plt.subplot2grid((2,2),(1,1))
    ax1b.plot(traffic_ber.index, mobility_ber['Volume'], lw=1.5, 
        ls='-', color=agnavy, zorder=8)
    ax1.plot(traffic_ber['Count'], lw=1.5, 
        ls='-', color=agorange)
    ax2b.plot(traffic_mil['2019-01-01':'2020-06-30'].index, 
        mobility_mil['Volume'], lw=1.5, ls='-', color=agnavy, zorder=8)
    ax2.plot(traffic_mil['Count'], lw=1.5, 
        ls='-', color=agorange)
    ax3.plot(rawber['NO2'], ls='--', color='darkgrey')
    ax3.plot(bcmber['predicted'], '--', color=agnavy)
    ax3.plot(bcmberism['predicted'], '--', color=agorange)
    ax3.plot(bcmber['observed'], '-k')
    # Calculate percentage change in NO2 during different traffic datasets
    ldstart = focuscities.loc[focuscities['City']=='Berlin']['start'].values[0]
    ldstart = pd.to_datetime(ldstart)
    ldend = focuscities.loc[focuscities['City']=='Berlin']['end'].values[0]
    ldend = pd.to_datetime(ldend)
    before = np.nanmean(bcmber.loc[ldstart-relativedelta(years=1):
        ldend-relativedelta(years=1)]['predicted'])
    after = np.abs(np.nanmean(bcmber.loc[ldstart:ldend]['anomaly']))
    pchange = -after/before*100
    ax3.text(0.65, 0.82,'$\mathregular{\Delta}$NO$_{\mathregular{2}}$ = %.1f%%'%(
        pchange), color=agnavy, transform=ax3.transAxes)
    before = np.nanmean(bcmberism.loc[ldstart-relativedelta(years=1):
        ldend-relativedelta(years=1)]['predicted'])
    after = np.abs(np.nanmean(bcmberism.loc[ldstart:ldend]['anomaly']))
    pchangeism = -after/before*100
    ax3.text(0.65, 0.92,'$\mathregular{\Delta}$NO$_{\mathregular{2}}$ = %.1f%%'%(
        pchangeism), color=agorange, ha='left', transform=ax3.transAxes)
    ax4.plot(rawmil['NO2'], ls='--', color='darkgrey', label='GEOS-CF')
    ax4.plot(bcmmil['predicted'], '--', color=agnavy, 
        label='Business-as-usual with Apple')
    ax4.plot(bcmmilism['predicted'], '--', color=agorange, 
        label=r'Business-as-usual with $\mathit{\mathregular{in-situ}}$')
    ax4.plot(bcmmil['observed'], '-k', label='Observed')
    # Calculate percentage change in NO2 during different traffic datasets
    ldstart = focuscities.loc[focuscities['City']=='Milan']['start'].values[0]
    ldstart = pd.to_datetime(ldstart)
    ldend = focuscities.loc[focuscities['City']=='Milan']['end'].values[0]
    ldend = pd.to_datetime(ldend)
    before = np.nanmean(bcmmil.loc[ldstart-relativedelta(years=1):
        ldend-relativedelta(years=1)]['predicted'])
    after = np.abs(np.nanmean(bcmmil.loc[ldstart:ldend]['anomaly']))
    pchange = -after/before*100
    ax4.text(0.65, 0.82,'$\mathregular{\Delta}$NO$_{\mathregular{2}}$ = %.1f%%'%(
        pchange), color=agnavy, transform=ax4.transAxes)
    before = np.nanmean(bcmmilism.loc[ldstart-relativedelta(years=1):
        ldend-relativedelta(years=1)]['predicted'])
    after = np.abs(np.nanmean(bcmmilism.loc[ldstart:ldend]['anomaly']))
    pchangeism = -after/before*100
    ax4.text(0.65, 0.92,'$\mathregular{\Delta}$NO$_{\mathregular{2}}$ = %.1f%%'%(
        pchangeism), color=agorange, ha='left', transform=ax4.transAxes)
    ax4.legend(bbox_to_anchor=(-1.4, -0.38), frameon=False, ncol=4, loc=3, 
        fontsize=10)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(['2019-11-01', '2020-06-30'])
        ax.set_xticks(['2019-11-01', '2019-12-01', 
            '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', 
            '2020-05-01', '2020-06-01']) 
        ax.set_xticklabels(['Nov\n2019', 'Dec', 'Jan\n2020', 'Feb', 
            'Mar', 'Apr', 'May', 'Jun'], fontsize=9) 
    ax1.spines['top'].set_visible(False)
    ax1b.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2b.spines['top'].set_visible(False)
    ax1b.spines['left'].set_visible(False)
    ax2b.spines['left'].set_visible(False)
    # Y-ticks
    ax1.set_ylim([0,35000])
    ax1.set_yticks(np.linspace(0,35000,11))
    ax1.set_yticklabels(['0','','7000','','14000','','21000','','28000',
        '','35000'])
    ax2.set_ylim([0,150000])
    ax2.set_yticks(np.linspace(0,150000,11))
    ax2.set_yticklabels(['0','','30000','','60000','','90000','','120000',
        '','150000'])
    ax1b.set_ylim([30,130])
    ax1b.set_yticks(np.linspace(30,130,11))
    ax1b.set_yticklabels(['30','','50','','70','','90','','110','','130'])
    ax2b.set_ylim([0,140])
    ax2b.set_yticks(np.linspace(0,140,11))
    ax2b.set_yticklabels(['0','','28','','56','','84','','112','','140'])
    # Labels
    ax1.set_ylabel('In-situ traffic counts', color=agorange)
    ax2.set_ylabel('In-situ traffic counts', color=agorange)
    ax1b.set_ylabel('Apple Mobility Trends Reports [%]', rotation=270, 
        color=agnavy)
    ax1b.yaxis.set_label_coords(1.16,0.5)
    ax2b.set_ylabel('Apple Mobility Trends Reports [%]', rotation=270, 
        color=agnavy)
    ax2b.yaxis.set_label_coords(1.16,0.5)
    ax3.set_ylabel('NO$_{2}$ [ppbv]')
    ax4.set_ylabel('NO$_{2}$ [ppbv]')
    # Spine/tick colors
    ax1.tick_params(axis='y', colors=agorange)
    ax1.spines['left'].set_color(agorange)
    ax2.tick_params(axis='y', colors=agorange)
    ax2.spines['left'].set_color(agorange)
    ax1b.tick_params(axis='y', colors=agnavy)
    ax1b.spines['right'].set_color(agnavy)
    ax2b.tick_params(axis='y', colors=agnavy)
    ax2b.spines['right'].set_color(agnavy)
    ax3.set_ylim([0, 24])
    ax3.set_yticks(np.linspace(0,24,4))
    ax4.set_ylim([0, 42])
    ax4.set_yticks(np.linspace(0,42,4))
    ax1.set_title('(a)', loc='left')
    ax2.set_title('(b)', loc='left')
    ax3.set_title('(c)', loc='left')
    ax4.set_title('(d)', loc='left')
    ax1.text(0.5, 1.1, 'Berlin', fontsize=12, transform=ax1.transAxes,
        ha='center')
    ax2.text(0.5, 1.1, 'Milan', fontsize=12, transform=ax2.transAxes, 
        ha='center')
    plt.subplots_adjust(hspace=0.3, wspace=0.5)
    plt.savefig(DIR_FIG+'figS9_eu.png', dpi=1000)
    return

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
sys.path.append('/Users/ghkerr/GW/mobility/')
import readc40aq
import readc40mobility
focuscities = build_focuscities(True)

# # Load GEOSCF data (AQ and meteorology)
# aqc = pd.read_csv(DIR_MODEL+'aqc_tavg_1d_cities_v2openaq.csv', delimiter=',', 
#     header=0, engine='python', parse_dates=['time'], date_parser=lambda x: 
#     dt.datetime.strptime(x, '%Y-%m-%d'))
# met = pd.read_csv(DIR_MODEL+'met_tavg_1d_cities_v2openaq.csv', delimiter=',', 
#     header=0, engine='python', parse_dates=['time'], date_parser=lambda x: 
#     dt.datetime.strptime(x, '%Y-%m-%d'))    
# # Group by city and date and average 
# aqc = aqc.groupby(by=['city','time']).mean().reset_index()
# met = met.groupby(by=['city','time']).mean().reset_index()
# model = aqc.merge(met, how='left')
# # Add mobility information to observation DataFrame
# mobility = readc40mobility.read_applemobility('2019-01-01', '2020-06-30')
# mobility.reset_index(inplace=True)
# mobility = mobility.rename(columns={'Date':'time'})
# model = model.merge(mobility, on=['city', 'time'], how='right')
# model = model.rename(columns={'time':'Date'})
# model['Date'] = pd.to_datetime(model['Date'])
# model.loc[model['city']=='Berlin C40', 'city'] = 'Berlin'
# model.loc[model['city']=='Milan C40', 'city'] = 'Milan'
# # Repeat but for model with traffic volumes replaced by day-of-week integers
# mobilitydow = readc40mobility.read_applemobility('2019-01-01', '2020-06-30', 
#     dow=True)
# mobilitydow.reset_index(inplace=True)
# modeldow = model.loc[:, model.columns!='Volume'].merge(mobilitydow, 
#     on=['city', 'Date'], how='right')

# # Save all station coordinates 
# stationcoords = [] 
# # # # # Load observations from EEA
# obs_eea = pd.DataFrame([])
# for countryabbrev in ['AT', 'BG', 'CZ', 'DE', 'DK', 'ES', 'FI', 'FR', 'GR', 
#     'HR', 'HU', 'IT', 'LT', 'NL', 'PL', 'RO', 'SE']:
#     country = pd.read_csv(DIR_AQ+'eea/'+
#         '%s_no2_2018-2020_timeseries.csv'%countryabbrev, sep=',', 
#         engine='python', parse_dates=['DatetimeBegin'], 
#         date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
#     if country.empty != True:
#     # Drop Unnamed: 0 row
#         coords = pd.read_csv(DIR_AQ+'eea/'+'%s_no2_2018-2020_coords.csv'
#             %countryabbrev, sep=',', engine='python')
#         coords = coords.drop('Unnamed: 0', axis=1)
#         stationcoords.append(coords)
#         country = country.drop(['Unnamed: 0'], axis=1)
#         country = country.rename(columns={'DatetimeBegin':'Date'})
#         # Calculate city average
#         country = country.groupby(by=['City','Date']).agg({
#             'Concentration':'mean', 'Latitude':'mean',
#             'Longitude':'mean'}).reset_index()
#         obs_eea = obs_eea.append(country, ignore_index=False)
# # Convert from ug/m3 NO2 to ppb (1.88 ug/m3 = 1 ppb)
# obs_eea['Concentration'] = obs_eea['Concentration']/1.88

# obs_lon = readc40aq.read_london('NO2', '2019-01-01', '2020-12-31')
# coordstemp = obs_lon.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'London C40'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# stationcoords.append(coordstemp)
# obs_lon = obs_lon.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()
# obs_eea = obs_eea.append(obs_lon, ignore_index=False)
# # Milan observations from EEA are more or less identical to the ones supplied
# # by C40; however, the EEA observations have a few missing days' worth of data
# # that make the timeseries plots look a little weird, so replace the EEA
# # observations with C40 ones
# obs_eea = obs_eea[~obs_eea.City.isin(['Milan'])]
# obs_mil = readc40aq.read_milan('NO2', '2019-01-01', '2020-12-31')
# obs_mil['City'] = 'Milan'
# coordstemp = obs_mil.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'Milan'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# obs_mil = obs_mil.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()
# obs_eea = obs_eea.append(obs_mil, ignore_index=False)
# stationcoords = pd.concat(stationcoords)
# # Now add delete Milan's coordinates (from EEA) to this DataFrame and then
# # add Milan's coordinates from C40. Note that this command must be done 
# # after concatenating the DataFrame from a list
# stationcoords = stationcoords[~stationcoords.City.isin(['Milan'])]
# stationcoords = stationcoords.append(coordstemp)
# obs = obs_eea

# rorig, fac2orig, mfborig = [], [], []
# rtrain, fac2train, mfbtrain = [], [], []
# rvalid, fac2valid, mfbvalid = [], [], []
# bcm = pd.DataFrame()
# raw = pd.DataFrame()
# shaps_all, features_all, features_lon, shaps_lon = [], [], [], []
# # # Loop through cities and build bias-corrected model
# for index, row in focuscities.iterrows():
#     city = row['City']    
#     print(city)    
#     # Select city in model/observational dataset
#     gcf_city = model.loc[model['city']==city]
#     obs_city = obs.loc[obs['City']==city].copy(deep=True)
#     # There are some cities (e.g., Brussels) with observations equal to 0 ppb
#     # that appear to just be missing data. Change these to NaN!
#     obs_city.loc[obs_city['Concentration']==0,'Concentration']= np.nan
#     # QA/QC: Require that year 2019 has at least 65% of observations for 
#     # a given city; remove observations +/- 3 standard deviations 
#     std = np.nanstd(obs_city['Concentration'])
#     mean = np.nanmean(obs_city['Concentration'])
#     obs_city.loc[obs_city['Concentration']>mean+(3*std), 'Concentration'] = np.nan
#     qaqc = obs_city.loc[(obs_city['Date']>='2019-01-01') & 
#         (obs_city['Date']<='2019-12-31')]
#     if qaqc.shape[0] >= (365*0.65):
#         # Remove city column otherwise XGBoost will throw a ValueError (i.e., 
#         # DataFrame.dtypes for data must be int, float, bool or categorical.  
#         # When categorical type is supplied, DMatrix parameter 
#         # `enable_categorical` must be set to `True`.city)    
#         del gcf_city['city'], obs_city['City']
#         # Run XGBoost for site
#         merged_train, bias_train, obs_conc_train = \
#             prepare_model_obs(obs_city, gcf_city, '2019-01-01', '2019-12-31')
#         merged_full, bias_full, obs_conc_full = \
#             prepare_model_obs(obs_city, gcf_city, '2019-01-01', '2020-06-30')
#         (no2diff, shaps, features, ro, fac2o, mfbo, rt, fac2t, mfbt, rv, fac2v, 
#             mfbv) = run_xgboost(args, merged_train, bias_train, merged_full, 
#             obs_conc_full)
#         # For SHAP plot
#         shaps_all.append(shaps)
#         # Save off SHAP values for London
#         if city=='London C40':
#             # features_lon.append(features)
#             shaps_lon.append(shaps)
#         # Append evaluation metrics to multi-city lists
#         rorig.extend(ro)
#         fac2orig.extend(fac2o)
#         mfborig.extend(mfbo)
#         rtrain.extend(rt)
#         fac2train.extend(fac2t)
#         mfbtrain.extend(mfbt)
#         rvalid.extend(rv)
#         fac2valid.extend(fac2v)
#         mfbvalid.extend(mfbv)              
#         # Group data by date and average over all k-fold predictions
#         bcm_city = no2diff.groupby(['Date']).mean().reset_index()
#         # Add column corresponding to city name for each ID
#         bcm_city['City'] = city
#         bcm = bcm.append(bcm_city, ignore_index=True)
#         # Save off raw (non bias-corrected) observations
#         raw_city = merged_full[['Date','NO2']].copy(deep=True)
#         raw_city['City'] = city
#         raw = raw.append(raw_city, ignore_index=True)
#     else:
#         print('Skipping %s...'%city)
# # Concatenate features
# shaps_concat = pd.concat(shaps_all)
# shaps_lon = pd.concat(shaps_lon)

# fig2()
# fig3(focuscities, bcm)  
# fig4(focuscities, bcm, stationcoords, coarse=True)
# figS1()
# figS2()
# figS3()
# figS4()
# figS5()
# figS6(obs)
# figS7(obs)
# figS8(focuscities, bcm, model, mobility)
# figS9(focuscities, model, obs, mobility)