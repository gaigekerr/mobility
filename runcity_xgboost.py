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

# Load custom font
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

def build_focuscities():
    """Build table of focus cities for this study.

    Parameters
    ----------
    None

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
        ['Athens', 'Greece', 8.1], # ACEA 
        ['Auckland C40', 'New Zealand', 8.3], # C40 partnership
        ['Barcelona', 'Spain', 	58.7], # ACEA
        ['Berlin C40', 'Germany', 31.7], # ACEA
        ['Budapest', 'Hungary', 31.5], # ACEA
        ['Copenhagen', 'Denmark', 30.9], # ACEA 
        ['Helsinki', 'Finland', 27.9], # ACEA 
        ['Krakow', 'Poland', 31.6], # ACEA 
        ['London C40', 'United Kingdom', 39.0], # ACEA 
        ['Los Angeles C40', 'United States', 0.4], # C40 partnership
        ['Madrid', 'Spain', 58.7], # ACEA
        ['Marseille', 'France', 58.9], # ACEA
        ['Mexico City C40', 'Mexico', 0.2], # C40 partnership
        ['Milan C40', 'Italy', 44.2], # ACEA 
        ['Munich', 'Germany', 31.7], # ACEA
        ['Paris', 'France', 58.9], # ACEA
        ['Prague', 'Czechia', 35.9], # ACEA
        ['Rome', 'Italy', 44.2], # ACEA
        ['Rotterdam', 'Netherlands', 14.0], # ACEA
        ['Santiago C40', 'Chile', 7.1], # C40 partnership
        ['Sofia', 'Bulgaria',  43.1], # ICCT partnership
        ['Stockholm', 'Sweden', 35.5], # ACEA
        ['Vienna', 'Austria', 55.0], # ACEA
        ['Vilnius', 'Lithuania', 69.2], # ACEA
        ['Warsaw', 'Poland', 31.6], # ACEA
        ['Zagreb', 'Croatia', 52.4], # ACEA 
        ]    
    focuscities = pd.DataFrame(focuscities, columns=['City', 'Country', 
        'Diesel share'])    
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
    focuscities['level1start'] = np.nan
    focuscities['level1end'] = np.nan
    focuscities['level2start'] = np.nan
    focuscities['level2end'] = np.nan
    focuscities['level2start'] = np.nan
    focuscities['level3end'] = np.nan
    # Loop through cities and determine the dates of lockdowns
    for index, row in focuscities.iterrows():
        country = row['Country']
        sah_country = sah.loc[sah['Entity']==country]
        # First occurrences of level 2-3 stay at home requirements 
        where1 = np.where(sah_country['stay_home_requirements']==1.)[0]
        where2 = np.where(sah_country['stay_home_requirements']==2.)[0]
        where3 = np.where(sah_country['stay_home_requirements']==3.)[0]
        # Code from above function adapted from https://stackoverflow.com/
        # questions/2154249/identify-groups-of-continuous-numbers-in-a-list
        where1startend = list(ranges(where1))
        where2startend = list(ranges(where2))
        where3startend = list(ranges(where3))
        try: 
            level1initiate = where1startend[0][0]
            level1end = where1startend[0][-1]
            focuscities.loc[index, 'level1start'] = sah_country['Day'].values[
                level1initiate]
            focuscities.loc[index, 'level1end'] = sah_country['Day'].values[
                level1end]
        except IndexError: 
            level1initiate = np.nan
            level1end = np.nan
        try: 
            level2initiate = where2startend[0][0]
            level2end = where2startend[0][-1]
            focuscities.loc[index, 'level2start'] = sah_country['Day'].values[
                level2initiate]
            focuscities.loc[index, 'level2end'] = sah_country['Day'].values[
                level2end]
        except IndexError: 
            level2initiate = np.nan
            level2end = np.nan
        # Last occurrence for the level 2-3 stay at home requirements after 
        # the initial implementation (note that this code won't pick up if, 
        # for example, there is a level 3 stay at home order that is lifted and 
        # then reinstated. It will only index the end of the *first*
        # implementation)
        try:
            level3initiate = where3startend[0][0]
            level3end = where3startend[0][-1]
            focuscities.loc[index, 'level3start'] = sah_country['Day'].values[
                level3initiate]
            focuscities.loc[index, 'level3end'] = sah_country['Day'].values[
                level3end]
        except IndexError: 
            level3initiate = np.nan
            level3end = np.nan
    # Add empty column for city-specific ratio of transportation NOx to 
    # total NOx
    focuscities['Ratio'] = np.nan
    return focuscities

def fig1(): 
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
    ld = ['level1start', 'level1end', 'level2start', 'level2end', 'level3end', 
        'level3start']
    ld_lon = focuscities.loc[focuscities['City']=='London C40']
    ld_lon = ld_lon[ld]
    ldstart = pd.to_datetime(ld_lon[ld].values[0]).min()
    ldend = pd.to_datetime(ld_lon[ld].values[0]).max()
    x = pd.date_range(ldstart,ldend)
    y = range(37)
    z = [[z] * len(x) for z in range(len(y))]
    num_bars = 100 # More bars = smoother gradient
    cmap = plt.get_cmap('Reds')
    new_cmap = truncate_colormap(cmap, 0.0, 0.35)
    ax1.contourf(x, y, z, num_bars, cmap=new_cmap, zorder=0)
    ax1.text(x[int(x.shape[0]/2.)-2], 31, 'LOCK-\nDOWN', ha='center', 
        rotation=270, va='center', fontsize=14) 
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
            widths=[0.5], patch_artist=True, whis=(20,80), vert=False, 
            showfliers=False)
        ax2.text(np.percentile(shaps_concat[var].values, 80)+0.03, i, 
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
            widths=[0.5], patch_artist=True, whis=(20,80), vert=False, 
            showfliers=False)
        ax2.text(np.percentile(shaps_lon[var].values, 80)+0.03, i+12, 
            vardict[var], ha='left', va='center', fontsize=9)    
        for item in ['boxes', 'whiskers', 'fliers']:
            plt.setp(bplot_lon[item], color=agnavy)  
        for item in ['medians', 'caps']:
            plt.setp(bplot_lon[item], color='w')      
    draw_brace(ax2, (0, 9), 0., agorange)
    draw_brace(ax2, (12, 21), 0., agnavy)
    ax2.text(-0.2, 16.5, 'London', rotation=90, ha='center', va='center', 
        color=agnavy)
    ax2.text(-0.2, 4.5, 'All', rotation=90, ha='center', va='center', 
        color=agorange)
    ax2.set_xlim([0,1.5])
    ax2.set_xticks([0,0.25,0.5,0.75,1.,1.25,1.5])
    ax2.set_xticklabels(['0.0','','0.5','','1.0','','1.5'], fontsize=9)
    ax2.set_ylim([-0.5,22.])
    ax2.set_yticks([])
    ax2.invert_yaxis()
    for side in ['left', 'right', 'top']:
        ax2.spines[side].set_visible(False)
    plt.subplots_adjust(left=0.05, right=0.92)
    ax1.set_title('(a) London', x=0.1, y=1.02, fontsize=12)
    ax2.set_title('(b) Absolute SHAP values', y=1.02, loc='left', fontsize=12)
    plt.savefig(DIR_FIG+'fig1.png', dpi=1000)
    return

def fig2(focuscities, bcm):
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
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from dateutil.relativedelta import relativedelta
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import scipy.odr    
    import scipy.stats
    dno2, no2, diesel, cities = [], [], [], []
    for index, row in focuscities.iterrows():
        city = row['City']
        print(city)
        bcm_city = bcm.loc[bcm['City']==city]
        bcm_city.set_index('Date', inplace=True)
        # Figure out lockdown dates
        ld = ['level1start', 'level1end', 'level2start', 'level2end', 
            'level3end', 'level3start']
        ld_city = focuscities.loc[focuscities['City']==city]
        ld_city = ld_city[ld]        
        ldstart = pd.to_datetime(ld_city[ld].values[0]).min()
        ldend = pd.to_datetime(ld_city[ld].values[0]).max()  
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
    # Calculate slope with total least squares (ODR)
    idx = np.isfinite(diesel) & np.isfinite(dno2)
    Model = scipy.odr.Model(fit_func)
    odr = scipy.odr.RealData(diesel[idx], dno2[idx])
    odr = scipy.odr.ODR(odr, Model,[np.polyfit(diesel[idx], dno2[idx], 1)[1],
        np.polyfit(diesel[idx], dno2[idx], 1)[0]], maxit=10000)
    output = odr.run() 
    beta = output.beta
    print('Equation coefficients should match the following:')
    print(beta)    
    # Plotting
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,1),(0,0))
    mb = ax1.scatter(diesel, dno2, c=no2, s=18, cmap=cmap, norm=norm, 
        clip_on=False)
    # Cities exceeding WHO guidelines
    ax1.scatter(diesel[np.where(no2>40/1.88)], dno2[np.where(no2>40/1.88)], 
        s=19, ec='r', fc='None', norm=norm, clip_on=False)
    ax1.set_xlabel(r'Market shares of diesel-powered passenger vehicles [%]')
    ax1.set_ylabel(r'$\mathregular{\Delta}$ NO$_{\mathregular{2}}$ [%]')
    ax1.plot(np.sort(diesel), fit_func(beta, np.sort(diesel)), 'darkgrey', 
        ls='--', lw=1, zorder=0)    
    txtstr = r'$\mathregular{\Delta}\:$NO$_{\mathregular{2}}$ = -0.58'+\
        r'$\:\mathregular{\times}\:$MSDPV$\:-\:$3.23'
    ax1.text(38, 1, txtstr, color='darkgrey')
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
    axins1 = inset_axes(ax1, width='40%', height='5%', loc='lower left', 
        bbox_to_anchor=(0.02, 0.04, 1, 1), bbox_transform=ax1.transAxes,
        borderpad=0)
    fig.colorbar(mb, cax=axins1, orientation="horizontal", extend='both', 
        label='NO$_{\mathregular{2}}$ [ppbv]')#, ticks=[1, 2, 3])
    axins1.xaxis.set_ticks_position('top')
    axins1.xaxis.set_label_position('top')
    ax1.tick_params(labelsize=9)
    ax1.set_xlim([-1,71])
    ax1.set_ylim([-65,5])
    # plt.savefig(DIR_FIG+'fig2_citynames.png', dpi=1000)    
    plt.savefig(DIR_FIG+'fig2.png', dpi=1000)
    return  

def figS1():
    """    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.io import shapereader
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.io.img_tiles as cimgt
    # Options that work and look halfway decent are: 'GoogleTiles',
    # 'GoogleWTS', 'QuadtreeTiles', 'Stamen'
    request = cimgt.Stamen()    
    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=5, ncols=4, 
        subplot_kw={'projection':request.crs}) 
    axes = np.hstack(axes)
    # Loop through cities for which we've built BCM/BAU concentrations
    citiesunique = np.unique(bcm.City)
    citiesunique = np.sort(citiesunique)
    for i, city in enumerate(citiesunique[:20]):
        citycoords = stationcoords.loc[stationcoords['City']==city]
        # Plot station coordinates   
        axes[i].plot(citycoords['Longitude'], citycoords['Latitude'], 
            marker='o', lw=0, markersize=3, color=agred, 
            transform=ccrs.PlateCarree())
        # Aesthetics
        if city=='London C40':    
            axes[i].set_title('London', loc='left')
        elif city=='Milan C40':    
            axes[i].set_title('Milan', loc='left')        
        elif city=='Berlin C40':    
            axes[i].set_title('Berlin', loc='left')
        elif city=='Santiago C40':    
            axes[i].set_title('Santiago', loc='left')
        elif city=='Mexico City C40':    
            axes[i].set_title('Mexico City', loc='left')
        elif city=='Los Angeles C40':    
            axes[i].set_title('Los Angeles', loc='left')
        elif city=='Auckland C40':    
            axes[i].set_title('Auckland', loc='left')        
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
    plt.savefig(DIR_FIG+'figS1a.png', dpi=1000)
    plt.show()
    # For the rest of the focus cities
    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=5, ncols=4, 
        subplot_kw={'projection':request.crs}) 
    axes = np.hstack(axes)
    for i, city in enumerate(citiesunique[20:]):
        citycoords = stationcoords.loc[stationcoords['City']==city]
        axes[i].plot(citycoords['Longitude'], citycoords['Latitude'], 
            marker='o', lw=0, markersize=3, color=agred, 
            transform=ccrs.PlateCarree())
        if city=='London C40':    
            axes[i].set_title('London', loc='left')
        elif city=='Milan C40':    
            axes[i].set_title('Milan', loc='left')        
        elif city=='Berlin C40':    
            axes[i].set_title('Berlin', loc='left')
        elif city=='Santiago C40':    
            axes[i].set_title('Santiago', loc='left')
        elif city=='Mexico City C40':    
            axes[i].set_title('Mexico City', loc='left')
        elif city=='Los Angeles C40':    
            axes[i].set_title('Los Angeles', loc='left')
        elif city=='Auckland C40':    
            axes[i].set_title('Auckland', loc='left')        
        else:
            axes[i].set_title(city, loc='left')
        extent = [citycoords['Longitude'].min()-0.2, 
            citycoords['Longitude'].max()+0.2, 
            citycoords['Latitude'].min()-0.2, 
            citycoords['Latitude'].max()+0.2]
        request = cimgt.Stamen()
        axes[i].set_extent(extent)
        axes[i].add_image(request, 11)
        axes[i].set_adjustable('datalim')
    # Remove blank axes
    for i in np.arange(6,20,1):
        axes[i].axis('off')     
    plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    plt.savefig(DIR_FIG+'figS1b.png', dpi=1000)
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
        if cityloc < 13.*2:
            ax1.plot(mobility_city.index, (filler+
                mobility_city['Volume'].values/100.*-1), color='k')
            pcolorax1.append(sah_country.values)
            ysax1.append(cityloc+1)
            if city=='Auckland C40':
                city='Auckland'
            elif city=='Berlin C40':
                city='Berlin'
            elif city=='London C40':
                city='London'
            elif city=='Los Angeles C40':
                city='Los Angeles'
            elif city=='Mexico City C40':
                city='Mexico City'
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
    plt.savefig(DIR_FIG+'figS2.png', dpi=1000)
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
    plt.savefig(DIR_FIG+'figS3.png', dpi=1000)
    return 

def figS4():
    """
    """    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from dateutil.relativedelta import relativedelta
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import scipy.odr    
    import scipy.stats
    dno2, no2, diesel, cities = [], [], [], []
    for index, row in focuscities.iterrows():
        city = row['City']
        print(city)
        bcm_city = bcm.loc[bcm['City']==city]
        bcm_city.set_index('Date', inplace=True)
        before = np.nanmean(bcm_city.loc[(
            pd.to_datetime('2020-03-15')-relativedelta(years=1)):
            (pd.to_datetime('2020-06-15')-relativedelta(years=1))]['predicted'])
        after = np.abs(np.nanmean(bcm_city.loc['2020-03-15':'2020-06-15']['anomaly']))
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
    # Calculate slope with total least squares (ODR)
    idx = np.isfinite(diesel) & np.isfinite(dno2)
    Model = scipy.odr.Model(fit_func)
    odr = scipy.odr.RealData(diesel[idx], dno2[idx])
    odr = scipy.odr.ODR(odr, Model,[np.polyfit(diesel[idx], dno2[idx], 1)[1],
        np.polyfit(diesel[idx], dno2[idx], 1)[0]], maxit=10000)
    output = odr.run() 
    beta = output.beta
    print('Equation coefficients should match the following:')
    print(beta)
    # Plotting
    fig = plt.figure(figsize=(6,4))
    ax1 = plt.subplot2grid((1,1),(0,0))
    mb = ax1.scatter(diesel, dno2, c=no2, s=18, cmap=cmap, norm=norm, 
        clip_on=False)
    # Cities exceeding WHO guidelines
    ax1.scatter(diesel[np.where(no2>40/1.88)], dno2[np.where(no2>40/1.88)], 
        s=19, ec='r', fc='None', norm=norm, clip_on=False)
    ax1.set_xlabel(r'Market shares of diesel-powered passenger vehicles [%]')
    ax1.set_ylabel(r'$\mathregular{\Delta}$ NO$_{\mathregular{2}}$ [%]')
    ax1.plot(np.sort(diesel), fit_func(beta, np.sort(diesel)), 'darkgrey', 
        ls='--', lw=1, zorder=0)    
    txtstr = r'$\mathregular{\Delta}\:$NO$_{\mathregular{2}}$ = -0.72'+\
        r'$\:\mathregular{\times}\:$MSDPV$\:+\:$2.15'
    ax1.text(38, 1, txtstr, color='darkgrey')
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
    axins1 = inset_axes(ax1, width='40%', height='5%', loc='lower left', 
        bbox_to_anchor=(0.02, 0.04, 1, 1), bbox_transform=ax1.transAxes,
        borderpad=0)
    fig.colorbar(mb, cax=axins1, orientation="horizontal", extend='both', 
        label='NO$_{\mathregular{2}}$ [ppbv]')#, ticks=[1, 2, 3])
    axins1.xaxis.set_ticks_position('top')
    axins1.xaxis.set_label_position('top')
    ax1.tick_params(labelsize=9)
    ax1.set_xlim([-1,71])
    ax1.set_ylim([-65,5])
    # plt.savefig(DIR_FIG+'figS4_citynames.png', dpi=1000)    
    plt.savefig(DIR_FIG+'figS4.png', dpi=1000)
    return

# import datetime as dt
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import sys
# sys.path.append('/Users/ghkerr/GW/mobility/')
# import readc40aq
# import readc40mobility
# focuscities = build_focuscities()
# # Load GEOSCF data (AQ and meteorology)
# aqc = pd.read_csv(DIR_MODEL+'aqc_tavg_1d_cities_v2openaq.csv', delimiter=',', 
#     header=0, engine='python', parse_dates=['time'],date_parser=lambda x: 
#     dt.datetime.strptime(x, '%Y-%m-%d'))
# met = pd.read_csv(DIR_MODEL+'met_tavg_1d_cities_v2openaq.csv', delimiter=',', 
#     header=0, engine='python', parse_dates=['time'],date_parser=lambda x: 
#     dt.datetime.strptime(x, '%Y-%m-%d'))    
# # Group by city and date and average 
# aqc = aqc.groupby(by=['city','time']).mean().reset_index()
# met = met.groupby(by=['city','time']).mean().reset_index()
# model = aqc.merge(met, how='left')
# # Delete Berlin and Milan (this retains the C40 version of these cities) 
# # and fix spelling of Taipei City
# model.drop(model.loc[model['city']=='Berlin'].index, inplace=True)
# model.drop(model.loc[model['city']=='Milan'].index, inplace=True)
# model.loc[model.city=='Tapei','city'] = 'Taipei City'
# # Also drop Mumbai (note that in the model dataset, the coordinates appear
# # to be surrounding Pune, not Mumbai 
# # Try this (before averaging by city/date),
# # mum = aqc.loc[aqc.city=='Mumbai'] 
# # plt.plot(mum['longitude'], mum['latitude'], 'k*')
# # Plus, the city's data for the measuring period doesn't meet our QA/QC
# # so skip for now!
# model.drop(model.loc[model['city']=='Mumbai'].index, inplace=True)
# # Add mobility information to observation DataFrame
# mobility = readc40mobility.read_applemobility('2019-01-01', '2020-06-30')
# mobility.reset_index(inplace=True)
# mobility = mobility.rename(columns={'Date':'time'})
# model = model.merge(mobility, on=['city', 'time'], how='right')
# model = model.rename(columns={'time':'Date'})
# model['Date'] = pd.to_datetime(model['Date'])

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
#         # # Find transportation/total NOx ratio in each city 
#         # ratio = edgar_sectorimportance(country)
#         # for index, row in ratio.iterrows():
#         #     city = row['City']
#         #     focuscities.loc[focuscities['City']==city,'Ratio'] = row['Ratio']
#         # Calculate city average
#         country = country.groupby(by=['City','Date']).agg({
#             'Concentration':'mean', 'Latitude':'mean',
#             'Longitude':'mean'}).reset_index()
#         # Remove Berlin and Milan since these cities' observations exist
#         # from the C40 dataset. Note, though, that the EEA dataset goes back
#         # to 2018 whereas the C40 dataset as of 21 April 2021 only includes
#         # 2019-2020
#         country.drop(country.loc[country['City']=='Berlin'].index, inplace=True)
#         country.drop(country.loc[country['City']=='Milan'].index, inplace=True)
#         obs_eea = obs_eea.append(country, ignore_index=False)
# # Convert from ug/m3 NO2 to ppb (1.88 ug/m3 = 1 ppb)
# obs_eea['Concentration'] = obs_eea['Concentration']/1.88

# # # # # Load observations from openAQ
# obs_openaq = pd.DataFrame([])
# for country in ['AU', 'BE', 'CA', 'CH', 'CN', 'CO', 'IE', 'LU',  
#     # Remove IN on 22 April 2021
#     'LV', 'NO', 'RS', 'SK', 'TH', 'TW']:
#     country = pd.read_csv(DIR_AQ+'openaq/'+
#         '%s_no2_2018-2020_timeseries.csv'%country, sep=',', 
#         engine='python', parse_dates=['DatetimeBegin'], 
#         date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
#     if country.empty != True:
#         country = country.drop(['Unnamed: 0'], axis=1)
#         country = country.rename(columns={'DatetimeBegin':'Date'})
#         # # Find transportation/total NOx ratio in each city 
#         # ratio = edgar_sectorimportance(country)
#         # for index, row in ratio.iterrows():
#         #     city = row['City']
#         #     focuscities.loc[focuscities['City']==city,'Ratio'] = row['Ratio']
#         # Calculate city average
#         country = country.groupby(by=['Date']).agg({
#             'Concentration':'mean', 'Latitude':'mean',
#             'Longitude':'mean', 'City':'first'}).reset_index()
#         obs_openaq = obs_openaq.append(country, ignore_index=False)
# # obs_openaq.loc[obs_openaq.City=='Tapei','City'] = 'Taipei City'
    
# # # # # Load observations from C40 cities and calculate city-wide averages
# obs_los = readc40aq.read_losangeles('NO2', '2018-01-01', '2020-12-31')
# coordstemp = obs_los.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'Los Angeles C40'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# stationcoords.append(coordstemp)
# obs_los = obs_los.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()
# obs_mex = readc40aq.read_mexicocity('NO2', '2018-01-01', '2020-12-31')
# coordstemp = obs_mex.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'Mexico City C40'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# stationcoords.append(coordstemp)
# obs_mex = obs_mex.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()
# obs_san = readc40aq.read_santiago('NO2', '2018-01-01', '2020-12-31')
# coordstemp = obs_san.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'Santiago C40'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# stationcoords.append(coordstemp)
# obs_san = obs_san.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()
# obs_ber = readc40aq.read_berlin('NO2', '2019-01-01', '2020-12-31')
# coordstemp = obs_ber.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'Berlin C40'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# stationcoords.append(coordstemp)
# obs_ber = obs_ber.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()
# obs_mil = readc40aq.read_milan('NO2', '2019-01-01', '2020-12-31')
# coordstemp = obs_mil.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'Milan C40'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# stationcoords.append(coordstemp)
# obs_mil = obs_mil.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()
# obs_lon = readc40aq.read_london('NO2', '2019-01-01', '2020-12-31')
# coordstemp = obs_lon.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'London C40'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# stationcoords.append(coordstemp)
# obs_lon = obs_lon.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()
# obs_auc = readc40aq.read_auckland('NO2', '2019-01-01', '2020-12-31')
# coordstemp = obs_auc.groupby(['Longitude','Latitude']).size().reset_index()
# coordstemp['City'] = 'Auckland C40'
# coordstemp.rename({0:'Count'}, axis=1, inplace=True)
# stationcoords.append(coordstemp)
# obs_auc = obs_auc.groupby(by=['Date']).agg({
#     'Concentration':'mean', 'Latitude':'mean',
#     'Longitude':'mean', 'City':'first'}).reset_index()

# # # # # Combine EEA, C40, and openAQ observations
# obs_eea = obs_eea.append(obs_los, ignore_index=False)
# obs_eea = obs_eea.append(obs_mex, ignore_index=False)
# obs_eea = obs_eea.append(obs_san, ignore_index=False)
# obs_eea = obs_eea.append(obs_ber, ignore_index=False)
# obs_eea = obs_eea.append(obs_mil, ignore_index=False)
# obs_eea = obs_eea.append(obs_lon, ignore_index=False)
# obs_eea = obs_eea.append(obs_auc, ignore_index=False)
# obs = obs_eea
# stationcoords = pd.concat(stationcoords)
# # obs = obs_eea.append(obs_openaq)

# rorig, fac2orig, mfborig = [], [], []
# rtrain, fac2train, mfbtrain = [], [], []
# rvalid, fac2valid, mfbvalid = [], [], []
# bcm = pd.DataFrame()
# raw = pd.DataFrame()
# shaps_all, features_all, features_lon, shaps_lon = [], [], [], []
# shaps_all, shaps_lon = [], []

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
        
agorange = '#CD5733'
agtan = '#F4E7C5'
agnavy = '#678096'
agblue = '#ACC2CF'
agpuke = '#979461'
agred = '#A12A19'
        
# fig1() 
# fig2(focuscities, bcm)  
# figS1()
# figS2()
# figS3()
# figS4()

"""CEDS INVENTORY COMPARISON"""
# import matplotlib.pyplot as plt
# import netCDF4 as nc
# import sys
# sys.path.append('/Users/ghkerr/phd/GMI/')
# from geo_idx import geo_idx
# # def cedsopen(fn):
# fn = 'NO-em-total-anthro_CEDS_2017.nc'
# ceds = nc.Dataset('/Users/ghkerr/Downloads/2017/'+fn)
# lng_ceds = ceds.variables['lon'][:]
# lat_ceds = ceds.variables['lat'][:]
# agr_ceds = np.nansum(ceds.variables['NO_agr'][:], axis=0)
# ene_ceds = np.nansum(ceds.variables['NO_ene'][:], axis=0)
# ind_ceds = np.nansum(ceds.variables['NO_ind'][:], axis=0)
# nrtr_ceds = np.nansum(ceds.variables['NO_nrtr'][:], axis=0)
# rcoc_ceds = np.nansum(ceds.variables['NO_rcoc'][:], axis=0)
# rcoo_ceds = np.nansum(ceds.variables['NO_rcoo'][:], axis=0)
# rcor_ceds = np.nansum(ceds.variables['NO_rcor'][:], axis=0)
# road_ceds = np.nansum(ceds.variables['NO_road'][:], axis=0)
# shp_ceds = np.nansum(ceds.variables['NO_shp'][:], axis=0)
# slv_ceds = np.nansum(ceds.variables['NO_slv'][:], axis=0)
# wst_ceds = np.nansum(ceds.variables['NO_wst'][:], axis=0)
# no_ceds = (agr_ceds+ene_ceds+ind_ceds+nrtr_ceds+rcoc_ceds+rcoo_ceds+
#     rcor_ceds+road_ceds+shp_ceds+slv_ceds+wst_ceds)
# # return no_ceds, road_ceds
# road_ceds = road_ceds+nrtr_ceds
# ratio_all = []
# diesel_all = []
# city_all = []
# ct = ['Athens','Mexico City C40', 'Santiago C40', 'Auckland C40', 
#       'Rotterdam', 'Hamburg', 'Berlin C40', 'Krakow', 'Prague', 'Sofia',
#       'Warsaw', 'Copenhagen', 'Helsinki', 'London C40', 'Stockholm', 
#       'Milan C40', 'Vienna', 'Vilnius', 'Rome', 'Marseille', 'Paris', 'Barcelona',
#       'Zagreb', 'Madrid']
# for city in ct:
#     gcf_city = model.loc[model['city']==city]
#     lat_city = gcf_city['latitude'].mean()
#     lng_city = gcf_city['longitude'].mean()
#     lng_closest = geo_idx(lng_city, lng_ceds)
#     lat_closest = geo_idx(lat_city, lat_ceds)

#     # ratio = (road_ceds[lat_closest-1:lat_closest+2,lng_closest-1:lng_closest+2]
#         # /no_ceds[lat_closest-1:lat_closest+2,lng_closest-1:lng_closest+2])
#     ratio = (road_ceds/no_ceds)[lat_closest-1:lat_closest+2,lng_closest-1:lng_closest+2]
#     ratio_all.append(ratio.mean())
#     city_all.append(city)
#     diesel_all.append(focuscities.loc[focuscities['City']==city]['Diesel share'].values[0])
# fig = plt.figure(figsize=(6,4))
# ax = plt.subplot2grid((1,1),(0,0))
# mb = ax.scatter(diesel_all, ratio_all, s=12)
# # plt.colorbar(mb, label='Percent traffic NOx [%]')
# # ax.set_xlabel(r'Percent diesel')#x fraction traffic
# # ax.set_ylabel(r'Lockdown NO$_{\mathregular{2}}$ change [%]')
# for i, txt in enumerate(city_all):
#     ax.annotate(txt, (diesel_all[i], ratio_all[i]), fontsize=7)

""""TRAFFIC DATA SENSITIVITY"""
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.transforms import Bbox, TransformedBbox,\
#     blended_transform_factory
# from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
#     BboxConnectorPatch
# import sys
# sys.path.append('/Users/ghkerr/GW/mobility/')
# import readc40mobility
# # Adapted from 
# # https://matplotlib.org/3.2.2/gallery/subplots_axes_and_figures/
# # axes_zoom_effect.html 
# def connect_bbox(bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, prop_lines, 
#     prop_patches=None):
#     if prop_patches is None:
#         prop_patches = prop_lines.copy()
#         prop_patches["alpha"] = prop_patches.get("alpha", 1)*0.2
#     c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
#     c1.set_clip_on(False)
#     c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
#     c2.set_clip_on(False)
#     bbox_patch1 = BboxPatch(bbox1, **prop_patches)
#     bbox_patch2 = BboxPatch(bbox2, **prop_patches)
#     p = BboxConnectorPatch(bbox1, bbox2, loc1a=loc1a, loc2a=loc2a, 
#         loc1b=loc1b, loc2b=loc2b, **prop_patches)
#     p.set_clip_on(False)
#     return c1, c2, bbox_patch1, bbox_patch2, p
# def zoom_effect02(ax1, ax2, **kwargs):
#     """
#     ax2 : the big main axes
#     ax1 : the zoomed axes
#     The xmin & xmax will be taken from the
#     ax1.viewLim.
#     """
#     tt = ax1.transScale + (ax1.transLimits+ax2.transAxes)
#     trans = blended_transform_factory(ax2.transData, tt)
#     mybbox1 = ax1.bbox
#     mybbox2 = TransformedBbox(ax1.viewLim, trans)
#     prop_patches = kwargs.copy()
#     prop_patches["ec"] = "none"
#     prop_patches["alpha"] = 0.0
#     # prop_patches['fc'] = 'lightgrey'
#     c1, c2, bbox_patch1, bbox_patch2, p = \
#         connect_bbox(mybbox1, mybbox2, loc1a=2, loc2a=3, loc1b=1, loc2b=4, 
#         prop_lines=kwargs, prop_patches=prop_patches)
#     ax1.add_patch(bbox_patch1)
#     ax2.add_patch(bbox_patch2)
#     ax2.add_patch(c1)
#     ax2.add_patch(c2)
#     ax2.add_patch(p)
#     return c1, c2, bbox_patch1, bbox_patch2, p
# def adjust_spines(ax, spines):
#     for loc, spine in ax.spines.items():
#         if loc in spines:
#             spine.set_position(('outward', 10))
#         # else:
#         #     spine.set_color('none')  # don't draw spine
#     # turn off ticks where there is no spine
#     if 'left' in spines:
#         ax.yaxis.set_ticks_position('left')
#     else:
#         # no yaxis ticks
#         ax.yaxis.set_ticks([])
#     # if 'bottom' in spines:
#     #     ax.xaxis.set_ticks_position('bottom')
#     # else:
#     #     # no xaxis ticks
#     #     ax.xaxis.set_ticks([])
# def adjust_spines_right(ax, spines):
#     for loc, spine in ax.spines.items():
#         if loc in spines:
#             spine.set_position(('outward', 10))
#         # else:
#         #     spine.set_color('none')  # don't draw spine
#     # # turn off ticks where there is no spine
#     # if 'left' in spines:
#     #     ax.yaxis.set_ticks_position('left')
#     # else:
#     #     # no yaxis ticks
#     #     ax.yaxis.set_ticks([])

# startdate, enddate = '2019-01-01','2020-12-31'
# # Fetch traffic data
# traffic_mil = readc40mobility.read_milan(startdate, enddate)
# traffic_ber = readc40mobility.read_berlin(startdate, enddate)
# traffic_ber = traffic_ber.groupby(traffic_ber.index).mean()
# traffic_mil = traffic_mil.groupby(traffic_mil.index).mean()

# # Fetch model/observational data
# gcf_ber = model.loc[model['city']=='Berlin']
# gcf_ber = gcf_ber.groupby([gcf_ber['time'].dt.date]).mean()
# gcf_ber.reset_index(inplace=True)
# gcf_ber = gcf_ber.rename(columns={'time':'Date'})
# gcf_ber['Date'] = pd.to_datetime(gcf_ber['Date'])
# obs_ber = obs_eea.loc[obs_eea['City']=='Berlin']

# # Run XGBoost for Berlin ***with Apple mobility data***
# train_ber, bias_train_ber, obs_train_ber = \
#     prepare_model_obs(obs_ber, gcf_ber, '2019-01-01', '2019-12-31')
# full_ber, bias_full_ber, obs_full_ber = \
#     prepare_model_obs(obs_ber, gcf_ber, '2019-01-01', '2020-06-30')
# no2diff_ber, shaps_ber = run_xgboost(args, train_ber, bias_train_ber, 
#     full_ber, obs_full_ber)
# dat_ber = no2diff_ber.groupby(['Date']).mean().reset_index()
# dat_ber = dat_ber.set_index('Date').resample('1D').mean().rolling(window=7,
#     min_periods=1).mean()

# # Run XGBoost for Berlin ***with in-situ mobility data***
# del gcf_ber['Volume']
# gcf_ber = pd.merge(gcf_ber, traffic_ber['Count'], left_on='Date', 
#     right_index=True)
# gcf_ber = gcf_ber.rename(columns={'Count':'Volume'})

# train_ber_ISM, bias_train_ber_ISM, obs_train_ber_ISM = \
#     prepare_model_obs(obs_ber, gcf_ber, '2019-01-01', '2019-12-31')
# full_ber_ISM, bias_full_ber_ISM, obs_full_ber_ISM = \
#     prepare_model_obs(obs_ber, gcf_ber, '2019-01-01', '2020-06-30')
# no2diff_ber_ISM, shaps_ber_ISM = run_xgboost(args, train_ber_ISM, \
#     bias_train_ber_ISM, full_ber_ISM, obs_full_ber_ISM)
# dat_ber_ISM = no2diff_ber_ISM.groupby(['Date']).mean().reset_index()
# dat_ber_ISM = dat_ber_ISM.set_index('Date').resample('1D').mean().rolling(
#     window=7, min_periods=1).mean()

# # Find SHAP values for in-situ vs Apple mobility data
# shaps_ber_ISM_melt = shaps_ber_ISM.melt(var_name='Feature',
#     value_name='Shap')
# shaps_ber_melt = shaps_ber.melt(var_name='Feature', value_name='Shap')
# # Median value for each feature, used to rank the SHAP values in the plot
# medians_ISM = shaps_ber_ISM_melt.groupby('Feature').median().sort_values(
#     by='Shap',ascending=False).reset_index()
# medians = shaps_ber_melt.groupby('Feature').median().sort_values(
#     by='Shap',ascending=False).reset_index()
# # Reduce data to first 10 number of features
# features_ISM = list(medians_ISM['Feature'].values[:10])
# features = list(medians['Feature'].values[:10])
# medians_ISM = medians_ISM.loc[medians_ISM['Feature'].isin(features_ISM)]
# medians = medians.loc[medians['Feature'].isin(features)]

# # Merge medians DataFrames
# medians_merged = medians.merge(medians_ISM, left_on='Feature', 
#     right_on='Feature')
# medians_merged = medians_merged.rename(columns={'Shap_x':'Apple', 
#     'Shap_y':'ISM'})
    
# fig = plt.figure(figsize=(10,6))
# ax1 = plt.subplot2grid((2,3),(0,0),colspan=3)
# ax1b = ax1.twinx()
# ax2 = plt.subplot2grid((2,3),(1,1),colspan=2)
# axshap = plt.subplot2grid((2,3),(1,0),colspan=1)
# app = '#56B4E9'
# ism = '#E69F00'
# # Traffic count/volume plot
# mobility_ber = mobility.loc[mobility['city']=='Berlin']
# mobility_ber.set_index('time', inplace=True)
# ax1b.plot(mobility_ber.loc['2020-01-13':'2020-06-30']['Volume'], lw=1.5, 
#     ls='-', color=app, zorder=8)
# ax1b.plot(mobility_ber.loc['2019-01-01':'2020-01-12']['Volume'], lw=1.5, 
#     ls='-', color=app, zorder=10)
# ax1.plot(traffic_ber.loc['2019-01-01':'2020-07-01']['Count'], lw=1.5, 
#     ls='-', color=ism)
# ax1.annotate('CSD/\nPride', xy=('2019-07-27',2324), 
#     xytext=('2019-06-20',9500), arrowprops={'arrowstyle': '-|>', 'fc':'k'}, 
#     va='center')
# ax1.annotate('Festival\nof\nLights', xy=('2019-10-20',653), 
#     xytext=('2019-11-20',9500), arrowprops={'arrowstyle': '-|>', 'fc':'k'}, 
#     va='center')
# ax1.axvspan('2020-01-01','2020-06-30', alpha=0.15, fc='grey', zorder=0)
# ax1.set_xlim(['2019-01-01','2020-06-30'])
# ax1.set_xticks(['2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01',
#     '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01', 
#     '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01', 
#     '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', 
#     '2020-05-01', '2020-06-01']) 
# ax1.set_xticklabels(['JAN\n2019', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
#     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', '', 'JAN\n2020', 'FEB', 
#     'MAR', 'APR', 'MAY', 'JUN'], fontsize=8) 
# ax1.spines['top'].set_visible(False)
# ax1b.spines['top'].set_visible(False)
# # Move vertical spines outwards for in-situ observations
# ax1b.spines['left'].set_visible(False)
# adjust_spines(ax1, ['left'])
# ax1.set_ylim([0,35000])
# ax1.set_yticks(np.linspace(0,35000,11))
# ax1.set_yticklabels(['0','','7000','','14000','','21000','','28000',
#     '','35000'])
# ax1.tick_params(axis='y', colors=ism)
# ax1.spines['left'].set_color(ism)
# ax1.set_ylabel('Berlin traffic counts [$\cdot$]', color=ism)
# # Move vertical spines outwards for Apple mobility
# ax1.spines['right'].set_visible(False)
# adjust_spines_right(ax1b, ['right'])
# ax1b.set_ylim([30,130])
# ax1b.set_yticklabels(['30','','50','','70','','90','','110','','130'])
# ax1b.set_yticks(np.linspace(30,130,11))
# ax1b.tick_params(axis='y', colors=app)
# ax1b.spines['right'].set_color(app)
# ax1b.set_ylabel('Apple mobility [%]', fontsize=12, color=app)
# # ML NO2 concentration plot 
# ax2.plot(full_ber.set_index('Date').resample('1D').mean().rolling(
#       window=7,min_periods=1).mean()['NO2'], ls='--', 
#       color='darkgrey', label='GEOS-CF')
# ax2.plot(dat_ber['observed'], lw=1.5, ls='-', color='k', label='Observed')
# ax2.plot(dat_ber['predicted'], lw=1.5, ls='-', color=app, 
#     label='Business-as-usual:\nApple mobility')
# ax2.plot(dat_ber_ISM['predicted'], lw=1.5, ls='-', color=ism, 
#     label='Business-as-usual:\nBerlin traffic counts')
# ax2.set_ylim([0,30])
# ax2.set_ylabel('NO$_{2}$ [ppbv]')
# ax2.yaxis.set_label_position("right")
# ax2.yaxis.tick_right()
# ax2.set_xlim(['2020-01-01','2020-06-30'])
# ax2.set_xticks(['2020-01-01', '2020-01-15', '2020-02-01', 
#     '2020-02-15', '2020-03-01', '2020-03-15', '2020-04-01', 
#     '2020-04-15', '2020-05-01', '2020-05-15', '2020-06-01',
#     '2020-06-15']) 
# ax2.set_xticklabels(['JAN\n2020', '', 'FEB', '', 'MAR', '', 'APR', 
#     '', 'MAY', '', 'JUN', ''], fontsize=8)
# ax2.set_ylim([0,18])
# ax2.axvspan('2020-01-01','2020-06-30', alpha=0.15, fc='grey', zorder=0)
# # # # # SHAP values
# axshap.barh(np.arange(0,len(medians_merged),1), medians_merged['Apple'], 
#     color=app, clip_on=False)
# axshap.barh(np.arange(0,len(medians_merged),1), medians_merged['ISM'], 
#     facecolor='None', edgecolor=ism, linewidth=2, clip_on=False)
# axshap.set_yticks(np.arange(0, len(medians_merged['Feature'].values), 1))
# axshap.set_yticklabels(medians_merged['Feature'].values)
# print('Subplot (b) y-axis labels should have the following order: %s'%(
#     medians_merged['Feature'].values))
# axshap.set_ylim([-1, len(medians_merged)])
# axshap.set_yticks(range(len(medians_merged)))
# axshap.set_yticklabels(['Traffic', 'Northward wind', 'Eastward wind', 
#     'Precipitation', 'O$_{\mathregular{3}}$', 'NO$_{\mathregular{2}}$', 
#     'Boundary layer height', 'Pressure', 'Relative humidity', 
#     'Specific humidity'])
# plt.gca().invert_yaxis()
# axshap.set_xlim([0,1])
# # Hide the right and top spines
# for side in ['right', 'top']:
#     axshap.spines[side].set_visible(False)
# # Subplot titles 
# ax1.set_title(r'(a)', loc='left', fontsize=16)
# axshap.set_title(r'(b)', loc='left', fontsize=16)
# plt.subplots_adjust(wspace=0.3, hspace=0.35, bottom=0.15, right=0.85, 
#     top=0.93)
# # Adjust position of axes
# pos1 = ax2.get_position()
# pos2 = [pos1.x0, pos1.y0, pos1.width+0.05, pos1.height] 
# ax2.set_position(pos2) # Set a new position
# zoom_effect02(ax2, ax1)
# pos1 = axshap.get_position()
# pos2 = [pos1.x0+0.05, pos1.y0, pos1.width-0.04, pos1.height] 
# axshap.set_position(pos2)
# ax2.legend(loc=1, ncol=4, bbox_to_anchor=(1.0, -0.1), frameon=False, 
#     fontsize=14)
# plt.savefig('/Users/ghkerr/Desktop/'+'traffic_berlin_trial.png', dpi=350)