#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:54:57 2021

@author: ghkerr
"""

DIR_MODEL = '/Users/ghkerr/Downloads/'

import pandas as pd
from scipy import stats 
from sklearn.metrics import mean_squared_error
import xgboost as xgb 
import argparse
import numpy as np
import time

def prepare_model_obs(obs, model, start, end):
    """Harmonize GEOS-CF model output (and driving emissions and meteorological 
    variables) with city-averaged NO2 observations and calculate (observed - 
    model) for XGBoost

    Parameters
    ----------
    obs : pandas.core.frame.DataFrame
        Observed NO2 concentrations
    model : pandas.core.frame.DataFrame
        Modeled NO2, meteorology, emissions, and control information (e.g., 
        day, latitude, longitude, etc.)
    start : str
        Start date of measuring period (YYYY-mm-dd format).
    end : str
        End date of measuring period (YYYY-mm-dd format).

    Returns
    -------
    merged : pandas.core.frame.DataFrame
        Merged observation-model dataset
    bias : pandas.core.series.Series
        NO2 bias (observed - modeled)
    obs_conc : pandas.core.series.Series
        Observed NO2 concentrations for the city/period of interest.
    """
    import pandas as pd
    model = model.copy(deep=True)
    # # # # For observations
    # Compute city-wide average 
    obs = obs.groupby(['Date']).mean()
    obs.reset_index(inplace=True)
    obs.rename({'Date': 'ISO8601'}, axis=1, inplace=True)
    # Restrict to specified time window
    obs = obs.loc[(obs['ISO8601']>=start) & (obs['ISO8601']<=end)]
    # # # # For GEOS-CF
    # From Christoph's _read_model function
    SKIPVARS = ['ISO8601','weekday','trendday','location','lat','lon',
        'CLDTT','Q10M','T10M','TS','U10M','V10M','ZPBL',
        'year','month','day','hour']
    # Data columns to be excluded from the machine learning
    # DROPVARS = ['location','original_station_name','lat','lon','unit','year']    
    DROPVARS = ['Latitude', 'Longitude', 'NO',
       'NOy', 'O3', 'CO', 'ACET', 'ALK4', 'ALD2', 'HCHO', 'C2H6', 'C3H8',
       'BCPI', 'BCPO', 'OCPI', 'OCPO', 'EOH', 'DST1', 'DST2', 'DST3', 'DST4',
       'H2O2', 'HNO3', 'HNO4', 'ISOP', 'MACR', 'MEK', 'MVK', 'N2O5', 'NH3',
       'NH4', 'NIT', 'PAN', 'PRPE', 'RCHO', 'SALA', 'SALC', 'SO2', 'SOAP',
       'SOAS', 'TOLU', 'XYLE', 'PM25_RH35_GCC', 'PM25ni_RH35_GCC',
       'PM25su_RH35_GCC', 'PM25ss_RH35_GCC', 'PM25du_RH35_GCC',
       'PM25bc_RH35_GCC', 'PM25oc_RH35_GCC', 'PM25soa_RH35_GCC',
       'PM25_RH35_GOCART', 'EMIS_NO', 'EMIS_CO', 'EMIS_ACET', 'EMIS_ALD2',
       'EMIS_ALK4', 'EMIS_BENZ', 'EMIS_C2H6', 'EMIS_C3H8', 'EMIS_HCHO',
       'EMIS_EOH', 'EMIS_MEK', 'EMIS_NH3', 'EMIS_PRPE', 'EMIS_TOLU',
       'EMIS_XYLE', 'EMIS_ISOP', 'EMIS_BCPI', 'EMIS_BCPO', 'EMIS_OCPI',
       'EMIS_OCPO', 'EMIS_DST1', 'EMIS_DST2', 'EMIS_DST3', 'EMIS_DST4',
       'EMIS_SALA', 'EMIS_SALC', 'EMIS_SO2', 'EMIS_SOAP', 'EMIS_SOAS',
       'EMIS_I2', 'EMIS_CHBr3', 'month', 'day', 'hour', 'weekday', 'trendday']    
    # Scale factors used to scale the model emissions and concentrations,
    # respectively.
    EMISSCAL = 1.0e6*3600.0
    CONCSCAL = 1.0e9
    # Scale concentrations to ppbv (from mol/mol) and emissions to mg/m2/h 
    # (from kg/m2/s)
    for v in model:
        if v in SKIPVARS:
            continue 
        if v == 'TPREC':
            scal = 1.0e6
        elif v == 'PS': 
            scal = 0.01 
        elif 'EMIS_' in v:
            scal = EMISSCAL 
        else:
            scal = CONCSCAL
        model[v] = model[v].values * scal 
    model['ISO8601'] = pd.to_datetime(model['ISO8601'])
    # # # # Merge model and observations
    merged = obs.merge(model, how='inner', on='ISO8601')
    # drop values not needed
    _ = [merged.pop(var) for var in DROPVARS if var in merged]
    # Machine learning algorithm is trained on (observation - 
    # model) difference
    bias = merged['Concentration'] - merged['NO2']
    obs_conc = merged.pop('Concentration')
    return merged, bias, obs_conc

def train(args,Xtrain,Ytrain):
    '''train XGBoost model'''
    Xt = Xtrain.copy()
    Xt.pop('ISO8601')
    train = xgb.DMatrix(Xt,np.array(Ytrain))
    params = {'booster':'gbtree'}
    bst = xgb.train(params,train)
    return bst

def predict(args,bst,Xpredict):
    """make prediction using XGBoost model and return predicted bias and 
    bias-corrected concentration"""
    Xp = Xpredict.copy()
    dates = Xp.pop('ISO8601')
    predict = xgb.DMatrix(Xp)
    predicted_bias = bst.predict(predict)
    predicted_conc = Xpredict['NO2'].values + predicted_bias    
    shap_values = _get_shap_values(args,bst,Xp)
    return predicted_bias, predicted_conc, dates, shap_values

def _get_shap_values(args,bst,X):
    '''Get SHAP values for given xgboost object bst and set of input features X'''
    import shap
    explainer = shap.TreeExplainer(bst)
    shap_array = np.abs(explainer.shap_values(X))
    shap_values = pd.DataFrame(data = shap_array,columns=list(bst.feature_names))
    return shap_values 

def valid(args,bst,Xvalid,Yvalid,instance):
    '''make prediction using XGboost model'''
    bias,conc,dates,shap_values = predict(args,bst,Xvalid)
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    axs[0] = _plot_scatter(axs[0],bias,Yvalid,-60.,60.,'Predicted bias [ppbv]','True bias [ppbv]','Bias')
    axs[1] = _plot_scatter(axs[1],Xvalid['NO2'],Xvalid['NO2'].values+Yvalid,0.,60.,'Model concentration [ppbv]','Observed concentration [ppbv]','Original')
    axs[2] = _plot_scatter(axs[2],conc,Xvalid['NO2'].values+Yvalid,0.,60.,'Model concentration [ppbv]','Observed concentration [ppbv]','Adjusted (XGBoost)')
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()
    return

def run_xgboost(args, merged_train, bias_train, merged_full, obs_conc_full): 
    """Conduct eXtreme Gradient Boosting/XGBoost to predict the bias between 
    modeled (GEOS-CF) and observed NO2. XGBoost leverages GEOS-CF inputs (emissions, 
    meteorology) and outputs (gases, aerosols) as well as covariates (mobility 
    changes, traffic counts) as input features. Input data (for this project, 
    pre-pandemic values) are split in N parts ("folds").  The XGBoost algorithm 
    is trained on N-1 folds with one held back and tested on the held back fold. 
    This is repeated so that each fold of the dataset is given a chance to be 
    the held back test set. The model derived from each fold is then used to 
    predict the (GEOS-CF - observation) NO2 bias for the entire period of 
    interest (pre-pandemic + lockdowns). These N different predictions 
    are output. 

    Parameters
    ----------
    args : argparse.Namespace
        Input arguments
    merged_train : pandas.core.frame.DataFrame
        Inputs for training time period
    bias_train : pandas.core.series.Series
        (Observed - modeled) concentrations for training time period
    merged_full : pandas.core.frame.DataFrame
        Inputs for full time period
    obs_conc_full : pandas.core.series.Series
        Observed concentration for full time period.

    Returns
    -------
    no2diff : pandas.core.frame.DataFrame
        For each XGBoost training, the predicted concentrations are returned
        alongside the observed concentrations (and the bias).
    shaps : pandas.core.frame.DataFrame
        Shapley values for each training
    """
    shap_list = []
    anomalies = []
    N = 6
    for n in range(N):
        # Split into training and validation by splitting into N chunks
        Xsplit = np.array_split(merged_train, N)
        Ysplit = np.array_split(bias_train, N)
        # Set one aside for validation
        Xvalid = Xsplit.pop(n)
        Yvalid = Ysplit.pop(n)
        # Remaining segments form the training data
        Xtrain = pd.concat(Xsplit)
        Ytrain = np.concatenate(Ysplit)
        # Train model
        bst = train(args,Xtrain,Ytrain)
        # Validate 
        #valid(args,bst,Xvalid,Yvalid,n) 
        # Apply bias correction to model output to obtain 'business-as-usual' 
        # estimate and compare this value against observations
        bias_pred, conc_pred, dates, shap_values = predict(args, bst, merged_full)
        anomaly = obs_conc_full - conc_pred    
        pred = pd.DataFrame({'ISO8601':dates, 'predicted':conc_pred,
            'observed':obs_conc_full,'anomaly':anomaly})
        anomalies.append(pred)
        shap_list.append(shap_values)
    # anomalies is a list with individual DataFrames comprised of dates, the
    # predicted concentrations (BCM = BAU modeled NO2 + predicted bias), the 
    # observed NO2 concentrations (this is somewhat of a repeat from 
    # above variables), and the difference between observed and predicted
    # concentrations. Each item in the list corresponds to a different 
    # set of training data
    no2diff = pd.concat(anomalies)    
    # Concatenate Shapely values (e.g., https://medium.com/@gabrieltseng/
    # interpreting-complex-models-with-shap-values-1c187db6ec83). In essence, 
    # Shapely values calculate the importance of a feature by comparing 
    # what a model predicts with and without the feature. All SHAP values 
    # have the same unit (the unit of the prediction space).
    shaps = pd.concat(shap_list)
    return no2diff, shaps

def _plot_scatter(ax,x,y,minval,maxval,xlab,ylab,title):
    '''make scatter plot of XGBoost prediction vs. true values'''
    r,p = stats.pearsonr(x,y)
    nrmse = np.sqrt(mean_squared_error(x,y))/np.std(x)
    mb = np.sum(y-x)/np.sum(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    ax.hexbin(x,y,cmap=plt.cm.gist_earth_r,bins='log')
    ax.set_xlim(minval,maxval)
    ax.set_ylim(minval,maxval)
    ax.plot((0.95*minval,1.05*maxval),(0.95*minval,1.05*maxval),color='grey',linestyle='dashed')
    # regression line
    ax.plot((0.95*minval,1.05*maxval),(intercept+(0.95*minval*slope),intercept+(1.05*maxval*slope)),color='blue',linestyle='dashed')
    ax.set_xlabel(xlab)
    if ylab != '-':
        ax.set_ylabel(ylab)
    istr = 'N = {:,}'.format(y.shape[0])
    _ = ax.text(0.05,0.95,istr,transform=ax.transAxes)
    istr = '{0:.2f}'.format(r**2)
    istr = 'R$^{2}$ = '+istr
    _ = ax.text(0.05,0.90,istr,transform=ax.transAxes)
    istr = 'NRMSE [%] = {0:.2f}'.format(nrmse*100)
    _ = ax.text(0.05,0.85,istr,transform=ax.transAxes)
    _ = ax.set_title(title)
    return ax

def plot_timeseries(no2diff, merged_full):
    """Plot timeseries of modeled NO2 (from GEOS-CF), observed NO2, and the 
    bias-corrected model. 

    Parameters
    ----------
    no2diff : pandas.core.frame.DataFrame
        Predicted concentrations, the observed concentrations, and the bias for
        each XGBoost training.
    merged_full : pandas.core.frame.DataFrame
        Inputs for full time period

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    # Group data by date and average over all N predictions
    dat = no2diff.groupby(['ISO8601']).mean().reset_index()
    dat = dat.set_index('ISO8601').resample('1D').mean().rolling(window=7,
        min_periods=1).mean()
    # Plotting
    fig = plt.figure(figsize=(7,2))
    ax = plt.subplot2grid((1,1),(0,0))
    ax.plot(merged_full.set_index('ISO8601').resample('1D').mean().rolling(
        window=7,min_periods=1).mean()['NO2'], ls='--', 
        color='darkgrey', label='GEOS-CF')
    ax.plot(dat['observed'], '-k', label='Observed')
    ax.plot(dat['predicted'], '--k', label='BCM')
    # Fill red for positive difference between , blue for negative difference
    y1positive=(dat['observed']-dat['predicted'])>0
    y1negative=(dat['observed']-dat['predicted'])<=0
    ax.fill_between(dat.index, dat['predicted'],
        dat['observed'], where=y1positive, color='red', alpha=0.5)
    ax.fill_between(dat.index, dat['predicted'], 
        dat['observed'], where=y1negative, color='blue', alpha=0.5,
        interpolate=True)
    # Legend
    ax.legend(loc=1, ncol=3, bbox_to_anchor=(1.,1.4), frameon=False)
    ax.set_ylim([0, 50])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.set_xlim([dat.index.values.min(), dat.index.values.max()])
    ax.set_ylabel('NO$_{2}$ [ppbv]')
    ax.set_title('London', x=0.05, fontsize=12)
    fig.tight_layout()
    plt.savefig('/Users/ghkerr/Desktop/london_bcm.png', dpi=400)
    plt.show()
    return 

# import datetime as dt
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sys
# sys.path.append('/Users/ghkerr/GW/mobility/')
# import readc40aq
import readc40mobility

# # Load model
# gcf = pd.read_csv(DIR_MODEL+'model.csv', delimiter=',', header=0, 
#     engine='python', parse_dates=['ISO8601'],date_parser=lambda x: 
#     dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

# # # # For Milan 
# Model grid cells corresponding to Milan 
lng_mil = [9., 9.25, 9.5]
lat_mil = [45.25, 45.5, 45.75]
gcf_mil = gcf.loc[(gcf['lon'].isin(lng_mil)) & (gcf['lat'].isin(lat_mil))]
# Compute daily average
gcf_mil = gcf_mil.groupby([gcf_mil['ISO8601'].dt.date]).mean()
gcf_mil.reset_index(inplace=True)
# Load NO2 observations
obs = readc40aq.read_milan('NO2', '2019-01-01', '2020-12-31')
# Load Milan traffic counts and calculate daily average
traffic_mil = readc40mobility.read_milan('2019-01-01', '2020-12-31')
traffic_mil = traffic_mil.drop(['Site'], axis=1).reset_index()
traffic_mil = traffic_mil.rename(columns={'Date':'ISO8601'})
traffic_mil['ISO8601'] = traffic_mil['ISO8601'].dt.date.astype(object)
gcf_mil = traffic_mil.merge(gcf_mil, how='inner', on='ISO8601')

merged_train, bias_train, obs_conc_train = \
    prepare_model_obs(obs, gcf_mil, '2019-01-01', '2019-12-31')
merged_full, bias_full, obs_conc_full = \
    prepare_model_obs(obs, gcf_mil, '2019-01-01', '2020-06-30')
no2diff, shaps = run_xgboost(args, merged_train, bias_train, merged_full, obs_conc_full)
plot_timeseries(no2diff, merged_full)

# # # # # For Los Angeles 
# lat_los = [34., 34.25, 33.75]
# lng_los = [-118., -118.25, -118.5]
# gcf_los = gcf.loc[(gcf['lon'].isin(lng_los)) & (gcf['lat'].isin(lat_los))]
# gcf_los = gcf_los.groupby([gcf_los['ISO8601'].dt.date]).mean()
# gcf_los.reset_index(inplace=True)
# obs = readc40aq.read_losangeles('NO2', '2019-01-01', '2020-12-31')
# merged_train, bias_train, obs_conc_train = \
#     prepare_model_obs(obs, gcf_los, '2019-01-01', '2019-12-31')
# merged_full, bias_full, obs_conc_full = \
#     prepare_model_obs(obs, gcf_los, '2019-01-01', '2020-06-30')
# no2diff, shaps = run_xgboost(args, merged_train, bias_train, merged_full, obs_conc_full)
# plot_timeseries(no2diff, merged_full)

# # # # # For London
# lat_lon = [51.5]
# lng_lon = [-0.25, 0., 0.25]
# gcf_lon = gcf.loc[(gcf['lon'].isin(lng_lon)) & (gcf['lat'].isin(lat_lon))]
# gcf_lon = gcf_lon.groupby([gcf_lon['ISO8601'].dt.date]).mean()
# gcf_lon.reset_index(inplace=True)
# obs = readc40aq.read_london('NO2', '2019-01-01', '2020-12-31')
# merged_train, bias_train, obs_conc_train = \
#     prepare_model_obs(obs, gcf_lon, '2019-01-01', '2019-12-31')
# merged_full, bias_full, obs_conc_full = \
#     prepare_model_obs(obs, gcf_lon, '2019-01-01', '2020-06-30')
# no2diff, shaps = run_xgboost(args, merged_train, bias_train, merged_full, obs_conc_full)
# plot_timeseries(no2diff, merged_full)






NFEATURES = 15
df = shaps.melt(var_name='Feature',value_name='Shap')
# median value for each feature, used to rank the SHAP values in the boxplot
medians = df.groupby('Feature').median().sort_values(by='Shap',ascending=False).reset_index()
# reduce data to first NFEATURES number of features
features = list(medians['Feature'].values[:NFEATURES])
medians = medians.loc[medians['Feature'].isin(features)]

fig = plt.figure()
ax= plt.subplot2grid((1,1),(0,0))
ax.bar(np.arange(0,len(medians),1), medians['Shap'])
ax.set_xticks(np.arange(0, len(medians['Feature'].values), 1))
ax.set_xticklabels(medians['Feature'].values, rotation=90)



# WASHINGTON DC
# [ -77.  ,   38.75],
# [ -77.  ,   39.  ],
# [ -76.75,   39.  ],
# [ -76.75,   39.25],
# [ -74.25,   40.75],
# [ -74.  ,   40.75],
# [ -73.75,   40.75],
# [ -73.75,   41.  ],

# MADRID
# [  -4.  ,   40.25],
# [  -3.75,   40.25],
# [  -3.75,   40.5 ],
# [  -3.5 ,   40.25],
# [  -3.5 ,   40.5 ],

# LONDON 
# [  -0.25,   51.5 ],
# [   0.  ,   51.5 ],
# [   0.25,   51.5 ],

# PARIS
# [   2.25,   48.75],
# [   2.25,   49.  ],
# [   2.5 ,   48.75],
# [   2.5 ,   49.  ],
# [   2.75,   48.75],

# WUHAN
# [ 114.25,   30.5 ],
# [ 114.25,   30.75],
# [ 114.5 ,   30.5 ],

# BEIJING
# [ 116.25,   39.75],
# [ 116.25,   40.  ],
# [ 116.5 ,   39.75],
# [ 116.5 ,   40.  ]])







# import argparse
# def parse_args():
#     p = argparse.ArgumentParser(description='Undef certain variables')
#     p.add_argument('-o','--obsfile',type=str,help='observation file',default='https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/obs.csv')
#     p.add_argument('-m','--modfile',type=str,help='model file',default='https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/model.csv')
#     p.add_argument('-c','--cities',type=str,nargs="+",help='city names',default='NewYork')
#     p.add_argument('-n','--nsplit',type=int,help='number of cross-fold validations',default=8)
#     p.add_argument('-v','--validate',type=int,help='make validation figures (1=yes; 0=no)?',default=0)
#     p.add_argument('-s','--shap',type=int,help='plot shap values for each city (1=yes; 0=no)?',default=0)
#     p.add_argument('-mn','--minnobs',type=int,help='minimum number of required observations (for training)',default=8760)
#     return p.parse_args()


# args = parse_args()


