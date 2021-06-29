#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:54:57 2021

@author: ghkerr
"""
DIR = '/Users/ghkerr/GW/'
DIR_MODEL = DIR+'data/GEOSCF/'

import pandas as pd
from scipy import stats 
from sklearn.metrics import mean_squared_error
import xgboost as xgb 
import argparse
import numpy as np
import time
import numpy.ma as ma

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
    model = model.copy(deep=True)
    # # # # # For observations
    # # Compute city-wide average 
    # obs = obs.groupby(['Date']).mean()
    # obs.reset_index(inplace=True)
    # obs.rename({'Date': 'ISO8601'}, axis=1, inplace=True)
    # Restrict to specified time window
    obs = obs.loc[(obs['Date']>=start) & (obs['Date']<=end)]
    # # # # For GEOS-CF
    # From Christoph's _read_model function
    SKIPVARS = ['Date', 'latitude', 'longitude', 'PM25', 'city', 'CLDTT', 
        'Q', 'Q10M', 'Q2M', 'RH', 'SLP', 'T', 'T10M', 'T2M', 'TS', 'U', 
        'U10M', 'U2M', 'V', 'V10M', 'V2M', 'ZPBL', 'Volume']
    # Data columns to be excluded from the machine learning
    DROPVARS = ['latitude', 'longitude', 'Q10M', 'Q2M', 'T10M', 'T2M',
        'TS', 'U10M', 'U2M', 'V10M', 'V2M']  
    # Scale model concentrations
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
        else:
            scal = CONCSCAL
        model[v] = model[v].values*scal
    # Merge model and observations
    merged = obs.merge(model, how='inner', on='Date')
    # Drop values not needed
    _ = [merged.pop(var) for var in DROPVARS if var in merged]
    # Machine learning algorithm is trained on (observation - 
    # model) difference
    bias = merged['Concentration'] - merged['NO2']
    obs_conc = merged.pop('Concentration')
    return merged, bias, obs_conc

def train(args,Xtrain,Ytrain):
    '''train XGBoost model'''
    Xt = Xtrain.copy()
    Xt.pop('Date')
    # Because of the occassional NaN in the observations, xgb.DMatrix will 
    # throw the following error: 
    # Degrees of freedom <= 0 for slice.
    # Mean of empty slice
    # To ameliorate this, replace NaNs with arbitrary values (from 
    # https://github.com/dmlc/xgboost/issues/822)
    # Ytrain = np.nan_to_num(Ytrain, nan=-999.)
    train = xgb.DMatrix(Xt, Ytrain)#, missing=-999.)
    params = {'booster':'gbtree'}
    bst = xgb.train(params,train)
    return bst

def predict(args,bst,Xpredict):
    """make prediction using XGBoost model and return predicted bias and 
    bias-corrected concentration"""
    Xp = Xpredict.copy()
    dates = Xp.pop('Date')
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
    rorig : list
        Correlation coefficient calculated between model and observations, [1]
    fac2orig : list
        Factor-of-2 fraction calculated between model and observations, [1]    
    mfborig : list
        Mean fractional bias calculated between model and observations, [1]
    rtrain : list
        Correlation coefficient calculated between training bias-corrected
        model and observations, [N]   
    fac2train : list
        Factor-of-2 fraction calculated between training bias-corrected
        model and observations, [N]
    mfbtrain : list
        Mean fractional bias calculated between training bias-corrected
        model and observations, [N]
    rvalid : list
        Correlation coefficient calculated between valdiation bias-corrected
        model and observations, [N]    
    fac2valid : list
        Factor-of-2 fraction calculated between valdiation bias-corrected
        model and observations, [N]    
    mfbvalid: list
        Mean fractional bias calculated between valdiation bias-corrected
        model and observations, [N]        
    """
    shap_list = []
    features = []
    anomalies = []
    # The following lists will be filled with evaluations metrics comparing the 
    # raw (non-bias-corrected) model with observations 
    rorig, fac2orig, mfborig = [], [], []
    pm = ma.masked_invalid(merged_train['NO2'].values)
    om = ma.masked_invalid(bias_train.values)    
    msk = (~pm.mask & ~om.mask)
    p = merged_train['NO2'].values[msk]
    o = (merged_train['NO2'].values[msk]+bias_train.values[msk])
    # Correlation coefficient 
    rorig.append(np.corrcoef(p,o)[0,1])
    # Factor-of-2 fraction (atmosphere.copernicus.eu/sites/default/files/
    # 2018-11/2_3rd_ECCC_NOAA_ECMWF_v06.pdf)
    fac2 = (p/o)
    fac2 = np.where((fac2>0.5) & (fac2<2.))[0]
    fac2orig.append(len(fac2)/len(p))
    # Mean fractional bias (see"Fractional bias" in Table 1 on
    # https://rmets.onlinelibrary.wiley.com/doi/10.1002/asl.125)
    mfborig.append(2*(np.nansum(p-o)/np.nansum(p+o)))
    del p, o
    # The following lists will be filled with values of machine learning 
    # evaluation metrics during each iteration of k-means cross validation 
    # (for both training and testing datasets)
    rtrain, fac2train, mfbtrain = [], [], []
    rvalid, fac2valid, mfbvalid = [], [], []    
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
        # Evaluation metrics for training data; note that "conc_train" and 
        # "conc_valid" are the observed concentrations for the training and
        # validation datasets, respectively. Xtrain['NO2'] is the model NO2
        # and Ytrain is the machine-learned bias, so Xtrain['NO2']+Ytrain
        # would represent the bias corrected model NO2
        biast, conct, datest, svt = predict(args, bst, Xtrain)
        p = Xtrain['NO2'].values+Ytrain
        o = conct
        pm = ma.masked_invalid(p)
        om = ma.masked_invalid(o)
        msk = (~pm.mask & ~om.mask)        
        rtrain.append(np.corrcoef(p[msk],o[msk])[0,1])
        fac2 = (p[msk]/o[msk])
        fac2 = np.where((fac2>0.5) & (fac2<2.))[0]
        fac2train.append(len(fac2)/len(p[msk]))
        mfbtrain.append(2*(np.nansum(p[msk]-o[msk])/np.nansum(p[msk]+o[msk])))
        del p, o
        # Evaluation metrics for testing (validation) data
        biasv, concv, datesv, svv = predict(args, bst, Xvalid)
        p = Xvalid['NO2'].values+Yvalid
        o = concv
        pm = ma.masked_invalid(p)
        om = ma.masked_invalid(o)
        msk = (~pm.mask & ~om.mask)
        # Depending on how much missing data a city has, there might be a case
        # where a particular held-out fold/set has no observations (an example
        # of this is Zurich validation dataset for n=1...in this case, all 
        # observations for the validation set are NaN.)
        if len(o[msk])!=0:
            rvalid.append(np.corrcoef(p[msk],o[msk])[0,1])
            fac2 = (p[msk]/o[msk])
            fac2 = np.where((fac2>0.5) & (fac2<2.))[0]
            fac2valid.append(len(fac2)/len(p[msk]))
            mfbvalid.append(2*(np.nansum(p[msk]-o[msk])/np.nansum(
                p[msk]+o[msk]))) 
        del p, o
        ## Validate 
        #valid(args,bst,Xvalid,Yvalid,n) 
        # Apply bias correction to model output to obtain 'business-as-usual' 
        # estimate and compare this value against observations
        bias_pred, conc_pred, dates, shap_values = predict(args, bst, 
            merged_full)
        anomaly = obs_conc_full - conc_pred    
        pred = pd.DataFrame({'Date':dates, 'predicted':conc_pred,
            'observed':obs_conc_full,'anomaly':anomaly})
        anomalies.append(pred)
        shap_list.append(shap_values)
        features.append(merged_full)
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
    features = pd.concat(features)
    return (no2diff, shaps, features, rorig, fac2orig, mfborig, rtrain, fac2train, 
        mfbtrain, rvalid, fac2valid, mfbvalid)

# def _plot_scatter(ax,x,y,minval,maxval,xlab,ylab,title):
#     '''make scatter plot of XGBoost prediction vs. true values'''
#     r,p = stats.pearsonr(x,y)
#     nrmse = np.sqrt(mean_squared_error(x,y))/np.std(x)
#     mb = np.sum(y-x)/np.sum(x)
#     slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#     ax.hexbin(x,y,cmap=plt.cm.gist_earth_r,bins='log')
#     ax.set_xlim(minval,maxval)
#     ax.set_ylim(minval,maxval)
#     ax.plot((0.95*minval,1.05*maxval),(0.95*minval,1.05*maxval),color='grey',linestyle='dashed')
#     # regression line
#     ax.plot((0.95*minval,1.05*maxval),(intercept+(0.95*minval*slope),intercept+(1.05*maxval*slope)),color='blue',linestyle='dashed')
#     ax.set_xlabel(xlab)
#     if ylab != '-':
#         ax.set_ylabel(ylab)
#     istr = 'N = {:,}'.format(y.shape[0])
#     _ = ax.text(0.05,0.95,istr,transform=ax.transAxes)
#     istr = '{0:.2f}'.format(r**2)
#     istr = 'R$^{2}$ = '+istr
#     _ = ax.text(0.05,0.90,istr,transform=ax.transAxes)
#     istr = 'NRMSE [%] = {0:.2f}'.format(nrmse*100)
#     _ = ax.text(0.05,0.85,istr,transform=ax.transAxes)
#     _ = ax.set_title(title)
#     return ax

# def plot_timeseries(no2diff, merged_full):
#     """Plot timeseries of modeled NO2 (from GEOS-CF), observed NO2, and the 
#     bias-corrected model. 

#     Parameters
#     ----------
#     no2diff : pandas.core.frame.DataFrame
#         Predicted concentrations, the observed concentrations, and the bias for
#         each XGBoost training.
#     merged_full : pandas.core.frame.DataFrame
#         Inputs for full time period

#     Returns
#     -------
#     None
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.dates as mdates
#     # Group data by date and average over all N predictions
#     dat = no2diff.groupby(['Date']).mean().reset_index()
#     dat = dat.set_index('Date').resample('1D').mean().rolling(window=7,
#         min_periods=1).mean()
#     # Plotting
#     fig = plt.figure(figsize=(7,2))
#     ax = plt.subplot2grid((1,1),(0,0))
#     ax.plot(merged_full.set_index('Date').resample('1D').mean().rolling(
#         window=7,min_periods=1).mean()['NO2'], ls='--', 
#         color='darkgrey', label='GEOS-CF')
#     ax.plot(dat['observed'], '-k', label='Observed')
#     ax.plot(dat['predicted'], '--k', label='BCM')
#     # Fill red for positive difference between , blue for negative difference
#     y1positive=(dat['observed']-dat['predicted'])>0
#     y1negative=(dat['observed']-dat['predicted'])<=0
#     ax.fill_between(dat.index, dat['predicted'],
#         dat['observed'], where=y1positive, color='red', alpha=0.5)
#     ax.fill_between(dat.index, dat['predicted'], 
#         dat['observed'], where=y1negative, color='blue', alpha=0.5,
#         interpolate=True)
#     # Legend
#     ax.legend(loc=1, ncol=3, bbox_to_anchor=(1.,1.4), frameon=False)
#     ax.set_ylim([0, 50])
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
#     ax.set_xlim([dat.index.values.min(), dat.index.values.max()])
#     ax.set_ylabel('NO$_{2}$ [ppbv]')
#     ax.set_title('London', x=0.05, fontsize=12)
#     fig.tight_layout()
#     plt.savefig('/Users/ghkerr/Desktop/london_bcm.png', dpi=400)
#     plt.show()
#     return 

# def valid(args,bst,Xvalid,Yvalid,instance):
#     '''make prediction using XGboost model'''
#     bias,conc,dates,shap_values = predict(args,bst,Xvalid)
#     fig, axs = plt.subplots(1,3,figsize=(15,5))
#     axs[0] = _plot_scatter(axs[0],bias,Yvalid,-60.,60.,
#         'Predicted bias [ppbv]','True bias [ppbv]','Bias')
#     axs[1] = _plot_scatter(axs[1],Xvalid['NO2'],Xvalid['NO2'
#         ].values+Yvalid,0.,60.,'Model concentration [ppbv]','Observed concentration [ppbv]','Original')
#     axs[2] = _plot_scatter(axs[2],conc,Xvalid['NO2'].values+Yvalid,0.,60.,'Model concentration [ppbv]','Observed concentration [ppbv]','Adjusted (XGBoost)')
#     plt.tight_layout(rect=[0,0.03,1,0.95])
#     plt.show()
#     return

import argparse
def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-o','--obsfile',type=str,help='observation file',default='https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/obs.csv')
    p.add_argument('-m','--modfile',type=str,help='model file',default='https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/COVID_NO2/examples/model.csv')
    p.add_argument('-c','--cities',type=str,nargs="+",help='city names',default='NewYork')
    p.add_argument('-n','--nsplit',type=int,help='number of cross-fold validations',default=8)
    p.add_argument('-v','--validate',type=int,help='make validation figures (1=yes; 0=no)?',default=0)
    p.add_argument('-s','--shap',type=int,help='plot shap values for each city (1=yes; 0=no)?',default=0)
    p.add_argument('-mn','--minnobs',type=int,help='minimum number of required observations (for training)',default=8760)
    return p.parse_args()
args = parse_args()


