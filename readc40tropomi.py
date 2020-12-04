#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:49:08 2020

@author: ghkerr
"""
DIR = '/Users/ghkerr/GW/'
DIR_TROPOMI = DIR+'data/tropomi/'
DIR_GEOGRAPHY = DIR+'data/geography/'
DIR_TYPEFACE = '/Users/ghkerr/Library/Fonts/'
DIR_FIGS = DIR+'mobility/figs/'
# Load custom font
import sys
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
    
def read_citytropomi(city, atype, period):
    """Open regridded TROPOMI NO2 files for city/region and time period of 
    interest.
    
    Parameters
    ----------
    city : str
        City/region of interest
    atype : str
        Based on output file format, an 'avg' should be supplied when the 
        time period-averaged file is desired, and '' should be passed when the
        daily regridded NO2 is preferred
    period : str
        Time period of interest, YYYYMMDD-YYYYMMDD format where the first value
        is the start date and the second value is the end date

    Returns
    -------
    lng : numpy.ma.core.MaskedArray
        Longitude for city/region of interest, units of degrees east
    lat : numpy.ma.core.MaskedArray
        Latitude for city/region of interest, units of degrees north    
    no2 : numpy.ma.core.MaskedArray
        Tropospheric column NO2 for city/region of interest, units of molecules
        per cm^2
    """
    import netCDF4 as nc
    infile = nc.Dataset(DIR_TROPOMI+'S5P_NO2_%s_%s%s_0.01grid_QA75.nc'%(
        city,atype,period), 'r')
    lng = infile['longitude'][:]
    lat = infile['latitude'][:]
    no2 = infile['nitrogendioxide_tropospheric_column'][:]
    # Convert from native units to molec/cm2 
    no2 = no2*6.02214e+19
    return lng, lat, no2

# # Load baseline periods
# baseline = '20190313-20190613'
# lng_auc, lat_auc, no2_auc_base = read_citytropomi('auckland', 'avg', baseline)
# lng_ber, lat_ber, no2_ber_base = read_citytropomi('berlin', 'avg', baseline)
# lng_lon, lat_lon, no2_lon_base = read_citytropomi('london', 'avg', baseline)
# lng_los, lat_los, no2_los_base = read_citytropomi('losangeles', 'avg', baseline)
# lng_mex, lat_mex, no2_mex_base = read_citytropomi('mexicocity', 'avg', baseline)
# lng_mil, lat_mil, no2_mil_base = read_citytropomi('milan', 'avg', baseline)
# lng_san, lat_san, no2_san_base = read_citytropomi('santiago', 'avg', baseline)

# # Load lockdown periods
# lockdown = '20200313-20200613'
# lng_auc, lat_auc, no2_auc_lock = read_citytropomi('auckland', 'avg', lockdown)
# lng_ber, lat_ber, no2_ber_lock = read_citytropomi('berlin', 'avg', lockdown)
# lng_lon, lat_lon, no2_lon_lock = read_citytropomi('london', 'avg', lockdown)
# lng_los, lat_los, no2_los_lock = read_citytropomi('losangeles', 'avg', lockdown)
# lng_mex, lat_mex, no2_mex_lock = read_citytropomi('mexicocity', 'avg', lockdown)
# lng_mil, lat_mil, no2_mil_lock = read_citytropomi('milan', 'avg', lockdown)
# lng_san, lat_san, no2_san_lock = read_citytropomi('santiago', 'avg', lockdown)

"""TROPOMI NO2 CHANGE IN SEVEN CITIES"""
# import numpy as np
# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon
# import cartopy.crs as ccrs
# import shapely.ops as so
# from cartopy.io import shapereader
# from fiona.crs import from_epsg
# # from pyproj import Transformer, CRS, Proj
# # import fiona
# # from fiona.crs import from_epsg
# # from pyproj import Transformer
# # from cartopy.feature import ShapelyFeature
# # from cartopy.io.shapereader import Reader
# import matplotlib
# import matplotlib.gridspec as gridspec
# import cartopy.feature as cfeature
# # Initialize figure, axes
# fig = plt.figure(figsize=(10,8))
# gs = gridspec.GridSpec(3, 3)
# axworld = plt.subplot(gs[1, :2], 
#     projection=ccrs.InterruptedGoodeHomolosine(central_longitude=0.))
# axlos = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
# axmex = plt.subplot(gs[2, 0], projection=ccrs.PlateCarree())
# axlon = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
# axber = plt.subplot(gs[0, 2], projection=ccrs.PlateCarree())
# axmil = plt.subplot(gs[1, 2], projection=ccrs.PlateCarree())
# axsan = plt.subplot(gs[2, 1], projection=ccrs.PlateCarree())
# axauc = plt.subplot(gs[2, 2], projection=ccrs.PlateCarree())

# # # # # # World map 
# transform = ccrs.PlateCarree()._as_mpl_transform(axworld)
#  # Los Angeles
# axworld.plot(-118.2437, 34.052, 'ko', transform=ccrs.PlateCarree(), 
#     markersize=3)
# axworld.annotate('(a)', (-141, 32), ha='left', xycoords=transform)
# # Mexico City
# axworld.plot(-99.1332, 19.4326, 'ko', markersize=3, 
#     transform=ccrs.PlateCarree()) 
# axworld.annotate('(e)', (-90, 20), ha='left', xycoords=transform)
#  # Milan 
# axworld.plot(9.1900, 45.4642, 'ko', markersize=3, 
#     transform=ccrs.PlateCarree())
# axworld.annotate('(d)', (9, 31), ha='left', xycoords=transform)
# # Santiago
# axworld.plot(-70.6693, -33.4489, 'ko', markersize=3, 
#     transform=ccrs.PlateCarree()) 
# axworld.annotate('(f)', (-63, -38), ha='left', xycoords=transform)
# # London
# axworld.plot(-0.127, 51.5074, 'ko', markersize=3, 
#     transform=ccrs.PlateCarree()) 
# axworld.annotate('(b)', (-29, 50), ha='left', xycoords=transform)
# # Berlin
# axworld.plot(13.4050, 52.5200, 'ko', markersize=3, 
#     transform=ccrs.PlateCarree()) 
# axworld.annotate('(c)', (17, 56), ha='left', xycoords=transform)
# # Auckland
# axworld.plot(174.7625, -36.84836, 'ko', markersize=3, 
#     transform=ccrs.PlateCarree()) 
# axworld.annotate('(g)', (155, -30), ha='left', xycoords=transform)
# axworld.set_global()
# ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
#     edgecolor='face', facecolor='lightgrey')
# axworld.add_feature(ocean_10m)

# # # # # Los Angeles
# # https://geohub.lacity.org/datasets/lacounty::county-boundaries/data?
# # geometry=-122.278%2C33.172%2C-114.807%2C34.766
# filename = DIR_GEOGRAPHY+'losangeles/County_Boundaries-shp/County_Boundaries.shp'
# shp = shapereader.Reader(filename)
# losangeles = shp.geometries()
# losangeles = list(losangeles)
# rec = shp.records()
# rec = list(rec)
# losangeles = so.cascaded_union([losangeles[1],losangeles[8],losangeles[9]])
# axlos.add_feature(ocean_10m)
# axlos.add_geometries([losangeles], ccrs.PlateCarree(), edgecolor='k', 
#     facecolor='None')
# # cmaplos = plt.get_cmap('bwr', 10)
# # normlos = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # cmaplos = plt.get_cmap('bwr', 10)
# # normlos = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # mb = axlos.pcolormesh(lng_los, lat_los, (no2_los_lock-no2_los_base), 
# #     cmap=cmaplos, norm=normlos, transform=ccrs.PlateCarree())
# cmaplos = plt.get_cmap('bwr', 10)
# normlos = matplotlib.colors.Normalize(vmin=-50, vmax=50)
# mb = axlos.pcolormesh(lng_los, lat_los, ((no2_los_lock-no2_los_base)/
#     no2_los_base)*100., cmap=cmaplos, norm=normlos, 
#     transform=ccrs.PlateCarree())
# axlos.set_extent([losangeles.bounds[0]-0.1, losangeles.bounds[2]+0.1, 
#     losangeles.bounds[1]-0.1, losangeles.bounds[3]+0.1], ccrs.PlateCarree())
# axlos.set_aspect('auto')

# # # # # Mexico City
# # https://datacatalog.worldbank.org/dataset/urban-cities-mexico
# # filename = DIR_GEOGRAPHY+'mexicocity/native/'+'LocalidadUrbana.shp'
# # shp = shapereader.Reader(filename)
# # cdmx = shp.geometries()
# # cdmx = list(cdmx)
# # rec = shp.records()
# # rec = list(rec)
# # # Fetch the NOMMUN codes of each shape; these correspond to the urban 
# # # areas in Mexico. state[566] is CDMX
# # state = []
# # for s in np.arange(0, len(cdmx), 1):
# #     state.append(rec[s].attributes['NOMMUN'])
# # shape = fiona.open(filename)
# # original = CRS(shape.crs)
# # destination = CRS('EPSG:4326')
# # transformer = Transformer.from_crs(original, destination)
# # with fiona.open('cdmx_transformed.shp', 'w', 'ESRI Shapefile', 
# #     shape.schema.copy(), crs=from_epsg(4326)) as output:
# #     for feat in shape:
# #         long = np.array(feat['geometry']['coordinates'][0])[:,0]
# #         lat = np.array(feat['geometry']['coordinates'][0])[:,1]    
# #         y, x = transformer.transform(long, lat)
# #         # change only the coordinates of the feature
# #         feat['geometry']['coordinates'] = [list(zip(x,y))]
# #         output.write(feat)
# filename = DIR_GEOGRAPHY+'mexicocity/'+'cdmx_transformed.shp'
# shp = shapereader.Reader(filename)
# cdmx = shp.geometries()
# cdmx = list(cdmx)[566]
# axmex.add_geometries([cdmx], ccrs.PlateCarree(), edgecolor='k', 
#     facecolor='None')
# # # For saturated colorbar
# # cmapmex = plt.get_cmap('bwr', 10)
# # normmex = matplotlib.colors.Normalize(vmin=-3e15, vmax=3e15)
# # # For uniform colorbar
# # cmapmex = plt.get_cmap('bwr', 10)
# # normmex = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # mb = axmex.pcolormesh(lng_mex, lat_mex, no2_mex_lock-no2_mex_base, 
# #     cmap=cmapmex, norm=normmex, transform=ccrs.PlateCarree())
# cmapmex = plt.get_cmap('bwr', 10)
# normmex = matplotlib.colors.Normalize(vmin=-50, vmax=50)
# mb = axmex.pcolormesh(lng_mex, lat_mex, ((no2_mex_lock-no2_mex_base)/
#     no2_mex_base)*100., cmap=cmapmex, norm=normmex, 
#     transform=ccrs.PlateCarree())
# axmex.set_extent([cdmx.bounds[0]-0.1, cdmx.bounds[2]+0.1, 
#     cdmx.bounds[1]-0.1, cdmx.bounds[3]+0.1], ccrs.PlateCarree())
# axmex.set_aspect('auto')

# # # # # Santiago
# # ; n.b., only need admin1 https://data.humdata.org/dataset/
# # chile-administrative-level-0-country-1-region-region
# filename = DIR_GEOGRAPHY+'santiago/'+'chl_admbnda_adm1_bcn2018.shp'
# shp = shapereader.Reader(filename)
# santiago = shp.geometries()
# # Querying rec = list(shp.records()) shows that santiago[-1] is the 
# # Region Metropolitana de Santiago 
# santiago = list(santiago)[-1]
# axsan.add_geometries([santiago], ccrs.PlateCarree(), edgecolor='k', 
#     facecolor='None')
# # cmapsan = plt.get_cmap('bwr', 10)
# # normsan = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # cmapsan = plt.get_cmap('bwr', 10)
# # normsan = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # mb = axsan.pcolormesh(lng_san, lat_san, no2_san_lock-no2_san_base, 
# #     cmap=cmapsan, norm=normsan, transform=ccrs.PlateCarree())
# cmapsan = plt.get_cmap('bwr', 10)
# normsan = matplotlib.colors.Normalize(vmin=-50, vmax=50)
# mb = axsan.pcolormesh(lng_san, lat_san, ((no2_san_lock-no2_san_base)/
#     no2_san_base)*100., cmap=cmapsan, norm=normsan, 
#     transform=ccrs.PlateCarree())
# axsan.set_extent([santiago.bounds[0]-0.1, santiago.bounds[2]+0.1, 
#     santiago.bounds[1]-0.1, santiago.bounds[3]+0.1], ccrs.PlateCarree())
# axsan.set_aspect('auto')

# # # # # Berlin
# # https://maps.princeton.edu/catalog/tufts-berlin-bezirke-boroughs01
# filename = DIR_GEOGRAPHY+'berlin/'+\
#     'GISPORTAL_GISOWNER01_BERLIN_BEZIRKE_BOROUGHS01.shp'
# shp = shapereader.Reader(filename)
# berlin = shp.geometries()
# berlin = list(berlin)
# # Cascaded union can work on a list of shapes, adapted from 
# # stackoverflow.com/questions/34475431/plot-unions-of-polygons-in-matplotlib
# berlin = so.cascaded_union(berlin)
# axber.add_geometries(berlin, ccrs.PlateCarree(), edgecolor='k', 
#     facecolor='None')
# # cmapber = plt.get_cmap('bwr', 10)
# # normber = matplotlib.colors.Normalize(vmin=-1e15, vmax=1e15)
# # cmapber = plt.get_cmap('bwr', 10)
# # normber = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # mb = axber.pcolormesh(lng_ber, lat_ber, no2_ber_lock-no2_ber_base, 
# #     cmap=cmapber, norm=normber, transform=ccrs.PlateCarree())
# cmapber = plt.get_cmap('bwr', 10)
# normber = matplotlib.colors.Normalize(vmin=-50, vmax=50)
# mb = axber.pcolormesh(lng_ber, lat_ber, ((no2_ber_lock-no2_ber_base)/
#     no2_ber_base)*100., cmap=cmapber, norm=normber, 
#     transform=ccrs.PlateCarree())
# axber.set_extent([berlin.bounds[0]-0.08, berlin.bounds[2]+0.08, 
#     berlin.bounds[1]-0.01, berlin.bounds[3]+0.02], ccrs.PlateCarree())
# axber.set_aspect('auto')

# # # # # London
# # https://data.london.gov.uk/dataset/inner-and-outer-london-
# # boundaries-london-plan-consultation-2009
# # filename = DIR_GEOGRAPHY+'london/native/'+\
# #     'lp-consultation-oct-2009-inner-outer-london.shp'
# # shape = fiona.open(filename)
# # # Define projections; taken from https://gis.stackexchange.com/questions/
# # # 121441/convert-shapely-polygon-coordinates
# # original = CRS(27700)
# # destination = CRS('EPSG:4326')
# # transformer = Transformer.from_crs(original, destination)
# # with fiona.open('london_transformed.shp', 'w', 'ESRI Shapefile', 
# #     shape.schema.copy(), crs=from_epsg(3857)) as output:
# #     for feat in shape:
# #         long = np.array(feat['geometry']['coordinates'][0])[:,0]
# #         lat = np.array(feat['geometry']['coordinates'][0])[:,1]    
# #         y, x = transformer.transform(long, lat)
# #         # change only the coordinates of the feature
# #         feat['geometry']['coordinates'] = [list(zip(x,y))]
# #         output.write(feat)
# filename = DIR_GEOGRAPHY+'london/'+'london_transformed.shp'
# shp = shapereader.Reader(filename)
# london = shp.geometries()
# london = list(london)
# london = so.cascaded_union(london) 
# axlon.add_geometries([london], ccrs.PlateCarree(), edgecolor='k', 
#     facecolor='None')
# # cmaplon = plt.get_cmap('bwr', 10)
# # normlon = matplotlib.colors.Normalize(vmin=-6e15, vmax=6e15)
# # cmaplon = plt.get_cmap('bwr', 10)
# # normlon = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # mb = axlon.pcolormesh(lng_lon, lat_lon, no2_lon_lock-no2_lon_base, 
# #     cmap=cmaplon, norm=normlon, transform=ccrs.PlateCarree())
# cmaplon = plt.get_cmap('bwr', 10)
# normlon = matplotlib.colors.Normalize(vmin=-50, vmax=50)
# mb = axlon.pcolormesh(lng_lon, lat_lon, ((no2_lon_lock-no2_lon_base)/
#     no2_lon_base)*100., cmap=cmaplon, norm=normlon, 
#     transform=ccrs.PlateCarree())
# axlon.set_extent([london.bounds[0]-0.1, london.bounds[2]+0.1, 
#     london.bounds[1]-0.1, london.bounds[3]+0.1], ccrs.PlateCarree())
# axlon.set_aspect('auto')

# # # # # Milan 
# # https://maps.princeton.edu/catalog/stanford-pd101kz6162
# filename = DIR_GEOGRAPHY+'milan/'+'pd101kz6162.shp'
# shp = shapereader.Reader(filename)
# milan = shp.geometries()
# milan = list(milan)
# axmil.add_geometries(milan, ccrs.PlateCarree(), edgecolor='k', 
#     facecolor='None')
# # cmapmil = plt.get_cmap('bwr', 10)
# # normmil = matplotlib.colors.Normalize(vmin=-5e15, vmax=5e15)
# # cmapmil = plt.get_cmap('bwr', 10)
# # normmil = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # mb = axmil.pcolormesh(lng_mil, lat_mil, no2_mil_lock-no2_mil_base, 
#     # cmap=cmapmil, norm=normmil, transform=ccrs.PlateCarree())
# cmapmil = plt.get_cmap('bwr', 10)
# normmil = matplotlib.colors.Normalize(vmin=-50, vmax=50)
# mb = axmil.pcolormesh(lng_mil, lat_mil, ((no2_mil_lock-no2_mil_base)/
#     no2_mil_base)*100., cmap=cmapmil, norm=normmil, 
#     transform=ccrs.PlateCarree())
# axmil.set_extent([milan[0].bounds[0]-0.1, milan[0].bounds[2]+0.1, 
#     milan[0].bounds[1]-0.1, milan[0].bounds[3]+0.1], ccrs.PlateCarree())
# axmil.set_aspect('auto')

# # # # # Auckland 
# # https://geodata.lib.berkeley.edu/catalog/stanford-pv084qp8629
# filename = DIR_GEOGRAPHY+'auckland/'+'NZL_adm1.shp'
# shp = shapereader.Reader(filename)
# auckland_all = shp.geometries()
# auckland = list(auckland_all)[0]
# # For the ocean around Auckland, the built-in Cartopy coastline doesn't
# # jive with the Auckland shapefile, so create a custom land-ocean mask 
# # by creating another rectangular polygon outside the Auckland 
# # shapefile and then calculate the difference
# poly = Polygon([(166.7,-40.6), (166.7, -32.5), (179., -32.5), 
#     (179.,-40.6)])
# shp = shapereader.Reader(filename)
# ocean_auc = shp.geometries()
# ocean_auc = so.cascaded_union(list(ocean_auc))
# axauc.add_geometries([poly.difference(ocean_auc)], ccrs.PlateCarree(), 
#     edgecolor='face', facecolor='lightgrey')
# axauc.add_geometries(auckland, ccrs.PlateCarree(), edgecolor='k', 
#     facecolor='None', zorder=12)
# # cmapauc = plt.get_cmap('bwr', 10)
# # normauc = matplotlib.colors.Normalize(vmin=-1e15, vmax=1e15)
# # cmapauc = plt.get_cmap('bwr', 10)
# # normauc = matplotlib.colors.Normalize(vmin=-4e15, vmax=4e15)
# # mb = axauc.pcolormesh(lng_auc, lat_auc, no2_auc_lock-no2_auc_base, 
# #     cmap=cmapauc, norm=normauc, transform=ccrs.PlateCarree())
# cmapauc = plt.get_cmap('bwr', 10)
# normauc = matplotlib.colors.Normalize(vmin=-50, vmax=50)
# mb = axauc.pcolormesh(lng_auc, lat_auc, ((no2_auc_lock-no2_auc_base)/
#     no2_auc_base)*100., cmap=cmapauc, norm=normauc, 
#     transform=ccrs.PlateCarree())
# axauc.set_extent([auckland.bounds[0]-0.08, auckland.bounds[2]-0.2, 
#     auckland.bounds[1]-0.1, auckland.bounds[3]-0.1], ccrs.PlateCarree())
# axauc.set_aspect('auto')

# # # # # Set titles
# plt.subplots_adjust(wspace=0.4)
# axlos.set_title('(a) Los Angeles', loc='left')
# axlon.set_title('(b) London', loc='left')
# axber.set_title('(c) Berlin', loc='left')
# axmil.set_title('(d) Milan', loc='left')
# axmex.set_title('(e) Mexico City', loc='left')
# axsan.set_title('(f) Santiago', loc='left')
# axauc.set_title('(g) Auckland', loc='left')

# # # # # Add colorbars
# # Los Angeles
# cax = fig.add_axes([axlos.get_position().x1+0.01, axlos.get_position().y0, 
#     0.015, (axlos.get_position().y1-axlos.get_position().y0)])
# cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmaplos, norm=normlos, 
#     spacing='proportional', orientation='vertical', extend='both')
# # cax.yaxis.offsetText.set_visible(False)
# # cax.text(x=0.0, y=1.03, s=r'$\times$ 10$^{\mathregular{15}}$', 
# #     transform=cax.transAxes)
# cax.text(x=0.0, y=1.03, s='[%]', transform=cax.transAxes)
# # London
# cax = fig.add_axes([axlon.get_position().x1+0.01, axlon.get_position().y0, 
#     0.015, (axlon.get_position().y1-axlon.get_position().y0)])
# cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmaplon, norm=normlon, 
#     spacing='proportional', orientation='vertical', extend='both')   
# # cax.yaxis.offsetText.set_visible(False)
# # cax.text(x=0.0, y=1.03, s=r'$\times$ 10$^{\mathregular{15}}$', 
# #     transform=cax.transAxes)
# cax.text(x=0.0, y=1.03, s='[%]', transform=cax.transAxes)
# # Mexico City
# cax = fig.add_axes([axmex.get_position().x1+0.01, axmex.get_position().y0, 
#     0.015, (axmex.get_position().y1-axmex.get_position().y0)])
# cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmapmex, norm=normmex, 
#     spacing='proportional', orientation='vertical', extend='both')   
# # cax.yaxis.offsetText.set_visible(False)
# # cax.text(x=0.0, y=1.03, s=r'$\times$ 10$^{\mathregular{15}}$', 
# #     transform=cax.transAxes)
# cax.text(x=0.0, y=1.03, s='[%]', transform=cax.transAxes)
# # Santiago 
# cax = fig.add_axes([axsan.get_position().x1+0.01, axsan.get_position().y0, 
#     0.015, (axsan.get_position().y1-axsan.get_position().y0)])
# cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmapsan, norm=normsan, 
#     spacing='proportional', orientation='vertical', extend='both')   
# # cax.yaxis.offsetText.set_visible(False)
# # cax.text(x=0.0, y=1.03, s=r'$\times$ 10$^{\mathregular{15}}$', 
# #     transform=cax.transAxes)
# cax.text(x=0.0, y=1.03, s='[%]', transform=cax.transAxes)
# # Berlin
# cax = fig.add_axes([axber.get_position().x1+0.01, axber.get_position().y0, 
#     0.015, (axber.get_position().y1-axber.get_position().y0)])
# cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmapber, norm=normber, 
#     spacing='proportional', orientation='vertical', extend='both')   
# # cax.yaxis.offsetText.set_visible(False)
# # cax.text(x=0.0, y=1.03, s=r'$\times$ 10$^{\mathregular{15}}$', 
# #     transform=cax.transAxes)
# cax.text(x=0.0, y=1.03, s='[%]', transform=cax.transAxes)
# # Milan
# cax = fig.add_axes([axmil.get_position().x1+0.01, axmil.get_position().y0, 
#     0.015, (axmil.get_position().y1-axmil.get_position().y0)])
# cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmapmil, norm=normmil, 
#     spacing='proportional', orientation='vertical', extend='both')   
# # cax.yaxis.offsetText.set_visible(False)
# # cax.text(x=0.0, y=1.03, s=r'$\times$ 10$^{\mathregular{15}}$', 
# #     transform=cax.transAxes)
# cax.text(x=0.0, y=1.03, s='[%]', transform=cax.transAxes)
# # Auckland 
# cax = fig.add_axes([axauc.get_position().x1+0.01, axauc.get_position().y0, 
#     0.015, (axauc.get_position().y1-axauc.get_position().y0)])
# cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmapauc, norm=normauc, 
#     spacing='proportional', orientation='vertical', extend='both')   
# # cax.yaxis.offsetText.set_visible(False)
# # cax.text(x=0.0, y=1.03, s=r'$\times$ 10$^{\mathregular{15}}$', 
# #     transform=cax.transAxes)
# cax.text(x=0.0, y=1.03, s='[%]', transform=cax.transAxes)
# plt.savefig(DIR_FIGS+'tropomino2_cities_percentchange.png', dpi=500)

""" PLOT COMPARISON OF LOCKDOWN AND BASELINE NO2 CHANGES FROM OBSERVATIONS
AND TROPOMI """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sys
# sys.path.append('/Users/ghkerr/GW/mobility/')
# import readc40aq
# sys.path.append('/Users/ghkerr/phd/utils/')
# from geo_idx import geo_idx

# # Load NO2 observations
# obsno2_los_base = readc40aq.read_losangeles('NO2', '2019-03-13', '2019-06-13')
# obsno2_los_lock = readc40aq.read_losangeles('NO2', '2020-03-13', '2020-06-13')
# obsno2_mex_base = readc40aq.read_mexicocity('NO2', '2019-03-13', '2019-06-13')
# obsno2_mex_lock = readc40aq.read_mexicocity('NO2', '2020-03-13', '2020-06-13')
# obsno2_san_base = readc40aq.read_santiago('NO2', '2019-03-13', '2019-06-13')
# obsno2_san_lock = readc40aq.read_santiago('NO2', '2020-03-13', '2020-06-13')
# obsno2_lon_base = readc40aq.read_london('NO2', '2019-03-13', '2019-06-13')
# obsno2_lon_lock = readc40aq.read_london('NO2', '2020-03-13', '2020-06-13')
# obsno2_mil_base = readc40aq.read_milan('NO2', '2019-03-13', '2019-06-13')
# obsno2_mil_lock = readc40aq.read_milan('NO2', '2020-03-13', '2020-06-13')
# obsno2_ber_base = readc40aq.read_berlin('NO2', '2019-03-13', '2019-06-13')
# obsno2_ber_lock = readc40aq.read_berlin('NO2', '2020-03-13', '2020-06-13')
# obsno2_auc_base = readc40aq.read_auckland('NO2', '2019-03-13', '2019-06-13')
# obsno2_auc_lock = readc40aq.read_auckland('NO2', '2020-03-13', '2020-06-13')

# # Time average baseline and lockdown periods and rename NO2 columns to 
# # indicate baseline or lockdown periods (otherwise there are issues with 
# # merging)
# obsno2_los_base = obsno2_los_base.groupby(['Site']).mean()
# obsno2_los_base.rename({'Concentration': 'Concentration_base'}, axis=1, 
#     inplace=True)
# obsno2_los_lock = obsno2_los_lock.groupby(['Site']).mean()
# obsno2_los_lock.rename({'Concentration': 'Concentration_lock'}, axis=1, 
#     inplace=True)
# obsno2_mex_base = obsno2_mex_base.groupby(['Site']).mean()
# obsno2_mex_base.rename({'Concentration': 'Concentration_base'}, axis=1, 
#     inplace=True)
# obsno2_mex_lock = obsno2_mex_lock.groupby(['Site']).mean()
# obsno2_mex_lock.rename({'Concentration': 'Concentration_lock'}, axis=1, 
#     inplace=True)
# obsno2_san_base = obsno2_san_base.groupby(['Site']).mean()
# obsno2_san_base.rename({'Concentration': 'Concentration_base'}, axis=1, 
#     inplace=True)
# obsno2_san_lock = obsno2_san_lock.groupby(['Site']).mean()
# obsno2_san_lock.rename({'Concentration': 'Concentration_lock'}, axis=1, 
#     inplace=True)
# obsno2_lon_base = obsno2_lon_base.groupby(['Site']).mean()
# obsno2_lon_base.rename({'Concentration': 'Concentration_base'}, axis=1, 
#     inplace=True)
# obsno2_lon_lock = obsno2_lon_lock.groupby(['Site']).mean()
# obsno2_lon_lock.rename({'Concentration': 'Concentration_lock'}, axis=1, 
#     inplace=True)
# obsno2_mil_base = obsno2_mil_base.groupby(['Site']).mean()
# obsno2_mil_base.rename({'Concentration': 'Concentration_base'}, axis=1, 
#     inplace=True)
# obsno2_mil_lock = obsno2_mil_lock.groupby(['Site']).mean()
# obsno2_mil_lock.rename({'Concentration': 'Concentration_lock'}, axis=1, 
#     inplace=True)
# obsno2_ber_base = obsno2_ber_base.groupby(['Site']).mean()
# obsno2_ber_base.rename({'Concentration': 'Concentration_base'}, axis=1, 
#     inplace=True)
# obsno2_ber_lock = obsno2_ber_lock.groupby(['Site']).mean()
# obsno2_ber_lock.rename({'Concentration': 'Concentration_lock'}, axis=1, 
#     inplace=True)
# obsno2_auc_base = obsno2_auc_base.groupby(['Site']).mean()
# obsno2_auc_base.rename({'Concentration': 'Concentration_base'}, axis=1, 
#     inplace=True)
# obsno2_auc_lock = obsno2_auc_lock.groupby(['Site']).mean()
# obsno2_auc_lock.rename({'Concentration': 'Concentration_lock'}, axis=1, 
#     inplace=True)

# # Exclude observations with no coordinate information 
# obsno2_los_base = obsno2_los_base.dropna(subset=['Latitude', 'Longitude'])
# obsno2_los_lock = obsno2_los_lock.dropna(subset=['Latitude', 'Longitude'])
# obsno2_mex_base = obsno2_mex_base.dropna(subset=['Latitude', 'Longitude'])
# obsno2_mex_lock = obsno2_mex_lock.dropna(subset=['Latitude', 'Longitude'])
# obsno2_san_base = obsno2_san_base.dropna(subset=['Latitude', 'Longitude'])
# obsno2_san_lock = obsno2_san_lock.dropna(subset=['Latitude', 'Longitude'])
# obsno2_lon_base = obsno2_lon_base.dropna(subset=['Latitude', 'Longitude'])
# obsno2_lon_lock = obsno2_lon_lock.dropna(subset=['Latitude', 'Longitude'])
# obsno2_mil_base = obsno2_mil_base.dropna(subset=['Latitude', 'Longitude'])
# obsno2_mil_lock = obsno2_mil_lock.dropna(subset=['Latitude', 'Longitude'])
# obsno2_ber_base = obsno2_ber_base.dropna(subset=['Latitude', 'Longitude'])
# obsno2_ber_lock = obsno2_ber_lock.dropna(subset=['Latitude', 'Longitude'])
# obsno2_auc_base = obsno2_auc_base.dropna(subset=['Latitude', 'Longitude'])
# obsno2_auc_lock = obsno2_auc_lock.dropna(subset=['Latitude', 'Longitude'])

# # Merge DataFrames
# obsno2_los = pd.merge(obsno2_los_base, obsno2_los_lock, left_index=True, 
#     right_index=True)
# obsno2_mex = pd.merge(obsno2_mex_base, obsno2_mex_lock, left_index=True, 
#     right_index=True)
# obsno2_lon = pd.merge(obsno2_lon_base, obsno2_lon_lock, left_index=True, 
#     right_index=True)
# obsno2_san = pd.merge(obsno2_san_base, obsno2_san_lock, left_index=True, 
#     right_index=True)
# obsno2_mil = pd.merge(obsno2_mil_base, obsno2_mil_lock, left_index=True, 
#     right_index=True)
# obsno2_ber = pd.merge(obsno2_ber_base, obsno2_ber_lock, left_index=True, 
#     right_index=True)
# obsno2_auc = pd.merge(obsno2_auc_base, obsno2_auc_lock, left_index=True, 
#     right_index=True)

# # Find collocated TROPOMI retrivals for each city
# for obscity, lat, lng, city_base, city_lock in zip(
#     [obsno2_los, obsno2_mex, obsno2_san, obsno2_lon, obsno2_mil, obsno2_ber, 
#     obsno2_auc], 
#     [lat_los, lat_mex, lat_san, lat_lon, lat_mil, lat_ber, lat_auc], 
#     [lng_los, lng_mex, lng_san, lng_lon, lng_mil, lng_ber, lng_auc],
#     [no2_los_base, no2_mex_base, no2_san_base, no2_lon_base, no2_mil_base,
#     no2_ber_base, no2_auc_base],
#     [no2_los_lock, no2_mex_lock, no2_san_lock, no2_lon_lock, no2_mil_lock,
#     no2_ber_lock, no2_auc_lock]):
#     # Add empty columns to merged observation DataFrame for city with 
#     # TROPOMI baseline/lockdown levels at each site 
#     obscity['TROPOMI_base'] = np.nan
#     obscity['TROPOMI_lock'] = np.nan
#     for index, row in obscity.iterrows():
#         # For a given AQ site in a city, find the closest lat/lon in the 
#         # regridded TROPOMI product
#         lat_close = geo_idx(row['Latitude_x'], lat)
#         lng_close = geo_idx(row['Longitude_x'], lng)
#         # Collocated TROPOMI NO2
#         row['TROPOMI_base'] = city_base[lat_close, lng_close]
#         row['TROPOMI_lock'] = city_lock[lat_close, lng_close]

# # Initialize figure, axes
# fig = plt.figure(figsize=(11,5))
# axlos = plt.subplot2grid((3,7),(0,0),rowspan=3)
# axmex = plt.subplot2grid((3,7),(0,1),rowspan=3)
# axsan = plt.subplot2grid((3,7),(0,2),rowspan=3)
# axlon = plt.subplot2grid((3,7),(0,3),rowspan=3)
# axmil = plt.subplot2grid((3,7),(0,4),rowspan=3)
# axber = plt.subplot2grid((3,7),(0,5),rowspan=3)
# axauc = plt.subplot2grid((3,7),(0,6),rowspan=3)
# # Set titles
# axlos.set_title('(a) Los Angeles', loc='left')
# axmex.set_title('(b) Mexico City', loc='left')
# axsan.set_title('(c) Santiago', loc='left')
# axlon.set_title('(d) London', loc='left')
# axmil.set_title('(e) Milan', loc='left')
# axber.set_title('(f) Berlin', loc='left')
# axauc.set_title('(g) Auckland', loc='left')
# # Twin axes for observations
# axlost = axlos.twinx()
# axmext = axmex.twinx()
# axsant = axsan.twinx()
# axlont = axlon.twinx()
# axmilt = axmil.twinx()
# axbert = axber.twinx()
# axauct = axauc.twinx()
# # Define color schemes (use the vibrant color scheme on 
# # https://personal.sron.nl/~pault/)
# color_obs = 'k'
# color_tropomi = '#0077BB'
# # # Plot TROPOMI 
# for city, ax in zip([obsno2_los, obsno2_mex, obsno2_san, obsno2_lon, 
#     obsno2_mil, obsno2_ber, obsno2_auc], [axlos, axmex, axsan, axlon, 
#     axmil, axber, axauc]):
#     ax.errorbar([1,2], [city.mean()['TROPOMI_base'],city.mean()[
#         'TROPOMI_lock']], yerr=[city.std()['TROPOMI_base'], 
#         city.std()['TROPOMI_lock']], fmt='ko', ecolor='k', capthick=2)
#     ax.axvspan(0, 2.5, facecolor=color_tropomi, alpha=0.2)
#     # Calculate percentage change 
#     pc = ((city.mean()['TROPOMI_lock']-city.mean()['TROPOMI_base'])/
#         city.mean()['TROPOMI_base'])*100.
#     ax.text(0.25, 0.02, '%d%%'%pc, ha='center',
#         transform=ax.transAxes, fontsize=14, color=color_tropomi, 
#         fontweight='bold')    
# # Plot observations
# i = 0
# for city, ax in zip([obsno2_los, obsno2_mex, obsno2_san, obsno2_lon, 
#     obsno2_mil, obsno2_ber, obsno2_auc], [axlost, axmext, axsant, axlont, 
#     axmilt, axbert, axauct]):
#     ax.errorbar([3,4], [city.mean()['Concentration_base'],city.mean()[
#         'Concentration_lock']], yerr=[city.std()['Concentration_base'], 
#         city.std()['Concentration_lock']], fmt='ko', ecolor='k', capthick=2)
#     ax.axvspan(2.5, 5, facecolor=color_obs, alpha=0.12) 
#     pc = ((city.mean()['Concentration_lock']-city.mean()[
#         'Concentration_base'])/city.mean()['Concentration_base'])*100.
#     if i==6:
#         ax.text(0.75, 0.23, '%d%%'%pc, ha='center', transform=ax.transAxes, 
#             fontsize=14, color=color_obs, fontweight='bold')

#     else: 
#         ax.text(0.75, 0.02, '%d%%'%pc, ha='center', transform=ax.transAxes, 
#             fontsize=14, color=color_obs, fontweight='bold')
#     i = i+1
# # Add legends to correspond to which half of plot corresponds to 
# # TROPOMI versus observations
# axlos.text(0.13, 0.735, 'TROPOMI', ha='left',
#     transform=axlos.transAxes, fontsize=14, color=color_tropomi, 
#     rotation=270, fontweight='bold')
# axlos.text(0.6, 0.687, 'Observations', ha='left',
#     transform=axlos.transAxes, fontsize=14, color=color_obs, 
#     rotation=270, fontweight='bold')
# # Set axis limits and labels
# for ax in [axlos, axmex, axsan, axlon, axmil, axber, axauc]:
#     ax.set_ylim([0e16, 1.5e16])
#     ax.set_xticks([1, 2, 3, 4])
#     ax.set_xticklabels([])
#     ax.set_yticks([0,0.3e16,0.6e16,0.9e16,1.2e16,1.5e16])
#     ax.set_yticklabels([])
# for ax in [axlost, axmext, axsant, axlont, axmilt, axbert, axauct]:
#     ax.set_xlim([0.5, 4.5])
#     ax.set_ylim([0, 80]) 
#     ax.set_yticks([0,16,32,48,64,80])
#     ax.set_yticklabels([])    
# axlos.set_xticklabels(['Baseline', 'Lockdown', 'Baseline', 'Lockdown'],
#     rotation=45, ha='right')
# axlos.set_yticklabels(['0','3','6','9','12','15 x 10$^{15}$\nmolec cm$^{-2}$'])
# axauct.set_yticklabels(['0','16','32','48','64','80 ppbv'])
# plt.subplots_adjust(left=0.09, right=0.93, top=0.94)
# plt.savefig(DIR_FIGS+'tropomino2_obsno2_comparison.png', dpi=500)


# # pollutant = 'PM2.5'    
# startdate = '2019-01-01'
# enddate = '2020-06-30'
# pollutant = 'NO2'
# no2_auckland = read_auckland(pollutant, startdate, enddate)
# no2_santiago = read_santiago(pollutant, startdate, enddate)
# no2_berlin = read_berlin(pollutant, startdate, enddate)
# no2_cdmx = read_mexicocity(pollutant, startdate, enddate)


# # https://clauswilke.com/dataviz/color-pitfalls.html
# color_auckland = '#E69F00'
# color_santiago = '#56B4E9'
# color_berlin = '#009E73'
# color_cdmx = '#CC79A7'
# color_london = '#0072B2'
# color_la = '#D55E00'

# color_auckland = '#1b9e77'
# color_santiago = '#d95f02'
# color_berlin = '#7570b3'
# color_cdmx = '#e7298a'
# # '#66a61e
# # '#a6761d
# # '#e6ab02


# fig = plt.figure(figsize=(8,6))
# ax1 = plt.subplot2grid((3,3),(0,0), colspan=3)
# ax2 = plt.subplot2grid((3,3),(1,0), colspan=3)
# ax3 = plt.subplot2grid((3,3),(2,0), colspan=3)


# startdate = '2019-01-01'
# enddate = '2020-06-30'
# # for ax, pollutant in zip([ax1,ax2,ax3],['NO2','O3','PM2.5']):
#     # Load pollutants
    # auckland = read_auckland(pollutant, startdate, enddate)
#     santiago = read_santiago(pollutant, startdate, enddate)
#     berlin = read_berlin(pollutant, startdate, enddate)
#     cdmx = read_mexicocity(pollutant, startdate, enddate)
#     # Compute citywide averages 
#     auckland_ca = auckland.groupby([auckland.index]).mean()
#     santiago_ca = santiago.groupby([santiago.index]).mean()
#     berlin_ca = berlin.groupby([berlin.index]).mean()
#     cdmx_ca = cdmx.groupby([cdmx.index]).mean()
#     # Make index datetime
#     for df in [auckland_ca, santiago_ca, berlin_ca, cdmx_ca]:
#         df.index = pd.to_datetime(df.index)
#     # Calculate March-June 2019 average; window_size is the length of window 
#     # for which the simple moving average will be calculated
#     avg_start = '2019-03-01'
#     avg_end = '2019-06-30'
#     window_size = 28
#     auckland_2019avg = auckland_ca.loc[avg_start:avg_end]['Concentration'].mean()
#     santiago_2019avg = santiago_ca.loc[avg_start:avg_end]['Concentration'].mean()
#     berlin_2019avg = berlin_ca.loc[avg_start:avg_end]['Concentration'].mean()
#     cdmx_2019avg = cdmx_ca.loc[avg_start:avg_end]['Concentration'].mean()
#     # Calculate percentage difference
#     auckland_ca['Concentration'] = ((auckland_ca['Concentration']-
#         auckland_2019avg)/auckland_2019avg)*100.
#     santiago_ca['Concentration'] = ((santiago_ca['Concentration']-
#         santiago_2019avg)/santiago_2019avg)*100.
#     berlin_ca['Concentration'] = ((berlin_ca['Concentration']-
#         berlin_2019avg)/berlin_2019avg)*100.
#     cdmx_ca['Concentration'] = ((cdmx_ca['Concentration']-
#         cdmx_2019avg)/cdmx_2019avg)*100.
#     # Determine rolling mean of percentage difference
#     auckland_ca['Concentration_SMA'] = auckland_ca['Concentration'
#         ].rolling(window=window_size).mean()
#     santiago_ca['Concentration_SMA'] = santiago_ca['Concentration'
#         ].rolling(window=window_size).mean()
#     berlin_ca['Concentration_SMA'] = berlin_ca['Concentration'
#         ].rolling(window=window_size).mean()
#     cdmx_ca['Concentration_SMA'] = cdmx_ca['Concentration'
#         ].rolling(window=window_size).mean()
#     ax.plot(auckland_ca['Concentration_SMA']['2020-01-01':'2020-06-01'], 
#         color=color_auckland, label='Auckland')
#     ax.plot(santiago_ca['Concentration_SMA']['2020-01-01':'2020-06-01'], 
#         color=color_santiago, label='Santiago')
#     ax.plot(berlin_ca['Concentration_SMA']['2020-01-01':'2020-06-01'], 
#         color=color_berlin, label='Berlin')
#     ax.plot(cdmx_ca['Concentration_SMA']['2020-01-01':'2020-06-01'], 
#         color=color_cdmx, label='Mexico City')
#     # Plot Percent change = 0% line
#     ax.axhline(y=0, color='darkgrey', linestyle='--', zorder=0)
#     ax.set_xlim([pd.to_datetime('2020-01-01'), pd.to_datetime('2020-06-01')])
#     # Plot start of lockdown 
#     ax.plot(pd.to_datetime('2020-03-26'), auckland_ca.loc['2020-03-26'
#         ]['Concentration_SMA'], 'o', markeredgecolor=color_auckland, 
#         markerfacecolor='w', markersize=5)
#     ax.plot(pd.to_datetime('2020-03-26'), santiago_ca.loc['2020-03-26'
#         ]['Concentration_SMA'], 'o', markeredgecolor=color_santiago, 
#         markerfacecolor='w', markersize=5)
#     ax.plot(pd.to_datetime('2020-03-23'), berlin_ca.loc['2020-03-23'
#         ]['Concentration_SMA'], 'o', markeredgecolor=color_berlin, 
#         markerfacecolor='w', markersize=5)
#     ax.plot(pd.to_datetime('2020-03-23'), cdmx_ca.loc['2020-03-23'
#         ]['Concentration_SMA'], 'o', markeredgecolor=color_cdmx, 
#         markerfacecolor='w', markersize=5)    
    
# plt.legend(ncol=4, bbox_to_anchor=(0.9, -0.25), frameon=False)
# ax1.set_ylabel('NO$_{2}$ [%]')
# ax2.set_ylabel('O$_{3}$ [%]')
# ax3.set_ylabel('PM$_{2.5}$ [%]')
# plt.subplots_adjust(hspace=0.6)
# plt.savefig('/Users/ghkerr/Desktop/obs_pc.png', dpi=500)














