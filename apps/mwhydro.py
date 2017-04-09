from netCDF4 import Dataset
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import glob

nc_base = 'data/mw-hydro_v01_2.5-deg_201610'

def load_area(nc_path):
    with Dataset(nc_path, mode='r') as ds:
        lats, lons, ice_cov = ds.variables['lat'][:], \
            ds.variables['lon'][:],\
            ds.variables['sea-ice_cover'][:]
        ice_cov_unit = ds.variables['sea-ice_cover'].units
        title = ds.title
    return lats, lons, ice_cov, ice_cov_unit, title

def draw_area(lats, lons, ice_cov, ice_cov_unit):
    lat_0, lon_0 = lats.mean(), lons.mean()
    bmap = Basemap(\
#        width=12000, height=5000,
        resolution='l', projection='robin',
        lat_0=lat_0, lon_0=lon_0)

    lons_2d, lats_2d = np.meshgrid(lons, lats)
    xi, yi = bmap(lons_2d, lats_2d)
    cs = bmap.pcolor(xi, yi, np.squeeze(ice_cov))
    bmap.drawparallels(np.arange(-80., 81., 60,), labels=[1,0,1,0], fontsize=7)
    bmap.drawmeridians(np.arange(0, 360, 60,), labels=[0,0,0,1], fontsize=7)
    bmap.drawcoastlines()

    cbar = bmap.colorbar(cs, location='bottom', pad='2%')
    cbar.set_label(ice_cov_unit)

def draw_ice_cov():
    fig = plt.figure()
    fig.tight_layout()
    nc_fmt = '{}/{}'.format(nc_base, 'mw-hydro_v01_2.5-deg_ice_*.nc')
    cnt = 1
    for nc in glob.glob(nc_fmt):
        lats, lons, ice_cov, ice_cov_unit, title = load_area(nc)
        ax = fig.add_subplot(2, 1, cnt)
        ax.set_title(title)
        draw_area(lats, lons, ice_cov, ice_cov_unit)
        cnt += 1
    plt.show()

draw_ice_cov()

