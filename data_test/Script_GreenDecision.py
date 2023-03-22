# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:50:25 2023

@author: calga
"""


# Carica file input ed estrai variabile

NETCDF_mfdataset = xr.open_mfdataset(simpath+'/Concentration_file_water_ug_L_fin.nc')
NETCDF_mfdataset=NETCDF_mfdataset.rename({'lat': 'latitude','lon': 'longitude'}) # Change name of dimentions lon, lat if needed
# NETCDF_mfdataset.__setitem__('time',NETCDF_mfdataset.time + np.timedelta64(30,'D')) # Shift time coordinate from 02/07AM to 02/08 AM for emissions TODO make new files


# Find spatial resolution of dataset
lat_resultion_data = NETCDF_mfdataset.latitude 
lat_resol_Wat1 = np.array(abs(lat_resultion_data[0]-lat_resultion_data[1]))


lon_resultion_data = NETCDF_mfdataset.longitude
lon_resol_Wat1 = np.array(abs(lon_resultion_data[0]-lon_resultion_data[1]))


lat_grid_m = np.array([6.371e6 * lat_resol * (2 * np.pi) / 360]) # fixed

lon_grid_m =  np.array([(6.371e6 * (np.cos(2 * (np.pi) * lo[i] / 360)) * lon_resol * (2 * np.pi) / 360)])  # 6.371e6: radius of Earth in m


Bathimetry_mfdataset_Wat1 = xr.open_mfdataset(simpath+'/Bathimetry_conc.nc')
# print(Bathimetry_mfdataset)
Bathimetry_conc_Wat1 = Bathimetry_mfdataset_Wat1.elevation
Bathimetry_conc_Wat1 = Bathimetry_conc_Wat1.where((Bathimetry_conc_Wat1.longitude > long_min) & (Bathimetry_conc_Wat1.longitude < long_max) &
                                (Bathimetry_conc_Wat1.latitude > lat_min) & (Bathimetry_conc_Wat1.latitude < lat_max), drop=True)


Bathimetry_conc = np.array([(Bathimetry_data.sel(latitude=la[i],longitude=lo[i],method='nearest'))]) # m

pixel_volume = Bathimetry_conc * lon_grid_m * lat_grid_m
# concentration is ug/L, volume is m: m3 * 1e3 = L
mass_ug = (data[i] * (pixel_volume * 1e3))



