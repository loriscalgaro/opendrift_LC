#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Thu Jul  9 12:43:02 2020

# @author: manuel
# """

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os as os
import opendrift
#import glob
from datetime import date
from datetime import datetime
from datetime import timedelta
from opendrift.readers import reader_global_landmask
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.chemicaldrift import ChemicalDrift
from opendrift.readers.unstructured import shyfem
from opendrift.readers import reader_shape
from opendrift.readers.reader_constant import Reader as ConstantReader

def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)




    
#%% Specify parameters for simulation
### Choose chemical compound 

chemical_compound="Benzo-a-pyrene"

### Set degradation
degradation=True
ReactionMode = "OverallRateConstants"
### Set volatilization
volatilization=True
### Set partitioning and dissociation
DissociationMode = 'nondiss'
### Set time interval for simulation 
step=12 # 3                   # simulation time step in hours
Time_step_output = (60*60*24) # simulation output time step in seconds
simulated_days=4         # days of simulation: should be two days longer than the emissions
### Set region for simulation
region="NA_Total"
long_min = 12.0
long_max = 15.35
lat_min = 44.35
lat_max = 45.8

mass_element_ug=1e8    # maximum mass of elements (micrograms)
number_of_elements_max=None #Max number of elements estimated for all the simulation is to be written here  
number_of_elements=None # Number of elements produced for each grid cell


strandingON=False        # 
offline=True             # offline with local NCDF files  


# Period of emissions
Time_emiss_START = np.datetime64('2018-08-02T00:00:00.000000000')
Time_emiss_END = np.datetime64('2018-08-04T00:00:00.000000000')

print("emissions start on: ", to_datetime(Time_emiss_START))
print("emissions end on: ", to_datetime(Time_emiss_END))

emiss_duration = Time_emiss_END - Time_emiss_START
emiss_duration = emiss_duration.astype('timedelta64[D]')
print("emissions: ", emiss_duration)

if simulated_days < (((to_datetime(Time_emiss_END).date() - to_datetime(Time_emiss_START).date()).days)+2): # Difference in days between start and end of shippin emissions + 2 days of safety margin
  print("WARNING: Emissions are not two days shorter then the simulation! ")
else:
  print("Simulation is at least two days longer than emissions")

emission_days = int(Time_emiss_END - Time_emiss_START)/8.64e13 # difference between START and END of emissions in ns / ns in a day 
print("emission_days = ", emission_days, "days" )
print("simulated_days = ", simulated_days, "days" )



#%% Specify output filenames and folders

# TODO Change path of input files here
simpath='C:/Users/calga/Documents/GitHub/opendrift_LC/data_test/' 
# TODO Change path of output files here
simname=os.path.basename("D:/ChemicalDrift_output/")[0:-3]+strandingON*"-str"+degradation*"-deg"+volatilization*"-vol"+"-"+region+"-"+chemical_compound+"-"+ReactionMode+"-"+DissociationMode+"-step_"+str(step)+"h-sim_"+str(simulated_days)+"d-em_"+str(emission_days)+"d"

simname=simname.translate(str.maketrans({'(':'-',')':'-'}))

simtime=datetime.now().strftime("%Y%m%d-%H%M%S")
# TODO Change path of output files here
simoutfolder="D:/ChemicalDrift_output/"
simoutputpath=simoutfolder+simtime+simname

os.makedirs(simoutputpath)

#%% Define model and configuration parameters

o = ChemicalDrift(loglevel=20,logfile=simoutputpath+"/"+simname+".log", seed=0)  # Set loglevel to 0 for debug information
#o = ChemicalDrift(loglevel=20, seed=0)  # Set loglevel to 0 for debug information

o.set_config('drift:vertical_mixing', True)
o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
o.set_config('vertical_mixing:background_diffusivity',0.0001)
o.set_config('vertical_mixing:timestep', 60)
#o.set_config('drift:horizontal_diffusivity', 10)

### Set SPM properties 
o.set_config('chemical:particle_diameter',25.e-6)
o.set_config('chemical:doc_particle_diameter',5.0e-6) # m
o.set_config('chemical:particle_concentration_half_depth',20) # m
o.set_config('chemical:doc_concentration_half_depth',20) # m
#o.set_config('chemical:particle_diameter_uncertainty',1.e-7) # m

### Parameters from radionuclides (Magne Simonsen 2019) for sediment interactions 

o.set_config('chemical:sediment:mixing_depth',0.03) # m 
o.set_config('chemical:sediment:density',2600)  # kg/m3
o.set_config('chemical:sediment:effective_fraction',0.9)
# o.set_config('chemical:sediment:corr_factor',0.1) # Only for metals
o.set_config('chemical:sediment:porosity',0.6)
o.set_config('chemical:sediment:layer_thickness',1.) # m, distance from seabed from which chemicals can be boud to sediments
o.set_config('chemical:sediment:desorption_depth',1.) # m, distance from seabed where desorbed elements are created
o.set_config('chemical:sediment:desorption_depth_uncert',0.1) # m
o.set_config('chemical:sediment:resuspension_depth',1.) # m, , distance from seabed where resuspended elements are created
o.set_config('chemical:sediment:resuspension_depth_uncert',0.1) # m
o.set_config('chemical:sediment:resuspension_critvel',0.15) # m/s
o.set_config('chemical:sediment:burial_rate',0.00003) # m/year

if ~strandingON:
    o.set_config('general:coastline_action', 'previous')
    
#o.fallback_values['sea_floor_depth_below_sea_level'] = 300  # 250m depth

### Volatilization and degradation  
o.set_config('chemical:transformations:volatilization', volatilization)
o.set_config('chemical:transformations:degradation', degradation)
o.set_config('chemical:transformations:degradation_mode', ReactionMode) # Choose between degradation modes

### Set transfer set-up

o.set_config('chemical:transfer_setup','organics')



#%%  Set chemical properties: Select "o.init_chemical_compound(chemical_compound)" for PAHs in database, else set manually

o.init_chemical_compound(chemical_compound)


#%% Set initial partitioning

o.set_config('seed:LMM_fraction', 0.95)
o.set_config('seed:particle_fraction', 0.05)

#%% Adding readers with Input data  
  
reader_sea_depth = reader_netCDF_CF_generic.Reader(simpath+'/Bathimetry.nc')
o.add_reader(reader_sea_depth)
o.set_config('general:use_auto_landmask',True)

print("Loading currents, T, and S Readers....", datetime.now().strftime("%Y_%m_%d-%H_%M_%S")) 

r_uovo_glo  = reader_netCDF_CF_generic.Reader(simpath+'/Test_uo_vo.nc')
r_so        = reader_netCDF_CF_generic.Reader(simpath+'/Test_so.nc')
r_thetao    = reader_netCDF_CF_generic.Reader(simpath+'/Test_thetao.nc')
o.add_reader(r_uovo_glo)
o.add_reader(r_so)
o.add_reader(r_thetao)

print("Loading currents, T, and S Readers....","END:", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

print("Loading wind, mlost, spm, depth, and doc Readers....","START:", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

r_wind      = reader_netCDF_CF_generic.Reader(simpath+'/Test_wind.nc')   
r_mlotst    = reader_netCDF_CF_generic.Reader(simpath+'/Test_mlotst.nc')
r_spm       = reader_netCDF_CF_generic.Reader(simpath+'/Test_SPM.nc')
r_doc       = reader_netCDF_CF_generic.Reader(simpath+'/Test_DOC.nc') # Changed input with no time dimention and variable

o.add_reader(r_wind)
o.add_reader(r_mlotst)
o.add_reader(r_spm)
o.add_reader(r_doc)

print("Loading wind, mlost, spm, depth, and doc Readers....","END:", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))


#%%
# o.list_configspec()

#%% Seeding elements from STEAM scrubbers discharge data: WARNING: This loads all the file to memory!!!!
# Original function called here
print("Start loading data from STEAM....",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

SCRUB_W_OPEN_mfdataset = xr.open_mfdataset(simpath+'/Test_SCRUB_W_OPEN.nc')
SCRUB_W_OPEN=SCRUB_W_OPEN_mfdataset.SCRUB_W_OPEN

SCRUB_W_OPEN=SCRUB_W_OPEN.where((SCRUB_W_OPEN.longitude > long_min) & (SCRUB_W_OPEN.longitude < long_max) &
                                (SCRUB_W_OPEN.latitude > lat_min) & (SCRUB_W_OPEN.latitude < lat_max) &
                                (SCRUB_W_OPEN.time >= Time_emiss_START) &
                                (SCRUB_W_OPEN.time <= Time_emiss_END), drop=True)

print(SCRUB_W_OPEN)
SCRUB_W_OPEN.attrs
SCRUB_W_OPEN.data

print("Finished loading data, start seeding from data....",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

o.seed_from_STEAM(SCRUB_W_OPEN, lowerbound=20000, higherbound=700000, radius=50, # l=1000, h=200000
                  scrubber_type="open_loop", chemical_compound=chemical_compound,
                  mass_element_ug=1e8, number_of_elements=number_of_elements_max, origin_marker = 1)
        
print("Finished seeding from STEAM...",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

#%%
# TODO New function
print("Start loading data from NETCDF....",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

# TODO Change name of input here
NETCDF_mfdataset = xr.open_mfdataset(simpath+'/Emissions_kg.nc')
Bathimetry_mfdataset = xr.open_mfdataset(simpath+'/Bathimetry_conc.nc')

Bathimetry_conc = Bathimetry_mfdataset.elevation

print(Bathimetry_mfdataset)
print(Bathimetry_conc)


# 'Concentration_file_sed_ug_kg_fin.nc'
# 'Concentration_file_water_ug_L_fin.nc'
# 'Emissions_kg.nc'

# TODO Change name of data variable here (sed_conc_ug_kg, wat_conc_ug_L, emission_kg)
# NETCDF=NETCDF_mfdataset.sed_conc_ug_kg 
# NETCDF=NETCDF_mfdataset.wat_conc_ug_L
NETCDF=NETCDF_mfdataset.emission_kg
print(NETCDF)

# For emission data

NETCDF=NETCDF.where((NETCDF.longitude > long_min) & (NETCDF.longitude < long_max) &
                                (NETCDF.latitude > lat_min) & (NETCDF.latitude < lat_max), drop=True)


Bathimetry_conc = Bathimetry_conc.where((NETCDF.longitude > long_min) & (NETCDF.longitude < long_max) &
                                (NETCDF.latitude > lat_min) & (NETCDF.latitude < lat_max), drop=True)

# For concentration data

# NETCDF=NETCDF_mfdataset.where((NETCDF.lon > long_min) & (NETCDF.lon < long_max) &
#                                 (NETCDF.lat > lat_min) & (NETCDF.lat < lat_max) &
#                                 (NETCDF.time >= Time_emiss_START) &
#                                 (NETCDF.time <= Time_emiss_END), drop=True)

# Code as is, "TypeError: cannot directly convert an xarray.Dataset into a numpy array. Instead, create an xarray.DataArray first, either with indexing on the Dataset or by invoking the `to_array()` method
# Changed NETCDF to DataArray
# NETCDF = NETCDF.to_array(dim = "data")
# AttributeError: 'DataArray' object has no attribute 'keys', -> temporaly removed control from code in chemicaldrift.py
# START sel test

lowerbound=0.
higherbound=1.0e15

NETCDF_data = NETCDF


# print(NETCDF_data.lat.data)
# NETCDF_data.plot()
# sel = np.where((NETCDF_data.data > lowerbound) & (NETCDF_data.data < higherbound))
sel = np.where((NETCDF_data > lowerbound) & (NETCDF_data < higherbound))
print('sel NECDF')
print(sel)

t = NETCDF_data.time[sel[0]].data
print('t')
print(t)
la = NETCDF_data.lat[sel[1]].data
print('la')
print(la)
# This print only one value of latitude for all the dataset

lo = NETCDF_data.lon[sel[2]].data
print('lo')
print(lo)
        
        
        
# print(NETCDF2.keys())
# NETCDF.data
# print(NETCDF2)


print("Finished loading data, start seeding from data....",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

o.seed_from_NETCDF(NETCDF_data = NETCDF,
                   Bathimetry_data = Bathimetry_conc,
                   mode = 'water_conc',
                   gen_mode = 'mass',# mass, fixed, tot
                   lon_resol= 0.05, lat_resol = 0.05, 
                   lowerbound=0., higherbound=1.0e15, 
                   radius=0, 
                   mass_element_ug=100e3, 
                   number_of_elements=number_of_elements, # specified at the beginning of the script
                   number_of_elements_max=number_of_elements_max, # specified at the beginning of the script
                   origin_marker=1)



print("Finished seeding from NETCDF...",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

#%% Run simulation

print("Running simulation....","Start:", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

o.run(time_step=step*3600, time_step_output = Time_step_output, duration=timedelta(hours=simulated_days*24), outfile=simoutputpath+"/"+simtime+simname+".nc")

print("End of simulation....","End:", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
#%% Open results after closing the program 
# import opendrift
# o=opendrift.open('D:/ChemicalDrift_output/20220827-171954-deg-vol-NORWAY_Coast-Benzo-a-pyrene-SingleRateConstants-nondiss-Riv-ST-step_8h-sim_365d-em_363d/20220827-171954-deg-vol-NORWAY_Coast-Benzo-a-pyrene-SingleRateConstants-nondiss-Riv-ST-step_8h-sim_365d-em_363d.nc')

#%%
# Print and plot results

o.simulation_summary(chemical_compound)

#%% Making video to check if elements moved correctly
print("Making video....","Start:", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

legend=['dissolved', '', 'SPM', 'sediment', '']
o.animation(color='specie',
            markersize=10,
            vmin=0,vmax=4, # o.nspecies-1 = 4
            colorbar=False,
            fast = True,
            filename = simoutputpath+"/"+simname+"-species.mp4",
            legend = legend,
            legend_loc = 3,
            corners=[long_min, long_max, lat_min, lat_max] # 11, 16, 44, 46
            )

print("Making video....","End:", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

#%%
llcrnrlon=long_min
urcrnrlon=long_max
llcrnrlat=lat_min
urcrnrlat=lat_max

grid = np.meshgrid(np.linspace(llcrnrlon, urcrnrlon, 1000), np.linspace(llcrnrlat, urcrnrlat, 1000))
print(grid)



# This opens a new Chemdrift object! 
o1=opendrift.open(simpath+'/ChemDrift_output.nc')
o1.write_netcdf_chemical_density_map(filename=simpath+'/ChemDrift_conc_output.nc',
                                pixelsize_m=1000.,
                                zlevels=None,
                                mass_unit='ug',
                                horizontal_smoothing=False,
                                smoothing_cells=1,
                                time_avg_conc=True,
                                deltat=24*1, # hours
                                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                                reader_sea_depth=simpath+'/Bathimetry.nc',
                                landmask_shapefile=simpath+'/Landmask.shp')

#%% For debugging

# mass = o.get_property('mass')
# mass_wat = o.get_property('mass_degraded_water')
# mass_sed = o.get_property("mass_degraded_sediment")

# m1 = mass[0].shape
# print(m1)

# m_w1 = mass_wat[0].shape
# print(m_w1)

# m_s1 =mass_sed[0].shape
# print(m_s1)

# m2 = mass[0][:, 25]
# print(m2)

# m_w2 = mass_wat[0][:, 5]
# print(m_w2)
# type(m_w2)

# m_s2 = mass_sed[0][:, 5]
# print(m_s2) 


# np.subtract(m2, m_w2)
