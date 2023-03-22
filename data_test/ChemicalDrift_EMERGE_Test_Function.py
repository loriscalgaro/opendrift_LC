#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Thu Jul  9 12:43:02 2020

# @author: loris
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
step=8 # 3                   # simulation time step in hours
Time_step_output = (60*60*8) # simulation output time step in seconds
simulated_days=6         # days of simulation: should be two days longer than the emissions
### Set region for simulation
region="NA_Total"
long_min = 12.0
long_max = 15.35
lat_min = 44.35
lat_max = 45.8

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
o.set_config('chemical:sediment:resuspension_critvel',0.15) # m/s 0.15
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
# print(r_uovo_glo)
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
mass_element_ug_S1=1e8    # maximum mass of elements (micrograms)
number_of_elements_max=None


# print("Start loading data from STEAM....",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

# SCRUB_W_OPEN_mfdataset = xr.open_mfdataset(simpath+'/Test_SCRUB_W_OPEN.nc')
# SCRUB_W_OPEN=SCRUB_W_OPEN_mfdataset.SCRUB_W_OPEN

# SCRUB_W_OPEN=SCRUB_W_OPEN.where((SCRUB_W_OPEN.longitude > long_min) & (SCRUB_W_OPEN.longitude < long_max) &
#                                 (SCRUB_W_OPEN.latitude > lat_min) & (SCRUB_W_OPEN.latitude < lat_max) &
#                                 (SCRUB_W_OPEN.time >= Time_emiss_START) &
#                                 (SCRUB_W_OPEN.time <= Time_emiss_END), drop=True)

# # SCRUB_W_OPEN[1,:,:].plot(cmap=plt.cm.viridis, robust = True)

# # # print(SCRUB_W_OPEN)
# # # SCRUB_W_OPEN.attrs
# # # SCRUB_W_OPEN.data
# # steam = SCRUB_W_OPEN
# # lowerbound = 0.
# # higherbound = 1e15


# # sel = np.where((steam > lowerbound) & (steam < higherbound))
# # t = steam.time[sel[0]].data
# # la = steam.latitude[sel[1]].data
# # lo = steam.longitude[sel[2]].data

# # print(t)
# # print('len t', len(t))
# # print(la)
# # print('len la', len(la))
# # print(lo)
# # print('len lo', len(lo))


# # data = np.array(steam.data)
# # print(data.shape)

# # data_2 = data[sel]
# # print(data_2)
# # print(data_2.shape)



# print("Finished loading data, start seeding from data....",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

# o.seed_from_STEAM(SCRUB_W_OPEN, lowerbound=20000, higherbound=700000, radius=50, # l=1000, h=200000
#                   scrubber_type="open_loop", chemical_compound=chemical_compound,
#                   mass_element_ug_S1=1e8, number_of_elements=number_of_elements_max, origin_marker = 1)
        
# print("Finished seeding from STEAM...",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

#%%
print("Start loading data from NETCDF....",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

# mode_N1 = "sed_conc" # water_conc, sed_conc, emission
mode_N1 = "water_conc"
# mode_N1 = "emission"

# TODO Check this for error
gen_mode_N1 = "fixed"
# gen_mode_N1 = "mass"

if gen_mode_N1 == "mass":
    mass_element_ug_N1=1e18    # maximum mass of elements (micrograms)
    number_of_elements_N1=None # Number of elements produced for each grid cell
if gen_mode_N1 == "fixed":
    number_of_elements_N1=10 # Number of elements produced for each grid cell
    mass_element_ug_N1=None    # maximum mass of elements (micrograms)
    
# TODO Change name of input here
# NETCDF_mfdataset = xr.open_mfdataset(simpath+'/Concentration_file_sed_ug_kg_fin.nc')
NETCDF_mfdataset = xr.open_mfdataset(simpath+'/Concentration_file_water_ug_L_fin.nc')
NETCDF_mfdataset=NETCDF_mfdataset.rename({'lat': 'latitude','lon': 'longitude'}) # Change name of dimentions lon, lat if needed
# NETCDF_mfdataset.__setitem__('time',NETCDF_mfdataset.time + np.timedelta64(30,'D')) # Shift time coordinate from 02/07AM to 02/08 AM for emissions TODO make new files


# Find spatial resolution of dataset
lat_resultion_data = NETCDF_mfdataset.latitude 
lat_resol_N1 = np.array(abs(lat_resultion_data[0]-lat_resultion_data[1]))
lat_grid_radius = (6.371e6 * lat_resol_N1 * (2 * np.pi) / 360)

lon_resultion_data = NETCDF_mfdataset.longitude
lon_resol_N1 = np.array(abs(lon_resultion_data[0]-lon_resultion_data[1]))
lon_grid_radius_ls = []
for i in range(len(NETCDF_mfdataset.longitude)-2):
    lon_grid_radius_ls.append((6.371e6 * (np.cos(2 * (np.pi) * lon_resultion_data[i] / 360)) * lon_resol_N1 * (2 * np.pi) / 360))

lon_grid_radius = np.mean(np.array(lon_grid_radius_ls))
# Set radius of dataset cell
# radius_N1 = min(np.array([lat_grid_radius, lon_resol_radius]))
radius_N1 = 1500


# Load bathimetry for mass calculation (same grid of concentration)
Bathimetry_mfdataset = xr.open_mfdataset(simpath+'/Bathimetry_conc.nc')
# print(Bathimetry_mfdataset)
Bathimetry_conc = Bathimetry_mfdataset.elevation
Bathimetry_conc = Bathimetry_conc.where((Bathimetry_conc.longitude > long_min) & (Bathimetry_conc.longitude < long_max) &
                                (Bathimetry_conc.latitude > lat_min) & (Bathimetry_conc.latitude < lat_max), drop=True)
# print(Bathimetry_conc)

# Load bathimetry for seeding (same as simulation)
Bathimetry_seed_data_mfdataset = xr.open_mfdataset(simpath+'/Bathimetry.nc')
Bathimetry_seed_data_mfdataset=Bathimetry_seed_data_mfdataset.rename({'lat': 'latitude','lon': 'longitude'})
Bathimetry_seed_data = Bathimetry_seed_data_mfdataset.elevation
# print(Bathimetry_seed_data)

# Change name of data variable (sed_conc_ug_kg, wat_conc_ug_L, emission_kg)
if mode_N1 == 'sed_conc': # water_conc, sed_conc, emission
    NETCDF=NETCDF_mfdataset.sed_conc_ug_kg
elif mode_N1 == 'water_conc':
    NETCDF=NETCDF_mfdataset.wat_conc_ug_L
elif mode_N1 == 'emission':
    NETCDF=NETCDF_mfdataset.emission_kg


NETCDF=NETCDF.where((NETCDF.longitude > long_min) & (NETCDF.longitude < long_max) &
                                (NETCDF.latitude > lat_min) & (NETCDF.latitude < lat_max)&
                                (NETCDF.time >= Time_emiss_START) &
                                (NETCDF.time <= Time_emiss_END), drop=True)

# NETCDF=NETCDF.where((NETCDF.longitude > long_min) & (NETCDF.longitude < long_max) &
#                                 (NETCDF.latitude > lat_min) & (NETCDF.latitude < lat_max), drop=True)

o.seed_from_NETCDF(NETCDF_data = NETCDF,
                   Bathimetry_data = Bathimetry_conc,
                   Bathimetry_seed_data = Bathimetry_seed_data,
                   mode = mode_N1, # converts gen_mode_N1 from tuple to string if needed
                   gen_mode = gen_mode_N1,
                   lon_resol= lon_resol_N1, lat_resol = lat_resol_N1, 
                   lowerbound=0., higherbound=1.0e15, 
                   radius=radius_N1, 
                   mass_element_ug=mass_element_ug_N1, 
                   number_of_elements=number_of_elements_N1, # specified at the beginning of the script
                   origin_marker=2)




print("Finished seeding from NETCDF...",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
#%%
# NETCDF_data = NETCDF
# lowerbound = 0.
# higherbound = 1e15
# Bathimetry_data = Bathimetry_conc
# mode = 'water_conc'
# gen_mode = 'mass'# mass, fixed, tot
# lon_resol= 0.05
# lat_resol = 0.05
# lowerbound=0. 
# higherbound=1.0e15
# radius=0.
# mass_element_ug=100e3
# number_of_elements=number_of_elements # specified at the beginning of the script
# number_of_elements_max=number_of_elements_max # specified at the beginning of the script
# origin_marker=1

# sel = np.where((NETCDF_data > lowerbound) & (NETCDF_data < higherbound))
# t = NETCDF_data.time[sel[0]].data
# la = NETCDF_data.lat[sel[1]].data
# lo = NETCDF_data.lon[sel[2]].data

# t.size

# print(t)
# print('len t', len(t))
# print(la)
# print('len la', len(la))
# print(lo)
# print('len lo', len(lo))

# data = np.array(NETCDF_data.data)
# print(data.shape)

# data_2 = data[sel]
# # data = data[sel]
# print(data_2)
# print(data_2.shape)

# lon_grid_m = []
# lat_grid_m = []
# lon_grid_m.append(6.371e6 * (np.cos(2 * (np.pi) * lo[1] / 360)) * lon_resol * (
#         2 * np.pi) / 360)  # 6.371e6: radius of Earth in m
# lat_grid_m.append(6.371e6 * la[1] * (2 * np.pi) / 360)

# lat_grid_m = np.array(lat_grid_m)
# lon_grid_m = np.array(lon_grid_m)
# Bathimetry_conc = []
# Bathimetry_conc.append(Bathimetry_data.sel(latitude=la[1],longitude=lo[1],method='nearest'))
# Bathimetry_conc = np.array(Bathimetry_conc)

# print("Finished loading data, start seeding from data....",": ", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

# lon_grid_m_seed = (6.371e6 * (np.cos(2 * (np.pi) * lo[1] / 360)) * lon_resol * (2 * np.pi) / 360)  # 6.371e6: radius of Earth in m
# lat_grid_m_seed = (6.371e6 * lat_resol * (2 * np.pi) / 360)




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

# grid = np.meshgrid(np.linspace(llcrnrlon, urcrnrlon, 1000), np.linspace(llcrnrlat, urcrnrlat, 1000))
# print(grid)



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

#%% Calculate and verify mass of chemical from sediments

# NETCDF_data = NETCDF
# lowerbound = 0.
# higherbound = 1e25
# sel = np.where((NETCDF_data > lowerbound) & (NETCDF_data < higherbound))
# t = NETCDF_data.time[sel[0]].data
# la = NETCDF_data.latitude[sel[1]].data
# lo = NETCDF_data.longitude[sel[2]].data


# lon_resultion_data = NETCDF_data.longitude
# lon_resol_N1 = np.array(abs(lon_resultion_data[0]-lon_resultion_data[1]))
# lon_grid_radius_ls = []
# for i in range(len(lo)):
#     lon_grid_radius_ls.append((6.371e6 * (np.cos(2 * (np.pi) * lo[i] / 360)) * lon_resol_N1 * (2 * np.pi) / 360))

# lon_grid_radius = np.array(np.array(lon_grid_radius_ls))
# print(lon_grid_radius)

# lat_resultion_data = NETCDF_data.latitude 
# lat_resol_N1 = np.array(abs(lat_resultion_data[0]-lat_resultion_data[1]))
# lat_grid_radius = (6.371e6 * lat_resol_N1 * (2 * np.pi) / 360)


# sed_mixing_depth = 0.03
# sed_porosity = np.array(0.6)  # fraction of sediment volume made of water
# sed_density_dry = np.array(2600)  # kg/m3 d.w.
# sed_density_wet = np.array((sed_density_dry * (1 - sed_porosity)) * 1e-3)  # kg/L wet weight, kg/m3 * 1e-3 = kg/L

# Bathimetry_mfdataset = xr.open_mfdataset(simpath+'/Bathimetry_conc.nc')
# # print(Bathimetry_mfdataset)
# Bathimetry_conc = Bathimetry_mfdataset.elevation
# Bathimetry_conc = Bathimetry_conc.where((Bathimetry_conc.longitude > long_min) & (Bathimetry_conc.longitude < long_max) &
#                                 (Bathimetry_conc.latitude > lat_min) & (Bathimetry_conc.latitude < lat_max), drop=True)

# Volume = np.array(sed_mixing_depth * lat_grid_radius * lon_grid_radius)  # m3
# print(len(Volume))
# print((Volume))

# sed_data = np.array(NETCDF_data.data)
# sed_data = sed_data[sel]
# # print(sed_data)
# data_sed = sed_data * ((1 - sed_porosity) * sed_density_wet)
# print(data_sed)

# mass_sed = []
# for j in range(len(data_sed)):
#     mass_sed.append(Volume[j] * data_sed[j]* 1e3)

# mass_sed = np.array(mass_sed)
# # print(mass_sed)

# mass_sed_g = mass_sed/1e6 # from ug to g
# print('mass_sed_g test: ', sum(mass_sed_g))

# run o.simulation_summary(chemical_compound) to get final mass of elements during simulation

#%% Calculate and verify mass of chemical from water

# NETCDF_data = NETCDF
# lowerbound = 0.
# higherbound = 1e25
# sel = np.where((NETCDF_data > lowerbound) & (NETCDF_data < higherbound))
# t = NETCDF_data.time[sel[0]].data
# la = NETCDF_data.latitude[sel[1]].data
# lo = NETCDF_data.longitude[sel[2]].data


# lon_resultion_data = NETCDF_data.longitude
# lon_resol_N1 = np.array(abs(lon_resultion_data[0]-lon_resultion_data[1]))
# lon_grid_radius_ls = []
# for i in range(len(lo)):
#     lon_grid_radius_ls.append((6.371e6 * (np.cos(2 * (np.pi) * lo[i] / 360)) * lon_resol_N1 * (2 * np.pi) / 360))

# lon_grid_radius = np.array(np.array(lon_grid_radius_ls))
# # print(lon_grid_radius)

# lat_resultion_data = NETCDF_data.latitude 
# lat_resol_N1 = np.array(abs(lat_resultion_data[0]-lat_resultion_data[1]))
# lat_grid_radius = (6.371e6 * lat_resol_N1 * (2 * np.pi) / 360)



# Bathimetry_mfdataset = xr.open_mfdataset(simpath+'/Bathimetry_conc.nc')
# # print(Bathimetry_mfdataset)
# Bathimetry_conc = Bathimetry_mfdataset.elevation
# Bathimetry_conc = Bathimetry_conc.where((Bathimetry_conc.longitude > long_min) & (Bathimetry_conc.longitude < long_max) &
#                                 (Bathimetry_conc.latitude > lat_min) & (Bathimetry_conc.latitude < lat_max), drop=True)


# Bathimetry_conc_data = []

# for i in range(len(la)):
#     Bathimetry_conc_data.append(np.array([(Bathimetry_conc.sel(latitude=la[i],longitude=lo[i],method='nearest'))]))

# Bathimetry_conc_data = np.array(Bathimetry_conc_data)
# print(Bathimetry_conc_data)
# Volume = np.array(Bathimetry_conc_data * lat_grid_radius * lon_grid_radius)  # m3

# Volume = np.mean(Volume, axis=1)

# wat_data = np.array(NETCDF_data.data)
# wat_data = wat_data[sel]
# print(wat_data)

# # print(wat_data)

# mass_wat = []
# for j in range(len(wat_data)):
#     mass_wat.append(Volume[j] * wat_data[j]* 1e3)

# mass_wat = np.array(mass_wat)
# # print(mass_wat)

# mass_wat_g = mass_wat/1e6 # from ug to g
# print('mass_wat_g test: ', sum(mass_wat_g))

# run o.simulation_summary(chemical_compound) to get final mass of elements during simulation

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
