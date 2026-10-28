# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

from datetime import datetime
import pyproj
import numpy as np
from netCDF4 import num2date
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import xarray as xr
from opendrift.readers.basereader.structured import StructuredReader
from opendrift.readers import open_dataset_opendrift, datetime_from_variable, \
                              add_standard_name_for_surface_grib_variables


class Reader(StructuredReader):
    """
    A reader for `CF-compliant <https://cfconventions.org/>`_ netCDF files. It can take a single file, a file pattern, a URL or an xarray Dataset.

    Args:
        :param filename: A single netCDF file, a pattern of files, or a xr.Dataset. The
                         netCDF file can also be an URL to an OPeNDAP server.
        :type filename: string, xr.Dataset (required).

        :param name: Name of reader
        :type name: string, optional

        :param proj4: PROJ.4 string describing projection of data.
        :type proj4: string, optional

        :param bottom_water_sampling: Opt-in water-side boundary sampling for
               3-D environmental point requests. Requires bathymetry on the
               same grid and compatible metre-based vertical coordinates.
               Defaults to False to preserve the behavior of other models.
        :type bottom_water_sampling: bool, optional

    Read-only forcing contract:
        The input NetCDF/xarray Dataset is assumed immutable after Reader
        initialization. A bounded one-frame cache is used only for categorical
        `wetdry_state` sampling. Normal time variation is supported: changing
        the selected source-time frame replaces the cached frame. Applications
        must not alter the Dataset contents in place while using this reader;
        if forcing data are replaced or edited, create a new Reader instance.
        No runtime cache-invalidation machinery is provided or required for
        ordinary read-only forcing workflows.

    Example:

    .. code::

       from opendrift.readers.reader_netCDF_CF_generic import Reader
       r = Reader("arome_subset_16Nov2015.nc")

    Several files can be specified by using a pattern:

    .. code::

       from opendrift.readers.reader_netCDF_CF_generic import Reader
       r = Reader("*.nc")

    An OPeNDAP URL can be used:

    .. code::

       from opendrift.readers.reader_netCDF_CF_generic import Reader
       r = Reader('https://thredds.met.no/thredds/dodsC/mepslatest/meps_lagged_6_h_latest_2_5km_latest.nc')

    A xr.Dataset or a zarr dataset in an object store with auth can be used:

    .. code::

        from opendrift.readers.reader_netCDF_CF_generic import Reader
        r = Reader(ds, zarr_storage_options)
    """

    def __init__(self, filename=None, zarr_storage_options=None, name=None, proj4=None,
                 standard_name_mapping={}, ensemble_member=None,
                 bottom_water_sampling=False):

        if not isinstance(bottom_water_sampling, (bool, np.bool_)):
            raise TypeError('bottom_water_sampling must be boolean')
        self.bottom_water_sampling = bool(bottom_water_sampling)
        self._cf_wetdry_frame_cache = None


        #if isinstance(filename, xr.Dataset):
        #    self.Dataset = filename
        #    if name is not None:
        #        self.name = name
        #    elif hasattr(self.Dataset, 'name'):
        #        self.name = self.Dataset.name
        #    else:
        #        self.name = str(filename)
        #else:
        #    if zarr_storage_options is not None:
        #        self.Dataset = xr.open_zarr(filename, storage_options=zarr_storage_options)
        #        if name is None:
        #            self.name = filename
        #        else:
        #            self.name = name
        #    else:
        #        if filename is None:
        #            raise ValueError('Need filename as argument to constructor')

        #        filestr = str(filename)
        #        if name is None:
        #            self.name = filestr
        #        else:
        #            self.name = name

        #        try:
        #            # Open file, check that everything is ok
        #            logger.info('Opening dataset: ' + filestr)
        #            if ('*' in filestr) or ('?' in filestr) or ('[' in filestr):
        #                logger.info('Opening files with MFDataset')
        #                self.Dataset = xr.open_mfdataset(filename, data_vars='minimal', coords='minimal',
        #                                                chunks={'time': 1}, decode_times=False)
        #            elif ensemble_member is not None:
        #                self.Dataset = xr.open_dataset(filename, decode_times=False).isel(ensemble_member=ensemble_member)
        #            else:
        #                self.Dataset = xr.open_dataset(filename, decode_times=False)
        #        except Exception as e:
        #            raise ValueError(e)

        self.Dataset = open_dataset_opendrift(source=filename, zarr_storage_options=zarr_storage_options)

        self.Dataset = add_standard_name_for_surface_grib_variables(self.Dataset)

        if name is None:
            self.name = str(filename)
        else:
            self.name = name

        # NB: check below might not be waterproof
        if 'ocean_time' in self.Dataset.dims and 'eta_u' in self.Dataset.dims and \
                'eta_rho' in self.Dataset.dims:
            raise ValueError('This seems to be a ROMS native file, should use ROMS native reader instead')

        logger.debug('Finding coordinate variables.')
        if proj4 is not None:  # If user has provided a projection apriori
            self.proj4 = proj4
        # Find x, y and z coordinates
        lon_var_name = None
        lat_var_name = None
        self.unitfactor = 1
        self.realizations = None
        self.ensemble_dimension = None
        self.dimensions = {}
        for var_name in self.Dataset.variables:
            var = self.Dataset.variables[var_name]

            if self.proj4 is None:
                if 'grid_mapping_name' in var.attrs:
                    logger.debug(
                        ('Parsing CF grid mapping dictionary:'
                        ' ' + str(var.attrs)))
                    try:  # parse proj4 with pyproj.CRS
                        crs = pyproj.CRS.from_cf(var.attrs)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.proj4 = crs.to_proj4()
                    except:
                        logger.info('Could not parse CF grid_mapping')
                if self.proj4 is None:
                    if 'proj4' in var.attrs:
                        self.proj4 = str(var.attrs['proj4'])
                    elif 'proj4_string' in var.attrs:
                        self.proj4 = str(var.attrs['proj4_string'])

            standard_name = var.attrs['standard_name'] if 'standard_name' in var.attrs else ''
            long_name = var.attrs['long_name'] if 'long_name' in var.attrs else ''
            axis = var.attrs['axis'] if 'axis' in var.attrs else ''
            units = var.attrs['units'] if 'units' in var.attrs else ''
            CoordinateAxisType = var.attrs['CoordinateAxisType'] if 'CoordinateAxisType' in var.attrs else ''
            if standard_name == 'longitude' or \
                    CoordinateAxisType.lower() == 'lon' or \
                    long_name.lower() == 'longitude' or \
                    var_name.lower() in ['longitude', 'lon']:
                lon_var_name = var_name
            if standard_name == 'latitude' or \
                    CoordinateAxisType.lower() == 'lat' or \
                    long_name.lower() == 'latitude' or \
                    var_name.lower() in ['latitude', 'lat']:
                lat_var_name = var_name
            if (axis == 'X' or standard_name == 'projection_x_coordinate' or standard_name == 'grid_longitude') \
                    and var.ndim == 1:
                if len(var.dims)==1:
                    self.dimensions['x'] = var.dims[0]
                # Fix for units; should ideally use udunits package
                if units == 'km':
                    self.unitfactor = 1000
                elif units == '100  km':
                    self.unitfactor = 100000
                var_data = var.values
                x = var_data*self.unitfactor
            if (axis == 'Y' or standard_name == 'projection_y_coordinate' or standard_name == 'grid_latitude') \
                    and var.ndim == 1:
                if len(var.dims)==1:
                    self.dimensions['y'] = var.dims[0]
                # Fix for units; should ideally use udunits package
                if units == 'km':
                    self.unitfactor = 1000
                elif units == '100  km':
                    self.unitfactor = 100000
                var_data = var.values
                y = var_data*self.unitfactor
            if (standard_name == 'depth' or axis == 'Z') and var.ndim==1:
                var_data = var.values
                if len(var.dims)==1:
                    self.dimensions['z'] = var.dims[0]
                if var_data.ndim == 1:  # Earlier this was not a requirement above
                    if 'positive' not in var.attrs or \
                            var.attrs['positive'] == 'up':
                        self.z = var_data
                    else:
                        self.z = -var_data
            if standard_name == 'time' or axis == 'T' or var_name in ['time', 'vtime']:
                # Read and store time coverage (of this particular file)
                if len(var.dims)==1:
                    if 'time' in self.dimensions:
                        if not isinstance(self.dimensions['time'], list):
                            self.dimensions['time'] = [self.dimensions['time']]
                        self.dimensions['time'].append(var.dims[0])
                    else:
                        self.dimensions['time'] = var.dims[0]

                self.times = datetime_from_variable(var)
                if len(self.times) > 1:
                    if len(self.times) > 5:  # Check that times are not corrupted, necessary hack for NOAA winds on thredds
                        first_timestep = self.times[1] - self.times[0]
                        second_timestep = self.times[2] - self.times[1]
                        if first_timestep/second_timestep > 2:
                            logger.warning('First time steps is larger than second timestep, starting at second timestep')
                            self.times = self.times[1:]
                            self.Dataset = self.Dataset.isel({var.dims[0]: slice(1, len(var))})
                        second_last_timestep = self.times[-2] - self.times[-3]
                        last_timestep = self.times[-1] - self.times[-2]
                        if last_timestep/second_last_timestep > 2:
                            logger.warning('Last time steps is larger than second last timestep, stopping at second last timestep')
                            self.times = self.times[0:-2]
                            self.Dataset = self.Dataset.isel({var.dims[0]: slice(0, len(var)-1)})
                    self.time_step = self.times[1] - self.times[0]
                else:
                    self.time_step = None
                self.start_time = self.times[0]
                self.end_time = self.times[-1]

            if standard_name == 'realization':
                if ensemble_member == None:
                    var_data = np.atleast_1d(var.values)
                    self.realizations = var_data
                    logger.debug('%i ensemble members available'
                                % len(self.realizations))
                if len(var.dims)==1:
                    self.ensemble_dimension = var.dims[0]

        # Temporary workaround for Barents EPS model
        if self.realizations is None and 'ensemble_member' in self.Dataset.dims:
            self.realizations = np.arange(self.Dataset.dims['ensemble_member'])

        #########################
        # Summary of geolocation
        #########################
        if 'x' in locals():
            if x[1]-x[0] == 1 and y[1]-y[0] == 1:
                logger.info('deltaX and deltaY are 1, interpreting dataset as unprojected')
                projected = False
            else:
                projected = True
        if 'x' not in locals() or projected is False:  # No x/y-coordinates were detected
            if lon_var_name is None:
                raise ValueError('No geospatial coordinates were detected, cannot geolocate dataset')
            # We load lon and lat arrays into memory
            lon_var = self.Dataset.variables[lon_var_name]
            lat_var = self.Dataset.variables[lat_var_name]
            if lon_var.ndim == 1:
                logger.debug('Lon and lat are 1D arrays - using as projection coordinates')
                x = lon_var.data
                y = lat_var.data
                self.dimensions['x'] = lon_var.dims[0]
                self.dimensions['y'] = lat_var.dims[0]
                if self.proj4 is None:
                    self.proj4 = '+proj=latlong'
            elif lon_var.ndim == 2:
                logger.debug('Lon and lat are 2D arrays - dataset is unprojected')
                self.lon = lon_var.data
                self.lat = lat_var.data
                # Check if 2D arrays are repeated 1D arrays
                if np.array_equal(self.lon[0,:], self.lon[-1,:]) and np.array_equal(self.lat[:,0], self.lat[:,-1]):
                    logger.info('Lon and lat are repeated 1D arrays - i.e. lonlat projection')
                    x = self.lon[0,:]
                    y = self.lat[:,0]
                    self.dimensions['x'] = lon_var.dims[1]
                    self.dimensions['y'] = lat_var.dims[0]
                    if self.proj4 is None:
                        self.proj4 = '+proj=latlong'
                else:
                    self.dimensions['x'] = lon_var.dims[1]
                    self.dimensions['y'] = lat_var.dims[0]
                    self.projected = False
                    self.proj = None
                    self.proj4 = None
            elif lon_var.ndim == 3:
                logger.debug('Lon lat are 3D arrays, reading first time')
                self.lon = lon_var[0,:,:].data
                self.lat = lat_var[0,:,:].data
                self.dimensions['x'] = lon_var.dims[1]
                self.dimensions['y'] = lat_var.dims[2]
                self.projected = False
                self.proj = None
                self.proj4 = None
        else:
            if self.proj4 is None:
                logger.info('Grid coordinates are detected, but proj4 string not given: assuming latlong')
                self.proj4 = '+proj=latlong'

        if 'x' in locals() and 'y' in locals():
            self.numx = len(x)
            self.numy = len(y)
            self.xmin, self.xmax = x.min(), x.max()
            self.ymin, self.ymax = y.min(), y.max()
            self.delta_x = np.abs(x[1] - x[0])
            self.delta_y = np.abs(y[1] - y[0])
            rel_delta_x = (x[1::] - x[0:-1])
            rel_delta_x = np.abs((rel_delta_x.max() -
                                  rel_delta_x.min())/self.delta_x)
            rel_delta_y = (y[1::] - y[0:-1])
            rel_delta_y = np.abs((rel_delta_y.max() -
                                  rel_delta_y.min())/self.delta_y)
            if rel_delta_x > 0.05:  # Allow 5 % deviation
                print(rel_delta_x)
                print(x[1::] - x[0:-1])
                raise ValueError('delta_x is not constant!')
            if rel_delta_y > 0.05:
                print(rel_delta_y)
                print(y[1::] - y[0:-1])
                raise ValueError('delta_y is not constant!')
            self.x = x  # Store coordinate vectors
            self.y = y

        if self.proj4 is not None and 'latlong' in self.proj4 and self.xmax is not None and self.xmax > 360:
            logger.info('Longitudes > 360 degrees, subtracting 360')
            self.xmin -= 360
            self.xmax -= 360
            self.x -= 360

        logger.info(f'Detected dimensions: {self.dimensions}')

        ##########################################
        # Find all variables having standard_name
        ##########################################
        self.variable_mapping = {}
        skipvars = []
        for var_name in self.Dataset.variables:
            var = self.Dataset.variables[var_name]
            if var.ndim < 2:
                continue
            if var_name in standard_name_mapping:
                # User may specify mapping if standard_name is missing, or to override existing
                standard_name = standard_name_mapping[var_name]
                self.variable_mapping[standard_name] = str(var_name)
            elif 'standard_name' in var.attrs and 'hybrid' not in var.dims:
                # Skipping hybrid dim is workaround to prevent parsing upper winds from ECMWF
                # A permanent solution for selecting correct variable is needed
                standard_name = str(var.attrs['standard_name'])
                if standard_name in self.variable_aliases:  # Mapping if needed
                    standard_name = self.variable_aliases[standard_name]
                self.variable_mapping[standard_name] = str(var_name)
            else:
                skipvars.append(var_name)
        # Necessary hack for ECMWF with winds at several levels
        # TODO: must provide variable mapping along with URLs to create readers
        if self.variable_mapping.get('x_wind') is not None:
            if 'x_wind_200m' in self.Dataset.variables and 'x_wind_10m' in self.Dataset.variables:
                logger.warning('Shifting variable names of ECMWF data, to get winds at 10m')
                self.variable_mapping['x_wind'] = 'x_wind_10m'
                self.variable_mapping['y_wind'] = 'y_wind_10m'
                if self.dimensions.get('y') == 'latitude1' and 'latitude' in self.Dataset.variables:
                    self.dimensions['x'] = 'longitude'
                    self.dimensions['y'] = 'latitude'

        if len(skipvars) > 0:
            logger.debug('Skipped variables without standard_name: %s' % skipvars)

        # Optional categorical wet/dry channel. Recognize a native variable
        # explicitly named wetdry_state, or an existing user mapping to that
        # standard environmental key. No inference from a static land mask.
        if ('wetdry_state' not in self.variable_mapping and
                'wetdry_state' in self.Dataset.data_vars):
            self.variable_mapping['wetdry_state'] = 'wetdry_state'
        if 'wetdry_state' in self.variable_mapping:
            state_var = self.Dataset[self.variable_mapping['wetdry_state']]
            spatial = {self.dimensions.get('x'), self.dimensions.get('y')}
            allowed = spatial | {self.dimensions.get('time')}
            if (None in spatial or not spatial.issubset(set(state_var.dims)) or
                    any(d not in allowed for d in state_var.dims) or
                    state_var.dtype.kind not in 'iufb'):
                raise ValueError('wetdry_state requires a numeric 2-D CF x/y field, optionally time-dependent')
            if not hasattr(self, 'x') or not hasattr(self, 'y'):
                raise ValueError('wetdry_state requires a rectilinear CF x/y grid for categorical nearest-cell sampling')
            for axis in (np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float)):
                if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() or
                        not (np.all(np.diff(axis)>0) or np.all(np.diff(axis)<0))):
                    raise ValueError('wetdry_state requires strictly monotone rectilinear x/y axes')

        self.variables = list(self.variable_mapping.keys())

        for vn, va in self.variable_mapping.items():
            if vn == 'sea_floor_depth_below_sea_level':
                var = self.Dataset.variables[va]
                if 'ensemble_member' in var.dims:
                    # Workaround for datasets with unnecessary ensemble dimension for static variables
                    logger.info(f'Removing ensemble dimension from {vn}')
                    var = var.isel(ensemble_member=0).squeeze()
                    self.Dataset[va] = var

        if isinstance(self.dimensions.get('time', None), list):
            warning = f'Several time dimensions detected: {self.dimensions["time"]}'
            variable_time_dimensions = []  # Check which time dimensions are actually used
            for vn, va in self.variable_mapping.items():
                variable_time_dimensions += [x for x in self.Dataset[va].coords if x in self.dimensions['time']]
                variable_time_dimensions = list(set(variable_time_dimensions))
            if len(variable_time_dimensions) == 1:
                logger.debug(f'{warning}: Using single time dimension found in detected variables: {variable_time_dimensions}')
                self.dimensions['time'] = variable_time_dimensions[0]
            elif len(variable_time_dimensions) > 1:
                if 'time' in variable_time_dimensions:
                    self.dimensions['time'] = 'time'
                else:
                    raise ValueError(f'{warning}, and also several time dimensions used by detected variables: {variable_time_dimensions}')

        # An explicit opt-in; no existing CF reader changes behavior by default.
        if self.bottom_water_sampling:
            if 'sea_floor_depth_below_sea_level' not in self.variable_mapping:
                raise ValueError('bottom_water_sampling requires CF bathymetry')
            if not hasattr(self, 'z') or 'z' not in self.dimensions:
                raise ValueError('bottom_water_sampling requires a native 1-D depth axis')
            zz = np.asarray(self.z, dtype=float)
            if (zz.ndim != 1 or not len(zz) or np.any(~np.isfinite(zz))
                    or np.any(zz > 0) or np.any(np.diff(zz) == 0)
                    or not (np.all(np.diff(zz) < 0) or np.all(np.diff(zz) > 0))):
                raise ValueError('Unqualified CF native sampling depths')
            depth_var = self.Dataset[self.dimensions['z']]
            # Explicit opt-in is qualified ONLY for measured physical-depth
            # sampling points. Never mistake sigma/hybrid/formula coordinates,
            # or an undocumented depth datum/unit, for metre-valued z samples.
            if (depth_var.attrs.get('standard_name') != 'depth' or
                    'formula_terms' in depth_var.attrs or
                    depth_var.attrs.get('positive') not in ('up', 'down')):
                raise ValueError('bottom_water_sampling requires a certified physical-depth axis (not sigma/hybrid)')
            if depth_var.attrs.get('units') not in ('m', 'meter', 'meters', 'metre', 'metres'):
                raise ValueError('bottom_water_sampling requires explicit depth units in metres')
            bathy_var = self.Dataset[self.variable_mapping['sea_floor_depth_below_sea_level']]
            if bathy_var.attrs.get('units', 'm') not in ('m', 'meter', 'meters', 'metre', 'metres'):
                raise ValueError('bottom_water_sampling requires bathymetry in metres')
            self._bottom_water_eta_variable = next(
                (n for n in ('sea_surface_height',
                             'water_surface_height_above_reference_datum')
                 if n in self.variable_mapping), None)
            if self._bottom_water_eta_variable is not None:
                eta_var = self.Dataset[self.variable_mapping[self._bottom_water_eta_variable]]
                if eta_var.attrs.get('units', 'm') not in ('m', 'meter', 'meters', 'metre', 'metres'):
                    raise ValueError('bottom_water_sampling requires sea level in metres')
            else:
                logger.warning('bottom_water_sampling: no sea-surface-height '
                               'field; treating supplied bathymetry as current water depth')

        # Run constructor of parent Reader class
        super().__init__()

    @staticmethod
    def _cf_batched_brackets(coords, points):
        """Conservative native stencil equivalents for an array of query points.

        Preserves the original left-search, per-axis absolute
        coordinate tolerance, and exact-coordinate precedence. No invalid
        query can become admissible merely through index clipping.
        """
        coords = np.asarray(coords, dtype=float)
        points = np.atleast_1d(np.asarray(points, dtype=float))
        count = len(points)
        lower = np.zeros(count, dtype=np.intp)
        upper = np.zeros(count, dtype=np.intp)
        ok = np.isfinite(points)
        if coords.ndim != 1 or not len(coords) or not np.all(np.isfinite(coords)):
            return lower, upper, np.zeros(count, dtype=bool)
        ascending = coords[0] < coords[-1]
        values = coords if ascending else -coords
        p = points if ascending else -points
        indices = np.searchsorted(values, p, side='left')
        spacing = np.min(np.abs(np.diff(values))) if len(values) > 1 else 1.0
        atol = np.maximum(1.e-7, np.minimum(.01*spacing,
                                           3.e-7*np.maximum(1.0, np.abs(p))))
        at_right = indices < len(values)
        safe_right = np.clip(indices, 0, len(values)-1)
        match_right = at_right & (np.abs(p-values[safe_right]) <= atol)
        at_left = indices > 0
        safe_left = np.clip(indices-1, 0, len(values)-1)
        match_left = (~match_right) & at_left & (np.abs(p-values[safe_left]) <= atol)
        interior = (~match_right) & (~match_left) & (indices > 0) & at_right
        lower[:] = np.where(match_right, safe_right, safe_left)
        upper[:] = np.where(interior, safe_right, lower)
        ok &= match_right | match_left | interior
        return lower, upper, ok

    def _cf_native_source_validity(self, source_time, x, y, depths, names,
                                  profile_names=()):
        """Batch native raw-mask checks without interpolation/downfill.

        Shape and source-time admission match the unoptimized opt-in reader.
        Reuse horizontal/vertical stencil indices across all fields; test all
        stencil corners including native source masks and nonfinite entries.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        depths = np.asarray(depths, dtype=float)
        all_names = sorted(set(names) | set(profile_names))
        if not all_names:
            return {}, {}
        raw_x, raw_y, raw_z = x, y, depths
        if profile_names:
            raw_x = np.concatenate((x, x[:1], x[:1]))
            raw_y = np.concatenate((y, y[:1], y[:1]))
            raw_z = np.concatenate((depths, [np.max(self.z), np.min(self.z)]))
        raw = self.get_variables(all_names, source_time, raw_x, raw_y, raw_z)
        rz = np.atleast_1d(np.asarray(raw['z'], dtype=float))
        rx = np.asarray(raw['x'])
        ry = np.asarray(raw['y'])
        xlo, xhi, xok = self._cf_batched_brackets(rx, x)
        ylo, yhi, yok = self._cf_batched_brackets(ry, y)
        zlo, zhi, zok = self._cf_batched_brackets(rz, depths)
        stencil_ok = xok & yok & zok
        point_valid = {}
        profile_valid = {}
        for name in all_names:
            arr = raw[name]
            if isinstance(arr, list):
                raise ValueError('Unqualified ensemble native mask in bottom-water sampling')
            arr = np.ma.asarray(arr)
            if arr.ndim == 2 and len(rz) == 1:
                arr = arr[np.newaxis, :, :]
            if arr.ndim != 3:
                raise ValueError('Unqualified CF native 3-D sample shape for ' + name)
            validity = np.isfinite(np.ma.filled(arr, np.nan)) & ~np.ma.getmaskarray(arr)
            good = stencil_ok.copy()
            # Exactly the same native 2 x 2 x 2 corner conjunction as
            # validity[np.ix_(iz, iy, ix)].all() per point, without Python
            # per-particle loops or repeated bracket searches.
            for kz in (zlo, zhi):
                for jy in (ylo, yhi):
                    for ix in (xlo, xhi):
                        good &= validity[kz, jy, ix]
            point_valid[name] = good
            # Native profiles retain source-level masks at every z; no
            # synthetic values or interpolation across invalid corners.
            pg = np.broadcast_to(stencil_ok, (len(rz),len(x))).copy()
            for jy in (ylo, yhi):
                for ix in (xlo, xhi):
                    pg &= validity[:, jy, ix]
            profile_valid[name] = (rz, pg)
        return point_valid, profile_valid

    def prepare(self, *args, **kwargs):
        self._cf_fast_blocks = []
        self._cf_geometry_block_cache = []
        return super().prepare(*args, **kwargs)

    def _cf_source_geometry_at_time(self, source_time, x, y, names):
        """Reuse source-record *grid blocks* for 2-D bathymetry and eta.

        The original OpenDrift ReaderBlock still performs the interpolation
        and native invalid-data handling.  No per-particle interpolation is
        memoized.  An out-of-coverage, large, mixed-dimensional or unsupported
        request falls back to the inherited StructuredReader method; the native source-time
        and mask policies are not changed.
        """
        from opendrift.readers.interpolation.structured import ReaderBlock
        key = (source_time, tuple(names))
        if (not len(x) or not np.isfinite(x).all() or
                not np.isfinite(y).all() or self.global_coverage()):
            raise ValueError('CF_FAST_UNSUPPORTED_GEOMETRY_REQUEST')
        snapshots = getattr(self, '_cf_geometry_block_cache', [])
        snapshot = None
        for entry in snapshots:
            if entry['key'] == key and entry['block'].covers_positions(x, y):
                snapshot = entry
                break
        if snapshot is None:
            # Keep the original reader's subset and mask admission unchanged.
            raw = self.get_variables(names, source_time, x, y,
                                     np.zeros(len(x), dtype=float))
            xx = np.asarray(raw['x']); yy = np.asarray(raw['y'])
            if (xx.ndim != 1 or yy.ndim != 1 or
                    len(xx) < 2 or len(yy) < 2 or
                    not np.isfinite(xx).all() or not np.isfinite(yy).all()):
                raise ValueError('CF_FAST_UNSUPPORTED_GEOMETRY_GRID')
            size = xx.nbytes + yy.nbytes
            for name in names:
                value = raw[name]
                if isinstance(value, list):
                    raise ValueError('CF_FAST_UNSUPPORTED_GEOMETRY_ENSEMBLE')
                array = np.ma.asarray(value)
                if array.ndim != 2 or array.shape != (len(yy), len(xx)):
                    raise ValueError('CF_FAST_UNSUPPORTED_GEOMETRY_SHAPE')
                size += array.nbytes + np.ma.getmaskarray(array).nbytes
            # Do not cache oversized snapshots.  Retain the parent behavior
            # even in this case via the ephemeral original ReaderBlock.
            block = ReaderBlock(dict(raw), interpolation_horizontal=self.interpolation,
                                wrap_x=False)
            snapshot = {'key':key,'block':block,'bytes':size}
            if size <= 16*1024*1024:
                snapshots = [q for q in snapshots if q['key'] != key]
                snapshots.append(snapshot)
                while (len(snapshots) > 2 or
                       sum(q['bytes'] for q in snapshots) > 16*1024*1024):
                    snapshots.pop(0)
                self._cf_geometry_block_cache = snapshots
        values, _ = snapshot['block'].interpolate(x, y, np.zeros(len(x)),
                                                   names, None, None)
        return values

    def _cf_fast_sample_valid_3d(self, source_time, x, y, depths, names):
        """Vectorized direct native interpolation for explicitly admitted CF 3-D
        fields without profiles. Only original source-valid stencils are sampled.
        The opt-in-only caller retains per-timestamp local bottom-depth geometry.

        No second source read for validity, and no waterward NaN synthesis.
        An unsupported shape or interpolation method takes the original path.
        """
        from scipy.ndimage import map_coordinates
        from scipy.interpolate import interp1d
        key = (source_time, tuple(names))
        blocks = getattr(self, '_cf_fast_blocks', [])
        snapshot = None
        for cached in blocks:
            if (cached['key'] == key and
                    x.min() >= cached['rx'].min() and x.max() <= cached['rx'].max() and
                    y.min() >= cached['ry'].min() and y.max() <= cached['ry'].max() and
                    depths.min() >= cached['rz'].min() and depths.max() <= cached['rz'].max()):
                snapshot = cached
                break
        if snapshot is None:
            raw = self.get_variables(names, source_time, x, y, depths)
            rx = np.asarray(raw['x'])
            ry = np.asarray(raw['y'])
            rz = np.atleast_1d(np.asarray(raw['z'], dtype=float))
            if (len(rz) < 2 or len(rz) >= 120 or len(rx) < 2 or len(ry) < 2 or
                    not np.all(np.isfinite(rx)) or not np.all(np.isfinite(ry)) or
                    not np.all(np.isfinite(rz)) or self.global_coverage()):
                raise ValueError('CF_FAST_UNSUPPORTED_GRID')
            fields = {}
            size = rx.nbytes + ry.nbytes + rz.nbytes
            for name in names:
                arr = raw[name]
                if isinstance(arr, list):
                    raise ValueError('CF_FAST_UNSUPPORTED_ENSEMBLE')
                arr = np.ma.asarray(arr)
                if arr.ndim != 3 or arr.shape != (len(rz), len(ry), len(rx)):
                    raise ValueError('CF_FAST_UNSUPPORTED_SHAPE')
                finite = np.isfinite(np.ma.filled(arr, np.nan)) & ~np.ma.getmaskarray(arr)
                # ReaderBlock first fills missing native samples downward,
                # then expands holes horizontally if needed for requested
                # positions. Precomputing the same expansions once over the
                # bounded source block preserves wet-neighbour sampling for
                # moving near-coastal points; source validity remains raw.
                from opendrift.readers.interpolation.interpolators import (
                    fill_NaN_towards_seafloor, expand_numpy_array)
                data = np.array(np.ma.filled(arr, np.nan), copy=True)
                fill_NaN_towards_seafloor(data)
                for k in range(data.shape[0]):
                    # Valid native point stencils need at most their directly
                    # adjacent masked neighbour filled for float-rounding.
                    if np.isfinite(data[k]).any() and not np.isfinite(data[k]).all():
                        expand_numpy_array(data[k])
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                fields[name] = (data, finite)
                size += data.nbytes + finite.nbytes
            snapshot = {'key':key, 'rx':rx,'ry':ry,'rz':rz,'fields':fields,'bytes':size}
            # Only store bounded immutable raw-source snapshots. Never cache
            # interpolated particle values, geometry or temporal composites.
            if size <= 64*1024*1024:
                blocks = [b for b in blocks if b['key'] != key]
                blocks.append(snapshot)
                while len(blocks) > 2 or sum(b['bytes'] for b in blocks) > 64*1024*1024:
                    blocks.pop(0)
                self._cf_fast_blocks = blocks
        rx, ry, rz = snapshot['rx'], snapshot['ry'], snapshot['rz']
        if (len(rz) < 2 or len(rz) >= 120 or len(rx) < 2 or len(ry) < 2 or
                not np.all(np.isfinite(rx)) or not np.all(np.isfinite(ry)) or
                not np.all(np.isfinite(rz)) or self.global_coverage()):
            raise ValueError('CF_FAST_UNSUPPORTED_GRID')
        n = len(x)
        xlo, xhi, xok = self._cf_batched_brackets(rx, x)
        ylo, yhi, yok = self._cf_batched_brackets(ry, y)
        zlo, zhi, zok = self._cf_batched_brackets(rz, depths)
        stencil_ok = xok & yok & zok
        # These are exactly the coordinates used by Linear2DInterpolator.
        xi = (x - rx[0]) / (rx[-1] - rx[0]) * (len(rx)-1)
        yi = (y - ry[0]) / (ry[-1] - ry[0]) * (len(ry)-1)
        # Linear1DInterpolator truncates z to the native sampling range and
        # interpolates the corresponding *index*, supporting descending z.
        clipped_z = np.clip(depths, rz.min(), rz.max())
        if rz[0] < rz[-1]:
            zfi = interp1d(rz, np.arange(len(rz)))(clipped_z)
        else:
            zfi = interp1d(rz[::-1], np.arange(len(rz))[::-1])(clipped_z)
        below = np.minimum(np.floor(zfi).astype(np.intp) + 1, len(rz)-1)
        above = np.clip(np.floor(zfi).astype(np.intp), 0, len(rz)-1)
        wa = 1 - (zfi - above)
        output = {}
        for name in names:
            data, finite = snapshot['fields'][name]
            good = stencil_ok.copy()
            for kz in (zlo, zhi):
                for jy in (ylo, yhi):
                    for ix in (xlo, xhi):
                        good &= finite[kz, jy, ix]
            # Keep NaNs away from valid stencils; their invalidity is preserved
            # by `good`, so no originally invalid 3-D source becomes admissible.
            # Interpolating only the two native layers needed by each particle
            # avoids constructing 3-D horizontal profiles for every particle.
            aa = map_coordinates(data, [above, yi, xi], order=1,
                                 mode='constant', cval=np.nan, prefilter=False)
            bb = map_coordinates(data, [below, yi, xi], order=1,
                                 mode='constant', cval=np.nan, prefilter=False)
            # Match Linear2DInterpolator's second attempt for points
            # infinitesimally outside the native grid due to rounding:
            # its source-time fill uses nearest-edge mode, not zeroing.
            ma = ~np.isfinite(aa)
            mb = ~np.isfinite(bb)
            if ma.any():
                aa[ma] = map_coordinates(data, [above[ma], yi[ma], xi[ma]],
                                         order=1, mode='nearest', prefilter=False)
            if mb.any():
                bb[mb] = map_coordinates(data, [below[mb], yi[mb], xi[mb]],
                                         order=1, mode='nearest', prefilter=False)
            interp = aa * wa + bb * (1 - wa)
            output[name] = np.ma.masked_where(~good, np.ma.masked_invalid(interp))
        return output

    @staticmethod
    def _cf_wetdry_nearest_indices(axis, positions):
        """Nearest native grid cells; do not interpolate categorical flags."""
        axis = np.asarray(axis, dtype=float)
        positions = np.asarray(positions, dtype=float)
        ascending = axis[0] < axis[-1]
        ordered = axis if ascending else axis[::-1]
        right = np.searchsorted(ordered, positions, side='left')
        hi = np.clip(right, 0, len(ordered) - 1)
        lo = np.clip(right - 1, 0, len(ordered) - 1)
        picked = np.where(np.abs(positions - ordered[lo]) <=
                          np.abs(ordered[hi] - positions), lo, hi)
        if not ascending:
            picked = len(axis) - 1 - picked
        return picked, (np.isfinite(positions) &
                        (positions >= ordered[0]) &
                        (positions <= ordered[-1]))

    def _cf_wetdry_categorical(self, time, reader_x, reader_y):
        """Read the preceding *source* frame at nearest native grid cells.

        The categorical 1/0/-1 values must never be interpolated in space or
        time.  Invalid/missing source cells remain masked, for the normal
        OpenDrift environment fallback/provenance mechanism to handle.
        The underlying forcing Dataset must remain immutable for this Reader's
        lifetime. The single-frame cache changes with source time, not when
        external code mutates an xarray Dataset in place.
        """
        x = np.atleast_1d(np.asarray(reader_x, dtype=float))
        y = np.atleast_1d(np.asarray(reader_y, dtype=float))
        if x.ndim != 1 or y.shape != x.shape:
            raise ValueError('wetdry_state point coordinates must be aligned 1-D arrays')
        source = self.Dataset[self.variable_mapping['wetdry_state']]
        tdim = self.dimensions.get('time')
        source_index = None
        source_time = None
        if tdim in source.dims:
            _nearest, before, _after, _ni, before_i, _ai = self.nearest_time(time)
            if before is None or before_i is None:
                raise ValueError('wetdry_state has no preceding CF source time')
            source_index = int(before_i)
            if (source_index < 0 or source_index >= len(self.times) or
                    self.times[source_index] != before):
                raise ValueError('wetdry_state source time/index mismatch')
            source_time = before
            source = source.isel({tdim: source_index})
        elif time is not None:
            source_time = 'STATIC'
        # The general CF reader's own projection stage supplies reader_x/y.
        # Reject unsupported curvilinear/cyclic coordinate transforms rather
        # than silently selecting a nearby but physically unrelated cell.
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            raise ValueError('wetdry_state lacks rectilinear native coordinates')
        ix, xin = self._cf_wetdry_nearest_indices(self.x, x)
        iy, yin = self._cf_wetdry_nearest_indices(self.y, y)
        in_domain = xin & yin
        # Benchmark-qualified single-frame cache: forcing must be immutable
        # after reader initialization; no in-place xarray Dataset mutation.
        # Only the categorical field is cached, not particle samples or normal
        # structured interpolator data. Cap at 32 MiB; larger frames use xarray.
        ydim, xdim = self.dimensions['y'], self.dimensions['x']
        if set(source.dims) != {ydim, xdim} or len(source.dims) != 2:
            raise ValueError('wetdry_state must be a 2-D categorical field')
        cache_key = (id(self.Dataset), self.variable_mapping['wetdry_state'],
                     source_index, tuple(source.shape), tuple(source.dims))
        existing = getattr(self, '_cf_wetdry_frame_cache', None)
        if existing is not None and existing[0] == cache_key:
            frame = existing[1]
        else:
            frame_bytes = int(source.size) * int(np.dtype(source.dtype).itemsize)
            if frame_bytes <= 32 * 1024 * 1024:
                frame = np.asarray(source.transpose(ydim, xdim).values)
                if frame.shape != (len(self.y), len(self.x)):
                    raise ValueError('wetdry_state native frame shape mismatch')
                self._cf_wetdry_frame_cache = (cache_key, frame)
            else:
                frame = None
                self._cf_wetdry_frame_cache = None
        if frame is not None:
            values = np.asarray(frame[iy, ix], dtype=float)
        else:
            selected = source.isel({
                xdim: xr.DataArray(ix, dims=('_wetdry_point',)),
                ydim: xr.DataArray(iy, dims=('_wetdry_point',))})
            values = np.asarray(selected.values, dtype=float)
        if values.shape != x.shape:
            raise ValueError('wetdry_state unexpected source shape')
        invalid = ~in_domain | ~np.isfinite(values)
        for attr in ('_FillValue', 'missing_value'):
            fill = source.attrs.get(attr, source.encoding.get(attr))
            if fill is not None:
                invalid |= values == fill
        # Bad categorical codes are missing/UNKNOWN, not fractional states.
        invalid |= ~np.isin(values, (-1., 0., 1.))
        self.last_wetdry_state_source = {
            'requested_time': str(time),
            'source_time': str(source_time),
            'source_index': source_index,
            'interpolation': 'preceding_source_frame_nearest_native_cell',
            'valid_count': int(np.count_nonzero(~invalid))}
        return np.ma.array(values, mask=invalid)

    def _get_variables_interpolated_(self, variables, profiles, profiles_depth,
                                     time, reader_x, reader_y, z):
        names = [variables] if isinstance(variables, str) else list(variables)
        if 'wetdry_state' not in names:
            return self._cf_interpolate_source_valid_bottom_water(
                variables, profiles, profiles_depth, time, reader_x, reader_y, z)
        profile_names = ([] if profiles is None else
                         ([profiles] if isinstance(profiles, str) else list(profiles)))
        if 'wetdry_state' in profile_names:
            raise ValueError('wetdry_state cannot be requested as a vertical profile')
        other_names = [name for name in names if name != 'wetdry_state']
        if other_names:
            values, profile = self._cf_interpolate_source_valid_bottom_water(
                other_names, profiles, profiles_depth, time, reader_x, reader_y, z)
        elif profile_names:
            _unused, profile = self._cf_interpolate_source_valid_bottom_water(
                profile_names, profiles, profiles_depth, time, reader_x, reader_y, z)
            values = {}
        else:
            values, profile = {}, None
        values['wetdry_state'] = self._cf_wetdry_categorical(time, reader_x, reader_y)
        return values, profile

    def _cf_interpolate_source_valid_bottom_water(self, variables, profiles, profiles_depth,
                                     time, reader_x, reader_y, z):
        """Reader-local opt-in water-side sampling without altering particle z.

        Native CF z coordinates identify sampling depths, NOT cell bottoms.
        Below the deepest physically wet sample, use that original sample
        (never interpolate with a below-bed native sample). Resolve H and
        the source-level constraint separately at each temporal bracket.
        No change is made to generic CF behavior unless explicitly enabled.
        """
        parent = super()._get_variables_interpolated_
        if not self.bottom_water_sampling:
            return parent(variables, profiles, profiles_depth,
                          time, reader_x, reader_y, z)
        names = [variables] if isinstance(variables, str) else list(variables)
        profile_names = [] if profiles is None else (
            [profiles] if isinstance(profiles, str) else list(profiles))
        depth_dim = self.dimensions['z']
        if not any(depth_dim in self.Dataset[self.variable_mapping[n]].dims
                   for n in names + profile_names):
            return parent(variables, profiles, profiles_depth,
                          time, reader_x, reader_y, z)
        requested_z = np.asarray(z, dtype=float)
        n = len(np.atleast_1d(reader_x))
        if requested_z.ndim != 1 or requested_z.size not in (1, n):
            raise ValueError('Invalid generic CF point-depth shape')
        requested_z = np.broadcast_to(requested_z, (n,))
        _nearest, before, after, *_ = self.nearest_time(time)
        if before is None:
            raise ValueError('No CF time before requested time')
        times = [before]
        if after is not None and before < time < after:
            times.append(after)
        snapshots = []
        for source_time in times:
            geometry_fields = ['sea_floor_depth_below_sea_level']
            if self._bottom_water_eta_variable:
                geometry_fields.append(self._bottom_water_eta_variable)
            geometry_fast = (profiles is None and
                             self.interpolation == 'linearNDFast' and
                             self.convolve is None and
                             all(self.dimensions['z'] not in
                                 self.Dataset[self.variable_mapping[nm]].dims
                                 for nm in geometry_fields))
            if geometry_fast:
                try:
                    geometry = self._cf_source_geometry_at_time(
                        source_time, np.asarray(reader_x, dtype=float),
                        np.asarray(reader_y, dtype=float), geometry_fields)
                except ValueError as exc:
                    if not str(exc).startswith('CF_FAST_UNSUPPORTED_'):
                        raise
                    geometry_fast = False
            if not geometry_fast:
                geometry, _ = parent(geometry_fields, None, None,
                                     source_time, reader_x, reader_y,
                                     np.zeros(n, dtype=float))
            h = np.ma.asarray(geometry['sea_floor_depth_below_sea_level'], dtype=float)
            eta = (np.ma.asarray(geometry[self._bottom_water_eta_variable], dtype=float)
                   if self._bottom_water_eta_variable else np.ma.zeros(n))
            H = np.ma.masked_invalid(h + eta)
            wet = (~np.ma.getmaskarray(H)) & np.isfinite(np.ma.filled(H, np.nan)) & (H > 0)
            native_z = np.asarray(self.z, dtype=float)
            # Only native sampling points above the instantaneous bed can be used.
            eligible = (native_z.reshape(-1, 1) <= 0) & (
                native_z.reshape(-1, 1) >= -np.ma.filled(H, np.nan).reshape(1, -1))
            has_wet_sample = np.any(eligible, axis=0)
            deepest = np.min(np.where(eligible, native_z[:, None], np.inf), axis=0)
            valid = wet & has_wet_sample & np.isfinite(requested_z) & (requested_z <= 0)
            # Effective source-sampling depth is strictly local to this reader;
            # never mutate original z or pre-clamp it for any other reader.
            local_z = np.maximum(requested_z, np.where(valid, deepest, 0.0))
            # StructuredReader caches time blocks by variable, not by z range.
            # Expire ONLY opt-in 3-D blocks that cannot bracket the required
            # local sampling depths; otherwise a shallower previous request
            # could silently change the result of a later deeper request.
            needed = local_z[valid]
            if needed.size:
                for cache in (self.var_block_before, self.var_block_after):
                    for key, block in list(cache.items()):
                        bz = getattr(block, 'z', None)
                        if bz is None:
                            continue
                        bz = np.asarray(bz, dtype=float)
                        if (bz.ndim == 1 and bz.size > 0 and
                                (needed.min() < bz.min() or needed.max() > bz.max())):
                            cache.pop(key, None)
            # Keep the original parent path for profiles, mixed 2-D/3-D
            # variables, nondefault interpolators and convolved grids.
            fast_admitted = (profiles is None and self.interpolation == 'linearNDFast'
                             and self.convolve is None
                             and all(depth_dim in self.Dataset[self.variable_mapping[nm]].dims
                                     for nm in names))
            if fast_admitted:
                try:
                    sampled = self._cf_fast_sample_valid_3d(
                        source_time, np.asarray(reader_x, dtype=float),
                        np.asarray(reader_y, dtype=float), local_z, names)
                    prof = None
                except ValueError as exc:
                    if not str(exc).startswith('CF_FAST_UNSUPPORTED_'):
                        raise
                    fast_admitted = False
            if not fast_admitted:
                sampled, prof = parent(names, profile_names if profiles is not None else None,
                                       profiles_depth, source_time,
                                       reader_x, reader_y, local_z)
                raw_names = [nm for nm in names + profile_names
                             if depth_dim in self.Dataset[self.variable_mapping[nm]].dims]
                source_validity, source_profiles = self._cf_native_source_validity(
                    source_time, reader_x, reader_y, local_z, raw_names,
                    profile_names=[nm for nm in profile_names if nm in raw_names])
                for name in names:
                    if depth_dim in self.Dataset[self.variable_mapping[name]].dims:
                        sampled[name] = np.ma.masked_where(
                            ~valid | ~source_validity[name],
                            np.ma.masked_invalid(np.ma.asarray(sampled[name])))
            else:
                for name in names:
                    sampled[name] = np.ma.masked_where(
                        ~valid, np.ma.masked_invalid(np.ma.asarray(sampled[name])))
            if prof is not None:
                profile_depths = np.asarray(prof['z'], dtype=float)
                for name in profile_names:
                    if depth_dim in self.Dataset[self.variable_mapping[name]].dims:
                        p = np.ma.asarray(prof[name])
                        if p.ndim == 2 and p.shape[1] == n:
                            bad = (~valid[None, :] |
                                   (profile_depths[:, None] < -np.ma.filled(H, np.nan)[None, :]))
                            source_z, source_mask = source_profiles[name]
                            for kk, zval in enumerate(profile_depths):
                                match = np.flatnonzero(np.isclose(source_z, zval, atol=1.e-7))
                                if not len(match):
                                    bad[kk, :] = True
                                else:
                                    bad[kk, :] |= ~source_mask[int(match[0]), :]
                            prof[name] = np.ma.masked_where(bad, np.ma.masked_invalid(p))
            snapshots.append((sampled, prof))
        if len(snapshots) == 1:
            return snapshots[0]
        alpha = (time - before).total_seconds() / (after - before).total_seconds()
        p0, pr0 = snapshots[0]
        p1, pr1 = snapshots[1]
        values = {name: np.ma.masked_invalid(
            (1 - alpha) * np.ma.asarray(p0[name]) +
            alpha * np.ma.asarray(p1[name])) for name in names}
        if profiles is None:
            return values, None
        if pr0 is None or pr1 is None or not np.array_equal(pr0['z'], pr1['z']):
            raise ValueError('Incompatible CF native temporal profile levels')
        output_profiles = {'z': np.asarray(pr0['z']).copy()}
        for name in profile_names:
            output_profiles[name] = np.ma.masked_invalid(
                (1 - alpha) * np.ma.asarray(pr0[name]) +
                alpha * np.ma.asarray(pr1[name]))
        return values, output_profiles

    def get_variables(self, requested_variables, time=None,
                      x=None, y=None, z=None,
                      indrealization=None):

        requested_variables, time, x, y, z, _outside = self.check_arguments(
            requested_variables, time, x, y, z)

        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
            self.nearest_time(time)

        if hasattr(self, 'z') and (z is not None):
            # Find z-index range
            if self.z[0] > self.z[-1]:  # descending
                indices = np.searchsorted(-self.z, [-z.min(), -z.max()])
            else:  # ascending
                indices = np.searchsorted(self.z, [z.min(), z.max()])
            indz = np.arange(np.maximum(0, indices.min() - 1 -
                                        self.verticalbuffer),
                             np.minimum(len(self.z), indices.max() + 1 +
                                        self.verticalbuffer))
            if len(indz) == 1:
                indz = indz[0]  # Extract integer to read only one layer
        else:
            indz = 0

        if indrealization == None:
            if self.realizations is not None:
                indrealization = range(len(self.realizations))
            else:
                indrealization = None

        # Find indices corresponding to requested x and y
        if hasattr(self, 'clipped'):
            clipped = self.clipped
        else: clipped = 0

        buffer = self.buffer  # Adding buffer, to cover also future positions of elements
        indy = np.floor(np.abs(y-self.y[0])/self.delta_y-clipped).astype(int) + clipped
        indy = np.arange(np.max([0, indy.min()-buffer]),
                         np.min([indy.max()+buffer, self.numy]))

        if self.global_coverage():  # Treatment of cyclic longitudes (x-coordinate)
            if self.lon_range() == '0to360':
                x = np.mod(x, 360)  # Shift x/lons to 0-360
            elif self.lon_range() == '-180to180':
                x = np.mod(x + 180, 360) - 180 # Shift x/lons to -180-180
        indx = np.floor(np.abs(x-self.x[0])/self.delta_x-clipped).astype(int) + clipped

        split = False
        if self.global_coverage():  # Check if need to split in two blocks
            uniqx = np.unique(indx)
            diff_xind = np.diff(uniqx)
            # We split if >800 pixels between left/west and right/east blocks
            if len(diff_xind)>=1 and diff_xind.max() > np.minimum(800, 0.6*self.numx):
                logger.debug('Requested data block crosses lon-border, reading and concatinating two parts')
                split = True
                splitind = np.argmax(diff_xind)
                indx_left = np.arange(0, uniqx[splitind] + buffer)
                indx_right = np.arange(uniqx[splitind+1] - buffer, self.numx)
                indx = np.concatenate((indx_right, indx_left))
        if split is False:
            indx = np.arange(np.maximum(0, indx.min()-buffer),
                             np.minimum(indx.max()+buffer+1, self.numx))

        variables = {}

        for par in requested_variables:
            if hasattr(self, 'rotate_mapping') and par in self.rotate_mapping:
                logger.debug('Using %s to retrieve %s' %
                    (self.rotate_mapping[par], par))
                if par not in self.variable_mapping:
                    self.variable_mapping[par] = \
                        self.variable_mapping[
                            self.rotate_mapping[par]]
            var = self.Dataset.variables[self.variable_mapping[par]]

            ensemble_dim = None
            dimindices = {'x': indx, 'y': indy, 'time': indxTime, 'z': indz}
            dimorder = list(var.dims)
            xnum = dimorder.index(self.dimensions['x'])
            ynum = dimorder.index(self.dimensions['y'])
            if xnum < ynum:
                # We must have y before x, since returning numpy arrays and not Xarrays
                logger.debug(f'Swapping order of x-y dimensions for {par}')
                dimorder[xnum] = self.dimensions['y']
                dimorder[ynum] = self.dimensions['x']
                var = var.permute_dims(*dimorder)

            # Remove any unknown dimensions
            for dim in var.dims:
                if dim not in self.dimensions.values() and dim != self.ensemble_dimension:
                    logger.debug(f'Removing unknown dimension: {dim}')
                    var = var.squeeze(dim=dim)
            if self.ensemble_dimension is not None and self.ensemble_dimension in var.dims:
                ensemble_dim = 0  # hardcoded, may not work for MEPS
            if split is False:
                subset = {vdim:dimindices[dim] for dim,vdim in self.dimensions.items() if vdim in var.dims}
                variables[par] = var.isel(subset)
            else:  # We need to read left and right parts separately
                d_left = dimindices.copy()
                d_right = dimindices.copy()
                d_left.update({'x': indx_left})
                d_right.update({'x': indx_right})
                subset_left = {vdim:d_left[dim] for dim,vdim in self.dimensions.items()
                               if vdim in var.dims}
                subset_right = {vdim:d_right[dim] for dim,vdim in self.dimensions.items()
                               if vdim in var.dims}
                left = var.isel(subset_left)
                right = var.isel(subset_right)

                variables[par] = xr.Variable.concat([right, left], dim=self.dimensions['x'])
                variables[par] = np.ma.masked_invalid(variables[par])

            # Mask values outside domain
            variables[par] = np.ma.array(variables[par],
                                         ndmin=2, mask=False)
            # Mask extreme values which might have slipped through
            with np.errstate(invalid='ignore'):
                variables[par] = np.ma.masked_outside(
                    variables[par], -30000, 30000)

            # Ensemble blocks are split into lists
            if ensemble_dim is not None:
                num_ensembles = variables[par].shape[ensemble_dim]
                logger.debug(f'Num ensembles for {par}: {num_ensembles}')
                newvar = [0]*num_ensembles
                for ensemble_num in range(num_ensembles):
                    newvar[ensemble_num] = \
                        np.take(variables[par],
                                ensemble_num, ensemble_dim)
                variables[par] = newvar

        # Store coordinates of returned points
        try:
            variables['z'] = self.z[indz]
        except:
            variables['z'] = None
        if self.projected is True:
            variables['x'] = self.x[indx]
            variables['y'] = self.y[indy]
            if split is True and not np.all(np.diff(variables['x'])>0):  # not monotonously increasing
                variables['x'] = np.mod(variables['x'], 360)  # Shift to 0-360
                if not np.all(np.diff(variables['x'])>0):  # not increasing, shift again to -180-180
                    variables['x'] = np.mod(variables['x'] + 180, 360) - 180
        else:
            variables['x'] = indx
            variables['y'] = indy
        variables['x'] = np.asarray(variables['x'], dtype=np.float32)
        variables['y'] = np.asarray(variables['y'], dtype=np.float32)

        variables['time'] = nearestTime

        # Rotate any east/north vectors if necessary
        if hasattr(self, 'rotate_mapping'):
            if self.y_is_north() is True:
                logger.debug('North is up, no rotation necessary')
            else:
                rx, ry = np.meshgrid(variables['x'], variables['y'])
                lon, lat = self.xy2lonlat(rx, ry)
                from opendrift.readers.basereader import vector_pairs_xy
                for vectorpair in vector_pairs_xy:
                    if vectorpair[0] in self.rotate_mapping and vectorpair[0] in variables.keys():
                        if self.proj.__class__.__name__ == 'fakeproj':
                            logger.warning('Rotation from fakeproj is not yet implemented, skipping.')
                            continue
                        logger.debug(f'Rotating vector from east/north to xy orientation: {vectorpair[0:2]}')
                        variables[vectorpair[0]], variables[vectorpair[1]] = self.rotate_vectors(
                            lon, lat, variables[vectorpair[0]], variables[vectorpair[1]],
                            pyproj.Proj('+proj=latlong'), self.proj)

        if hasattr(self, 'shift_x'):
            # "hidden feature": if reader.shift_x and reader.shift_y are defined,
            # the returned fields are shifted this many meters in the x- and y directions
            # E.g. reader.shift_x=10000 gives a shift 10 km eastwards (if x is east direction)
            if self.proj.crs.is_geographic:  # meters to degrees
                shift_y = (self.shift_y/111000)
                shift_x = (self.shift_x/111000)*np.cos(np.radians(variables['y']))
                logger.info('Shifting x between %s and %s' % (shift_x.min(), shift_x.max()))
                logger.info('Shifting y with %s m' % shift_y)
            else:
                shift_x = self.shift_x
                shift_y = self.shift_y
                logger.info('Shifting x with %s m' % shift_x)
                logger.info('Shifting y with %s m' % shift_y)
            variables['x'] += shift_x
            variables['y'] += shift_y

        return variables
