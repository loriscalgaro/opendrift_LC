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
# Copyright 2021, Gaute Hope, MET Norway

import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset, MFDataset
import logging
logger = logging.getLogger(__name__)

from opendrift.readers.basereader.unstructured import UnstructuredReader


class Reader(UnstructuredReader):
    """
    A reader for unstructured SHYFEM (irregularily gridded) `CF compliant
    <https://cfconventions.org/>`_ netCDF files.

    http://www.ismar.cnr.it/shyfem

    Args:
        :param filename: A single netCDF file, or a pattern of files. The
                         netCDF file can also be an URL to an OPeNDAP server.
        :type filename: string, requiered.

        :param name: Name of reader
        :type name: string, optional

    .. seealso::

        py:mod:`opendrift.readers.basereader.unstructured`.
    """

    variable_aliases = {
        'eastward_sea_water_velocity': 'x_sea_water_velocity',
        'northward_sea_water_velocity': 'y_sea_water_velocity',
        'sea_floor_depth_below_sea_surface': 'sea_floor_depth_below_sea_level'
    }

    dataset = None
    vertical_dimension_names = ('level', 'levels')

    def __init__(self, filename=None, name=None):
        if filename is None:
            raise ValueError('Filename is missing')
        filestr = str(filename)
        if name is None:
            self.name = filestr
        else:
            self.name = name

        # xarray currently does not handle this type of grid:
        # https://github.com/pydata/xarray/issues/2233

        self.timer_start("open dataset")
        logger.info('Opening dataset: ' + filestr)
        if ('*' in filestr) or ('?' in filestr) or ('[' in filestr):
            logger.info('Opening files with MFDataset')
            self.dataset = MFDataset(filename)
        else:
            logger.info('Opening file with Dataset')
            self.dataset = Dataset(filename, 'r')

        self.proj4 = '+proj=lonlat'

        logger.info('Reading grid and coordinate variables..')

        self.x, self.y = self.dataset['longitude'][:], self.dataset[
            'latitude'][:]

        self._init_time_coordinates()

        self.xmin = np.min(self.x)
        self.xmax = np.max(self.x)
        self.ymin = np.min(self.y)
        self.ymax = np.max(self.y)

        self._init_vertical_coordinates()
        self._init_variable_mapping()

        # Run constructor of parent Reader class
        super().__init__()

        x_boundary = np.ma.asarray(self.x).compressed()
        y_boundary = np.ma.asarray(self.y).compressed()

        self.boundary = self._build_boundary_polygon_(x_boundary, y_boundary)

        self.timer_start("build index")
        logger.debug("building index of nodes..")
        self.nodes_idx = self._build_ckdtree_(self.x, self.y)
        self.timer_end("build index")

        self.timer_end("open dataset")

    def _init_time_coordinates(self):
        """Initialize time coordinates, allowing fully static node files."""
        if 'time' in self.dataset.variables:
            time_var = self.dataset['time']
            units = getattr(time_var, 'units', '')

            # Expected common format: "seconds since YYYY-MM-DDTHH:MM:SS"
            if 'since' in units:
                ref_string = units.split('since', 1)[1].strip()
                # Be tolerant of trailing timezone or missing T
                ref_string = ref_string.replace('Z', '').strip()
                ref_string = ref_string.strip()
                if ref_string.upper().endswith(" UTC"):
                    ref_string = ref_string[:-4].strip() + "+00:00"
                ref_time = datetime.fromisoformat(ref_string)

                # Keep reader times timezone-naive
                if ref_time.tzinfo is not None:
                    ref_time = ref_time.replace(tzinfo=None)
            else:
                raise ValueError(
                    "time variable exists but has no CF-like 'units' attribute "
                    "containing 'since'."
                )

            self.times = np.array([
                ref_time + timedelta(seconds=float(d))
                for d in time_var[:]
            ])
            self.start_time = self.times[0]
            self.end_time = self.times[-1]
            self.has_time = True

        else:
            # Fully static unstructured node file.
            # Static variables do not depend on requested time.
            logger.info(
                'No time variable found; treating dataset as fully static '
                'unstructured node data.'
            )
            self.times = np.array([datetime(1970, 1, 1)])
            self.start_time = self.times[0]
            self.end_time = self.times[-1]
            self.has_time = False
            self.always_valid = True

    def _init_vertical_coordinates(self):
        """Initialize z coordinates, allowing depth-collapsed 2D files."""
        self.level_var_name = None
        for candidate in self.vertical_dimension_names:
            if candidate in self.dataset.variables:
                self.level_var_name = candidate
                break

        if self.level_var_name is not None:
            # Levels are the depth of the bottom of each layer. Re-assign to
            # middle of layer for nearest interpolation.
            levels = self.dataset[self.level_var_name][:]
            self.z = -levels
            self.z = np.insert(self.z, 0, [0.])
            self.z = self.z[:-1] + (np.diff(self.z) / 2)
            assert len(self.z) == len(levels)
            self.zmin, self.zmax = np.min(self.z), 0.
            assert (self.z <= 0).all()
            self.has_vertical_levels = True
        else:
            # Files may contain only (time, node) variables.
            # For those variables, z is not used.
            logger.info(
                'No level/levels variable found; treating dataset as '
                'depth-collapsed unstructured node data. Requested z values '
                'will be ignored for variables without a vertical dimension.'
            )
            self.z = np.array([0.])
            # Keep a permissive range so BaseReader.check_arguments does not
            # discard particles solely because a 2D variable is requested with
            # a nonzero z value.
            self.zmin, self.zmax = -1.0e12, 1.0e12
            self.has_vertical_levels = False

    def _init_variable_mapping(self):
        """Map standard_name to dataset variable name."""
        self.variable_mapping = {}
        coordinate_variables = set(['time', 'longitude', 'latitude', 'node'])
        coordinate_variables.update(self.vertical_dimension_names)

        for var_name in self.dataset.variables:
            # Skip coordinate variables.
            if var_name in coordinate_variables:
                continue

            var = self.dataset[var_name]
            if 'standard_name' in var.ncattrs():
                std_name = getattr(var, 'standard_name')
                std_name = self.variable_aliases.get(std_name, std_name)
                self.variable_mapping[std_name] = str(var_name)

        self.variables = list(self.variable_mapping.keys())

    def _variable_has_vertical_dimension(self, standard_name):
        """Return True if the mapped variable needs vertical interpolation."""
        var_name = self.variable_mapping.get(standard_name)
        if var_name is None:
            # Unknown variables will be handled by check_arguments. Treat as
            # vertical to avoid bypassing normal validation.
            return True

        var = self.dataset[var_name]
        dims = getattr(var, 'dimensions', ())
        if any(dim in self.vertical_dimension_names for dim in dims):
            return True

        # Backward-compatible fallback for datasets with unnamed/nonstandard
        # vertical axes but original SHYFEM variable shape (time, node, level).
        return len(var.shape) > 2

    def _variable_has_time_dimension(self, standard_name):
        """Return True if the mapped variable has a time dimension."""
        var_name = self.variable_mapping.get(standard_name)
        if var_name is None:
            return True

        var = self.dataset[var_name]
        dims = getattr(var, 'dimensions', ())

        if 'time' in dims:
            return True

        # Backward-compatible fallback:
        # 2D variables are often (time, node),
        # but only interpret 2D as time-dependent if the dataset actually has time.
        return bool(getattr(self, 'has_time', True)) and len(var.shape) == 2

    def _all_requested_variables_are_z_independent(self, requested_variables):
        """Return True if all requested variables can safely ignore z."""
        if isinstance(requested_variables, str):
            requested = [requested_variables]
        else:
            requested = list(requested_variables)

        return all(
            not self._variable_has_vertical_dimension(var)
            for var in requested
        )

    def plot_mesh(self,corners=None):
        """
        Plot the grid mesh. Does not automatically show the figure.
        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(self.x, self.y, marker='x', color='blue', label='nodes')

        x, y = getattr(self.boundary, 'context').exterior.xy
        plt.plot(x, y, color='green', label='boundary')

        plt.legend()
        plt.title('Unstructured grid: %s\n%s' % (self.name, self.proj))
        plt.xlabel('lon [deg E]')
        plt.ylabel('lat [deg N]')

        if corners is not None:
            plt.xlim(corners[0],corners[1])
            plt.ylim(corners[2],corners[3])

    def get_variables(self,
                      requested_variables,
                      time=None,
                      x=None,
                      y=None,
                      z=None):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if z is None:
            z = np.zeros_like(x, dtype=float)
        else:
            z = np.atleast_1d(z)
        #if len(z) == 1:
        #    z = z[0] * np.ones(x.shape)

        logger.debug("Requested variabels: %s, lengths: %d, %d, %d" %
                     (requested_variables, len(x), len(y), len(z)))

        # Normalize requested_variables only for the preliminary checks below.
        # check_arguments() may return its own normalized list, so we do not
        # replace requested_variables permanently yet.
        if isinstance(requested_variables, str):
            requested_for_flags = [requested_variables]
        else:
            requested_for_flags = list(requested_variables)

        # If every requested variable is 1D/2D, the requested z must not affect
        # either argument validation or returned values. This guarantees that
        # particles at the same lon/lat/time receive the same bottom-variable
        # value even when their requested z values differ.
        z_independent_request = self._all_requested_variables_are_z_independent(
            requested_for_flags
        )
        if z_independent_request:
            z_for_check = np.zeros_like(x, dtype=float)
        else:
            z_for_check = z

        # If every requested variable is time-independent, the requested model time
        # must not affect either argument validation or returned values.
        time_independent_request = all(
            not self._variable_has_time_dimension(var)
            for var in requested_for_flags
        )

        if time_independent_request:
            # Use the reader's internal static time for validation. This avoids
            # check_arguments() rejecting valid static variables only because the
            # model time is outside the artificial static time interval.
            time_for_check = self.times[0]
        else:
            time_for_check = time

        requested_variables, time, x, y, z_checked, _outside = \
            self.check_arguments(
                requested_variables,
                time_for_check,
                x,
                y,
                z_for_check
            )

        if z_independent_request:
            z = z_checked
        else:
            z = z_checked

        # Recalculate after check_arguments(), because requested_variables may have
        # been normalized or filtered by the base reader.
        time_independent_request = all(
            not self._variable_has_time_dimension(var)
            for var in requested_variables
        )

        if time_independent_request:
            indx_nearest = 0
            logger.debug("Time-independent request: using static index 0")
        else:
            nearest_time, _time_before, _time_after, indx_nearest, _indx_before, _indx_after = \
                self.nearest_time(time)
            logger.debug("Nearest time: %s" % nearest_time)

        variables = {}

        logger.debug("Interpolating node-variables..")

        nodes = self._nearest_node_(x, y)
        assert len(nodes) == len(x)

        for var in requested_variables:
            dvar_name = self.variable_mapping.get(var)

            if dvar_name is None:
                raise KeyError(
                    "Requested variable %s is not available in reader %s. "
                    "Available variables are: %s" %
                    (var, self.name, self.variables)
                )

            logger.debug("Interpolating: %s (%s)" % (var, dvar_name))
            dvar = self.dataset[dvar_name]
            dims = getattr(dvar, 'dimensions', ())

            has_time_dim = 'time' in dims
            has_vertical_dim = any(
                dim in self.vertical_dimension_names
                for dim in dims
            )

            # Identify the node dimension robustly. For SHYFEM node variables this
            # is normally named "node", but we also allow the common fallback where
            # longitude/latitude have a single shared dimension.
            if 'node' in dims:
                node_dim_name = 'node'
            else:
                lon_dims = getattr(self.dataset['longitude'], 'dimensions', ())
                node_dim_name = lon_dims[0] if len(lon_dims) == 1 and lon_dims[0] in dims else None

            if node_dim_name is None:
                raise ValueError(
                    "Variable %s (%s) has dimensions %s, but no node dimension "
                    "matching longitude/latitude could be identified." %
                    (var, dvar_name, dims)
                )

            node_axis = dims.index(node_dim_name)

            if has_vertical_dim:
                if not self.has_vertical_levels:
                    raise ValueError(
                        'Variable %s has more than two dimensions, but dataset '
                        'has no level/levels coordinate variable.' % var
                    )

                level_ind = self.__nearest_level__(z)
                vertical_dim_name = next(
                    dim for dim in dims
                    if dim in self.vertical_dimension_names
                )
                vertical_axis = dims.index(vertical_dim_name)

                # Reading the smallest block covering the actual data
                indexer = [slice(None)] * len(dims)

                if has_time_dim:
                    indexer[dims.index('time')] = indx_nearest

                indexer[node_axis] = slice(nodes.min(), nodes.max() + 1)
                indexer[vertical_axis] = slice(level_ind.min(), level_ind.max() + 1)

                block = dvar[tuple(indexer)]

                # Picking the nearest value
                # After indexing with an integer time index, dimensions after the
                # time axis shift left by one. Convert original axes to block axes.
                block_dims = list(dims)
                if has_time_dim:
                    time_axis = dims.index('time')
                    block_dims.pop(time_axis)
                else:
                    time_axis = None

                block_node_axis = block_dims.index(node_dim_name)
                block_vertical_axis = block_dims.index(vertical_dim_name)

                node_local = nodes - nodes.min()
                level_local = level_ind - level_ind.min()

                if block_node_axis == 0 and block_vertical_axis == 1:
                    variables[var] = block[node_local, level_local]
                elif block_node_axis == 1 and block_vertical_axis == 0:
                    variables[var] = block[level_local, node_local]
                else:
                    raise ValueError(
                        "Variable %s (%s) has unsupported node/vertical axis "
                        "order after slicing: %s" %
                        (var, dvar_name, block_dims)
                    )

            elif has_time_dim:
                # Reading the smallest block covering the actual data
                # Variables with dimensions (time, node) have no vertical
                # dependence. z is intentionally ignored.
                indexer = [slice(None)] * len(dims)
                indexer[dims.index('time')] = indx_nearest
                indexer[node_axis] = slice(nodes.min(), nodes.max() + 1)

                block = dvar[tuple(indexer)]

                # Picking the nearest value
                variables[var] = block[
                        nodes - nodes.min(),
                        ]

            else:
                # Reading the smallest block covering the actual data
                # Variables with dimensions (node) have no time or vertical
                # dependence. z is intentionally ignored.
                indexer = [slice(None)] * len(dims)
                indexer[node_axis] = slice(nodes.min(), nodes.max() + 1)

                block = dvar[tuple(indexer)]

                # Picking the nearest value
                variables[var] = block[
                        nodes - nodes.min(),
                        ]

        return variables

    def __nearest_level__(self, z):
        """
        Find nearest index of z in levels.
        """
        if not self.has_vertical_levels:
            raise ValueError(
                'Cannot interpolate vertically because dataset has no '
                'level/levels variable.'
            )
        return np.argmin(np.abs(self.z[:, None] - z), axis=0)
