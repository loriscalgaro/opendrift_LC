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

from opendrift.readers.basereader import BaseReader, UnstructuredReader


class Reader(BaseReader, UnstructuredReader):
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

        ref_time = datetime.fromisoformat(self.dataset['time'].units[14:33])

        self.times = np.array([
            ref_time + timedelta(seconds=d.item())
            for d in self.dataset['time'][:]
        ])
        self.start_time = self.times[0]
        self.end_time = self.times[-1]
        # time steps are not constant

        self.xmin = np.min(self.x)
        self.xmax = np.max(self.x)
        self.ymin = np.min(self.y)
        self.ymax = np.max(self.y)

        self._init_vertical_coordinates()
        self._init_variable_mapping()

        # Run constructor of parent Reader class
        super().__init__()

        self.boundary = self._build_boundary_polygon_(self.x.compressed(),
                                                      self.y.compressed())

        self.timer_start("build index")
        logger.debug("building index of nodes..")
        self.nodes_idx = self._build_ckdtree_(self.x, self.y)
        self.timer_end("build index")

        self.timer_end("open dataset")

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
        coordinate_variables = set(['time', 'longitude', 'latitude'])
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

        # If every requested variable is 1D/2D, the requested z must not affect
        # either argument validation or returned values. This guarantees that
        # particles at the same lon/lat/time receive the same bottom-variable
        # value even when their requested z values differ.
        z_independent_request = self._all_requested_variables_are_z_independent(
            requested_variables
        )
        if z_independent_request:
            z_for_check = np.zeros_like(x, dtype=float)
        else:
            z_for_check = z

        requested_variables, time, x, y, z_checked, _outside = \
            self.check_arguments(requested_variables, time, x, y, z_for_check)

        if z_independent_request:
            z = z_checked
        else:
            z = z_checked

        nearest_time, _time_before, _time_after, indx_nearest, _indx_before, _indx_after = self.nearest_time(
            time)

        logger.debug("Nearest time: %s" % nearest_time)

        variables = {}

        logger.debug("Interpolating node-variables..")

        nodes = self._nearest_node_(x, y)
        assert len(nodes) == len(x)

        for var in requested_variables:
            dvar_name = self.variable_mapping.get(var)
            logger.debug("Interpolating: %s (%s)" % (var, dvar_name))
            dvar = self.dataset[dvar_name]

            if len(dvar.shape) > 2:
                if not self.has_vertical_levels:
                    raise ValueError(
                        'Variable %s has more than two dimensions, but dataset '
                        'has no level/levels coordinate variable.' % var
                    )

                level_ind = self.__nearest_level__(z)

                # Reading the smallest block covering the actual data
                block = dvar[indx_nearest,
                             slice(nodes.min(),
                                   nodes.max() + 1),
                             slice(level_ind.min(),
                                   level_ind.max() + 1), ]

                # Picking the nearest value
                variables[var] = block[
                        nodes - nodes.min(),
                        level_ind - level_ind.min(),
                        ]
            elif len(dvar.shape) == 2:
                # Reading the smallest block covering the actual data
                # Variables with dimensions (time, node) have no vertical
                # dependence. z is intentionally ignored.
                block = dvar[indx_nearest,
                             slice(nodes.min(),
                                   nodes.max() + 1), ]

                # Picking the nearest value
                variables[var] = block[
                        nodes - nodes.min(),
                        ]
            elif len(dvar.shape) == 1:
                # Reading the smallest block covering the actual data
                # Variables with dimensions (node) have no time or vertical
                # dependence. z is intentionally ignored.
                block = dvar[slice(nodes.min(),
                                   nodes.max() + 1), ]

                # Picking the nearest value
                variables[var] = block[
                        nodes - nodes.min(),
                        ]
            else:
                logger.error('unknown dimensionality')

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
