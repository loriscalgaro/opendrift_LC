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
from collections import OrderedDict
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
        'sea_floor_depth_below_sea_surface': 'sea_floor_depth_below_sea_level',
        'water_surface_height_above_reference_datum': 'sea_surface_height'
    }

    dataset = None
    vertical_dimension_names = ('level', 'levels')

    def __init__(self, filename=None, name=None, *, wetdry=None,
                 horizontal_interpolation='nearest', mesh_filename=None):
        if horizontal_interpolation not in ('nearest', 'barycentric'):
            raise ValueError('horizontal_interpolation must be nearest or barycentric')
        if horizontal_interpolation == 'barycentric' and wetdry is not None and wetdry.get('method') != 'element_depth_hysteresis':
            raise ValueError('Barycentric D19 requires SHYFEM element_depth_hysteresis')
        if horizontal_interpolation == 'barycentric' and mesh_filename is None:
            raise ValueError('Barycentric interpolation requires explicitly supplied native mesh_filename')
        self._horizontal_interpolation = horizontal_interpolation
        self._barycentric_mesh_filename = mesh_filename
        self._barycentric_geometry_cache = None
        self._barycentric_cache_hits = 0
        self._barycentric_cache_misses = 0
        self._barycentric_wet_state_matrix = None
        self._barycentric_element_source_depths = None
        self._last_dry_current_profile_provenance = {}
        self._last_barycentric_vertex_source = {}
        self._configure_optional_wetdry(wetdry)
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
        if (self._horizontal_interpolation == 'barycentric' and
                self._wetdry_config is not None and
                self._wetdry_config['method'] == 'element_depth_hysteresis'):
            self.variables.append('wetdry_state')
        self.last_wetdry_state_source = None

        # Run constructor of parent Reader class
        super().__init__()

        x_boundary = np.ma.asarray(self.x).compressed()
        y_boundary = np.ma.asarray(self.y).compressed()

        self.boundary = self._build_boundary_polygon_(x_boundary, y_boundary)

        self.timer_start("build index")
        logger.debug("building index of nodes..")
        self.nodes_idx = self._build_ckdtree_(self.x, self.y)
        self.timer_end("build index")

        if self._horizontal_interpolation == 'barycentric':
            self._initialize_native_barycentric_mesh()
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
            # Midpoints select a layer, but its bottom is the physical coverage
            # boundary. The deepest midpoint is not the bottom boundary.
            self.native_level_bottoms_m = np.asarray(levels, dtype=float)
            if (not np.isfinite(self.native_level_bottoms_m).all()
                    or np.any(self.native_level_bottoms_m <= 0)
                    or np.any(np.diff(self.native_level_bottoms_m) <= 0)):
                raise ValueError('SHYFEM layer-bottom depths must increase and be positive')
            # The nominal deepest interface is measured from the source
            # datum. Its instantaneous surface-relative depth moves with eta.
            # Avoid a stale GLOBAL zmin veto; per-record physical wet depth,
            # native interfaces and masks are checked after node/time lookup.
            self.zmin, self.zmax = -1.0e12, 0.
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

    @staticmethod
    def _validate_wetdry(config):
        """Explicit opt-in. No run-specific defaults or undocumented href conversion."""
        if config is None:
            return None
        required = {'method', 'href', 'hzmin', 'hzoff', 'hzon',
                    'depth_variable', 'water_level_variable', 'datum',
                    'dry_current'}
        optional = {'dry_zero_time_mode'}
        if (not isinstance(config, dict) or
                not required <= set(config) or
                set(config) - required - optional):
            raise ValueError('wetdry requires prescribed keys; optional dry_zero_time_mode')
        if config['method'] not in ('nodal_depth_hysteresis','element_depth_hysteresis'):
            raise ValueError('Unsupported wetdry method')
        if config['datum'] != 'shared_reference_datum':
            raise ValueError('Only verified shared-reference-datum convention is supported')
        if config['dry_current'] != 'zero':
            raise ValueError('D19 permits only explicitly configured dry_current=zero')
        if config.get('dry_zero_time_mode', 'piecewise') not in ('piecewise','linear'):
            raise ValueError('Unsupported dry_zero_time_mode')
        for var in ('depth_variable','water_level_variable'):
            if not isinstance(config[var], str) or not config[var]:
                raise ValueError('wetdry variable names must be nonempty strings')
        for v in ('href','hzmin','hzoff','hzon'):
            if isinstance(config[v],bool) or not isinstance(config[v], (int,float,np.number)) or not np.isfinite(config[v]):
                raise ValueError(v+' must be a finite numeric value')
        if float(config['href']) != 0.0:
            # The 2018 run is verified only for href=0. The semantics of
            # nonzero href must be source-verified, not guessed as bathy offset.
            raise ValueError('Nonzero href semantics are not qualified by the nodal-depth method')
        if not (0 <= config['hzmin'] <= config['hzoff'] < config['hzon']):
            raise ValueError('Require 0 <= hzmin <= hzoff < hzon')
        result=dict(config)
        result.setdefault('dry_zero_time_mode','piecewise')
        return result

    def _configure_optional_wetdry(self, config):
        self._wetdry_config = self._validate_wetdry(config)
        self._wetdry_states = {}
        self._wetdry_depths = {}
        self._last_dry_current_provenance = {}
        self._last_dry_current_profile_provenance = {}

    def _optional_wet_state(self, nearest_time_index, nodes):
        """Cached, chronology-independent nodal proxy: 0=dry, 1=wet, -1=unknown.

        Not equivalent to native SHYFEM element wet/dry flags. Exact-equality
        thresholds remain unknown until original source comparison is verified.
        """
        cfg = self._wetdry_config
        if cfg is None:
            raise RuntimeError('Optional wetdry not configured')
        dv, ev = cfg['depth_variable'], cfg['water_level_variable']
        if dv not in self.dataset.variables or ev not in self.dataset.variables:
            raise ValueError('Configured wetdry source variable is missing')
        if (self.dataset[dv].dimensions != ('node',) or
                self.dataset[ev].dimensions != ('time','node')):
            raise ValueError('Unqualified wetdry nodal depth / source-time water-level dimensions')
        if len(self.dataset[ev]) != len(self.times):
            raise ValueError('wetdry state-time dimension does not match the reader source timestamps')
        result = np.full(len(nodes), -1, dtype=np.int8)
        for i,node in enumerate(np.asarray(nodes,dtype=int)):
            node = int(node)
            if node not in self._wetdry_states:
                bathy = np.ma.filled(np.ma.asarray(self.dataset[dv][node]), np.nan)
                eta = np.ma.filled(np.ma.asarray(self.dataset[ev][:, node]), np.nan)
                H = np.asarray(eta, dtype=float) + float(bathy)
                states = np.full(len(H), -1, dtype=np.int8)
                previous = -1
                for t, depth in enumerate(H):
                    if not np.isfinite(depth):
                        previous = -1
                    elif depth < cfg['hzoff']:
                        previous = 0
                    elif depth > cfg['hzon']:
                        previous = 1
                    elif depth == cfg['hzoff'] or depth == cfg['hzon']:
                        previous = -1
                    states[t] = previous
                if len(self._wetdry_states) >= 256:
                    oldest = next(iter(self._wetdry_states))
                    self._wetdry_states.pop(oldest)
                    self._wetdry_depths.pop(oldest, None)
                self._wetdry_states[node] = states
                self._wetdry_depths[node] = H
            if 0 <= nearest_time_index < len(self._wetdry_states[node]):
                result[i] = self._wetdry_states[node][nearest_time_index]
        return result

    def _barycentric_vertex_states(self, time_index, vertices, face_ids=None, *, include_depths=True):
        """SHYFEM-compatible hysteretic ELEMENT states from native nodal depths.

        This deliberately preserves the historic call name. It DOES NOT classify
        vertices individually. Return (element_state, nodal_depths_at_time).
        Element DRY: at least one shallow vertex (min H < hzoff).
        Element WET: all vertices meet rewetting threshold (min H > hzon).
        Between thresholds, retain the most recent KNOWN element state.
        Unknown input/initialization/equality without known state => UNKNOWN.
        Equality uses the previous known element state, never an invented state.
        Source output only has hourly nodal depths: reconstructed source-frame
        state is NOT authenticated native sub-timestep SHYFEM element flags.
        """
        if self._wetdry_config is None or self._horizontal_interpolation != 'barycentric':
            raise RuntimeError('Element hysteresis requires explicit barycentric D19 opt-in')
        cfg = self._wetdry_config
        if cfg['method'] != 'element_depth_hysteresis':
            raise ValueError('Barycentric D19 requires element_depth_hysteresis; independent nodal states are not native SHYFEM')
        vertices = np.asarray(vertices, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError('Expected three original native vertices')
        if np.any(vertices<0) or np.any(vertices>=len(self.x)):
            raise ValueError('Native vertex out of range')
        ti = int(time_index)
        if ti < 0 or ti >= len(self.times):
            raise ValueError('D19 time index out of range')
        if self._barycentric_wet_state_matrix is None:
            dv, ev = cfg['depth_variable'], cfg['water_level_variable']
            if dv not in self.dataset.variables or ev not in self.dataset.variables:
                raise ValueError('Original native nodal bathymetry and water level required')
            dvar, evar = self.dataset[dv], self.dataset[ev]
            if dvar.dimensions != ('node',) or evar.dimensions != ('time','node') or len(evar)!=len(self.times):
                raise ValueError('Unqualified native source dimensions/time')
            bathy=np.ma.filled(np.ma.asarray(dvar[:]).astype(float),np.nan)
            eta=np.ma.filled(np.ma.asarray(evar[:]).astype(float),np.nan)
            H=eta+bathy[None,:]
            if H.shape != (len(self.times),len(self.x)):
                raise ValueError('Invalid native water depth array geometry')
            faces=self._barycentric_faces
            states=np.full((len(self.times),len(faces)),-1,dtype=np.int8)
            previous=np.full(len(faces),-1,dtype=np.int8)
            # Vectorized over native original ELEMENTS. Iterate in source time
            # only; cache is independent of the order in which particles query.
            for it in range(len(self.times)):
                H3=H[it,faces]
                finite=np.all(np.isfinite(H3),axis=1)
                hmin=np.min(np.where(np.isfinite(H3),H3,np.inf),axis=1)
                dry=finite & (hmin < cfg['hzoff'])
                wet=finite & (hmin > cfg['hzon'])
                previous=np.where(dry,0,np.where(wet,1,previous)).astype(np.int8)
                # Missing present-frame nodal depths make PRESENT state unknown.
                # They do not fabricate or delete the earlier KNOWN hysteresis state.
                states[it]=np.where(finite,previous,-1)
            self._barycentric_wet_state_matrix=states
            self._barycentric_element_source_depths=H
        if face_ids is None:
            face_ids=np.fromiter((self._barycentric_face_index_by_vertices.get(
                  tuple(sorted(map(int, row))),-1) for row in vertices),dtype=np.int64,count=len(vertices))
        face_ids=np.asarray(face_ids,dtype=np.int64)
        if face_ids.shape!=(len(vertices),) or np.any((face_ids<0)|(face_ids>=len(self._barycentric_faces))):
            raise ValueError('Barycentric vertices must map to authenticated native faces')
        if not np.array_equal(np.sort(self._barycentric_faces[face_ids],axis=1),np.sort(vertices,axis=1)):
            raise ValueError('Barycentric native element/vertex identity mismatch')
        states = self._barycentric_wet_state_matrix[ti,face_ids].copy()
        if not include_depths:
            return states, None
        return states, self._barycentric_element_source_depths[ti,vertices].copy()

    def _barycentric_apply_d19_currents(self, env, time_index, vertices):
        if self._wetdry_config is None:
            return env
        state, source_depths = self._barycentric_vertex_states(time_index, vertices, self._last_barycentric_face_ids)
        self._last_barycentric_vertex_source = {
            'source_time_index': int(time_index),
            'source_vertex_depths_m': source_depths.copy(),
            'face_state': state.copy(),
            'status': 'SHYFEM_ELEMENT_HYSTERESIS_RECONSTRUCTED_FROM_HOURLY_OUTPUT_NOT_NATIVE_FLAGS'}
        dry, wet, unknown = state == 0, state == 1, state < 0
        # Explicit opt-in D19 transport forcing ONLY. UNKNOWN remains a
        # physical UNKNOWN (-1); its operational zero is NOT a native dry state
        # or an observed SHYFEM zero. Do not replace genuinely missing WET
        # observations, alter noncurrent fields, or install global fallbacks.
        effective_zero = dry | unknown
        for v in ('x_sea_water_velocity', 'y_sea_water_velocity'):
            if v not in env:
                continue
            original = np.ma.masked_invalid(np.ma.asarray(env[v]))
            vals = np.asarray(np.ma.getdata(original), dtype=float).copy()
            masked = np.ma.getmaskarray(original).copy()
            # A dry value is EFFECTIVE configured forcing, never observed.
            # Do not require a genuine source current from a dry vertex.
            vals[effective_zero] = 0.
            masked = np.where(effective_zero, False, masked)
            env[v] = np.ma.array(vals, mask=masked, copy=False)
        return env

    @staticmethod
    def _zero_at_masked_indices(arr, dry):
        out = np.ma.array(arr, copy=True)
        mask = np.ma.getmaskarray(out).copy()
        vals = np.array(np.ma.getdata(out), copy=True)
        vals[dry] = 0.0
        mask[dry] = False
        return np.ma.array(vals, mask=mask, copy=False)

    def _native_local_wet_depth(self, time_index, nodes):
        """Depth h+eta at selected source nodes; never derive from padded 3-D data.

        Strictly require explicit native CF bathymetry and water-level metadata,
        time/node indexing and the SHYFEM shared-datum convention exercised by
        the 2018 witness. Do not infer an unknown model's vertical datum.
        """
        keys = ('sea_floor_depth_below_sea_level',
                'water_surface_height_above_reference_datum')
        names = [self.variable_mapping.get(keys[0]),
                 self.variable_mapping.get(keys[1]) or
                 self.variable_mapping.get('sea_surface_height')]
        if None in names:
            raise ValueError('SHYFEM local wet-depth contract requires source bathymetry and water level; 3D validity unqualified')
        vals = []
        for name in names:
            var = self.dataset[name]
            dims = var.dimensions
            if dims not in (('node',), ('time', 'node')):
                raise ValueError('Unqualified SHYFEM local wet-depth source dimensions: %s %s' % (name, dims))
            data = var[nodes] if dims == ('node',) else var[time_index, nodes]
            vals.append(np.ma.filled(np.ma.asarray(data).astype(float), np.nan))
        h, eta = vals
        return np.asarray(h, dtype=float) + np.asarray(eta, dtype=float), h, eta

    def _initialize_native_barycentric_mesh(self):
        """Load AUTHENTICATED, explicitly supplied native 1-based connectivity.

        No mesh-related work is performed in legacy nearest mode.
        """
        from matplotlib.tri import Triangulation
        from netCDF4 import Dataset
        with Dataset(self._barycentric_mesh_filename, 'r') as mesh:
            conn = mesh.variables.get('Mesh2_face_nodes')
            if conn is None or int(getattr(conn, 'start_index', -1)) != 1:
                raise ValueError('Native mesh requires 1-based UGRID Mesh2_face_nodes')
            faces = np.asarray(conn[:], dtype=np.int64) - 1
            if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
                raise ValueError('Native mesh must contain 3-node triangular faces')
            if (np.any(faces < 0) or np.any(faces >= len(self.x)) or
                np.any(faces[:,0] == faces[:,1]) or
                np.any(faces[:,0] == faces[:,2]) or
                np.any(faces[:,1] == faces[:,2])):
                raise ValueError('Mesh face index out of range or invalid')
            for name, ref in (('longitude', self.x), ('latitude', self.y),
                              ('total_depth', self.dataset['total_depth'][:])):
                a = np.asarray(mesh.variables[name][:])
                b = np.asarray(ref)
                if a.shape != b.shape or not np.array_equal(a, b):
                    raise ValueError('Native mesh %s values disagree with hydrodynamics' % name)
            if 'source_hydro_node_index' in mesh.variables and 'source_node_index' in self.dataset.variables:
                if not np.array_equal(mesh.variables['source_hydro_node_index'][:],
                                      self.dataset['source_node_index'][:]):
                    raise ValueError('Original hydrodynamic node-index provenance disagreement')
            self._barycentric_source_faces = (np.asarray(mesh.variables['source_wave_face_index'][:]).copy()
                  if 'source_wave_face_index' in mesh.variables else None)
        self._barycentric_faces = faces.astype(np.int32, copy=False)
        self._barycentric_face_index_by_vertices = {tuple(sorted(map(int, row))):i for i,row in enumerate(self._barycentric_faces)}
        self._barycentric_triangulation = Triangulation(np.asarray(self.x), np.asarray(self.y), self._barycentric_faces)
        # Matplotlib's compiled trapezoid-map finder respects the ORIGINAL
        # triangulation, unlike nearest-centroid or synthetic Delaunay search.
        self._barycentric_trifinder = self._barycentric_triangulation.get_trifinder()

    def _barycentric_geometry(self, x, y):
        """Vectorized point-in-native-triangle with one last-batch cache.

        Return (face_ids, vertex_indices, lambda[point,vertex]).  Cache compares
        actual coordinates, so in-place particle motion invalidates it.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if x.shape != y.shape or x.ndim != 1 or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError('Barycentric point coordinates must be finite matching 1D arrays')
        cache = self._barycentric_geometry_cache
        if (cache is not None and np.array_equal(cache[0], x) and
                np.array_equal(cache[1], y)):
            self._barycentric_cache_hits += 1
            return cache[2], cache[3], cache[4]
        self._barycentric_cache_misses += 1
        face = np.asarray(self._barycentric_trifinder(x, y), dtype=np.int32)
        # Uncovered ROI edge and domain-external requests fail CLOSED.
        if np.any(face < 0):
            raise ValueError('Barycentric sampling outside authenticated native ROI triangulation')
        verts = self._barycentric_faces[face]
        xx = np.asarray(self.x, dtype=float)[verts]
        yy = np.asarray(self.y, dtype=float)[verts]
        den = (yy[:,1]-yy[:,2])*(xx[:,0]-xx[:,2]) + (xx[:,2]-xx[:,1])*(yy[:,0]-yy[:,2])
        if np.any(~np.isfinite(den) | (np.abs(den) < 1.e-17)):
            raise ValueError('Degenerate triangle in native connectivity')
        a = ((yy[:,1]-yy[:,2])*(x-xx[:,2]) + (xx[:,2]-xx[:,1])*(y-yy[:,2]))/den
        b = ((yy[:,2]-yy[:,0])*(x-xx[:,2]) + (xx[:,0]-xx[:,2])*(y-yy[:,2]))/den
        w = np.stack((a,b,1-a-b), axis=1)
        if np.any(~np.isfinite(w) | (w < -1.e-9) | (w > 1+1.e-9)):
            raise ValueError('Out-of-triangle weights / invalid mesh topology')
        # Exclude minuscule numerical outside excursions; weights sum to 1.
        w = np.maximum(w, 0.0)
        w /= w.sum(axis=1, keepdims=True)
        self._barycentric_geometry_cache = (x.copy(),y.copy(),face,verts,w)
        return face,verts,w

    @staticmethod
    def _barycentric_read_sparse(dvar, dims, nodes, levels, time_index):
        """Fetch only required original nodal cell values (no NxN fancy indexing).

        Sparse node groups are read per native level.  NetCDF's orthogonal
        indexing is never asked to pair two 1D index arrays.
        """
        node_dim = 'node'
        if node_dim not in dims:
            raise ValueError('Barycentric native fields require source node dimension')
        node_axis = dims.index(node_dim)
        level_dim = next((d for d in dims if d in ('level','levels')), None)
        if set(dims) - {'time',node_dim,'level','levels'}:
            raise ValueError('Unsupported native barycentric field axes '+str(dims))
        if len(set(dims)) != len(dims):
            raise ValueError('Repeated native dimensions not qualified')
        out = np.ma.masked_all(nodes.shape, dtype=float)
        if level_dim is not None:
            groups = np.unique(levels)
        else:
            groups = [None]
        for level in groups:
            select = np.ones(len(nodes), dtype=bool) if level is None else (levels == level)
            unique, inv = np.unique(nodes[select], return_inverse=True)
            ix = [slice(None)]*len(dims)
            # Many disjoint NetCDF reads are much slower than one contiguous
            # native row slab.  For sufficiently large nodal batches, read the
            # 1-D bounding node span ONCE and gather locally, preserving masks.
            use_contiguous = len(unique) >= 24
            node_start = int(unique.min()) if use_contiguous else 0
            ix[node_axis] = (slice(node_start, int(unique.max())+1)
                             if use_contiguous else unique)
            if 'time' in dims:
                ix[dims.index('time')] = int(time_index)
            if level_dim is not None:
                ix[dims.index(level_dim)] = int(level)
            raw = np.ma.asarray(dvar[tuple(ix)])
            if raw.ndim != 1 or (not use_contiguous and raw.size != unique.size):
                raise ValueError('Unexpected native orthogonal NetCDF slice shape '+str(raw.shape))
            out[select] = raw[unique[inv]-node_start] if use_contiguous else raw[inv]
        return out

    def _get_variables_native_barycentric(self, requested_variables, time, x, y, z):
        """Strict, user-opted-in native triangular interpolation.

        Each physical native layer is selected at EACH VERTEX at THIS source
        time.  Only original stored wet nodal values contribute.  No vertical
        value interpolation, invented masks, or masked-vertex renormalization.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        z = np.zeros_like(x) if z is None else np.atleast_1d(np.asarray(z, dtype=float))
        if z.size == 1 and x.size > 1: z = np.full(x.size, z.item())
        if not (x.shape == y.shape == z.shape):
            raise ValueError('Barycentric native sample x/y/z array shapes differ')
        names = [requested_variables] if isinstance(requested_variables,str) else list(requested_variables)
        # Follow original admission without changing requested particle depths.
        two_d_only = self._all_requested_variables_are_z_independent(names)
        static = all(not self._variable_has_time_dimension(v) for v in names)
        names, t, x, y, checked_z, outside = self.check_arguments(
            names, self.times[0] if static else time,
            x, y, np.zeros_like(z) if two_d_only else z)
        if x.size == 0:
            return {name:np.ma.array([]) for name in names}
        if static:
            it=0
        else:
            _,_,_,it,_,_=self.nearest_time(t)
        face,verts,weights=self._barycentric_geometry(x,y)
        n=len(x)
        flat_nodes=verts.reshape(-1)
        H_vertex=None
        eta_vertex=None
        valid_vertex=None
        results={}
        for v in names:
            name=self.variable_mapping.get(v)
            if name is None:
                raise KeyError('Unknown native SHYFEM variable '+str(v))
            dvar=self.dataset[name]
            dims=dvar.dimensions
            is_3d=any(d in self.vertical_dimension_names for d in dims)
            if is_3d:
                if not self.has_vertical_levels:
                    raise ValueError('Native 3D field without authenticated native levels')
                if H_vertex is None:
                    H_vertex,h_vertex,eta_vertex=self._native_local_wet_depth(it,flat_nodes)
                    H_vertex=np.asarray(H_vertex).reshape((n,3))
                    eta_vertex=np.asarray(eta_vertex).reshape((n,3))
                    z_vertex=np.broadcast_to(z[:,None],(n,3))
                    valid_vertex=(np.isfinite(H_vertex) & (H_vertex > 0) &
                        np.isfinite(eta_vertex) & np.isfinite(z_vertex) & (z_vertex<=0))
                    # D18: deliberate below-bed water-side requests are
                    # limited to local terminal wet layer for EACH vertex.
                    sample_z=np.maximum(z_vertex,-H_vertex)
                    nominal_depth=-sample_z-eta_vertex
                    valid_vertex &= (np.isfinite(nominal_depth) & (nominal_depth <= self.native_level_bottoms_m[-1]))
                    native_level=np.searchsorted(self.native_level_bottoms_m,
                        np.maximum(nominal_depth,0),side='left')
                    native_level=np.clip(native_level,0,len(self.native_level_bottoms_m)-1)
                    native_level=native_level.reshape(-1)
                flat_level=native_level
            else:
                flat_level=None
            raw=self._barycentric_read_sparse(dvar,dims,flat_nodes,flat_level,it).reshape((n,3))
            raw=np.ma.masked_invalid(np.ma.asarray(raw))
            if is_3d:
                raw=np.ma.masked_where(~valid_vertex,raw)
            # All 3 source vertices must be valid; never renormalize 2 out of 3.
            raw_mask=np.ma.getmaskarray(raw)
            bad=np.any(raw_mask | (~np.isfinite(np.ma.getdata(raw))),axis=1)
            vals=np.einsum('ij,ij->i',weights,np.ma.getdata(raw))
            results[v]=np.ma.array(vals,mask=bad,copy=False)
        self._last_barycentric_face_ids=face.copy()
        return self._barycentric_apply_d19_currents(results,it,verts)

    def get_variables(self,
                      requested_variables,
                      time=None,
                      x=None,
                      y=None,
                      z=None):
        if self._horizontal_interpolation == 'barycentric':
            return self._get_variables_native_barycentric(requested_variables, time, x, y, z)
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

                # D17: native SHYFEM levels label the BOTTOM of a cell, not
                # sampling midpoints. OpenDrift z is instantaneous-surface
                # relative, while nominal SHYFEM levels are reference-datum
                # depths. Convert using time-specific eta before cell lookup.
                H_native, _h_native, eta_native = self._native_local_wet_depth(
                    indx_nearest, nodes)
                # D18: do not mutate requested particle z. A request at or
                # below the instantaneous bed samples the last physically wet
                # layer on the WATER side; each time frame resolves its own H.
                # This only changes where a native water-column field is read.
                requested_z = np.asarray(z, dtype=float)
                sample_z = np.maximum(requested_z, -H_native)
                nominal_depth = -sample_z - eta_native
                level_ind = np.searchsorted(
                    self.native_level_bottoms_m,
                    np.maximum(nominal_depth, 0.0), side='left')
                # Never turn below-domain requests into a real array index.
                level_ind = np.clip(level_ind, 0,
                                    len(self.native_level_bottoms_m) - 1)
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

                # A global native level is not evidence of local wet-domain
                # coverage: inactive deep layers may be zero-padded. Apply
                # the source-aware local h+eta contract for ALL 3D requests,
                # independently of optional A5 dry-current zeroing.
                H_local, _h, _eta = self._native_local_wet_depth(indx_nearest, nodes)
                # D18: below-bed REQUESTS are valid when an active water-side
                # layer exists; missing/dry NATIVE cells stay invalid.
                invalid_local = (~np.isfinite(H_local) | (H_local <= 0) |
                                 ~np.isfinite(z) | (z > 0))
                variables[var] = np.ma.masked_where(
                    invalid_local, np.ma.asarray(variables[var]))

                # Native direct get_variables() previously returned finite
                # nearest layers outside the documented physical level range.
                # Do not silently clamp. Preserve per-point masks, including
                # mixed in-domain / out-of-domain batches.
                invalid_depth = (~np.isfinite(z) | (z > 0) |
                                 ~np.isfinite(nominal_depth) |
                                 (nominal_depth > self.native_level_bottoms_m[-1]))
                variables[var] = np.ma.masked_where(
                    invalid_depth, np.ma.asarray(variables[var]))
                if (self._wetdry_config is not None and
                        var in ('x_sea_water_velocity', 'y_sea_water_velocity')):
                    # The optional categorical dry-current override is
                    # strictly restricted to native horizontal currents.
                    # Noncurrent fields retain the exact legacy native sample
                    # and its geometry/missing masks, without categorical
                    # wet/dry zeroing OR status-driven masking.
                    status = self._optional_wet_state(indx_nearest, nodes)
                    H = np.asarray([
                        self._wetdry_depths[int(n)][indx_nearest]
                        for n in nodes], dtype=float)
                    invalid_column = (~np.isfinite(H)) | (H <= 0)
                    out = np.ma.masked_where(
                        (status != 1) | invalid_column,
                        np.ma.asarray(variables[var]))
                    if self._wetdry_config['dry_current'] == 'zero':
                        out = self._zero_at_masked_indices(out, status == 0)
                    variables[var] = np.ma.masked_where(invalid_local, out)

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

    def _published_wetdry_state(self, time, lon, lat):
        """Categorical native element state at the preceding *source* frame.

        Hysteresis stays entirely within this reader. The state is never
        interpolated numerically between WET/DRY/UNKNOWN categories.
        """
        if 'wetdry_state' not in self.variables:
            raise RuntimeError('Native element wetdry_state is not configured')
        if time is None:
            raise ValueError('A source time is required for wetdry_state')
        _near, _before, _after, near_i, before_i, _after_i = self.nearest_time(time)
        frame = int(before_i if before_i is not None else near_i)
        face, vertices, _weights = self._barycentric_geometry(
            np.atleast_1d(np.asarray(lon, dtype=float)),
            np.atleast_1d(np.asarray(lat, dtype=float)))
        states, _depths = self._barycentric_vertex_states(
            frame, vertices, face, include_depths=False)
        states = np.asarray(states, dtype=np.int8)
        if states.shape != (len(np.atleast_1d(lon)),) or not np.all(
                np.isin(states, (-1, 0, 1))):
            raise RuntimeError('Unqualified native wetdry_state output')
        self.last_wetdry_state_source = {
            'source_frame_index': frame,
            'source_time': self.times[frame],
            'requested_time': time,
            'method': 'element_depth_hysteresis',
            'state_not_interpolated': True,
            'unknown_count': int(np.count_nonzero(states == -1))}
        return states.astype(float)

    def get_variables_interpolated(self, variables, profiles=None,
                                   profiles_depth=None, time=None, lon=None,
                                   lat=None, z=None, rotate_to_proj=None):
        names = [variables] if isinstance(variables, str) else list(variables)
        if 'wetdry_state' in names:
            if 'wetdry_state' not in self.variables:
                raise ValueError('wetdry_state not available without native opt-in')
            if profiles is not None and 'wetdry_state' in profiles:
                raise ValueError('Categorical wetdry_state is not a vertical profile')
            other = [v for v in names if v != 'wetdry_state']
            if other:
                env, prof = self.get_variables_interpolated(
                    other, profiles=profiles, profiles_depth=profiles_depth,
                    time=time, lon=lon, lat=lat, z=z,
                    rotate_to_proj=rotate_to_proj)
            else:
                if profiles:
                    raise ValueError('Profile variables require ordinary reader request')
                env, prof = {}, None
            env['wetdry_state'] = self._published_wetdry_state(time, lon, lat)
            return env, prof
        # Depth-independent variables must remain readable at any particle z,
        # including when other readers cannot cover the particle vertically.
        if self._all_requested_variables_are_z_independent(variables) and (
                profiles is None or self._all_requested_variables_are_z_independent(profiles)):
            z = np.zeros_like(np.atleast_1d(lon), dtype=float)
        return super().get_variables_interpolated(
            variables, profiles=profiles, profiles_depth=profiles_depth,
            time=time, lon=lon, lat=lat, z=z,
            rotate_to_proj=rotate_to_proj)

    def get_variables_interpolated_xy(self, variables, profiles=None,
                                      profiles_depth=None, time=None, x=None,
                                      y=None, z=None, rotate_to_proj=None):
        """Keep valid 2-D fields available in a mixed 2-D/3-D request.

        The common reader wrapper performs global z coverage masking before
        the SHYFEM native extractor sees the request. For a mixed request
        only, safely sample invalid-z positions at the surface, then mask
        every 3-D variable at those original invalid depths. No extrapolated
        3-D values are returned to the caller and 2-D fields are untouched.
        This is limited to the isolated SHYFEM reader, not a global patch.
        """
        names = list(variables) if not isinstance(variables, str) else [variables]
        vertical = [v for v in names if self._variable_has_vertical_dimension(v)]
        horizontal = [v for v in names if not self._variable_has_vertical_dimension(v)]
        original_z = np.atleast_1d(z) if z is not None else np.zeros(1)
        x_array = np.atleast_1d(x)
        if original_z.size == 1 and x_array.size > 1:
            original_z = np.repeat(original_z, x_array.size)
        if not self.has_vertical_levels or not vertical or not horizontal:
            return super().get_variables_interpolated_xy(
                variables, profiles=profiles, profiles_depth=profiles_depth,
                time=time, x=x, y=y, z=z, rotate_to_proj=rotate_to_proj)
        out_of_depth = (~np.isfinite(original_z) |
                        (original_z < self.zmin) | (original_z > self.zmax))
        if not np.any(out_of_depth):
            return super().get_variables_interpolated_xy(
                variables, profiles=profiles, profiles_depth=profiles_depth,
                time=time, x=x, y=y, z=z, rotate_to_proj=rotate_to_proj)
        checked_z = np.where(out_of_depth, self.zmax, original_z)
        environment, profile_data = super().get_variables_interpolated_xy(
            variables, profiles=profiles, profiles_depth=profiles_depth,
            time=time, x=x, y=y, z=checked_z, rotate_to_proj=rotate_to_proj)
        for variable in vertical:
            if variable in environment:
                original_values = np.ma.asarray(environment[variable])
                environment[variable] = np.ma.masked_where(out_of_depth,
                                                           original_values)
        return environment, profile_data

    def _native_wetdry_exact_time_snapshot(self, variables, profiles, profiles_depth,
                            time, reader_x, reader_y, z):
        env, profile = self._get_variables_interpolated_native_snapshot(
            variables, profiles, profiles_depth, time, reader_x, reader_y, z)
        self._last_dry_current_provenance = {}
        if self._wetdry_config is None:
            return env, profile
        _, _, _, ti, _, _ = self.nearest_time(time)
        if self._horizontal_interpolation == 'barycentric':
            face, verts, _ = self._barycentric_geometry(reader_x, reader_y)
            state, _depths = self._barycentric_vertex_states(ti, verts, face)
        else:
            nodes = self._nearest_node_(np.atleast_1d(reader_x), np.atleast_1d(reader_y))
            state = self._optional_wet_state(ti, nodes)
        for v in ('x_sea_water_velocity','y_sea_water_velocity'):
            if v not in env:
                continue
            values = np.ma.masked_invalid(np.ma.asarray(env[v]))
            valid = (~np.ma.getmaskarray(values)) & np.isfinite(np.ma.getdata(values))
            if values.shape != state.shape:
                raise ValueError('Native exact-time state/current shape mismatch')
            is_bary = self._horizontal_interpolation == 'barycentric'
            # Nearest mode is unchanged. In opt-in barycentric mode only, the
            # UNKNOWN current was already replaced by operational zero in the
            # source snapshot. Keep categorical state -1 in provenance.
            env[v] = (values if is_bary else
                      np.ma.masked_where((state < 0), values))
            self._last_dry_current_provenance[v] = {
                'source_state':state.copy(),
                'effective_override':((state == 0) & valid).copy(),
                'unknown_effective_override':((state < 0) & valid & is_bary).copy(),
                'native_wet_valid':((state == 1) & valid).copy(),
                'selected_source_time_index':int(ti),
                'selected_provenance':np.where(~valid | ((state < 0) & ~is_bary),'INVALID',
                    np.where(state < 0,'SHYFEM_UNKNOWN_CURRENT_ZERO_OVERRIDE',
                        np.where(state == 0,'SHYFEM_DRY_CURRENT_ZERO_OVERRIDE',
                                 'SOURCE_OBSERVED_VALID'))).copy(),
                'mode':self._wetdry_config['dry_zero_time_mode']}
        return env, profile

    def _get_variables_interpolated_(self, variables, profiles,
                                     profiles_depth, time, reader_x, reader_y, z):
        """D17: independent native spatial/depth samples, then one time blend.

        Unlike regular gridded readers the native SHYFEM z coordinates mark
        cell bottoms. Never interpolate vertically between values. Each
        record's eta, wet geometry and masks must be resolved separately.
        """
        names = [variables] if isinstance(variables, str) else list(variables)
        profile_names = [] if profiles is None else ([profiles] if isinstance(profiles, str) else list(profiles))
        has_dynamic = any(self._variable_has_time_dimension(v)
                          for v in names + profile_names)
        if not has_dynamic or not self.has_time:
            return self._native_wetdry_exact_time_snapshot(
                variables, profiles, profiles_depth, time, reader_x, reader_y, z)
        nearest, before, after, nearest_i, before_i, after_i = self.nearest_time(time)
        if before is None or after is None or before == after or time == before:
            return self._native_wetdry_exact_time_snapshot(
                variables, profiles, profiles_depth, time, reader_x, reader_y, z)
        if time == after:
            return self._native_wetdry_exact_time_snapshot(
                variables, profiles, profiles_depth, time, reader_x, reader_y, z)
        alpha = (time - before).total_seconds() / (after - before).total_seconds()
        if not 0.0 < alpha < 1.0:
            raise ValueError('SHYFEM interpolation requires a valid bracketing time')
        e0,p0 = self._get_variables_interpolated_native_snapshot(
            variables, profiles, profiles_depth, before, reader_x, reader_y, z)
        e1,p1 = self._get_variables_interpolated_native_snapshot(
            variables, profiles, profiles_depth, after, reader_x, reader_y, z)
        result = {}
        native_wetdry_bracket_states = None
        current_fields = {'x_sea_water_velocity', 'y_sea_water_velocity'} & (set(names) | set(profile_names))
        if self._wetdry_config is not None and current_fields:
            if self._horizontal_interpolation == 'barycentric':
                face, verts, _ = self._barycentric_geometry(reader_x, reader_y)
                state0, _v0 = self._barycentric_vertex_states(before_i, verts, face)
                state1, _v1 = self._barycentric_vertex_states(after_i, verts, face)
            else:
                nodes = self._nearest_node_(np.atleast_1d(reader_x), np.atleast_1d(reader_y))
                state0 = self._optional_wet_state(before_i, nodes)
                state1 = self._optional_wet_state(after_i, nodes)
            native_wetdry_bracket_states = (state0, state1)
        self._last_dry_current_provenance = {}
        for v in names:
            if not self._variable_has_time_dimension(v):
                result[v] = e0[v]
                continue
            left = np.ma.masked_invalid(np.ma.asarray(e0[v]))
            right = np.ma.masked_invalid(np.ma.asarray(e1[v]))
            if v not in current_fields or native_wetdry_bracket_states is None:
                result[v] = np.ma.masked_invalid((1.0-alpha)*left + alpha*right)
                continue
            st0, st1 = native_wetdry_bracket_states
            st0, st1 = np.asarray(st0),np.asarray(st1)
            if left.shape != st0.shape or right.shape != st1.shape:
                raise ValueError('Native source-status/current shape mismatch')
            valid0 = (~np.ma.getmaskarray(left)) & np.isfinite(np.ma.getdata(left))
            valid1 = (~np.ma.getmaskarray(right)) & np.isfinite(np.ma.getdata(right))
            # Only the opted-in barycentric D19 policy maps UNKNOWN to an
            # EFFECTIVE zero-current bracket, without changing its state.
            is_bary = self._horizontal_interpolation == 'barycentric'
            eff0 = np.where(st0 < 0, 0, st0) if is_bary else st0
            eff1 = np.where(st1 < 0, 0, st1) if is_bary else st1
            pair_valid = valid0 & valid1 & (eff0 >= 0) & (eff1 >= 0)
            both_wet = (eff0 == 1) & (eff1 == 1)
            both_dry = (eff0 == 0) & (eff1 == 0)
            mixed = ((eff0 == 0) & (eff1 == 1)) | ((eff0 == 1) & (eff1 == 0))
            linear_value = (1.0-alpha)*left + alpha*right
            selected_after = nearest_i == after_i
            selected_value = right if selected_after else left
            is_linear = self._wetdry_config['dry_zero_time_mode'] == 'linear'
            numerical = np.ma.where(mixed & (not is_linear), selected_value, linear_value)
            out = np.ma.masked_where(~pair_valid | ~(both_wet | both_dry | mixed),
                                     np.ma.masked_invalid(numerical))
            result[v] = out
            selected_state = st1 if selected_after else st0
            # Piecewise WET/UNKNOWN selects the bracket's actual effective
            # current. Explicit linear mode blends effective boundary zero
            # with valid wet current exactly once, preserving original states.
            if is_bary:
                unknown_used = np.where(mixed & (not is_linear),
                    selected_state < 0, (st0 < 0) | (st1 < 0))
            else:
                unknown_used = np.zeros(st0.shape, dtype=bool)
            origin = np.where(~pair_valid, 'INVALID', np.where(
                mixed & (not is_linear), np.where(selected_state == 0,
                    'SHYFEM_DRY_CURRENT_ZERO_OVERRIDE', 'SOURCE_OBSERVED_VALID'),
                np.where(both_dry, 'SHYFEM_DRY_CURRENT_ZERO_OVERRIDE',
                    np.where(both_wet, 'SOURCE_OBSERVED_VALID', 'MIXED_SOURCE_AND_DRY_ZERO'))))
            origin = np.where(pair_valid & unknown_used,
                'SHYFEM_UNKNOWN_CURRENT_ZERO_OVERRIDE', origin)
            self._last_dry_current_provenance[v] = {
                'source_state_before':st0.copy(),
                'source_state_after':st1.copy(),
                'native_wet_valid_before': ((st0 == 1) & valid0).copy(),
                'native_wet_valid_after': ((st1 == 1) & valid1).copy(),
                'effective_override_before': ((st0 == 0) & valid0).copy(),
                'effective_override_after': ((st1 == 0) & valid1).copy(),
                'unknown_effective_override_before': ((st0 < 0) & valid0 & is_bary).copy(),
                'unknown_effective_override_after': ((st1 < 0) & valid1 & is_bary).copy(),
                'selected_source_time_index':int(after_i if selected_after else before_i),
                'selected_provenance':origin.copy(),
                'mode':self._wetdry_config['dry_zero_time_mode']}
        if profiles is None:
            return result,None
        if p0 is None or p1 is None:
            raise ValueError('SHYFEM temporal profile missing one source record')
        # Preserve only original native profile z coordinates present in BOTH
        # snapshots; no synthetic vertical coordinates or nearest-z mapping.
        z0=np.asarray(p0['z']); z1=np.asarray(p1['z'])
        ii0=[];ii1=[]
        for i,zz in enumerate(z0):
            matches=np.flatnonzero(z1 == zz)
            if len(matches): ii0.append(i);ii1.append(int(matches[0]))
        out={'z':z0[ii0].copy()}
        for v in profiles:
            profile0=np.ma.masked_invalid(np.ma.asarray(p0[v])[ii0,:])
            profile1=np.ma.masked_invalid(np.ma.asarray(p1[v])[ii1,:])
            linear_profile=(1-alpha)*profile0 + alpha*profile1
            if native_wetdry_bracket_states is None or v not in current_fields:
                out[v]=np.ma.masked_invalid(linear_profile)
                continue
            st0, st1=native_wetdry_bracket_states
            if profile0.shape[1] != len(st0) or profile1.shape[1] != len(st1):
                raise ValueError('Native current profile/node dimension mismatch')
            is_bary = self._horizontal_interpolation == 'barycentric'
            eff0=np.where(st0<0,0,st0) if is_bary else st0
            eff1=np.where(st1<0,0,st1) if is_bary else st1
            valid=(~np.ma.getmaskarray(profile0) & ~np.ma.getmaskarray(profile1) &
                   np.isfinite(np.ma.getdata(profile0)) & np.isfinite(np.ma.getdata(profile1)) &
                   (eff0[None,:]>=0) & (eff1[None,:]>=0))
            mixed=(((eff0==0)&(eff1==1))|((eff0==1)&(eff1==0)))[None,:]
            if self._wetdry_config['dry_zero_time_mode']=='piecewise':
                selected_profile=profile1 if nearest_i==after_i else profile0
                value=np.ma.where(mixed,selected_profile,linear_profile)
            else:
                value=linear_profile
            out[v]=np.ma.masked_invalid(np.ma.masked_where(~valid,value))
            if is_bary:
                self._last_dry_current_profile_provenance[v]={
                    'source_element_state_before':st0.copy(),
                    'source_element_state_after':st1.copy(),
                    'native_layer_bottom_z':out['z'].copy(),
                    'unknown_zero_before':(st0<0).copy(),
                    'unknown_zero_after':(st1<0).copy(),
                    'mode':self._wetdry_config['dry_zero_time_mode'],
                    'provenance':'SHYFEM_D19_EFFECTIVE_ZERO_NOT_NATIVE_MEASUREMENT'}
        return result,out

    def _barycentric_native_snapshot_profiles(self, variables, profiles,
                                            profiles_depth, time, reader_x, reader_y, z):
        """Preserve native z profile coordinates with strict 3-vertex wet masks.

        This is a separate native-layer profile pathway; it never fabricates
        interpolated vertical coordinates or values between source layers.
        """
        env = self.get_variables(variables, time, reader_x, reader_y, z)
        if profiles is None:
            return env, None
        if (not self.has_vertical_levels or profiles_depth is None or
                not np.isfinite(profiles_depth) or profiles_depth <= 0):
            raise ValueError('Barycentric native profile requires valid positive profiles_depth')
        x = np.atleast_1d(reader_x); y = np.atleast_1d(reader_y)
        face, nodes, weights = self._barycentric_geometry(x,y)
        _,_,_,it,_,_ = self.nearest_time(time)
        H, _h, _eta = self._native_local_wet_depth(it,nodes.ravel())
        H = np.asarray(H,dtype=float).reshape(nodes.shape)
        bottoms = np.asarray(self.native_level_bottoms_m)
        uppers = np.r_[0.,bottoms[:-1]]
        # A layer is valid in the triangular stencil ONLY if all 3 native
        # nodal columns physically contain that layer's wet portion.
        vertex_wet = ((np.isfinite(H) & (H>0))[:,:,None] &
                      (H[:,:,None]>uppers[None,None,:]) &
                      (uppers[None,None,:]<abs(float(profiles_depth))))
        all_wet = np.all(vertex_wet,axis=1)  # particle, level
        chosen = np.flatnonzero(np.any(all_wet,axis=0))
        if not chosen.size:
            chosen = np.array([0],dtype=int)
        # Native original layer-bottom interfaces, NOT synthesized midpoints.
        ret={'z':-bottoms[chosen].copy()}
        if self._wetdry_config is not None:
            element_states, _vertex_depths = self._barycentric_vertex_states(it,nodes,face)
        else:
            element_states=np.ones(len(x),dtype=np.int8)
        self._last_dry_current_profile_provenance={}
        for var in profiles:
            if not self._variable_has_vertical_dimension(var):
                raise ValueError('Barycentric SHYFEM profiles require native 3D field '+str(var))
            dvar = self.dataset[self.variable_mapping[var]]
            out=[]
            for k in chosen:
                src=self._barycentric_read_sparse(dvar,dvar.dimensions,nodes.ravel(),
                         np.full(nodes.size,int(k),dtype=int),it).reshape(nodes.shape)
                src=np.ma.masked_invalid(np.ma.asarray(src))
                bad=np.any(np.ma.getmaskarray(src),axis=1) | ~all_wet[:,k]
                values=np.einsum('ij,ij->i',weights,np.ma.getdata(src))
                out.append(np.ma.array(values,mask=bad))
            val=np.ma.vstack(out)
            if self._wetdry_config is not None and var in ('x_sea_water_velocity','y_sea_water_velocity'):
                # Effective D19 current zeros are allowed ONLY for genuine
                # existing physically wet layer portions. Below-bed/padded
                # source levels remain masked, even for DRY or UNKNOWN states.
                override=(element_states!=1)[None,:] & all_wet[:,chosen].T
                val=self._zero_at_masked_indices(val,np.broadcast_to(override,val.shape))
                self._last_dry_current_profile_provenance[var]={
                    'source_time_index':int(it),
                    'native_layer_bottom_z':ret['z'].copy(),
                    'source_element_state':element_states.copy(),
                    'dry_zero_mask':((element_states==0)[None,:] & all_wet[:,chosen].T).copy(),
                    'unknown_zero_mask':((element_states<0)[None,:] & all_wet[:,chosen].T).copy(),
                    'provenance':'SHYFEM_D19_EFFECTIVE_ZERO_NOT_NATIVE_MEASUREMENT'}
            ret[var]=val
        return env,ret

    def _get_variables_interpolated_native_snapshot(self, variables, profiles,
                                     profiles_depth, time, reader_x, reader_y, z):
        if self._horizontal_interpolation == 'barycentric':
            return self._barycentric_native_snapshot_profiles(
                variables, profiles, profiles_depth, time, reader_x, reader_y, z)
        env = self.get_variables(variables, time, reader_x, reader_y, z)
        if profiles is None:
            return env, None
        # I5: native cell means are piecewise constant over each *wet part*
        # of their source layer. A nominal midpoint below the bed must not
        # erase an active terminal cell. Keep EXACT original native z values;
        # do not request midpoint point-samples (which correctly mask below-bed).
        if (not self.has_vertical_levels or profiles_depth is None or
                not np.isfinite(profiles_depth) or profiles_depth <= 0):
            raise ValueError('Native SHYFEM profile requires valid layer metadata and positive profiles_depth')
        x = np.atleast_1d(reader_x)
        y = np.atleast_1d(reader_y)
        nodes = self._nearest_node_(x, y)
        nearest_time, before, after, ti, ib, ia = self.nearest_time(time)
        bottoms = np.asarray(self.native_level_bottoms_m, dtype=float)
        uppers = np.r_[0.0, bottoms[:-1]]
        H, _h, _eta = self._native_local_wet_depth(ti, nodes)
        H = np.asarray(H, dtype=float)
        # Strict overlap: only cells with positive physical wet extent, with
        # actual source masks checked below. The virtual profile request-depth
        # limit selects existing indices but never creates or moves coordinates.
        wet = ((np.isfinite(H) & (H > 0))[:, None] &
               (H[:, None] > uppers[None, :]) &
               (uppers[None, :] < abs(float(profiles_depth))))
        selected_indices = np.flatnonzero(np.any(wet, axis=0))
        if not selected_indices.size:
            # Preserve a native level and all-masked data for dry columns so
            # existing environment fallbacks can operate; never make up a z.
            selected_indices = np.array([0], dtype=int)
        result = {'z': np.asarray(self.z)[selected_indices].copy()}
        for var in profiles:
            if var not in self.variables or not self._variable_has_vertical_dimension(var):
                raise ValueError('Native 3D SHYFEM profile variable unavailable: '+str(var))
            v = self.dataset[self.variable_mapping[var]]
            dims = v.dimensions
            if 'node' not in dims or not any(d in self.vertical_dimension_names for d in dims):
                raise ValueError('Unsupported native SHYFEM profile dimensions '+str(dims))
            def one_layer(k):
                ii=[slice(None)]*len(dims)
                if 'time' in dims:ii[dims.index('time')]=ti
                ii[dims.index('node')]=nodes
                for candidate in self.vertical_dimension_names:
                    if candidate in dims:ii[dims.index(candidate)]=int(k);break
                vals=np.ma.asarray(v[tuple(ii)])
                if vals.ndim != 1 or vals.size != nodes.size:
                    raise ValueError('Unexpected native layer data shape '+str(vals.shape))
                return vals
            values=np.ma.vstack([one_layer(k) for k in selected_indices])
            inactive=~wet[:, selected_indices].T
            values=np.ma.masked_where(inactive, values)
            if (self._wetdry_config is not None and
                    var in ('x_sea_water_velocity','y_sea_water_velocity')):
                state=self._optional_wet_state(ti,nodes)
                valid_state=np.asarray(state)==1
                values=np.ma.masked_where(~valid_state[None,:],values)
                if (var in ('x_sea_water_velocity','y_sea_water_velocity') and
                        self._wetdry_config['dry_current']=='zero'):
                    # Preserve A5 dry-zero only for geometrically wet cells.
                    dry=(np.asarray(state)==0)[None,:] & ~inactive
                    values=self._zero_at_masked_indices(values,dry)
            result[var]=values
        return env, result

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
