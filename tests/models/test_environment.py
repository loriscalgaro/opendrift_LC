from datetime import datetime, timedelta
import pytest
import numpy as np
from opendrift.models.basemodel.environment import Environment
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_oscillating
from opendrift.config import Configurable


def test_add_readers(test_data_roms):
    c = Configurable()
    required_variables = {
        'x_sea_water_velocity': {
            'fallback': 0
        },
        'y_sea_water_velocity': {
            'fallback': 10
        },
    }
    env = Environment(required_variables, c._config)
    env.add_reader(test_data_roms)
    env.finalize()

    e, p, k = env.get_environment(
        ['x_sea_water_velocity', 'y_sea_water_velocity'],
        test_data_roms.start_time, [10], [50], [0], [])
    assert e[0][0] == 0
    assert e[0][1] == 10

def test_previous():
    o = OceanDrift()  # With coastline_action = none
    o.set_config('general:coastline_action', 'none')
    o.set_config('drift:vertical_advection', False)
    o.set_config('environment:constant:land_binary_mask', 0)
    o.set_config('environment:constant:x_sea_water_velocity', 1)
    o.seed_elements(lon=3, lat=60, time=datetime.now())
    o.run(steps=1)
    assert o.elements.lon == pytest.approx(3.0645, .001)
    assert o.elements_previous is None
    assert o.environment_previous is None

    o = OceanDrift()  # With coastline_action = previous
    o.set_config('general:coastline_action', 'previous')
    o.set_config('drift:vertical_advection', False)
    o.set_config('environment:constant:land_binary_mask', 0)
    o.set_config('environment:constant:x_sea_water_velocity', 1)
    o.seed_elements(lon=3, lat=60, time=datetime.now())
    o.run(steps=1)
    assert o.elements.lon == pytest.approx(3.0645, .001)
    assert o.elements_previous.lon == pytest.approx(3.0, .001)
    assert o.elements_previous is not None
    assert o.environment_previous is None
 
    o = OceanDrift()  # For newly seeded, previous shall equal present
    o.seed_elements(lon=3, lat=60, time=[datetime.now(), datetime.now()+timedelta(hours=1)], number=10)
    o.run(steps=1, time_step=timedelta(minutes=30))
    assert len(o.environment.sea_surface_height) == 5
    assert len(o.environment_previous.sea_surface_height) == 5
    assert len(o._environment_previous.sea_surface_height) == 10
    assert o.environment_previous.sea_surface_height[0] == 0
    assert np.isnan(o._environment_previous.sea_surface_height[-1])

    o = OceanDrift()  # Check gradient
    time = datetime.now()
    reader_tidal = reader_oscillating.Reader('sea_surface_height', amplitude=1,
                                             period=timedelta(hours=6), phase=0, zero_time=time)
    o.add_reader(reader_tidal)
    o.seed_elements(lon=3, lat=60, time=[time, time+timedelta(hours=1)], number=10)
    o.run(steps=2, time_step=timedelta(minutes=30))
    assert len(o.environment.sea_surface_height) == 9
    assert len(o.environment_previous.sea_surface_height) == 9
    assert len(o._environment_previous.sea_surface_height) == 10
    assert o.environment.sea_surface_height[0] == pytest.approx(0.2588, 3)
    assert o.environment_previous.sea_surface_height[0] == 0
    assert np.isnan(o._environment_previous.sea_surface_height[-1])
    assert not np.isnan(o._environment_previous.sea_surface_height[-2])


def test_skip_env_variable():
    o = OceanDrift()
    o.set_config('drift:vertical_mixing', True)  # Diffusivity shall be included
    o.set_config('environment:constant:land_binary_mask', 0)
    o.seed_elements(lon=3, lat=60, time=datetime.now())
    o.run(steps=1)
    assert 'ocean_vertical_diffusivity' in o.required_variables

    o = OceanDrift()
    o.set_config('drift:vertical_mixing', False)  # Diffusivity shall be skipped
    o.set_config('environment:constant:land_binary_mask', 0)
    o.seed_elements(lon=3, lat=60, time=datetime.now())
    o.run(steps=1)
    assert 'ocean_vertical_diffusivity' not in o.required_variables


class _ProvenanceDummyReader:
    """Minimal Environment reader for provenance-only unit tests."""
    __module__ = 'tests.models.test_environment'

    def __init__(self, name, variable, values, profile=None):
        self.name = name
        self.variables = [variable]
        self.variable = variable
        self.values = np.asarray(values, dtype=float)
        self.profile = profile
        self._element_ID = None
        self.number_of_fails = 0
        self.is_lazy = False
        self.start_time = None
        self.end_time = None
        self.always_valid = True
        self.checked_for_overlap = True

    def covers_time(self, time):
        return True

    def get_variables_interpolated(
            self, variable_group, profiles=None, profiles_depth=None,
            time=None, lon=None, lat=None, z=None, rotate_to_proj=None):
        n = len(lon)
        values = (self.values[:n] if self.values.size != 1
                  else np.full(n, float(self.values[0])))
        env = {self.variable: values}
        env_profiles = {}
        if profiles and self.variable in profiles and self.profile is not None:
            profile = np.asarray(self.profile, dtype=float)
            if profile.ndim == 1:
                profile = profile[:, None]
            if profile.shape[1] == 1 and n > 1:
                profile = np.repeat(profile, n, axis=1)
            env_profiles = {
                'z': np.arange(profile.shape[0], dtype=float),
                self.variable: np.ma.masked_invalid(profile[:, :n]),
            }
        return env, env_profiles


class _ProvenanceConstantReader(_ProvenanceDummyReader):
    __module__ = 'opendrift.readers.reader_constant'


def _provenance_environment(fallback, readers=(), profiles=None):
    config = Configurable()
    env = Environment({'sea_water_temperature': {'fallback': fallback}}, config._config)
    env.readers.clear()
    env.priority_list.clear()
    for reader in readers:
        env.readers[reader.name] = reader
    if readers:
        env.priority_list['sea_water_temperature'] = [r.name for r in readers]
    env.required_profiles = profiles or []
    # These unit tests exercise get_environment directly without the full model
    # finalization lifecycle.
    env.__finalized__ = True
    return env


def _get_temperature_environment(env, n=3, profiles=None):
    return env.get_environment(
        ['sea_water_temperature'], datetime(2026, 1, 1),
        np.arange(n, dtype=float), np.zeros(n), np.zeros(n),
        profiles=profiles or [])


def test_environment_provenance_distinguishes_fallback_from_equal_reader_value():
    fallback_env = _provenance_environment(10.0)
    values, _, _ = _get_temperature_environment(fallback_env)
    assert np.all(values.sea_water_temperature == 10.0)
    assert np.all(
        fallback_env.last_environment_provenance['sea_water_temperature']
        == fallback_env.PROVENANCE_FALLBACK)

    reader = _ProvenanceDummyReader(
        'reader', 'sea_water_temperature', [10.0, 10.0, 10.0])
    reader_env = _provenance_environment(10.0, [reader])
    values, _, _ = _get_temperature_environment(reader_env)
    assert np.all(values.sea_water_temperature == 10.0)
    assert np.all(
        reader_env.last_environment_provenance['sea_water_temperature']
        == reader_env.PROVENANCE_READER)


def test_environment_provenance_preserves_later_reader_fill_semantics():
    first = _ProvenanceDummyReader(
        'first', 'sea_water_temperature', [np.nan, np.nan, np.nan])
    second = _ProvenanceDummyReader(
        'second', 'sea_water_temperature', [7.0, 8.0, 9.0])
    env = _provenance_environment(10.0, [first, second])
    values, _, _ = _get_temperature_environment(env)
    np.testing.assert_allclose(values.sea_water_temperature, [7.0, 8.0, 9.0])
    assert np.all(
        env.last_environment_provenance['sea_water_temperature']
        == env.PROVENANCE_READER)


def test_environment_provenance_marks_explicit_constant_as_source():
    reader = _ProvenanceConstantReader(
        'constant_reader', 'sea_water_temperature', [10.0])
    env = _provenance_environment(10.0, [reader])
    values, _, _ = _get_temperature_environment(env)
    assert np.all(values.sea_water_temperature == 10.0)
    assert np.all(
        env.last_environment_provenance['sea_water_temperature']
        == env.PROVENANCE_CONSTANT)


def test_environment_provenance_distinguishes_hard_missing_from_fallback():
    reader = _ProvenanceDummyReader(
        'reader', 'sea_water_temperature', [np.nan, 2.0, np.nan])
    env = _provenance_environment(None, [reader])
    values, _, missing = _get_temperature_environment(env)
    np.testing.assert_array_equal(
        env.last_environment_provenance['sea_water_temperature'],
        [env.PROVENANCE_HARD_MISSING,
         env.PROVENANCE_READER,
         env.PROVENANCE_HARD_MISSING])
    np.testing.assert_array_equal(np.asarray(missing, dtype=bool), [True, False, True])
    assert np.isnan(values.sea_water_temperature[[0, 2]]).all()


def test_environment_profile_provenance_tracks_reader_and_fallback_cells():
    reader = _ProvenanceDummyReader(
        'reader', 'sea_water_temperature', [1.0, 1.0],
        profile=[[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
    env = _provenance_environment(10.0, [reader], profiles=['sea_water_temperature'])
    _, profiles, _ = _get_temperature_environment(
        env, n=2, profiles=['sea_water_temperature'])
    provenance = env.last_environment_profile_provenance['sea_water_temperature']
    assert provenance.shape == profiles['sea_water_temperature'].shape
    assert np.count_nonzero(provenance == env.PROVENANCE_READER) > 0
    assert np.count_nonzero(provenance == env.PROVENANCE_FALLBACK) > 0


def test_environment_provenance_is_current_call_uint8_and_filters_with_elements():
    reader = _ProvenanceDummyReader(
        'reader', 'sea_water_temperature', [1.0, 2.0, 3.0])
    env = _provenance_environment(10.0, [reader])
    _get_temperature_environment(env)
    provenance = env.get_last_environment_provenance('sea_water_temperature')
    assert provenance.dtype == np.uint8
    provenance[0] = 99
    assert env.last_environment_provenance['sea_water_temperature'][0] == env.PROVENANCE_READER

    env.filter_last_environment_provenance(np.array([True, False, True]))
    np.testing.assert_array_equal(
        env.last_environment_provenance['sea_water_temperature'],
        [env.PROVENANCE_READER, env.PROVENANCE_READER])
