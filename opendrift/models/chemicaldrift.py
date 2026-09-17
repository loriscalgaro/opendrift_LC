# -*- coding: utf-8 -*-
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
# Copyright 2020, Manuel Aghito, MET Norway

"""
ChemicalDrift is an OpenDrift module for drift and fate of chemicals.
The module is under development within the scope of the Horizon2020 project EMERGE
Manuel Aghito. Norwegian Meteorological Institute. 2021.
The initial version is based on Radionuclides module by Magne Simonsen
"""

import numpy as np
import logging; logger = logging.getLogger(__name__)

from opendrift.models.physics_methods import seawater_dynamic_viscosity
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray
from opendrift.models.chemicaldrift_postprocess import ChemicalDriftPostProcessMixin
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC, CONFIG_LEVEL_ADVANCED
import pyproj
from datetime import datetime, timezone


class Chemical(Lagrangian3DArray):
    """Container for Chemical element variable definitions."""

    BASE_CHEMICAL_VARIABLES = [
        ('diameter', {'dtype': np.float32, 'units': 'm', 'default': 0.}),
        ('d50', {'dtype': np.float32, 'units': 'm', 'default': 0.}),
        ('density', {'dtype': np.float32, 'units': 'kg/m^3', 'default': 2650.}),
        ('critstress_factor', {'dtype': np.float32, 'units': '', 'default': 1.0}),
        ('f_OC', {'dtype': np.float32, 'units': '', 'default': 0.01}),
        ('specie', {'dtype': np.int32, 'units': '', 'default': 0}),
        ('mass', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 1e3}),
        ('mass_degraded', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_degraded_water', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_degraded_sediment', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_volatilized', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
    ]
    SINGLE_DEGRADATION_VARIABLES = [
        ('mass_photodegraded', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_biodegraded', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_biodegraded_water', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_biodegraded_sediment', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_hydrolyzed', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_hydrolyzed_water', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
        ('mass_hydrolyzed_sediment', {'dtype': np.float32, 'units': 'ug', 'seed': True, 'default': 0}),
    ]
    SINGLE_DEGRADATION_VARIABLE_NAMES = (
    'mass_photodegraded',
    'mass_biodegraded', 'mass_biodegraded_water', 'mass_biodegraded_sediment',
    'mass_hydrolyzed', 'mass_hydrolyzed_water', 'mass_hydrolyzed_sediment',
        )

    BED_INTERACTION_VARIABLES = [
        ('tau_bx', {'dtype': np.float32, 'units': 'Pa', 'seed': True, 'default': np.nan}),
        ('tau_by', {'dtype': np.float32, 'units': 'Pa', 'seed': True, 'default': np.nan}),
        ('tau_current', {'dtype': np.float32, 'units': 'Pa', 'seed': True, 'default': np.nan}),
        ('tau_effective', {'dtype': np.float32, 'units': 'Pa', 'seed': True, 'default': np.nan}),
        ('tau_effective_x', {'dtype': np.float32, 'units': 'Pa', 'seed': True, 'default': np.nan}),
        ('tau_effective_y', {'dtype': np.float32, 'units': 'Pa', 'seed': True, 'default': np.nan}),
        ('ustar_effective', {'dtype': np.float32, 'units': 'm/s', 'seed': True, 'default': np.nan}),
        ('rho', {'dtype': np.float32, 'units': 'kg/m3', 'seed': True, 'default': np.nan}),
        ('Cd', {'dtype': np.float32, 'units': '1', 'seed': True, 'default': np.nan}),
        ('speed', {'dtype': np.float32, 'units': 'm/s', 'seed': True, 'default': np.nan}),
        ('z_ref', {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan}),
        ('z0', {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan}),
        ('tau_wave', {'dtype': np.float32, 'units': 'Pa', 'seed': True, 'default': np.nan}),
        ('wave_orbital_velocity', {'dtype': np.float32, 'units': 'm/s', 'seed': True, 'default': np.nan}),
        ('wave_excursion', {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan}),
        ('wave_number', {'dtype': np.float32, 'units': '1/m', 'seed': True, 'default': np.nan}),
        ('wave_friction_factor', {'dtype': np.float32, 'units': '1', 'seed': True, 'default': np.nan}),
        ('wave_z0', {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan}),
        ('wave_water_depth', {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan}),
        ('p_res', {'dtype': np.float32, 'units': '1', 'seed': True, 'default': np.nan}),
        ('p_dep', {'dtype': np.float32, 'units': '1', 'seed': True, 'default': np.nan}),
        ('tau_cr_res', {'dtype': np.float32, 'units': 'Pa', 'seed': True, 'default': np.nan}),
    ]
    BED_INTERACTION_VARIABLE_NAMES = (
        'tau_bx','tau_by',
        'tau_current', 'tau_effective',
        'tau_effective_x', 'tau_effective_y',
        'ustar_effective',
        'rho', 'Cd',
        'speed', 'z_ref',
        'z0','tau_wave',
        'wave_orbital_velocity', 'wave_excursion', 'wave_number',
        'wave_friction_factor', 'wave_z0', 'wave_water_depth',
        'p_res', 'p_dep',
        'tau_cr_res',
            )

    # Optional sediment-oxygen diagnostics. Only the subset required by the
    # selected oxygen model is added to the element type when requested.
    SEDIMENT_OXYGEN_DIAGNOSTIC_DEFINITIONS = {
        'sed_o2_used': {'dtype': np.float32, 'units': 'mmol/m3', 'seed': True, 'default': np.nan},
        'sed_o2_bottom': {'dtype': np.float32, 'units': 'mmol/m3', 'seed': True, 'default': np.nan},
        'sed_o2_active_layer_thickness': {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan},
        'sed_o2_porosity': {'dtype': np.float32, 'units': '1', 'seed': True, 'default': np.nan},
        'sed_o2_diffusivity': {'dtype': np.float32, 'units': 'm2/s', 'seed': True, 'default': np.nan},
        'sed_o2_dbl_thickness': {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan},
        'sed_o2_k_bl': {'dtype': np.float32, 'units': 'm/s', 'seed': True, 'default': np.nan},
        'sed_o2_surface': {'dtype': np.float32, 'units': 'mmol/m3', 'seed': True, 'default': np.nan},
        'sed_o2_penetration_depth': {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan},
        'sed_o2_consumption_rate': {'dtype': np.float32, 'units': 'mmol O2 m-3 s-1', 'seed': True, 'default': np.nan},
        'sed_o2_flux': {'dtype': np.float32, 'units': 'mmol O2 m-2 s-1', 'seed': True, 'default': np.nan},
        'sed_o2_R_upper': {'dtype': np.float32, 'units': 'mmol O2 m-3 s-1', 'seed': True, 'default': np.nan},
        'sed_o2_R_lower': {'dtype': np.float32, 'units': 'mmol O2 m-3 s-1', 'seed': True, 'default': np.nan},
        'sed_o2_transition_depth': {'dtype': np.float32, 'units': 'm', 'seed': True, 'default': np.nan},
    }

    @classmethod
    def sediment_oxygen_diagnostic_variable_names(
            cls, oxygen_model, oxygen_demand_mode='VOLUMETRIC_RATE'):
        """Return the exact diagnostic schema for one sediment-O2 setup."""
        common = ['sed_o2_used', 'sed_o2_bottom']
        geometry = ['sed_o2_active_layer_thickness']
        transport = ['sed_o2_porosity', 'sed_o2_diffusivity']
        penetration = ['sed_o2_penetration_depth']
        flux = ['sed_o2_flux']
        dbl = ['sed_o2_dbl_thickness', 'sed_o2_k_bl', 'sed_o2_surface']

        if oxygen_model == 'FIXED_FRACTION':
            names = common
        elif oxygen_model == 'PRESCRIBED_OPD':
            names = common + geometry + penetration
        elif oxygen_model in ('ZERO_ORDER', 'ZERO_ORDER_DBL'):
            if oxygen_demand_mode not in ('VOLUMETRIC_RATE', 'BENTHIC_FLUX'):
                raise ValueError(
                    'Unknown sediment oxygen demand mode: '
                    f'{oxygen_demand_mode!r}'
                )
            names = common + geometry + transport
            if oxygen_model == 'ZERO_ORDER_DBL':
                names += dbl
            names += penetration
            if oxygen_demand_mode == 'VOLUMETRIC_RATE':
                names += ['sed_o2_consumption_rate']
            names += flux
        elif oxygen_model == 'TWO_LAYER_DBL':
            names = (
                common + geometry + transport + dbl + penetration + flux
                + ['sed_o2_R_upper', 'sed_o2_R_lower',
                   'sed_o2_transition_depth']
            )
        else:
            raise ValueError(
                f'Unknown sediment oxygen model: {oxygen_model!r}'
            )

        # Preserve deterministic NetCDF variable order and reject accidental
        # duplicate names in future edits.
        if len(names) != len(set(names)):
            raise RuntimeError(
                'Duplicate sediment-oxygen diagnostic variable in schema.'
            )
        return tuple(names)

    @classmethod
    def sediment_oxygen_diagnostic_variables(
            cls, oxygen_model, oxygen_demand_mode='VOLUMETRIC_RATE'):
        """Return Lagrangian variable definitions for the selected schema."""
        names = cls.sediment_oxygen_diagnostic_variable_names(
            oxygen_model, oxygen_demand_mode
        )
        return [
            (name, dict(cls.SEDIMENT_OXYGEN_DIAGNOSTIC_DEFINITIONS[name]))
            for name in names
        ]

    @classmethod
    def sediment_oxygen_diagnostic_schema_key(
            cls, oxygen_model, oxygen_demand_mode='VOLUMETRIC_RATE'):
        """Return the configuration components that change output schema."""
        # Validation is delegated to the schema builder.
        cls.sediment_oxygen_diagnostic_variable_names(
            oxygen_model, oxygen_demand_mode
        )
        if oxygen_model in ('ZERO_ORDER', 'ZERO_ORDER_DBL'):
            return (oxygen_model, oxygen_demand_mode)
        return (oxygen_model,)

    @classmethod
    def make_element_type(
        cls,
        save_single_degr_mass=False,
        save_bed_interaction=False,
        save_sediment_oxygen_diagnostics=False,
        sediment_oxygen_model='FIXED_FRACTION',
        sediment_oxygen_demand_mode='VOLUMETRIC_RATE',
    ):
        """
        Build a concrete Lagrangian3DArray element type from the active
        optional-output settings.
        Must be called before seeding, because element arrays are allocated
        when particles are seeded.
        """
        variables = list(cls.BASE_CHEMICAL_VARIABLES)
        if save_single_degr_mass:
            variables.extend(cls.SINGLE_DEGRADATION_VARIABLES)
        if save_bed_interaction:
            variables.extend(cls.BED_INTERACTION_VARIABLES)
        if save_sediment_oxygen_diagnostics:
            variables.extend(cls.sediment_oxygen_diagnostic_variables(
                sediment_oxygen_model,
                sediment_oxygen_demand_mode,
            ))
        class ChemicalElement(Lagrangian3DArray):
            """Concrete Chemical element type for this run configuration."""
            pass
        ChemicalElement.variables = Lagrangian3DArray.add_variables(variables)
        return ChemicalElement



class ChemicalDrift(ChemicalDriftPostProcessMixin, OceanDrift):
    """Chemical particle trajectory model based on the OpenDrift framework.
        Developed at MET Norway

        Generic module for particles that are subject to vertical turbulent
        mixing with the possibility for positive or negative buoyancy

        Particles could be e.g. oil droplets, plankton, or sediments

        Chemical functionality include interactions with solid matter
        (particles and sediments) through transformation processes, implemented
        with stochastic approach for dynamic partitioning.

        Under construction.
    """
    ###########################################################################
    # Class metadata, element type, and required-variable groups
    ###########################################################################

    ElementType = Chemical.make_element_type(
        save_single_degr_mass=False,
        save_bed_interaction=False,
        save_sediment_oxygen_diagnostics=False,
    )

    required_variables = {}
    BASE_REQUIRED_VARIABLES = {
        # Hydrodynamics
        'x_sea_water_velocity': {'fallback': None},                            # m/s
        'y_sea_water_velocity': {'fallback': None},                            # m/s
        'upward_sea_water_velocity': {'fallback': 0},                          # m/s
        'sea_surface_height': {'fallback': 0},                                 # m
        'x_wind': {'fallback': 0},                                             # m/s
        'y_wind': {'fallback': 0},                                             # m/s
        'ocean_vertical_diffusivity': {'fallback': 0.0001, 'profiles': True},  # m2/s
        'horizontal_diffusivity': {'fallback': 0, 'important': False},         # m2/s
        'land_binary_mask': {'fallback': None},                                # unitless, 0/1 or bool
        'sea_floor_depth_below_sea_level': {'fallback': 10000},                # m, positive downward
        # Needed by terminal velocity and many temperature/salinity corrections
        'sea_water_temperature': {'fallback': 10, 'profiles': True},           # degC
        'sea_water_salinity': {'fallback': 34, 'profiles': True},              # PSU
    }
    PARTITIONING_REQUIRED_VARIABLES = {
        # Dynamic partitioning / carrier concentrations
        'doc': {'fallback': 0.0},                                              # mmol C/kg
        'spm': {'fallback': 1},                                                # g/m3
        # Organic-carbon fractions
        'f_OC_spm': {'fallback': 0.01},                                        # gOC/g
        'f_OC_sed': {'fallback': 0.01},                                        # gOC/g
        # pH fields used for pH-dependent sorption/speciation corrections
        'sea_water_ph_reported_on_total_scale': {'fallback': 8.1, 'profiles': True,}, # pH units, dimensionless
        'pH_sediment': {'fallback': 6.9, 'profiles': False,},                         # pH units, dimensionless
        # Needed by _spm_g_m3() and _doc_mmolkg() when vertical reader levels  are not supplied.
        'ocean_mixed_layer_thickness': {'fallback': 50,'important': False,},   # m
    }
    SEDIMENT_EXCHANGE_REQUIRED_VARIABLES = {
        # Sediment-layer geometry
        'active_sediment_layer_thickness': {'fallback': 0},                    # m
        'interaction_sediment_layer_thickness': {'fallback': 0},               # m
        # Needed to check sediment exchange by adsorption/desorption.
        'ocean_mixed_layer_thickness': {'fallback': 50,'important': False,},   # m
        # Local organic-carbon fractions used when updating f_OC after particle/sediment transitions.
        'f_OC_spm': {'fallback': 0.01},                                        # gOC/g
        'f_OC_sed': {'fallback': 0.01},                                        # gOC/g
    }
    DIRECT_CURRENT_STRESS_REQUIRED_VARIABLES = {
        # Optional preferred hydro-model current bed stress
        'sea_floor_current_stress': {'fallback': -1.0, 'important': False},    # Pa
        # Finite negative fallback is used as physically invalid sentinel and is therefore rejected explicitly by
        # _direct_current_stress_array() at runtime
    }
    SEDIMENT_BOTTOM_VELOCITY_REQUIRED_VARIABLES = {
    # Needed by LOG_Z0 and GRAIN_D50 fallback stress modes
    'x_bottom_sea_water_velocity': {'fallback': 0, 'important': False,},       # m/s
    'y_bottom_sea_water_velocity': {'fallback': 0, 'important': False,},       # m/s
    'bottom_layer_thickness': {'fallback': 0.0, 'important': False,},          # m
    }
    SEDIMENT_LOG_Z0_REQUIRED_VARIABLES = {
        # Roughness for the LOG_Z0 current law and LOG_Z0 wave friction.
        'sea_floor_roughness_length': {'fallback': 0.0, 'important': False,},  # m
    }
    SEDIMENT_GRAIN_D50_REQUIRED_VARIABLES = {
        # Needed by GRAIN_D50 mode if a mapped bed grain size is available.
        # In USER critical-stress mode, mapped d50 is rejected
        'sea_floor_d50': {'fallback': 0, 'important': False,},                 # m
    }
    SEDIMENT_BULK_FLOW_REQUIRED_VARIABLES = {
        # Needed by MANNING, CHEZY, WHITE_COLEBROOK modes
        'x_depth_averaged_sea_water_velocity': {'fallback': 0, 'important': False,},  # m/s
        'y_depth_averaged_sea_water_velocity': {'fallback': 0, 'important': False,},  # m/s
        'hydraulic_radius': {'fallback': 0, 'important': False,},                     # m
    }
    SEDIMENT_RESUSPENSION_REQUIRED_VARIABLES = {
        # Optional mapped cohesive erodibility
        'sea_floor_erodibility_M': {'fallback': 0, 'important': False,},       # kg m-2 s-1 Pa-1
        # Optional mapped critical shear stress
        'sea_floor_resuspension_critstress': {'fallback': 0, 'important': False,}, # Pa
    }
    OTHER_STRESS_REQUIRED_VARIABLES = {
        'sea_floor_other_stress': {'fallback': np.nan, 'important': False,},        # Pa
    }
    WAVE_STRESS_REQUIRED_VARIABLES = {
    # Surface-wave forcing; no direct wave-stress/orbital-velocity inputs.
    'sea_surface_wave_significant_height': {'fallback': None},
    # A missing period is tolerated only at genuinely calm/dry locations.
    'sea_surface_wave_period_at_variance_spectral_density_maximum': {
        'fallback': np.nan, 'important': False},
    }

    WAVE_DIRECTION_REQUIRED_VARIABLES = {
        'sea_surface_wave_to_direction': {
            'fallback': np.nan, 'important': False},
        'sea_surface_wave_from_direction': {
            'fallback': np.nan, 'important': False},
    }
    DIRECT_WAVE_STRESS_REQUIRED_VARIABLES = {
        # Wave-only stress amplitude [Pa], not total wave-current stress.
        'sea_floor_wave_stress': {'fallback': None},
    }
    VOLATILIZATION_REQUIRED_VARIABLES = {
        'ocean_mixed_layer_thickness': {'fallback': 50,'important': False,},          # m
        'sea_water_ph_reported_on_total_scale': {'fallback': 8.1, 'profiles': True,}, # pH units
    }
    HYDROLYSIS_REQUIRED_VARIABLES = {
        'sea_water_ph_reported_on_total_scale': {'fallback': 8.1, 'profiles': True,}, # pH units, dimensionless
        'pH_sediment': {'fallback': 6.9, 'profiles': False,},                         # pH units, dimensionless
    }
    BIODEGRADATION_REQUIRED_VARIABLES = {
        # Dissolved oxygen used for biodegradation oxygen correction
        'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water': {
            'fallback': 225, 'profiles': True,},                               # mmol/m3
        'sea_water_ph_reported_on_total_scale': {'fallback': 8.1, 'profiles': True,}, # pH units, dimensionless
        'pH_sediment': {'fallback': 6.9, 'profiles': False,},                         # pH units, dimensionless
    }
    # Optional mapped sediment-oxygen fields used only by the selected
    # sediment oxygen model. np.nan fallbacks intentionally preserve the
    # distinction between a reader-supplied map and the configured fallback.
    SEDIMENT_OXYGEN_GEOMETRY_REQUIRED_VARIABLES = {
        'active_sediment_layer_thickness': {'fallback': 0,
            'important': False, 'profiles': False,},  # m
    }

    SEDIMENT_OXYGEN_OPD_REQUIRED_VARIABLES = {
        'sediment_oxygen_penetration_depth': {'fallback': np.nan,
            'important': False,'profiles': False,},  # m
    }

    SEDIMENT_OXYGEN_TRANSPORT_REQUIRED_VARIABLES = {
        'sea_floor_porosity': {'fallback': np.nan,
            'important': False,'profiles': False,},  # 1, m3 pore water / m3 bulk sediment
        'sediment_oxygen_diffusivity': {'fallback': np.nan,
            'important': False, 'profiles': False,},  # m2/s, effective pore-water diffusivity
    }

    SEDIMENT_OXYGEN_VOLUMETRIC_REQUIRED_VARIABLES = {
        'sediment_oxygen_consumption_rate': {'fallback': np.nan,
            'important': False, 'profiles': False,},  # mmol O2 m-3 bulk sediment s-1
    }

    SEDIMENT_OXYGEN_FLUX_REQUIRED_VARIABLES = {
        'benthic_oxygen_flux': {'fallback': np.nan,
            'important': False, 'profiles': False,},  # mmol O2 m-2 s-1, positive into sediment
    }

    SEDIMENT_OXYGEN_DBL_REQUIRED_VARIABLES = {
        'diffusive_boundary_layer_thickness': {'fallback': np.nan,
            'important': False, 'profiles': False,},  # m
    }

    SEDIMENT_OXYGEN_TWO_LAYER_REQUIRED_VARIABLES = {
        'sediment_oxygen_consumption_rate_upper': {'fallback': np.nan,
            'important': False, 'profiles': False,},  # mmol O2 m-3 bulk sediment s-1
        'sediment_oxygen_consumption_rate_lower': {'fallback': np.nan,
            'important': False, 'profiles': False,},  # mmol O2 m-3 bulk sediment s-1
        'sediment_oxygen_reactivity_transition_depth': {'fallback': np.nan,
            'important': False, 'profiles': False,},  # m
    }

    PHOTODEGRADATION_REQUIRED_VARIABLES = {
        # Light and water-column attenuation fields
        'solar_irradiance': {'fallback': 241,'important': False,},             # W/m2, or Ly/day if chemical:transformations:solar_input_unit = 'Ly_day'
        'mole_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water': {
            'fallback': 0, 'profiles': True,},                                 # mmol C/m3
        'ocean_mixed_layer_thickness': {'fallback': 50,'important': False,},   # m
        'doc': {'fallback': 0.0},                                              # mmol C/kg
        'spm': {'fallback': 1},                                                # g/m3
    }

    # Class-level superset of all environmental variables that can be required.
    # OpenDrift registers environment:fallback:* configuration keys during
    # BaseModel/OceanDrift construction from the class-level required_variables.
    # The runtime subset is still rebuilt later in prepare_run().
    required_variables = {}
    for _required_group in (
        BASE_REQUIRED_VARIABLES,
        PARTITIONING_REQUIRED_VARIABLES,
        SEDIMENT_EXCHANGE_REQUIRED_VARIABLES,
        DIRECT_CURRENT_STRESS_REQUIRED_VARIABLES,
        SEDIMENT_BOTTOM_VELOCITY_REQUIRED_VARIABLES,
        SEDIMENT_LOG_Z0_REQUIRED_VARIABLES,
        SEDIMENT_GRAIN_D50_REQUIRED_VARIABLES,
        SEDIMENT_BULK_FLOW_REQUIRED_VARIABLES,
        SEDIMENT_RESUSPENSION_REQUIRED_VARIABLES,
        OTHER_STRESS_REQUIRED_VARIABLES,
        WAVE_STRESS_REQUIRED_VARIABLES,
        WAVE_DIRECTION_REQUIRED_VARIABLES,
        DIRECT_WAVE_STRESS_REQUIRED_VARIABLES,
        VOLATILIZATION_REQUIRED_VARIABLES,
        HYDROLYSIS_REQUIRED_VARIABLES,
        BIODEGRADATION_REQUIRED_VARIABLES,
        SEDIMENT_OXYGEN_GEOMETRY_REQUIRED_VARIABLES,
        SEDIMENT_OXYGEN_OPD_REQUIRED_VARIABLES,
        SEDIMENT_OXYGEN_TRANSPORT_REQUIRED_VARIABLES,
        SEDIMENT_OXYGEN_VOLUMETRIC_REQUIRED_VARIABLES,
        SEDIMENT_OXYGEN_FLUX_REQUIRED_VARIABLES,
        SEDIMENT_OXYGEN_DBL_REQUIRED_VARIABLES,
        SEDIMENT_OXYGEN_TWO_LAYER_REQUIRED_VARIABLES,
        PHOTODEGRADATION_REQUIRED_VARIABLES,
    ):
        required_variables.update(_required_group)
    del _required_group

    ###########################################################################
    # Set-up and OpenDrift lifecycle
    ###########################################################################
    def __init__(self, *args, **kwargs):
        # Calling general constructor of parent class
        super(ChemicalDrift, self).__init__(*args, **kwargs)

        self._add_config(
            {
            'chemical:transfer_setup': {'type': 'enum',
                'enum': ['Sandnesfj_Al','metals', '137Cs_rev', 'custom', 'organics'], 'default': 'custom',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Define partitioning scheme'},
            'chemical:dynamic_partitioning': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle dynamic partitioning'},
            'chemical:slowly_fraction': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:irreversible_fraction': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:dissolved_diameter': {'type': 'float', 'default': 0,
                'min': 0, 'max': 100e-6, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Effective diameter assigned to dissolved elements.'},
            'chemical:particle_diameter': {'type': 'float', 'default': 5e-6,
                'min': 0, 'max': 100e-6, 'units': 'm',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Median diameter of sediment/SPM carrier particles, used as the median of the log-normal particle-diameter distribution.'},
            'chemical:doc_particle_diameter': {'type': 'float', 'default': 5e-6,
                'min': 0, 'max': 100e-6, 'units': 'm',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Median diameter of DOC/DOM aggregates in marine water, used as the median of the log-normal DOC-aggregate diameter distribution.'}, # https://doi.org/10.1038/246170a0
            'chemical:particle_concentration_half_depth': {'type': 'float', 'default': 20,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:doc_concentration_half_depth': {'type': 'float', 'default': 1000, # TODO: check better
                'min': 0, 'max': 3000, 'units': 'm',                                    # Vertical conc drops more slowly slower than for SPM
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},                     # example: 10.3389/fmars.2017.00436. lower limit around 40 umol/L
            'chemical:particle_diameter_uncertainty': {'type': 'float', 'default': 0.35,# https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/latest/model-description/water-and-sediment-properties/sediment-properties
                'min': 0, 'max': 5, 'units': '',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Log-space standard deviation sigma_ln of the particle-diameter distribution, used in a log-normal sampling around chemical:particle_diameter interpreted as the median diameter.'},
            'chemical:doc_particle_diameter_uncertainty': {'type': 'float', 'default': 0.70, # https://doi.org/10.1038/35248
                'min': 0, 'max': 5, 'units': '',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Log-space standard deviation sigma_ln of the DOC-aggregate diameter distribution, used in a log-normal sampling around chemical:doc_particle_diameter interpreted as the median diameter.'},
            'seed:LMM_fraction': {'type': 'float','default': .1,
                'min': 0, 'max': 1, 'units': '',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Fraction of dissolved elements at seeding'},
            'seed:particle_fraction': {'type': 'float','default': 0.9,
                'min': 0, 'max': 1, 'units': '',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Fraction of SPM elements at seeding'},
            # Species
            'chemical:species:LMM': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle LMM specie'},
            'chemical:species:LMMcation': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle LMMcation specie'},
            'chemical:species:LMManion': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle LMManion specie'},
            'chemical:species:Colloid': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle Colloid specie'},
            'chemical:species:Humic_colloid': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle Humic_colloid specie'},
            'chemical:species:Polymer': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle Polymer specie'},
            'chemical:species:Particle_reversible': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle Particle_reversible specie'},
            'chemical:species:Particle_slowly_reversible': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle Particle_slowly_reversible specie'},
            'chemical:species:Particle_irreversible': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle Particle_irreversible specie'},
            'chemical:species:Sediment_reversible': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle Sediment_reversible specie'},
            'chemical:species:Sediment_slowly_reversible': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle Sediment_slowly_reversible specie'},
            'chemical:species:Sediment_buried': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle Sediment_buried specie'},
            'chemical:species:Sediment_irreversible': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle Sediment_irreversible specie'},
            # Transformations
            'chemical:transformations:Kd': {'type': 'float', 'default': 2.0,
                'min': 0, 'max': 1e9, 'units': 'm3/kg',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Water/sediment partitioning coefficient for metals'},
            'chemical:transformations:irreversible_rate': {'type': 'float', 'default': 0.0,
                'min': 0, 'max': 1e6, 'units': 's-1',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': 'First-order transfer rate from reversible to irreversible particle/sediment pools.'},
            'chemical:transformations:S0': {'type': 'float', 'default': 0.0,
                'min': 0, 'max': 100, 'units': 'PSU',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Parameter controlling salinity dependency of Kd for metals'},
            'chemical:transformations:Dc': {'type': 'float', 'default': 1.16e-5,                 # Simonsen 2019
                'min': 0, 'max': 1e6, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Desorption rate of metals from particles'},
            'chemical:transformations:slow_coeff': {'type': 'float', 'default': 0,     # 1.2e-7, # Simonsen 2019
                'min': 0, 'max': 1e6, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Adsorption coefficient to slowly reversible fractions (metals)'},
            'chemical:transformations:slow_coeff_des': {'type': 'float', 'default': 0, # 2.77e-7 1/s (up to 1.11e-6 1/s) # doi.org/10.1021/es960300+ Cornelissen et al. (1997) # doi.org/10.1897/06-104R.1 Birdwell et al. (2007)
                'min': 0, 'max': 1e6, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Desorption coefficient from slowly reversible fractions (organics)'},
            'chemical:transformations:slow_coeff_ads': {'type': 'float', 'default': 0, # 1.88e-7 1/s (up to 7.43e-6 1/s) (40% of chem in slow fraction) # doi:10.1016/j.chemosphere.2005.02.092 Dunnivant et al. (2005)
                'min': 0, 'max': 1e6, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Adsorption coefficient to slowly reversible fractions (organics)'},
            'chemical:transformations:volatilization': {'type': 'bool', 'default': False,
                'description': 'Chemical is evaporated.',
                'level': CONFIG_LEVEL_BASIC},
            'chemical:transformations:degradation': {'type': 'bool', 'default': False,
                'description': 'Chemical mass is degraded.',
                'level': CONFIG_LEVEL_BASIC},
            'chemical:transformations:degradation_mode': {'type': 'enum',
                'enum': ['OverallRateConstants', 'SingleRateConstants'], 'default': 'OverallRateConstants',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Select degradation mode'},
            'chemical:transformations:mass_checks': {'type': 'bool', 'default': False,
                'description': 'Check consistency of degraded mass across mechanisms and wat/sed.',
                'level': CONFIG_LEVEL_BASIC},
            # Sorption/desorption
            'chemical:transformations:dissociation': {'type': 'enum',
                'enum': ['nondiss','acid', 'base', 'amphoteric'], 'default': 'nondiss',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Select dissociation mode'},
            'chemical:transformations:LogKOW': {'type': 'float', 'default': 3.361,           # Naphthalene
                'min': -3, 'max': 10, 'units': '',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Log10 of Octanol/Water partitioning coefficient'},
            'chemical:transformations:TrefKOW': {'type': 'float', 'default': 25.,            # Naphthalene
                'min': -3, 'max': 30, 'units': 'C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Reference temperature of KOW'},
            'chemical:transformations:DeltaH_KOC_Sed': {'type': 'float', 'default': -21036., # Naphthalene
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of sorption to sediments'},
            'chemical:transformations:DeltaH_KOC_DOM': {'type': 'float', 'default': -25900., # Naphthalene
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of sorption to DOM'},
            'chemical:transformations:Setchenow': {'type': 'float', 'default': 0.2503,       # Naphthalene
                'min': -2, 'max': 1, 'units': 'L/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Setchenow constant of organic chemicals'},
            'chemical:transformations:pKa_acid': {'type': 'float', 'default': -1,
                'min': -1, 'max': 14, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'pKa of chemical'},
            'chemical:transformations:pKa_base': {'type': 'float', 'default': -1,
                'min': -1, 'max': 14, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': "pKa of chemical's conjugated acid"},
            'chemical:transformations:KOC_DOM': {'type': 'float', 'default': -1,
                'min': -1, 'max': 10000000000, 'units': 'L/KgOC',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'DOM Organic carbon/Water partitioning coefficient'},
            'chemical:transformations:KOC_sed': {'type': 'float', 'default': -1,
                'min': -1, 'max': 10000000000, 'units': 'L/KgOC',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'SPM/sed Organic carbon/Water partitioning coefficient'},
            'chemical:transformations:KOC_sed_acid': {'type': 'float', 'default': -1,
                'min': -1, 'max': 10000000000, 'units': 'L/KgOC',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'SPM/sed Organic carbon/Water partitioning coefficient for acid anionic species'},
            'chemical:transformations:KOC_sed_base': {'type': 'float', 'default': -1,
                'min': -1, 'max': 10000000000, 'units': 'L/KgOC',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'SPM/sed Organic carbon/Water partitioning coefficient for base cationic species'},
            'chemical:transformations:KOC_DOM_acid': {'type': 'float', 'default': -1,
                'min': -1, 'max': 10000000000, 'units': 'L/KgOC',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'DOM Organic carbon/Water partitioning coefficient for acid anionic species'},
            'chemical:transformations:KOC_DOM_base': {'type': 'float', 'default': -1,
                'min': -1, 'max': 10000000000, 'units': 'L/KgOC',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'DOM Organic carbon/Water partitioning coefficient for base cationic species'},
            'chemical:transformations:fOC_SPM': {'type': 'float', 'default': 0.05,
                'min': 0.01, 'max': 0.1, 'units': 'gOC/g',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Organic carbon fraction of SPM'},
            'chemical:transformations:fOC_sed': {'type': 'float', 'default': 0.05,
                'min': 0.01, 'max': 0.1, 'units': 'gOC/g',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Organic carbon fraction of sediments'},
            'chemical:transformations:aggregation_rate': {'type': 'float', 'default': 0,
                'min': 0, 'max': 1, 'units': 's-1',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Aggregation rate of DOM in marine water'},
            # Degradation in water column
            'chemical:transformations:t12_W_tot': {'type': 'float', 'default': 224.08,       # Naphthalene
                'min': 1, 'max': None, 'units': 'hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Half life in water, total'},
            'chemical:transformations:Tref_kWt': {'type': 'float', 'default': 25.,           # Naphthalene
                'min': -3, 'max': 30, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Reference temperature of t12_W_tot'},
            'chemical:transformations:DeltaH_kWt': {'type': 'float', 'default': 50000.,      # Generic
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of t12_W_tot'},
            # Degradation in sediment layer
            'chemical:transformations:t12_S_tot': {'type': 'float', 'default': 5012.4,       # Naphthalene
                'min': 1, 'max': None, 'units': 'hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Half life in sediments, total'},
            'chemical:transformations:ssrev_slow_deg_factor': {'type': 'float', 'default': 1, # No slow degradation
                'min': 0, 'max': 1, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Correction factor for slower degradation in buried sediments'},
            'chemical:transformations:Tref_kSt': {'type': 'float', 'default': 25.,            # Naphthalene
                'min': -3, 'max': 30, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Reference temperature of t12_S_tot'},
            'chemical:transformations:DeltaH_kSt': {'type': 'float', 'default': 50000.,       # Generic
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of t12_S_tot'},
            # Volatilization
            'chemical:transformations:MolWt': {'type': 'float', 'default': 128.1705,          # Naphthalene
                'min': 50, 'max': 1000, 'units': 'g/mol',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Molecular weight'},
            'chemical:transformations:Henry': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'atm m3 mol-1',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Henry constant (uses Tref_Slb as Tref)'},
            'chemical:transformations:DeltaH_Henry': {'type': 'float','default': 47000.,    # 8–93 kJ mol https://doi.org/10.1016/S0045-6535(00)00505-1
                'min': 0., 'max': 100000.,'units': 'J/mol', 'level': CONFIG_LEVEL_ESSENTIAL,
                'description': 'Enthalpy of air-water volatilization used for temperature correction of Henry constant' },
            # Vapour pressure
            'chemical:transformations:Vpress': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'Pa',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Vapour pressure'},
            'chemical:transformations:Tref_Vpress': {'type': 'float', 'default': 25.,       # Naphthalene
                'min': None, 'max': None, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Vapour pressure ref temp'},
            'chemical:transformations:DeltaH_Vpress': {'type': 'float', 'default': 55925.,  # Naphthalene
                'min': -100000., 'max': 150000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of volatilization'},
            # Solubility
            'chemical:transformations:Solub': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'g/m3',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Solubility'},
            'chemical:transformations:Tref_Solub': {'type': 'float', 'default': 25.,        # Naphthalene
                'min': None, 'max': None, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Solubility ref temp'},
            'chemical:transformations:DeltaH_Solub': {'type': 'float', 'default': 25300.,   # Naphthalene
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of solubilization'},
            'chemical:transformations:Tref_Henry': {'type': 'float', 'default': 25.,        # Naphthalene
                'min': None, 'max': None, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Henry constant ref temp'},
            # Sedimentation/Resuspension
            'chemical:sediment:mixing_depth': {'type': 'float', 'default': 0.03,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Height of sediments active layer'},
            'chemical:sediment:density': {'type': 'float', 'default': 2600,
                'min': 0, 'max': 10000, 'units': 'kg/m3',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Density of sediments'},
            'chemical:sediment:effective_fraction': {'type': 'float', 'default': 0.9,
                'min': 0, 'max': 1, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Fraction of effective sediments acting as sorbents'},
            'chemical:sediment:corr_factor': {'type': 'float', 'default': 0.1,
                'min': 0, 'max': 10, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Correction factor for ads/des-orption from sediments'},
            'chemical:sediment:porosity': {'type': 'float', 'default': 0.6,
                'min': 0, 'max': 1, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Fraction of sediment volume made of water, adimensional'},
            'chemical:sediment:layer_thickness': {'type': 'float', 'default': 1,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Thickness of seabed interaction layer'},
            'chemical:sediment:desorption_depth': {'type': 'float', 'default': 1,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Distance from seabed where desorbed elements are moved'},
            'chemical:sediment:desorption_depth_uncert': {'type': 'float', 'default': 0.5,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:sediment:resuspension_depth': {'type': 'float', 'default': 1,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Distance from seabed where resuspended elements are moved'},
            'chemical:sediment:resuspension_depth_uncert': {'type': 'float', 'default': 0.5,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:sediment:burial_rate': {'type': 'float', 'default': 0.0003,   # Parnis, J.M. Mackay, D. (2020) # doi.org/10.1201/9780367809829
                'min': 0, 'max': 10, 'units': 'm/year',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Rate of sediment burial'},
            'chemical:sediment:buried_leaking_rate': {'type': 'float', 'default': 0,
                'min': 0, 'max': 10, 'units': 's-1',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Rate of resuspension of buried sediments'},
            'chemical:sediment:buried_leak_to_ssrev_fraction': {'type': 'float', 'default': 0.0,
                'min': 0, 'max': 1, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Fraction of buried-sediment leaking that returns to Sediment slowly reversible (rest goes to Sediment reversible).'},
            'chemical:compound': {'type': 'str', 'default': '', 'min_length': 0, 'max_length': 256,
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Name of modelled chemical' },
            # Bed shear stress / roughness
            'chemical:sediment:stress_param_mode': {'type': 'enum',
                'enum': ['LOG_Z0', 'MANNING', 'CHEZY', 'WHITE_COLEBROOK', 'GRAIN_D50'],
                'default': 'LOG_Z0', 'level': CONFIG_LEVEL_ESSENTIAL,
                'description': 'Bed-stress parameterization. Use direct hydro-model bed stress if available; otherwise use the selected fallback mode.'},
            'chemical:sediment:roughness_length': {'type': 'float', 'default': -1.0,
                'min': -1.0, 'max': 10, 'units': 'm', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Uniform fallback bottom roughness length z0 [m]. Used only when sea_floor_roughness_length is not supplied by a reader, or to replace invalid reader values. Must be > 0 when used.'},
            'chemical:sediment:bottom_layer_thickness': {'type': 'float', 'default': 1.0,
                'min': 0, 'max': 100, 'units': 'm', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Fallback bottom layer thickness if environment.bottom_layer_thickness is unavailable. For LOG_Z0, the reference height is taken as 0.5*bottom_layer_thickness.'},
            'chemical:sediment:bulk_hydraulic_radius': {'type': 'float', 'default': 0.0,
                'min': 0, 'max': 1e6, 'units': 'm', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Fallback hydraulic radius for MANNING/CHEZY/WHITE_COLEBROOK. If <= 0, use sea-floor depth as approximation.'},
            'chemical:sediment:manning_n': {'type': 'float', 'default': 0.02,
                'min': 0, 'max': 1, 'units': 's m-1/3', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Manning n for MANNING mode, used with depth-averaged velocity and hydraulic radius/depth.'},
            'chemical:sediment:chezy_C': {'type': 'float', 'default': 65.0,
                'min': 1e-6, 'max': 1e4, 'units': 'm1/2 s-1', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Chezy coefficient for CHEZY mode, used with depth-averaged velocity and hydraulic radius/depth.'},
            'chemical:sediment:nikuradse_ks': {'type': 'float', 'default': 0.02,
                'min': 1e-8, 'max': 10, 'units': 'm', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Nikuradse roughness for WHITE_COLEBROOK mode, used with depth-averaged velocity and hydraulic radius/depth.'},
            'chemical:sediment:d50': {'type': 'float', 'default': 2.0e-4,
                'min': 1e-8, 'max': 1, 'units': 'm', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Median grain size for GRAIN_D50 mode and d50-based critical stress. The per-element d50 is initialized from particle diameter at seeding and is then used for d50-based critical-stress calculations, while diameter continues to control terminal velocity. In USER mode, mapped sea_floor_d50 is rejected and chemical:sediment:d50 must equal chemical:particle_diameter so that carrier size remains uniform without hidden d50-driven jumps.'},
            'chemical:sediment:cd_min': {'type': 'float', 'default': 0.0,
                'min': 0, 'max': 1, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Lower bound for drag coefficient.'},
            'chemical:sediment:cd_max': {'type': 'float', 'default': 0.2,
                'min': 0, 'max': 10, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Upper bound for drag coefficient.'},
            'chemical:sediment:include_wave_stress': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Include wave bed stress from the selected CALCULATED or DIRECT source.'},
            'chemical:sediment:wave_stress_source': {'type': 'enum',
                'enum': ['CALCULATED', 'DIRECT'], 'default': 'CALCULATED',
                'level': CONFIG_LEVEL_BASIC,
                'description':
                    'CALCULATED derives wave-only bed stress from explicitly supplied significant wave height Hs, peak period, depth and roughness. '
                    'Supplied wave forcing is authoritative: Hs=0 is retained as a valid calm-wave state and is not replaced by a wind-derived estimate; '
                    'wave direction 0 degrees is retained as a valid geographic bearing. DIRECT uses reader-supplied sea_floor_wave_stress: a finite '
                    'non-negative wave-only stress amplitude in Pa. Sources are mutually exclusive.'},
            'chemical:sediment:calm_wave_wind_warning_threshold': {
                'type': 'float', 'default': 5.0, 'min': -1.0, 'max': 100.0,
                'units': 'm/s', 'level': CONFIG_LEVEL_ADVANCED,
                'description':
                    'For CALCULATED wave stress, emit a one-time forcing-consistency warning when explicitly supplied Hs=0 coincides with local wind speed '
                    'above this threshold. The supplied Hs remains authoritative and is never replaced. Set a negative value to disable the warning.'},
            'chemical:sediment:wave_height_convention': {'type': 'enum',
                'enum': ['RMS_EQUIVALENT', 'SIGNIFICANT_HEIGHT'],
                'default': 'RMS_EQUIVALENT', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Representative regular-wave height: Hs/sqrt(2) for RMS_EQUIVALENT, or Hs for SIGNIFICANT_HEIGHT. Both use the supplied peak period.'},
            'chemical:sediment:wave_depth_convention': {'type': 'enum',
                'enum': ['MEAN_SEA_LEVEL', 'INSTANTANEOUS'],
                'default': 'MEAN_SEA_LEVEL', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'MEAN_SEA_LEVEL adds sea_surface_height to bathymetry; INSTANTANEOUS uses the supplied depth directly. Missing surface elevation defaults to zero.'},
            'chemical:sediment:wave_roughness_mode': {'type': 'enum',
                'enum': ['CURRENT_MODE', 'LOG_Z0', 'GRAIN_D50', 'NIKURADSE'],
                'default': 'CURRENT_MODE', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Wave roughness source. CURRENT_MODE follows the current-stress mode; LOG_Z0 uses mapped/configured z0; GRAIN_D50 uses ks=2.5*d50; NIKURADSE uses configured nikuradse_ks.'},
            'chemical:sediment:include_other_stress': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Whether to include externally provided other bed stress when available.'},
            'chemical:sediment:shear_stress_combination': {'type': 'enum',
                'enum': ['sum', 'max', 'rss', 'SOULSBY_CLARKE'], 'default': 'sum',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': 'sum/max/rss combine stress magnitudes heuristically. SOULSBY_CLARKE is the legacy name for the simplified Soulsby wave-current peak formulation, evaluated over both wave half-cycles; other stress is then added in quadrature.'},
            'chemical:sediment:use_critstress_heterogeneity': {
                'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Apply persistent sub-grid heterogeneity factor to resuspension critical stress.'},
            'chemical:sediment:critstress_heterogeneity_lnsigma': {
                'type': 'float', 'default': 0.15, 'min': 0.0, 'max': 2.0, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Log-space standard deviation of the persistent multiplicative heterogeneity factor for resuspension critical stress.'},
            # Probabilistic exchange scheme
            'chemical:sediment:save_bed_interaction': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC, 'description': 'Toggle save of sediment probabilistic exchange scheme arrays'},
            'chemical:sediment:erodibility_M': {'type': 'float', 'default': 1.0e-4,
                'min': 0.0, 'max': 1.0, 'units': 'kg m-2 s-1 Pa-1', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Partheniades erodibility coefficient for cohesive resuspension. Used as a uniform fallback when no sea_floor_erodibility_M reader is supplied; a mapped erodibility field is preferred when available.'},
            'chemical:sediment:noncohesive_resuspension_timescale': {'type': 'float', 'default': 3600.0,
                'min': 1e-6, 'max': 1e9, 'units': 's', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Characteristic timescale for noncohesive excess-shear pickup probability.'},
            'chemical:sediment:noncohesive_excess_shear_exponent': {'type': 'float', 'default': 1.0,
                'min': 0.1, 'max': 10.0, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Exponent for noncohesive excess-shear pickup probability.'},
            'chemical:sediment:resuspension_probability_model': {'type': 'enum',
                'enum': ['INSTANTANEOUS_EXCESS_SHEAR', 'TIMESTEP_DEPENDENT'],
                'default': 'TIMESTEP_DEPENDENT',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Resuspension probability model. INSTANTANEOUS_EXCESS_SHEAR reproduces the old time-independent excess-shear probability; TIMESTEP_DEPENDENT uses timestep-dependent pickup/erosion probability.'},
            'chemical:sediment:exchange_scheme': {'type': 'enum',
                'enum': ['EXCESS_SHEAR_PROBABILITY'],'default': 'EXCESS_SHEAR_PROBABILITY',
                'level': CONFIG_LEVEL_ESSENTIAL,
                'description': 'Probabilistic sediment exchange scheme.'},
            'chemical:sediment:deposition_reduction_factor': {'type': 'float', 'default': -1.0,
                'min': -1.0, 'max': 1.0, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'If >= 0, use a constant deposition reduction factor. If < 0, use Krone-type stress reduction.'},
            'chemical:sediment:enable_deposition': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Enable suspended-to-bed deposition probability.'},
            'chemical:sediment:enable_resuspension': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Enable bed-to-suspended resuspension probability.'},
            'chemical:sediment:deposition_critstress': {'type': 'float', 'default': 0.05,
                'min': 0, 'max': 1e6, 'units': 'Pa', 'level': CONFIG_LEVEL_ESSENTIAL,
                'description': 'Critical shear stress for deposition/sedimentation.'},
            'chemical:sediment:resuspension_critstress': {'type': 'float', 'default': 0.5,
                'min': 0, 'max': 1e6, 'units': 'Pa', 'level': CONFIG_LEVEL_ESSENTIAL,
                'description': 'Critical shear stress for resuspension/erosion. If a mapped sea_floor_resuspension_critstress reader is supplied, that mapped value is preferred locally; otherwise this configured value or the d50-based calculation is used.'},
            'chemical:sediment:resuspension_critustar': {'type': 'float', 'default': -1.0,
                'min': -1.0, 'max': 100.0, 'units': 'm/s', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Critical shear velocity for resuspension. If >= 0, override resuspension_critstress using tau_cr = rho * ustar^2. If < 0, use resuspension_critstress directly.'},
            # D50-based critical stress
            'chemical:sediment:resuspension_critstress_mode': {'type': 'enum',
                'enum': ['USER', 'FROM_D50'], 'default': 'USER',
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Use user-specified resuspension_critstress or compute it from d50.'},
            'chemical:sediment:resuspension_critstress_branch': {'type': 'enum',
                'enum': ['AUTO', 'NONCOHESIVE', 'COHESIVE'], 'default': 'COHESIVE',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Branch used when computing resuspension_critstress from d50.'},
            'chemical:sediment:resuspension_critstress_method': {'type': 'enum',
                'enum': ['soulsby_whitehouse', 'van_rijn', 'laursen', 'mpm', 'wu'],
                'default': 'soulsby_whitehouse', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Noncohesive method used when computing resuspension_critstress from d50.'},
            'chemical:sediment:critstress_rho_s': {'type': 'float', 'default': 2650.0,
                'min': 1000.0, 'max': 10000.0, 'units': 'kg/m3', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Sediment density used in d50-based critical stress calculation.'},
            'chemical:sediment:critstress_nu': {'type': 'float', 'default': 1.004e-6,
                'min': 1e-8, 'max': 1e-3, 'units': 'm2/s', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Kinematic viscosity used in d50-based critical stress calculation.'},
            'chemical:sediment:critstress_owen_rho_d': {'type': 'float', 'default': -1.0,
                'min': -1.0, 'max': 1e6, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Dry bulk density for cohesive Owen-type critical stress.'},
            'chemical:sediment:critstress_owen_a': {'type': 'float', 'default': -1.0,
                'min': -1.0, 'max': 1e6, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Coefficient a in tau_ce = a * rho_d^b for cohesive sediments.'},
            'chemical:sediment:critstress_owen_b': {'type': 'float', 'default': -1.0,
                'min': -1.0, 'max': 10.0, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Exponent b in tau_ce = a * rho_d^b for cohesive sediments.'},
            # Single process degradation
            #  main implementation from AQUATOX https://www.epa.gov/sites/default/files/2014-03/documents/technical-documentation-3-1.pdf
            'chemical:transformations:Save_single_degr_mass': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle save of mass degraded by single mechanism'},
            'chemical:transformations:Photodegradation': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle photodegradation'},
            'chemical:transformations:Biodegradation': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle biodegradation'},
            'chemical:transformations:Hydrolysis': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle hydrolysis'},
            # Biodegradation
            'chemical:transformations:k_DecayMax_water': {'type': 'float', 'default': 0,          # Default for no aerobic biodegradation
                'min': 0, 'max': None, 'units': '1/hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Max first-order rate constant for biodegradation in aerobic condition'},
            'chemical:transformations:k_Anaerobic_water': {'type': 'float', 'default': 0,         # Default for no anaerobic biodegradation
                'min': 0, 'max': None, 'units': '1/hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Max first-order rate constant for biodegradation in anaerobic condition '},
            'chemical:transformations:HalfSatO_w': {'type': 'float', 'default': 0.5,              # Default from AQUATOX
                'min': 0.01, 'max': None, 'units': 'g/m3',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Half-saturation constant for oxygen'},
            'chemical:transformations:T_Max_bio': {'type': 'float', 'default': 50,                # Default from AQUATOX
                'min': 1, 'max': None, 'units': 'C',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Maximum temperature at which biodegradation process will occur'},
            'chemical:transformations:T_Opt_bio': {'type': 'float', 'default': 22,                # Default from AQUATOX
                 'min': 1, 'max': None, 'units': 'C',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Optimal temperature for biodegradation'},
            'chemical:transformations:T_Adp_bio': {'type': 'float', 'default': 2,                 # Default from AQUATOX
                 'min': 0.1, 'max': None, 'units': 'C',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': '“adaptation” temperature below which there is no acclimation for biobegradation'},
            'chemical:transformations:Max_Accl_bio': {'type': 'float', 'default': 2,              # Default from AQUATOX
                 'min': 0.1, 'max': None, 'units': 'C',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Maximum acclimation allowed for biodegratation'},
            'chemical:transformations:Dec_Accl_bio': {'type': 'float', 'default': 0.5,            # Default from AQUATOX
                 'min': 0.1, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Coefficient for decreasing acclimation as temperature approaches T_Adp_bio'},
            'chemical:transformations:Q10_bio': {'type': 'float', 'default': 2,                   # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Slope or rate of change per 10°C temperature change for biodegradation'},
            'chemical:transformations:pH_min_bio': {'type': 'float', 'default': 5,                # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Minimum pH below which limitation on biodegradation rate occurs'},
            'chemical:transformations:pH_max_bio': {'type': 'float', 'default': 8.5,              # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Maximum pH over which limitation on biodegradation rate occurs'},
            # Sediment oxygen used for sediment biodegradation
            'chemical:sediment:oxygen_model': {'type': 'enum',
                'enum': ['FIXED_FRACTION', 'PRESCRIBED_OPD', 'ZERO_ORDER',
                    'ZERO_ORDER_DBL', 'TWO_LAYER_DBL',],
                'default': 'FIXED_FRACTION', 'level': CONFIG_LEVEL_ADVANCED,
                'description': ('Reduced-order model used to estimate dissolved oxygen in '
                    'the active sediment layer for biodegradation. Buried sediment '
                    'is handled separately and forced to zero oxygen.'),},
            'chemical:sediment:save_oxygen_diagnostics': {'type': 'bool',
                'default': False, 'level': CONFIG_LEVEL_BASIC,
                'description': ('Save mode-specific per-element sediment-oxygen diagnostic '
                    'arrays. Only variables required by the selected oxygen model '
                    'and oxygen-demand mode are added to the output schema.'),},
            'chemical:sediment:oxygen_active_fraction': {'type': 'float',
                'default': 1.0, 'min': 0.0, 'max': 1.0, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': ('Fraction of bottom-water dissolved oxygen assigned to active '
                    'sediment in FIXED_FRACTION mode.'),},
            'chemical:sediment:oxygen_penetration_depth': {
                'type': 'float', 'default': 0.003, 'min': 0.0, 'max': 100.0,
                'units': 'm', 'level': CONFIG_LEVEL_ADVANCED, 'description': (
                'Uniform fallback oxygen penetration depth used in PRESCRIBED_OPD mode when no valid '
                'sediment_oxygen_penetration_depth map is supplied.'),},
            'chemical:sediment:oxygen_demand_mode': {'type': 'enum', 'enum': ['VOLUMETRIC_RATE', 'BENTHIC_FLUX'],
                'default': 'VOLUMETRIC_RATE', 'level': CONFIG_LEVEL_ADVANCED,
                'description': ('Oxygen-demand input used by ZERO_ORDER and ZERO_ORDER_DBL: '
                    'volumetric sediment oxygen consumption or benthic oxygen flux.'),},
            'chemical:sediment:oxygen_consumption_rate': {'type': 'float',
                'default': 0.03, 'min': 0.0, 'max': None, 'units': 'mmol O2 m-3 s-1',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ('Uniform fallback zero-order oxygen consumption rate per bulk '
                    'sediment volume when no valid sediment_oxygen_consumption_rate map is supplied.'),},
            'chemical:sediment:benthic_oxygen_flux': {'type': 'float', 'default': 9.0e-5,
                'min': 0.0, 'max': None, 'units': 'mmol O2 m-2 s-1', 'level': CONFIG_LEVEL_ADVANCED,
                'description': ('Uniform fallback benthic oxygen flux, positive into sediment, '
                    'when no valid benthic_oxygen_flux map is supplied.'),},
            'chemical:sediment:oxygen_molecular_diffusivity': {'type': 'float', 'default': 2.0e-9,
                'min': 1.0e-12, 'max': 1.0e-7,
                'units': 'm2/s', 'level': CONFIG_LEVEL_ADVANCED,
                'description': ('Molecular diffusivity of dissolved oxygen in free water used '
                    'to derive sediment diffusivity when no valid sediment_oxygen_diffusivity map is supplied.'),},
            'chemical:sediment:oxygen_dbl_thickness': {'type': 'float',
                'default': 0.001, 'min': 1.0e-6, 'max': 0.1, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ('Uniform fallback diffusive boundary-layer thickness used in '
                    'DBL modes when no valid diffusive_boundary_layer_thickness map is supplied.'),},
            'chemical:sediment:oxygen_consumption_rate_upper': {'type': 'float',
                'default': 0.03, 'min': 0.0, 'max': None, 'units': 'mmol O2 m-3 s-1',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ('Uniform fallback zero-order oxygen consumption rate in the '
                    'upper reactive sediment layer for TWO_LAYER_DBL.'),},
            'chemical:sediment:oxygen_consumption_rate_lower': {'type': 'float',
                'default': 0.03, 'min': 0.0, 'max': None, 'units': 'mmol O2 m-3 s-1',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ('Uniform fallback zero-order oxygen consumption rate in the '
                'lower reactive sediment layer for TWO_LAYER_DBL. Equal upper/lower defaults reduce the model to the one-layer case.'),},
            'chemical:sediment:oxygen_reactivity_transition_depth': {'type': 'float',
                'default': 0.001, 'min': 0.0, 'max': 100.0, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ('Uniform fallback depth of the upper/lower reactivity transition '
                    'for TWO_LAYER_DBL when no valid map is supplied.'),},
            # Hydrolysis
            # Based on the approach reported by Mabey, W., & Mill, T. (1978) https://doi.org/10.1063/1.555572 (Figure 1)
            'chemical:transformations:k_Acid': {'type': 'float', 'default': 0,        # Default: no acid catalyzed hydrolysis
                 'min': None, 'max': None, 'units': 'L/mol*h',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'acid-catalyzed pseudo-second-order coefficient'},
            'chemical:transformations:k_Base': {'type': 'float', 'default': 0,        # Default: no base catalyzed hydrolysis
                 'min': None, 'max': None, 'units': 'L/mol*h',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'base-catalyzed pseudo-second-order coefficient'},
            'chemical:transformations:k_Hydr_Uncat': {'type': 'float', 'default': 0,  # Default: no hydrolysis
                 'min': 0, 'max': None, 'units': '1/hours',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Measured first-order hydrolysis rate at pH 7'},
            # Photolysis
            'chemical:transformations:solar_input_unit': { 'type': 'enum', 'enum': ['W_m2', 'Ly_day'], 'default': 'W_m2',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Unit of environment.solar_irradiance. AQUATOX photolysis uses Ly/day; W/m2 is converted using 1 Ly/day = 0.4843 W/m2.'},
            'chemical:transformations:k_Photo': {'type': 'float', 'default': 0,            # Default: no photolysis
                 'min': 0, 'max': None, 'units': '1/hours',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Measured first-order photolysis rate'},
            'chemical:transformations:RadDistr': {'type': 'float', 'default': 1.6,         # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Radiance distribution function, which is the ratio of the average pathlength to the depth'},
            'chemical:transformations:RadDistr0_ml': {'type': 'float', 'default': 1.2,     # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Standard radiance distribution function in the Mixed Layer'},
            'chemical:transformations:RadDistr0_bml': {'type': 'float', 'default': 1.6,    # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Standard radiance distribution function below the Mixed Layer'},
            'chemical:transformations:WaterExt': {'type': 'float', 'default': 0.02,        # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '1/m',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Extinction coefficient of light in the water with depht due to water'},
            'chemical:transformations:ExtCoeffDOM': {'type': 'float', 'default': 0.03,    # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '1/(m*g/m3)',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Extinction coefficient of light in the water with depht due to DOM'},
            'chemical:transformations:ExtCoeffSPM': {'type': 'float', 'default': 0.17,     # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '1/(m*g/m3)',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Extinction coefficient of light in the water with depht due to SPM'},
            'chemical:transformations:ExtCoeffPHY': {'type': 'float', 'default': 0.099,    # Default from AQUATOX
                 'min': 0, 'max': None, 'units': '1/(m*g/m3)',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Extinction coefficient of light in the water with depht due to blue-gree phytoplankton'},
            'chemical:transformations:C2PHYC': {'type': 'float', 'default': 0.44,          # Default from https://doi.org/10.1007/BF00006636
                 'min': 0, 'max': None, 'units': 'g_Caron/g_Biomass',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Phytoplankton carbon content'},
            'chemical:transformations:AveSolar': {'type': 'float', 'default': 500,         # Default from AQUATOX
                 'min': 0, 'max': None, 'units': 'Ly/day',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Average light intensity for late spring or early summer, corresponding to time when photolytic half-life is often measured'},
            })

        self._set_config_default('drift:vertical_mixing', True)
        self._set_config_default('drift:vertical_mixing_at_surface', True)
        self._set_config_default('drift:vertical_advection_at_surface', True)
        self._sync_required_variables_from_config()

    def _build_required_variables(self):
        """
        Build required_variables from active configuration.

        Called in prepare_run(), after user configs have been set and before
        super().prepare_run() prepares/interpolates environment readers.
        """
        req = dict(self.BASE_REQUIRED_VARIABLES)

        dynamic_partitioning = bool(
            self.get_config('chemical:dynamic_partitioning')
        )
        degradation_enabled = bool(
            self.get_config('chemical:transformations:degradation')
        )
        degradation_mode = self.get_config(
            'chemical:transformations:degradation_mode'
        )
        volatilization_enabled = bool(
            self.get_config('chemical:transformations:volatilization')
        )
        do_dep = bool(
            self.get_config('chemical:sediment:enable_deposition')
        )
        do_res = bool(
            self.get_config('chemical:sediment:enable_resuspension')
        )
        sediment_exchange_enabled = do_dep or do_res
        include_wave_stress = bool(
            self.get_config('chemical:sediment:include_wave_stress')
        )
        include_other_stress = bool(
            self.get_config('chemical:sediment:include_other_stress')
        )
        stress_mode = self.get_config('chemical:sediment:stress_param_mode')
        resuspension_critstress_mode = self.get_config(
            'chemical:sediment:resuspension_critstress_mode'
        )
        resuspension_branch = self.get_config(
            'chemical:sediment:resuspension_critstress_branch'
        )
        # Dynamic partitioning needs SPM/DOC, fOC, pH, and mixed-layer depth.
        if dynamic_partitioning:
            req.update(self.PARTITIONING_REQUIRED_VARIABLES)
        # Single-rate degradation may need extra environmental fields.
        if degradation_enabled and degradation_mode == 'SingleRateConstants':
            if bool(self.get_config('chemical:transformations:Biodegradation')):
                req.update(self.BIODEGRADATION_REQUIRED_VARIABLES)

                sediment_oxygen_model = self.get_config('chemical:sediment:oxygen_model')

                if sediment_oxygen_model == 'PRESCRIBED_OPD':
                    req.update(self.SEDIMENT_OXYGEN_GEOMETRY_REQUIRED_VARIABLES)
                    req.update(self.SEDIMENT_OXYGEN_OPD_REQUIRED_VARIABLES)

                elif sediment_oxygen_model in ('ZERO_ORDER', 'ZERO_ORDER_DBL'):
                    req.update(self.SEDIMENT_OXYGEN_GEOMETRY_REQUIRED_VARIABLES)
                    req.update(self.SEDIMENT_OXYGEN_TRANSPORT_REQUIRED_VARIABLES)

                    oxygen_demand_mode = self.get_config('chemical:sediment:oxygen_demand_mode')
                    if oxygen_demand_mode == 'VOLUMETRIC_RATE':
                        req.update(self.SEDIMENT_OXYGEN_VOLUMETRIC_REQUIRED_VARIABLES)
                    else:
                        req.update(self.SEDIMENT_OXYGEN_FLUX_REQUIRED_VARIABLES)

                    if sediment_oxygen_model == 'ZERO_ORDER_DBL':
                        req.update(self.SEDIMENT_OXYGEN_DBL_REQUIRED_VARIABLES)

                elif sediment_oxygen_model == 'TWO_LAYER_DBL':
                    req.update(self.SEDIMENT_OXYGEN_GEOMETRY_REQUIRED_VARIABLES)
                    req.update(self.SEDIMENT_OXYGEN_TRANSPORT_REQUIRED_VARIABLES)
                    req.update(self.SEDIMENT_OXYGEN_DBL_REQUIRED_VARIABLES)
                    req.update(self.SEDIMENT_OXYGEN_TWO_LAYER_REQUIRED_VARIABLES)
            if bool(self.get_config('chemical:transformations:Photodegradation')):
                req.update(self.PHOTODEGRADATION_REQUIRED_VARIABLES)
            if bool(self.get_config('chemical:transformations:Hydrolysis')):
                req.update(self.HYDROLYSIS_REQUIRED_VARIABLES)
        if volatilization_enabled:
            req.update(self.VOLATILIZATION_REQUIRED_VARIABLES)
        if sediment_exchange_enabled:
            req.update(self.SEDIMENT_EXCHANGE_REQUIRED_VARIABLES)

            # Direct current bed stress is an optional preferred source.  Its
            # reader availability is resolved once in prepare_run(); only then is
            # it promoted into the runtime required-variable subset.  If no reader
            # supplies it, compute_bottom_shear_stress() uses the selected
            # calculated-current parameterization instead.
            if bool(getattr(self, '_direct_current_stress_reader_available', False)):
                req.update(self.DIRECT_CURRENT_STRESS_REQUIRED_VARIABLES)

            if stress_mode in ('LOG_Z0', 'GRAIN_D50'):
                req.update(self.SEDIMENT_BOTTOM_VELOCITY_REQUIRED_VARIABLES)
            if stress_mode == 'LOG_Z0':
                req.update(self.SEDIMENT_LOG_Z0_REQUIRED_VARIABLES)
            if stress_mode == 'GRAIN_D50':
                req.update(self.SEDIMENT_GRAIN_D50_REQUIRED_VARIABLES)
            if stress_mode in ('MANNING', 'CHEZY', 'WHITE_COLEBROOK'):
                req.update(self.SEDIMENT_BULK_FLOW_REQUIRED_VARIABLES)

            # WHITE_COLEBROOK uses configured nikuradse_ks for z0 internally, so no
            # sea_floor_roughness_length reader is needed unless wave stress needs
            # reader z0 through another mode.
            if do_res:
                req.update(self.SEDIMENT_RESUSPENSION_REQUIRED_VARIABLES)
                # Mapped d50 is useful for FROM_D50 or AUTO d50-based branch logic.
                # In USER mode, the model intentionally rejects mapped sea_floor_d50.
                if (
                    resuspension_critstress_mode == 'FROM_D50' or
                    resuspension_branch == 'AUTO'
                ):
                    req.update(self.SEDIMENT_GRAIN_D50_REQUIRED_VARIABLES)

            if include_other_stress:
                req.update(self.OTHER_STRESS_REQUIRED_VARIABLES)
            if include_wave_stress:
                source = self._wave_stress_source()

                if source == 'CALCULATED':
                    req.update(self.WAVE_STRESS_REQUIRED_VARIABLES)

                    wave_roughness = self._resolved_wave_roughness_mode()
                    if wave_roughness == 'LOG_Z0':
                        req.update(self.SEDIMENT_LOG_Z0_REQUIRED_VARIABLES)
                    elif wave_roughness == 'GRAIN_D50':
                        req.update(self.SEDIMENT_GRAIN_D50_REQUIRED_VARIABLES)

                else:  # DIRECT
                    req.update(self.DIRECT_WAVE_STRESS_REQUIRED_VARIABLES)
                    # Bathymetry/roughness may still be needed by current stress
                    # and sediment exchange, but are not required by DIRECT waves.
                combo = self.get_config(
                    'chemical:sediment:shear_stress_combination')

                if combo == 'SOULSBY_CLARKE':
                    # Wave direction is required for the directional wave-current
                    # combination, but either "to" or "from" direction is sufficient.
                    for name, spec in self.WAVE_DIRECTION_REQUIRED_VARIABLES.items():
                        if self._has_explicit_environment_source(name):
                            req[name] = dict(spec)
                    # Direct scalar current stress still needs a current direction.
                    req.update(self.SEDIMENT_BOTTOM_VELOCITY_REQUIRED_VARIABLES)
                    req.update(self.SEDIMENT_BULK_FLOW_REQUIRED_VARIABLES)

                # Missing mapped roughness must remain distinguishable from zero.
                if 'sea_floor_roughness_length' in req:
                    req['sea_floor_roughness_length'] = {
                        'fallback': np.nan,
                        'important': False,
                    }
        return req

    def _sync_required_variables_from_config(self):
        """
        Synchronize the configuration-dependent ChemicalDrift environmental
        requirements with OpenDrift's Environment object.

        ChemicalDrift keeps a class-level superset of environmental variables so
        OpenDrift can register all environment:constant:* and
        environment:fallback:* configuration keys during construction.

        Runtime configuration selects the actual subset. Because readers may have
        been added before this subset is finalized, rebuild the relevant parts of
        Environment.priority_list from the readers already attached to the model.
        """
        req = self._build_required_variables()

        self.required_variables = req

        if not hasattr(self, 'env'):
            return req

        self.env.required_variables = req

        # Keep profile/desired-variable metadata synchronized.
        self.env.required_profiles = [
            name for name, spec in req.items()
            if spec.get('profiles', False) is True
        ]

        self.env.desired_variables = [
            name for name, spec in req.items()
            if spec.get('important', True) is False
        ]

        self.required_profiles = list(self.env.required_profiles)
        self.desired_variables = list(self.env.desired_variables)

        # Remove priority entries that are no longer required.
        for variable in list(self.env.priority_list):
            if variable not in req:
                del self.env.priority_list[variable]

        # Re-register already attached non-lazy readers for variables that have
        # become required after configuration changes.
        for reader_name, reader in self.env.readers.items():
            if getattr(reader, 'is_lazy', False):
                continue

            reader_variables = set(getattr(reader, 'variables', []))

            for variable in req:
                if variable not in reader_variables:
                    continue

                if variable not in self.env.priority_list:
                    self.env.priority_list[variable] = [reader_name]
                elif reader_name not in self.env.priority_list[variable]:
                    self.env.priority_list[variable].append(reader_name)

        return req

    @staticmethod
    def _chemical_element_array_length(elements_obj):
        """Best-effort length of an OpenDrift element array.

        Empty element buffers are created during model initialization using the
        class-level ElementType. When optional Chemical variables are enabled
        later from configuration, these empty buffers must be rebuilt with the
        new concrete ElementType before delayed/scheduled particles are moved
        into them.
        """
        if elements_obj is None:
            return 0
        try:
            return len(elements_obj)
        except Exception:
            pass
        try:
            return int(len(elements_obj.lon))
        except Exception:
            return 0

    def _chemical_sync_element_buffer(self, attr_name, required_names):
        """Rebuild an empty element buffer if it lacks optional variables.

        This keeps self.elements and self.elements_scheduled consistent with
        self.ElementType after optional output settings are read from
        configuration. Non-empty buffers are never silently rebuilt, because
        that would drop already seeded particles.
        """
        elements_obj = getattr(self, attr_name, None)
        if elements_obj is None:
            return

        missing = [name for name in required_names if not hasattr(elements_obj, name)]
        if not missing:
            return

        n = self._chemical_element_array_length(elements_obj)
        if n == 0:
            setattr(self, attr_name, self.ElementType())
            logger.debug(
                'Rebuilt empty Chemical element buffer %s with optional variables: %s',
                attr_name, ', '.join(required_names)
            )
            return

        raise RuntimeError(
            f"Chemical element buffer '{attr_name}' already contains {n} element(s) "
            f"but is missing required optional variable(s): {missing}. "
            'Set chemical:transformations:Save_single_degr_mass, '
            'chemical:sediment:save_bed_interaction, and '
            'chemical:sediment:save_oxygen_diagnostics before any seeding, '
            'or restart the model.'
        )

    def _validate_sediment_oxygen_diagnostics_config(self):
        """Reject diagnostic configurations that would never be populated."""
        if not bool(self.get_config('chemical:sediment:save_oxygen_diagnostics')):
            return

        transfer_setup = self.get_config('chemical:transfer_setup')
        if transfer_setup not in ('organics', 'custom'):
            raise ValueError(
                'chemical:sediment:save_oxygen_diagnostics=True requires '
                "chemical:transfer_setup to be 'organics' or 'custom', because "
                'degradation() is otherwise not called by ChemicalDrift.update().'
            )

        if not bool(self.get_config('chemical:transformations:degradation')):
            raise ValueError(
                'chemical:sediment:save_oxygen_diagnostics=True requires '
                'chemical:transformations:degradation=True.'
            )
        if self.get_config('chemical:transformations:degradation_mode') != 'SingleRateConstants':
            raise ValueError(
                'chemical:sediment:save_oxygen_diagnostics=True requires '
                "chemical:transformations:degradation_mode='SingleRateConstants'."
            )
        if not bool(self.get_config('chemical:transformations:Biodegradation')):
            raise ValueError(
                'chemical:sediment:save_oxygen_diagnostics=True requires '
                'chemical:transformations:Biodegradation=True.'
            )

        k_aer = float(self.get_config(
            'chemical:transformations:k_DecayMax_water'
        ))
        k_ana = float(self.get_config(
            'chemical:transformations:k_Anaerobic_water'
        ))
        if k_aer <= 0.0 and k_ana <= 0.0:
            raise ValueError(
                'chemical:sediment:save_oxygen_diagnostics=True requires at '
                'least one positive biodegradation endpoint rate so the '
                'sediment-oxygen calculation is actually evaluated.'
            )

        # Also validate the model/demand selector combination before seeding.
        Chemical.sediment_oxygen_diagnostic_schema_key(
            self.get_config('chemical:sediment:oxygen_model'),
            self.get_config('chemical:sediment:oxygen_demand_mode'),
        )

    def _configure_element_type_from_config(self):
        save_single = bool(
            self.get_config('chemical:transformations:Save_single_degr_mass')
        )
        save_bed = bool(
            self.get_config('chemical:sediment:save_bed_interaction')
        )
        save_o2 = bool(
            self.get_config('chemical:sediment:save_oxygen_diagnostics')
        )

        oxygen_model = self.get_config('chemical:sediment:oxygen_model')
        oxygen_demand_mode = self.get_config(
            'chemical:sediment:oxygen_demand_mode'
        )

        if save_o2:
            self._validate_sediment_oxygen_diagnostics_config()
            o2_schema_key = Chemical.sediment_oxygen_diagnostic_schema_key(
                oxygen_model, oxygen_demand_mode
            )
            o2_names = Chemical.sediment_oxygen_diagnostic_variable_names(
                oxygen_model, oxygen_demand_mode
            )
        else:
            o2_schema_key = None
            o2_names = ()

        # Only configuration changes that alter the concrete element schema
        # participate in this key. For example, oxygen_demand_mode does not
        # change the schema for FIXED_FRACTION, PRESCRIBED_OPD, or TWO_LAYER_DBL.
        key = (save_single, save_bed, o2_schema_key)

        required_names = []
        if save_single:
            required_names.extend(Chemical.SINGLE_DEGRADATION_VARIABLE_NAMES)
        if save_bed:
            required_names.extend(Chemical.BED_INTERACTION_VARIABLE_NAMES)
        if save_o2:
            required_names.extend(o2_names)

        if getattr(self, '_chemical_element_type_key', None) != key:
            if hasattr(self, 'elements') and self.num_elements_active() > 0:
                raise RuntimeError(
                    'Cannot change optional Chemical element variables after seeding. '
                    'Set chemical:transformations:Save_single_degr_mass, '
                    'chemical:sediment:save_bed_interaction, '
                    'chemical:sediment:save_oxygen_diagnostics, oxygen_model, and '
                    'oxygen_demand_mode before seed_elements().'
                )

            self.ElementType = Chemical.make_element_type(
                save_single_degr_mass=save_single,
                save_bed_interaction=save_bed,
                save_sediment_oxygen_diagnostics=save_o2,
                sediment_oxygen_model=oxygen_model,
                sediment_oxygen_demand_mode=oxygen_demand_mode,
            )
            self._chemical_element_type_key = key

        # Important: OpenDrift may already have created empty active/scheduled
        # element arrays from the class-level default ElementType before runtime
        # configuration is applied. Rebuild only empty buffers so that delayed
        # release can move scheduled elements without AttributeError.
        if required_names:
            self._chemical_sync_element_buffer('elements', required_names)
            self._chemical_sync_element_buffer('elements_scheduled', required_names)
            self._chemical_sync_element_buffer('elements_deactivated', required_names)

    def init_species(self):
        """Initialize specie types and build the species list.

        For predefined transfer_setup values, species toggles are reset to a consistent
        state to avoid stale/leftover activated species from previous runs or manual
        configuration changes that are not compatible with the selected scheme.

        For transfer_setup == 'custom', species toggles must be set manually.
        """
        transfer_setup = self.get_config('chemical:transfer_setup')

        # Reset derived convenience flags
        self.set_config('chemical:slowly_fraction', False)
        self.set_config('chemical:irreversible_fraction', False)

        all_species_keys = [
            'LMM', 'LMMcation', 'LMManion',
            'Colloid', 'Humic_colloid', 'Polymer',
            'Particle_reversible', 'Particle_slowly_reversible', 'Particle_irreversible',
            'Sediment_reversible', 'Sediment_slowly_reversible', 'Sediment_buried', 'Sediment_irreversible',
        ]

        if transfer_setup != 'custom':
            # Preserve user choice for optional irreversible pools where supported
            keep_particle_irrev = self.get_config('chemical:species:Particle_irreversible')
            keep_sediment_irrev = self.get_config('chemical:species:Sediment_irreversible')

            # Reset all known species toggles first (prevents stale state)
            for key in all_species_keys:
                self.set_config(f'chemical:species:{key}', False)

            if transfer_setup == 'metals':
                self.set_config('chemical:species:LMM', True)
                self.set_config('chemical:species:Particle_reversible', True)
                self.set_config('chemical:species:Particle_slowly_reversible', True)
                self.set_config('chemical:species:Sediment_reversible', True)
                self.set_config('chemical:species:Sediment_slowly_reversible', True)
                self.set_config('chemical:species:Sediment_buried', True)

                # Optional irreversible pools (only if user enabled them)
                if keep_particle_irrev:
                    self.set_config('chemical:species:Particle_irreversible', True)
                if keep_sediment_irrev:
                    self.set_config('chemical:species:Sediment_irreversible', True)

            elif transfer_setup == '137Cs_rev':
                self.set_config('chemical:species:LMM', True)
                self.set_config('chemical:species:Particle_reversible', True)
                self.set_config('chemical:species:Sediment_reversible', True)

            elif transfer_setup == 'Sandnesfj_Al':
                self.set_config('chemical:species:LMMcation', True)
                self.set_config('chemical:species:LMManion', True)
                self.set_config('chemical:species:Humic_colloid', True)
                self.set_config('chemical:species:Polymer', True)
                self.set_config('chemical:species:Particle_reversible', True)
                self.set_config('chemical:species:Sediment_reversible', True)

            elif transfer_setup == 'organics':
                self.set_config('chemical:species:LMM', True)
                self.set_config('chemical:species:Humic_colloid', True)
                self.set_config('chemical:species:Particle_reversible', True)
                self.set_config('chemical:species:Particle_slowly_reversible', True)
                self.set_config('chemical:species:Sediment_reversible', True)
                self.set_config('chemical:species:Sediment_slowly_reversible', True)
                self.set_config('chemical:species:Sediment_buried', True)

                # Optional irreversible pools (only if user enabled them)
                if keep_particle_irrev:
                    self.set_config('chemical:species:Particle_irreversible', True)
                if keep_sediment_irrev:
                    self.set_config('chemical:species:Sediment_irreversible', True)

            else:
                logger.error('No valid transfer_setup {}'.format(transfer_setup))
        else:
            # Custom setup: species must be set manually by the user
            pass

        # Build species list in fixed order
        self.name_species=[]
        if self.get_config('chemical:species:LMM'):
            self.name_species.append('LMM')
        if self.get_config('chemical:species:LMMcation'):
            self.name_species.append('LMMcation')
        if self.get_config('chemical:species:LMManion'):
            self.name_species.append('LMManion')
        if self.get_config('chemical:species:Colloid'):
            self.name_species.append('Colloid')
        if self.get_config('chemical:species:Humic_colloid'):
            self.name_species.append('Humic colloid')
        if self.get_config('chemical:species:Polymer'):
            self.name_species.append('Polymer')
        if self.get_config('chemical:species:Particle_reversible'):
            self.name_species.append('Particle reversible')
        if self.get_config('chemical:species:Particle_slowly_reversible'):
            self.name_species.append('Particle slowly reversible')
        if self.get_config('chemical:species:Particle_irreversible'):
            self.name_species.append('Particle irreversible')
        if self.get_config('chemical:species:Sediment_reversible'):
            self.name_species.append('Sediment reversible')
        if self.get_config('chemical:species:Sediment_slowly_reversible'):
            self.name_species.append('Sediment slowly reversible')
        if self.get_config('chemical:species:Sediment_buried'):
            self.name_species.append('Sediment buried')
        if self.get_config('chemical:species:Sediment_irreversible'):
            self.name_species.append('Sediment irreversible')

        # Derived convenience flags (always recomputed to avoid stale state)
        self.set_config(
            'chemical:slowly_fraction',
            bool(self.get_config('chemical:species:Sediment_slowly_reversible') and
                 self.get_config('chemical:species:Particle_slowly_reversible')))
        self.set_config(
            'chemical:irreversible_fraction',
            bool(self.get_config('chemical:species:Sediment_irreversible') and
                 self.get_config('chemical:species:Particle_irreversible')))

        self.nspecies      = len(self.name_species)
        if self.nspecies == 0:
            raise ValueError("No active species configured for chemical:transfer_setup='custom'.")

    def _build_transition_destination_cache(self):
        """
        Cache possible destination species for each source species.

        This assumes update_transfer_rates() modifies rates for existing transitions
        but does not create entirely new destination columns that are zero in
        self.transfer_rates. That matches the intended structure: init_transfer_rates()
        defines the topology, update_transfer_rates() applies local corrections.
        """
        rates = np.asarray(self.transfer_rates)

        if rates.ndim == 3:
            # Sandnesfj_Al: shape = salinity_bin x source x destination
            nonzero = np.any(rates != 0.0, axis=0)   # source x destination
        else:
            nonzero = rates != 0.0                   # source x destination

        self._transition_destinations = [
            np.flatnonzero(nonzero[src]).astype(np.int32)
            for src in range(self.nspecies)
        ]

    def init_transfer_rates(self):
        ''' Initialize the background species-to-species transfer-rate matrix.
        This routine builds self.transfer_rates as the baseline transition-rate tensor
        used by update_transfer_rates() and update_partitioning().

        1) Clear any stale cached species indices from previous setups.
        2) Allocate:
               transfer_rates[i, j]      = background rate from species i to species j [1/s]
               ntransformations[i, j]    = cumulative realized transitions i -> j
        3) Populate the matrix according to chemical:transfer_setup:
               - 'organics'
               - 'metals'
               - '137Cs_rev'
               - 'custom'
               - 'Sandnesfj_Al'
        4) Enforce zero diagonals so self-transitions are impossible:
               transfer_rates[i, i] = 0

        Organics setup
        --------------
        For organics, the matrix represents reversible exchange among:
            dissolved (LMM), humic colloid / DOM, suspended particles, sediments,
            slowly reversible pools, buried sediment, and optional irreversible pools.
        Main steps are:
        1) Build phase-specific partition coefficients
           - KOC_sed, KOC_SPM, KOC_DOM [L/kgOC]
           - optionally pH-dependent using speciation-weighted KOC
           - convert to:
                 Kd_sed = KOC_sed * fOC_sed
                 Kd_SPM = KOC_SPM * fOC_SPM
                 Kd_DOM = KOC_DOM * Org2C
        2) Compute adsorption/desorption constants
           - adsorption: k_ads = 33.3 / 3600     [L kg-1 s-1]
           - desorption: k_des = k_ads / Kd      [1/s]
        3) Apply background temperature and salinity normalization
           - background matrix entries are stored at representative Tref/Sref conditions
           - local corrections are later applied in update_transfer_rates()
        4) Fill matrix entries such as:
           - dissolved -> DOM: k12 = k_ads * concDOM
           - DOM -> dissolved: k21 = k_des_DOM / TcorrDOM / Scorr
           - dissolved -> SPM: k13 = k_ads * concSPM
           - SPM -> dissolved: k31 = k_des_SPM / TcorrSed / Scorr
           - dissolved -> sediment: k14 = (k_ads * 1e-3) * sed_L * sed_dens * (1-poro) * sed_phi / sed_H
           - sediment -> dissolved: k41 = k_des_sed * sed_phi / TcorrSed / Scorr
        5) Add optional slowly reversible and buried-sediment exchanges
           - reversible <-> slowly reversible
           - active sediment -> buried sediment
           - buried sediment -> active sediment pools

        Metals / 137Cs / custom / Sandnesfj_Al
        --------------------------------------
        These setups use fixed empirical transfer schemes specific to each model family.
        Typical forms are:
            dissolved -> particle  ~ Dc * Kd * concSPM
            particle  -> dissolved ~ Dc
            dissolved -> sediment  ~ Dc * Kd * sediment_inventory / layer_thickness
            sediment  -> dissolved ~ Dc * sediment_correction
        The Sandnesfj_Al setup stores a salinity-binned 3D rate tensor:
            transfer_rates[salinity_bin, i, j]
        '''
        transfer_setup=self.get_config('chemical:transfer_setup')
        irreversible_rate = self.get_config('chemical:transformations:irreversible_rate')
        logger.info( 'transfer setup: %s' % transfer_setup)
        # Clear any stale species indices from previous runs / setups
        for _attr in (
            'num_lmm', 'num_lmmcation', 'num_lmmanion',
            'num_col', 'num_humcol', 'num_polymer',
            'num_prev', 'num_psrev', 'num_pirrev',
            'num_srev', 'num_ssrev', 'num_sirrev',
            'num_sburied',
        ):
            if hasattr(self, _attr):
                delattr(self, _attr)

        self.transfer_rates = np.zeros((self.nspecies, self.nspecies), dtype=np.float32,)
        self.ntransformations = np.zeros((self.nspecies, self.nspecies), dtype=np.int64,)

        if transfer_setup == 'organics':
            self.num_lmm    = self.specie_name2num('LMM')
            self.num_humcol = self.specie_name2num('Humic colloid')
            self.num_prev   = self.specie_name2num('Particle reversible')
            self.num_srev   = self.specie_name2num('Sediment reversible')
            if self.get_config('chemical:species:Particle_slowly_reversible'):
                self.num_psrev  = self.specie_name2num('Particle slowly reversible')
            self.num_ssrev  = self.specie_name2num('Sediment slowly reversible')
            if self.get_config('chemical:species:Sediment_buried'):
                self.num_sburied = self.specie_name2num('Sediment buried')
            # Optional irreversible compartments (if enabled)
            if self.get_config('chemical:species:Sediment_irreversible'):
                self.num_sirrev = self.specie_name2num('Sediment irreversible')
            if self.get_config('chemical:species:Particle_irreversible'):
                self.num_pirrev = self.specie_name2num('Particle irreversible')

            # Values from EMERGE-Aquatox
            Org2C      = 0.526  # kgOC/KgOM
            #Kd         = self.get_config('chemical:transformations:Kd')
            KOW        = 10**self.get_config('chemical:transformations:LogKOW')
            KOWTref    = self.get_config('chemical:transformations:TrefKOW')
            DH_KOC_Sed = self.get_config('chemical:transformations:DeltaH_KOC_Sed')
            DH_KOC_DOM = self.get_config('chemical:transformations:DeltaH_KOC_DOM')
            Setchenow  = self.get_config('chemical:transformations:Setchenow')

            diss       = self.get_config('chemical:transformations:dissociation')
            pKa_acid   = self.get_config('chemical:transformations:pKa_acid')
            pKa_base   = self.get_config('chemical:transformations:pKa_base')
            if diss in ["acid", "amphoteric"] and pKa_acid < 0:
                raise ValueError("pKa_acid must be positive")
            if diss in ["base", "amphoteric"] and pKa_base < 0:
                raise ValueError("pKa_base must be positive")
            if diss == "amphoteric" and abs(pKa_acid - pKa_base) < 2:
                raise ValueError("pKa_base and pKa_acid must differ of at least two units")

            # NOTE: pKa_base is pKa of conjugate acid BH+ (base speciation: BH+ <-> B + H+)
            # representative pH for background (updated later per-particle in update_transfer_rates)
            pH_water   = 8.1
            pH_sed     = 6.9

            fOC_SPM    = self.get_config('chemical:transformations:fOC_SPM')       # typical values from 0.01 to 0.1 gOC/g
            fOC_sed    = self.get_config('chemical:transformations:fOC_sed')       # typical values from 0.01 to 0.1 gOC/g

            # environmental / sediment constants
            # Values from Simonsen et al (2019a)
            slow_coeff_ads = self.get_config('chemical:transformations:slow_coeff_ads')
            slow_coeff_des = self.get_config('chemical:transformations:slow_coeff_des')
            concSPM     = 50.e-3                                                # available SPM (kg/m3)
            sed_L       = self.get_config('chemical:sediment:mixing_depth')     # sediment mixing depth (m)
            sed_dens    = self.get_config('chemical:sediment:density')          # default particle density (kg/m3)
            sed_phi     = self.get_config('chemical:sediment:corr_factor')      # sediment correction factor
            sed_poro    = self.get_config('chemical:sediment:porosity')         # sediment porosity
            sed_H       = self.get_config('chemical:sediment:layer_thickness')  # thickness of seabed interaction layer (m)
            sed_burial  = self.get_config('chemical:sediment:burial_rate')      # sediment burial rate (m/y)
            sed_leaking_rate = self.get_config( 'chemical:sediment:buried_leaking_rate')
            concDOM   = 1.e-3 / Org2C    # concentration of available dissolved organic matter (kg/m3)
                                         # rough initial estimate for coastal waters, doi: 10.1002/lom3.10118
            #concDOM   = 50.e-3          # HIGHER VALUE FOR TESTING!!!!!!!!!!!!

            if diss == "nondiss":
                # direct KOC inputs or fallback estimates

                KOC_sed = self.get_config("chemical:transformations:KOC_sed")
                if KOC_sed < 0:
                    KOC_sed = 2.62 * KOW ** 0.82     # (L/kgOC), Park and Clough, 2014
                    #KOC_Sed    = 1.26 * kOW**0.81   # (L/KgOC), Ragas et al., 2019

                KOC_SPM = KOC_sed

                KOC_DOM = self.get_config("chemical:transformations:KOC_DOM")
                if KOC_DOM < 0:
                    KOC_DOM = 2.88 * KOW ** 0.67  # (L/kgOC), Park and Clough, 2014
            else:
                # Sediment KOC components (L/kgOC)
                # (Franco et al., 2008) https://doi.org/10.1897/07-583.1 (Table 3)
                KOC_sed_n = self.get_config("chemical:transformations:KOC_sed")
                if KOC_sed_n < 0:
                    if diss == 'acid':
                        KOC_sed_n = 10 ** ((0.54 * np.log10(KOW)) + 1.11)
                    elif diss == 'base':
                        KOC_sed_n = 10 ** ((0.37 * np.log10(KOW)) + 1.70)
                    elif diss == 'amphoteric':
                        KOC_sed_n = 10 ** ((0.50 * np.log10(KOW)) + 1.13)

                # anion / cation KOC (keep as neutral if not needed)
                KOC_sed_acid = self.get_config("chemical:transformations:KOC_sed_acid")
                if KOC_sed_acid < 0:
                        KOC_sed_acid = 10 ** (0.11 * np.log10(KOW) + 1.54)

                KOC_sed_base = self.get_config("chemical:transformations:KOC_sed_base")
                if KOC_sed_base < 0:
                    KOC_sed_base = 10.0 ** ((pKa_base ** 0.65) * ((KOW / (KOW + 1.0)) ** 0.14)) # Franco et al. (2008)  https://doi.org/10.1897/07-583.1

                # DOM KOC components (L/kgOC)
                # Kd_DOM = KOC_DOM * Org2C,
                KOC_DOM_n = self.get_config("chemical:transformations:KOC_DOM")
                if KOC_DOM_n < 0:
                    KOC_DOM_n = (0.08 * KOW) / Org2C   # from DOC to DOM, Burkhard L.P. (2000) https://doi.org/10.1021/es001269l
                    # KOC_DOM_n   = 2.88 * KOW**0.67   # (L/KgOC), Park and Clough, 2014

                KOC_DOM_acid = self.get_config("chemical:transformations:KOC_DOM_acid")
                if KOC_DOM_acid < 0:
                    KOC_DOM_acid = (0.08 * 10 ** (np.log10(KOW) - 3.5)) / Org2C  # KOC_DOC/Org2C, Trapp, S., Horobin, R.W., (2005) https://doi.org/ 10.1007/s00249-005-0472-1

                KOC_DOM_base = self.get_config("chemical:transformations:KOC_DOM_base")
                if KOC_DOM_base < 0:
                    KOC_DOM_base = (0.08 * 10 ** (np.log10(KOW) - 3.5)) / Org2C  # KOC_DOC/Org2C, Trapp, S., Horobin, R.W., (2005) https://doi.org/ 10.1007/s00249-005-0472-1

                # Background KOC for each phase at representative pH
                # SPM uses sediment KOC components by assumption.
                KOC_SPM = self.koc_updated(KOC_sed_n, KOC_sed_acid, KOC_sed_base,
                                       pH=pH_water, diss=diss, pKa_acid=pKa_acid, pKa_base=pKa_base)

                KOC_DOM = self.koc_updated(KOC_DOM_n, KOC_DOM_acid, KOC_DOM_base,
                                       pH=pH_water, diss=diss, pKa_acid=pKa_acid, pKa_base=pKa_base)

                KOC_sed = self.koc_updated(KOC_sed_n, KOC_sed_acid, KOC_sed_base,
                                       pH=pH_sed, diss=diss, pKa_acid=pKa_acid, pKa_base=pKa_base)



            # Convert to Kd (L/kg) using fOC (sed/SPM) and Org2C (DOM)
            #KOM_sed = KOC_sed * Org2C #  L/KgOC * KgOC/KgOM = L/KgOM
            #KOM_SPM = KOC_sed * Org2C #  L/KgOC * KgOC/KgOM = L/KgOM
            #KOM_DOM = KOC_DOM * Org2C #  L/KgOC * KgOC/KgOM = L/KgOM

            # to be calculated separately for sed, SPM, dom (different KOC, pH, fOC)
            self.Kd_sed = Kd_sed = KOC_sed * fOC_sed    # L/KgOC * KgOC/KG = L/Kg
            self.Kd_SPM = Kd_SPM = KOC_SPM * fOC_SPM    # L/KgOC * KgOC/KG = L/Kg
            self.Kd_DOM = Kd_DOM = KOC_DOM * Org2C      # L/KgOC * KgOC/KgOM * 1KgOM/Kg = L/Kg (=KOM_DOM)
            # TODO Use setconfig() to store these?

            logger.info("transfer setup: %s", transfer_setup)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Partitioning coefficients (Tref,freshwater)')
                logger.debug("KOC_sed: %s L/KgOC", KOC_sed)
                logger.debug("KOC_SPM: %s L/KgOC", KOC_SPM)
                logger.debug("KOC_DOM: %s L/KgOC", KOC_DOM)
                logger.debug("Kd_sed: %s L/Kg", Kd_sed)
                logger.debug("Kd_SPM: %s L/Kg", Kd_SPM)
                logger.debug("Kd_DOM: %s L/Kg", Kd_DOM)

            # Base adsorption/desorption rates and background corrections
            # From Karickhoff and Morris 1985
            k_ads = 33.3 / (60*60) # L/(Kg*s) = 33 L/(kgOM*h)

            k_des_sed = k_ads / Kd_sed # 1/s
            k_des_SPM = k_ads / Kd_SPM # 1/s
            k_des_DOM = k_ads / Kd_DOM # 1/s

            # Default corrections, assuming temperature 25 salinity 35
            TcorrSed = self.tempcorr("Arrhenius",DH_KOC_Sed,25,KOWTref)
            TcorrDOM = self.tempcorr("Arrhenius",DH_KOC_DOM,25,KOWTref)
            Scorr    = self.salinitycorr(Setchenow,KOWTref,35)

            concSPM = concSPM * 1e-3 # (Kg/L)
            concDOM = concDOM * 1e-3 # (Kg/L)

            self.k_ads = k_ads
            self.k21_0 = k_des_DOM
            self.k31_0 = k_des_SPM
            self.k41_0 = k_des_sed * sed_phi
            self.k64_0 = slow_coeff_des
            self.k53_0 = slow_coeff_des
            # self.k46_0 = slow_coeff_ads
            # self.k35_0 = slow_coeff_ads

            # TODO Use setconfig() to store these?

            # Fill transfer matrix (background at Tref, Sref)
            #   Dissolved  <->  Humic colloid
            self.transfer_rates[self.num_lmm,self.num_humcol] = k_ads * concDOM             # k12
            self.transfer_rates[self.num_humcol,self.num_lmm] = k_des_DOM / TcorrDOM / Scorr# k21

            #   Dissolved  <->  Particle reversible
            self.transfer_rates[self.num_lmm,self.num_prev] = k_ads * concSPM               # k13
            self.transfer_rates[self.num_prev,self.num_lmm] = k_des_SPM / TcorrSed / Scorr  # k31

            #   Dissolved  <->  Sediment reversible
            self.transfer_rates[self.num_lmm,self.num_srev] = \
                (k_ads *1e-3) * sed_L * sed_dens * (1.-sed_poro) * sed_phi / sed_H          # k14 # *1e-3, from L to m3, so k14 is 1/s
            self.transfer_rates[self.num_srev,self.num_lmm] = \
                k_des_sed * sed_phi / TcorrSed / Scorr                                      # k41

            #   Sediment reversible  <->  Sediment slowly reversible
            self.transfer_rates[self.num_srev,self.num_ssrev] = slow_coeff_ads                     # k46
            self.transfer_rates[self.num_ssrev,self.num_srev] = slow_coeff_des / TcorrSed / Scorr  # k64

            #   Particle reversible  <->  Particle slowly reversible
            if hasattr(self, 'num_psrev'):
                self.transfer_rates[self.num_prev,self.num_psrev] = slow_coeff_ads                     # k35
                self.transfer_rates[self.num_psrev,self.num_prev] = slow_coeff_des / TcorrSed / Scorr  # k53

            #   Reversible -> irreversible
            if hasattr(self, 'num_pirrev'):
                self.transfer_rates[self.num_prev, self.num_pirrev] = irreversible_rate
            if hasattr(self, 'num_sirrev'):
                self.transfer_rates[self.num_srev, self.num_sirrev] = irreversible_rate

            # Burial/leaking using dedicated sediment buried species
            # (m/y) / m / (s/y) = 1/s
            if hasattr(self, 'num_sburied'):
                burial_rate = sed_burial / sed_L / 31556926.0
                self.transfer_rates[self.num_srev, self.num_sburied] = burial_rate             # k47
                self.transfer_rates[self.num_ssrev, self.num_sburied] = burial_rate            # k57
                # Also bury irreversible sediment pool if present
                if hasattr(self, 'num_sirrev'):
                    self.transfer_rates[self.num_sirrev, self.num_sburied] = burial_rate

                leak_frac_to_ssrev = self.get_config('chemical:sediment:buried_leak_to_ssrev_fraction')
                leak_frac_to_ssrev = np.clip(leak_frac_to_ssrev, 0.0, 1.0)
                self.transfer_rates[self.num_sburied, self.num_srev] = sed_leaking_rate * (1.0 - leak_frac_to_ssrev)  # k74
                self.transfer_rates[self.num_sburied, self.num_ssrev] = sed_leaking_rate * leak_frac_to_ssrev         # k75

            #   Humic colloid   <->  Particle reversible (aggregation of DOC)
            self.transfer_rates[self.num_humcol,self.num_prev] = self.get_config('chemical:transformations:aggregation_rate')
            self.transfer_rates[self.num_prev,self.num_humcol] = 0

        elif transfer_setup == 'metals':    # renamed from radionuclides Bokna_137Cs

            self.num_lmm    = self.specie_name2num('LMM')
            self.num_prev   = self.specie_name2num('Particle reversible')
            self.num_srev   = self.specie_name2num('Sediment reversible')
            self.num_psrev  = self.specie_name2num('Particle slowly reversible')
            self.num_ssrev  = self.specie_name2num('Sediment slowly reversible')
            if self.get_config('chemical:species:Sediment_buried'):
                self.num_sburied = self.specie_name2num('Sediment buried')
            # Optional irreversible compartments (if enabled)
            if self.get_config('chemical:species:Sediment_irreversible'):
                self.num_sirrev = self.specie_name2num('Sediment irreversible')
            if self.get_config('chemical:species:Particle_irreversible'):
                self.num_pirrev = self.specie_name2num('Particle irreversible')

            # Values from Simonsen et al (2019a)
            Kd         = self.get_config('chemical:transformations:Kd')          # (m3/Kg)
            Dc         = self.get_config('chemical:transformations:Dc')          # (1/s)
            slow_coeff = self.get_config('chemical:transformations:slow_coeff')
            concSPM    = 1.e-3   # concentration of available suspended particulate matter (kg/m3)
            sed_L = self.get_config('chemical:sediment:mixing_depth')            # sediment mixing depth (m)
            sed_dens =  self.get_config('chemical:sediment:density')             # default particle density (kg/m3)
            sed_f           =  self.get_config('chemical:sediment:effective_fraction')      # fraction of effective sorbents
            sed_phi         =  self.get_config('chemical:sediment:corr_factor')   # sediment correction factor
            sed_poro        =  self.get_config('chemical:sediment:porosity')      # sediment porosity
            sed_H =  self.get_config('chemical:sediment:layer_thickness')         # thickness of seabed interaction layer (m)

            #self.k_ads = Dc * Kd * 1e3 # L/(Kg*s)
            self.transfer_rates[self.num_lmm,self.num_prev] = Dc * Kd * concSPM
            self.transfer_rates[self.num_prev,self.num_lmm] = Dc
            self.transfer_rates[self.num_lmm,self.num_srev] = \
                Dc * Kd * sed_L * sed_dens * (1.-sed_poro) * sed_f * sed_phi / sed_H
            self.transfer_rates[self.num_srev,self.num_lmm] = Dc * sed_phi

            # Slow reversible sediment partitioning
            self.transfer_rates[self.num_srev, self.num_ssrev] = slow_coeff
            self.transfer_rates[self.num_ssrev, self.num_srev] = slow_coeff * 0.1

            #   Reversible -> irreversible
            if hasattr(self, 'num_pirrev'):
                self.transfer_rates[self.num_prev, self.num_pirrev] = irreversible_rate
            if hasattr(self, 'num_sirrev'):
                self.transfer_rates[self.num_srev, self.num_sirrev] = irreversible_rate

            # Burial/leaking to/from buried-sediment species
            if hasattr(self, 'num_sburied'):
                sed_burial = self.get_config('chemical:sediment:burial_rate')
                sed_leaking_rate = self.get_config('chemical:sediment:buried_leaking_rate')
                burial_rate = sed_burial / sed_L / 31556926.0
                self.transfer_rates[self.num_srev, self.num_sburied] = burial_rate
                self.transfer_rates[self.num_ssrev, self.num_sburied] = burial_rate
                # Also bury irreversible sediment pool if present
                if hasattr(self, 'num_sirrev'):
                    self.transfer_rates[self.num_sirrev, self.num_sburied] = burial_rate

                leak_frac_to_ssrev = self.get_config('chemical:sediment:buried_leak_to_ssrev_fraction')
                leak_frac_to_ssrev = np.clip(leak_frac_to_ssrev, 0.0, 1.0)
                self.transfer_rates[self.num_sburied, self.num_srev] = sed_leaking_rate * (1.0 - leak_frac_to_ssrev)
                self.transfer_rates[self.num_sburied, self.num_ssrev] = sed_leaking_rate * leak_frac_to_ssrev

            self.transfer_rates[self.num_prev,self.num_psrev] = slow_coeff
            self.transfer_rates[self.num_psrev,self.num_prev] = slow_coeff*.1

        elif transfer_setup == '137Cs_rev':

            self.num_lmm    = self.specie_name2num('LMM')
            self.num_prev   = self.specie_name2num('Particle reversible')
            self.num_srev   = self.specie_name2num('Sediment reversible')

            # Simpler version of Values from Simonsen et al (2019a)
            # Only consider the reversible fraction
            Kd         = self.get_config('chemical:transformations:Kd')
            Dc         = self.get_config('chemical:transformations:Dc')
            concSPM    = 1.e-3   # concentration of available suspended particulate matter (kg/m3)
            sed_L           = self.get_config('chemical:sediment:mixing_depth')# sediment mixing depth (m)
            sed_dens        = self.get_config('chemical:sediment:density')     # default particle density (kg/m3)
            sed_f           = self.get_config('chemical:sediment:effective_fraction') # fraction of effective sorbents
            sed_phi         = self.get_config('chemical:sediment:corr_factor') # sediment correction factor
            sed_poro        = self.get_config('chemical:sediment:porosity')    # sediment porosity
            sed_H =  self.get_config('chemical:sediment:layer_thickness')      # thickness of seabed interaction layer (m)

            self.transfer_rates[self.num_lmm,self.num_prev] = Dc * Kd * concSPM
            self.transfer_rates[self.num_prev,self.num_lmm] = Dc
            self.transfer_rates[self.num_lmm,self.num_srev] = \
                Dc * Kd * sed_L * sed_dens * (1.-sed_poro) * sed_f * sed_phi / sed_H
            self.transfer_rates[self.num_srev,self.num_lmm] = Dc * sed_phi

        elif transfer_setup=='custom':
        # Set of custom values for testing/development
            if not self.get_config('chemical:species:LMM'):
                raise ValueError("custom transfer_setup currently requires chemical:species:LMM = True")

            self.num_lmm   = self.specie_name2num('LMM')
            if self.get_config('chemical:species:Colloid'):
                self.num_col = self.specie_name2num('Colloid')
            if self.get_config('chemical:species:Particle_reversible'):
                self.num_prev  = self.specie_name2num('Particle reversible')
            if self.get_config('chemical:species:Sediment_reversible'):
                self.num_srev  = self.specie_name2num('Sediment reversible')
            # Index optional species if present (independent toggles)
            if self.get_config('chemical:species:Particle_slowly_reversible'):
                self.num_psrev  = self.specie_name2num('Particle slowly reversible')
            if self.get_config('chemical:species:Sediment_slowly_reversible'):
                self.num_ssrev  = self.specie_name2num('Sediment slowly reversible')
            if self.get_config('chemical:species:Particle_irreversible'):
                self.num_pirrev  = self.specie_name2num('Particle irreversible')
            if self.get_config('chemical:species:Sediment_irreversible'):
                self.num_sirrev  = self.specie_name2num('Sediment irreversible')
            if self.get_config('chemical:species:Sediment_buried'):
                self.num_sburied = self.specie_name2num('Sediment buried')

            if self.get_config('chemical:species:Particle_reversible'):
                self.transfer_rates[self.num_lmm,self.num_prev] = 5.e-6 #*0.
                self.transfer_rates[self.num_prev,self.num_lmm] = \
                    self.get_config('chemical:transformations:Dc')
            if self.get_config('chemical:species:Sediment_reversible'):
                self.transfer_rates[self.num_lmm,self.num_srev] = 1.e-5 #*0.
                self.transfer_rates[self.num_srev,self.num_lmm] = \
                    self.get_config('chemical:transformations:Dc') * self.get_config('chemical:sediment:corr_factor')
                # self.transfer_rates[self.num_srev,self.num_lmm] = 5.e-6

            # Slow reversible partitioning
            if hasattr(self, 'num_psrev'):
                self.transfer_rates[self.num_prev, self.num_psrev] = 2.e-6
                self.transfer_rates[self.num_psrev, self.num_prev] = 2.e-7
            if hasattr(self, 'num_ssrev'):
                self.transfer_rates[self.num_srev, self.num_ssrev] = 2.e-6
                self.transfer_rates[self.num_ssrev, self.num_srev] = 2.e-7

            #   Reversible -> irreversible
            if hasattr(self, 'num_pirrev'):
                self.transfer_rates[self.num_prev, self.num_pirrev] = irreversible_rate
            if hasattr(self, 'num_sirrev'):
                self.transfer_rates[self.num_srev, self.num_sirrev] = irreversible_rate

            # Burial/leaking using dedicated buried-sediment species (optional for custom setup)
            if hasattr(self, 'num_sburied'):
                sed_L = self.get_config('chemical:sediment:mixing_depth')  # m
                sed_burial = self.get_config('chemical:sediment:burial_rate')  # m/y
                sed_leaking_rate = self.get_config('chemical:sediment:buried_leaking_rate')  # 1/s
                if sed_L > 0:
                    burial_rate = sed_burial / sed_L / 31556926.0  # 1/s
                    # Bury from all sediment pools present (srev, ssrev, sirrev)
                    if hasattr(self, 'num_srev'):
                        self.transfer_rates[self.num_srev, self.num_sburied] = burial_rate
                    if hasattr(self, 'num_ssrev'):
                        self.transfer_rates[self.num_ssrev, self.num_sburied] = burial_rate
                    if hasattr(self, 'num_sirrev'):
                        self.transfer_rates[self.num_sirrev, self.num_sburied] = burial_rate

                    leak_frac_to_ssrev = self.get_config('chemical:sediment:buried_leak_to_ssrev_fraction')
                    leak_frac_to_ssrev = np.clip(leak_frac_to_ssrev, 0.0, 1.0)

                    # Leak back only to srev and (if present) ssrev
                    if hasattr(self, 'num_srev'):
                        if hasattr(self, 'num_ssrev'):
                            self.transfer_rates[self.num_sburied, self.num_srev] = sed_leaking_rate * (1.0 - leak_frac_to_ssrev)
                            self.transfer_rates[self.num_sburied, self.num_ssrev] = sed_leaking_rate * leak_frac_to_ssrev
                        else:
                            self.transfer_rates[self.num_sburied, self.num_srev] = sed_leaking_rate

        elif transfer_setup=='Sandnesfj_Al':
            # Use values from Simonsen et al (2019b)
            self.num_lmmanion    = self.specie_name2num('LMManion')
            self.num_lmmcation   = self.specie_name2num('LMMcation')
            self.num_humcol      = self.specie_name2num('Humic colloid')
            self.num_polymer     = self.specie_name2num('Polymer')
            self.num_prev        = self.specie_name2num('Particle reversible')
            self.num_srev        = self.specie_name2num('Sediment reversible')

            Dc         = self.get_config('chemical:transformations:Dc')

            self.salinity_intervals = [0,1,10,20]

            # Resize transfer rates array
            self.transfer_rates = np.zeros([len(self.salinity_intervals),self.transfer_rates.shape[0],self.transfer_rates.shape[1]])

            # Salinity interval 0-1 psu
            self.transfer_rates[0,self.num_lmmcation, self.num_humcol]    = 1.2e-5
            self.transfer_rates[0,self.num_lmmcation, self.num_prev]      = 4.e-6
            self.transfer_rates[0,self.num_humcol,    self.num_lmmcation] = .3*Dc
            self.transfer_rates[0,self.num_humcol,    self.num_prev]      = 2.e-6
            self.transfer_rates[0,self.num_prev,      self.num_lmmcation] = .3*Dc
            self.transfer_rates[0,self.num_srev,      self.num_lmmcation] = .03*Dc

            # Salinity interval 1-10 psu
            self.transfer_rates[1,self.num_lmmcation, self.num_humcol]    = 1.e-5
            self.transfer_rates[1,self.num_lmmcation, self.num_prev]      = 3.e-6
            self.transfer_rates[1,self.num_lmmcation, self.num_polymer]   = 1.2e-4
            self.transfer_rates[1,self.num_humcol,    self.num_lmmcation] = 7.*Dc
            self.transfer_rates[1,self.num_humcol,    self.num_prev]      = 4.e-6
            self.transfer_rates[1,self.num_prev,      self.num_lmmcation] = .5*Dc
            self.transfer_rates[1,self.num_srev,      self.num_lmmcation] = .05*Dc
            self.transfer_rates[1,self.num_lmmanion,  self.num_polymer]   = 5.e-6
            self.transfer_rates[1,self.num_polymer,   self.num_lmmanion]  = 12.*Dc
            self.transfer_rates[1,self.num_polymer,   self.num_prev]      = 2.4e-5

            # Salinity interval 10-20 psu
            self.transfer_rates[2,self.num_lmmcation, self.num_humcol]    = 8.e-6
            self.transfer_rates[2,self.num_lmmcation, self.num_prev]      = 2.e-6
            self.transfer_rates[2,self.num_lmmcation, self.num_polymer]   = 1.4e-4
            self.transfer_rates[2,self.num_humcol,    self.num_lmmcation] = 7.*Dc
            self.transfer_rates[2,self.num_humcol,    self.num_prev]      = 6.e-6
            self.transfer_rates[2,self.num_prev,      self.num_lmmcation] = .6*Dc
            self.transfer_rates[2,self.num_srev,      self.num_lmmcation] = .06*Dc
            self.transfer_rates[2,self.num_lmmanion,  self.num_polymer]   = 5.e-6
            self.transfer_rates[2,self.num_polymer,   self.num_lmmanion]  = 12.*Dc
            self.transfer_rates[2,self.num_polymer,   self.num_prev]      = 6.e-5

            # Salinity interval >20 psu
            self.transfer_rates[3,self.num_lmmcation, self.num_humcol]    = 6.e-6
            self.transfer_rates[3,self.num_lmmcation, self.num_prev]      = 1.8e-6
            self.transfer_rates[3,self.num_lmmcation, self.num_polymer]   = 1.5e-4
            self.transfer_rates[3,self.num_humcol,    self.num_lmmcation] = 7.*Dc
            self.transfer_rates[3,self.num_humcol,    self.num_prev]      = 1.e-5
            self.transfer_rates[3,self.num_prev,      self.num_lmmcation] = .8*Dc
            self.transfer_rates[3,self.num_srev,      self.num_lmmcation] = .08*Dc
            self.transfer_rates[3,self.num_lmmanion,  self.num_polymer]   = 5.e-6
            self.transfer_rates[3,self.num_polymer,   self.num_lmmanion]  = 12.*Dc
            self.transfer_rates[3,self.num_polymer,   self.num_prev]      = 8.e-5

        else:
            logger.error('No transfer setup available')

        # Set diagonal to 0. (not possible to transform to present specie)
        if len(self.transfer_rates.shape) == 3:
            for ii in range(self.transfer_rates.shape[0]):
                np.fill_diagonal(self.transfer_rates[ii,:,:],0.)
        else:
            np.fill_diagonal(self.transfer_rates,0.)

        self._build_transition_destination_cache()

        ## HACK :
        # self.transfer_rates[:] = 0.
        # print ('\n ###### \n IMPORTANT:: \n transfer rates have been hacked! \n#### \n ')
        logger.debug('nspecies: %s' % self.nspecies)
        logger.debug("Transfer rates:\n%s", self.transfer_rates)

    def _store_model_state_for_export(self):
        """Store ChemicalDrift model-level state using collision-safe dimensions.
          * ordinary transfer matrices keep the historical
            (specie_0, specie_1) dimensions for backward compatibility;
          * Sandnesfj_Al uses a dedicated salinity_interval axis, preventing
            the 4-bin salinity dimension from colliding with nspecies;
          * ntransformations always uses (specie_0, specie_1).
        """
        nspecies = int(self.nspecies)

        transfer_rates = np.asarray(self.transfer_rates)
        if transfer_rates.ndim == 2:
            expected = (nspecies, nspecies)
            if transfer_rates.shape != expected:
                raise ValueError(
                    'transfer_rates has incompatible shape for export: '
                    f'{transfer_rates.shape}; expected {expected}.')
            self.result['transfer_rates'] = (
                ('specie_0', 'specie_1'), transfer_rates)

        elif transfer_rates.ndim == 3:
            if transfer_rates.shape[-2:] != (nspecies, nspecies):
                raise ValueError(
                    '3-D transfer_rates must end with the two species axes. '
                    f'Got shape {transfer_rates.shape} for nspecies={nspecies}.')
            if not hasattr(self, 'salinity_intervals'):
                raise ValueError(
                    '3-D transfer_rates requires self.salinity_intervals for '
                    'an unambiguous model-state export.')

            salinity_intervals = np.asarray(
                self.salinity_intervals, dtype=np.float32).ravel()
            if salinity_intervals.size != transfer_rates.shape[0]:
                raise ValueError(
                    'salinity_intervals length does not match the leading '
                    'transfer_rates dimension: '
                    f'{salinity_intervals.size} != {transfer_rates.shape[0]}.')

            # A coordinate named like the dimension is interpreted by xarray
            # as the coordinate for that dimension.
            self.result['salinity_interval'] = (
                ('salinity_interval',), salinity_intervals)
            self.result['transfer_rates'] = (
                ('salinity_interval', 'specie_0', 'specie_1'),
                transfer_rates)
        else:
            raise ValueError(
                'transfer_rates must be a 2-D species matrix or a 3-D '
                'salinity-by-species matrix. '
                f'Got ndim={transfer_rates.ndim}.')

        ntransformations = np.asarray(self.ntransformations)
        expected = (nspecies, nspecies)
        if ntransformations.shape != expected:
            raise ValueError(
                'ntransformations has incompatible shape for export: '
                f'{ntransformations.shape}; expected {expected}.')
        self.result['ntransformations'] = (
            ('specie_0', 'specie_1'), ntransformations)

        self.result['nspecies'] = self.nspecies
        self.result['name_species'] = self.name_species

        # Species-index attributes are scalar model state. Keep their existing
        # names and values so downstream readers remain backward compatible.
        for var_name in sorted(k for k in vars(self) if k.startswith('num_')):
            self.result[var_name] = getattr(self, var_name)

    def _resolve_reader_dependent_requirements(self):
        """Finalize reader-dependent environmental requirements for this run.

        OpenDrift allocates the trajectory/output schema before ``prepare_run()``
        is called. Reader-dependent variables therefore have to be promoted into
        ``required_variables`` before entering ``OceanDrift.run()`` if they are
        to be stored in the trajectory NetCDF as well as used by the physics.

        At present the only reader-dependent source choice is the optional direct
        current bed stress. Keeping the resolution in one helper makes the pre-run
        and ``prepare_run()`` paths identical and idempotent.
        """
        self._reader_variables = set()
        for _, reader in self.env.readers.items():
            self._reader_variables.update(getattr(reader, 'variables', []))

        self._resolve_current_stress_source()
        return self._sync_required_variables_from_config()

    def run(self, *args, **kwargs):
        """Finalize reader-dependent requirements before OpenDrift builds output.

        ``BaseModel.run()`` creates ``self.result`` from the then-current
        ``required_variables`` before it calls ``prepare_run()``. Resolving the
        optional direct-current-stress source here ensures that a reader-supplied
        ``sea_floor_current_stress`` field is included in that output schema.
        """
        self._resolve_reader_dependent_requirements()
        return super(ChemicalDrift, self).run(*args, **kwargs)

    def prepare_run(self):
        self._configure_element_type_from_config()
        if not hasattr(self, "name_species"):
            self.init_species()
        if not hasattr(self, "transfer_rates"):
            self.init_transfer_rates()

        # Normally already resolved immediately before OceanDrift.run() creates
        # the trajectory/output schema. Repeat the same idempotent resolution
        # here to protect direct prepare_run() use during testing/development.
        self._resolve_reader_dependent_requirements()

        logger.info('Required variables for this run:')
        for name in sorted(self.required_variables):
            logger.info('  %s', name)

        logger.info("Number of species: %s", self.nspecies)
        for i,sp in enumerate(self.name_species):
            logger.info("%3s %s", i, sp)

        logger.info("transfer setup: %s", self.get_config('chemical:transfer_setup'))
        logger.info("nspecies: %s", self.nspecies)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Transfer rates:\n%s", self.transfer_rates)

        self.SPM_vertical_levels_given = False
        for key, value in self.env.readers.items():
            if 'spm' in value.variables:
                if (hasattr(value,'sigma') or hasattr(value,'z') ):
                    self.SPM_vertical_levels_given = True

        self.DOC_vertical_levels_given = False
        for key, value in self.env.readers.items():
            if 'doc' in value.variables:
                if (hasattr(value,'sigma') or hasattr(value,'z') ):
                    self.DOC_vertical_levels_given = True

        if (self.get_config('chemical:sediment:include_wave_stress') and
                (self.get_config('chemical:sediment:enable_deposition') or
                 self.get_config('chemical:sediment:enable_resuspension'))):
            self._validate_wave_stress_source()

        if self._user_resuspension_threshold_overrides_d50() and self._d50_map_reader_present():
            raise ValueError(
                'A mapped sea_floor_d50 reader cannot be supplied when '
                'chemical:sediment:resuspension_critstress_mode == USER. In USER mode '
                'critical stress is user-prescribed, so the d50 map would be unused and '
                'chemical:sediment:d50 is required to remain uniform and equal to '
                'chemical:particle_diameter. Remove the d50 reader or switch to FROM_D50.'
            )

        if self._user_resuspension_threshold_overrides_d50():
            sed_d50 = float(self.get_config('chemical:sediment:d50'))
            part_diam = float(self.get_config('chemical:particle_diameter'))
            if not np.isclose(sed_d50, part_diam, rtol=0.0, atol=0.0):
                raise ValueError(
                    'chemical:sediment:d50 must equal chemical:particle_diameter when '
                    'chemical:sediment:resuspension_critstress_mode == USER, because '
                    'USER mode forbids mapped d50 and assumes one uniform carrier/bed size. '
                    f'Got chemical:sediment:d50={sed_d50} and chemical:particle_diameter={part_diam}.'
                )

        # Model-level ChemicalDrift state is exported separately from element
        # trajectories using semantic/collision-safe dimensions.
        self._store_model_state_for_export()

        super(ChemicalDrift, self).prepare_run()

    def _infer_seed_element_count(self, args, kwargs):
        """Infer ChemicalDrift seed cardinality without overriding OpenDrift semantics.

        Explicit ``number`` remains authoritative. Otherwise, lon/lat arrays,
        number_per_point, time-series releases (>2 times), or per-element
        Chemical properties can determine the cardinality. Two-element time
        arrays are intentionally treated as start/end intervals, matching
        OpenDrift rather than as two particles.

        Returns
        -------
        (num_elements, inject_number)
            ``inject_number`` is True only when ChemicalDrift inferred the
            count from per-element properties but the parent OpenDrift seeder
            would otherwise fall back to seed:number.
        """
        def _non_scalar_size(value):
            if value is None or isinstance(value, (str, bytes)) or np.isscalar(value):
                return None
            try:
                arr = np.asarray(value)
            except Exception:
                return None
            if arr.ndim == 0:
                return None
            return int(arr.size)

        lon = kwargs.get('lon', args[0] if len(args) > 0 else None)
        lat = kwargs.get('lat', args[1] if len(args) > 1 else None)
        time = kwargs.get('time', args[2] if len(args) > 2 else None)

        lon_n = _non_scalar_size(lon)
        lat_n = _non_scalar_size(lat)
        if lon_n is not None and lat_n is not None and lon_n != lat_n:
            raise ValueError(
                f"'lon' length ({lon_n}) must equal 'lat' length ({lat_n}).")

        point_count = lon_n if lon_n is not None else lat_n
        if point_count is not None and point_count < 1:
            raise ValueError('Seeding coordinates must contain at least one point.')

        explicit_number = kwargs.get('number', None)
        number_per_point = kwargs.get('number_per_point', None)

        if explicit_number is not None and number_per_point is not None:
            raise ValueError(
                "'number' and 'number_per_point' cannot both be supplied.")

        if explicit_number is not None:
            num_elements = int(explicit_number)
            if num_elements < 1:
                raise ValueError("'number' must be >= 1.")
            if point_count is not None and point_count > 1:
                if num_elements % point_count != 0:
                    raise ValueError(
                        f"Coordinate length ({point_count}) must divide number "
                        f"of elements ({num_elements}).")
            inject_number = False

        elif number_per_point is not None:
            npp = int(number_per_point)
            if npp < 1:
                raise ValueError("'number_per_point' must be >= 1.")
            if point_count is None or point_count <= 1:
                raise ValueError(
                    "'number_per_point' requires array-like lon/lat with more "
                    "than one seeding point.")
            num_elements = point_count * npp
            inject_number = False  # parent OpenDrift handles this expansion

        else:
            # OpenDrift interprets >2 times as a release time series. Exactly
            # two times mean start/end of a continuous release and therefore do
            # not determine particle count.
            time_n = _non_scalar_size(time)
            time_series_count = time_n if time_n is not None and time_n > 2 else None

            coordinate_count = point_count if point_count is not None else None

            property_sizes = {}
            element_names = set(getattr(self.ElementType, 'variables', {}).keys())
            for name in element_names:
                if name in ('lon', 'lat') or name not in kwargs:
                    continue
                size = _non_scalar_size(kwargs[name])
                if size is not None:
                    property_sizes[name] = size

            inferred_candidates = {}
            if coordinate_count is not None:
                inferred_candidates['coordinates'] = coordinate_count
            if time_series_count is not None:
                inferred_candidates['time'] = time_series_count
            inferred_candidates.update(property_sizes)

            unique_sizes = sorted(set(inferred_candidates.values()))
            if len(unique_sizes) > 1:
                detail = ', '.join(
                    f'{name}={size}' for name, size in sorted(inferred_candidates.items()))
                raise ValueError(
                    'Inconsistent per-element seed-array lengths: ' + detail)

            if unique_sizes:
                num_elements = unique_sizes[0]
                if num_elements < 1:
                    raise ValueError('Per-element seed arrays cannot be empty.')

                # Parent OpenDrift automatically infers count from coordinate
                # arrays of length >1 and from time arrays of length >2. It does
                # not infer count from arbitrary element-property arrays.
                parent_can_infer = (
                    (coordinate_count is not None and coordinate_count > 1)
                    or (time_series_count is not None)
                )
                inject_number = not parent_can_infer
            else:
                num_elements = int(self.get_config('seed:number'))
                inject_number = False

        # Per-element properties must match the total number of elements after
        # any point replication implied by explicit number/number_per_point.
        element_names = set(getattr(self.ElementType, 'variables', {}).keys())
        for name in sorted(element_names):
            if name in ('lon', 'lat') or name not in kwargs:
                continue
            size = _non_scalar_size(kwargs[name])
            if size is not None and size != num_elements:
                raise ValueError(
                    f"'{name}' length ({size}) must equal number of elements "
                    f"({num_elements}).")

        return num_elements, inject_number

    def _prepare_restart_lifecycle(self):
        """Prepare ChemicalDrift restart state before OpenDrift freezes config.

        OpenDrift advances from Config to Ready before its restart seeders run.
        Predefined ChemicalDrift transfer setups may call set_config() while
        initializing species, so all such preparation must happen while the
        model is still in Config mode.
        """
        mode_name = getattr(getattr(self, 'mode', None), 'name', None)

        if mode_name == 'Config':
            self._configure_element_type_from_config()
            self._sync_required_variables_from_config()
            if not hasattr(self, 'name_species'):
                self.init_species()
            if not hasattr(self, 'transfer_rates'):
                self.init_transfer_rates()
            return

        # If the model is already Ready, configuration is frozen. We can only
        # proceed when species and transfer state were prepared earlier.
        missing = [
            name for name in ('name_species', 'nspecies', 'transfer_rates',
                              'ntransformations')
            if not hasattr(self, name)
        ]
        if missing:
            raise RuntimeError(
                'ChemicalDrift restart reached OpenDrift Ready mode before '
                'Chemical model state was initialized. Missing: '
                + ', '.join(missing)
                + '. Call seed_from_file/seed_from_dataset on a newly configured '
                  'ChemicalDrift instance, before any other seeding operation.')

        # Reapplying an unchanged schema is safe and verifies that the concrete
        # optional ElementType still matches the frozen configuration.
        self._configure_element_type_from_config()

    @staticmethod
    def _restart_dataset_string_list(value):
        """Return a robust list of strings from an xarray/NumPy value."""
        raw = getattr(value, 'values', value)
        arr = np.asarray(raw)

        # Older NetCDF encodings may expose a 2-D S1/U1 character array.
        if arr.ndim == 2 and arr.dtype.kind in ('S', 'U'):
            out = []
            for row in arr:
                chars = []
                for item in row.tolist():
                    if isinstance(item, bytes):
                        item = item.decode('utf-8', errors='replace')
                    chars.append(str(item))
                out.append(''.join(chars).rstrip('\x00 ').strip())
            return out

        out = []
        for item in arr.ravel().tolist():
            if isinstance(item, bytes):
                item = item.decode('utf-8', errors='replace')
            out.append(str(item))
        return out

    def _capture_restart_model_state(self, ds, trajectory_time_index, keep_properties):
        """Capture run-level ChemicalDrift state before parent reseeding.

        Older ChemicalDrift files may not contain model-level metadata. Such
        files remain usable: element properties are still restored by OpenDrift
        and missing run-level history stays freshly initialized with a warning.
        """
        if not keep_properties:
            # keep_properties=False is ordinary reseeding, not an exact restart.
            return None

        time_size = int(ds.sizes.get('time', 0)) if hasattr(ds, 'sizes') else 0
        state = {
            'trajectory_time_index': trajectory_time_index,
            'time_size': time_size,
            'restore_ntransformations': True,
        }

        for name in ('nspecies', 'name_species', 'transfer_rates',
                     'ntransformations', 'salinity_interval'):
            if name in ds:
                value = ds[name]
                if name == 'name_species':
                    state[name] = self._restart_dataset_string_list(value)
                else:
                    state[name] = np.asarray(value.values).copy()

        # Accept an alternate/legacy plural spelling if encountered.
        if 'salinity_interval' not in state and 'salinity_intervals' in ds:
            state['salinity_interval'] = np.asarray(
                ds['salinity_intervals'].values).copy()

        # Preserve any exported species-index mapping for later compatibility
        # validation. These are run-level scalars, not element properties.
        state['num_indices'] = {}
        for name in ds.variables:
            if not str(name).startswith('num_'):
                continue
            value = np.asarray(ds[name].values)
            if value.size == 1:
                state['num_indices'][str(name)] = int(value.reshape(-1)[0])

        model_state_names = (
            'nspecies', 'name_species', 'transfer_rates', 'ntransformations')
        if not any(name in state for name in model_state_names):
            logger.warning(
                'Restart source contains no ChemicalDrift run-level model-state '
                'metadata. Element properties can still be restored, but exact '
                'global history (e.g. ntransformations) is unavailable.')

        # ntransformations is exported as run-level final metadata, without a
        # time axis. Restoring it from a non-final trajectory record would copy
        # transitions that occurred after the selected restart time. Keep the
        # newly initialized counter in that case and warn explicitly.
        if 'ntransformations' in state and time_size > 0:
            try:
                selected = int(trajectory_time_index)
            except Exception:
                selected = -1
            if selected < 0:
                selected = time_size + selected
            if selected != time_size - 1:
                state['restore_ntransformations'] = False
                logger.warning(
                    'Restart requested from non-final time index %s. '
                    'ntransformations is stored only as final run-level metadata, '
                    'so it will not be restored for this branch restart.',
                    trajectory_time_index)

        if ('transfer_rates' in state
                and np.asarray(state['transfer_rates']).ndim == 3
                and 'salinity_interval' not in state):
            logger.warning(
                'Restart source has 3-D salinity-dependent transfer_rates but no '
                'exported salinity_interval coordinate. Compatibility will be '
                'checked from transfer-rate shape/values; current configured '
                'salinity intervals will be retained.')

        return state

    def _validate_restart_model_state(self, state):
        """Validate source model-state metadata against the configured model.

        The restart must not silently reinterpret species indices or transfer
        matrices produced by a chemically incompatible simulation.
        """
        if not state:
            return

        if 'nspecies' in state:
            source = np.asarray(state['nspecies'])
            if source.size != 1:
                raise ValueError(
                    'Restart metadata nspecies must be scalar; got shape '
                    f'{source.shape}.')
            source_nspecies = int(source.reshape(-1)[0])
            if source_nspecies != int(self.nspecies):
                raise ValueError(
                    'ChemicalDrift restart species-count mismatch: source '
                    f'nspecies={source_nspecies}, current nspecies={self.nspecies}.')

        if 'name_species' in state:
            source_names = list(state['name_species'])
            current_names = [str(v) for v in self.name_species]
            if source_names != current_names:
                raise ValueError(
                    'ChemicalDrift restart species ordering mismatch. '
                    f'Source={source_names}; current={current_names}.')

        if 'transfer_rates' in state:
            source_rates = np.asarray(state['transfer_rates'])
            current_rates = np.asarray(self.transfer_rates)
            if source_rates.shape != current_rates.shape:
                raise ValueError(
                    'ChemicalDrift restart transfer_rates shape mismatch: '
                    f'source={source_rates.shape}, current={current_rates.shape}.')
            if not np.allclose(
                    source_rates, current_rates, rtol=1e-6, atol=1e-12,
                    equal_nan=True):
                raise ValueError(
                    'ChemicalDrift restart transfer_rates differ from the '
                    'currently configured model. Use the same chemical setup '
                    'as the source simulation for an exact restart.')

        if 'ntransformations' in state:
            source_counts = np.asarray(state['ntransformations'])
            expected = (int(self.nspecies), int(self.nspecies))
            if source_counts.shape != expected:
                raise ValueError(
                    'ChemicalDrift restart ntransformations shape mismatch: '
                    f'source={source_counts.shape}, expected={expected}.')

        if 'salinity_interval' in state:
            source_salinity = np.asarray(
                state['salinity_interval'], dtype=float).ravel()
            if not hasattr(self, 'salinity_intervals'):
                raise ValueError(
                    'Restart source contains salinity-dependent transfer rates, '
                    'but the current ChemicalDrift setup has no salinity_intervals.')
            current_salinity = np.asarray(
                self.salinity_intervals, dtype=float).ravel()
            if (source_salinity.shape != current_salinity.shape
                    or not np.allclose(source_salinity, current_salinity,
                                       rtol=0.0, atol=0.0)):
                raise ValueError(
                    'ChemicalDrift restart salinity intervals differ from the '
                    f'current setup: source={source_salinity.tolist()}, '
                    f'current={current_salinity.tolist()}.')

        for name, source_index in state.get('num_indices', {}).items():
            if not hasattr(self, name):
                raise ValueError(
                    'ChemicalDrift restart species-index metadata is '
                    f'incompatible: source contains {name}={source_index}, '
                    'but the current setup has no such index.')
            current_index = int(getattr(self, name))
            if int(source_index) != current_index:
                raise ValueError(
                    'ChemicalDrift restart species-index mismatch for '
                    f'{name}: source={source_index}, current={current_index}.')

    def _restore_restart_model_state(self, state):
        """Restore captured run-level ChemicalDrift state after reseeding."""
        if not state:
            return

        if 'transfer_rates' in state:
            source_rates = np.asarray(state['transfer_rates'])
            target_dtype = np.asarray(self.transfer_rates).dtype
            self.transfer_rates = source_rates.astype(target_dtype, copy=True)
            self._build_transition_destination_cache()

        if ('ntransformations' in state
                and state.get('restore_ntransformations', True)):
            self.ntransformations = np.asarray(
                state['ntransformations'], dtype=np.int64).copy()

        if 'salinity_interval' in state:
            self.salinity_intervals = np.asarray(
                state['salinity_interval'], dtype=float).ravel().tolist()

    def seed_from_dataset(self, ds, trajectory_time_index=-1, time=None,
                          keep_properties=True, **kwargs):
        """Restart/reseed from an OpenDrift dataset with Chemical lifecycle setup.

        Element properties are still delegated to OpenDrift's implementation.
        This only prepares the concrete Chemical schema/model state before
        the OpenDrift Config->Ready transition and captures run-level state for
        restoration after seeding.
        """
        self._prepare_restart_lifecycle()
        restart_state = self._capture_restart_model_state(
            ds, trajectory_time_index, keep_properties)
        self._validate_restart_model_state(restart_state)

        super(ChemicalDrift, self).seed_from_dataset(
            ds,
            trajectory_time_index=trajectory_time_index,
            time=time,
            keep_properties=keep_properties,
            **kwargs)

        self._restore_restart_model_state(restart_state)

    def seed_from_file(self, filename, trajectory_time_index=-1, time=None,
                       keep_properties=True, **kwargs):
        """Prepare ChemicalDrift while Config is mutable, then delegate file I/O."""
        self._prepare_restart_lifecycle()
        return super(ChemicalDrift, self).seed_from_file(
            filename,
            trajectory_time_index=trajectory_time_index,
            time=time,
            keep_properties=keep_properties,
            **kwargs)

    def seed_elements(self, *args, **kwargs):
        import numpy as np

        self._configure_element_type_from_config()
        # OpenDrift finalizes the environment when super().seed_elements()
        # is entered, so synchronize the active required-variable subset first.
        self._sync_required_variables_from_config()

        if hasattr(self, 'name_species') is False:
            self.init_species()
            self.init_transfer_rates()

        # Number of elements. OpenDrift can infer this from lon/lat arrays,
        # whereas the historical ChemicalDrift wrapper fell back immediately
        # to seed:number. That broke seed_from_dataset()/seed_from_file() when
        # restarting more than one particle.
        num_elements, inject_number = self._infer_seed_element_count(args, kwargs)
        if inject_number and 'number' not in kwargs:
            # Only inject when OpenDrift cannot infer the same count from its
            # own coordinate/time semantics (e.g. scalar coordinates plus a
            # per-element Chemical property array).
            kwargs['number'] = num_elements

        def _as_per_element_int_array(x, n, name):
            if x is None or np.isscalar(x):
                return None
            try:
                arr = np.asarray(x, dtype=int).ravel()
            except Exception as e:
                raise ValueError(f"Could not convert '{name}' to an int array: {e}")
            if arr.size != n:
                raise ValueError(
                    f"'{name}' length ({arr.size}) must equal number of elements ({n}).")
            return arr

        def _as_per_element_array(x, n, name):
            if x is None or np.isscalar(x):
                return None
            try:
                arr = np.asarray(x, dtype=float).ravel()
            except Exception as e:
                raise ValueError(f"Could not convert '{name}' to a float array: {e}")
            if arr.size != n:
                raise ValueError(
                    f"'{name}' length ({arr.size}) must equal number of elements ({n}).")
            return arr

        def _default_foc_from_specie(specie_arr):
            specie_arr = np.asarray(specie_arr, dtype=int).ravel()

            foc_spm = float(self.get_config('chemical:transformations:fOC_SPM'))
            foc_sed = float(self.get_config('chemical:transformations:fOC_sed'))

            if foc_spm <= 0.0:
                raise ValueError(
                    f"chemical:transformations:fOC_SPM must be > 0, got {foc_spm}")
            if foc_sed <= 0.0:
                raise ValueError(
                    f"chemical:transformations:fOC_sed must be > 0, got {foc_sed}")

            name_to_idx = {name: i for i, name in enumerate(self.name_species)}
            sediment_like_names = {
                "Sediment reversible",
                "Sediment slowly reversible",
                "Sediment buried",
                "Sediment irreversible",
            }
            sediment_like_idx = {name_to_idx[n] for n in sediment_like_names if n in name_to_idx}

            # Default for all non-sediment species in the water column
            foc_default = np.full(specie_arr.size, foc_spm, dtype=float)

            # Sediment species use sediment OC fraction
            if sediment_like_idx:
                sedmask = np.isin(specie_arr, list(sediment_like_idx))
                foc_default[sedmask] = foc_sed

            return foc_default

        def _default_critstress_from_specie(specie_arr):
            specie_arr = np.asarray(specie_arr, dtype=int).ravel()
            crit_default = np.ones(specie_arr.size, dtype=float)

            if self.get_config('chemical:sediment:use_critstress_heterogeneity'):
                name_to_idx = {name: i for i, name in enumerate(self.name_species)}
                sediment_like_names = {
                    "Sediment reversible",
                    "Sediment slowly reversible",
                    "Sediment buried",
                    "Sediment irreversible",
                }
                sediment_like_idx = {name_to_idx[n] for n in sediment_like_names if n in name_to_idx}

                if sediment_like_idx:
                    sedmask = np.isin(specie_arr, list(sediment_like_idx))
                    nsed = int(np.sum(sedmask))
                    if nsed > 0:
                        crit_default[sedmask] = self._sample_critstress_factor(nsed)

            return crit_default

        def _default_d50_from_specie(specie_arr, diameter_arr, sediment_d50_arr=None):
            specie_arr = np.asarray(specie_arr, dtype=int).ravel()
            diameter_arr = np.asarray(diameter_arr, dtype=float).ravel()
            d50_default = np.zeros(specie_arr.size, dtype=float)
            name_to_idx = {name: i for i, name in enumerate(self.name_species)}

            particle_names = {
                "Particle reversible", "Particle slowly reversible", "Particle irreversible",
            }
            sediment_names = {
                "Sediment reversible", "Sediment slowly reversible", "Sediment buried", "Sediment irreversible",
            }
            particle_idx = {name_to_idx[n] for n in particle_names if n in name_to_idx}
            sediment_idx = {name_to_idx[n] for n in sediment_names if n in name_to_idx}

            if particle_idx:
                mask_part = np.isin(specie_arr, list(particle_idx))
                d50_default[mask_part] = np.maximum(diameter_arr[mask_part], 0.0)

            if sediment_idx:
                mask_sed = np.isin(specie_arr, list(sediment_idx))
                if np.any(mask_sed):
                    if sediment_d50_arr is None:
                        sedvals = np.full(int(np.sum(mask_sed)), float(self.get_config('chemical:sediment:d50')), dtype=float)
                    else:
                        sedvals = np.asarray(sediment_d50_arr, dtype=float).ravel()
                        if sedvals.size != int(np.sum(mask_sed)):
                            raise ValueError(
                                f"sediment_d50_arr has size {sedvals.size}, expected {int(np.sum(mask_sed))}")
                    d50_default[mask_sed] = np.maximum(sedvals, 0.0)

            return d50_default

        # Speciation handling
        if 'specie' in kwargs and kwargs['specie'] is not None:
            sp = kwargs['specie']

            if np.isscalar(sp):
                init_specie = np.full(num_elements, int(sp), dtype=int)
            else:
                init_specie = _as_per_element_int_array(sp, num_elements, "specie")
        else:
            # Ensure specie key doesn't accidentally influence downstream logic
            kwargs.pop('specie', None)

            # Config-driven initial partitioning
            particle_frac = kwargs.get('particle_fraction', self.get_config('seed:particle_fraction'))
            lmm_frac = kwargs.get('LMM_fraction', self.get_config('seed:LMM_fraction'))

            if not np.isclose(lmm_frac + particle_frac, 1.0, rtol=0, atol=1e-12):
                logger.error('Fraction does not sum up to 1: %s' % str(lmm_frac + particle_frac))
                logger.error('LMM fraction: %s ' % str(lmm_frac))
                logger.error('Particle fraction %s ' % str(particle_frac))
                raise ValueError(
                    'Illegal specie fraction combination : ' + str(lmm_frac) + ' ' + str(particle_frac))

            transfer_setup = self.get_config('chemical:transfer_setup')
            dissolved_idx = None
            particle_idx = None

            if lmm_frac > 0:
                if transfer_setup == 'Sandnesfj_Al':
                    if not hasattr(self, 'num_lmmcation'):
                        raise ValueError(
                            "Default seeding requires 'num_lmmcation' when "
                            "seed:LMM_fraction > 0 for transfer_setup='Sandnesfj_Al'. "
                            "Enable chemical:species:LMMcation or pass 'specie' explicitly.")
                    dissolved_idx = self.num_lmmcation
                else:
                    if not hasattr(self, 'num_lmm'):
                        raise ValueError(
                            "Default seeding requires 'num_lmm' when seed:LMM_fraction > 0. "
                            "Enable chemical:species:LMM or pass 'specie' explicitly.")
                    dissolved_idx = self.num_lmm

            if particle_frac > 0:
                if not hasattr(self, 'num_prev'):
                    raise ValueError(
                        "Default seeding requires 'num_prev' when seed:particle_fraction > 0. "
                        "Enable chemical:species:Particle_reversible or pass 'specie' explicitly.")
                particle_idx = self.num_prev

            init_specie = np.empty(num_elements, dtype=int)

            if lmm_frac == 1.0:
                init_specie[:] = dissolved_idx
            elif particle_frac == 1.0:
                init_specie[:] = particle_idx
            else:
                dissolved = np.random.rand(num_elements) < lmm_frac
                init_specie[dissolved] = dissolved_idx
                init_specie[~dissolved] = particle_idx

        if np.any(init_specie < 0) or np.any(init_specie >= len(self.name_species)):
            bad = init_specie[(init_specie < 0) | (init_specie >= len(self.name_species))][:5]
            raise ValueError(
                f"'specie' contains out-of-range indices. "
                f"Valid range is 0..{len(self.name_species)-1}. Examples: {bad.tolist()}")

        kwargs['specie'] = init_specie

        if logger.isEnabledFor(logging.DEBUG):
            counts = np.bincount(
                np.asarray(init_specie, dtype=np.int64),
                minlength=len(self.name_species),
            )
            logger.debug("Initial partitioning:")
            for i, sp_name in enumerate(self.name_species):
                logger.debug("%9s %3s %-24s", int(counts[i]), i, sp_name)

        # Diameter assignment (respect explicit per-element diameters)
        diam_in = kwargs.get("diameter", None)

        arr = _as_per_element_array(diam_in, num_elements, "diameter")
        sediment_seed_d50 = None
        if arr is not None:
            kwargs["diameter"] = np.maximum(arr, 0.0)
        else:
            dia_diss = float(self.get_config('chemical:dissolved_diameter'))
            dia_part = float(self.get_config('chemical:particle_diameter'))
            dia_doc = float(self.get_config('chemical:doc_particle_diameter'))
            sigma_part_ln = float(self.get_config('chemical:particle_diameter_uncertainty'))
            sigma_doc_ln = float(self.get_config('chemical:doc_particle_diameter_uncertainty'))

            init_diam = np.full(num_elements, dia_diss, dtype=float)

            name_to_idx = {name: i for i, name in enumerate(self.name_species)}

            particle_names = {
                "Particle reversible",
                "Particle slowly reversible",
                "Particle irreversible",
            }
            sediment_names = {
                "Sediment reversible",
                "Sediment slowly reversible",
                "Sediment buried",
                "Sediment irreversible",
            }
            particle_idx = {name_to_idx[n] for n in particle_names if n in name_to_idx}
            sediment_idx = {name_to_idx[n] for n in sediment_names if n in name_to_idx}

            humic_idx = name_to_idx.get("Humic colloid", None)
            polymer_idx = name_to_idx.get("Polymer", None)
            colloid_idx = name_to_idx.get("Colloid", None)

            if particle_idx:
                mask_part = np.isin(init_specie, list(particle_idx))
                nmask = int(mask_part.sum())
                if nmask > 0:
                    if diam_in is not None and np.isscalar(diam_in):
                        dia_seed = float(diam_in)
                    else:
                        dia_seed = dia_part
                    init_diam[mask_part] = self._sample_lognormal_diameter(
                        median_diameter=dia_seed,
                        sigma_ln=sigma_part_ln,
                        n=nmask)

            if sediment_idx:
                mask_sed = np.isin(init_specie, list(sediment_idx))
                nsed = int(mask_sed.sum())
                if nsed > 0:
                    # During seeding, self.environment may not exist yet.
                    # Therefore sediment-seeded elements use the configured fallback d50,
                    # while mapped/local bed d50 can still be used later during runtime.
                    sediment_seed_d50 = np.full(
                        nsed,
                        float(self.get_config('chemical:sediment:d50')),
                        dtype=float
                    )
                    init_diam[mask_sed] = np.maximum(sediment_seed_d50, 0.0)

            if humic_idx is not None:
                mask = init_specie == humic_idx
                if np.any(mask):
                    init_diam[mask] = self._sample_lognormal_diameter(
                        median_diameter=dia_doc,
                        sigma_ln=sigma_doc_ln,
                        n=int(np.sum(mask)))

            if polymer_idx is not None:
                mask = init_specie == polymer_idx
                if np.any(mask):
                    init_diam[mask] = self._sample_lognormal_diameter(
                        median_diameter=dia_doc,
                        sigma_ln=sigma_doc_ln,
                        n=int(np.sum(mask)))

            if colloid_idx is not None:
                mask = init_specie == colloid_idx
                if np.any(mask):
                    init_diam[mask] = dia_diss

            kwargs["diameter"] = np.maximum(init_diam, 0.0)

        # d50 assignment (respect explicit per-element values; default: particles use current diameter, sediments use local bed d50, dissolved/doc-like species use 0)
        d50_in = kwargs.get("d50", None)
        d50_default = _default_d50_from_specie(init_specie, kwargs["diameter"], sediment_d50_arr=sediment_seed_d50)
        d50_arr = _as_per_element_array(d50_in, num_elements, "d50")
        if d50_arr is not None:
            d50_arr = np.asarray(d50_arr, dtype=float).copy()
            bad = (~np.isfinite(d50_arr)) | (d50_arr < 0.0)
            if np.any(bad):
                logger.warning("Replacing %s negative/non-finite d50 values during seeding with species-based defaults.", int(np.sum(bad)))
                d50_arr[bad] = d50_default[bad]
            kwargs["d50"] = d50_arr
        elif d50_in is not None and np.isscalar(d50_in):
            d50_scalar = float(d50_in)
            if (not np.isfinite(d50_scalar)) or (d50_scalar < 0.0):
                logger.warning("Received scalar d50=%s during seeding; replacing with species-based defaults.", d50_scalar)
                kwargs["d50"] = d50_default
            else:
                kwargs["d50"] = np.full(num_elements, d50_scalar, dtype=float)
        else:
            kwargs["d50"] = d50_default

        # f_OC assignment (respect explicit per-element values, but never keep <= 0)
        foc_in = kwargs.get("f_OC", None)
        foc_default = _default_foc_from_specie(init_specie)
        foc_arr = _as_per_element_array(foc_in, num_elements, "f_OC")

        if foc_arr is not None:
            foc_arr = np.asarray(foc_arr, dtype=float).copy()
            bad = (~np.isfinite(foc_arr)) | (foc_arr <= 0.0)
            if np.any(bad):
                logger.warning(
                    "Replacing %s non-positive/non-finite f_OC values during seeding "
                    "with species-based defaults (fOC_SPM / fOC_sed).",
                    int(np.sum(bad)))
                foc_arr[bad] = foc_default[bad]
            kwargs["f_OC"] = foc_arr

        elif foc_in is not None and np.isscalar(foc_in):
            foc_scalar = float(foc_in)
            if (not np.isfinite(foc_scalar)) or (foc_scalar <= 0.0):
                logger.warning(
                    "Received scalar f_OC=%s during seeding; replacing with "
                    "species-based defaults (fOC_SPM / fOC_sed).",
                    foc_scalar)
                kwargs["f_OC"] = foc_default
            else:
                kwargs["f_OC"] = np.full(num_elements, foc_scalar, dtype=float)

        else:
            kwargs["f_OC"] = foc_default

        # critstress_factor assignment (respect explicit per-element values)
        crit_in = kwargs.get("critstress_factor", None)
        crit_default = _default_critstress_from_specie(init_specie)
        crit_arr = _as_per_element_array(crit_in, num_elements, "critstress_factor")

        if crit_arr is not None:
            crit_arr = np.asarray(crit_arr, dtype=float).copy()
            bad = (~np.isfinite(crit_arr)) | (crit_arr <= 0.0)
            if np.any(bad):
                logger.warning(
                    "Replacing %s non-positive/non-finite critstress_factor values during seeding "
                    "with 1.0 / sampled heterogeneity defaults.",
                    int(np.sum(bad)))
                crit_arr[bad] = crit_default[bad]
            kwargs["critstress_factor"] = crit_arr

        elif crit_in is not None and np.isscalar(crit_in):
            crit_scalar = float(crit_in)
            if (not np.isfinite(crit_scalar)) or (crit_scalar <= 0.0):
                logger.warning(
                    "Received scalar critstress_factor=%s during seeding; replacing with defaults.",
                    crit_scalar)
                kwargs["critstress_factor"] = crit_default
            else:
                kwargs["critstress_factor"] = np.full(num_elements, crit_scalar, dtype=float)

        else:
            kwargs["critstress_factor"] = crit_default

        super(ChemicalDrift, self).seed_elements(*args, **kwargs)

    ###########################################################################
    # Common helpers
    ###########################################################################

    def specie_num2name(self,num):
        return self.name_species[num]

    def specie_name2num(self,name):
        num = self.name_species.index(name)
        return num

    def _dbg_arr(self, name, arr, idx=None, max_items=20):
        '''
        Print an array for debug
        '''
        import numpy as np

        a = np.asarray(arr)
        if idx is not None:
            a = a[idx]

        flat = a.ravel()
        n = flat.size
        if n == 0:
            print(f"[DEBUG {name}: shape={a.shape}, n=0, sample=[]")
            logger.debug(f"[DEBUG {name}: shape={a.shape}, n=0, sample=[]")
            return

        finite = np.isfinite(flat)
        if finite.any():
            vmin = np.nanmin(flat[finite])
            vmax = np.nanmax(flat[finite])
        else:
            vmin = "all_nonfinite"
            vmax = "all_nonfinite"

        head = flat[:max_items]
        logger.debug(f"[DEBUG {name}: shape={a.shape}, n={n}, min={vmin}, max={vmax}, sample={head}")
        print(f"[DEBUG {name}: shape={a.shape}, n={n}, min={vmin}, max={vmax}, sample={head}")

    @staticmethod
    def _validate_scalar_param(name, value, *, finite=True, gt=None, ge=None, lt=None, le=None):
        x = float(value)
        if finite and not np.isfinite(x):
            raise ValueError(f"{name} must be finite, got {value}")
        if gt is not None and not (x > gt):
            raise ValueError(f"{name} must be > {gt}, got {x}")
        if ge is not None and not (x >= ge):
            raise ValueError(f"{name} must be >= {ge}, got {x}")
        if lt is not None and not (x < lt):
            raise ValueError(f"{name} must be < {lt}, got {x}")
        if le is not None and not (x <= le):
            raise ValueError(f"{name} must be <= {le}, got {x}")
        return x

    @staticmethod
    def _validate_array_param(name, value, *, finite=True, ge=None, gt=None, le=None, lt=None):
        arr = np.asarray(value, dtype=float)
        if finite and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if gt is not None and np.any(arr <= gt):
            raise ValueError(f"{name} must be > {gt} everywhere")
        if ge is not None and np.any(arr < ge):
            raise ValueError(f"{name} must be >= {ge} everywhere")
        if lt is not None and np.any(arr >= lt):
            raise ValueError(f"{name} must be < {lt} everywhere")
        if le is not None and np.any(arr > le):
            raise ValueError(f"{name} must be <= {le} everywhere")
        return arr

    def _has_reader_variable(self, name):
        """
        Return True if an environment variable is actually provided by at least
        one currently attached reader.

        The cached set is refreshed from attached readers so that readers added
        after model construction are also detected.
        """
        reader_vars = set(getattr(self, '_reader_variables', set()) or set())

        env = getattr(self, 'env', None)
        if env is not None:
            for _, reader in getattr(env, 'readers', {}).items():
                reader_vars.update(getattr(reader, 'variables', []))

        self._reader_variables = reader_vars
        return name in reader_vars

    def _has_explicit_environment_source(self, name):
        """Return True for an attached reader or an explicit environment constant.

        OpenDrift materializes ``environment:constant:<name>`` as a constant reader
        during environment finalization. ChemicalDrift also needs to recognize such
        constants before that lifecycle step, because zero is a valid explicit value
        for wave height and direction.
        """
        if self._has_reader_variable(name):
            return True

        try:
            constant = self.get_config(f'environment:constant:{name}')
        except (KeyError, ValueError):
            return False

        return constant is not None

    def _optional_env_array(self, name, idx=None):
        """
        Return an environment array only if the variable is truly supplied by a reader.
        1) If the variable is not present in any reader:
               return None
        2) Otherwise:
               return _env_array(name, fallback=None, idx=idx)
        This prevents silent use of fallback values when the caller needs to know
        whether a field is genuinely available.
        """
        if not self._has_reader_variable(name):
            return None
        return self._env_array(name, fallback=None, idx=idx)

    def _env_array(self, name, fallback=None, idx=None):
        """Return an environment variable as a float array.
        1) Read:
               val = self.environment.<name>
        2) If missing, use fallback
        3) Convert to ndarray(dtype=float)
        4) If idx is None:
               - scalar values are broadcast to all active elements
               - arrays are returned as-is
        5) If idx is provided:
               - scalar values are broadcast to len(idx)
               - arrays are indexed by idx
        """
        # Determine requested output length early, so we can still return a sensible
        # fallback even when self.environment does not yet exist (e.g. during seeding).
        if idx is None:
            try:
                n = int(self.num_elements_active())
            except Exception:
                n = 0
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()
            n = idx.size

        env = getattr(self, 'environment', None)
        val = getattr(env, name, None) if env is not None else None

        if val is None:
            val = fallback
        if val is None:
            return None

        arr = np.asarray(val, dtype=float)

        if idx is None:
            if arr.ndim == 0:
                return np.full(n, float(arr), dtype=float)
            return arr

        if arr.ndim == 0:
            return np.full(idx.size, float(arr), dtype=float)
        return arr[idx]

    def _sanitize_positive_with_fallback(self, values, fallback):
        """Return finite positive values, replacing non-finite or <=0 with fallback."""
        values = np.asarray(values, dtype=float)
        out = values.copy()
        invalid = (~np.isfinite(out)) | (out <= 0.0)
        if np.any(invalid):
            out[invalid] = fallback
        return out

    def _z_array(self, idx=None):
        """
        Return element depths z as a float array.
        If idx is None:
            returns all active-element z values
        else:
            returns z[idx]
        """
        if idx is None:
            return np.asarray(self.elements.z, dtype=float)
        idx = np.asarray(idx, dtype=np.int64).ravel()
        return np.asarray(self.elements.z[idx], dtype=float)

    def _apply_halfdepth_profile(self, values, z, mld, half_depth):
        """
        Apply an exponential half-depth profile below the mixed layer.

        For elements deeper than the mixed layer:
            values(z) = values(z_mld) * exp(-(z + mld) * ln(0.5) / half_depth)
        Since z is negative downward in OpenDrift coordinates, the expression:
            -(z + mld)
        is the depth below the mixed-layer base.
        """
        values = np.asarray(values, dtype=float).copy()
        if half_depth <= 0:
            return values

        lower_mld = z < -mld
        if np.any(lower_mld):
            values[lower_mld] *= np.exp( -(z[lower_mld] + mld[lower_mld]) * np.log(0.5) / half_depth)
        return values

    def _spm_g_m3(self, idx=None):
        """
        Return local suspended particulate matter concentration [g/m3].
        1) Start from environment.spm
        2) If no SPM vertical levels are provided by the reader:
               apply an exponential half-depth profile below the mixed layer
        3) Otherwise:
               use the reader-provided vertical structure directly
        """
        conc_spm = self._env_array('spm', 1.0, idx=idx)
        if not self.SPM_vertical_levels_given:
            z = self._z_array(idx)
            mld = self._env_array('ocean_mixed_layer_thickness', 50.0, idx=idx)
            conc_spm = self._apply_halfdepth_profile(conc_spm, z, mld,
                float(self.get_config('chemical:particle_concentration_half_depth')),)
        return conc_spm

    def _doc_mmolkg(self, idx=None):
        """
        Return local dissolved organic carbon concentration [mmol C / kg].
        1) Start from environment.doc
        2) If no DOC vertical levels are provided by the reader:
               apply an exponential half-depth profile below the mixed layer
        3) Otherwise:
               use the reader-provided vertical structure directly
        """
        conc_doc = self._env_array('doc', 0.0, idx=idx)
        if not self.DOC_vertical_levels_given:
            z = self._z_array(idx)
            mld = self._env_array('ocean_mixed_layer_thickness', 50.0, idx=idx)
            conc_doc = self._apply_halfdepth_profile(conc_doc, z, mld,
                float(self.get_config('chemical:doc_concentration_half_depth')),)
        return conc_doc

    def _sample_lognormal_diameter(self, median_diameter, sigma_ln, n):
        """
        Sample positive diameters from a log-normal distribution.
          - median_diameter is the median (or nominal) diameter [m]
          - sigma_ln is the standard deviation of ln(diameter)
        For sigma_ln <= 0, return a constant array.
        """
        median_diameter = float(median_diameter)
        sigma_ln = float(sigma_ln)

        if median_diameter < 0:
            raise ValueError("median_diameter must be >= 0")
        if sigma_ln < 0:
            raise ValueError("sigma_ln must be >= 0")

        if n <= 0:
            return np.empty(0, dtype=float)

        if median_diameter == 0.0:
            return np.zeros(n, dtype=float)

        if sigma_ln == 0.0:
            return np.full(n, median_diameter, dtype=float)

        return median_diameter * np.exp(np.random.normal(0.0, sigma_ln, n))

    ###########################################################################
    # Physical and chemical correction helpers
    ###########################################################################

    ### General temperature and salinity correction
    def tempcorr(self,mode,DeltaH,T_C,Tref_C):
        """Temperature correction factor for a process rate or partition coefficient.
            1) Arrhenius
               Applies a thermodynamic temperature correction relative to a reference temperature:
                   corr = exp( -(DeltaH / R) * (1/T - 1/Tref) )
            2) Q10
               Applies an empirical factor-of-two-per-10C-type scaling:
                   corr = 2^((T_C - Tref_C)/10)
            where:
                DeltaH: enthalpy-like parameter [J/mol]
                R:      8.3145 [J mol-1 K-1]
                T:      ambient temperature [K]
                Tref:   reference temperature [K]
            """
        if mode == 'Arrhenius':
            R = 8.3145 # J/(mol*K)
            T_K = T_C + 273.15
            Tref_K = Tref_C + 273.15
            corr = np.exp(-(DeltaH/R)*(1/T_K - 1/Tref_K))
        elif mode =='Q10':
            corr = 2**((T_C - Tref_C)/10)
        else:
            raise ValueError(f"Unknown tempcorr mode: {mode}")
        return corr

    def salinitycorr(self,Setschenow,Temperature,Salinity):
        """Salinity correction factor using a Setschenow-type relation.
        The correction is based on the salt concentration in seawater and modifies
        a partition coefficient or solubility-related quantity according to:
           Log(Kd_fin)=(Setschenow ∙ ConcSalt)+Log(Kd_T)
           corr = K_final / K_T = 10^(Setschenow * ConcSalt)
        and
           ConcSalt = (Salinity / MWsalt) * rho_sw
        where:
            Salinity:    practical salinity, treated here as g salt / kg seawater [PSU]
            MWsalt:      68.35 g/mol, representative mean molar mass of sea salts
            rho_sw:      seawater density [kg/L]
            Setschenow:  Setschenow constant [L/mol].
            Temperature: Water temperature [C]
        """

        MWsalt = 68.35 # average mass of sea water salt (g/mol) Schwarzenbach Gschwend Imboden Environmental Organic Chemistry
        Dens_sw = self.sea_water_density(T=Temperature, S=Salinity)*1e-3 # (Kg/L)

        ConcSalt = (Salinity/MWsalt)*Dens_sw
        corr = 10**(Setschenow*ConcSalt)
        return corr

    ###########################################################################
    # Helpers for partitioning and species transitions
    ###########################################################################

    ### Partitioning coefficients
    def speciation_fractions(self, diss, pH, pKa_acid, pKa_base):
        """
        Return speciation fractions as (phi_neutral, phi_anion, phi_cation).

          - acid: HA (neutral) <-> A- + H+      pKa_acid = pKa(HA)
          - base: BH+ (cation) <-> B + H+       pKa_base = pKa(BH+)  (conjugate-acid pKa)
          - amphoteric: one acidic site + one basic site, ignoring zwitterion:
                phi_neutral corresponds to the uncharged form (e.g. HA/B)
                phi_anion corresponds to deprotonated acid site (A-)
                phi_cation corresponds to protonated base site (BH+)
        """
        pH = np.asarray(pH, dtype=float)

        if diss == "nondiss":
            phi_neutral = np.ones_like(pH)
            phi_anion   = np.zeros_like(pH)
            phi_cation  = np.zeros_like(pH)
            return phi_neutral, phi_anion, phi_cation

        if diss == "acid":
            # HA neutral fraction
            phi_neutral = 1.0 / (1.0 + 10.0 ** (pH - pKa_acid))  # HA
            phi_anion   = 1.0 - phi_neutral                      # A-
            phi_cation  = np.zeros_like(pH)
            return phi_neutral, phi_anion, phi_cation

        if diss == "base":
            # pKa_base is pKa of BH+ (conjugate acid). Protonated/cationic fraction:
            phi_cation  = 1.0 / (1.0 + 10.0 ** (pH - pKa_base))  # BH+
            phi_neutral = 1.0 - phi_cation                       # B
            phi_anion   = np.zeros_like(pH)
            return phi_neutral, phi_anion, phi_cation

        if diss == "amphoteric":
            # Ignoring zwitterion: neutral + anion + cation = 1
            denom = 1.0 + 10.0 ** (pH - pKa_acid) + 10.0 ** (pKa_base - pH)
            phi_neutral = 1.0 / denom
            phi_anion   = phi_neutral * 10.0 ** (pH - pKa_acid)
            phi_cation  = phi_neutral * 10.0 ** (pKa_base - pH)
            return phi_neutral, phi_anion, phi_cation

        raise ValueError(f"Unknown dissociation mode: {diss!r}")

    def koc_correction(self, KOC_initial, KOC_neutral, KOC_anion, KOC_cation,
                      pH, diss, pKa_acid, pKa_base, eps=1e-10):
        """
        Compute a pH-dependent correction factor for KOC based on species fractions.
        The updated organic-carbon partition coefficient is obtained as a
        speciation-weighted average of the phase-specific coefficients:
            KOC_updated = (KOC_neutral * phi_neutral) +(KOC_anion * phi_anion) + (KOC_cation * phi_cation)

        KOC_initial:                        Baseline KOC used as reference.
        KOC_neutral, KOC_anion, KOC_cation: Species-specific KOC values for neutral, anionic, and cationic forms.
        pH:                                 Ambient pH controlling speciation.
        diss:                               Dissociation model {'nondiss', 'acid', 'base', 'amphoteric'}.
        pKa_acid, pKa_base :                Acid/base dissociation constants.
        eps:                                Small positive floor to avoid division by zero.
        """
        phi_neu, phi_an, phi_cat = self.speciation_fractions(diss, pH, pKa_acid, pKa_base)

        KOC_updated = (KOC_neutral * phi_neu) + (KOC_anion * phi_an) + (KOC_cation * phi_cat)
        KOC_initial = np.asarray(KOC_initial, dtype=float)

        return KOC_updated / np.maximum(KOC_initial, eps)

    def calc_KOC_sedcorr(self, KOC_sed_initial, KOC_sed_n, pKa_acid, pKa_base, pH_sed, diss,
                         KOC_sed_acid, KOC_sed_base):
        """
        Correction of KOC in sediments due to pH.

        KOC_sed_n    : neutral form KOC in sediments
        KOC_sed_acid : anionic form KOC (A-)
        KOC_sed_base : cationic form KOC (BH+)
        """
        return self.koc_correction(
            KOC_initial=KOC_sed_initial,
            KOC_neutral=KOC_sed_n,
            KOC_anion=KOC_sed_acid,
            KOC_cation=KOC_sed_base,
            pH=pH_sed,
            diss=diss,
            pKa_acid=pKa_acid,
            pKa_base=pKa_base,
        )

    def calc_KOC_watcorrSPM(self, KOC_SPM_initial, KOC_sed_n, pKa_acid, pKa_base, pH_water_SPM, diss,
                            KOC_sed_acid, KOC_sed_base):
        """
        Correction of KOC for SPM due to water pH (speciation in the water).

        KOC_sed_n: neutral-form KOC for SPM
        KOC_sed_acid: anionic-form KOC for SPM (A-)
        KOC_sed_base: cationic-form KOC for SPM (BH+)
        """
        return self.koc_correction(
            KOC_initial=KOC_SPM_initial,
            KOC_neutral=KOC_sed_n,
            KOC_anion=KOC_sed_acid,
            KOC_cation=KOC_sed_base,
            pH=pH_water_SPM,
            diss=diss,
            pKa_acid=pKa_acid,
            pKa_base=pKa_base,
        )

    def calc_KOC_watcorrDOM(self, KOC_DOM_initial, KOC_DOM_n, pKa_acid, pKa_base, pH_water_DOM, diss,
                            KOC_DOM_acid, KOC_DOM_base):
        """
        Correction of KOC for DOM due to water pH (speciation in the water).

        KOC_DOM_n    : neutral form KOC for DOM
        KOC_DOM_acid : anionic form KOC for DOM (A-)
        KOC_DOM_base : cationic form KOC for DOM (BH+)
        """
        return self.koc_correction(
            KOC_initial=KOC_DOM_initial,
            KOC_neutral=KOC_DOM_n,
            KOC_anion=KOC_DOM_acid,
            KOC_cation=KOC_DOM_base,
            pH=pH_water_DOM,
            diss=diss,
            pKa_acid=pKa_acid,
            pKa_base=pKa_base,
        )

    def koc_updated(self, KOC_neutral, KOC_anion, KOC_cation, pH, diss, pKa_acid, pKa_base):
        """Return pH-updated KOC = sum_i(KOC_i * phi_i).."""
        phi_neu, phi_an, phi_cat = self.speciation_fractions(diss, pH, pKa_acid, pKa_base)
        return (KOC_neutral * phi_neu) + (KOC_anion * phi_an) + (KOC_cation * phi_cat)

    def update_transfer_rates(self):
        """
        Update per-element transfer rates from the background matrix using local conditions.
        This routine starts from the baseline rates prepared in init_transfer_rates()
        and then modifies them for each active element based on:
            current species, pH, temperature, salinity,SPM / DOC concentrations
            sediment geometry, element f_OC values

        General workflow
        1) Copy the baseline row for each element:
               transfer_rates1D[e, :] = transfer_rates[current_species[e], :]
        2) Apply transfer_setup-specific environmental corrections
        3) For sediment adsorption:
               zero rates outside the local interaction layer
        4) For burial:
               rescale buried-sediment transfer rates by local active-layer thickness
        """
        transfer_setup = self.get_config('chemical:transfer_setup')

        if transfer_setup == 'Sandnesfj_Al':
            sal = self._env_array('sea_water_salinity', 34.0)
            sali = np.searchsorted(self.salinity_intervals, sal, side='right') - 1
            sali = np.clip(sali, 0, len(self.salinity_intervals) - 1)
            self.elements.transfer_rates1D = self.transfer_rates[sali, self.elements.specie, :]
            return

        if transfer_setup not in ('metals', 'custom', '137Cs_rev', 'organics'):
            raise ValueError(f"Unsupported transfer_setup: {transfer_setup}")

        specie = self.elements.specie
        self.elements.transfer_rates1D = self.transfer_rates[specie, :].astype(np.float32, copy=False,)

        diss = self.get_config('chemical:transformations:dissociation')

        Sed_H_0 = float(self.get_config('chemical:sediment:layer_thickness'))
        Sed_L_0 = float(self.get_config('chemical:sediment:mixing_depth'))

        def _local_sediment_geometry(idx):
            '''
            Returns local sediment geometry:
                Sed_H_int   = interaction-layer thickness [m]
                Sed_L_eff   = active sediment layer thickness [m]
                water_depth = local depth [m]
                mld         = mixed-layer depth [m]
            '''
            idx = np.asarray(idx, dtype=np.int64).ravel()
            if idx.size == 0:
                empty = np.empty(0, dtype=np.float32)
                return empty, empty, empty, empty

            Sed_H_env = self._env_array('interaction_sediment_layer_thickness', 0.0, idx=idx)
            Sed_L_env = self._env_array('active_sediment_layer_thickness', 0.0, idx=idx)
            water_depth = self._env_array('sea_floor_depth_below_sea_level', 10000.0, idx=idx)
            mld = self._env_array('ocean_mixed_layer_thickness', 0.0, idx=idx)

            water_depth = np.asarray(water_depth, dtype=np.float32)
            water_depth = np.maximum(water_depth, 0.0)

            mld = np.asarray(mld, dtype=np.float32)
            mld = np.maximum(mld, 0.0)

            # Physical near-bed interaction layer used for geometric cutoff.
            # Use reader-provided interaction thickness when available, otherwise
            # fall back to the configured value, but never let it exceed local depth.
            Sed_H_nominal = np.where(Sed_H_env > 0, Sed_H_env, Sed_H_0).astype(np.float32, copy=False)
            Sed_H_int = np.minimum(Sed_H_nominal, water_depth).astype(np.float32, copy=False)

            # Active sediment layer thickness used in k14 numerator
            Sed_L_eff = np.where(Sed_L_env > 0, Sed_L_env, Sed_L_0).astype(np.float32, copy=False)

            return Sed_H_int, Sed_L_eff, water_depth, mld

        def _local_k14_and_cutoff(idx):
            '''
            computes:
                k14_corr = (Sed_L_eff / H_rate_eff) / (Sed_L_0 / Sed_H_0)
            and the geometric cutoff used to disable dissolved->sediment exchange
            for elements too far above the bed.
            '''
            Sed_H_int, Sed_L_eff, water_depth, mld = _local_sediment_geometry(idx)

            # Keep adsorption restricted to the actual interaction layer
            interaction_cutoff = np.maximum(Sed_H_int, 0.0)

            # Use a separate effective water thickness for the k14 scaling
            # If the water column is fully mixed down to the bed, use full local water depth
            # in the probability/rate calculation, but do not enlarge the geometric cutoff.
            H_rate_eff = Sed_H_int.copy()
            full_mix = (water_depth > 0.0) & (mld >= water_depth)
            H_rate_eff[full_mix] = water_depth[full_mix]

            k14_corr = np.zeros_like(Sed_L_eff, dtype=np.float32)
            if Sed_L_0 > 0 and Sed_H_0 > 0:
                valid = (Sed_L_eff > 0) & (H_rate_eff > 0)
                k14_corr[valid] = (
                    (Sed_L_eff[valid] / H_rate_eff[valid]) /
                    (Sed_L_0 / Sed_H_0)
                )

            return k14_corr, interaction_cutoff, Sed_H_int, Sed_L_eff, H_rate_eff, water_depth

        def _local_burial_corr(idx):
            '''
            rescales burial by:
                burial_corr = Sed_L_0 / Sed_L_eff
            '''
            _, Sed_L_eff, _, _ = _local_sediment_geometry(idx)

            burial_corr = np.zeros_like(Sed_L_eff, dtype=np.float32)
            if Sed_L_0 > 0:
                valid = Sed_L_eff > 0
                burial_corr[valid] = Sed_L_0 / Sed_L_eff[valid]

            return burial_corr

        def _element_foc(idx, fallback):
            """
            Return per-element f_OC for the selected elements.
            Invalid/non-positive values fall back to the provided config value.
            """
            idx = np.asarray(idx, dtype=np.int64).ravel()
            if idx.size == 0:
                return np.empty(0, dtype=np.float32)

            foc = np.asarray(self.elements.f_OC[idx], dtype=float)
            invalid = (~np.isfinite(foc)) | (foc <= 0.0)
            if np.any(invalid):
                foc = foc.copy()
                foc[invalid] = fallback
            return foc.astype(np.float32, copy=False)

        if transfer_setup == 'organics':
            KOWTref = self.get_config('chemical:transformations:TrefKOW')
            DH_KOC_Sed = self.get_config('chemical:transformations:DeltaH_KOC_Sed')
            DH_KOC_DOM = self.get_config('chemical:transformations:DeltaH_KOC_DOM')
            Setchenow = self.get_config('chemical:transformations:Setchenow')

            idx_DOM = np.flatnonzero(specie == self.num_humcol)
            idx_SPM = np.flatnonzero(specie == self.num_prev)
            idx_SED = np.flatnonzero(specie == self.num_srev)

            psrev_active = hasattr(self, 'num_psrev') and self.get_config('chemical:species:Particle_slowly_reversible')
            ssrev_active = hasattr(self, 'num_ssrev') and self.get_config('chemical:species:Sediment_slowly_reversible')

            idx_PSREV = np.flatnonzero(specie == self.num_psrev) if psrev_active else np.empty(0, dtype=np.int64)
            idx_SSREV = np.flatnonzero(specie == self.num_ssrev) if ssrev_active else np.empty(0, dtype=np.int64)

            if idx_SPM.size and psrev_active:
                self.elements.transfer_rates1D[idx_SPM, self.num_psrev] = self.transfer_rates[self.num_prev, self.num_psrev]
            if idx_SED.size and ssrev_active:
                self.elements.transfer_rates1D[idx_SED, self.num_ssrev] = self.transfer_rates[self.num_srev, self.num_ssrev]

            # Config fallbacks for f_OC if element values are invalid/unset
            fOC_SPM_cfg = float(self.get_config('chemical:transformations:fOC_SPM'))
            fOC_sed_cfg = float(self.get_config('chemical:transformations:fOC_sed'))

            # Needed below for KOC/Kd
            Org2C = 0.526
            KOW = 10 ** self.get_config('chemical:transformations:LogKOW')
            eps = 1e-30

            # --- DOM / particle / sediment KOC definitions ---
            if diss == 'nondiss':
                KOC_sed = self.get_config("chemical:transformations:KOC_sed")
                if KOC_sed < 0:
                    KOC_sed = 2.62 * KOW ** 0.82  # L/kgOC

                KOC_SPM = KOC_sed

                KOC_DOM = self.get_config("chemical:transformations:KOC_DOM")
                if KOC_DOM < 0:
                    KOC_DOM = 2.88 * KOW ** 0.67  # L/kgOC

                # DOM -> LMM : unchanged (DOM uses Org2C, not element f_OC)
                if idx_DOM.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_DOM)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_DOM)

                    Kd_DOM = np.maximum(KOC_DOM * Org2C, eps)
                    k_des_DOM = self.k_ads / Kd_DOM

                    self.elements.transfer_rates1D[idx_DOM, self.num_lmm] = (
                        k_des_DOM /
                        self.tempcorr("Arrhenius", DH_KOC_DOM, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

                # Particle reversible -> LMM : per-element f_OC
                if idx_SPM.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_SPM)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_SPM)
                    foc_spm = _element_foc(idx_SPM, fOC_SPM_cfg)

                    Kd_SPM = np.maximum(KOC_SPM * foc_spm, eps)
                    k_des_SPM = self.k_ads / Kd_SPM

                    self.elements.transfer_rates1D[idx_SPM, self.num_lmm] = (
                        k_des_SPM /
                        self.tempcorr("Arrhenius", DH_KOC_Sed, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

                # Sediment reversible -> LMM : per-element f_OC
                if idx_SED.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_SED)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_SED)
                    foc_sed = _element_foc(idx_SED, fOC_sed_cfg)

                    Kd_sed = np.maximum(KOC_sed * foc_sed, eps)
                    k_des_sed = self.k_ads / Kd_sed

                    self.elements.transfer_rates1D[idx_SED, self.num_lmm] = (
                        k_des_sed /
                        self.tempcorr("Arrhenius", DH_KOC_Sed, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

                # Slowly reversible compartments unchanged
                if idx_PSREV.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_PSREV)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_PSREV)
                    self.elements.transfer_rates1D[idx_PSREV, self.num_prev] = (
                        self.k53_0 /
                        self.tempcorr("Arrhenius", DH_KOC_Sed, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

                if idx_SSREV.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_SSREV)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_SSREV)
                    self.elements.transfer_rates1D[idx_SSREV, self.num_srev] = (
                        self.k64_0 /
                        self.tempcorr("Arrhenius", DH_KOC_Sed, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

            else:
                pKa_acid = self.get_config('chemical:transformations:pKa_acid')
                if pKa_acid < 0 and diss in ['acid', 'amphoteric']:
                    raise ValueError("pKa_acid must be positive")

                pKa_base = self.get_config('chemical:transformations:pKa_base')
                if pKa_base < 0 and diss in ['base', 'amphoteric']:
                    raise ValueError("pKa_base must be positive")

                KOC_sed_n = self.get_config('chemical:transformations:KOC_sed')
                if KOC_sed_n < 0:
                    if diss == 'acid':
                        KOC_sed_n = 10 ** ((0.54 * np.log10(KOW)) + 1.11)
                    elif diss == 'base':
                        KOC_sed_n = 10 ** ((0.37 * np.log10(KOW)) + 1.70)
                    elif diss == 'amphoteric':
                        KOC_sed_n = 10 ** ((0.50 * np.log10(KOW)) + 1.13)

                KOC_sed_acid = self.get_config('chemical:transformations:KOC_sed_acid')
                if KOC_sed_acid < 0:
                    KOC_sed_acid = 10 ** (0.11 * np.log10(KOW) + 1.54)

                KOC_sed_base = self.get_config('chemical:transformations:KOC_sed_base')
                if KOC_sed_base < 0:
                    KOC_sed_base = 10.0 ** ((pKa_base ** 0.65) * ((KOW / (KOW + 1.0)) ** 0.14))

                KOC_DOM_n = self.get_config('chemical:transformations:KOC_DOM')
                if KOC_DOM_n < 0:
                    KOC_DOM_n = (0.08 * KOW) / Org2C

                KOC_DOM_acid = self.get_config('chemical:transformations:KOC_DOM_acid')
                if KOC_DOM_acid < 0:
                    KOC_DOM_acid = (0.08 * 10 ** (np.log10(KOW) - 3.5)) / Org2C

                KOC_DOM_base = self.get_config('chemical:transformations:KOC_DOM_base')
                if KOC_DOM_base < 0:
                    KOC_DOM_base = (0.08 * 10 ** (np.log10(KOW) - 3.5)) / Org2C

                # DOM -> LMM : unchanged (DOM uses Org2C, not element f_OC)
                if idx_DOM.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_DOM)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_DOM)
                    pH = self._env_array('sea_water_ph_reported_on_total_scale', 8.1, idx=idx_DOM)

                    KOC_DOM_loc = self.koc_updated(
                        KOC_DOM_n, KOC_DOM_acid, KOC_DOM_base,
                        pH=pH, diss=diss, pKa_acid=pKa_acid, pKa_base=pKa_base
                    )
                    Kd_DOM = np.maximum(KOC_DOM_loc * Org2C, eps)
                    k_des_DOM = self.k_ads / Kd_DOM

                    self.elements.transfer_rates1D[idx_DOM, self.num_lmm] = (
                        k_des_DOM /
                        self.tempcorr("Arrhenius", DH_KOC_DOM, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

                # Particle reversible -> LMM : per-element f_OC
                if idx_SPM.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_SPM)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_SPM)
                    pH = self._env_array('sea_water_ph_reported_on_total_scale', 8.1, idx=idx_SPM)
                    foc_spm = _element_foc(idx_SPM, fOC_SPM_cfg)

                    KOC_SPM_loc = self.koc_updated(
                        KOC_sed_n, KOC_sed_acid, KOC_sed_base,
                        pH=pH, diss=diss, pKa_acid=pKa_acid, pKa_base=pKa_base
                    )
                    Kd_SPM = np.maximum(KOC_SPM_loc * foc_spm, eps)
                    k_des_SPM = self.k_ads / Kd_SPM

                    self.elements.transfer_rates1D[idx_SPM, self.num_lmm] = (
                        k_des_SPM /
                        self.tempcorr("Arrhenius", DH_KOC_Sed, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

                # Sediment reversible -> LMM : per-element f_OC
                if idx_SED.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_SED)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_SED)
                    pH = self._env_array('pH_sediment', 6.9, idx=idx_SED)
                    foc_sed = _element_foc(idx_SED, fOC_sed_cfg)

                    KOC_sed_loc = self.koc_updated(
                        KOC_sed_n, KOC_sed_acid, KOC_sed_base,
                        pH=pH, diss=diss, pKa_acid=pKa_acid, pKa_base=pKa_base
                    )
                    Kd_sed = np.maximum(KOC_sed_loc * foc_sed, eps)
                    k_des_sed = self.k_ads / Kd_sed

                    self.elements.transfer_rates1D[idx_SED, self.num_lmm] = (
                        k_des_sed /
                        self.tempcorr("Arrhenius", DH_KOC_Sed, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

                # Slowly reversible compartments unchanged
                if idx_PSREV.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_PSREV)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_PSREV)
                    self.elements.transfer_rates1D[idx_PSREV, self.num_prev] = (
                        self.k53_0 /
                        self.tempcorr("Arrhenius", DH_KOC_Sed, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

                if idx_SSREV.size:
                    T = self._env_array('sea_water_temperature', 10.0, idx=idx_SSREV)
                    S = self._env_array('sea_water_salinity', 34.0, idx=idx_SSREV)
                    self.elements.transfer_rates1D[idx_SSREV, self.num_srev] = (
                        self.k64_0 /
                        self.tempcorr("Arrhenius", DH_KOC_Sed, T, KOWTref) /
                        self.salinitycorr(Setchenow, T, S)
                    )

            # Adsorption-side rates remain unchanged in this formulation
            idx_LMM = np.flatnonzero(specie == self.num_lmm)
            if idx_LMM.size:
                concSPM = self._spm_g_m3(idx_LMM) * 1e-6
                self.elements.transfer_rates1D[idx_LMM, self.num_prev] = self.k_ads * concSPM

                concDOM = self._doc_mmolkg(idx_LMM) * 12e-3 / 1.025 / 0.526 * 1e-3
                self.elements.transfer_rates1D[idx_LMM, self.num_humcol] = self.k_ads * concDOM

                if hasattr(self, 'num_srev'):
                    k14_corr, interaction_cutoff, _, _, _, water_depth = _local_k14_and_cutoff(idx_LMM)
                    rate_local = self.transfer_rates[self.num_lmm, self.num_srev] * k14_corr
                    Zmin = -water_depth
                    dist_to_seabed = self._z_array(idx_LMM) - Zmin
                    rate_local[dist_to_seabed > interaction_cutoff] = 0.0
                    self.elements.transfer_rates1D[idx_LMM, self.num_srev] = rate_local

        elif transfer_setup == 'metals':
            idx_LMM = np.flatnonzero(specie == self.num_lmm)
            if idx_LMM.size:
                concSPM = self._spm_g_m3(idx_LMM) * 1e-3

                Kd0 = self.get_config('chemical:transformations:Kd')
                S0 = self.get_config('chemical:transformations:S0')
                Dc = self.get_config('chemical:transformations:Dc')

                sed_dens = self.get_config('chemical:sediment:density')
                sed_f = self.get_config('chemical:sediment:effective_fraction')
                sed_phi = self.get_config('chemical:sediment:corr_factor')
                sed_poro = self.get_config('chemical:sediment:porosity')

                salinity = self._env_array('sea_water_salinity', 34.0, idx=idx_LMM)
                if S0 > 0:
                    Kd = Kd0 * (S0 + salinity) / S0
                else:
                    Kd = np.full(idx_LMM.size, Kd0, dtype=np.float32)

                self.elements.transfer_rates1D[idx_LMM, self.num_prev] = Dc * Kd * concSPM

                if hasattr(self, 'num_srev'):
                    k14_corr, interaction_cutoff, Sed_H_int, Sed_L_eff, H_rate_eff, water_depth = _local_k14_and_cutoff(idx_LMM)

                    k14_local = np.zeros_like(Sed_L_eff, dtype=np.float32)
                    valid = (Sed_L_eff > 0) & (H_rate_eff > 0)
                    k14_local[valid] = (
                        Dc * Kd[valid] * Sed_L_eff[valid] * sed_dens *
                        (1.0 - sed_poro) * sed_f * sed_phi / H_rate_eff[valid])

                    Zmin = -water_depth
                    dist_to_seabed = self._z_array(idx_LMM) - Zmin
                    k14_local[dist_to_seabed > interaction_cutoff] = 0.0

                    self.elements.transfer_rates1D[idx_LMM, self.num_srev] = k14_local

        elif transfer_setup in ('137Cs_rev', 'custom'):
            if hasattr(self, 'num_lmm') and hasattr(self, 'num_srev'):
                idx_LMM = np.flatnonzero(specie == self.num_lmm)
                if idx_LMM.size:
                    k14_corr, interaction_cutoff, _, _, _, water_depth = _local_k14_and_cutoff(idx_LMM)
                    rate_local = self.transfer_rates[self.num_lmm, self.num_srev] * k14_corr

                    Zmin = -water_depth
                    dist_to_seabed = self._z_array(idx_LMM) - Zmin
                    rate_local[dist_to_seabed > interaction_cutoff] = 0.0

                    self.elements.transfer_rates1D[idx_LMM, self.num_srev] = rate_local

        if hasattr(self, 'num_sburied'):
            if hasattr(self, 'num_srev'):
                idx = np.flatnonzero(specie == self.num_srev)
                if idx.size:
                    burial_corr = _local_burial_corr(idx)
                    self.elements.transfer_rates1D[idx, self.num_sburied] = (
                        self.transfer_rates[self.num_srev, self.num_sburied] * burial_corr)

            if hasattr(self, 'num_ssrev'):
                idx = np.flatnonzero(specie == self.num_ssrev)
                if idx.size:
                    burial_corr = _local_burial_corr(idx)
                    self.elements.transfer_rates1D[idx, self.num_sburied] = (
                        self.transfer_rates[self.num_ssrev, self.num_sburied] * burial_corr)

            if hasattr(self, 'num_sirrev'):
                idx = np.flatnonzero(specie == self.num_sirrev)
                if idx.size:
                    burial_corr = _local_burial_corr(idx)
                    self.elements.transfer_rates1D[idx, self.num_sburied] = (
                        self.transfer_rates[self.num_sirrev, self.num_sburied] * burial_corr)

    def _compact_transition_arrays(self, changed_idx=None, old_species=None, new_species=None):
        """
        Normalize compact transition inputs.
        Returns only true species changes:
            idx, sp_in, sp_out
        Each returned array has the same length.
        """
        if changed_idx is None and old_species is None and new_species is None:
            return None, None, None

        if changed_idx is not None and old_species is not None and new_species is None:
            raise TypeError(
                "Compact transition methods now expect "
                "changed_idx, old_species, and new_species. "
                "Do not call them with full sp_in, sp_out arrays."
            )

        if changed_idx is None or old_species is None or new_species is None:
            raise ValueError(
                "changed_idx, old_species, and new_species must all be provided."
            )

        idx = np.asarray(changed_idx, dtype=np.int64).ravel()
        sp_in = np.asarray(old_species, dtype=int).ravel()
        sp_out = np.asarray(new_species, dtype=int).ravel()

        if not (idx.size == sp_in.size == sp_out.size):
            raise ValueError(
                "changed_idx, old_species, and new_species must have the same length."
            )

        changed = sp_in != sp_out

        return idx[changed], sp_in[changed], sp_out[changed]

    def _assign_bed_critstress_factor(self, idx):
        '''
        Assign persistent heterogeneity factors for bed critical stress.
        If sub-grid heterogeneity is enabled:
            critstress_factor ~ lognormal(mean=1)
        otherwise:
            critstress_factor = 1
        This factor multiplies the nominal resuspension critical stress and is used to
        represent unresolved spatial variability in bed erodibility.
        '''
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return

        if self.get_config('chemical:sediment:use_critstress_heterogeneity'):
            self.elements.critstress_factor[idx] = self._sample_critstress_factor(idx.size)
        else:
            self.elements.critstress_factor[idx] = 1.0

    def update_chemical_fOC(self, changed_idx=None, old_species=None, new_species=None):
        """
        Update element organic-carbon fractions only for elements that changed species.

        Compact transition input:
            changed_idx : global element indices that changed species
            old_species : species before transition
            new_species : species after transition

        Rules:
          1) Entering particle family from a non-particle family:
                 f_OC <- local particle f_OC
          2) Entering sediment family from a non-sediment family:
                 f_OC <- local sediment f_OC
          3) Transitions within the same family preserve f_OC.
          4) Dissolved / DOM-like species do not receive a new f_OC.
        """
        idx_all, sp_in, sp_out = self._compact_transition_arrays(
            changed_idx=changed_idx,
            old_species=old_species,
            new_species=new_species,
        )

        if idx_all is None or idx_all.size == 0:
            return

        particle_species = []
        if hasattr(self, 'num_prev'):
            particle_species.append(self.num_prev)
        if hasattr(self, 'num_psrev'):
            particle_species.append(self.num_psrev)
        if hasattr(self, 'num_pirrev'):
            particle_species.append(self.num_pirrev)

        sediment_species = []
        if hasattr(self, 'num_srev'):
            sediment_species.append(self.num_srev)
        if hasattr(self, 'num_ssrev'):
            sediment_species.append(self.num_ssrev)
        if hasattr(self, 'num_sirrev'):
            sediment_species.append(self.num_sirrev)
        if hasattr(self, 'num_sburied'):
            sediment_species.append(self.num_sburied)

        def _isin(values, species):
            if len(species) == 0:
                return np.zeros(values.shape, dtype=bool)
            return np.isin(values, species)

        was_particle = _isin(sp_in, particle_species)
        is_particle = _isin(sp_out, particle_species)

        was_sediment = _isin(sp_in, sediment_species)
        is_sediment = _isin(sp_out, sediment_species)

        entered_particle = is_particle & (~was_particle)
        if np.any(entered_particle):
            idx = idx_all[entered_particle]
            self.elements.f_OC[idx] = self._local_particle_fOC(idx)

        entered_sediment = is_sediment & (~was_sediment)
        if np.any(entered_sediment):
            idx = idx_all[entered_sediment]
            self.elements.f_OC[idx] = self._local_sediment_fOC(idx)

    def update_chemical_diameter(self, changed_idx=None, old_species=None, new_species=None):
        """
        Update particle diameter when an element changes species.

        changed_idx : array-like of int
            Global element indices that changed species.
        old_species : array-like of int
            Species before the transition, same length as changed_idx.
        new_species : array-like of int
            Species after the transition, same length as changed_idx.

        Carrier-diameter behavior
        -------------------------
        1) Seeded elements start with d50 = diameter.
        2) Suspended-particle phase:
               terminal velocity uses diameter.
        3) Particle -> sediment transitions interpreted as deposition:
               keep the previous diameter unchanged.
        4) Direct non-particle -> sediment association:
               diameter <- local bed d50 (mapped sea_floor_d50 if available,
               otherwise config chemical:sediment:d50)
        5) Sediment -> particle transitions interpreted as resuspension:
               keep the previous diameter unchanged.
        6) Entering DOC-like carrier species:
               diameter ~ lognormal(median=doc_particle_diameter, sigma=sigma_doc_ln)
        7) Entering dissolved / colloid species:
               diameter = dissolved_diameter.

        Note
        ----
        Diameter controls terminal velocity, while per-element d50 controls any
        d50-based bed-threshold calculation. Deposition and resuspension preserve the
        current diameter. Only direct bed-associated sorption / direct sediment
        association rewrites diameter to the local sediment d50. In USER mode,
        mapped sea_floor_d50 is rejected and chemical:sediment:d50 must equal
        chemical:particle_diameter, so these sediment-association updates remain
        spatially uniform and consistent.

        Updated families
        ----------------
        Particle family:            prev, psrev, pirrev
        Sediment family:            srev, ssrev, sirrev, sburied
        Dissolved family:           lmm, lmmanion, lmmcation, colloid
        DOC-like family:            humic colloid, polymer
        """
        if changed_idx is None or old_species is None or new_species is None:
            return

        idx_all = np.asarray(changed_idx, dtype=np.int64).ravel()
        sp_in = np.asarray(old_species, dtype=int).ravel()
        sp_out = np.asarray(new_species, dtype=int).ravel()

        if idx_all.size == 0:
            return

        if not (idx_all.size == sp_in.size == sp_out.size):
            raise ValueError(
                "changed_idx, old_species, and new_species must have the same length."
            )

        # Keep only actual species changes.
        changed = sp_in != sp_out
        if not np.any(changed):
            return

        idx_all = idx_all[changed]
        sp_in = sp_in[changed]
        sp_out = sp_out[changed]

        dia_part = float(self.get_config('chemical:particle_diameter'))
        dia_doc = float(self.get_config('chemical:doc_particle_diameter'))
        dia_diss = float(self.get_config('chemical:dissolved_diameter'))

        sigma_part_ln = float(self.get_config('chemical:particle_diameter_uncertainty'))
        sigma_doc_ln = float(self.get_config('chemical:doc_particle_diameter_uncertainty'))

        particle_species = []
        if hasattr(self, 'num_prev'):
            particle_species.append(self.num_prev)
        if hasattr(self, 'num_psrev'):
            particle_species.append(self.num_psrev)
        if hasattr(self, 'num_pirrev'):
            particle_species.append(self.num_pirrev)

        sediment_species = []
        if hasattr(self, 'num_srev'):
            sediment_species.append(self.num_srev)
        if hasattr(self, 'num_ssrev'):
            sediment_species.append(self.num_ssrev)
        if hasattr(self, 'num_sirrev'):
            sediment_species.append(self.num_sirrev)
        if hasattr(self, 'num_sburied'):
            sediment_species.append(self.num_sburied)

        dissolved_species = []
        if hasattr(self, 'num_lmm'):
            dissolved_species.append(self.num_lmm)
        if hasattr(self, 'num_lmmanion'):
            dissolved_species.append(self.num_lmmanion)
        if hasattr(self, 'num_lmmcation'):
            dissolved_species.append(self.num_lmmcation)
        if hasattr(self, 'num_col'):
            dissolved_species.append(self.num_col)

        doclike_species = []
        if hasattr(self, 'num_humcol'):
            doclike_species.append(self.num_humcol)
        if hasattr(self, 'num_polymer'):
            doclike_species.append(self.num_polymer)

        def _isin(values, species):
            if len(species) == 0:
                return np.zeros(values.shape, dtype=bool)
            return np.isin(values, species)

        def _assign_lognormal(idx, median_diameter, sigma_ln, label):
            idx = np.asarray(idx, dtype=np.int64).ravel()
            n = idx.size
            if n == 0:
                return

            self.elements.diameter[idx] = self._sample_lognormal_diameter(
                median_diameter=median_diameter, sigma_ln=sigma_ln,
                n=n,)
            logger.debug("Updated %s diameter for %s elements", label, n)

        def _assign_constant_or_array(idx, values, label):
            idx = np.asarray(idx, dtype=np.int64).ravel()
            n = idx.size
            if n == 0:
                return

            arr = np.asarray(values, dtype=float)

            if arr.ndim == 0:
                self.elements.diameter[idx] = float(arr)
            else:
                if arr.size != n:
                    raise ValueError(
                        f"{label} diameter array has size {arr.size}, expected {n}"
                    )
                self.elements.diameter[idx] = arr

            logger.debug("Updated %s diameter for %s elements", label, n)

        was_particle = _isin(sp_in, particle_species)
        is_particle = _isin(sp_out, particle_species)

        was_sediment = _isin(sp_in, sediment_species)
        is_sediment = _isin(sp_out, sediment_species)

        was_dissolved = _isin(sp_in, dissolved_species)
        is_dissolved = _isin(sp_out, dissolved_species)

        was_doclike = _isin(sp_in, doclike_species)
        is_doclike = _isin(sp_out, doclike_species)

        # Entering suspended-particle family.
        entered_particle = is_particle & (~was_particle)
        if np.any(entered_particle):
            # Sediment -> particle: resuspension, preserve diameter.
            resuspended_idx = idx_all[entered_particle & was_sediment]
            if resuspended_idx.size:
                logger.debug(
                    "Preserved diameter for %s resuspended particle elements",
                    resuspended_idx.size,
                )

            # Other -> particle: assign new carrier diameter.
            from_other = entered_particle & (~was_sediment)
            if np.any(from_other):
                from_other_idx = idx_all[from_other]
                from_other_old_sp = sp_in[from_other]

                if hasattr(self, 'num_humcol'):
                    from_humic = from_other_old_sp == self.num_humcol
                else:
                    from_humic = np.zeros(from_other_old_sp.shape, dtype=bool)

                humic_idx = from_other_idx[from_humic]
                other_idx = from_other_idx[~from_humic]

                if humic_idx.size:
                    _assign_lognormal(
                        humic_idx,
                        dia_doc,
                        sigma_doc_ln,
                        'particle-from-humic',
                    )

                if other_idx.size:
                    _assign_lognormal(
                        other_idx,
                        dia_part,
                        sigma_part_ln,
                        'particle',
                    )

        # Entering sediment family.
        entered_sediment = is_sediment & (~was_sediment)
        if np.any(entered_sediment):
            # Particle -> sediment: deposition, preserve diameter.
            deposited_idx = idx_all[entered_sediment & was_particle]
            if deposited_idx.size:
                logger.debug(
                    "Preserved diameter for %s deposited sediment elements",
                    deposited_idx.size,
                )

            # Non-particle -> sediment: direct bed association, use local bed d50.
            direct_assoc_idx = idx_all[entered_sediment & (~was_particle)]
            if direct_assoc_idx.size:
                local_bed_d50 = self._local_bed_d50(idx=direct_assoc_idx)
                _assign_constant_or_array(
                    direct_assoc_idx,
                    local_bed_d50,
                    'sediment-local-bed-d50',
                )

        # Entering dissolved / colloid family.
        entered_dissolved = is_dissolved & (~was_dissolved)
        if np.any(entered_dissolved):
            _assign_constant_or_array(
                idx_all[entered_dissolved],
                dia_diss,
                'dissolved',
            )

        # Entering DOC-like family.
        entered_doclike = is_doclike & (~was_doclike)
        if np.any(entered_doclike):
            entered_doclike_idx = idx_all[entered_doclike]
            entered_doclike_new_sp = sp_out[entered_doclike]

            if hasattr(self, 'num_humcol'):
                humic_idx = entered_doclike_idx[entered_doclike_new_sp == self.num_humcol]
                if humic_idx.size:
                    _assign_lognormal(
                        humic_idx,
                        dia_doc,
                        sigma_doc_ln,
                        'humic colloid',
                    )

            if hasattr(self, 'num_polymer'):
                polymer_idx = entered_doclike_idx[entered_doclike_new_sp == self.num_polymer]
                if polymer_idx.size:
                    _assign_lognormal(
                        polymer_idx,
                        dia_doc,
                        sigma_doc_ln,
                        'polymer',
                    )

    def update_chemical_d50(self, changed_idx=None, old_species=None, new_species=None):
        """
        Update per-element d50 only for elements that changed species.

        Rules
        -----
        1) Sediment -> particle resuspension:
               preserve bed-associated d50.
        2) Particle -> sediment deposition:
               preserve d50.
        3) Non-sediment -> particle:
               assign particle_diameter as d50.
        4) Non-particle -> sediment:
               assign local bed d50.
        5) Entering dissolved-like species:
               d50 = 0.
        """
        if changed_idx is None or old_species is None or new_species is None:
            return

        idx_all = np.asarray(changed_idx, dtype=np.int64).ravel()
        sp_in = np.asarray(old_species, dtype=int).ravel()
        sp_out = np.asarray(new_species, dtype=int).ravel()

        if idx_all.size == 0:
            return

        if not (idx_all.size == sp_in.size == sp_out.size):
            raise ValueError(
                "changed_idx, old_species, and new_species must have the same length."
            )

        # Keep only actual species changes.
        changed = sp_in != sp_out
        if not np.any(changed):
            return

        idx_all = idx_all[changed]
        sp_in = sp_in[changed]
        sp_out = sp_out[changed]

        particle_species = []
        if hasattr(self, 'num_prev'):
            particle_species.append(self.num_prev)
        if hasattr(self, 'num_psrev'):
            particle_species.append(self.num_psrev)
        if hasattr(self, 'num_pirrev'):
            particle_species.append(self.num_pirrev)

        sediment_species = []
        if hasattr(self, 'num_srev'):
            sediment_species.append(self.num_srev)
        if hasattr(self, 'num_ssrev'):
            sediment_species.append(self.num_ssrev)
        if hasattr(self, 'num_sirrev'):
            sediment_species.append(self.num_sirrev)
        if hasattr(self, 'num_sburied'):
            sediment_species.append(self.num_sburied)

        dissolved_like_species = []
        if hasattr(self, 'num_lmm'):
            dissolved_like_species.append(self.num_lmm)
        if hasattr(self, 'num_lmmanion'):
            dissolved_like_species.append(self.num_lmmanion)
        if hasattr(self, 'num_lmmcation'):
            dissolved_like_species.append(self.num_lmmcation)
        if hasattr(self, 'num_col'):
            dissolved_like_species.append(self.num_col)
        if hasattr(self, 'num_humcol'):
            dissolved_like_species.append(self.num_humcol)
        if hasattr(self, 'num_polymer'):
            dissolved_like_species.append(self.num_polymer)

        def _isin(values, species):
            if len(species) == 0:
                return np.zeros(values.shape, dtype=bool)
            return np.isin(values, species)

        was_particle = _isin(sp_in, particle_species)
        is_particle = _isin(sp_out, particle_species)

        was_sediment = _isin(sp_in, sediment_species)
        is_sediment = _isin(sp_out, sediment_species)

        was_dissolved = _isin(sp_in, dissolved_like_species)
        is_dissolved = _isin(sp_out, dissolved_like_species)

        # Entering particle family.
        entered_particle = is_particle & (~was_particle)
        if np.any(entered_particle):
            # Sediment -> particle: resuspension, preserve d50.
            resuspended_idx = idx_all[entered_particle & was_sediment]
            if resuspended_idx.size:
                logger.debug(
                    "Preserved d50 for %s resuspended particle elements",
                    resuspended_idx.size,
                )

            # Other -> particle: assign configured particle d50.
            from_other_idx = idx_all[entered_particle & (~was_sediment)]
            if from_other_idx.size:
                self._assign_d50_to_elements(
                    from_other_idx,
                    particle_d50=float(self.get_config('chemical:particle_diameter')),
                    dissolved_value=0.0,
                )

        # Entering sediment family.
        entered_sediment = is_sediment & (~was_sediment)
        if np.any(entered_sediment):
            # Particle -> sediment: deposition, preserve d50.
            deposited_idx = idx_all[entered_sediment & was_particle]
            if deposited_idx.size:
                logger.debug(
                    "Preserved d50 for %s deposited sediment elements",
                    deposited_idx.size,
                )

            # Other -> sediment: assign local bed d50.
            direct_assoc_idx = idx_all[entered_sediment & (~was_particle)]
            if direct_assoc_idx.size:
                self._assign_d50_to_elements(
                    direct_assoc_idx,
                    sediment_d50=self._local_bed_d50(idx=direct_assoc_idx),
                    dissolved_value=0.0,
                )

        # Entering dissolved-like family.
        entered_dissolved = is_dissolved & (~was_dissolved)
        if np.any(entered_dissolved):
            self.elements.d50[idx_all[entered_dissolved]] = 0.0

    def sorption_to_sediments(self, changed_idx=None, old_species=None, new_species=None):
        """
        Move newly sorbed dissolved elements onto the seabed.

        Compact transition input:
            changed_idx : global element indices that changed species
            old_species : species before transition
            new_species : species after transition

        If an element transitions:
            LMM / LMMcation -> Sediment reversible

        then:
            z      <- -local_water_depth
            moving <- 0

        Only changed elements are inspected and modified.
        """
        if not hasattr(self, 'num_srev'):
            logger.debug(
                "No sediment reversible specie initiated, "
                "sorption_to_sediments was skipped"
            )
            return

        idx_all, sp_in, sp_out = self._compact_transition_arrays(
            changed_idx=changed_idx,
            old_species=old_species,
            new_species=new_species,
        )

        if idx_all is None or idx_all.size == 0:
            return

        touched_parts = []

        if self.get_config('chemical:species:LMM') and hasattr(self, 'num_lmm'):
            loc = (sp_out == self.num_srev) & (sp_in == self.num_lmm)
            if np.any(loc):
                idx = idx_all[loc]
                depth = np.asarray(
                    self._env_array(
                        'sea_floor_depth_below_sea_level',
                        10000.0,
                        idx=idx,
                    ),
                    dtype=float,
                )
                depth = np.maximum(depth, 0.0)

                self.elements.z[idx] = -depth
                self.elements.moving[idx] = 0
                touched_parts.append(idx)

        if self.get_config('chemical:species:LMMcation') and hasattr(self, 'num_lmmcation'):
            loc = (sp_out == self.num_srev) & (sp_in == self.num_lmmcation)
            if np.any(loc):
                idx = idx_all[loc]
                depth = np.asarray(
                    self._env_array(
                        'sea_floor_depth_below_sea_level',
                        10000.0,
                        idx=idx,
                    ),
                    dtype=float,
                )
                depth = np.maximum(depth, 0.0)

                self.elements.z[idx] = -depth
                self.elements.moving[idx] = 0
                touched_parts.append(idx)

        # Avoid global O(N) z clipping.
        # Only these touched elements could have been modified here.
        if touched_parts:
            touched = np.concatenate(touched_parts)
            above_surface = self.elements.z[touched] > 0.0

            if np.any(above_surface):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Number of sorbed elements lowered down to sea surface: %s",
                        int(np.count_nonzero(above_surface)),
                    )
                self.elements.z[touched[above_surface]] = 0.0

    def desorption_from_sediments(self, changed_idx=None, old_species=None, new_species=None):
        """
        Move newly desorbed sediment elements back into the water column.

        Compact transition input:
            changed_idx : global element indices that changed species
            old_species : species before transition
            new_species : species after transition

        If an element transitions:
            Sediment reversible -> LMM / LMMcation
        then:
            z      <- -local_water_depth + desorption_depth
            moving <- 1

        Optional Gaussian perturbation:
            z <- z + Normal(0, desorption_depth_uncert)
        Only changed elements are inspected and modified.
        """
        if not hasattr(self, 'num_srev'):
            logger.debug(
                "No sediment reversible specie initiated, "
                "desorption_from_sediments was skipped"
            )
            return

        idx_all, sp_in, sp_out = self._compact_transition_arrays(
            changed_idx=changed_idx,
            old_species=old_species,
            new_species=new_species,
        )

        if idx_all is None or idx_all.size == 0:
            return

        desorption_depth = float(
            self.get_config('chemical:sediment:desorption_depth')
        )
        std = float(
            self.get_config('chemical:sediment:desorption_depth_uncert')
        )

        def _apply_desorption(idx):
            """
            Apply sediment -> dissolved release for a compact index subset.
            """
            idx = np.asarray(idx, dtype=np.int64).ravel()

            if idx.size == 0:
                return

            depth = np.asarray(
                self._env_array(
                    'sea_floor_depth_below_sea_level',
                    10000.0,
                    idx=idx,
                ),
                dtype=float,
            )
            depth = np.maximum(depth, 0.0)
            # In shallow water, do not release above a distance larger than local depth.
            desorption_depth_eff = np.minimum(desorption_depth, depth)
            z_new = -depth + desorption_depth_eff

            if std > 0:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Adding uncertainty for desorption from sediments: %s m",
                        std,
                    )
                z_new += np.random.normal(0.0, std, idx.size)
            # Keep desorbed elements inside the local water column.
            z_new = np.maximum(z_new, -depth)
            z_new = np.minimum(z_new, 0.0)

            self.elements.z[idx] = z_new
            self.elements.moving[idx] = 1

        if self.get_config('chemical:species:LMM') and hasattr(self, 'num_lmm'):
            loc = (sp_out == self.num_lmm) & (sp_in == self.num_srev)
            if np.any(loc):
                _apply_desorption(idx_all[loc])

        if self.get_config('chemical:species:LMMcation') and hasattr(self, 'num_lmmcation'):
            loc = (sp_out == self.num_lmmcation) & (sp_in == self.num_srev)
            if np.any(loc):
                _apply_desorption(idx_all[loc])

    ###########################################################################
    # Main partitioning function
    ###########################################################################

    def update_partitioning(self):
        '''
        Apply stochastic dynamic partitioning for one timestep.

        This routine uses a one-jump-per-timestep approximation to the
        continuous-time Markov process defined by the transfer-rate matrix.
        Its first-event probability and destination draw are exact for rates
        frozen at the start of the step; subsequent within-step jumps are omitted.
        Timestep convergence should be checked when outgoing_rate * dt is large.
          - process elements grouped by their current/source species
          - for each source species, use only its possible destination species
          - avoid building full temporary arrays of shape N x nspecies
          - avoid building K_sel, probs, and cdf over all species

        Mathematical steps for each element e:
        1) Current outgoing rates
               K[e, j] = rate from current species to species j   [1/s]
        2) Total leaving rate
               k_tot[e] = sum_j K[e, j]
        3) Probability that at least one transition occurs during dt
               p_any[e] = 1 - exp(-k_tot[e] * dt)
           Numerically this is evaluated as:
               p_any[e] = -expm1(-k_tot[e] * dt)
           which is more accurate than 1 - exp(-x) for small x.
        4) Monte Carlo draw for whether a transition occurs
               phaseshift[e] ~ Bernoulli(p_any[e])
        5) Conditional destination probabilities, given that a transition occurred
               P(dest=j | transition) = K[e, j] / k_tot[e]
           In this implementation we do not explicitly form probabilities.
           Instead we sample from cumulative raw rates, which is equivalent:
               P(dest=j | transition) = K[e, j] / sum_j K[e, j]
        6) Use inverse-transform sampling on the cumulative rate distribution
           to choose the destination species.

        Post-processing after species are updated:
          - assign critstress heterogeneity to elements entering sediment pools
          - update cumulative transition counters ntransformations[i, j]
          - update diameter via update_chemical_diameter()
          - update d50 via update_chemical_d50()
          - update f_OC via update_chemical_fOC()
          - update z/moving state for sorption/desorption relative to sediments
        '''
        dt = float(self.time_step.total_seconds())
        n_active = self.num_elements_active()

        if n_active == 0:
            return

        if not hasattr(self, '_transition_destinations'):
            self._build_transition_destination_cache()

        # Store the initial species for post-processing and bookkeeping.
        specie_in = np.asarray(self.elements.specie, dtype=np.int32).copy()

        # K: per-element transition rates between species.
        # Shape is still (N, nspecies), units [1/s].
        # Convention:
        #   for element e, K[e, j] is the rate of jumping
        #   FROM its current species TO species j.
        K = self.elements.transfer_rates1D
        changed_idx_parts = []
        old_species_parts = []
        new_species_parts = []

        # Process one current/source species at a time.
        # Instead of computing:
        #   k_tot = np.sum(K, axis=1)
        # for every element over every species, we only inspect the possible
        # destination columns for the current source species.
        for src in range(self.nspecies):
            dests = self._transition_destinations[src]

            if dests.size == 0:
                continue
            idx_src = np.flatnonzero(specie_in == src)
            if idx_src.size == 0:
                continue
            # Current outgoing rates for elements whose current species is src.
            # rates[m, q] = rate from source species src to destination dests[q]
            # for element idx_src[m].
            rates = K[np.ix_(idx_src, dests)]

            # Total rate of leaving the current state for each selected element:
            #   k_tot[e] = sum_j K[e, j]  [1/s]
            k_tot = rates.sum(axis=1)

            active = k_tot > 0.0
            if not np.any(active):
                continue

            idx_active = idx_src[active]
            rates_active = rates[active]
            k_tot_active = k_tot[active]

            # Probability that at least one transition occurs within dt for a
            # Poisson process:
            #   P(no transition in dt)       = exp(-k_tot * dt)
            #   P(at least one transition)   = 1 - exp(-k_tot * dt)
            # Use -expm1(-x) instead of 1 - exp(-x) for numerical stability.
            p_any = -np.expm1(-k_tot_active * dt)

            # First Monte Carlo draw:
            # decide which elements undergo a phase/species change this step.
            hit = np.random.random(idx_active.size) < p_any
            if not np.any(hit):
                continue

            idx_hit = idx_active[hit]
            rates_hit = rates_active[hit]
            k_tot_hit = k_tot_active[hit]

            # Conditional destination probabilities, given that a transition occurs:
            #   P(dest=j | transition) = K[e, j] / k_tot[e]
            cdf_rates = np.cumsum(rates_hit, axis=1)

            # Second Monte Carlo draw:
            # pick destination species using inverse-transform sampling.
            # Since cdf_rates ends at k_tot, draw u in [0, k_tot).
            u = np.random.random(idx_hit.size) * k_tot_hit

            # chosen_pos is the first index where cdf_rates >= u.
            # Equivalently, (cdf_rates < u).sum gives the count of bins strictly
            # below u, i.e. the selected destination position within dests.
            chosen_pos = (cdf_rates < u[:, None]).sum(axis=1)
            # Safety clamp in case of tiny floating-point deficits.
            chosen_pos = np.minimum(chosen_pos, dests.size - 1)

            # Convert destination positions back to actual species numbers.
            new_species = dests[chosen_pos].astype(np.int32, copy=False)
            changed_idx_parts.append(idx_hit)
            old_species_parts.append(np.full(idx_hit.size, src, dtype=np.int32))
            new_species_parts.append(new_species)

        if not changed_idx_parts:
            logger.info("Number of transformations: 0")
            return

        changed_idx = np.concatenate(changed_idx_parts)
        old_species = np.concatenate(old_species_parts)
        new_species = np.concatenate(new_species_parts)

        ntr = changed_idx.size
        logger.info("Number of transformations: %s", ntr)

        # Apply new species only to transformed elements.
        self.elements.specie[changed_idx] = new_species

        # Assign bed_critstress_factor to elements that entered the sediment pool.
        bed_species = []
        for attr in ('num_srev', 'num_ssrev', 'num_sirrev', 'num_sburied'):
            if hasattr(self, attr):
                bed_species.append(getattr(self, attr))

        if bed_species:
            entered_bed = np.zeros(ntr, dtype=bool)
            for bed_sp in bed_species:
                entered_bed |= (new_species == bed_sp) & (old_species != bed_sp)
            if np.any(entered_bed):
                self._assign_bed_critstress_factor(changed_idx[entered_bed])

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("old species: %s", old_species)
            logger.debug("new species: %s", new_species)

        # Bookkeeping:
        # count transitions iin -> iout among transformed elements.
        #   flat_index = iin * nspecies + iout
        # Counts all pairs in one bincount call.
        flat = old_species.astype(np.int64) * self.nspecies + new_species.astype(np.int64)
        counts = np.bincount(flat, minlength=self.nspecies * self.nspecies)
        self.ntransformations += counts.reshape(self.nspecies, self.nspecies)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Number of transformations total:\n%s", self.ntransformations)

        # Update chemical properties after transformations.
        self.update_chemical_diameter(
            changed_idx=changed_idx, old_species=old_species,
            new_species=new_species,)

        self.update_chemical_d50(
            changed_idx=changed_idx, old_species=old_species,
            new_species=new_species,)

        self.update_chemical_fOC(
            changed_idx=changed_idx, old_species=old_species,
            new_species=new_species,)

        self.sorption_to_sediments(
            changed_idx=changed_idx, old_species=old_species,
            new_species=new_species,)

        self.desorption_from_sediments(
            changed_idx=changed_idx, old_species=old_species,
            new_species=new_species,
        )

    ###########################################################################
    # Helpers for terminal velocity and particle transport properties
    ###########################################################################

    def update_terminal_velocity(self, Tprofiles=None,
                                 Sprofiles=None, z_index=None):
        """Calculate terminal velocity only for elements that can actually settle/rise.

        according to
        S. Sundby (1983): A one-dimensional model for the vertical
        distribution of pelagic fish eggs in the mixed layer
        Deep Sea Research (30) pp. 645-661

        Method copied from ibm.f90 module of LADIM:
        Vikebo, F., S. Sundby, B. Aadlandsvik and O. Otteraa (2007),
        Fish. Oceanogr. (16) pp. 216-228

        - terminal_velocity is set to zero for all elements first
        - only moving elements with diameter > 0 are evaluated
        - immobile sediment/buried elements and dissolved zero-diameter elements
          do not require temperature, salinity, density, viscosity, or Stokes
          settling calculations

      Sign convention:
        W = (1 / mu) * (1 / 18) * g * d^2 * (rho_water - rho_particle)

        If rho_particle > rho_water, then W is negative, corresponding to
        downward settling in the OpenDrift z convention.
        """
        g = 9.81  # m s-2
        # Always reset first.
        # This prevents stale terminal velocities for elements that became immobile
        # or changed to zero diameter since the previous timestep.
        self.elements.terminal_velocity[:] = 0.0

        # Only particles that are moving and have a nonzero diameter can have
        # nonzero terminal velocity.
        idx = np.flatnonzero(
            (np.asarray(self.elements.moving) != 0) &
            (np.asarray(self.elements.diameter) > 0.0)
        )
        if idx.size == 0:
            return

        # Particle properties that determine settling/rising velocity.
        partsize = np.asarray(self.elements.diameter[idx], dtype=float)
        DENSpart = np.asarray(self.elements.density[idx], dtype=float)

        # Prepare interpolation of temperature and salinity profiles only for the
        # selected elements. This avoids interpolating profiles for sedimented,
        # buried, dissolved, or otherwise non-moving zero-diameter elements.
        profile_ref = Tprofiles if Tprofiles is not None else Sprofiles

        if profile_ref is not None:
            if z_index is None:
                from scipy.interpolate import interp1d
                z_i = range(profile_ref.shape[0])
                z_index = interp1d(
                    -self.environment_profiles['z'],
                    z_i,
                    bounds_error=False,
                )
            zi = z_index(-np.asarray(self.elements.z[idx], dtype=float))
            upper = np.maximum(np.floor(zi).astype(np.int64), 0)
            lower = np.minimum(upper + 1, profile_ref.shape[0] - 1)
            weight_upper = 1.0 - (zi - upper)

        # Temperature at selected element positions.
        # If profiles are not passed, use the already interpolated reader values
        # from self.environment. Otherwise interpolate from the provided profiles.
        if Tprofiles is None:
            T0 = np.asarray(self.environment.sea_water_temperature[idx], dtype=float)
        else:
            T0 = (
                Tprofiles[upper, idx] * weight_upper +
                Tprofiles[lower, idx] * (1.0 - weight_upper)
            )
        # Salinity at selected element positions.
        if Sprofiles is None:
            S0 = np.asarray(self.environment.sea_water_salinity[idx], dtype=float)
        else:
            S0 = (
                Sprofiles[upper, idx] * weight_upper +
                Sprofiles[lower, idx] * (1.0 - weight_upper)
            )
        # Water density and particle-water density difference.
        DENSw = self.sea_water_density(T=T0, S=S0)
        dr = DENSw - DENSpart

        # Dynamic viscosity of seawater.
        # Typical value is around 0.0014 kg m-1 s-1.
        my_w = seawater_dynamic_viscosity(T0, S0)

        # Terminal velocity for low Reynolds numbers / Stokes settling:
        W = (1.0 / my_w) * (1.0 / 18.0) * g * partsize**2 * dr
        # Write only selected elements. All others remain zero.
        self.elements.terminal_velocity[idx] = W

    ###########################################################################
    # Helpers for sediment properties and bed maps
    ###########################################################################

    def _user_resuspension_threshold_overrides_d50(self):
        """Return True when USER mode is selected for resuspension critical stress."""
        mode = self.get_config('chemical:sediment:resuspension_critstress_mode')
        return mode == 'USER'

    def _d50_map_reader_present(self):
        """Return True if any active reader advertises a mapped bed-d50 variable."""
        return 'sea_floor_d50' in getattr(self, '_reader_variables', set())

    def _local_bed_d50(self, idx=None):
        """
        Return local bed median grain size d50 [m].

        If USER mode is active, mapped d50 is intentionally ignored and a spatially
        uniform value from chemical:sediment:d50 is returned. In that mode critical
        stress is user-prescribed, mapped d50 is forbidden, and the uniform fallback
        is used wherever a local bed d50 is queried.

        Otherwise the priority is:
          1) environment.sea_floor_d50 if supplied by a reader
          2) config fallback chemical:sediment:d50
        """
        fallback = float(self.get_config('chemical:sediment:d50'))
        if idx is None:
            try:
                n = self.num_elements_active()
            except Exception:
                n = 0
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()
            n = idx.size
        if self._user_resuspension_threshold_overrides_d50():
            return np.full(n, fallback, dtype=float)
        if not hasattr(self, 'environment'):
            return np.full(n, fallback, dtype=float)
        d50 = self._optional_env_array('sea_floor_d50', idx=idx)
        if d50 is None:
            return np.full(n, fallback, dtype=float)
        return self._sanitize_positive_with_fallback(d50, fallback)

    def _local_erodibility_M(self, idx=None):
        """
        Return local cohesive erodibility coefficient M [kg m-2 s-1 Pa-1].

        Priority:
          1) environment.sea_floor_erodibility_M if supplied by a reader
          2) config fallback chemical:sediment:erodibility_M

        Negative or non-finite mapped values are replaced by the config fallback.
        Zero is allowed and locally disables cohesive erosion.
        """
        fallback = float(self.get_config('chemical:sediment:erodibility_M'))
        n = self.num_elements_active() if idx is None else np.asarray(idx, dtype=np.int64).ravel().size
        M = self._optional_env_array('sea_floor_erodibility_M', idx=idx)
        if M is None:
            return np.full(n, fallback, dtype=float)
        out = np.asarray(M, dtype=float).copy()
        invalid = (~np.isfinite(out)) | (out < 0.0)
        if np.any(invalid):
            out[invalid] = fallback
        return out

    def _local_resuspension_critstress_map(self, idx=None):
        """
        Return an optional local mapped resuspension critical shear stress [Pa].

        Priority:
          1) environment.sea_floor_resuspension_critstress

        Returns None if no mapped critical-stress reader is available.
        Invalid mapped values (non-finite or <= 0) are left as NaN so the caller can
        fall back element-wise to the configured / computed threshold.
        """
        tau = self._optional_env_array('sea_floor_resuspension_critstress', idx=idx)
        if tau is None:
            return None
        out = np.asarray(tau, dtype=float).copy()
        invalid = (~np.isfinite(out)) | (out <= 0.0)
        if np.any(invalid):
            out[invalid] = np.nan
        return out

    def _element_or_local_d50(self, idx=None):
        """
        Return per-element d50 [m] for bed-physics calculations.
        Priority:
          1) self.elements.d50 if > 0
          2) local mapped bed d50 from reader
          3) config fallback chemical:sediment:d50
        """
        if idx is None:
            elem = np.asarray(self.elements.d50, dtype=float)
            n = self.num_elements_active()
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()
            elem = np.asarray(self.elements.d50[idx], dtype=float)
            n = idx.size
        out = np.asarray(elem, dtype=float).copy()
        invalid = (~np.isfinite(out)) | (out <= 0.0)
        if np.any(invalid):
            fallback = self._local_bed_d50(idx=idx)
            if np.asarray(fallback).ndim == 0:
                fallback = np.full(n, float(fallback), dtype=float)
            out[invalid] = np.asarray(fallback, dtype=float)[invalid]
        return out

    def _assign_d50_to_elements(self, idx, particle_d50=None, sediment_d50=None, dissolved_value=0.0):
        """Assign per-element d50 values after seeding or species reassignment."""
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return
        specie = np.asarray(self.elements.specie[idx], dtype=int)
        particle_species = []
        if hasattr(self, 'num_prev'):
            particle_species.append(self.num_prev)
        if hasattr(self, 'num_psrev'):
            particle_species.append(self.num_psrev)
        if hasattr(self, 'num_pirrev'):
            particle_species.append(self.num_pirrev)
        sediment_species = []
        if hasattr(self, 'num_srev'):
            sediment_species.append(self.num_srev)
        if hasattr(self, 'num_ssrev'):
            sediment_species.append(self.num_ssrev)
        if hasattr(self, 'num_sirrev'):
            sediment_species.append(self.num_sirrev)
        if hasattr(self, 'num_sburied'):
            sediment_species.append(self.num_sburied)
        particle_mask = np.isin(specie, particle_species) if len(particle_species) > 0 else np.zeros(idx.size, dtype=bool)
        sediment_mask = np.isin(specie, sediment_species) if len(sediment_species) > 0 else np.zeros(idx.size, dtype=bool)
        other_mask = ~(particle_mask | sediment_mask)
        if particle_d50 is not None and np.any(particle_mask):
            pd = np.asarray(particle_d50, dtype=float)
            if pd.ndim == 0:
                self.elements.d50[idx[particle_mask]] = float(pd)
            else:
                self.elements.d50[idx[particle_mask]] = pd[particle_mask]
        if sediment_d50 is not None and np.any(sediment_mask):
            sd = np.asarray(sediment_d50, dtype=float)
            if sd.ndim == 0:
                self.elements.d50[idx[sediment_mask]] = float(sd)
            else:
                self.elements.d50[idx[sediment_mask]] = sd[sediment_mask]
        if np.any(other_mask):
            self.elements.d50[idx[other_mask]] = float(dissolved_value)

    def _sanitize_foc(self, values, fallback):
        """
        Return finite local f_OC values, replacing invalid entries with a fallback.
            invalid = non-finite OR negative
            out[invalid] = fallback

        This helper is used to sanitize environmental or per-element organic-carbon
        fractions before using them in Kd = KOC * f_OC calculations.
    """
        values = np.asarray(values, dtype=float)
        out = values.copy()
        invalid = (~np.isfinite(out)) | (out <= 0.0)
        if np.any(invalid):
            out[invalid] = fallback
        return out

    def _local_particle_fOC(self, idx=None):
        """
        Return local organic-carbon fraction for particle-bound water-column carriers.
        Priority:
          1) use environment.f_OC_spm if actually provided by a reader
          2) otherwise use config fallback chemical:transformations:fOC_SPM
        Invalid values are sanitized with _sanitize_foc().
        """
        fallback = float(self.get_config('chemical:transformations:fOC_SPM'))
        foc = self._optional_env_array('f_OC_spm', idx=idx)
        if foc is None:
            if idx is None:
                n = self.num_elements_active()
            else:
                n = np.asarray(idx, dtype=np.int64).ravel().size
            return np.full(n, fallback, dtype=float)
        return self._sanitize_foc(foc, fallback)

    def _local_sediment_fOC(self, idx=None):
        """
        Return local organic-carbon fraction for sediment-bound species.
        Priority:
          1) use environment.f_OC_sed if actually provided by a reader
          2) otherwise use config fallback chemical:transformations:fOC_sed
        Invalid values are sanitized with _sanitize_foc().
        """
        fallback = float(self.get_config('chemical:transformations:fOC_sed'))
        foc = self._optional_env_array('f_OC_sed', idx=idx)
        if foc is None:
            if idx is None:
                n = self.num_elements_active()
            else:
                n = np.asarray(idx, dtype=np.int64).ravel().size
            return np.full(n, fallback, dtype=float)
        return self._sanitize_foc(foc, fallback)

    def _assign_fOC_to_elements(self, idx, particle_foc=None, sediment_foc=None):
        """
        Assign per-element f_OC values after seeding or species reassignment (self.elements.f_OC[idx]).
        1) Identify which selected elements belong to a particle family:
               {prev, psrev, pirrev}
        2) Identify which selected elements belong to a sediment family:
               {srev, ssrev, sirrev, sburied}
        3) If particle_foc is provided:
               assign it only to particle-family elements
        4) If sediment_foc is provided:
               assign it only to sediment-family elements
        5) Dissolved / colloidal / non-carrier species are left unchanged
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return

        specie = np.asarray(self.elements.specie[idx], dtype=int)

        particle_species = []
        if hasattr(self, 'num_prev'):
            particle_species.append(self.num_prev)
        if hasattr(self, 'num_psrev'):
            particle_species.append(self.num_psrev)
        if hasattr(self, 'num_pirrev'):
            particle_species.append(self.num_pirrev)
        sediment_species = []
        if hasattr(self, 'num_srev'):
            sediment_species.append(self.num_srev)
        if hasattr(self, 'num_ssrev'):
            sediment_species.append(self.num_ssrev)
        if hasattr(self, 'num_sirrev'):
            sediment_species.append(self.num_sirrev)
        if hasattr(self, 'num_sburied'):
            sediment_species.append(self.num_sburied)

        if particle_foc is not None and len(particle_species) > 0:
            mask = np.isin(specie, particle_species)
            if np.any(mask):
                pf = np.asarray(particle_foc, dtype=float)
                if pf.ndim == 0:
                    self.elements.f_OC[idx[mask]] = float(pf)
                else:
                    self.elements.f_OC[idx[mask]] = pf[mask]

        if sediment_foc is not None and len(sediment_species) > 0:
            mask = np.isin(specie, sediment_species)
            if np.any(mask):
                sf = np.asarray(sediment_foc, dtype=float)
                if sf.ndim == 0:
                    self.elements.f_OC[idx[mask]] = float(sf)
                else:
                    self.elements.f_OC[idx[mask]] = sf[mask]

    ###########################################################################
    # Helpers for bed shear stress and wave/current stress
    ###########################################################################

    def _wave_reader_array(self, name, idx=None):
        """Explicitly supplied field only, preserving masked values as NaN.

        Accept attached-reader fields and explicit ``environment:constant:*``
        values, including exact zero. Generic environment fallbacks are never
        interpreted as supplied wave forcing.
        """
        if not self._has_explicit_environment_source(name):
            return None
        raw = getattr(self.environment, name, None)
        if raw is None:
            return None
        values = np.ma.asarray(raw, dtype=float).filled(np.nan)
        n = self.num_elements_active()
        if values.ndim == 0 or values.size == 1:
            count = n if idx is None else np.asarray(idx).size
            return np.full(count, float(values.reshape(-1)[0]), dtype=float)
        if values.ndim != 1 or values.size != n:
            raise ValueError(f'{name} must be scalar or have one value per active element.')
        return values.copy() if idx is None else values[np.asarray(idx, dtype=np.int64)]

    def _required_wave_environment_array(self, name, idx=None):
        """Return already-loaded wave forcing without re-checking reader availability.

        Reader capability is validated once by _validate_wave_stress_source()
        before the run starts. This helper is therefore only responsible for
        retrieving the values already placed in ``self.environment`` for the
        current timestep and active elements.

        A missing attribute here is an internal lifecycle/environment-loading
        inconsistency, not a reader-availability decision. Masked values are
        preserved as NaN so the calling physics routine can reject invalid local
        values before they enter a calculation.
        """
        raw = getattr(self.environment, name, None)
        if raw is None:
            raise RuntimeError(
                f'Required wave forcing {name!r} was validated before the run, '
                'but is absent from the current environment state.'
            )
        values = np.ma.asarray(raw, dtype=float).filled(np.nan)
        n = self.num_elements_active()
        if values.ndim == 0 or values.size == 1:
            count = n if idx is None else np.asarray(idx).size
            return np.full(count, float(values.reshape(-1)[0]), dtype=float)
        if values.ndim != 1 or values.size != n:
            raise ValueError(f'{name} must be scalar or have one value per active element.')
        return values.copy() if idx is None else values[np.asarray(idx, dtype=np.int64)]

    def _bottom_velocity_components(self, idx=None):
        """Return bottom-layer velocity components.
        This helper only accepts dedicated bottom-layer velocity fields:
            - x_bottom_sea_water_velocity
            - y_bottom_sea_water_velocity
        No fallback to depth-averaged or full-column velocity is used.
        """
        reader = (self._wave_reader_array
                  if self.get_config('chemical:sediment:include_wave_stress')
                  else self._optional_env_array)
        u_b = reader('x_bottom_sea_water_velocity', idx=idx)
        v_b = reader('y_bottom_sea_water_velocity', idx=idx)

        if u_b is None or v_b is None:
            return None, None

        return np.asarray(u_b, dtype=float), np.asarray(v_b, dtype=float)

    def _depth_averaged_velocity_components(self, idx=None):
        """Return depth-averaged velocity components for bulk friction laws.
        This helper reads:
            - x_depth_averaged_sea_water_velocity
            - y_depth_averaged_sea_water_velocity
    """
        reader = (self._wave_reader_array
                  if self.get_config('chemical:sediment:include_wave_stress')
                  else self._optional_env_array)
        u_da = reader('x_depth_averaged_sea_water_velocity', idx=idx)
        v_da = reader('y_depth_averaged_sea_water_velocity', idx=idx)

        if u_da is None or v_da is None:
            return None, None

        return np.asarray(u_da, dtype=float), np.asarray(v_da, dtype=float)

    def _hydraulic_radius_array(self, idx=None):
        """Returns hydraulic radius [m] for bulk friction laws: valid reader value, configured value, then depth.

        The priority applies separately to each element. Generic environment
        fallback zeros do not take precedence over bulk_hydraulic_radius.
        compute_bottom_shear_stress retains its existing cap at local depth.
        """
        n = self.num_elements_active() if idx is None else np.asarray(idx).size
        radius = self._wave_reader_array('hydraulic_radius', idx=idx)
        if radius is None:
            radius = np.full(n, np.nan, dtype=float)
        else:
            radius = np.asarray(radius, dtype=float).copy()
        valid = np.isfinite(radius) & (radius > 0.0)
        configured = float(self.get_config('chemical:sediment:bulk_hydraulic_radius'))
        if not np.isfinite(configured):
            raise ValueError('bulk_hydraulic_radius must be finite; use zero to select depth.')
        if configured > 0.0:
            radius[~valid] = configured
        else:
            depth = np.asarray(self._env_array(
                'sea_floor_depth_below_sea_level', 1.0, idx=idx), dtype=float)
            radius[~valid] = depth[~valid]
        if np.any(~np.isfinite(radius)):
            raise ValueError('Hydraulic-radius fallback contains non-finite water depth.')
        return np.maximum(radius, 1e-12)

    def _resolve_current_stress_source(self):
        """Resolve the optional direct-current-stress source once per run.

        Reader capability belongs to prepare_run(), not to the timestep physics.
        If an attached reader advertises ``sea_floor_current_stress``, DIRECT is
        selected for the run.  Otherwise the configured current-stress
        parameterization is used.  Invalid/missing local DIRECT values never
        trigger a fallback; they are rejected explicitly at runtime.
        """
        sediment_exchange_enabled = bool(
            self.get_config('chemical:sediment:enable_deposition') or
            self.get_config('chemical:sediment:enable_resuspension')
        )
        has_direct = bool(
            sediment_exchange_enabled and
            self._has_reader_variable('sea_floor_current_stress')
        )
        self._direct_current_stress_reader_available = has_direct
        self._current_stress_source = 'DIRECT' if has_direct else 'CALCULATED'

        if sediment_exchange_enabled:
            if has_direct:
                logger.info(
                    'Current bed-stress source=DIRECT (reader-supplied '
                    'sea_floor_current_stress).')
            else:
                logger.info(
                    'Current bed-stress source=CALCULATED (mode=%s); no reader '
                    'supplies sea_floor_current_stress.',
                    self.get_config('chemical:sediment:stress_param_mode'))
        return self._current_stress_source

    def _direct_current_stress_array(self, idx=None):
        """Return current local DIRECT stress values selected in prepare_run().

        This routine does not inspect reader availability.  ``prepare_run()`` has
        already frozen whether DIRECT current stress is available for the run.
        Here we only retrieve the already-loaded local environment values and
        reject missing, non-finite or negative stresses before they enter the
        physics.  Zero is a valid stress magnitude.
        """
        if not bool(getattr(self, '_direct_current_stress_reader_available', False)):
            return None

        name = 'sea_floor_current_stress'
        raw = getattr(self.environment, name, None)
        if raw is None:
            raise RuntimeError(
                f'Required direct current-stress forcing {name!r} was resolved '
                'from the readers in prepare_run(), but is absent from the '
                'current environment state.')

        values = np.ma.asarray(raw, dtype=float).filled(np.nan)
        n = self.num_elements_active()
        if values.ndim == 0 or values.size == 1:
            count = n if idx is None else np.asarray(idx).size
            tau = np.full(count, float(values.reshape(-1)[0]), dtype=float)
        else:
            if values.ndim != 1 or values.size != n:
                raise ValueError(
                    f'{name} must be scalar or have one value per active element.')
            tau = values.copy() if idx is None else values[np.asarray(idx, dtype=np.int64)]

        invalid = ~np.isfinite(tau) | (tau < 0.0)
        if np.any(invalid):
            raise ValueError(
                f'{name} contains {int(invalid.sum())} invalid stress values; '
                'expected finite, non-negative magnitudes in Pa.')
        return tau

    def _bed_stress_array(self, name, idx=None):
        """Read an optional non-current stress magnitude [Pa], preserving masks.

        This helper retains reader-availability discovery for optional stress
        fields such as ``sea_floor_other_stress``.  Direct current stress uses
        ``_direct_current_stress_array()`` instead, because its source is frozen
        once in prepare_run().
        """
        tau = self._wave_reader_array(name, idx=idx)
        if tau is None:
            if self._has_reader_variable(name):
                raise ValueError(f'{name} is advertised by a reader but has no available data.')
            return None
        invalid = ~np.isfinite(tau) | (tau < 0.0)
        if np.any(invalid):
            raise ValueError(
                f'{name} contains {int(invalid.sum())} invalid stress values; '
                'expected finite, non-negative magnitudes in Pa.')
        return tau

    def _wave_to_direction_array(self, idx=None):
        """Geographic propagation bearing, clockwise from north; NaN if absent.

        Prefer finite to-directions locally; fill gaps with finite from-directions
        converted by 180 degrees. Never accept the generic environment fallback
        as evidence that a direction field was actually supplied.
        """
        n = self.num_elements_active() if idx is None else np.asarray(idx).size
        direction = np.full(n, np.nan, dtype=float)
        used = []
        for name, offset in (('sea_surface_wave_to_direction', 0.0),
                             ('sea_surface_wave_from_direction', 180.0)):
            values = self._wave_reader_array(name, idx=idx)
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            # Accepted geographic bearings: [0, 360]. Reject finite fill values
            # rather than silently wrapping, e.g. -99999, into a valid bearing.
            valid = np.isfinite(values) & (values >= 0) & (values <= 360)
            take = ~np.isfinite(direction) & valid
            direction[take] = (values[take] + offset) % 360.0
            if np.any(take):
                used.append(name)
        self._wave_direction_source_name = ','.join(used) if used else None
        self._wave_direction_source_convention = 'to' if used else None
        return direction

    def _bearing_to_unit_vector(self, bearing_deg):
        """
        Convert a geographic bearing (degrees clockwise from north, 'to' direction)
        to unit-vector components (x eastward, y northward).
        """
        theta = np.deg2rad(np.asarray(bearing_deg, dtype=float))
        ex = np.sin(theta)
        ey = np.cos(theta)
        return ex, ey

    def _wave_water_depth(self, idx=None):
        """
        Return actual water-column thickness [m] for CALCULATED wave stress.

        The generic ChemicalDrift/OpenDrift bathymetry fallback is retained for
        normal model operation, but it is not valid physical bathymetry for wave
        attenuation. CALCULATED wave stress therefore rejects local bathymetry
        values that are missing, negative, or equal to the configured generic
        bathymetry fallback.

        A non-positive final water-column thickness after applying sea-surface
        elevation is allowed and represents a locally dry point.

        DIRECT wave stress never calls this routine.
        """
        bathy = self._required_wave_environment_array(
            'sea_floor_depth_below_sea_level',
            idx=idx,)
        bathy = np.asarray(bathy, dtype=float)

        bad = ~np.isfinite(bathy) | (bathy < 0.0)

        # The configured generic fallback is a software fallback, not physical
        # bathymetry for CALCULATED wave attenuation.
        bathy_fallback = self.get_config(
            'environment:fallback:sea_floor_depth_below_sea_level')

        if bathy_fallback is not None:
            try:
                bathy_fallback = float(bathy_fallback)
            except (TypeError, ValueError):
                bathy_fallback = np.nan

            if np.isfinite(bathy_fallback):
                bad |= np.isclose(bathy,
                    bathy_fallback, rtol=0.0, atol=1.0e-12)

        if np.any(bad):
            raise ValueError(
                'CALCULATED wave stress requires valid physical bathymetry; '
                f'{int(np.count_nonzero(bad))} element(s) contain missing, '
                'negative, or fallback-valued sea_floor_depth_below_sea_level.')

        mode = self.get_config('chemical:sediment:wave_depth_convention')

        if mode == 'MEAN_SEA_LEVEL':
            eta = self._wave_reader_array('sea_surface_height', idx=idx,)

            if eta is None:
                eta = np.asarray( self._env_array('sea_surface_height',
                        0.0, idx=idx,), dtype=float,)

            if np.any(~np.isfinite(eta)):
                raise ValueError(
                    'Wave-stress surface elevation contains non-finite values.')

            depth = bathy + eta

        elif mode == 'INSTANTANEOUS':
            depth = bathy.copy()

        else:
            raise ValueError(
                f'Unknown wave_depth_convention: {mode!r}')

        return depth

    def _wave_number(self, period, depth):
        """Solve omega**2=g*k*tanh(k*h) for positive period/depth arrays.

        Safeguarded Newton iteration in q=k*h; bisection completes any
        unconverged entries. Relative stopping criteria also cover shallow water.
        """
        period, depth = np.broadcast_arrays(np.asarray(period, dtype=float),
                                            np.asarray(depth, dtype=float))
        if np.any(~np.isfinite(period) | (period <= 0) |
                  ~np.isfinite(depth) | (depth <= 0)):
            raise ValueError('Dispersion requires finite, positive period and depth.')
        x = (2.0 * np.pi / period)**2 * depth / 9.81
        if np.any(~np.isfinite(x) | (x <= 0)):
            raise ValueError('Wave period/depth outside numerical dispersion range.')
        lo = np.maximum(x, np.sqrt(x))
        hi = x + np.sqrt(x)
        q = 0.5 * (lo + hi)
        for _ in range(12):
            tq = np.tanh(q)
            residual = q * tq - x
            done = np.abs(residual) <= 1e-12 * x
            if np.all(done):
                return q / depth
            lo = np.where((residual < 0) & ~done, q, lo)
            hi = np.where((residual >= 0) & ~done, q, hi)
            proposal = q - residual / (tq + q * (1.0 - tq*tq))
            proposal = np.where((proposal > lo) & (proposal < hi),
                                proposal, 0.5 * (lo + hi))
            q = np.where(done, q, proposal)
        for _ in range(60):
            residual = q * np.tanh(q) - x
            done = np.abs(residual) <= 1e-12 * x
            if np.all(done):
                return q / depth
            lo = np.where((residual < 0) & ~done, q, lo)
            hi = np.where((residual >= 0) & ~done, q, hi)
            q = np.where(done, q, 0.5 * (lo + hi))
        raise RuntimeError('Wave dispersion solver failed to converge.')

    def _wave_orbital_parameters(self, height, period, depth):
        """Equivalent-wave orbital amplitude and excursion; positive inputs only.

        Retain logarithms for stress evaluation in strongly attenuated waves.
        RMS_EQUIVALENT uses Hs/sqrt(2), a narrow-band approximation at Tp.
        The returned amplitude is not the maximum of a random sea state.
        """
        height, period, depth = np.broadcast_arrays(
            np.asarray(height, dtype=float), np.asarray(period, dtype=float),
            np.asarray(depth, dtype=float))
        if np.any(~np.isfinite(height) | (height <= 0)):
            raise ValueError('Orbital calculation requires finite, positive wave height.')
        mode = self.get_config('chemical:sediment:wave_height_convention')
        if mode == 'RMS_EQUIVALENT':
            scale = np.sqrt(2.0)
        elif mode == 'SIGNIFICANT_HEIGHT':
            scale = 1.0
        else:
            raise ValueError(f'Unknown wave_height_convention: {mode!r}')
        k = self._wave_number(period, depth)
        q = k * depth
        # log(1/sinh(q)) without overflow for large q or cancellation for small q.
        log_transfer = np.log(2.0) - q - np.log(-np.expm1(-2.0*q))
        log_u = np.log(np.pi) + np.log(height) - np.log(scale) - np.log(period) + log_transfer
        log_a = log_u + np.log(period) - np.log(2.0*np.pi)
        with np.errstate(under='ignore'):
            u = np.exp(log_u)
            a = np.exp(log_a)
        if np.any(~np.isfinite(u) | ~np.isfinite(a)):
            raise ValueError('Wave orbital parameters outside numerical range.')
        return u, a, k, log_u, log_a

    def _soulsby_wave_stress(self, log_u, log_a, ks, rho, nu):
        """Soulsby rough/smooth maximum closure; stress amplitude in Pa.

        Logarithms avoid overflow in f_w*U_w**2 at negligible orbital motion.
        A friction factor beyond float range is reported as NaN in diagnostics;
        stress remains evaluated from its finite logarithm. No empirical cap
        from the old A/z0 clipping is retained.
        """
        log_u, log_a, ks, rho, nu = np.broadcast_arrays(
            np.asarray(log_u, dtype=float), np.asarray(log_a, dtype=float),
            np.asarray(ks, dtype=float), np.asarray(rho, dtype=float),
            np.asarray(nu, dtype=float))
        if np.any(~np.isfinite(log_u) | ~np.isfinite(log_a) |
                  ~np.isfinite(ks) | (ks <= 0) | ~np.isfinite(rho) | (rho <= 0) |
                  ~np.isfinite(nu) | (nu <= 0)):
            raise ValueError('Wave friction requires finite orbital logarithms and positive ks/rho/nu.')
        log_re = log_u + log_a - np.log(nu)
        log_fr = np.log(0.237) - 0.52*(log_a - np.log(ks))
        log_fs = np.where(log_re <= np.log(5e5),
                          np.log(2.0) - 0.5*log_re,
                          np.log(0.0521) - 0.187*log_re)
        log_fw = np.maximum(log_fr, log_fs)
        log_tau = np.log(0.5) + np.log(rho) + log_fw + 2.0*log_u
        if np.any(~np.isfinite(log_tau) | (log_tau > np.log(np.finfo(float).max))):
            raise ValueError('Calculated wave stress exceeds the numerical range.')
        fw = np.full(log_fw.shape, np.nan, dtype=float)
        representable = log_fw <= np.log(np.finfo(float).max)
        with np.errstate(under='ignore'):
            fw[representable] = np.exp(log_fw[representable])
            tau = np.exp(log_tau)
        return tau, fw

    def _wave_stress_source(self):
        """Selected source; never fall back between DIRECT and CALCULATED."""
        source = self.get_config('chemical:sediment:wave_stress_source')
        if source not in ('CALCULATED', 'DIRECT'):
            raise ValueError(f'Unknown wave_stress_source: {source!r}')
        return source

    def _wave_forcing_active(self):
        """Whether wave bed stress can participate in sediment exchange."""
        return (
            bool(self.get_config('chemical:sediment:include_wave_stress'))
            and (
                bool(self.get_config('chemical:sediment:enable_deposition'))
                or bool(self.get_config('chemical:sediment:enable_resuspension'))
            )
        )

    def _warn_calm_wave_with_wind(self):
        """Warn once about Hs=0 with substantial wind; never alter forcing."""
        if getattr(self, '_calm_wave_wind_warning_emitted', False):
            return
        if self._wave_stress_source() != 'CALCULATED':
            return

        threshold = float(self.get_config(
            'chemical:sediment:calm_wave_wind_warning_threshold'))
        if threshold < 0.0:
            return

        height = self._required_wave_environment_array(
            'sea_surface_wave_significant_height')
        height = np.asarray(height, dtype=float)
        x_wind = np.asarray(self._env_array('x_wind', 0.0), dtype=float)
        y_wind = np.asarray(self._env_array('y_wind', 0.0), dtype=float)
        wind_speed = np.hypot(x_wind, y_wind)

        calm_with_wind = (
            np.isfinite(height)
            & (height == 0.0)
            & np.isfinite(wind_speed)
            & (wind_speed > threshold)
        )
        if np.any(calm_with_wind):
            self._calm_wave_wind_warning_emitted = True
            logger.warning(
                'CALCULATED wave stress received explicitly supplied Hs=0 for '
                '%d active element(s) while local wind speed exceeds %.3g m/s '
                '(maximum %.3g m/s). The supplied wave field is authoritative: '
                'Hs remains zero and no wind-derived wave height is generated.',
                int(np.count_nonzero(calm_with_wind)),
                threshold,
                float(np.nanmax(wind_speed[calm_with_wind])),
            )

    def calculate_missing_environment_variables(self):
        """Preserve explicit wave forcing against wind-derived substitution.

        OpenDrift generic missing-environment handling may interpret all-zero wave
        fields as missing and derive significant wave height or wave-from direction
        from wind. ChemicalDrift instead treats explicitly supplied wave forcing as
        authoritative:

        * Hs == 0 is a valid supplied calm-wave state;
        * wave direction == 0 degrees is a valid geographic bearing;
        * zero local wind does not imply zero waves, and non-zero wind does not
          authorize replacement of an explicitly supplied zero Hs.

        The parent processing is retained for all other environment variables.
        """
        if not self._wave_forcing_active():
            return super(
                ChemicalDrift, self
            ).calculate_missing_environment_variables()

        protected_names = []
        if self._wave_stress_source() == 'CALCULATED':
            protected_names.extend((
                'sea_surface_wave_significant_height',
                'sea_surface_wave_period_at_variance_spectral_density_maximum',
            ))

        # Direction may be needed by SOULSBY_CLARKE for either wave-stress source.
        protected_names.extend((
            'sea_surface_wave_to_direction',
            'sea_surface_wave_from_direction',
        ))

        protected = {}
        for name in protected_names:
            if not self._has_explicit_environment_source(name):
                continue
            raw = getattr(self.environment, name, None)
            if raw is None:
                continue
            try:
                protected[name] = raw.copy()
            except AttributeError:
                protected[name] = raw

        try:
            result = super(
                ChemicalDrift, self
            ).calculate_missing_environment_variables()
        finally:
            for name, value in protected.items():
                setattr(self.environment, name, value)

        if (
            self._wave_stress_source() == 'CALCULATED'
            and 'sea_surface_wave_significant_height' in protected
        ):
            self._warn_calm_wave_with_wind()

        return result

    def _validate_wave_stress_source(self):
        """
        Validate wave-reader capability once before the simulation starts.
        Runtime wave-stress routines then deal only with local values: they
        decide which inputs are physically needed for the current elements and
        reject invalid values before those values enter a calculation.
        """
        source = self._wave_stress_source()
        if source == 'DIRECT':
            names = ('sea_floor_wave_stress',)
        else:
            try:
                use_tabularised_stokes = bool(
                    self.get_config('drift:use_tabularised_stokes_drift'))
            except (KeyError, ValueError):
                use_tabularised_stokes = False

            if use_tabularised_stokes:
                raise ValueError(
                    "chemical:sediment:wave_stress_source='CALCULATED' is "
                    "incompatible with drift:use_tabularised_stokes_drift=True. "
                    "CALCULATED wave stress requires explicitly supplied wave "
                    "forcing and does not permit wind-derived Hs replacement."
                )

            names = (
                'sea_surface_wave_significant_height',
                'sea_surface_wave_period_at_variance_spectral_density_maximum',
                'sea_floor_depth_below_sea_level',)
        for name in names:
            if not self._has_explicit_environment_source(name):
                raise ValueError(
                    f'{source} wave stress requires an explicit source for {name}; '
                    'attached readers and environment constants are allowed.')
        logger.info('Wave stress source=%s; combination=%s', source,
                    self.get_config('chemical:sediment:shear_stress_combination'))
        if source == 'CALCULATED':
            logger.info('Wave calculation: height=%s; depth=%s; roughness=%s',
                        self.get_config('chemical:sediment:wave_height_convention'),
                        self.get_config('chemical:sediment:wave_depth_convention'),
                        self._resolved_wave_roughness_mode())
        combo = self.get_config(
            'chemical:sediment:shear_stress_combination')

        if combo == 'SOULSBY_CLARKE':
            has_to = self._has_explicit_environment_source(
                'sea_surface_wave_to_direction')
            has_from = self._has_explicit_environment_source(
                'sea_surface_wave_from_direction')
            if not (has_to or has_from):
                raise ValueError(
                    'SOULSBY_CLARKE requires a reader supplying either '
                    'sea_surface_wave_to_direction or '
                    'sea_surface_wave_from_direction.')

    def _wave_stress(self, rho, idx=None):
        """Return one wave-stress source and its available diagnostics.

        DIRECT bypasses all wave orbital, roughness, depth and viscosity
        calculations. Reader availability is validated before the run; this
        routine validates only the current local stress values. The input must
        be a wave-only stress amplitude in Pa; a combined wave-current field
        would double-count current stress. Masked, negative or non-finite direct
        values raise an error; zero is valid. Derived diagnostics remain NaN,
        including at zero stress. Direction is handled by
        compute_bottom_shear_stress after source selection.
        """
        if self._wave_stress_source() == 'CALCULATED':
            return self._wave_stress_from_surface(rho=rho, idx=idx)
        if idx is None:
            idx = np.arange(self.num_elements_active(), dtype=np.int64)
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()
        n = idx.size
        result = {name: np.full(n, np.nan, dtype=float) for name in
                  ('wave_orbital_velocity', 'wave_excursion', 'wave_number',
                   'wave_friction_factor', 'wave_z0', 'wave_water_depth')}
        if n == 0:
            result['tau_wave'] = np.empty(0, dtype=float)
            return result
        tau = self._required_wave_environment_array('sea_floor_wave_stress', idx=idx)
        bad = ~np.isfinite(tau) | (tau < 0)
        if np.any(bad):
            raise ValueError(
                f'Invalid DIRECT wave stress for {int(bad.sum())} elements: '
                'values must be finite and non-negative [Pa].'
            )
        result['tau_wave'] = tau.copy()
        return result

    def _wave_stress_from_surface(self, rho, idx=None):
        """Derive wave stress from local Hs, Tp, depth and bed roughness.

        Source: Soulsby (1997), rough/smooth friction closure; linear-wave
        transfer and equivalent-wave definitions: Soulsby (2006), TR155.

        Reader capability is checked once before the run. At each timestep this
        routine then uses the local Hs field to decide whether the more expensive
        period/roughness/viscosity branch is needed. Only positive-height, wet
        entries consume Tp and enter the orbital/stress calculation. Invalid Hs
        or Tp values are rejected before they can enter those calculations.
        Dry entries contribute zero wave stress; this does not perform particle
        stranding or replace OceanDrift's wet/dry treatment.
        """
        if idx is None:
            idx = np.arange(self.num_elements_active(), dtype=np.int64)
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()
        n = idx.size
        result = {name: np.zeros(n, dtype=float) for name in
                  ('tau_wave', 'wave_orbital_velocity', 'wave_excursion', 'wave_number')}
        result.update({name: np.full(n, np.nan, dtype=float) for name in
                       ('wave_friction_factor', 'wave_z0', 'wave_water_depth')})
        if n == 0:
            return result

        # Depth and Hs are always needed to determine whether wave stress exists
        # for the current elements at this timestep.
        depth = self._wave_water_depth(idx=idx)
        result['wave_water_depth'] = depth.copy()
        wet = depth > 0

        height = self._required_wave_environment_array(
            'sea_surface_wave_significant_height', idx=idx)
        height = np.asarray(height, dtype=float)

        # Hs must be finite and non-negative wherever the element is wet. Invalid
        # values are stopped here and never enter the wave calculations.
        bad_height = wet & (~np.isfinite(height) | (height < 0.0))
        if np.any(bad_height):
            raise ValueError(
                f'Invalid/missing significant wave height for '
                f'{int(bad_height.sum())} wet elements.'
            )

        # Only wet elements with strictly positive Hs need Tp, roughness,
        # viscosity, dispersion, orbital velocity or wave-friction calculations.
        active = wet & (height > 0.0)
        if not np.any(active):
            return result

        active_idx = idx[active]

        # Tp is deliberately accessed only after active waves have been identified.
        period = self._required_wave_environment_array(
            'sea_surface_wave_period_at_variance_spectral_density_maximum',
            idx=active_idx,
        )
        period = np.asarray(period, dtype=float)

        # Invalid Tp must not enter dispersion/orbital/stress calculations.
        bad_period = ~np.isfinite(period) | (period <= 0.0)
        if np.any(bad_period):
            raise ValueError(
                'Positive waves at wet elements require finite, positive peak periods; '
                f'found {int(bad_period.sum())} invalid value(s).'
            )

        # The remaining inputs/calculations are needed only for active waves.
        z0 = self._wave_roughness_length_array(idx=active_idx)
        rho = np.broadcast_to(np.asarray(rho, dtype=float), (n,))[active]
        temp = self._env_array('sea_water_temperature', 10.0, idx=active_idx)
        salt = self._env_array('sea_water_salinity', 34.0, idx=active_idx)
        nu = np.asarray(seawater_dynamic_viscosity(temp, salt), dtype=float) / rho
        u, a, k, log_u, log_a = self._wave_orbital_parameters(
            height[active], period, depth[active])
        tau, fw = self._soulsby_wave_stress(log_u, log_a, 30.0*z0, rho, nu)
        for name, value in (('tau_wave', tau), ('wave_orbital_velocity', u),
                            ('wave_excursion', a), ('wave_number', k),
                            ('wave_friction_factor', fw), ('wave_z0', z0)):
            result[name][active] = value
        return result

    def _bottom_roughness_length_array(self, idx=None):
        """
        Return bottom roughness length z0 [m].
        Priority:
          1) reader-provided sea_floor_roughness_length
          2) config chemical:sediment:roughness_length

        Mssing reader returns None instead of the required_variables fallback.
        If reader values are invalid (non-finite or <= 0), they are replaced by the
        config fallback, which must then be finite and > 0.
        """
        eps = 1e-12

        if idx is None:
            n = self.num_elements_active()
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()
            n = idx.size

        z0_cfg = float(self.get_config('chemical:sediment:roughness_length'))
        cfg_ok = np.isfinite(z0_cfg) and (z0_cfg > 0.0)

        z0_reader = self._optional_env_array('sea_floor_roughness_length', idx=idx)

        # No reader supplied this variable: use config fallback
        if z0_reader is None:
            if not cfg_ok:
                raise ValueError(
                    "sea_floor_roughness_length is not provided by any reader, and "
                    "chemical:sediment:roughness_length must then be finite and > 0.")
            return np.full(n, z0_cfg, dtype=float)

        # Reader supplied it: validate values, replace bad cells with config fallback
        z0 = np.asarray(z0_reader, dtype=float).copy()
        bad = (~np.isfinite(z0)) | (z0 <= 0.0)

        if np.any(bad):
            if not cfg_ok:
                raise ValueError(
                    f"sea_floor_roughness_length contains {int(np.sum(bad))} invalid values, and "
                    "chemical:sediment:roughness_length must be finite and > 0 to replace them.")
            logger.warning(
                "Replacing %s invalid sea_floor_roughness_length values with "
                "chemical:sediment:roughness_length=%s", int(np.sum(bad)), z0_cfg)
            z0[bad] = z0_cfg

        return np.maximum(z0, eps)

    def _resolved_wave_roughness_mode(self):
        """Resolve CURRENT_MODE independently of whether direct current stress exists."""
        mode = self.get_config('chemical:sediment:wave_roughness_mode')
        if mode == 'CURRENT_MODE':
            current = self.get_config('chemical:sediment:stress_param_mode')
            mode = {'LOG_Z0': 'LOG_Z0', 'MANNING': 'LOG_Z0', 'CHEZY': 'LOG_Z0',
                    'GRAIN_D50': 'GRAIN_D50', 'WHITE_COLEBROOK': 'NIKURADSE'}[current]
        if mode not in ('LOG_Z0', 'GRAIN_D50', 'NIKURADSE'):
            raise ValueError(f'Unknown wave_roughness_mode: {mode!r}')
        return mode

    def _wave_roughness_length_array(self, idx=None):
        """Positive z0 [m] for wave friction; ks=30*z0, or ks=2.5*d50.

        Reuses existing mapped/configured fallback policy and USER d50 rules.
        Current roughness and its z0 diagnostic retain their existing meaning.
        """
        mode = self._resolved_wave_roughness_mode()
        n = self.num_elements_active() if idx is None else np.asarray(idx).size
        if mode == 'LOG_Z0':
            z0 = self._wave_reader_array('sea_floor_roughness_length', idx=idx)
            fallback = float(self.get_config('chemical:sediment:roughness_length'))
            if z0 is None:
                z0 = np.full(n, fallback, dtype=float)
            else:
                bad = ~np.isfinite(z0) | (z0 <= 0)
                if np.any(bad):
                    logger.warning('Replacing %s invalid wave z0 values with configured roughness_length.', int(bad.sum()))
                    z0 = np.where(bad, fallback, z0)
        elif mode == 'GRAIN_D50':
            # Preserve USER semantics; otherwise keep masks until fallback selection.
            d50 = None if self._user_resuspension_threshold_overrides_d50() else self._wave_reader_array('sea_floor_d50', idx=idx)
            fallback = float(self.get_config('chemical:sediment:d50'))
            if d50 is None:
                d50 = np.full(n, fallback, dtype=float)
            else:
                d50 = np.where(np.isfinite(d50) & (d50 > 0), d50, fallback)
            z0 = d50 / 12.0
        else:
            z0 = np.full(n, float(self.get_config('chemical:sediment:nikuradse_ks')) / 30.0)
        z0 = np.asarray(z0, dtype=float)
        if np.any(~np.isfinite(z0) | (z0 <= 0)):
            raise ValueError('Wave friction requires positive, finite roughness z0.')
        return z0

    def compute_bottom_shear_stress(self, idx=None):
        """
        Compute bed shear stress.
          1) use direct hydro-model current bed stress if available: sea_floor_current_stress
          2) otherwise:
             - LOG_Z0 / GRAIN_D50 use bottom-layer velocity
               (x/y_bottom_sea_water_velocity) at z_ref = 0.5 * bottom_layer_thickness
             - MANNING / CHEZY / WHITE_COLEBROOK use depth-averaged velocity
               (x/y_depth_averaged_sea_water_velocity) with hydraulic_radius
               (or sea depth as fallback approximation)

        Wave stress uses the selected CALCULATED (default) or DIRECT source.
        sum/max/rss are scalar heuristic combinations. Their vector diagnostics
        use a surrogate current direction (or wave axis if current is absent).
        SOULSBY_CLARKE is used to name the simplified Soulsby peak formulation applied,
        not the full Soulsby-Clarke boundary-layer model.
        It evaluates both +/- wave half-cycles and saves the larger vector.
        With no physical direction, diagnostic components are NaN, while the
        scalar stress remains usable. Other stress is non-directional and is
        added in quadrature in the SOULSBY_CLARKE branch.
        """
        if idx is None:
            idx = np.arange(self.num_elements_active(), dtype=np.int64)
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()

        n = idx.size
        if n == 0:
            empty = np.empty(0, dtype=float)
            return {
                'tau_bx': empty,
                'tau_by': empty,
                'tau_current': empty,
                'tau_effective': empty,
                'tau_effective_x': empty,
                'tau_effective_y': empty,
                'ustar_effective': empty,
                'rho': empty,
                'Cd': empty,
                'speed': empty,
                'z_ref': empty,
                'z0': empty,
                'tau_wave': empty,
                'wave_orbital_velocity': empty,
                'wave_excursion': empty,
                'wave_number': empty,
                'wave_friction_factor': empty,
                'wave_z0': empty,
                'wave_water_depth': empty,
            }

        kappa = 0.41
        g = 9.81
        eps = 1e-12

        T = self._env_array('sea_water_temperature', 10.0, idx=idx)
        S = self._env_array('sea_water_salinity', 34.0, idx=idx)
        rho = self.sea_water_density(T=T, S=S)

        # Local water depth [m], capped at >= 0
        depth = np.asarray(
            self._env_array('sea_floor_depth_below_sea_level', 10000.0, idx=idx),
            dtype=float
        )
        depth = np.maximum(depth, 0.0)

        dz_env = self._env_array('bottom_layer_thickness', 0.0, idx=idx)
        # z0_env = self._env_array('sea_floor_roughness_length', 0.0, idx=idx)

        Param_mode = self.get_config('chemical:sediment:stress_param_mode')
        dz_cfg = float(self.get_config('chemical:sediment:bottom_layer_thickness'))
        n_cfg = float(self.get_config('chemical:sediment:manning_n'))
        C_cfg = float(self.get_config('chemical:sediment:chezy_C'))
        ks_cfg = float(self.get_config('chemical:sediment:nikuradse_ks'))
        d50_local = self._local_bed_d50(idx=idx)
        cd_min = float(self.get_config('chemical:sediment:cd_min'))
        cd_max = float(self.get_config('chemical:sediment:cd_max'))

        # Bottom-layer thickness for LOG_Z0 / GRAIN_D50 fallback.
        # Use reader if provided, otherwise config fallback, but never let it exceed local depth.
        dz_bot = np.where(np.asarray(dz_env, dtype=float) > 0.0, dz_env, dz_cfg)
        dz_bot = np.asarray(dz_bot, dtype=float)
        dz_bot = np.minimum(np.maximum(dz_bot, eps), np.maximum(depth, eps))

        z0_default = None

        # Preferred path: direct hydro-model current bed stress, but only when
        # its reader was resolved as available once in prepare_run().  Runtime
        # logic validates local values only and never re-checks reader capability.
        tau_c = self._direct_current_stress_array(idx=idx)

        if tau_c is not None:
            # Preserve current-only behavior; wave-enabled runs need local validity.
            u_ref, v_ref = self._bottom_velocity_components(idx=idx)
            if u_ref is None or v_ref is None:
                u_ref, v_ref = self._depth_averaged_velocity_components(idx=idx)
            if u_ref is None or v_ref is None:
                u_ref = np.zeros(n, dtype=float)
                v_ref = np.zeros(n, dtype=float)
            if self.get_config('chemical:sediment:include_wave_stress'):
                u_ref = np.asarray(u_ref, dtype=float).copy()
                v_ref = np.asarray(v_ref, dtype=float).copy()
                invalid = ~np.isfinite(u_ref) | ~np.isfinite(v_ref) | (np.hypot(u_ref, v_ref) <= eps)
                u_da, v_da = self._depth_averaged_velocity_components(idx=idx)
                if u_da is not None and v_da is not None:
                    valid_da = np.isfinite(u_da) & np.isfinite(v_da) & (np.hypot(u_da, v_da) > eps)
                    take = invalid & valid_da
                    u_ref[take], v_ref[take] = u_da[take], v_da[take]
                invalid = ~np.isfinite(u_ref) | ~np.isfinite(v_ref)
                u_ref[invalid], v_ref[invalid] = 0.0, 0.0
            speed = np.hypot(u_ref, v_ref)

            tau_bx = np.zeros(n, dtype=float)
            tau_by = np.zeros(n, dtype=float)
            moving = speed > eps
            tau_bx[moving] = tau_c[moving] * u_ref[moving] / speed[moving]
            tau_by[moving] = tau_c[moving] * v_ref[moving] / speed[moving]

            Cd = np.full(n, np.nan, dtype=float)
            Cd[moving] = tau_c[moving] / np.maximum(rho[moving] * speed[moving] ** 2, eps)

            z_ref = 0.5 * dz_bot
            z0 = z0_default

        # Fallback path: compute current bed stress from selected mode
        else:
            if Param_mode in ('LOG_Z0', 'GRAIN_D50'):
                u_b, v_b = self._bottom_velocity_components(idx=idx)
                if u_b is None or v_b is None:
                    raise ValueError(
                        f"{Param_mode} requires x_bottom_sea_water_velocity and "
                        f"y_bottom_sea_water_velocity when sea_floor_current_stress is unavailable.")

                speed = np.hypot(u_b, v_b)
                z_ref = 0.5 * dz_bot

                if Param_mode == 'GRAIN_D50':
                    z0 = np.maximum(np.asarray(d50_local, dtype=float) / 12.0, eps)
                else:
                    z0 = self._bottom_roughness_length_array(idx=idx)

                z0_default = z0.copy()

                z_ref_eff = np.maximum(z_ref, 1.01 * z0)
                ratio = np.maximum(z_ref_eff / z0, 1.0 + 1e-6)
                Cd = (kappa / np.log(ratio)) ** 2

                if cd_min > 0.0:
                    Cd = np.maximum(Cd, cd_min)
                if cd_max > 0.0:
                    Cd = np.minimum(Cd, cd_max)

                tau_bx = rho * Cd * speed * u_b
                tau_by = rho * Cd * speed * v_b
                tau_c = rho * Cd * speed ** 2

            elif Param_mode in ('MANNING', 'CHEZY', 'WHITE_COLEBROOK'):
                u_da, v_da = self._depth_averaged_velocity_components(idx=idx)
                if u_da is None or v_da is None:
                    raise ValueError(
                        f"{Param_mode} requires x_depth_averaged_sea_water_velocity and "
                        f"y_depth_averaged_sea_water_velocity when sea_floor_current_stress is unavailable.")

                speed = np.hypot(u_da, v_da)

                # Hydraulic radius for MANNING / CHEZY / WHITE_COLEBROOK.
                # Use helper, but never let it exceed local depth.
                Rh = np.asarray(self._hydraulic_radius_array(idx=idx), dtype=float)
                Rh = np.minimum(np.maximum(Rh, eps), np.maximum(depth, eps))

                z_ref = np.full(n, np.nan, dtype=float)
                z0 = np.full(n, np.nan, dtype=float)

                if Param_mode == 'MANNING':
                    Cd = g * (n_cfg ** 2) / np.maximum(Rh, eps) ** (1.0 / 3.0)

                elif Param_mode == 'CHEZY':
                    Cd = np.full(n, g / max(C_cfg, eps) ** 2, dtype=float)

                elif Param_mode == 'WHITE_COLEBROOK':
                    ks = max(ks_cfg, eps)
                    Chezy = 18.0 * np.log10(np.maximum(12.0 * Rh / ks, 1.000001))
                    Cd = g / (Chezy ** 2)
                    z0 = np.full(n, ks / 30.0, dtype=float)

                if cd_min > 0.0:
                    Cd = np.maximum(Cd, cd_min)
                if cd_max > 0.0:
                    Cd = np.minimum(Cd, cd_max)

                tau_bx = rho * Cd * speed * u_da
                tau_by = rho * Cd * speed * v_da
                tau_c = rho * Cd * speed ** 2

            else:
                raise ValueError(f"Unknown stress_param_mode: {Param_mode!r}")

        combo = self.get_config('chemical:sediment:shear_stress_combination')
        use_wave = self.get_config('chemical:sediment:include_wave_stress')
        use_other = self.get_config('chemical:sediment:include_other_stress')

        tau_other = self._bed_stress_array('sea_floor_other_stress', idx=idx) if use_other else None

        # Defaults: effective = current-only
        tau_eff_x = tau_bx.copy()
        tau_eff_y = tau_by.copy()
        tau_eff = tau_c.copy()

        # Current stress direction is valid only with finite nonzero components.
        ec_x = np.zeros(n, dtype=float)
        ec_y = np.zeros(n, dtype=float)
        tau_vec = np.hypot(tau_bx, tau_by)
        has_current = np.isfinite(tau_vec) & (tau_vec > eps)
        ec_x[has_current] = tau_bx[has_current] / tau_vec[has_current]
        ec_y[has_current] = tau_by[has_current] / tau_vec[has_current]

        tau_wave = None
        ew_x = np.zeros(n, dtype=float)
        ew_y = np.zeros(n, dtype=float)
        has_wave_dir = np.zeros(n, dtype=bool)
        wave = {name: np.full(n, np.nan, dtype=float) for name in
                ('wave_orbital_velocity', 'wave_excursion', 'wave_number',
                 'wave_friction_factor', 'wave_z0', 'wave_water_depth')}
        if use_wave:
            if np.any(~np.isfinite(tau_c)):
                raise ValueError('Current stress contains non-finite values; check current velocity forcing.')
            wave = self._wave_stress(rho=rho, idx=idx)
            tau_wave = wave['tau_wave']
            wave_to_dir = self._wave_to_direction_array(idx=idx)
            if wave_to_dir is not None:
                has_wave_dir = np.isfinite(wave_to_dir)
                ew_x[has_wave_dir], ew_y[has_wave_dir] = self._bearing_to_unit_vector(
                    wave_to_dir[has_wave_dir])

        def _fallback_direction():
            """
            Return a surrogate unit direction for effective bed-stress components.
            Priority:
              1) current-stress direction, if current stress is nonzero
              2) wave direction, but only where wave direction is available
                 and wave stress is actively contributing (tau_wave > eps)
              3) no direction (0, 0) if neither current nor active wave
                 direction is available

            This helper is used only when the effective stress magnitude has been
            computed by a non-directional heuristic combination, or when a
            directional combination later loses direction information
            (for example, adding non-directional 'other' stress to a zero vector).
            It does not invent a direction for purely non-directional stress.
            """
            dir_x = np.zeros(n, dtype=float)
            dir_y = np.zeros(n, dtype=float)

            use_current_dir = has_current
            dir_x[use_current_dir] = ec_x[use_current_dir]
            dir_y[use_current_dir] = ec_y[use_current_dir]

            active_wave_dir = has_wave_dir
            if tau_wave is not None:
                active_wave_dir = has_wave_dir & (tau_wave > eps)
            else:
                active_wave_dir = np.zeros(n, dtype=bool)

            use_wave_dir = (~use_current_dir) & active_wave_dir
            dir_x[use_wave_dir] = ew_x[use_wave_dir]
            dir_y[use_wave_dir] = ew_y[use_wave_dir]

            return dir_x, dir_y

        if combo == 'SOULSBY_CLARKE':
            if tau_wave is not None:
                both = (tau_wave > 0) & (tau_c > 0)
                if np.any(both & (~has_wave_dir | ~has_current)):
                    raise ValueError(
                        'SOULSBY_CLARKE needs finite wave and current directions '
                        'where both stresses are positive. Supply to/from wave '
                        'direction and bottom or depth-averaged current velocity, '
                        'or select the scalar sum/max/rss combination.')
                total = tau_c + tau_wave
                fraction = np.divide(tau_wave, total, out=np.zeros(n), where=total > 0)
                tau_m = tau_c * (1.0 + 1.2*fraction**3.2)
                # Choose the half-cycle that reinforces the current projection.
                dot = ec_x*ew_x + ec_y*ew_y
                sign = np.where(dot < 0, -1.0, 1.0)
                tau_eff_x = tau_m*ec_x + sign*tau_wave*ew_x
                tau_eff_y = tau_m*ec_y + sign*tau_wave*ew_y
                tau_eff = np.hypot(tau_eff_x, tau_eff_y)
                # Preserve scalar forcing even if its direction is unknowable.
                current_only = (tau_c > 0) & (tau_wave == 0)
                wave_only = (tau_wave > 0) & (tau_c == 0)
                tau_eff[current_only] = tau_c[current_only]
                tau_eff[wave_only] = tau_wave[wave_only]

            if tau_other is not None:
                tau_eff = np.sqrt(tau_eff ** 2 + tau_other ** 2)

                # keep same direction if possible, otherwise fall back to current then wave direction
                mag = np.hypot(tau_eff_x, tau_eff_y)
                nonzero = mag > eps
                if np.any(nonzero):
                    tau_eff_x[nonzero] *= tau_eff[nonzero] / mag[nonzero]
                    tau_eff_y[nonzero] *= tau_eff[nonzero] / mag[nonzero]

                zero_with_mag = (~nonzero) & (tau_eff > eps)
                if np.any(zero_with_mag):
                    dir_x, dir_y = _fallback_direction()
                    tau_eff_x[zero_with_mag] = tau_eff[zero_with_mag] * dir_x[zero_with_mag]
                    tau_eff_y[zero_with_mag] = tau_eff[zero_with_mag] * dir_y[zero_with_mag]

        elif combo == 'sum':
            if tau_wave is not None:
                tau_eff = tau_eff + tau_wave
            if tau_other is not None:
                tau_eff = tau_eff + tau_other

        elif combo == 'max':
            if tau_wave is not None:
                tau_eff = np.maximum(tau_eff, tau_wave)
            if tau_other is not None:
                tau_eff = np.maximum(tau_eff, tau_other)

        elif combo == 'rss':
            tau_sq = tau_eff ** 2
            if tau_wave is not None:
                tau_sq = tau_sq + tau_wave ** 2
            if tau_other is not None:
                tau_sq = tau_sq + tau_other ** 2
            tau_eff = np.sqrt(tau_sq)

        else:
            raise ValueError(f"Unknown shear_stress_combination: {combo!r}")

        # For non-directional combinations, rebuild consistent vector components
        if combo != 'SOULSBY_CLARKE':
            dir_x, dir_y = _fallback_direction()
            tau_eff_x = tau_eff * dir_x
            tau_eff_y = tau_eff * dir_y

        if use_wave:
            # NaNs distinguish unknown direction from an actual zero stress.
            unknown_direction = (tau_eff > 0) & (np.hypot(tau_eff_x, tau_eff_y) == 0)
            tau_eff_x[unknown_direction] = np.nan
            tau_eff_y[unknown_direction] = np.nan

        ustar_eff = np.sqrt(np.maximum(tau_eff, 0.0) / np.maximum(rho, eps))

        return {
            'tau_bx': tau_bx,                  # current-only x-component
            'tau_by': tau_by,                  # current-only y-component
            'tau_current': tau_c,              # current-only magnitude
            'tau_wave': tau_wave if tau_wave is not None else np.zeros(n, dtype=float),
            **{name: wave[name] for name in
               ('wave_orbital_velocity', 'wave_excursion', 'wave_number',
                'wave_friction_factor', 'wave_z0', 'wave_water_depth')},
            'tau_effective': tau_eff,          # combined magnitude
            'tau_effective_x': tau_eff_x,      # combined x-component
            'tau_effective_y': tau_eff_y,      # combined y-component
            'ustar_effective': ustar_eff,
            'rho': rho,
            'Cd': Cd,
            'speed': speed,
            'z_ref': z_ref,
            'z0': z0,
        }

    ###########################################################################
    # Helpers for resuspension thresholds and probabilities
    ###########################################################################

    def _resuspension_branch(self, idx=None):
        """
        Return the resuspension branch used by the timestep-dependent erosion model.
        Explicit NONCOHESIVE/COHESIVE settings are returned directly.
        For AUTO, d50 >= 0.0625 mm is treated as NONCOHESIVE, finer material as COHESIVE.
        If no usable d50 exists, erodibility_M > 0 implies COHESIVE, else NONCOHESIVE.
        Returns either a scalar branch string or an array of branch strings.
        """
        branch_cfg = self.get_config('chemical:sediment:resuspension_critstress_branch')
        if branch_cfg in ('NONCOHESIVE', 'COHESIVE'):
            return branch_cfg
        d50_m = self._element_or_local_d50(idx=idx)
        d50_m = np.asarray(d50_m, dtype=float)
        branch = np.full(d50_m.shape, '', dtype='<U12')
        valid = np.isfinite(d50_m) & (d50_m > 0.0)
        if np.any(valid):
            d50_mm = d50_m[valid] * 1e3
            branch[valid] = np.where(d50_mm >= 0.0625, 'NONCOHESIVE', 'COHESIVE')
        if np.any(~valid):
            M_loc = np.asarray(self._local_erodibility_M(idx=idx), dtype=float)
            if M_loc.ndim == 0:
                M_loc = np.full(d50_m.shape, float(M_loc), dtype=float)
            branch[~valid] = np.where(M_loc[~valid] > 0.0, 'COHESIVE', 'NONCOHESIVE')
        return branch if branch.ndim > 0 else str(branch)

    def classify_sediment(self, d50_mm: float) -> str:
        """
        Classify sediment from median grain size d50 [mm].
            d50 < 0.0039     -> clay
            d50 < 0.0625     -> silt
            d50 < 2.0        -> sand
            otherwise        -> gravel
        """
        if d50_mm < 0:
            raise ValueError('d50_mm must be nonnegative')
        if d50_mm < 0.0039:
            return 'clay'
        elif d50_mm < 0.0625:
            return 'silt'
        elif d50_mm < 2.0:
            return 'sand'
        else:
            return 'gravel'

    def dimensionless_grain_size(self, d_m, rho_s=2650.0, rho_w=1000.0, nu=1.004e-6, g=9.81):
        """
        Compute dimensionless grain size D*.
        Let:
            s = rho_s / rho_w
        Then:
            D* = d * [ g * (s - 1) / nu^2 ]^(1/3)
        where:
            d     = grain diameter [m]
            rho_s = sediment density [kg/m3]
            rho_w = water density [kg/m3]
            nu    = kinematic viscosity [m2/s]
            g     = gravity [m/s2]

        D* is used in empirical Shields-type threshold relations.
        """
        d_m = np.asarray(d_m, dtype=float)
        rho_w = np.asarray(rho_w, dtype=float)
        if np.any(d_m <= 0):
            raise ValueError('d_m must be > 0')
        if np.any(rho_s <= rho_w):
            raise ValueError('rho_s must be > rho_w for sediment in water')
        if nu <= 0:
            raise ValueError('nu must be > 0')
        s = rho_s / rho_w
        return d_m * ((g * (s - 1.0)) / (nu ** 2)) ** (1.0 / 3.0)

    def theta_cr_soulsby_whitehouse(self, d_m, rho_s=2650.0, rho_w=1000.0, nu=1.004e-6, g=9.81):
        """
        Compute critical Shields parameter using the Soulsby-Whitehouse relation.
            theta_cr = 0.30 / (1 + 1.2 * D*)
                     + 0.055 * (1 - exp(-0.02 * D*))

        where D* is the dimensionless grain size.
        This relation gives the threshold dimensionless bed shear stress for
        noncohesive sediment motion.
        """
        D_star = self.dimensionless_grain_size(d_m, rho_s=rho_s, rho_w=rho_w, nu=nu, g=g)
        return 0.30 / (1.0 + 1.2 * D_star) + 0.055 * (1.0 - np.exp(-0.02 * D_star))

    def theta_cr_van_rijn(self, d_m, rho_s=2650.0, rho_w=1000.0, nu=1.004e-6, g=9.81):
        """
        Compute critical Shields parameter using the Van Rijn piecewise relation.
        For D* = dimensionless grain size:
            theta = 0.24  * D*^-1.00     for D* <= 4
            theta = 0.14  * D*^-0.64     for 4   < D* <= 10
            theta = 0.04  * D*^-0.10     for 10  < D* <= 20
            theta = 0.013 * D*^0.29      for 20  < D* <= 150
            theta = 0.056                for D* > 150
        """
        D_star = self.dimensionless_grain_size(d_m, rho_s=rho_s, rho_w=rho_w, nu=nu, g=g)
        theta = np.empty_like(D_star, dtype=float)
        m1 = D_star <= 4.0
        m2 = (D_star > 4.0) & (D_star <= 10.0)
        m3 = (D_star > 10.0) & (D_star <= 20.0)
        m4 = (D_star > 20.0) & (D_star <= 150.0)
        m5 = D_star > 150.0
        theta[m1] = 0.24 * D_star[m1] ** -1.0
        theta[m2] = 0.14 * D_star[m2] ** -0.64
        theta[m3] = 0.04 * D_star[m3] ** -0.10
        theta[m4] = 0.013 * D_star[m4] ** 0.29
        theta[m5] = 0.056
        return theta

    def theta_cr_constant(self, method):
        """
        Return a constant critical Shields parameter for selected literature formulas.
            laursen -> 0.039
            mpm     -> 0.047
            wu      -> 0.030
        """
        if method == 'laursen':
            return 0.039
        elif method == 'mpm':
            return 0.047
        elif method == 'wu':
            return 0.030
        raise ValueError("method must be one of: 'laursen', 'mpm', 'wu'")

    def tau_ce_owen(self, rho_d, a, b):
        """
        Compute cohesive critical erosion stress [Pa] using an Owen-type dry-density relation.
            tau_ce = a * rho_d^b
        where:
            rho_d = dry bulk density
            a, b  = empirical coefficients
        """
        if rho_d <= 0:
            raise ValueError('rho_d must be > 0')
        return a * (rho_d ** b)

    def get_resuspension_critstress(self, idx=None):
        """
        Return the local resuspension critical shear stress [Pa].
        1) USER
           a) If critical shear velocity is provided:
                  tau_cr = rho_w * ustar_cr^2
           b) Otherwise use the configured constant:
                  tau_cr = resuspension_critstress
        2) FROM_D50
           a) Determine cohesive vs noncohesive branch
           b) NONCOHESIVE:
                  theta_cr <- selected Shields relation
                  tau_cr   = theta_cr * (rho_s - rho_w) * g * d50
           c) COHESIVE:
                  tau_cr = a * rho_d^b
        Optional heterogeneity
        If enabled:
            tau_cr <- tau_cr * critstress_factor
        This lets each bed element carry a persistent multiplicative modifier.
        """
        if idx is None:
            n = self.num_elements_active()
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()
            n = idx.size

        mode = self.get_config('chemical:sediment:resuspension_critstress_mode')

        if mode == 'USER':
            ustar_cr = float(self.get_config('chemical:sediment:resuspension_critustar'))

            if ustar_cr >= 0.0:
                # Convert user-specified critical shear velocity to critical shear stress
                T = self._env_array('sea_water_temperature', 10.0, idx=idx)
                S = self._env_array('sea_water_salinity', 34.0, idx=idx)
                rho_w = self.sea_water_density(T=T, S=S)
                tau_cr = rho_w * (ustar_cr ** 2)
            else:
                tau_cr = np.full(n,
                    float(self.get_config('chemical:sediment:resuspension_critstress')),
                    dtype=float)

        elif mode == 'FROM_D50':
            d50_m = self._element_or_local_d50(idx=idx)
            d50_m = np.asarray(d50_m, dtype=float)
            if np.any(d50_m <= 0.0):
                raise ValueError('A positive element/local d50 is required when resuspension_critstress_mode == FROM_D50')
            branch = self._resuspension_branch(idx=idx)
            if isinstance(branch, str):
                branch_arr = np.full(n, branch, dtype='<U12')
            else:
                branch_arr = np.asarray(branch, dtype='<U12')
                if branch_arr.size != n:
                    raise ValueError('Local resuspension branch array has wrong size')
            tau_cr = np.zeros(n, dtype=float)
            rho_s = float(self.get_config('chemical:sediment:critstress_rho_s'))
            nu = float(self.get_config('chemical:sediment:critstress_nu'))
            T = self._env_array('sea_water_temperature', 10.0, idx=idx)
            S = self._env_array('sea_water_salinity', 34.0, idx=idx)
            rho_w = self.sea_water_density(T=T, S=S)
            mask_non = branch_arr == 'NONCOHESIVE'
            if np.any(mask_non):
                method = self.get_config('chemical:sediment:resuspension_critstress_method')
                d50_non = d50_m[mask_non]
                rho_w_non = rho_w[mask_non]
                if method == 'soulsby_whitehouse':
                    theta = self.theta_cr_soulsby_whitehouse(d50_non, rho_s=rho_s, rho_w=rho_w_non, nu=nu, g=9.81)
                elif method == 'van_rijn':
                    theta = self.theta_cr_van_rijn(d50_non, rho_s=rho_s, rho_w=rho_w_non, nu=nu, g=9.81)
                elif method in {'laursen', 'mpm', 'wu'}:
                    theta = np.full(np.sum(mask_non), self.theta_cr_constant(method), dtype=float)
                else:
                    raise ValueError(f'Unknown noncohesive method: {method!r}')
                tau_cr[mask_non] = np.maximum(theta * (rho_s - rho_w_non) * 9.81 * d50_non, 0.0)
            mask_coh = branch_arr == 'COHESIVE'
            if np.any(mask_coh):
                rho_d = float(self.get_config('chemical:sediment:critstress_owen_rho_d'))
                a = float(self.get_config('chemical:sediment:critstress_owen_a'))
                b = float(self.get_config('chemical:sediment:critstress_owen_b'))
                if rho_d <= 0 or a <= 0 or b <= 0:
                    raise ValueError('Cohesive resuspension_critstress requires positive critstress_owen_rho_d, critstress_owen_a, and critstress_owen_b')
                tau0 = self.tau_ce_owen(rho_d=rho_d, a=a, b=b)
                tau_cr[mask_coh] = tau0

        else:
            raise ValueError(f'Unknown resuspension_critstress_mode: {mode!r}')

        tau_cr_map = self._local_resuspension_critstress_map(idx=idx)
        if tau_cr_map is not None:
            tau_cr_map = np.asarray(tau_cr_map, dtype=float)
            if tau_cr_map.ndim == 0:
                tau_cr_map = np.full(n, float(tau_cr_map), dtype=float)
            elif tau_cr_map.size != n:
                raise ValueError('Local resuspension critical-stress map has wrong size')
            valid_map = np.isfinite(tau_cr_map) & (tau_cr_map > 0.0)
            if np.any(valid_map):
                tau_cr = np.where(valid_map, tau_cr_map, tau_cr)

        if self.get_config('chemical:sediment:use_critstress_heterogeneity'):
            if idx is None:
                eta = np.asarray(self.elements.critstress_factor, dtype=float)
            else:
                eta = np.asarray(self.elements.critstress_factor[idx], dtype=float)
            tau_cr = tau_cr * np.maximum(eta, 1e-12)

        return tau_cr

    def krone_deposition_reduction(self, tau, tau_cr_dep):
        """
        Krone-type deposition reduction factor.
            alpha_d = max(0, 1 - tau / tau_cr_dep)
        """
        tau = np.asarray(tau, dtype=float)
        tau_cr_dep = np.asarray(tau_cr_dep, dtype=float)
        return np.maximum(0.0, 1.0 - tau / np.maximum(tau_cr_dep, 1e-30))

    def linear_excess_shear_factor(self, tau, tau_cr_res):
        """
        Linear excess-shear factor above the resuspension threshold.
            factor = max(0, tau / tau_cr_res - 1)
        This is the time-independent excess-stress form used by the
        INSTANTANEOUS_EXCESS_SHEAR resuspension model.
        """
        tau = np.asarray(tau, dtype=float)
        tau_cr_res = np.asarray(tau_cr_res, dtype=float)
        return np.maximum(0.0, tau / np.maximum(tau_cr_res, 1e-30) - 1.0)

    def deposition_probability(self, tau, ws, dt, h_b):
        """
        Compute deposition probability over one timestep.

        1) Deposition reduction factor:
           a) if a constant deposition_reduction_factor is configured:
                  alpha_d = constant
           b) otherwise use Krone-type reduction:
                  alpha_d = max(0, 1 - tau / tau_cr_dep)
        2) Deposition hazard:
               k_dep = ws * alpha_d / h_b
           where:
               ws   = settling velocity magnitude toward the bed [m/s]
               h_b  = near-bed interaction-layer thickness [m]
        3) Timestep probability:
               p_dep = 1 - exp(-k_dep * dt)
        """
        tau_cr_dep = float(self.get_config('chemical:sediment:deposition_critstress'))
        h_b = np.maximum(np.asarray(h_b, dtype=float), 1e-30)
        ws = np.maximum(np.asarray(ws, dtype=float), 0.0)

        dep_red = float(self.get_config('chemical:sediment:deposition_reduction_factor'))
        if dep_red >= 0.0:
            alpha_d = np.full_like(np.asarray(tau, dtype=float), dep_red, dtype=float)
        else:
            alpha_d = self.krone_deposition_reduction(tau, tau_cr_dep)

        k_dep = ws * alpha_d / h_b
        p_dep = (1.0 - np.exp(-k_dep * dt))

        return np.clip(p_dep, 0.0, 1.0)

    def _local_erodible_mass_per_area(self, idx=None):
        """
        Return the locally erodible bed mass per unit area.
            m_erodible = rho_s * (1 - porosity) * Sed_L_eff * effective_fraction
        where:
            rho_s              = sediment density [kg/m3]
            Sed_L_eff          = active sediment thickness [m]
            effective_fraction = fraction of sorbing / erodible sediment
        This quantity is used to convert cohesive erosion flux [kg m-2 s-1]
        into a resuspension hazard [1/s].
        """
        if idx is None:
            idx = np.arange(self.num_elements_active(), dtype=np.int64)
        else:
            idx = np.asarray(idx, dtype=np.int64).ravel()

        Sed_L_env = self._env_array('active_sediment_layer_thickness', 0.0, idx=idx)
        Sed_L_0 = float(self.get_config('chemical:sediment:mixing_depth'))
        Sed_L_eff = np.where(Sed_L_env > 0.0, Sed_L_env, Sed_L_0)

        rho_s = float(self.get_config('chemical:sediment:density'))
        poro = float(self.get_config('chemical:sediment:porosity'))
        f_eff = float(self.get_config('chemical:sediment:effective_fraction'))

        m_erodible = (
            rho_s *
            np.maximum(1.0 - poro, 0.0) *
            np.maximum(Sed_L_eff, 0.0) *
            np.maximum(f_eff, 0.0)
        )
        return np.asarray(m_erodible, dtype=float)

    def resuspension_probability(self, tau, tau_cr_res=None, dt=None, idx=None):
        """
        Resuspension probability for a single timestep.

        Two selectable models:
        1) INSTANTANEOUS_EXCESS_SHEAR
           Reproduces the previous purely time-independent implementation:
               p = max(0, tau/tau_cr - 1)
           then clipped to [0, 1].
        2) TIMESTEP_DEPENDENT
           Uses timestep-dependent pickup/erosion probability:
               NONCOHESIVE:
                   lambda = ((tau/tau_cr) - 1)^n / T_pickup
                   p = 1 - exp(-lambda * dt)
               COHESIVE:
                   E = M * max(tau - tau_cr, 0)                  [kg m-2 s-1]
                   lambda = E / m_erodible                       [1/s]
                   p = 1 - exp(-lambda * dt)
               where:
                   m_erodible ~= rho_s * (1-porosity) * mixing_depth * effective_fraction
        """
        tau = np.asarray(tau, dtype=float)

        if tau_cr_res is None:
            tau_cr_res = self.get_resuspension_critstress()
        tau_cr_res = np.asarray(tau_cr_res, dtype=float)

        prob_model = self.get_config('chemical:sediment:resuspension_probability_model')

        # time-independent formulation
        if prob_model == 'INSTANTANEOUS_EXCESS_SHEAR':
            p_res = self.linear_excess_shear_factor(tau, tau_cr_res)
            return np.clip(p_res, 0.0, 1.0)

        # timestep-dependent formulation
        if prob_model != 'TIMESTEP_DEPENDENT':
            raise ValueError(f'Unknown resuspension_probability_model: {prob_model!r}')

        if dt is None:
            dt = self.time_step.total_seconds()
        dt = float(dt)

        branch = self._resuspension_branch(idx=idx)
        if isinstance(branch, str):
            branch_arr = np.full(tau.shape, branch, dtype='<U12')
        else:
            branch_arr = np.asarray(branch, dtype='<U12')
            if branch_arr.shape != tau.shape:
                branch_arr = np.asarray(branch_arr).reshape(tau.shape)

        hazard = np.zeros_like(tau, dtype=float)

        mask_coh = branch_arr == 'COHESIVE'
        if np.any(mask_coh):
            M = np.asarray(self._local_erodibility_M(idx=idx), dtype=float)
            if M.ndim == 0:
                M = np.full(tau.shape, float(M), dtype=float)
            m_erodible = self._local_erodible_mass_per_area(idx=idx)
            m_erodible = np.asarray(m_erodible, dtype=float)
            if m_erodible.ndim == 0:
                m_erodible = np.full(tau.shape, float(m_erodible), dtype=float)
            erosion_flux = np.maximum(M[mask_coh], 0.0) * np.maximum(tau[mask_coh] - tau_cr_res[mask_coh], 0.0)
            hazard[mask_coh] = np.where(
                m_erodible[mask_coh] > 0.0,
                erosion_flux / np.maximum(m_erodible[mask_coh], 1e-30),
                0.0,
            )

        mask_non = branch_arr == 'NONCOHESIVE'
        if np.any(mask_non):
            T_pickup = float(self.get_config('chemical:sediment:noncohesive_resuspension_timescale'))
            expo = float(self.get_config('chemical:sediment:noncohesive_excess_shear_exponent'))

            T_pickup = max(T_pickup, 1e-12)
            expo = max(expo, 1e-12)

            excess = np.maximum(tau[mask_non] / np.maximum(tau_cr_res[mask_non], 1e-30) - 1.0, 0.0)
            hazard[mask_non] = (excess ** expo) / T_pickup                   # 1/s

        unknown = ~(mask_coh | mask_non)
        if np.any(unknown):
            raise ValueError(f'Unknown resuspension branch values: {np.unique(branch_arr[unknown]).tolist()}')

        p_res = 1.0 - np.exp(-hazard * dt)
        return np.clip(p_res, 0.0, 1.0)

    def _sample_critstress_factor(self, n):
        """
        Sample a persistent multiplicative heterogeneity factor for critical stress.

        If sigma_ln <= 0:
            eta = 1
        Otherwise:
            eta ~ LogNormal(mu_ln, sigma_ln)
        with:
            mu_ln = -0.5 * sigma_ln^2
        so that:
            E[eta] = 1
        This preserves the mean critical stress while introducing sub-grid spatial variability.
        """
        sigma_ln = float(self.get_config('chemical:sediment:critstress_heterogeneity_lnsigma'))
        if sigma_ln <= 0.0:
            return np.ones(n, dtype=np.float32)

        mu_ln = -0.5 * sigma_ln**2   # ensures mean(eta)=1
        return np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=n).astype(np.float32)

    ###########################################################################
    # Helpers for bed-interaction diagnostics
    ###########################################################################

    def _save_bed_interaction(self):
        return (
            bool(self.get_config('chemical:sediment:save_bed_interaction'))
            and hasattr(self.elements, 'tau_effective')
        )

    def _reset_bed_interaction_variables(self):
        """
        Reset dense bed-interaction diagnostics for this timestep.

        This is intentionally O(N) when diagnostics are enabled. If these variables
        are written to output, resetting is required to avoid stale values from the
        previous timestep.
        """
        if not self._save_bed_interaction():
            return
        for name in Chemical.BED_INTERACTION_VARIABLE_NAMES:
            if hasattr(self.elements, name):
                getattr(self.elements, name).fill(np.nan)

    def _store_bed_interaction(
        self,
        idx,
        stress,
        p_dep=None,
        p_res=None,
        tau_cr_res=None,
    ):
        if not self._save_bed_interaction():
            return

        self.elements.tau_bx[idx] = stress['tau_bx']
        self.elements.tau_by[idx] = stress['tau_by']
        self.elements.tau_current[idx] = stress['tau_current']
        self.elements.tau_effective[idx] = stress['tau_effective']
        self.elements.tau_effective_x[idx] = stress['tau_effective_x']
        self.elements.tau_effective_y[idx] = stress['tau_effective_y']
        self.elements.ustar_effective[idx] = stress['ustar_effective']
        self.elements.rho[idx] = stress['rho']
        self.elements.Cd[idx] = stress['Cd']
        self.elements.speed[idx] = stress['speed']
        self.elements.z_ref[idx] = stress['z_ref']

        if stress.get('z0', None) is not None:
            self.elements.z0[idx] = stress['z0']

        self.elements.tau_wave[idx] = stress['tau_wave']
        for name in ('wave_orbital_velocity', 'wave_excursion', 'wave_number',
                     'wave_friction_factor', 'wave_z0', 'wave_water_depth'):
            values = np.asarray(stress[name], dtype=float)
            target = getattr(self.elements, name)
            # Float32 output may not represent very large deep-water friction
            # factors; retain finite stress but mark unavailable diagnostics NaN.
            limit = np.finfo(target.dtype).max
            values = np.where(np.isfinite(values) & (np.abs(values) <= limit), values, np.nan)
            target[idx] = values

        if p_dep is not None:
            self.elements.p_dep[idx] = p_dep

        if p_res is not None:
            self.elements.p_res[idx] = p_res

        if tau_cr_res is not None:
            self.elements.tau_cr_res[idx] = tau_cr_res

    ###########################################################################
    # Main sediment dynamics functions
    ###########################################################################

    def _empty_bed_update(self):
        """Return empty compact update arrays for bed_exchange()."""
        return (
            np.empty(0, dtype=np.int64),                         # idx
            np.empty(0, dtype=np.asarray(self.elements.specie).dtype),
            np.empty(0, dtype=np.asarray(self.elements.z).dtype),
            np.empty(0, dtype=np.asarray(self.elements.moving).dtype),
            np.empty(0, dtype=np.asarray(self.elements.critstress_factor).dtype),
        )

    def deposition(self, specie0, z0, dt):
        """
        Helper for bed_exchange().
        Decide suspended -> bed deposition using only the start-of-timestep state.
        Inputs:
          - specie0, z0: frozen initial species and depth arrays
          - dt: timestep [s]
        Eligible initial suspended species:
          - Particle reversible           -> Sediment reversible
          - Particle slowly reversible    -> Sediment slowly reversible
          - Particle irreversible         -> Sediment irreversible

        Only initially suspended elements inside the local near-bed interaction layer
        are considered. Deposition decisions are based only on the initial state, so
        elements newly resuspended later in the same timestep are not eligible here.

        Returns compact update arrays:
            idx, new_specie, new_z, new_moving, new_critstress_factor
        """
        if not self.get_config('chemical:sediment:enable_deposition'):
            return self._empty_bed_update()

        has_rev = hasattr(self, 'num_prev') and hasattr(self, 'num_srev')
        has_slow = hasattr(self, 'num_psrev') and hasattr(self, 'num_ssrev')
        has_irrev = hasattr(self, 'num_pirrev') and hasattr(self, 'num_sirrev')

        if not (has_rev or has_slow or has_irrev):
            logger.debug('No particle elements initiated, deposition was skipped')
            return self._empty_bed_update()

        # Build candidate indices directly instead of creating a full N-sized boolean mask.
        idx_parts = []

        if has_rev:
            idx = np.flatnonzero(specie0 == self.num_prev)
            if idx.size:
                idx_parts.append(idx)

        if has_slow:
            idx = np.flatnonzero(specie0 == self.num_psrev)
            if idx.size:
                idx_parts.append(idx)

        if has_irrev:
            idx = np.flatnonzero(specie0 == self.num_pirrev)
            if idx.size:
                idx_parts.append(idx)

        if not idx_parts:
            logger.debug('No particle elements present, deposition was skipped')
            return self._empty_bed_update()

        idx_susp = np.concatenate(idx_parts)

        z_susp = np.asarray(z0[idx_susp], dtype=float)
        Zmin_susp = -self._env_array(
            'sea_floor_depth_below_sea_level',
            10000.0,
            idx=idx_susp,
        )

        # Near-bed interaction thickness.
        h_env = self._env_array(
            'interaction_sediment_layer_thickness',
            0.0,
            idx=idx_susp,
        )
        h_cfg = float(self.get_config('chemical:sediment:layer_thickness'))

        # Use reader value if available, otherwise config fallback.
        h_b_nominal = np.where(h_env > 0.0, h_env, h_cfg)

        # Local bathymetry / water-column thickness limit.
        depth_susp = np.asarray(
            self._env_array(
                'sea_floor_depth_below_sea_level',
                10000.0,
                idx=idx_susp,
            ),
            dtype=float,
        )
        depth_susp = np.maximum(depth_susp, 0.0)

        # Effective near-bed interaction thickness cannot exceed local depth.
        h_b_susp = np.minimum(h_b_nominal, depth_susp)
        h_b_susp = np.maximum(h_b_susp, 1e-12)

        dist_above_bed = z_susp - Zmin_susp
        near_bed_local = (dist_above_bed >= 0.0) & (dist_above_bed <= h_b_susp)

        if not np.any(near_bed_local):
            logger.debug('No particle elements present near seabed, deposition was skipped')
            return self._empty_bed_update()

        idx_dep = idx_susp[near_bed_local]
        Zmin_dep = Zmin_susp[near_bed_local]
        h_b_dep = h_b_susp[near_bed_local]

        stress_dep = self.compute_bottom_shear_stress(idx=idx_dep)
        tau_dep = stress_dep['tau_effective']

        ws_dep = np.maximum(
            -np.asarray(self.elements.terminal_velocity[idx_dep], dtype=float),
            0.0,
        )

        p_dep = self.deposition_probability(
            tau=tau_dep,
            ws=ws_dep,
            dt=dt,
            h_b=h_b_dep,
        )

        # Optional debug storage.
        self._store_bed_interaction(
            idx_dep,
            stress_dep,
            p_dep=p_dep,
        )

        hit_dep_local = np.random.random(idx_dep.size) < p_dep

        if not np.any(hit_dep_local):
            logger.debug('No particle elements hit deposition probability, deposition was skipped')
            return self._empty_bed_update()

        ii = idx_dep[hit_dep_local]
        Zhit = Zmin_dep[hit_dep_local]

        new_specie = np.empty(ii.size, dtype=np.asarray(self.elements.specie).dtype)
        new_z = np.empty(ii.size, dtype=np.asarray(self.elements.z).dtype)
        new_moving = np.empty(ii.size, dtype=np.asarray(self.elements.moving).dtype)
        new_crit = np.empty(
            ii.size,
            dtype=np.asarray(self.elements.critstress_factor).dtype,
        )

        new_z[:] = Zhit
        new_moving[:] = 0

        use_heterogeneity = self.get_config(
            'chemical:sediment:use_critstress_heterogeneity'
        )

        if has_rev:
            loc = specie0[ii] == self.num_prev
            if np.any(loc):
                n = np.count_nonzero(loc)
                new_specie[loc] = self.num_srev
                if use_heterogeneity:
                    new_crit[loc] = self._sample_critstress_factor(n)
                else:
                    new_crit[loc] = 1.0
                self.ntransformations[self.num_prev, self.num_srev] += n

        if has_slow:
            loc = specie0[ii] == self.num_psrev
            if np.any(loc):
                n = np.count_nonzero(loc)
                new_specie[loc] = self.num_ssrev
                if use_heterogeneity:
                    new_crit[loc] = self._sample_critstress_factor(n)
                else:
                    new_crit[loc] = 1.0
                self.ntransformations[self.num_psrev, self.num_ssrev] += n

        if has_irrev:
            loc = specie0[ii] == self.num_pirrev
            if np.any(loc):
                n = np.count_nonzero(loc)
                new_specie[loc] = self.num_sirrev
                if use_heterogeneity:
                    new_crit[loc] = self._sample_critstress_factor(n)
                else:
                    new_crit[loc] = 1.0
                self.ntransformations[self.num_pirrev, self.num_sirrev] += n

        return ii, new_specie, new_z, new_moving, new_crit

    def resuspension(self, specie0, z0, dt):
        """
        Helper for bed_exchange().
        Decide bed -> suspended resuspension using only the start-of-timestep state.
        Inputs:
          - specie0, z0: frozen initial species and depth arrays
          - dt: timestep [s]
        Eligible initial bed species:
          - Sediment reversible           -> Particle reversible
          - Sediment slowly reversible    -> Particle slowly reversible
          - Sediment irreversible         -> Particle irreversible
        Only initially bed-bound elements are considered. Resuspension decisions are
        based only on the initial state, so elements newly deposited earlier in the
        same timestep are not eligible here.
        Returns compact update arrays:
            idx, new_specie, new_z, new_moving, new_critstress_factor
        """
        if not self.get_config('chemical:sediment:enable_resuspension'):
            return self._empty_bed_update()

        has_rev = hasattr(self, 'num_prev') and hasattr(self, 'num_srev')
        has_slow = hasattr(self, 'num_psrev') and hasattr(self, 'num_ssrev')
        has_irrev = hasattr(self, 'num_pirrev') and hasattr(self, 'num_sirrev')

        if not (has_rev or has_slow or has_irrev):
            logger.debug('No sediment species initiated, resuspension was skipped')
            return self._empty_bed_update()

        # Build candidate indices directly instead of creating a full N-sized boolean mask.
        idx_parts = []
        if has_rev:
            idx = np.flatnonzero(specie0 == self.num_srev)
            if idx.size:
                idx_parts.append(idx)
        if has_slow:
            idx = np.flatnonzero(specie0 == self.num_ssrev)
            if idx.size:
                idx_parts.append(idx)
        if has_irrev:
            idx = np.flatnonzero(specie0 == self.num_sirrev)
            if idx.size:
                idx_parts.append(idx)
        if not idx_parts:
            logger.debug('No sediment elements present, resuspension was skipped')
            return self._empty_bed_update()

        idx_bed = np.concatenate(idx_parts)

        z_bed = np.asarray(z0[idx_bed], dtype=float)
        Zmin_bed = -self._env_array(
            'sea_floor_depth_below_sea_level',
            10000.0,
            idx=idx_bed,
        )

        on_bed_local = z_bed <= Zmin_bed

        if not np.any(on_bed_local):
            return self._empty_bed_update()

        idx_res = idx_bed[on_bed_local]
        Zmin_res = Zmin_bed[on_bed_local]

        stress_res = self.compute_bottom_shear_stress(idx=idx_res)
        tau_res = stress_res['tau_effective']
        tau_cr_res = self.get_resuspension_critstress(idx=idx_res)

        p_res = self.resuspension_probability(
            tau=tau_res,
            tau_cr_res=tau_cr_res,
            dt=dt,
            idx=idx_res,
        )

        # Optional debug storage.
        self._store_bed_interaction(
            idx_res,
            stress_res,
            p_res=p_res,
            tau_cr_res=tau_cr_res,
        )

        hit_res_local = np.random.random(idx_res.size) < p_res

        if not np.any(hit_res_local):
            return self._empty_bed_update()

        ii = idx_res[hit_res_local]
        Zhit = Zmin_res[hit_res_local]

        resuspension_depth = float(
            self.get_config('chemical:sediment:resuspension_depth')
        )
        std = float(
            self.get_config('chemical:sediment:resuspension_depth_uncert')
        )

        new_specie = np.empty(ii.size, dtype=np.asarray(self.elements.specie).dtype)
        new_z = np.empty(ii.size, dtype=np.asarray(self.elements.z).dtype)
        new_moving = np.empty(ii.size, dtype=np.asarray(self.elements.moving).dtype)
        new_crit = np.empty(
            ii.size,
            dtype=np.asarray(self.elements.critstress_factor).dtype,
        )

        new_moving[:] = 1
        new_crit[:] = 1.0

        if has_rev:
            loc = specie0[ii] == self.num_srev
            if np.any(loc):
                n = np.count_nonzero(loc)
                depth_jj = -Zhit[loc]
                depth_jj = np.maximum(depth_jj, 0.0)

                resuspension_depth_eff = np.minimum(
                    resuspension_depth,
                    depth_jj,
                )

                z_tmp = -depth_jj + resuspension_depth_eff

                if std > 0:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            'Adding uncertainty for resuspension from sediments: %s m',
                            std,
                        )
                    z_tmp += np.random.normal(0.0, std, n)

                z_tmp = np.maximum(z_tmp, -depth_jj)
                z_tmp = np.minimum(z_tmp, 0.0)

                new_specie[loc] = self.num_prev
                new_z[loc] = z_tmp
                self.ntransformations[self.num_srev, self.num_prev] += n

        if has_slow:
            loc = specie0[ii] == self.num_ssrev
            if np.any(loc):
                n = np.count_nonzero(loc)
                depth_jj = -Zhit[loc]
                depth_jj = np.maximum(depth_jj, 0.0)

                resuspension_depth_eff = np.minimum(
                    resuspension_depth,
                    depth_jj,
                )

                z_tmp = -depth_jj + resuspension_depth_eff

                if std > 0:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            'Adding uncertainty for resuspension from sediments: %s m',
                            std,
                        )
                    z_tmp += np.random.normal(0.0, std, n)

                z_tmp = np.maximum(z_tmp, -depth_jj)
                z_tmp = np.minimum(z_tmp, 0.0)

                new_specie[loc] = self.num_psrev
                new_z[loc] = z_tmp
                self.ntransformations[self.num_ssrev, self.num_psrev] += n

        if has_irrev:
            loc = specie0[ii] == self.num_sirrev
            if np.any(loc):
                n = np.count_nonzero(loc)
                depth_jj = -Zhit[loc]
                depth_jj = np.maximum(depth_jj, 0.0)

                resuspension_depth_eff = np.minimum(
                    resuspension_depth,
                    depth_jj,
                )

                z_tmp = -depth_jj + resuspension_depth_eff

                if std > 0:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            'Adding uncertainty for resuspension from sediments: %s m',
                            std,)
                    z_tmp += np.random.normal(0.0, std, n)

                z_tmp = np.maximum(z_tmp, -depth_jj)
                z_tmp = np.minimum(z_tmp, 0.0)

                new_specie[loc] = self.num_pirrev
                new_z[loc] = z_tmp
                self.ntransformations[self.num_sirrev, self.num_pirrev] += n

        return ii, new_specie, new_z, new_moving, new_crit

    def bed_exchange(self):
        """
        Apply probabilistic bed exchange using a single start-of-timestep snapshot.
        Deposition and resuspension are both evaluated from the same initial
        element state at the beginning of the timestep, so an element cannot undergo
        two bed-exchange transitions within one timestep.

        Start-of-step snapshot:
        The following fields are copied once at the beginning of the routine:
            specie, z, moving, critstress_factor
        All eligibility tests and Monte Carlo draws are based on that initial state.
        Updates are accumulated in new output arrays and applied only once at the end.

        Deposition branch:
        Eligible initial suspended species:
            prev -> srev, psrev -> ssrev, pirrev -> sirrev
        Steps:
          1) Select initially suspended elements.
          2) Restrict to elements inside the local near-bed interaction layer:
                 0 <= z - Zmin <= h_b
          3) Compute local effective bed shear stress:
                 tau_b = compute_bottom_shear_stress(...)
          4) Compute local settling speed toward the bed:
                 ws = max(-terminal_velocity, 0)
          5) Compute timestep deposition probability:
                 p_dep = deposition_probability(tau_b, ws, dt, h_b)
          6) Perform a Monte Carlo deposition draw.
          7) For deposited elements:
                 specie <- corresponding sediment species
                 z      <- Zmin
                 moving <- 0
                 critstress_factor <- sampled heterogeneity factor, or 1.0

        Resuspension branch:
        Eligible initial bed species:
            srev -> prev, ssrev -> pssrev, sirrev -> pirrev
        Steps:
          1) Select initially bed-bound elements.
          2) Restrict to elements that are at or below the local seabed:
                 z <= Zmin
          3) Compute local effective bed shear stress:
                 tau_b = compute_bottom_shear_stress(...)
          4) Compute local critical resuspension stress:
                 tau_cr = get_resuspension_critstress(...)
          5) Compute timestep resuspension probability:
                 p_res = resuspension_probability(tau_b, tau_cr, dt)
          6) Perform a Monte Carlo resuspension draw.
          7) For resuspended elements:
                 specie <- corresponding particle species
                 z      <- Zmin + resuspension_depth
                 moving <- 1
                 critstress_factor <- 1.0
          8) Optionally add Gaussian resuspension-depth uncertainty and clip to:
                 -local_depth <= z <= 0

        Additional rules
          - Buried sediment remains immobile.
          - Elements are clipped so they cannot end above the sea surface.
          - f_OC is preserved across deposition and resuspension.
          - Diameter is updated once at the end using initial -> final species.

          - copy only specie0, because it is needed after in-place updates
          - use z0 as a read-only start-of-step view
          - do not create full-size specie_new, z_new, moving_new, or crit_new arrays
          - deposition() and resuspension() return compact updates only for hit elements
          - apply updates directly to self.elements at the end
        """
        # Reset first, before early returns, so disabled/no-hit timesteps
        # do not keep diagnostic values from the previous timestep.
        self._reset_bed_interaction_variables()

        do_dep = self.get_config('chemical:sediment:enable_deposition')
        do_res = self.get_config('chemical:sediment:enable_resuspension')

        if not (do_dep or do_res):
            return

        has_rev = hasattr(self, 'num_prev') and hasattr(self, 'num_srev')
        has_slow = hasattr(self, 'num_psrev') and hasattr(self, 'num_ssrev')
        has_irrev = hasattr(self, 'num_pirrev') and hasattr(self, 'num_sirrev')

        if not (has_rev or has_slow or has_irrev):
            return

        dt = float(self.time_step.total_seconds())
        # Start-of-step snapshot.
        #
        # specie0 must be copied because self.elements.specie will be updated
        # in place below, while specie0 is still needed for diameter/d50 updates.
        specie0 = np.asarray(self.elements.specie, dtype=np.int32).copy()

        # z0 is only read before any in-place z update is applied, so a view is enough.
        # If future code modifies self.elements.z inside deposition/resuspension,
        # change this to .copy().
        z0 = np.asarray(self.elements.z)

        # Compute compact updates from the same start-of-step snapshot.
        dep_update = self.deposition(
            specie0=specie0,
            z0=z0,
            dt=dt,
        )
        res_update = self.resuspension(
            specie0=specie0,
            z0=z0,
            dt=dt,
        )
        updates = [u for u in (dep_update, res_update) if u[0].size > 0]

        # Buried sediment remains immobile.
        # This updates only buried indices rather than making a full moving_new copy.
        if hasattr(self, 'num_sburied'):
            idx_buried = np.flatnonzero(specie0 == self.num_sburied)
            if idx_buried.size:
                self.elements.moving[idx_buried] = 0

        if not updates:
            return

        idx_all = np.concatenate([u[0] for u in updates])
        specie_all = np.concatenate([u[1] for u in updates])
        z_all = np.concatenate([u[2] for u in updates])
        moving_all = np.concatenate([u[3] for u in updates])
        crit_all = np.concatenate([u[4] for u in updates])

        # Global surface clip
        z_all = np.minimum(z_all, 0.0)

        old_species_all = specie0[idx_all]
        new_species_all = specie_all

        self.elements.specie[idx_all] = specie_all
        self.elements.z[idx_all] = z_all
        self.elements.moving[idx_all] = moving_all
        self.elements.critstress_factor[idx_all] = crit_all
        self.update_chemical_diameter(
            changed_idx=idx_all,
            old_species=old_species_all,
            new_species=new_species_all,
        )
        self.update_chemical_d50(
            changed_idx=idx_all,
            old_species=old_species_all,
            new_species=new_species_all,
        )

    ###########################################################################
    # Helpers for optional sediment-oxygen diagnostics
    ###########################################################################

    def _save_sediment_oxygen_diagnostics(self):
        """Return True when the selected O2 diagnostic schema is allocated."""
        return (
            bool(self.get_config('chemical:sediment:save_oxygen_diagnostics'))
            and hasattr(self.elements, 'sed_o2_used')
        )

    def _reset_sediment_oxygen_diagnostics(self):
        """Reset allocated sediment-O2 diagnostics to NaN for this timestep."""
        if not self._save_sediment_oxygen_diagnostics():
            return
        for name in Chemical.SEDIMENT_OXYGEN_DIAGNOSTIC_DEFINITIONS:
            if hasattr(self.elements, name):
                getattr(self.elements, name).fill(np.nan)

    def _store_sediment_oxygen_diagnostics(self, idx, diagnostics):
        """Store one compact diagnostic dictionary at global element indices."""
        if not self._save_sediment_oxygen_diagnostics():
            return

        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return

        for name, values in diagnostics.items():
            if not hasattr(self.elements, name):
                raise RuntimeError(
                    'Sediment-oxygen diagnostic schema mismatch: calculation '
                    f'produced {name!r}, but that variable is not allocated.'
                )

            values = np.asarray(values, dtype=float)
            if values.ndim == 0:
                values = np.full(idx.size, float(values), dtype=float)
            else:
                values = values.ravel()
                if values.size != idx.size:
                    raise ValueError(
                        f'Sediment-oxygen diagnostic {name!r} has {values.size} '
                        f'value(s) for {idx.size} element index/indices.'
                    )

            getattr(self.elements, name)[idx] = values

    ###########################################################################
    # Reduced-order sediment oxygen helpers
    ###########################################################################

    def _sediment_oxygen_map_or_config(self, env_name, config_name, idx, *,
                                       ge=None, gt=None, le=None, lt=None):
        """
        Return a local sediment-O2 input using the model-wide map -> config rule.

        A mapped value is used only when the variable is genuinely supplied by
        a reader and the local value is finite and satisfies the requested
        bounds. Missing/invalid mapped values are replaced element-by-element by
        the configured fallback.

        This helper intentionally uses _optional_env_array(), so OpenDrift
        environment fallbacks are not mistaken for genuine mapped inputs.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return np.empty(0, dtype=float)

        fallback = self._validate_scalar_param(
            config_name,
            self.get_config(config_name),
            ge=ge, gt=gt, le=le, lt=lt,
        )
        out = np.full(idx.size, fallback, dtype=float)

        mapped = self._optional_env_array(env_name, idx=idx)
        if mapped is None:
            return out

        mapped = np.asarray(mapped, dtype=float)
        valid = np.isfinite(mapped)
        if ge is not None:
            valid &= mapped >= ge
        if gt is not None:
            valid &= mapped > gt
        if le is not None:
            valid &= mapped <= le
        if lt is not None:
            valid &= mapped < lt

        out[valid] = mapped[valid]

        if np.any(~valid):
            logger.debug(
                "%s: replaced %d invalid mapped value(s) with %s",
                env_name, int(np.count_nonzero(~valid)), config_name,
            )
        return out

    def _bottom_water_oxygen_for_sediment(self, idx):
        """
        Return bottom-water dissolved O2 for sediment calculations [mmol/m3].

        Values must be finite and non-negative.
        Conversion to g/m3 for the biodegradation O2-rate calculation is deliberately left to the degradation coupling.
        """
        oxygen = self._env_array(
            'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water',
            225.0, idx=idx,)
        return self._validate_array_param(
            "bottom_water_oxygen", oxygen, ge=0.0)

    def _local_active_sediment_layer_thickness(self, idx):
        """
        Return active sediment-layer thickness H_a [m].

        Priority:
          1) reader-supplied active_sediment_layer_thickness when finite >= 0
          2) chemical:sediment:mixing_depth

        H_a = 0 is allowed. In the averaging helpers it is interpreted as the
        surface-limit case, so the returned active-layer concentration equals
        the sediment-surface concentration.
        """
        return self._sediment_oxygen_map_or_config(
            'active_sediment_layer_thickness',
            'chemical:sediment:mixing_depth',
            idx,
            ge=0.0,
        )

    def _local_sediment_oxygen_porosity(self, idx):
        """
        Return sediment porosity phi [-].

        Priority:
          1) reader-supplied sea_floor_porosity, requiring 0 < phi <= 1
          2) chemical:sediment:porosity
        """
        return self._sediment_oxygen_map_or_config(
            'sea_floor_porosity',
            'chemical:sediment:porosity',
            idx,
            gt=0.0,
            le=1.0,
        )

    def _local_sediment_oxygen_diffusivity(self, idx, porosity=None):
        """
        Return effective O2 diffusivity in sediment pore water D_s [m2/s].

        If sediment_oxygen_diffusivity is genuinely mapped and positive, use it.
        Otherwise derive D_s from the configured free-water molecular
        diffusivity D_0 and porosity using Boudreau (1996):

            theta^2 = 1 - ln(phi^2)
            D_s     = D_0 / theta^2

        Source:
          Boudreau, B.P. (1996), Geochimica et Cosmochimica Acta 60,
          3139-3142. DOI: 10.1016/0016-7037(96)00158-5
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return np.empty(0, dtype=float)

        if porosity is None:
            porosity = self._local_sediment_oxygen_porosity(idx)
        porosity = self._validate_array_param(
            "sediment_oxygen_porosity", porosity, gt=0.0, le=1.0
        )

        D0 = self._validate_scalar_param(
            "chemical:sediment:oxygen_molecular_diffusivity",
            self.get_config('chemical:sediment:oxygen_molecular_diffusivity'),
            gt=0.0,
        )

        theta2 = 1.0 - np.log(porosity * porosity)
        derived = D0 / theta2

        mapped = self._optional_env_array('sediment_oxygen_diffusivity', idx=idx)
        if mapped is None:
            return derived

        mapped = np.asarray(mapped, dtype=float)
        valid = np.isfinite(mapped) & (mapped > 0.0)
        out = derived.copy()
        out[valid] = mapped[valid]

        if np.any(~valid):
            logger.debug(
                "sediment_oxygen_diffusivity: replaced %d invalid mapped "
                "value(s) with Boudreau-derived D_s",
                int(np.count_nonzero(~valid)),
            )
        return out

    @staticmethod
    def _mean_parabolic_sediment_oxygen(C_surface, H_active, L_oxygen):
        """
        Depth-average a zero-order parabolic O2 profile over the active layer.

        Assumed profile:
            C(z) = C_surface * (1 - z/L)^2,       0 <= z <= L
            C(z) = 0,                             z > L

        ChemicalDrift does not track an element's depth inside the active
        sediment layer. The concentration assigned to an active sediment
        element is therefore the layer mean:

            C_active = (1/H) * integral_0^H C(z) dz
        which gives:

            H <= L: C_active = C_surface * [1 - H/L + H^2/(3 L^2)]
            H >  L: C_active = C_surface * L/(3 H)

        The parabolic zero-order diffusion-reaction profile is consistent with
        classical steady sediment O2 models (e.g. Bouldin, 1968) and with the
        uniform-reactivity interpretation discussed by Cai & Sayles (1996).

        Sources:
          Bouldin, D.R. (1968), Journal of Ecology 56, 77-87.
          DOI: 10.2307/2258068
          Cai, W.-J. & Sayles, F.L. (1996), Marine Chemistry 52, 123-131.
          DOI: 10.1016/0304-4203(95)00081-X
        """
        C_surface, H_active, L_oxygen = np.broadcast_arrays(
            np.asarray(C_surface, dtype=float),
            np.asarray(H_active, dtype=float),
            np.asarray(L_oxygen, dtype=float),
        )

        if not np.all(np.isfinite(C_surface)):
            raise ValueError("C_surface contains non-finite values")
        if not np.all(np.isfinite(H_active)):
            raise ValueError("H_active contains non-finite values")
        if np.any(C_surface < 0.0):
            raise ValueError("C_surface must be >= 0")
        if np.any(H_active < 0.0):
            raise ValueError("H_active must be >= 0")
        if np.any(np.isnan(L_oxygen)) or np.any(L_oxygen < 0.0):
            raise ValueError("L_oxygen must be >= 0 or +inf")

        out = np.zeros_like(C_surface, dtype=float)

        # No O2 at the sediment surface, or zero penetration depth -> zero mean.
        usable = (C_surface > 0.0) & (L_oxygen > 0.0)
        if not np.any(usable):
            return out

        # H -> 0 is the sediment-surface limit.
        surface_limit = usable & (H_active == 0.0)
        out[surface_limit] = C_surface[surface_limit]

        finite_layer = usable & (H_active > 0.0)

        within_oxic = finite_layer & (H_active <= L_oxygen)
        if np.any(within_oxic):
            x = H_active[within_oxic] / L_oxygen[within_oxic]
            out[within_oxic] = C_surface[within_oxic] * (
                1.0 - x + x*x / 3.0
            )

        extends_below_oxic = finite_layer & (H_active > L_oxygen)
        if np.any(extends_below_oxic):
            out[extends_below_oxic] = (
                C_surface[extends_below_oxic]
                * L_oxygen[extends_below_oxic]
                / (3.0 * H_active[extends_below_oxic])
            )

        # Protect against tiny floating-point excursions outside physical bounds.
        return np.clip(out, 0.0, C_surface)

    @staticmethod
    def _zero_order_dbl_surface_depth(C_bottom, porosity, D_s, R_oxygen, k_bl):
        """
        Solve the single-layer zero-order diffusion-reaction model with a DBL.

        Sediment equations:
            C_s = R * L^2 / (2 phi D_s)
            F   = R * L = sqrt(2 phi D_s R C_s)

        DBL transfer equation:
            F = k_bl * (C_bottom - C_s)
            k_bl = D_0 / delta_DBL

        Solving analytically for y = sqrt(C_s):
            B = sqrt(2 phi D_s R)
            y = 2 k_bl C_bottom /
                [sqrt(B^2 + 4 k_bl^2 C_bottom) + B]

        and then:
            C_s = y^2
            L   = sqrt(2 phi D_s C_s / R)

        R = 0 is treated as no sediment O2 consumption:
            C_s = C_bottom, L = +inf, F = 0.

        DBL transport basis:
          Jorgensen & Revsbech (1985), Limnology and Oceanography 30,
          111-122. DOI: 10.4319/lo.1985.30.1.0111
        """
        C_bottom, porosity, D_s, R_oxygen, k_bl = np.broadcast_arrays(
            np.asarray(C_bottom, dtype=float),
            np.asarray(porosity, dtype=float),
            np.asarray(D_s, dtype=float),
            np.asarray(R_oxygen, dtype=float),
            np.asarray(k_bl, dtype=float),
        )

        if np.any(~np.isfinite(C_bottom)) or np.any(C_bottom < 0.0):
            raise ValueError("C_bottom must be finite and >= 0")
        if np.any(~np.isfinite(porosity)) or np.any(porosity <= 0.0) or np.any(porosity > 1.0):
            raise ValueError("porosity must be finite with 0 < porosity <= 1")
        if np.any(~np.isfinite(D_s)) or np.any(D_s <= 0.0):
            raise ValueError("D_s must be finite and > 0")
        if np.any(~np.isfinite(R_oxygen)) or np.any(R_oxygen < 0.0):
            raise ValueError("R_oxygen must be finite and >= 0")
        if np.any(~np.isfinite(k_bl)) or np.any(k_bl <= 0.0):
            raise ValueError("k_bl must be finite and > 0")

        C_surface = np.zeros_like(C_bottom, dtype=float)
        L_oxygen = np.zeros_like(C_bottom, dtype=float)
        oxygen_flux = np.zeros_like(C_bottom, dtype=float)

        no_consumption = (R_oxygen == 0.0)
        C_surface[no_consumption] = C_bottom[no_consumption]
        L_oxygen[no_consumption & (C_bottom > 0.0)] = np.inf
        L_oxygen[no_consumption & (C_bottom == 0.0)] = 0.0

        reacting = (R_oxygen > 0.0) & (C_bottom > 0.0)
        if np.any(reacting):
            B = np.sqrt(
                2.0 * porosity[reacting] * D_s[reacting] * R_oxygen[reacting]
            )
            kb = k_bl[reacting]
            Cb = C_bottom[reacting]

            sqrt_C_surface = (
                2.0 * kb * Cb
                / (np.sqrt(B*B + 4.0*kb*kb*Cb) + B)
            )
            Cs = sqrt_C_surface * sqrt_C_surface
            L = np.sqrt(
                2.0 * porosity[reacting] * D_s[reacting] * Cs
                / R_oxygen[reacting]
            )

            C_surface[reacting] = Cs
            L_oxygen[reacting] = L
            oxygen_flux[reacting] = R_oxygen[reacting] * L

        return C_surface, L_oxygen, oxygen_flux

    def _sediment_oxygen_fixed_fraction(self, idx, return_diagnostics=False):
        """FIXED_FRACTION: C_active = f_active * C_bottom."""
        C_bottom = self._bottom_water_oxygen_for_sediment(idx)
        fraction = self._validate_scalar_param(
            "chemical:sediment:oxygen_active_fraction",
            self.get_config('chemical:sediment:oxygen_active_fraction'),
            ge=0.0,
            le=1.0,
        )
        oxygen = fraction * C_bottom
        if not return_diagnostics:
            return oxygen
        return oxygen, {
            'sed_o2_bottom': C_bottom,
        }

    def _sediment_oxygen_prescribed_opd(self, idx, return_diagnostics=False):
        """
        PRESCRIBED_OPD: use a prescribed/mapped oxygen penetration depth L.

        The active-layer oxygen concentration is the analytical mean of the
        parabolic zero-order profile over H_active.
        """
        C_bottom = self._bottom_water_oxygen_for_sediment(idx)
        H_active = self._local_active_sediment_layer_thickness(idx)
        L_oxygen = self._sediment_oxygen_map_or_config(
            'sediment_oxygen_penetration_depth',
            'chemical:sediment:oxygen_penetration_depth',
            idx,
            ge=0.0,
        )
        oxygen = self._mean_parabolic_sediment_oxygen(
            C_bottom, H_active, L_oxygen
        )
        if not return_diagnostics:
            return oxygen
        return oxygen, {
            'sed_o2_bottom': C_bottom,
            'sed_o2_active_layer_thickness': H_active,
            'sed_o2_penetration_depth': L_oxygen,
        }

    def _sediment_oxygen_zero_order(self, idx, return_diagnostics=False):
        """
        ZERO_ORDER: calculate oxygen penetration depth dynamically without DBL.

        Volumetric-demand mode uses the steady zero-order equation:
            phi D_s d2C/dz2 = R
            L = sqrt(2 phi D_s C_bottom / R)

        Benthic-flux mode uses:
            L = 2 phi D_s C_bottom / F_O2

        The latter relationship is reported and evaluated for marine sediments
        by Cai & Sayles (1996), DOI: 10.1016/0304-4203(95)00081-X.

        For R = 0 or F_O2 = 0, oxygen is not depleted in this reduced model and
        L is represented as +inf.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        C_bottom = self._bottom_water_oxygen_for_sediment(idx)
        H_active = self._local_active_sediment_layer_thickness(idx)
        porosity = self._local_sediment_oxygen_porosity(idx)
        D_s = self._local_sediment_oxygen_diffusivity(idx, porosity=porosity)

        L_oxygen = np.full(idx.size, np.inf, dtype=float)
        oxygen_flux = np.zeros(idx.size, dtype=float)
        R_oxygen = None
        demand_mode = self.get_config('chemical:sediment:oxygen_demand_mode')

        if demand_mode == 'VOLUMETRIC_RATE':
            R_oxygen = self._sediment_oxygen_map_or_config(
                'sediment_oxygen_consumption_rate',
                'chemical:sediment:oxygen_consumption_rate',
                idx,
                ge=0.0,
            )
            reacting = (R_oxygen > 0.0) & (C_bottom > 0.0)
            L_oxygen[C_bottom == 0.0] = 0.0
            if np.any(reacting):
                L_oxygen[reacting] = np.sqrt(
                    2.0 * porosity[reacting] * D_s[reacting]
                    * C_bottom[reacting] / R_oxygen[reacting]
                )
                oxygen_flux[reacting] = (
                    R_oxygen[reacting] * L_oxygen[reacting]
                )

        elif demand_mode == 'BENTHIC_FLUX':
            oxygen_flux = self._sediment_oxygen_map_or_config(
                'benthic_oxygen_flux',
                'chemical:sediment:benthic_oxygen_flux',
                idx,
                ge=0.0,
            )
            consuming = (oxygen_flux > 0.0) & (C_bottom > 0.0)
            L_oxygen[(oxygen_flux > 0.0) & (C_bottom == 0.0)] = 0.0
            if np.any(consuming):
                L_oxygen[consuming] = (
                    2.0 * porosity[consuming] * D_s[consuming]
                    * C_bottom[consuming] / oxygen_flux[consuming]
                )
        else:
            raise ValueError(
                f"Unknown chemical:sediment:oxygen_demand_mode: {demand_mode!r}"
            )

        oxygen = self._mean_parabolic_sediment_oxygen(
            C_bottom, H_active, L_oxygen
        )
        if not return_diagnostics:
            return oxygen

        diagnostics = {
            'sed_o2_bottom': C_bottom,
            'sed_o2_active_layer_thickness': H_active,
            'sed_o2_porosity': porosity,
            'sed_o2_diffusivity': D_s,
            'sed_o2_penetration_depth': L_oxygen,
            'sed_o2_flux': oxygen_flux,
        }
        if demand_mode == 'VOLUMETRIC_RATE':
            diagnostics['sed_o2_consumption_rate'] = R_oxygen
        return oxygen, diagnostics

    def _sediment_oxygen_zero_order_dbl(self, idx, return_diagnostics=False):
        """
        ZERO_ORDER_DBL: single-reactivity sediment plus diffusive boundary layer.

        The DBL is represented as a linear transfer resistance:
            F = k_bl (C_bottom - C_surface)
            k_bl = D_0 / delta_DBL

        Source for the DBL concept and its control on sediment O2 uptake:
          Jorgensen & Revsbech (1985), Limnology and Oceanography 30,
          111-122. DOI: 10.4319/lo.1985.30.1.0111

        In BENTHIC_FLUX mode a prescribed flux greater than k_bl*C_bottom is
        physically incompatible with non-negative C_surface and is rejected.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        C_bottom = self._bottom_water_oxygen_for_sediment(idx)
        H_active = self._local_active_sediment_layer_thickness(idx)
        porosity = self._local_sediment_oxygen_porosity(idx)
        D_s = self._local_sediment_oxygen_diffusivity(idx, porosity=porosity)

        D0 = self._validate_scalar_param(
            "chemical:sediment:oxygen_molecular_diffusivity",
            self.get_config('chemical:sediment:oxygen_molecular_diffusivity'),
            gt=0.0,
        )
        dbl = self._sediment_oxygen_map_or_config(
            'diffusive_boundary_layer_thickness',
            'chemical:sediment:oxygen_dbl_thickness',
            idx,
            gt=0.0,
        )
        k_bl = D0 / dbl

        demand_mode = self.get_config('chemical:sediment:oxygen_demand_mode')
        R_oxygen = None

        if demand_mode == 'VOLUMETRIC_RATE':
            R_oxygen = self._sediment_oxygen_map_or_config(
                'sediment_oxygen_consumption_rate',
                'chemical:sediment:oxygen_consumption_rate',
                idx,
                ge=0.0,
            )
            C_surface, L_oxygen, oxygen_flux = self._zero_order_dbl_surface_depth(
                C_bottom, porosity, D_s, R_oxygen, k_bl
            )

        elif demand_mode == 'BENTHIC_FLUX':
            oxygen_flux = self._sediment_oxygen_map_or_config(
                'benthic_oxygen_flux',
                'chemical:sediment:benthic_oxygen_flux',
                idx,
                ge=0.0,
            )

            transport_capacity = k_bl * C_bottom
            tolerance = 1.0e-12 * np.maximum(1.0, transport_capacity)
            impossible = oxygen_flux > (transport_capacity + tolerance)
            if np.any(impossible):
                raise ValueError(
                    "Benthic oxygen flux exceeds DBL transport capacity "
                    "k_bl*C_bottom for one or more sediment elements."
                )

            C_surface = np.maximum(
                C_bottom - oxygen_flux / k_bl,
                0.0,
            )
            L_oxygen = np.full(idx.size, np.inf, dtype=float)
            consuming = oxygen_flux > 0.0
            if np.any(consuming):
                L_oxygen[consuming] = (
                    2.0 * porosity[consuming] * D_s[consuming]
                    * C_surface[consuming] / oxygen_flux[consuming]
                )
        else:
            raise ValueError(
                f"Unknown chemical:sediment:oxygen_demand_mode: {demand_mode!r}"
            )

        oxygen = self._mean_parabolic_sediment_oxygen(
            C_surface, H_active, L_oxygen
        )
        if not return_diagnostics:
            return oxygen

        diagnostics = {
            'sed_o2_bottom': C_bottom,
            'sed_o2_active_layer_thickness': H_active,
            'sed_o2_porosity': porosity,
            'sed_o2_diffusivity': D_s,
            'sed_o2_dbl_thickness': dbl,
            'sed_o2_k_bl': k_bl,
            'sed_o2_surface': C_surface,
            'sed_o2_penetration_depth': L_oxygen,
            'sed_o2_flux': oxygen_flux,
        }
        if demand_mode == 'VOLUMETRIC_RATE':
            diagnostics['sed_o2_consumption_rate'] = R_oxygen
        return oxygen, diagnostics

    @classmethod
    def _mean_two_layer_sediment_oxygen(cls, C_surface, H_active, L_oxygen,
                                        porosity, D_s, R_upper, R_lower,
                                        transition_depth):
        """
        Return the active-layer mean O2 concentration for TWO_LAYER_DBL.

        For finite L > h1, the piecewise zero-order profile is:

          lower layer, h1 <= z <= L:
            C2(z) = R2/(2 phi D_s) * (L - z)^2

          upper layer, 0 <= z <= h1:
            C1(z) = C_h + g_h (z-h1)
                    + R1/(2 phi D_s) * (z-h1)^2

          with:
            C_h = R2/(2 phi D_s) * (L-h1)^2
            g_h = -R2/(phi D_s) * (L-h1)

        The returned concentration is the exact analytical integral of this
        piecewise profile divided by the full active-layer thickness H_active;
        any part of H_active below finite L contributes zero oxygen.

        If R_lower = 0 and oxygen reaches below h1, no finite penetration depth
        exists in this reduced model. The lower layer then has a constant O2
        concentration equal to C(h1), and L is represented as +inf.

        Two-reactivity-layer description is described by:
          Epping, E.H.G. & Helder, W. (1997), Continental Shelf Research 17,
          1737-1764. DOI: 10.1016/S0278-4343(97)00039-3
        """
        (C_surface, H_active, L_oxygen, porosity, D_s,
         R_upper, R_lower, transition_depth) = np.broadcast_arrays(
            np.asarray(C_surface, dtype=float),
            np.asarray(H_active, dtype=float),
            np.asarray(L_oxygen, dtype=float),
            np.asarray(porosity, dtype=float),
            np.asarray(D_s, dtype=float),
            np.asarray(R_upper, dtype=float),
            np.asarray(R_lower, dtype=float),
            np.asarray(transition_depth, dtype=float),
        )

        if np.any(~np.isfinite(C_surface)) or np.any(C_surface < 0.0):
            raise ValueError("C_surface must be finite and >= 0")
        if np.any(~np.isfinite(H_active)) or np.any(H_active < 0.0):
            raise ValueError("H_active must be finite and >= 0")
        if np.any(np.isnan(L_oxygen)) or np.any(L_oxygen < 0.0):
            raise ValueError("L_oxygen must be >= 0 or +inf")
        if np.any(~np.isfinite(porosity)) or np.any(porosity <= 0.0) or np.any(porosity > 1.0):
            raise ValueError("porosity must be finite with 0 < porosity <= 1")
        if np.any(~np.isfinite(D_s)) or np.any(D_s <= 0.0):
            raise ValueError("D_s must be finite and > 0")
        if np.any(~np.isfinite(R_upper)) or np.any(R_upper < 0.0):
            raise ValueError("R_upper must be finite and >= 0")
        if np.any(~np.isfinite(R_lower)) or np.any(R_lower < 0.0):
            raise ValueError("R_lower must be finite and >= 0")
        if np.any(~np.isfinite(transition_depth)) or np.any(transition_depth < 0.0):
            raise ValueError("transition_depth must be finite and >= 0")

        out = np.zeros_like(C_surface, dtype=float)
        oxygenated = C_surface > 0.0
        if not np.any(oxygenated):
            return out

        surface_limit = oxygenated & (H_active == 0.0)
        out[surface_limit] = C_surface[surface_limit]

        finite_H = oxygenated & (H_active > 0.0)
        if not np.any(finite_H):
            return np.clip(out, 0.0, C_surface)

        # If oxygen is exhausted in the upper layer, the profile is simply the
        # single-layer parabolic solution and can use the common exact average.
        upper_only = finite_H & np.isfinite(L_oxygen) & (L_oxygen <= transition_depth)
        if np.any(upper_only):
            out[upper_only] = cls._mean_parabolic_sediment_oxygen(
                C_surface[upper_only],
                H_active[upper_only],
                L_oxygen[upper_only],
            )

        # Finite penetration into the lower reactive layer.
        lower_finite = finite_H & np.isfinite(L_oxygen) & (L_oxygen > transition_depth)
        if np.any(lower_finite):
            H = H_active[lower_finite]
            L = L_oxygen[lower_finite]
            phi = porosity[lower_finite]
            Ds = D_s[lower_finite]
            R1 = R_upper[lower_finite]
            R2 = R_lower[lower_finite]
            h1 = transition_depth[lower_finite]

            a1 = R1 / (2.0 * phi * Ds)
            a2 = R2 / (2.0 * phi * Ds)
            C_h = a2 * (L - h1)**2
            g_h = -R2 * (L - h1) / (phi * Ds)

            x1 = np.minimum(H, h1)
            u0 = -h1
            u1 = x1 - h1
            I_upper = (
                C_h * (u1 - u0)
                + 0.5 * g_h * (u1*u1 - u0*u0)
                + (a1 / 3.0) * (u1*u1*u1 - u0*u0*u0)
            )

            I_lower = np.zeros_like(H)
            reaches_lower = H > h1
            if np.any(reaches_lower):
                x2 = np.minimum(H[reaches_lower], L[reaches_lower])
                I_lower[reaches_lower] = (
                    a2[reaches_lower] / 3.0
                    * (
                        (L[reaches_lower] - h1[reaches_lower])**3
                        - (L[reaches_lower] - x2)**3
                    )
                )

            out[lower_finite] = (I_upper + I_lower) / H

        # Infinite penetration occurs only when the lower-layer consumption is
        # zero and enough O2 remains after crossing the upper layer.
        lower_nonreactive = finite_H & np.isinf(L_oxygen)
        if np.any(lower_nonreactive):
            H = H_active[lower_nonreactive]
            Cs = C_surface[lower_nonreactive]
            phi = porosity[lower_nonreactive]
            Ds = D_s[lower_nonreactive]
            R1 = R_upper[lower_nonreactive]
            h1 = transition_depth[lower_nonreactive]

            a1 = R1 / (2.0 * phi * Ds)
            C_h = np.maximum(Cs - a1*h1*h1, 0.0)

            x1 = np.minimum(H, h1)
            u0 = -h1
            u1 = x1 - h1
            I_upper = (
                C_h * (u1 - u0)
                + (a1 / 3.0) * (u1*u1*u1 - u0*u0*u0)
            )

            I_lower = np.zeros_like(H)
            reaches_lower = H > h1
            I_lower[reaches_lower] = (
                C_h[reaches_lower]
                * (H[reaches_lower] - h1[reaches_lower])
            )

            out[lower_nonreactive] = (I_upper + I_lower) / H

        return np.clip(out, 0.0, C_surface)

    def _sediment_oxygen_two_layer_dbl(self, idx, return_diagnostics=False):
        """
        TWO_LAYER_DBL: two zero-order sediment reactivities plus a DBL.

        Governing equations:

          phi D_s d2C/dz2 = R1,  0 < z < h1
          phi D_s d2C/dz2 = R2,  h1 < z < L

          C(L) = 0,  dC/dz|L = 0
          k_bl (C_bottom - C_surface) = F

        For L > h1:
          C_surface = [R2 L^2 + (R1-R2) h1^2] / (2 phi D_s)
          F = R1 h1 + R2 (L-h1)

        The lower-layer solution for R2 > 0 reduces to one quadratic equation
        in L, solved analytically below. If R2 = 0 and oxygen passes h1, no
        finite penetration depth exists and the lower layer remains at constant
        O2 concentration.

        THe two-layer oxic-zone model with discrete reactivities was taken from
        Epping & Helder (1997), DOI: 10.1016/S0278-4343(97)00039-3.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        C_bottom = self._bottom_water_oxygen_for_sediment(idx)
        H_active = self._local_active_sediment_layer_thickness(idx)
        porosity = self._local_sediment_oxygen_porosity(idx)
        D_s = self._local_sediment_oxygen_diffusivity(idx, porosity=porosity)

        D0 = self._validate_scalar_param(
            "chemical:sediment:oxygen_molecular_diffusivity",
            self.get_config('chemical:sediment:oxygen_molecular_diffusivity'),
            gt=0.0,
        )
        dbl = self._sediment_oxygen_map_or_config(
            'diffusive_boundary_layer_thickness',
            'chemical:sediment:oxygen_dbl_thickness',
            idx,
            gt=0.0,
        )
        k_bl = D0 / dbl

        R_upper = self._sediment_oxygen_map_or_config(
            'sediment_oxygen_consumption_rate_upper',
            'chemical:sediment:oxygen_consumption_rate_upper',
            idx,
            ge=0.0,
        )
        R_lower = self._sediment_oxygen_map_or_config(
            'sediment_oxygen_consumption_rate_lower',
            'chemical:sediment:oxygen_consumption_rate_lower',
            idx,
            ge=0.0,
        )
        h1 = self._sediment_oxygen_map_or_config(
            'sediment_oxygen_reactivity_transition_depth',
            'chemical:sediment:oxygen_reactivity_transition_depth',
            idx,
            ge=0.0,
        )

        C_surface = np.zeros(idx.size, dtype=float)
        L_oxygen = np.zeros(idx.size, dtype=float)
        oxygen_flux = np.zeros(idx.size, dtype=float)

        # Bottom-water O2 required for oxygen to just reach h1. At L=h1 the
        # lower layer has zero thickness and the upper layer controls both
        # surface concentration and flux.
        C_transition = (
            R_upper * h1*h1 / (2.0 * porosity * D_s)
            + R_upper * h1 / k_bl
        )

        upper_only = (C_bottom > 0.0) & (C_bottom <= C_transition)
        if np.any(upper_only):
            Cs_u, L_u, F_u = self._zero_order_dbl_surface_depth(
                C_bottom[upper_only],
                porosity[upper_only],
                D_s[upper_only],
                R_upper[upper_only],
                k_bl[upper_only],
            )
            C_surface[upper_only] = Cs_u
            L_oxygen[upper_only] = L_u
            oxygen_flux[upper_only] = F_u

        enters_lower = C_bottom > C_transition
        finite_lower = enters_lower & (R_lower > 0.0)
        if np.any(finite_lower):
            phi = porosity[finite_lower]
            Ds = D_s[finite_lower]
            kb = k_bl[finite_lower]
            Cb = C_bottom[finite_lower]
            R1 = R_upper[finite_lower]
            R2 = R_lower[finite_lower]
            ht = h1[finite_lower]
            dR = R1 - R2

            a = kb * R2 / (2.0 * phi * Ds)
            b = R2
            c = (
                dR * ht
                + kb * dR * ht*ht / (2.0 * phi * Ds)
                - kb * Cb
            )
            discriminant = b*b - 4.0*a*c
            # Only roundoff-sized negatives are admissible here.
            disc_scale = b*b + np.abs(4.0*a*c)
            bad_disc = discriminant < (
                -1.0e-12 * np.maximum(1.0, disc_scale)
            )
            if np.any(bad_disc):
                raise ValueError(
                    "Negative discriminant in TWO_LAYER_DBL oxygen solution."
                )
            discriminant = np.maximum(discriminant, 0.0)

            # Stable positive root of a*L^2 + b*L + c = 0. In this physical
            # branch c < 0 and the positive root is -2c/(b + sqrt(discriminant)).
            L = -2.0 * c / (b + np.sqrt(discriminant))

            Cs = (
                R2 * L*L + dR * ht*ht
            ) / (2.0 * phi * Ds)

            C_surface[finite_lower] = np.maximum(Cs, 0.0)
            L_oxygen[finite_lower] = L
            oxygen_flux[finite_lower] = R1*ht + R2*(L-ht)

        # If the lower layer has zero demand, oxygen that reaches h1 is not
        # exhausted at any finite depth. Flux is then only the integrated upper
        # layer demand R1*h1 and the lower layer has constant concentration.
        nonreactive_lower = enters_lower & (R_lower == 0.0)
        if np.any(nonreactive_lower):
            F_upper = R_upper[nonreactive_lower] * h1[nonreactive_lower]
            Cs = (
                C_bottom[nonreactive_lower]
                - F_upper / k_bl[nonreactive_lower]
            )
            C_surface[nonreactive_lower] = np.maximum(Cs, 0.0)
            L_oxygen[nonreactive_lower] = np.inf
            oxygen_flux[nonreactive_lower] = F_upper

        oxygen = self._mean_two_layer_sediment_oxygen(
            C_surface,
            H_active,
            L_oxygen,
            porosity,
            D_s,
            R_upper,
            R_lower,
            h1,
        )
        if not return_diagnostics:
            return oxygen

        # Cross-check the independently calculated reaction flux against the
        # DBL transport balance before exporting it.
        flux_dbl = k_bl * (C_bottom - C_surface)
        flux_scale = np.maximum(
            1.0, np.maximum(np.abs(oxygen_flux), np.abs(flux_dbl))
        )
        if np.any(np.abs(oxygen_flux - flux_dbl) > 1.0e-10 * flux_scale):
            raise ValueError(
                'TWO_LAYER_DBL diagnostic flux does not satisfy the DBL balance.'
            )

        return oxygen, {
            'sed_o2_bottom': C_bottom,
            'sed_o2_active_layer_thickness': H_active,
            'sed_o2_porosity': porosity,
            'sed_o2_diffusivity': D_s,
            'sed_o2_dbl_thickness': dbl,
            'sed_o2_k_bl': k_bl,
            'sed_o2_surface': C_surface,
            'sed_o2_penetration_depth': L_oxygen,
            'sed_o2_flux': oxygen_flux,
            'sed_o2_R_upper': R_upper,
            'sed_o2_R_lower': R_lower,
            'sed_o2_transition_depth': h1,
        }

    def calculate_active_sediment_oxygen(self, idx, return_diagnostics=False):
        """
        Calculate effective dissolved O2 for active sediment elements [mmol/m3].

        When return_diagnostics is True, also return a dictionary containing
        exactly the mode-specific internal arrays allocated by the selected
        sediment-oxygen diagnostic schema. The common sed_o2_used field is
        stored by degradation(), because only that caller knows which sediment
        elements are buried and must therefore receive O2 = 0.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            empty = np.empty(0, dtype=float)
            if return_diagnostics:
                return empty, {}
            return empty

        model = self.get_config('chemical:sediment:oxygen_model')

        if model == 'FIXED_FRACTION':
            result = self._sediment_oxygen_fixed_fraction(
                idx, return_diagnostics=return_diagnostics
            )
        elif model == 'PRESCRIBED_OPD':
            result = self._sediment_oxygen_prescribed_opd(
                idx, return_diagnostics=return_diagnostics
            )
        elif model == 'ZERO_ORDER':
            result = self._sediment_oxygen_zero_order(
                idx, return_diagnostics=return_diagnostics
            )
        elif model == 'ZERO_ORDER_DBL':
            result = self._sediment_oxygen_zero_order_dbl(
                idx, return_diagnostics=return_diagnostics
            )
        elif model == 'TWO_LAYER_DBL':
            result = self._sediment_oxygen_two_layer_dbl(
                idx, return_diagnostics=return_diagnostics
            )
        else:
            raise ValueError(
                f"Unknown chemical:sediment:oxygen_model: {model!r}"
            )

        if return_diagnostics:
            oxygen, diagnostics = result
        else:
            oxygen = result
            diagnostics = None

        oxygen = self._validate_array_param(
            "active_sediment_oxygen", oxygen, ge=0.0
        )

        if not return_diagnostics:
            return oxygen

        demand_mode = self.get_config('chemical:sediment:oxygen_demand_mode')
        expected = set(Chemical.sediment_oxygen_diagnostic_variable_names(
            model, demand_mode
        )) - {'sed_o2_used'}
        actual = set(diagnostics)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                'Sediment-oxygen diagnostic dictionary does not match the '
                f'selected schema. Missing={missing}, extra={extra}.'
            )

        return oxygen, diagnostics

    ###########################################################################
    # Helpers for biodegradation, photolysis, hydrolysis
    ###########################################################################

    ### Biodegradation
    def calc_DO_biodegradation_rate(
            self,
            HalfSatO_w,
            k_Anaerobic_water,
            k_Aerobic,
            Ox):
        """
        Return the dissolved-oxygen-dependent biodegradation rate [1/h]
        before pH and temperature corrections.

        The formulation is the direct-rate form of the AQUATOX-type
        aerobic/anaerobic interpolation previously implemented through
        a dimensionless dissolved-oxygen correction factor:

            f_O2 = O2 / (HalfSatO_w + O2)

            k_eff = f_O2 * k_Aerobic
                  + (1 - f_O2) * k_Anaerobic_water

        This is algebraically equivalent to:

            k_eff = k_Aerobic * [
                f_O2
                + (1 - f_O2) * k_Anaerobic_water / k_Aerobic
            ]

        when k_Aerobic > 0, but the direct-rate form is numerically safer
        and also remains valid when k_Aerobic == 0.

        The formulation intentionally permits:

            k_Anaerobic_water > k_Aerobic

        because anaerobic degradation can be faster than aerobic degradation
        for some compounds. In that case the effective rate decreases as O2
        increases, approaching k_Aerobic at high O2.

        Parameters
        ----------
        HalfSatO_w : float
            Oxygen half-saturation constant [g/m3]. Must be > 0.
        k_Anaerobic_water : float
            Anaerobic biodegradation rate constant [1/h]. Must be >= 0.
            The same constant is used for water and sediment.
        k_Aerobic : float
            Aerobic endpoint rate constant [1/h]. Must be >= 0.
            For water this is k_DecayMax_water.
            For sediment this is 4 * k_DecayMax_water.
        Ox : array-like
            Dissolved oxygen concentration [g/m3]. Must be >= 0.

        """
        HalfSatO_w = self._validate_scalar_param(
            "HalfSatO_w", HalfSatO_w, gt=0.0
        )
        k_Anaerobic_water = self._validate_scalar_param(
            "k_Anaerobic_water", k_Anaerobic_water, ge=0.0
        )
        k_Aerobic = self._validate_scalar_param(
            "k_Aerobic", k_Aerobic, ge=0.0
        )
        Ox = self._validate_array_param(
            "Ox", Ox, ge=0.0
        )

        Ox = np.asarray(Ox, dtype=float)
        original_shape = Ox.shape
        Ox_flat = Ox.ravel()
        k_eff_flat = np.empty_like(Ox_flat, dtype=float)

        chunk_size = int(1e5)
        for i in range(0, Ox_flat.size, chunk_size):
            end = min(i + chunk_size, Ox_flat.size)
            Ox_chunk = Ox_flat[i:end]

            f_O2 = Ox_chunk / (HalfSatO_w + Ox_chunk)

            k_eff_flat[i:end] = (
                f_O2 * k_Aerobic
                + (1.0 - f_O2) * k_Anaerobic_water
            )

        k_eff = k_eff_flat.reshape(original_shape)

        if np.any(~np.isfinite(k_eff)):
            raise ValueError(
                "Calculated dissolved-oxygen biodegradation rate contains "
                "non-finite values."
            )

        # Inputs and interpolation weights are non-negative, so negative output
        # should only be possible through an unexpected numerical/code error.
        if np.any(k_eff < -1e-15):
            raise ValueError(
                "Calculated dissolved-oxygen biodegradation rate is negative."
            )

        return np.maximum(k_eff, 0.0)

    def _debug_biodegradation_rate_regime(
            self,
            k_DecayMax_water,
            k_Anaerobic_water):
        """
        Log unusual aerobic/anaerobic rate relationships at DEBUG level only.

        These relationships are scientifically permitted and are therefore
        not validation errors. Messages are emitted only when DEBUG logging is
        enabled and only once for each parameter pair, preventing repeated
        messages at every model timestep.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        k_DecayMax_water = float(k_DecayMax_water)
        k_Anaerobic_water = float(k_Anaerobic_water)
        key = (k_DecayMax_water, k_Anaerobic_water)

        if getattr(self, '_biodegradation_rate_debug_key', None) == key:
            return

        k_DecayMax_sediment = 4.0 * k_DecayMax_water

        if k_DecayMax_water == 0.0 and k_Anaerobic_water == 0.0:
            logger.debug(
                "Biodegradation is enabled but both k_DecayMax_water and "
                "k_Anaerobic_water are 0 1/h; biodegradation rate is zero."
            )

        elif k_DecayMax_water == 0.0 and k_Anaerobic_water > 0.0:
            logger.debug(
                "Biodegradation parameter note: k_DecayMax_water is 0 1/h "
                "while k_Anaerobic_water=%g 1/h. This is permitted; the "
                "oxygen interpolation gives maximum degradation at O2=0 and "
                "approaches zero under strongly aerobic conditions.",
                k_Anaerobic_water,
            )

        if k_Anaerobic_water > k_DecayMax_water:
            logger.debug(
                "Biodegradation parameter note: k_Anaerobic_water=%g 1/h "
                "exceeds the aerobic water endpoint k_DecayMax_water=%g 1/h. "
                "This is permitted; water biodegradation decreases with "
                "increasing O2 between these endpoints.",
                k_Anaerobic_water,
                k_DecayMax_water,
            )

        if k_Anaerobic_water > k_DecayMax_sediment:
            logger.debug(
                "Biodegradation parameter note: k_Anaerobic_water=%g 1/h "
                "exceeds the aerobic sediment endpoint "
                "4*k_DecayMax_water=%g 1/h. This is permitted; sediment "
                "biodegradation decreases with increasing O2 between these "
                "endpoints.",
                k_Anaerobic_water,
                k_DecayMax_sediment,
            )

        self._biodegradation_rate_debug_key = key

    def calc_TCorr(self, T_Max_bio, T_Opt_bio, T_Adp_bio, Max_Accl_bio, Dec_Accl_bio, Q10_bio, TW):
        """
        Correction for the effects of water temperature on biodegradation
          Acclimation magnitude = XM * (1 - exp(-KT * abs(T - TRef)))
          sign(Acclimation) is negative if T < TRef, positive otherwise

          VT = ((TMax + Acclimation) - T) / ((TMax + Acclimation) - (TOpt + Acclimation))
          TCorr = 0 if VT < 0
          else TCorr = (VT**XT) * exp(XT * (1 - VT))

          WT = ln(Q10) * ((TMax + A) - (TOpt + A))
          YT = ln(Q10) * ((TMax + A) - (TOpt + A) + 2)
          XT = (WT**2 * (1 + sqrt(1 + 40/YT))**2) / 400
        """
        T_Max_bio = self._validate_scalar_param("T_Max_bio", T_Max_bio, gt=-273.15)
        T_Opt_bio = self._validate_scalar_param("T_Opt_bio", T_Opt_bio, gt=-273.15)
        T_Adp_bio = self._validate_scalar_param("T_Adp_bio", T_Adp_bio, gt=-273.15)
        Max_Accl_bio = self._validate_scalar_param("Max_Accl_bio", Max_Accl_bio, ge=0.0)
        Dec_Accl_bio = self._validate_scalar_param("Dec_Accl_bio", Dec_Accl_bio, ge=0.0)
        Q10_bio = self._validate_scalar_param("Q10_bio", Q10_bio, gt=1.0)
        TW = self._validate_array_param("TW", TW)

        if T_Max_bio <= T_Opt_bio:
            raise ValueError("T_Max_bio must be > T_Opt_bio")
        if Dec_Accl_bio < 0:
            raise ValueError("Dec_Accl_bio must be >= 0")
        if Max_Accl_bio < 0:
            raise ValueError("Max_Accl_bio must be >= 0")

        TW = np.asarray(TW, dtype=float)
        TCorr = np.zeros_like(TW, dtype=float)

        lnQ10 = np.log(Q10_bio)
        chunk_size = int(1e5)

        for i in range(0, TW.size, chunk_size):
            end = min(i + chunk_size, TW.size)
            T = TW[i:end]
            # Acclimation
            accl_mag = Max_Accl_bio * (1.0 - np.exp(-Dec_Accl_bio * np.abs(T - T_Adp_bio)))
            Acclimation = np.where(T < T_Adp_bio, -accl_mag, accl_mag)

            VT = ((T_Max_bio + Acclimation) - T) / (
                (T_Max_bio + Acclimation) - (T_Opt_bio + Acclimation))

            WT = lnQ10 * ((T_Max_bio + Acclimation) - (T_Opt_bio + Acclimation))
            YT = lnQ10 * ((T_Max_bio + Acclimation) - (T_Opt_bio + Acclimation) + 2.0)
            XT = ((WT ** 2) * (1.0 + np.sqrt(1.0 + 40.0 / YT)) ** 2) / 400.0

            out = np.zeros_like(T, dtype=float)
            valid = VT > 0.0
            out[valid] = (VT[valid] ** XT[valid]) * np.exp(XT[valid] * (1.0 - VT[valid]))

            if np.any((out < 0.0) | (out > 1.0001)):
                bad = np.where((out < 0.0) | (out > 1.0001))[0]
                raise ValueError(
                    f"TCorr outside expected range at local indices {bad[:10].tolist()}")

            TCorr[i:end] = out

        return TCorr

    def calc_pHCorr(self, pH_min_bio, pH_max_bio, pH_water):
        ''' Correction for the effects of water pH on biodegradation
        A piecewise exponential limitation is applied outside the optimal pH range:

            pHCorr = exp(pH - pH_min_bio)    if pH < pH_min_bio
            pHCorr = 1                       if pH_min_bio <= pH <= pH_max_bio
            pHCorr = exp(pH_max_bio - pH)    if pH > pH_max_bio
        '''
        pH_min_bio = self._validate_scalar_param("pH_min_bio", pH_min_bio, ge=0.0, le=14.0)
        pH_max_bio = self._validate_scalar_param("pH_max_bio", pH_max_bio, ge=0.0, le=14.0)
        pH_water = self._validate_array_param("pH_water", pH_water, ge=0.0, le=14.0)

        if pH_min_bio > pH_max_bio:
            raise ValueError("pH_min_bio must be <= pH_max_bio")

        pHCorr = np.ones_like(pH_water)
        N = len(pH_water)
        chunk_size = int(1e5)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N) # Ensure last chunk fits correctly
            # Slice chunk
            pH_chunk = pH_water[i:end]
            # Compute pHCorr based on conditions
            pHCorr[i:end] = np.where(
                pH_chunk < pH_min_bio, np.exp(pH_chunk - pH_min_bio),  # Below range
                np.where(pH_chunk > pH_max_bio, np.exp(pH_max_bio - pH_chunk), 1))  # Above range & default to 1

            if np.any((pHCorr[i:end] < 0) | (pHCorr[i:end] > 1)):
                raise ValueError("pHCorr is not between 0 and 1")
            else:
                pass

        return pHCorr

    # Hydrolysis
    def calc_k_hydro_water(self, k_Acid, k_Base, k_Hydr_Uncat, pH_water):
        ''' Hydrolysis rate constant in water from acid-, base-, and uncatalyzed pathways.

                k_W_hydro = k_hy_Ac + k_hy_Base + k_Hydr_Uncat
            with:
                k_hy_Ac   = k_Acid * 10^(-pH)
                k_hy_Base = k_Base * 10^(pH - 14)
            where:
                k_Acid        = acid-catalyzed pseudo-second-order coefficient [L mol-1 h-1]
                k_Base        = base-catalyzed pseudo-second-order coefficient [L mol-1 h-1]
                k_Hydr_Uncat  = uncatalyzed first-order hydrolysis rate [1/h]
        '''
        # k_Acid and k_Base may be negative by design.
        # Negative values allow the pH-dependent acid/base terms to cancel
        # the uncatalyzed term at specific pH values. The final total rate is
        # clipped to >= 0 after summing the terms.
        k_Acid = self._validate_scalar_param("k_Acid", k_Acid)
        k_Base = self._validate_scalar_param("k_Base", k_Base)
        # Keep uncatalyzed hydrolysis non-negative
        k_Hydr_Uncat = self._validate_scalar_param("k_Hydr_Uncat", k_Hydr_Uncat, ge=0.0)

        pH_water = self._validate_array_param("pH_water", pH_water, ge=0.0, le=14.0)

        k_W_hydro = np.zeros_like(pH_water)

        if (k_Acid == 0 and k_Base == 0 and k_Hydr_Uncat == 0):
            logger.debug("k_Acid, k_Base, k_Hydr_Uncat are set to 0 1/h, therefore no hydolysis occurs in the water")
            return k_W_hydro
        else:
            logger.debug("k_Acid or k_Base or k_Hydr_Uncat are set != 0 1/h, therefore hydolysis occurs in the water")
            N = len(pH_water)  # Total number of elements
            chunk_size = int(1e5)
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)  # Ensure last chunk fits correctly
                # Slice chunk
                pH_chunk = pH_water[i:end]
                # Compute k_hy_Ac and k_hy_Base
                k_hy_Ac = k_Acid * 10**(-pH_chunk)
                k_hy_Base = k_Base * 10**(pH_chunk - 14)
                # Compute final k_W_hydro values
                k_W_hydro[i:end] = k_hy_Ac + k_hy_Base + k_Hydr_Uncat
                # Avoid setting negative values
                k_W_hydro[i:end] = np.clip(k_W_hydro[i:end], 0, None)

        return k_W_hydro

    def calc_k_hydro_sed(self, k_Acid, k_Base, k_Hydr_Uncat, pH_sed):
        ''' Hydrolysis rate constant in sediments from acid-, base-, and uncatalyzed pathways.

                k_S_hydro = k_hy_Ac + k_hy_Base + k_Hydr_Uncat
            with:
                k_hy_Ac   = k_Acid * 10^(-pH_sed)
                k_hy_Base = k_Base * 10^(pH_sed - 14)
            where:
                k_Acid        = acid-catalyzed pseudo-second-order coefficient [L mol-1 h-1]
                k_Base        = base-catalyzed pseudo-second-order coefficient [L mol-1 h-1]
                k_Hydr_Uncat  = uncatalyzed first-order hydrolysis rate [1/h]
    '''
        # k_Acid and k_Base may be negative by design.
        # The final total hydrolysis rate is clipped to >= 0 after summing.
        k_Acid = self._validate_scalar_param("k_Acid", k_Acid)
        k_Base = self._validate_scalar_param("k_Base", k_Base)

        # Keep uncatalyzed hydrolysis non-negative
        k_Hydr_Uncat = self._validate_scalar_param("k_Hydr_Uncat", k_Hydr_Uncat, ge=0.0)

        pH_sed = self._validate_array_param("pH_sed", pH_sed, ge=0.0, le=14.0)

        k_S_hydro = np.zeros_like(pH_sed)

        if (k_Acid == 0 and k_Base == 0 and k_Hydr_Uncat == 0):
            logger.debug("k_Acid, k_Base, k_Hydr_Uncat are set to 0 1/h, therefore no hydolysis occurs in the sediments")
            return k_S_hydro
        else:
            logger.debug("k_Acid or k_Base or k_Hydr_Uncat are set != 0 1/h, therefore hydolysis occurs in the sediments")
            N = len(pH_sed)  # Total number of elements
            chunk_size = int(1e5)
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)  # Ensure last chunk fits correctly
                # Slice chunk
                pH_chunk = pH_sed[i:end]
                # Compute k_hy_Ac and k_hy_Base
                k_hy_Ac = k_Acid * 10**(-pH_chunk)
                k_hy_Base = k_Base * 10**(pH_chunk - 14)
                # Compute final k_S_hydro values
                k_S_hydro[i:end] = k_hy_Ac + k_hy_Base + k_Hydr_Uncat
                # Avoid setting negative values
                k_S_hydro[i:end] = np.clip(k_S_hydro[i:end], 0, None)

        return k_S_hydro

    #Photolysis
    def calc_ScreeningFactor(self, RadDistr, RadDistr0_ml, RadDistr0_bml,
                           WaterExt, ExtCoeffDOM, ExtCoeffSPM, ExtCoeffPHY,
                           C2PHYC, concDOC, concSPM, Conc_Phyto_water, Depth, MLDepth):
        """
        Screening factor for photolysis attenuation within the local water column.
            ScreeningFactor =
                (RadDistr / RadDistr0) *
                (1 - exp(-Extinct * Depth)) / (Extinct * Depth)
            with:
                Extinct = WaterExt
                        + ExtCoeffDOM * ConcDOM
                        + ExtCoeffSPM * concSPM
                        + ExtCoeffPHY * ConcPHYTO
        The factor combines:
        1) A radiance-distribution ratio
               RadDistr / RadDistr0
           where RadDistr0 is chosen according to depth regime:
               - mixed layer / epilimnion:     RadDistr0_ml
               - below mixed layer / hypolimnion: RadDistr0_bml
        2) A vertically averaged exponential attenuation term
               (1 - exp(-Extinct * Depth)) / (Extinct * Depth)
        """

        RadDistr = self._validate_scalar_param("RadDistr", RadDistr, gt=0.0)
        RadDistr0_ml = self._validate_scalar_param("RadDistr0_ml", RadDistr0_ml, gt=0.0)
        RadDistr0_bml = self._validate_scalar_param("RadDistr0_bml", RadDistr0_bml, gt=0.0)
        WaterExt = self._validate_scalar_param("WaterExt", WaterExt, ge=0.0)
        ExtCoeffDOM = self._validate_scalar_param("ExtCoeffDOM", ExtCoeffDOM, ge=0.0)
        ExtCoeffSPM = self._validate_scalar_param("ExtCoeffSPM", ExtCoeffSPM, ge=0.0)
        ExtCoeffPHY = self._validate_scalar_param("ExtCoeffPHY", ExtCoeffPHY, ge=0.0)
        C2PHYC = self._validate_scalar_param("C2PHYC", C2PHYC, gt=0.0)

        concDOC = self._validate_array_param("concDOC", concDOC, ge=0.0)
        concSPM = self._validate_array_param("concSPM", concSPM, ge=0.0)
        Conc_Phyto_water = self._validate_array_param("Conc_Phyto_water", Conc_Phyto_water, ge=0.0)
        # OpenDrift z is negative downward. Photolysis formulas need positive depth
        # measured downward from the sea surface.
        Depth = np.asarray(Depth, dtype=float)
        Depth = np.where(Depth < 0.0, -Depth, Depth)
        Depth = self._validate_array_param("Depth", Depth, ge=0.0)

        MLDepth = self._validate_array_param("MLDepth", MLDepth, ge=0.0)

        if not (concDOC.shape == concSPM.shape == Conc_Phyto_water.shape == Depth.shape == MLDepth.shape):
            raise ValueError("concDOC, concSPM, Conc_Phyto_water, Depth, and MLDepth must have the same shape")

        if RadDistr0_ml <= 0 or RadDistr0_bml <= 0:
            raise ValueError("RadDistr0_ml and RadDistr0_bml must be > 0")
        if RadDistr <= 0:
            raise ValueError("RadDistr must be > 0")

        N = len(Depth)
        ScreeningFactor = np.ones_like(Depth, dtype=float)
        chunk_size = int(1e5)
        eps = 1e-30
        rho_w = 1025.0  # kg/m3 or local density

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)

            Depth_chunk = np.abs(np.asarray(Depth[i:end], dtype=float))
            MLDepth_chunk = np.abs(np.asarray(MLDepth[i:end], dtype=float))
            concDOC_chunk = np.asarray(concDOC[i:end], dtype=float)
            concSPM_chunk = np.asarray(concSPM[i:end], dtype=float)
            Conc_Phyto_chunk = np.asarray(Conc_Phyto_water[i:end], dtype=float)
            ConcDOM = concDOC_chunk * 12e-3 * rho_w / 0.526 # g DOM / m3
            ConcPHYTO = (((Conc_Phyto_chunk * 1e-6) * 12.01) / C2PHYC) * 1000.0  # g biomass / m3

            Extinct = (WaterExt + (ExtCoeffDOM * ConcDOM) + (ExtCoeffSPM * concSPM_chunk)
                    + (ExtCoeffPHY * ConcPHYTO))
            Extinct = np.maximum(Extinct, 0.0)

            valid_depth = Depth_chunk > 0.0
            if np.any(valid_depth):
                RadDistr0 = np.where(Depth_chunk <= MLDepth_chunk,
                    RadDistr0_ml, RadDistr0_bml)

                x = Extinct[valid_depth] * Depth_chunk[valid_depth]

                safe_factor = np.ones_like(x)
                nz = np.abs(x) > eps
                safe_factor[nz] = (1.0 - np.exp(-x[nz])) / x[nz]

                sf = (RadDistr / RadDistr0[valid_depth]) * safe_factor

                idx = np.where(valid_depth)[0] + i
                ScreeningFactor[idx] = sf

                # Clip tiny numerical overshoots (or larger values) to 1 and log them
                too_high = ScreeningFactor[i:end] > 1.0
                if np.any(too_high):
                    if logger.isEnabledFor(logging.INFO):
                        n_high = int(np.sum(too_high))
                        max_high = float(np.nanmax(ScreeningFactor[i:end][too_high]))
                        logger.info(
                            "Clipping %s ScreeningFactor values > 1 to 1.0 "
                            "(max before clip: %s)",
                            n_high,
                            max_high,
                        )
                if np.any(ScreeningFactor[i:end] < 0):
                    raise ValueError("ScreeningFactor is negative")

                ScreeningFactor[i:end] = np.clip(ScreeningFactor[i:end], 0.0, 1.0)

        return ScreeningFactor

    def _solar_to_ly_day(self, solar):
        """
        Convert solar forcing to Langley per day [Ly/day].
        1) 'W_m2'
           Converts from W/m2 using: Solar_ly_day = Solar_W_m2 / 0.4843
               1 Ly/day = 0.4843 W/m2
        2) 'Ly_day'
           Input is already in Ly/day and is returned unchanged.
        """
        solar = np.asarray(solar, dtype=float)

        unit = self.get_config('chemical:transformations:solar_input_unit')
        if unit == 'W_m2':
            return solar / 0.4843
        elif unit == 'Ly_day':
            return solar
        else:
            raise ValueError(f"Unknown solar_input_unit: {unit!r}")

    def _photolysis_alpha(self, idx=None):
        """
        Dynamic bulk light-extinction coefficient for photolysis. [1/m]

        The total extinction coefficient is computed as the sum of background water
        attenuation and constituent-specific contributions:
            Alpha = WaterExt
                  + ExtCoeffDOM * ConcDOM
                  + ExtCoeffSPM * concSPM
                  + ExtCoeffPHY * ConcPHYTO
        where:
            WaterExt    = background attenuation by pure water [1/m]
            ConcDOM     = dissolved organic matter concentration [g/m3]
            concSPM     = suspended particulate matter concentration [g/m3]
            ConcPHYTO   = phytoplankton biomass concentration [g/m3]
        """
        WaterExt = self.get_config('chemical:transformations:WaterExt')
        ExtCoeffDOM = self.get_config('chemical:transformations:ExtCoeffDOM')
        ExtCoeffSPM = self.get_config('chemical:transformations:ExtCoeffSPM')
        ExtCoeffPHY = self.get_config('chemical:transformations:ExtCoeffPHY')
        C2PHYC = self.get_config('chemical:transformations:C2PHYC')

        # DOC input is mmol C / kg
        concDOC = self._doc_mmolkg(idx)
        rho_w = 1025.0  # kg/m3 or local density
        ConcDOM = concDOC * 12e-3 * rho_w / 0.526 # g DOM / m3
        # SPM as g / m3
        concSPM = self._spm_g_m3(idx)
        # Phytoplankton input is mmol C / m3
        Conc_Phyto_water = self._env_array(
            'mole_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water',
            0.0, idx=idx)
        ConcPHYTO = (((Conc_Phyto_water * 1e-6) * 12.01) / C2PHYC) * 1000.0  # g biomass / m3

        Alpha = (WaterExt + (ExtCoeffDOM * ConcDOM) + (ExtCoeffSPM * concSPM) + (ExtCoeffPHY * ConcPHYTO))

        return np.maximum(np.asarray(Alpha, dtype=float), 0.0)

    def calc_LightFactor(self, AveSolar, Solar_ly_day, Depth, MLDepth, Alpha):
        """
        Light factor for photolysis attenuation relative to a reference solar forcing.
            LightFactor = Solar0 / AveSolar
        where:
            AveSolar = reference average solar intensity [Ly/day]
            Solar0   = effective solar forcing reaching the top of the local segment

        The segment-top forcing is defined as:
        1) Epilimnion / unstratified water
               Solar0 = Solar_ly_day
        2) Hypolimnion (below mixed layer)
               Solar0 = Solar_ly_day * exp(-Alpha * MLDepth)
        where:
            Alpha   = bulk extinction coefficient [1/m]
            MLDepth = mixed-layer depth, used here as epilimnion thickness [m]
        """
        AveSolar = self._validate_scalar_param("AveSolar", AveSolar, gt=0.0)
        Solar_ly_day = self._validate_array_param("Solar_ly_day", Solar_ly_day, ge=0.0)
        Depth = np.asarray(Depth, dtype=float)
        Depth = np.where(Depth < 0.0, -Depth, Depth)
        Depth = self._validate_array_param("Depth", Depth, ge=0.0)
        MLDepth = self._validate_array_param("MLDepth", MLDepth, ge=0.0)

        Alpha = np.asarray(Alpha, dtype=float)
        if Alpha.ndim == 0:
            Alpha = np.full_like(Depth, float(Alpha))
        else:
            Alpha = self._validate_array_param("Alpha", Alpha, ge=0.0)

        if Solar_ly_day.shape != Depth.shape:
            raise ValueError("Solar_ly_day and Depth must have the same shape")
        if MLDepth.shape != Depth.shape:
            raise ValueError("MLDepth and Depth must have the same shape")
        if Alpha.shape != Depth.shape:
            raise ValueError("Alpha must be scalar or have the same shape as Depth")

        MLDepth = np.maximum(MLDepth, 0.0)
        Alpha = np.maximum(Alpha, 0.0)
        # epilimnion/unstratified -> Solar0 = Solar
        Solar0 = Solar_ly_day.copy()
        # particles below the mixed layer are treated as "hypolimnion"
        below_ml = Depth > MLDepth
        # hypolimnion -> Solar0 = Solar * exp(-Alpha * MaxZMix)
        if np.any(below_ml):
            Solar0[below_ml] = Solar_ly_day[below_ml] * np.exp(
                -Alpha[below_ml] * MLDepth[below_ml])

        return Solar0 / AveSolar

    def assert_degradation_balance(self, degraded_now, W, S, check_single_mech=False):
        """Common consistency checks for both degradation modes (cumulative-only).
        Checks:
          1) computed degraded_now equals computed (water+sediment) split for this step
          2) stored cumulative mass_degraded equals stored cumulative (water+sediment)
          3) if check_single_mech and Save_single_degr_mass: cumulative mechanisms sum equals cumulative degraded
        """
        # 1) computed step totals
        degraded_step_sum = float(degraded_now.sum())
        degraded_step_ws_sum = float(degraded_now[W].sum() + degraded_now[S].sum())
        assert np.isclose(degraded_step_sum, degraded_step_ws_sum, rtol=1e-5, atol=1e-8), \
            "Computed degraded_now is inconsistent with computed water+sediment split"
        # 2) stored cumulative totals
        stored_tot_sum = float(self.elements.mass_degraded.sum())
        stored_ws_sum = float(self.elements.mass_degraded_water.sum() + self.elements.mass_degraded_sediment.sum())
        assert np.isclose(stored_tot_sum, stored_ws_sum, rtol=1e-5, atol=1e-8), \
            "Inconsistent cumulative mass_degraded vs (water+sediment)"
        # 3) single-mechanism checks (cumulative)
        if not (check_single_mech and self.get_config('chemical:transformations:Save_single_degr_mass') is True):
            return

        mech_sum = 0.0

        # Hydrolysis
        if self.get_config('chemical:transformations:Hydrolysis'):
            assert hasattr(self.elements, "mass_hydrolyzed")
            assert hasattr(self.elements, "mass_hydrolyzed_water")
            assert hasattr(self.elements, "mass_hydrolyzed_sediment")

            hyd = float(self.elements.mass_hydrolyzed.sum())
            hyd_ws = float(self.elements.mass_hydrolyzed_water.sum() + self.elements.mass_hydrolyzed_sediment.sum())
            assert np.isclose(hyd, hyd_ws, rtol=1e-5, atol=1e-8), \
                "Inconsistent sum of hydrolyzed mass: total vs (water+sediment)"
            mech_sum += hyd
        # Biodegradation
        if self.get_config('chemical:transformations:Biodegradation'):
            assert hasattr(self.elements, "mass_biodegraded")
            assert hasattr(self.elements, "mass_biodegraded_water")
            assert hasattr(self.elements, "mass_biodegraded_sediment")

            bio = float(self.elements.mass_biodegraded.sum())
            bio_ws = float(self.elements.mass_biodegraded_water.sum() + self.elements.mass_biodegraded_sediment.sum())
            assert np.isclose(bio, bio_ws, rtol=1e-5, atol=1e-8), \
                "Inconsistent sum of biodegraded mass: total vs (water+sediment)"
            mech_sum += bio
        # Photodegradation (water-only)
        if self.get_config('chemical:transformations:Photodegradation'):
            if hasattr(self.elements, "mass_photodegraded"):
                mech_sum += float(self.elements.mass_photodegraded.sum())

        # Mechanisms reconstruct total degraded (cumulative)
        assert np.isclose(stored_tot_sum, mech_sum, rtol=1e-5, atol=1e-8), \
            "Inconsistent sum: cumulative degraded vs (enabled mechanisms)"

    ###########################################################################
    # Main transformation-loss functions
    ###########################################################################

    def _degradation_rate_fraction(self, mechanism_per_hour, total_per_second):
        """Mechanism share of a total rate, with zero share for zero total rate.

        Convert h^-1 to s^-1 before division. A positive denominator must not
        be raised to a numerical floor: that loses mass from slow mechanisms.
        """
        rate, total = np.broadcast_arrays(
            np.asarray(mechanism_per_hour, dtype=float) / 3600.0,
            np.asarray(total_per_second, dtype=float))
        if np.any(~np.isfinite(rate) | ~np.isfinite(total) | (rate < 0) | (total < 0)):
            raise ValueError('Degradation rates must be finite and non-negative.')
        share = np.divide(rate, total, out=np.zeros_like(total), where=total > 0)
        return np.clip(share, 0.0, 1.0)

    def degradation(self):
        """
        Apply chemical mass loss by degradation.
        Two degradation modes are supported:

        1) OverallRateConstants
           Uses a single effective first-order decay rate in water and sediment:
               k_W_tot = ln(2) / t12_W_tot
               k_S_tot = ln(2) / t12_S_tot
           corrected for temperature by Arrhenius scaling:
               k_fin = k_tot * tempcorr(...)
           degraded mass over dt:
               degraded = mass * (1 - exp(-k_fin * dt))

        Water degradation applies to: LMM, humic colloid
        Sediment degradation applies to: srev, ssrev, sburied, sirrev
        An optional burial slowdown factor can reduce degradation in buried sediment.

        2) SingleRateConstants
           Computes separate contributions from:
               - biodegradation
               - photodegradation
               - hydrolysis
           Water-column rate:
               k_W_fin = (k_W_bio + k_W_photo + k_W_hydro) / 3600
           Sediment rate:
               k_S_fin = (k_S_bio + k_S_hydro) / 3600
           degraded mass:
               degraded = mass * (1 - exp(-k_fin * dt))
           Optional bookkeeping stores mechanism-specific degraded masses.

        Elements with very small remaining mass are deactivated when:
            mass < (mass + mass_degraded + mass_volatilized) / 500
        If mass_checks is enabled, internal balance checks are applied.
        """
        # Optional sediment-O2 diagnostics are dense per-element arrays. Reset
        # them every degradation call so particles that leave the sediment do
        # not retain values from the previous timestep.
        self._reset_sediment_oxygen_diagnostics()

        if self.get_config('chemical:transformations:degradation') is True:

            if self.get_config('chemical:transformations:degradation_mode') == 'OverallRateConstants':

                logger.debug('Calculating overall degradation using overall rate constants')
                degraded_now = np.zeros(self.num_elements_active())

                # Degradation in the water
                k_W_tot = -np.log(0.5) / (self.get_config('chemical:transformations:t12_W_tot') * (60 * 60))  # (1/s)
                Tref_kWt = self.get_config('chemical:transformations:Tref_kWt')
                DH_kWt = self.get_config('chemical:transformations:DeltaH_kWt')

                W = np.zeros(self.num_elements_active(), dtype=bool)
                if hasattr(self, 'num_lmm'):
                    W |= (self.elements.specie == self.num_lmm)
                if hasattr(self, 'num_humcol'):
                    W |= (self.elements.specie == self.num_humcol)
                idx_W = np.flatnonzero(W)
                W_deg = idx_W.size > 0

                if W_deg:
                    TW = self._env_array('sea_water_temperature', 10.0, idx=idx_W)
                    # if np.any(TW==0):
                    #     TW[TW==0]=np.median(TW)
                    #     logger.debug("Temperature in degradation was 0, set to median value")

                    k_W_fin = k_W_tot * self.tempcorr("Arrhenius", DH_kWt, TW, Tref_kWt)
                    k_W_fin = np.maximum(k_W_fin, 0.0)

                    degraded_now[W] = np.minimum(self.elements.mass[W],
                        self.elements.mass[W] * (-np.expm1(-k_W_fin * self.time_step.total_seconds())))
                    # avoid degrading more mass than present

                # Degradation in the sediments
                k_S_tot = -np.log(0.5) / (
                    self.get_config('chemical:transformations:t12_S_tot') * (60 * 60))  # (1/s)
                Tref_kSt = self.get_config('chemical:transformations:Tref_kSt')
                DH_kSt = self.get_config('chemical:transformations:DeltaH_kSt')

                # Sediment degradation applies to the active sediment layer (srev/ssrev)
                # and (if enabled) the buried sediment compartment.
                S = np.zeros(self.num_elements_active(), dtype=bool)
                if hasattr(self, 'num_srev'):
                    S |= (self.elements.specie == self.num_srev)
                if hasattr(self, 'num_ssrev'):
                    S |= (self.elements.specie == self.num_ssrev)
                if hasattr(self, 'num_sburied'):
                    S |= (self.elements.specie == self.num_sburied)
                if hasattr(self, 'num_sirrev'):
                    S |= (self.elements.specie == self.num_sirrev)

                idx_S = np.flatnonzero(S)
                S_deg = idx_S.size > 0

                if S_deg:
                    TS = self._env_array('sea_water_temperature', 10.0, idx=idx_S)

                    k_S_fin = k_S_tot * self.tempcorr("Arrhenius", DH_kSt, TS, Tref_kSt)
                    k_S_fin = np.maximum(k_S_fin, 0.0)

                    # Apply slower degradation to buried sediments due to anoxic conditions
                    ssrev_slow_deg = self.get_config('chemical:transformations:ssrev_slow_deg_factor')
                    if ssrev_slow_deg < 1:
                        if ssrev_slow_deg < 0:
                            ssrev_slow_deg = 0.0

                        # Apply the slowdown to the buried sediment compartment
                        if hasattr(self, 'num_sburied'):
                            S_is_buried = (self.elements.specie[S] == self.num_sburied)
                            k_S_fin[S_is_buried] *= ssrev_slow_deg
                        else:
                            # Backward compatibility (older setups used ssrev as buried)
                            # S_is_buried = (self.elements.specie[S] == self.num_ssrev)
                            # k_S_fin[S_is_buried] *= ssrev_slow_deg
                            pass

                    degraded_now[S] = np.minimum(self.elements.mass[S],
                        self.elements.mass[S] * (-np.expm1(-k_S_fin * self.time_step.total_seconds()))
                        )  # avoid degrading more mass than present

                if W_deg or S_deg:
                    self.elements.mass_degraded_water[W] += degraded_now[W]
                    self.elements.mass_degraded_sediment[S] += degraded_now[S]

                    self.elements.mass_degraded += degraded_now
                    self.elements.mass -= degraded_now
                    self.elements.mass = np.maximum(self.elements.mass, 0.0)

                    self.deactivate_elements(
                        self.elements.mass <
                        (self.elements.mass + self.elements.mass_degraded + self.elements.mass_volatilized) / 500,
                        reason='removed')

                if self.get_config("chemical:transformations:mass_checks"):
                    # Consistency checks (overall mode)
                    self.assert_degradation_balance(degraded_now, W, S, check_single_mech=False)

            elif self.get_config('chemical:transformations:degradation_mode') == 'SingleRateConstants':
                # Calculations here are for single process degradation including
                # biodegradation, photodegradation, and hydrolysys
                logger.debug('Calculating single degradation rates in water')

                Photo_degr = self.get_config('chemical:transformations:Photodegradation')
                Bio_degr = self.get_config('chemical:transformations:Biodegradation')
                Hydro_degr = self.get_config('chemical:transformations:Hydrolysis')

                degraded_now = np.zeros(self.num_elements_active())

                # Only "dissolved" and "DOC" elements will degrade in the water column
                W = np.zeros(self.num_elements_active(), dtype=bool)
                if hasattr(self, 'num_lmm'):
                    W |= (self.elements.specie == self.num_lmm)
                if hasattr(self, 'num_humcol'):
                    W |= (self.elements.specie == self.num_humcol)
                idx_W = np.flatnonzero(W)

                # All elements in the active sediment layer will degrade (srev/ssrev),
                # and (if enabled) the buried sediment compartment as well.
                S = np.zeros(self.num_elements_active(), dtype=bool)
                if hasattr(self, 'num_srev'):
                    S |= (self.elements.specie == self.num_srev)
                if hasattr(self, 'num_ssrev'):
                    S |= (self.elements.specie == self.num_ssrev)
                if hasattr(self, 'num_sburied'):
                    S |= (self.elements.specie == self.num_sburied)
                if hasattr(self, 'num_sirrev'):
                    S |= (self.elements.specie == self.num_sirrev)
                idx_S = np.flatnonzero(S)

                W_deg = idx_W.size > 0
                S_deg = idx_S.size > 0

                k_Photo = self.get_config('chemical:transformations:k_Photo')
                k_DecayMax_water = self.get_config('chemical:transformations:k_DecayMax_water')
                k_Anaerobic_water = self.get_config('chemical:transformations:k_Anaerobic_water')

                # Biodegradation is active when at least one endpoint rate is
                # positive. Do not require the aerobic endpoint to be > 0:
                # an anaerobic-only compound is a valid case.
                bio_rate_enabled = (
                    Bio_degr is True and (
                        k_DecayMax_water > 0.0
                        or k_Anaerobic_water > 0.0))

                if k_Photo == 0:
                    logger.debug(
                        "k_Photo is set to 0 1/h, therefore no photodegradation occurs"
                    )

                if Bio_degr is True:
                    # This helper uses logger.debug only. No warning/error is
                    # raised when anaerobic degradation exceeds aerobic degradation.
                    self._debug_biodegradation_rate_regime(
                        k_DecayMax_water, k_Anaerobic_water,
                    )

                if k_Anaerobic_water == 0:
                    logger.debug(
                        "k_Anaerobic_water is set to 0 1/h, therefore no biodegradation occurs at O2 = 0"
                    )
                if W_deg or S_deg:
                    Tref_kWt = self.get_config('chemical:transformations:Tref_kWt')
                    DH_kWt = self.get_config('chemical:transformations:DeltaH_kWt')
                    Tref_kSt = self.get_config('chemical:transformations:Tref_kSt')
                    DH_kSt = self.get_config('chemical:transformations:DeltaH_kSt')

                    if bio_rate_enabled:
                        HalfSatO_w = self.get_config('chemical:transformations:HalfSatO_w')
                        T_Max_bio = self.get_config('chemical:transformations:T_Max_bio')
                        T_Opt_bio = self.get_config('chemical:transformations:T_Opt_bio')
                        T_Adp_bio = self.get_config('chemical:transformations:T_Adp_bio')
                        Max_Accl_bio = self.get_config('chemical:transformations:Max_Accl_bio')
                        Dec_Accl_bio = self.get_config('chemical:transformations:Dec_Accl_bio')
                        Q10_bio = self.get_config('chemical:transformations:Q10_bio')
                        pH_min_bio = self.get_config('chemical:transformations:pH_min_bio')
                        pH_max_bio = self.get_config('chemical:transformations:pH_max_bio')
                        Ox_water = (
                            self._env_array(
                                'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water',
                                7.25,
                                idx=idx_W,
                            )
                            if W_deg else np.empty(0, dtype=float)
                        ) * 31.9988e-3  # mmol/m3 -> g/m3 O2

                    if Hydro_degr is True:
                        k_Acid = self.get_config('chemical:transformations:k_Acid')
                        k_Base = self.get_config('chemical:transformations:k_Base')
                        k_Hydr_Uncat = self.get_config('chemical:transformations:k_Hydr_Uncat')
                        if (k_Acid <= 0 and k_Base <= 0 and k_Hydr_Uncat == 0):
                            logger.debug("k_Acid, k_Base, and k_Hydr_Uncat are set to 0 1/h, therefore no hydrolysis occurs")

                # Water-column degradation
                if W_deg:
                    TW = self._env_array('sea_water_temperature', 10.0, idx=idx_W)
                    # if np.any(TW==0):
                    #     TW[TW==0]=np.median(TW)
                    #     logger.debug("Temperature in degradation was 0, set to median value")

                    if Photo_degr is True and k_Photo > 0:
                        RadDistr = self.get_config('chemical:transformations:RadDistr')
                        RadDistr0_ml = self.get_config('chemical:transformations:RadDistr0_ml')
                        RadDistr0_bml = self.get_config('chemical:transformations:RadDistr0_bml')
                        WaterExt = self.get_config('chemical:transformations:WaterExt')
                        ExtCoeffDOM = self.get_config('chemical:transformations:ExtCoeffDOM')
                        ExtCoeffSPM = self.get_config('chemical:transformations:ExtCoeffSPM')
                        ExtCoeffPHY = self.get_config('chemical:transformations:ExtCoeffPHY')
                        C2PHYC = self.get_config('chemical:transformations:C2PHYC')
                        AveSolar = self.get_config('chemical:transformations:AveSolar')
                        Solar_radiation = self._env_array('solar_irradiance', 241.0, idx=idx_W)
                        Solar_ly_day = self._solar_to_ly_day(Solar_radiation)
                        Alpha = self._photolysis_alpha(idx_W)
                        # Concentration of phytoplankton in the water column (mmol_C/m3)
                        Conc_Phyto_water = self._env_array('mole_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water',
                            0.0, idx=idx_W)
                        # Concentration of SPM (g/m3)
                        concSPM = self._spm_g_m3(idx_W)
                        # Mixed Layer depth (m)
                        MLDepth = self._env_array('ocean_mixed_layer_thickness', 50.0, idx=idx_W)
                        # Depth of element below sea surface (m).
                        # OpenDrift z is negative downward, so convert z -> positive depth.

                        Depth = np.maximum(-self._z_array(idx_W), 0.0)  # positive depth from surface
                        # Concentration of DOC (mmol[C]/Kg)
                        concDOC = self._doc_mmolkg(idx_W)

                    if bio_rate_enabled or Hydro_degr is True:
                        # pH water
                        pH_water = self._env_array('sea_water_ph_reported_on_total_scale',
                            8.1, idx=idx_W)

                        if np.any(pH_water == 0):
                            pH_water[pH_water == 0] = np.median(pH_water)
                            logger.debug("pH_water in degradation was 0, set to median value")

                    # Calculate correction factors for degradation rates
                    if bio_rate_enabled:
                        # Direct aerobic/anaerobic interpolation [1/h].
                        # This formulation permits k_Anaerobic_water to exceed k_DecayMax_water and remains valid when the aerobic
                        # endpoint is exactly zero.
                        k_W_bio = self.calc_DO_biodegradation_rate(
                            HalfSatO_w=HalfSatO_w, k_Anaerobic_water=k_Anaerobic_water,
                            k_Aerobic=k_DecayMax_water, Ox=Ox_water,
                        )

                        k_W_bio = k_W_bio * self.calc_pHCorr(
                            pH_min_bio, pH_max_bio, pH_water,
                            )
                        k_W_bio = k_W_bio * self.calc_TCorr(
                            T_Max_bio, T_Opt_bio, T_Adp_bio, Max_Accl_bio, Dec_Accl_bio, Q10_bio, TW,
                            )

                        k_W_bio = np.maximum(k_W_bio, 0.0)
                    else:
                        k_W_bio = np.zeros_like(TW)

                    if Photo_degr is True and k_Photo > 0:
                        k_W_photo = k_Photo * self.calc_LightFactor(AveSolar=AveSolar,
                            Solar_ly_day=Solar_ly_day, Depth=Depth, MLDepth=MLDepth, Alpha=Alpha)

                        k_W_photo = k_W_photo * self.calc_ScreeningFactor(
                            RadDistr, RadDistr0_ml, RadDistr0_bml, WaterExt,
                            ExtCoeffDOM, ExtCoeffSPM, ExtCoeffPHY, C2PHYC,
                            concDOC, concSPM, Conc_Phyto_water, Depth, MLDepth)

                        k_W_photo = k_W_photo * self.tempcorr("Arrhenius", DH_kWt, TW, Tref_kWt)
                        k_W_photo = np.maximum(k_W_photo, 0.0)
                    else:
                        k_W_photo = np.zeros_like(TW)

                    if Hydro_degr is True:
                        k_W_hydro = self.calc_k_hydro_water(k_Acid, k_Base, k_Hydr_Uncat, pH_water)
                        k_W_hydro = k_W_hydro * self.tempcorr("Arrhenius", DH_kWt, TW, Tref_kWt)
                        k_W_hydro = np.maximum(k_W_hydro, 0.0)
                    else:
                        k_W_hydro = np.zeros_like(TW)

                    k_W_fin = (k_W_bio + k_W_hydro + k_W_photo) / (60 * 60)  # 1/h -> 1/s
                    k_W_fin_sum = np.sum(k_W_fin)

                    if k_W_fin_sum > 0:
                        degraded_now[W] = np.minimum(self.elements.mass[W],
                            self.elements.mass[W] * (-np.expm1(-k_W_fin * self.time_step.total_seconds())))
                else:
                    k_W_bio = 0
                    k_W_hydro = 0
                    k_W_photo = 0
                    k_W_fin = 0
                    k_W_fin_sum = 0

                # Sediment degradation
                if S_deg:
                    TS = self._env_array('sea_water_temperature', 10.0, idx=idx_S)
                    # if np.any(TS==0):
                    #     TS[TS==0]=np.median(TS)
                    #     logger.debug("Temperature in degradation was 0, set to median value")

                    if bio_rate_enabled or Hydro_degr is True:
                        # pH sediments
                        pH_sed = self._env_array('pH_sediment', 6.9, idx=idx_S)
                        if np.any(pH_sed == 0):
                            pH_sed[pH_sed == 0] = np.median(pH_sed)
                            logger.debug("pH_sed in degradation was 0, set to median value")

                    if bio_rate_enabled:
                        # Only the aerobic endpoint is increased in sediment.
                        # The anaerobic endpoint remains exactly the configured k_Anaerobic_water value.
                        k_DecayMax_sediment = 4.0 * k_DecayMax_water

                        # Build one sediment-O2 array aligned with idx_S.
                        # Active sediment uses the selected reduced-order oxygen
                        # model. Buried sediment is forced to exactly O2 = 0.
                        S_is_buried = np.zeros(idx_S.size, dtype=bool)
                        if hasattr(self, 'num_sburied'):
                            S_is_buried = (
                                self.elements.specie[idx_S] == self.num_sburied
                            )
                        S_is_active = ~S_is_buried

                        Ox_sed_mmol_m3 = np.zeros(idx_S.size, dtype=float)
                        save_o2_diag = self._save_sediment_oxygen_diagnostics()

                        if np.any(S_is_active):
                            idx_S_active = idx_S[S_is_active]
                            if save_o2_diag:
                                active_oxygen, active_o2_diag = (
                                    self.calculate_active_sediment_oxygen(
                                        idx_S_active,
                                        return_diagnostics=True,
                                    )
                                )
                                Ox_sed_mmol_m3[S_is_active] = active_oxygen
                                self._store_sediment_oxygen_diagnostics(
                                    idx_S_active, active_o2_diag
                                )
                            else:
                                Ox_sed_mmol_m3[S_is_active] = (
                                    self.calculate_active_sediment_oxygen(
                                        idx_S_active
                                    )
                                )

                        if save_o2_diag:
                            # This is the O2 value actually passed to sediment
                            # biodegradation: active-layer mean for active sediment,
                            # exactly zero for Sediment_buried. Non-sediment
                            # particles remain NaN after the timestep reset.
                            self._store_sediment_oxygen_diagnostics(
                                idx_S,
                                {'sed_o2_used': Ox_sed_mmol_m3},
                            )

                        # The biodegradation oxygen interpolation uses O2 in g/m3.
                        # Molecular weight O2 = 31.9988 g/mol:
                        # mmol/m3 * 31.9988e-3 = g/m3.
                        Ox_sed = Ox_sed_mmol_m3 * 31.9988e-3

                        # Direct aerobic/anaerobic interpolation [1/h].
                        # At O2 = 0 this returns exactly k_Anaerobic_water,
                        # irrespective of the fourfold aerobic sediment factor.
                        k_S_bio = self.calc_DO_biodegradation_rate(
                            HalfSatO_w=HalfSatO_w, k_Anaerobic_water=k_Anaerobic_water,
                            k_Aerobic=k_DecayMax_sediment, Ox=Ox_sed,
                        )

                        k_S_bio = k_S_bio * self.calc_pHCorr(
                            pH_min_bio, pH_max_bio, pH_sed,
                        )
                        k_S_bio = k_S_bio * self.tempcorr(
                            "Arrhenius", DH_kSt, TS, Tref_kSt,
                        )
                        k_S_bio = np.maximum(k_S_bio, 0.0)

                    else:
                        k_S_bio = np.zeros_like(TS)

                    if Hydro_degr is True:
                        k_S_hydro = self.calc_k_hydro_sed(k_Acid, k_Base, k_Hydr_Uncat, pH_sed)
                        k_S_hydro = k_S_hydro * self.tempcorr("Arrhenius", DH_kSt, TS, Tref_kSt)
                    else:
                        k_S_hydro = np.zeros_like(TS)

                    k_S_fin = (k_S_bio + k_S_hydro) / (60 * 60)   # from 1/h to 1/s
                    k_S_fin_sum = k_S_fin.sum()

                    if k_S_fin_sum > 0:
                        degraded_now[S] = np.minimum(self.elements.mass[S],
                            self.elements.mass[S] * (-np.expm1(-k_S_fin * self.time_step.total_seconds())))
                else:
                    k_S_bio = 0
                    k_S_hydro = 0
                    k_S_fin = 0
                    k_S_fin_sum = 0

                # Save single-mechanism masses
                if bool(self.get_config('chemical:transformations:Save_single_degr_mass')):
                    k_W_photo_fraction = 0
                    k_W_bio_fraction = 0
                    k_S_bio_fraction = 0
                    k_W_hydro_fraction = 0
                    k_S_hydro_fraction = 0

                    if Photo_degr is True and k_Photo > 0:
                        missing = [name for name in ['mass_photodegraded'] if not hasattr(self.elements, name)]
                        if missing:
                            raise RuntimeError(
                                'Save_single_degr_mass and Photo_degr are True, but these element variables are missing: '
                                + ', '.join(missing)
                            )
                        if np.sum(k_W_photo) > 0:
                            photo_degraded_now = np.zeros(self.num_elements_active())
                            k_W_photo_fraction = self._degradation_rate_fraction(k_W_photo, k_W_fin)
                            photo_degraded_now[W] = degraded_now[W] * k_W_photo_fraction

                            if W_deg:
                                self.elements.mass_photodegraded[W] += photo_degraded_now[W]

                    if bio_rate_enabled:
                        missing = [name for name in ['mass_biodegraded', 'mass_biodegraded_water', 'mass_biodegraded_sediment',] if not hasattr(self.elements, name)]
                        if missing:
                            raise RuntimeError(
                                'Save_single_degr_mass and Bio_degr are True, but these element variables are missing: '
                                + ', '.join(missing)
                            )
                        if np.sum(k_W_bio) > 0 or np.sum(k_S_bio) > 0:
                            bio_degraded_now = np.zeros(self.num_elements_active())
                            if np.sum(k_W_bio) > 0:
                                k_W_bio_fraction = self._degradation_rate_fraction(k_W_bio, k_W_fin)

                                bio_degraded_now[W] = degraded_now[W] * k_W_bio_fraction
                            if np.sum(k_S_bio) > 0:
                                k_S_bio_fraction = self._degradation_rate_fraction(k_S_bio, k_S_fin)

                                bio_degraded_now[S] = degraded_now[S] * k_S_bio_fraction

                            if W_deg:
                                self.elements.mass_biodegraded[W] += bio_degraded_now[W]
                                self.elements.mass_biodegraded_water[W] += bio_degraded_now[W]
                            if S_deg:
                                self.elements.mass_biodegraded[S] += bio_degraded_now[S]
                                self.elements.mass_biodegraded_sediment[S] += bio_degraded_now[S]

                    if Hydro_degr is True:
                        missing = [name for name in ['mass_hydrolyzed', 'mass_hydrolyzed_water', 'mass_hydrolyzed_sediment',] if not hasattr(self.elements, name)]
                        if missing:
                            raise RuntimeError(
                                'Save_single_degr_mass and Hydro_degr are True, but these element variables are missing: '
                                + ', '.join(missing)
                            )
                        if np.sum(k_W_hydro) > 0 or np.sum(k_S_hydro) > 0:
                            hydro_degraded_now = np.zeros(self.num_elements_active())
                            if np.sum(k_W_hydro) > 0:
                                k_W_hydro_fraction = self._degradation_rate_fraction(k_W_hydro, k_W_fin)

                                hydro_degraded_now[W] = degraded_now[W] * k_W_hydro_fraction
                            if np.sum(k_S_hydro) > 0:
                                k_S_hydro_fraction = self._degradation_rate_fraction(k_S_hydro, k_S_fin)

                                hydro_degraded_now[S] = degraded_now[S] * k_S_hydro_fraction

                            if W_deg:
                                self.elements.mass_hydrolyzed[W] += hydro_degraded_now[W]
                                self.elements.mass_hydrolyzed_water[W] += hydro_degraded_now[W]
                            if S_deg:
                                self.elements.mass_hydrolyzed[S] += hydro_degraded_now[S]
                                self.elements.mass_hydrolyzed_sediment[S] += hydro_degraded_now[S]

                    total_W_fraction = (k_W_photo_fraction + k_W_bio_fraction + k_W_hydro_fraction)
                    assert np.all(total_W_fraction <= 1.0 + 1e-6), "Degradation fractions in water exceed 100%"
                    total_S_fraction = (k_S_bio_fraction + k_S_hydro_fraction)
                    assert np.all(total_S_fraction <= 1.0 + 1e-6), "Degradation fractions in sediments exceed 100%"

                # Apply total degraded mass
                if (k_S_fin_sum > 0) or (k_W_fin_sum > 0):
                    self.elements.mass_degraded += degraded_now
                    if W_deg:
                        self.elements.mass_degraded_water[W] += degraded_now[W]
                    if S_deg:
                        self.elements.mass_degraded_sediment[S] += degraded_now[S]

                    # Update mass and clamp to 0
                    self.elements.mass -= degraded_now
                    self.elements.mass = np.maximum(self.elements.mass, 0.0)

                    self.deactivate_elements(
                        self.elements.mass <
                        (self.elements.mass + self.elements.mass_degraded + self.elements.mass_volatilized) / 500,
                        reason='removed')

                if self.get_config("chemical:transformations:mass_checks"):
                    # Consistency checks (single-mechanism mode)
                    self.assert_degradation_balance(degraded_now, W, S, check_single_mech=True)

    def volatilization(self):
        """
            Volatilization is currently applied only to dissolved LMM elements located
            within a positive-thickness mixed layer.
            1) Restrict to:
                   specie == LMM
                   depth <= mixed_layer_depth
            2) Compute Henry constant
               a) If Henry is provided directly:
                      Henry(T) from enthalpy correction
               b) Otherwise estimate from vapor pressure and solubility:
                      Henry ~ Vpress(T) / Solub(T) * MolWt / 101325
            3) Compute neutral / volatilizable fraction
               depending on dissociation mode:
                   nondiss   -> 1
                   acid      -> fraction of neutral HA
                   base      -> fraction of neutral B
                   amphoteric  -> neutral fraction only
            4) Compute water-side transfer coefficient:
                   MTCw = (9e-4 + 7.2e-6 * wind^3) * (MolWtCO2 / MolWt)^0.25 * Undiss_n
            5) Compute air-side transfer coefficient:
                   MTCa = MTCaH2O * (MolWtH2O / MolWt)^(1/3)
            6) Convert Henry constant to dimensionless air-water partitioning:
                   HenryLaw = Henry * (1 + 0.01143*S) / (R * T_K)
            7) Combine air- and water-side resistances:
                   MTCvol = 1 / (1/MTCw + 1/(MTCa * HenryLaw))    [cm/s]
            8) Convert to first-order volatilization rate:
                   K_vol = 0.01 * MTCvol / Thick                  [1/s]
            9) Volatilized mass over dt:
                   volatilized = mass * (1 - exp(-K_vol * dt))
            """
        if self.get_config('chemical:transformations:volatilization') is not True:
            return

        # Volatilization is currently defined only for dissolved LMM
        if not hasattr(self, 'num_lmm'):
            return

        idx_lmm = np.flatnonzero(self.elements.specie == self.num_lmm)
        if idx_lmm.size == 0:
            return

        mixedlayerdepth_all = self._env_array('ocean_mixed_layer_thickness', 50.0, idx=idx_lmm)
        mixedlayerdepth_all = np.asarray(mixedlayerdepth_all, dtype=float)

        # Only dissolved elements inside a positive-thickness mixed layer can volatilize
        z_lmm = np.asarray(self.elements.z[idx_lmm], dtype=float)
        in_mixed_layer = ((-z_lmm <= mixedlayerdepth_all) & (mixedlayerdepth_all > 0.0))
        if not np.any(in_mixed_layer):
            return

        idx = idx_lmm[in_mixed_layer]

        # Local environmental values only for the selected elements
        Thick = mixedlayerdepth_all[in_mixed_layer]  # m
        T = self._env_array('sea_water_temperature', 10.0, idx=idx)     # degC
        S = self._env_array('sea_water_salinity', 34.0, idx=idx)
        pH_water = self._env_array('sea_water_ph_reported_on_total_scale', 8.1, idx=idx)

        x_wind = self._env_array('x_wind', 0.0, idx=idx)
        y_wind = self._env_array('y_wind', 0.0, idx=idx)
        wind = np.hypot(x_wind, y_wind)

        MolWtCO2 = 44.009
        MolWtH2O = 18.015
        MolWt = self.get_config('chemical:transformations:MolWt')

        diss = self.get_config('chemical:transformations:dissociation')

        pKa_acid = self.get_config('chemical:transformations:pKa_acid')
        if pKa_acid < 0 and diss in ['amphoteric', 'acid']:
            raise ValueError("pKa_acid must be positive")

        pKa_base = self.get_config('chemical:transformations:pKa_base')
        if pKa_base < 0 and diss in ['amphoteric', 'base']:
            raise ValueError("pKa_base must be positive")

        if diss == 'amphoteric' and abs(pKa_acid - pKa_base) < 2:
            raise ValueError("pKa_base and pKa_acid must differ of at least two units")

        Vp = self.get_config('chemical:transformations:Vpress')
        Tref_Vp = self.get_config('chemical:transformations:Tref_Vpress')
        DH_Vp = self.get_config('chemical:transformations:DeltaH_Vpress')

        Slb = self.get_config('chemical:transformations:Solub')
        Tref_Slb = self.get_config('chemical:transformations:Tref_Solub')
        DH_Slb = self.get_config('chemical:transformations:DeltaH_Solub')

        H0 = self.get_config('chemical:transformations:Henry')  # atm m3/mol
        Tref_H0 = self.get_config('chemical:transformations:Tref_Henry')

        R = 8.206e-05  # (atm m3)/(mol K)
        # Henry constant
        if H0 < 0:
            if Vp > 0 and Slb > 0:
                logger.debug("Henry constant calculated from Vp and Slb")
                Henry = (
                    (Vp * self.tempcorr("Arrhenius", DH_Vp, T, Tref_Vp)) /
                    (Slb * self.tempcorr("Arrhenius", DH_Slb, T, Tref_Slb))
                ) * MolWt / 101325.0  # atm m3 mol-1
            else:
                raise ValueError("Vp, Slb, and Henry not specified")
        else:
            logger.debug("Henry constant calculated from chemical:transformations:Henry")
            DH_H0 = self.get_config('chemical:transformations:DeltaH_Henry')
            Henry = H0 * self.tempcorr("Arrhenius", DH_H0, T, Tref_H0)

        # Fraction in volatilizable neutral/undissociated form
        if diss == 'nondiss':
            Undiss_n = np.ones_like(pH_water)
        elif diss == 'acid':
            Undiss_n = 1.0 / (1.0 + 10.0 ** (pH_water - pKa_acid))
        elif diss == 'base':
            # Neutral base B is the volatilizing form; pKa_base is pKa of BH+
            Undiss_n = 1.0 - (1.0 / (1.0 + 10.0 ** (pH_water - pKa_base)))
        elif diss == 'amphoteric':
            # Neutral fraction only; zwitterion ignored as in the original implementation
            Undiss_n = 1.0 / (1.0 + 10.0 ** (pH_water - pKa_acid) + 10.0 ** (pKa_base - pH_water))
        else:
            raise ValueError(f"Unknown dissociation mode: {diss!r}")

        # Water-side mass transfer coefficient
        # Schwarzenbach et al., 2016 Eq. (19-20)
        MTCw = (((9e-4) + (7.2e-6 * wind ** 3)) * (MolWtCO2 / MolWt) ** 0.25) * Undiss_n

        # Air-side mass transfer coefficient
        # Schwarzenbach et al., 2016 Eq. (19-17)(19-18)(19-19)
        Sca_H2O = 0.62
        MTCaH2O = (0.1+ wind * (6.1 + 0.63 * wind) ** 0.5
            / (13.3 * (Sca_H2O ** 0.5) + (6.1e-4 + (6.3e-5) * wind) ** -0.5 - 5.0
            + 1.25 * np.log(Sca_H2O)))
        MTCa = MTCaH2O * (MolWtH2O / MolWt) ** (1.0 / 3.0)

        # Overall volatilization mass transfer coefficient
        HenryLaw = Henry * (1.0 + 0.01143 * S) / (R * (T + 273.15))
        HenryLaw = np.maximum(HenryLaw, 1e-30)

        MTCvol = 1.0 / (1.0 / np.maximum(MTCw, 1e-30) +
            1.0 / np.maximum(MTCa * HenryLaw, 1e-30))  # cm/s

        K_volatilization = 0.01 * MTCvol / Thick  # 1/s

        dt = self.time_step.total_seconds()
        mass_local = np.asarray(self.elements.mass[idx], dtype=float)

        volatilized_now_local = np.minimum(mass_local,
            mass_local * (1.0 - np.exp(-K_volatilization * dt)))

        # Update cumulative volatilized mass and remaining mass
        self.elements.mass_volatilized[idx] += volatilized_now_local
        self.elements.mass[idx] -= volatilized_now_local
        self.elements.mass[idx] = np.maximum(self.elements.mass[idx], 0.0)

        self.deactivate_elements(self.elements.mass <
            (self.elements.mass + self.elements.mass_degraded + self.elements.mass_volatilized) / 500.0,
            reason='removed')

    ###########################################################################
    # Runtime post-seeding map updates
    ###########################################################################

    def _apply_mapped_bed_d50_to_new_elements(self):
        """
        Seeding can occur before self.environment exists, so sediment elements are
        initially assigned the configured fallback d50 in seed_elements(). This helper
        replaces that fallback with the local mapped/configured bed d50 as soon as
        readers/environment are available.

        The update is applied only once, using:
            self.elements.age_seconds <= 0.0
        as the one-shot condition for "first update after seeding".

        Only sediment species are updated:
          - Sediment reversible
          - Sediment slowly reversible
          - Sediment buried
          - Sediment irreversible
        Both:
          - self.elements.d50
          - self.elements.diameter
        are updated to the same value, explicitly avoiding any mismatch between
        particle diameter and d50 for sediment elements.
        Values are taken from self._local_bed_d50(idx=...), which already handles:
          - mapped bed d50 when available at runtime
          - configured fallback chemical:sediment:d50
          - USER-mode override behavior, if applicable
        """
        # Environment/readers may not exist during seeding; this helper is runtime-only
        if not hasattr(self, 'environment'):
            return
        # One-shot trigger: only newly active elements at their first runtime update
        ages = np.asarray(self.elements.age_seconds, dtype=float)
        dt_s = float(abs(self.time_step.total_seconds()))

        if logger.isEnabledFor(logging.DEBUG) and ages.size > 0:
            logger.debug(
                "self.elements.age_seconds: mean [%s] min [%s] max [%s]",
                float(ages.mean()), float(ages.min()), float(ages.max()),
            )
        eps = 1e-9
        newmask = np.asarray(ages <= dt_s + eps, dtype=bool)
        if not np.any(newmask):
            return
        name_to_idx = {name: i for i, name in enumerate(self.name_species)}
        sediment_names = {
            "Sediment reversible",
            "Sediment slowly reversible",
            "Sediment buried",
            "Sediment irreversible",
        }
        sediment_idx = {name_to_idx[n] for n in sediment_names if n in name_to_idx}
        if not sediment_idx:
            return
        sedmask = newmask & np.isin(self.elements.specie, list(sediment_idx))
        if not np.any(sedmask):
            return
        ii = np.flatnonzero(sedmask)
        # _local_bed_d50 already handles map/config fallback and USER-mode logic.
        d50_local = np.asarray(self._local_bed_d50(idx=ii), dtype=float).ravel()
        d50_local = np.maximum(d50_local, 0.0)
        # Keep sediment d50 and diameter explicitly identical.
        self.elements.d50[ii] = d50_local
        self.elements.diameter[ii] = d50_local

    def _apply_mapped_fOC_to_new_elements(self):
        """
        Seeding can occur before self.environment exists, so elements are initially
        assigned fallback f_OC values in seed_elements() using:
          - chemical:transformations:fOC_SPM   for particle-family species
          - chemical:transformations:fOC_sed   for sediment-family species

        This helper replaces those fallback values with local mapped/configured values
        as soon as readers/environment are available.

        The update is applied only once, using:
            age_seconds <= one time step
        as the one-shot condition for "first update after seeding".

        Particle family species:
          - Particle reversible
          - Particle slowly reversible
          - Particle irreversible
        Sediment family species:
          - Sediment reversible
          - Sediment slowly reversible
          - Sediment buried
          - Sediment irreversible

        Values are taken from:
          - self._local_particle_fOC(idx=...)
          - self._local_sediment_fOC(idx=...)
        These helpers already handle:
          - mapped f_OC when available at runtime
          - configured fallbacks when maps are absent/invalid
        """
        import numpy as np

        # Environment/readers may not exist during seeding; this helper is runtime-only.
        if not hasattr(self, 'environment'):
            return
        ages = np.asarray(self.elements.age_seconds, dtype=float)
        dt_s = float(abs(self.time_step.total_seconds()))
        eps = 1e-9

        if logger.isEnabledFor(logging.DEBUG) and ages.size > 0:
            logger.debug(
                "self.elements.age_seconds: mean [%s] min [%s] max [%s]",
                float(ages.mean()), float(ages.min()), float(ages.max()),
            )
        # One-shot trigger: elements at their first runtime update
        newmask = np.asarray(ages <= dt_s + eps, dtype=bool)
        if not np.any(newmask):
            return

        name_to_idx = {name: i for i, name in enumerate(self.name_species)}
        particle_names = {
            "Particle reversible",
            "Particle slowly reversible",
            "Particle irreversible",
        }
        sediment_names = {
            "Sediment reversible",
            "Sediment slowly reversible",
            "Sediment buried",
            "Sediment irreversible",
        }
        particle_idx = {name_to_idx[n] for n in particle_names if n in name_to_idx}
        sediment_idx = {name_to_idx[n] for n in sediment_names if n in name_to_idx}

        if particle_idx:
            partmask = newmask & np.isin(self.elements.specie, list(particle_idx))
            if np.any(partmask):
                ii = np.flatnonzero(partmask)
                self.elements.f_OC[ii] = self._local_particle_fOC(ii)
        if sediment_idx:
            sedmask = newmask & np.isin(self.elements.specie, list(sediment_idx))
            if np.any(sedmask):
                ii = np.flatnonzero(sedmask)
                self.elements.f_OC[ii] = self._local_sediment_fOC(ii)

    ###########################################################################
    # Main update loop
    ###########################################################################

    def update(self):
        """Update positions and properties of Chemical particles."""
        # Workaround due to conversion of datatype
        self.elements.specie = self.elements.specie.astype(np.int32)

        # First-step remap of sediment d50 from reader/configured local bed values
        self._apply_mapped_bed_d50_to_new_elements()
        # First-step remap of carrier f_OC from reader/configured local values
        self._apply_mapped_fOC_to_new_elements()

        # Degradation and Volatilization
        if self.get_config('chemical:transfer_setup')=='organics' or self.get_config('chemical:transfer_setup')=='custom':
            self.degradation()
            self.volatilization()

        # Dynamic Partitioning
        if self.get_config('chemical:dynamic_partitioning') is True:
            self.update_transfer_rates()
            self.update_partitioning()

        # Turbulent Mixing
        if self.get_config('drift:vertical_mixing') is True:
            self.update_terminal_velocity()
            self.vertical_mixing()
        else:
            self.update_terminal_velocity()
            self.vertical_buoyancy()

        # Deposition and resuspension
        self.bed_exchange()

        if logger.isEnabledFor(logging.DEBUG):
            specie = np.asarray(self.elements.specie, dtype=np.int64)
            counts = np.bincount(specie, minlength=self.nspecies).tolist()
            logger.debug("partitioning: %s %s", counts, self.name_species)

        # Horizontal advection
        self.advect_ocean_current()

        # Vertical advection
        if self.get_config('drift:vertical_advection') is True:
            self.vertical_advection()

        # Update transfer rates after last time step
        if self.time == (self.expected_end_time - self.time_step) or \
           self.time == (self.expected_end_time) or \
           self.num_elements_active() == 0 :
               self.update_transfer_rates()

    ###########################################################################
    # POSTPROCESSING
    ###########################################################################

# Bind post-processing metadata after both final classes exist.
ChemicalDrift._initialize_chemicaldrift_postprocess_registry(Chemical)
