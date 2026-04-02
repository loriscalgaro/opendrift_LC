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
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC, CONFIG_LEVEL_ADVANCED
import pyproj
from datetime import datetime

# Defining the Chemical element properties
class Chemical(Lagrangian3DArray):
    """Extending Lagrangian3DArray with specific properties for chemicals
    """

    variables = Lagrangian3DArray.add_variables([
        ('diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.}),
        #('neutral_buoyancy_salinity', {'dtype': np.float32,
        #                               'units': '[]',
        #                               'default': 31.25}),  # for NEA Cod
        ('density', {'dtype': np.float32,
                     'units': 'kg/m^3',
                     'default': 2650.}),  # Mineral particles
        ('critstress_factor', {'dtype': np.float32,
                               'units': '',
                               'default': 1.0}),
        ('f_OC', {'dtype': np.float32,
                               'units': '',
                               'default': 0.01}),
        ('specie', {'dtype': np.int32,
                    'units': '',
                    'default': 0}),
        ('mass', {'dtype': np.float32,
                      'units': 'ug',
                      'seed': True,
                      'default': 1e3}),
        ('mass_degraded', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_degraded_water', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_degraded_sediment', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_volatilized', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_photodegraded', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_biodegraded', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_biodegraded_water', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_biodegraded_sediment', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_hydrolyzed', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_hydrolyzed_water', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0}),
        ('mass_hydrolyzed_sediment', {'dtype': np.float32,
                             'units': 'ug',
                             'seed': True,
                             'default': 0})
        ])


class ChemicalDrift(OceanDrift):
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

    ElementType = Chemical

    required_variables = {
         # Hydodinamics
        'x_sea_water_velocity': {'fallback': None},                            # m/s
        'y_sea_water_velocity': {'fallback': None},                            # m/s
        'upward_sea_water_velocity': {'fallback': 0},                          # m/s
        'sea_surface_height': {'fallback': 0},                                 # m
        'x_wind': {'fallback': 0},                                             # m/s
        'y_wind': {'fallback': 0},                                             # m/s
        'ocean_vertical_diffusivity': {'fallback': 0.0001, 'profiles': True},  # m2/s
        'horizontal_diffusivity': {'fallback': 0, 'important': False},         # m2/s
        'land_binary_mask': {'fallback': None},                                # unitless (0/1 or boolean mask)
        'sea_floor_depth_below_sea_level': {'fallback': 10000},                # m (positive downward)
         # Overall degradation and partitioning
        'sea_water_temperature': {'fallback': 10, 'profiles': True},           # degC
        'sea_water_salinity': {'fallback': 34, 'profiles': True},              # PSU
        'doc': {'fallback': 0.0},                                              # mmol C/kg
        'spm': {'fallback': 1},                                                # g/m3
        'f_OC_spm': {'fallback': 0.01},                                        # gOC/g (dimensionless mass fraction)
        'f_OC_sed': {'fallback': 0.01},                                        # gOC/g (dimensionless mass fraction)
        'ocean_mixed_layer_thickness': {'fallback': 50},                       # m
        'sea_water_ph_reported_on_total_scale':{'fallback': 8.1, 'profiles': True}, # pH units (dimensionless)
        'pH_sediment':{'fallback': 6.9, 'profiles': False},
         # # Hydodinamics for sediment interaction
        'x_bottom_sea_water_velocity': {'fallback': 0, 'important': False},    # m/s
        'y_bottom_sea_water_velocity': {'fallback': 0, 'important': False},    # m/s
        'bottom_layer_thickness': {'fallback': 0.0, 'important': False},       # m
        'sea_floor_roughness_length': {'fallback': 0.0, 'important': False},   # m
        # Direct bed stresses from hydro model (preferred if available)
        'sea_floor_current_stress': {'fallback': 0, 'important': False},       # Pa
        'sea_floor_wave_stress': {'fallback': 0, 'important': False},          # Pa
        'sea_floor_other_stress': {'fallback': 0, 'important': False},         # Pa
        'sea_floor_wave_orbital_velocity': {'fallback': 0, 'important': False},# m/s
        'sea_surface_wave_period_at_variance_spectral_density_maximum': {'fallback': 0, 'important': False}, # s
        'sea_surface_wave_to_direction': {'fallback': 0, 'important': False},  # degrees clockwise from north, “to” direction
        # Bulk-flow inputs for Manning/Chezy/White-Colebrook modes
        'x_depth_averaged_sea_water_velocity': {'fallback': 0, 'important': False}, # m/s
        'y_depth_averaged_sea_water_velocity': {'fallback': 0, 'important': False}, # m/s
        'hydraulic_radius': {'fallback': 0, 'important': False},               # m
        # Sediment interaction
            # Map of 'chemical:sediment:mixing_depth'
        'active_sediment_layer_thickness': {'fallback': 0},                    # m
            # Map of 'chemical:sediment:layer_thickness'
        'interaction_sediment_layer_thickness': {'fallback': 0},               # m
         # Single-process degradation
        'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water':{'fallback': 225, 'profiles': True},      # mmol/m3
        'mole_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water':{'fallback': 0, 'profiles': True}, # mmol C/m3
        'solar_irradiance':{'fallback': 241} # W/m2 by default, or Ly/day if chemical:transformations:solar_input_unit = 'Ly_day'
        }

    def specie_num2name(self,num):
        return self.name_species[num]

    def specie_name2num(self,name):
        num = self.name_species.index(name)
        return num

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
                'min': 0, 'max': 1200, 'units': 'm',                                    # Vertical conc drops more slowly slower than for SPM
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},                     # example: 10.3389/fmars.2017.00436. lower limit around 40 umol/L
            'chemical:particle_diameter_uncertainty': {'type': 'float', 'default': 0.35,  # https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/latest/model-description/water-and-sediment-properties/sediment-properties
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
            'chemical:transformations:Dc': {'type': 'float', 'default': 1.16e-5,                # Simonsen 2019
                'min': 0, 'max': 1e6, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Desorption rate of metals from particles'},
            'chemical:transformations:slow_coeff': {'type': 'float', 'default': 0, #1.2e-7,         # Simonsen 2019
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
                'enum': ['nondiss','acid', 'base', 'amphoter'], 'default': 'nondiss',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Select dissociation mode'},
            'chemical:transformations:LogKOW': {'type': 'float', 'default': 3.361,          # Naphthalene
                'min': -3, 'max': 10, 'units': '',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Log10 of Octanol/Water partitioning coefficient'},
            'chemical:transformations:TrefKOW': {'type': 'float', 'default': 25.,           # Naphthalene
                'min': -3, 'max': 30, 'units': 'C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Reference temperature of KOW'},
            'chemical:transformations:DeltaH_KOC_Sed': {'type': 'float', 'default': -21036., # Naphthalene
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of sorption to sediments'},
            'chemical:transformations:DeltaH_KOC_DOM': {'type': 'float', 'default': -25900., # Naphthalene
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of sorption to DOM'},
            'chemical:transformations:Setchenow': {'type': 'float', 'default': 0.2503,      # Naphthalene
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
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Organic carbon fration of sediments'},
            'chemical:transformations:aggregation_rate': {'type': 'float', 'default': 0,
                'min': 0, 'max': 1, 'units': 's-1',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Aggregation rate of DOM in marine water'},
            # Degradation in water column
            'chemical:transformations:t12_W_tot': {'type': 'float', 'default': 224.08,      # Naphthalene
                'min': 1, 'max': None, 'units': 'hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Half life in water, total'},
            'chemical:transformations:Tref_kWt': {'type': 'float', 'default': 25.,          # Naphthalene
                'min': -3, 'max': 30, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Reference temperature of t12_W_tot'},
            'chemical:transformations:DeltaH_kWt': {'type': 'float', 'default': 50000.,     # generic
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Entalpy of t12_W_tot'},
            # Degradation in sediment layer
            'chemical:transformations:t12_S_tot': {'type': 'float', 'default': 5012.4,      # Naphthalene
                'min': 1, 'max': None, 'units': 'hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Half life in sediments, total'},
            'chemical:transformations:ssrev_slow_deg_factor': {'type': 'float', 'default': 1,      # No slow degradation
                'min': 0, 'max': 1, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Correction factor for slower degradation in buried sediments'},
            'chemical:transformations:Tref_kSt': {'type': 'float', 'default': 25.,          # Naphthalene
                'min': -3, 'max': 30, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Reference temperature of t12_S_tot'},
            'chemical:transformations:DeltaH_kSt': {'type': 'float', 'default': 50000.,     # generic
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Entalpy of t12_S_tot'},
            # Volatilization
            'chemical:transformations:MolWt': {'type': 'float', 'default': 128.1705,         # Naphthalene
                'min': 50, 'max': 1000, 'units': 'g/mol',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Molecular weight'},
            'chemical:transformations:Henry': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'atm m3 mol-1',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Henry constant (uses Tref_Slb as Tref)'},
            'chemical:transformations:DeltaH_Henry': {'type': 'float', 'default': 0.0,
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL,
                'description': 'Effective enthalpy used to temperature-correct a directly supplied Henry constant.'},
            # Vapour pressure
            'chemical:transformations:Vpress': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'Pa',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Vapour pressure'},
            'chemical:transformations:Tref_Vpress': {'type': 'float', 'default': 25.,        # Naphthalene
                'min': None, 'max': None, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Vapour pressure ref temp'},
            'chemical:transformations:DeltaH_Vpress': {'type': 'float', 'default': 55925.,   # Naphthalene
                'min': -100000., 'max': 150000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of volatilization'},
            # Solubility
            'chemical:transformations:Solub': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'g/m3',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Solubility'},
            'chemical:transformations:Tref_Solub': {'type': 'float', 'default': 25.,         # Naphthalene
                'min': None, 'max': None, 'units': '°C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Solubility ref temp'},
            'chemical:transformations:DeltaH_Solub': {'type': 'float', 'default': 25300.,    # Naphthalene
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of solubilization'},
            'chemical:transformations:Tref_Henry': {'type': 'float', 'default': 25.,         # Naphthalene
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
            'chemical:sediment:desorption_depth_uncert': {'type': 'float', 'default': .5,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:sediment:resuspension_depth': {'type': 'float', 'default': 1,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Distance from seabed where resuspended elements are moved'},
            'chemical:sediment:resuspension_depth_uncert': {'type': 'float', 'default': .5,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:sediment:burial_rate': {'type': 'float', 'default': .0003,   # MacKay
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
                'description': 'Median grain size for GRAIN_D50 mode and d50-based critical stress.'},
            'chemical:sediment:cd_min': {'type': 'float', 'default': 0.0,
                'min': 0, 'max': 1, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Lower bound for drag coefficient.'},
            'chemical:sediment:cd_max': {'type': 'float', 'default': 0.2,
                'min': 0, 'max': 10, 'units': '', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Upper bound for drag coefficient.'},
            'chemical:sediment:include_wave_stress': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Whether to include externally provided wave bed stress when available.'},
            'chemical:sediment:include_other_stress': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Whether to include externally provided other bed stress when available.'},
            'chemical:sediment:shear_stress_combination': {'type': 'enum',
                'enum': ['sum', 'max', 'rss', 'SOULSBY_CLARKE'], 'default': 'sum',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Heuristic scalar combination of current, wave, and other bed-stress magnitudes. Not a full nonlinear wave-current interaction model.'},
            'chemical:sediment:use_critstress_heterogeneity': {
                'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_BASIC,
                'description': 'Apply persistent sub-grid heterogeneity factor to resuspension critical stress.'},
            'chemical:sediment:critstress_heterogeneity_lnsigma': {
                'type': 'float', 'default': 0.15, 'min': 0.0, 'max': 2.0, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Log-space standard deviation of the persistent multiplicative heterogeneity factor for resuspension critical stress.'},
            # Probabilistic exchange scheme
            'chemical:sediment:erodibility_M': {'type': 'float', 'default': 1.0e-4,
                'min': 0.0, 'max': 1.0, 'units': 'kg m-2 s-1 Pa-1', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Partheniades erodibility coefficient for cohesive resuspension.'},
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
                'description': 'Critical shear stress for resuspension/erosion.'},
            'chemical:sediment:resuspension_critustar': {'type': 'float', 'default': -1.0,
                'min': -1.0, 'max': 100.0, 'units': 'm/s', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Critical shear velocity for resuspension. If >= 0, override resuspension_critstress using tau_cr = rho * ustar^2. If < 0, use resuspension_critstress directly.'},
            'chemical:sediment:bed_interaction_thickness': {'type': 'float', 'default': 1.0,
                'min': 0, 'max': 100, 'units': 'm', 'level': CONFIG_LEVEL_ADVANCED,
                'description': 'Near-bed water-layer thickness used in deposition probability.'},
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
            'chemical:transformations:Save_single_degr_mass': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle save of mass degraded by single mechanism'},
            'chemical:transformations:Photodegradation': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle photodegradation'},
            'chemical:transformations:Biodegradation': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle biodegradation'},
            'chemical:transformations:Hydrolysis': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle hydrolysis'},
            # Biodegradation
            'chemical:transformations:k_DecayMax_water': {'type': 'float', 'default': 0.054,      # from AQUATOX Database (0.13 1/day)
                'min': 0, 'max': None, 'units': '1/hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Max first-order rate constant for biodegradation in aerobic condition'},
            'chemical:transformations:k_Anaerobic_water': {'type': 'float', 'default': 0,      # Default for no anaerobic biodegradation
                'min': 0, 'max': None, 'units': '1/hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Max first-order rate constant for biodegradation in anaerobic condition '},
            'chemical:transformations:HalfSatO_w': {'type': 'float', 'default': 0.5,      # Half-saturation constant for oxygen, default from AQUATOX Database
                'min': 0.01, 'max': None, 'units': 'g/m3',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Half-saturation constant for oxygen, default from AQUATOX Database'},
            'chemical:transformations:T_Max_bio': {'type': 'float', 'default': 50,     # Default from AQUATOX Database
                'min': 1, 'max': None, 'units': 'C',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Maximum temperature at which biodegradation process will occur, default from AQUATOX Database'},
            'chemical:transformations:T_Opt_bio': {'type': 'float', 'default': 24,     # Default from AQUATOX Database
                 'min': 1, 'max': None, 'units': 'C',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Optimal temperature for biodegradation, default from AQUATOX Database'},
            'chemical:transformations:T_Adp_bio': {'type': 'float', 'default': 2,     # Default from AQUATOX Database
                 'min': 0.1, 'max': None, 'units': 'C',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': '“adaptation” temperature below which there is no acclimation for biobegradation, default from AQUATOX Database'},
            'chemical:transformations:Max_Accl_bio': {'type': 'float', 'default': 2,     # Default from AQUATOX Database
                 'min': 0.1, 'max': None, 'units': 'C',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Maximum acclimation allowed for biodegratation, default from AQUATOX Database'},
            'chemical:transformations:Dec_Accl_bio': {'type': 'float', 'default': 0.5,     # Default from AQUATOX Database
                 'min': 0.1, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Coefficient for decreasing acclimation as temperature approaches T_Adp_bio, default from AQUATOX Database'},
            'chemical:transformations:Q10_bio': {'type': 'float', 'default': 2,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Slope or rate of change per 10°C temperature change for biodegradation, default from AQUATOX Database'},
            'chemical:transformations:pH_min_bio': {'type': 'float', 'default': 5,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Minimum pH below which limitation on biodegradation rate occurs, default from AQUATOX Database'},
            'chemical:transformations:pH_max_bio': {'type': 'float', 'default': 8.5,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Maximum pH over which limitation on biodegradation rate occurs, default from AQUATOX Database'},
            # Hydrolysis
            # Based on the approach reported by Mabey, W., & Mill, T. (1978) https://doi.org/10.1063/1.555572 (Figure 1)
            'chemical:transformations:k_Acid': {'type': 'float', 'default': 0,     # Default: no acid catalyzed hydrolysis
                 'min': None, 'max': None, 'units': 'L/mol*h',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'acid-catalyzed pseudo-second-order coefficient'},
            'chemical:transformations:k_Base': {'type': 'float', 'default': 0,     # Default: no base catalyzed hydrolysis
                 'min': None, 'max': None, 'units': 'L/mol*h',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'base-catalyzed pseudo-second-order coefficient'},
            'chemical:transformations:k_Hydr_Uncat': {'type': 'float', 'default': 0,     # Default: no hydrolysis
                 'min': 0, 'max': None, 'units': '1/hours',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Measured first-order hydrolysis rate at pH 7'},
            # Photolysis
            'chemical:transformations:solar_input_unit': { 'type': 'enum', 'enum': ['W_m2', 'Ly_day'], 'default': 'W_m2',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Unit of environment.solar_irradiance. AQUATOX photolysis uses Ly/day; W/m2 is converted using 1 Ly/day = 0.4843 W/m2.'},
            'chemical:transformations:k_Photo': {'type': 'float', 'default': 0,     # Default: no photolysis
                 'min': 0, 'max': None, 'units': '1/hours',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Measured first-order photolysis rate'},
            'chemical:transformations:RadDistr': {'type': 'float', 'default': 1.6,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Radiance distribution function, which is the ratio of the average pathlength to the depth'},
            'chemical:transformations:RadDistr0_ml': {'type': 'float', 'default': 1.2,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Standard radiance distribution function in the Mixed Layer'},
            'chemical:transformations:RadDistr0_bml': {'type': 'float', 'default': 1.6,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Standard radiance distribution function below the Mixed Layer'},
            'chemical:transformations:WaterExt': {'type': 'float', 'default': 0.21,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '1/m',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Extinction coefficient of light in the water with depht due to water'},
            'chemical:transformations:ExtCoeffDOM': {'type': 'float', 'default': 0.028,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '1/(m*g/m3)',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Extinction coefficient of light in the water with depht due to DOM'},
            'chemical:transformations:ExtCoeffSPM': {'type': 'float', 'default': 0.17,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '1/(m*g/m3)',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Extinction coefficient of light in the water with depht due to SPM'},
            'chemical:transformations:ExtCoeffPHY': {'type': 'float', 'default': 0.14,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '1/(m*g/m3)',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Extinction coefficient of light in the water with depht due to phytoplankton'},
            'chemical:transformations:C2PHYC': {'type': 'float', 'default': 0.44,     # Default from https://doi.org/10.1007/BF00006636
                 'min': 0, 'max': None, 'units': 'g_Caron/g_Biomass',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Phytoplankton carbon content'},
            'chemical:transformations:AveSolar': {'type': 'float', 'default': 500,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': 'Ly/day',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Average light intensity for late spring or early summer, corresponding to time when photolytic half-life is often measured'},
            })

        self._set_config_default('drift:vertical_mixing', True)
        self._set_config_default('drift:vertical_mixing_at_surface', True)
        self._set_config_default('drift:vertical_advection_at_surface', True)

    def prepare_run(self):
        if not hasattr(self, "name_species"):
            self.init_species()
        if not hasattr(self, "transfer_rates"):
            self.init_transfer_rates()

        logger.info( 'Number of species: {}'.format(self.nspecies) )
        for i,sp in enumerate(self.name_species):
            logger.info( '{:>3} {}'.format( i, sp ) )

        logger.info('transfer setup: %s' % self.get_config('chemical:transfer_setup'))
        logger.info('nspecies: %s' % self.nspecies)
        logger.info('Transfer rates:\n %s' % self.transfer_rates)

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

        # Keep track of supplied readers
        self._reader_variables = set()
        for _, reader in self.env.readers.items():
            self._reader_variables.update(getattr(reader, 'variables', []))

        # List of additional custom variables to be saved in self.result
        # TODO: These could now be moved to post_run() which should be
        # more robust in case variables are changed during run()

        savelist = ['nspecies', 'name_species',
                    'transfer_rates', 'ntransformations']

        # Add all variables starting with "num_"
        savelist.extend(k for k in vars(self) if k.startswith("num_"))

        # Saving the variables
        for var_name in savelist:
            var_value = getattr(self, var_name)
            if isinstance(var_value, np.ndarray):
                dims = tuple(f'specie_{i}' for i in range(var_value.ndim))
                self.result[var_name] = (dims, var_value)
            else:
                self.result[var_name] = var_value

        super(ChemicalDrift, self).prepare_run()

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

#         logger.info( 'Number of species: {}'.format(self.nspecies) )
#         for i,sp in enumerate(self.name_species):
#             logger.info( '{:>3} {}'.format( i, sp ))

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

    def seed_elements(self, *args, **kwargs):
        import numpy as np

        if hasattr(self, 'name_species') is False:
            self.init_species()
            self.init_transfer_rates()

        # Number of elements
        if 'number' in kwargs:
            num_elements = int(kwargs['number'])
        else:
            num_elements = int(self.get_config('seed:number'))

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

        logger.debug('Initial partitioning:')
        for i, sp_name in enumerate(self.name_species):
            logger.debug('{:>9} {:>3} {:24} '.format(np.sum(init_specie == i), i, sp_name))

        # Diameter assignment (respect explicit per-element diameters)
        diam_in = kwargs.get("diameter", None)

        arr = _as_per_element_array(diam_in, num_elements, "diameter")
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

            particle_like_names = {
                "Particle reversible",
                "Particle slowly reversible",
                "Particle irreversible",
                "Sediment reversible",
                "Sediment slowly reversible",
                "Sediment buried",
                "Sediment irreversible",
            }
            particle_like_idx = {name_to_idx[n] for n in particle_like_names if n in name_to_idx}

            humic_idx = name_to_idx.get("Humic colloid", None)
            polymer_idx = name_to_idx.get("Polymer", None)
            colloid_idx = name_to_idx.get("Colloid", None)

            if particle_like_idx:
                mask_part = np.isin(init_specie, list(particle_like_idx))
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

    ### Functions for temperature and salinity correction
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

    ### Functions to update partitioning coefficients
    def speciation_fractions(self, diss, pH, pKa_acid, pKa_base):
        """
        Return speciation fractions as (phi_neutral, phi_anion, phi_cation).

          - acid: HA (neutral) <-> A- + H+      pKa_acid = pKa(HA)
          - base: BH+ (cation) <-> B + H+       pKa_base = pKa(BH+)  (conjugate-acid pKa)
          - amphoter: one acidic site + one basic site, ignoring zwitterion:
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

        if diss == "amphoter":
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
        diss:                               Dissociation model {'nondiss', 'acid', 'base', 'amphoter'}.
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

    ### Biodegradation
    def calc_DOCorr(self, HalfSatO_w, k_Anaerobic_water, k_DecayMax_water, Ox_water):
        '''
        Dissolved-oxygen correction factor for biodegradation.
        Follows a Monod-type oxygen limitation with optional anaerobic fallback:
            MMFact_w = O2 / (HalfSatO_w + O2)
            DOCorr = MMFact_w + (1 - MMFact_w) * (k_Anaerobic_water / k_DecayMax_water)
        where:
            O2:                dissolved oxygen concentration [g/m3]
            HalfSatO_w:        half-saturation oxygen concentration [g/m3]
            k_DecayMax_water:  maximum aerobic biodegradation rate [1/h]
            k_Anaerobic_water: anaerobic biodegradation rate [1/h]
                '''
        HalfSatO_w = self._validate_scalar_param("HalfSatO_w", HalfSatO_w, gt=0.0)
        k_Anaerobic_water = self._validate_scalar_param("k_Anaerobic_water", k_Anaerobic_water, ge=0.0)
        k_DecayMax_water = self._validate_scalar_param("k_DecayMax_water", k_DecayMax_water, ge=0.0)
        Ox_water = self._validate_array_param("Ox_water", Ox_water, ge=0.0)

        if k_DecayMax_water == 0.0:
            if k_Anaerobic_water > 0.0:
                raise ValueError(
                    "k_Anaerobic_water cannot be > 0 when k_DecayMax_water == 0.")
            return np.zeros_like(Ox_water, dtype=float)

        if k_Anaerobic_water > k_DecayMax_water:
            raise ValueError(
                "k_Anaerobic_water should not exceed k_DecayMax_water.")

        DOCorr = np.zeros_like(Ox_water)
        N = len(DOCorr)
        chunk_size = int(1e5)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            Ox_water_chunk = Ox_water[i:end]

            if k_DecayMax_water == 0:
                logger.debug("k_DecayMax_water is set to 0 1/h, therefore  DOCorr = 0 and no biodegradation occurs")
            elif k_DecayMax_water < 0:
                raise ValueError("k_DecayMax_water is set < 0 1/h, this is not possible")
            elif k_DecayMax_water > 0:
                MMFact_w = Ox_water_chunk / (HalfSatO_w + Ox_water_chunk)
                DOCorr[i:end] = MMFact_w + (1 - MMFact_w) * (k_Anaerobic_water / k_DecayMax_water)

            if np.any((DOCorr[i:end] < 0) | (DOCorr[i:end] > 1)):
                raise ValueError('DOCorr is not between 0 and 1')

        return DOCorr

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
        k_Acid = self._validate_scalar_param("k_Acid", k_Acid, ge=0.0)
        k_Base = self._validate_scalar_param("k_Base", k_Base, ge=0.0)
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
        k_Acid = self._validate_scalar_param("k_Acid", k_Acid, ge=0.0)
        k_Base = self._validate_scalar_param("k_Base", k_Base, ge=0.0)
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
                    n_high = int(np.sum(too_high))
                    max_high = float(np.nanmax(ScreeningFactor[i:end][too_high]))
                    logger.info(
                        "Clipping %s ScreeningFactor values > 1 to 1.0 (max before clip: %s)",
                        n_high, max_high)
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

    def koc_updated(self, KOC_neutral, KOC_anion, KOC_cation, pH, diss, pKa_acid, pKa_base):
        """Return pH-updated KOC = sum_i(KOC_i * phi_i).."""
        phi_neu, phi_an, phi_cat = self.speciation_fractions(diss, pH, pKa_acid, pKa_base)
        return (KOC_neutral * phi_neu) + (KOC_anion * phi_an) + (KOC_cation * phi_cat)

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

        self.transfer_rates = np.zeros([self.nspecies,self.nspecies])
        self.ntransformations = np.zeros([self.nspecies,self.nspecies])

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
            if diss in ["acid", "amphoter"] and pKa_acid < 0:
                raise ValueError("pKa_acid must be positive")
            if diss in ["base", "amphoter"] and pKa_base < 0:
                raise ValueError("pKa_base must be positive")
            if diss == "amphoter" and abs(pKa_acid - pKa_base) < 2:
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
                    elif diss == 'amphoter':
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

            logger.debug('Partitioning coefficients (Tref,freshwater)')
            logger.debug('KOC_sed: %s L/KgOC' % KOC_sed)
            logger.debug('KOC_SPM: %s L/KgOC' % KOC_SPM)
            logger.debug('KOC_DOM: %s L/KgOC' % KOC_DOM)

            # Convert to Kd (L/kg) using fOC (sed/SPM) and Org2C (DOM)
            #KOM_sed = KOC_sed * Org2C #  L/KgOC * KgOC/KgOM = L/KgOM
            #KOM_SPM = KOC_sed * Org2C #  L/KgOC * KgOC/KgOM = L/KgOM
            #KOM_DOM = KOC_DOM * Org2C #  L/KgOC * KgOC/KgOM = L/KgOM

            # to be calculated separately for sed, SPM, dom (different KOC, pH, fOC)
            self.Kd_sed = Kd_sed = KOC_sed * fOC_sed    # L/KgOC * KgOC/KG = L/Kg
            self.Kd_SPM = Kd_SPM = KOC_SPM * fOC_SPM    # L/KgOC * KgOC/KG = L/Kg
            self.Kd_DOM = Kd_DOM = KOC_DOM * Org2C      # L/KgOC * KgOC/KgOM * 1KgOM/Kg = L/Kg (=KOM_DOM)
            # TODO Use setconfig() to store these?

            logger.debug('Kd_sed: %s L/Kg' % Kd_sed)
            logger.debug('Kd_SPM: %s L/Kg' % Kd_SPM)
            logger.debug('Kd_DOM: %s L/Kg' % Kd_DOM)

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

        ## HACK :
        # self.transfer_rates[:] = 0.
        # print ('\n ###### \n IMPORTANT:: \n transfer rates have been hacked! \n#### \n ')

        logger.debug('nspecies: %s' % self.nspecies)
        logger.debug('Transfer rates:\n %s' % self.transfer_rates)

    def update_terminal_velocity(self, Tprofiles=None,
                                 Sprofiles=None, z_index=None):
        """Calculate terminal velocity for Pelagic Egg

        according to
        S. Sundby (1983): A one-dimensional model for the vertical
        distribution of pelagic fish eggs in the mixed layer
        Deep Sea Research (30) pp. 645-661

        Method copied from ibm.f90 module of LADIM:
        Vikebo, F., S. Sundby, B. Aadlandsvik and O. Otteraa (2007),
        Fish. Oceanogr. (16) pp. 216-228
        """
        g = 9.81  # ms-2

        # Particle properties that determine settling velocity
        partsize = self.elements.diameter
        # prepare interpolation of temp, salt
        if not (Tprofiles is None and Sprofiles is None):
            if z_index is None:
                from scipy.interpolate import interp1d
                z_i = range(Tprofiles.shape[0])  # evtl. move out of loop
                z_index = interp1d(-self.environment_profiles['z'],
                                   z_i, bounds_error=False)
            zi = z_index(-self.elements.z)
            upper = np.maximum(np.floor(zi).astype(np.uint8), 0)
            lower = np.minimum(upper+1, Tprofiles.shape[0]-1)
            weight_upper = 1 - (zi - upper)

        # do interpolation of temp, salt if profiles were passed into
        # this function, if not, use reader by calling self.environment
        if Tprofiles is None:
            T0 = self.environment.sea_water_temperature
        else:
            T0 = Tprofiles[upper, range(Tprofiles.shape[1])] * \
                weight_upper + \
                Tprofiles[lower, range(Tprofiles.shape[1])] * \
                (1-weight_upper)
        if Sprofiles is None:
            S0 = self.environment.sea_water_salinity
        else:
            S0 = Sprofiles[upper, range(Sprofiles.shape[1])] * \
                weight_upper + \
                Sprofiles[lower, range(Sprofiles.shape[1])] * \
                (1-weight_upper)

        DENSw = self.sea_water_density(T=T0, S=S0)
        DENSpart = self.elements.density
        dr = DENSw-DENSpart  # density difference

        # water viscosity
        my_w = 0.001*(1.7915 - 0.0538*T0 + 0.0007*(T0**2.0) + 0.0023*S0)
        # ~0.0014 kg m-1 s-1

        # terminal velocity for low Reynolds numbers
        W = (1.0/my_w)*(1.0/18.0)*g*partsize**2 * dr

        #W=np.zeros_like(W) #Setting to zero for debugging
        self.elements.terminal_velocity = W
        self.elements.terminal_velocity = W * self.elements.moving

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

    def update_chemical_fOC(self, sp_in=None, sp_out=None):
        """
        Update element organic-carbon fractions when dynamic partitioning changes carrier phase.
        1) Entering particle family from a non-particle family:    f_OC <- local particle f_OC
        2) Entering sediment family from a non-sediment family:    f_OC <- local sediment f_OC
        3) Transitions within the same family preserve the current f_OC:
               particle -> particle    : unchanged
               sediment -> sediment    : unchanged
        4) Dissolved / DOM-like species do not receive a new f_OC

        Deposition/resuspension preserve carrier f_OC.
        """
        if sp_in is None or sp_out is None:
            return

        sp_in = np.asarray(sp_in, dtype=int)
        sp_out = np.asarray(sp_out, dtype=int)

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

        was_particle = np.isin(sp_in, particle_species) if len(particle_species) > 0 else np.zeros_like(sp_in, dtype=bool)
        is_particle  = np.isin(sp_out, particle_species) if len(particle_species) > 0 else np.zeros_like(sp_out, dtype=bool)

        was_sediment = np.isin(sp_in, sediment_species) if len(sediment_species) > 0 else np.zeros_like(sp_in, dtype=bool)
        is_sediment  = np.isin(sp_out, sediment_species) if len(sediment_species) > 0 else np.zeros_like(sp_out, dtype=bool)

        entered_particle = np.flatnonzero(is_particle & (~was_particle))
        if entered_particle.size > 0:
            self.elements.f_OC[entered_particle] = self._local_particle_fOC(entered_particle)

        entered_sediment = np.flatnonzero(is_sediment & (~was_sediment))
        if entered_sediment.size > 0:
            self.elements.f_OC[entered_sediment] = self._local_sediment_fOC(entered_sediment)

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
        self.elements.transfer_rates1D = self.transfer_rates[specie, :]

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

            # Physical near-bed interaction layer used for geometric cutoff
            Sed_H_int = np.where(Sed_H_env > 0, Sed_H_env, Sed_H_0).astype(np.float32, copy=False)

            # Active sediment layer thickness used in k14 numerator
            Sed_L_eff = np.where(Sed_L_env > 0, Sed_L_env, Sed_L_0).astype(np.float32, copy=False)

            water_depth = np.asarray(water_depth, dtype=np.float32)
            mld = np.asarray(mld, dtype=np.float32)

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
                k14_corr[valid] = ((Sed_L_eff[valid] / H_rate_eff[valid]) /
                    (Sed_L_0 / Sed_H_0))

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
                if pKa_acid < 0 and diss in ['acid', 'amphoter']:
                    raise ValueError("pKa_acid must be positive")

                pKa_base = self.get_config('chemical:transformations:pKa_base')
                if pKa_base < 0 and diss in ['base', 'amphoter']:
                    raise ValueError("pKa_base must be positive")

                KOC_sed_n = self.get_config('chemical:transformations:KOC_sed')
                if KOC_sed_n < 0:
                    if diss == 'acid':
                        KOC_sed_n = 10 ** ((0.54 * np.log10(KOW)) + 1.11)
                    elif diss == 'base':
                        KOC_sed_n = 10 ** ((0.37 * np.log10(KOW)) + 1.70)
                    elif diss == 'amphoter':
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


    def update_partitioning(self):
        '''
        Apply stochastic dynamic partitioning for one timestep.
        This routine interprets the per-element transfer-rate matrix as a
        continuous-time Markov jump process among species.

        Mathematical steps for each element e:
        1) Current outgoing rates
               K[e, j] = rate from current species to species j   [1/s]
        2) Total leaving rate
               k_tot[e] = sum_j K[e, j]
        3) Probability that at least one transition occurs during dt
               p_any[e] = 1 - exp(-k_tot[e] * dt)
        4) Monte Carlo draw for whether a transition occurs
               phaseshift[e] ~ Bernoulli(p_any[e])
        5) Conditional destination probabilities, given that a transition occurred
               P(dest=j | transition) = K[e, j] / k_tot[e]
        6) Use inverse-transform sampling on the cumulative distribution function
           to choose the destination species.

        Post-processing after species are updated:
          - assign critstress heterogeneity to elements entering sediment pools
          - update cumulative transition counters ntransformations[i, j]
          - update diameter via update_chemical_diameter()
          - update f_OC via update_chemical_fOC()
          - update z/moving state for sorption/desorption relative to sediments
                '''
        specie_in  = self.elements.specie.copy()  # for storage of the initial partitioning
        specie_out = specie_in.copy()             # for storage of the final partitioning
        dt = self.time_step.total_seconds()       # length of a time step [s]

        # K: per-element transition rates between species (continuous-time Markov chain)
        # shape (N, nspecies), units [1/s]
        # Convention: for element e, K[e, j] is the rate of jumping FROM its current species TO species j
        # (typically with K[e, current_species] = 0)
        K = self.elements.transfer_rates1D        # shape (N, nspecies), rates [1/s]

        # Total rate of leaving the current state for each element:
        # k_tot[e] = sum_j K[e, j]  [1/s]
        # This is tranformation rate of "any transition happens" for that element.
        k_tot = np.sum(K, axis=1)                 # total probability of change out of current state

        # Probability that at least one transition occurs within dt for a Poisson process:
        # If leaving events occur with rate k_tot, then
        #   P(no transition in dt) = exp(-k_tot * dt)
        #   P(at least one transition in dt) = 1 - exp(-k_tot * dt)
        p_any = 1.0 - np.exp(-k_tot * dt)

        # First Monte Carlo draw: decide which elements will undergo a phase/species change this step
        u = np.random.random(self.num_elements_active())
        phaseshift = u < p_any                    # Denotes which trajectory that shall be transformed
        ntr = np.count_nonzero(phaseshift)
        logger.info("Number of transformations: %s", ntr)
        if ntr == 0:
            return

        # Restrict to transformed elements only
        K_sel = K[phaseshift]
        k_tot_sel = k_tot[phaseshift]

        # Conditional destination probabilities given that a transition occurs:
        # For continuous-time Markov chains, given that an event happens,
        # the probability it goes to state j is proportional to its rate:
        #   P(dest=j | transition) = K_sel[:, j] / k_tot_sel
        probs = K_sel / k_tot_sel[:, None]

        # Convert destination probabilities to CDF for inverse-transform sampling
        # cdf[e, j] = sum_{m<=j} probs[e, m]
        cdf = np.cumsum(probs, axis=1)

        # Second Monte Carlo draw: pick destination species using the CDF
        u2 = np.random.random(ntr)

        # idx is the first index where cdf >= u2
        # (cdf < u2).sum gives the count of bins strictly below u2, i.e. the chosen bin index
        idx = (cdf < u2[:, None]).sum(axis=1)

        # Safety clamp in case of tiny floating point deficits where last cdf < 1 by ~1e-16
        idx = np.minimum(idx, self.nspecies - 1)

        # Apply new species only to the transformed elements
        specie_out[phaseshift] = idx
        # Set the new partitioning
        self.elements.specie = specie_out

        # assign bed_critstress_factor to elements that entered based the sediment pool
        entered_bed = np.zeros(self.num_elements_active(), dtype=bool)

        if hasattr(self, 'num_srev'):
            entered_bed |= (specie_out == self.num_srev) & (specie_in != self.num_srev)
        if hasattr(self, 'num_ssrev'):
            entered_bed |= (specie_out == self.num_ssrev) & (specie_in != self.num_ssrev)
        if hasattr(self, 'num_sirrev'):
            entered_bed |= (specie_out == self.num_sirrev) & (specie_in != self.num_sirrev)
        if hasattr(self, 'num_sburied'):
            entered_bed |= (specie_out == self.num_sburied) & (specie_in != self.num_sburied)

        self._assign_bed_critstress_factor(np.flatnonzero(entered_bed))

        logger.debug('old species: %s' % specie_in[phaseshift])
        logger.debug('new species: %s' % specie_out[phaseshift])

        # Bookkeeping: count transitions iin -> iout among transformed elements
        for iin in range(self.nspecies):
            for iout in range(self.nspecies):
                self.ntransformations[iin, iout] += np.count_nonzero(
                    (specie_in[phaseshift] == iin) & (specie_out[phaseshift] == iout)
                )

        logger.debug('Number of transformations total:\n %s' % self.ntransformations )

        # Update Chemical properties after transformations
        self.update_chemical_diameter(specie_in, specie_out)
        self.update_chemical_fOC(specie_in, specie_out)
        self.sorption_to_sediments(specie_in, specie_out)
        self.desorption_from_sediments(specie_in, specie_out)

    def sorption_to_sediments(self, sp_in=None, sp_out=None):
        """
        Move newly sorbed dissolved elements onto the seabed.
        If an element transitions:
            dissolved -> sediment reversible
        then:
            z      <- -local_water_depth
            moving <- 0
        The final z field is clipped so no element ends above the sea surface.
        """

        # If sediment reversible compartment is not present, nothing to do
        if not hasattr(self, 'num_srev'):
            return

        # Set z to local sea depth for particles that have sorbed to sediments
        if self.get_config('chemical:species:LMM') and hasattr(self, 'num_lmm'):
            idx = np.flatnonzero((sp_out == self.num_srev) & (sp_in == self.num_lmm))
            if idx.size:
                depth = self._env_array('sea_floor_depth_below_sea_level', 10000.0, idx=idx)
                self.elements.z[idx] = -depth
                self.elements.moving[idx] = 0

        if self.get_config('chemical:species:LMMcation') and hasattr(self, 'num_lmmcation'):
            idx = np.flatnonzero((sp_out == self.num_srev) & (sp_in == self.num_lmmcation))
            if idx.size:
                depth = self._env_array('sea_floor_depth_below_sea_level', 10000.0, idx=idx)
                self.elements.z[idx] = -depth
                self.elements.moving[idx] = 0

        # avoid setting positive z values
        if np.nansum(self.elements.z > 0):
            logger.debug('Number of elements lowered down to sea surface: %s' % np.nansum(self.elements.z > 0))
        self.elements.z[self.elements.z > 0] = 0

    def desorption_from_sediments(self, sp_in=None, sp_out=None):
        """
        Move newly desorbed sediment elements back into the water column.
        If an element transitions:
            sediment reversible -> dissolved
        then:
            z      <- -local_water_depth + desorption_depth
            moving <- 1
        An optional Gaussian perturbation is added:
            z <- z + Normal(0, desorption_depth_uncert)
        The final z field is clipped so no element ends above the sea surface.
        """
        # If sediment reversible compartment is not present, nothing to do
        if not hasattr(self, 'num_srev'):
            return

        desorption_depth = self.get_config('chemical:sediment:desorption_depth')
        std = self.get_config('chemical:sediment:desorption_depth_uncert')

        if self.get_config('chemical:species:LMM') and hasattr(self, 'num_lmm'):
            idx = np.flatnonzero((sp_out == self.num_lmm) & (sp_in == self.num_srev))
            if idx.size:
                depth = self._env_array('sea_floor_depth_below_sea_level', 10000.0, idx=idx)
                self.elements.z[idx] = -depth + desorption_depth
                self.elements.moving[idx] = 1
                if std > 0:
                    logger.debug('Adding uncertainty for desorption from sediments: %s m' % std)
                    self.elements.z[idx] += np.random.normal(0, std, idx.size)
                    self.elements.z[idx] = np.maximum(self.elements.z[idx], -depth)
                    self.elements.z[idx] = np.minimum(self.elements.z[idx], 0.0)

        if self.get_config('chemical:species:LMMcation') and hasattr(self, 'num_lmmcation'):
            idx = np.flatnonzero((sp_out == self.num_lmmcation) & (sp_in == self.num_srev))
            if idx.size:
                depth = self._env_array('sea_floor_depth_below_sea_level', 10000.0, idx=idx)
                self.elements.z[idx] = -depth + desorption_depth
                self.elements.moving[idx] = 1
                if std > 0:
                    logger.debug('Adding uncertainty for desorption from sediments: %s m' % std)
                    self.elements.z[idx] += np.random.normal(0, std, idx.size)
                    self.elements.z[idx] = np.maximum(self.elements.z[idx], -depth)
                    self.elements.z[idx] = np.minimum(self.elements.z[idx], 0.0)

        # avoid setting positive z values
        if np.nansum(self.elements.z > 0):
            logger.debug('Number of elements lowered down to sea surface: %s' % np.nansum(self.elements.z > 0))
        self.elements.z[self.elements.z > 0] = 0


    def update_chemical_diameter(self, sp_in=None, sp_out=None):
        """
        Update particle diameter when an element changes species.
        1) Particle-family species:
               diameter ~ lognormal(median=particle_diameter, sigma=sigma_part_ln)
        2) Sediment-family species:
               diameter ~ lognormal(median=particle_diameter, sigma=sigma_part_ln)
        3) DOC-like carrier species:
               diameter ~ lognormal(median=doc_particle_diameter, sigma=sigma_doc_ln)
        4) Dissolved / colloid species:
               diameter = dissolved_diameter

        If Humic colloid -> Particle reversible,
        the DOC-derived carrier size is preserved by sampling from the DOC-size
        distribution rather than the particle-size distribution.

        Updated:
        Particle family:            prev, psrev, pirrev
        Sediment family:            srev, ssrev, sirrev, sburied
        Dissolved family:           lmm, lmmanion, lmmcation, colloid
        DOC-like family:            humic colloid, polymer
        """
        if sp_in is None or sp_out is None:
            return

        dia_part = float(self.get_config('chemical:particle_diameter'))
        dia_doc = float(self.get_config('chemical:doc_particle_diameter'))
        dia_diss = float(self.get_config('chemical:dissolved_diameter'))

        sigma_part_ln = float(self.get_config('chemical:particle_diameter_uncertainty'))
        sigma_doc_ln = float(self.get_config('chemical:doc_particle_diameter_uncertainty'))

        def _assign_lognormal(mask, median_diameter, sigma_ln, label):
            nmask = int(np.sum(mask))
            if nmask > 0:
                self.elements.diameter[mask] = self._sample_lognormal_diameter(
                    median_diameter=median_diameter,
                    sigma_ln=sigma_ln,
                    n=nmask)
                logger.debug('Updated %s diameter for %s elements', label, nmask)

        def _assign_constant(mask, value, label):
            nmask = int(np.sum(mask))
            if nmask > 0:
                self.elements.diameter[mask] = value
                logger.debug('Updated %s diameter for %s elements', label, nmask)

        # Particle-bound species
        if hasattr(self, 'num_prev'):
            mask = (sp_out == self.num_prev) & (sp_in != self.num_prev)
            # Preserve diameter of DOC aggregates
            if self.get_config('chemical:species:Humic_colloid') and hasattr(self, 'num_humcol'):
                mask_from_humic = mask & (sp_in == self.num_humcol)
                _assign_lognormal(mask_from_humic, dia_doc, sigma_doc_ln, 'particle-from-humic')

                mask_other = mask & (sp_in != self.num_humcol)
                _assign_lognormal(mask_other, dia_part, sigma_part_ln, 'particle')
            else:
                _assign_lognormal(mask, dia_part, sigma_part_ln, 'particle')

        if hasattr(self, 'num_psrev'):
            mask = (sp_out == self.num_psrev) & (sp_in != self.num_psrev)
            _assign_lognormal(mask, dia_part, sigma_part_ln, 'slowly reversible particle')

        if hasattr(self, 'num_pirrev'):
            mask = (sp_out == self.num_pirrev) & (sp_in != self.num_pirrev)
            _assign_lognormal(mask, dia_part, sigma_part_ln, 'irreversible particle')

        # Sediment-bound species
        if hasattr(self, 'num_srev'):
            mask = (sp_out == self.num_srev) & (sp_in != self.num_srev)
            _assign_lognormal(mask, dia_part, sigma_part_ln, 'sediment reversible')

        if hasattr(self, 'num_ssrev'):
            mask = (sp_out == self.num_ssrev) & (sp_in != self.num_ssrev)
            _assign_lognormal(mask, dia_part, sigma_part_ln, 'sediment slowly reversible')

        if hasattr(self, 'num_sirrev'):
            mask = (sp_out == self.num_sirrev) & (sp_in != self.num_sirrev)
            _assign_lognormal(mask, dia_part, sigma_part_ln, 'sediment irreversible')

        if hasattr(self, 'num_sburied'):
            mask = (sp_out == self.num_sburied) & (sp_in != self.num_sburied)
            _assign_lognormal(mask, dia_part, sigma_part_ln, 'sediment buried')

        # Dissolved species
        if self.get_config('chemical:species:LMM') and hasattr(self, 'num_lmm'):
            mask = (sp_out == self.num_lmm) & (sp_in != self.num_lmm)
            _assign_constant(mask, dia_diss, 'dissolved')

        if self.get_config('chemical:species:LMManion') and hasattr(self, 'num_lmmanion'):
            mask = (sp_out == self.num_lmmanion) & (sp_in != self.num_lmmanion)
            _assign_constant(mask, dia_diss, 'dissolved anion')

        if self.get_config('chemical:species:LMMcation') and hasattr(self, 'num_lmmcation'):
            mask = (sp_out == self.num_lmmcation) & (sp_in != self.num_lmmcation)
            _assign_constant(mask, dia_diss, 'dissolved cation')

        # Colloids / DOM-like species
        if self.get_config('chemical:species:Colloid') and hasattr(self, 'num_col'):
            mask = (sp_out == self.num_col) & (sp_in != self.num_col)
            _assign_constant(mask, dia_diss, 'colloid')

        if self.get_config('chemical:species:Humic_colloid') and hasattr(self, 'num_humcol'):
            mask = (sp_out == self.num_humcol) & (sp_in != self.num_humcol)
            _assign_lognormal(mask, dia_doc, sigma_doc_ln, 'humic colloid')

        if self.get_config('chemical:species:Polymer') and hasattr(self, 'num_polymer'):
            mask = (sp_out == self.num_polymer) & (sp_in != self.num_polymer)
            _assign_lognormal(mask, dia_doc, sigma_doc_ln, 'polymer')

    def _has_reader_variable(self, name):
        """
        Return True if an environment variable is actually provided by at least one reader.
        This checks the cached set of reader variables collected in prepare_run().
        """
        return name in getattr(self, '_reader_variables', set())

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
        val = getattr(self.environment, name, None)
        if val is None:
            val = fallback
        if val is None:
            return None

        arr = np.asarray(val, dtype=float)

        if idx is None:
            if arr.ndim == 0:
                return np.full(self.num_elements_active(), float(arr), dtype=float)
            return arr

        idx = np.asarray(idx, dtype=np.int64).ravel()
        if arr.ndim == 0:
            return np.full(idx.size, float(arr), dtype=float)
        return arr[idx]

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

    def _bottom_velocity_components(self, idx=None):
        """Return bottom-layer velocity components.
        This helper only accepts dedicated bottom-layer velocity fields:
            - x_bottom_sea_water_velocity
            - y_bottom_sea_water_velocity
        No fallback to depth-averaged or full-column velocity is used.
"""
        u_b = self._optional_env_array('x_bottom_sea_water_velocity', idx=idx)
        v_b = self._optional_env_array('y_bottom_sea_water_velocity', idx=idx)

        if u_b is None or v_b is None:
            return None, None

        return np.asarray(u_b, dtype=float), np.asarray(v_b, dtype=float)

    def _depth_averaged_velocity_components(self, idx=None):
        """Return depth-averaged velocity components for bulk friction laws.
        This helper reads:
            - x_depth_averaged_sea_water_velocity
            - y_depth_averaged_sea_water_velocity
    """
        u_da = self._optional_env_array('x_depth_averaged_sea_water_velocity', idx=idx)
        v_da = self._optional_env_array('y_depth_averaged_sea_water_velocity', idx=idx)

        if u_da is None or v_da is None:
            return None, None

        return np.asarray(u_da, dtype=float), np.asarray(v_da, dtype=float)

    def _hydraulic_radius_array(self, idx=None):
        """
        Return hydraulic radius for bulk friction laws.
        1) environment.hydraulic_radius
        2) config('chemical:sediment:bulk_hydraulic_radius') if > 0
        3) local water depth as fallback approximation
        All returned values are floored to a small positive epsilon.
        """
        eps = 1e-12

        Rh = self._env_array('hydraulic_radius', None, idx=idx)
        if Rh is not None:
            Rh = np.asarray(Rh, dtype=float)
            depth = np.asarray(
                self._env_array('sea_floor_depth_below_sea_level', 1.0, idx=idx),
                dtype=float)
            Rh = np.where(np.isfinite(Rh) & (Rh > 0.0), Rh, depth)
            return np.maximum(Rh, eps)

        Rh_cfg = float(self.get_config('chemical:sediment:bulk_hydraulic_radius'))
        if idx is None:
            n = self.num_elements_active()
        else:
            n = np.asarray(idx, dtype=np.int64).ravel().size

        if Rh_cfg > 0.0:
            return np.full(n, Rh_cfg, dtype=float)

        depth = np.asarray(
            self._env_array('sea_floor_depth_below_sea_level', 1.0, idx=idx),
            dtype=float)
        return np.maximum(depth, eps)

    def _bed_stress_array(self, name, idx=None):
        """
        Return a non-negative bed-stress magnitude field if available.
        1) Read the requested stress field only if it is provided by a reader
        2) Replace non-finite values with 0
        3) Clip negative values to 0

        Used for sea_floor_current_stress, sea_floor_wave_stress, sea_floor_other_stress
        """
        tau = self._optional_env_array(name, idx=idx)
        if tau is None:
            return None
        tau = np.asarray(tau, dtype=float)
        tau = np.where(np.isfinite(tau), tau, 0.0)
        return np.maximum(tau, 0.0)

    def _resuspension_branch(self):
        """
        Return the resuspension branch used by the timestep-dependent erosion model.
        1) If branch is explicitly configured as:
               'NONCOHESIVE' or 'COHESIVE'
           use it directly
        2) If branch == 'AUTO' and d50 is available:
               classify sediment from d50
               sand/gravel -> NONCOHESIVE
               clay/silt   -> COHESIVE
        3) If AUTO and no usable d50:
               erodibility_M > 0 -> COHESIVE
               otherwise         -> NONCOHESIVE
        """
        branch_cfg = self.get_config('chemical:sediment:resuspension_critstress_branch')
        if branch_cfg in ('NONCOHESIVE', 'COHESIVE'):
            return branch_cfg

        d50_m = float(self.get_config('chemical:sediment:d50'))
        if d50_m > 0.0:
            grain_class = self.classify_sediment(d50_m * 1e3)
            if grain_class in {'sand', 'gravel'}:
                return 'NONCOHESIVE'
            return 'COHESIVE'

        if float(self.get_config('chemical:sediment:erodibility_M')) > 0.0:
            return 'COHESIVE'

        return 'NONCOHESIVE'

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
            d50_m = float(self.get_config('chemical:sediment:d50'))
            if d50_m <= 0.0:
                raise ValueError('chemical:sediment:d50 must be > 0 when '
                                 'resuspension_critstress_mode == FROM_D50')

            branch_cfg = self.get_config('chemical:sediment:resuspension_critstress_branch')
            if branch_cfg == 'AUTO':
                grain_class = self.classify_sediment(d50_m * 1e3)
                branch = 'NONCOHESIVE' if grain_class in {'sand', 'gravel'} else 'COHESIVE'
            else:
                branch = branch_cfg

            if branch == 'NONCOHESIVE':
                method = self.get_config('chemical:sediment:resuspension_critstress_method')
                rho_s = float(self.get_config('chemical:sediment:critstress_rho_s'))
                nu = float(self.get_config('chemical:sediment:critstress_nu'))
                T = self._env_array('sea_water_temperature', 10.0, idx=idx)
                S = self._env_array('sea_water_salinity', 34.0, idx=idx)
                rho_w = self.sea_water_density(T=T, S=S)

                if method == 'soulsby_whitehouse':
                    theta = self.theta_cr_soulsby_whitehouse(
                        d50_m, rho_s=rho_s, rho_w=rho_w, nu=nu, g=9.81)
                elif method == 'van_rijn':
                    theta = self.theta_cr_van_rijn(
                        d50_m, rho_s=rho_s, rho_w=rho_w, nu=nu, g=9.81)
                elif method in {'laursen', 'mpm', 'wu'}:
                    theta = np.full(n, self.theta_cr_constant(method), dtype=float)
                else:
                    raise ValueError(f'Unknown noncohesive method: {method!r}')

                tau_cr = np.maximum(theta * (rho_s - rho_w) * 9.81 * d50_m, 0.0)

            elif branch == 'COHESIVE':
                rho_d = float(self.get_config('chemical:sediment:critstress_owen_rho_d'))
                a = float(self.get_config('chemical:sediment:critstress_owen_a'))
                b = float(self.get_config('chemical:sediment:critstress_owen_b'))
                if rho_d <= 0 or a <= 0 or b <= 0:
                    raise ValueError('Cohesive resuspension_critstress requires positive '
                        'critstress_owen_rho_d, critstress_owen_a, and critstress_owen_b')
                tau0 = self.tau_ce_owen(rho_d=rho_d, a=a, b=b)
                tau_cr = np.full(n, tau0, dtype=float)

            else:
                raise ValueError(f'Unknown resuspension_critstress_branch: {branch!r}')

        else:
            raise ValueError(f'Unknown resuspension_critstress_mode: {mode!r}')

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
        p_dep = 1.0 - np.exp(-k_dep * dt)
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

        branch = self._resuspension_branch()

        if branch == 'COHESIVE':
            M = float(self.get_config('chemical:sediment:erodibility_M'))  # kg m-2 s-1 Pa-1

            if M <= 0.0:
                return np.zeros_like(tau, dtype=float)

            m_erodible = self._local_erodible_mass_per_area(idx=idx)

            if np.all(m_erodible <= 0.0):
                return np.zeros_like(tau, dtype=float)

            erosion_flux = M * np.maximum(tau - tau_cr_res, 0.0)   # kg m-2 s-1
            hazard = erosion_flux / np.maximum(m_erodible, 1e-30)  # 1/s

        elif branch == 'NONCOHESIVE':
            T_pickup = float(self.get_config('chemical:sediment:noncohesive_resuspension_timescale'))
            expo = float(self.get_config('chemical:sediment:noncohesive_excess_shear_exponent'))

            T_pickup = max(T_pickup, 1e-12)
            expo = max(expo, 1e-12)

            excess = np.maximum(tau / np.maximum(tau_cr_res, 1e-30) - 1.0, 0.0)
            hazard = (excess ** expo) / T_pickup                   # 1/s

        else:
            raise ValueError(f'Unknown resuspension branch: {branch!r}')

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

    def _bearing_to_unit_vector(self, bearing_deg):
        """
        Convert a geographic bearing (degrees clockwise from north, 'to' direction)
        to unit-vector components (x eastward, y northward).
        """
        theta = np.deg2rad(np.asarray(bearing_deg, dtype=float))
        ex = np.sin(theta)
        ey = np.cos(theta)
        return ex, ey

    def _wave_stress_from_orbital(self, rho, z0, idx=None):
        """
        Wave-only bed-stress amplitude tau_w [Pa].

        Priority:
          1) use sea_floor_wave_stress directly if available
          2) otherwise estimate from near-bed orbital velocity amplitude and wave period:
               A   = Uw * T / (2*pi)
               fwr = 1.39 * (A/z0)^(-0.52)   # rough turbulent
               tau_w = 0.5 * rho * fwr * Uw^2
        """
        eps = 1e-12

        tau_wave = self._bed_stress_array('sea_floor_wave_stress', idx=idx)
        if tau_wave is not None:
            return tau_wave

        Uw = self._optional_env_array('sea_floor_wave_orbital_velocity', idx=idx)
        T = self._optional_env_array('sea_surface_wave_period_at_variance_spectral_density_maximum', idx=idx)

        if Uw is None or T is None:
            return None

        Uw = np.abs(np.asarray(Uw, dtype=float))
        T = np.maximum(np.asarray(T, dtype=float), eps)
        z0 = np.maximum(np.asarray(z0, dtype=float), eps)

        A = Uw * T / (2.0 * np.pi)
        rel_rough = np.maximum(A / z0, 1.000001)

        fwr = 1.39 * rel_rough ** (-0.52)
        fwr = np.maximum(fwr, 0.0)

        tau_wave = 0.5 * rho * fwr * Uw**2
        return np.maximum(tau_wave, 0.0)

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

        Combination options:
          - sum / max / rss: heuristic scalar combinations
          - SOULSBY_CLARKE: current-wave combination with
                tau_m   = tau_c * (1 + 1.2 * (tau_w/(tau_c + tau_w))**3.2)
                tau_max = | tau_m * e_current + tau_w * e_wave |
            where tau_w is wave-alone bed-stress amplitude and e_wave is the wave
            propagation unit vector from sea_surface_wave_to_direction.

        For 'sum', 'max', and 'rss', tau_effective is a scalar heuristic combination.
        The returned tau_effective_x/y are assigned along the current-stress direction
        (or wave direction if current stress is zero), so that the vector components are
        internally consistent with tau_effective.

        'other' stress has no directional information; preserve current combined direction
        when possible, and only inflate the magnitude heuristically.
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
            }

        kappa = 0.41
        g = 9.81
        eps = 1e-12

        T = self._env_array('sea_water_temperature', 10.0, idx=idx)
        S = self._env_array('sea_water_salinity', 34.0, idx=idx)
        rho = self.sea_water_density(T=T, S=S)

        dz_env = self._env_array('bottom_layer_thickness', 0.0, idx=idx)
        # z0_env = self._env_array('sea_floor_roughness_length', 0.0, idx=idx)

        Param_mode = self.get_config('chemical:sediment:stress_param_mode')
        dz_cfg = float(self.get_config('chemical:sediment:bottom_layer_thickness'))
        n_cfg = float(self.get_config('chemical:sediment:manning_n'))
        C_cfg = float(self.get_config('chemical:sediment:chezy_C'))
        ks_cfg = float(self.get_config('chemical:sediment:nikuradse_ks'))
        d50_cfg = float(self.get_config('chemical:sediment:d50'))
        cd_min = float(self.get_config('chemical:sediment:cd_min'))
        cd_max = float(self.get_config('chemical:sediment:cd_max'))

        dz_bot = np.where(np.asarray(dz_env, dtype=float) > 0.0, dz_env, dz_cfg)
        dz_bot = np.maximum(np.asarray(dz_bot, dtype=float), eps)

        z0_default = None

        # Preferred path: direct hydro-model current bed stress
        tau_c = self._bed_stress_array('sea_floor_current_stress', idx=idx)

        if tau_c is not None:
            # Use bottom-layer direction if available, otherwise depth-averaged direction.
            u_ref, v_ref = self._bottom_velocity_components(idx=idx)
            if u_ref is None or v_ref is None:
                u_ref, v_ref = self._depth_averaged_velocity_components(idx=idx)

            if u_ref is None or v_ref is None:
                u_ref = np.zeros(n, dtype=float)
                v_ref = np.zeros(n, dtype=float)

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
                    z0 = np.full(n, max(d50_cfg / 12.0, eps), dtype=float)
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
                Rh = self._hydraulic_radius_array(idx=idx)

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

        # Current direction
        ec_x = np.zeros(n, dtype=float)
        ec_y = np.zeros(n, dtype=float)
        has_current = tau_c > eps
        ec_x[has_current] = tau_bx[has_current] / tau_c[has_current]
        ec_y[has_current] = tau_by[has_current] / tau_c[has_current]

        # Wave magnitude and direction, if requested
        tau_wave = None
        ew_x = np.zeros(n, dtype=float)
        ew_y = np.zeros(n, dtype=float)
        has_wave_dir = np.zeros(n, dtype=bool)

        if use_wave:
            z0_wave = z0_default
            if z0_wave is None:
                if Param_mode == 'WHITE_COLEBROOK':
                    z0_wave = np.full(n, max(ks_cfg / 30.0, eps), dtype=float)
                elif Param_mode == 'GRAIN_D50':
                    z0_wave = np.full(n, max(d50_cfg / 12.0, eps), dtype=float)
                else:
                    z0_wave = self._bottom_roughness_length_array(idx=idx)

            tau_wave = self._wave_stress_from_orbital(rho=rho, z0=z0_wave, idx=idx)

            wave_to_dir = self._optional_env_array('sea_surface_wave_to_direction', idx=idx)
            if wave_to_dir is not None:
                ew_x, ew_y = self._bearing_to_unit_vector(wave_to_dir)
                has_wave_dir[:] = True

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
                need_wave_dir = tau_wave > eps
                if np.any(need_wave_dir & (~has_wave_dir)):
                    raise ValueError(
                        "SOULSBY_CLARKE requires sea_surface_wave_to_direction for all elements "
                        "where wave stress is used.")

                denom = np.maximum(tau_c + tau_wave, eps)
                tau_m = tau_c * (1.0 + 1.2 * (tau_wave / denom) ** 3.2)

                tau_eff_x = tau_m * ec_x + tau_wave * ew_x
                tau_eff_y = tau_m * ec_y + tau_wave * ew_y
                tau_eff = np.hypot(tau_eff_x, tau_eff_y)

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

        ustar_eff = np.sqrt(np.maximum(tau_eff, 0.0) / np.maximum(rho, eps))

        return {
            'tau_bx': tau_bx,                  # current-only x-component
            'tau_by': tau_by,                  # current-only y-component
            'tau_current': tau_c,              # current-only magnitude
            'tau_wave': tau_wave if tau_wave is not None else np.zeros(n, dtype=float),
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

    def deposition(self, specie0, z0, specie_new, z_new, moving_new, crit_new, dt):
        """
        Helper for bed_exchange().
        Decide suspended -> bed deposition using only the start-of-timestep state.
        Inputs:
          - specie0, z0: frozen initial species and depth arrays
          - specie_new, z_new, moving_new, crit_new: output arrays modified in place
          - dt: timestep [s]
        Eligible initial suspended species:
          - Particle reversible           -> Sediment reversible
          - Particle slowly reversible    -> Sediment slowly reversible
          - Particle irreversible         -> Sediment irreversible

        Only initially suspended elements inside the local near-bed interaction layer
        are considered. Deposition decisions are based only on the initial state, so
        elements newly resuspended later in the same timestep are not eligible here.
        """
        if not self.get_config('chemical:sediment:enable_deposition'):
            return

        has_rev = hasattr(self, 'num_prev') and hasattr(self, 'num_srev')
        has_slow = hasattr(self, 'num_psrev') and hasattr(self, 'num_ssrev')
        has_irrev = hasattr(self, 'num_pirrev') and hasattr(self, 'num_sirrev')
        if not (has_rev or has_slow or has_irrev):
            return

        susp_mask0 = np.zeros(self.num_elements_active(), dtype=bool)
        if has_rev:
            susp_mask0 |= (specie0 == self.num_prev)
        if has_slow:
            susp_mask0 |= (specie0 == self.num_psrev)
        if has_irrev:
            susp_mask0 |= (specie0 == self.num_pirrev)

        if not np.any(susp_mask0):
            return

        idx_susp = np.flatnonzero(susp_mask0)

        z_susp = np.asarray(z0[idx_susp], dtype=float)
        Zmin_susp = -self._env_array('sea_floor_depth_below_sea_level', 10000.0, idx=idx_susp)

        h_env = self._env_array('interaction_sediment_layer_thickness', 0.0, idx=idx_susp)
        h_cfg = float(self.get_config('chemical:sediment:bed_interaction_thickness'))
        h_b_susp = np.where(h_env > 0.0, h_env, h_cfg)

        dist_above_bed = z_susp - Zmin_susp
        near_bed_local = (dist_above_bed >= 0.0) & (dist_above_bed <= h_b_susp)

        if not np.any(near_bed_local):
            return

        idx_dep = idx_susp[near_bed_local]
        Zmin_dep = Zmin_susp[near_bed_local]
        h_b_dep = h_b_susp[near_bed_local]

        stress_dep = self.compute_bottom_shear_stress(idx=idx_dep)
        tau_dep = stress_dep['tau_effective']

        ws_dep = np.maximum(-np.asarray(self.elements.terminal_velocity[idx_dep], dtype=float), 0.0)
        p_dep = self.deposition_probability(tau=tau_dep, ws=ws_dep, dt=dt, h_b=h_b_dep)

        hit_dep_local = np.random.rand(idx_dep.size) < p_dep
        if not np.any(hit_dep_local):
            return

        ii = idx_dep[hit_dep_local]
        Zhit = Zmin_dep[hit_dep_local]

        if has_rev:
            loc = (specie0[ii] == self.num_prev)
            if np.any(loc):
                jj = ii[loc]
                specie_new[jj] = self.num_srev
                z_new[jj] = Zhit[loc]
                moving_new[jj] = 0
                if self.get_config('chemical:sediment:use_critstress_heterogeneity'):
                    crit_new[jj] = self._sample_critstress_factor(jj.size)
                else:
                    crit_new[jj] = 1.0
                self.ntransformations[self.num_prev, self.num_srev] += jj.size

        if has_slow:
            loc = (specie0[ii] == self.num_psrev)
            if np.any(loc):
                jj = ii[loc]
                specie_new[jj] = self.num_ssrev
                z_new[jj] = Zhit[loc]
                moving_new[jj] = 0
                if self.get_config('chemical:sediment:use_critstress_heterogeneity'):
                    crit_new[jj] = self._sample_critstress_factor(jj.size)
                else:
                    crit_new[jj] = 1.0
                self.ntransformations[self.num_psrev, self.num_ssrev] += jj.size

        if has_irrev:
            loc = (specie0[ii] == self.num_pirrev)
            if np.any(loc):
                jj = ii[loc]
                specie_new[jj] = self.num_sirrev
                z_new[jj] = Zhit[loc]
                moving_new[jj] = 0
                if self.get_config('chemical:sediment:use_critstress_heterogeneity'):
                    crit_new[jj] = self._sample_critstress_factor(jj.size)
                else:
                    crit_new[jj] = 1.0
                self.ntransformations[self.num_pirrev, self.num_sirrev] += jj.size

    def resuspension(self, specie0, z0, specie_new, z_new, moving_new, crit_new, dt):
        """
        Helper for bed_exchange().
        Decide bed -> suspended resuspension using only the start-of-timestep state.
        Inputs:
          - specie0, z0: frozen initial species and depth arrays
          - specie_new, z_new, moving_new, crit_new: output arrays modified in place
          - dt: timestep [s]
        Eligible initial bed species:
          - Sediment reversible           -> Particle reversible
          - Sediment slowly reversible    -> Particle slowly reversible
          - Sediment irreversible         -> Particle irreversible
        Only initially bed-bound elements are considered. Resuspension decisions are
        based only on the initial state, so elements newly deposited earlier in the
        same timestep are not eligible here.
        """
        if not self.get_config('chemical:sediment:enable_resuspension'):
            return

        has_rev = hasattr(self, 'num_prev') and hasattr(self, 'num_srev')
        has_slow = hasattr(self, 'num_psrev') and hasattr(self, 'num_ssrev')
        has_irrev = hasattr(self, 'num_pirrev') and hasattr(self, 'num_sirrev')
        if not (has_rev or has_slow or has_irrev):
            return

        bed_mask0 = np.zeros(self.num_elements_active(), dtype=bool)
        if has_rev:
            bed_mask0 |= (specie0 == self.num_srev)
        if has_slow:
            bed_mask0 |= (specie0 == self.num_ssrev)
        if has_irrev:
            bed_mask0 |= (specie0 == self.num_sirrev)

        if not np.any(bed_mask0):
            return

        idx_bed = np.flatnonzero(bed_mask0)

        z_bed = np.asarray(z0[idx_bed], dtype=float)
        Zmin_bed = -self._env_array('sea_floor_depth_below_sea_level', 10000.0, idx=idx_bed)
        on_bed_local = z_bed <= Zmin_bed

        if not np.any(on_bed_local):
            return

        idx_res = idx_bed[on_bed_local]
        Zmin_res = Zmin_bed[on_bed_local]

        stress_res = self.compute_bottom_shear_stress(idx=idx_res)
        tau_res = stress_res['tau_effective']
        tau_cr_res = self.get_resuspension_critstress(idx=idx_res)

        p_res = self.resuspension_probability(
            tau=tau_res,
            tau_cr_res=tau_cr_res,
            dt=dt,
            idx=idx_res,)

        hit_res_local = np.random.rand(idx_res.size) < p_res
        if not np.any(hit_res_local):
            return

        ii = idx_res[hit_res_local]
        Zhit = Zmin_res[hit_res_local]

        resusp_depth = float(self.get_config('chemical:sediment:resuspension_depth'))
        std = float(self.get_config('chemical:sediment:resuspension_depth_uncert'))

        if has_rev:
            loc = (specie0[ii] == self.num_srev)
            if np.any(loc):
                jj = ii[loc]
                specie_new[jj] = self.num_prev
                moving_new[jj] = 1
                z_new[jj] = Zhit[loc] + resusp_depth
                crit_new[jj] = 1.0
                self.ntransformations[self.num_srev, self.num_prev] += jj.size

        if has_slow:
            loc = (specie0[ii] == self.num_ssrev)
            if np.any(loc):
                jj = ii[loc]
                specie_new[jj] = self.num_psrev
                moving_new[jj] = 1
                z_new[jj] = Zhit[loc] + resusp_depth
                crit_new[jj] = 1.0
                self.ntransformations[self.num_ssrev, self.num_psrev] += jj.size

        if has_irrev:
            loc = (specie0[ii] == self.num_sirrev)
            if np.any(loc):
                jj = ii[loc]
                specie_new[jj] = self.num_pirrev
                moving_new[jj] = 1
                z_new[jj] = Zhit[loc] + resusp_depth
                crit_new[jj] = 1.0
                self.ntransformations[self.num_sirrev, self.num_pirrev] += jj.size

        if std > 0.0:
            z_new[ii] += np.random.normal(0.0, std, ii.size)
            z_new[ii] = np.maximum(z_new[ii], Zhit)
            z_new[ii] = np.minimum(z_new[ii], 0.0)

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
        """
        do_dep = self.get_config('chemical:sediment:enable_deposition')
        do_res = self.get_config('chemical:sediment:enable_resuspension')
        if not (do_dep or do_res):
            return

        has_rev = hasattr(self, 'num_prev') and hasattr(self, 'num_srev')
        has_slow = hasattr(self, 'num_psrev') and hasattr(self, 'num_ssrev')
        has_irrev = hasattr(self, 'num_pirrev') and hasattr(self, 'num_sirrev')
        if not (has_rev or has_slow or has_irrev):
            return

        dt = self.time_step.total_seconds()

        # Snapshot start-of-step state
        specie0 = np.asarray(self.elements.specie, dtype=int).copy()
        z0 = np.asarray(self.elements.z, dtype=float).copy()
        moving0 = np.asarray(self.elements.moving).copy()
        crit0 = np.asarray(self.elements.critstress_factor, dtype=float).copy()

        # Working output arrays
        specie_new = specie0.copy()
        z_new = z0.copy()
        moving_new = moving0.copy()
        crit_new = crit0.copy()

        # Both helpers use the same snapshot
        self.deposition(specie0=specie0, z0=z0,
            specie_new=specie_new, z_new=z_new,
            moving_new=moving_new, crit_new=crit_new,
            dt=dt,)

        self.resuspension(specie0=specie0,
            z0=z0, specie_new=specie_new,
            z_new=z_new, moving_new=moving_new,
            crit_new=crit_new,
            dt=dt,)

        if hasattr(self, 'num_sburied'):
            buried = (specie_new == self.num_sburied)
            moving_new[buried] = 0

        z_new[z_new > 0.0] = 0.0
        # Update element properties
        self.elements.specie = specie_new
        self.elements.z = z_new
        self.elements.moving = moving_new
        self.elements.critstress_factor = crit_new
        self.update_chemical_diameter(specie0, specie_new)

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
                        self.elements.mass[W] * ( 1 - np.exp(-k_W_fin * self.time_step.total_seconds())))
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
                        self.elements.mass[S] * (1 - np.exp(-k_S_fin * self.time_step.total_seconds()))
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

                if k_Photo == 0:
                    logger.debug("k_Photo is set to 0 1/h, therefore no photodegradation occurs")
                if k_DecayMax_water == 0:
                    logger.debug("k_DecayMax_water is set to 0 1/h, therefore DOCorr = 0 and no biodegradation occurs")
                if k_Anaerobic_water == 0:
                    logger.debug("k_Anaerobic_water is set to 0 1/h, therefore no biodegradation occurs without oxygen")

                if W_deg or S_deg:
                    Tref_kWt = self.get_config('chemical:transformations:Tref_kWt')
                    DH_kWt = self.get_config('chemical:transformations:DeltaH_kWt')
                    Tref_kSt = self.get_config('chemical:transformations:Tref_kSt')
                    DH_kSt = self.get_config('chemical:transformations:DeltaH_kSt')

                    if Bio_degr is True and k_DecayMax_water > 0:
                        HalfSatO_w = self.get_config('chemical:transformations:HalfSatO_w')
                        T_Max_bio = self.get_config('chemical:transformations:T_Max_bio')
                        T_Opt_bio = self.get_config('chemical:transformations:T_Opt_bio')
                        T_Adp_bio = self.get_config('chemical:transformations:T_Adp_bio')
                        Max_Accl_bio = self.get_config('chemical:transformations:Max_Accl_bio')
                        Dec_Accl_bio = self.get_config('chemical:transformations:Dec_Accl_bio')
                        Q10_bio = self.get_config('chemical:transformations:Q10_bio')
                        pH_min_bio = self.get_config('chemical:transformations:pH_min_bio')
                        pH_max_bio = self.get_config('chemical:transformations:pH_max_bio')
                        Ox_water = (self._env_array('mole_concentration_of_dissolved_molecular_oxygen_in_sea_water',
                            7.25, idx=idx_W) if W_deg else np.empty(0, dtype=float)) * 31.9988e-3 # from mmol/m3 to to g/m3

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
                        # Depth of element (m)
                        Depth = -self._z_array(idx_W)  # positive depth
                        # Concentration of DOC (mmol[C]/Kg)
                        concDOC = self._doc_mmolkg(idx_W)

                    if (Bio_degr is True and k_DecayMax_water > 0) or Hydro_degr is True:
                        # pH water
                        pH_water = self._env_array('sea_water_ph_reported_on_total_scale',
                            8.1, idx=idx_W)

                        if np.any(pH_water == 0):
                            pH_water[pH_water == 0] = np.median(pH_water)
                            logger.debug("pH_water in degradation was 0, set to median value")

                    # Calculate correction factors for degradation rates
                    if Bio_degr is True and k_DecayMax_water > 0:
                        k_W_bio = k_DecayMax_water * self.calc_DOCorr(
                            HalfSatO_w, k_Anaerobic_water, k_DecayMax_water, Ox_water)

                        k_W_bio = k_W_bio * self.calc_pHCorr(pH_min_bio, pH_max_bio, pH_water)
                        k_W_bio = k_W_bio * self.calc_TCorr(
                            T_Max_bio, T_Opt_bio, T_Adp_bio, Max_Accl_bio,
                            Dec_Accl_bio, Q10_bio, TW)

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
                            self.elements.mass[W] * (1 - np.exp(-k_W_fin * self.time_step.total_seconds())))
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

                    if (Bio_degr is True and k_DecayMax_water > 0) or Hydro_degr is True:
                        # pH sediments
                        pH_sed = self._env_array('pH_sediment', 6.9, idx=idx_S)
                        if np.any(pH_sed == 0):
                            pH_sed[pH_sed == 0] = np.median(pH_sed)
                            logger.debug("pH_sed in degradation was 0, set to median value")

                    if Bio_degr is True and k_DecayMax_water > 0:
                        # From AQUATOX: k_DecayMax_water is a rate (1/h), and k_S_bio is four times slower than k_DecayMax_water
                        k_S_bio = self.get_config('chemical:transformations:k_DecayMax_water') / 4
                        k_S_bio = k_S_bio * self.calc_pHCorr(pH_min_bio, pH_max_bio, pH_sed)
                        k_S_bio = k_S_bio * self.tempcorr("Arrhenius", DH_kSt, TS, Tref_kSt)

                        # Apply slower degradation to buried sediments due to anoxic conditions
                        ssrev_slow_deg = self.get_config('chemical:transformations:ssrev_slow_deg_factor')
                        if ssrev_slow_deg < 1:
                            if ssrev_slow_deg < 0:
                                ssrev_slow_deg = 0.0
                            # Apply the slowdown to the buried sediment compartment
                            if hasattr(self, 'num_sburied'):
                                S_is_buried = (self.elements.specie[S] == self.num_sburied)
                                k_S_bio[S_is_buried] *= ssrev_slow_deg
                            else:
                                # Backward compatibility (older setups used ssrev as buried)
                                # S_is_buried = (self.elements.specie[S] == self.num_ssrev)
                                # k_S_bio[S_is_buried] *= ssrev_slow_deg
                                pass
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
                            self.elements.mass[S] * (1 - np.exp(-k_S_fin * self.time_step.total_seconds())))
                else:
                    k_S_bio = 0
                    k_S_hydro = 0
                    k_S_fin = 0
                    k_S_fin_sum = 0

                # Save single-mechanism masses
                if self.get_config('chemical:transformations:Save_single_degr_mass') is True:
                    k_W_photo_fraction = 0
                    k_W_bio_fraction = 0
                    k_S_bio_fraction = 0
                    k_W_hydro_fraction = 0
                    k_S_hydro_fraction = 0

                    if Photo_degr is True and k_Photo > 0:
                        if np.sum(k_W_photo) > 0:
                            photo_degraded_now = np.zeros(self.num_elements_active())
                            k_W_photo_fraction = np.minimum(
                                (k_W_photo / 3600) / np.maximum(k_W_fin, 1e-12), # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation
                                1.0
                            )
                            photo_degraded_now[W] = degraded_now[W] * k_W_photo_fraction

                            if W_deg:
                                self.elements.mass_photodegraded[W] += photo_degraded_now[W]

                    if Bio_degr is True and k_DecayMax_water > 0:
                        if np.sum(k_W_bio) > 0 or np.sum(k_S_bio) > 0:
                            bio_degraded_now = np.zeros(self.num_elements_active())
                            if np.sum(k_W_bio) > 0:
                                k_W_bio_fraction = np.minimum((k_W_bio / 3600) / np.maximum(k_W_fin, 1e-12),
                                    1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation

                                bio_degraded_now[W] = degraded_now[W] * k_W_bio_fraction
                            if np.sum(k_S_bio) > 0:
                                k_S_bio_fraction = np.minimum(
                                    (k_S_bio / 3600) / np.maximum(k_S_fin, 1e-12),
                                    1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation

                                bio_degraded_now[S] = degraded_now[S] * k_S_bio_fraction

                            if W_deg:
                                self.elements.mass_biodegraded[W] += bio_degraded_now[W]
                                self.elements.mass_biodegraded_water[W] += bio_degraded_now[W]
                            if S_deg:
                                self.elements.mass_biodegraded[S] += bio_degraded_now[S]
                                self.elements.mass_biodegraded_sediment[S] += bio_degraded_now[S]

                    if Hydro_degr is True:
                        if np.sum(k_W_hydro) > 0 or np.sum(k_S_hydro) > 0:
                            hydro_degraded_now = np.zeros(self.num_elements_active())
                            if np.sum(k_W_hydro) > 0:
                                k_W_hydro_fraction = np.minimum(
                                    (k_W_hydro / 3600) / np.maximum(k_W_fin, 1e-12),
                                    1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation

                                hydro_degraded_now[W] = degraded_now[W] * k_W_hydro_fraction
                            if np.sum(k_S_hydro) > 0:
                                k_S_hydro_fraction = np.minimum(
                                    (k_S_hydro / 3600) / np.maximum(k_S_fin, 1e-12),
                                    1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation

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
                   amphoter  -> neutral fraction only
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
        if pKa_acid < 0 and diss in ['amphoter', 'acid']:
            raise ValueError("pKa_acid must be positive")

        pKa_base = self.get_config('chemical:transformations:pKa_base')
        if pKa_base < 0 and diss in ['amphoter', 'base']:
            raise ValueError("pKa_base must be positive")

        if diss == 'amphoter' and abs(pKa_acid - pKa_base) < 2:
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
        elif diss == 'amphoter':
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

    def update(self):
        """Update positions and properties of Chemical particles."""

        # Workaround due to conversion of datatype
        self.elements.specie = self.elements.specie.astype(np.int32)

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

        logger.info('partitioning: {} {}'.format([sum(self.elements.specie==ii) for ii in range(self.nspecies)], self.name_species))

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



##################
# POSTPROCESSING
    #
    def simulation_summary(self, chemical_compound):
        '''Print a summary of the simulation: number of elements, number of transformations
        and final speciation
        '''

        print(chemical_compound)

        if hasattr(self,'name_species'):
            print('Final speciation:')
            for isp,sp in enumerate(self.name_species):
                print ('{:32}: {:>6}'.format(sp,sum(self.elements.specie==isp)))

        if hasattr(self,'ntransformations'):
            print('Number of transformations:')
            for isp in range(self.nspecies):
                print('{}'.format(['{:>9}'.format(np.int32(item)) for item in self.ntransformations[isp,:]]))

        base_attrs = [
            "mass", "mass_degraded", "mass_degraded_water", "mass_degraded_sediment",
            "mass_volatilized", "mass_photodegraded", "mass_biodegraded",
            "mass_biodegraded_water", "mass_biodegraded_sediment", "mass_hydrolyzed",
            "mass_hydrolyzed_water", "mass_hydrolyzed_sediment"
            ]

        mass_values = {
            attr: (sum(np.nan_to_num(getattr(self.elements, attr, []), nan = 0))
            # + sum(np.nan_to_num(getattr(self.elements_deactivated, attr, []), nan = 0))
            )
            for attr in base_attrs
        }

        # Ensure missing attributes default to zero
        m_pre = mass_values["mass"]
        m_deg = mass_values["mass_degraded"]
        m_deg_w = mass_values["mass_degraded_water"]
        m_deg_s = mass_values["mass_degraded_sediment"]
        m_vol = mass_values["mass_volatilized"]
        m_photo = mass_values.get("mass_photodegraded", 0)
        m_bio = mass_values.get("mass_biodegraded", 0)
        m_bio_w = mass_values.get("mass_biodegraded_water", 0)
        m_bio_s = mass_values.get("mass_biodegraded_sediment", 0)
        m_hydro = mass_values.get("mass_hydrolyzed", 0)
        m_hydro_w = mass_values.get("mass_hydrolyzed_water", 0)
        m_hydro_s = mass_values.get("mass_hydrolyzed_sediment", 0)
        m_tot = m_pre + m_deg + m_vol

        print("Mass balance:")
        print(f"mass total                : {m_tot * 1e-6:.3e} g   ")

        mass_components = {
            "mass preserved            ": (m_pre, m_tot),
            "mass degraded             ": (m_deg, m_tot),
            "     degr in water        ": (m_deg_w, m_deg),
            "     degr in sediments    ": (m_deg_s, m_deg),
            "mass volatilized          ": (m_vol, m_tot),
            "mass photodegraded        ": (m_photo, m_tot),
            "mass biodegraded          ": (m_bio, m_tot),
            "     biodegr in water     ": (m_bio_w, m_bio if m_bio > 0 else 1),
            "     biodegr in sediments ": (m_bio_s, m_bio if m_bio > 0 else 1),
            "mass hydrolyzed           ": (m_hydro, m_tot),
            "     hydr in water        ": (m_hydro_w, m_hydro if m_hydro > 0 else 1),  # Avoid division by zero
            "     hydr in sediments    ": (m_hydro_s, m_hydro if m_hydro > 0 else 1)   # Avoid division by zero
            }

        for label, (value, divisor) in mass_components.items():
            if value > 0:  # Only print if nonzero
                if not any(keyword in label for keyword in ['hydrolyzed', 'biodegraded', 'photodegraded', 'volatilized', 'degraded', 'preserved']):
                    print(f"{label}: {value * 1e-6:.3e} g   {value / m_tot * 100:.2f} % of m_tot  ({value / divisor * 100:.2f} %)")
                else:
                    print(f"{label}: {value * 1e-6:.3e} g   {value / m_tot * 100:.2f} % of m_tot")

    def write_netcdf_chemical_density_map(self, filename, pixelsize_m='auto', zlevels=None,
                                          lat_resol=None, lon_resol=None,
                                          deltat=None,
                                          density_proj=None,
                                          llcrnrlon=None, llcrnrlat=None,
                                          urcrnrlon=None, urcrnrlat=None,
                                          mass_unit=None,
                                          time_avg_conc=False,
                                          time_start=None,
                                          time_end=None,
                                          time_chunk_size=50,
                                          horizontal_smoothing=False,
                                          smoothing_cells=0,
                                          reader_sea_depth=None,
                                          reader_active_sediment_layer_thickness=None,
                                          landmask_shapefile=None,
                                          landmask_bathymetry_thr=None,
                                          origin_marker=None,
                                          elements_density=False,
                                          active_status=False,
                                          weight=None,
                                          sim_description=None,
                                          timestep_values=False,
                                          compress_species=False,
                                          weight_mode="extensive",
                                          weight_unit=None):
        """
        Arguments:
            pixelsize_m:           float32, lenght of gridcells in m (default mode)
            lat_resol:             float32, latitude resolution of gricells (in degrees using EPSG 4326)
            lon_resol:             float32, longitude resolution of gricells (in degrees using EPSG 4326)
            zlevels:               list of float32, depth levels at which concentration will be calculated
                           Values must be negative and ordered from the lowest depth (e.g. [-50., -10., -5.])
                           In the .nc file "depth" value will indicate the start of the vertical slice
                           i.e. "depth = 0" indicates slice from 0 to 5 m, and "depth = 50" indicates
                           slice from 50m to bathimietry
            density_proj:          None: add default projection with equal-area property (proj=moll +ellps=WGS84 +lon_0=0.0')
                                   <proj4_string>: <longlat +datum=WGS84 +no_defs> for EPSG 4326
                                   int, 4326 to indicate use of EPSG 4326
            llcrnrlon:             float32, min longitude of grid (in degrees using EPSG 4326)
            llcrnrlat:             float32, min latitude of grid (in degrees using EPSG 4326)
            urcrnrlon:             float32, max longitude of grid (in degrees using EPSG 4326)
            urcrnrlat:             float32, max latitude of grid (in degrees using EPSG 4326)
            mass_unit:             string, mass unit of input, used for output concentration label(ug/mg/g/kg)
            time_avg_conc:         boolean, calculate concentration averaged each deltat
            time_start:            string, datetime64[ns] string for start time of concentration map
            time_end:              string, datetime64[ns] string for end time of concentration map
            time_chunk_size        int, number of timesteps computed per chunk
            horizontal_smoothing:  boolean, smooth concentration horizontally
            smoothing_cells:       int, number of cells for horizontal smoothing,
            reader_sea_depth:      string, path of bathimethy .nc file
            reader_active_sediment_layer_thicknes:
                                   string, path of .nc file containing variable
                                   'active_sediment_layer_thickness'; if not provided,
                                   chemical:sediment:mixing_depth is used
            landmask_shapefile:    string, path of bathimethylandmask .shp file
            landmask_bathymetry_thr:   float32, if set the value is the threshold used to extract the landmask from reader_sea_depth
            elements_density:      boolean, add number of elements present in each grid cell to output
            origin_marker:         int/list/tuple/np.ndarray, only elements with these values of "origin_marker" will be considered
            active_status:         boolean, only active elements will be considered
            weight:                string, elements property to be extracted to produce maps
            sim_description:       string, descrition of simulation to be included in netcdf attributes
            timestep_values:       boolean, only active elements will be considered
            compress_species:      boolean, only species present in self.result will be used to construct netCDF file
            weight_mode:           string, one of:
                                   - "extensive": histogram is interpreted as an extensive sum and converted to concentration
                                   - "mean": histogram is interpreted as sum(property) and divided by particle count per cell
        Supported output modes
        ----------------------
        weight_mode="extensive"
            Interprets the histogrammed quantity as an extensive mass-like quantity.
            The raw per-cell sum is converted to:
              - water-column concentration using cell water volume
              - sediment concentration using dry sediment mass
        weight_mode="mean"
            Interprets the histogrammed quantity as sum(property) per cell and converts
            it to a per-snapshot cell mean:
                mean(property) = sum(property) / number_of_elements
            Empty cells are stored as NaN.
            When block-averaging snapshots, the already-computed per-snapshot mean field
            is averaged with nanmean; snapshots where a cell is empty remain NaN and are
            ignored in the block mean for that cell.

        Auxiliary-reader limitation
        ---------------------------
        Auxiliary bathymetry and active sediment layer thickness readers used by
        this function are currently expected to be rectilinear lon/lat datasets
        in the ChemicalDrift [-180, 180] longitude convention.
        Projected auxiliary x/y readers and curvilinear 2D auxiliary coordinates
        are not supported by the auxiliary-reader opening/subsetting helper here.

        Time averaging
        --------------
        If `time_avg_conc=True`, the output is a block mean of discrete model snapshots,
        not a continuous-time integral average.
        - Uniform model timesteps are assumed by design.
        - The averaging window length (`deltat`, hours) does not need to be an exact
          multiple of the model output interval.
        - Blocks are formed with ndt = ceil(deltat / model_dt).
        - The final block may be shorter.
        - No synthetic or shifted timestamps are introduced.
        - `avg_time` stores the timestamp of the last snapshot included in each block.

        Depth coordinate in the saved NetCDF
        ------------------------------------
        Internally, the model uses z coordinates with surface at 0 and negative values
        downward.
        In the saved NetCDF:
          - `depth` is written as a CF-style positive-down coordinate
          - it is monotonic increasing from 0 downward
          - it represents the top of each saved layer
          - `depth_bounds` stores the corresponding positive-down layer bounds
          - `depth_model_top` and `depth_model_bounds` preserve the original model-z
            layer description explicitly

        Dateline / Coordinates convention
        -------------------------------
        This writer expects ChemicalDrift longitudes in the native [-180, 180] convention.
        For geographic CRS, lat_resol/lon_resol are written in degrees.
        For projected CRS, dx/dy are written in projected units (typically meters).
        If geographic lat_resol/lon_resol inputs are converted internally to projected
        dx/dy, the original inputs are also saved as input_lat_resol_deg/input_lon_resol_deg.
        It does not support:
          - 0..360 output corners
          - dateline-crossing output domains in this function
          - automatic bounds inference for dateline-spanning data

        Interpolation note
        ------------------
        Bathymetry and optional active sediment layer thickness are interpolated from the
        auxiliary reader field to the output grid using:
          1) linear interpolation
          2) nearest-neighbour fill for any remaining NaNs
        This is not conservative. If the output grid is finer than the auxiliary field,
        the apparent spatial resolution may increase without adding information.

        Compatibility note (weight_mode="extensive" only)
        -----------------------------------------------
        One variable stores both water-like and sediment-like species.
        Use units_water, units_sediment and specie_phase for machine-readable interpretation.
        mass_unit and weight_unit are labels only here. No unit conversion is performed.
        """
        import numpy as np
        from netCDF4 import Dataset, date2num
        import opendrift
        from pyproj import CRS, Proj, Transformer
        import pandas as pd
        import gc

        def is_valid_proj4(density_proj):
            try:
                CRS.from_string(density_proj)
                return density_proj
            except Exception as e1:
                logger.debug("CRS.from_string failed for density_proj=%r: %s", density_proj, e1, exc_info=True)
                try:
                    density_proj = CRS.from_epsg(density_proj).to_proj4()
                    return density_proj
                except Exception as e2:
                    logger.debug("CRS.from_epsg failed for density_proj=%r: %s", density_proj, e2, exc_info=True)
                    raise ValueError(f"Invalid density_proj: {density_proj}") from e2

        def _normalize_unit_string(u):
            if u is None:
                return None
            return str(u).strip()

        def _resolve_weight_units():
            src_unit = None
            if weight is not None and weight in self.result.variables:
                try:
                    src_unit = _normalize_unit_string(self.result[weight].attrs.get("units", None))
                except Exception as e:
                    logger.debug("Could not read source units for weight=%r: %s", weight, e, exc_info=True)
                    src_unit = None

            if weight_mode == "extensive":
                if mass_unit is None:
                    if not src_unit:
                        raise ValueError(
                            "mass_unit must be specified when source weight units are unavailable "
                            "for weight_mode='extensive'.")
                    return src_unit
                out_unit = _normalize_unit_string(mass_unit)
                if src_unit and out_unit != src_unit:
                    raise ValueError(
                        f"mass_unit={out_unit!r} does not match source units {src_unit!r}. "
                        "This writer does not perform unit conversion.")
                return out_unit

            out_unit = _normalize_unit_string(weight_unit)
            if not out_unit:
                raise ValueError(
                    "weight_unit must be specified explicitly for weight_mode='mean'. "
                    "This writer does not infer or convert units in mean mode.")
            if src_unit and out_unit != src_unit:
                raise ValueError(
                    f"weight_unit={out_unit!r} does not match source units {src_unit!r}. "
                    "This writer does not perform unit conversion.")
            return out_unit

        def _resolve_var_name(ds, requested_name_or_standard_name):
            if requested_name_or_standard_name in ds.data_vars:
                return requested_name_or_standard_name
            matches = [
                v for v in ds.data_vars
                if str(ds[v].attrs.get("standard_name", "")).strip() == requested_name_or_standard_name]

            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                raise ValueError(
                    f"More than one variable in dataset matches standard_name "
                    f"{requested_name_or_standard_name!r}: {matches}")
            else:
                raise ValueError(
                    f"No variable found with name or standard_name "
                    f"{requested_name_or_standard_name!r}")

        def _open_cf_reader_with_var(nc_path, var_name,
                                     llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
                                     allow_corner_adjust=False):
            """
            Open an auxiliary CF-like reader and subset it with lon/lat corners.

            This helper currently supports only auxiliary datasets whose selection
            coordinates are rectilinear longitude/latitude in the ChemicalDrift
            [-180, 180] convention.
            It does not support:
              - projected auxiliary x/y coordinates in meters
              - 0..360 longitude convention
              - dateline-crossing auxiliary domains
              - curvilinear 2D lon/lat coordinate selection
            """
            from opendrift.readers import reader_netCDF_CF_generic
            import xarray as xr

            for nm, vv in {
                "llcrnrlon": llcrnrlon,
                "urcrnrlon": urcrnrlon,
                "llcrnrlat": llcrnrlat,
                "urcrnrlat": urcrnrlat,
            }.items():
                if vv is None or not np.isfinite(vv):
                    raise ValueError(f"{nm} must be finite, got {vv!r}")

            if not (-180.0 <= llcrnrlon <= 180.0 and -180.0 <= urcrnrlon <= 180.0):
                raise ValueError(
                    "Output corners must use the ChemicalDrift longitude convention [-180, 180]. "
                    "0..360 output domains are not supported here.")
            if llcrnrlon > urcrnrlon:
                raise ValueError(
                    "Dateline-crossing output domains are not supported in this writer. "
                    "Use non-crossing corners with llcrnrlon <= urcrnrlon.")

            with xr.open_dataset(nc_path) as ds:
                resolved_var = _resolve_var_name(ds, var_name)
                da = ds[resolved_var]

                lat_names = ["latitude", "lat"]
                lon_names = ["longitude", "lon", "long"]
                lat_name = next((name for name in lat_names if name in da.coords), None)
                lon_name = next((name for name in lon_names if name in da.coords), None)
                if any(x is None for x in [lat_name, lon_name]):
                    raise ValueError(f"Latitude/longitude coordinate names not found in {nc_path}")

                lat_values = np.asarray(da.coords[lat_name].values)
                lon_values = np.asarray(da.coords[lon_name].values)
                lat_f = lat_values[np.isfinite(lat_values)]
                lon_f = lon_values[np.isfinite(lon_values)]

                if lon_f.size == 0 or lat_f.size == 0:
                    raise ValueError(f"No finite lat/lon values found in {nc_path}")

                if np.nanmin(lon_f) < -180.0 or np.nanmax(lon_f) > 180.0:
                    raise ValueError(
                        f"{nc_path} appears not to use the ChemicalDrift longitude convention [-180, 180]. "
                        "0..360 reader longitudes are not supported by this writer.")

                lat_sorted = np.sort(lat_f)
                lon_sorted = np.sort(lon_f)
                if allow_corner_adjust:
                    if lat_sorted.size < 3 or lon_sorted.size < 3:
                        raise ValueError(
                            f"Grid in {nc_path} is too small to safely adjust corners.")
                lat_min, lat_max = lat_sorted[0], lat_sorted[-1]
                lon_min, lon_max = lon_sorted[0], lon_sorted[-1]

                if allow_corner_adjust:
                    if llcrnrlat < lat_min:
                        new = lat_sorted[1]
                        logger.warning(f"Changed llcrnrlat from {llcrnrlat} to {new} for {var_name}")
                        llcrnrlat = new
                    if urcrnrlat > lat_max:
                        new = lat_sorted[-2]
                        logger.warning(f"Changed urcrnrlat from {urcrnrlat} to {new} for {var_name}")
                        urcrnrlat = new
                    if llcrnrlon < lon_min:
                        new = lon_sorted[1]
                        logger.warning(f"Changed llcrnrlon from {llcrnrlon} to {new} for {var_name}")
                        llcrnrlon = new
                    if urcrnrlon > lon_max:
                        new = lon_sorted[-2]
                        logger.warning(f"Changed urcrnrlon from {urcrnrlon} to {new} for {var_name}")
                        urcrnrlon = new
                else:
                    if not (lat_min <= llcrnrlat <= urcrnrlat <= lat_max):
                        raise ValueError(f"Corners are outside latitude bounds of {var_name!r} in {nc_path}")
                    if not (lon_min <= llcrnrlon <= urcrnrlon <= lon_max):
                        raise ValueError(f"Corners are outside longitude bounds of {var_name!r} in {nc_path}")

                da_sel = da.where(
                    (da[lon_name] >= llcrnrlon) &
                    (da[lon_name] <= urcrnrlon) &
                    (da[lat_name] >= llcrnrlat) &
                    (da[lat_name] <= urcrnrlat),
                    drop=True)

                num_x = da_sel.coords[lon_name].size
                num_y = da_sel.coords[lat_name].size
                if num_x == 0 or num_y == 0:
                    raise ValueError(f"Empty selection for {var_name!r} in {nc_path}")

            rdr = reader_netCDF_CF_generic.Reader(nc_path)
            return rdr, num_x, num_y, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat

        def _safe_divide_num_count(num, den):
            num = np.asarray(num, dtype=np.float32)
            den = np.asarray(den, dtype=np.float32)
            out = np.full(num.shape, np.nan, dtype=np.float32)
            np.divide(num, den, out=out, where=(den > 0))
            return out

        def _compute_snapshot_block_layout(times, deltat_hours):
            times = np.asarray(times)
            Tsnap = len(times)
            if Tsnap == 0:
                return 1, 0, None

            if Tsnap == 1 or deltat_hours is None:
                ndt = 1
            else:
                dt_hours = (times[1] - times[0]).total_seconds() / 3600.0
                if dt_hours <= 0:
                    raise ValueError("Model output times must be strictly increasing.")
                ndt = max(1, int(np.ceil(float(deltat_hours) / dt_hours)))

            odt = int(np.ceil(Tsnap / ndt))
            block_end_idx = np.minimum((np.arange(odt) + 1) * ndt - 1, Tsnap - 1)
            avg_times = times[block_end_idx]
            return ndt, odt, avg_times

        def _species_category_ids():
            sediment_ids = set()
            for attr in ("num_srev", "num_ssrev", "num_sirrev", "num_sburied"):
                if hasattr(self, attr):
                    sediment_ids.add(int(getattr(self, attr)))
            particle_ids = set()
            for attr in ("num_prev", "num_psrev", "num_pirrev"):
                if hasattr(self, attr):
                    particle_ids.add(int(getattr(self, attr)))
            return sediment_ids, particle_ids

        def _reorder_depth_for_output(arr, depth_order):
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.ndim == 5:
                return arr[:, :, depth_order, :, :]
            if arr.ndim == 3:
                return arr[depth_order, :, :]
            raise ValueError(f"Unexpected array rank for depth reorder: {arr.ndim}")

        def _make_cf_depth_axis(z_edges_model):
            z_edges_model = np.asarray(z_edges_model, dtype=np.float64)
            nlay = z_edges_model.size - 1
            if nlay <= 0:
                raise ValueError("z_edges_model must contain at least two edges.")

            depth_order = np.arange(nlay - 1, -1, -1)

            z_top_model = z_edges_model[1:]
            z_bot_model = z_edges_model[:-1]

            z_top_model = z_top_model[depth_order]
            z_bot_model = z_bot_model[depth_order]

            depth = -z_top_model
            depth_bounds = np.column_stack((-z_top_model, -z_bot_model)).astype(np.float64)
            depth_model_top = z_top_model.astype(np.float64)
            depth_model_bounds = np.column_stack((z_top_model, z_bot_model)).astype(np.float64)

            return depth_order, depth, depth_bounds, depth_model_top, depth_model_bounds

        def _write_cf_crs_variable(nc, density_proj_str):
            crs_obj = CRS.from_user_input(density_proj_str)

            crs_var = nc.createVariable('crs', 'i4')
            crs_var.long_name = 'Coordinate reference system'
            crs_var.spatial_ref = crs_obj.to_wkt()
            crs_var.crs_wkt = crs_obj.to_wkt()
            crs_var.proj4_params = density_proj_str
            crs_var.actual_density_proj_str = density_proj_str
            try:
                cf_dict = crs_obj.to_cf()
                for k, v in cf_dict.items():
                    try:
                        setattr(crs_var, k, v)
                    except Exception as e:
                        logger.debug("Could not set CRS attribute %r=%r: %s", k, v, e, exc_info=True)
            except Exception as e:
                logger.debug("Could not build CF CRS dict for %r: %s", density_proj_str, e, exc_info=True)

            return crs_obj, crs_var

        # Input checks
        if sum(x is None for x in [lat_resol, lon_resol]) == 1:
            raise ValueError("Both lat_resol and lon_resol must be specified together.")
        elif sum(x is None for x in [lat_resol, lon_resol]) == 0:
            if pixelsize_m is not None:
                raise ValueError("If lat_resol/lon_resol are specified, pixelsize_m must be None.")

        weight_mode = str(weight_mode).lower()
        if weight_mode not in {"extensive", "mean"}:
            raise ValueError("weight_mode must be one of: 'extensive', 'mean'")

        if weight is None:
            raise ValueError("weight must be specified explicitly for both weight modes.")

        if weight_mode == "extensive" and "mass" not in str(weight).lower():
            raise ValueError(
                "weight_mode='extensive' currently supports only mass-like weights. "
                f"Got weight={weight!r}.")

        if timestep_values and weight_mode != "extensive":
            raise ValueError("timestep_values=True is only valid with weight_mode='extensive'.")

        if timestep_values and str(weight) == "mass":
            raise ValueError(
                "timestep_values=True is intended for cumulative extensive quantities, not plain 'mass'.")

        resolved_weight_unit = _resolve_weight_units()

        if any(v is not None for v in (llcrnrlon, urcrnrlon)):
            if llcrnrlon is None or urcrnrlon is None:
                raise ValueError("llcrnrlon and urcrnrlon must be specified together.")
            if not (-180.0 <= llcrnrlon <= 180.0 and -180.0 <= urcrnrlon <= 180.0):
                raise ValueError("Supplied output longitudes must be in the ChemicalDrift [-180, 180] convention.")
            if llcrnrlon > urcrnrlon:
                raise ValueError("Dateline-crossing output domains are not supported here.")

        if self.mode != opendrift.models.basemodel.Mode.Config:
            self.mode = opendrift.models.basemodel.Mode.Config
            logger.debug("Changed self.mode to Config")

        # Time filtering
        all_times = pd.to_datetime(self.result.time).to_pydatetime()
        if time_start is not None:
            time_start = pd.to_datetime(time_start).to_pydatetime()
        if time_end is not None:
            time_end = pd.to_datetime(time_end).to_pydatetime()

        if (time_start is not None) or (time_end is not None):
            tmask = np.ones(len(all_times), dtype=bool)
            if time_start is not None:
                tmask &= np.array(all_times) >= time_start
            if time_end is not None:
                tmask &= np.array(all_times) <= time_end
            if not tmask.any():
                logger.warning(f"No timesteps fall within time_start={time_start} and time_end={time_end}.")
                return
            filtered_times = np.array(all_times)[tmask]
        else:
            filtered_times = np.asarray(all_times)

        # Landmask readers
        if landmask_shapefile is not None:
            if 'shape' in self.env.readers:
                del self.env.readers['shape']
            from opendrift.readers import reader_shape
            custom_landmask = reader_shape.Reader.from_shpfiles(landmask_shapefile)
            self.add_reader(custom_landmask)
        elif 'global_landmask' not in self.env.readers:
            from opendrift.readers import reader_global_landmask
            global_landmask = reader_global_landmask.Reader()
            self.add_reader(global_landmask)

        # Bathymetry reader
        if reader_sea_depth is not None:
            if any(v is None for v in (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)):
                raise ValueError("Corners must be provided when reader_sea_depth is used.")

            reader_sea_depth, num_x, num_y, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = _open_cf_reader_with_var(
                reader_sea_depth,
                'sea_floor_depth_below_sea_level',
                llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
                allow_corner_adjust=True)
        else:
            raise ValueError("A reader for 'sea_floor_depth_below_sea_level' must be specified.")

        active_sediment_layer_thickness_reader = None
        if reader_active_sediment_layer_thickness is not None:
            if any(v is None for v in (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)):
                raise ValueError("Corners must be provided when reader_active_sediment_layer_thickness is used.")

            active_sediment_layer_thickness_reader, _, _, _, _, _, _ = _open_cf_reader_with_var(
                reader_active_sediment_layer_thickness,
                'active_sediment_layer_thickness',
                llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
                allow_corner_adjust=False)

        # Species init
        if not hasattr(self, "name_species"):
            self.init_species()
        if not hasattr(self, "nspecies"):
            self.nspecies = 4
        if not hasattr(self, "name_species"):
            self.name_species = ['dissolved', 'DOC', 'SPM', 'sediment']

        if self.mode != opendrift.models.basemodel.Mode.Result:
            self.mode = opendrift.models.basemodel.Mode.Result
            logger.debug("Changed self.mode to Result")

        logger.info("Postprocessing: write density/concentration/property maps to netCDF")

        bathimetry_res = 500
        if num_x > 500 and num_y > 500:
            bathimetry_res = min(num_x, num_y) - 1
            logger.warning(f"Changed intermediate bathymetry interpolation resolution to {bathimetry_res}")

        grid = np.meshgrid(
            np.linspace(llcrnrlon, urcrnrlon, bathimetry_res),
            np.linspace(llcrnrlat, urcrnrlat, bathimetry_res))
        self.conc_lon = grid[0]
        self.conc_lat = grid[1]

        topo_raw = reader_sea_depth.get_variables_interpolated_xy(
            ['sea_floor_depth_below_sea_level'],
            x=self.conc_lon.flatten(),
            y=self.conc_lat.flatten(),
            time=reader_sea_depth.times[0] if reader_sea_depth.times is not None else None
        )[0]['sea_floor_depth_below_sea_level'].reshape(self.conc_lon.shape)

        self.conc_topo_source_invalid = (
            np.ma.getmaskarray(topo_raw) |
            ~np.isfinite(np.asarray(topo_raw, dtype=float)))
        self.conc_topo = np.asarray(np.ma.filled(topo_raw, np.nan), dtype=np.float64)

        self.conc_active_sediment_layer_thickness = None
        self.conc_active_sediment_layer_thickness_source_invalid = None

        if active_sediment_layer_thickness_reader is not None:
            aslt_raw = active_sediment_layer_thickness_reader.get_variables_interpolated_xy(
                ['active_sediment_layer_thickness'],
                x=self.conc_lon.flatten(),
                y=self.conc_lat.flatten(),
                time=active_sediment_layer_thickness_reader.times[0] if active_sediment_layer_thickness_reader.times is not None else None
            )[0]['active_sediment_layer_thickness'].reshape(self.conc_lon.shape)

            self.conc_active_sediment_layer_thickness_source_invalid = (
                np.ma.getmaskarray(aslt_raw) |
                ~np.isfinite(np.asarray(aslt_raw, dtype=float)))
            self.conc_active_sediment_layer_thickness = np.asarray(
                np.ma.filled(aslt_raw, np.nan), dtype=np.float64)

        if pixelsize_m == 'auto':
            lat = self.result.lat
            latspan = lat.max() - lat.min()
            pixelsize_m = 30
            if latspan > 0.05:
                pixelsize_m = 50
            if latspan > 0.1:
                pixelsize_m = 300
            if latspan > 0.3:
                pixelsize_m = 500
            if latspan > 0.7:
                pixelsize_m = 1000
            if latspan > 2:
                pixelsize_m = 2000
            if latspan > 5:
                pixelsize_m = 4000

        if density_proj is None:
            density_proj_str = '+proj=moll +ellps=WGS84 +lon_0=0.0'
        else:
            density_proj_str = is_valid_proj4(density_proj)

        density_proj = Proj(density_proj_str)
        crs_obj = CRS.from_user_input(density_proj_str)

        is_moll = "+proj=moll" in density_proj_str
        is_latlon = crs_obj.is_geographic

        if not is_moll and (lat_resol is None or lon_resol is None):
            raise ValueError("lat_resol and lon_resol must be set for non-moll projections.")

        input_lat_resol_deg = None
        input_lon_resol_deg = None

        if sum(x is None for x in [lat_resol, lon_resol]) == 0:
            if (not is_moll) and (not is_latlon):
                input_lon_resol_deg = float(lon_resol)
                input_lat_resol_deg = float(lat_resol)

                source_proj = Proj("+proj=longlat +datum=WGS84 +no_defs")
                transformer = Transformer.from_proj(source_proj, density_proj, always_xy=True)

                dummy_lon = (llcrnrlon + urcrnrlon) / 2.0
                dummy_lat = (llcrnrlat + urcrnrlat) / 2.0
                x1, y1 = transformer.transform(dummy_lon, dummy_lat)
                x2, y2 = transformer.transform(dummy_lon + lon_resol, dummy_lat + lat_resol)

                lon_resol = abs(x2 - x1)  # projected dx
                lat_resol = abs(y2 - y1)  # projected dy

                logger.warning(
                    "Converted supplied lon_resol/lat_resol from degrees to projected grid spacing "
                    f"(dx={lon_resol}, dy={lat_resol}) in the target CRS.")

        z_array = self._build_z_array(zlevels, zmin_cap=-10000.0, ztop=0.0)
        logger.info("Internal model z grid boundaries: %s", [str(item) for item in z_array])

        need_counts = elements_density or (weight_mode == "mean")
        write_density = elements_density or (weight_mode == "mean")

        H, x_centers, y_centers, lon_center_2d, lat_center_2d, H_count, keep_species, name_species_out = self.get_chemical_density_array(
            pixelsize_m=pixelsize_m,
            is_moll=is_moll, is_latlon=is_latlon,
            z_array=z_array,
            lat_resol=lat_resol,
            lon_resol=lon_resol,
            density_proj=density_proj,
            llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
            weight=weight,
            origin_marker=origin_marker,
            active_status=active_status,
            elements_density=need_counts,
            time_start=time_start,
            time_end=time_end,
            time_chunk_size=time_chunk_size,
            timestep_values=timestep_values,
            compress_species=compress_species,
            weight_mode=weight_mode,)

        nspecies_out = len(name_species_out)

        # Landmask on output centers: keep true land separate from invalid auxiliary domain
        if (landmask_bathymetry_thr is not None) and (reader_sea_depth is not None):
            vals = reader_sea_depth.get_variables_interpolated_xy(
                ['sea_floor_depth_below_sea_level'],
                x=np.clip(lon_center_2d.ravel(), reader_sea_depth.xmin, reader_sea_depth.xmax),
                y=np.clip(lat_center_2d.ravel(), reader_sea_depth.ymin, reader_sea_depth.ymax),
                time=reader_sea_depth.times[0] if reader_sea_depth.times is not None else None
            )[0]['sea_floor_depth_below_sea_level']
            landmask_raw = (vals <= landmask_bathymetry_thr).reshape(lon_center_2d.shape)
        elif landmask_shapefile is not None:
            landmask_raw = self.env.readers['shape'].get_variables(
                'land_binary_mask', x=lon_center_2d, y=lat_center_2d
            )['land_binary_mask']
        else:
            landmask_raw = self.env.readers['global_landmask'].get_variables(
                'land_binary_mask', x=lon_center_2d, y=lat_center_2d
            )['land_binary_mask']

        if landmask_raw.shape != lon_center_2d.shape:
            landmask_raw = landmask_raw.reshape(lon_center_2d.shape)

        raw = np.asarray(landmask_raw)
        if raw.dtype != np.bool_:
            if np.issubdtype(raw.dtype, np.floating):
                raw_f = raw[np.isfinite(raw)]
            else:
                raw_f = raw.ravel()
            vals = np.unique(raw_f)[:10]
            if vals.size and not np.all(np.isin(vals, [0, 1])):
                logger.warning(
                    f"Landmask contained non-binary values (sample: {vals}). "
                    "Normalizing with policy='auto'.")

        pixel_mean_depth, pixel_area, pixel_active_sediment_layer_thickness, bathy_invalid_mask = self.get_pixel_mean_depth(
            lon_center_2d, lat_center_2d,
            is_moll, is_latlon, lat_resol, lon_resol)

        landmask = self._normalize_landmask(landmask_raw, policy="auto", thr=0.5)

        if bathy_invalid_mask is None:
            bathy_invalid_mask = np.zeros_like(landmask, dtype=bool)
        else:
            bathy_invalid_mask = np.asarray(bathy_invalid_mask, dtype=bool)
            if bathy_invalid_mask.shape != landmask.shape:
                raise ValueError(
                    f"bathy_invalid_mask shape {bathy_invalid_mask.shape} != landmask shape {landmask.shape}")

        combined_mask = landmask | bathy_invalid_mask

        landmask_yx = landmask.T
        domain_invalid_yx = bathy_invalid_mask.T
        combined_mask_yx = combined_mask.T

        if (lon_center_2d.shape[0] > bathimetry_res) or (lon_center_2d.shape[1] > bathimetry_res):
            logger.warning(
                "The requested output grid is finer than the intermediate auxiliary interpolation grid. "
                "This may increase apparent spatial resolution without adding information.")

        def _remove_reader_by_key(self, key):
            rdr = self.env.readers.pop(key, None)
            if rdr is not None and hasattr(rdr, "close"):
                try:
                    rdr.close()
                except Exception as e:
                    logger.debug("Could not close reader %r: %s", key, e, exc_info=True)

        _remove_reader_by_key(self, 'shape')
        _remove_reader_by_key(self, 'global_landmask')

        pixel_volume = np.zeros_like(H[0, 0, :, :, :], dtype=np.float32)

        for zi, zz in enumerate(z_array[:-1]):
            topotmp = -pixel_mean_depth.copy()
            topotmp[topotmp < zz] = zz
            topotmp = z_array[zi + 1] - topotmp
            topotmp[topotmp < 0.1] = 0.0

            if is_moll:
                pixel_volume[zi, :, :] = topotmp * (pixelsize_m ** 2)
            else:
                pixel_volume[zi, :, :] = topotmp * pixel_area

        pixel_volume[pixel_volume == 0.0] = np.nan
        pixel_volume = np.where(~combined_mask[None, :, :], pixel_volume, np.nan)

        sed_L_cfg = self.get_config('chemical:sediment:mixing_depth')
        sed_dens = self.get_config('chemical:sediment:density')
        sed_poro = self.get_config('chemical:sediment:porosity')

        if pixel_active_sediment_layer_thickness is None:
            sed_L = sed_L_cfg
        else:
            sed_L = np.asarray(pixel_active_sediment_layer_thickness, dtype=np.float32)
            sed_L = np.where(np.isfinite(sed_L), sed_L, sed_L_cfg)
            sed_L = np.maximum(sed_L, 0.0)

        if is_moll:
            pixel_sed_mass = ((pixelsize_m ** 2) * sed_L) * (1 - sed_poro) * sed_dens
        else:
            pixel_sed_mass = (pixel_area * sed_L) * (1 - sed_poro) * sed_dens

        pixel_sed_mass = np.where(pixel_sed_mass > 0.0, pixel_sed_mass, np.nan)
        pixel_sed_mass = np.where(~combined_mask, pixel_sed_mass, np.nan)

        sediment_species_ids, _ = _species_category_ids()
        sediment_mask_out = np.array(
            [int(old_id) in sediment_species_ids for old_id in keep_species],
            dtype=bool)

        if weight_mode == "extensive":
            for sp_out in range(nspecies_out):
                if not sediment_mask_out[sp_out]:
                    H[:, sp_out, :, :, :] = H[:, sp_out, :, :, :] / pixel_volume[None, :, :, :]
                else:
                    H[:, sp_out, :, :, :] = H[:, sp_out, :, :, :] / pixel_sed_mass[None, None, :, :]
        else:
            if H_count is None:
                raise ValueError("H_count is required for weight_mode='mean'.")
            H = _safe_divide_num_count(H, H_count)

        Hsm = None
        Hcount_sm = None
        if horizontal_smoothing and not time_avg_conc:
            Hsm = np.empty_like(H, dtype=np.float32)
            for ti in range(H.shape[0]):
                for sp in range(nspecies_out):
                    for zi in range(len(z_array) - 1):
                        Hsm[ti, sp, zi, :, :] = self.horizontal_smooth(
                            H[ti, sp, zi, :, :],
                            n=smoothing_cells,
                            landmask=combined_mask,
                        ).astype(np.float32, copy=False)

            if write_density and H_count is not None:
                Hcount_sm = np.empty(H_count.shape, dtype=np.float32)
                for ti in range(H_count.shape[0]):
                    for sp in range(H_count.shape[1]):
                        for zi in range(H_count.shape[2]):
                            Hcount_sm[ti, sp, zi, :, :] = self.horizontal_smooth(
                                H_count[ti, sp, zi, :, :].astype(np.float32, copy=False),
                                n=smoothing_cells,
                                landmask=combined_mask,
                            ).astype(np.float32, copy=False)

        mean_field = None
        mean_dens = None
        mean_field_sm = None
        mean_dens_sm = None
        avg_times = None

        if time_avg_conc:
            ndt, odt, avg_times = _compute_snapshot_block_layout(filtered_times, deltat)
            if odt == 0:
                logger.warning("No snapshots available for block averaging.")
                return

            mean_field = np.full(
                (odt, H.shape[1], H.shape[2], H.shape[3], H.shape[4]),
                np.nan, dtype=np.float32)

            for ii in range(odt):
                s0 = ii * ndt
                s1 = min((ii + 1) * ndt, H.shape[0])
                if s0 >= s1:
                    continue

                if weight_mode == "mean":
                    mean_field[ii, :, :, :, :] = np.nanmean(
                        H[s0:s1, :, :, :, :], axis=0
                    ).astype(np.float32, copy=False)
                else:
                    mean_field[ii, :, :, :, :] = np.mean(
                        H[s0:s1, :, :, :, :], axis=0, dtype=np.float32)

            if write_density and H_count is not None:
                mean_dens = np.zeros(
                    (odt, H_count.shape[1], H_count.shape[2], H_count.shape[3], H_count.shape[4]),
                    dtype=np.float32)
                for ii in range(odt):
                    s0 = ii * ndt
                    s1 = min((ii + 1) * ndt, H_count.shape[0])
                    if s0 >= s1:
                        continue
                    mean_dens[ii, :, :, :, :] = np.mean(
                        H_count[s0:s1, :, :, :, :], axis=0, dtype=np.float32)

            if horizontal_smoothing:
                mean_field_sm = np.empty_like(mean_field, dtype=np.float32)
                for ti in range(mean_field.shape[0]):
                    for sp in range(nspecies_out):
                        for zi in range(len(z_array) - 1):
                            mean_field_sm[ti, sp, zi, :, :] = self.horizontal_smooth(
                                mean_field[ti, sp, zi, :, :],
                                n=smoothing_cells,
                                landmask=combined_mask,
                            ).astype(np.float32, copy=False)

                if write_density and mean_dens is not None:
                    mean_dens_sm = np.empty_like(mean_dens, dtype=np.float32)
                    for ti in range(mean_dens.shape[0]):
                        for sp in range(mean_dens.shape[1]):
                            for zi in range(mean_dens.shape[2]):
                                mean_dens_sm[ti, sp, zi, :, :] = self.horizontal_smooth(
                                    mean_dens[ti, sp, zi, :, :],
                                    n=smoothing_cells,
                                    landmask=combined_mask,
                                ).astype(np.float32, copy=False)

        depth_order, depth_cf, depth_bounds_cf, depth_model_top, depth_model_bounds = _make_cf_depth_axis(z_array)

        H_out = _reorder_depth_for_output(H, depth_order)
        Hsm_out = _reorder_depth_for_output(Hsm, depth_order)
        mean_field_out = _reorder_depth_for_output(mean_field, depth_order)
        mean_field_sm_out = _reorder_depth_for_output(mean_field_sm, depth_order)
        H_count_out = _reorder_depth_for_output(H_count, depth_order)
        mean_dens_out = _reorder_depth_for_output(mean_dens, depth_order)
        Hcount_sm_out = _reorder_depth_for_output(Hcount_sm, depth_order)
        mean_dens_sm_out = _reorder_depth_for_output(mean_dens_sm, depth_order)
        pixel_volume_out = _reorder_depth_for_output(pixel_volume, depth_order)

        compound = self.get_config('chemical:compound')
        if compound is None:
            compound = "None"
        species_str = ' '.join([f"{isp}:{sp}" for isp, sp in enumerate(name_species_out)])

        nc = Dataset(filename, 'w')
        nc.Conventions = "CF-1.10"
        nc.title = "ChemicalDrift gridded chemical field"
        nc.source = "OpenDrift ChemicalDrift"
        nc.history = "Created by write_netcdf_chemical_density_map"
        nc.actual_density_proj_str = density_proj_str
        nc.weight_name = str(weight)
        nc.weight_mode = weight_mode
        if weight_mode == "extensive":
            nc.phase_units_metadata_mode = "compatibility"
        if weight_mode == "extensive":
            nc.comment = (
                "Depth is written as a CF-style positive-down coordinate derived from model z coordinates. "
                "Auxiliary-reader interpolation is linear first, then nearest-neighbour fill for remaining NaNs; "
                "this is not conservative. "
                "For concentration outputs in compatibility mode, use units_water, "
                "units_sediment and specie_phase to interpret phase-specific units.")
        else:
            nc.comment = (
                "Depth is written as a CF-style positive-down coordinate derived from model z coordinates. "
                "Auxiliary-reader interpolation is linear first, then nearest-neighbour fill for remaining NaNs; "
                "this is not conservative. "
                "This file stores per-cell mean property fields; no mixed-phase "
                "compatibility-unit interpretation is required.")

        nc.createDimension('x', x_centers.size)
        nc.createDimension('y', y_centers.size)
        nc.createDimension('depth', len(depth_cf))
        nc.createDimension('depth_bounds_dim', 2)
        nc.createDimension('specie', nspecies_out)

        nc.species_axis_compressed = int(bool(compress_species))

        nc.createVariable('specie_original_id', 'i4', ('specie',))
        nc.variables['specie_original_id'][:] = keep_species.astype('i4')
        nc.variables['specie_original_id'].long_name = 'Original species id in model indexing'

        nc.createVariable('specie_phase', 'i4', ('specie',))
        nc.variables['specie_phase'][:] = sediment_mask_out.astype('i4')
        nc.variables['specie_phase'].long_name = 'Species phase class'
        nc.variables['specie_phase'].flag_values = np.array([0, 1], dtype='i4')
        nc.variables['specie_phase'].flag_meanings = 'water_like sediment_like'

        maxlen = max(len(s) for s in name_species_out) if name_species_out else 1
        nc.createDimension('name_strlen', maxlen)
        specie_name_var = nc.createVariable('specie_name', 'S1', ('specie', 'name_strlen'))
        specie_name_bytes = np.array([list(s.ljust(maxlen)) for s in name_species_out], dtype='S1')
        specie_name_var[:] = specie_name_bytes
        specie_name_var.long_name = 'Species name aligned with specie axis'

        timestr = 'seconds since 1970-01-01 00:00:00'
        if not time_avg_conc:
            nc.createDimension('time', H_out.shape[0])
            nc.createVariable('time', 'f8', ('time',))
            nc.variables['time'][:] = date2num(filtered_times, timestr)
            nc.variables['time'].units = timestr
            nc.variables['time'].standard_name = 'time'
            nc.variables['time'].long_name = 'Model output snapshot time'
        else:
            nc.createDimension('avg_time', mean_field_out.shape[0])
            nc.createVariable('avg_time', 'f8', ('avg_time',))
            nc.variables['avg_time'][:] = date2num(avg_times, timestr)
            nc.variables['avg_time'].units = timestr
            nc.variables['avg_time'].standard_name = 'time'
            nc.variables['avg_time'].long_name = 'Timestamp of last snapshot included in the averaging block'
            nc.variables['avg_time'].comment = (
                "This is a block mean of discrete model snapshots; no synthetic or shifted timestamps are used.")

        _crs_obj, _ = _write_cf_crs_variable(nc, density_proj_str)

        nc.createVariable('x', 'f8', ('x',))
        nc.createVariable('y', 'f8', ('y',))
        nc.variables['x'][:] = x_centers
        nc.variables['y'][:] = y_centers

        if crs_obj.is_geographic:
            nc.variables['x'].standard_name = 'longitude'
            nc.variables['x'].long_name = 'longitude of cell center'
            nc.variables['x'].units = 'degrees_east'
            nc.variables['x'].axis = 'X'

            nc.variables['y'].standard_name = 'latitude'
            nc.variables['y'].long_name = 'latitude of cell center'
            nc.variables['y'].units = 'degrees_north'
            nc.variables['y'].axis = 'Y'
        else:
            nc.variables['x'].standard_name = 'projection_x_coordinate'
            nc.variables['x'].long_name = 'x coordinate of cell center in target CRS'
            nc.variables['x'].units = 'm'
            nc.variables['x'].axis = 'X'

            nc.variables['y'].standard_name = 'projection_y_coordinate'
            nc.variables['y'].long_name = 'y coordinate of cell center in target CRS'
            nc.variables['y'].units = 'm'
            nc.variables['y'].axis = 'Y'

        nc.createVariable('lon', 'f8', ('y', 'x'))
        nc.createVariable('lat', 'f8', ('y', 'x'))
        nc.variables['lon'][:] = lon_center_2d.T
        nc.variables['lat'][:] = lat_center_2d.T
        nc.variables['lon'].standard_name = 'longitude'
        nc.variables['lon'].long_name = 'longitude of cell center'
        nc.variables['lon'].units = 'degrees_east'
        nc.variables['lat'].standard_name = 'latitude'
        nc.variables['lat'].long_name = 'latitude of cell center'
        nc.variables['lat'].units = 'degrees_north'

        nc.createVariable('depth', 'f8', ('depth',))
        nc.variables['depth'][:] = depth_cf
        nc.variables['depth'].standard_name = 'depth'
        nc.variables['depth'].long_name = 'Top of vertical layer, positive downward, derived from model z coordinates'
        nc.variables['depth'].units = 'm'
        nc.variables['depth'].positive = 'down'
        nc.variables['depth'].axis = 'Z'
        nc.variables['depth'].bounds = 'depth_bounds'

        nc.createVariable('depth_bounds', 'f8', ('depth', 'depth_bounds_dim'))
        nc.variables['depth_bounds'][:] = depth_bounds_cf
        nc.variables['depth_bounds'].long_name = 'Positive-down depth bounds for each vertical layer'
        nc.variables['depth_bounds'].units = 'm'

        nc.createVariable('depth_model_top', 'f8', ('depth',))
        nc.variables['depth_model_top'][:] = depth_model_top
        nc.variables['depth_model_top'].long_name = 'Top of vertical layer in original model z coordinates'
        nc.variables['depth_model_top'].units = 'm'
        nc.variables['depth_model_top'].positive = 'up'

        nc.createVariable('depth_model_bounds', 'f8', ('depth', 'depth_bounds_dim'))
        nc.variables['depth_model_bounds'][:] = depth_model_bounds
        nc.variables['depth_model_bounds'].long_name = 'Layer bounds in original model z coordinates'
        nc.variables['depth_model_bounds'].units = 'm'

        nc.createVariable('specie', 'i4', ('specie',))
        nc.variables['specie'][:] = np.arange(nspecies_out)
        nc.variables['specie'].long_name = 'Species index'
        nc.variables['specie'].comment = ' '.join([f"{isp}:{sp}" for isp, sp in enumerate(name_species_out)])

        if pixelsize_m is not None:
            nc.createVariable('cell_size', 'f8')
            nc.variables['cell_size'][:] = pixelsize_m
            nc.variables['cell_size'].long_name = 'Nominal grid cell size'
            nc.variables['cell_size'].units = 'm'
        else:
            if crs_obj.is_geographic:
                nc.createVariable('lat_resol', 'f8')
                nc.variables['lat_resol'][:] = lat_resol
                nc.variables['lat_resol'].long_name = 'Latitude resolution used to build the output grid'
                nc.variables['lat_resol'].units = 'degrees_north'

                nc.createVariable('lon_resol', 'f8')
                nc.variables['lon_resol'][:] = lon_resol
                nc.variables['lon_resol'].long_name = 'Longitude resolution used to build the output grid'
                nc.variables['lon_resol'].units = 'degrees_east'
            else:
                nc.createVariable('dy', 'f8')
                nc.variables['dy'][:] = lat_resol
                nc.variables['dy'].long_name = 'Grid spacing in projected y direction'
                nc.variables['dy'].units = 'm'

                nc.createVariable('dx', 'f8')
                nc.variables['dx'][:] = lon_resol
                nc.variables['dx'].long_name = 'Grid spacing in projected x direction'
                nc.variables['dx'].units = 'm'

                if input_lat_resol_deg is not None:
                    nc.createVariable('input_lat_resol_deg', 'f8')
                    nc.variables['input_lat_resol_deg'][:] = input_lat_resol_deg
                    nc.variables['input_lat_resol_deg'].long_name = 'Original user-supplied latitude increment before projected conversion'
                    nc.variables['input_lat_resol_deg'].units = 'degrees_north'
                if input_lon_resol_deg is not None:
                    nc.createVariable('input_lon_resol_deg', 'f8')
                    nc.variables['input_lon_resol_deg'][:] = input_lon_resol_deg
                    nc.variables['input_lon_resol_deg'].long_name = 'Original user-supplied longitude increment before projected conversion'
                    nc.variables['input_lon_resol_deg'].units = 'degrees_east'

        if horizontal_smoothing:
            nc.createVariable('smoothing_cells', 'i8')
            nc.variables['smoothing_cells'][:] = smoothing_cells
            nc.variables['smoothing_cells'].long_name = 'Horizontal smoothing half-width in grid cells'
            nc.variables['smoothing_cells'].units = '1'

        def _mask5d(arr_5d_tsd_yx, mask_yx):
            mm = np.asarray(mask_yx, dtype=bool)
            if mm.shape != arr_5d_tsd_yx.shape[-2:]:
                if mm.T.shape == arr_5d_tsd_yx.shape[-2:]:
                    mm = mm.T
                else:
                    raise ValueError(
                        f"mask_yx shape {mm.shape} not compatible with data YX {arr_5d_tsd_yx.shape[-2:]}")
            m = mm[None, None, None, :, :]
            return np.broadcast_to(m, arr_5d_tsd_yx.shape)

        def _mask3d(arr_3d_d_yx, mask_yx):
            mm = np.asarray(mask_yx, dtype=bool)
            if mm.shape != arr_3d_d_yx.shape[-2:]:
                if mm.T.shape == arr_3d_d_yx.shape[-2:]:
                    mm = mm.T
                else:
                    raise ValueError(
                        f"mask_yx shape {mm.shape} not compatible with data YX {arr_3d_d_yx.shape[-2:]}")
            m = mm[None, :, :]
            return np.broadcast_to(m, arr_3d_d_yx.shape)

        def _write_masked_5d(var_nc, arr_5d_tsdxy, *, mask_yx, fill_value):
            arr = np.swapaxes(arr_5d_tsdxy, 3, 4)
            mask = _mask5d(arr, mask_yx)
            marr = np.ma.MaskedArray(arr, mask=mask, copy=False)
            if fill_value is not None:
                try:
                    marr.set_fill_value(fill_value)
                except Exception as e:
                    logger.debug("Could not set masked-array fill_value=%r: %s", fill_value, e, exc_info=True)
            var_nc[:] = marr

        def _write_masked_3d(var_nc, arr_3d_dxy, *, mask_yx, fill_value):
            arr = np.swapaxes(arr_3d_dxy, 1, 2)
            mask = _mask3d(arr, mask_yx)
            marr = np.ma.MaskedArray(arr, mask=mask, copy=False)
            if fill_value is not None:
                try:
                    marr.set_fill_value(fill_value)
                except Exception as e:
                    logger.debug("Could not set masked-array fill_value=%r: %s", fill_value, e, exc_info=True)
                    pass
            var_nc[:] = marr

        if write_density:
            if not time_avg_conc:
                nc.createVariable('density', 'i4', ('time', 'specie', 'depth', 'y', 'x'), fill_value=99999)
                H_count_i4 = np.nan_to_num(H_count_out, nan=0, posinf=0, neginf=0).astype('i4', copy=False)
                _write_masked_5d(nc.variables['density'], H_count_i4, mask_yx=combined_mask_yx, fill_value=99999)
                nc.variables['density'].long_name = 'Number of elements in grid cell'
                nc.variables['density'].grid_mapping = 'crs'
                nc.variables['density'].coordinates = 'lat lon'
                nc.variables['density'].units = '1'
                if sim_description is not None:
                    nc.variables['density'].sim_description = str(sim_description)

                if horizontal_smoothing and Hcount_sm_out is not None:
                    nc.createVariable('density_smooth', 'f4', ('time', 'specie', 'depth', 'y', 'x'),
                                      fill_value=np.float32(1.0e36))
                    _write_masked_5d(
                        nc.variables['density_smooth'],
                        Hcount_sm_out.astype(np.float32, copy=False),
                        mask_yx=combined_mask_yx,
                        fill_value=np.float32(1.0e36),)
                    nc.variables['density_smooth'].long_name = 'Horizontally smoothed number of elements in grid cell'
                    nc.variables['density_smooth'].grid_mapping = 'crs'
                    nc.variables['density_smooth'].coordinates = 'lat lon'
                    nc.variables['density_smooth'].units = '1'
                    if sim_description is not None:
                        nc.variables['density_smooth'].sim_description = str(sim_description)
            else:
                nc.createVariable('density_avg', 'f4', ('avg_time', 'specie', 'depth', 'y', 'x'),
                                  fill_value=np.float32(1.0e36))
                _write_masked_5d(
                    nc.variables['density_avg'],
                    mean_dens_out.astype(np.float32, copy=False),
                    mask_yx=combined_mask_yx,
                    fill_value=np.float32(1.0e36),)
                nc.variables['density_avg'].long_name = 'Mean number of elements per snapshot in averaging block'
                nc.variables['density_avg'].grid_mapping = 'crs'
                nc.variables['density_avg'].coordinates = 'lat lon'
                nc.variables['density_avg'].units = '1'
                if sim_description is not None:
                    nc.variables['density_avg'].sim_description = str(sim_description)

                if horizontal_smoothing and mean_dens_sm_out is not None:
                    nc.createVariable('density_smooth_avg', 'f4', ('avg_time', 'specie', 'depth', 'y', 'x'),
                                      fill_value=np.float32(1.0e36))
                    _write_masked_5d(
                        nc.variables['density_smooth_avg'],
                        mean_dens_sm_out.astype(np.float32, copy=False),
                        mask_yx=combined_mask_yx,
                        fill_value=np.float32(1.0e36),)
                    nc.variables['density_smooth_avg'].long_name = (
                        'Horizontally smoothed mean number of elements per snapshot in averaging block')
                    nc.variables['density_smooth_avg'].grid_mapping = 'crs'
                    nc.variables['density_smooth_avg'].coordinates = 'lat lon'
                    nc.variables['density_smooth_avg'].units = '1'
                    if sim_description is not None:
                        nc.variables['density_smooth_avg'].sim_description = str(sim_description)

        if weight_mode == "extensive":
            units_water = f"{resolved_weight_unit} m-3"
            units_sediment = f"{resolved_weight_unit} kg-1 dry_weight"

            if not time_avg_conc:
                nc.createVariable('concentration', 'f8', ('time', 'specie', 'depth', 'y', 'x'), fill_value=1.e36)
                _write_masked_5d(nc.variables['concentration'], H_out, mask_yx=combined_mask_yx, fill_value=1.e36)
                nc.variables['concentration'].long_name = f"{compound} concentration from weight={weight}\nspecie {species_str}"
                nc.variables['concentration'].grid_mapping = 'crs'
                nc.variables['concentration'].coordinates = 'lat lon'
                nc.variables['concentration'].units = f"{units_water} | {units_sediment}"
                nc.variables['concentration'].units_water = units_water
                nc.variables['concentration'].units_sediment = units_sediment
                nc.variables['concentration'].comment = (
                    "Compatibility mode: one variable stores both water-like and sediment-like species. "
                    "Use units_water, units_sediment and specie_phase.")
                if sim_description is not None:
                    nc.variables['concentration'].sim_description = str(sim_description)
            else:
                nc.createVariable('concentration_avg', 'f8', ('avg_time', 'specie', 'depth', 'y', 'x'), fill_value=1.e36)
                _write_masked_5d(nc.variables['concentration_avg'], mean_field_out, mask_yx=combined_mask_yx, fill_value=1.e36)
                nc.variables['concentration_avg'].long_name = f"{compound} snapshot-block mean concentration from weight={weight}\nspecie {species_str}"
                nc.variables['concentration_avg'].grid_mapping = 'crs'
                nc.variables['concentration_avg'].coordinates = 'lat lon'
                nc.variables['concentration_avg'].units = f"{units_water} | {units_sediment}"
                nc.variables['concentration_avg'].units_water = units_water
                nc.variables['concentration_avg'].units_sediment = units_sediment
                nc.variables['concentration_avg'].comment = (
                    "Compatibility mode block mean. Use units_water, units_sediment and specie_phase.")
                if sim_description is not None:
                    nc.variables['concentration_avg'].sim_description = str(sim_description)
        else:
            if not time_avg_conc:
                nc.createVariable('property_mean', 'f8', ('time', 'specie', 'depth', 'y', 'x'), fill_value=1.e36)
                _write_masked_5d(nc.variables['property_mean'], H_out, mask_yx=combined_mask_yx, fill_value=1.e36)
                nc.variables['property_mean'].long_name = f"Per-snapshot mean particle property {weight!r} in grid cell\nspecie {species_str}"
                nc.variables['property_mean'].grid_mapping = 'crs'
                nc.variables['property_mean'].coordinates = 'lat lon'
                nc.variables['property_mean'].units = resolved_weight_unit
                if sim_description is not None:
                    nc.variables['property_mean'].sim_description = str(sim_description)
            else:
                nc.createVariable('property_mean_avg', 'f8', ('avg_time', 'specie', 'depth', 'y', 'x'), fill_value=1.e36)
                _write_masked_5d(nc.variables['property_mean_avg'], mean_field_out, mask_yx=combined_mask_yx, fill_value=1.e36)
                nc.variables['property_mean_avg'].long_name = f"Block mean of per-snapshot mean particle property {weight!r}\nspecie {species_str}"
                nc.variables['property_mean_avg'].grid_mapping = 'crs'
                nc.variables['property_mean_avg'].coordinates = 'lat lon'
                nc.variables['property_mean_avg'].units = resolved_weight_unit
                if sim_description is not None:
                    nc.variables['property_mean_avg'].sim_description = str(sim_description)

        if horizontal_smoothing:
            if weight_mode == "extensive":
                if not time_avg_conc:
                    nc.createVariable('concentration_smooth', 'f8', ('time', 'specie', 'depth', 'y', 'x'), fill_value=1.e36)
                    _write_masked_5d(nc.variables['concentration_smooth'], Hsm_out, mask_yx=combined_mask_yx, fill_value=1.e36)
                    nc.variables['concentration_smooth'].long_name = f"Horizontally smoothed concentration from weight={weight}\nspecie {species_str}"
                    nc.variables['concentration_smooth'].grid_mapping = 'crs'
                    nc.variables['concentration_smooth'].coordinates = 'lat lon'
                    nc.variables['concentration_smooth'].units = nc.variables['concentration'].units
                    nc.variables['concentration_smooth'].units_water = units_water
                    nc.variables['concentration_smooth'].units_sediment = units_sediment
                    if sim_description is not None:
                        nc.variables['concentration_smooth'].sim_description = str(sim_description)
                else:
                    nc.createVariable('concentration_smooth_avg', 'f8', ('avg_time', 'specie', 'depth', 'y', 'x'), fill_value=1.e36)
                    _write_masked_5d(nc.variables['concentration_smooth_avg'], mean_field_sm_out, mask_yx=combined_mask_yx, fill_value=1.e36)
                    nc.variables['concentration_smooth_avg'].long_name = f"Horizontally smoothed block-mean concentration from weight={weight}\nspecie {species_str}"
                    nc.variables['concentration_smooth_avg'].grid_mapping = 'crs'
                    nc.variables['concentration_smooth_avg'].coordinates = 'lat lon'
                    nc.variables['concentration_smooth_avg'].units = nc.variables['concentration_avg'].units
                    nc.variables['concentration_smooth_avg'].units_water = units_water
                    nc.variables['concentration_smooth_avg'].units_sediment = units_sediment
                    if sim_description is not None:
                        nc.variables['concentration_smooth_avg'].sim_description = str(sim_description)
            else:
                if not time_avg_conc:
                    nc.createVariable('property_mean_smooth', 'f8', ('time', 'specie', 'depth', 'y', 'x'), fill_value=1.e36)
                    _write_masked_5d(nc.variables['property_mean_smooth'], Hsm_out, mask_yx=combined_mask_yx, fill_value=1.e36)
                    nc.variables['property_mean_smooth'].long_name = f"Horizontally smoothed per-snapshot mean particle property {weight!r}\nspecie {species_str}"
                    nc.variables['property_mean_smooth'].grid_mapping = 'crs'
                    nc.variables['property_mean_smooth'].coordinates = 'lat lon'
                    nc.variables['property_mean_smooth'].units = resolved_weight_unit
                    if sim_description is not None:
                        nc.variables['property_mean_smooth'].sim_description = str(sim_description)
                else:
                    nc.createVariable('property_mean_smooth_avg', 'f8', ('avg_time', 'specie', 'depth', 'y', 'x'), fill_value=1.e36)
                    _write_masked_5d(nc.variables['property_mean_smooth_avg'], mean_field_sm_out, mask_yx=combined_mask_yx, fill_value=1.e36)
                    nc.variables['property_mean_smooth_avg'].long_name = f"Horizontally smoothed block-mean particle property {weight!r}\nspecie {species_str}"
                    nc.variables['property_mean_smooth_avg'].grid_mapping = 'crs'
                    nc.variables['property_mean_smooth_avg'].coordinates = 'lat lon'
                    nc.variables['property_mean_smooth_avg'].units = resolved_weight_unit
                    if sim_description is not None:
                        nc.variables['property_mean_smooth_avg'].sim_description = str(sim_description)

        nc.createVariable('volume', 'f8', ('depth', 'y', 'x'), fill_value=0)
        pv = np.ma.masked_invalid(pixel_volume_out)
        _write_masked_3d(nc.variables['volume'], pv, mask_yx=combined_mask_yx, fill_value=0)
        nc.variables['volume'].long_name = 'Water volume of grid cell'
        nc.variables['volume'].grid_mapping = 'crs'
        nc.variables['volume'].coordinates = 'lat lon'
        nc.variables['volume'].units = 'm3'

        nc.createVariable('topo', 'f8', ('y', 'x'), fill_value=0)
        topo_mask = combined_mask_yx | ~np.isfinite(pixel_mean_depth.T)
        topo_ma = np.ma.array(pixel_mean_depth.T, mask=topo_mask, copy=False)
        nc.variables['topo'][:] = topo_ma
        nc.variables['topo'].long_name = 'Water depth at grid-cell center'
        nc.variables['topo'].standard_name = 'sea_floor_depth_below_sea_level'
        nc.variables['topo'].grid_mapping = 'crs'
        nc.variables['topo'].coordinates = 'lat lon'
        nc.variables['topo'].units = 'm'

        if pixel_active_sediment_layer_thickness is not None:
            nc.createVariable('active_sediment_layer_thickness', 'f8', ('y', 'x'), fill_value=0)
            aslt_mask = combined_mask_yx | ~np.isfinite(pixel_active_sediment_layer_thickness.T)
            aslt_ma = np.ma.array(pixel_active_sediment_layer_thickness.T, mask=aslt_mask, copy=False)
            nc.variables['active_sediment_layer_thickness'][:] = aslt_ma
            nc.variables['active_sediment_layer_thickness'].long_name = 'Thickness of active sediment layer'
            nc.variables['active_sediment_layer_thickness'].grid_mapping = 'crs'
            nc.variables['active_sediment_layer_thickness'].coordinates = 'lat lon'
            nc.variables['active_sediment_layer_thickness'].units = 'm'

        if pixelsize_m is None:
            nc.createVariable('area', 'f8', ('y', 'x'), fill_value=0)
            area_ma = np.ma.array(pixel_area.T, mask=combined_mask_yx, copy=False)
            nc.variables['area'][:] = area_ma
            nc.variables['area'].long_name = 'Horizontal area of grid cell'
            nc.variables['area'].grid_mapping = 'crs'
            nc.variables['area'].coordinates = 'lat lon'
            nc.variables['area'].units = 'm2'

        nc.createVariable('land', 'i4', ('y', 'x'), fill_value=-1)
        nc.variables['land'][:] = landmask_yx.astype('i4', copy=False)
        nc.variables['land'].long_name = 'Binary land mask'
        nc.variables['land'].grid_mapping = 'crs'
        nc.variables['land'].coordinates = 'lat lon'
        nc.variables['land'].units = '1'
        nc.variables['land'].flag_values = np.array([0, 1], dtype='i4')
        nc.variables['land'].flag_meanings = 'water land'

        nc.createVariable('domain_invalid', 'i4', ('y', 'x'), fill_value=-1)
        nc.variables['domain_invalid'][:] = domain_invalid_yx.astype('i4', copy=False)
        nc.variables['domain_invalid'].long_name = 'Mask of cells outside the valid auxiliary-data domain'
        nc.variables['domain_invalid'].grid_mapping = 'crs'
        nc.variables['domain_invalid'].coordinates = 'lat lon'
        nc.variables['domain_invalid'].units = '1'
        nc.variables['domain_invalid'].flag_values = np.array([0, 1], dtype='i4')
        nc.variables['domain_invalid'].flag_meanings = 'valid invalid'

        nc.close()
        logger.info('Wrote to %s', filename)

        del H, H_out, pixel_volume, pixel_volume_out, pixel_mean_depth, x_centers, y_centers, lon_center_2d, lat_center_2d
        del landmask, landmask_yx, combined_mask, combined_mask_yx, bathy_invalid_mask, domain_invalid_yx
        if pixel_active_sediment_layer_thickness is not None:
            del pixel_active_sediment_layer_thickness
        if H_count is not None:
            del H_count
        if H_count_out is not None:
            del H_count_out
        if mean_dens is not None:
            del mean_dens
        if mean_dens_out is not None:
            del mean_dens_out
        if Hsm is not None:
            del Hsm
        if Hsm_out is not None:
            del Hsm_out
        if mean_field is not None:
            del mean_field
        if mean_field_out is not None:
            del mean_field_out
        if mean_field_sm is not None:
            del mean_field_sm
        if mean_field_sm_out is not None:
            del mean_field_sm_out
        if Hcount_sm is not None:
            del Hcount_sm
        if Hcount_sm_out is not None:
            del Hcount_sm_out
        if mean_dens_sm is not None:
            del mean_dens_sm
        if mean_dens_sm_out is not None:
            del mean_dens_sm_out
        gc.collect()

    ##### Helpers for write_netcdf_chemical_density_map
    @staticmethod
    def _normalize_landmask(mask, policy="auto", thr=0.5):
        """
        Returns boolean array: True = land, False = water.
        policy:
          - "auto"    : choose based on values
          - "eq1"     : land iff == 1
          - "nonzero" : land iff != 0
          - "thr"     : land iff >= thr
        NaNs/Infs/masked -> water
        """
        import numpy as np

        m_ma = np.ma.asarray(mask)
        m = np.asarray(np.ma.filled(m_ma, np.nan))

        if m.dtype == np.bool_:
            return m
        if np.issubdtype(m.dtype, np.floating):
            m = np.where(np.isfinite(m), m, 0.0)
        if policy == "auto":
            v = m[np.isfinite(m)].ravel()
            if v.size == 0:
                return np.zeros(m.shape, dtype=bool)
            if v.size > 100000:
                step = max(1, v.size // 100000)
                v = v[::step]
            u = np.unique(v)
            if np.all(np.isin(u, [0, 1])):
                policy = "eq1"
            elif np.issubdtype(m.dtype, np.integer) and u.max() > 1:
                policy = "nonzero"
            elif np.issubdtype(m.dtype, np.floating) and (v.min() >= 0.0) and (v.max() <= 1.0):
                policy = "thr"
            else:
                policy = "nonzero"

        if policy == "eq1":
            return (m == 1)
        elif policy == "nonzero":
            return (m != 0)
        elif policy == "thr":
            return (m >= thr)
        else:
            raise ValueError(f"Unknown policy: {policy}")

    @staticmethod
    def _build_z_array(zlevels, *, zmin_cap=-10000.0, ztop=0.0):
        """
        Build strictly increasing bin edges (more negative -> less negative),
        always including zmin_cap and ztop exactly once.
        """
        import numpy as np
        if zlevels is None:
            return np.array([zmin_cap, ztop], dtype=np.float32)

        zlev = np.asarray(zlevels, dtype=np.float32)
        # Remove NaNs/Infs
        zlev = zlev[np.isfinite(zlev)]
        # Force anything > ztop down to ztop
        zlev = np.minimum(zlev, ztop)
        # Ensure caps are included
        zlev = np.concatenate(([zmin_cap], zlev, [ztop]))
        # Prevents duplicate 0
        zlev = np.unique(np.sort(zlev))
        # Ensure strictly increasing
        if zlev.size < 2:
            zlev = np.array([zmin_cap, ztop], dtype=np.float32)

        return zlev.astype(np.float32, copy=False)

    @staticmethod
    def _make_edges(lo, hi, step, dtype=np.float32):
        """
        Build monotonic bin edges including the last edge explicitly.
        """
        import numpy as np

        vals = np.array([lo, hi, step], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Non-finite lo/hi/step: lo={lo}, hi={hi}, step={step}")
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}")
        if hi <= lo:
            raise ValueError(f"hi must be > lo, got lo={lo}, hi={hi}")

        n = int(np.ceil((hi - lo) / step))
        edges = lo + np.arange(n + 1, dtype=np.float64) * step

        if edges[-1] < hi:
            edges = np.append(edges, hi)
        else:
            edges[-1] = hi

        return edges.astype(dtype, copy=False)

    @staticmethod
    def _cumulative_to_deltas_ffill(w_cum, chunk_cols=20000, out=None, mode="deltas", out_ff=None):
        """
        Process cumulative (T,N) with NaNs after deactivation using forward-fill per element.

        w_cum:         np.ndarray (T,N), Cumulative values with NaNs possible (post-deactivation and/or at start).
        chunk_cols:    int, number of columns processed per chunk.
        out:           np.ndarray or None, Preallocated output for mode="deltas" or mode="both". Shape (T,N), dtype float32.
                       If None and mode in {"deltas","both"}, allocated.
        mode:          string, {"deltas","ffill","both"}
                        - "deltas": return per-timestep increments (float32), negatives/non-finite -> 0.
                        - "ffill":  return forward-filled cumulative array (same dtype as w_cum unless out_ff provided).
                        - "both":   return (deltas_out, ffill_out).
        out_ff:        np.ndarray or None, Preallocated output for forward-filled array when mode in {"ffill","both"}.
                       If None, allocated with dtype=w_cum.dtype.
        """
        w_cum = np.asarray(w_cum)
        if w_cum.ndim != 2:
            raise ValueError("w_cum must be a 2D array (T, N)")

        T, N = w_cum.shape

        want_deltas = mode in ("deltas", "both")
        want_ffill  = mode in ("ffill", "both")

        if mode not in ("deltas", "ffill", "both"):
            raise ValueError("mode must be one of {'deltas','ffill','both'}")

        # Allocate outputs as needed
        if want_deltas:
            if out is None:
                out = np.empty((T, N), dtype=np.float32)
            else:
                if out.shape != (T, N):
                    raise ValueError(f"out must have shape {(T, N)}, got {out.shape}")
                if out.dtype != np.float32:
                    raise ValueError("out must be dtype float32")

        if want_ffill:
            if out_ff is None:
                out_ff = np.empty((T, N), dtype=w_cum.dtype)
            else:
                if out_ff.shape != (T, N):
                    raise ValueError(f"out_ff must have shape {(T, N)}, got {out_ff.shape}")

        # Process in chunks
        for start in range(0, N, chunk_cols):
            end = min(start + chunk_cols, N)
            width = end - start

            prev_ff = np.zeros((width,), dtype=w_cum.dtype)
            cur_ff = np.empty((width,), dtype=w_cum.dtype)

            for t in range(T):
                cur = w_cum[t, start:end]
                finite = np.isfinite(cur)
                # cur_ff = prev_ff; then overwrite finite entries with current
                np.copyto(cur_ff, prev_ff)
                if finite.any():
                    cur_ff[finite] = cur[finite]

                if want_ffill:
                    out_ff[t, start:end] = cur_ff

                if want_deltas:
                    out_row = out[t, start:end]  # float32 view
                    np.subtract(cur_ff, prev_ff, out=out_row, casting="unsafe")
                    np.maximum(out_row, 0.0, out=out_row)
                    out_row[~np.isfinite(out_row)] = 0.0

                np.copyto(prev_ff, cur_ff)

        if mode == "deltas":
            return out
        if mode == "ffill":
            return out_ff
        return out, out_ff

    def get_chemical_density_array(self, pixelsize_m, z_array,
                                   is_moll, is_latlon,
                                   lat_resol=None, lon_resol=None,
                                   density_proj=None, llcrnrlon=None, llcrnrlat=None,
                                   urcrnrlon=None, urcrnrlat=None,
                                   weight=None, origin_marker=None,
                                   active_status=False,
                                   elements_density=False,
                                   time_start=None, time_end=None,
                                   time_chunk_size=50,
                                   timestep_values=False,
                                   compress_species=False,
                                   weight_mode="extensive"):
        """
        Compute a gridded species-resolved histogram from particle positions.
        Returns
        H : ndarray
            Raw histogram sum of `weight` in each cell, shape (T, S, D, X, Y).
        x_centers, y_centers : 1D ndarrays
            Cell-center coordinates in the target grid coordinate system.
            - If the target CRS is geographic, these are lon/lat centers in degrees.
            - Otherwise, these are projected x/y centers in target CRS units.
        lon_center_2d, lat_center_2d : 2D ndarrays
            Geographic longitude/latitude of cell centers.
            Centers are computed in projected x/y space first, then inverse-transformed.
            For geographic grids, the returned centers are the rectilinear lon/lat centers.
        H_count : ndarray or None
            Number of elements in each bin, shape (T, S, D, X, Y).
            Allocated if:
              - elements_density=True
              - or weight_mode='mean'
        keep_species, name_species_out
            Species compression metadata.

        Notes / limitations
        - This function expects ChemicalDrift longitudes in the native [-180, 180] convention.
        - It does not support 0..360 longitudes.
        - Automatic bounds inference is disabled for dateline-spanning data.
        - Supplied output corners must be non-crossing in [-180, 180].
        - timestep_values is intended only for cumulative extensive mass-like quantities.
        """
        import numpy as np
        import pandas as pd
        import gc
        from pyproj import Proj, Transformer

        def _is_supported_cumulative_mass_weight(name):
            cumulative_mass_names = {
                'mass_degraded',
                'mass_degraded_water',
                'mass_degraded_sediment',
                'mass_volatilized',
                'mass_photodegraded',
                'mass_biodegraded',
                'mass_biodegraded_water',
                'mass_biodegraded_sediment',
                'mass_hydrolyzed',
                'mass_hydrolyzed_water',
                'mass_hydrolyzed_sediment',
            }
            return str(name) in cumulative_mass_names

        def _should_drop_all_zero_trajectories(weight_name, weight_mode, timestep_values):
            if weight_mode != "extensive":
                return False
            if timestep_values:
                return True
            return ("mass" in str(weight_name).lower())

        def _validate_longitude_convention(lon_values, *, allow_dateline_span=False):
            lon_values = np.asarray(lon_values, dtype=float)
            lon_f = lon_values[np.isfinite(lon_values)]
            if lon_f.size == 0:
                return

            if lon_f.min() < -180.0 or lon_f.max() > 180.0:
                raise ValueError(
                    "ChemicalDrift uses longitudes in the [-180, 180] convention. "
                    "0..360 longitudes are not supported here.")

            if not allow_dateline_span and (lon_f.max() - lon_f.min() > 180.0):
                raise ValueError(
                    "Automatic bounds inference is disabled for dateline-spanning data "
                    "in the [-180, 180] convention.")

        if density_proj is None:
            raise ValueError("density_proj must be a pyproj.Proj instance (or compatible).")

        weight_mode = str(weight_mode).lower()
        if weight_mode not in {"extensive", "mean"}:
            raise ValueError("weight_mode must be one of: 'extensive', 'mean'")

        if weight is None:
            raise ValueError("weight must be specified explicitly.")

        if weight_mode == "extensive" and "mass" not in str(weight).lower():
            raise ValueError("weight_mode='extensive' currently supports only mass-like weights.")

        if timestep_values and weight_mode != "extensive":
            raise ValueError("timestep_values=True is only valid with weight_mode='extensive'.")

        if timestep_values and not _is_supported_cumulative_mass_weight(weight):
            raise ValueError(
                f"timestep_values=True is intended only for known cumulative extensive mass quantities. "
                f"Got weight={weight!r}.")

        Nspecies_total = int(self.nspecies)
        need_counts = elements_density or (weight_mode == "mean")

        if is_moll and pixelsize_m is None:
            raise ValueError("pixelsize_m must be provided for Mollweide grids (is_moll=True).")

        if (not is_moll) and (lat_resol is None or lon_resol is None):
            raise ValueError("lat_resol and lon_resol must be provided for non-moll grids (is_moll=False).")

        times_1d = pd.to_datetime(self.result.time.values)
        if time_start is not None:
            time_start = pd.to_datetime(time_start)
        if time_end is not None:
            time_end = pd.to_datetime(time_end)

        if (time_start is not None) or (time_end is not None):
            tmask = np.ones(times_1d.shape, dtype=bool)
            if time_start is not None:
                tmask &= (times_1d >= time_start)
            if time_end is not None:
                tmask &= (times_1d <= time_end)
            if not tmask.any():
                raise ValueError("No timesteps fall within [time_start, time_end].")
            time_sel = np.flatnonzero(tmask)
            time_filtered = True
        else:
            time_sel = slice(None)
            time_filtered = False

        weight_2d = None
        keep_idx = None

        if weight is not None:
            w_T = self.result[weight].transpose("time", "trajectory").data

            if timestep_values:
                if isinstance(time_sel, slice):
                    idx_ext = slice(None)
                    drop_first = False
                else:
                    idx_sel = time_sel
                    i0 = int(idx_sel[0])
                    if i0 > 0:
                        idx_ext = np.concatenate(([i0 - 1], idx_sel))
                        drop_first = True
                    else:
                        idx_ext = idx_sel
                        drop_first = False

                w_cum_ext = w_T[idx_ext, :]
                dW_ext = self._cumulative_to_deltas_ffill(
                    w_cum_ext, chunk_cols=20000, out=None, mode="deltas", out_ff=None
                )
                dW = dW_ext[1:, :] if drop_first else dW_ext

                elem_keep = np.any(dW > 0, axis=0)
                keep_idx = np.flatnonzero(elem_keep)
                if keep_idx.size == 0:
                    raise ValueError("All trajectories have zero timestep increments in the selected window.")
                weight_2d = dW[:, keep_idx].astype(np.float32, copy=False)

            else:
                w_sel = w_T[time_sel, :]

                if time_filtered:
                    elem_keep = np.any(np.isfinite(w_sel), axis=0)

                    if _should_drop_all_zero_trajectories(weight, weight_mode, timestep_values):
                        elem_keep &= np.any(w_sel != 0, axis=0)

                    keep_idx = np.flatnonzero(elem_keep)
                    if keep_idx.size == 0:
                        raise ValueError("All trajectories are non-finite in the selected window.")
                    w_sel = w_sel[:, keep_idx]

                weight_2d = w_sel.astype(np.float32, copy=False)

        def _slice_time_traj(arr_T, time_sel, keep_idx, *, readonly=False):
            out = arr_T[time_sel, :]
            if keep_idx is not None:
                out = out[:, keep_idx]
            if readonly:
                try:
                    out.setflags(write=False)
                except Exception as e:
                    logger.debug("Could not mark sliced array readonly: %s", e, exc_info=True)
            return out

        lon_T = self.result.lon.transpose("time", "trajectory").data
        lat_T = self.result.lat.transpose("time", "trajectory").data
        z_T = self.result.z.transpose("time", "trajectory").data
        sp_T = self.result.specie.transpose("time", "trajectory").data

        lon_2d = _slice_time_traj(lon_T, time_sel, keep_idx, readonly=True)
        lat_2d = _slice_time_traj(lat_T, time_sel, keep_idx, readonly=True)
        z_2d = _slice_time_traj(z_T, time_sel, keep_idx, readonly=True)
        specie_2d = _slice_time_traj(sp_T, time_sel, keep_idx, readonly=True)

        _validate_longitude_convention(lon_2d, allow_dateline_span=False)

        originmarker_2d = None
        status_2d = None
        if origin_marker is not None:
            om_T = self.result.origin_marker.transpose("time", "trajectory").data
            originmarker_2d = _slice_time_traj(om_T, time_sel, keep_idx, readonly=True)
        if active_status:
            st_T = self.result.status.transpose("time", "trajectory").data
            status_2d = _slice_time_traj(st_T, time_sel, keep_idx, readonly=True)

        n_timef, n_elem = lon_2d.shape

        transformer = None
        if not is_latlon:
            source_proj = Proj("+proj=longlat +datum=WGS84 +no_defs")
            transformer = Transformer.from_proj(source_proj, density_proj, always_xy=True)

        present_species = np.zeros(Nspecies_total, dtype=bool)

        bounds_from_corners = all(v is not None for v in (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat))
        if any(v is not None for v in (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)) and not bounds_from_corners:
            raise ValueError("Either provide all four corners or none.")

        if bounds_from_corners:
            if not (-180.0 <= llcrnrlon <= 180.0 and -180.0 <= urcrnrlon <= 180.0):
                raise ValueError("Supplied output longitudes must use the ChemicalDrift [-180, 180] convention.")
            if llcrnrlon > urcrnrlon:
                raise ValueError("Dateline-crossing output domains are not supported here.")
            llcrnrx, llcrnry = density_proj(llcrnrlon, llcrnrlat)
            urcrnrx, urcrnry = density_proj(urcrnrlon, urcrnrlat)
            xmin = xmax = ymin = ymax = None
        else:
            xmin, ymin = np.inf, np.inf
            xmax, ymax = -np.inf, -np.inf

        m = np.empty(n_elem, dtype=bool)
        tmp = np.empty(n_elem, dtype=bool)

        if active_status:
            status_categories = self.status_categories
            if "active" not in status_categories:
                raise ValueError("No active elements in simulation.")
            active_index = status_categories.index("active")
        else:
            active_index = None

        # Pass 1: discover species and bounds
        for t0i in range(0, n_timef, time_chunk_size):
            t1i = min(t0i + time_chunk_size, n_timef)
            blk_T = t1i - t0i

            lon_blk = lon_2d[t0i:t1i, :]
            lat_blk = lat_2d[t0i:t1i, :]
            z_blk = z_2d[t0i:t1i, :]
            sp_blk = specie_2d[t0i:t1i, :]

            w_blk = weight_2d[t0i:t1i, :] if weight_2d is not None else None
            om_blk = originmarker_2d[t0i:t1i, :] if originmarker_2d is not None else None
            st_blk = status_2d[t0i:t1i, :] if status_2d is not None else None

            for t_rel in range(blk_T):
                lon_row = lon_blk[t_rel, :]
                lat_row = lat_blk[t_rel, :]
                z_row = z_blk[t_rel, :]
                sp_row = sp_blk[t_rel, :]
                w_row = None if w_blk is None else w_blk[t_rel, :]
                om_row = None if om_blk is None else om_blk[t_rel, :]
                st_row = None if st_blk is None else st_blk[t_rel, :]

                m.fill(True)

                if w_row is not None:
                    np.isfinite(w_row, out=tmp)
                    m &= tmp
                    if timestep_values:
                        np.greater(w_row, 0.0, out=tmp)
                        m &= tmp

                if om_row is not None:
                    if isinstance(origin_marker, (list, tuple, np.ndarray)):
                        m &= np.isin(om_row, origin_marker)
                    else:
                        m &= (om_row == origin_marker)

                if st_row is not None:
                    m &= (st_row == active_index)

                np.isfinite(lon_row, out=tmp)
                m &= tmp
                np.isfinite(lat_row, out=tmp)
                m &= tmp
                np.isfinite(z_row, out=tmp)
                m &= tmp

                np.greater_equal(lon_row, -180.0, out=tmp)
                m &= tmp
                np.less_equal(lon_row, 180.0, out=tmp)
                m &= tmp
                np.greater_equal(lat_row, -90.0, out=tmp)
                m &= tmp
                np.less_equal(lat_row, 90.0, out=tmp)
                m &= tmp

                if not m.any():
                    continue

                idx = np.flatnonzero(m)
                sp_vals = sp_row[idx]

                if np.issubdtype(sp_vals.dtype, np.integer):
                    sp_int = sp_vals.astype(np.int32, copy=False)
                    valid_sp = (sp_int >= 0) & (sp_int < Nspecies_total)
                else:
                    finite_sp = np.isfinite(sp_vals)
                    sp_round = np.rint(sp_vals)
                    intlike = finite_sp & (sp_vals == sp_round)
                    sp_int = sp_round.astype(np.int32, copy=False)
                    valid_sp = intlike & (sp_int >= 0) & (sp_int < Nspecies_total)

                if not valid_sp.any():
                    continue

                idx = idx[valid_sp]
                sp_int = sp_int[valid_sp]
                present_species[sp_int] = True

                if not bounds_from_corners:
                    xs = lon_row[idx]
                    ys = lat_row[idx]
                    if not is_latlon:
                        xs, ys = transformer.transform(xs, ys)

                    xmin = min(xmin, xs.min())
                    xmax = max(xmax, xs.max())
                    ymin = min(ymin, ys.min())
                    ymax = max(ymax, ys.max())

        if compress_species:
            keep_species = np.flatnonzero(present_species).astype(np.int32)
            if keep_species.size == 0:
                raise ValueError("No species found after filtering.")
            name_species_out = [self.name_species[i] for i in keep_species]
        else:
            keep_species = np.arange(Nspecies_total, dtype=np.int32)
            name_species_out = list(self.name_species)

        old2new = -np.ones(Nspecies_total, dtype=np.int32)
        old2new[keep_species] = np.arange(keep_species.size, dtype=np.int32)
        Nout = int(keep_species.size)

        if not bounds_from_corners:
            if not np.isfinite([xmin, xmax, ymin, ymax]).all():
                raise ValueError("Could not infer bounds from data after filtering.")

            if is_moll:
                llcrnrx, llcrnry = xmin - pixelsize_m, ymin - pixelsize_m
                urcrnrx, urcrnry = xmax + pixelsize_m, ymax + pixelsize_m
            else:
                llcrnrx, llcrnry = xmin - lon_resol, ymin - lat_resol
                urcrnrx, urcrnry = xmax + lon_resol, ymax + lat_resol

        if is_moll:
            x_edges = self._make_edges(llcrnrx, urcrnrx, pixelsize_m, dtype=np.float32)
            y_edges = self._make_edges(llcrnry, urcrnry, pixelsize_m, dtype=np.float32)
        else:
            x_edges = self._make_edges(llcrnrx, urcrnrx, lon_resol, dtype=np.float32)
            y_edges = self._make_edges(llcrnry, urcrnry, lat_resol, dtype=np.float32)

        x_edges = np.asarray(x_edges, dtype=np.float32)
        y_edges = np.asarray(y_edges, dtype=np.float32)
        z_array = np.asarray(z_array, dtype=np.float32)

        if x_edges.size < 2 or y_edges.size < 2 or z_array.size < 2:
            raise ValueError(
                f"Invalid bin edges: x={x_edges.size}, y={y_edges.size}, z={z_array.size}. "
                "Check bounds/resolution and/or data coverage.")

        Zx = len(z_array) - 1
        Xx = len(x_edges) - 1
        Yx = len(y_edges) - 1

        H = np.zeros((n_timef, Nout, Zx, Xx, Yx), dtype=np.float32)
        H_count = np.zeros_like(H, dtype=np.uint32) if need_counts else None

        m.fill(False)
        tmp.fill(False)

        # Pass 2: fill histogram
        for t0i in range(0, n_timef, time_chunk_size):
            t1i = min(t0i + time_chunk_size, n_timef)
            blk_T = t1i - t0i

            lon_blk = lon_2d[t0i:t1i, :]
            lat_blk = lat_2d[t0i:t1i, :]
            z_blk = z_2d[t0i:t1i, :]
            sp_blk = specie_2d[t0i:t1i, :]

            w_blk = weight_2d[t0i:t1i, :] if weight_2d is not None else None
            om_blk = originmarker_2d[t0i:t1i, :] if originmarker_2d is not None else None
            st_blk = status_2d[t0i:t1i, :] if status_2d is not None else None

            for t_rel in range(blk_T):
                ti = t0i + t_rel

                lon_row = lon_blk[t_rel, :]
                lat_row = lat_blk[t_rel, :]
                z_row = z_blk[t_rel, :]
                sp_row = sp_blk[t_rel, :]
                w_row = None if w_blk is None else w_blk[t_rel, :]
                om_row = None if om_blk is None else om_blk[t_rel, :]
                st_row = None if st_blk is None else st_blk[t_rel, :]

                m.fill(True)

                if w_row is not None:
                    np.isfinite(w_row, out=tmp)
                    m &= tmp
                    if timestep_values:
                        np.greater(w_row, 0.0, out=tmp)
                        m &= tmp

                if om_row is not None:
                    if isinstance(origin_marker, (list, tuple, np.ndarray)):
                        m &= np.isin(om_row, origin_marker)
                    else:
                        m &= (om_row == origin_marker)

                if st_row is not None:
                    m &= (st_row == active_index)

                np.isfinite(lon_row, out=tmp)
                m &= tmp
                np.isfinite(lat_row, out=tmp)
                m &= tmp
                np.isfinite(z_row, out=tmp)
                m &= tmp

                np.greater_equal(lon_row, -180.0, out=tmp)
                m &= tmp
                np.less_equal(lon_row, 180.0, out=tmp)
                m &= tmp
                np.greater_equal(lat_row, -90.0, out=tmp)
                m &= tmp
                np.less_equal(lat_row, 90.0, out=tmp)
                m &= tmp

                if not m.any():
                    continue

                idx = np.flatnonzero(m)
                sp_vals = sp_row[idx]

                if np.issubdtype(sp_vals.dtype, np.integer):
                    sp_int = sp_vals.astype(np.int32, copy=False)
                    valid_sp = (sp_int >= 0) & (sp_int < Nspecies_total)
                else:
                    finite_sp = np.isfinite(sp_vals)
                    sp_round = np.rint(sp_vals)
                    intlike = finite_sp & (sp_vals == sp_round)
                    sp_int = sp_round.astype(np.int32, copy=False)
                    valid_sp = intlike & (sp_int >= 0) & (sp_int < Nspecies_total)

                if not valid_sp.any():
                    continue

                idx = idx[valid_sp]
                sp_int = sp_int[valid_sp]

                sp_new = old2new[sp_int]
                ok = (sp_new >= 0)
                if not ok.any():
                    continue

                idx = idx[ok]
                sp_new = sp_new[ok]

                for sp_out in np.unique(sp_new):
                    sel = (sp_new == sp_out)
                    if not sel.any():
                        continue

                    idx_sp = idx[sel]
                    xs = lon_row[idx_sp]
                    ys = lat_row[idx_sp]
                    zs = z_row[idx_sp].astype(np.float32, copy=True)

                    pos = (zs >= 0.0)
                    if pos.any():
                        zs[pos] = -1e-4

                    ws = None
                    if w_row is not None:
                        ws = w_row[idx_sp]

                    if not is_latlon:
                        xs, ys = transformer.transform(xs, ys)

                    if ws is None:
                        H_xyz, _ = np.histogramdd((xs, ys, zs), bins=(x_edges, y_edges, z_array))
                        H[ti, sp_out, :, :, :] += H_xyz.transpose(2, 0, 1).astype(np.float32, copy=False)
                        if H_count is not None:
                            H_count[ti, sp_out, :, :, :] += H_xyz.transpose(2, 0, 1).astype(np.uint32, copy=False)
                    else:
                        H_xyz_w, _ = np.histogramdd((xs, ys, zs), bins=(x_edges, y_edges, z_array), weights=ws)
                        H[ti, sp_out, :, :, :] += H_xyz_w.transpose(2, 0, 1).astype(np.float32, copy=False)
                        if H_count is not None:
                            H_xyz_c, _ = np.histogramdd((xs, ys, zs), bins=(x_edges, y_edges, z_array))
                            H_count[ti, sp_out, :, :, :] += H_xyz_c.transpose(2, 0, 1).astype(np.uint32, copy=False)

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='xy')
        Xc = Xc.T
        Yc = Yc.T

        if is_latlon:
            lon_center_2d = Xc.astype(np.float64, copy=False)
            lat_center_2d = Yc.astype(np.float64, copy=False)
        else:
            lon_center_2d, lat_center_2d = density_proj(Xc, Yc, inverse=True)

        gc.collect()

        return H, x_centers, y_centers, lon_center_2d, lat_center_2d, H_count, keep_species, name_species_out

    def get_pixel_mean_depth(self, lons, lats,
                             is_moll, is_latlon,
                             lat_resol, lon_resol):
        """
        Interpolate auxiliary gridded fields to the output cell centers.
        Returns
        h : 2D array
            Water depth at output cell centers [m].
        area : 2D array or None
            Horizontal cell area [m2].
            - None for Mollweide with fixed square pixels, because volume is handled upstream.
            - spherical area for geographic grids
            - constant projected-cell area for projected grids
        active_sediment_layer_thickness : 2D array or None
            Interpolated active sediment layer thickness [m], if available.
        bathy_invalid_mask : 2D bool array
            Invalid-domain mask inherited from the auxiliary bathymetry field after
            interpolating the source mask to the output grid.
            True means the source bathymetry was masked/NaN there and the target cell
            should remain masked.

        Interpolation method
        For both bathymetry and active sediment layer thickness:
          1) interpolate linearly
          2) fill remaining NaNs with nearest neighbour
        This is not conservative.

        The invalid mask from the auxiliary bathymetry is preserved separately and
        re-applied after interpolation so that masked/NaN source regions remain masked
        on the output grid.
        """
        from scipy import interpolate
        import numpy as np
        import gc

        def _interp_linear_then_nearest(x_src, y_src, v_src, x_tgt, y_tgt, field_name):
            x_src = np.asarray(x_src, dtype=float).ravel()
            y_src = np.asarray(y_src, dtype=float).ravel()
            v_src = np.asarray(v_src, dtype=float).ravel()

            valid = np.isfinite(x_src) & np.isfinite(y_src) & np.isfinite(v_src)
            if not np.any(valid):
                raise ValueError(f"No finite source values available for interpolating {field_name!r}.")

            pts = (x_src[valid], y_src[valid])
            vals = v_src[valid]

            out = interpolate.griddata(pts, vals, (x_tgt, y_tgt), method='linear')

            missing = ~np.isfinite(out)
            if np.any(missing):
                out_nearest = interpolate.griddata(pts, vals, (x_tgt, y_tgt), method='nearest')
                out[missing] = out_nearest[missing]

            return out

        def _interp_mask_nearest(x_src, y_src, mask_src, x_tgt, y_tgt, field_name):
            """
            Interpolate a categorical mask to target grid using nearest neighbour.
            True = invalid/masked, False = valid.
            """
            x_src = np.asarray(x_src, dtype=float).ravel()
            y_src = np.asarray(y_src, dtype=float).ravel()
            mask_src = np.asarray(mask_src, dtype=bool).ravel()

            good_xy = np.isfinite(x_src) & np.isfinite(y_src)
            if not np.any(good_xy):
                raise ValueError(f"No finite source coordinates available for interpolating mask {field_name!r}.")

            pts = (x_src[good_xy], y_src[good_xy])
            vals = mask_src[good_xy].astype(np.float64)

            out = interpolate.griddata(pts, vals, (x_tgt, y_tgt), method='nearest')
            return np.asarray(out >= 0.5, dtype=bool)

        # Auxiliary bathymetry grid in lon/lat
        h_grd = np.asarray(self.conc_topo, dtype=float)
        nx = h_grd.shape[0]
        ny = h_grd.shape[1]
        lat_grd = np.asarray(self.conc_lat[:nx, :ny], dtype=float)
        lon_grd = np.asarray(self.conc_lon[:nx, :ny], dtype=float)

        topo_src_invalid = getattr(self, 'conc_topo_source_invalid', None)
        if topo_src_invalid is None:
            topo_src_invalid = np.ma.getmaskarray(self.conc_topo) | ~np.isfinite(h_grd)

        bathy_invalid_mask = _interp_mask_nearest(
            lon_grd, lat_grd, topo_src_invalid,
            lons, lats,
            field_name='sea_floor_depth_below_sea_level mask')
        # Bathymetry: interpolate finite values, then restore invalid source mask on target
        h = _interp_linear_then_nearest(
            lon_grd, lat_grd, h_grd,
            lons, lats,
            field_name='sea_floor_depth_below_sea_level')
        h[bathy_invalid_mask] = np.nan

        # Active sediment layer thickness
        active_sediment_layer_thickness = None
        if hasattr(self, 'conc_active_sediment_layer_thickness') and self.conc_active_sediment_layer_thickness is not None:
            blt_grd = np.asarray(self.conc_active_sediment_layer_thickness[:nx, :ny], dtype=float)

            active_sediment_layer_thickness = _interp_linear_then_nearest(
                lon_grd, lat_grd, blt_grd,
                lons, lats,
                field_name='active_sediment_layer_thickness')

            blt_src_invalid = getattr(self, 'conc_active_sediment_layer_thickness_source_invalid', None)
            if blt_src_invalid is None:
                blt_src_invalid = (
                    np.ma.getmaskarray(self.conc_active_sediment_layer_thickness) |
                    ~np.isfinite(blt_grd))

            blt_invalid_mask = _interp_mask_nearest(
                lon_grd, lat_grd, blt_src_invalid,
                lons, lats,
                field_name='active_sediment_layer_thickness mask')

            active_sediment_layer_thickness[blt_invalid_mask] = np.nan
            active_sediment_layer_thickness[bathy_invalid_mask] = np.nan

        # Release cached intermediate auxiliary grids
        for attr in (
            'conc_lon',
            'conc_lat',
            'conc_topo',
            'conc_topo_source_invalid',
            'conc_active_sediment_layer_thickness',
            'conc_active_sediment_layer_thickness_source_invalid',):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()

        if is_moll:
            return h, None, active_sediment_layer_thickness, bathy_invalid_mask

        if is_latlon:
            Radius = 6.371e6
            lat_resol_rad = np.radians(lat_resol)
            lon_resol_rad = np.radians(lon_resol)
            lat_array_rad = np.radians(lats)

            lat1 = lat_array_rad - (lat_resol_rad / 2.0)
            lat2 = lat_array_rad + (lat_resol_rad / 2.0)

            area = (Radius ** 2) * lon_resol_rad * (np.sin(lat2) - np.sin(lat1))
            return h, area, active_sediment_layer_thickness, bathy_invalid_mask

        area = np.full_like(lats, abs(lat_resol * lon_resol), dtype=np.float64)
        return h, area, active_sediment_layer_thickness, bathy_invalid_mask

    def horizontal_smooth(self, a, n=0, landmask=None, pad_mode="edge", land_value=0.0):
        """
        2D box smoothing (mean over (2n+1)x(2n+1)), no wrap.
        - Ignores NaNs
        - If landmask is provided (True=land), land is excluded from averages
        - Land cells in output are set to land_value (default 0.0)
        """
        import numpy as np
        a = np.asarray(a, dtype=np.float64)
        if n <= 0:
            return a

        n = int(n)
        ydm, xdm = a.shape

        valid = np.isfinite(a)
        lm = None
        if landmask is not None:
            lm = np.asarray(landmask, dtype=bool)
            if lm.shape != a.shape and lm.T.shape == a.shape:
                lm = lm.T
            if lm.shape != a.shape:
                raise ValueError(f"landmask shape {lm.shape} != a shape {a.shape}")
            valid &= ~lm

        a0 = np.where(valid, a, 0.0)
        w0 = valid.astype(np.float64)

        a_pad = np.pad(a0, ((n, n), (n, n)), mode=pad_mode)
        w_pad = np.pad(w0, ((n, n), (n, n)), mode=pad_mode)

        # integral images with leading zeros
        S = np.pad(a_pad, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        W = np.pad(w_pad, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

        k = 2 * n + 1
        num = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
        den = W[k:, k:] - W[:-k, k:] - W[k:, :-k] + W[:-k, :-k]

        out = np.full((ydm, xdm), np.nan, dtype=np.float64)
        ok = den > 0
        out[ok] = num[ok] / den[ok]

        if lm is not None:
            out[lm] = land_value # keep land clean

        return out

    ##### Helpers for calculate_water_sediment_conc
    @staticmethod
    def _property_weight_var(variable):
        """
        Return the density/count variable to use as weights for property_mean-like variables.
        """
        variable = str(variable)
        mapping = {
            "property_mean": "density",
            "property_mean_smooth": "density_smooth",
            "property_mean_avg": "density_avg",
            "property_mean_smooth_avg": "density_smooth_avg",
        }
        return mapping.get(variable, None)

    @staticmethod
    def _water_units(src_da, variable, default="unknown"):
        """
        Tolerant reader for water-like units.
        Priority:
          1) src_da.attrs["units_water"]
          2) new compatibility format: "<water> | <sediment>"
          3) legacy format where water-like units are the text before "(sed ...)"
          4) plain src_da.attrs["units"] for property-like variables
        """
        import re

        variable = str(variable)

        if "density" in variable:
            return "1"

        if "property_mean" in variable:
            units_raw = src_da.attrs.get("units", None)
            return str(units_raw).strip() if units_raw else default

        units_water = src_da.attrs.get("units_water", None)
        if units_water:
            return str(units_water).strip()

        units_raw = src_da.attrs.get("units", None)
        if not units_raw:
            return default

        s = str(units_raw).strip()

        if "|" in s:
            return s.split("|", 1)[0].strip() or default

        # legacy style: water-like units before "(sed ...)"
        s = re.sub(r"\s*/\s*", "/", s)
        s = s.split("(", 1)[0].strip()
        return s or default

    @staticmethod
    def _sed_units(src_da, variable, default="unknown"):
        """
        Tolerant reader for sediment-like units.
        Priority:
          1) src_da.attrs["units_sediment"]
          2) new compatibility format: "<water> | <sediment>"
          3) legacy format "(sed <units>)"
          4) plain src_da.attrs["units"] for property-like variables
        """
        import re

        variable = str(variable)

        if "density" in variable:
            return "1"

        if "property_mean" in variable:
            units_raw = src_da.attrs.get("units", None)
            return str(units_raw).strip() if units_raw else default

        units_sed = src_da.attrs.get("units_sediment", None)
        if units_sed:
            return str(units_sed).strip()

        units_raw = src_da.attrs.get("units", None)
        if not units_raw:
            return default

        s = str(units_raw).strip()

        if "|" in s:
            return s.split("|", 1)[1].strip() or default

        m = re.search(r"\(\s*sed\s*([^)]+)\)", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip() or default

        return default

    @staticmethod
    def _apply_common_attrs(out_da, Sim_description,
                            projection_proj4=None, grid_mapping=None,
                            longitude=None, latitude=None,
                            x=None, y=None,
                            x_units=None, y_units=None):
        """
        Apply common metadata to a derived DataArray.
        - sim_description
        - projection
        - grid_mapping
        - lon_resol / lat_resol only for 1D geographic coords
        - dx / dy only for 1D projected x/y coords
        """
        import numpy as np

        if Sim_description is not None:
            out_da.attrs["sim_description"] = str(Sim_description)

        if projection_proj4 is not None:
            out_da.attrs["projection"] = str(projection_proj4)

        if grid_mapping is not None:
            out_da.attrs["grid_mapping"] = str(grid_mapping)

        if longitude is not None:
            lon_arr = np.asarray(longitude)
            if lon_arr.ndim == 1 and lon_arr.size > 1:
                out_da.attrs["lon_resol"] = f"{np.around(abs(lon_arr[1] - lon_arr[0]), decimals=8)} degrees_east"

        if latitude is not None:
            lat_arr = np.asarray(latitude)
            if lat_arr.ndim == 1 and lat_arr.size > 1:
                out_da.attrs["lat_resol"] = f"{np.around(abs(lat_arr[1] - lat_arr[0]), decimals=8)} degrees_north"

        if x is not None:
            x_arr = np.asarray(x)
            if x_arr.ndim == 1 and x_arr.size > 1:
                xu = str(x_units) if x_units is not None else "m"
                out_da.attrs["dx"] = f"{np.around(abs(x_arr[1] - x_arr[0]), decimals=8)} {xu}"

        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim == 1 and y_arr.size > 1:
                yu = str(y_units) if y_units is not None else "m"
                out_da.attrs["dy"] = f"{np.around(abs(y_arr[1] - y_arr[0]), decimals=8)} {yu}"

        return out_da

    @staticmethod
    def _rename_common_dims(da, time_name=None):
        """
        Normalize only non-spatial/common dimensions.

        Current behavior:
          - avg_time -> time
          - preserve provenance via attrs["source_time_coordinate"] = "avg_time"
        """
        out = da.copy()

        renamed_avg_time = False
        if time_name == "avg_time" and "avg_time" in out.dims:
            out = out.rename({"avg_time": "time"})
            renamed_avg_time = True
        if renamed_avg_time:
            out.attrs["source_time_coordinate"] = "avg_time"

        return out

    @staticmethod
    def _build_unique_rename_map(dims, pairs, context="dimension rename"):
        rename_pairs = [(old, new) for old, new in pairs if old in dims]

        target_to_sources = {}
        for old, new in rename_pairs:
            target_to_sources.setdefault(new, []).append(old)

        duplicate_targets = {
            new: olds for new, olds in target_to_sources.items()
            if len(olds) > 1
        }
        if duplicate_targets:
            raise ValueError(
                f"Ambiguous {context}: multiple source dims map to the same "
                f"target {duplicate_targets}. Input dims are {tuple(dims)}")

        rename_map = dict(rename_pairs)

        for old, new in rename_pairs:
            if new in dims and old != new:
                raise ValueError(
                    f"Cannot rename {old!r} -> {new!r}: "
                    f"{new!r} already exists in dims {tuple(dims)}"
                )
        return rename_map

    def _prepare_rectilinear_output_da(self, da,
                                       lat_var=None, lon_var=None,
                                       lat1d_from_2d=None, lon1d_from_2d=None):
        """
        Prepare a rectilinear geographic output:
          - dimensions: latitude, longitude
          - coordinates: 1D latitude, 1D longitude

        Accepts either:
          1) native 1D lat/lon source coordinates
          2) separable 2D lon/lat converted to 1D vectors
          3) existing 1D geographic coordinates already present on `da`

        Any pre-existing geographic coords named
        ('lat', 'lon', 'latitude', 'longitude') are removed before the final
        canonical 1D coordinates are attached.
        """
        import numpy as np

        out = da.copy()

        def _coord_attrs_or_default(coord, defaults):
            attrs = {}
            if coord is not None and hasattr(coord, "attrs"):
                attrs.update(dict(coord.attrs))
            for k, v in defaults.items():
                attrs.setdefault(k, v)
            return attrs

        def _maybe_get_existing_1d_coord(out_da, names):
            for nm in names:
                if nm in out_da.coords and getattr(out_da.coords[nm], "ndim", 0) == 1:
                    return out_da.coords[nm]
            return None

        # Preserve existing 1D geographic coords as fallback before dropping them
        existing_lat_1d = _maybe_get_existing_1d_coord(out, ("latitude", "lat"))
        existing_lon_1d = _maybe_get_existing_1d_coord(out, ("longitude", "lon"))

        # Remove all existing geographic coords so the output naming is rebuilt cleanly
        stale_geo_coords = [nm for nm in ("lat", "lon", "latitude", "longitude") if nm in out.coords]
        if stale_geo_coords:
            out = out.drop_vars(stale_geo_coords)

        # Rename spatial dims explicitly to rectilinear geographic names
        rename_map = self._build_unique_rename_map(
            out.dims,
            (
                ("lat", "latitude"),
                ("lon", "longitude"),
                ("y", "latitude"),
                ("x", "longitude"),
            ),
            context="rectilinear spatial dimension rename",
        )
        if rename_map:
            out = out.rename(rename_map)

        if "latitude" not in out.dims or "longitude" not in out.dims:
            raise ValueError(
                "Rectilinear geographic output requires dimensions that can be mapped "
                "to ('latitude', 'longitude')."
            )

        lat_vals = None
        lon_vals = None
        lat_attrs = None
        lon_attrs = None

        # Case A: native 1D source coords
        if (lat_var is not None and getattr(lat_var, "ndim", 0) == 1 and
                lon_var is not None and getattr(lon_var, "ndim", 0) == 1):
            lat_vals = np.asarray(lat_var.values)
            lon_vals = np.asarray(lon_var.values)
            lat_attrs = _coord_attrs_or_default(
                lat_var,
                {"standard_name": "latitude", "long_name": "latitude",
                 "units": "degrees_north", "axis": "Y"}
            )
            lon_attrs = _coord_attrs_or_default(
                lon_var,
                {"standard_name": "longitude", "long_name": "longitude",
                 "units": "degrees_east", "axis": "X"}
            )

        # Case B: separable 2D promoted to 1D
        elif lat1d_from_2d is not None and lon1d_from_2d is not None:
            lat_vals = np.asarray(lat1d_from_2d, dtype=float)
            lon_vals = np.asarray(lon1d_from_2d, dtype=float)
            lat_attrs = {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
                "axis": "Y",
            }
            lon_attrs = {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
                "axis": "X",
            }

        # Case C: fallback to existing 1D coords that were already on the input DA
        elif existing_lat_1d is not None and existing_lon_1d is not None:
            lat_vals = np.asarray(existing_lat_1d.values)
            lon_vals = np.asarray(existing_lon_1d.values)
            lat_attrs = _coord_attrs_or_default(
                existing_lat_1d,
                {"standard_name": "latitude", "long_name": "latitude",
                 "units": "degrees_north", "axis": "Y"}
            )
            lon_attrs = _coord_attrs_or_default(
                existing_lon_1d,
                {"standard_name": "longitude", "long_name": "longitude",
                 "units": "degrees_east", "axis": "X"}
            )

        else:
            raise ValueError(
                "Could not build rectilinear geographic output: need either "
                "1D latitude/longitude source coordinates, separable 2D lon/lat, "
                "or existing 1D geographic coordinates on the input DataArray."
            )

        if out.sizes["latitude"] != lat_vals.size:
            raise ValueError(
                f"Latitude size mismatch: data has {out.sizes['latitude']} rows, "
                f"but derived latitude has {lat_vals.size} values."
            )
        if out.sizes["longitude"] != lon_vals.size:
            raise ValueError(
                f"Longitude size mismatch: data has {out.sizes['longitude']} cols, "
                f"but derived longitude has {lon_vals.size} values."
            )

        out = out.assign_coords(
            latitude=("latitude", lat_vals),
            longitude=("longitude", lon_vals),
        )
        out.coords["latitude"].attrs = lat_attrs
        out.coords["longitude"].attrs = lon_attrs

        return out


    @staticmethod
    def _prepare_curvilinear_output_da(da,
                                       y_var=None, x_var=None,
                                       lon2d=None, lat2d=None,
                                       separable=False,
                                       Verbose=True):
        """
        Prepare a projected/curvilinear output:
          - dimensions: y, x
          - coordinates: 1D y, 1D x when available
          - optional auxiliary coordinates: 2D lon, 2D lat

        Any pre-existing geographic coords named
        ('lat', 'lon', 'latitude', 'longitude') are removed before the intended
        auxiliary 2D lon/lat coordinates are attached.
        """
        import numpy as np
        import xarray as xr

        out = da.copy()

        def _coord_attrs_or_default(coord, defaults):
            attrs = {}
            if coord is not None and hasattr(coord, "attrs"):
                attrs.update(dict(coord.attrs))
            for k, v in defaults.items():
                attrs.setdefault(k, v)
            return attrs

        if "lat" in out.dims or "lon" in out.dims or "latitude" in out.dims or "longitude" in out.dims:
            raise ValueError(
                "Curvilinear/projected output helper expects y/x dimensions. "
                "Rectilinear geographic inputs should be handled by _prepare_rectilinear_output_da(...)."
            )

        if "y" not in out.dims or "x" not in out.dims:
            raise ValueError(
                "Curvilinear/projected mode requires data dimensions containing both 'y' and 'x'."
            )

        stale_geo_coords = [nm for nm in ("lat", "lon", "latitude", "longitude") if nm in out.coords]
        if stale_geo_coords:
            out = out.drop_vars(stale_geo_coords)

        if y_var is not None and getattr(y_var, "ndim", 0) == 1:
            y_vals = np.asarray(y_var.values)
            if out.sizes["y"] != y_vals.size:
                raise ValueError(
                    f"y size mismatch: data has {out.sizes['y']} rows, "
                    f"but source y has {y_vals.size} values."
                )
            out = out.assign_coords(y=("y", y_vals))
            out.coords["y"].attrs = _coord_attrs_or_default(
                y_var,
                {"standard_name": "projection_y_coordinate",
                 "long_name": "y", "units": "m", "axis": "Y"}
            )

        if x_var is not None and getattr(x_var, "ndim", 0) == 1:
            x_vals = np.asarray(x_var.values)
            if out.sizes["x"] != x_vals.size:
                raise ValueError(
                    f"x size mismatch: data has {out.sizes['x']} cols, "
                    f"but source x has {x_vals.size} values."
                )
            out = out.assign_coords(x=("x", x_vals))
            out.coords["x"].attrs = _coord_attrs_or_default(
                x_var,
                {"standard_name": "projection_x_coordinate",
                 "long_name": "x", "units": "m", "axis": "X"}
            )

        if lon2d is not None and lat2d is not None:
            if lon2d.shape != lat2d.shape:
                raise ValueError(f"lon2d/lat2d shape mismatch: {lon2d.shape} vs {lat2d.shape}")

            if tuple(lon2d.dims) != tuple(lat2d.dims):
                raise ValueError(f"lon2d/lat2d dims mismatch: {lon2d.dims} vs {lat2d.dims}")

            if tuple(lon2d.dims) != ("y", "x"):
                raise ValueError(
                    f"Expected 2D lon/lat auxiliary coords on dims ('y', 'x'), got {lon2d.dims}"
                )

            if lon2d.shape != (out.sizes["y"], out.sizes["x"]):
                raise ValueError(
                    f"2D longitude shape mismatch: expected {(out.sizes['y'], out.sizes['x'])}, "
                    f"got {lon2d.shape}"
                )

            lon_aux = xr.DataArray(
                np.asarray(lon2d.values),
                dims=("y", "x"),
                attrs={
                    "standard_name": "longitude",
                    "long_name": "longitude of cell center",
                    "units": "degrees_east",
                    **dict(getattr(lon2d, "attrs", {})),
                },
            )
            lat_aux = xr.DataArray(
                np.asarray(lat2d.values),
                dims=("y", "x"),
                attrs={
                    "standard_name": "latitude",
                    "long_name": "latitude of cell center",
                    "units": "degrees_north",
                    **dict(getattr(lat2d, "attrs", {})),
                },
            )
            out = out.assign_coords(lon=lon_aux, lat=lat_aux)

        if Verbose and lon2d is not None and lat2d is not None and not separable:
            print("Preserving projected/curvilinear y/x grid with 2D lon/lat auxiliary coordinates")

        return out

    def prepare_derived_output_da(self, da, src_ds, time_name=None,
                                  spatial_mode="auto",
                                  rectilinear_tol=1e-10,
                                  Verbose=True):
        """
        Prepare a derived output DataArray with explicit handling of 3 spatial cases.

        spatial_mode:
          - "auto": choose rectilinear when possible, otherwise curvilinear/projected
          - "rectilinear": require a true/separable geographic lon/lat grid
          - "curvilinear": preserve y/x and attach 2D lon/lat auxiliaries when available

        Cases handled:
          1) Rectilinear geographic: 1D latitude/longitude
          2) Projected/curvilinear: y/x with optional 2D lon/lat
          3) Legacy 2D lon/lat that are separable into 1D vectors
        """
        import numpy as np

        spatial_mode = str(spatial_mode).lower()
        if spatial_mode not in {"auto", "rectilinear", "curvilinear"}:
            raise ValueError("spatial_mode must be one of: 'auto', 'rectilinear', 'curvilinear'")

        out = self._rename_common_dims(da, time_name=time_name)

        def _get_src_var(names):
            for name in names:
                if name in src_ds.coords:
                    return src_ds.coords[name]
                if name in src_ds.data_vars:
                    return src_ds[name]
                if name in src_ds.variables:
                    return src_ds[name]
            return None

        def _is_separable_2d(lon_da, lat_da, tol):
            lon2d = np.asarray(lon_da.values, dtype=float)
            lat2d = np.asarray(lat_da.values, dtype=float)

            if lon2d.ndim != 2 or lat2d.ndim != 2 or lon2d.shape != lat2d.shape:
                return False, None, None

            lon1d = lon2d[0, :]
            lat1d = lat2d[:, 0]

            lon_ok = np.allclose(
                lon2d,
                np.broadcast_to(lon1d[None, :], lon2d.shape),
                rtol=0.0, atol=tol, equal_nan=True
            )
            lat_ok = np.allclose(
                lat2d,
                np.broadcast_to(lat1d[:, None], lat2d.shape),
                rtol=0.0, atol=tol, equal_nan=True
            )
            return bool(lon_ok and lat_ok), lon1d, lat1d

        lat_var = _get_src_var(["latitude", "lat"])
        lon_var = _get_src_var(["longitude", "lon"])
        y_var = _get_src_var(["y"])
        x_var = _get_src_var(["x"])

        lon2d = lon_var if (lon_var is not None and getattr(lon_var, "ndim", 0) == 2) else None
        lat2d = lat_var if (lat_var is not None and getattr(lat_var, "ndim", 0) == 2) else None

        separable = False
        lon1d_from_2d = None
        lat1d_from_2d = None
        if lon2d is not None and lat2d is not None:
            separable, lon1d_from_2d, lat1d_from_2d = _is_separable_2d(
                lon2d, lat2d, rectilinear_tol
            )

        has_native_1d_geo = (
            lat_var is not None and getattr(lat_var, "ndim", 0) == 1 and
            lon_var is not None and getattr(lon_var, "ndim", 0) == 1
        )

        already_rectilinear_geo_dims = (
            ("latitude" in out.dims or "lat" in out.dims) and
            ("longitude" in out.dims or "lon" in out.dims)
        )

        if spatial_mode == "auto":
            if has_native_1d_geo or separable:
                target_mode = "rectilinear"
            else:
                target_mode = "curvilinear"
        else:
            target_mode = spatial_mode

        if target_mode == "rectilinear":
            return self._prepare_rectilinear_output_da(
                out,
                lat_var=lat_var,
                lon_var=lon_var,
                lat1d_from_2d=lat1d_from_2d if separable else None,
                lon1d_from_2d=lon1d_from_2d if separable else None,
            )

        # target_mode == "curvilinear"
        # Preserve already-rectilinear geographic inputs as rectilinear rather than
        # forcing them into y/x.
        if already_rectilinear_geo_dims:
            return self._prepare_rectilinear_output_da(
                out,
                lat_var=lat_var,
                lon_var=lon_var,
            )

        return self._prepare_curvilinear_output_da(
            out,
            y_var=y_var,
            x_var=x_var,
            lon2d=lon2d,
            lat2d=lat2d,
            separable=separable,
            Verbose=Verbose,
        )

    @staticmethod
    def _species_dict(string, allowed=None):
        """
        Parse "...\\nspecie 0:LMM 1:Humic colloid 2:Particle reversible ..."
        into {"LMM": 0, "Humic colloid": 1, ...}

        If allowed is provided (list/set), names are validated against it.
        """
        import re
        if allowed is None:
             allowed = ["LMM", "LMMcation", "LMManion",
                        "Colloid", "Humic colloid",
                        "Polymer", "Particle reversible",
                        "Particle slowly reversible",
                        "Particle irreversible",
                        "dissolved", "DOC", "SPM",
                        "Sediment reversible",
                        "Sediment slowly reversible",
                     	"Sediment buried",
                        "Sediment irreversible",
                        "sediment", "buried"]

        tail = string.split("\n", 1)[1] if "\n" in string else string
        pairs = re.findall(r'(\d+)\s*:\s*(.*?)(?=\s+\d+\s*:|$)', tail)

        out = {}
        allowed_set = set(a.strip() for a in allowed) if allowed is not None else None

        for idx_str, raw_name in pairs:
            name = raw_name.strip()
            if allowed_set is not None and name not in allowed_set:
                raise ValueError("Unknown species name: {!r}".format(name))
            out[name] = int(idx_str)

        return out

    @staticmethod
    def _format_species_pairs(pairs):
        # pairs is list of (name, idx)
        return ", ".join(f'"{name}": {idx}' for name, idx in pairs)

    @staticmethod
    def _excluded_pairs(specie_ids_num, excluded_names, TOT_Conc=None):
        """
        Return [(name, idx), ...] for excluded species that are present in specie_ids_num
        and (if TOT_Conc provided) whose idx exists in TOT_Conc.specie coords.
        """
        coord_vals = None
        if TOT_Conc is not None and "specie" in TOT_Conc.coords:
            coord_vals = set(TOT_Conc["specie"].values)

        pairs = []
        for name in excluded_names:
            idx = specie_ids_num.get(name)
            if idx is None:
                continue
            if coord_vals is not None and idx not in coord_vals:
                continue
            pairs.append((name, idx))
        return pairs

    @staticmethod
    def _water_column_conc(TOT_Conc, specie_ids_num, water_species, Verbose=False):
        """
        Compute the concentration in the water column as sum of all species listed in `water_species` if:
          - the name exists in `specie_ids_num` (name -> index), and
          - that index is present in TOT_Conc.specie coordinate.
        Returns an xarray DataArray (or None if nothing matched), and the list of included species.
        """

        out = None
        included = []
        specie_coord_vals = set(TOT_Conc["specie"].values) if "specie" in TOT_Conc.coords else None

        for name in water_species:
            idx = specie_ids_num.get(name)
            if idx is None:
                if Verbose:
                    print(f"Skipping {name!r}: not found in specie_ids_num")
                continue

            if specie_coord_vals is not None and idx not in specie_coord_vals:
                if Verbose:
                    print(f"Skipping {name!r}: specie index {idx} not in TOT_Conc.specie")
                continue

            da = TOT_Conc.sel(specie=idx)
            out = da if out is None else (out + da)
            included.append((name, idx))

        if out is None:
            if Verbose:
                print("No water species matched; returning None.")
            return None, []
        if Verbose:
            print("Included water species:", ", ".join(name for name, _ in included))
        # Drop singleton depth if present
        if "depth" in out.dims and out.sizes.get("depth", 0) == 1:
            out = out.isel(depth=0, drop=True)

        return out, included

    @staticmethod
    def _sediment_conc_sum(TOT_Conc, specie_ids_num, sed_species,
                           Verbose=False,
                           debug_check_single_depth=False,
                           nonzero_tol=0.0):
        """
        Compute a depth-collapsed sediment concentration as the sum of sediment species.

        Assumption:
          sediment concentration occupies only one depth bin for each horizontal cell.
        If debug_check_single_depth=True, an assertion is performed before the depth collapse.

        Returns:
          (DataArray or None, included_species_list)
        """
        import xarray as xr
        import numpy as np

        specie_coord_vals = set(TOT_Conc["specie"].values) if "specie" in TOT_Conc.coords else None

        da_sed = None
        included = []

        for name in sed_species:
            idx = specie_ids_num.get(name)
            if idx is None:
                if Verbose:
                    print(f"Skipping {name!r}: not found in specie_ids_num")
                continue

            if specie_coord_vals is not None and idx not in specie_coord_vals:
                if Verbose:
                    print(f"Skipping {name!r}: specie index {idx} not in TOT_Conc.specie")
                continue

            part = TOT_Conc.sel(specie=idx)
            da_sed = part if da_sed is None else (da_sed + part)
            included.append((name, idx))

        if da_sed is None:
            if Verbose:
                print("No sediment species matched; returning None.")
            return None, []

        if Verbose:
            print("Included sediment species:", ", ".join(name for name, _ in included))

        mask = np.isnan(TOT_Conc)

        if "depth" in da_sed.dims and debug_check_single_depth:
            occupied = xr.where(
                np.isfinite(da_sed) & (np.abs(da_sed) > float(nonzero_tol)),
                1, 0
            ).sum(dim="depth")
            if bool((occupied > 1).any()):
                raise AssertionError(
                    "Sediment concentration occupies more than one depth bin in at least one horizontal cell.")

        if "depth" in da_sed.dims:
            if Verbose:
                print("depth included in sediment DA -> summing over depth")
            da_sed = da_sed.sum(dim="depth")
            da_sed = xr.where(mask.isel(specie=0, depth=0), np.nan, da_sed)
        else:
            da_sed = xr.where(mask.isel(specie=0), np.nan, da_sed)

        return da_sed, included

    @staticmethod
    def _sim_description_attr(src_da, Sim_description):
        """
        Resolve the simulation description string to attach to derived outputs.
          1) If `src_da` (typically an xarray DataArray from the concentration file)
             has attribute `sim_description`, return it.
          2) Else, if `Sim_description` argument is provided, return it as `str(...)`.
          3) Else return None.

        src_da : xarray.DataArray, Source data array from which metadata may be inherited.
        Sim_description : Any, User-provided simulation description.
        """
        sim_desc = src_da.attrs.get("sim_description", None)
        if sim_desc is not None:
            return sim_desc
        if Sim_description is not None:
            return str(Sim_description)
        return None

    @staticmethod
    def _base_long_name(src_da, Chemical_name, variable):
        """
        Build a base `long_name` prefix for derived water/sediment products using the `long_name` attribute,
        otherwise, it constructs a fallback base name using the provided chemical name and  and variable label: "<Chemical_name> <variable>".

        src_da:         xarray.DataArray, Source data array from which metadata may be inherited.
        Chemical_name:  str or None, Chemical name used in fallback naming.
        variable:       str, variable name (e.g., "concentration", "density", ...).
        """
        long_name = src_da.attrs.get("long_name", None)
        if long_name is not None:
            return str(long_name).split("specie")[0].strip()
        return (Chemical_name or "") + f" {variable}"

    def specie_ids_num_from_ds(self, DS, TOT_Conc=None, legacy_alias=None):
        """
        Build {species_name: specie_coord_value} from a dataset.

        Normalization is applied consistently to dataset-derived names using legacy_alias.
        Both the original and canonical names are accepted as lookup keys when possible.
        """
        import numpy as np

        if legacy_alias is None:
            legacy_alias = {}

        def _clean_name(x) -> str:
            if isinstance(x, (bytes, bytearray, np.bytes_)):
                s = bytes(x).decode("utf-8", errors="ignore")
            else:
                s = str(x)
            s = s.replace("\x00", "").strip()
            if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
                s = s[2:-1].replace("\x00", "").strip()
            return s

        def _decode_specie_name_array(a):
            if np.ma.isMaskedArray(a):
                a = a.filled(0)

            names = []

            if a.ndim == 1:
                if a.dtype.kind == "S":
                    names = [_clean_name(s) for s in a.tolist()]
                elif a.dtype.kind == "U":
                    names = [_clean_name(s) for s in a.tolist()]
                else:
                    names = [_clean_name(x) for x in a.tolist()]

            elif a.ndim == 2:
                if a.dtype.kind == "S":
                    for row in a:
                        names.append(_clean_name(row.tobytes()))
                elif a.dtype.kind == "U":
                    for row in a:
                        names.append(_clean_name("".join(row)))
                elif a.dtype.kind in ("i", "u"):
                    a8 = a.astype(np.uint8, copy=False)
                    for row in a8:
                        names.append(_clean_name(bytes(row.tolist())))
                else:
                    for row in a:
                        if len(row) == 0:
                            names.append("")
                        else:
                            first = row[0]
                            if isinstance(first, (bytes, np.bytes_)):
                                b = b"".join([x if isinstance(x, (bytes, np.bytes_)) else bytes([int(x)]) for x in row])
                                names.append(_clean_name(b))
                            elif isinstance(first, (int, np.integer)):
                                names.append(_clean_name(bytes([int(x) & 0xFF for x in row])))
                            else:
                                names.append(_clean_name("".join([str(x) for x in row])))
            else:
                names = [_clean_name(x) for x in a.reshape(-1).tolist()]

            return names

        if "specie_name" in DS.variables:
            a = np.asarray(DS["specie_name"].values)
            names = _decode_specie_name_array(a)

            if "specie" in DS.coords:
                specie_vals = np.asarray(DS["specie"].values)
            else:
                specie_vals = np.arange(len(names), dtype=int)

            if len(names) != len(specie_vals):
                raise ValueError(
                    f"Length mismatch between specie_name ({len(names)}) and specie coord ({len(specie_vals)})")

            out = {}
            for coord_val, raw_name in zip(specie_vals, names):
                if not raw_name:
                    continue

                canonical = legacy_alias.get(raw_name, raw_name)
                for key in {raw_name, canonical}:
                    if key in out and out[key] != int(coord_val):
                        raise ValueError(
                            f"Species alias collision for {key!r}: {out[key]} vs {int(coord_val)}")
                    out[key] = int(coord_val)

            return out

        long_name = TOT_Conc.attrs.get("long_name", None) if TOT_Conc is not None else None
        if long_name is not None:
            raw = self._species_dict(long_name)
            out = {}
            for name, idx in raw.items():
                name_clean = _clean_name(name)
                canonical = legacy_alias.get(name_clean, name_clean)
                for key in {name_clean, canonical}:
                    if key in out and out[key] != int(idx):
                        raise ValueError(
                            f"Species alias collision in long_name parsing for {key!r}: {out[key]} vs {int(idx)}")
                    out[key] = int(idx)
            return out

        raise ValueError("Cannot build species mapping: missing specie_name and long_name.")

    def fallback_specie_ids_num_from_flags(self):
        """
        Build fallback specie_ids_num using ONLY:
          - transfer_setup = self.get_config('chemical:transfer_setup')
          - slowly_fraction flag (chemical:slowly_fraction)
          - irreversible_fraction flag (chemical:irreversible_fraction)
        Returns:
            dict: {species_name: idx} matching the canonical build order
        """
        transfer_setup = self.get_config("chemical:transfer_setup")
        slowly = bool(self.get_config("chemical:slowly_fraction"))
        irrev  = bool(self.get_config("chemical:irreversible_fraction"))

        # Canonical order (must match self.name_species builder)
        order = [
            "LMM", "LMMcation", "LMManion",
            "Colloid", "Humic colloid",
            "Polymer",
            "Particle reversible", "Particle slowly reversible", "Particle irreversible",
            "Sediment reversible", "Sediment slowly reversible", "Sediment buried",
            "Sediment irreversible",]
        # Base sets per transfer_setup (minimum guaranteed pools)
        base = {
            "metals": {
                "LMM",
                "Particle reversible",
                "Sediment reversible",
                "Sediment buried",},
            "organics": {
                "LMM",
                "Humic colloid",
                "Particle reversible",
                "Sediment reversible",
                "Sediment buried",},
            "137Cs_rev": {
                "LMM",
                "Particle reversible",
                "Sediment reversible",},
            "Sandnesfj_Al": {
                "LMMcation",
                "LMManion",
                "Humic colloid",
                "Polymer",
                "Particle reversible",
                "Sediment reversible",
            },}

        if transfer_setup == "custom":
            raise ValueError("Cannot build fallback for transfer_setup='custom' from flags alone.")
        if transfer_setup not in base:
            raise ValueError(f"Unknown transfer_setup: {transfer_setup!r}")

        species = set(base[transfer_setup])
        # Slowly pools exist only for metals/organics setups and require BOTH particle+sediment slow
        if slowly and transfer_setup in ("metals", "organics"):
            species.add("Particle slowly reversible")
            species.add("Sediment slowly reversible")
        # Irreversible pools require both particle+sediment irreversible
        if irrev and transfer_setup in ("metals", "organics"):
            species.add("Particle irreversible")
            species.add("Sediment irreversible")

        # Build list+mapping in canonical order
        name_species_out = [s for s in order if s in species]
        specie_ids_num = {name: i for i, name in enumerate(name_species_out)}
        return specie_ids_num

    @staticmethod
    def _normalize_species_names(names, legacy_alias=None):
        """
        Normalize a list of species names:
          - drop None
          - strip whitespace
          - map legacy aliases to canonical names
          - preserve order
          - deduplicate
        """
        if legacy_alias is None:
            legacy_alias = {}

        if names is None:
            return []
        out = []
        seen = set()
        for name in names:
            if name is None:
                continue
            name_norm = str(name).strip()
            if not name_norm:
                continue
            name_norm = legacy_alias.get(name_norm, name_norm)
            if name_norm not in seen:
                out.append(name_norm)
                seen.add(name_norm)
        return out

    @staticmethod
    def _species_weighted_mean(TOT_Value, TOT_Count, specie_ids_num, species_names,
                               Verbose=False,
                               collapse_depth=False,
                               debug_check_single_depth=False):
        """
        Count-weighted aggregation across species:
            out = sum(value_i * count_i) / sum(count_i)

        Only species present both in specie_ids_num and in the specie coordinate are used.
        Cells with zero total count are returned as NaN.

        Parameters
        ----------
        TOT_Value : xarray.DataArray
            Value field with a specie dimension, e.g. property_mean*
        TOT_Count : xarray.DataArray
            Count field with the same specie axis, e.g. density or density_avg
        specie_ids_num : dict
            Mapping {species_name: specie_index}
        species_names : list
            Species names to aggregate
        collapse_depth : bool
            If True, collapse depth after weighted aggregation. Intended for sediment.
        debug_check_single_depth : bool
            If True and collapse_depth=True, assert that the aggregated count support
            occupies at most one depth bin per horizontal cell.
        """
        import numpy as np
        import xarray as xr

        value_specie_vals = set(TOT_Value["specie"].values) if "specie" in TOT_Value.coords else None
        count_specie_vals = set(TOT_Count["specie"].values) if "specie" in TOT_Count.coords else None

        num = None
        den = None
        included = []

        for name in species_names:
            idx = specie_ids_num.get(name)
            if idx is None:
                if Verbose:
                    print(f"Skipping {name!r}: not found in specie_ids_num")
                continue

            if value_specie_vals is not None and idx not in value_specie_vals:
                if Verbose:
                    print(f"Skipping {name!r}: specie index {idx} not in TOT_Value.specie")
                continue

            if count_specie_vals is not None and idx not in count_specie_vals:
                if Verbose:
                    print(f"Skipping {name!r}: specie index {idx} not in TOT_Count.specie")
                continue

            val = TOT_Value.sel(specie=idx)
            cnt = TOT_Count.sel(specie=idx)

            valid = np.isfinite(val) & np.isfinite(cnt) & (cnt > 0)
            num_i = xr.where(valid, val * cnt, 0.0)
            den_i = xr.where(valid, cnt, 0.0)

            num = num_i if num is None else (num + num_i)
            den = den_i if den is None else (den + den_i)
            included.append((name, idx))

        if num is None or den is None:
            if Verbose:
                print("No weighted species matched; returning None.")
            return None, []

        if Verbose:
            print("Included weighted species:", ", ".join(name for name, _ in included))

        if collapse_depth and "depth" in den.dims:
            if debug_check_single_depth:
                occupied = xr.where(den > 0, 1, 0).sum(dim="depth")
                if bool((occupied > 1).any()):
                    raise AssertionError(
                        "Weighted sediment support occupies more than one depth bin "
                        "in at least one horizontal cell."
                    )

            num = num.sum(dim="depth")
            den = den.sum(dim="depth")

        out = xr.where(den > 0, num / den, np.nan)

        if (not collapse_depth) and "depth" in out.dims and out.sizes.get("depth", 0) == 1:
            out = out.isel(depth=0, drop=True)

        return out, included

    def calculate_water_sediment_conc(self,
                                   File_Path,
                                   File_Name,
                                   File_Path_out,
                                   Chemical_name,
                                   Origin_marker_name,
                                   File_Name_out = None,
                                   variables = None,
                                   Concentration_file = None,
                                   Excluded_species = None,
                                   Sim_description=None,
                                   Save_files = True,
                                   Return_datasets = True,
                                   spatial_mode="auto",
                                   which_conc = "both",
                                   debug_check_single_depth=False,
                                   Verbose = True,):
        """
        Sum concentration/property/density by species into water and sediment arrays.

        DataSets can be returned or saved as netCDF files.
        Results can be used as inputs by "seed_from_NETCDF" when the chosen
        spatial_mode is compatible with the expected target workflow.

        Concentration_file:    "write_netcdf_chemical_density_map" output if already loaded (original or after regrid_conc)
        File_Path:             string, path of "write_netcdf_chemical_density_map" output
        File_Name:             string, name of "write_netcdf_chemical_density_map" output
        File_Name_out:         string, suffix of wat/sed output files
        File_Path_out:         string, path where created concentration files will be saved, must end with "/"
        Chemical_name:         string, name of modelled chemical
        Origin_marker_name:    string, name of source indicated by "origin_marker" parameter
        variables:             list, variables to process. If None, all supported variables are considered.
        Excluded_species:      dict, {"water": [...], "sediment":[...]} lists of names of species to be excluded from water column or sediment concentration
        Sim_description:       string, description of simulation to be included in output metadata
        Save_files:            boolean, if True saves outputs to disk, otherwise return xr.Datasets
        Return_datasets:       boolean, if True return xr.Datasets
        which_conc:            string, one of "water" (compute only water concentration) "sediment" compute only sediment concentration, or "both".
        spatial_mode:          string, one of "auto", "rectilinear", "curvilinear". Passed to prepare_derived_output_da(...).
        debug_check_single_depth: bool, If True, assert that sediment concentration occupies at most one depth
                               bin per horizontal cell before depth collapsing.
        Verbose:               boolean, if True prints progress messages

        For property_mean-like variables, species aggregation is count-weighted using:
          density for property_mean
          density_smooth for property_mean_smooth
          density_avg for property_mean_avg
          density_smooth_avg for property_mean_smooth_avg
        """
        from datetime import datetime
        import numpy as np
        import xarray as xr
        import time

        processed_variables = []
        which_conc = str(which_conc).lower()
        if which_conc not in {"water", "sediment", "both"}:
            raise ValueError("which_conc must be one of: 'water', 'sediment', 'both'")

        do_water = which_conc in {"water", "both"}
        do_sediment = which_conc in {"sediment", "both"}

        water_species = [
            "LMM", "LMMcation", "LMManion",
            "Colloid", "Humic colloid",
            "Polymer", "Particle reversible",
            "Particle slowly reversible",
            "Particle irreversible",
        ]
        sed_species = [
            "Sediment reversible",
            "Sediment slowly reversible",
            "Sediment buried",
            "Sediment irreversible",
        ]
        legacy_alias = {
            "dissolved": "LMM",
            "DOC": "Humic colloid",
            "SPM": "Particle reversible",
            "sediment": "Sediment reversible",
            "buried": "Sediment buried",
        }

        if Excluded_species is None:
            Excluded_species = {"water": [None], "sediment": [None]}

        excluded_water_names = self._normalize_species_names(
            Excluded_species.get("water", []),
            legacy_alias=legacy_alias,
        )
        excluded_sed_names = self._normalize_species_names(
            Excluded_species.get("sediment", Excluded_species.get("sed", [])),
            legacy_alias=legacy_alias,
        )

        excluded_set_w = set(excluded_water_names)
        removed_water = [sp for sp in water_species if sp in excluded_set_w]
        water_species = [sp for sp in water_species if sp not in excluded_set_w]

        excluded_set_s = set(excluded_sed_names)
        removed_sed = [sp for sp in sed_species if sp in excluded_set_s]
        sed_species = [sp for sp in sed_species if sp not in excluded_set_s]

        if Verbose and removed_water:
            print("Excluded water species:", ", ".join(removed_water))
        if Verbose and removed_sed:
            print("Excluded sediment species:", ", ".join(removed_sed))

        _t0 = time.perf_counter()
        _start_wall = datetime.now()

        def log(msg: str) -> None:
            elapsed_s = time.perf_counter() - _t0
            h, rem = divmod(int(elapsed_s), 3600)
            m, s = divmod(rem, 60)
            print(
                f"{msg} | elapsed {h:02d}:{m:02d}:{s:02d} "
                f"(started {_start_wall.strftime('%Y-%m-%d %H:%M:%S')})"
            )

        if Concentration_file is None and File_Path is not None and File_Name is not None:
            if Verbose:
                log("Loading Concentration_file from File_Path")
            DS = xr.open_dataset(File_Path + File_Name)
        elif Concentration_file is not None:
            DS = Concentration_file
        else:
            raise ValueError("Incorrect file or file/path not specified")

        if not Save_files and not Return_datasets:
            raise ValueError("Nothing to do: set Save_files=True and/or Return_datasets=True.")

        supported_vars = [
            "concentration", "concentration_avg",
            "concentration_smooth", "concentration_smooth_avg",
            "density", "density_avg",
            "density_smooth", "density_smooth_avg",
            "property_mean", "property_mean_avg",
            "property_mean_smooth", "property_mean_smooth_avg",]

        if not any(var in DS.data_vars for var in supported_vars):
            raise ValueError("No valid variables found in dataset")

        if "time" in DS.dims:
            time_name = "time"
        elif "avg_time" in DS.dims:
            time_name = "avg_time"
        else:
            raise ValueError("No time or avg_time dimension found in dataset")

        variable_ls = supported_vars if variables is None else list(variables)

        sum_vars_wat_dict = {}
        sum_vars_sed_dict = {}
        first_var = True

        def _alias_and_filter(species_list, specie_ids_num, legacy_alias_map):
            out = []
            seen = set()
            for s in species_list:
                s2 = legacy_alias_map.get(s, s)
                if s2 in specie_ids_num and s2 not in seen:
                    out.append(s2)
                    seen.add(s2)
            return out

        def _copy_source_metadata(dst_ds, src_ds):
            for attr_name in (
                "Conventions", "title", "source", "history",
                "actual_density_proj_str", "weight_name", "weight_mode",
                "comment", "phase_units_metadata_mode",):
                if attr_name in src_ds.attrs:
                    dst_ds.attrs[attr_name] = src_ds.attrs[attr_name]

        def _copy_support_vars(dst_ds, src_ds, extra_grid_mapping_names=None):
            support_vars = {
                "crs", "topo", "land", "domain_invalid",
                "active_sediment_layer_thickness", "area", "volume",
                "depth", "depth_bounds", "depth_model_top", "depth_model_bounds",
                "specie_name", "specie_original_id", "specie_phase",
                "x", "y", "lon", "lat",}
            if extra_grid_mapping_names:
                support_vars |= {gm for gm in extra_grid_mapping_names if gm}

            for name in support_vars:
                if name in src_ds.variables and name not in dst_ds.variables and name not in dst_ds.coords:
                    dst_ds[name] = src_ds[name]

        def _coord_getter(da_obj, name):
            if name in da_obj.coords:
                return da_obj.coords[name]
            return None

        for variable in variable_ls:
            if variable not in DS.data_vars or variable == "topo":
                continue

            if Verbose:
                print(variable)

            var_wat_name = variable + "_wat"
            var_sed_name = variable + "_sed"

            TOT_Conc = DS[variable]
            weight_var = self._property_weight_var(variable)
            TOT_Weights = None

            if weight_var is not None:
                if weight_var not in DS.data_vars:
                    raise ValueError(
                        f"Variable {variable!r} requires weight/count variable {weight_var!r}, "
                        "but it is not present in the dataset."
                    )
                TOT_Weights = DS[weight_var]

            try:
                specie_ids_num = self.specie_ids_num_from_ds(
                    DS, TOT_Conc, legacy_alias=legacy_alias
                )
            except (ValueError, KeyError, AttributeError) as e:
                if Verbose:
                    print(f"Could not derive species mapping from dataset metadata: {e}")
                    print("Trying config-derived fallback species mapping.")
                try:
                    specie_ids_num = self.fallback_specie_ids_num_from_flags()
                except ValueError as e2:
                    raise ValueError(
                        f"Species mapping failed from dataset metadata ({e}) "
                        f"and fallback mapping also failed ({e2})."
                    ) from e2

            water_species_eff = _alias_and_filter(water_species, specie_ids_num, legacy_alias)
            sed_species_eff = _alias_and_filter(sed_species, specie_ids_num, legacy_alias)

            DA_Conc_array_wat = None
            DA_Conc_array_sed = None
            included_wat = []
            included_sed = []

            if weight_var is None:
                if do_water:
                    if Verbose:
                        log("Running sum of water concentration/density")
                    DA_Conc_array_wat, included_wat = self._water_column_conc(
                        TOT_Conc=TOT_Conc,
                        specie_ids_num=specie_ids_num,
                        water_species=water_species_eff,
                        Verbose=Verbose,
                    )

                if do_sediment:
                    if Verbose:
                        log("Running sum of sediment concentration/density")
                    DA_Conc_array_sed, included_sed = self._sediment_conc_sum(
                        TOT_Conc=TOT_Conc,
                        specie_ids_num=specie_ids_num,
                        sed_species=sed_species_eff,
                        Verbose=Verbose,
                        debug_check_single_depth=debug_check_single_depth,
                    )

                aggregation_kind = "sum"
            else:
                if do_water:
                    if Verbose:
                        log(f"Running count-weighted water aggregation using {weight_var}")
                    DA_Conc_array_wat, included_wat = self._species_weighted_mean(
                        TOT_Value=TOT_Conc,
                        TOT_Count=TOT_Weights,
                        specie_ids_num=specie_ids_num,
                        species_names=water_species_eff,
                        Verbose=Verbose,
                        collapse_depth=False,
                        debug_check_single_depth=False,
                    )

                if do_sediment:
                    if Verbose:
                        log(f"Running count-weighted sediment aggregation using {weight_var}")
                    DA_Conc_array_sed, included_sed = self._species_weighted_mean(
                        TOT_Value=TOT_Conc,
                        TOT_Count=TOT_Weights,
                        specie_ids_num=specie_ids_num,
                        species_names=sed_species_eff,
                        Verbose=Verbose,
                        collapse_depth=True,
                        debug_check_single_depth=debug_check_single_depth,
                    )

                aggregation_kind = "count_weighted_mean"

            missing = []
            if do_water and DA_Conc_array_wat is None:
                missing.append(f"DA_Conc_array_wat (water_species={water_species_eff})")
            if do_sediment and DA_Conc_array_sed is None:
                missing.append(f"DA_Conc_array_sed (sed_species={sed_species_eff})")

            if missing:
                raise ValueError(
                    "Missing: " + " | ".join(missing) +
                    ". Reason: no correspondence between TOT_Conc.specie and specie_ids_num "
                    "or all matching species were excluded.")

            if do_water:
                DA_Conc_array_wat.attrs["species_included"] = self._format_species_pairs(included_wat)
                excluded_wat_pairs = self._excluded_pairs(
                    specie_ids_num, excluded_water_names, TOT_Conc=TOT_Conc
                )
                DA_Conc_array_wat.attrs["species_excluded"] = self._format_species_pairs(excluded_wat_pairs)
                DA_Conc_array_wat.attrs["species_aggregation"] = aggregation_kind
                if weight_var is not None:
                    DA_Conc_array_wat.attrs["species_aggregation_weights"] = weight_var
                    DA_Conc_array_wat.attrs["species_aggregation_note"] = (
                            f"Species aggregated with count-weighted mean using {weight_var}.")
                DA_Conc_array_wat.attrs.pop("coordinates", None)

            if do_sediment:
                DA_Conc_array_sed.attrs["species_included"] = self._format_species_pairs(included_sed)
                excluded_sed_pairs = self._excluded_pairs(
                    specie_ids_num, excluded_sed_names, TOT_Conc=TOT_Conc
                )
                DA_Conc_array_sed.attrs["species_excluded"] = self._format_species_pairs(excluded_sed_pairs)
                DA_Conc_array_sed.attrs["species_aggregation"] = aggregation_kind
                if weight_var is not None:
                    DA_Conc_array_sed.attrs["species_aggregation_weights"] = weight_var
                    DA_Conc_array_sed.attrs["species_aggregation_note"] = (
                            f"Species aggregated with count-weighted mean using {weight_var}.")
                DA_Conc_array_sed.attrs.pop("coordinates", None)

            if Verbose:
                log("Preparing coordinates/metadata")

            if do_water:
                DA_Conc_array_wat = self.prepare_derived_output_da(
                    DA_Conc_array_wat,
                    src_ds=DS,
                    time_name=time_name,
                    spatial_mode=spatial_mode,
                    Verbose=Verbose,
                )

            if do_sediment:
                DA_Conc_array_sed = self.prepare_derived_output_da(
                    DA_Conc_array_sed,
                    src_ds=DS,
                    time_name=time_name,
                    spatial_mode=spatial_mode,
                    Verbose=Verbose,
                )

            if ("topo" in DS.data_vars) and first_var:
                DC_topo = self.prepare_derived_output_da(
                    DS["topo"],
                    src_ds=DS,
                    time_name=time_name,
                    spatial_mode=spatial_mode,
                    Verbose=Verbose,
                )
                first_var = False
                if do_water:
                    sum_vars_wat_dict["topo"] = DC_topo
                if do_sediment:
                    sum_vars_sed_dict["topo"] = DC_topo

            src_da = TOT_Conc
            sim_desc = self._sim_description_attr(src_da, Sim_description)
            base_ln = self._base_long_name(src_da, Chemical_name, variable)

            projection_proj4 = DS.attrs.get("actual_density_proj_str", None)
            grid_mapping = src_da.attrs.get("grid_mapping", None)

            if do_water:
                DA_Conc_array_wat.name = var_wat_name
                DA_Conc_array_wat.attrs["long_name"] = base_ln + " in water"
                DA_Conc_array_wat.attrs["units"] = self._water_units(src_da, variable)

                lon_coord = _coord_getter(DA_Conc_array_wat, "longitude")
                lat_coord = _coord_getter(DA_Conc_array_wat, "latitude")
                x_coord = _coord_getter(DA_Conc_array_wat, "x")
                y_coord = _coord_getter(DA_Conc_array_wat, "y")

                x_units = x_coord.attrs.get("units", None) if x_coord is not None else None
                y_units = y_coord.attrs.get("units", None) if y_coord is not None else None

                DA_Conc_array_wat = self._apply_common_attrs(
                    out_da=DA_Conc_array_wat,
                    Sim_description=sim_desc,
                    projection_proj4=projection_proj4,
                    grid_mapping=grid_mapping,
                    longitude=np.asarray(lon_coord.values) if (lon_coord is not None and lon_coord.ndim == 1) else None,
                    latitude=np.asarray(lat_coord.values) if (lat_coord is not None and lat_coord.ndim == 1) else None,
                    x=np.asarray(x_coord.values) if (x_coord is not None and x_coord.ndim == 1) else None,
                    y=np.asarray(y_coord.values) if (y_coord is not None and y_coord.ndim == 1) else None,
                    x_units=x_units,
                    y_units=y_units,
                )

                sum_vars_wat_dict[var_wat_name] = DA_Conc_array_wat

            if do_sediment:
                DA_Conc_array_sed.name = var_sed_name
                DA_Conc_array_sed.attrs["long_name"] = base_ln + " in sediments"
                DA_Conc_array_sed.attrs["units"] = self._sed_units(src_da, variable)

                lon_coord = _coord_getter(DA_Conc_array_sed, "longitude")
                lat_coord = _coord_getter(DA_Conc_array_sed, "latitude")
                x_coord = _coord_getter(DA_Conc_array_sed, "x")
                y_coord = _coord_getter(DA_Conc_array_sed, "y")

                x_units = x_coord.attrs.get("units", None) if x_coord is not None else None
                y_units = y_coord.attrs.get("units", None) if y_coord is not None else None

                DA_Conc_array_sed = self._apply_common_attrs(
                    out_da=DA_Conc_array_sed,
                    Sim_description=sim_desc,
                    projection_proj4=projection_proj4,
                    grid_mapping=grid_mapping,
                    longitude=np.asarray(lon_coord.values) if (lon_coord is not None and lon_coord.ndim == 1) else None,
                    latitude=np.asarray(lat_coord.values) if (lat_coord is not None and lat_coord.ndim == 1) else None,
                    x=np.asarray(x_coord.values) if (x_coord is not None and x_coord.ndim == 1) else None,
                    y=np.asarray(y_coord.values) if (y_coord is not None and y_coord.ndim == 1) else None,
                    x_units=x_units,
                    y_units=y_units,
                )

                sum_vars_sed_dict[var_sed_name] = DA_Conc_array_sed

            processed_variables.append(variable)

        if not processed_variables:
            raise ValueError(
                "No requested variables were processed. "
                f"Requested={variable_ls}, available={list(DS.data_vars)}"
            )
        if do_water and not any(name != "topo" for name in sum_vars_wat_dict):
            raise ValueError("No water output variables were produced.")
        if do_sediment and not any(name != "topo" for name in sum_vars_sed_dict):
            raise ValueError("No sediment output variables were produced.")

        DS_wat_fin = xr.Dataset(sum_vars_wat_dict) if do_water else None
        DS_sed_fin = xr.Dataset(sum_vars_sed_dict) if do_sediment else None

        if do_water:
            _copy_source_metadata(DS_wat_fin, DS)
            gm_names = {da.attrs.get("grid_mapping") for da in DS_wat_fin.data_vars.values()}
            _copy_support_vars(DS_wat_fin, DS, extra_grid_mapping_names=gm_names)

        if do_sediment:
            _copy_source_metadata(DS_sed_fin, DS)
            gm_names = {da.attrs.get("grid_mapping") for da in DS_sed_fin.data_vars.values()}
            _copy_support_vars(DS_sed_fin, DS, extra_grid_mapping_names=gm_names)

        wat_file = None
        sed_file = None

        if do_water:
            if File_Name_out is not None:
                wat_file = File_Path_out + "wat_" + File_Name_out
                if not wat_file.endswith(".nc"):
                    wat_file += ".nc"
            else:
                wat_file = (
                    File_Path_out + "water_conc_" +
                    (Chemical_name or "") + "_" +
                    (Origin_marker_name or "") + ".nc"
                )

        if do_sediment:
            if File_Name_out is not None:
                sed_file = File_Path_out + "sed_" + File_Name_out
                if not sed_file.endswith(".nc"):
                    sed_file += ".nc"
            else:
                sed_file = (
                    File_Path_out + "sediments_conc_" +
                    (Chemical_name or "") + "_" +
                    (Origin_marker_name or "") + ".nc"
                )

        if Save_files:
            if do_water:
                if Verbose:
                    log("Saving water concentration file")
                DS_wat_fin.to_netcdf(wat_file)

            if do_sediment:
                if Verbose:
                    log("Saving sediment concentration file")
                DS_sed_fin.to_netcdf(sed_file)

        if Return_datasets:
            if Verbose:
                if do_water and do_sediment:
                    log("Returning water and sediment DS")
                elif do_water:
                    log("Returning water DS")
                else:
                    log("Returning sediment DS")
            return DS_wat_fin, DS_sed_fin

    ##### Seeding of elements
    def emission_factors(self, scrubber_type, chemical_compound):
        """Emission factors for heavy metals and PAHs in
            open loop and closed loop scrubbers

            Hermansson et al 2021
            https://doi.org/10.1016/j.trd.2021.102912

            bilge water, gray water, anti fouling paint,
            sewage, food waster

            from EMERGE Deliverable 2.1

            ash (atmospheric depositions)
            from EMERGE Deliverable 3.2

        """
        emission_factors_open_loop = {
            #                           mean    +/-95%
            #                           ug/L    ug/L
            "Arsenic":                  [6.8,    3.4],
            "Cadmium":                  [0.8,    0.3],
            "Chromium":                 [15.,    6.5],
            "Copper":                   [36.,    12.],
            "Iron":                     [260.,   250.],
            "Lead":                     [8.8,    4.4],
            "Mercury":                  [0.09,   0.01],
            "Nickel":                   [48.,    12.],
            "Vanadium":                 [170.,   49.],
            "Zinc":                     [110.,   59.],
            "Cobalt":                   [0.17,   0.14],
            "Selenium":                 [97.,    38],
            #
            "Naphthalene":              [2.81,   0.77],
            "Phenanthrene":             [1.51,   0.29],
            "Fluoranthene":             [0.16,   0.04],
            "Benzo-a-anthracene":       [0.12,   0.05],
            "Benzo-a-pyrene":           [0.05,   0.02],
            "Dibenzo-ah-anthracene":    [0.03,   0.01],
            #
            "Acenaphthylene":           [0.12,   0.07],
            "Acenaphthene":             [0.19,   0.07],
            "Fluorene":                 [0.46,   0.10],
            "Anthracene":               [0.08,   0.04],
            "Pyrene":                   [0.31,   0.11],
            "Chrysene":                 [0.19,   0.07],
            "Benzo-b-fluoranthene":     [0.04,   0.02],
            "Benzo-k-fluoranthene":     [0.01,   0.01],
            "Indeno-123cd-pyrene":      [0.07,   0.06],
            "Benzo-ghi-perylene":       [0.02,   0.01],
            #
            "Nitrate":                  [2830.,    2060.],
            "Nitrite":                  [760.,     680.],
            "Ammonium":                 [730.,     30.],
            "Sulphur":                  [2200000., 446000.],
            "Nitrogen":                 [1400.,    0.0],
            #
            "Alkalinity":               [142.39,   0.0], # H+ ions concentration form pH
            }

        emission_factors_closed_loop = {
            #                           mean    +/-95%
            #                           ug/L    ug/L
            "Arsenic":                  [22.,    9.4],
            "Cadmium":                  [0.55,   0.19],
            "Chromium":                 [1300.,  1700.],
            "Copper":                   [480.,   230.],
            "Iron":                     [490.,   82.],
            "Lead":                     [7.7,    3.1],
            "Mercury":                  [0.07,   0.02],
            "Nickel":                   [2700.,  860.],
            "Vanadium":                 [9100.,  3200.],
            "Zinc":                     [370.,   200.],
            "Cobalt":                   [0.,     0.],
            "Selenium":                 [0.,     0.],
            #
            "Naphthalene":              [2.08,   1.05],
            "Phenanthrene":             [5.00,   2.30],
            "Fluoranthene":             [0.63,	 0.41],
            "Benzo-a-anthracene":       [0.30,	 0.29],
            "Benzo-a-pyrene":           [0.06,	 0.05],
            "Dibenzo-ah-anthracene":    [0.03,	 0.02],
            #
            "Acenaphthylene":           [0.09,   0.06],
            "Acenaphthene":             [0.47,   0.31],
            "Fluorene":                 [1.32,   0.54],
            "Anthracene":               [1.55,   2.00],
            "Pyrene":                   [0.76,   0.59],
            "Chrysene":                 [0.50,   0.45],
            "Benzo-b-fluoranthene":     [0.14,   0.12],
            "Benzo-k-fluoranthene":     [0.02,   0.02],
            "Indeno-123-cd-pyrene":     [0.04,   0.03],
            "Benzo-ghi-perylene":       [0.07,   0.07],
            #
            "Nitrate":                  [110980.,   100000.],
            "Nitrite":                  [55760.,    55000.],
            "Ammonium":                 [0.,        0.],
            "Sulphur":                  [12280000., 10104000.],
            "Nitrogen":                 [42030.,    0.0],
            #
            "Alkalinity":               [29.07, 0.0], # H+ ions concentration form pH
            }

        emission_factors_grey_water = {
            #                           mean    +/-95%
            #                           ug/L    ug/L
            "Arsenic":                  [5.98,    3.17],
            "Cadmium":                  [0.16,    0.09],
            "Chromium":                 [7.28,    2.06],
            "Copper":                   [267.,    97.],
            "Lead":                     [25.6,    21.01],
            "Mercury":                  [0.16,    0.09],
            "Nickel":                   [25.0,    19.36],
            "Selenium":                 [16.1,    10.64],
            "Zinc":                     [517.,    112.],
            #
            "Nitrogen":                 [28900.,  0.0],
         }

        emission_factors_bilge_water = {
            #                           mean    +/-95%
            #                           ug/L    ug/L
            "Arsenic":                  [35.9,    33.2],
            "Cadmium":                  [0.32,    0.07],
            "Chromium":                 [16.3,    15.4],
            "Copper":                   [49.7,    22.9],
            "Lead":                     [3.0,     1.24],
            "Nickel":                   [71.1,    11.8],
            "Selenium":                 [2.95,    1.01],
            "Vanadium":                 [76.5,    22.4],
            "Zinc":                     [949.,    660.],
            #
            "Nitrate":                  [110980.,   100000.],
            "Nitrite":                  [55760.,    55000.],
            "Ammonium":                 [0.,        0.],
            "Sulphur":                  [12280000., 10104000.],
            "Nitrogen": 	                [42047.,    39335.],
            #
            "Naphthalene":              [50.6,   34.3],
            "Phenanthrene":             [3.67,   2.51],
            "Fluoranthene":             [0.60,   0.96],
            "Benzo(a)anthracene":       [0.10,   0.18],
            "Benzo(a)pyrene":           [0.10,   0.15],
            "Dibenzo(a,h)anthracene":   [0.02,   0.01],
            #
            "Acenaphthylene":           [0.29,   0.17],
            "Acenaphthene":             [1.42,   0.86],
            "Fluorene":                 [3.33,   2.43],
            "Anthracene":               [0.22,   0.14],
            "Pyrene":                   [1.23,   1.33],
            "Chrysene":                 [0.17,   0.25],
            "Benzo-b-fluoranthene":     [0.09,   0.13],
            "Benzo-k-fluoranthene":     [0.03,   0.00],
            "Indeno-123-cd-pyrene":     [0.05,   0.06],
            "Benzo-ghi-perylene":       [0.13,   0.16],
         }

        emission_factors_sewage_water = {
            #                           mean    +/-95%
            #                           ug/L    ug/L
            "Arsenic":                  [22.9,    7.4],
            "Cadmium":                  [0.12,   0.10],
            "Chromium":                 [11.9,    8.2],
            "Copper":                   [319,     190],
            "Lead":                     [6.5,     3.1],
            "Mercury":                  [0.22,   0.12],
            "Nickel":                   [32.3,   21.3],
            "Selenium":                 [43.7,   18.3],
            "Zinc":                     [395.,   174.],
            #
            "Nitrogen":                 [430.,  0.],
         }

        emission_factors_NOx = {
            #                           mean    +/-95%
            #                           ug/L    ug/L
            "Alkalinity": [(1.0080/46.005), 0.0], # H+ ions from NOx, MW H+/MW NOx, from kg_NOx to kg_H+
        }

        emission_factors_SOx = {
            #                           mean    +/-95%
            #                           ug/L    ug/L
            "Alkalinity": [(1.0080/64.066)* 2, 0.0], # H+ ions from SOx, MW H+/MW SOx, from kg_SOx to kg_H+
        }

        emission_factors_AFP = {
            # Copper = 63.546 g/mol
            # Zinc = 65.38 g/mol
            # CuPyr = 315.86 g/mol = Copper(II) pyrithione = 0.2112 of Cu
            # CuO = 79.55 g/mol = Copper(II) oxide = 0.7989 of Cu
            # Zineb = 275.7 g/mol = Zinc ethylenebis(dithiocarbamate) = 0.2371 of Zn
            # ZnO = 81.38 g/mol = Zinc(II) oxide = 0.8033 of Zn
            # ZPyr = 317.70 g/mol = Zinc(II) pyrithione = 0.2058 of Zn

            #                           mean    +/-95%
            #                           ug/L    ug/L
            "CuO_AFP":                  [0.7989,    0.],
            "CuPyr_AFP":                [0.2112,    0.],
            "Zineb_AFP":                [0.2371,    0.],
            "ZnO_AFP":                  [0.8033,    0.],
            "ZnPyr_AFP":                [0.2058,    0.],
         }

        emission_factors_SILAM_ash = {
            #                           g/g
            "Aresenic":                 [8.09E-5],
            "Cadmium":                  [6.30E-6],
            "Chromium":                 [2.10E-4],
            "Copper":                   [2.52E-4],
            "Iron":                     [2.52E-2],
            "Mercury":                  [6.30E-6],
            "Nickel":                   [4.10E-2],
            "Lead":                     [1.16E-4],
            "Vanadium":                 [8.30E-2],
            "Zinc":                     [2.42E-3],
         }

        if scrubber_type=="open_loop":
            Emission_factors = emission_factors_open_loop.get(chemical_compound)[0]
        elif scrubber_type=="closed_loop":
            Emission_factors = emission_factors_closed_loop.get(chemical_compound)[0]
        elif scrubber_type=="bilge_water":
            Emission_factors = emission_factors_bilge_water.get(chemical_compound)[0]
        elif scrubber_type=="grey_water":
            Emission_factors = emission_factors_grey_water.get(chemical_compound)[0]
        elif scrubber_type=="sewage_water":
            Emission_factors = emission_factors_sewage_water.get(chemical_compound)[0]
        elif scrubber_type=="AFP": # Copper and Zinc from antifouling paint
            Emission_factors = 1e6*emission_factors_AFP.get(chemical_compound)[0]  # 1g = 1e6 ug: AFP is expressed as g
        elif scrubber_type=="AFP_metals_total":
            Emission_factors = 1e6 # g to ug
        elif scrubber_type=="N_sewage": # Nitrogen from sewage
            Emission_factors = 1e6  # 1kg = 1e9 ug: N_sewage is expressed as g
        elif scrubber_type=="N_foodwaste": # Nitrogen from foodwaste
            Emission_factors = 1e6  # 1kg = 1e9 ug: N_sewage is expressed as g
        elif scrubber_type == "N_NOx":  # Nitrogen from engine's NOx emissions
            Emission_factors = 1e9 * (14.0067 / 46.005)  # 1kg = 1e9 ug: NOx is expressed in kg, then tranformed to kg of nitrogen # MW of NOx: 46.005 g/mol # https://www.epa.gov/air-emissions-inventories/how-are-oxides-nitrogen-nox-defined-nei
        elif scrubber_type == "NOx":  # Nitrogen from engine's NOx emissions
            Emission_factors = 1e9 *emission_factors_NOx.get(chemical_compound)[0]
        elif scrubber_type == "SOx":  # Nitrogen from engine's NOx emissions
            Emission_factors = 1e9 *emission_factors_SOx.get(chemical_compound)[0]
        elif scrubber_type == "emission_kg":  # Generic emission expresses as kg
        	Emission_factors = 1e9
        elif scrubber_type=="SILAM_metals":
            Emission_factors = 1e9  #+ 1kg = 1e9 ug: Lead and Cadmium depositions given in kg
        elif scrubber_type=="SILAM_metals_from_ash":
            Emission_factors = 1e9*emission_factors_SILAM_ash.get(chemical_compound)[0] # 1kg=1e9ug: Ash depositions given in kg

        return Emission_factors
        # TODO: Add emission uncertainty based on 95% confidence interval

    def seed_from_DataArray(self, steam, lowerbound=0, higherbound=np.inf, radius=0, scrubber_type="open_loop", chemical_compound="Copper", mass_element_ug=100e3, number_of_elements=None, **kwargs):
            """Seed elements based on a dataarray with STEAM emission data

            Arguments:
                steam: dataarray with steam emission data, with coordinates
                    * latitude   (latitude) float32
                    * longitude  (longitude) float32
                    * time       (time) datetime64[ns]
                radius:      scalar, unit: meters
                lowerbound:  scalar, elements with lower values are discarded
            """

            if chemical_compound is None:
                chemical_compound = self.get_config('chemical:compound')

            #mass_element_ug=1e3      # 1e3 - 1 element is 1mg chemical
            #mass_element_ug=20e3      # 100e3 - 1 element is 100mg chemical
            #mass_element_ug=100e3      # 100e3 - 1 element is 100mg chemical
            #mass_element_ug=1e6     # 1e6 - 1 element is 1g chemical

            sel=np.where((steam > lowerbound) & (steam < higherbound))
            t=steam.time[sel[0]].data
            la=steam.latitude[sel[1]].data
            lo=steam.longitude[sel[2]].data

            data=np.array(steam.data)

            if number_of_elements is not None:
                total_volume = np.sum(data[sel])
                total_mass = total_volume * self.emission_factors(scrubber_type, chemical_compound)
                mass_element_ug = total_mass / number_of_elements
                mass_element_ug_0 = total_mass / number_of_elements

            for i in range(0,t.size):
                scrubberwater_vol_l=data[sel][i]
                mass_ug=scrubberwater_vol_l * self.emission_factors(scrubber_type, chemical_compound)

                if number_of_elements is None:
                    number=np.array(mass_ug / mass_element_ug).astype('int')
                else:
                    number=np.ceil(np.array(mass_ug / mass_element_ug_0)).astype('int')
                    mass_element_ug=mass_ug/number

                time = datetime.utcfromtimestamp((t[i] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))

                if number>0:
                    z = -1*np.random.uniform(0, 1, number)
                    self.seed_elements(lon=lo[i]*np.ones(number), lat=la[i]*np.ones(number),
                                radius=radius, number=number, time=time,
                                mass=mass_element_ug,mass_degraded=0,mass_volatilized=0, z=z, origin_marker=1)

                mass_residual = mass_ug - number*mass_element_ug

                if mass_residual>0 and number_of_elements is None:
                    z = -1*np.random.uniform(0, 1, 1)
                    self.seed_elements(lon=lo[i], lat=la[i],
                                radius=radius, number=1, time=time,
                                mass=mass_residual,mass_degraded=0,mass_volatilized=0, z=z, origin_marker=1)

    seed_from_STEAM = seed_from_DataArray
    ''' Alias of seed_from_DataArray method for backward compatibility
    '''

    ### Helpers for seed_from_NETCDF ###
    @staticmethod
    def _get_number_of_elements(g_mode, mass_element_ug=None, data_point=None, n_elements=None):
        """
        Returns number of elements to generate.
        For g_mode == "mass":
            Checks inputs. Number of full elements and residuals
            are handles speparately.
        For g_mode == "fixed":
            Returns n_elements.
        """
        import numpy as np

        if g_mode == "mass":
            if mass_element_ug is None or data_point is None:
                raise ValueError("'mass' mode requires mass_element_ug and data_point")
            if mass_element_ug <= 0:
                raise ValueError("'mass' mode requires mass_element_ug > 0")
            if data_point <= 0:
                return 0
            return int(np.floor(float(data_point) / float(mass_element_ug)))
        elif g_mode == "fixed":
            if n_elements is None or n_elements <= 0:
                raise ValueError("fixed mode requires n_elements > 0")
            return int(n_elements)
        else:
            raise ValueError("Incorrect combination of mode and input - undefined inputs")

    @staticmethod
    def _get_z(
        mode,
        number,
        NETCDF_data_dim_names,
        depth_seed=None, depth_min=None, depth_max=None,
        emission_placement="upper_1m",
        emission_seafloor_eps=None,
        emission_depth_value=None,
    ):
        """
        Compute initial vertical positions (z) for seeded elements.
        Returns:
          - np.ndarray of shape (number,) with negative depths (meters) for
            water_conc / emission / emission_depth
          - a string "seafloor+X" for sed_conc or emission when requested
        """
        import numpy as np

        if number is None:
            raise ValueError("number is None")
        number = int(number)
        if number < 0:
            raise ValueError(f"number must be >= 0, got {number}")
        if number == 0:
            if mode == "sed_conc":
                return "seafloor+0.0"
            if mode == "emission" and emission_placement == "seafloor":
                eps = emission_seafloor_eps
                if eps is None or not np.isfinite(eps) or eps < 0:
                    raise ValueError(f"emission_seafloor_eps must be finite and >= 0, got {eps}")
                return f"seafloor+{float(eps)}"
            return np.empty((0,), dtype=float)

        dim_names = set(NETCDF_data_dim_names or [])

        if mode == "water_conc":
            if depth_seed is None:
                raise ValueError("water_conc requires depth_seed (bathymetry at seed point).")
            if not np.isfinite(depth_seed) or float(depth_seed) <= 0.0:
                raise ValueError(f"water_conc requires finite depth_seed > 0, got {depth_seed}")

            if "depth" in dim_names:
                if depth_min is None or depth_max is None:
                    raise ValueError(
                        "depth_min and depth_max must be provided when 'depth' is a dimension.")
                if (not np.isfinite(depth_min)) or (not np.isfinite(depth_max)):
                    raise ValueError(
                        f"depth_min/depth_max must be finite, got {depth_min}, {depth_max}")

                lo = float(depth_min)
                hi = float(depth_max)
                if hi < lo:
                    lo, hi = hi, lo

                eps = 1e-4
                if hi - lo <= eps:
                    zpos = max(lo, eps)
                    return -np.full(number, zpos, dtype=float)

                lo2 = max(lo, eps)
                hi2 = max(hi, lo2 + eps)
                return -np.random.uniform(lo2, hi2, number)

            eps = 1e-4
            hi = float(depth_seed) - eps
            if not np.isfinite(hi) or hi <= eps:
                return -np.full(number, eps, dtype=float)
            return -np.random.uniform(eps, hi, number)

        if mode == "sed_conc":
            return "seafloor+0.0"

        if mode == "emission":
            if emission_placement == "upper_1m":
                eps = 1e-4
                hi = 1.0 - eps

                if depth_seed is not None:
                    if not np.isfinite(depth_seed) or float(depth_seed) <= 0.0:
                        raise ValueError(
                            f"emission upper_1m requires finite depth_seed > 0 when provided, got {depth_seed}"
                        )
                    hi = min(hi, float(depth_seed) - eps)

                if hi <= eps:
                    return -np.full(number, eps, dtype=float)

                return -np.random.uniform(eps, hi, number)

            if emission_placement == "seafloor":
                eps = emission_seafloor_eps
                if eps is None or not np.isfinite(eps) or eps < 0:
                    raise ValueError(f"emission_seafloor_eps must be finite and >= 0, got {eps}")
                return f"seafloor+{float(eps)}"

            raise ValueError(
                f"Unsupported emission_placement '{emission_placement}', "
                f"use 'upper_1m' or 'seafloor'")

        if mode == "emission_depth":
            eps = 1e-4

            if emission_depth_value is None:
                raise ValueError("emission_depth requires emission_depth_value.")
            if not np.isfinite(emission_depth_value) or float(emission_depth_value) < 0.0:
                raise ValueError(
                    f"emission_depth requires finite emission_depth_value >= 0, got {emission_depth_value}"
                )

            zpos = float(emission_depth_value)

            if depth_seed is not None:
                if not np.isfinite(depth_seed) or float(depth_seed) <= 0.0:
                    raise ValueError(
                        f"emission_depth requires finite depth_seed > 0 when provided, got {depth_seed}"
                    )
                if zpos > float(depth_seed) - eps:
                    raise ValueError(
                        f"Requested emission_depth_value={zpos} exceeds local water depth {depth_seed}."
                    )

            zpos = max(zpos, eps)
            return -np.full(number, zpos, dtype=float)

        raise ValueError(f"Unsupported mode '{mode}' in _get_z")

    @staticmethod
    def _speciation_is_explicit(speciation):
        """
        True only for explicit seeding speciation.
        Explicit forms:
          - int
          - species-name str
          - array-like of int/str
          - dict {species: fraction}
        Non-explicit forms:
          - None
          - "config"
          - "default"
        """
        if speciation is None:
            return False
        if isinstance(speciation, str) and speciation in ("config", "default"):
            return False
        return True

    @staticmethod
    def _validate_depth_layer_convention(depth_coord, depth_layer_convention, tol=1e-8):
        """
        Validate that the provided depth coordinate is consistent with the
        declared layer convention.

        depth_layer_convention == "top":
            depth values are interpreted as TOP of layers
            -> shallowest depth must be 0
        depth_layer_convention == "bottom":
            depth values are interpreted as BOTTOM of layers
            -> depth coordinate must not contain 0
        """
        import numpy as np

        if depth_layer_convention not in ("top", "bottom"):
            raise ValueError(
                "depth_layer_convention must be 'top' or 'bottom' when "
                "NETCDF_data has a depth dimension.")

        raw = np.asarray(depth_coord, dtype=float).ravel()
        if raw.size == 0:
            raise ValueError("NETCDF_data depth coordinate is empty.")
        if not np.all(np.isfinite(raw)):
            raise ValueError("NETCDF_data depth coordinate contains non-finite values.")

        depth_abs = np.sort(np.unique(np.abs(raw)))
        if depth_abs.size == 0:
            raise ValueError("NETCDF_data depth coordinate is empty after processing.")

        if np.any(np.diff(depth_abs) <= tol):
            raise ValueError(
                "NETCDF_data depth coordinate must contain distinct, strictly increasing "
                "depth levels in magnitude.")

        has_zero = np.isclose(depth_abs[0], 0.0, rtol=0.0, atol=tol)

        if depth_layer_convention == "top":
            if not has_zero:
                raise ValueError(
                    "depth_layer_convention='top' is inconsistent with the provided depth "
                    f"coordinate: shallowest depth is {depth_abs[0]}, but top-of-layer "
                    "coordinates must start at 0.0 m.")

        elif depth_layer_convention == "bottom":
            if has_zero:
                raise ValueError(
                    "depth_layer_convention='bottom' is inconsistent with the provided depth "
                    "coordinate: it contains 0.0 m, but bottom-of-layer coordinates must be "
                    "strictly greater than 0.0 m.")

    @staticmethod
    def _remove_positions(items, positions):
        """
        Remove indices in `positions` from each item in `items`.

        Each item can be:
          - a 1D array
          - a tuple/list of arrays (e.g. `sel` from np.where)
          - None (passes through)
        Returns a tuple of pruned items in same order.
        """
        import numpy as np
        positions = np.asarray(positions, dtype=int)

        def prune_one(x):
            if x is None:
                return None
            if isinstance(x, (tuple, list)):
                out = tuple(np.delete(np.asarray(a), positions) for a in x)
                return out if isinstance(x, tuple) else list(out)
            return np.delete(np.asarray(x), positions)

        return tuple(prune_one(x) for x in items)

    @staticmethod
    def _coord_values_and_sel_index(da, dim_name):
        """
        Return (coord_values, index_in_dims or None).

        - If dim_name is a dimension: coord_values is the coordinate values (1D, len=dim)
          and dim_index is the position in sel tuple.
        - If dim_name is not a dimension but exists as a coordinate:
          coord_values is np.array(da.coords[dim_name]) and dim_index is None.
        """
        import numpy as np

        dim_names = list(da.dims)
        if dim_name in dim_names:
            dim_index = dim_names.index(dim_name)
            coord_vals = np.asarray(da[dim_name].data)
            return coord_vals, dim_index

        # coordinate but not a dimension
        if dim_name in da.coords:
            coord_vals = np.asarray(da.coords[dim_name].data)
            return coord_vals, None

        raise ValueError(f"'{dim_name}' not found as dimension or coordinate in dataarray.")

    def build_specie_array(self, n, speciation, default_specie):
        '''
        Builds the initial per-element species index array (length n) used when seeding particles.
        It interprets speciation as:
         - None / "config": returns None (meaning: don’t pass specie, let config-driven partitioning decide).

         - "default": returns an array filled with default_specie.
         - scalar int or species-name str: returns an array with that single species for all n.
         - array-like of int/str: (length n): returns the corresponding per-element indices (mapping names via self.name_species).
         - dict {species: fraction}:→ randomly samples n species according to the normalized fractions.
        '''
        import numpy as np

        # ChemicalDrift config-driven partitioning
        if speciation is None or (isinstance(speciation, str) and speciation == "config"):
            return None
        if isinstance(speciation, str) and speciation == "default":
            return np.full(n, int(default_specie), dtype=int)

        # scalar: all same (int or species-name string)
        if np.isscalar(speciation):
            if isinstance(speciation, str):
                if not hasattr(self, "name_species"):
                    self.init_species()
                try:
                    sp_idx = int(self.name_species.index(speciation))
                except ValueError:
                    raise ValueError(f"Unknown species name '{speciation}'. Valid: {self.name_species}")
                return np.full(n, sp_idx, dtype=int)
            else:
                return np.full(n, int(speciation), dtype=int)

        # explicit array/list of indices OR names (length must be n)
        if isinstance(speciation, (list, tuple, np.ndarray)):
            arr = np.asarray(speciation).ravel()
            if arr.size != n:
                raise ValueError(f"Explicit speciation array has length {arr.size}, expected {n}.")

            # If any strings present, map by name_species
            if arr.dtype.kind in ("U", "S", "O"):
                if not hasattr(self, "name_species"):
                    self.init_species()

                out = np.empty(arr.size, dtype=int)
                for i, v in enumerate(arr):
                    if isinstance(v, str):
                        try:
                            out[i] = int(self.name_species.index(v))
                        except ValueError:
                            raise ValueError(f"Unknown species name '{v}'. Valid: {self.name_species}")
                    else:
                        idx = int(v)
                        if idx < 0 or idx >= len(self.name_species):
                            raise ValueError(f"Species index {idx} out of range 0..{len(self.name_species)-1}")

                        out[i] = int(v)
                return out.astype(int)

            # Pure numeric
            if not hasattr(self, "name_species"):
                self.init_species()

            arr_i = arr.astype(int)
            if np.any(arr_i < 0) or np.any(arr_i >= len(self.name_species)):
                bad = arr_i[(arr_i < 0) | (arr_i >= len(self.name_species))][:5]
                raise ValueError(f"Species index out of range 0..{len(self.name_species)-1}. Examples: {bad.tolist()}")
            return arr_i

        # dict of fractions: {species: fraction, ...} (keys int or str)
        if isinstance(speciation, dict):
            keys = list(speciation.keys())
            fracs = np.asarray([speciation[k] for k in keys], dtype=float)

            if np.any(fracs < 0):
                raise ValueError("Speciation fractions must be >= 0.")
            s = fracs.sum()
            if not np.isfinite(s) or s <= 0:
                raise ValueError("Speciation fractions must sum to a positive number.")
            fracs = fracs / s  # normalize

            if not hasattr(self, "name_species"):
                self.init_species()

            idx = []
            for k in keys:
                if isinstance(k, str):
                    try:
                        idx.append(int(self.name_species.index(k)))
                    except ValueError:
                        raise ValueError(f"Unknown species name '{k}'. Valid: {self.name_species}")
                else:
                    if int(k) < 0 or int(k) >= len(self.name_species):
                        raise ValueError(f"Species index {k} out of range 0..{len(self.name_species)-1}")

                    idx.append(int(k))

            idx = np.asarray(idx, dtype=int)
            return np.random.choice(idx, size=n, p=fracs).astype(int)

        raise TypeError(
            "speciation must be None/'config'/'default', int, str, array-like of int/str, or dict of fractions.")

    def speciation_to_indices_set(self, speciation, name_to_idx, active_names, label="speciation"):
        """
        Return a set of species indices referenced by speciation input (without sampling).
        speciation may be: None/"config"/"default", int, str, array-like of int/str, dict with int/str keys.
        """
        import numpy as np

        missing = set()

        def to_index(v):
            if isinstance(v, str):
                if v not in name_to_idx:
                    missing.add(v)
                    return None
                return int(name_to_idx[v])
            # numeric: accept as index but check bounds
            i = int(v)
            if i < 0 or i >= len(active_names):
                raise ValueError(
                    f"{label}: species index {i} out of range for active species list (0..{len(active_names)-1}).")
            return i

        if isinstance(speciation, dict):
            vals = list(speciation.keys())
        elif np.isscalar(speciation):
            vals = [speciation]
        elif isinstance(speciation, (list, tuple, np.ndarray)):
            vals = list(np.asarray(speciation, dtype=object).ravel())
        else:
            raise TypeError(f"{label}: speciation must be None/'config', int, str, array-like, or dict.")

        used = set()
        for v in vals:
            idx = to_index(v)
            if idx is not None:
                used.add(idx)

        return used, missing

    def validate_speciation_input_allowed(self, speciation, allowed_names, label="speciation"):
        """
        Validate that the user-provided speciation references only species allowed for this context,
        AND that requested species are available (active) in self.name_species for this run.

        - water_speciation accepts speciation as None/"config"/"default" (config-driven) and int/str/array/dict
        - sediment_speciation accepts speciation as "default"/int/str/array/dict, but not None/"config"
        - allowed_names may include species that are not active in this run; those are ignored for building allowed_idx.
        - If the user requests a species name not active in this run -> raises ValueError with a helpful message.
        """
        if speciation is None or (isinstance(speciation, str) and speciation in ("config", "default")):
            if label == "sed_speciation" and (speciation is None or speciation == "config"):
                raise ValueError(
                    f"{label} cannot be None or 'config', it is: {speciation}")
            return

        if not hasattr(self, "name_species"):
            self.init_species()

        active_names = list(self.name_species)
        active_set = set(active_names)
        allowed_set = set(allowed_names)

        # Allowed AND active for this run
        allowed_active = allowed_set & active_set

        # If none of the allowed species are active, validation is impossible / meaningless
        if not allowed_active:
            raise ValueError(
                f"{label}: none of the allowed species are active in this run. "
                f"Allowed: {sorted(allowed_set)}. Active: {active_names}")

        # Build index sets safely
        name_to_idx = {name: i for i, name in enumerate(active_names)}
        allowed_idx = {name_to_idx[n] for n in allowed_active}

        # Collect which species the user referenced (as indices), and also track missing names
        used_idx, missing_names = self.speciation_to_indices_set(
            speciation, name_to_idx=name_to_idx, active_names=active_names, label=label)

        if missing_names:
            raise ValueError(
                f"{label}: requested species are not active in this run: {sorted(missing_names)}. "
                f"Active species: {active_names}")

        bad_idx = sorted(set(used_idx) - set(allowed_idx))
        if bad_idx:
            bad_names = [active_names[i] for i in bad_idx]
            raise ValueError(
                f"{label}: contains species not allowed here: {bad_names}. "
                f"Allowed (active subset): {sorted(allowed_active)}")

    def _default_seed_specie(self, mode):
        """
        Resolve the canonical default specie for seeding, based on the active
        transfer setup and the species actually created by init_species().
        - water_conc / emission / emission_depth:
            Sandnesfj_Al -> LMMcation
            otherwise    -> prefer LMM
        - sed_conc:
            prefer Sediment reversible
        Falls back to the first active allowed specie in the relevant family.
        """
        if not hasattr(self, 'name_species'):
            self.init_species()

        transfer_setup = self.get_config('chemical:transfer_setup')

        if transfer_setup == 'custom':
            raise ValueError(
                "Default seed specie is undefined for chemical:transfer_setup='custom'. "
                "Use explicit water_speciation / sed_speciation when seeding.")

        if mode in ('water_conc', 'emission', 'emission_depth'):
            if transfer_setup == 'Sandnesfj_Al':
                preferred = [
                    'LMMcation', 'LMManion',
                    'Humic colloid', 'Polymer', 'Colloid',
                    'Particle reversible', 'Particle slowly reversible',
                    'Particle irreversible',]
            else:
                preferred = [
                    'LMM', 'LMMcation', 'LMManion',
                    'Humic colloid', 'Polymer', 'Colloid',
                    'Particle reversible',
                    'Particle slowly reversible',
                    'Particle irreversible',]
        elif mode == 'sed_conc':
            preferred = [
                'Sediment reversible',
                'Sediment slowly reversible',
                'Sediment irreversible',
                'Sediment buried',]
        else:
            raise ValueError(f"Unsupported mode for default seed specie: {mode!r}")

        for name in preferred:
            if name in self.name_species:
                return self.specie_name2num(name)

        raise ValueError(
            f"Could not resolve a default seed specie for mode={mode!r}. "
            f"Active species are: {self.name_species}")

    @staticmethod
    def _record_seed_failure(fail_records, stage, point_index, exc, **context):

        rec = {
            "stage": stage,
            "point_index": int(point_index),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        rec.update(context)
        fail_records.append(rec)

        logger.exception(
            "seed_from_NETCDF failure at stage=%s point=%s context=%s",
            stage, point_index, context
        )

    def seed_from_NETCDF(
            self,
            NETCDF_data,
            Bathimetry_data=None,
            Bathimetry_seed_data=None,
            mode='water_conc',
            lon_resol=None,
            lat_resol=None,
            lowerbound=0,
            higherbound=float("inf"),
            radius=50,
            mass_element_ug=100e3,
            number_of_elements=None,
            origin_marker=1,
            gen_mode="mass",
            water_speciation=None,
            sed_speciation="default",
            last_depth_until_bathimetry=True,
            depth_layer_convention=None,
            emission_placement="upper_1m",
            emission_seafloor_eps=None,
            active_sediment_layer_thickness_data=None,
            f_OC_particle_data=None,
            f_OC_sediment_data=None,
    ):
        """
        Seed elements based on a dataarray with water/sediment concentration or direct emissions to water.

        Arguments:
        NETCDF_data:          dataarray with concentration or emission data, with coordinates
            * latitude        (latitude) float32
            * longitude       (longitude) float32
            * time            (time) datetime64[ns]
        Bathimetry_data:      dataarray with bathimetry data to calcuate volume of gridcells for water_conc
                              MUST have the same grid of NETCDF_data, no time dimension, and positive values
            * latitude        (latitude) float32
            * longitude       (longitude) float32
        Bathimetry_seed_data: dataarray with bathimetry data to check seeding of elements,
                              MUST be the same used for running the simulation, no time dimension, and positive downward.
                              Required for "water_conc", "sed_conc", and "emission_depth".
            * latitude        (latitude) float32
            * longitude       (longitude) float32
        f_OC_particle_data:   dataarray with local f_OC for water-column seeding.
                              If not given, seed_elements() will build default f_OC internally.
            * latitude        (latitude) float32
            * longitude       (longitude) float32
            * time            (time) datetime64[ns]
        f_OC_sediment_data:   dataarray with local f_OC for sediment seeding.
                              If not given, seed_elements() will build default f_OC internally.
            * latitude        (latitude) float32
            * longitude       (longitude) float32
            * time            (time) datetime64[ns]
        active_sediment_layer_thickness_data: dataarray with local active sediment layer thickness [m]
                              MUST be on the same spatial grid as NETCDF_data. Used only for mode='sed_conc'.
                              If not given, or if local values are invalid, the global self.get_config('chemical:sediment:mixing_depth') is used.
            * latitude        (latitude) float32
            * longitude       (longitude) float32
        mode:                 "water_conc" (seed from concentration in water column, in ug/L),
                      "sed_conc" (seed from sediment concentration, in ug/kg d.w.),
                      "emission" (seed from direct discharge to water, in kg),
                      "emission_depth" (seed from direct discharge to water, in kg,
                                        at depth given by NETCDF_data.depth)
        radius:               float32, unit: meters, elements will be created in a circular area around coordinates
        lowerbound:           float32 elements with lower values are discarded
        higherbound:          float32, elements with higher values are discarded
        number_of_elements:   int, number of elements created for each vertical layer at each gridpoint
        mass_element_ug:      float32, maximum mass of elements if number_of_elements is not specificed
        lon_resol:            float32, longitude resolution of the NETCDF dataset
        lat_resol:            float32, latitude resolution of the NETCDF dataset
        gen_mode:             string, "mass" (elements generated from mass), "fixed" (fixed number of elements for each data point)
        water_speciation:     Controls initial species assignment for elements seeded in water_conc, emission, and emission_depth modes.
                              Accepted values:
                                - None or "config": config-driven partitioning
                                - "default": transfer_setup-dependent canonical default specie
                                - int / str / array-like / dict: explicit species selection
                              For chemical:transfer_setup='custom', only explicit species selection is allowed. None/"config"/"default" are rejected.
        sed_speciation:       Controls initial species assignment for elements seeded in sed_conc mode.
                              Accepted values:
                                - "default"
                                - int / str / array-like / dict: explicit species selection
                              For chemical:transfer_setup='custom', only explicit species
                              selection is allowed. None/"config"/"default" are rejected.
        depth_layer_convention: string,"top" if depth coordinate stores TOP of each layer, "bottom" if depth coordinate stores BOTTOM of each layer.
                                  Used only for mode='water_conc'.
        last_depth_until_bathimetry: boolean, when depth is specified in NETCDF_data using "water_conc" mode
                                    the water column below the highest depth value is considered the same as the last
                                    available layer (True) or is consedered without chemical (False)
        emission_placement:   "upper_1m" or "seafloor"
                              Controls where emission-mode particles are initialized. Used only for mode='emission'.
        emission_seafloor_eps:                              float32, optional offset above seabed used only when
                              mode='emission' and emission_placement='seafloor'.
        origin_marker:        int, or string "single", assign a marker to seeded elements. If "single" a different origin_marker will be assigned to each datapoint
        """
        import opendrift
        from datetime import datetime
        import numpy as np
        from collections import Counter
        fail_records = []

        def _time_repr(x):
            if x is None:
                return None
            if hasattr(x, "isoformat"):
                try:
                    return x.isoformat()
                except Exception:
                    pass
            try:
                return str(np.datetime64(x))
            except Exception:
                return str(x)

        if mode not in ['water_conc', 'sed_conc', 'emission', 'emission_depth']:
            raise ValueError(
                f"Invalid mode: '{mode}', only 'water_conc', 'sed_conc', "
                "'emission', and 'emission_depth' are permitted")
        if mode == "emission" and emission_placement not in ("upper_1m", "seafloor"):
            raise ValueError(
                f"Invalid emission_placement='{emission_placement}'. "
                f"Use 'upper_1m' or 'seafloor'.")
        if mode == "emission" and emission_placement == "seafloor" and emission_seafloor_eps is None:
            raise ValueError(
                "emission_seafloor_eps must be provided when mode='emission' "
                "and emission_placement='seafloor'.")

        # Validate speciation inputs
        if not np.all([(hasattr(self, attr)) for attr in ['nspecies', 'name_species', 'transfer_rates']]):
            if self.mode != opendrift.models.basemodel.Mode.Config:
                self.mode = opendrift.models.basemodel.Mode.Config
            self.init_species()
            self.init_transfer_rates()

        transfer_setup = self.get_config('chemical:transfer_setup')
        if transfer_setup == 'custom':
            if mode in ('water_conc', 'emission', 'emission_depth'):
                if not self._speciation_is_explicit(water_speciation):
                    raise ValueError(
                        "For chemical:transfer_setup='custom', water_conc/emission/"
                        "emission_depth seeding requires explicit water_speciation. "
                        "None, 'config', and 'default' are not allowed.")
            elif mode == 'sed_conc':
                if not self._speciation_is_explicit(sed_speciation):
                    raise ValueError(
                        "For chemical:transfer_setup='custom', sed_conc seeding requires "
                        "explicit sed_speciation. None, 'config', and 'default' are not allowed.")

        WATER_ALLOWED = {
            "LMM", "LMMcation", "LMManion",
            "Colloid", "Humic colloid",
            "Polymer",
            "Particle reversible",
            "Particle slowly reversible",
            "Particle irreversible",
        }
        SED_ALLOWED = {
            "Sediment reversible",
            "Sediment slowly reversible",
            "Sediment irreversible",
            "Sediment buried"
        }

        self.validate_speciation_input_allowed(water_speciation, WATER_ALLOWED, label="water_speciation")
        self.validate_speciation_input_allowed(sed_speciation,   SED_ALLOWED,   label="sed_speciation")

        # Validate f_OC inputs
        def _build_uniform_foc_array(n, foc_value, label):
            foc_value = float(foc_value)
            if not np.isfinite(foc_value) or foc_value <= 0.0:
                raise ValueError(f"{label} must be finite and > 0, got {foc_value}")
            return np.full(n, foc_value, dtype=float)

        def _lookup_scalar_map_value(da, lat, lon, fallback, label, time=None, point_index=None):
            if da is None:
                return fallback

            try:
                sel_kwargs = {
                    "latitude": float(lat),
                    "longitude": float(lon),
                }

                has_time = ("time" in da.dims) or ("time" in da.coords)
                if has_time:
                    if time is None:
                        raise ValueError(
                            f"{label} has a time coordinate/dimension, but no time was provided.")
                    sel_kwargs["time"] = np.datetime64(time)

                value = da.sel(method='nearest', **sel_kwargs).values
                arr = np.asarray(value)

                if arr.size != 1:
                    raise ValueError(
                        f"{label} selection did not reduce to a scalar; remaining shape={arr.shape}")

                value = float(arr.squeeze())
                if np.isfinite(value) and value > 0.0:
                    return value

                logger.debug(
                    "Invalid %s at lon=%s lat=%s time=%s (value=%s), using fallback=%s",
                    label, lon, lat, _time_repr(time), value, fallback
                )
                return fallback

            except Exception as e:
                if point_index is not None:
                    self._record_seed_failure(
                        fail_records=fail_records,
                        stage=f"{label}_lookup",
                        point_index=point_index,
                        exc=e,
                        lon=float(lon),
                        lat=float(lat),
                        time=_time_repr(time),
                        fallback=float(fallback),
                    )
                else:
                    logger.exception(
                        "Could not read %s at lon=%s lat=%s time=%s. Using fallback=%s",
                        label, lon, lat, _time_repr(time), fallback
                    )
                return fallback

        def _build_seed_foc_array(n, mode, particle_foc, sediment_foc):
            # Only override f_OC during seeding when a local map was provided.
            # Otherwise let seed_elements() assign defaults internally from init_specie.
            if mode in ("water_conc", "emission", "emission_depth"):
                if f_OC_particle_data is None:
                    return None
                return _build_uniform_foc_array(n, particle_foc, "water-column f_OC")

            if mode == "sed_conc":
                if f_OC_sediment_data is None:
                    return None
                return _build_uniform_foc_array(n, sediment_foc, "sediment f_OC")

            return None

        # Select data
        sel = np.where((NETCDF_data > lowerbound) & (NETCDF_data < higherbound))

        if ("time" not in NETCDF_data.dims) and ("time" not in NETCDF_data.coords):
            raise ValueError("NETCDF_data has no 'time' dimension/coord.")
        time_check = NETCDF_data.time.size

        NETCDF_data_dim_names = list(NETCDF_data.dims)

        # Build per-point lat/lon vectors aligned with selected datapoints
        if lon_resol is None or lat_resol is None:
            raise ValueError("lat/lon_resol must be specified")

        lat_vals, lat_sel_idx = self._coord_values_and_sel_index(NETCDF_data, "latitude")
        lon_vals, lon_sel_idx = self._coord_values_and_sel_index(NETCDF_data, "longitude")

        if lat_sel_idx is not None:
            la = lat_vals[sel[lat_sel_idx]]
        else:
            if lat_vals.size != 1:
                raise ValueError("latitude is not a dimension: only scalar latitude coordinate is supported.")
            la = np.full(len(sel[0]), float(lat_vals.item()))

        if lon_sel_idx is not None:
            lo = lon_vals[sel[lon_sel_idx]]
        else:
            if lon_vals.size != 1:
                raise ValueError("longitude is not a dimension: only scalar longitude coordinate is supported.")
            lo = np.full(len(sel[0]), float(lon_vals.item()))

        if "time" in NETCDF_data_dim_names:
            time_name_index = NETCDF_data_dim_names.index("time")
        elif time_check > 1:
            raise ValueError("Dimension [time] is not present in NETCDF_data_dim_names")
        else:
            time_name_index = None

        depth = None
        depth_name_index = None
        if "depth" in NETCDF_data.dims:
            depth_name_index = NETCDF_data_dim_names.index("depth")

            if mode == "water_conc":
                if depth_layer_convention is None:
                    raise ValueError(
                        "depth_layer_convention must be explicitly set to 'top' or 'bottom' "
                        "when mode='water_conc' and NETCDF_data has a depth dimension.")
                self._validate_depth_layer_convention(
                    NETCDF_data.depth.values,
                    depth_layer_convention,
                )
            elif depth_layer_convention is not None:
                logger.info(
                    "Ignoring depth_layer_convention=%s because it is only used for mode='water_conc'.",
                    depth_layer_convention)
        elif depth_layer_convention is not None:
            logger.info(
                "Ignoring depth_layer_convention=%s because NETCDF_data has no depth dimension.",
                depth_layer_convention)

        # Time vector
        if time_check == 1:
            try:
                t = np.datetime64(str(np.array(NETCDF_data.time.data)))
                t = np.array(t, dtype='datetime64[s]')
            except Exception:
                t = np.datetime64(str(np.array(NETCDF_data.time[0])))
                t = np.array(t, dtype='datetime64[s]')
        else:
            t = NETCDF_data.time[sel[time_name_index]].data
            t = np.array(t, dtype='datetime64[s]')

        # Depth vector for selected points (if any)
        if mode in ("water_conc", "emission_depth") and depth_name_index is not None:
            depth = np.absolute(NETCDF_data.depth[sel[depth_name_index]].data)

        if mode == "emission_depth":
            if "depth" not in NETCDF_data.dims:
                raise ValueError(
                    "mode='emission_depth' requires NETCDF_data to have a 'depth' dimension.")
            if Bathimetry_seed_data is None:
                raise ValueError(
                    "Bathimetry_seed_data is required for mode='emission_depth'")

        # Find center of pixel for volume of water / sediments
        lon_array = lo + lon_resol / 2
        lat_array = la + lat_resol / 2

        # Prune datapoints where Bathimetry_seed_data at the pixel-center is NaN/<=0
        if mode in ("water_conc", "sed_conc", "emission_depth"):
            if Bathimetry_seed_data is None:
                raise ValueError(
                    "Bathimetry_seed_data is required for mode='water_conc', "
                    "mode='sed_conc', and mode='emission_depth'")

            check_bad = []
            ncheck = int(np.size(lon_array))
            for i in range(ncheck):
                try:
                    Bseed = float(
                        Bathimetry_seed_data.sel(
                            latitude=float(lat_array[i]),
                            longitude=float(lon_array[i]),
                            method='nearest'
                        ).values
                    )
                    if (not np.isfinite(Bseed)) or (Bseed <= 0.0):
                        check_bad.append(i)
                except Exception as e:
                    self._record_seed_failure(
                        fail_records=fail_records,
                        stage="bathymetry_seed_prune",
                        point_index=i,
                        exc=e,
                        lon=float(lon_array[i]),
                        lat=float(lat_array[i]),
                        mode=mode,
                    )
                    check_bad.append(i)

            if check_bad:
                bad = np.asarray(check_bad, dtype=int)
                depth_vec = depth if depth is not None else None
                t_vec = t if (hasattr(t, "size") and t.size > 1) else None

                sel, la, lo, lat_array, lon_array, depth_vec, t_vec = self._remove_positions(
                    (sel, la, lo, lat_array, lon_array, depth_vec, t_vec),
                    bad)

                if depth_vec is not None:
                    depth = depth_vec
                if t_vec is not None:
                    t = t_vec

                logger.info("%s datapoints removed due to inconsistent bathymetry (seed grid)", bad.size)

        if mode == 'water_conc' and Bathimetry_data is None:
            raise ValueError("Bathimetry_data is required for mode='water_conc'")

        # Build 1D values aligned with sel/la/lo/lat_array/lon_array (and depth/t if present)
        values = np.asarray(NETCDF_data.data)[sel]
        npts = int(values.size)

        if npts == 0:
            logger.info("No datapoints left after filtering/pruning; nothing to seed.")
            self._last_seed_from_netcdf_failures = fail_records
            return

        print(f"Seeding {npts} datapoints")
        list_index_print = self._print_progress_list(npts)

        if mode == 'sed_conc':
            sed_mixing_depth_global = float(self.get_config('chemical:sediment:mixing_depth'))
            sed_density = float(self.get_config('chemical:sediment:density'))
            sed_porosity = float(self.get_config('chemical:sediment:porosity'))

            if active_sediment_layer_thickness_data is not None:
                thickness_dims = set(active_sediment_layer_thickness_data.dims)
                if not {'latitude', 'longitude'}.issubset(thickness_dims):
                    raise ValueError(
                        "active_sediment_layer_thickness_data must have at least "
                        "'latitude' and 'longitude' dimensions.")

        if f_OC_particle_data is not None:
            focp_dims = set(f_OC_particle_data.dims)
            if not {'latitude', 'longitude'}.issubset(focp_dims):
                raise ValueError("f_OC_particle_data must have at least 'latitude' and 'longitude' dimensions.")
            if 'time' in focp_dims and ('time' not in NETCDF_data.dims and 'time' not in NETCDF_data.coords):
                raise ValueError("f_OC_particle_data has a time dimension, but NETCDF_data has no usable time coordinate.")
        if f_OC_sediment_data is not None:
            focs_dims = set(f_OC_sediment_data.dims)
            if not {'latitude', 'longitude'}.issubset(focs_dims):
                raise ValueError("f_OC_sediment_data must have at least 'latitude' and 'longitude' dimensions.")
            if 'time' in focs_dims and ('time' not in NETCDF_data.dims and 'time' not in NETCDF_data.coords):
                raise ValueError("f_OC_sediment_data has a time dimension, but NETCDF_data has no usable time coordinate.")

        # origin markers aligned with filtered points
        if origin_marker == "single":
            origin_marker_np = np.arange(npts)

        fOC_SPM_global = float(self.get_config('chemical:transformations:fOC_SPM'))
        fOC_sed_global = float(self.get_config('chemical:transformations:fOC_sed'))

        FAIL_PREVIEW = 20

        for i in range(npts):
            lai = None
            loi = None
            elem_lat = None
            elem_lon = None
            time = None
            mass_ug = None
            number = None
            residual_to_seed = None
            emission_depth_value = None

            try:
                if i == 0:
                    time_start_0 = datetime.now()
                if i == 1:
                    time_start_1 = datetime.now()
                    estimated_time = (time_start_1 - time_start_0) * npts
                    print(f"Estimated time (h:min:s): {estimated_time}")

                if i in list_index_print:
                    print(".", end="")
                # Per-point lat/lon (scalar)
                lai = float(la[i])
                loi = float(lo[i])
                # Grid size in meters (scalar)
                lon_grid_m = float(
                    6.371e6 * np.cos(2.0 * np.pi * float(lai) / 360.0)
                    * lon_resol * (2.0 * np.pi) / 360.0)
                lat_grid_m_scalar = float(6.371e6 * lat_resol * (2.0 * np.pi) / 360.0)

                depth_min = None
                depth_max = None
                depth_layer_high = None
                Bathimetry_seed = None

                Bathimetry_seed = None
                if mode in ('water_conc', 'emission', 'emission_depth') and Bathimetry_seed_data is not None:
                    Bathimetry_seed = float(
                        Bathimetry_seed_data.sel(
                            latitude=float(lat_array[i]),
                            longitude=float(lon_array[i]),
                            method='nearest'
                        ).values
                    )

                    if (not np.isfinite(Bathimetry_seed)) or (Bathimetry_seed <= 0.0):
                        logger.debug(
                            "Skipping point: invalid Bathimetry_seed_data at lon %.6f, lat %.6f (value=%s)",
                            lon_array[i], lat_array[i], Bathimetry_seed
                        )
                        continue

                if mode == 'water_conc':
                    Bathimetry_datapoint = float(
                        Bathimetry_data.sel(
                            latitude=float(lai),
                            longitude=float(loi),
                            method='nearest'
                        ).values
                    )

                    if (not np.isfinite(Bathimetry_datapoint)) or (Bathimetry_datapoint <= 0.0):
                        logger.debug(
                            "Skipping point: invalid Bathimetry_data at lon %.6f, lat %.6f (value=%s)",
                            loi, lai, Bathimetry_datapoint)
                        continue

                    if depth is not None:
                        depth_datapoint = float(np.absolute(depth[i]))

                        depth_datapoint_np = np.sort(np.abs(
                            NETCDF_data.sel(latitude=lai, longitude=loi, method="nearest").depth.values
                        ).astype(float))

                        depth_datapoint_np_index = int(np.argmin(np.abs(depth_datapoint_np - depth_datapoint)))

                        if depth_layer_convention == "top":
                            depth_min = float(depth_datapoint)

                            if Bathimetry_datapoint < depth_min:
                                logger.info(
                                    "Skipping point: bathymetry (%.3f m) is shallower than layer top "
                                    "depth_min (%.3f m) at lon %.6f, lat %.6f",
                                    Bathimetry_datapoint, depth_min, loi, lai
                                )
                                continue

                            is_last = depth_datapoint_np_index >= len(depth_datapoint_np) - 1

                            if is_last:
                                if last_depth_until_bathimetry:
                                    depth_max = float(Bathimetry_datapoint)
                                else:
                                    continue
                            else:
                                depth_max = min(
                                    float(depth_datapoint_np[depth_datapoint_np_index + 1]),
                                    float(Bathimetry_datapoint))

                            if depth_max <= depth_min:
                                continue

                            depth_layer_high = depth_max - depth_min

                        elif depth_layer_convention == "bottom":
                            if depth_datapoint_np_index > 0:
                                depth_min = float(depth_datapoint_np[depth_datapoint_np_index - 1])
                            else:
                                depth_min = 0.0

                            if (depth_datapoint_np_index == len(depth_datapoint_np) - 1) and last_depth_until_bathimetry:
                                depth_max = float(Bathimetry_datapoint)
                            else:
                                depth_max = float(depth_datapoint)

                            depth_max = min(depth_max, float(Bathimetry_datapoint))

                            if depth_max <= depth_min:
                                logger.info(
                                    "Skipping point: bathymetry (%.3f m) is shallower than layer top "
                                    "depth_min (%.3f m) at lon %.6f, lat %.6f",
                                    Bathimetry_datapoint, depth_min, loi, lai
                                )
                                continue

                            depth_layer_high = depth_max - depth_min

                        else:
                            raise RuntimeError(
                                "depth_layer_convention must have been validated before entering the seeding loop.")
                    else:
                        depth_layer_high = Bathimetry_datapoint

                    if (not np.isfinite(depth_layer_high)) or (depth_layer_high <= 0.0):
                        logger.debug(
                            "Skipping point: non-positive depth_layer_high at lon %.6f, lat %.6f (value=%s)",
                            loi, lai, depth_layer_high)
                        continue

                elif mode == 'emission_depth':
                    if depth is None:
                        raise ValueError("mode='emission_depth' requires NETCDF_data to have a depth dimension.")

                    emission_depth_value = float(np.absolute(depth[i]))
                    eps = 1e-4

                    if (not np.isfinite(emission_depth_value)) or (emission_depth_value < 0.0):
                        logger.debug(
                            "Skipping point: invalid emission depth at lon %.6f, lat %.6f (value=%s)",
                            loi, lai, emission_depth_value)
                        continue

                    if Bathimetry_seed is None:
                        raise ValueError("Bathimetry_seed_data is required for mode='emission_depth'")

                    if emission_depth_value > float(Bathimetry_seed) - eps:
                        logger.info(
                            "Skipping point: requested emission depth %.3f m is deeper than/equal to "
                            "local water depth %.3f m at lon %.6f, lat %.6f",
                            emission_depth_value, Bathimetry_seed, loi, lai
                        )
                        continue

                v = values[i]

                if mode == 'water_conc' and depth_layer_high is None:
                    raise ValueError("depth_layer_high is None for water_conc; check bathymetry/depth handling.")

                if mode == 'water_conc':
                    pixel_volume = depth_layer_high * lon_grid_m * lat_grid_m_scalar
                    mass_ug = float(v) * (pixel_volume * 1e3)

                elif mode == 'sed_conc':
                    sed_mixing_depth_local = sed_mixing_depth_global
                    if active_sediment_layer_thickness_data is not None:
                        try:
                            sed_thickness_value = active_sediment_layer_thickness_data.sel(
                                latitude=float(lai),
                                longitude=float(loi),
                                method='nearest'
                            ).values
                            sed_thickness_value = float(np.asarray(sed_thickness_value).squeeze())

                            if np.isfinite(sed_thickness_value) and sed_thickness_value > 0.0:
                                sed_mixing_depth_local = sed_thickness_value
                            else:
                                logger.debug(
                                    "Invalid active_sediment_layer_thickness at lon=%s lat=%s "
                                    "(value=%s), using global chemical:sediment:mixing_depth=%s",
                                    loi, lai, sed_thickness_value, sed_mixing_depth_global)
                        except Exception as e:
                            self._record_seed_failure(
                                fail_records=fail_records,
                                stage="active_sediment_layer_thickness_lookup",
                                point_index=i,
                                exc=e,
                                lon=loi,
                                lat=lai,
                                fallback=sed_mixing_depth_global,
                            )

                    pixel_volume = sed_mixing_depth_local * (lon_grid_m * lat_grid_m_scalar)
                    pixel_sed_mass = pixel_volume * (1 - sed_porosity) * sed_density
                    mass_ug = float(v) * pixel_sed_mass

                elif mode in ('emission', 'emission_depth'):
                    mass_ug = float(v) * 1e9

                else:
                    raise ValueError("Incorrect mode")

                if mass_ug <= 0:
                    continue


                if getattr(t, "size", 1) == 1:
                    time = datetime.utcfromtimestamp(
                        int((np.array(t - np.datetime64('1970-01-01T00:00:00'))) / np.timedelta64(1, 's')))
                else:
                    time = datetime.utcfromtimestamp(
                        int((np.array(t[i] - np.datetime64('1970-01-01T00:00:00'))) / np.timedelta64(1, 's')))

                speciation_ctrl = (
                sed_speciation if mode == "sed_conc"
                else (water_speciation if mode in ("water_conc", "emission", "emission_depth") else None))

                moving_element = False if mode == 'sed_conc' else True

                needs_default_specie = (
                    speciation_ctrl is None or
                    (isinstance(speciation_ctrl, str) and speciation_ctrl in ("config", "default")))

                if needs_default_specie:
                    default_specie = self._default_seed_specie(mode)
                else:
                    default_specie = None

                if gen_mode == "mass":
                    number_full = self._get_number_of_elements(
                        g_mode="mass",
                        mass_element_ug=mass_element_ug,
                        data_point=mass_ug)

                    mass_residual = float(mass_ug - number_full * float(mass_element_ug))
                    seed_single_residual_only = (number_full == 0 and mass_residual > 0)

                    number = int(number_full)
                    mass_element_seed_ug = float(mass_element_ug)

                elif gen_mode == "fixed":
                    number = self._get_number_of_elements(g_mode="fixed", n_elements=number_of_elements)
                    if number <= 0:
                        continue
                    number = int(number)
                    mass_element_seed_ug = float(mass_ug / number)
                    mass_residual = 0.0
                    seed_single_residual_only = False

                else:
                    raise ValueError("Invalid gen_mode")

                elem_lat = float(lat_array[i])
                elem_lon = float(lon_array[i])
                origin_marker_seed = origin_marker_np[i] if origin_marker == "single" else origin_marker

                local_particle_foc = _lookup_scalar_map_value(
                    f_OC_particle_data,
                    elem_lat, elem_lon,
                    fOC_SPM_global,
                    'f_OC_particle_data',
                    time=time,
                    point_index=i,
                )
                local_sediment_foc = _lookup_scalar_map_value(
                    f_OC_sediment_data,
                    elem_lat, elem_lon,
                    fOC_sed_global,
                    'f_OC_sediment_data',
                    time=time,
                    point_index=i,
                )

                # main batch
                if number > 0:
                    z = self._get_z(
                    mode=mode,
                    number=number,
                    NETCDF_data_dim_names=NETCDF_data_dim_names,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth_seed=Bathimetry_seed if mode in ('water_conc', 'emission', 'emission_depth') else None,
                    emission_placement=emission_placement,
                    emission_seafloor_eps=emission_seafloor_eps,
                    emission_depth_value=emission_depth_value if mode == 'emission_depth' else None,)

                    spec_arr = self.build_specie_array(
                        n=number,
                        speciation=speciation_ctrl,
                        default_specie=default_specie)

                    foc_seed = _build_seed_foc_array(
                        number, mode, local_particle_foc, local_sediment_foc)

                    kwargs_seed = dict(
                        lon=elem_lon,
                        lat=elem_lat,
                        radius=radius,
                        number=number,
                        time=time,
                        mass=mass_element_seed_ug,
                        mass_degraded=0,
                        mass_volatilized=0,
                        moving=moving_element,
                        z=z,
                        origin_marker=origin_marker_seed)

                    if spec_arr is not None:
                        kwargs_seed["specie"] = spec_arr
                    elif mode == "sed_conc":
                        raise RuntimeError(
                            "sed_speciation resolved to None, which is not supported for sediment seeding.")

                    if foc_seed is not None:
                        kwargs_seed["f_OC"] = foc_seed

                    try:
                        self.seed_elements(**kwargs_seed)

                    except Exception as e_batch:
                        self._record_seed_failure(
                            fail_records=fail_records,
                            stage="main_batch_seed",
                            point_index=i,
                            exc=e_batch,
                            mode=mode,
                            lon=elem_lon,
                            lat=elem_lat,
                            time=_time_repr(time),
                            mass_ug=float(mass_ug),
                            number=int(number),
                            seed_mass_ug=float(mass_element_seed_ug),
                            origin_marker=origin_marker_seed,
                            z_kind="string" if isinstance(z, str) else "array",
                        )

                        # Fallback for sediments only (coastal failures due to radius sampling)
                        if mode == "sed_conc":
                            spec_choices = kwargs_seed.get("specie", None)

                            for k in range(number):
                                try:
                                    kwargs_one = dict(kwargs_seed)
                                    kwargs_one["number"] = 1
                                    kwargs_one["mass"] = mass_element_seed_ug

                                    if spec_choices is not None:
                                        kwargs_one["specie"] = int(np.asarray(spec_choices).ravel()[k])
                                    else:
                                        kwargs_one["specie"] = int(default_specie)

                                    if isinstance(z, str):
                                        kwargs_one["z"] = z
                                    else:
                                        kwargs_one["z"] = np.asarray(z, dtype=float).ravel()[k:k+1]

                                    if "f_OC" in kwargs_seed:
                                        kwargs_one["f_OC"] = np.asarray(
                                            kwargs_seed["f_OC"], dtype=float).ravel()[k:k+1]

                                    self.seed_elements(**kwargs_one)

                                except Exception as e_single:
                                    self._record_seed_failure(
                                        fail_records=fail_records,
                                        stage="sediment_single_fallback_seed",
                                        point_index=i,
                                        exc=e_single,
                                        mode=mode,
                                        lon=elem_lon,
                                        lat=elem_lat,
                                        time=_time_repr(time),
                                        mass_ug=float(mass_ug),
                                        element_offset=int(k),
                                        seed_mass_ug=float(mass_element_seed_ug),
                                        origin_marker=origin_marker_seed,
                                    )

                # residual (mass mode only)
                if gen_mode == "mass":
                    residual_to_seed = float(mass_ug) if seed_single_residual_only else float(mass_residual)

                    if residual_to_seed > 0:
                        z_res = self._get_z(
                                    mode=mode,
                                    number=1,
                                    NETCDF_data_dim_names=NETCDF_data_dim_names,
                                    depth_min=depth_min,
                                    depth_max=depth_max,
                                    depth_seed=Bathimetry_seed if mode in ('water_conc', 'emission', 'emission_depth') else None,
                                    emission_placement=emission_placement,
                                    emission_seafloor_eps=emission_seafloor_eps,
                                    emission_depth_value=emission_depth_value if mode == 'emission_depth' else None,
                                )

                        spec_res = self.build_specie_array(
                            n=1,
                            speciation=speciation_ctrl,
                            default_specie=default_specie)

                        foc_res = _build_seed_foc_array(
                            1, mode, local_particle_foc, local_sediment_foc)

                        kwargs_res = dict(
                            lon=elem_lon,
                            lat=elem_lat,
                            radius=radius,
                            number=1,
                            time=time,
                            mass=residual_to_seed,
                            mass_degraded=0,
                            mass_volatilized=0,
                            moving=moving_element,
                            z=z_res,
                            origin_marker=origin_marker_seed)

                        if spec_res is not None:
                            kwargs_res["specie"] = spec_res
                        elif mode == "sed_conc":
                            raise RuntimeError(
                                "sed_speciation resolved to None, which is not supported for sediment seeding.")

                        if foc_res is not None:
                            kwargs_res["f_OC"] = foc_res

                        try:
                            self.seed_elements(**kwargs_res)

                        except Exception as e_res:
                            self._record_seed_failure(
                                fail_records=fail_records,
                                stage="residual_seed",
                                point_index=i,
                                exc=e_res,
                                mode=mode,
                                lon=elem_lon,
                                lat=elem_lat,
                                time=_time_repr(time),
                                mass_ug=float(mass_ug),
                                residual_mass_ug=float(residual_to_seed),
                                origin_marker=origin_marker_seed,
                                z_kind="string" if isinstance(z_res, str) else "array",
                            )

            except Exception as e_point:
                self._record_seed_failure(
                    fail_records=fail_records,
                    stage="point_processing",
                    point_index=i,
                    exc=e_point,
                    mode=mode,
                    lon=loi,
                    lat=lai,
                    elem_lon=elem_lon,
                    elem_lat=elem_lat,
                    time=_time_repr(time),
                    mass_ug=float(mass_ug) if mass_ug is not None and np.isfinite(mass_ug) else None,
                    number=int(number) if number is not None else None,
                    residual_mass_ug=float(residual_to_seed)
                    if residual_to_seed is not None and np.isfinite(residual_to_seed) else None,
                )
                continue

        self._last_seed_from_netcdf_failures = fail_records

        if fail_records:
            stage_counts = Counter(rec["stage"] for rec in fail_records)
            logger.warning(
                "Seeding completed with %s recorded failures. Stage counts: %s",
                len(fail_records), dict(stage_counts))

            for rec in fail_records[:FAIL_PREVIEW]:
                logger.warning("seed_from_NETCDF failure record: %s", rec)

            if len(fail_records) > FAIL_PREVIEW:
                logger.warning(
                    "... %s additional seed_from_NETCDF failure records omitted from log preview",
                    len(fail_records) - FAIL_PREVIEW)
        else:
            logger.info("Seeding completed with 0 failures.")

    ### Helpers for regrid_conc
    def interp_weights(self, xyz, uvw):
        """
        Calculate interpolation weights within regrid_conc function
        # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
        """
        import scipy.spatial as sp

        tri = sp.Delaunay(xyz)
        simplex = tri.find_simplex(uvw)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        d=2                                               ## CHECK
        delta = uvw - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def interpolate_regrid(self, values, vtx, wts):
        """
        Interpolate the value of each concentration gridpoint within regrid_conc function
        """
        return np.einsum('nj,nj->n', np.take(values, vtx), wts)

    def regrid_dataarray(self,
                         mode,
                         ds,
                         new_lat, new_lon,
                         lonlat_2d, lonlat_2d_new,
                         new_lon_coords, new_lat_coords,
                         variable,
                         time_name, **kwargs):
        '''
        Regrid output of "write_netcdf_chemical_density_map" or "calculate_water_sediment_conc" functions
        depending on the value of "mode"
        Used inside "regrid_conc" function

        mode:         string, "chemical_density_map" or "wat_sed_map"
        ds:           xarray DataSet containing "variable", topo (density, optional) dataarrays to be regridded
        variable:     string, name of concentration variable to be regridded within ds
        time_name:    string, name of time dimension (time or time_avg)

        '''
        import xarray as xr

        # create an empty arrays to store the regridded data
        if mode == "chemical_density_map":
            regridded_data = np.zeros((ds.sizes[time_name], ds.sizes['specie'], ds.sizes['depth'], new_lat.shape[1], new_lon.shape[0]))
        else:
            regridded_data = np.zeros((ds.sizes['time'], ds.sizes['depth'], new_lat.shape[1], new_lon.shape[0]))

        first=True
        if mode == "chemical_density_map":
            # loop over every value of avg_time, specie, and depth
            for t, s, d in np.ndindex(ds.sizes[time_name], ds.sizes['specie'], ds.sizes['depth']):
                # select the data for the current time, specie, and depth
                points = ds[variable][t,s,d,:,:].values.reshape(-1)

                if first:
                    # Store the weights for the interpolation
                    vtx, wts = self.interp_weights(lonlat_2d, lonlat_2d_new)
                    first=False

                # interpolate the concentration data onto the new grid using griddata
                new_concentration_2d = self.interpolate_regrid(points, vtx, wts)
                # store the interpolated data in the regridded_data array
                regridded_data[t, s, d] = np.transpose(np.reshape(new_concentration_2d,(len(new_lon_coords),len(new_lat_coords))))
        else:
            # loop over every value of time, and depth
            for t, d in np.ndindex(ds.sizes['time'], ds.sizes['depth']):
                points = ds[variable][t,d,:,:].values.reshape(-1)

                if first:
                    # Store the weights for the interpolation
                    vtx, wts = self.interp_weights(lonlat_2d, lonlat_2d_new)
                    first=False

                # interpolate the concentration data onto the new grid using griddata
                new_concentration_2d = self.interpolate_regrid(points, vtx, wts)
                # store the interpolated data in the regridded_avg_data array
                regridded_data[t, d] = np.transpose(np.reshape(new_concentration_2d,(len(new_lon_coords),len(new_lat_coords))))


        if mode == "chemical_density_map":
            # create a new xarray dataarray with the regridded data
            regridded_data = xr.DataArray(regridded_data, coords=[ds[time_name], ds['specie'], ds['depth'], new_lat_coords, new_lon_coords], dims=[time_name, 'specie', 'depth', 'latitude', 'longitude'])
        else:
            regridded_data = xr.DataArray(regridded_data, coords=[ds['time'], ds['depth'], new_lat_coords, new_lon_coords], dims=['time', 'depth', 'latitude', 'longitude'])

        regridded_data['latitude'] = regridded_data['latitude'].assign_attrs(standard_name='latitude')
        regridded_data['latitude'] = regridded_data['latitude'].assign_attrs(long_name='latitude')
        regridded_data['latitude'] = regridded_data['latitude'].assign_attrs(units='degrees_north')
        regridded_data['latitude'] = regridded_data['latitude'].assign_attrs(axis='Y')

        regridded_data['longitude'] = regridded_data['longitude'].assign_attrs(standard_name='longitude')
        regridded_data['longitude'] = regridded_data['longitude'].assign_attrs(long_name='longitude')
        regridded_data['longitude'] = regridded_data['longitude'].assign_attrs(units='degrees_east')
        regridded_data['longitude'] = regridded_data['longitude'].assign_attrs(axis='X')

        return regridded_data, vtx, wts

    def regrid_conc(self, filename, filename_regridded, latmin, latmax, latstep, lonmin, lonmax, lonstep, mode = None,
                    variables = None, concfile = None,
                    lon_2d_ncdm = None, lat_2d_ncdm = None):
        """
        Interpolate "write_netcdf_chemical_density_map" or "calculate_water_sediment_conc" output to regular lat/lon grid
            filename:               string, path or filename of "write_netcdf_chemical_density_map" output file to be regridded
            filename_regridded:     string, path or filename of regridded output
            latmin:                 float 32, min latitude of new grid
            latmax:                 float 32, max latitude of new grid
            latstep:                float 32, latitude resolution of new grid, in degrees
            lonmin:                 float 32, min longitude of new grid
            lonmax:                 float 32, max longitude of new grid
            lonstep:                float 32 longitude resolution of new grid, in degrees
            variables:              list, list of variables' name to be regridded within DataSet
            concfile:               xarray Dataset of "write_netcdf_chemical_density_map" or
                                    "calculate_water_sediment_conc" output file to be regridded
            lon_2d_ncdm:            array of float 64, flattened array of ds['lon'].values.flatten() from ncdm
            lat_2d_ncdm:            array of float 64, flattened array of ds['lat'].values.flatten() from ncdm
        """
        import numpy as np
        import xarray as xr
        from datetime import datetime as dt

        if ((concfile is None) and (filename is not None)):
            print("Loading concentration file from filename")
            ds = xr.open_dataset(filename)
        else:
            ds = concfile

        ds.load()
        start=dt.now()
        variable_ls = ['concentration', 'concentration_avg',
                       'concentration_smooth', 'concentration_smooth_avg',
                       'concentration_avg_sediments', 'concentration_avg_sediments',
                       'density', 'density_avg', 'topo']

        if variables is not None:
            variable_ls = variables

        if mode is None:
            # Define if output of "write_netcdf_chemical_density_map" or
            # of "calculate_water_sediment_conc" was given as input
            if "latitude" not in ds.dims:
                mode = "chemical_density_map"
                lat_name = "lat"
                lon_name = "lon"
            else:
                mode = "wat_sed_map"
                lat_name = "latitude"
                lon_name = "longitude"
                if ((lon_2d_ncdm is None) or (lat_2d_ncdm is None)):
                    raise ValueError("lat/lon_2d_ncdm unspecified")
        print(f"mode: {mode}, variables: {variable_ls}")

        if (latmin < min(ds[lat_name].values.flatten()) or latmax > max(ds[lat_name].values.flatten())\
        or lonmin < min(ds[lon_name].values.flatten()) or lonmax > max(ds[lon_name].values.flatten())):
            if latmin < min(ds[lat_name].values.flatten()):
                print(f"latmin ({latmin}) is not in range, should not be lower than: {min(ds[lat_name].values.flatten())}")
            if latmax > max(ds[lat_name].values.flatten()):
                print(f"latmax ({latmax}) is not in range, should not be higher than: {max(ds[lat_name].values.flatten())}")
            if lonmin < min(ds[lon_name].values.flatten()):
                print(f"lonmin ({lonmin}) is not in range: should not be lower than: {min(ds[lon_name].values.flatten())}")
            if lonmax > max(ds[lon_name].values.flatten()):
                print(f"lonmax ({lonmax}) is not in range, should not be higher than: {max(ds[lon_name].values.flatten())}")

            raise ValueError("Regrid coordinates out of bounds from input file range")
        else:
            pass

        if "time" in ds.dims:
            time_name = "time"
        else:
            time_name = "avg_time"

        new_lat_coords = np.arange(latmin,latmax,latstep)
        new_lon_coords = np.arange(lonmin,lonmax,lonstep)

        # define new grid of latitude and longitude
        new_lat, new_lon = np.meshgrid(new_lat_coords, new_lon_coords)

        # create 2D array of (y*x,) coordinates from the 2D (y,x) lat-lon grid
        if mode == "chemical_density_map":
            lon_2d = ds[lon_name].values.flatten()
            lat_2d = ds[lat_name].values.flatten()
        else:
            lat_2d = lat_2d_ncdm
            lon_2d = lon_2d_ncdm

        lonlat_2d = np.column_stack((lat_2d, lon_2d))

        # create 2D array of the new coordinates
        lonlat_2d_new=np.column_stack((new_lat.flatten(),new_lon.flatten()))

        regridded_vars_dict = {}
        for variable in variable_ls:
            if (variable in ds.data_vars) and (variable != 'topo'):
                print(variable)
                ds_attrs = ds[variable].attrs
                for key in ["lon_resol", "lat_resol", "grid_mapping"]:
                    ds_attrs.pop(key, None)

                regridded_variable, vtx, wts = self.regrid_dataarray(mode = mode,
                                                         ds = ds,
                                                         variable = variable,
                                                         new_lat = new_lat, new_lon = new_lon,
                                                         lonlat_2d = lonlat_2d,
                                                         lonlat_2d_new = lonlat_2d_new,
                                                         new_lon_coords = new_lon_coords,
                                                         new_lat_coords = new_lat_coords,
                                                         time_name = time_name)

                regridded_variable.name = variable
                regridded_variable.attrs['grid_mapping'] = "+proj=longlat +datum=WGS84 +no_defs"
                regridded_variable.attrs.update(ds_attrs)
                regridded_variable.attrs['lon_resol'] = str(np.around(abs(new_lon[0][0]-new_lon[1][0]), decimals = 8)) + " degrees E"
                regridded_variable.attrs['lat_resol'] = str(np.around(abs(new_lat[0][0]-new_lat[0][1]), decimals = 8)) + " degrees N"
                # change negative values to 0
                regridded_variable =  regridded_variable.clip(min = 0)
                regridded_vars_dict[variable] = regridded_variable

        if "topo" in list(ds.keys()):
            topo_attrs = ds["topo"].attrs
            for key in ["lon_resol", "lat_resol", "grid_mapping"]:
                topo_attrs.pop(key, None)

            regridded_topo_data = np.zeros((new_lat.shape[1], new_lon.shape[0]))
            # regrid topography
            points = ds.topo[:,:].values.reshape(-1)

            # interpolate the concentration data onto the new grid using griddata
            new_topo_2d = self.interpolate_regrid(points, vtx, wts)
            # store the interpolated data in the regridded_topo array
            regridded_topo_data = np.transpose(np.reshape(new_topo_2d,(len(new_lon_coords),len(new_lat_coords))))

            # create a new xarray dataarray with the topography regridded data
            regridded_topo = xr.DataArray(regridded_topo_data, coords=[new_lat_coords, new_lon_coords], dims=['latitude', 'longitude'])

            regridded_topo.name = "topo"
            regridded_topo.attrs['grid_mapping'] = "+proj=longlat +datum=WGS84 +no_defs"
            regridded_topo.attrs.update(topo_attrs)
            regridded_topo.attrs['lon_resol'] = str(np.around(abs(new_lon[0][0]-new_lon[1][0]), decimals = 8)) + " degrees E"
            regridded_topo.attrs['lat_resol'] = str(np.around(abs(new_lat[0][0]-new_lat[0][1]), decimals = 8)) + " degrees N"
            # change negative topography values to np.nan
            regridded_topo = xr.where(regridded_topo < 0, np.nan, regridded_topo)
            regridded_topo['latitude'] = regridded_topo['latitude'].assign_attrs(standard_name='latitude')
            regridded_topo['latitude'] = regridded_topo['latitude'].assign_attrs(long_name='latitude')
            regridded_topo['latitude'] = regridded_topo['latitude'].assign_attrs(units='degrees_north')
            regridded_topo['latitude'] = regridded_topo['latitude'].assign_attrs(axis='Y')

            regridded_topo['longitude'] = regridded_topo['longitude'].assign_attrs(standard_name='longitude')
            regridded_topo['longitude'] = regridded_topo['longitude'].assign_attrs(long_name='longitude')
            regridded_topo['longitude'] = regridded_topo['longitude'].assign_attrs(units='degrees_east')
            regridded_topo['longitude'] = regridded_topo['longitude'].assign_attrs(axis='X')
            regridded_vars_dict['topo'] = regridded_topo

        print(f"Time elapsed (hr:min:sec): {dt.now()-start}")
        print("Saving to netcdf")
        # save regridded data
        regridded_dataset = xr.Dataset(regridded_vars_dict)
        regridded_dataset.to_netcdf(filename_regridded)

        print(f"Time elapsed (hr:min:sec): {dt.now()-start}")

    @staticmethod
    def _save_masked_DataArray(DataArray_masked,
                              file_output_path,
                              file_output_name):
        import os
        import xarray as xr

        if not os.path.exists(file_output_path):
            os.makedirs(file_output_path)
            print("file_output_path did not exist and was created")
        else:
            pass
        print(f"Saving to {file_output_path}")

        if 'grid_mapping' in DataArray_masked.attrs:
            del DataArray_masked.attrs['grid_mapping'] # delete grid_mapping attribute to avoid "ValueError in safe_setitem" from xarray
        else:
            pass
        full_output_path = os.path.join(file_output_path, file_output_name)
        try:
            DataArray_masked.to_netcdf(full_output_path)
        except Exception as e:
            print(f"Error writing netcdf ({e}), trying fallback path...")
            # Change DataArray_masked to dataset if xarray.core.dataarray.DataArray
            if isinstance(DataArray_masked, xr.DataArray):
                DataArray_masked = DataArray_masked.to_dataset()
                print("Changed DataArray_masked from DataArray to DataSet")
            # Remove "_FillValue" = np.nan from data_vars and coordinates attributes
            # Change "_FillValue" to -9999, to avoid "ValueError: cannot convert float NaN to integer"
            for var_name, var in DataArray_masked.variables.items():
                if "_FillValue" in var.attrs:
                    del var.attrs["_FillValue"]
                    var.attrs["_FillValue"] = -9999
                    print(f"Changed _FillValue of {var_name} from NaN to -9999")

            for coord_name, coord in DataArray_masked.coords.items():
                if "_FillValue" in coord.attrs:
                    del coord.attrs["_FillValue"]
                    coord.attrs["_FillValue"] = -9999
                    print(f"Changed _FillValue of {coord_name} from NaN to -9999")

            DataArray_masked.to_netcdf(file_output_path + file_output_name)

    @staticmethod
    def _mask_DataArray(DataArray,
                       shp_mask,
                       shp_epsg = "epsg:4326",
                       invert_shp = False,
                       drop_data = False):
        '''
        Mask xarray DataArray using shapefile, return a masked xarray DataArray
        See mask_netcdf_map function for details
        '''

        from shapely.geometry import mapping

        if ("latitude" in DataArray.dims) and ("longitude" in DataArray.dims):
            DataArray = DataArray.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            print("latitude/longitude dimensions used")
        elif ("lat" in DataArray.dims) and ("lon" in DataArray.dims):
            DataArray = DataArray.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            print("lat/lon dimensions used")
        elif ("x" in DataArray.dims) and ("y" in DataArray.dims):
            DataArray = DataArray.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
            print("x/y dimensions used")
        else:
            raise ValueError("Unspecified lat/lon dimensions in DataArray")

        if DataArray.rio.crs is None:
            DataArray = DataArray.rio.write_crs(shp_epsg, inplace=True)
            print(f"DataArray.rio.crs not present, shp_epsg used: {shp_epsg}")
        else:
            if DataArray.rio.crs != shp_epsg:
                # shp_mask = shp_mask.to_crs(DataArray.rio.crs)
                raise ValueError("DataArray and shp have different crs systems")

        DataArray_masked = DataArray.rio.clip(shp_mask.geometry.apply(mapping), shp_mask.crs, drop=drop_data, invert = invert_shp)

        if "lat" in DataArray_masked.dims:
            DataArray_masked = DataArray_masked.rename({'lat': 'latitude','lon': 'longitude'})
        if "x" in DataArray_masked.dims:
            DataArray_masked = DataArray_masked.rename({'y': 'latitude','x': 'longitude'})

        DataArray_masked['latitude'] = DataArray_masked['latitude'].assign_attrs(standard_name='latitude')
        DataArray_masked['latitude'] = DataArray_masked['latitude'].assign_attrs(long_name='latitude')
        DataArray_masked['latitude'] = DataArray_masked['latitude'].assign_attrs(units='degrees_north')
        DataArray_masked['latitude'] = DataArray_masked['latitude'].assign_attrs(axis='Y')

        DataArray_masked['longitude'] = DataArray_masked['longitude'].assign_attrs(standard_name='longitude')
        DataArray_masked['longitude'] = DataArray_masked['longitude'].assign_attrs(long_name='longitude')
        DataArray_masked['longitude'] = DataArray_masked['longitude'].assign_attrs(units='degrees_east')
        DataArray_masked['longitude'] = DataArray_masked['longitude'].assign_attrs(axis='X')

        return DataArray_masked

    @staticmethod
    def _check_extra_dimensions(Dataset,
                                permitted_dims):
        '''
        Check if dimensions other than lat/lon/time/depth are present.
        Other permitted_dims can be spedified as a list
        If extra dimensions are present data_var will not be masked
        '''
        acceptable_dimensions = set(['lat', 'lon', 'latitude', 'longitude', 'x', 'y',
                                     'time', 'avg_time', 'depth', 'z'] + permitted_dims)
        Dataset_dimensions = set(Dataset.dims)

        extra_dimensions = Dataset_dimensions - acceptable_dimensions
        return extra_dimensions

    @staticmethod
    def _merge_masked_dataset(DataArray_ls):
        '''
        Store DataArrays into a single DataSet
        '''
        import xarray as xr

        merged_dataset = xr.Dataset({name[0]: data_array.to_dataarray() for name, data_array in DataArray_ls})
        merged_dataset = merged_dataset.drop_vars(['variable', 'spatial_ref'])
        # Remove dimensions without coordinates
        for variable in merged_dataset.data_vars:
            dims_with_coords = set([dim for dim in merged_dataset[variable].dims if dim in merged_dataset[variable].coords])
            all_dims = set(list(merged_dataset[variable].dims))
            uncommon_dimensions = list(all_dims.symmetric_difference(dims_with_coords))

            for uncommon_dim in uncommon_dimensions:
                if uncommon_dim in merged_dataset.dims:
                    merged_dataset[variable] = merged_dataset[variable].sel({uncommon_dim: merged_dataset[uncommon_dim][0]})

        return(merged_dataset)

    def mask_netcdf_map(self,
                        shp_mask_file,
                        file_path = None,
                        file_name = None,
                        DataArray = None,
                        shp_epsg = "epsg:4326",
                        invert_shp = False,
                        drop_data = False,
                        save_masked_file = False,
                        file_output_path = None,
                        file_output_name = None,
                        permitted_dims = []
                         ):
        '''
        Mask xarray DataArray using shapefile, return a masked xarray DataArray
            Used for xarray DataArray with regular lat/lon coordinates.
            "write_netcdf_chemical_density_map" output must be regridded to regular lat/lon coordinates with "regrid_conc" function

        shp_mask_file:       string, full path to mask shapefile
        DataArray:           xarray DataArray to be masked loaded with rioxarray.open_rasterio(DataArray)
                                 *latitude/longitude, lat/lon, y/x are accepted as coordinates
        shp_epsg:            string, reference system of shp file (e.g. "epsg:4326")
        invert_shp:          boolean, select if values inside (False) or outside (True) shp are masked
        drop_data:           boolean, select if spatial extent of DataArray is mantained (False) or reduced to the extent of shp (True)
        save_masked_file:    boolean,select if DataArray_masked is saved (True) or returned (False)
        file_path:           string, path of the file to be masked. Must end with /
        file_name:           string, name of the DataArray to be masked (.nc)
        file_output_path:    string, path of the file to be saved. Must end with /
        file_output_name:    string, name of the DataArray_masked output file (.nc)
        permitted_dims:      list, name of dimensions in input file acceped for masking
        '''
        import geopandas as gpd
        import rioxarray

        shp_mask = gpd.read_file(shp_mask_file)
        if hasattr(shp_mask, "crs"):
            shp_epsg = shp_mask.crs
            print(f"shp_crs taken from shapefile: {shp_epsg} ")
        elif shp_epsg is not None:
            print(f"shp_mask.crs not present, specified shp_epsg used: {shp_epsg}")
        else:
            raise ValueError("shp_mask.crs not present and shp_epsg not specified ")

        if DataArray is not None:

            extra_dims = self._check_extra_dimensions(Dataset = DataArray,
                                                      permitted_dims = permitted_dims)
            if extra_dims:
                raise ValueError(f'Unpermitted dimensions {extra_dims} are present, check permitted_dims')

            # Mask input DataArray
            DataArray_masked = self._mask_DataArray(DataArray = DataArray,
                               shp_mask = shp_mask,
                               shp_epsg = shp_epsg,
                               invert_shp = invert_shp,
                               drop_data = drop_data)
            if save_masked_file is True:
                self._save_masked_DataArray(DataArray_masked = DataArray_masked,
                                      file_output_path = file_output_path,
                                      file_output_name = file_output_name)
            else:
                return(DataArray_masked)

        elif file_path is not None and file_name is not None:
                print("Loading DataArray from disk")
                DataArray = rioxarray.open_rasterio(file_path + file_name)

                if not isinstance(DataArray, list):
                    # File to mask contains only one variable
                    extra_dims = self._check_extra_dimensions(Dataset = DataArray,
                                                              permitted_dims = permitted_dims)
                    if extra_dims:
                        raise ValueError(f'Unpermitted dimensions {extra_dims} are present, check permitted_dims')

                    DataArray_masked = self._mask_DataArray(DataArray = DataArray,
                                           shp_mask = shp_mask,
                                           shp_epsg = shp_epsg,
                                           invert_shp = invert_shp,
                                           drop_data = drop_data)
                    if save_masked_file is True:
                        self._save_masked_DataArray(DataArray_masked = DataArray_masked,
                                              file_output_path = file_output_path,
                                              file_output_name = file_output_name)
                    else:
                        return(DataArray_masked)
                else:
                    # Masked file contains more than one variable
                    # Store masked and not masked DataArrays
                    DataArray_masked_ls = []
                    DataArray_not_masked_ls = []
                    masked_dataset = None
                    not_masked_dataset = None

                    for Dataset in DataArray:
                        extra_dims = self._check_extra_dimensions(Dataset = Dataset,
                                                                  permitted_dims = permitted_dims)
                        if extra_dims:
                            print(f"Extra dimensions found for {str(Dataset.data_vars)[20:]}")
                            print("Returning original DataArray")
                            DataArray_not_masked_ls.append([list(Dataset.data_vars), Dataset])
                        else:
                            print(f"Masking {str(Dataset.data_vars)[20:]}")
                            DataArray_masked = self._mask_DataArray(DataArray = Dataset,
                                                   shp_mask = shp_mask,
                                                   shp_epsg = shp_epsg,
                                                   invert_shp = invert_shp,
                                                   drop_data = drop_data)
                            DataArray_masked_ls.append([list(DataArray_masked.data_vars), DataArray_masked])

                    if len(DataArray_masked_ls) > 0:
                        masked_dataset = self._merge_masked_dataset(DataArray_masked_ls)
                    if len(DataArray_not_masked_ls) > 0:
                        not_masked_dataset = self._merge_masked_dataset(DataArray_not_masked_ls)

                    if save_masked_file is True:
                        if masked_dataset is not None:
                            self._save_masked_DataArray(DataArray_masked = masked_dataset,
                                                  file_output_path = file_output_path,
                                                  file_output_name = file_output_name[:-3] + "_MASKED_VARS.nc")
                        else:
                            print("No data_var was masked")
                        if not_masked_dataset is not None:
                            self._save_masked_DataArray(DataArray_masked = not_masked_dataset,
                                                  file_output_path = file_output_path,
                                                  file_output_name = file_output_name[:-3] + "_NOT_MASKED_VARS.nc")
                        else:
                            pass
                    else:
                        if masked_dataset is not None:
                            return(masked_dataset)
                        else:
                            print("No data_var was masked")
        else:
            raise ValueError("DataArray or file_path/file_name not specified")

    ##### Helpers for create_images
    @staticmethod
    def _simmetrical_colormap(cmap):
        '''
        Take a colormap and create a new one, as the concatenation of itself by a symmetrical fold around 0
        from https://stackoverflow.com/questions/28439251/symmetric-colormap-matplotlib

        cmap:     matplotlib colormap that will be returned symmetrical with respect to 0
        '''
        import numpy as np
        import matplotlib.colors as mcolors

        new_cmap_name = "sym_" + cmap.name
        # Define the roughness of the colormap, default is 128
        n= 128
        # get the list of color from colormap
        colors_r = cmap(np.linspace(0, 1, n))    # take the standard colormap # 'right-part'
        colors_l = colors_r[::-1]                # take the first list of color and flip the order # "left-part"

        # combine them and build a new colormap
        colors = np.vstack((colors_l, colors_r))
        new_cmap = mcolors.LinearSegmentedColormap.from_list(new_cmap_name, colors)

        return new_cmap

    @staticmethod
    def _remove_white_borders(image, padding_r, padding_c, tol=1e-6):
        '''
        Remove white borders from an image

        image:     np.array of float32, rgb array of image with white = 1
        '''
        # supports RGB or RGBA
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        mask = np.any(image < 1 - tol, axis=2) if image.ndim == 3 else (image < 1 - tol)
        if not np.any(mask):
            return image  # nothing to trim
        # Get the non-zero pixels along each axis
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        # Get the bounding box of non-zero pixels
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(rmin - padding_r, 0); rmax = min(rmax + padding_r, image.shape[0]-1)
        cmin = max(cmin - padding_c, 0); cmax = min(cmax + padding_c, image.shape[1]-1)
        # Crop the image to the bounding box
        return image[rmin:rmax+1, cmin:cmax+1]

    @staticmethod
    def _create_animation(load_img_from_folder,
                       trim_images,
                       figure_ls,
                       file_out_path,
                       file_out_sub_folder,
                       anim_prefix,
                       figure_file_name,
                       animation_format,
                       fps,
                       width_fig, high_fig,
                       low_quality):
        '''
        Make .mp4 or .gif animation of figures created with create_images
        '''
        # https://stackoverflow.com/questions/67420158/how-do-you-make-a-matplotlib-funcanimation-animation-out-of-matplotlib-image-axe
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        from datetime import datetime as dt
        import matplotlib.pyplot as plt
        import numpy as np

        start = dt.now()
        def update(frame):
            # frame is rgb np.array
            # Update the image in the plot
            art = (figure_ls[frame])
            draw_image.set_array(art)
            ax.set_axis_off()
            return [draw_image]


        if low_quality == True:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize = (width_fig,high_fig))

        ax = plt.gca()
        draw_image = ax.imshow((figure_ls[0]),animated=True)

        # Create the animation
        print("Creating animation")
        animation = FuncAnimation(fig, update, frames=len(figure_ls), interval=1000/fps, blit = True)

        output_video = file_out_path + file_out_sub_folder + anim_prefix + figure_file_name + animation_format
        print(f"Time to create animation (hr:min:sec): {dt.now()-start}")
        print(f"Saving animation to {file_out_path + file_out_sub_folder}")
        start = dt.now()
        writer = FFMpegWriter(fps=fps) if animation_format == ".mp4" else PillowWriter(fps=fps)
        animation.save(output_video, writer=writer)

        print(f"Time to save animation (hr:min:sec): {dt.now()-start}")

    @staticmethod
    def _flatten_list(matrix):
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list

    @staticmethod
    def _check_nested_list(input_list):
        '''
        Check if fig_numbers is a nested list
        '''
        for element in input_list:
            if isinstance(element, list):
                return True
        return False

    @staticmethod
    def _print_progress_list(length):
        '''
        Create list of indexes to print progress of creating/saving images

        length:       float, lenght of figures array
        '''
        elem_print = []
        if length < 10:
             for index in range(0, length):
                 elem_print.append(index)
             return elem_print

        interval = length // 10  # Calculate the interval
        for i in range(1, 11):
             index = i * interval
             if index < length:
                 elem_print.append(index)
             else:
                 break
        return elem_print


    @staticmethod
    def _check_imgs_memory_use(width_fig, high_fig,
                              fig_dpi, figures_number,
                              trim_images,
                              make_animation,
                              color_depth = 4): # Bytes per pixel (default is 4 for RGBA)
        '''
        Check if enought memory can be allocated to create/trim/animate images

        color_depth :     int, Bytes per pixel (default is 4 for RGBA)

        '''
        import psutil

        total_ram = psutil.virtual_memory().total
        total_ram_gb = (total_ram / (1024 ** 3))

        # Calculate dimensions in pixels of figure
        width_px = width_fig * fig_dpi
        height_px = high_fig * fig_dpi
        # Total pixels
        total_pixels = (width_px * height_px) * figures_number
        # Memory used in gigabytes
        fig_memory_gb = (total_pixels * color_depth)/ (1024 ** 3)
        mult = 1 + (1 if make_animation else 0) + (0.1 if trim_images else 0)  # memory for trimmed images and animation
        fig_memory_gb *= mult

        if fig_memory_gb >= total_ram_gb*0.8:
            print("WARNING: More than 80% of available RAM will be necessary")
            if fig_memory_gb >= total_ram_gb:
                raise MemoryError(f"Memory needed to create/trim/animate figures {fig_memory_gb} GB exceeds available RAM {total_ram_gb} GB")


    def create_images(self,
                      Conc_Dataset,
                      time_start,
                      time_end,
                      long_min, long_max,
                      lat_min, lat_max,
                      file_out_path,
                      file_out_sub_folder,
                      figure_file_name,
                      shp_file_path,
                      title_caption,
                      unit_measure,
                      full_title = None,
                      vmin = None,
                      vmax = None,
                      selected_colormap = None,
                      colormap_norm = None,
                      levels_colormap = None,
                      simmetrical_cmap = False,
                      scientific_colorbar = False,
                      colorbar_title = None,
                      selected_depth = 0,
                      fig_format = ".jpg",
                      fig_dpi = 100,
                      make_animation = False,
                      concat_animation = False,
                      animation_format = ".mp4",
                      fps = 8,
                      load_img_from_folder = False,
                      fig_numbers = None,
                      add_shp_to_figure = False,
                      variable_name = None,
                      labels_font_sizes = [30,30,30,25,25,25,25],
                      shp_color = "black",
                      trim_images = True,
                      save_figures = True,
                      shading = None,
                      date_str_lenght = 10,
                      width_fig = 26, high_fig =15,
                      padding_r = 0, padding_c = 0,
                      low_quality = False):
        '''
        Create a series of .jpg or .png for each timestep of a concentration map
        from REGRIDDED "calculate_water_sediment_conc" function output

        Conc_Dataset:         xarray dataset of concentration after calculate_water_sediment_conc
                                *latitude, degrees N
                                *longitude, degrees E
                                *time, datetime64[ns]
                                *depth, meters (optional)
        time_start:           datetime64[ns], start time of figures
        time_end:             datetime64[ns], end time of figures
        long_min:             float64, min longitude of figure
        long_max              float64, max longitude of figure
        lat_min:              float64, min latitude of figure
        lat_max:              float64, max latitude of figure
        vmin:                 float64, min value of concentration in the figure, specify to keep colorscale constant
        vmax:                 float64, max value of concentration in the figure, specify to keep colorscale constant
        file_out_path:        string, main output path of figure produced, must end with /
        file_out_sub_folder:  string, subforlder of file_out_path, must end with /
        figure_file_name:     string, name of figure
        shp_file_path:        string, full path and name of shp file
        title_caption:        string, first part of figure title before date and unit_measure
        full_title:           string, full title of figure. It overwrites title_caption if specified
        unit_measure:         string, (ug/m3) or (ug/kg d.w), between parenthesis
        levels_colormap:      list of float64, levels used for colorbar (e.g., [0., 1., 15.])
        selected_colormap:    e.g. plt.cm.Blues
        colorbar_title:       string, title of colorbar
        simmetrical_cmap:     boolean,select if cmap is simmetrical to 0 (True) or not (False)
        scientific_colorbar:  boolean,select if colorbar is written in scientific notation (True) or not (False)
        selected_depth:       float32, depth selected when creating map if "depth" in Conc_Dataset.dims
                                   If no depth was selected when creating conc map, use 0
        fig_format:           string, format of produced images (e.g.,".jpg", ".png")
        fig_dpi:              int, dots per inch resolution of figure
        make_animation:       boolean,select if animation (.mp4 or .gif) is created (True) or not (False)
        concat_animation:     boolean,select if animations (.mp4 or .gif) are loaded and concatenated (True) or not (False)
        animation_format:     string, format of produced animation (".mp4" or .gif")
        fps:                  integer, frames per second of animation
        load_img_from_folder: boolean,select if images are created or loaded
        fig_numbers:          list of lists (of int) that specifyies numbers to create fig_name of figures to load and in
                              in prefix of animations to concatenate (e.g., [[0, 8], [9, 15]]))
        add_shp_to_figure:    boolean,select if shp is added to the figure (True) or not (False)
        variable_name:        string, name of Conc_Datasetdata variable to plot if not concentration_avg_water/sediments
        labels_font_sizes:    list of int, [title_font_size, x_label_font_size, y_label_font_size, x_ticks_font_size
                                           y_ticks_font_size, cbar_label_font_size, cbar_ticks_font_size]
        width_fig:              int, lenght of the figure (inches)
        high_fig:             int, height of the figure (inches)
        shp_color:            string, color of shapefile when plotted
        trim_images:          boolean,select if white borders of images is removed
        padding_r, padding_c: int, number of pixels not trimmed (_r for height, _c for width)
        shading:              string, interpolation format using plt. pcolormesh (None, 'flat', 'nearest', 'gouraud', 'auto')
        save_figures:         boolean,select if figures are saved
        date_str_lenght:      int, number date string charcters kept in title [10 for YYYY-MM-DD, 19 for hour shown]
        low_quality:          boolean, select if figures are sized down for animation
        '''

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import os as os
        from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
        import geopandas as gpd
        from datetime import datetime as dt
        import gc

        if all([not e for e in [save_figures, make_animation, concat_animation]]) is True:
            raise ValueError("No output (save_figures/make_animation/concat_animation) was selected ")

        if load_img_from_folder == True and fig_numbers is None:
            raise ValueError("fig_numbers must be specified when loading images or concatenating animations")

        if fig_numbers is not None:
            fig_num_ls = self._flatten_list(fig_numbers)
            if  (all(fig_num_ls[i] <= fig_num_ls[i + 1] for i in range(len(fig_num_ls) - 1))) is False:
                raise ValueError("fig_numbers are not ordered increasingly")

        if concat_animation is True and ((self._check_nested_list(fig_numbers) is False) or (fig_numbers is None)):
            raise ValueError("No fig_numbers lists were specified for concat_animation")

        if load_img_from_folder == False and Conc_Dataset is not None:
            start=dt.now()
            aspect = 15
            pad_fraction1 = -0.08
            pad_fraction2 = 0.0384
            file_output_path = file_out_path + file_out_sub_folder
            print(f"Figures saved to: {file_output_path}")

            def fmt(x, pos):
                '''
                Define scientific notation for colorbar if scientific_colorbar is True
                '''
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)

            title_font_size = labels_font_sizes[0]
            x_label_font_size = labels_font_sizes[1]
            y_label_font_size = labels_font_sizes[2]
            x_ticks_font_size = labels_font_sizes[3]
            # y_ticks_font_size = labels_font_sizes[4]
            cbar_label_font_size = labels_font_sizes[5]
            # cbar_ticks_font_size = labels_font_sizes[6]

            if simmetrical_cmap == True:
                if selected_colormap == None:
                    selected_colormap = plt.colormaps["viridis"]
                selected_colormap = self._simmetrical_colormap(cmap = selected_colormap)

            if not os.path.exists(file_output_path):
                os.makedirs(file_output_path)
                print("file_output_path did not exist and was created")
            else:
                pass

            if add_shp_to_figure:
                print("shp was added over the figures")
                shp = gpd.read_file(shp_file_path)
                cax_pad= pad_fraction1 * width_fig
            else:
                print("shp was not added over the figures")
                # Create an empty GeoDataFrame (shp) to plot
                from shapely.geometry import Point
                crs = 'epsg:4326'
                # Create an empty GeoDataFrame
                columns = ['geometry']
                shp =  gpd.GeoDataFrame(columns=columns, crs=crs)
                point = Point(0, 0)
                shp.loc[0, 'geometry'] = point
                cax_pad= pad_fraction2 * width_fig

            if "longitude" not in Conc_Dataset.dims:
                if 'lat' in Conc_Dataset.dims:
                    Conc_Dataset = Conc_Dataset.rename({'lat': 'latitude','lon': 'longitude'})
                elif 'x' in Conc_Dataset.dims:
                    Conc_Dataset = Conc_Dataset.rename({'y': 'latitude','x': 'longitude'})
                else:
                    raise ValueError("Unknown spatial coordinates")

            if "depth" not in Conc_Dataset.dims:
                if 'z' in Conc_Dataset.dims:
                    Conc_Dataset = Conc_Dataset.rename({'z': 'depth'})
            if 'avg_time' in Conc_Dataset.dims:
                Conc_Dataset = Conc_Dataset.rename({'avg_time': 'time'})

            if "concentration_avg_water" in Conc_Dataset.keys():
                Conc_DataArray = Conc_Dataset.concentration_avg_water
            elif "concentration_avg_sediments" in Conc_Dataset.keys():
                Conc_DataArray = Conc_Dataset.concentration_avg_sediments
            elif variable_name is not None:
                Conc_DataArray = Conc_Dataset[variable_name]
            else:
                raise ValueError("specified variable_name is not present in Conc_Dataset")


            Conc_DataArray_tstart =np.array(Conc_DataArray.time[0])
            Conc_DataArray_tend =np.array(Conc_DataArray.time[-1])

            if colorbar_title is None:
                if "concentration_avg_water" in Conc_Dataset.keys():
                    colorbar_title = "concentration_avg_water"
                elif "concentration_avg_sediments" in Conc_Dataset.keys():
                    colorbar_title = "concentration_avg_sediments"
                elif variable_name is not None:
                    colorbar_title = variable_name
                else:
                    raise ValueError("colorbar_title or variable_name are not specified")
            del Conc_Dataset

            if colormap_norm is not None:
                vmax = colormap_norm.boundaries[-1]

            if 'time' not in Conc_DataArray.dims:
                if "year" in Conc_DataArray.dims:
                    # Change "year" dimension to "time", at the January, 1st
                    Conc_DataArray['year'] = pd.to_datetime(np.char.add(np.array(Conc_DataArray['year']).astype(str), '-01-01'))
                    Conc_DataArray = Conc_DataArray.rename({'year': 'time'})
                    Conc_DataArray = Conc_DataArray.assign_coords(time=Conc_DataArray['time'])
                elif "season" in Conc_DataArray.dims and time_start is not None:
                    # Change "season" dimension to "time", at the first day of each season
                    time_start_year = time_start.astype('datetime64[Y]').astype(int) + 1970
                    time_season_dict = {"DJF":"-12-21", "JJA":"-06-21", "MAM":"-03-21", "SON":"-09-23"}
                    time_season = [time_season_dict.get(season) for season in list(Conc_DataArray.season.values)]
                    Conc_DataArray["season"] = pd.to_datetime(np.char.add(str(time_start_year), time_season))
                    Conc_DataArray = Conc_DataArray.rename({'season': 'time'})
                    Conc_DataArray = Conc_DataArray.assign_coords(time=Conc_DataArray['time'])
                else:
                    # Check if other dimensions than the ones to be allowed are present and add time_start as time
                    acceptable_dimensions = set(['latitude', 'longitude', 'time', 'depth'])
                    Dataset_dimensions = set(Conc_DataArray.dims)
                    extra_dimensions = (Dataset_dimensions - acceptable_dimensions)
                    if len(extra_dimensions) > 0:
                        raise ValueError(f"Dimensions other than {acceptable_dimensions} are present: f{extra_dimensions}")
                    else:
                        if time_start is not None:
                            Conc_DataArray['time'] = time_start
                        else:
                            raise ValueError("Conc_DataArray.time is missing, time_start must be specified")
            else:
                # Check if other dimensions than the ones to be loaded are present
                acceptable_dimensions = set(['latitude', 'longitude', 'time', 'depth'])
                Dataset_dimensions = set(Conc_DataArray.dims)
                extra_dimensions = (Dataset_dimensions - acceptable_dimensions)
                if len(extra_dimensions) > 0:
                    raise ValueError(f"Dimensions other than {acceptable_dimensions} are present: f{extra_dimensions}")

            # Remove timesteps before time_start and after time_end
            if time_start is not None:
                Conc_DataArray = Conc_DataArray.where((Conc_DataArray.time >= time_start), drop=True)
            if time_end is not None:
                Conc_DataArray = Conc_DataArray.where((Conc_DataArray.time <= time_end), drop=True)

            if Conc_DataArray.time.size == 0:
                raise ValueError(f"Conc_DataArray.time {Conc_DataArray_tstart} - {Conc_DataArray_tend} is out of time_start/end interval")

            attribute_list = list(Conc_DataArray.attrs)
            for attr in attribute_list:
                del Conc_DataArray.attrs[attr]

            fig_num = []
            figure_ls = []
            figure_name_ls = []
            Conc_DataArray_time_size = (Conc_DataArray.time.to_numpy()).size
            figures_number = (Conc_DataArray_time_size)
            if fig_numbers is not None:
                if fig_numbers[-1][1] > figures_number:
                    raise ValueError(f"fig_numbers selects more figures ({fig_numbers[-1][1] + 1}) that were created ({figures_number})")

            self._check_imgs_memory_use(width_fig = width_fig,
                                        high_fig = high_fig,
                                        fig_dpi = fig_dpi,
                                        figures_number = figures_number,
                                        trim_images = trim_images,
                                        make_animation = make_animation)

            for num in [0, figures_number]:
                fig_num.append(str(f"{num:03d}"))
            anim_prefix = fig_num[0] + "_" + fig_num[1] + "_"

            for timestep in range(0, figures_number):
                figure_name_ls.append(str(f"{timestep:03d}")+"_"+figure_file_name+fig_format)

            list_index_print = self._print_progress_list(figures_number)

            if "depth" in Conc_DataArray.dims and selected_depth is None:
                raise ValueError("selected_depth must be specified")
            elif "depth" in Conc_DataArray.dims:
                all_depth_values = np.sort((np.unique(np.array(Conc_DataArray.depth)))) # Change depth to positive values
                idx = np.where(np.isclose(all_depth_values, selected_depth))[0]
                if idx.size == 0:
                    raise ValueError(f"selected_depth {selected_depth} not found in available depths {all_depth_values}")
                selected_depth_index = int(idx[0])

            start = dt.now()

            for timestep in range(0, figures_number):
                if timestep in list_index_print:
                     print(f"creating image n° {str(timestep+1)} out of {str(figures_number)}")

                if Conc_DataArray_time_size > 1 and "depth" in Conc_DataArray.dims:
                    Conc_DataArray_selected = Conc_DataArray.isel(time=timestep, depth=selected_depth_index, drop=True)
                elif Conc_DataArray_time_size > 1 and "depth" not in Conc_DataArray.dims:
                    Conc_DataArray_selected = Conc_DataArray.isel(time=timestep, drop=True)
                elif Conc_DataArray_time_size <= 1 and "depth" in Conc_DataArray.dims:
                    Conc_DataArray_selected = Conc_DataArray.isel(depth=selected_depth_index, drop=True)
                else:
                    Conc_DataArray_selected = Conc_DataArray  # still may have time=1
                    Conc_DataArray_selected = Conc_DataArray_selected.squeeze(drop=True)


                fig, ax = plt.subplots(figsize = (width_fig, high_fig), dpi=fig_dpi)
                shp.plot(ax = ax, zorder = 10, edgecolor = 'black', facecolor = shp_color)

                if shading in [None, "flat", "auto"]:
                    ax2 = Conc_DataArray_selected.plot.pcolormesh(
                                                x = 'longitude',
                                                y = 'latitude',
                                                cmap = selected_colormap,
                                                norm = colormap_norm,
                                                vmin = vmin, vmax = vmax,
                                                levels = levels_colormap,
                                                shading = shading,
                                                add_colorbar = False, # colorbar is added ex-post
                                                zorder = 0)
                else:
                    if shading not in ["gouraud", "nearest", "flat", "auto"]:
                        raise ValueError("Incorrect shading specified")
                    if shading in ["gouraud", "nearest"]:
                        X = Conc_DataArray_selected.coords['longitude'].to_numpy()
                        Y = Conc_DataArray_selected.coords['latitude'].to_numpy()
                        Conc_DataArray_selected = Conc_DataArray_selected.to_numpy()

                        ax2 = plt.pcolormesh(X,
                                             Y,
                                             Conc_DataArray_selected,
                                             cmap = selected_colormap,
                                             vmin = vmin, vmax = vmax,
                                             shading = shading,
                                             zorder = 0)
                        del X, Y
                ax.set_xlim(long_min, long_max)
                ax.set_ylim(lat_min, lat_max)
                ax.set_xlabel("Longitude", fontsize = x_label_font_size, labelpad = high_fig*2) # Change here size of ax labels
                ax.set_ylabel("Latitude", fontsize = y_label_font_size, labelpad = high_fig*2) # Change here size of ax labels
                ax.tick_params(labelsize=x_ticks_font_size) # Change here size of ax ticks
                if full_title is not None:
                    fig_title = full_title
                else:
                    if Conc_DataArray_time_size > 1:
                        ts = pd.to_datetime(Conc_DataArray.time[timestep].item())
                        um = f" {unit_measure}" if unit_measure else ""
                        fig_title = (
                            f"{title_caption} {ts:%Y-%m-%d %H:%M:%S}{um}"
                            if date_str_lenght >= 19
                            else f"{title_caption} {ts:%Y-%m-%d}{um}"
                        )
                    else:
                        fig_title = (title_caption + " " + unit_measure)

                ax.set_title(fig_title, pad=high_fig*1.5, fontsize = title_font_size, weight = "bold", wrap= True)
                # from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
                divider = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1./aspect)
                cax = divider.append_axes("right", size=width, pad = cax_pad)
                cax.yaxis.offsetText.set_fontsize(24)
                cax.tick_params(labelsize=cbar_label_font_size)
                y_formatter = ticker.ScalarFormatter(useMathText=True)
                cax.yaxis.set_major_formatter(y_formatter)

                if scientific_colorbar is True:
                    cbar = plt.colorbar(ax2, cax=cax, format=ticker.FuncFormatter(fmt), label=colorbar_title)
                else:
                    cbar = plt.colorbar(ax2, cax=cax, label=colorbar_title)

                cbar.set_label(colorbar_title, fontsize=cbar_label_font_size, labelpad = 20, fontweight ="bold")

                fig_path = file_out_path + file_out_sub_folder + figure_name_ls[timestep]

                # Draw the canvas and grab an RGB array
                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
                # Convert RGBA to RGB by removing the alpha channel
                rgb = rgba[..., :3].astype(np.float32) / 255.0

                # Optionally trim white borders
                if trim_images:
                    rgb = self._remove_white_borders(rgb, padding_r=padding_r, padding_c=padding_c)

                # Save to disk if requested
                if save_figures:
                    plt.imsave(fig_path, rgb, cmap=selected_colormap)

                # Collect frames for animation if requested
                if make_animation:
                    figure_ls.append(rgb)

                # Close and clean up
                plt.close(fig)
                del fig, ax, cax, cbar, ax2, Conc_DataArray_selected, rgba, rgb
                gc.collect()

            print(f"Time to create figures (hr:min:sec): {dt.now()-start}")

            if make_animation is True:
                if fig_numbers is None:

                    self._create_animation(load_img_from_folder = load_img_from_folder,
                                       trim_images = trim_images,
                                       figure_ls = figure_ls,
                                       file_out_path = file_out_path,
                                       file_out_sub_folder = file_out_sub_folder,
                                       anim_prefix = anim_prefix,
                                       figure_file_name = figure_file_name,
                                       animation_format = animation_format,
                                       fps = fps,
                                       width_fig = width_fig,
                                       high_fig = high_fig,
                                       low_quality = low_quality
                                       )
                else:
                    for num_list in fig_numbers:
                        # Create prefix for animation name
                        fig_num = []
                        for num in num_list:
                            fig_num.append(str(f"{num:03d}"))
                        anim_prefix = fig_num[0] + "_" + fig_num[1] + "_"

                        print(f"Creating animation {anim_prefix}")
                        figure_ls_split = figure_ls[num_list[0]:num_list[1]+1]
                        self._create_animation(load_img_from_folder = load_img_from_folder,
                                           trim_images = trim_images,
                                           figure_ls = figure_ls_split,
                                           file_out_path = file_out_path,
                                           file_out_sub_folder = file_out_sub_folder,
                                           anim_prefix = anim_prefix,
                                           figure_file_name = figure_file_name,
                                           animation_format = animation_format,
                                           fps = fps,
                                           width_fig = width_fig,
                                           high_fig = high_fig,
                                           low_quality = low_quality
                                           )
                        del figure_ls_split

        elif load_img_from_folder == True and fig_numbers is not None:

            for num_list in fig_numbers:
                # Create prefix for animation name
                fig_num = []
                for num in num_list:
                    fig_num.append(str(f"{num:03d}"))
                anim_prefix = fig_num[0] + "_" + fig_num[1] + "_"

                # Prepare progress messages
                figures_number = (num_list[1] - num_list[0]) + 1

                self._check_imgs_memory_use(width_fig = width_fig,
                                            high_fig = high_fig,
                                            fig_dpi = fig_dpi,
                                            figures_number = figures_number,
                                            trim_images = trim_images,
                                            make_animation = make_animation)

                list_index_print = self._print_progress_list(figures_number)

                # Create figures names
                figure_name_ls = []
                for timestep in range(num_list[0], num_list[1] +1):
                    figure_name_ls.append(str(f"{timestep:03d}")+"_"+figure_file_name+fig_format)

                # Load figures
                figure_ls = []
                print(f"Loading images {anim_prefix}")
                for fig_name in figure_name_ls:
                    fig_path = (file_out_path + file_out_sub_folder + fig_name)
                    image = plt.imread(fig_path)
                    # fig, ax = plt.subplots(figsize = (width_fig,high_fig))
                    # ax.set_axis_off()
                    figure_ls.append(image)
                    plt.close('all')
                if trim_images == True:
                    for img_index in range(0, len(figure_ls)):
                        if img_index in list_index_print:
                            print(f"trim image n° {str(img_index+1)} out of {str(figures_number)}")
                        rgb = self._remove_white_borders(figure_ls[img_index], padding_r = padding_r, padding_c = padding_c)
                        figure_ls[img_index] = rgb

                    if save_figures == True:
                        for img_index in range(0, len(figure_ls)):
                            if img_index in list_index_print:
                                print(f"saving image n° {str(img_index+1)} out of {str(figures_number)}")
                            fig_path = (file_out_path + file_out_sub_folder + figure_name_ls[img_index][:-4]+"_trim"+fig_format)
                            fig, ax = plt.subplots(figsize = (width_fig,high_fig))
                            ax.set_axis_off()
                            plt.imsave(fig_path, figure_ls[img_index], cmap=selected_colormap)
                            plt.close('all')
                    else:
                        pass

                if make_animation is True:
                    self._create_animation(load_img_from_folder = load_img_from_folder,
                                           trim_images = trim_images,
                                           fps=fps,
                                           figure_ls = figure_ls,
                                           file_out_path = file_out_path,
                                           file_out_sub_folder = file_out_sub_folder,
                                           anim_prefix = anim_prefix,
                                           figure_file_name = figure_file_name,
                                           animation_format = animation_format,
                                           width_fig = width_fig,
                                           high_fig = high_fig,
                                           low_quality = low_quality
                                           )
                else:
                    pass

        elif concat_animation is True:
            pass
        else:
            raise ValueError("No image was set to be created/loaded, and no animation was set to be concatenated")

        if concat_animation is True and fig_numbers is not None:
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            from natsort import natsorted
            # Prepare animation names to load
            anim_name_ls = []
            for num_list in fig_numbers:

                fig_num = []
                for num in num_list:
                    fig_num.append(str(f"{num:03d}"))
                anim_prefix = fig_num[0] + "_" + fig_num[1] + "_"
                anim_name_ls.append(anim_prefix+figure_file_name+animation_format)

            # Order names and load animations
            print("Loading animations")
            L_anim = []
            files = natsorted(anim_name_ls)
            for file in files:
                    video = VideoFileClip(file_out_path + file_out_sub_folder + file)
                    L_anim.append(video)

            final_clip = concatenate_videoclips(L_anim)
            if animation_format == ".gif":
                video_codec ='gif'
            elif  animation_format == ".mp4":
                video_codec ='libx264'
            else:
                raise ValueError("Unsupported animation_format")

            merged_prefix = "Merged_" + (str(f"{fig_numbers[0][0]:03d}")) + "_" + (str(f"{fig_numbers[-1][1]:03d}")) + "_"
            print("Saving concatenated animation")
            output_video = file_out_path + file_out_sub_folder + merged_prefix + figure_file_name + animation_format

            final_clip.to_videofile(output_video, fps=12, remove_temp=False, codec = video_codec)

    @staticmethod
    def _plot_emission_data_frequency(emissions, title, n_bins=100, zoom_max=100, zoom_min=0):
        '''
        Plot distribution of emissions dataset values, mass, and cumulative mass

        emissions:   array-like, selected data points to plot
        title:       string, title of main plot
        n_bins:      int, number of bins to group datapoints
        zoom_max:    int, % of histogram bins where the zoomed area stops
        zoom_min:    int, % of histogram bins where the zoomed area starts
        '''
        import matplotlib.pyplot as plt
        import numpy as np

        emissions = np.asarray(emissions).ravel()
        emissions = emissions[np.isfinite(emissions)]

        if emissions.size == 0:
            raise ValueError("No valid emission values available for plotting.")

        counts, bin_edges = np.histogram(emissions, bins=n_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        bin_mass = counts * bin_centers
        cumulative_mass = np.cumsum(bin_mass)

        def _normalize(arr):
            arr_max = np.max(arr) if arr.size > 0 else 0
            if arr_max == 0:
                return np.zeros_like(arr, dtype=float)
            return 100.0 * arr / arr_max

        def _normalize_cumulative(arr):
            total = arr[-1] if arr.size > 0 else 0
            if total == 0:
                return np.zeros_like(arr, dtype=float)
            return 100.0 * arr / total

        counts_norm = _normalize(counts)
        mass_norm = _normalize(bin_mass)
        cumulative_norm = _normalize_cumulative(cumulative_mass)

        zoom_min = float(np.clip(zoom_min, 0, 100))
        zoom_max = float(np.clip(zoom_max, 0, 100))

        if zoom_max <= zoom_min:
            zoom_min, zoom_max = 0.0, 100.0

        n_hist_bins = len(bin_centers)
        i_min = int(np.floor((zoom_min / 100.0) * n_hist_bins))
        i_max = int(np.ceil((zoom_max / 100.0) * n_hist_bins))
        i_max = max(i_max, i_min + 1)

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), constrained_layout=True)

        ax1.plot(bin_centers, mass_norm)
        ax1.plot(bin_centers, cumulative_norm)
        ax1.plot(bin_centers, counts_norm)
        ax1.set_title(title)
        ax1.set_ylabel("Frequency (%)")
        ax1.legend(['Mass', 'Cumulative mass', 'Data points'])

        ax2.plot(bin_centers[i_min:i_max], mass_norm[i_min:i_max])
        ax2.plot(bin_centers[i_min:i_max], cumulative_norm[i_min:i_max])
        ax2.plot(bin_centers[i_min:i_max], counts_norm[i_min:i_max])
        ax2.set_title(f"Zoom from {zoom_min:.1f}% to {zoom_max:.1f}% of histogram bins")
        ax2.set_ylabel("Frequency (%)")
        ax2.set_xlabel("Value")
        ax2.legend(['Mass', 'Cumulative mass', 'Data points'])

        plt.show()


    def summary_created_elements(
        self,
        file_folder,
        file_name,
        variable_name,
        mass_factor,
        upper_limit,
        lower_limit,
        name_dataset,
        long_min=None,
        long_max=None,
        lat_min=None,
        lat_max=None,
        time_start=None,
        time_end=None,
        range_max=None,
        range_min=None,
        n_bins=100,
        zoom_max=100,
        zoom_min=0,
        print_results=False,
        mass_unit=None,
        mass_element_ug=100e3,
        number_of_elements=None,
        estimate_mode="mass",
    ):
        '''
        Calculate the maximum number of elements in a simulation created by seed_from_NETCDF from xarray DataArray.
        Produce histograms with frequency of datapoints values within the specified limits.
        Estimate the number of seeded elements ignoring bathymetry consistency checks.

        file_folder:         string, path to file, must end with /
        file_name:           string, name of file, must end with .nc
        variable_name:       string, name of xarray DataArray variable
        mass_factor:         float32, multiplicative conversion factor between data values and mass units used for seeded elements
        upper_limit:         float32, upper selection limit; datapoints >= upper_limit are ignored by seed_from_NETCDF
        lower_limit:         float32, lower selection limit; datapoints <= lower_limit are ignored by seed_from_NETCDF
        name_dataset:        string, name of data to be reported in the title of figures
        time_start:          datetime64[ns] or None, start time of dataset considered; if None, start from first available time
        time_end:            datetime64[ns] or None, end time of dataset considered; if None, end at last available time
        long_min:            float32 or None, min longitude of dataset considered; if None, use whole longitude range
        long_max:            float32 or None, max longitude of dataset considered; if None, use whole longitude range
        lat_min:             float32 or None, min latitude of dataset considered; if None, use whole latitude range
        lat_max:             float32 or None, max latitude of dataset considered; if None, use whole latitude range
        range_max:           float32 or None, max value shown in the figure on data frequency for the selected dataset
        range_min:           float32 or None, min value shown in the figure on data frequency for the selected dataset
        n_bins:              int, number of bins used for histograms
        zoom_max:            int, % of histogram bins where the zoomed area stops
        zoom_min:            int, % of histogram bins where the zoomed area starts
        print_results:       boolean, select if results are printed or returned as dictionary
        mass_unit:           string or None, unit label used when printing masses; if None, results are reported as "mass units"
        mass_element_ug:     float32 or None, mass of one seeded element when estimate_mode includes "mass"
        number_of_elements:  int or None, fixed number of elements per selected datapoint when estimate_mode includes "fixed"
        estimate_mode:       string, choose which estimate to calculate:
                             "mass", "fixed", or "both"
        '''
        import xarray as xr
        import matplotlib.pyplot as plt
        import numpy as np

        if estimate_mode not in ["mass", "fixed", "both"]:
            raise ValueError("estimate_mode must be 'mass', 'fixed', or 'both'.")

        file_path = file_folder + file_name
        mass_unit_label = mass_unit if mass_unit is not None else "mass units"

        with xr.open_dataset(file_path) as ds:
            rename_dict = {}
            if "lat" in ds.dims:
                rename_dict["lat"] = "latitude"
            if "lon" in ds.dims:
                rename_dict["lon"] = "longitude"
            if "x" in ds.dims:
                rename_dict["x"] = "longitude"
            if "y" in ds.dims:
                rename_dict["y"] = "latitude"
            if rename_dict:
                ds = ds.rename(rename_dict)

            da = ds[variable_name]

            mask = xr.ones_like(da, dtype=bool)

            if "longitude" in da.coords:
                if long_min is not None:
                    mask = mask & (da.longitude >= long_min)
                if long_max is not None:
                    mask = mask & (da.longitude <= long_max)

            if "latitude" in da.coords:
                if lat_min is not None:
                    mask = mask & (da.latitude >= lat_min)
                if lat_max is not None:
                    mask = mask & (da.latitude <= lat_max)

            if "time" in da.dims or "time" in da.coords:
                if time_start is not None:
                    mask = mask & (da.time >= time_start)
                if time_end is not None:
                    mask = mask & (da.time <= time_end)

            da = da.where(mask, drop=True)

            emissions = da.to_masked_array()
            emissions = np.asarray(emissions).ravel()
            emissions = emissions[np.isfinite(emissions) & (emissions > 0)]

        if emissions.size == 0:
            raise ValueError("No positive emission values found in the selected dataset.")

        if mass_factor is None or mass_factor <= 0:
            raise ValueError("mass_factor must be specified and > 0.")

        selected = (emissions > lower_limit) & (emissions < upper_limit)
        selected_emissions = emissions[selected]

        ds_max = float(np.max(emissions))
        ds_min = float(np.min(emissions))

        emissions_sum = float(np.sum(emissions))
        total_mass = emissions_sum * mass_factor

        selected_mass_array = selected_emissions * mass_factor
        selected_mass = float(np.sum(selected_mass_array))

        num_tot = int(emissions.size)
        num_selected = int(np.count_nonzero(selected))

        perc_num_selected = 100.0 * num_selected / num_tot if num_tot else 0.0
        perc_mass_selected = 100.0 * selected_mass / total_mass if total_mass else 0.0

        est_elements_mass_mode = None
        est_elements_fixed_mode = None

        if estimate_mode in ["mass", "both"]:
            if mass_element_ug is None or mass_element_ug <= 0:
                raise ValueError("mass_element_ug must be > 0 when estimate_mode is 'mass' or 'both'.")
            if num_selected > 0:
                est_elements_mass_mode_per_point = np.ceil(selected_mass_array / mass_element_ug).astype(int)
                est_elements_mass_mode = int(np.sum(est_elements_mass_mode_per_point))
            else:
                est_elements_mass_mode = 0

        if estimate_mode in ["fixed", "both"]:
            if number_of_elements is None or number_of_elements <= 0:
                raise ValueError("number_of_elements must be > 0 when estimate_mode is 'fixed' or 'both'.")
            est_elements_fixed_mode = int(num_selected * int(number_of_elements))

        results_dict = {}
        results_dict["name_dataset"] = name_dataset
        results_dict["upper_limit"] = upper_limit
        results_dict["lower_limit"] = lower_limit
        results_dict["DS_max"] = ds_max
        results_dict["DS_min"] = ds_min
        results_dict["Num_tot"] = num_tot
        results_dict["Num_selected"] = num_selected
        results_dict["Mass_tot"] = total_mass
        results_dict["Mass_selected"] = selected_mass
        results_dict["Mass_unit"] = mass_unit_label
        results_dict["Perc_num_selected"] = perc_num_selected
        results_dict["Perc_mass_selected"] = perc_mass_selected
        results_dict["mass_factor"] = mass_factor
        results_dict["estimate_mode"] = estimate_mode
        results_dict["mass_element_ug"] = mass_element_ug if estimate_mode in ["mass", "both"] else None
        results_dict["number_of_elements"] = number_of_elements if estimate_mode in ["fixed", "both"] else None
        results_dict["Estimated_elements_mass_mode"] = est_elements_mass_mode
        results_dict["Estimated_elements_fixed_mode"] = est_elements_fixed_mode

        if print_results is False:
            return results_dict

        print("##START " + name_dataset + " ##")

        print(f"DS_max: {ds_max}")
        print(f"DS_min: {ds_min}\n")

        print(f"number of data-points without limits: {num_tot}")
        print(f"upper limit: {upper_limit}")
        print(f"lower limit: {lower_limit}\n")

        print(f"number of data-points selected within the limits: {num_selected}\n")
        print(f"total mass of chemical: {total_mass} {mass_unit_label}")
        print(f"selected mass of chemical: {selected_mass} {mass_unit_label}")
        print(f"% of total mass selected: {perc_mass_selected} %")
        print(f"% of total data-points selected: {perc_num_selected} %\n")

        if estimate_mode in ["mass", "both"]:
            print(f"estimated number of elements with gen_mode='mass': {est_elements_mass_mode}")
            print(f"mass_factor used for estimate: {mass_factor}")
            print(f"mass_element_ug used for estimate: {mass_element_ug}")

        if estimate_mode in ["fixed", "both"]:
            print(f"estimated number of elements with gen_mode='fixed': {est_elements_fixed_mode}")
            print(f"number_of_elements used for estimate: {number_of_elements}")

        print("bathymetry checks ignored in these estimates\n")

        num_upper_lim = int(np.count_nonzero(emissions >= upper_limit))
        mass_over_limit = float(np.sum(emissions[emissions >= upper_limit]) * mass_factor)
        perc_upper_num = 100.0 * num_upper_lim / num_tot if num_tot else 0.0
        perc_upper_mass = 100.0 * mass_over_limit / total_mass if total_mass else 0.0

        print(f"n° of data-points over upper limit: {num_upper_lim}")
        print(f"% of data-points over upper limit: {perc_upper_num} %")
        print(f"mass of chemical over upper limit: {mass_over_limit} {mass_unit_label}")
        print(f"% of total mass of the elements over upper limit: {perc_upper_mass} %\n")

        num_lower_lim = int(np.count_nonzero(emissions <= lower_limit))
        mass_below_limit = float(np.sum(emissions[emissions <= lower_limit]) * mass_factor)
        perc_lower_num = 100.0 * num_lower_lim / num_tot if num_tot else 0.0
        perc_lower_mass = 100.0 * mass_below_limit / total_mass if total_mass else 0.0

        print(f"n° of data-points under lower limit: {num_lower_lim}")
        print(f"% of data-points under lower limit: {perc_lower_num} %")
        print(f"mass of chemical under lower limit: {mass_below_limit} {mass_unit_label}")
        print(f"% of total mass of the elements under lower limit: {perc_lower_mass} %\n")

        mass_below_upper = float(np.sum(emissions[emissions < upper_limit]) * mass_factor)
        perc_below_lower_with_upper = (
            100.0 * mass_below_limit / mass_below_upper if mass_below_upper else 0.0
        )

        print(f"% of total mass of elements under lower limit considering also upper limit {perc_below_lower_with_upper}\n")

        self._plot_emission_data_frequency(
            emissions=emissions,
            title="Complete dataset",
            n_bins=n_bins,
            zoom_max=zoom_max,
            zoom_min=zoom_min
        )

        if num_selected > 0:
            self._plot_emission_data_frequency(
                emissions=selected_emissions,
                title="Selected dataset between lower and upper limit",
                n_bins=n_bins,
                zoom_max=zoom_max,
                zoom_min=zoom_min
            )

        if range_max is not None and range_min is not None:
            ranged = emissions[(emissions >= range_min) & (emissions <= range_max)]

            if ranged.size > 0:
                self._plot_emission_data_frequency(
                    emissions=ranged,
                    title="Selected dataset between range_min and range_max",
                    n_bins=n_bins,
                    zoom_max=100,
                    zoom_min=0
                )

        plt.hist(x=emissions, bins=n_bins, range=(0, lower_limit))
        plt.title("Datapoints between 0 and lower limit for " + name_dataset)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

        print("##END##")

        return results_dict

    def init_chemical_compound(self, chemical_compound = None):
        ''' Chemical parameters for a selection of PAHs:
            Naphthalene, Phenanthrene, Fluorene,
            Benzo-a-anthracene, Benzo-a-pyrene, Dibenzo-ah-anthracene

            Data collected from literature by
            Isabel Hanstein (University of Heidelberg / Norwegian Meteorological Insitute)
            Mattia Boscherini, Loris Calgaro (University Ca' Foscari, Venezia)
            Manuel Aghito (Norwegian Meteorological Institute / University of Bergen)
        '''

        if chemical_compound is not None:
            self.set_config('chemical:compound',chemical_compound)

        if self.get_config('chemical:compound') is None:
            raise ValueError("Chemical compound not defined")

        if  self.get_config('chemical:compound') in ["Naphthalene",'C1-Naphthalene','Acenaphthene','Acenaphthylene','Fluorene']:

            #partitioning
            self.set_config('chemical:transfer_setup','organics')
            self.set_config('chemical:transformations:dissociation','nondiss')
            self.set_config('chemical:transformations:LogKOW',3.361)
            self.set_config('chemical:transformations:TrefKOW',25)
            self.set_config('chemical:transformations:KOC_sed',959)
            self.set_config('chemical:transformations:KOC_DOM',1798)
            self.set_config('chemical:transformations:DeltaH_KOC_Sed',-21036)
            self.set_config('chemical:transformations:DeltaH_KOC_DOM',-25900)                 ### Phenanthrene value
            self.set_config('chemical:transformations:Setchenow', 0.2503)

            #degradation
            self.set_config('chemical:transformations:t12_W_tot', 224.08)
            self.set_config('chemical:transformations:Tref_kWt', 25)
            self.set_config('chemical:transformations:DeltaH_kWt', 50000)                     ### generic
            self.set_config('chemical:transformations:t12_S_tot', 5012.4)
            self.set_config('chemical:transformations:Tref_kSt', 25)
            self.set_config('chemical:transformations:DeltaH_kSt', 50000)                     ### generic

            #volatilization
            self.set_config('chemical:transformations:MolWt', 128.1705)
            self.set_config('chemical:transformations:Henry', 4.551e-4)

            self.set_config('chemical:transformations:Vpress', 11.2)
            self.set_config('chemical:transformations:Tref_Vpress', 25)
            self.set_config('chemical:transformations:DeltaH_Vpress', 55925)

            self.set_config('chemical:transformations:Solub', 31.4)
            self.set_config('chemical:transformations:Tref_Solub', 25)
            self.set_config('chemical:transformations:DeltaH_Solub', 25300)

        elif self.get_config('chemical:compound') in ["Phenanthrene",'Dibenzothiophene','C2-Naphthalene','Anthracene','C3-Naphthalene','C1-Dibenzothiophene']:

            #partitioning
            self.set_config('chemical:transfer_setup','organics')
            self.set_config('chemical:transformations:dissociation','nondiss')
            self.set_config('chemical:transformations:LogKOW',4.505)
            self.set_config('chemical:transformations:TrefKOW',25)
            self.set_config('chemical:transformations:KOC_sed',25936)
            self.set_config('chemical:transformations:KOC_DOM',39355)
            self.set_config('chemical:transformations:DeltaH_KOC_Sed',-24900)
            self.set_config('chemical:transformations:DeltaH_KOC_DOM',-25900)
            self.set_config('chemical:transformations:Setchenow', 0.3026)

            #degradation
            self.set_config('chemical:transformations:t12_W_tot', 1125.79)
            self.set_config('chemical:transformations:Tref_kWt', 25)
            self.set_config('chemical:transformations:DeltaH_kWt', 50000)                     ### generic
            self.set_config('chemical:transformations:t12_S_tot', 29124.96)
            self.set_config('chemical:transformations:Tref_kSt', 25)
            self.set_config('chemical:transformations:DeltaH_kSt', 50000)                     ### generic

            #volatilization
            self.set_config('chemical:transformations:MolWt', 178.226)
            self.set_config('chemical:transformations:Henry', 4.294e-5)

            self.set_config('chemical:transformations:Vpress', 0.0222)
            self.set_config('chemical:transformations:Tref_Vpress', 25)
            self.set_config('chemical:transformations:DeltaH_Vpress', 71733)

            self.set_config('chemical:transformations:Solub', 1.09)
            self.set_config('chemical:transformations:Tref_Solub', 25)
            self.set_config('chemical:transformations:DeltaH_Solub', 34800)

        elif self.get_config('chemical:compound') in ["Fluoranthene",'Pyrene','C1-Phenanthrene','C2-Dibenzothiophene']:

            #partitioning
            self.set_config('chemical:transfer_setup','organics')
            self.set_config('chemical:transformations:dissociation','nondiss')
            self.set_config('chemical:transformations:LogKOW',5.089)
            self.set_config('chemical:transformations:TrefKOW',25)
            self.set_config('chemical:transformations:KOC_sed',170850)
            self.set_config('chemical:transformations:KOC_DOM',53700)
            self.set_config('chemical:transformations:DeltaH_KOC_Sed',-47413)
            self.set_config('chemical:transformations:DeltaH_KOC_DOM',-27900)
            self.set_config('chemical:transformations:Setchenow', 0.2885)

            #degradation
            self.set_config('chemical:transformations:t12_W_tot', 1705.02)
            self.set_config('chemical:transformations:Tref_kWt', 25)
            self.set_config('chemical:transformations:DeltaH_kWt', 50000)                     ### generic
            self.set_config('chemical:transformations:t12_S_tot', 55000)
            self.set_config('chemical:transformations:Tref_kSt', 25)
            self.set_config('chemical:transformations:DeltaH_kSt', 50000)                     ### generic

            #volatilization
            self.set_config('chemical:transformations:MolWt', 202.26)
            self.set_config('chemical:transformations:Henry', 1.439e-5)

            self.set_config('chemical:transformations:Vpress', 0.00167)
            self.set_config('chemical:transformations:Tref_Vpress', 25)
            self.set_config('chemical:transformations:DeltaH_Vpress', 79581)

            self.set_config('chemical:transformations:Solub', 0.231)
            self.set_config('chemical:transformations:Tref_Solub', 25)
            self.set_config('chemical:transformations:DeltaH_Solub', 30315)

        elif self.get_config('chemical:compound') in ["Benzo-a-anthracene",'C2-Phenanthrene','Benzo-b-fluoranthene','Chrysene']:

            #partitioning
            self.set_config('chemical:transfer_setup','organics')
            self.set_config('chemical:transformations:dissociation','nondiss')
            self.set_config('chemical:transformations:LogKOW',5.724)
            self.set_config('chemical:transformations:TrefKOW',20)
            self.set_config('chemical:transformations:KOC_sed',732824)
            self.set_config('chemical:transformations:KOC_DOM',309029)
            self.set_config('chemical:transformations:DeltaH_KOC_Sed', -38000)                ### Pyrene value
            self.set_config('chemical:transformations:DeltaH_KOC_DOM', -25400)                ### Pyrene value
            self.set_config('chemical:transformations:Setchenow', 0.3605)

            #degradation
            self.set_config('chemical:transformations:t12_W_tot', 1467.62)
            self.set_config('chemical:transformations:Tref_kWt', 25)
            self.set_config('chemical:transformations:DeltaH_kWt', 50000)                     ### generic
            self.set_config('chemical:transformations:t12_S_tot', 46600)
            self.set_config('chemical:transformations:Tref_kSt', 25)
            self.set_config('chemical:transformations:DeltaH_kSt', 50000)                     ### generic

            #volatilization
            self.set_config('chemical:transformations:MolWt', 228.29)
            self.set_config('chemical:transformations:Henry', 6.149e-6)

            self.set_config('chemical:transformations:Vpress', 0.0000204)
            self.set_config('chemical:transformations:Tref_Vpress', 25)
            self.set_config('chemical:transformations:DeltaH_Vpress', 100680)

            self.set_config('chemical:transformations:Solub', 0.011)
            self.set_config('chemical:transformations:Tref_Solub', 25)
            self.set_config('chemical:transformations:DeltaH_Solub', 46200)

        elif self.get_config('chemical:compound') in ["Benzo-a-pyrene",'C3-Dibenzothiophene','C3-Phenanthrene']:

            #partitioning
            self.set_config('chemical:transfer_setup','organics')
            self.set_config('chemical:transformations:dissociation','nondiss')
            self.set_config('chemical:transformations:LogKOW', 6.124)
            self.set_config('chemical:transformations:TrefKOW',25)
            self.set_config('chemical:transformations:KOC_sed',1658700)
            self.set_config('chemical:transformations:KOC_DOM',172499)
            self.set_config('chemical:transformations:DeltaH_KOC_Sed', -43700)                ### mean value 16 PAHs
            self.set_config('chemical:transformations:DeltaH_KOC_DOM', -31280)
            self.set_config('chemical:transformations:Setchenow', 0.171)

            #degradation
            self.set_config('chemical:transformations:t12_W_tot', 1491.42)
            self.set_config('chemical:transformations:Tref_kWt', 25)
            self.set_config('chemical:transformations:DeltaH_kWt', 50000)                     ### generic
            self.set_config('chemical:transformations:t12_S_tot', 44934.76)
            self.set_config('chemical:transformations:Tref_kSt', 25)
            self.set_config('chemical:transformations:DeltaH_kSt', 50000)                     ### generic

            #volatilization
            self.set_config('chemical:transformations:MolWt', 252.32)
            self.set_config('chemical:transformations:Henry', 6.634e-7)

            self.set_config('chemical:transformations:Vpress', 0.00000136)
            self.set_config('chemical:transformations:Tref_Vpress', 25)
            self.set_config('chemical:transformations:DeltaH_Vpress', 107887)

            self.set_config('chemical:transformations:Solub', 0.00229)
            self.set_config('chemical:transformations:Tref_Solub', 25)
            self.set_config('chemical:transformations:DeltaH_Solub', 38000)

        elif self.get_config('chemical:compound') in ["Dibenzo-ah-anthracene",'Benzo-k-fluoranthene','Benzo-ghi-perylene','Indeno-123cd-pyrene']:

            #partitioning
            self.set_config('chemical:transfer_setup','organics')
            self.set_config('chemical:transformations:dissociation','nondiss')
            self.set_config('chemical:transformations:LogKOW', 6.618)
            self.set_config('chemical:transformations:TrefKOW',20)
            self.set_config('chemical:transformations:KOC_sed',5991009)
            self.set_config('chemical:transformations:KOC_DOM',4120975)
            self.set_config('chemical:transformations:DeltaH_KOC_Sed', -43700)                ### mean value 16 PAHs
            self.set_config('chemical:transformations:DeltaH_KOC_DOM', -30900)
            self.set_config('chemical:transformations:Setchenow', 0.338)

            #degradation
            self.set_config('chemical:transformations:t12_W_tot', 1464.67)
            self.set_config('chemical:transformations:Tref_kWt', 25)
            self.set_config('chemical:transformations:DeltaH_kWt', 50000)                     ### generic
            self.set_config('chemical:transformations:t12_S_tot', 40890.08)
            self.set_config('chemical:transformations:Tref_kSt', 25)
            self.set_config('chemical:transformations:DeltaH_kSt', 50000)                     ### generic

            #volatilization
            self.set_config('chemical:transformations:MolWt', 278.35)
            self.set_config('chemical:transformations:Henry', 4.894e-8)

            self.set_config('chemical:transformations:Vpress', 0.0000000427)
            self.set_config('chemical:transformations:Tref_Vpress', 25)
            self.set_config('chemical:transformations:DeltaH_Vpress', 112220)

            self.set_config('chemical:transformations:Solub', 0.00142)
            self.set_config('chemical:transformations:Tref_Solub', 25)
            self.set_config('chemical:transformations:DeltaH_Solub', 38000)                   ### Benzo-a-pyrene value

        elif self.get_config('chemical:compound') in ["Tralopyril",'Econea']:

            # https://downloads.regulations.gov/EPA-HQ-OPP-2013-0217-0017/content.pdf
            # https://downloads.regulations.gov/EPA-HQ-OPP-2013-0217-0005/content.pdf
            # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB81210138.htm

            #partitioning
            self.set_config('chemical:transfer_setup','organics')
            self.set_config('chemical:transformations:dissociation','nondiss')  # weak base
            #self.set_config('chemical:transformations:pKa_base', 7.08)         # EPA (pka 10.24 chemicalbook.com ?!?!)
            #self.set_config('chemical:transformations:pKa_acid', -1)           # ?
            self.set_config('chemical:transformations:LogKOW', 3.5)             # EPA-HQ-OPP-2013-0217-0005/17 experimental  ( Chemspider > 5 ?!? predicted )
            self.set_config('chemical:transformations:TrefKOW',25)              # ?
            self.set_config('chemical:transformations:DeltaH_KOC_Sed',-21036)   # Naphthalene (similar KOW)
            self.set_config('chemical:transformations:DeltaH_KOC_DOM',-25900)   # Naphthalene (similar KOW)
            self.set_config('chemical:transformations:Setchenow', -1.17)        # Artificial value to match KOC 4585 in sea water Tralopyril E-fate data review (freshwater)
            self.set_config('chemical:transformations:KOC_sed', 18586.5)        # Tralopyril E-fate data review (freshwater)

            #degradation
            self.set_config('chemical:transformations:t12_W_tot', 16)           # (9-16 hours)
            self.set_config('chemical:transformations:Tref_kWt', 10)            # ?
            self.set_config('chemical:transformations:DeltaH_kWt', 50000)       # PAH generic
            self.set_config('chemical:transformations:t12_S_tot', 16*30)        # ? for PAHs degradation in sediments is typically 20-30 times slower than in water
            self.set_config('chemical:transformations:Tref_kSt', 10)            # ?
            self.set_config('chemical:transformations:DeltaH_kSt', 50000)       # PAH generic

            #volatilization
            self.set_config('chemical:transformations:MolWt', 349.53)           # EPA-HQ-OPP-2013-0217-0005/17
            self.set_config('chemical:transformations:Henry', 9.3e-10)          # EPA-HQ-OPP-2013-0217-0005/17
            self.set_config('chemical:transformations:Vpress', 4.60e-8)         # Pa = 3.45e-10 mmHg EPA-HQ-OPP-2013-0217-0017
            self.set_config('chemical:transformations:Tref_Vpress', 25)         # EPA-HQ-OPP-2013-0217-005/17
            self.set_config('chemical:transformations:DeltaH_Vpress', 70000)    # https://www.chemspider.com/Chemical-Structure.159609.html

            self.set_config('chemical:transformations:Solub', 0.16)             # EPA-HQ-OPP-2013-0217-005/17 (sea water)
            self.set_config('chemical:transformations:Tref_Solub', 25)          # ?
            self.set_config('chemical:transformations:DeltaH_Solub', 30315)     # Fluoranthene (similar solubility)

        elif self.get_config('chemical:compound') == "Copper":
            self.set_config('chemical:transfer_setup','metals')
            self.set_config('chemical:transformations:Kd', 60.1)            # Tomczak et Al 2019
            #self.set_config('chemical:transformations:Kd', 50)             # Merlin Expo, high confidence
            self.set_config('chemical:transformations:S0', 17.0)            # note below

        elif self.get_config('chemical:compound') == "Zinc":
            self.set_config('chemical:transfer_setup','metals')
            self.set_config('chemical:transformations:Kd', 173)             # Tomczak et Al 2019
            #self.set_config('chemical:transformations:Kd', 100)            # Merlin Expo, high confidence
            self.set_config('chemical:transformations:S0', 17.0)            # note below

        elif self.get_config('chemical:compound') == "Lead":
            self.set_config('chemical:transfer_setup','metals')
            self.set_config('chemical:transformations:Kd', 369)             # Tomczak et Al 2019
            #self.set_config('chemical:transformations:Kd', 500)            # Merlin Expo, strong confidence
            self.set_config('chemical:transformations:S0', 17.0)            # note below

        elif self.get_config('chemical:compound') == "Vanadium":
            self.set_config('chemical:transfer_setup','metals')
            self.set_config('chemical:transformations:Kd', 42.9)            # Tomczak et Al 2019
            #self.set_config('chemical:transformations:Kd', 5)              # Merlin Expo, weak confidence
            self.set_config('chemical:transformations:S0', 17.0)            # note below

        elif self.get_config('chemical:compound') == "Cadmium":
            self.set_config('chemical:transfer_setup','metals')
            self.set_config('chemical:transformations:Kd', 134)             # Tomczak et Al 2019
            #self.set_config('chemical:transformations:Kd', 79)             # Merlin Expo, strong confidence
            #self.set_config('chemical:transformations:Kd', 6.6)            # Turner Millward 2002
            self.set_config('chemical:transformations:S0', 17.0)            # note below

        elif self.get_config('chemical:compound') == "Chromium":
            self.set_config('chemical:transfer_setup','metals')
            self.set_config('chemical:transformations:Kd', 124)             # Tomczak et Al 2019
            #self.set_config('chemical:transformations:Kd', 130)            # Cr(III) Merlin Expo, moderate confidence
            #self.set_config('chemical:transformations:Kd', 180)            # Turner Millward 2002
            self.set_config('chemical:transformations:S0', 17.0)            # note below

        elif self.get_config('chemical:compound') == "Nickel":
            self.set_config('chemical:transfer_setup','metals')
            self.set_config('chemical:transformations:Kd', 31.1)            # Tomczak et Al 2019
            #self.set_config('chemical:transformations:Kd', 25)             # Merlin Expo, strong confidence
            #self.set_config('chemical:transformations:Kd', 5.3)            # Turner Millward 2002
            self.set_config('chemical:transformations:S0', 17.0)            # note below

# Default value for S0 is set to 17.0. This correspond to a Kd at salinity 35 being 32.7%
# of the fresh water value, which was the average reduction obtained comparing the values
# in Tomczak et Al 2019 to the "ocean margins" recommended values in IAEA TRS no.422, for a
# selection of metals (Cd, Cr, Hg, Ni, Pb, Zn). This gives very similar results the value
# 15.8, suggested in Perianez 2018.
# https://doi.org/10.1016/j.apgeochem.2019.04.003
# https://www-pub.iaea.org/MTCD/Publications/PDF/TRS422_web.pdf
# https://doi.org/10.1016/j.jenvrad.2018.02.014
#
# Merlin Expo Kd values are mean values from Allison and Allison 2005
# https://cfpub.epa.gov/si/si_public_record_report.cfm?dirEntryId=135783

        elif self.get_config('chemical:compound') == "Nitrogen":
            self.set_config('chemical:transfer_setup', 'metals')
            self.set_config('chemical:transformations:Kd', 0.)  # Nitrogen does not interact with particulate matter or sediments
            self.set_config('chemical:transformations:S0', 17.0)

        elif self.get_config('chemical:compound') == "Alkalinity":
            self.set_config('chemical:transfer_setup', 'metals')
            self.set_config('chemical:transformations:Kd', 0.)  # Alkalinity does not interact with particulate matter or sediments
            self.set_config('chemical:transformations:S0', 17.0)

    def plot_mass(self,
                  legend=['dissolved','SPM','sediment'],
                  mass_unit='g',
                  time_unit='hours',
                  title=None,
                  filename=None,
                  start_date=None):
        """Plot chemical mass distribution between the different species
            legend      list of specie labels, for example ['dissolved','SPM','sediment']
            mass_unit   'g','mg','ug'
            time_unit   'seconds', 'minutes', 'hours' , 'days'
            title       figure title string
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        if not title == []:
            fig.suptitle(title)

        mass=self.result.mass
        sp=self.result.specie

        steps=len(self.result.time)

        bars=np.zeros((steps,5))

        mass_conversion_factor=1e-6
        if mass_unit=='g' and self.elements.variables['mass']['units']=='ug':
            mass_conversion_factor=1e-6
        if mass_unit=='mg' and self.elements.variables['mass']['units']=='ug':
            mass_conversion_factor=1e-3
        if mass_unit=='ug' and self.elements.variables['mass']['units']=='ug':
            mass_conversion_factor=1
        if mass_unit=='kg' and self.elements.variables['mass']['units']=='ug':
            mass_conversion_factor=1e-9

        time_conversion_factor = self.time_step_output.total_seconds() / (60*60)
        if time_unit=='seconds':
            time_conversion_factor = self.time_step_output.total_seconds()
        if time_unit=='minutes':
            time_conversion_factor = self.time_step_output.total_seconds() / 60
        if time_unit=='hours':
            time_conversion_factor = self.time_step_output.total_seconds() / (60*60)
        if time_unit=='days':
            time_conversion_factor = self.time_step_output.total_seconds() / (24*60*60)

        for i in range(steps):

            bars[i]=[np.sum(mass[:,i]*(sp[:,i]==0))*mass_conversion_factor,
                     np.sum(mass[:,i]*(sp[:,i]==1))*mass_conversion_factor,
                     np.sum(mass[:,i]*(sp[:,i]==2))*mass_conversion_factor,
                     np.sum(mass[:,i]*(sp[:,i]==3))*mass_conversion_factor,
                     np.sum(mass[:,i]*(sp[:,i]==4))*mass_conversion_factor]
        bottom=np.zeros_like(bars[:,0])
        if 'dissolved' in legend:
            ax.bar(np.arange(steps),bars[:,self.result.num_lmm],width=1.01,color='midnightblue')
            bottom=bars[:,self.result.num_lmm]
            print(f'dissolved: {str(bars[-1,self.result.num_lmm])} {mass_unit} ({str(100*bars[-1,self.result.num_lmm]/np.sum(bars[-1,:]))} %)')
        if 'DOC' in legend:
            ax.bar(np.arange(steps),bars[:,self.num_humcol],bottom=bottom,width=1.25,color='royalblue')
            bottom=bottom+bars[:,self.num_humcol]
            print(f'DOC: {str(bars[-1,self.result.num_humcol])} {mass_unit} ({str(100*bars[-1,self.result.num_humcol]/np.sum(bars[-1,:]))} %)')
        if 'SPM' in legend:
            ax.bar(np.arange(steps),bars[:,self.num_prev],bottom=bottom,width=1.25,color='palegreen')
            bottom=bottom+bars[:,self.num_prev]
            print(f'SPM: {str(bars[-1,self.result.num_prev])} {mass_unit} ({str(100*bars[-1,self.result.num_prev]/np.sum(bars[-1,:]))} %)')
        if 'sediment' in legend:
            ax.bar(np.arange(steps),bars[:,self.num_srev],bottom=bottom,width=1.25,color='orange')
            bottom=bottom+bars[:,self.num_srev]
            print(f'sediment: {str(bars[-1,self.result.num_srev])} {mass_unit} ({str(100*bars[-1,self.result.num_srev]/np.sum(bars[-1,:]))} %)')

        ax.legend(list(filter(None, legend)))
        ax.set_ylabel('mass (' + mass_unit + ')')
        ax.set_xlim(-.49,steps-1+0.49)
        if start_date is None:
            # Get current tick positions
            xticks = ax.get_xticks()
            # Set the ticks explicitly before setting labels
            ax.set_xticks(xticks)
            ax.set_xticklabels(np.round(xticks * time_conversion_factor))
            ax.set_xlabel('time (' + time_unit + ')')
        else:
            date_values = [datetime.strptime(start_date,"%Y-%m-%d") + i*self.time_step for i in range(steps)]

            # Set a fraction of datetime values as labels for the x-axis
            fraction = 24*7  # Show every second datetime value
            ax.set_xticks(np.arange(0, steps, fraction))
            ax.set_xticklabels([date.strftime('%m-%d') for date in date_values[::fraction]])
            ax.set_xlabel('time (month-day)')
        fig.show()

        if filename is not None:
            plt.savefig(filename, format=filename[-3:], transparent=True, bbox_inches="tight", dpi=300)

    ### Helpers for extract_summary_timeseries
    def calc_mass_conversion_factor(self, mass_unit):
        """
        Returns factor f such that:
            value_in_requested_unit = value_in_current_unit * f
        where current unit is self.elements.variables['mass']['units'].
        """
        _MASS_TO_GRAMS = {
                        "ug": 1e-6,
                        "µg": 1e-6,
                        "mg": 1e-3,
                        "g" : 1.0,
                        "kg": 1e3
                        }

        src = str(self.elements.variables["mass"]["units"]).strip()
        dst = str(mass_unit).strip()

        # normalize common microgram spelling
        src = "ug" if src == "µg" else src
        dst = "ug" if dst == "µg" else dst

        if src not in _MASS_TO_GRAMS:
            raise ValueError(f"Unsupported source mass unit: {src!r}")
        if dst not in _MASS_TO_GRAMS:
            raise ValueError(f"Unsupported destination mass unit: {dst!r}")

        # convert src -> grams -> dst
        return _MASS_TO_GRAMS[src] / _MASS_TO_GRAMS[dst]

    def calc_time_conversion_factor(self, time_unit):
        """
        Returns factor f such that:
            value_in_requested_unit = value_in_seconds * f
        where value_in_seconds is based on self.time_step_output.
        """
        _TIME_TO_SECONDS = {
                        "s":  1.0,
                        "m":  60.0,
                        "hr": 3600.0,
                        "h":  3600.0,
                        "d":  86400.0,
                    }

        u = str(time_unit).strip()
        # normalize optional aliases
        if u == "h":
            u = "hr"

        if u not in _TIME_TO_SECONDS:
            raise ValueError(
                f"Incorrect time_unit: {time_unit!r}, can be only {list(_TIME_TO_SECONDS.keys())}"
            )

        seconds = self.time_step_output.total_seconds()
        return seconds / _TIME_TO_SECONDS[u]

    @staticmethod
    def _cumulative_to_deltas_sum_ffill(w_cum, chunk_cols=20000, m_ts_out=None):
        """
        Compute m_ts[t] = sum_j max(ffill(w_cum[t,j]) - ffill(w_cum[t-1,j]), 0)
        where forward-fill is per-column (element), with NaNs after deactivation.

        w_cum: (T,N) float32 preferred
        Returns m_ts float64.
        """
        w_cum = np.asarray(w_cum)
        if w_cum.ndim != 2:
            raise ValueError("w_cum must be 2D (T,N)")

        T, N = w_cum.shape

        if m_ts_out is None:
            m_ts = np.zeros(T, dtype=np.float64)
        else:
            if m_ts_out.shape != (T,):
                raise ValueError(f"m_ts_out must have shape {(T,)}, got {m_ts_out.shape}")
            m_ts_out.fill(0.0)
            m_ts = m_ts_out

        # Process in chunks across columns
        for start in range(0, N, chunk_cols):
            end = min(start + chunk_cols, N)
            width = end - start
            # float32 buffers
            prev_ff = np.zeros(width, dtype=np.float32)
            cur_ff  = np.empty(width, dtype=np.float32)
            delta   = np.empty(width, dtype=np.float32)

            for t in range(T):
                cur = w_cum[t, start:end]
                # cur_ff = prev_ff
                np.copyto(cur_ff, prev_ff)
                # forward-fill: overwrite only finite entries with cur
                finite = np.isfinite(cur)
                np.copyto(cur_ff, cur, where=finite)
                # delta = cur_ff - prev_ff
                np.subtract(cur_ff, prev_ff, out=delta)
                # clamp negatives to 0
                np.maximum(delta, 0.0, out=delta)
                # clear any non-finite
                delta[~np.isfinite(delta)] = 0.0
                # accumulate sum in float64 for stability
                m_ts[t] += delta.sum(dtype=np.float64)
                # update prev
                np.copyto(prev_ff, cur_ff)

        return m_ts

    @staticmethod
    def adv_out_mass_ts(mass, adv_out, mass_conversion_factor=1.0, chunk_cols=20000):
        """
        mass:   (T,N) float array with NaNs after deactivation
        adv_out:(T,N) bool array (True == outside)
        Ignores initial outside at t=0 (only counts False->True transitions for t>0).
        Returns:
            exit_outside_at_t: (T,) float64
            cum_exit_outside:  (T,) float64
        """
        mass = np.asarray(mass)
        adv_out = np.asarray(adv_out, dtype=bool)

        T, N = mass.shape
        exit_outside_at_t = np.zeros(T, dtype=np.float64)

        # No advection out
        if not adv_out.any():
            return (exit_outside_at_t * mass_conversion_factor,
                    exit_outside_at_t.copy() * mass_conversion_factor)

        for start in range(0, N, chunk_cols):
            end = min(start + chunk_cols, N)
            width = end - start

            prev_out = adv_out[0, start:end].copy()        # prev_out[0] == adv_out[0] -> ignore initial outside
            prev_ff  = np.zeros(width, dtype=mass.dtype)   # last finite mass per element
            cur_ff   = np.empty(width, dtype=mass.dtype)

            # initialize prev_ff from t=0 if finite
            cur0 = mass[0, start:end]
            finite0 = np.isfinite(cur0)
            if finite0.any():
                prev_ff[finite0] = cur0[finite0]

            # t=0 contributes nothing by design (ignore initial outside), so start from t=1
            for t in range(1, T):
                out_now = adv_out[t, start:end]
                became_out = (~prev_out) & out_now          # transitions only

                if became_out.any():
                    cur = mass[t, start:end]
                    finite = np.isfinite(cur)
                    # forward-fill mass
                    np.copyto(cur_ff, prev_ff)
                    if finite.any():
                        cur_ff[finite] = cur[finite]
                    # sum mass of those that became outside
                    exit_outside_at_t[t] += np.nansum(cur_ff[became_out], dtype=np.float64)
                    # update prev_ff
                    np.copyto(prev_ff, cur_ff)
                else:
                    # still need to update prev_ff if mass has new finite values this step
                    cur = mass[t, start:end]
                    finite = np.isfinite(cur)
                    if finite.any():
                        prev_ff[finite] = cur[finite]
                # update prev_out for transitions
                prev_out = out_now

        exit_outside_at_t *= mass_conversion_factor
        cum_exit_outside = np.cumsum(exit_outside_at_t, dtype=np.float64)
        return exit_outside_at_t, cum_exit_outside

    @staticmethod
    def _event_mass_ts(mass, event_mask, adv_out=None,
                       allow_initial_event=False,
                       require_inside_now=False,
                       require_inside_prev=False,
                       chunk_cols=50000,
                       mass_conversion_factor=1.0,
                    ):
        """
        Generic event counter: sums mass (forward-filled) for elements that become event==True at each timestep.

        mass:       (T,N) float array with NaNs after deactivation
        event_mask: (T,N) bool array (e.g., stranded, in_buried_sed)
        adv_out:    (T,N) bool array, used if require_inside_* is True

        allow_initial_event: if True, count event at t=0 when event_mask[0]==True
        require_inside_now:  if True, count only when ~adv_out[t]
        require_inside_prev: if True, count only when ~adv_out[t-1] (for t>0)

        Returns:
          event_ts (T,) float64 in converted units, and cumulative (T,) float64.
        """
        mass = np.asarray(mass)
        event_mask = np.asarray(event_mask, dtype=bool)
        T, N = mass.shape

        if require_inside_now or require_inside_prev:
            if adv_out is None:
                raise ValueError("adv_out must be provided when require_inside_* is True")
            adv_out = np.asarray(adv_out, dtype=bool)

        event_ts = np.zeros(T, dtype=np.float64)

        if not event_mask.any():
            event_ts *= mass_conversion_factor
            return event_ts, np.cumsum(event_ts)

        for start in range(0, N, chunk_cols):
            end = min(start + chunk_cols, N)
            width = end - start

            prev_event = event_mask[0, start:end].copy()  # used to detect transitions
            prev_ff = np.zeros(width, dtype=mass.dtype)
            cur_ff  = np.empty(width, dtype=mass.dtype)
            # initialize prev_ff from t=0 if finite
            cur0 = mass[0, start:end]
            finite0 = np.isfinite(cur0)
            if finite0.any():
                prev_ff[finite0] = cur0[finite0]
            # t=0 event handling
            if allow_initial_event:
                ev0 = event_mask[0, start:end]
                if require_inside_now:
                    ev0 &= ~adv_out[0, start:end]
                # require_inside_prev doesn't apply at t=0
                if ev0.any():
                    event_ts[0] += np.nansum(prev_ff[ev0], dtype=np.float64)
            #  t>=1
            for t in range(1, T):
                ev_now = event_mask[t, start:end]
                became = (~prev_event) & ev_now  # False->True transitions

                if became.any():
                    if require_inside_now:
                        became &= ~adv_out[t, start:end]
                    if require_inside_prev:
                        became &= ~adv_out[t-1, start:end]
                if became.any():
                    cur = mass[t, start:end]
                    finite = np.isfinite(cur)

                    np.copyto(cur_ff, prev_ff)
                    if finite.any():
                        cur_ff[finite] = cur[finite]

                    event_ts[t] += np.nansum(cur_ff[became], dtype=np.float64)
                    np.copyto(prev_ff, cur_ff)
                else:
                    # still update forward-fill state if new finite values appear
                    cur = mass[t, start:end]
                    finite = np.isfinite(cur)
                    if finite.any():
                        prev_ff[finite] = cur[finite]

                prev_event = ev_now

        event_ts *= mass_conversion_factor
        return event_ts, np.cumsum(event_ts, dtype=np.float64)

    @staticmethod
    def _transition_mass_ts(mass, specie, inside_mask, src_idx, dst_idx,
                            steps, N_elem,
                            mass_conversion_factor=1.0,
                            require_inside_now=True,
                            require_inside_prev=True,
                            chunk_cols=50000):
        """
        Per-timestep mass for exact specie transitions src_idx -> dst_idx.

        Only transitions inside the domain are counted:
          - require_inside_now=True  -> element must be inside at timestep t
          - require_inside_prev=True -> element must be inside at timestep t-1
        """
        if (src_idx is None) or (dst_idx is None):
            return np.zeros(steps, dtype=np.float32)


        mass = np.asarray(mass)
        specie = np.asarray(specie)
        inside_mask = np.asarray(inside_mask, dtype=bool)

        if mass.ndim != 2 or specie.ndim != 2 or inside_mask.ndim != 2:
            raise ValueError("mass, specie, and inside_mask must all be 2D (T,N)")
        if mass.shape != specie.shape or mass.shape != inside_mask.shape:
            raise ValueError("mass, specie, and inside_mask must have the same shape")

        ts = np.zeros(steps, dtype=np.float64)

        if steps < 2:
            return ts.astype(np.float32, copy=False)

        for start in range(0, N_elem, chunk_cols):
            end = min(start + chunk_cols, N_elem)
            width = end - start

            prev_ff = np.zeros(width, dtype=mass.dtype)
            cur_ff = np.empty(width, dtype=mass.dtype)

            cur0 = mass[0, start:end]
            finite0 = np.isfinite(cur0)
            if finite0.any():
                prev_ff[finite0] = cur0[finite0]

            for t in range(1, steps):
                trans = ((specie[t-1, start:end] == src_idx) &
                    (specie[t,   start:end] == dst_idx))

                if trans.any():
                    if require_inside_now:
                        trans &= inside_mask[t, start:end]
                    if require_inside_prev:
                        trans &= inside_mask[t-1, start:end]

                cur = mass[t, start:end]
                finite = np.isfinite(cur)

                np.copyto(cur_ff, prev_ff)
                if finite.any():
                    cur_ff[finite] = cur[finite]

                if trans.any():
                    ts[t] += np.nansum(cur_ff[trans], dtype=np.float64)

                np.copyto(prev_ff, cur_ff)

        ts *= mass_conversion_factor
        return ts.astype(np.float32, copy=False)


    def sum_transition_pairs(self, pairs, mass, specie, inside_mask, steps, N_elem,
                              mass_conversion_factor=1.0, chunk_cols=50000):
        out = np.zeros(steps, dtype=np.float32)
        for src_idx, dst_idx in pairs:
            out += self._transition_mass_ts(
                mass=mass,
                specie=specie,
                inside_mask=inside_mask,
                src_idx=src_idx,
                dst_idx=dst_idx,
                steps=steps,
                N_elem=N_elem,
                mass_conversion_factor=mass_conversion_factor,
                require_inside_now=True,
                require_inside_prev=True,
                chunk_cols=chunk_cols,
            )
        return out

    def extract_summary_timeseries(self,
            timeseries_file_path,
            mass_unit='g',
            time_unit='h',
            shp_file_path = None,
            lon_min = None,
            lon_max = None,
            lat_min = None,
            lat_max = None,
            start_date = None,
            end_date = None,
            time_start=None,
            time_end=None,
            extra_fields=None,
            save_files = True,
            verbose = False
            ):
        """
        Extract, aggregate, and optionally export (.csv) a mass-budget summary time series from a
        ChemicalDrift simulation stored in self.result.

        This routine builds 1D time series (one value per output timestep) from element-wise
        (trajectory,time) variables, taking care of:
        - unit conversion (mass and time),
        - removal of invalid trajectories (those with no valid lat/lon),
        - definition of the “inside simulation domain” to avoid double-counting mass that has left
          the domain,
        - robust handling of NaNs after element deactivation (forward-fill per element when needed).

        The resulting DataFrame contains:
        1) Mass currently inside the system (water, active sediment layer, total)
        2) Mass split by species (absolute and % of inside-system mass)
        3) Emitted mass (when each element first appears)
        4) Eliminated/removed mass mechanisms (per-timestep increments and cumulative)
        5) Exiting events (advection out, stranding, burial) as per-timestep increments and cumulative
        6) Aggregated per-timestep phase-transfer mass fluxes reconstructed from species transitions

        Parameters
        ----------
        timeseries_file_path: str, Output CSV file path. Must end with ".csv" when save_files is True.
        mass_unit:            str, Mass unit for outputs. Allowed: 'kg', 'g', 'mg', 'ug', 'µg'. Default is 'g'.
                                       The conversion is performed from the unit stored in self.elements.variables['mass']['units'].
        time_unit:            str, Time unit for the exported time axis.
                                       Allowed: 's' (seconds), 'm' (minutes), 'hr' or 'h' (hours), 'd' (days). Default is 'h'.
                                       The conversion is based on self.time_step_output (the model output interval).
        shp_file_path:        str, Path to a shapefile defining the “inside” domain polygon(s).
                                       If provided, adv_out is computed as NOT-within shapefile.
        lon_min, lon_max,
        lat_min, lat_max:     float, Geographic bounding box used to define the inside domain when no shapefile is provided.
                                      Takes precedence over deactivate_coords. Default is None.
        start_date, end_date: pandas.Timestamp/np.datetime64, Start/end dates for slicing the exported period.
        time_start, time_end: int/float, Start/end of slicing window for the exported period, in requested time_unit
                                         (e.g. hours if time_unit='h').
                                         Takes precedence over start_date, end_date.
        extra_fields:        dict, optional Extra 2D fields from self.result to aggregate over ALL elements
                             in the selected period, with no active/in/out/species masks.
                             Format: { "field1": {"Name": "AAA","Unit": "UM", "mode": "mean"},}
                             where:
                               - dict key = variable name in self.result
                               - "Name" = output column label
                               - "Unit" = output unit label
                               - "mode" can be:
                                   * "mean" -> per-timestep nanmean across elements
                                              <Name> [Unit] = per-timestep mean over valid elements
                                              <Name> [Unit]__n_valid = number of valid elements used in that mean
                                   * "sum"  -> per-timestep nansum across elements
        save_files:           bool, If True, export the resulting DataFrame to timeseries_file_path and return None.
                                    If False, return the DataFrame and do not write files.
        verbose:              bool, Print progress information.

        Processes / logic
        -----------------
        1) Ensures species and transfer-rate metadata exist (init_species/init_transfer_rates) if not
           already available.
        2) Detects whether drift:deactivate_north_of/south_of/east_of/west_of were used
           (deactivate_coords).
        3) Computes conversion factors:
           - mass_conversion_factor: from self.elements.variables['mass']['units'] -> mass_unit
           - time_conversion_factor: from seconds (self.time_step_output) -> time_unit
        4) Cleans trajectories: drops any trajectory with no finite lat and no finite lon over the run.
        5) Extracts required 2D arrays (T,N), e.g. mass and cumulative eliminated-mass arrays.
           NaNs are expected after deactivation.
        6) Builds masks:
           - adv_out (True == element considered outside domain) using one of:
               (i) shp_file_path polygon test
               (ii) bounding box (lon_min/lon_max/lat_min/lat_max)
               (iii) if deactivate_coords: status == 'outside'
               (iv) fallback: status == 'missing_data' (if present)
             If none apply, the function raises an error (domain cannot be determined).
           - in_water_column: element is in a water-column species at each timestep
           - in_sediment_layer: element is in the active/mixed sediment layer species at each timestep
           - in_buried_sed: element is in the buried sediment compartment at each timestep (if enabled)
        7) Defines the inside-system mask:
             inside_mask = ~adv_out
             and additionally excludes elements with status 'seeded_on_land' and/or 'stranded'
             (if those status categories exist).
        8) Aggregates “current” inside-system mass time series:
           - mass_water_ts, mass_sed_ts, mass_actual_ts (water + sediment), all in mass_unit
        9) Aggregates mass by species (inside-system only) and percent of inside-system mass:
           - mass_sp_<species>_ts and perc_sp_<species>_ts (0–100)
        10) Emitted mass:
            - Detects the first timestep each element has finite mass, sums that mass into
              mass_emitted_ts, and produces mass_emitted_cumulative.
        11) Eliminated mass mechanisms:
            - Many eliminated-mass variables in self.result are cumulative per element.
              The function converts cumulative -> per-timestep increments by forward-filling each
              element’s cumulative series across NaNs and then summing positive deltas.
            - Produces both *_ts (increment at each timestep) and *_cumulative
              (cumulative sum of increments).
        12) Exiting events (priority to adv_out where applicable):
            - Advection out: counts only False->True transitions in adv_out for t>0
              (initial outside at t=0 is ignored). Uses forward-filled mass at the transition time.
            - Stranding: counts False->True transitions of 'stranded' elements, only if the element
              is inside now AND was inside at the previous timestep
              (requires ~adv_out[t] and ~adv_out[t-1]).
              Initial stranded at t=0 is ignored.
            - Burial: counts transitions into buried state (can count at t=0 if starts buried),
              only when inside now (~adv_out[t]). Multiple burial events per element are allowed if
              it leaves and re-enters burial.
        13) Aggregated phase-transfer time series:
            - Reconstructs per-timestep transferred mass from exact species transitions
              src -> dst using forward-filled mass at the transition time.
            - Counts only transitions occurring inside the domain both before and after the
              transition.
            - Produces the following aggregated transition series:
              * mass_ads_to_sed_ts
              * mass_des_from_sed_ts
              * mass_ads_to_spm_ts
              * mass_des_from_spm_ts
              * mass_aggr_doc_poly_to_spm_ts
              * mass_disaggr_poly_ts
              * mass_ads_to_doc_ts
              * mass_des_from_doc_ts

        Returns
        -------
        If save_files is True:
            None
            Writes CSV to timeseries_file_path.

        If save_files is False:
            pandas.DataFrame with the following columns:

            Time axis
            ---------
            time [<time_unit>]
                Time since simulation start, converted to time_unit.
            date_of_timestep
                Calendar date for each output step.

            Inside-system mass (in mass_unit)
            --------------------------------
            mass_water_ts
                Total mass in water column (inside_mask).
            mass_sed_ts
                Total mass in active sediment layer (inside_mask).
            mass_actual_ts
                Total inside-system mass (= water + active sediment).

            Emissions (in mass_unit)
            ------------------------
            mass_emitted_ts
                Mass first appearing at each timestep.
            mass_emitted_cumulative
                Cumulative emitted mass.

            Eliminated mass mechanisms (in mass_unit)
            -----------------------------------------
            For each available cumulative variable in self.result among:
              mass_degraded, mass_degraded_water, mass_degraded_sediment,
              mass_volatilized,
              mass_photodegraded, mass_biodegraded,
              mass_biodegraded_water, mass_biodegraded_sediment,
              mass_hydrolyzed, mass_hydrolyzed_water, mass_hydrolyzed_sediment

            the function provides:
              <var>_ts
                  Per-timestep increment.
              <var>_cumulative
                  Cumulative sum of increments.

            Domain-exit / event-based removals (in mass_unit)
            -------------------------------------------------
            mass_adv_out_ts
                Mass exiting by advection out (False->True transitions in adv_out).
            mass_adv_out_cumulative
                Cumulative advected-out mass.
            mass_stranded_ts
                Mass stranded (False->True transitions), only if inside now and previously.
            mass_stranded_cumulative
                Cumulative stranded mass.
            mass_buried_ts
                Mass entering buried sediment (transition into buried), burial at t=0 allowed.
            mass_buried_cumulative
                Cumulative buried mass.

            Aggregated phase-transfer time series (in mass_unit)
            ----------------------------------------------------
            mass_ads_to_sed_ts
                Adsorption to sediments, aggregated from dissolved/cationic dissolved -> sediment
                reversible transitions.
            mass_des_from_sed_ts
                Desorption from sediments, aggregated from sediment reversible ->
                dissolved/cationic dissolved transitions.
            mass_ads_to_spm_ts
                Adsorption to suspended particles, aggregated from dissolved/cationic dissolved ->
                particle reversible transitions.
            mass_des_from_spm_ts
                Desorption from suspended particles, aggregated from particle reversible ->
                dissolved/cationic dissolved transitions.
            mass_aggr_doc_poly_to_spm_ts
                Aggregation from DOC/polymer pools to suspended particles.
            mass_disaggr_poly_ts
                Disaggregation of polymer to dissolved anionic form.
            mass_ads_to_doc_ts
                Adsorption/binding to DOC-like pools (humic colloid and polymer), aggregated across
                the corresponding active transitions.
            mass_des_from_doc_ts
                Desorption/unbinding from DOC-like pools (humic colloid -> dissolved forms).

            Species mass and percent (inside-system only; in mass_unit and %)
            -----------------------------------------------------------------
            For each species key present in the model (excluding 'sed_buried' as it is
            outside-system storage):
              mass_sp_<key>_ts
                  Inside-system mass of that species.
              perc_sp_<key>_ts
                  Percent of inside-system mass
                  (= mass_sp_<key>_ts / mass_actual_ts * 100).

            Species keys may include (depending on model configuration):
              dissolved, dissolved_anion, dissolved_cation, doc, colloid, polymer,
              spm_rev, spm_srev, spm_irrev, sed_rev, sed_srev, sed_irrev, sed_buried
        """
        import opendrift
        import pandas as pd
        import numpy as np

        if shp_file_path is not None:
            import geopandas as gpd

        if save_files:
            if timeseries_file_path is None:
                raise ValueError("timeseries_file_path is unspecified when save_file is True")
            if not timeseries_file_path.endswith(".csv"):
                raise ValueError("timeseries_file_path must end with .csv")

        # Initialize init_species() and init_transfer_rates() if they were not stored in self.result
        required_meta = ['nspecies', 'name_species', 'transfer_rates']
        need_init = (
        (not all(hasattr(self.result, attr) for attr in required_meta)) or
        (not hasattr(self, 'num_lmm') and not hasattr(self, 'num_lmmcation'))    )

        if need_init:
            if self.mode != opendrift.models.basemodel.Mode.Config:
                self.mode = opendrift.models.basemodel.Mode.Config
            self.init_species()
            self.init_transfer_rates()
            if self.mode != opendrift.models.basemodel.Mode.Result:
                self.mode = opendrift.models.basemodel.Mode.Result

        # Check if deactivate_N/S/E/W_of where specified
        deactivate_coords = any(self.get_config(k) for k in (
                'drift:deactivate_north_of', 'drift:deactivate_south_of',
                'drift:deactivate_east_of', 'drift:deactivate_west_of'))

        # Define id and name of species
        specie_ids_num = {
                          "dissolved" : (self.num_lmm if hasattr(self, "num_lmm") else None),
                          "dissolved_anion": (self.num_lmmanion if hasattr(self, "num_lmmanion") else None),
                          "dissolved_cation": (self.num_lmmcation if hasattr(self, "num_lmmcation") else None),
                          "doc" : (self.num_humcol if hasattr(self, "num_humcol") else None),
                          "colloid": (self.num_col if hasattr(self, "num_col") else None),
                          "polymer": (self.num_polymer if hasattr(self, "num_polymer") else None),
                          "spm_rev" : (self.num_prev if hasattr(self, "num_prev") else None),
                          "spm_srev": (self.num_psrev if hasattr(self, "num_psrev") else None),
                          "spm_irrev": (self.num_pirrev if hasattr(self, "num_pirrev") else None),
                          "sed_rev" : (self.num_srev if hasattr(self, "num_srev") else None),
                          "sed_srev": (self.num_ssrev if hasattr(self, "num_ssrev") else None),
                          "sed_irrev": (self.num_sirrev if hasattr(self, "num_sirrev") else None),
                          "sed_buried": (self.num_sburied if hasattr(self, "num_sburied") else None),
                          }

        status_categories = self.status_categories

        outside_idx       = status_categories.index('outside')         if 'outside'  in status_categories else None
        stranded_idx      = status_categories.index('stranded')        if 'stranded' in status_categories else None
        missing_data_idx  = status_categories.index('missing_data')    if 'missing_data'   in status_categories else None
        seeded_on_land_idx = status_categories.index('seeded_on_land') if 'seeded_on_land' in status_categories else None
        # active_idx        = status_categories.index('active')          if 'active'   in status_categories else None
        # removed_idx       = status_categories.index('removed')         if 'removed'  in status_categories else None
        # specie_ids_name = {v: k for k, v in specie_ids_num.items() if v is not None}

        # Define time and mass convertions
        if mass_unit not in ['g','mg','ug','µg','kg']:
            raise ValueError(f"Incorrect mass_unit: '{mass_unit}', can be only 'g','mg','ug','µg','kg'")
        mass_conversion_factor = self.calc_mass_conversion_factor(mass_unit)

        if time_unit not in ['s','m','hr','h','d']:
            raise ValueError(f"Incorrect time_unit: '{time_unit}', can be only 's', 'm', 'hr', 'h', 'd'")
        time_conversion_factor = self.calc_time_conversion_factor(time_unit)

        if verbose:
            print("Extracting data from simulation")
        # Extract properties from simulation

        ds = self.result
        vars_time = [v for v in ds.data_vars if set(ds[v].dims) == {"trajectory", "time"}]
        if not vars_time:
            raise ValueError("No (trajectory,time) variables found (in any dim order).")

        # Check only lat/lon
        valid_traj = (
            ds["lon"].notnull().any("time") & ds["lat"].notnull().any("time"))
        removed = ds.trajectory.where(~valid_traj, drop=True).values
        if verbose:
            if len(removed) > 0:
                print(f"Removed IDs {removed} from self.result  as lat/lon were NaN")

        # keep only valid trajectories
        ds_clean = ds.isel(trajectory=valid_traj)
        self.result = ds_clean
        del ds_clean, ds

        # Optional time filtering
        result_ds = self.result

        # Build full time axes on the *current* (cleaned) dataset
        full_time_date = pd.to_datetime(result_ds.time.values)
        full_steps = full_time_date.size
        full_time_steps = (np.arange(full_steps, dtype=np.float64) * time_conversion_factor)  # in requested time_unit

        mask_t = np.ones(full_steps, dtype=bool)
        use_time_window = (time_start is not None) or (time_end is not None)

        # Calendar-date filtering (only if no explicit time window)
        if (not use_time_window) and (start_date is not None):
            sd = pd.to_datetime(start_date)
            mask_t &= (full_time_date >= sd)
        if (not use_time_window) and (end_date is not None):
            ed = pd.to_datetime(end_date)
            mask_t &= (full_time_date <= ed)

        # Time-since-start filtering (inclusive)
        if time_start is not None:
            mask_t &= (full_time_steps >= float(time_start))
        if time_end is not None:
            mask_t &= (full_time_steps <= float(time_end))

        # Apply slicing only if any filter was requested
        filter_requested = (start_date is not None) or (end_date is not None) or (time_start is not None) or (time_end is not None)

        pad_rows = 0
        time_index0 = 0
        window_starts_at_sim0 = True

        if filter_requested:
            idx = np.flatnonzero(mask_t)
            if idx.size == 0:
                raise ValueError("Requested time/date filter returned an empty time window.")

            i0, i1 = int(idx[0]), int(idx[-1])

            # include one step before i0 as padding (if possible) for deltas/transitions
            slice_start = max(i0 - 1, 0)
            slice_end   = i1 + 1  # python slice end is exclusive

            pad_rows = i0 - slice_start          # 1 if i0>0 else 0
            time_index0 = slice_start            # offset to keep 'time since simulation start' consistent
            window_starts_at_sim0 = (i0 == 0)

            self.result = result_ds.isel(time=slice(slice_start, slice_end))
            result_ds = self.result

        # Number of timesteps (after optional filtering)
        steps = len(self.result.time)

        # Dynamically extract attributes and create dictionary
        attrs_ls = ['lat', 'lon', 'mass',
                    'mass_degraded','mass_degraded_water', 'mass_degraded_sediment',
                    'mass_volatilized', 'mass_photodegraded', 'mass_biodegraded',
                    'mass_biodegraded_water', 'mass_biodegraded_sediment',
                    'mass_hydrolyzed', 'mass_hydrolyzed_water', 'mass_hydrolyzed_sediment'
                    ]

        result_ds=self.result
        extracted_attrs_dict_2d = {}
        for attr in attrs_ls:
            extracted_attrs_dict_2d[attr] = ((getattr(result_ds, attr).T.values)
                               if hasattr(result_ds, attr) else None)

        # Get mask for elements in water column and sedments for each timestep
        N_elem = extracted_attrs_dict_2d['mass'].shape[1]
        status = result_ds.status.T.values                     # (T,N)
        specie = result_ds.specie.T.values                     # (T,N)

        in_water_column_ls = []
        in_water_column_ls = [
            val for name in ('num_lmm', 'num_lmmcation', 'num_lmmanion',
                             'num_humcol', 'num_col', 'num_polymer',
                             'num_prev', 'num_psrev', 'num_pirrev')
            if (val := getattr(self, name, None)) is not None
        ]
        in_water_column = np.any([specie == value for value in in_water_column_ls], axis=0)

        # Active (mixed) sediment layer includes all sediment pools that are in the mixed layer:
        # reversible, slowly reversible, and (if enabled) irreversible.
        in_sediment_column_ls = []
        in_sediment_column_ls = [
            val for name in ('num_srev', 'num_ssrev', 'num_sirrev')
            if (val := getattr(self, name, None)) is not None
        ]
        in_sediment_layer = np.any([specie == value for value in in_sediment_column_ls], axis=0)

        # Buried sediment compartment (deep storage)
        in_buried_sed = (specie == self.num_sburied) if hasattr(self, 'num_sburied') else np.zeros_like(specie, dtype=bool)
        def status_mask(idx):
            return (status == idx) if idx is not None else np.zeros_like(status, bool)
        # mask for elements active until the end of each timestep
        # active         = status_mask(active_idx)
        # mask for elements seeded_on_land or stranded
        # seeded_on_land = status_mask(seeded_on_land_idx) if seeded_on_land_idx is not None else None
        stranded       = status_mask(stranded_idx) if stranded_idx is not None else None

        # Find which elements are advected out of the domain
        if verbose:
            print("Calculating mass advected out of the simulation")
        # (i) Use shapefile if provided
        if shp_file_path is not None:
            if verbose:
                print("shapefile used")
            gdf_shapefile = gpd.read_file(shp_file_path)

            ntraj = extracted_attrs_dict_2d['lon'].shape[1]
            adv_out = np.empty((steps, ntraj), dtype=bool)

            for t in range(steps):
                lon_ = extracted_attrs_dict_2d['lon'][t]
                lat_ = extracted_attrs_dict_2d['lat'][t]

                pts = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(lon_, lat_, crs=gdf_shapefile.crs)
                )
                joined = gpd.sjoin(pts, gdf_shapefile, predicate='within', how='left')
                inside = joined["index_right"].notna().to_numpy()
                adv_out[t] = ~inside
        # (ii) polygon built from (lon_min/lon_max/lat_min/lat_max)
        elif (lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None):
            if verbose:
                    print(f"lat_min {lat_min}, lat_max: {lat_max}, lon_min: {lon_min}, lon_max: {lon_max} used")
            lon = extracted_attrs_dict_2d['lon']
            lat = extracted_attrs_dict_2d['lat']
            adv_out = ~( (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max))
            # adv_out |= ~(np.isfinite(lon) & np.isfinite(lat))  # mark NaNs as outside if desired
        # (iii) deactivate_coords + outside status
        elif deactivate_coords:
            if verbose:
                print("deactivate_north_of/south_of/_east_of/_west_of parameters used")
            if ('outside' in status_categories):
                adv_out = (status == outside_idx)
            else:
                adv_out = np.zeros_like(status, dtype=bool)
        # (iv) missing_data status
        elif 'missing_data' in status_categories:
            if verbose:
                print("missing_data used")
            adv_out = (status == missing_data_idx)
        else:
            error = (
                "shp_file_path and lat/lon limits not specified when "
                "deactivate_north_/south_/_east_/_west_of parameters "
                "were not used"
            )
            raise ValueError(error)

        # Keep time since simulation start (not since window start)
        time_steps = (np.arange(time_index0, time_index0 + steps, dtype=np.float64) * time_conversion_factor)
        # Use dataset time coordinate directly
        time_date_serie = pd.to_datetime(self.result.time.values)

        adv_out=adv_out.astype(bool)
        in_water_column=in_water_column.astype(bool)
        in_sediment_layer=in_sediment_layer.astype(bool)
        in_buried_sed=in_buried_sed.astype(bool)

        def masked_nansum(arr, mask, axis):
            """Sum arr over axis, only where mask==True; ignore NaNs inside mask."""
            return np.nansum(np.where(mask, arr, np.nan), axis=axis)

        # Dynamically extract 1D timeseries
        mass_dict_1d = {}
        mass_eliminated_dict_1d = {}
        mass_sp_dict_1d = {}
        perc_sp_dict_1d = {}
        perc_elim_dict_1d = {}
        mass_transition_dict_1d = {}
        extra_fields_dict_1d = {}
        extra_fields_count_dict_1d = {}

        # 1) Inside-system mass (not outside/seeded_on_land/stranded elements)
        inside_mask = ~adv_out
        if seeded_on_land_idx is not None:
            inside_mask &= (status != seeded_on_land_idx)
        if stranded_idx is not None:
            inside_mask &= (status != stranded_idx)

        mass_dict_1d["mass_water_ts"] = masked_nansum(extracted_attrs_dict_2d['mass'], in_water_column  & inside_mask, axis=1) * mass_conversion_factor
        mass_dict_1d["mass_sed_ts"] = masked_nansum(extracted_attrs_dict_2d['mass'], in_sediment_layer  & inside_mask, axis=1) * mass_conversion_factor
        mass_dict_1d["mass_actual_ts"] = mass_dict_1d["mass_water_ts"] + mass_dict_1d["mass_sed_ts"]

        # 2) Inside-system mass for each specie (not buried/outside/seeded_on_land/stranded elements)
        mass = np.asarray(extracted_attrs_dict_2d['mass'])   # (T,N)
        T = mass.shape[0]
        zeros_ts = np.zeros(T, dtype=float)

        den = mass_dict_1d["mass_actual_ts"].astype(float)
        den_safe = np.where(den > 0, den, np.nan)

        for key, sp_idx in specie_ids_num.items():
            # skip percent for buried sediments (outside system)
            if sp_idx is None:
                m_ts = zeros_ts
            else:
                m_ts = masked_nansum(mass, inside_mask & (specie == sp_idx), axis=1) * mass_conversion_factor
            # mass of each specie during each ts
            mass_sp_dict_1d[f"mass_sp_{key}_ts"] = m_ts
            if key == "sed_buried":
                # skip percent for buried sediments (outside system)
                # but keep track or buried mass in simulation
                continue
            # percentage of mass for each specie during each ts (vs "mass_actual_ts")
            if sp_idx is None:
                perc_sp_dict_1d[f"perc_sp_{key}_ts"] = zeros_ts
            else:
                perc_sp_dict_1d[f"perc_sp_{key}_ts"] = np.nan_to_num((m_ts / den_safe) * 100.0, nan=0.0)

        # 3) Emitted mass (first timestep each element is finite)
        valid = np.isfinite(extracted_attrs_dict_2d['mass'])
        first_seen_idx = valid.argmax(axis=0)                 # 0 if never seen; guard with has_seen
        has_seen = valid.any(axis=0)
        # weight by mass at first_seen
        rows = first_seen_idx[has_seen]
        cols = np.arange(N_elem)[has_seen]
        first_seen_mass = np.zeros(N_elem); first_seen_mass[has_seen] = mass[rows, cols]
        emitted_mass = np.bincount(first_seen_idx[has_seen], weights=first_seen_mass[has_seen], minlength=steps)
        mass_dict_1d["mass_emitted_ts"] = emitted_mass * mass_conversion_factor
        mass_dict_1d["mass_emitted_cumulative"] = np.cumsum(mass_dict_1d["mass_emitted_ts"])

        # 4) Mass eliminated
        # Always cumulative in self.result
        attrs_ls_1d = [
            'mass_degraded', 'mass_degraded_water', 'mass_degraded_sediment',
            'mass_volatilized',
            'mass_photodegraded', 'mass_biodegraded',
            'mass_biodegraded_water', 'mass_biodegraded_sediment',
            'mass_hydrolyzed', 'mass_hydrolyzed_water', 'mass_hydrolyzed_sediment'
        ]

        for attr in attrs_ls_1d:
            array2d = extracted_attrs_dict_2d.get(attr)
            if array2d is None:
                mass_eliminated_dict_1d[f"{attr}_ts"] = zeros_ts
                mass_eliminated_dict_1d[f"{attr}_cumulative"] = zeros_ts
                continue
            # 1D per-export-step total increment
            m_ts = self._cumulative_to_deltas_sum_ffill(array2d, chunk_cols=50000)
            # convert units
            m_ts = (m_ts * mass_conversion_factor).astype(np.float32)
            mass_eliminated_dict_1d[f"{attr}_ts"] = m_ts
            mass_eliminated_dict_1d[f"{attr}_cumulative"] = np.cumsum(m_ts, dtype=np.float64).astype(np.float32)

        # 5) Outside/advection
        # compute mass exiting by advection (ignore initial outside at t=0)
        if not adv_out.any():
            mass_eliminated_dict_1d["mass_adv_out_ts"] = np.zeros(steps)
            mass_eliminated_dict_1d["mass_adv_out_cumulative"] = np.zeros(steps)
        else:
            exit_outside_at_t, cum_exit_outside = self.adv_out_mass_ts(
                mass=mass,
                adv_out=adv_out,
                mass_conversion_factor=mass_conversion_factor,
                chunk_cols=50000,)
            mass_eliminated_dict_1d["mass_adv_out_ts"] = exit_outside_at_t.astype(np.float32, copy=False)
            mass_eliminated_dict_1d["mass_adv_out_cumulative"] = cum_exit_outside.astype(np.float32, copy=False)

        # 6) Stranding (priority to adv_out)
        # compute mass exiting by stranding (ignore initial stranded at t=0)
        if stranded_idx is None:
            mass_eliminated_dict_1d["mass_stranded_ts"] = np.zeros(steps)
            mass_eliminated_dict_1d["mass_stranded_cumulative"] = np.zeros(steps)
        else:
            exit_stranded_at_t, cum_exit_stranded = self._event_mass_ts(
                mass=mass,
                event_mask=stranded,
                adv_out=adv_out,
                allow_initial_event=False,      # ignore initial stranded at t=0
                require_inside_now=True,        # must be inside now
                require_inside_prev=True,       # must have been inside previous step too
                chunk_cols=50000,
                mass_conversion_factor=mass_conversion_factor,
            )
            mass_eliminated_dict_1d["mass_stranded_ts"] = exit_stranded_at_t.astype(np.float32, copy=False)
            mass_eliminated_dict_1d["mass_stranded_cumulative"] = cum_exit_stranded.astype(np.float32, copy=False)

        # 7) Burial (multiple events allowed)
        # Count mass each time an element transitions into buried state (sburied): specie id = num_sburied
        # - burial at t=0 is allowed (counts as an event if starts buried)
        # - multiple burial events per element are allowed (buried -> not buried -> buried again)
        # - only count events that occur while inside domain (priority to adv_out)
        if in_buried_sed is None or (not np.asarray(in_buried_sed).any()):
            mass_eliminated_dict_1d["mass_buried_ts"] = np.zeros(steps)
            mass_eliminated_dict_1d["mass_buried_cumulative"] = np.zeros(steps)
        else:
            exit_buried_at_t, cum_buried = self._event_mass_ts(
                mass=mass,
                event_mask=in_buried_sed,
                adv_out=adv_out,
                allow_initial_event=window_starts_at_sim0,  # burial at t=0 allowed
                require_inside_now=True,                    # must be inside now
                require_inside_prev=False,                  # not required for burial
                chunk_cols=50000,
                mass_conversion_factor=mass_conversion_factor,
            )
            mass_eliminated_dict_1d["mass_buried_ts"] = exit_buried_at_t.astype(np.float32, copy=False)
            mass_eliminated_dict_1d["mass_buried_cumulative"] = cum_buried.astype(np.float32, copy=False)

        # 7b) Aggregated specie-transition mass time series
        # Count only when the element is inside the domain both before and after the transition.
        mass_transition_dict_1d["mass_ads_to_sed_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_lmm", None), getattr(self, "num_srev", None)),
                (getattr(self, "num_lmmcation", None), getattr(self, "num_srev", None)),
            ],
            mass=mass,
            specie=specie,
            inside_mask=inside_mask,
            steps=steps,
            N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        mass_transition_dict_1d["mass_des_from_sed_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_srev", None), getattr(self, "num_lmm", None)),
                (getattr(self, "num_srev", None), getattr(self, "num_lmmcation", None)),
            ],
            mass=mass, specie=specie, inside_mask=inside_mask,
            steps=steps, N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        mass_transition_dict_1d["mass_ads_to_spm_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_lmm", None), getattr(self, "num_prev", None)),
                (getattr(self, "num_lmmcation", None), getattr(self, "num_prev", None)),
            ],
            mass=mass, specie=specie, inside_mask=inside_mask,
            steps=steps, N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        mass_transition_dict_1d["mass_des_from_spm_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_prev", None), getattr(self, "num_lmm", None)),
                (getattr(self, "num_prev", None), getattr(self, "num_lmmcation", None)),
            ],
            mass=mass, specie=specie, inside_mask=inside_mask,
            steps=steps, N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        mass_transition_dict_1d["mass_aggr_doc_poly_to_spm_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_humcol", None), getattr(self, "num_prev", None)),
                (getattr(self, "num_polymer", None), getattr(self, "num_prev", None)),
            ],
            mass=mass, specie=specie, inside_mask=inside_mask,
            steps=steps, N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        mass_transition_dict_1d["mass_disaggr_poly_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_polymer", None), getattr(self, "num_lmmanion", None)),
            ],
            mass=mass, specie=specie, inside_mask=inside_mask,
            steps=steps, N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        mass_transition_dict_1d["mass_ads_to_doc_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_lmm", None), getattr(self, "num_humcol", None)),
                (getattr(self, "num_lmmcation", None), getattr(self, "num_humcol", None)),
                (getattr(self, "num_lmmcation", None), getattr(self, "num_polymer", None)),
                (getattr(self, "num_lmmanion", None), getattr(self, "num_polymer", None)),
            ],
            mass=mass, specie=specie, inside_mask=inside_mask,
            steps=steps, N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        mass_transition_dict_1d["mass_des_from_doc_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_humcol", None), getattr(self, "num_lmm", None)),
                (getattr(self, "num_humcol", None), getattr(self, "num_lmmcation", None)),
            ],
            mass=mass, specie=specie, inside_mask=inside_mask,
            steps=steps, N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        mass_transition_dict_1d["mass_dep_to_sed_ts"] = self.sum_transition_pairs(
        pairs=[
            (getattr(self, "num_prev", None), getattr(self, "num_srev", None)),
            (getattr(self, "num_psrev", None), getattr(self, "num_ssrev", None)),
            (getattr(self, "num_pirrev", None), getattr(self, "num_sirrev", None)),
        ],
        mass=mass, specie=specie, inside_mask=inside_mask,
        steps=steps, N_elem=N_elem,
        mass_conversion_factor=mass_conversion_factor,
        chunk_cols=50000)

        mass_transition_dict_1d["mass_res_from_sed_ts"] = self.sum_transition_pairs(
            pairs=[
                (getattr(self, "num_srev", None), getattr(self, "num_prev", None)),
                (getattr(self, "num_ssrev", None), getattr(self, "num_psrev", None)),
                (getattr(self, "num_sirrev", None), getattr(self, "num_pirrev", None)),
            ],
            mass=mass, specie=specie, inside_mask=inside_mask,
            steps=steps, N_elem=N_elem,
            mass_conversion_factor=mass_conversion_factor,
            chunk_cols=50000)

        # 8) Percentage contribution of each elimination term (per time step)
        ts_keys = [k for k in mass_eliminated_dict_1d.keys() if k.endswith("_ts")]

        # total elimination during each ts
        total_elim_ts = np.zeros(steps, dtype=np.float64)
        for k in ["mass_degraded_ts", "mass_volatilized_ts", "mass_adv_out_ts", "mass_stranded_ts", "mass_buried_ts"]:
            total_elim_ts += np.asarray(mass_eliminated_dict_1d[k], dtype=np.float64)
        # compute per-term percentages
        for k in ts_keys:
            term_ts = np.asarray(mass_eliminated_dict_1d[k], dtype=np.float64)
            base = k[:-3]  # drop "_ts"
            perc_key = f"perc_elim_{base}_ts"

            perc = np.divide(term_ts, total_elim_ts,
                out=np.zeros_like(term_ts, dtype=np.float64),
                where=total_elim_ts > 0.0,) * 100.0
            perc_elim_dict_1d[perc_key] = perc.astype(np.float32, copy=False)

        # 9) Extra user-requested fields aggregated over all elements (no masks)
        # For mode="mean", also export a companion count series:
        #   <value_col>__n_valid
        for field, spec in (extra_fields or {}).items():
            if not hasattr(result_ds, field):
                raise ValueError(f"Requested extra field {field!r} not found in self.result")

            da = getattr(result_ds, field)
            if set(da.dims) != {"trajectory", "time"}:
                raise ValueError(
                    f"Requested extra field {field!r} must have dims "
                    f"('trajectory','time') in any order, got {da.dims}")

            spec = {} if spec is None else dict(spec)

            out_name = spec.get("Name", field)
            out_um = spec.get("Unit of measure", spec.get("Unit", getattr(da, "units", "")))
            mode = str(spec.get("mode", "mean")).strip().lower()

            arr2d = da.transpose("time", "trajectory").values  # (T, N)

            col_name = f"{out_name} [{out_um}]" if str(out_um).strip() else out_name

            if mode == "sum":
                ts = np.nansum(arr2d, axis=1).astype(np.float64, copy=False)
                extra_fields_dict_1d[col_name] = ts

            elif mode == "mean":
                valid_count = np.isfinite(arr2d).sum(axis=1).astype(np.float64, copy=False)
                ts_sum = np.nansum(arr2d, axis=1).astype(np.float64, copy=False)

                ts = np.divide(
                    ts_sum,
                    valid_count,
                    out=np.full(arr2d.shape[0], np.nan, dtype=np.float64),
                    where=valid_count > 0,)

                extra_fields_dict_1d[col_name] = ts
                extra_fields_count_dict_1d[f"{col_name}__n_valid"] = valid_count
            else:
                raise ValueError(
                    f"Incorrect mode for extra field {field!r}: {mode!r}. "
                    "Allowed: 'mean', 'sum'")

        # Drop padding row (if used) and recompute window cumulatives
        if pad_rows:
            # slice time axes
            time_steps = time_steps[pad_rows:]
            time_date_serie = time_date_serie[pad_rows:]
            # slice 1D dicts
            for d in (mass_dict_1d, mass_sp_dict_1d, perc_sp_dict_1d, mass_transition_dict_1d,
                      extra_fields_dict_1d, extra_fields_count_dict_1d):
                for k in list(d.keys()):
                    d[k] = np.asarray(d[k])[pad_rows:]

            # emitted cumulative should restart at 0 in the window
            if "mass_emitted_ts" in mass_dict_1d:
                mass_dict_1d["mass_emitted_cumulative"] = np.cumsum(
                    mass_dict_1d["mass_emitted_ts"], dtype=np.float64)
            # slice eliminated ts and recompute eliminated cumulative within the window
            for k in list(mass_eliminated_dict_1d.keys()):
                if k.endswith("_ts"):
                    ts = np.asarray(mass_eliminated_dict_1d[k])[pad_rows:]
                    mass_eliminated_dict_1d[k] = ts.astype(np.float32, copy=False)

                    cum_key = k[:-3] + "_cumulative"
                    if cum_key in mass_eliminated_dict_1d:
                        mass_eliminated_dict_1d[cum_key] = np.cumsum(ts, dtype=np.float64).astype(np.float32)

        # Assemble DataFrame
        # The function builds time-series arrays in *_dict_1d;
        # assemble them into a dataframe for convenience.
        mass_UM = f"[{mass_unit}]"
        time_UM = f"[{time_unit}]"

        # rename columns (dict keys)
        mass_dict              = {f"{k} {mass_UM}": v for k, v in mass_dict_1d.items()}
        mass_eliminated_dict   = {f"{k} {mass_UM}": v for k, v in mass_eliminated_dict_1d.items()}
        mass_sp_dict           = {f"{k} {mass_UM}": v for k, v in mass_sp_dict_1d.items()}
        mass_transition_dict   = {f"{k} {mass_UM}": v for k, v in mass_transition_dict_1d.items()}
        perc_sp_dict           = {f"{k} [%]":       v for k, v in perc_sp_dict_1d.items()}
        perc_elim_dict         = {f"{k} [%]":       v for k, v in perc_elim_dict_1d.items()}

        # UM column:
        # row 0 -> global mass/time units
        # row 1 -> extra field metadata (if any)
        n = len(time_steps)
        extra_fields_info = pd.NA
        if extra_fields:
            parts = []
            for field, spec in extra_fields.items():
                spec = {} if spec is None else dict(spec)
                out_name = spec.get("Name", field)
                out_um = spec.get("Unit of measure", spec.get("Unit", ""))
                mode = str(spec.get("mode", "mean")).strip().lower()

                if str(out_um).strip():
                    msg = f"{field} -> {out_name}: mode={mode}, UM={out_um}"
                else:
                    msg = f"{field} -> {out_name}: mode={mode}"

                if mode == "mean":
                    col_name = f"{out_name} [{out_um}]" if str(out_um).strip() else out_name
                    msg += f", count_col={col_name}__n_valid"

                parts.append(msg)
            extra_fields_info = " ; ".join(parts)

        if n == 0:
            um_col = []
        elif n == 1:
            # only one row available: merge both pieces of info into the first row
            if pd.isna(extra_fields_info):
                um_col = [f"{mass_UM} {time_UM}"]
            else:
                um_col = [f"{mass_UM} {time_UM} | {extra_fields_info}"]
        else:
            um_col = [f"{mass_UM} {time_UM}", extra_fields_info] + [pd.NA] * (n - 2)

        df = pd.DataFrame({
            f"time [{time_unit}]": time_steps,
            "date_of_timestep": time_date_serie,
            **mass_dict,
            **mass_eliminated_dict,
            **mass_transition_dict,
            **extra_fields_dict_1d,
            **extra_fields_count_dict_1d,
            **mass_sp_dict,
            **perc_sp_dict,
            **perc_elim_dict,
            "UM": um_col,
        })

        if save_files:
            df.to_csv(timeseries_file_path, index=False)
        else:
            return df

    ### Helpers for plot_summary_timeseries
    @staticmethod
    def _existing(df, cols):
        return [c for c in cols if c in df.columns]

    @staticmethod
    def _unit_from_col(label):
        import re
        m = re.search(r"\[([^\]]+)\]\s*$", str(label))
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_um_from_df(df):
        """
        Returns (mass_unit, time_unit) without brackets.
        Expects df['UM'] contains something like "[g] [hr]" in first non-null row.
        Fallback: infer from 'time [..]' and first mass col '[..]'.
        """
        import re
        # Primary: UM column
        if "UM" in df.columns:
            s = df["UM"].dropna()
            if len(s):
                txt = str(s.iloc[0])
                units = re.findall(r"\[([^\]]+)\]", txt)
                if len(units) >= 2:
                    mu = units[0].strip()
                    tu = units[1].strip()
                    # normalize time alias
                    tu = "hr" if tu == "h" else tu
                    # normalize micro sign
                    mu = "ug" if mu == "µg" else mu
                    return mu, tu
        # Fallback: time column header
        tu = None
        time_cols = [c for c in df.columns if c.startswith("time [") and c.endswith("]")]
        if time_cols:
            tu = time_cols[0].split("[", 1)[1].rstrip("]").strip()
            tu = "hr" if tu == "h" else tu
        # Fallback: mass column header
        mu = None
        for c in df.columns:
            m = re.search(r"\[([^\]]+)\]\s*$", c)
            if m and "perc_" not in c and "[%]" not in c and not c.startswith("time ["):
                mu = m.group(1).strip()
                mu = "ug" if mu == "µg" else mu
                break

        return mu, tu

    @staticmethod
    def _mass_factor(src, dst):
        _MASS_TO_GRAMS = {
            "ug": 1e-6,
            "µg": 1e-6,
            "mg": 1e-3,
            "g":  1.0,
            "kg": 1e3,
        }
        src = "ug" if src == "µg" else src
        dst = "ug" if dst == "µg" else dst
        if src not in _MASS_TO_GRAMS:
            raise ValueError(f"Unsupported source mass unit: {src!r}")
        if dst not in _MASS_TO_GRAMS:
            raise ValueError(f"Unsupported destination mass unit: {dst!r}")
        return _MASS_TO_GRAMS[src] / _MASS_TO_GRAMS[dst]

    @staticmethod
    def _time_factor(src, dst):
        _TIME_TO_SECONDS = {
            "s":  1.0,
            "m":  60.0,
            "hr": 3600.0,
            "h":  3600.0,
            "d":  86400.0,
        }
        src = "hr" if src == "h" else src
        dst = "hr" if dst == "h" else dst
        if src not in _TIME_TO_SECONDS:
            raise ValueError(f"Unsupported source time unit: {src!r}")
        if dst not in _TIME_TO_SECONDS:
            raise ValueError(f"Unsupported destination time unit: {dst!r}")
        return _TIME_TO_SECONDS[src] / _TIME_TO_SECONDS[dst]

    def convert_units(self, df, target_mass_unit=None, target_time_unit=None, time_col=None):
        """
        Returns: (df_converted, UM_mass, UM_time, time_col_name)
        """
        import pandas as pd
        dfc = df.copy()
        UM_mass, UM_time = self._parse_um_from_df(dfc)

        # determine time column (if not provided)
        if time_col is None:
            time_cols = [c for c in dfc.columns if c.startswith("time [") and c.endswith("]")]
            time_col = time_cols[0] if time_cols else None
        # time conversion
        if target_time_unit and time_col and UM_time:
            dst = "hr" if target_time_unit == "h" else target_time_unit
            f = self._time_factor(UM_time, dst)
            dfc[time_col] = pd.to_numeric(dfc[time_col], errors="coerce") * f
            new_time_col = f"time [{dst}]"
            dfc = dfc.rename(columns={time_col: new_time_col})
            time_col = new_time_col
            UM_time = dst
        # mass conversion
        if target_mass_unit and UM_mass:
            dst = "ug" if target_mass_unit == "µg" else target_mass_unit
            f = self._mass_factor(UM_mass, dst)
            # convert only columns that explicitly carry the source mass unit in their name
            src_tag = f"[{UM_mass}]"
            dst_tag = f"[{dst}]"
            mass_cols = [c for c in dfc.columns if src_tag in c and "[%]" not in c]

            if mass_cols:
                dfc[mass_cols] = dfc[mass_cols].apply(pd.to_numeric, errors="coerce") * f
                rename_map = {c: c.replace(src_tag, dst_tag) for c in mass_cols}
                dfc = dfc.rename(columns=rename_map)

            UM_mass = dst
        # update UM column (first row only)
        if "UM" in dfc.columns:
            dfc["UM"] = pd.NA
            if len(dfc):
                dfc.loc[dfc.index[0], "UM"] = f"[{UM_mass}] [{UM_time}]"

        return dfc, UM_mass, UM_time, time_col

    @staticmethod
    def _strip_units(label):
        import re
        # remove trailing " [something]" (e.g., "mass_x [g]" -> "mass_x")
        return re.sub(r"\s*\[[^\]]+\]\s*$", "", str(label))

    @staticmethod
    def _legend_below(ax, max_cols=6, y_offset=-0.22, fontsize=7,
                      handlelength=2.2, handletextpad=0.6, legend_lw=3.0,
                      box_height=0.22):
        """
        Legend below the subplot, constrained to axis width so layout can reserve space.
        """
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        if not labels:
            return None

        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h

        handles = list(uniq.values())
        labels = list(uniq.keys())
        ncol = min(len(labels), max_cols)

        leg = ax.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(0.0, y_offset, 1.0, box_height),
            mode="expand",
            ncol=ncol,
            frameon=False,
            fontsize=fontsize,
            handlelength=handlelength,
            handletextpad=handletextpad,
            columnspacing=1.0,
            borderaxespad=0.0,
        )

        leg_handles = getattr(leg, "legend_handles", None)
        if leg_handles is None:
            leg_handles = leg.get_lines()
        for h in leg_handles:
            if isinstance(h, Line2D):
                h.set_linewidth(legend_lw)

        return leg

    @staticmethod
    def _is_all_zero(series, atol=0.0):
        """True if series is all zeros (NaN treated as 0)."""
        import numpy as np
        import pandas as pd

        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        return np.all(np.isclose(arr, 0.0, atol=atol))


    def _mix_with_white(self, hex_color: str, amount: float) -> str:
        """
        amount=0 -> original color, amount=1 -> white
        """
        from matplotlib.colors import to_rgb, to_hex
        r, g, b = to_rgb(hex_color)
        r = r + (1 - r) * amount
        g = g + (1 - g) * amount
        b = b + (1 - b) * amount
        return to_hex((r, g, b))

    def _make_shades(self, base_hex: str, n: int, light_amt=(0.70, 0.20)):
        """
        Return n shades of one base color by mixing with white.
        light_amt is (lightest_mix, darkest_mix).
        """
        import numpy as np
        if n <= 0:
            return []
        a0, a1 = light_amt
        return [self._mix_with_white(base_hex, a) for a in np.linspace(a0, a1, n)]

    def _build_explicit_color_map(self, labels):
        """
        Returns dict: label -> color.
        labels must be unit-stripped (what you already put in all_labels).
        """
        import matplotlib.pyplot as plt

        labels = list(dict.fromkeys(labels))  # keep order, unique

        cmap = {}

        # 1) Fixed "main" variables
        fixed = {
            "mass_actual_ts": "#000000",
            "mass_water_ts":  "#0d47a1",
            "mass_sed_ts":    "#c68c1f",

            "mass_emitted_ts":         "#e53935",
            "mass_emitted_cumulative": "#0d47a1",

            "mass_volatilized_ts":         "#d4a62a",
            "mass_volatilized_cumulative": "#8a6d00",

            "mass_adv_out_ts":         "#546e7a",
            "mass_adv_out_cumulative": "#263238",

            "mass_stranded_ts":         "#d6d1c8",
            "mass_stranded_cumulative": "#a8a29a",

            "mass_buried_ts":         "#8d6e63",
            "mass_buried_cumulative": "#4e342e",
        }
        cmap.update(fixed)

        # 2) Process families with water/sed shading
        hydro_base = "#7a0000"  # dark red
        bio_base   = "#2e7d32"  # green
        photo_base = "#fb8c00"  # orange

        # Hydrolysis
        hydro_w = self._mix_with_white(hydro_base, 0.70)
        hydro_s = self._mix_with_white(hydro_base, 0.50)
        for k in ("mass_hydrolyzed_ts", "mass_hydrolyzed_cumulative"):
            cmap[k] = hydro_base
        for k in ("mass_hydrolyzed_water_ts", "mass_hydrolyzed_water_cumulative"):
            cmap[k] = hydro_w
        for k in ("mass_hydrolyzed_sediment_ts", "mass_hydrolyzed_sediment_cumulative"):
            cmap[k] = hydro_s
        # bar labels (used in bar_elim_summary legend)
        cmap["hydrolyzed water"] = hydro_w
        cmap["hydrolyzed sediment"] = hydro_s
        cmap["hydrolyzed"] = hydro_base

        # Biodegradation
        bio_w = self._mix_with_white(bio_base, 0.65)
        bio_s = self._mix_with_white(bio_base, 0.45)
        for k in ("mass_biodegraded_ts", "mass_biodegraded_cumulative"):
            cmap[k] = bio_base
        for k in ("mass_biodegraded_water_ts", "mass_biodegraded_water_cumulative"):
            cmap[k] = bio_w
        for k in ("mass_biodegraded_sediment_ts", "mass_biodegraded_sediment_cumulative"):
            cmap[k] = bio_s
        cmap["biodegraded water"] = bio_w
        cmap["biodegraded sediment"] = bio_s
        cmap["biodegraded"] = bio_base

        # Photodegradation
        for k in ("mass_photodegraded_ts", "mass_photodegraded_cumulative"):
            cmap[k] = photo_base
        cmap["photodegraded"] = photo_base

        # "total degraded"
        deg_base = "#4a148c"
        deg_w = self._mix_with_white(deg_base, 0.60)
        deg_s = self._mix_with_white(deg_base, 0.40)
        for k in ("mass_degraded_ts", "mass_degraded_cumulative"):
            cmap[k] = deg_base
        for k in ("mass_degraded_water_ts", "mass_degraded_water_cumulative"):
            cmap[k] = deg_w
        for k in ("mass_degraded_sediment_ts", "mass_degraded_sediment_cumulative"):
            cmap[k] = deg_s
        cmap["degraded water"] = deg_w
        cmap["degraded sediment"] = deg_s
        cmap["degraded"] = deg_base

        # Volatilization / advection / stranded bar labels
        cmap["volatilized"]  = fixed["mass_volatilized_cumulative"]
        cmap["advected out"] = fixed["mass_adv_out_cumulative"]
        cmap["stranded"]     = fixed["mass_stranded_cumulative"]

        # 3) Speciation palettes by topic
        #    (mass_sp_* and perc_sp_* share same colors)
        # Canonical SP keys are like: sp_dissolved_ts, sp_spm_rev_ts, sp_sed_irrev_ts, etc.
        sp_keys = set()
        for lab in labels:
            if lab.startswith("mass_sp_"):
                sp_keys.add(lab[len("mass_"):])
            elif lab.startswith("perc_sp_"):
                sp_keys.add(lab[len("perc_"):])

        blue_base  = "#1565c0"
        jade_base  = "#009688"
        brown_base = "#a66a2c"
        gray_base  = "#607d8b"

        def assign_sp_group_fixed(order, base_hex, mix_amounts):
            for i, k in enumerate(order):
                if k not in sp_keys:
                    continue
                amt = mix_amounts[i] if i < len(mix_amounts) else mix_amounts[-1]
                col = self._mix_with_white(base_hex, amt)
                cmap["mass_" + k] = col
                cmap["perc_" + k] = col

        diss_mix = [0.78, 0.62, 0.48, 0.36, 0.26, 0.18]
        spm_mix  = [0.70, 0.48, 0.28]
        sed_mix  = [0.78, 0.58, 0.38, 0.22]

        dissolved_order = [
            "sp_dissolved_ts", "sp_dissolved_anion_ts", "sp_dissolved_cation_ts",
            "sp_doc_ts", "sp_colloid_ts", "sp_polymer_ts",
        ]
        spm_order = ["sp_spm_rev_ts", "sp_spm_srev_ts", "sp_spm_irrev_ts"]
        sed_order = ["sp_sed_rev_ts", "sp_sed_srev_ts", "sp_sed_irrev_ts", "sp_sed_buried_ts"]

        assign_sp_group_fixed(dissolved_order, blue_base,  diss_mix)
        assign_sp_group_fixed(spm_order,      jade_base,  spm_mix)
        assign_sp_group_fixed(sed_order,      brown_base, sed_mix)

        other_sp_order = sorted([k for k in sp_keys if k not in set(dissolved_order + spm_order + sed_order)])
        other_mix = [0.75, 0.60, 0.48, 0.36, 0.26, 0.18, 0.10]
        assign_sp_group_fixed(other_sp_order, gray_base, other_mix)

        # 4) Speciation palettes by topic
        transition_fixed = {
            "mass_ads_to_sed_ts": "#8d6e63",
            "mass_des_from_sed_ts": "#bcaaa4",

            "mass_ads_to_spm_ts": "#00897b",
            "mass_des_from_spm_ts": "#80cbc4",

            "mass_dep_to_sed_ts": "#6d4c41",
            "mass_res_from_sed_ts": "#a1887f",

            "mass_ads_to_doc_ts": "#5e35b1",
            "mass_des_from_doc_ts": "#b39ddb",

            "mass_aggr_doc_poly_to_spm_ts": "#ef6c00",
            "mass_disaggr_poly_ts": "#ffb74d",
        }
        cmap.update(transition_fixed)

        # 5) Fallback for anything not covered
        # Use 60-color palette; deterministic assignment for remaining labels
        fallback_palette = (
            list(plt.get_cmap("tab20").colors) +
            list(plt.get_cmap("tab20b").colors) +
            list(plt.get_cmap("tab20c").colors)
        )
        fallback_idx = 0
        for lab in sorted(set(labels)):
            if lab not in cmap:
                if fallback_idx >= len(fallback_palette):
                    fallback_idx = 0
                cmap[lab] = fallback_palette[fallback_idx]
                fallback_idx += 1

        return cmap

    def _pretty_label(self, key: str) -> str:
        """
        Convert unit-stripped internal keys (e.g. 'mass_sp_spm_srev_ts') to a human-readable label.
        Colors should still be keyed by the original key, not this string.
        """
        k = str(key)

        # Bar-summary labels (used in bar_elim_summary legend)
        bar_map = {
            "degraded water": "Degraded (wat)",
            "degraded sediment": "Degraded (sed)",
            "volatilized": "Volatilized",
            "biodegraded water": "Biodegraded (wat)",
            "biodegraded sediment": "Biodegraded (sed)",
            "hydrolyzed water": "Hydrolyzed (wat)",
            "hydrolyzed sediment": "Hydrolyzed (sed)",
            "photodegraded": "Photodegraded",
        }
        if k in bar_map:
            return bar_map[k]

        # Speciation keys: mass_sp_* and perc_sp_*
        if k.startswith("mass_sp_") or k.startswith("perc_sp_"):
            # strip mass_/perc_
            sp = k.split("_", 1)[1]
            # strip trailing "_ts" if present (all your species are ts)
            if sp.endswith("_ts"):
                sp = sp[:-3]

            sp_map = {
                "sp_dissolved": "Dissolved",
                "sp_dissolved_anion": "Dissolved (anion)",
                "sp_dissolved_cation": "Dissolved (cation)",
                "sp_doc": "DOC",
                "sp_colloid": "Colloid",
                "sp_polymer": "Polymer",

                "sp_spm_rev": "SPM (rev)",
                "sp_spm_srev": "SPM (slow rev)",
                "sp_spm_irrev": "SPM (irrev)",

                "sp_sed_rev": "Sed (rev)",
                "sp_sed_srev": "Sed (slow rev)",
                "sp_sed_irrev": "Sed (irrev)",
                "sp_sed_buried": "Sed (buried)",
            }
            return sp_map.get(sp, sp.replace("_", " ").replace("sp ", "").title())

        #
        transition_map = {
            "mass_ads_to_sed_ts": "Ads. to sed (ts)",
            "mass_des_from_sed_ts": "Des. from sed (ts)",

            "mass_ads_to_spm_ts": "Ads. to SPM (ts)",
            "mass_des_from_spm_ts": "Des. from SPM (ts)",

            "mass_dep_to_sed_ts": "Dep. to sed (ts)",
            "mass_res_from_sed_ts": "Resusp. from sed (ts)",

            "mass_ads_to_doc_ts": "Ads. to DOC/polymer (ts)",
            "mass_des_from_doc_ts": "Des. from DOC/polymer (ts)",

            "mass_aggr_doc_poly_to_spm_ts": "Aggr. to SPM (ts)",
            "mass_disaggr_poly_ts": "Disaggr. from polymer (ts)",
        }
        if k in transition_map:
            return transition_map[k]

        # Other mass keys: merge time + compartment into one suffix like "(ts-wat)"
        time_tag = None
        base = k

        if k.endswith("_ts"):
            time_tag = "ts"
            base = k[:-3]
        elif k.endswith("_cumulative"):
            time_tag = "cumul"
            base = k[: -len("_cumulative")]

        # detect compartment tags in the base name
        comp_tag = None
        if base.endswith("_water"):
            comp_tag = "wat"
            base = base[:-len("_water")]
        elif base.endswith("_sediment"):
            comp_tag = "sed"
            base = base[:-len("_sediment")]

        base_map = {
            "mass_actual": "Actual mass",
            "mass_water": "Water mass",
            "mass_sed": "Sediment mass",

            "mass_emitted": "Emitted",
            "mass_volatilized": "Volatilized",
            "mass_adv_out": "Advection out",
            "mass_stranded": "Stranded",
            "mass_buried": "Buried",

            "mass_degraded": "Degraded",
            "mass_hydrolyzed": "Hydrolyzed",
            "mass_biodegraded": "Biodegraded",
            "mass_photodegraded": "Photodegraded",
        }

        nice = base_map.get(base, base.replace("_", " ").title())

        parts = []
        if time_tag:
            parts.append(time_tag)
        if comp_tag:
            parts.append(comp_tag)

        suffix = f" ({'-'.join(parts)})" if parts else ""
        return nice + suffix

    def plot_summary_timeseries(
        self, df=None,
        timeseries_file_path=None,
        target_mass_unit=None,
        target_time_unit=None,
        use_date=False,
        time_col=None,
        start_date=None,
        end_date=None,
        fig_title=None,
        font_sizes=None,
        font_scale=1.0,
        split_a4=False,
        pdf_path=None,
        row_mode="double"
    ):
        """
            Plot summary time-series panels for mass balance, elimination pathways, and speciation.

            df:                     pd.DataFrame, Input dataframe containing the time-series results.
                                    If None, `timeseries_file_path` must be provided.
            timeseries_file_path:   str or path-like, Path to a CSV file to read when `df` is None.
            target_mass_unit:       str, Target mass unit for conversion ("ug", "mg", "g", "kg").
            target_time_unit:       str, Target time unit for conversion ("s", "m", "hr", "d").
            use_date:               bool, If True and the column `date_of_timestep` exists,
                                    use it as x-axis. Otherwise use the numeric time column.
            time_col:               str, Name of the time column to use
                                    (for example: `time [hr]`).
                                    If None, it is inferred automatically.
            start_date:             date-like str, Start of the period to include in the figure. If `use_date=True`, interpreted as a date,
                                     otherwise interpreted as a numeric lower time bound.
            end_date:               date-like str, End of the period to include in the figure. If `use_date=True`, interpreted as a date,
                                     otherwise interpreted as a numeric upper time bound.
            fig_title:              str, Global figure title.
            font_sizes:             dict, Explicit font-size overrides.
            font_scale:             float, Global multiplicative scale applied to default
                                    font sizes before `font_sizes` overrides are applied.
            split_a4:               bool, If False, If True, split the output into multiple A4 portrait pages
                                    grouped by topic.
            pdf_path:               str or path-like, Output PDF path used only when `split_a4=True`. If provided, all pages are written
                                    to a multi-page PDF.
            row_mode:               {"double", "single"}, Panel layout mode.
                                    `"double"`: two panels per row, `"single"`: one wide panel per row
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.ticker import FuncFormatter
        import numpy as np
        import pandas as pd

        if row_mode not in ("double", "single"):
            raise ValueError(f"Unsupported row_mode: {row_mode!r}. Use 'double' or 'single'.")

        if df is None:
            if timeseries_file_path is not None:
                df = pd.read_csv(timeseries_file_path)
            else:
                raise ValueError("Both df and timeseries_file_path are None")

        # font sizes (defaults)
        fs = {
            "legend": 9,
            "xlabel": 9,
            "ylabel": 9,
            "xticks": 9,
            "yticks": 9,
            "subplot_title": 10,
            "figure_title": 11,
        }
        # apply global scale first
        if font_scale is None:
            font_scale = 1.0
        for k in fs:
            fs[k] = fs[k] * float(font_scale)

        # then apply explicit overrides (treated as absolute)
        if font_sizes:
            fs.update(font_sizes)

        # Convert units + get resulting units and time col name
        dfc, UM_mass, UM_time, time_col = self.convert_units(
            df, target_mass_unit=target_mass_unit, target_time_unit=target_time_unit, time_col=time_col
        )

        # choose x axis + optional period filter
        use_date_axis = bool(use_date and "date_of_timestep" in dfc.columns)

        if use_date_axis:
            x_raw = pd.to_datetime(dfc["date_of_timestep"], errors="coerce")
            xlab = "date_of_timestep"
        else:
            if time_col is None:
                raise ValueError("No time column found (expected something like 'time [hr]').")
            x_raw = pd.to_numeric(dfc[time_col], errors="coerce")
            xlab = time_col

        def _fmt_period_val(v, is_date_axis):
            if pd.isna(v):
                return "NaT" if is_date_axis else "NaN"
            if is_date_axis:
                try:
                    return pd.Timestamp(v).strftime("%Y-%m-%d")
                except Exception:
                    return str(v)
            try:
                return f"{float(v):g}"
            except Exception:
                return str(v)

        mask = pd.Series(True, index=dfc.index)

        if use_date_axis:
            if start_date is not None:
                start_bound = pd.to_datetime(start_date, errors="coerce")
                if pd.isna(start_bound):
                    raise ValueError(f"Could not parse start_date={start_date!r} as a date.")
                mask &= (x_raw >= start_bound)

            if end_date is not None:
                end_bound = pd.to_datetime(end_date, errors="coerce")
                if pd.isna(end_bound):
                    raise ValueError(f"Could not parse end_date={end_date!r} as a date.")
                mask &= (x_raw <= end_bound)
        else:
            if start_date is not None:
                try:
                    start_bound = float(start_date)
                except Exception:
                    raise ValueError(f"Could not parse start_date={start_date!r} as a numeric time bound.")
                mask &= (x_raw >= start_bound)

            if end_date is not None:
                try:
                    end_bound = float(end_date)
                except Exception:
                    raise ValueError(f"Could not parse end_date={end_date!r} as a numeric time bound.")
                mask &= (x_raw <= end_bound)

        if start_date is not None or end_date is not None:
            dfc = dfc.loc[mask].copy()
            x_raw = x_raw.loc[mask]
            if len(dfc) == 0:
                raise ValueError("No rows remain after applying start_date/end_date filter.")

        x = x_raw

        period_suffix = ""
        if start_date is not None or end_date is not None:
            x_start = _fmt_period_val(x.iloc[0], use_date_axis)
            x_end = _fmt_period_val(x.iloc[-1], use_date_axis)
            if use_date_axis:
                period_suffix = f" | period: {x_start} to {x_end}"
            else:
                period_suffix = f" | period: {x_start} to {x_end} ({xlab})"

        # name helpers (after conversion)
        def mcol(base):  # mass column with unit
            return f"{base} [{UM_mass}]"

        def pcol(base):  # percent col
            return f"{base} [%]"

        # scientific formatter
        SCI_THRESHOLD = 1000.0

        def _fmt_big_sci(v, pos=None):
            try:
                v = float(v)
            except Exception:
                return ""
            if v == 0.0:
                return "0"
            if abs(v) >= SCI_THRESHOLD:
                exp = int(np.floor(np.log10(abs(v))))
                mant = v / (10 ** exp)
                return f"{mant:.2f}e{exp}"
            return f"{v:g}"

        # xlabel strip helper
        def _place_xlabel_in_axis(ax_xlab, text):
            ax_xlab.set_xticks([])
            ax_xlab.set_yticks([])
            for sp in ax_xlab.spines.values():
                sp.set_visible(False)
            ax_xlab.set_facecolor("none")
            ax_xlab.text(
                0.5, 0.25, str(text),
                ha="center", va="center",
                fontsize=fs["xlabel"],
                transform=ax_xlab.transAxes
            )

        # All species masses at each timestep (after conversion)
        sp_key_order = [
            # dissolved
            "sp_dissolved_ts",
            "sp_dissolved_anion_ts",
            "sp_dissolved_cation_ts",
            "sp_doc_ts",
            "sp_colloid_ts",
            "sp_polymer_ts",
            # SPM
            "sp_spm_rev_ts",
            "sp_spm_srev_ts",
            "sp_spm_irrev_ts",
            # sediment
            "sp_sed_rev_ts",
            "sp_sed_srev_ts",
            "sp_sed_irrev_ts",
            "sp_sed_buried_ts",
        ]

        # Build ordered mass_sp columns (after unit conversion)
        sp_mass_cols = []
        for k in sp_key_order:
            col = f"mass_{k} [{UM_mass}]"
            if col in dfc.columns:
                sp_mass_cols.append(col)

        # Add any remaining mass_sp_*_ts columns not covered above
        extra_sp = [
            c for c in dfc.columns
            if c.startswith("mass_sp_") and c.endswith(f"_ts [{UM_mass}]") and c not in sp_mass_cols
        ]
        sp_mass_cols.extend(sorted(extra_sp))

        # panel specs
        panels = []

        # Row A1
        panels += [
            dict(title="Mass in system (ts)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_water_ts"), mcol("mass_sed_ts"), mcol("mass_actual_ts")]),
                 legend={"max_cols": 1}),
            dict(title="Emissions", kind="line",
                 cols=self._existing(dfc, [mcol("mass_emitted_ts"), mcol("mass_emitted_cumulative")]),
                 legend={"max_cols": 1}),
        ]

        # Row A2
        panels += [
            dict(title="Total mass removed (timestep)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_degraded_ts"), mcol("mass_adv_out_ts"), mcol("mass_volatilized_ts"),
                                           mcol("mass_stranded_ts"), mcol("mass_buried_ts")]),
                 legend={"max_cols": 1}),
            dict(title="Total mass removed (ts) – contribution (%)", kind="stack_share",
                 cols=self._existing(dfc, [mcol("mass_degraded_ts"), mcol("mass_adv_out_ts"), mcol("mass_volatilized_ts"),
                                           mcol("mass_stranded_ts"), mcol("mass_buried_ts")]),
                 legend={"max_cols": 1},
                 stack_style="bars", show_empty_outline=True),
        ]

        # Row B1 total degraded
        panels += [
            dict(title="Total degraded (ts + water/sed)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_degraded_ts"), mcol("mass_degraded_water_ts"), mcol("mass_degraded_sediment_ts")]),
                 legend={"max_cols": 1}),
            dict(title="Total degraded (ts) – water vs sediment (%)", kind="stack_share",
                 cols=self._existing(dfc, [mcol("mass_degraded_water_ts"), mcol("mass_degraded_sediment_ts")]),
                 legend={"max_cols": 1},
                 stack_style="bars", show_empty_outline=True),
        ]

        # Row B2
        panels += [
            dict(title="Volatilized (ts + cumulative)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_volatilized_ts"), mcol("mass_volatilized_cumulative")]),
                 legend={"max_cols": 1}),
            dict(title="Advected out (ts + cumulative)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_adv_out_ts"), mcol("mass_adv_out_cumulative")]),
                 legend={"max_cols": 1}),
        ]

        # Row B3
        panels += [
            dict(title="Stranded (ts + cumulative)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_stranded_ts"), mcol("mass_stranded_cumulative")]),
                 legend={"max_cols": 1}),
            dict(title="Buried (ts + cumulative)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_buried_ts"), mcol("mass_buried_cumulative")]),
                 legend={"max_cols": 1}),
        ]

        # Row C1 biodegradation
        panels += [
            dict(title="Biodegradation (ts + water/sed)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_biodegraded_ts"), mcol("mass_biodegraded_water_ts"), mcol("mass_biodegraded_sediment_ts")]),
                 legend={"max_cols": 1}),
            dict(title="Biodegradation (ts) – water vs sediment (%)", kind="stack_share",
                 cols=self._existing(dfc, [mcol("mass_biodegraded_water_ts"), mcol("mass_biodegraded_sediment_ts")]),
                 legend={"max_cols": 1},
                 stack_style="bars", show_empty_outline=True),
        ]

        # Row C2 hydrolysis
        panels += [
            dict(title="Hydrolysis (ts + water/sed)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_hydrolyzed_ts"), mcol("mass_hydrolyzed_water_ts"), mcol("mass_hydrolyzed_sediment_ts")]),
                 legend={"max_cols": 1}),
            dict(title="Hydrolysis (ts) – water vs sediment (%)", kind="stack_share",
                 cols=self._existing(dfc, [mcol("mass_hydrolyzed_water_ts"), mcol("mass_hydrolyzed_sediment_ts")]),
                 legend={"max_cols": 1},
                 stack_style="bars", show_empty_outline=True),
        ]

        # Photodegradation
        panels += [
            dict(title="Photodegradation (ts)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_photodegraded_ts")]),
                 legend={"max_cols": 1}),
            dict(title="Cumulative eliminated masses (final)", kind="bar_elim_summary",
                 cols=[], legend={"max_cols": 2}),
        ]

        # Transition processes: adsorption / desorption / deposition / resuspension / aggregation
        if self._existing(dfc, [mcol("mass_ads_to_spm_ts"), mcol("mass_des_from_spm_ts")]):
            panels += [
                dict(title="Adsorption to suspended particles", kind="line",
                     cols=self._existing(dfc, [mcol("mass_ads_to_spm_ts")]),
                     legend={"max_cols": 1}),
                dict(title="Desorption from suspended particles", kind="line",
                     cols=self._existing(dfc, [mcol("mass_des_from_spm_ts")]),
                     legend={"max_cols": 1}),
            ]

        if self._existing(dfc, [mcol("mass_ads_to_sed_ts"), mcol("mass_des_from_sed_ts")]):
            panels += [
                dict(title="Adsorption to sediments", kind="line",
                     cols=self._existing(dfc, [mcol("mass_ads_to_sed_ts")]),
                     legend={"max_cols": 1}),
                dict(title="Desorption from sediments", kind="line",
                     cols=self._existing(dfc, [mcol("mass_des_from_sed_ts")]),
                     legend={"max_cols": 1}),
            ]

        if self._existing(dfc, [mcol("mass_ads_to_doc_ts"), mcol("mass_des_from_doc_ts")]):
            panels += [
                dict(title="Adsorption to DOC / polymer", kind="line",
                     cols=self._existing(dfc, [mcol("mass_ads_to_doc_ts")]),
                     legend={"max_cols": 1}),
                dict(title="Desorption from DOC / polymer", kind="line",
                     cols=self._existing(dfc, [mcol("mass_des_from_doc_ts")]),
                     legend={"max_cols": 1}),
            ]

        if self._existing(dfc, [mcol("mass_dep_to_sed_ts"), mcol("mass_res_from_sed_ts")]):
            panels += [
                dict(title="Deposition to sediments", kind="line",
                     cols=self._existing(dfc, [mcol("mass_dep_to_sed_ts")]),
                     legend={"max_cols": 1}),
                dict(title="Resuspension from sediments", kind="line",
                     cols=self._existing(dfc, [mcol("mass_res_from_sed_ts")]),
                     legend={"max_cols": 1}),
            ]

        if self._existing(dfc, [mcol("mass_aggr_doc_poly_to_spm_ts"), mcol("mass_disaggr_poly_ts")]):
            panels += [
                dict(title="Aggregation to suspended particles", kind="line",
                     cols=self._existing(dfc, [mcol("mass_aggr_doc_poly_to_spm_ts")]),
                     legend={"max_cols": 1}),
                dict(title="Disaggregation from polymer", kind="line",
                     cols=self._existing(dfc, [mcol("mass_disaggr_poly_ts")]),
                     legend={"max_cols": 1}),
            ]

        # Speciation: mass and percent stacked bars
        panels += [
            dict(title="All species masses (stacked, ts)", kind="stack_mass", cols=sp_mass_cols,
                 legend={"max_cols": 2}, stack_style="bars", show_empty_outline=True),
            dict(title="All species masses – contribution (%)", kind="stack_share", cols=sp_mass_cols,
                 legend={"max_cols": 2}, stack_style="bars", show_empty_outline=True),
        ]

        # Speciation: mass lines vs percent stacked
        panels += [
            dict(title="SP masses (dissolved-phase)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_sp_dissolved_ts"), mcol("mass_sp_dissolved_anion_ts"),
                                           mcol("mass_sp_dissolved_cation_ts"), mcol("mass_sp_doc_ts"),
                                           mcol("mass_sp_colloid_ts"), mcol("mass_sp_polymer_ts")]),
                 legend={"max_cols": 2}),
            dict(title="SP composition (%) (dissolved-phase)", kind="stack_share",
                 cols=self._existing(dfc, [pcol("perc_sp_dissolved_ts"), pcol("perc_sp_dissolved_anion_ts"),
                                           pcol("perc_sp_dissolved_cation_ts"), pcol("perc_sp_doc_ts"),
                                           pcol("perc_sp_colloid_ts"), pcol("perc_sp_polymer_ts")]),
                 legend={"max_cols": 2},
                 stack_style="bars", show_empty_outline=True),

            dict(title="SP masses (SPM partition)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_sp_spm_rev_ts"), mcol("mass_sp_spm_srev_ts"), mcol("mass_sp_spm_irrev_ts")]),
                 legend={"max_cols": 2}),
            dict(title="SP composition (%) (SPM partition)", kind="stack_share",
                 cols=self._existing(dfc, [pcol("perc_sp_spm_rev_ts"), pcol("perc_sp_spm_srev_ts"), pcol("perc_sp_spm_irrev_ts")]),
                 legend={"max_cols": 2},
                 stack_style="bars", show_empty_outline=True),

            dict(title="SP masses (sediment partition)", kind="line",
                 cols=self._existing(dfc, [mcol("mass_sp_sed_rev_ts"), mcol("mass_sp_sed_srev_ts"), mcol("mass_sp_sed_irrev_ts")]),
                 legend={"max_cols": 1}),
            dict(title="SP composition (%) (sediment partition)", kind="stack_share",
                 cols=self._existing(dfc, [pcol("perc_sp_sed_rev_ts"), pcol("perc_sp_sed_srev_ts"), pcol("perc_sp_sed_irrev_ts")]),
                 legend={"max_cols": 1},
                 stack_style="bars", show_empty_outline=True),
        ]

        # Extra user-defined fields: dedicated section before mass-budget checks
        extra_panel_start_idx = None

        extra_line_cols = []
        skip_exact = {"UM", "date_of_timestep"}
        if time_col is not None:
            skip_exact.add(time_col)

        for c in dfc.columns:
            if c in skip_exact:
                continue
            if c.startswith("time [") and c.endswith("]"):
                continue
            if c.startswith("mass_"):
                continue
            if c.startswith("perc_"):
                continue
            if c.startswith("qc_"):
                continue
            if c.endswith("__n_valid"):
                continue
            if not pd.api.types.is_numeric_dtype(dfc[c]):
                continue

            s = pd.to_numeric(dfc[c], errors="coerce")
            if s.notna().sum() == 0:
                continue

            extra_line_cols.append(c)

        if extra_line_cols:
            # force "Additional fields" to start on a fresh row in double mode
            if row_mode == "double" and (len(panels) % 2 == 1):
                panels.append(dict(title="", kind="blank", cols=[], legend={"max_cols": 1}))

            extra_panel_start_idx = len(panels)

            for col in extra_line_cols:
                unit = self._unit_from_col(col)
                ylabel = f"Value [{unit}]" if unit else "Value"
                panels.append(
                    dict(
                        title=self._strip_units(col),
                        kind="line",
                        cols=[col],
                        ylabel=ylabel,
                        legend={"max_cols": 1},))

            # force following sanity section to start on a fresh row too
            if row_mode == "double" and (len(panels) % 2 == 1):
                panels.append(dict(title="", kind="blank", cols=[], legend={"max_cols": 1}))

        # Mass budget checks
        has_ts = all(c in dfc.columns for c in [mcol("mass_degraded_ts"), mcol("mass_photodegraded_ts"),
                                               mcol("mass_biodegraded_ts"), mcol("mass_hydrolyzed_ts")])
        has_cum = all(c in dfc.columns for c in [mcol("mass_degraded_cumulative"), mcol("mass_photodegraded_cumulative"),
                                                mcol("mass_biodegraded_cumulative"), mcol("mass_hydrolyzed_cumulative")])
        if has_ts:
            panels += [dict(title="Sanity (ts): total degraded vs sum of pathways", kind="sanity_ts", cols=[], legend={"max_cols": 1})]
        if has_cum:
            panels += [dict(title="Sanity (cumulative): total degraded vs sum of pathways", kind="sanity_cum", cols=[], legend={"max_cols": 1})]

        # Build layout rows
        if row_mode == "double":
            if len(panels) % 2 == 1:
                panels.append(dict(title="", kind="blank", cols=[], legend={"max_cols": 1}))
            rows = [panels[i:i + 2] for i in range(0, len(panels), 2)]
        else:
            rows = [[p] for p in panels]

        i_extra = None
        if extra_panel_start_idx is not None:
            i_extra = (extra_panel_start_idx // 2) if row_mode == "double" else extra_panel_start_idx

        # Build a global label list so colors stay consistent even if some series are omitted in a panel
        all_labels = []
        for spec in panels:
            for col in spec.get("cols", []) or []:
                all_labels.append(self._strip_units(col))

        # add bar-segment labels
        all_labels += [
            "degraded water", "degraded sediment", "volatilized",
            "biodegraded water", "biodegraded sediment",
            "hydrolyzed water", "hydrolyzed sediment",
            "photodegraded",]

        color_map = self._build_explicit_color_map(all_labels)

        # Pagination + rendering on A4 portrait with fixed panel sizes
        A4_PORTRAIT = (8.27, 11.69)  # inches
        ROWS_PER_PAGE = 2 if row_mode == "double" else 4

        def _place_legend_in_axis(ax_plot, ax_leg, max_cols, fontsize, legend_lw=3.0):
            """Put legend into a dedicated axis below the xlabel strip (never overlaps)."""
            from matplotlib.lines import Line2D

            handles, labels = ax_plot.get_legend_handles_labels()
            if not labels:
                ax_leg.axis("off")
                return None

            # de-duplicate while preserving order
            uniq = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h

            handles = list(uniq.values())
            labels = list(uniq.keys())
            ncol = min(len(labels), max_cols) if max_cols else len(labels)

            ax_leg.axis("off")
            leg = ax_leg.legend(
                handles, labels,
                loc="upper center",
                ncol=ncol,
                frameon=False,
                fontsize=fontsize,
                handlelength=2.2,
                handletextpad=0.85,
                columnspacing=1.2,
                borderaxespad=0.0,)

            # Thicken legend line samples
            for h in leg.get_lines():
                if isinstance(h, Line2D):
                    h.set_linewidth(legend_lw)

            return leg

        def _set_ylabel(ax, kind, cols, ylabel=None):
            cols = cols or []

            if ylabel:
                ax.set_ylabel(str(ylabel), fontsize=fs["ylabel"], labelpad=2)
                ax.tick_params(axis="x", labelsize=fs["xticks"])
                ax.tick_params(axis="y", labelsize=fs["yticks"])
                return

            is_percent_axis = (
                kind in ("stack_share", "stack100", "stack_to_total")
                or any("[%]" in c for c in cols))

            if is_percent_axis:
                ax.set_ylabel("Percentage [%]", fontsize=fs["ylabel"], labelpad=2)
            else:
                ax.set_ylabel(f"Mass [{UM_mass}]", fontsize=fs["ylabel"], labelpad=2)
                ax.yaxis.set_major_formatter(FuncFormatter(_fmt_big_sci))

            ax.tick_params(axis="x", labelsize=fs["xticks"])
            ax.tick_params(axis="y", labelsize=fs["yticks"])

        def _plot_one_panel(ax, spec):
            """Plot into ax only. xlabel + legend are handled in separate strip axes."""
            kind = spec.get("kind", "blank")
            cols = spec.get("cols", []) or []
            title = spec.get("title", "")

            if kind == "blank":
                ax.axis("off")
                return

            ax.set_title(title, fontsize=fs["subplot_title"])

            if kind == "line":
                for col in cols:
                    if col not in dfc.columns:
                        continue
                    if self._is_all_zero(dfc[col], atol=1e-12):
                        continue
                    key = self._strip_units(col)
                    if key.startswith(("mass_", "perc_")):
                        pretty = self._pretty_label(key)
                    else:
                        pretty = key
                    ax.plot(
                        x, dfc[col],
                        label=pretty,
                        color=color_map.get(key, None),
                        linewidth=1.8)

            elif kind == "stack100":
                keep = [col for col in cols if col in dfc.columns and not self._is_all_zero(dfc[col], atol=1e-12)]
                if keep:
                    y = np.nan_to_num(dfc[keep].to_numpy(dtype=float), nan=0.0)
                    keys = [self._strip_units(c) for c in keep]
                    labels = [self._pretty_label(k) for k in keys]
                    colors = [color_map.get(k, None) for k in keys]
                    ax.stackplot(x, y.T, labels=labels, colors=colors)
                    ax.set_ylim(0, 100)

            elif kind == "stack_share":
                keep = [col for col in cols if col in dfc.columns and not self._is_all_zero(dfc[col], atol=1e-12)]
                if keep:
                    y = np.nan_to_num(dfc[keep].to_numpy(dtype=float), nan=0.0)
                    denom = y.sum(axis=1, keepdims=True)
                    shares = np.divide(y, denom, out=np.zeros_like(y), where=denom != 0) * 100.0

                    keys = [self._strip_units(c) for c in keep]
                    labels = [self._pretty_label(k) for k in keys]
                    colors = [color_map.get(k, None) for k in keys]

                    stack_style = spec.get("stack_style", "area")
                    show_empty_outline = bool(spec.get("show_empty_outline", False))

                    if stack_style == "bars":
                        xpos = np.arange(len(x))
                        width = 1.0

                        if show_empty_outline:
                            ax.bar(
                                xpos, np.full(len(xpos), 100.0),
                                width=width, color="none",
                                edgecolor="0.85", linewidth=0.6, zorder=0)

                        bottom = np.zeros(len(xpos), dtype=float)
                        for lab, colr, j in zip(labels, colors, range(len(labels))):
                            h = shares[:, j]
                            if np.all(np.isclose(h, 0.0)):
                                continue
                            ax.bar(xpos, h, width=width, bottom=bottom, label=lab, color=colr, align="center")
                            bottom += h

                        ax.set_ylim(0, 100)
                        ax.set_xlim(-0.5, len(xpos) - 0.5)

                        n = len(xpos)
                        if n <= 30:
                            ax.set_xticks(xpos)
                            ax.set_xticklabels(
                                [f"{v:g}" if isinstance(v, (float, np.floating)) else str(v) for v in x],
                                rotation=0)
                        else:
                            step = max(1, n // 12)
                            idx = np.arange(0, n, step)
                            ax.set_xticks(idx)
                            xvals = x.iloc[idx] if hasattr(x, "iloc") else [x[i] for i in idx]
                            ax.set_xticklabels(
                                [f"{v:g}" if isinstance(v, (float, np.floating)) else str(v) for v in xvals],
                                rotation=0)
                    else:
                        ax.stackplot(x, shares.T, labels=labels, colors=colors)
                        ax.set_ylim(0, 100)

            elif kind == "stack_to_total":
                if cols and len(cols) >= 2 and cols[0] in dfc.columns:
                    total_col = cols[0]
                    part_cols = [c for c in cols[1:] if c in dfc.columns and not self._is_all_zero(dfc[c], atol=1e-12)]
                    if part_cols:
                        total = pd.to_numeric(dfc[total_col], errors="coerce").to_numpy(dtype=float)
                        total = np.nan_to_num(total, nan=0.0)
                        parts = dfc[part_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                        parts = np.nan_to_num(parts, nan=0.0)
                        denom = total.reshape(-1, 1)
                        shares = np.divide(parts, denom, out=np.zeros_like(parts), where=denom != 0) * 100.0

                        keys = [self._strip_units(c) for c in part_cols]
                        labels = [self._pretty_label(k) for k in keys]
                        colors = [color_map.get(k, None) for k in keys]
                        ax.stackplot(x, shares.T, labels=labels, colors=colors)
                        ax.set_ylim(0, 100)

            elif kind == "stack_mass":
                keep = [col for col in cols if col in dfc.columns and not self._is_all_zero(dfc[col], atol=1e-12)]
                if keep:
                    y = np.nan_to_num(dfc[keep].to_numpy(dtype=float), nan=0.0)
                    keys = [self._strip_units(c) for c in keep]
                    labels = [self._pretty_label(k) for k in keys]
                    colors = [color_map.get(k, None) for k in keys]

                    xpos = np.arange(len(x))
                    width = 1.0
                    show_empty_outline = bool(spec.get("show_empty_outline", False))

                    if show_empty_outline:
                        totals = y.sum(axis=1)
                        ax.bar(
                            xpos, totals,
                            width=width, color="none",
                            edgecolor="0.85", linewidth=0.6,
                            zorder=0, align="center")

                    bottom = np.zeros(len(xpos), dtype=float)
                    for lab, colr, j in zip(labels, colors, range(len(labels))):
                        h = y[:, j]
                        if np.all(np.isclose(h, 0.0)):
                            continue
                        ax.bar(xpos, h, width=width, bottom=bottom, label=lab, color=colr, align="center")
                        bottom += h

                    ax.set_xlim(-0.5, len(xpos) - 0.5)

                    n = len(xpos)
                    if n <= 30:
                        ax.set_xticks(xpos)
                        ax.set_xticklabels(
                            [f"{v:g}" if isinstance(v, (float, np.floating)) else str(v) for v in x],
                            rotation=0)
                    else:
                        step = max(1, n // 12)
                        idx = np.arange(0, n, step)
                        ax.set_xticks(idx)
                        xvals = x.iloc[idx] if hasattr(x, "iloc") else [x[i] for i in idx]
                        ax.set_xticklabels(
                            [f"{v:g}" if isinstance(v, (float, np.floating)) else str(v) for v in xvals],
                            rotation=0)

            elif kind == "bar_elim_summary":
                if len(dfc) == 0:
                    ax.text(0.5, 0.5, "Empty dataframe", ha="center", va="center", transform=ax.transAxes)
                else:
                    def _last(colname):
                        if colname not in dfc.columns:
                            return 0.0
                        s = pd.to_numeric(dfc[colname], errors="coerce")
                        return float(s.iloc[-1]) if len(s) else 0.0

                    deg_w = _last(mcol("mass_degraded_water_cumulative"))
                    deg_s = _last(mcol("mass_degraded_sediment_cumulative"))
                    vol   = _last(mcol("mass_volatilized_cumulative"))
                    bio_w = _last(mcol("mass_biodegraded_water_cumulative"))
                    bio_s = _last(mcol("mass_biodegraded_sediment_cumulative"))
                    hyd_w = _last(mcol("mass_hydrolyzed_water_cumulative"))
                    hyd_s = _last(mcol("mass_hydrolyzed_sediment_cumulative"))
                    photo = _last(mcol("mass_photodegraded_cumulative"))

                    xpos = np.arange(5)
                    xtxt = ["Degraded", "Volatilized", "Biodegraded", "Hydrolyzed", "Photodegraded"]
                    width = 0.75

                    bottom = 0.0
                    if deg_w:
                        ax.bar(xpos[0], deg_w, width, bottom=bottom,
                               label=self._pretty_label("degraded water"),
                               color=color_map["degraded water"])
                        bottom += deg_w
                    if deg_s:
                        ax.bar(xpos[0], deg_s, width, bottom=bottom,
                               label=self._pretty_label("degraded sediment"),
                               color=color_map["degraded sediment"])
                        bottom += deg_s
                    if vol:
                        ax.bar(xpos[1], vol, width,
                               label=self._pretty_label("volatilized"),
                               color=color_map["volatilized"])

                    bottom = 0.0
                    if bio_w:
                        ax.bar(xpos[2], bio_w, width, bottom=bottom,
                               label=self._pretty_label("biodegraded water"),
                               color=color_map["biodegraded water"])
                        bottom += bio_w
                    if bio_s:
                        ax.bar(xpos[2], bio_s, width, bottom=bottom,
                               label=self._pretty_label("biodegraded sediment"),
                               color=color_map["biodegraded sediment"])
                        bottom += bio_s

                    bottom = 0.0
                    if hyd_w:
                        ax.bar(xpos[3], hyd_w, width, bottom=bottom,
                               label=self._pretty_label("hydrolyzed water"),
                               color=color_map["hydrolyzed water"])
                        bottom += hyd_w
                    if hyd_s:
                        ax.bar(xpos[3], hyd_s, width, bottom=bottom,
                               label=self._pretty_label("hydrolyzed sediment"),
                               color=color_map["hydrolyzed sediment"])
                        bottom += hyd_s

                    if photo:
                        ax.bar(xpos[4], photo, width,
                               label=self._pretty_label("photodegraded"),
                               color=color_map["photodegraded"])

                    ax.axvline(1.5, linestyle=":")
                    ax.set_xticks(xpos)
                    ax.set_xticklabels(xtxt, rotation=20, ha="right")
                    ax.tick_params(axis="x", pad=3)
                    ax.set_xlim(-0.6, 4.6)

            elif kind == "sanity_ts":
                needed = [
                    mcol("mass_degraded_ts"),
                    mcol("mass_photodegraded_ts"),
                    mcol("mass_biodegraded_ts"),
                    mcol("mass_hydrolyzed_ts")]
                if all(n in dfc.columns for n in needed):
                    sum_path = (
                        dfc[mcol("mass_photodegraded_ts")] +
                        dfc[mcol("mass_biodegraded_ts")] +
                        dfc[mcol("mass_hydrolyzed_ts")])
                    key1 = self._strip_units(mcol("mass_degraded_ts"))
                    ax.plot(x, dfc[mcol("mass_degraded_ts")],
                            label=self._pretty_label(key1),
                            color=color_map.get(key1, None),
                            linewidth=1.8)
                    ax.plot(x, sum_path, label="Sum pathways (ts)", color="#000000", linewidth=1.8)

            elif kind == "sanity_cum":
                needed = [
                    mcol("mass_degraded_cumulative"),
                    mcol("mass_photodegraded_cumulative"),
                    mcol("mass_biodegraded_cumulative"),
                    mcol("mass_hydrolyzed_cumulative")]
                if all(n in dfc.columns for n in needed):
                    sum_path = (
                        dfc[mcol("mass_photodegraded_cumulative")] +
                        dfc[mcol("mass_biodegraded_cumulative")] +
                        dfc[mcol("mass_hydrolyzed_cumulative")])
                    key1 = self._strip_units(mcol("mass_degraded_cumulative"))
                    ax.plot(x, dfc[mcol("mass_degraded_cumulative")],
                            label=self._pretty_label(key1),
                            color=color_map.get(key1, None),
                            linewidth=1.8)
                    ax.plot(x, sum_path, label="Sum pathways (cumulative)", color="#000000", linewidth=1.8)

            # common styling
            _set_ylabel(ax, kind, cols, ylabel=spec.get("ylabel"))
            ax.set_axisbelow(True)

            # Remove vertical gridlines: only y-grid
            ax.grid(True, axis="y", alpha=0.25)

            # keep x-limits for non-bar panels
            stack_style = spec.get("stack_style", None)
            is_bar_style = (kind in ("stack_share", "stack_mass") and stack_style == "bars")
            if not is_bar_style and kind != "bar_elim_summary":
                try:
                    ax.set_xlim(x.min(), x.max())
                except Exception:
                    pass

            ax.set_xlabel("")

        def _render_fixed_page(page_rows, page_title=None):
            """
            Render one page.
            - row_mode='double': each page row contains 2 panels
            - row_mode='single': each page row contains 1 wide panel
            """
            fig = plt.figure(figsize=A4_PORTRAIT)

            title_txt = None
            if fig_title or page_title:
                title_txt = page_title if page_title and not fig_title else (
                    f"{fig_title} - {page_title}" if page_title else fig_title)
                title_txt = f"{title_txt}{period_suffix}"
                fig.suptitle(title_txt, fontsize=fs["figure_title"])

            top = 0.92 if title_txt else 0.97

            ncols_page = 2 if row_mode == "double" else 1

            outer = fig.add_gridspec(
                nrows=ROWS_PER_PAGE, ncols=ncols_page,
                left=0.17,
                right=0.95 if row_mode == "double" else 0.97,
                top=top, bottom=0.06,
                wspace=0.34 if row_mode == "double" else 0.0,
                hspace=0.24)

            blank = dict(kind="blank", cols=[], title="", legend={"max_cols": 1})

            page_rows = list(page_rows)
            while len(page_rows) < ROWS_PER_PAGE:
                page_rows.append([blank, blank] if row_mode == "double" else [blank])

            ax_plots = np.empty((ROWS_PER_PAGE, ncols_page), dtype=object)

            for rr in range(ROWS_PER_PAGE):
                row_specs = page_rows[rr]

                if row_mode == "double":
                    row_specs = row_specs + [blank] * (2 - len(row_specs))
                else:
                    row_specs = row_specs[:1] if row_specs else [blank]

                for cc in range(ncols_page):
                    spec = row_specs[cc]

                    legend_cfg = spec.get("legend", {}) or {}
                    legend_max_cols = legend_cfg.get("max_cols", 3)

                    if row_mode == "double":
                        hr = [0.72, 0.11, 0.29]
                    else:
                        if spec.get("kind") == "bar_elim_summary":
                            hr = [0.42, 0.18, 0.24]
                        else:
                            hr = [0.50, 0.08, 0.16]

                    sub = outer[rr, cc].subgridspec(
                        nrows=3, ncols=1,
                        height_ratios=hr,
                        hspace=0.03)

                    ax = fig.add_subplot(sub[0])
                    ax_xlab = fig.add_subplot(sub[1])
                    ax_leg = fig.add_subplot(sub[2])

                    ax_plots[rr, cc] = ax

                    _plot_one_panel(ax, spec)

                    if spec.get("kind") in ("blank", "bar_elim_summary"):
                        ax_xlab.axis("off")
                    else:
                        _place_xlabel_in_axis(ax_xlab, xlab)

                    if spec.get("kind") == "blank":
                        ax_leg.axis("off")
                    else:
                        _place_legend_in_axis(
                            ax, ax_leg,
                            max_cols=legend_max_cols if row_mode == "double" else max(legend_max_cols, 3),
                            fontsize=fs["legend"],
                            legend_lw=3.0)

            return fig, ax_plots

        # Topic grouping + pagination
        def find_row(startswith):
            for i, row in enumerate(rows):
                if row and row[0].get("title", "").startswith(startswith):
                    return i
            return None

        i_mass = find_row("Mass in system")
        i_deg = find_row("Total degraded")
        i_bio = find_row("Biodegradation")
        i_allsp = find_row("All species masses (stacked, ts)")
        i_sp_sed = find_row("SP masses (sediment partition)")
        i_sanity = find_row("Sanity (ts):")

        topic_pages = []

        # 1) Mass balance overview
        if i_mass is not None and i_deg is not None:
            topic_pages.append(("Mass balance overview", rows[i_mass:i_deg]))

        # 2) Fate & transport sinks
        if i_deg is not None and i_bio is not None:
            topic_pages.append(("Fate & transport sinks", rows[i_deg:i_bio]))

        i_trans = find_row("Adsorption to suspended particles")

        # 3) Degradation pathways
        if i_bio is not None:
            if i_trans is not None:
                topic_pages.append(("Degradation pathways", rows[i_bio:i_trans]))
            elif i_allsp is not None:
                topic_pages.append(("Degradation pathways", rows[i_bio:i_allsp]))

        # 4) Adsorption / desorption / deposition / resuspension
        if i_trans is not None and i_allsp is not None:
            topic_pages.append(("Adsorption / desorption and deposition / resuspension", rows[i_trans:i_allsp]))

        # 5) Speciation overview
        if i_allsp is not None and i_sp_sed is not None:
            topic_pages.append(("Speciation overview", rows[i_allsp:i_sp_sed]))

        # 6) Speciation in sediment
        i_sp_sed_comp = find_row("SP composition (%) (sediment partition)")
        if i_sp_sed is not None:
            if row_mode == "double":
                topic_pages.append(("Speciation in sediment", rows[i_sp_sed:i_sp_sed + 1]))
            else:
                end_idx = (i_sp_sed_comp + 1) if i_sp_sed_comp is not None else (i_sp_sed + 1)
                topic_pages.append(("Speciation in sediment", rows[i_sp_sed:end_idx]))

        # 7) Additional fields
        if i_extra is not None:
            end_extra = i_sanity if i_sanity is not None else len(rows)
            if end_extra > i_extra:
                topic_pages.append(("Additional fields", rows[i_extra:end_extra]))

        # 8) Mass budget checks
        if i_sanity is not None:
            topic_pages.append(("Mass-budget checks", rows[i_sanity:]))

        # If not splitting to multiple pages, render everything into one figure
        if not split_a4:
            nrows_local = len(rows)
            ncols_local = 2 if row_mode == "double" else 1

            fig = plt.figure(figsize=A4_PORTRAIT)

            title_txt = fig_title if fig_title else "All panels"
            if title_txt:
                title_txt = f"{title_txt}{period_suffix}"
                fig.suptitle(title_txt, fontsize=fs["figure_title"])

            top = 0.92 if title_txt else 0.97

            outer = fig.add_gridspec(
                nrows=nrows_local, ncols=ncols_local,
                left=0.17,
                right=0.95 if row_mode == "double" else 0.97,
                top=top, bottom=0.06,
                wspace=0.34 if row_mode == "double" else 0.0,
                hspace=0.24)

            ax_plots = np.empty((nrows_local, ncols_local), dtype=object)
            blank = dict(kind="blank", cols=[], title="", legend={"max_cols": 1})

            for rr in range(nrows_local):
                row_specs = rows[rr]

                if row_mode == "double":
                    row_specs = row_specs + [blank] * (2 - len(row_specs))
                else:
                    row_specs = row_specs[:1] if row_specs else [blank]

                for cc in range(ncols_local):
                    spec = row_specs[cc]

                    legend_cfg = spec.get("legend", {}) or {}
                    legend_max_cols = legend_cfg.get("max_cols", 3)

                    if row_mode == "double":
                        hr = [0.72, 0.11, 0.29]
                    else:
                        if spec.get("kind") == "bar_elim_summary":
                            hr = [0.42, 0.18, 0.24]
                        else:
                            hr = [0.50, 0.08, 0.16]

                    sub = outer[rr, cc].subgridspec(
                        nrows=3, ncols=1,
                        height_ratios=hr,
                        hspace=0.03)

                    ax = fig.add_subplot(sub[0])
                    ax_xlab = fig.add_subplot(sub[1])
                    ax_leg = fig.add_subplot(sub[2])

                    ax_plots[rr, cc] = ax

                    _plot_one_panel(ax, spec)

                    if spec.get("kind") in ("blank", "bar_elim_summary"):
                        ax_xlab.axis("off")
                    else:
                        _place_xlabel_in_axis(ax_xlab, xlab)

                    if spec.get("kind") == "blank":
                        ax_leg.axis("off")
                    else:
                        _place_legend_in_axis(
                            ax, ax_leg,
                            max_cols=legend_max_cols if row_mode == "double" else max(legend_max_cols, 3),
                            fontsize=fs["legend"],
                            legend_lw=3.0)

            return fig, ax_plots, dfc

        # split_a4: paginate
        def chunk_rows(row_list, n=ROWS_PER_PAGE):
            for i in range(0, len(row_list), n):
                yield row_list[i:i + n]

        figs = []
        pdf = PdfPages(pdf_path) if pdf_path else None
        blank = dict(kind="blank", cols=[], title="", legend={"max_cols": 1})

        for topic, topic_rows in topic_pages:
            chunks = list(chunk_rows(topic_rows, ROWS_PER_PAGE))
            for idx, chunk in enumerate(chunks, start=1):
                chunk = list(chunk)
                while len(chunk) < ROWS_PER_PAGE:
                    chunk.append([blank, blank] if row_mode == "double" else [blank])

                page_title = f"{topic} - {idx}" if len(chunks) > 1 else topic
                fig, _ = _render_fixed_page(chunk, page_title=page_title)
                figs.append(fig)

                if pdf:
                    pdf.savefig(fig)
                    plt.close(fig)

        if pdf:
            pdf.close()

        return figs, dfc

    ### Helpers for sum_summary_timeseries
    @staticmethod
    def _as_timedelta64(x):
        '''
        Convert a duration-like input into a NumPy timedelta64.

        x:   None|np.timedelta64|pandas.Timedelta|str, Duration value to convert.
                - None -> returns None
                - np.timedelta64 -> returned unchanged
                - pandas.Timedelta -> converted via to_timedelta64()
                - str -> parsed with pandas.to_timedelta (e.g. "3h", "24h", "2D")

        Returns:
            np.timedelta64|None, Converted duration, or None if x is None.

        Raises:
            TypeError, If x is not one of the supported types.
        '''
        import pandas as pd
        import numpy as np

        if x is None:
            return None
        if isinstance(x, np.timedelta64):
            return x
        if isinstance(x, pd.Timedelta):
            return x.to_timedelta64()
        if isinstance(x, str):
            return pd.to_timedelta(x).to_timedelta64()
        raise TypeError(f"Unsupported timedelta type: {type(x)}")

    @staticmethod
    def _infer_freq_step(times_list, verbose=False):
        '''
        Infer the minimal positive timestep across multiple time vectors.

        times_list:   list[array-like], List of time arrays.
                        Each element is converted with np.asarray and differenced with np.diff.
                        Works with datetime64 (returns np.timedelta64) and numeric time (returns float/int).
                        Non-positive diffs (duplicates/non-monotonic artifacts) are ignored.
        verbose:      bool, If True, prints whether a frequency was inferred.

        Returns:
            np.timedelta64|float|int|None, Minimal positive step if inferable; otherwise None.
        '''
        steps = []
        for t in times_list:
            t = np.asarray(t)
            if t.size > 1:
                d = np.diff(t)

                if np.issubdtype(d.dtype, np.timedelta64):
                    d = d[d > np.timedelta64(0, "ns")]
                else:
                    d = d[d > 0]

                if d.size:
                    steps.append(d.min())

        if steps:
            freq = np.min(steps)
            if verbose:
                print("freq inferred:", freq)
            return freq

        if verbose:
            print("freq could not be inferred")
        return None


    def _make_target_time(self, DataArray_ls, time_name,
                          start_date=None, end_date=None,
                          freq_time=None, verbose=False):
        '''
        Construct a common target time grid for aligning/summing multiple series.

        DataArray_ls: list[xarray.DataArray], List of DataArrays containing coordinate/dimension time_name.
                        Only the time coordinate values are used.
        time_name:    str, Name of the time coordinate/dimension to align on.
        start_date:   np.datetime64|float|int|None, Start of target grid.
                        If None, inferred as the minimum start across DataArray_ls.
        end_date:     np.datetime64|float|int|None, End of target grid.
                        If None, inferred as the maximum end across DataArray_ls.
        freq_time:    np.timedelta64|float|int|None, Target timestep.
                        If None, inferred from inputs using _infer_freq_step.
                        If it cannot be inferred, uses the union of timestamps (irregular grid).
                        Must be timedelta-like for datetime axes, numeric for numeric axes.
        verbose:      bool, If True, prints inference/progress information.

        Returns:
            np.ndarray, Target time coordinate (regular grid if freq_time available; else union of timestamps).
        '''
        times = [da[time_name].values for da in DataArray_ls]

        if start_date is None:
            start_date = np.min([t.min() for t in times])
            if verbose:
                print("start_date set from DataArray_ls:", start_date)

        if end_date is None:
            end_date = np.max([t.max() for t in times])
            if verbose:
                print("end_date set from DataArray_ls:", end_date)

        target_time = None

        if freq_time is None:
            freq_time = self._infer_freq_step(times, verbose=verbose)
            if freq_time is None:
                # fallback: union (non-regular)
                target_time = np.unique(np.concatenate(times))
                if verbose:
                    print("Using union of timestamps (non-regular grid)")

        if target_time is None:
            target_time = np.arange(start_date, end_date + freq_time, freq_time)

        return target_time


    @staticmethod
    def _reindex_ds(ds, time_name, target_time,
                    nearest_tol_time=None, align_mode="pad",
                    clip_to_original=True):
        '''
        Reindex an xarray.Dataset onto a target time grid.

        ds:                  xarray.Dataset, Dataset to reindex.
        time_name:           str, Name of the time coordinate/dimension.
        target_time:         np.ndarray, Target timestamps to reindex onto.
        nearest_tol_time:    np.timedelta64|None, Max allowed distance for align_mode="nearest".
                                           Ignored for "pad" and "exact".
        align_mode:          str, Reindexing mode. Allowed: "pad"|"nearest"|"exact".
                                           - "exact": exact matches only (no filling)
                                           - "pad": forward-fill from previous timestamp
                                           - "nearest": nearest timestamp (optionally within nearest_tol_time)
        clip_to_original:    bool, If True, masks values outside the original ds time range after reindexing.

        Returns:
            xarray.Dataset, Reindexed (and optionally clipped) Dataset.
        '''
        if align_mode in {"pad", "nearest"}:
            idx = ds[time_name].to_index()
            if not idx.is_monotonic_increasing:
                ds = ds.sortby(time_name)

        if align_mode == "exact":
            out = ds.reindex({time_name: target_time}, method=None)
        elif align_mode == "pad":
            out = ds.reindex({time_name: target_time}, method="pad")
        else:  # "nearest"
            kwargs = {}
            if nearest_tol_time is not None:
                kwargs["tolerance"] = nearest_tol_time
            out = ds.reindex({time_name: target_time}, method="nearest", **kwargs)

        if clip_to_original:
            t_orig = ds[time_name].values
            t_min = t_orig.min()
            t_max = t_orig.max()
            valid = (out[time_name] >= t_min) & (out[time_name] <= t_max)
            out = out.where(valid)

        return out

    @staticmethod
    def _speciation_qc_check(
        df,
        mass_unit,
        *,
        denom_base = "mass_actual_ts",
        exclude_keys = ("sed_buried",),
        abs_tol_mass = 1e-9,     # in target mass units
        rel_tol_mass = 1e-6,     # fraction of denom
        abs_tol_perc = 1e-6,     # percentage points
        add_qc_columns = False,   # default: don't add columns
        qc_prefix = "qc_",
        on_fail= "warn",          # "warn" | "raise" | "ignore"
        return_report = False,    # default: don't return anything extra
        time_hint_col = None,  # e.g. "date_of_timestep" for nicer reporting
    ):
        '''
        Run QC checks for speciation consistency (mass closure + percent sum).

        df:              pandas.DataFrame, DataFrame to validate.
                             Expected columns:
                               Denominator: f"{denom_base} [{mass_unit}]"
                               Species masses: "mass_sp_<key>_ts [{mass_unit}]"
                               Species percents: "perc_sp_<key>_ts [%]"
        mass_unit:       str, Mass unit token (without brackets), e.g. "g", "mg", "ug".
                             Used to find the unit-tagged columns in df.
        denom_base:      str, Base name for the denominator (default "mass_actual_ts").
                             If denom_base already contains brackets, it is used as-is.
                             Otherwise the denominator column is f"{denom_base} [{mass_unit}]".
        exclude_keys:    tuple[str,...], Species keys excluded from QC (default includes "sed_buried").
                             These are treated as outside-system and not part of the 100% partition.
        abs_tol_mass:    float, Absolute tolerance for mass closure residual (in target mass units).
        rel_tol_mass:    float, Relative tolerance for mass closure residual (fraction of denominator when denom>0).
        abs_tol_perc:    float, Absolute tolerance for percent-sum error (percentage points).
        add_qc_columns:  bool, If True, append diagnostic columns (prefixed with qc_prefix) to df.
        qc_prefix:       str, Prefix for diagnostic columns when add_qc_columns is True.
        on_fail:         str, Failure behavior. Allowed: "warn"|"raise"|"ignore".
                            "warn": emit RuntimeWarning
                            "raise": raise ValueError
                            "ignore": do nothing
        return_report:   bool, If True, return a summary report dict (and df if add_qc_columns=True).
        time_hint_col:   str|None, Optional column used only to include a timestamp/value in warning/error messages.

        Returns:
            None, If return_report=False and add_qc_columns=False (only warns/raises as configured).
            pandas.DataFrame, If return_report=False and add_qc_columns=True (df with QC columns).
            dict, If return_report=True and add_qc_columns=False (QC report).
            (pandas.DataFrame, dict), If return_report=True and add_qc_columns=True (df with QC columns + report).
        '''
        import numpy as np
        import pandas as pd
        import re
        import warnings

        out = df.copy() if add_qc_columns else df
        dst_tag = f"[{mass_unit}]"
        denom_col = denom_base if "[" in denom_base else f"{denom_base} {dst_tag}"

        report = {
            "denom_col": denom_col,
            "mass_unit": mass_unit,
            "n_rows": len(df),
            "mass": {},
            "perc": {},}

        if denom_col not in df.columns or len(df) == 0:
            # nothing to check
            if return_report:
                return (out, report) if add_qc_columns else report
            return out if add_qc_columns else None

        den = pd.to_numeric(df[denom_col], errors="coerce").to_numpy()
        den_safe = np.where(den > 0, den, np.nan)

        # MASS closure
        mass_sp_cols = [c for c in df.columns if c.startswith("mass_sp_") and c.endswith(dst_tag)]
        kept_mass_cols = []
        for c in mass_sp_cols:
            m = re.match(r"^mass_sp_(.+)_ts\s*\[[^\]]+\]\s*$", c)
            key = m.group(1) if m else None
            if key and (key not in exclude_keys):
                kept_mass_cols.append(c)

        if kept_mass_cols:
            mass_sp_sum = (
                df[kept_mass_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .sum(axis=1)
                .to_numpy(dtype=float))
        else:
            mass_sp_sum = np.full(len(df), np.nan, dtype=float)

        denom = pd.to_numeric(df[denom_col], errors="coerce").to_numpy(dtype=float)
        residual = denom - mass_sp_sum
        abs_res = np.abs(residual)

        # rel bound only meaningful when denom>0
        rel_bound = rel_tol_mass * np.where(denom > 0, denom, np.nan)
        mass_ok = (abs_res <= abs_tol_mass) | ((denom > 0) & (abs_res <= rel_bound))

        residual_rel = np.nan_to_num((residual / den_safe) * 100.0, nan=0.0)

        report["mass"].update({
            "kept_mass_cols": kept_mass_cols,
            "n_fail": int((~mass_ok).sum()),
            "max_abs_residual": float(np.nanmax(abs_res)) if np.isfinite(abs_res).any() else np.nan,
            "max_rel_residual_pct": float(np.nanmax(np.abs(residual_rel))) if np.isfinite(residual_rel).any() else np.nan,})

        # PERCENT sum
        perc_cols = [c for c in df.columns if c.startswith("perc_sp_") and c.endswith("[%]")]
        kept_perc_cols = []
        for c in perc_cols:
            m = re.match(r"^perc_sp_(.+)_ts\s*\[%\]\s*$", c)
            key = m.group(1) if m else None
            if key and (key not in exclude_keys):
                kept_perc_cols.append(c)

        if kept_perc_cols:
            perc_sum = (
                df[kept_perc_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .sum(axis=1)
                .to_numpy(dtype=float))
        else:
            perc_sum = np.full(len(df), np.nan, dtype=float)

        target = np.where(denom > 0, 100.0, 0.0)
        perc_ok = np.abs(perc_sum - target) <= abs_tol_perc

        report["perc"].update({
            "kept_perc_cols": kept_perc_cols,
            "n_fail": int((~perc_ok).sum()),
            "max_abs_err": float(np.nanmax(np.abs(perc_sum - target))) if np.isfinite(perc_sum).any() else np.nan,})

        # optional QC columns
        if add_qc_columns:
            out[f"{qc_prefix}mass_sp_sum_ts {dst_tag}"] = mass_sp_sum
            out[f"{qc_prefix}mass_sp_residual_ts {dst_tag}"] = residual
            out[f"{qc_prefix}mass_sp_residual_rel [%]"] = residual_rel
            out[f"{qc_prefix}mass_sp_closure_ok"] = mass_ok

            out[f"{qc_prefix}perc_sp_sum_ts [%]"] = perc_sum
            out[f"{qc_prefix}perc_sp_sum_ok"] = perc_ok

        # failure handling
        any_fail = (report["mass"]["n_fail"] > 0) or (report["perc"]["n_fail"] > 0)
        if any_fail and on_fail != "ignore":
            # find a representative “worst” index to point at
            worst_i = None
            if report["mass"]["n_fail"] > 0 and np.isfinite(abs_res).any():
                worst_i = int(np.nanargmax(abs_res))
            elif report["perc"]["n_fail"] > 0 and np.isfinite(np.abs(perc_sum - target)).any():
                worst_i = int(np.nanargmax(np.abs(perc_sum - target)))

            where = f"row={worst_i}"
            if time_hint_col and time_hint_col in df.columns:
                where = f"{where}, {time_hint_col}={df[time_hint_col].iloc[worst_i]!r}"

            msg = (
                f"Speciation QC failed ({where}). "
                f"Mass closure fails: {report['mass']['n_fail']}/{len(df)} "
                f"(max |res|={report['mass']['max_abs_residual']}, "
                f"max |res_rel|%={report['mass']['max_rel_residual_pct']}). "
                f"Percent-sum fails: {report['perc']['n_fail']}/{len(df)} "
                f"(max abs err={report['perc']['max_abs_err']}).")

            if on_fail == "raise":
                raise ValueError(msg)
            else:  # warn
                warnings.warn(msg, RuntimeWarning)

        if return_report:
            return (out, report) if add_qc_columns else report
        return out if add_qc_columns else None

    @staticmethod
    def _recompute_perc_sp(
        df,
        mass_unit,
        *,
        denom_base="mass_actual_ts",
        normalize_to_100=False,
        exclude_keys=("sed_buried",),
    ):
        '''
        Recompute perc_sp_*_ts [%] from mass_sp_*_ts and the denominator.

        df:               pandas.DataFrame, DataFrame containing denominator and mass_sp/perc_sp columns.
                            perc_sp columns are recomputed in the returned copy.
        mass_unit:        str, Mass unit token (without brackets), e.g. "g", "mg", "ug".
                            Used to map "mass_sp_*_ts [{mass_unit}]" columns.
        denom_base:       str, Base name for denominator (default "mass_actual_ts").
                            Denominator column is f"{denom_base} [{mass_unit}]" unless denom_base already has brackets.
                            Percent rule matches generator semantics:
                              den_safe = where(den > 0, den, nan)
                              perc = nan_to_num((mass_sp/den_safe)*100, nan=0)
        normalize_to_100: bool, If True, renormalize computed perc_sp columns so row-wise sum is exactly 100
                            (only when denom>0 and row-sum>0). Useful to remove tiny FP drift.
        exclude_keys:     tuple[str,...], Species keys to skip (default includes "sed_buried").

        Returns:
            pandas.DataFrame, Copy of df with recomputed perc_sp_*_ts [%] columns.
        '''
        import numpy as np
        import pandas as pd
        import re

        out = df.copy()
        dst_tag = f"[{mass_unit}]"
        denom_col = denom_base if "[" in denom_base else f"{denom_base} {dst_tag}"
        if denom_col not in out.columns:
            return out

        den = pd.to_numeric(out[denom_col], errors="coerce").to_numpy()
        den_safe = np.where(den > 0, den, np.nan)

        perc_cols = [c for c in out.columns if c.startswith("perc_sp_") and c.endswith("[%]")]
        if not perc_cols:
            return out

        def key_from_perc(col):
            m = re.match(r"^perc_sp_(.+)_ts\s*\[%\]\s*$", col)
            return m.group(1) if m else None

        computed = []
        for pc in sorted(perc_cols):
            k = key_from_perc(pc)
            if (k is None) or (k in exclude_keys):
                continue

            mass_col = pc.replace("perc_", "mass_").replace("[%]", dst_tag)
            if mass_col not in out.columns:
                out[pc] = np.nan
                continue

            num = pd.to_numeric(out[mass_col], errors="coerce").to_numpy()
            out[pc] = np.nan_to_num((num / den_safe) * 100.0, nan=0.0)
            computed.append(pc)

        if normalize_to_100 and computed:
            row_sum = out[computed].sum(axis=1).to_numpy(dtype=float)
            scale = np.where((den > 0) & (row_sum > 0), 100.0 / row_sum, 1.0)
            out[computed] = out[computed].multiply(scale, axis=0)

        return out

    @staticmethod
    def _recompute_perc_elim(
        df,
        mass_unit,
        *,
        perc_cols=None,
        denom_terms=("mass_degraded_ts", "mass_volatilized_ts",
                     "mass_adv_out_ts", "mass_stranded_ts", "mass_buried_ts"),
    ):
        '''
    Recompute perc_elim_* [%] from the corresponding mass_*_ts and a chosen elimination denominator.

    df:            pandas.DataFrame, DataFrame containing elimination mass time-series columns and perc_elim columns.
                      perc_elim columns are recomputed in the returned copy.
    mass_unit:     str, Mass unit token (without brackets), e.g. "g", "mg", "ug".
                      Used to build the unit tag "[{mass_unit}]" when mapping perc_elim columns
                      to their corresponding mass columns and when building the denominator.
    perc_cols:     list[str]|None, List of perc_elim columns to recompute.
                      If None, recomputes all columns in df matching:
                        startswith("perc_elim_") and endswith("[%]").
    denom_terms:   tuple[str,...], Base names (without unit tag) of elimination mass terms used
                      to form the denominator, e.g.: ("mass_degraded_ts", "mass_volatilized_ts",
                         "mass_adv_out_ts", "mass_stranded_ts", "mass_buried_ts")
                      The denominator is computed row-wise as:
                        den = sum( df[f"{term} [{mass_unit}]"] )   (missing terms treated as 0)

    Returns:
        pandas.DataFrame, Copy of df with recomputed perc_elim_* [%] columns.
    '''
        import numpy as np
        import pandas as pd

        out = df.copy()
        dst_tag = f"[{mass_unit}]"

        if perc_cols is None:
            perc_cols = [c for c in out.columns if c.startswith("perc_elim_") and c.endswith("[%]")]

        # denominator: sum of selected elimination mass terms (missing -> 0)
        den = np.zeros(len(out), dtype=float)
        for base in denom_terms:
            col = f"{base} {dst_tag}"
            if col in out.columns:
                den += pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        for pc in perc_cols:
            # "perc_elim_mass_degraded_ts [%]" -> "mass_degraded_ts [g]"
            mass_col = pc.replace("perc_elim_", "").replace("[%]", dst_tag)
            if mass_col not in out.columns:
                out[pc] = np.nan
                continue

            num = pd.to_numeric(out[mass_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            out[pc] = np.divide(num, den, out=np.zeros_like(num), where=den > 0.0) * 100.0

        return out

    def sum_summary_timeseries(
        self,
        df_list,
        target_mass_unit="g",
        target_time_unit="hr",
        date_col="date_of_timestep",
        time_col=None,
        align_mode="pad",
        nearest_tol_time=None,
        clip_to_original=True,
        start_date=None, end_date=None,
        freq_time=None,
        rebuild_time_from_date=True,
        verbose=False,
        qc_check=True,
        add_qc_columns=False,
        qc_on_fail="warn",
        return_qc_report=False,
        normalize_perc_to_100=False,
    ):
        '''
        Align and sum multiple summary time series DataFrames, then recompute perc_sp_*.

        df_list:             list[pandas.DataFrame], Input summary DataFrames to align and sum.
                             Each DF is converted to common units using convert_units().
                             Percent columns (perc_*) are dropped before summation and recomputed after.
        target_mass_unit:    str, Mass unit for outputs. Allowed (per your converter): 'kg','g','mg','ug','µg'.
        target_time_unit:    str, Time unit for output bookkeeping / optional rebuilt numeric time column.
                             Allowed: 's','m','hr'/'h','d' (per your converter).
        date_col:            str, Name of datetime column used as time axis when present in all inputs.
                             If present in all DFs, it is used as the alignment index.
        time_col:            str|None, Explicit numeric time column name to use.
                             If None, convert_units() auto-detects a column like "time [..]".
                             When date_col is used as index, any numeric "time [..]" columns are dropped
                             to prevent summing time values across runs.
        align_mode:          str, Time alignment mode for reindexing. Allowed: "pad"|"nearest"|"exact".
        nearest_tol_time:    np.timedelta64|pandas.Timedelta|str|None, Tolerance for align_mode="nearest".
                             Strings like "3h" are supported.
        clip_to_original:    bool, If True, masks values outside each run's original time range after reindexing.
        start_date,end_date: np.datetime64|pandas.Timestamp|float|int|None, Start/end bounds for the target grid.
                             If None, inferred from the input series.
        freq_time:           np.timedelta64|pandas.Timedelta|str|float|int|None, Target timestep.
                             If None, inferred; if not inferable, uses union of timestamps.
                             Must be timedelta-like when aligning on date_col; numeric when aligning on numeric time.
        rebuild_time_from_date: bool, If True and using date_col axis, rebuild a numeric "time [target_time_unit]" column
                             measured from the first timestamp.
        verbose:             bool, Print progress/inference information.
        qc_check:            bool, If True, run _speciation_qc_check after recomputing perc_sp.
        add_qc_columns:      bool, If True, append QC diagnostic columns to the output.
                             Default False (QC runs but output schema is unchanged).
        qc_on_fail:          str, QC failure behavior. Allowed: "warn"|"raise"|"ignore".
        return_qc_report:    bool, If True, return a QC report dict in addition to the output DataFrame.
        normalize_perc_to_100: bool, If True, renormalize perc_sp columns to sum exactly to 100 when denom>0.

        Returns:
            pandas.DataFrame, Summed and post-processed DataFrame (units converted, aligned, summed, perc_sp recomputed).
            (pandas.DataFrame, dict), If return_qc_report=True, returns (output_df, qc_report_dict).
        '''
        import numpy as np
        import pandas as pd
        import xarray as xr

        if not df_list:
            raise ValueError("df_list is empty")

        nearest_tol_time = self._as_timedelta64(nearest_tol_time)

        ds_list = []
        perc_sp_cols_union = set()
        perc_elim_cols_union = set()
        time_name = None
        UM_mass = UM_time = None
        used_date_axis = None

        # reference column order from the first converted df
        ref_cols = None
        ref_time_col = None  # exact "time [..]" label from ref_cols (if any)

        for df in df_list:
            dfc, UM_mass_i, UM_time_i, detected_time_col = self.convert_units(
                df,
                target_mass_unit=target_mass_unit,
                target_time_unit=target_time_unit,
                time_col=time_col,)

            if ref_cols is None:
                ref_cols = list(dfc.columns)
                ref_time_col = next(
                    (c for c in ref_cols if c.startswith("time [") and c.endswith("]")),
                    None)

            perc_sp_cols_union.update([c for c in dfc.columns if c.startswith("perc_sp_") and c.endswith("[%]")])
            perc_elim_cols_union.update([c for c in dfc.columns if c.startswith("perc_elim_") and c.endswith("[%]")])

            has_date = (date_col in dfc.columns)
            if used_date_axis is None:
                used_date_axis = has_date
            if used_date_axis != has_date:
                raise ValueError(f"Inconsistent time axis: {date_col!r} present in some dataframes but not all.")

            if has_date:
                tname = date_col
                dfc[tname] = pd.to_datetime(dfc[tname])
                dfc = dfc.set_index(tname)
                time_like_cols = [c for c in dfc.columns if c.startswith("time [") and c.endswith("]")]
            else:
                if detected_time_col is None:
                    raise ValueError("No date_col present and no time column detected.")
                tname = detected_time_col
                dfc[tname] = pd.to_numeric(dfc[tname], errors="coerce")
                dfc = dfc.set_index(tname)
                time_like_cols = []  # time axis is the index already

            drop_cols = []
            if "UM" in dfc.columns:
                drop_cols.append("UM")
            drop_cols += [c for c in dfc.columns if c.startswith("perc_") and c.endswith("[%]")]
            drop_cols += time_like_cols

            dfc2 = dfc.drop(columns=[c for c in drop_cols if c in dfc.columns], errors="ignore")

            ds = dfc2.to_xarray()
            numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
            if not numeric_vars:
                raise ValueError("A dataframe had no numeric variables after dropping perc/UM/time.")
            ds = ds[numeric_vars].astype("float64")

            ds_list.append(ds)

            if time_name is None:
                time_name = tname
            elif time_name != tname:
                raise ValueError(f"Time coordinate name mismatch: {time_name!r} vs {tname!r}")

            UM_mass, UM_time = UM_mass_i, UM_time_i

        # convert freq_time only if datetime axis
        freq_time_used = self._as_timedelta64(freq_time) if used_date_axis else freq_time

        DataArray_ls = [ds[next(iter(ds.data_vars))] for ds in ds_list]
        target_time = self._make_target_time(
            DataArray_ls=DataArray_ls,
            time_name=time_name,
            start_date=start_date,
            end_date=end_date,
            freq_time=freq_time_used,
            verbose=verbose,)

        reindexed = [
            self._reindex_ds(
                ds=ds,
                time_name=time_name,
                target_time=target_time,
                nearest_tol_time=nearest_tol_time,
                align_mode=align_mode,
                clip_to_original=clip_to_original,)
            for ds in ds_list]

        all_vars = sorted(set().union(*[set(ds.data_vars) for ds in reindexed]))
        standardized = []
        for ds in reindexed:
            for v in all_vars:
                if v not in ds:
                    ds[v] = xr.DataArray(
                        np.full(ds[time_name].shape, np.nan, dtype="float64"),
                        coords={time_name: ds[time_name]},
                        dims=(time_name,),)
            standardized.append(ds[all_vars])

        aligned = xr.align(*standardized, join="exact", copy=False)

        if verbose:
            print("Combining datasets...")
        count_suffix = "__n_valid"

        def _sum_da_list(arrays):
            acc = arrays[0].fillna(0.0)
            for da in arrays[1:]:
                acc = acc + da.fillna(0.0)
            return acc
        # A value column is treated as a weighted-mean field if a companion
        # "<value_col>__n_valid" column exists.
        weighted_mean_value_cols = {
            v for v in all_vars
            if (not v.endswith(count_suffix)) and (f"{v}{count_suffix}" in all_vars)}

        combined_vars = {}
        for v in all_vars:
            arrays = [ds[v] for ds in aligned]
            # Companion counts are always additive
            if v.endswith(count_suffix):
                combined_vars[v] = _sum_da_list(arrays)
                continue
            # Weighted mean reconstruction:
            # global_mean = sum(mean_i * count_i) / sum(count_i)
            if v in weighted_mean_value_cols:
                count_name = f"{v}{count_suffix}"
                count_arrays = [ds[count_name] for ds in aligned]

                numerator = arrays[0].fillna(0.0) * count_arrays[0].fillna(0.0)
                denominator = count_arrays[0].fillna(0.0)

                for val_da, cnt_da in zip(arrays[1:], count_arrays[1:]):
                    cnt = cnt_da.fillna(0.0)
                    numerator = numerator + val_da.fillna(0.0) * cnt
                    denominator = denominator + cnt

                combined_vars[v] = xr.where(denominator > 0.0, numerator / denominator, np.nan)
                continue

            # Default behavior: additive series
            combined_vars[v] = _sum_da_list(arrays)

        combined = xr.Dataset(combined_vars)
        out = combined.to_dataframe().reset_index()

        # enforce mass closure for denom
        dst_tag = f"[{UM_mass}]"
        mw = f"mass_water_ts {dst_tag}"
        ms = f"mass_sed_ts {dst_tag}"
        ma = f"mass_actual_ts {dst_tag}"
        if mw in out.columns and ms in out.columns:
            out[ma] = (
                pd.to_numeric(out[mw], errors="coerce").fillna(0.0)
                + pd.to_numeric(out[ms], errors="coerce").fillna(0.0))

        # ensure perc_sp columns exist, then recompute
        for pc in sorted(perc_sp_cols_union):
            if pc not in out.columns:
                out[pc] = np.nan

        out = self._recompute_perc_sp(
            out,
            mass_unit=UM_mass,
            denom_base="mass_actual_ts",
            normalize_to_100=normalize_perc_to_100,
            exclude_keys=("sed_buried",),)

        # ensure perc_elim columns exist, then recompute
        for pc in sorted(perc_elim_cols_union):
            if pc not in out.columns:
                out[pc] = np.nan

        out = self._recompute_perc_elim(
            out,
            mass_unit=UM_mass,
            perc_cols=sorted(perc_elim_cols_union),
            denom_terms=(
                "mass_degraded_ts",
                "mass_volatilized_ts",
                "mass_adv_out_ts",
                "mass_stranded_ts",
                "mass_buried_ts",),)

        # QC
        qc_report = None
        if qc_check:
            qc_result = self._speciation_qc_check(
                out,
                mass_unit=UM_mass,
                denom_base="mass_actual_ts",
                exclude_keys=("sed_buried",),
                add_qc_columns=add_qc_columns,
                on_fail=qc_on_fail,
                return_report=return_qc_report,
                time_hint_col=(date_col if date_col in out.columns else None),)
            if return_qc_report:
                if add_qc_columns:
                    out, qc_report = qc_result
                else:
                    qc_report = qc_result

        # rebuild time [unit] from date axis, using the exact original label if available
        if used_date_axis and rebuild_time_from_date and target_time_unit and date_col in out.columns:
            old_time_cols = [c for c in out.columns if c.startswith("time [") and c.endswith("]")]
            out = out.drop(columns=old_time_cols, errors="ignore")

            time_col_name = ref_time_col or f"time [{target_time_unit}]"
            unit = time_col_name[time_col_name.find("[") + 1 : time_col_name.rfind("]")].strip()

            unit_to_ns = {
                "s": 1e9,
                "m": 60e9,
                "min": 60e9,
                "h": 3600e9,
                "hr": 3600e9,
                "d": 86400e9,
            }
            if unit not in unit_to_ns:
                raise ValueError(f"Unsupported target_time_unit for rebuild: {unit!r}")

            t0 = pd.to_datetime(out[date_col]).min()
            dt_ns = (pd.to_datetime(out[date_col]) - t0).astype("timedelta64[ns]").astype("int64")
            out.insert(0, time_col_name, dt_ns / unit_to_ns[unit])

        # UM
        out["UM"] = pd.NA
        if len(out):
            out.loc[out.index[0], "UM"] = f"[{UM_mass}] [{UM_time}]"

        # force final DF to contain *all* ref_cols, in the same order
        if ref_cols is not None:
            for c in ref_cols:
                if c not in out.columns:
                    out[c] = pd.NA

            ordered = list(ref_cols)
            extras = [c for c in out.columns if c not in set(ordered)]
            out = out[ordered + extras]

        return (out, qc_report) if return_qc_report else out

    ### Helpers for sum_DataArray_list
    @staticmethod
    def _reindex_da(
        DataArray,
        time_name, target_time,
        nearest_tol_time=None,
        align_mode="pad",
        clip_to_original=True):
        """
        Reindex xr.DataArray along "time_name" dimension using target_time array.

        DataArray:        xarray DataArray
        time_name:        string, name of time dimension of all DataArray present in DataArray_ls
        target_time:      np.array of np.datetime64[ns]
        nearest_tol_time: np.timedelta64, max distance for "nearest" (e.g., np.timedelta64(3,'h'))
        align_mode:       string, mode of selecting timestamp in reconstructed sum ("pad"|"nearest"|"exact")
        clip_to_original: bool, select if contribution of DataArray before the start and after the end of that array is set to 0
        """
        valid_modes = {"pad", "nearest", "exact"}
        if align_mode not in valid_modes:
            raise ValueError(f"align_mode must be one of {valid_modes}, got {align_mode!r}")

        if time_name not in DataArray.dims:
            raise ValueError(f'DataArray is missing time dimension "{time_name}"')

        time_index = DataArray[time_name].to_index()
        if not time_index.is_unique:
            raise ValueError(f'Time coordinate "{time_name}" must be unique for reindexing')

        # sort for methods that need monotonic time
        if align_mode in {"pad", "nearest"} and not time_index.is_monotonic_increasing:
            DataArray = DataArray.sortby(time_name)

        if align_mode == "exact":
            out = DataArray.reindex({time_name: target_time}, method=None)
        elif align_mode == "pad":
            out = DataArray.reindex({time_name: target_time}, method="pad")
        else:  # "nearest"
            kwargs = {}
            if nearest_tol_time is not None:
                kwargs["tolerance"] = nearest_tol_time
            out = DataArray.reindex({time_name: target_time}, method="nearest", **kwargs)

        if clip_to_original:
            t_orig = DataArray[time_name].values
            t_min = t_orig.min()
            t_max = t_orig.max()
            valid = (out[time_name] >= t_min) & (out[time_name] <= t_max)
            out = out.where(valid)

        return out

    @staticmethod
    def _infer_decimals_from_res(res, extra = 1):
        """
        Infer decimals from the resolution: number of visible decimal places in |res| + `extra`.
        Examples: 0.005 -> 4, 0.001 -> 4, 0.2 -> 2, 0.25 -> 3
        """
        r = abs(float(res))
        s = f"{r:.12f}".rstrip("0").rstrip(".")
        d = len(s.split(".")[1]) if "." in s else 0
        return max(0, d + extra)

    def build_regular_series(self, *,
        first_value, res, length,
        direction, decimals) :
        """
        Round the first_value to the desired decimals (derived from |res| if None),
        then create: first_rounded + step * i  for i = 0..length-1,
        where step = +|res| for ascending, -|res| for descending.
        """
        import numpy as np

        if length < 0:
            raise ValueError("length must be >= 0")
        if res == 0:
            raise ValueError("res must be non-zero")

        # decide direction / step
        base = abs(float(res))
        if direction is None:
            step = base if res > 0 else -base
        elif direction.lower() == "asc":
            step = base
        elif direction.lower() == "desc":
            step = -base
        else:
            raise ValueError("direction must be 'asc', 'desc', or None")

        # decimals for rounding
        if decimals is None:
            decimals = self._infer_decimals_from_res(base, extra=1)

        first_rounded = round(float(first_value), decimals)
        idx = np.arange(length, dtype=np.float64)
        series = first_rounded + idx * step
        return np.round(series, decimals=decimals)

    @staticmethod
    def _canonical_nontime_dims(DataArray_ls,
        nontime_dims, dim_res_dict=None,
        coord_tolerance_dict=None,
        snap_mode=None, Verbose=True):
        """
        If coord_tolerance is None -> strict equality (np.array_equal).
        If coord_tolerance is float -> allow per-index |diff| <= tolerance across arrays

        Return canonical 1-D coord arrays for each dim in nontime_dims.
        """
        import numpy as np

        if snap_mode not in (None, "median", "ref"):
            raise ValueError("snap_mode must be None, 'median', or 'ref'")

        def _is_regular(vec, tol):
            """
            Check if a 1D coordinate vector is regularly spaced within tolerance.
            """
            v = np.asarray(vec, dtype=np.float64)
            if v.size < 2:
                return True, 0.0
            diffs = np.diff(v)
            step = np.median(np.abs(diffs))
            sign = np.sign(diffs.mean()) if diffs.size else 1.0
            err = np.max(np.abs(diffs - sign * step))
            return (err <= (tol if tol is not None else 1e-12)), step

        def _min_decimals(x, max_dec=12):
            """
            Return minimum number of decimals to represent x as a clean decimal within tolerance.
            """
            x = float(abs(x))
            for d in range(max_dec + 1):
                if np.isclose(x * (10**d), round(x * (10**d)), atol=1e-12):
                    return d
            return max_dec

        canonical = {}
        need_interpolation = {}

        for dim in sorted(nontime_dims):
            ref = np.asarray(DataArray_ls[0][dim].values)

            # length check
            for idx, da in enumerate(DataArray_ls[1:], start=1):
                if dim not in da.dims:
                    raise ValueError(f'Missing dimension "{dim}" in array {idx}')
                if da.sizes[dim] != ref.size:
                    raise ValueError(
                        f'Dimension "{dim}" length differs: {da.sizes[dim]} vs {ref.size} for array {idx}')

            tol = None if coord_tolerance_dict is None else coord_tolerance_dict.get(dim)
            all_exact = True

            # tolerance check
            if tol is not None:
                for idx, da in enumerate(DataArray_ls[1:], start=1):
                    vals = np.asarray(da[dim].values)
                    diff = np.abs(vals - ref)
                    if np.nanmax(diff) > tol:
                        raise ValueError(
                            f'Dimension "{dim}" values differ by > tolerance ({tol}) in array {idx}')
                    if not np.array_equal(vals, ref):
                        all_exact = False
            else:
                for idx, da in enumerate(DataArray_ls[1:], start=1):
                    vals = np.asarray(da[dim].values)
                    if not np.array_equal(vals, ref):
                        raise ValueError(f'Dimension "{dim}" values differ between arrays (array {idx})')

            if all_exact:
                # dim is identical across all DataArray_ls
                if Verbose:
                    print(f"Identical dim '{dim}' among DataArray_ls: no interpolation needed")
                canonical[dim] = ref
                need_interpolation[dim] = False
                continue

            need_interpolation[dim] = True

            # resolution
            dim_res = None if dim_res_dict is None else dim_res_dict.get(dim)
            if dim_res is None:
                regular, dim_res = _is_regular(ref, tol)
                dim_res = float(np.round(dim_res, _min_decimals(dim_res)))
                if Verbose:
                    print(f"dim_res: {dim_res} inferred for dim: {dim}")
                if not regular:
                    raise ValueError(f'dim_res for "{dim}" not specified and could not be inferred.')

            # build canonical
            if snap_mode == "ref":
                can = ref.astype(np.float64)
            else:
                stack = np.stack(
                    [np.asarray(da[dim].values, dtype=np.float64) for da in DataArray_ls],
                    axis=0,)
                can0 = np.nanmedian(stack, axis=0)
                inc = (can0[-1] >= can0[0])
                first = can0[0]
                step = abs(dim_res)
                series = first + np.arange(can0.size) * (step if inc else -step)
                can = np.round(series, decimals=_min_decimals(step))
                if Verbose:
                    print(f"dim '{dim}' will be interpolated within tolerance ({tol}): {can0[0]} -> {can[0]}")

            if dim.lower() in ("lon", "longitude"):
                lon = can
                first_lon = np.asarray(DataArray_ls[0][dim].values)
                if first_lon.min() >= 0:
                    lon = lon % 360.0
                else:
                    lon = ((lon + 180.0) % 360.0) - 180.0
                can = lon

            canonical[dim] = can

        return canonical, need_interpolation

    @staticmethod
    def _normalize_inputs(DataArray_dict, DataArray_ls, variable):
        """
        Ensure the correct format of DataArray_ls.
        """

        if DataArray_dict is None and DataArray_ls is None:
            raise ValueError("Either DataArray_dict or DataArray_ls must be provided.")

        index_keys = None

        if DataArray_dict is not None:
            if variable is None:
                raise ValueError(
                    "Key for extracting DataArrays from DataArray_dict must be specified (variable)"
                )
            if DataArray_ls is not None:
                raise ValueError("Both DataArray_dict and DataArray_ls were given.")

            # Build list from dict, keep mapping list index → dict key
            sorted_items = sorted(DataArray_dict.items())
            index_keys = []
            DataArray_ls = []

            for idx_key, inner in sorted_items:
                if variable in inner and inner[variable] is not None:
                    DataArray_ls.append(inner[variable])
                    index_keys.append(idx_key)

        # If DataArray_dict is None, we just keep the passed DataArray_ls as-is
        if DataArray_ls is None or len(DataArray_ls) == 0:
            raise ValueError("Empty DataArray_ls")

        return DataArray_ls, index_keys

    @staticmethod
    def _normalize_depth_and_specie(DataArray_ls):
        """
        Normalize 'depth' and 'specie' across a list of DataArrays.

        - 'specie' is always removed (dim and/or coord).
        - If a common single-valued depth exists, it is kept as a SCALAR coordinate
          (0-D), NOT a dimension.
        - If depth is multi-valued (len > 1), we keep it as-is (dimension/coord).
        - If depth is multi-valued (length > 1), every DataArray must already have
          'depth' present, and all must match; they are left multi-depth but their
          depth coord is set to the common reference.
        """
        import numpy as np
        import xarray as xr

        # find depth reference among inputs (if any)
        depth_ref = None
        depth_ref_idx = None
        has_depth_flags = []

        for i, da in enumerate(DataArray_ls):
            has_depth = ("depth" in da.coords) or ("depth" in da.dims)
            has_depth_flags.append(has_depth)
            if not has_depth:
                continue

            depth_vals = da.coords["depth"].values if "depth" in da.coords else da["depth"].values
            depth_arr = np.atleast_1d(np.asarray(depth_vals))

            if depth_ref is None:
                depth_ref = depth_arr
                depth_ref_idx = i
            else:
                ref_arr = np.atleast_1d(np.asarray(depth_ref))
                if depth_arr.shape != ref_arr.shape or not np.allclose(depth_arr, ref_arr):
                    raise ValueError(
                        f"Inconsistent depth coordinates between inputs {i} and {depth_ref_idx}:\n"
                        f"  depth[{i}] = {depth_arr}\n"
                        f"  depth[{depth_ref_idx}] = {ref_arr}\n"
                        "All depth coordinate values must be identical.")

        def _drop_specie(aa: xr.DataArray) -> xr.DataArray:
            if "specie" in aa.dims:
                if aa.sizes["specie"] != 1:
                    raise ValueError(
                        f"'specie' dimension must be singleton to be dropped, got size {aa.sizes['specie']}")
                aa = aa.squeeze("specie", drop=True)
            if "specie" in aa.coords:
                aa = aa.drop_vars("specie")
            return aa

        if depth_ref is None:
            return [_drop_specie(da) for da in DataArray_ls]

        depth_ref = np.atleast_1d(np.asarray(depth_ref))
        n_depth = depth_ref.size
        any_missing_depth = not all(has_depth_flags)

        if n_depth > 1 and any_missing_depth:
            raise ValueError(
                "Some DataArrays have multi-layer 'depth' while others have no 'depth'. "
                "Cannot safely normalize in this case.")

        normalized = []

        for i, da in enumerate(DataArray_ls):
            aa = _drop_specie(da)

            if n_depth == 1:
                depth_value = float(depth_ref[0])

                if "depth" in aa.dims:
                    if aa.sizes["depth"] != 1:
                        raise ValueError(
                            f"DataArray index {i} has 'depth' dim size {aa.sizes['depth']} "
                            "but reference depth is single-valued.")
                    aa = aa.isel(depth=0, drop=True)

                if "depth" in aa.coords:
                    aa = aa.drop_vars("depth")
                aa = aa.assign_coords(depth=depth_value)
            else:
                if "depth" in aa.dims:
                    aa = aa.assign_coords(depth=depth_ref)
                elif "depth" in aa.coords:
                    aa = aa.assign_coords(depth=depth_ref)

            normalized.append(aa)

        return normalized

    def _infer_horizontal_names(self, da):
        """
        Return (lat_name, lon_name) from a DataArray using common conventions.
        """
        lat_name = next((n for n in ("latitude", "lat") if n in da.dims or n in da.coords), None)
        lon_name = next((n for n in ("longitude", "lon", "long") if n in da.dims or n in da.coords), None)

        if lat_name is None or lon_name is None:
            raise ValueError(
                "Could not infer horizontal coordinates. Expected one of "
                "latitude/lat and longitude/lon/long.")
        return lat_name, lon_name

    def _cell_areas_from_1d(self, lat, lon, lat_name="latitude", lon_name="longitude", R=6371000.0):
        """
        Compute spherical cell areas from 1D lat/lon centers.
        """
        import numpy as np
        import xarray as xr

        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

        def _centers_to_edges(v):
            v = np.asarray(v, dtype=np.float64)
            if v.size == 1:
                half = 0.5
                return np.array([v[0] - half, v[0] + half], dtype=np.float64)

            e = np.empty(v.size + 1, dtype=np.float64)
            e[1:-1] = 0.5 * (v[:-1] + v[1:])
            e[0] = v[0] - (e[1] - v[0])
            e[-1] = v[-1] + (v[-1] - e[-2])
            return e

        lat_e = np.deg2rad(_centers_to_edges(lat))
        lon_e = np.deg2rad(_centers_to_edges(lon))

        dlambda = np.diff(lon_e)[None, :]
        area = R**2 * np.abs(np.sin(lat_e[1:, None]) - np.sin(lat_e[:-1, None])) * np.abs(dlambda)

        return xr.DataArray(
            area,
            dims=(lat_name, lon_name),
            coords={lat_name: lat, lon_name: lon},
            name="cell_area",)

    def _standardize_horizontal_names(self, da, lat_name="latitude", lon_name="longitude"):
        """
        Rename common horizontal coordinate names to canonical names.
        """
        lat0, lon0 = self._infer_horizontal_names(da)
        rename_map = {}
        if lat0 != lat_name:
            rename_map[lat0] = lat_name
        if lon0 != lon_name:
            rename_map[lon0] = lon_name
        return da.rename(rename_map) if rename_map else da

    def _extract_common_topo_from_dict(self, DataArray_dict, topo_key="topo"):
        """
        Return a common topography from DataArray_dict if all topographies are identical.
        Otherwise raise, because one common cell-volume field is required for a clean
        mass-based assert on the aligned grid.
        """
        import numpy as np
        import xarray as xr

        topo_items = []
        for k, inner in sorted(DataArray_dict.items()):
            topo = inner.get(topo_key)
            if topo is not None:
                topo_items.append((k, topo))

        if not topo_items:
            return None

        n_total = len(DataArray_dict)
        n_topo = len(topo_items)
        if 0 < n_topo < n_total:
            raise ValueError("Some inputs have topography and some do not; cannot run common mass check.")

        ref_key, ref_topo = topo_items[0]
        for k, topo in topo_items[1:]:
            a, b = xr.align(ref_topo, topo, join="exact", copy=False)
            if not np.array_equal(np.asarray(a.values), np.asarray(b.values), equal_nan=True):
                raise ValueError(
                    f"Mass check requires one common topography, but topo for key {k} "
                    f"differs from topo for key {ref_key}.")

        return ref_topo.copy(deep=False)

    @staticmethod
    def _snap_2d_field_to_ref_coords(field, ref, dims, atol=1e-8):
        """
        Make a 2-D field use the exact coordinates of `ref` on the given dims,
        but only if coordinates are already numerically equal within tolerance.

        This is meant for cases where data_var coordinates were normalized/snap-fixed
        but a companion field (e.g. topography) still carries slightly different
        coordinate objects/rounding.
        """
        import numpy as np

        d0, d1 = dims
        out = field.transpose(d0, d1)

        for dim in (d0, d1):
            if dim not in out.coords:
                raise ValueError(f"Field is missing coordinate '{dim}'")
            if dim not in ref.coords:
                raise ValueError(f"Reference is missing coordinate '{dim}'")

            vals_field = np.asarray(out[dim].values)
            vals_ref = np.asarray(ref[dim].values)

            if vals_field.shape != vals_ref.shape:
                raise ValueError(
                    f"Coordinate '{dim}' shape mismatch: "
                    f"field={vals_field.shape} vs ref={vals_ref.shape}")

            if np.array_equal(vals_field, vals_ref):
                continue

            if not np.allclose(vals_field, vals_ref, rtol=0.0, atol=atol, equal_nan=True):
                diff = np.abs(vals_field - vals_ref)
                max_abs = float(np.nanmax(diff)) if diff.size else 0.0
                raise ValueError(
                    f"Coordinate '{dim}' differs between topography and da_ref "
                    f"(max_abs_diff={max_abs:.6e}, atol={atol})")

            out = out.assign_coords({dim: ref[dim].values})

        return out

    def _build_common_cell_volume(self, da_ref, topo_ref):
        """
        Build cell volume/thickness on the common aligned grid.

        3D: da_ref has 'depth' as a dimension, interpreted as layer tops.
                cell_volume = dz * area
            Scalar depth coord:
                da_ref has 'depth' as a scalar coord but not a dimension.
                cell_volume = (topo - depth) * area
        2D: no depth present.
            cell_volume = topo * area
        """
        import numpy as np
        import xarray as xr

        lat_name_ref, lon_name_ref = self._infer_horizontal_names(da_ref)
        lat_name_topo, lon_name_topo = self._infer_horizontal_names(topo_ref)

        topo = topo_ref
        if lat_name_topo != lat_name_ref or lon_name_topo != lon_name_ref:
            topo = topo.rename({lat_name_topo: lat_name_ref, lon_name_topo: lon_name_ref})

        expected_dims = {lat_name_ref, lon_name_ref}
        if set(topo.dims) != expected_dims:
            raise ValueError(
                f"topo_ref must be 2-D on ({lat_name_ref}, {lon_name_ref}); got dims {topo.dims}")

        topo = self._snap_2d_field_to_ref_coords(
            topo,
            da_ref,
            dims=(lat_name_ref, lon_name_ref),
            atol=1e-8,)

        topo = topo.astype(np.float64).clip(min=0.0)

        area = self._cell_areas_from_1d(
            da_ref[lat_name_ref].values,
            da_ref[lon_name_ref].values,
            lat_name=lat_name_ref,
            lon_name=lon_name_ref,)

        topo, area = xr.align(topo, area, join="exact", copy=False)

        # 3D case: depth is a layer-top dimension
        if "depth" in da_ref.dims:
            depth_top = np.asarray(da_ref["depth"].values, dtype=np.float64)

            z_top = xr.DataArray(
                depth_top,
                dims=("depth",),
                coords={"depth": da_ref["depth"].values},
            ).broadcast_like(topo).transpose("depth", lat_name_ref, lon_name_ref)

            z_bottom = topo.expand_dims({"depth": [depth_top.size]})
            z_edges = xr.concat([z_top, z_bottom], dim="depth").rename({"depth": "depth_edge"})
            z_edges = z_edges.transpose("depth_edge", lat_name_ref, lon_name_ref)

            dz = z_edges.diff("depth_edge").rename({"depth_edge": "depth"}).clip(min=0.0)
            dz = dz.assign_coords(depth=da_ref["depth"].values)

            cell_volume = dz * area

        # scalar depth coord but not a dimension
        elif "depth" in da_ref.coords:
            depth0 = float(np.ravel(da_ref["depth"].values)[0])
            thickness = (topo - depth0).clip(min=0.0)
            cell_volume = thickness * area

        # 2D full-column case
        else:
            cell_volume = topo * area

        return cell_volume.astype(np.float64)

    def _mass_on_common_grid(self, da, cell_volume):
        """
        Convert concentration to mass on the common grid and reduce only over spatial dims.
        Preserves time and any other non-spatial dims.
        """
        import numpy as np
        import xarray as xr

        vol, data = xr.align(cell_volume, da, join="exact", copy=False)
        mass = data.fillna(0).astype(np.float64) * vol.astype(np.float64)

        spatial_dims = [d for d in vol.dims if d in mass.dims]
        if spatial_dims:
            mass = mass.sum(dim=spatial_dims, skipna=True)

        return mass

    def sum_DataArray_list(
        self,
        DataArray_dict,
        DataArray_ls=None,
        variable=None,
        start_date=None,
        end_date=None,
        freq_time=None,
        time_name="time",
        align_mode="pad",
        nearest_tol_time=None,
        dim_res_dict=None,
        coord_tolerance_dict=None,
        snap_mode="median",
        mask_mode="input0",
        sim_description=None,
        Verbose=True,
    ):
        """
        Sum a list of xarray DataArrays, with the same or different time step.
        If start_date, end_date, or freq_time are specified time dimension is reconstructed.

        DataArray_dict:      dictionary of {idx: {"topo": ..., "variable": ...}, ...}
        variable:            string, name of variable to be used as key for DataArray_dict
        DataArray_ls:        list of xarray DataArray
        start_date:          np.datetime64, start of reconstructed time dimension
        end_date:            np.datetime64, end of reconstructed time dimension
        freq_time:           np.timedelta64, frequency of reconstructed time dimension
        time_name:           string, name of time dimension of all DataArray present in DataArray_ls
        align_mode:          string, mode of selecting timestamp in reconstructed sum ("pad"|"nearest"|"exact")
        nearest_tol_time:    np.timedelta64, max distance for "nearest" (e.g., np.timedelta64(3,'h'))
        dim_res_dict:        dict {"dim": float32} resolution of each dimension. Will be inferred if None or {}
        mask_mode:           string, mode of masking final sum ("input0"|"union"|"intersection")
        sim_description:     string, description of simulation to be included in netcdf attributes
        """
        import hashlib
        import os
        import tempfile
        from datetime import datetime

        import numpy as np
        import xarray as xr

        # Convert time_name to string to allow indexing
        if not isinstance(time_name, str):
            time_name = str(time_name)

        valid_mask_modes = {"input0", "union", "intersection"}
        if mask_mode not in valid_mask_modes:
            raise ValueError(f"mask_mode must be one of {valid_mask_modes}, got {mask_mode!r}")

        if DataArray_dict is None and DataArray_ls is None:
            raise ValueError("Either DataArray_dict or DataArray_ls must be provided.")

        coord_tolerance_dict = {} if coord_tolerance_dict is None else coord_tolerance_dict

        # Normalize inputs
        DataArray_ls, index_keys = self._normalize_inputs(DataArray_dict, DataArray_ls, variable)
        DataArray_ls = self._normalize_depth_and_specie(DataArray_ls)
        DataArray_ls = [self._standardize_horizontal_names(da) for da in DataArray_ls]

        ordered_keys = index_keys if index_keys is not None else list(range(len(DataArray_ls)))

        DataArray_dict_work = None
        if DataArray_dict is not None:
            DataArray_dict_work = {}
            for key, da_norm in zip(ordered_keys, DataArray_ls):
                inner = dict(DataArray_dict[key])
                inner[variable] = da_norm
                if inner.get("topo") is not None:
                    inner["topo"] = self._standardize_horizontal_names(inner["topo"])
                DataArray_dict_work[key] = inner

        if not all(time_name in da.dims for da in DataArray_ls):
            missing = [i for i, da in enumerate(DataArray_ls) if time_name not in da.dims]
            raise ValueError(f'All DataArrays must contain time dimension "{time_name}". Missing in indexes: {missing}')

        if DataArray_dict_work is not None:
            nan_counts_inputs = {
                key: int(DataArray_dict_work[key][variable].isnull().sum().item())
                for key in ordered_keys}
        else:
            nan_counts_inputs = {
                i: int(da.isnull().sum().item())
                for i, da in enumerate(DataArray_ls)}

        if len(DataArray_ls) < 1:
            raise ValueError("Empty DataArray_ls")
        if len(DataArray_ls) == 1:
            if Verbose:
                print("len(DataArray_ls) is 1, returning DataArray_ls[0]")
            return DataArray_ls[0], None

        ### Dimension / metadata checks
        if Verbose:
            print("Checking input DataArray dimensions")

        all_dims_per_da = [set(da.dims) for da in DataArray_ls]
        common_dims = set.intersection(*all_dims_per_da)
        extra_dims = set.union(*all_dims_per_da) - common_dims
        if extra_dims:
            for dim in sorted(extra_dims):
                for idx, da in enumerate(DataArray_ls):
                    if dim in da.dims and Verbose:
                        print(f'Extra dimension "{dim}" in array {idx}')
            raise ValueError("Uncommon dimensions are present in DataArray_ls")

        names = [da.name for da in DataArray_ls]
        if any(n is None for n in names):
            raise ValueError(f"All DataArrays must have a name. Got: {names}")
        ref_name = next((n for n in names if n is not None), "data")

        mismatch = [n for n in names if n is not None and n != ref_name]
        if mismatch:
            raise ValueError(f"Mismatched DataArray names: {names}. Expected all '{ref_name}'.")
        if Verbose:
            print(f"var_name: {ref_name}")

        nontime_dims = set.union(*all_dims_per_da)
        nontime_dims.discard(time_name)

        # check equality of dimensions within tolerance
        canonical_dims_dict, need_interpolation = self._canonical_nontime_dims(
            DataArray_ls=DataArray_ls,
            nontime_dims=nontime_dims,
            coord_tolerance_dict=coord_tolerance_dict,
            snap_mode=snap_mode,
            dim_res_dict=dim_res_dict,
            Verbose=Verbose,)

        # Find common attributes to be added in Final_sum
        common_attrs = DataArray_ls[0].attrs.copy()
        for da in DataArray_ls[1:]:
            common_attrs = {
                key: value
                for key, value in common_attrs.items()
                if da.attrs.get(key) == value}

        H_can_out = None
        nan_counts_interp = None
        common_topo_for_mass = None

        # No interpolation path: try to get one common topo for mass check
        if not any(need_interpolation.values()):
            if DataArray_dict_work is not None:
                try:
                    common_topo_for_mass = self._extract_common_topo_from_dict(DataArray_dict_work)
                except ValueError as e:
                    common_topo_for_mass = None
                    if Verbose:
                        print(f"Skipping mass-based conservation check: {e}")

        ### Interpolation path
        if any(need_interpolation.values()):
            import xesmf as xe
            if DataArray_dict_work is None:
                raise ValueError("DataArray_dict {idx: {'var','topo'}, ...} must be specified for interpolation")

            if "latitude" not in canonical_dims_dict or "longitude" not in canonical_dims_dict:
                raise ValueError("Interpolation requires canonical 'latitude' and 'longitude' dimensions")

            lat_can = canonical_dims_dict["latitude"]
            lon_can = canonical_dims_dict["longitude"]

            need_horizontal_interp = any(
                bool(need_interpolation.get(k, False)) for k in ("latitude", "longitude"))
            depth_top_can = canonical_dims_dict.get("depth", None)
            need_vertical_interp = bool(need_interpolation.get("depth", False)) if depth_top_can is not None else False

            unsupported_interp_dims = {
                dim for dim, needed in need_interpolation.items()
                if needed and dim not in {"latitude", "longitude", "depth"}}
            if unsupported_interp_dims:
                raise ValueError(
                    f"Interpolation is only supported for latitude/longitude "
                    f"(and depth is reserved for future vertical remap). "
                    f"Unsupported dims requiring interpolation: {sorted(unsupported_interp_dims)}")

            ### Helpers for "regrid_topography_conservative"
            def _maybe_depth_dim(da, candidates=("depth", "z", "lev", "level")):
                """
                Return the first matching depth-like dimension name present in the DataArray.
                """
                for nm in candidates:
                    if nm in da.dims:
                        return nm
                return None

            def _centers_to_edges_1d(c):
                """
                Convert 1D cell centers to edges along an axis (used for conservative grids).
                """
                c = np.asarray(c, dtype=np.float64)
                if c.size < 2:
                    half = 0.5 * (c[0] if c[0] != 0 else 1.0)
                    return np.array([c[0] - half, c[0] + half], dtype=np.float64)
                mid = 0.5 * (c[:-1] + c[1:])
                first = c[0] - (mid[0] - c[0])
                last = c[-1] + (c[-1] - mid[-1])
                return np.concatenate([[first], mid, [last]])

            def _hash_grid(lon_in, lat_in, lon_out, lat_out, method="conservative"):
                """
                Compute a short hash identifier for a source/destination grid pair plus method.
                """
                h = hashlib.sha1()
                for arr in (lon_in, lat_in, lon_out, lat_out):
                    a = np.asarray(arr, dtype=np.float64)
                    h.update(a.tobytes())
                h.update(method.encode())
                return h.hexdigest()[:10]

            def _make_rect_grid(lon, lat):
                """
                Build a rectilinear xarray Dataset with cell centers and edges for xESMF.
                """
                lon = np.asarray(lon, dtype=np.float64)
                lat = np.asarray(lat, dtype=np.float64)
                lon_b = _centers_to_edges_1d(lon)
                lat_b = _centers_to_edges_1d(lat)
                # NB: dim names 'lon','lat','lon_b','lat_b' are what xESMF expects
                return xr.Dataset(
                    coords=dict(
                        lon=(["lon"], lon),
                        lat=(["lat"], lat),
                        lon_b=(["lon_b"], lon_b),
                        lat_b=(["lat_b"], lat_b),))

            def get_regridder(src_lon, src_lat, dst_lon, dst_lat, *, method="conservative", periodic=True, weights_dir=None, prefix="weights"):
                """
                Create (or reuse cached) xESMF Regridder for given src/dst grids and method.
                Optionally cache weights on disk under weights_dir.
                """
                src_grid = _make_rect_grid(src_lon, src_lat)
                dst_grid = _make_rect_grid(dst_lon, dst_lat)
                cache_id = _hash_grid(src_lon, src_lat, dst_lon, dst_lat, method=method)
                weights_path = None
                reuse = False
                if weights_dir:
                    os.makedirs(weights_dir, exist_ok=True)
                    weights_path = os.path.join(weights_dir, f"{prefix}_{method}_{cache_id}.nc")
                    reuse = os.path.exists(weights_path)
                R = xe.Regridder(
                    src_grid,
                    dst_grid,
                    method=method,
                    periodic=periodic,
                    filename=weights_path,
                    reuse_weights=reuse,)
                if weights_path is not None and not reuse:
                    R.to_netcdf(weights_path)
                return R, weights_path

            def regrid_topography_conservative(H_src, lat_out, lon_out, *, periodic=False, weights_dir=None):
                """
                Conservative remap of topography to (lat_out, lon_out).
                H_src must have dims named ('latitude','longitude') (any order). Extra dims broadcast.
                - Caches weights under weights_dir (default: system temp) unless weights_path is given.
                - Returns DataArray on the same non-horizontal dims as H_src, but with
                  coords named ('latitude','longitude') on output.
                """
                lat_out = np.asarray(lat_out, dtype=np.float64)
                lon_out = np.asarray(lon_out, dtype=np.float64)

                if "latitude" not in H_src.dims or "longitude" not in H_src.dims:
                    raise ValueError("H_src must have dims 'latitude' and 'longitude'")

                other = [d for d in H_src.dims if d not in ("latitude", "longitude")]
                H_ord = H_src.transpose(*other, "latitude", "longitude", missing_dims="ignore")

                topo_R, _ = get_regridder(
                    src_lon=H_src["longitude"].values,
                    src_lat=H_src["latitude"].values,
                    dst_lon=lon_out,
                    dst_lat=lat_out,
                    method="conservative",
                    periodic=periodic,
                    weights_dir=weights_dir,
                    prefix="topo_conservative",)

                H_regridded = topo_R(
                    H_ord.rename({"latitude": "lat", "longitude": "lon"})
                ).rename({"lat": "latitude", "lon": "longitude"})

                H_back = H_regridded.transpose(*other, "latitude", "longitude", missing_dims="ignore")
                return H_back.assign_coords(
                    latitude=("latitude", lat_out),
                    longitude=("longitude", lon_out),)

            def _safe_assign_depth(da, depth_dim, coord_vals):
                """
                Assign depth coordinates to a DataArray, enforcing size consistency
                between data along depth_dim and coord_vals.
                """
                coord_vals = np.asarray(coord_vals)
                if da.sizes[depth_dim] != coord_vals.size:
                    raise ValueError(
                        f"conflicting sizes for '{depth_dim}': data={da.sizes[depth_dim]} vs coord={coord_vals.size}")
                return da.assign_coords({depth_dim: (depth_dim, coord_vals)})

            def _regrid_horizontal_conservative_mass(mass_src, lon_out, lat_out, *, periodic=True, weights_dir=None, depth_dim=None):

                lon_name, lat_name = "longitude", "latitude"

                if depth_dim is None:
                    depth_dim = _maybe_depth_dim(mass_src)

                dims = list(mass_src.dims)
                front = [d for d in dims if d not in (lat_name, lon_name)]
                if depth_dim in front:
                    front.remove(depth_dim)
                    lead = front + [depth_dim]
                else:
                    lead = front

                M = mass_src.transpose(
                    *front,
                    *([depth_dim] if depth_dim else []),
                    lat_name,
                    lon_name,
                    missing_dims="ignore")

                def _edges_1d(x, periodic=False):
                    """
                    Convert 1D centers to edges, optionally periodic in longitude.
                    """
                    x = np.asarray(x, dtype=np.float64)
                    if x.size == 1:
                        half = 0.5 * (x[0] if x[0] != 0 else 1.0)
                        return np.array([x[0] - half, x[0] + half], dtype=np.float64)

                    dx = np.diff(x)
                    mids = x[:-1] + 0.5 * dx
                    first = x[0] - 0.5 * dx[0]
                    last = x[-1] + 0.5 * dx[-1]
                    e = np.concatenate(([first], mids, [last]))
                    if periodic:
                        span = x[-1] - x[0] + dx[-1]
                        e[0] = x[0] - 0.5 * dx[0]
                        e[-1] = e[0] + span
                    return e

                def make_rect_grid(lon_1d, lat_1d, *, periodic=False):
                    """
                    Create xESMF rectilinear grid Dataset from 1D lon/lat.
                    """
                    lon = np.asarray(lon_1d, dtype=np.float64)
                    lat = np.asarray(lat_1d, dtype=np.float64)
                    return xr.Dataset(
                        coords=dict(
                            lon=(("lon",), lon),
                            lat=(("lat",), lat),
                            lon_b=(("lon_b",), _edges_1d(lon, periodic=periodic)),
                            lat_b=(("lat_b",), _edges_1d(lat, periodic=False)),))

                src_grid = make_rect_grid(
                    np.asarray(M[lon_name].values),
                    np.asarray(M[lat_name].values),
                    periodic=periodic)

                dst_grid = make_rect_grid(
                    np.asarray(lon_out),
                    np.asarray(lat_out),
                    periodic=periodic)

                method = "conservative"
                weights_path, reuse = None, False
                if weights_dir is not None:
                    gid = _hash_grid(
                        np.asarray(M[lon_name].values),
                        np.asarray(M[lat_name].values),
                        np.asarray(lon_out),
                        np.asarray(lat_out),
                        method=method,)
                    os.makedirs(weights_dir, exist_ok=True)
                    weights_path = os.path.join(weights_dir, f"mass_{method}_{gid}.nc")
                    reuse = os.path.exists(weights_path)

                R = xe.Regridder(
                    src_grid,
                    dst_grid,
                    method=method,
                    periodic=periodic,
                    filename=weights_path,
                    reuse_weights=reuse,)
                if weights_path is not None and not reuse:
                    R.to_netcdf(weights_path)

                if lead:
                    # Save lead-dim metadata BEFORE stacking
                    M0 = M.transpose(*lead, lat_name, lon_name)
                    lead_sizes = [M0.sizes[d] for d in lead]
                    lead_coords = {
                        d: (np.asarray(M0.coords[d].values) if d in M0.coords else np.arange(M0.sizes[d]))
                        for d in lead}

                    # Stack for xESMF, but rebuild explicitly after regrid
                    M_stacked = M0.stack(_lead=lead).transpose("_lead", lat_name, lon_name)

                    out_stacked = R(M_stacked)

                    rename_map = {}
                    if "lat" in out_stacked.dims:
                        rename_map["lat"] = lat_name
                    if "lon" in out_stacked.dims:
                        rename_map["lon"] = lon_name
                    if rename_map:
                        out_stacked = out_stacked.rename(rename_map)

                    out_stacked = out_stacked.transpose("_lead", lat_name, lon_name)

                    out_vals = np.asarray(out_stacked.data).reshape(
                        *lead_sizes,
                        len(lat_out), len(lon_out),)

                    out = xr.DataArray(
                        out_vals,
                        dims=tuple(lead) + (lat_name, lon_name),
                        coords={
                            **lead_coords,
                            lat_name: np.asarray(lat_out),
                            lon_name: np.asarray(lon_out),
                        }, name=mass_src.name,)

                    # Preserve scalar/non-dimension coords if any
                    scalar_coords = {
                        c: mass_src.coords[c]
                        for c in mass_src.coords
                        if c not in mass_src.dims and c not in out.coords}
                    if scalar_coords:
                        out = out.assign_coords(scalar_coords)

                else:
                    out = R(M)

                    rename_map = {}
                    if "lat" in out.dims:
                        rename_map["lat"] = lat_name
                    if "lon" in out.dims:
                        rename_map["lon"] = lon_name
                    if rename_map:
                        out = out.rename(rename_map)

                    out = out.transpose(lat_name, lon_name)
                    out = out.assign_coords({
                        lat_name: (lat_name, lat_out),
                        lon_name: (lon_name, lon_out),})

                out.attrs.update(mass_src.attrs)
                out.attrs["regrid_method"] = "conservative"
                return out

            ### Horizontal and vertical interpolation ###
            def regrid3d_conservative_with_topo(
                da,
                topo_src,
                lat_can,
                lon_can,
                depth_top_can, # depth_top_can reserved for future vertical conservative remap
                need_horizontal_interp,
                need_vertical_interp,
                *,
                periodic_lon=True,
                depth_dim="depth",
                weights_dir=None,
                idx=None,
                idx_tot=None,
                Verbose=True,
            ):
                """
                Apply mass-conservative regrid to 3D/2D concentration field using topography:
                conservative xESMF horizontally, thickness-based vertically (not yet supported).
                """
                if Verbose:
                    print(f"Interpolating array {idx} out of {idx_tot - 1}")

                if topo_src is None:
                    raise ValueError("topo_src not specified")

                topo_src = topo_src.astype(np.float64).clip(min=0.0)

                dd = depth_dim if depth_dim in da.dims else None
                has_depth = dd is not None

                A_src = self._cell_areas_from_1d(
                    da["latitude"].values,
                    da["longitude"].values,
                    lat_name="latitude", lon_name="longitude",)

                if has_depth:
                    depth_top_src = np.asarray(da[dd].values, dtype=np.float64)
                    z_edges_src_col = xr.concat(
                        [
                            xr.DataArray(
                                depth_top_src,
                                dims=(dd,),
                                coords={dd: np.arange(depth_top_src.size)},
                            ).broadcast_like(topo_src),
                            topo_src.astype(np.float64).expand_dims({dd: [depth_top_src.size]}),
                        ],
                        dim=dd,).rename({dd: "depth_edge"}).transpose("depth_edge", "latitude", "longitude")

                    dz_src = z_edges_src_col.diff("depth_edge").rename({"depth_edge": dd}).clip(min=0.0).astype(np.float64)
                    dz_src = _safe_assign_depth(dz_src, dd, da[dd].values)

                    vol_src = dz_src * A_src
                    mass_src = da.fillna(0).astype(np.float64) * vol_src
                else:
                    vol_src = A_src * topo_src
                    mass_src = da.fillna(0).astype(np.float64) * vol_src

                if not need_horizontal_interp and not need_vertical_interp:
                    if Verbose:
                        print("No interpolation necessary")
                    return da

                if need_vertical_interp:
                    raise ValueError("Vertical interpolation not yet supported")

                mass_h = _regrid_horizontal_conservative_mass(
                    mass_src=mass_src,
                    lon_out=np.asarray(lon_can),
                    lat_out=np.asarray(lat_can),
                    periodic=periodic_lon,
                    weights_dir=weights_dir,
                    depth_dim=dd,)
                vol_h = _regrid_horizontal_conservative_mass(
                    mass_src=vol_src,
                    lon_out=np.asarray(lon_can),
                    lat_out=np.asarray(lat_can),
                    periodic=periodic_lon,
                    weights_dir=weights_dir,
                    depth_dim=dd,)

                if has_depth:
                    vol_h = _safe_assign_depth(vol_h, depth_dim, da[depth_dim].values)
                    mass_h = _safe_assign_depth(mass_h, depth_dim, da[depth_dim].values)

                spatial_dims = [d for d in mass_src.dims if d in ("depth", "latitude", "longitude")]
                m_in = mass_src.sum(dim=spatial_dims, skipna=True)
                m_out = mass_h.sum(dim=spatial_dims, skipna=True)
                xr.testing.assert_allclose(m_out, m_in, rtol=1e-6, atol=1e-12)

                if Verbose:
                    print(f"mass before regrid: {float(m_in.fillna(0).sum().item())}")
                    print(f"mass after regrid:  {float(m_out.fillna(0).sum().item())}")

                mass_fin = mass_h
                vol_fin = vol_h

                conc_out = (mass_fin / vol_fin.where(vol_fin > 0)).where(vol_fin > 0)

                if Verbose:
                    check_conc_original = np.array(da.sum())
                    check_conc_out = np.array(conc_out.sum())
                    print(f"check_conc_original: {check_conc_original}")
                    print(f"check_conc_out: {check_conc_out}")

                    if has_depth:
                        Cvw_in = float((da * dz_src * A_src).sum() / (dz_src * A_src).sum())
                    else:
                        Cvw_in = float(mass_src.sum() / vol_src.sum())
                    Cvw_out = float(mass_fin.sum() / vol_fin.sum())
                    print(f"Cvw_in: {Cvw_in}")
                    print(f"Cvw_out: {Cvw_out}")

                lead = [d for d in da.dims if d not in ("latitude", "longitude") and (dd is None or d != dd)]
                if has_depth:
                    conc_out = conc_out.transpose(*lead, dd, "latitude", "longitude", missing_dims="ignore")
                else:
                    conc_out = conc_out.transpose(*lead, "latitude", "longitude", missing_dims="ignore")

                conc_out.name = getattr(da, "name", "concentration")
                conc_out.attrs.update(da.attrs)
                conc_out.attrs["regrid_note"] = (
                    "mass-conserving: xESMF conservative (horizontal); "
                    "vertical conservative remap not yet implemented")
                return conc_out

            with tempfile.TemporaryDirectory() as tmpdir:
                # Conservative regrid of MASS

                DataArray_ls = [
                    regrid3d_conservative_with_topo(
                        da=DataArray_dict_work[key][variable],    # concentration [mass/volume], dims (..., depth, latitude, longitude)
                        topo_src=DataArray_dict_work[key]["topo"],# 2-D topography [m] (bottom on source), positve downword, dims include ('latitude','longitude')
                        lat_can=lat_can,                          # 1-D canonical lat
                        lon_can=lon_can,                          # 1-D canonical TOPS (m, positive down; include 0 if surface)
                        depth_top_can=depth_top_can,
                        need_horizontal_interp=need_horizontal_interp,
                        need_vertical_interp=need_vertical_interp,
                        periodic_lon=True,
                        depth_dim="depth",
                        weights_dir=tmpdir,
                        idx=key,
                        idx_tot=len(ordered_keys),
                        Verbose=Verbose,)
                    for key in ordered_keys]

                nan_counts_interp = {
                    key: int(np.count_nonzero(np.isnan(da.values)))
                    for key, da in zip(ordered_keys, DataArray_ls)}

                first_idx = ordered_keys[0]
                H_can_out = regrid_topography_conservative(
                    H_src=DataArray_dict_work[first_idx]["topo"],
                    lat_out=lat_can,
                    lon_out=lon_can,
                    periodic=True,
                    weights_dir=tmpdir,
                ).where(lambda x: x > 0, 0.0)

                common_topo_for_mass = H_can_out

        else:
            if Verbose:
                print("No interpolation needed")

        ### NaN-count consistency after interpolation
        if nan_counts_interp is not None:
            import math

            def report_interp_nan_changes(nan_counts_before, nan_counts_after, threshold_pct=10.0):
                print("Index | NaN before interp -> after interp | d(after-before) | % change")
                print("-" * 84)
                offenders = []

                all_keys = sorted(set(nan_counts_before) | set(nan_counts_after))
                for k in all_keys:
                    before = nan_counts_before.get(k, 0)
                    after = nan_counts_after.get(k, 0)
                    delta = after - before

                    # % change with explicit sign; handle before==0
                    if before == 0:
                        if after == 0:
                            pct = 0.0
                            pct_str = f"{pct:+.2f}%"
                        else:
                            pct = math.inf
                            pct_str = "+inf%"
                    else:
                        pct = (delta / before) * 100.0
                        pct_str = f"{pct:+.2f}%"

                    print(f"{k:5} | {before:18} -> {after:<18} | {delta:16} | {pct_str:>8}")

                    if math.isinf(pct) or abs(pct) > threshold_pct:
                        offenders.append((k, before, after, pct))

                if offenders:
                    details = ", ".join(
                        f"idx={k} (before={b}, after={a}, change="
                        f"{'+' if (math.isinf(p) or p >= 0) else ''}"
                        f"{'inf' if math.isinf(p) else f'{p:.2f}%'}"
                        f")"
                        for k, b, a, p in offenders)
                    raise ValueError(
                        f"Interpolation NaN count changed by more than {threshold_pct:.1f}% for: {details}")

            if Verbose:
                report_interp_nan_changes(nan_counts_inputs, nan_counts_interp)

        ### Re-check canonical non-time dims after interpolation
        canonical_dims_dict, need_interpolation = self._canonical_nontime_dims(
            DataArray_ls=DataArray_ls,
            nontime_dims=nontime_dims,
            coord_tolerance_dict=None,
            snap_mode=snap_mode,
            dim_res_dict=None,
            Verbose=Verbose,)

        time_equal = False
        if all(time_name in da.dims for da in DataArray_ls):
            tvals = [da[time_name].values for da in DataArray_ls]
            time_equal = all(np.array_equal(tv, tvals[0]) for tv in tvals)

        time_start = datetime.now()
        Final_sum = None

        ### Sum directly or on reconstructed time grid
        if time_equal and (start_date is None and end_date is None and freq_time is None):
            if Verbose:
                print("Time dimensions are all equal in DataArray_ls, skipping target_time setup")
                print("Run sum of DataArray_ls")
            aligned = DataArray_ls
            Final_sum = DataArray_ls[0].fillna(0)
            for da in DataArray_ls[1:]:
                Final_sum = Final_sum + da.fillna(0)
        else:
            if Verbose:
                print("Time dimensions not all equal (or reconstruction requested)")

            ### Infer start_date/end_date if missing
            if start_date is None:
                start_date = np.min([da[time_name].values.min() for da in DataArray_ls])
                if Verbose:
                    print("start_date set from DataArray_ls")
            if end_date is None:
                end_date = np.max([da[time_name].values.max() for da in DataArray_ls])
                if Verbose:
                    print("end_date set from DataArray_ls")

            ### Prefer a regular grid if a minimal step can be inferred; otherwise use union of times
            target_time = None
            if freq_time is None:
                steps = []
                for da in DataArray_ls:
                    t = np.sort(da[time_name].values)
                    if t.size > 1:
                        d = np.diff(t)
                        # filter out non-positive diffs just in case of duplicates
                        d = d[d > np.timedelta64(0, "ns")]
                        if d.size:
                            steps.append(d.min())

                if steps:
                    freq_time = np.min(steps)
                    if Verbose:
                        print("freq_time set from DataArray_ls")
                else:
                    # fallback: non-regular union of all timestamps
                    target_time = np.unique(
                        np.concatenate([da[time_name].values for da in DataArray_ls]))
                    if Verbose:
                        print("freq_time could not be inferred; using union of timestamps")

            if target_time is None:
                #  inclusive of final timestamp adding a timestep at the end
                target_time = np.arange(start_date, end_date + freq_time, freq_time)

            ### Print start_date, end_date, and freq_time
            if Verbose:
                try:
                    print(f"start_date: {np.datetime_as_string(start_date)}")
                    print(f"end_date:   {np.datetime_as_string(end_date)}")
                except Exception:
                    pass
                if freq_time is not None:
                    ns = (freq_time / np.timedelta64(1, "ns")).astype("int64")
                    if ns >= 3_600_000_000_000:
                        print(f"freq_time:  {ns / 3.6e12:.0f} hours")
                    else:
                        print(f"freq_time:  {ns / 6e10:.0f} min")

            ### Reindex DataArrays using target_time
            if Verbose:
                print("Reindex and align DataArrays using target_time")

            reindexed = [
                self._reindex_da(
                    DataArray=da,
                    time_name=time_name,
                    target_time=target_time,
                    nearest_tol_time=nearest_tol_time,
                    align_mode=align_mode,)
                for da in DataArray_ls]

            aligned = xr.align(*reindexed, join="exact", copy=False)

            if Verbose:
                print("Run sum of DataArray_ls")

            Final_sum = aligned[0].fillna(0)
            for da in aligned[1:]:
                Final_sum = Final_sum + da.fillna(0)

            ### Remove depth dimension but keep as coord
            if "depth" in Final_sum.dims and Final_sum.sizes.get("depth", None) == 1:
                depth_val = float(Final_sum["depth"].values)
                Final_sum = Final_sum.squeeze("depth", drop=True)
                Final_sum = Final_sum.assign_coords(depth=depth_val)

            ### Reorder dimensions with time first
            dims = list(Final_sum.dims)
            dims.remove(time_name)
            Final_sum = Final_sum.transpose(time_name, *dims)

        if Final_sum is None:
            raise ValueError("Final_sum is None")

        # Mass-based conservation check before masking
        if common_topo_for_mass is None:
            if Verbose:
                print("Skipping mass-based conservation check: no common topography available.")
        else:
            cell_volume_common = self._build_common_cell_volume(
                da_ref=aligned[0],
                topo_ref=common_topo_for_mass,)

            mass_aligned_parts = None
            for da in aligned:
                m = self._mass_on_common_grid(da, cell_volume_common)
                mass_aligned_parts = m if mass_aligned_parts is None else (mass_aligned_parts + m)

            mass_final = self._mass_on_common_grid(Final_sum, cell_volume_common)

            try:
                xr.testing.assert_allclose(
                    mass_final,
                    mass_aligned_parts,
                    rtol=1e-6,
                    atol=1e-12,)
            except AssertionError as e:
                diff = (mass_final - mass_aligned_parts).astype(np.float64)
                max_abs = float(np.nanmax(np.abs(diff.values)))

                denom = xr.where(np.abs(mass_aligned_parts) > 0, np.abs(mass_aligned_parts), np.nan)
                rel = diff / denom
                max_rel = float(np.nanmax(np.abs(rel.values))) if np.any(np.isfinite(rel.values)) else 0.0

                raise AssertionError(
                    "Mass conservation check failed on aligned grid: "
                    f"max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}") from e

            if Verbose:
                total_mass_inputs = float(mass_aligned_parts.fillna(0).sum().item())
                total_mass_final = float(mass_final.fillna(0).sum().item())
                print(f"Aligned input mass: {total_mass_inputs}")
                print(f"Final_sum mass:     {total_mass_final}")

                if total_mass_inputs == 0.0:
                    pct_change = 0.0 if total_mass_final == 0.0 else np.inf
                else:
                    pct_change = 100.0 * (total_mass_final - total_mass_inputs) / total_mass_inputs

                print(f"Mass relative change: {pct_change:+.6f}%")

        ### Masking
        if Verbose:
            print("Create landmask")

        per_input_valid = [~a.isnull() for a in aligned]
        per_input_valid_any_time = [v.any(dim=time_name) for v in per_input_valid]
        valid_stack = xr.concat(per_input_valid_any_time, dim="part")

        if mask_mode == "input0":
            final_mask = per_input_valid_any_time[0]
            mask_source_value = "input0-any-over-time"
        elif mask_mode == "union":
            final_mask = valid_stack.any("part")
            mask_source_value = "union-any-over-time"
        else:
            final_mask = valid_stack.all("part")
            mask_source_value = "intersection-any-over-time"

        # NaN audit on the SAME object before vs after masking
        Final_sum_before_mask = Final_sum
        nans_before_mask = int(np.count_nonzero(np.isnan(Final_sum_before_mask.values)))

        # Broadcast mask explicitly to Final_sum shape/order
        final_mask_full = final_mask.broadcast_like(Final_sum_before_mask)
        final_mask_full = final_mask_full.transpose(*Final_sum_before_mask.dims)

        Final_sum = Final_sum_before_mask.where(final_mask_full)
        nans_after_mask = int(np.count_nonzero(np.isnan(Final_sum.values)))

        # Masking should never reduce NaNs
        if nans_after_mask < nans_before_mask:
            raise ValueError(
                f"Masking reduced NaNs unexpectedly: before={nans_before_mask}, after={nans_after_mask}")

        # Check that newly created NaNs come only from mask=False where data was previously finite
        added_nan_mask = Final_sum.isnull() & Final_sum_before_mask.notnull()
        expected_added_nan_mask = (~final_mask_full) & Final_sum_before_mask.notnull()

        xr.testing.assert_equal(added_nan_mask, expected_added_nan_mask)

        Final_sum.attrs.update(common_attrs)
        if "sim_description" not in Final_sum.attrs and sim_description is not None:
            Final_sum.attrs["sim_description"] = str(sim_description)
        Final_sum.attrs["mask_source"] = mask_source_value

        if Verbose:
            added_nans = nans_after_mask - nans_before_mask
            print(
                f"Masking NaN check OK: before={nans_before_mask}, "
                f"after={nans_after_mask}, added={added_nans}")

        time_end = datetime.now()
        sum_time = time_end - time_start

        if Verbose:
            print(f"Sum_time (h:min:s): {sum_time}")

        # Pure consistency check only: do not modify Final_sum or H_can_out
        if H_can_out is not None:
            for dim in ("latitude", "longitude"):
                if dim not in Final_sum.coords or dim not in H_can_out.coords:
                    raise ValueError(f"Missing coordinate '{dim}' in Final_sum or H_can_out")
                if not Final_sum[dim].equals(H_can_out[dim]):
                    raise ValueError(f"Coordinate '{dim}' differs between Final_sum and H_can_out")

        return Final_sum, H_can_out

    ##### Helpers for vertical_depth_mean
    @staticmethod
    def _check_dims_values(dims_values):
        '''
        Check if each dimension within DataArrays_ls has the same values for all DataArrays in the list

        dims_values:        list of dict, [{dimension name: dimension values}, ...]
        '''
        # Get the intersection of keys from all dictionaries
        common_keys = set.intersection(*(set(d.keys()) for d in dims_values))
        if len(common_keys) == 0:
            raise ValueError('No common dimensions are present in DataArray_ls')
        else:
            pass

        for key in common_keys:
            # Get the value associated with the key in the first dictionary
            value = dims_values[0][key]
            # Check if all dictionaries have the same value for this key
            if not all(np.array_equal(d[key], value) for d in dims_values):
                raise ValueError(f'Dimension "{key}" has different values across DataArray_ls')
            else:
                pass

    @staticmethod
    def _rename_dimensions(DataArray):
        '''
        Rename latitude/longitude of xarray dataarray to standard format
        '''
        if "latitude" in DataArray.dims:
            pass
        else:
            if "lat" in DataArray.dims:
                DataArray = DataArray.rename({'lat': 'latitude'})
            elif "y" in DataArray.dims:
                DataArray = DataArray.rename({'y': 'latitude'})
            else:
                raise ValueError("Unknown spatial lat coordinates")

        if "longitude" in DataArray.dims:
            pass
        else:
            if "lon" in DataArray.dims:
                DataArray = DataArray.rename({'lon': 'longitude'})
            elif "x" in DataArray.dims:
                DataArray = DataArray.rename({'x': 'longitude'})
            else:
                raise ValueError("Unknown spatial lon coordinates")

        DataArray['latitude'] = DataArray['latitude'].assign_attrs(
                                standard_name="latitude",
                                long_name="latitude",
                                units="degrees_north",
                                axis="Y",)
        DataArray['longitude'] = DataArray['longitude'].assign_attrs(
                                standard_name="longitude",
                                long_name="longitude",
                                units="degrees_east",
                                axis="X",)
        return(DataArray)

    def vertical_depth_mean(self,
                            Dataset,
                            Topograpy_DA = None,
                            variable_name = None,
                            topograpy_name = None,
                            save_file = True,
                            file_output_path = None,
                            file_output_name = None
                            ):
        '''
        Calculate the weighted average over "depth" for a concentration dataarray using the bathimerty to calculate average weights.
        Use with outputs of "calculate_water_sediment_conc" and " write_netcdf_chemical_density_map" functions
        Depth must contain 0, and its value indicate the upper limit of the vertical level

        Dataset:           xarray DataSet, containing concentration (and topography) dataarray
            * latitude      (latitude) float32
            * longitude     (longitude) float32
            * depth         (depth) float32 (Expected like [-2, -1, 0])
            * other dims
        Topograpy_DA :     xarray DataArray, with topograpy corresponding to Dataset (positive downword)
            * latitude      (latitude) float32
            * longitude     (longitude) float32
        variable_name:     string, name of variable in DataSet to be averaged
        topograpy_name:    string, name of topograpy variable in Dataset
        save_file:         boolean, select if averege file is saved
        file_output_path:  string, path of the file to be saved. Must end with /
        file_output_name:  string, name of the average DataArray output file (.nc)
        '''

        import xarray as xr

        if "depth" not in Dataset.dims:
            raise ValueError("depth not in Dataset.dims")

        if save_file is True:
            if not ((file_output_path is not None) & (file_output_name is not None)):
                raise ValueError("file_output_path or file_output_name not specified")

        # Specify bathimetry if not included in Dataset
        if topograpy_name is not None:
            if topograpy_name not in Dataset.data_vars:
                raise ValueError(f"Topography variable '{topograpy_name}' not found in Dataset")
            Bathymetry_DA = Dataset[topograpy_name]
        else:
            if (Topograpy_DA is None) or (not isinstance(Topograpy_DA, xr.DataArray)):
                raise ValueError("topograpy array not in Dataset and not specified")
            Bathymetry_DA = Topograpy_DA

        # Bathimetry is expected positive downward
        if (Bathymetry_DA <= 0).any():
            print("Changed <= 0 to np.nan in bathimetry")
            Bathymetry_DA = xr.where(Bathymetry_DA <= 0,
                                     np.nan,
                                     Bathymetry_DA)

        Conc_DA = Dataset[variable_name]
        Conc_DA = self._rename_dimensions(Conc_DA)
        Bathymetry_DA = self._rename_dimensions(Bathymetry_DA)

        # Check if ["latitude", "longitude"] have the same values
        dims_values = []
        for DataArray in [Conc_DA, Bathymetry_DA]:
            # Initialize dict to store dimension values for this DataArray
            DataArray_dims_values = {}
            # Iterate through ["latitude", "longitude"]
            for dim in ["latitude", "longitude"]:
                # Get the dimension values if the dimension exists in the DataArray
                if dim in DataArray.dims:
                    DataArray_dims_values[dim] = DataArray[dim].values
                else:
                    raise ValueError("Uncommon dimensions are present in Conc_DA and Bathymetry_DA")
            # Append the dimension values for this DataArray to the list
            dims_values.append(DataArray_dims_values)

        self._check_dims_values(dims_values)

        Bathymetry_mask = ((Bathymetry_DA == 0) | np.isnan(Bathymetry_DA))
        Landmask = (Conc_DA.sel(**{"depth": 0})).isnull()
        # Prepare weights for avarage
        # Depth is expected like  [-2, -1, 0], then formatted to ascending positive [-1, 0] -> [0, 1]
        Conc_DA = Conc_DA.assign_coords(
            depth=("depth", np.abs(Conc_DA["depth"].values))
        )
        Conc_DA = Conc_DA.sortby("depth")

        depth_levels = np.array(Conc_DA.depth.values)
        if depth_levels[0] != 0:
            raise ValueError("Depth coordinate must end at 0 (surface level)")

        ### Construct weights
        # depth_levels: 1D numpy array, ascending, with 0 at the surface
        depth_vals = np.asarray(depth_levels, dtype=float)
        abs_depth = np.abs(depth_vals)

        if depth_vals[0] != 0:
            raise ValueError("Depth coordinate must start at 0 for this weighting scheme.")

        # "top" of each layer (in absolute depth)
        depth_top = xr.DataArray(
            abs_depth,
            dims=("depth",),
            coords={"depth": depth_vals},
        )

        # "bottom" of each layer is the next depth; last one left as NaN
        depth_bottom_vals = np.empty_like(abs_depth)
        depth_bottom_vals[:-1] = abs_depth[1:]
        depth_bottom_vals[-1] = np.nan  # last layer handled separately

        depth_bottom = xr.DataArray(
            depth_bottom_vals,
            dims=("depth",),
            coords={"depth": depth_vals},
        )

        # Broadcast bathymetry to have a depth dimension
        # Bathymetry_DA: (latitude, longitude)
        H = Bathymetry_DA  # (latitude, longitude)

        # Broadcast to 3D (depth, latitude, longitude)
        top3, H3 = xr.broadcast(depth_top, H)
        bottom3, _ = xr.broadcast(depth_bottom, H)

        # Layer thickness for all but last layer (last layer uses separate rule)
        layer_thickness = bottom3 - top3

        # Initialize weights array
        weights_array_fin = xr.zeros_like(H3)

        # Masks for "full" vs "partial" coverage for all but last layer
        full = H3 > bottom3                      # bathymetry deeper than layer bottom
        partial = (H3 > top3) & (H3 <= bottom3)  # bathymetry within layer

        # For full layers: thickness / H
        weights_array_fin = xr.where(
            full,
            layer_thickness / H3,
            weights_array_fin
        )

        # For partial layers: (H - top) / H
        weights_array_fin = xr.where(
            partial,
            (H3 - top3) / H3,
            weights_array_fin
        )

        # Handle the last (deepest) layer: index -1
        last_depth = depth_vals[-1]
        top_last = abs_depth[-1]

        H_last = H3.sel(depth=last_depth)
        w_last = xr.where(
            H_last > top_last, # Last depth layer is lower than bathymetry, so weight = 0
            (H_last - top_last) / H_last,  # Last depth layer accounts for water until bathymetry
            0.0
        )

        weights_array_fin.loc[dict(depth=last_depth)] = w_last

        # Apply bathymetry mask (0 or NaN) and clean up
        weights_array_fin = xr.where(
            Bathymetry_mask,
            0,          # bathymetry 0 or NaN -> weight 0
            weights_array_fin
        )

        weights_array_fin = weights_array_fin.fillna(0)

        # Check that weights sum to ~1 where non-zero
        weights_array_check = weights_array_fin.sum(dim="depth")
        weights_array_check = (weights_array_check != 0) & (np.abs(weights_array_check - 1) > 1e-10)

        if weights_array_check.sum() > 0:
            raise ValueError("weights_array sum higher >1 or <0")

        # Conc_DA can be (time, depth, lat, lon) or (depth, lat, lon)
        # weights_array_fin has no time dimension and will broadcast over time automatically
        weighted_avg = (Conc_DA * weights_array_fin).sum(dim="depth", skipna=True)

        # Re-apply landmask and set name
        weighted_avg = weighted_avg.where(~Landmask)
        weighted_avg.name = variable_name

        if save_file is True:
                self._save_masked_DataArray(DataArray_masked = weighted_avg,
                                                  file_output_path = file_output_path,
                                                  file_output_name = file_output_name)
        else:
            return(weighted_avg)


    @staticmethod
    def _get_dataset_size(dataset_path):
        '''
        Load an xarray dataset and return its size in bytes as integer.

        dataset_path:      string, path to the xarray dataset file (e.g., NetCDF format)
        '''
        import xarray as xr

        # Load the dataset in xarray
        ds = xr.open_dataset(dataset_path)
        size_in_bytes = ds.nbytes  # Get the size in bytes
        ds.close()  # Close the dataset to free up memory
        return size_in_bytes

    def divide_datasets_by_size(self, file_paths, max_size=16 * 1024 ** 3):
        '''
        Divide datasets into sub-lists, each not exceeding a specified max size.

        file_paths:       list, paths to xarray dataset files. Must end with "/"
        max_size:         int, maximum size for each sub-list in bytes (default is 16 GB)

        '''
        file_paths = sorted(file_paths, key = self._get_dataset_size)

        sub_lists = []     # List to store all sub-lists
        current_sub_list = []  # Current sub-list of dataset paths
        current_size = 0    # Track current sub-list size

        for file_path in file_paths:
            # Get the size of the dataset
            file_size = self._get_dataset_size(file_path)

            # Check if adding this file exceeds the max size
            if current_size + file_size > max_size:
                # Add the current sub-list to the list of sub-lists
                sub_lists.append(current_sub_list)
                # Start a new sub-list and reset the current size
                current_sub_list = []
                current_size = 0

            # Add the file to the current sub-list
            current_sub_list.append(file_path)
            current_size += file_size

        # Add the last sub-list if it's not empty
        if current_sub_list:
            sub_lists.append(current_sub_list)

        return sub_lists

    @staticmethod
    def _is_compressed_at_level_6(nc_file):
        '''
        Check if simulation files to be concatenated are already compressed.
        If true, files are only stored in .zip archive
        '''
        import h5py
        compression_status = []
        with h5py.File(nc_file, 'r') as f:
            for var_name in f.keys():
                if var_name in ['time', 'trajectory']:
                    continue  # Skip 'time' and 'trajectory'

                dataset = f[var_name]
                compression = dataset.compression
                compression_opts = dataset.compression_opts

                if compression != 'gzip' or compression_opts != 6:
                    compression_status.append(False)
                else:
                    compression_status.append(True)
        if all(compression_status):
            return True
        else:
            return False


    def concat_simulation(self,
                          sim_file_list,
                          simoutputpath,
                          sim_name = "sim",
                          max_size_GB = 16,
                          zip_files = True):
        '''
        Concatenate simulation slices into files not exceeding max_size_GB and compress results into a .zip file.
        A summary file of which slices were included in each concatenated file is saved

        sim_file_list:      list, strings with flinames of simulations files or .zip archives
                                  files already present in simoutputpath are not extracted from
                                 .zip archives
        sim_name:           string, name of simulation
        simoutputpath:      string, folder containing files to be concatenated.
        max_size_GB:        int, max size of concatenated files in GB (Default is 16)
        zip_files:          boolean, select if concatenated files are zipped and the originals deleted
        '''
        import xarray as xr
        import os
        import zipfile
        import warnings

        # Remove files that are not ".nc" or ".zip"
        sim_file_list = [file for file in sim_file_list if file.endswith((".zip", ".nc"))]
        if len(sim_file_list) == 0:
            raise ValueError("No files to be concatenated")
        if not simoutputpath.endswith("/"):
            simoutputpath = simoutputpath + "/"
        if sim_name is None:
            sim_name = "sim"

        # Check if any zip archive of simulation to concatenate is in sim_file_list
        zip_ls = []
        start=datetime.now()
        if any([".zip" in file for file in sim_file_list]):
            for file in sim_file_list:
                if file.endswith(".zip"):
                    zip_ls.append(file)
            # Remove each zip archive from sim_file_list and extract files to concatenate
            sim_file_list = [file for file in sim_file_list if ".zip" not in file]

            for index_zip, zipfilename in enumerate(zip_ls):
                with zipfile.ZipFile((simoutputpath + zipfilename), 'r') as zip_ref:
                    # Get a list of all files in the archive and add it to sim_file_list
                    files_in_archive = zip_ref.namelist()

                print(f"Extracting zip_file {index_zip + 1} out of {len(zip_ls)}")
                # Extract each file to concatenate
                for index_nc, nc_file in enumerate(files_in_archive):
                    sim_file_list.append(nc_file)
                    # Control if nc_file was already present in simoutputpath
                    if not os.path.exists(simoutputpath + nc_file):
                        print(f"Extracting nc_file {index_nc + 1} out of {len(files_in_archive)}")
                        zip_file_path = (simoutputpath + zipfilename)
                        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                                zip_ref.extract(nc_file, simoutputpath)
            end_zip = datetime.now()
            print(f"Extracting time : {end_zip-start}")

            # Flatten sim_file_list in a single list
            sim_file_list = [item for sublist in sim_file_list for item in (sublist if isinstance(sublist, list) else [sublist])]

        file_paths = [simoutputpath + file_name for file_name in sim_file_list]
        # Group .nc files into sublists for concatenation
        concat_parts = []
        concat_parts = self.divide_datasets_by_size(file_paths = file_paths,
                                                   max_size=max_size_GB * 1024 ** 3)

        concat_parts = [sublist for sublist in concat_parts if sublist != []]
        # Separate files to concatenate from files exceeding max size if concatenated
        concat_parts_ls = []
        files_not_concat = []
        for sublist in concat_parts:
            if len(sublist) == 1:
                files_not_concat.append(sublist)
            else:
                concat_parts_ls.append(sublist)
        # Flatten sim_file_list in a single list
        files_not_concat = [item for sublist in files_not_concat for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Write concat_parts_ls to a txt file
        with open(simoutputpath + sim_name+ "_concat_parts_ls.txt", 'w') as file:
            for i, sublist in enumerate(concat_parts_ls):
                # Write name of concatenated file as a header
                file.write(sim_name + f"_concatenated_{i}.nc:\n")
                for item in sublist:
                    item_size = "{:.2f}".format((self._get_dataset_size(item)/1024**3))
                    file_name_size_um = "GB"
                    if float(item_size) < 0.01:
                        item_size = str(float(item_size)*1000)
                        file_name_size_um = "MB"
                    item = item.replace(simoutputpath, "")
                    file.write(f"{item}, {item_size} {file_name_size_um}\n")
                file.write("\n")
            file.write("Files not concatenated:\n")
            for file_name in files_not_concat:
                file_name_size = "{:.2f}".format((self._get_dataset_size(file_name)/1024**3))
                file_name_size_um = "GB"
                if float(file_name_size) < 0.01:
                    file_name_size = str(float(file_name_size)*1000)
                    file_name_size_um = "MB"
                file_name = file_name.replace(simoutputpath, "")
                file.write(f"{file_name}, {file_name_size} {file_name_size_um}\n")
        # Load and concatenate slices
        if zip_files is True:
            if os.path.exists(file_paths[0]):
                compress_type = zipfile.ZIP_STORED if self._is_compressed_at_level_6(file_paths[0]) else zipfile.ZIP_DEFLATED

        concatenated_files = []

        for index_concat, concat_ls in enumerate(concat_parts_ls):
            concat_output_name = (sim_name + f"_concatenated_{index_concat}.nc")
            ds=[]
            print(f"Loading concat_ls {index_concat +1} out of {len(concat_parts_ls)}")
            for nc_file in concat_ls:
                if os.path.exists(nc_file):
                    dsi=xr.open_dataset(nc_file)
                    if len(dsi.time)>0:
                        ds.append(dsi)

            if len(ds) > 1:
                print("Concatenate slices")
                dsconc=xr.concat(ds,dim="trajectory")
                dsconc.attrs['steps_exported'] = max([ds[i].attrs['steps_exported'] if 'steps_exported' in ds[i].attrs
                             else ds[i].attrs.get('steps_output', 0)
                             for i in range(len(ds))])
                print("Save concatenated file")
                dsconc.to_netcdf(simoutputpath+concat_output_name)
                concatenated_files.append((simoutputpath+concat_output_name))
                print("Remove concatenated slices")
                for nc_file in concat_ls:
                    try:
                        os.remove(nc_file)
                    except FileNotFoundError:
                        warnings.warn(f"File not found (nothing to delete): {nc_file}")
                    except IsADirectoryError:
                        warnings.warn(f"Expected a file but found a directory: {nc_file}")
                    except PermissionError as e:
                        warnings.warn(f"Permission denied deleting {nc_file}: {e}")
                    except Exception as e:
                        warnings.warn(f"Could not remove {nc_file}: {e}")
            end=datetime.now()
            print(f"Concatenating time : {end-start}")

        if len(files_not_concat) > 0:
            for nc_file in files_not_concat:
                concatenated_files.append(nc_file)
        if zip_files is True:
            print("Zip concatenated files")


            zip_path = os.path.join(simoutputpath, sim_name + "_concatenated_files.zip")

            with zipfile.ZipFile(zip_path, mode='w', compression=compress_type) as myzip:
                for index_zip, f in enumerate(concatenated_files):
                    print(f"Zipping file {index_zip + 1} out of {len(concatenated_files)}")
                    arcname_f = f.replace(simoutputpath, "")
                    myzip.write(f, arcname=arcname_f)

            for file in os.listdir(simoutputpath):
                if file.endswith(".nc"):
                    os.remove(simoutputpath+"/" + file)

        end_zip=datetime.now()
        print(f"Zip files time :{end_zip-end}")
