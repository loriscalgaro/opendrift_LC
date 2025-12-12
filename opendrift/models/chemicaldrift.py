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
        ('mass_degraded_now', {'dtype': np.float32,
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
        ('mass_volatilized_now', {'dtype': np.float32,
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
        'x_sea_water_velocity': {'fallback': None},
        'y_sea_water_velocity': {'fallback': None},
        'sea_surface_height': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 10000},
        'ocean_vertical_diffusivity': {'fallback': 0.0001, 'profiles': True},
        'sea_water_temperature': {'fallback': 10, 'profiles': True},
        'sea_water_salinity': {'fallback': 34, 'profiles': True},
        'upward_sea_water_velocity': {'fallback': 0},
        'spm': {'fallback': 1},
        'ocean_mixed_layer_thickness': {'fallback': 50},
        'active_sediment_layer_thickness': {'fallback': 0.03}, # TODO - currently not used, redundant with 'chemical:sediment:mixing_depth'
        'doc': {'fallback': 0.0},
        # Variables for dissociation and single process degradation
        'sea_water_ph_reported_on_total_scale':{'fallback': 8.1, 'profiles': True}, # water_pH from CMENS with standard name #
        'pH_sediment':{'fallback': 6.9, 'profiles': False}, # supplied by the user, with pH_sediment as standard name
        'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water':{'fallback': 7.25, 'profiles': True}, # in g/m3 or mg/L from CMENS with standard name
        'mole_concentration_of_dissolved_inorganic_carbon_in_sea_water':{'fallback': 104, 'profiles': True}, # in concentration of carbon in the water (Conc_C) in mol/m3, nedded as ueq/L (conversion: 22.73 ueq/mg_C, MW_C = 12.01 g/mol. # DONE
        # From concentration of carbon in the water (Conc_C) in mol/m3: Conc_CO2 = ((Conc_C*MW_C)*1000)*22.73*1000;
        # from mol_C/m3, *12.01 g_C/mol = g_C/m3, *1000 = mg/m3, * 22.73 ueq/mg = ueq/m3, *1000 = ueq/L
        # default from https://www.soest.hawaii.edu/oceanography/faculty/zeebe_files/Publications/ZeebeWolfEnclp07.pdf, 2.3 mmol/kg
        'solar_irradiance':{'fallback': 241}, # Available in W/m2, in the function it is nedded in Ly/day. TO DO Check UM of input for convertion. 1 Ly = 41868 J/m2 -> 1 Ly/day =  41868 J/m2 / 86400 s = 0.4843 W/m2  # DONE
        'mole_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water':{'fallback': 0, 'profiles': True} # in mmol_carbon/m3 for CMENS. # TO DO *1e-6 to convert into mol/L. #  Concentration of phytoplankton as “mmol/m3 of phytoplankton expressed as carbon”

        }


    def specie_num2name(self,num):
        return self.name_species[num]

    def specie_name2num(self,name):
        num = self.name_species.index(name)
        return num

    def __init__(self, *args, **kwargs):

        # Calling general constructor of parent class
        super(ChemicalDrift, self).__init__(*args, **kwargs)

        self._add_config({
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
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:particle_diameter': {'type': 'float', 'default': 5e-6,
                'min': 0, 'max': 100e-6, 'units': 'm',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Diameter of SPM particles'},
			'chemical:doc_particle_diameter': {'type': 'float', 'default': 5e-6,
			    'min': 0, 'max': 100e-6, 'units': 'm',
			    'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Diameter of DOM aggregates for marine water'}, # https://doi.org/10.1038/246170a0
            'chemical:particle_concentration_half_depth': {'type': 'float', 'default': 20,
                'min': 0, 'max': 100, 'units': 'm',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},
            'chemical:doc_concentration_half_depth': {'type': 'float', 'default': 1000, # TODO: check better
                'min': 0, 'max': 1200, 'units': 'm',                                     # Vertical conc drops more slowly slower than for SPM
                'level': CONFIG_LEVEL_ADVANCED, 'description': ''},                # example: 10.3389/fmars.2017.00436. lower limit around 40 umol/L
            'chemical:particle_diameter_uncertainty': {'type': 'float', 'default': 1e-7,
                'min': 0, 'max': 100e-6, 'units': 'm',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': ''},
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
            'chemical:species:Sediment_irreversible': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle Sediment_irreversible specie'},
            # Transformations
            'chemical:transformations:Kd': {'type': 'float', 'default': 2.0,
                'min': 0, 'max': 1e9, 'units': 'm3/kg',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Water/sediment partitioning coefficient for metals'},
            'chemical:transformations:S0': {'type': 'float', 'default': 0.0,
                'min': 0, 'max': 100, 'units': 'PSU',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Parameter controlling salinity dependency of Kd for metals'},
            'chemical:transformations:Dc': {'type': 'float', 'default': 1.16e-5,                # Simonsen 2019
                'min': 0, 'max': 1e6, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Desorption rate of metals from particles'},
            'chemical:transformations:slow_coeff': {'type': 'float', 'default': 0, #1.2e-7,         # Simonsen 2019
                'min': 0, 'max': 1e6, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Desorption coefficient from slowly reversible fractions'},
            'chemical:transformations:volatilization': {'type': 'bool', 'default': False,
                'description': 'Chemical is evaporated.',
                'level': CONFIG_LEVEL_BASIC},
            'chemical:transformations:degradation': {'type': 'bool', 'default': False,
                'description': 'Chemical mass is degraded.',
                'level': CONFIG_LEVEL_BASIC},
            'chemical:transformations:degradation_mode': {'type': 'enum',
                'enum': ['OverallRateConstants', 'SingleRateConstants'], 'default': 'OverallRateConstants',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Select degradation mode'},
            # Sorption/desorption
            'chemical:transformations:dissociation': {'type': 'enum',
                'enum': ['nondiss','acid', 'base', 'amphoter'], 'default': 'nondiss',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Select dissociation mode'},
            'chemical:transformations:LogKOW': {'type': 'float', 'default': 3.361,          # Naphthalene
                'min': -3, 'max': 10, 'units': 'Log L/Kg',
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
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'pKa of chemical"s conjugated acid'},
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
                'min': -3, 'max': 30, 'units': 'C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Reference temperature of t12_W_tot'},
            'chemical:transformations:DeltaH_kWt': {'type': 'float', 'default': 50000.,     # generic
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Entalpy of t12_W_tot'},
            # Degradation in sediment layer
            'chemical:transformations:t12_S_tot': {'type': 'float', 'default': 5012.4,      # Naphthalene
                'min': 1, 'max': None, 'units': 'hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Half life in sediments, total'},
            'chemical:transformations:Tref_kSt': {'type': 'float', 'default': 25.,          # Naphthalene
                'min': -3, 'max': 30, 'units': 'C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Reference temperature of t12_S_tot'},
            'chemical:transformations:DeltaH_kSt': {'type': 'float', 'default': 50000.,     # generic
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Entalpy of t12_S_tot'},
            # Volatilization
            'chemical:transformations:MolWt': {'type': 'float', 'default': 128.1705,         # Naphthalene
                'min': 50, 'max': 1000, 'units': 'amu',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Molecular weight'},
            'chemical:transformations:Henry': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'atm m3 mol-1',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Henry constant (uses Tref_Slb as Tref)'},
            # vapour pressure
            'chemical:transformations:Vpress': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'Pa',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Vapour pressure'},
            'chemical:transformations:Tref_Vpress': {'type': 'float', 'default': 25.,        # Naphthalene
                'min': None, 'max': None, 'units': 'C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Vapour pressure ref temp'},
            'chemical:transformations:DeltaH_Vpress': {'type': 'float', 'default': 55925.,   # Naphthalene
                'min': -100000., 'max': 150000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of volatilization'},
            # solubility
            'chemical:transformations:Solub': {'type': 'float', 'default': -1,
                'min': None, 'max': None, 'units': 'g/m3',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Solubility'},
            'chemical:transformations:Tref_Solub': {'type': 'float', 'default': 25.,         # Naphthalene
                'min': None, 'max': None, 'units': 'C',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Solubility ref temp'},
            'chemical:transformations:DeltaH_Solub': {'type': 'float', 'default': 25300.,    # Naphthalene
                'min': -100000., 'max': 100000., 'units': 'J/mol',
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Enthalpy of solubilization'},
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
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Correction factor desorption, to calculate sed desorption from SPM desorption (metals only)'},
            'chemical:sediment:porosity': {'type': 'float', 'default': 0.6,
                'min': 0, 'max': 1, 'units': '',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Fraction of sediment volume made of water, adimentional'},
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
            'chemical:sediment:resuspension_critvel': {'type': 'float', 'default': .01,
                'min': 0, 'max': 1, 'units': 'm/s',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Critical velocity of water to resuspend sediments'},
            'chemical:sediment:burial_rate': {'type': 'float', 'default': .00003,   # MacKay
                'min': 0, 'max': 10, 'units': 'm/year',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Rate of sediment burial'},
            'chemical:sediment:buried_leaking_rate': {'type': 'float', 'default': 0,
                'min': 0, 'max': 10, 'units': 's-1',
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Rate of resuspension of buried sediments'},
            #
            'chemical:compound': {'type': 'enum',
                'enum': ['Naphthalene','Phenanthrene','Fluoranthene',
                         'Benzo-a-anthracene','Benzo-a-pyrene','Dibenzo-ah-anthracene',
                         'C1-Naphthalene','Acenaphthene','Acenaphthylene','Fluorene',
                         'Dibenzothiophene','C2-Naphthalene','Anthracene','C3-Naphthalene','C1-Dibenzothiophene',
                         'Pyrene','C1-Phenanthrene','C2-Dibenzothiophene',
                         'C2-Phenanthrene','Benzo-b-fluoranthene','Chrysene',
                         'C3-Dibenzothiophene','C3-Phenanthrene',
                         'Benzo-k-fluoranthene','Benzo-ghi-perylene','Indeno-123cd-pyrene',
                         'Copper','Cadmium','Chromium','Lead','Vanadium','Zinc','Nickel','Tralopyril','Econea',
                         'Nitrogen', 'Alkalinity','Azoxystrobin','Diflufenican','Metconazole','Penconazole','Tebuconazole', 'Metaflumizone',
                         'Tetraconazole','Methiocarb','test', 'Sulfamethoxazole','Trimethoprim','Clindamycin',
                         'Ofloxacin','Metformin','Venlafaxine',None],
                'default': None,
                'level': CONFIG_LEVEL_ESSENTIAL, 'description': 'Name of modelled chemical'},
            # Single process degradation
            # Save each degradation process output
            'chemical:transformations:Save_single_degr_mass': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle save of mass degraded by single mechanism'},
            'chemical:transformations:Photodegradation': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle photodegradation'},
            'chemical:transformations:Biodegradation': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle biodegradation'},
            'chemical:transformations:Hydrolysis': {'type': 'bool', 'default': True,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle hydrolysis'},
            'chemical:transformations:Save_degr_now': {'type': 'bool', 'default': False,
                'level': CONFIG_LEVEL_ADVANCED, 'description': 'Toggle save of of mass degraded during each timestep (True) or cumulative (False). Does not affect mass_degraded and mass_volatilized as they are saved as separate new variables *_now'},
            # Biodegradation
            'chemical:transformations:k_DecayMax_water': {'type': 'float', 'default': 0.054,      # from AQUATOX Database (0.13 1/day)
                'min': 0, 'max': None, 'units': '1/hours',
                'level': CONFIG_LEVEL_ADVANCED, 'description': ' Max first-order rate constant for biodegradation in aerobic condition'},
            'chemical:transformations:k_Anaerobic_water': {'type': 'float', 'default': 0,      # Defalt for no anaerobic biodegradation
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
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Pseudo-first-order acid-catalysed rate constant for a given pH for hydrolysis'},
             'chemical:transformations:k_Base': {'type': 'float', 'default': 0,     # Default: no base catalyzed hydrolysis
                 'min': None, 'max': None, 'units': 'L/mol*h',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Pseudo-first-order base-catalysed rate constant for a given pH'},
             'chemical:transformations:k_Hydr_Uncat': {'type': 'float', 'default': 0,     # Default: no hydrolysis
                 'min': 0, 'max': None, 'units': '1/hours',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Measured first-order hydrolysis rate at pH 7'},
             # Photolysis
             'chemical:transformations:k_Photo': {'type': 'float', 'default': 0,     # Default: no photolysis
                 'min': 0, 'max': None, 'units': '1/hours',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Measured first-order photolysis rate'},
             'chemical:transformations:RadDistr': {'type': 'float', 'default': 1.6,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Radiance distribution function, which is the ratio of the average pathlength to the depth'},
             'chemical:transformations:RadDistr0_ml': {'type': 'float', 'default': 1.6,     # Default from AQUATOX Database
                 'min': 0, 'max': None, 'units': '',
                 'level': CONFIG_LEVEL_ADVANCED, 'description': 'Standard radiance distribution function in the Mixed Layer'},
             'chemical:transformations:RadDistr0_bml': {'type': 'float', 'default': 1.2,     # Default from AQUATOX Database
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

        logger.info( 'Number of species: {}'.format(self.nspecies) )
        for i,sp in enumerate(self.name_species):
            logger.info( '{:>3} {}'.format( i, sp ) )


        logger.info( 'transfer setup: %s' % self.get_config('chemical:transfer_setup'))

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

        # List of additional custom variables to be saved in self.result
        # TODO: These could now be moved to post_run() which should be
        # more robust in case variables are changed during run()

        savelist = ['nspecies',
                    'name_species',
                    'transfer_rates',
                    'ntransformations']

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
        # Initialize specie types
        if self.get_config('chemical:transfer_setup')=='metals':
            self.set_config('chemical:species:LMM',True)
            self.set_config('chemical:species:Particle_reversible', True)
            self.set_config('chemical:species:Particle_slowly_reversible', True)
            self.set_config('chemical:species:Sediment_reversible', True)
            self.set_config('chemical:species:Sediment_slowly_reversible', True)
        elif self.get_config('chemical:transfer_setup')=='137Cs_rev':
            self.set_config('chemical:species:LMM',True)
            self.set_config('chemical:species:Particle_reversible', True)
            self.set_config('chemical:species:Sediment_reversible', True)
        elif self.get_config('chemical:transfer_setup')=='Sandnesfj_Al':
            self.set_config('chemical:species:LMM', False)
            self.set_config('chemical:species:LMMcation', True)
            self.set_config('chemical:species:LMManion', True)
            self.set_config('chemical:species:Humic_colloid', True)
            self.set_config('chemical:species:Polymer', True)
            self.set_config('chemical:species:Particle_reversible', True)
            self.set_config('chemical:species:Sediment_reversible', True)
        elif self.get_config('chemical:transfer_setup')=='organics':
            self.set_config('chemical:species:LMM',True)
            self.set_config('chemical:species:Particle_reversible', True)
            self.set_config('chemical:species:Particle_slowly_reversible', False)
            self.set_config('chemical:species:Sediment_reversible', True)
            self.set_config('chemical:species:Sediment_slowly_reversible', True)
            self.set_config('chemical:species:Humic_colloid', True)
        elif self.get_config('chemical:transfer_setup')=='custom':
            # Do nothing, species must be set manually
            pass
        else:
            logger.error('No valid transfer_setup {}'.format(self.get_config('chemical:transfer_setup')))


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
        if self.get_config('chemical:species:Sediment_irreversible'):
            self.name_species.append('Sediment irreversible')


        if self.get_config('chemical:species:Sediment_slowly_reversible') and \
                    self.get_config('chemical:species:Particle_slowly_reversible'):
            self.set_config('chemical:slowly_fraction', True)
        if self.get_config('chemical:species:Sediment_irreversible') and \
                    self.get_config('chemical:species:Particle_irreversible'):
            self.set_config('chemical:irreversible_fraction', True)


        self.nspecies      = len(self.name_species)

#         logger.info( 'Number of species: {}'.format(self.nspecies) )
#         for i,sp in enumerate(self.name_species):
#             logger.info( '{:>3} {}'.format( i, sp ))


    def seed_elements(self, *args, **kwargs):

        if hasattr(self,'name_species') == False:
            self.init_species()
            self.init_transfer_rates()


        if 'number' in kwargs:
            num_elements = kwargs['number']
        else:
            num_elements = self.get_config('seed:number')

        if 'specie' in kwargs:
            # print('num_elements', num_elements)
            # try:
            #     print('len specie:',len(kwargs['specie']))
            # except:
            #     print('specie:',kwargs['specie'])

            init_specie = np.ones(num_elements,dtype=int)
            init_specie[:] = kwargs['specie']

        else:

            # Set initial partitioning
            if 'particle_fraction' in kwargs:
                particle_frac = kwargs['particle_fraction']
            else:
                particle_frac = self.get_config('seed:particle_fraction')

            if 'LMM_fraction' in kwargs:
                lmm_frac = kwargs['LMM_fraction']
            else:
                lmm_frac = self.get_config('seed:LMM_fraction')

            if not lmm_frac + particle_frac == 1.:
                logger.error('Fraction does not sum up to 1: %s' % str(lmm_frac+particle_frac) )
                logger.error('LMM fraction: %s ' % str(lmm_frac))
                logger.error( 'Particle fraction %s '% str(particle_frac) )
                raise ValueError('Illegal specie fraction combination : ' + str(lmm_frac) + ' '+ str(particle_frac) )

            init_specie = np.ones(num_elements, int)

            dissolved=np.random.rand(num_elements)<lmm_frac
            if self.get_config('chemical:transfer_setup')=='Sandnesfj_Al':
                init_specie[dissolved]=self.num_lmmcation
            else:
                init_specie[dissolved]=self.num_lmm
            init_specie[~dissolved]=self.num_prev
            kwargs['specie'] = init_specie

        logger.debug('Initial partitioning:')
        for i,sp in enumerate(self.name_species):
            logger.debug( '{:>9} {:>3} {:24} '.format(  np.sum(init_specie==i), i, sp ) )

        # Set initial particle size
        if 'diameter' in kwargs:
            diameter = kwargs['diameter']
        else:
            diameter = self.get_config('chemical:particle_diameter')

        std = self.get_config('chemical:particle_diameter_uncertainty')

        init_diam = np.zeros(num_elements,float)
        init_diam[init_specie==self.num_prev] = diameter + np.random.normal(0, std, sum(init_specie==self.num_prev))
        kwargs['diameter'] = init_diam


        super(ChemicalDrift, self).seed_elements(*args, **kwargs)

    def tempcorr(self,mode,DeltaH,T_C,Tref_C):
        ''' Temperature correction using Arrhenius or Q10 method
        '''
        if mode == 'Arrhenius':
            R = 8.3145 # J/(mol*K)
            T_K = T_C + 273.15
            Tref_K = Tref_C + 273.15
            corr = np.exp(-(DeltaH/R)*(1/T_K - 1/Tref_K))
        elif mode =='Q10':
            corr = 2**((T_C - Tref_C)/10)
        return corr

    def salinitycorr(self,Setschenow,Temperature,Salinity):
        ''' Salinity correction
        '''
        # Setschenow constant for the given chemical (L/mol)
        # Salinity   (PSU =g/Kg)
        # Temperature (Celsius)

        MWsalt = 68.35 # average mass of sea water salt (g/mol) Schwarzenbach Gschwend Imboden Environmental Organic Chemistry

        Dens_sw = self.sea_water_density(T=Temperature, S=Salinity)*1e-3 # (Kg/L)

        # ConcSalt= (Salinitypsu/MWsalt)∙Dens_sw
        #         = (     g/Kg    /    g/mol  )∙  Kg/L
        #         = mol/Kg ∙ Kg/L = mol/L

        ConcSalt = (Salinity/MWsalt)*Dens_sw

        # Log(Kd_fin)=(Setschenow ∙ ConcSalt)+Log(Kd_T)
        # Kd_fin = 10^(Setschenow ∙ ConcSalt) * Kd_T

        corr = 10**(Setschenow*ConcSalt)

        return corr

### Functions to update partitioning coefficients

    def calc_KOC_sedcorr(self, KOC_sed_initial, KOC_sed_n, pKa_acid, pKa_base, KOW, pH_sed, diss,
                         KOC_sed_acid, KOC_sed_base):
        ''' Calculate correction of KOC due to pH of sediments
        '''

        if diss == 'acid':
            Phi_n_sed = 1/(1+10**(pH_sed-pKa_acid))
            Phi_diss_sed = 1-Phi_n_sed
            KOC_sed_updated = (KOC_sed_n*Phi_n_sed)+(Phi_diss_sed*KOC_sed_acid)

            KOC_sedcorr = KOC_sed_updated/KOC_sed_initial

        elif diss == 'base':
            # Undissociated form is positively charged
            Phi_n_sed = 1/(1+10**(pH_sed-pKa_base))
            # Dissociated form is neutral
            Phi_diss_sed = 1-Phi_n_sed
            KOC_sed_updated = (KOC_sed_n*Phi_diss_sed) + (Phi_n_sed*KOC_sed_base)

            KOC_sedcorr = KOC_sed_updated/KOC_sed_initial

        elif diss == 'amphoter':
            Phi_n_sed = 1/(1+10**(pH_sed-pKa_acid)+10**(pKa_base))
            Phi_anion_sed = Phi_n_sed*10**(pH_sed-pKa_acid)
            Phi_cation_sed = Phi_n_sed*10**(pKa_base-pH_sed)
            KOC_sed_updated = (KOC_sed_n*Phi_n_sed)+(Phi_anion_sed*KOC_sed_acid) + (Phi_cation_sed*KOC_sed_base)

            KOC_sedcorr = KOC_sed_updated/KOC_sed_initial

        elif diss == 'undiss':
            KOC_sedcorr = np.ones_like(pH_sed)

        return KOC_sedcorr

    def calc_KOC_watcorrSPM(self, KOC_SPM_initial, KOC_sed_n, pKa_acid, pKa_base, KOW, pH_water_SPM, diss,
                            KOC_sed_acid, KOC_sed_base):
        ''' Calculate correction of KOC due to pH of water for SPM
        '''

        if diss == 'acid':
            Phi_n_SPM = 1 / (1 + 10 ** (pH_water_SPM - pKa_acid))
            Phi_diss_SPM = 1 - Phi_n_SPM
            KOC_SPM_updated = (KOC_sed_n * Phi_n_SPM) + (Phi_diss_SPM * KOC_sed_acid)

            KOC_SPMcorr = KOC_SPM_updated / KOC_SPM_initial

        elif diss == 'base':
            # Undissociated form is positively charged
            Phi_n_SPM = 1 / (1 + 10 ** (pH_water_SPM - pKa_base))
            # Dissociated form is neutral
            Phi_diss_SPM = 1 - Phi_n_SPM
            KOC_SPM_updated = (KOC_sed_n * Phi_n_SPM) + (Phi_diss_SPM * KOC_sed_base)

            KOC_SPMcorr = KOC_SPM_updated / KOC_SPM_initial

        elif diss == 'amphoter':

            Phi_n_SPM = 1 / (1 + 10 ** (pH_water_SPM - pKa_acid) + 10 ** (pKa_base))
            Phi_anion_SPM = Phi_n_SPM * 10 ** (pH_water_SPM - pKa_acid)
            Phi_cation_SPM = Phi_n_SPM * 10 ** (pKa_base - pH_water_SPM)
            KOC_SPM_updated = (KOC_sed_n * Phi_n_SPM) + (Phi_anion_SPM * KOC_sed_acid) + (Phi_cation_SPM * KOC_sed_base)

            KOC_SPMcorr = KOC_SPM_updated / KOC_SPM_initial

        elif diss == 'undiss':
            KOC_SPMcorr = np.ones_like(pH_water_SPM)

        return KOC_SPMcorr

    def calc_KOC_watcorrDOM(self, KOC_DOM_initial, KOC_DOM_n, pKa_acid, pKa_base, KOW, pH_water_DOM, diss,
                            KOC_DOM_acid, KOC_DOM_base):
        ''' Calculate correction of KOC due to pH of water for DOM
        '''

        if diss == 'acid':

            Phi_n_DOM = 1 / (1 + 10 ** (pH_water_DOM - pKa_acid))
            Phi_diss_DOM = 1 - Phi_n_DOM
            KOC_DOM_updated = ((KOC_DOM_n * Phi_n_DOM) + (Phi_diss_DOM * KOC_DOM_acid))

            KOC_DOMcorr = KOC_DOM_updated / KOC_DOM_initial

        elif diss == 'base':

            # Undissociated form is positively charged
            Phi_n_DOM = 1 / (1 + 10 ** (pH_water_DOM - pKa_base))
            # Dissociated form is neutral
            Phi_diss_DOM = 1 - Phi_n_DOM
            KOC_DOM_updated = ((KOC_DOM_base * Phi_n_DOM) + (Phi_diss_DOM * KOC_DOM_n))

            KOC_DOMcorr = KOC_DOM_updated / KOC_DOM_initial

        elif diss == 'amphoter':

            Phi_n_DOM      = 1/(1 + 10**(pH_water_DOM-pKa_acid) + 10**(pKa_base))
            Phi_anion_DOM  = Phi_n_DOM * 10**(pH_water_DOM-pKa_acid)
            Phi_cation_DOM = Phi_n_DOM * 10**(pKa_base-pH_water_DOM)
            KOC_DOM_updated = (KOC_DOM_n * Phi_n_DOM) + (Phi_anion_DOM * KOC_DOM_acid) + (Phi_cation_DOM * KOC_DOM_base)

            KOC_DOMcorr = KOC_DOM_updated / KOC_DOM_initial

        elif diss == 'undiss':
            KOC_DOMcorr = np.ones_like(pH_water_DOM)

        return KOC_DOMcorr

    def calc_DOCorr(self, HalfSatO_w, k_Anaerobic_water, k_DecayMax_water, Ox_water):
        ''' Correction for the effects of Dissolved Ox concentration on biodegradation
        '''
        DOCorr = np.zeros_like(Ox_water)
        N = len(DOCorr)  # Total number of elements
        chunk_size = int(1e5)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)  # Ensure last chunk fits correctly
            # Slice chunk
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
            else:
                pass

        return DOCorr

    def calc_TCorr(self, T_Max_bio, T_Opt_bio, T_Adp_bio, Max_Accl_bio, Dec_Accl_bio, Q10_bio, TW):
        ''' Correction for the effects of water temperature on biodegradation
        '''

        TCorr = np.zeros_like(TW)
        N = len(TCorr)
        chunk_size = int(1e5)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)  # Ensure last chunk fits correctly
            # Slice chunk
            TW_chunk = TW[i:end]
            # Compute Acclimation for this chunk
            Acclimation = Max_Accl_bio * (1 - np.exp(-Dec_Accl_bio * np.abs(TW_chunk - T_Adp_bio)))
            # Compute VT
            VT = ((T_Max_bio + Acclimation) - TW_chunk) / ((T_Max_bio + Acclimation) - (T_Opt_bio + Acclimation))
            # Compute WT, YT, XT
            WT = np.log(Q10_bio) * ((T_Max_bio + Acclimation) - (T_Opt_bio + Acclimation))
            YT = np.log(Q10_bio) * ((T_Max_bio + Acclimation) - (T_Opt_bio + Acclimation) + 2)
            XT = ((WT**2) * (1 + ((1 + 40 / YT) ** 0.5)) ** 2) / 400
            # Compute TCorr only where VT > 0, otherwise keep as 0
            TCorr[i:end] = np.where(VT > 0, (VT**XT) * np.exp(XT * (1 - VT)), 0)

            if np.any((TCorr[i:end] < 0) | (TCorr[i:end] > 1.0001)): # Allow for 0.01% rounding error
                invalid_indices = np.where((TCorr[i:end] < 0) | (TCorr[i:end] > 1.0001))
                print("Invalid TCorr values and corresponding TW values:")
                print(f"TCorr[{invalid_indices}] = {TCorr[i:end][invalid_indices]}")
                print(f"TW[{invalid_indices}] = {TW[i:end][invalid_indices]}")

                raise ValueError("TCorr is not between 0 and 1")
            else:
                pass

        return TCorr

    def calc_pHCorr(self, pH_min_bio, pH_max_bio, pH_water):
        ''' Correction for the effects of water pH on biodegradation
        '''

        pHCorr = np.ones_like(pH_water)
        N = len(pH_water)
        chunk_size = int(1e5)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)  # Ensure last chunk fits correctly
            # Slice chunk
            pH_chunk = pH_water[i:end]
            # Compute pHCorr based on conditions
            pHCorr[i:end] = np.where(
                pH_chunk < pH_min_bio, np.exp(pH_chunk - pH_min_bio),  # Below range
                np.where(pH_chunk > pH_max_bio, np.exp(pH_max_bio - pH_chunk), 1)  # Above range & default to 1
            )

            if np.any((pHCorr[i:end] < 0) | (pHCorr[i:end] > 1)):
                raise ValueError("pHCorr is not between 0 and 1")
            else:
                pass

        return pHCorr

    def calc_k_hydro_water(self, k_Acid, k_Base, k_Hydr_Uncat, pH_water):
        ''' Hydrolysis rate in water
        '''

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
        ''' Hydrolysis rate in sediments
        '''
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

    def calc_ScreeningFactor(self, RadDistr, RadDistr0_ml, RadDistr0_bml, WaterExt, ExtCoeffDOM, ExtCoeffSPM, ExtCoeffPHY, C2PHYC, concDOC, concSPM, Conc_Phyto_water, Depth, MLDepth):
        ''' Screening Factor for photolisis attenuation with depth due to DOM, SPM, and Pythoplankton
        '''
        N = len(Depth)  # Total number of elements
        ScreeningFactor = np.ones_like(Depth)  # Initialize ScreeningFactor with 1 (default case for Depth == 0)
        chunk_size = int(1e5)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)  # Ensure last chunk fits correctly

            # Slice chunks
            Depth_chunk = np.abs(Depth[i:end])
            MLDepth_chunk = np.abs(MLDepth[i:end])
            concDOC_chunk = concDOC[i:end]
            concSPM_chunk = concSPM[i:end]
            Conc_Phyto_chunk = Conc_Phyto_water[i:end]

            # Compute intermediate values
            ConcDOM = (concDOC_chunk * 12e-6 / 1.025 / 0.526 * 1e-3) * 1e-6  # ((Kg[OM]/L) from (umol[C]/Kg))* 1e-6 = g_DOM/m3
            # ConcSPM is already esxpressed in g_SPM/m3
            ConcPHYTO = (((Conc_Phyto_chunk * 1e-6) * 12.01) / C2PHYC) * 1000 # mmol/m3*1e-6 = mol/L, *12.01 g_C/mol = g_C/L, / (g_Caron/g_Biomass) = g_Biomass/L, *1000 = g_BiomassPHYTO/m3
            Extinct = WaterExt + ExtCoeffDOM * ConcDOM + ExtCoeffSPM * concSPM_chunk + ExtCoeffPHY * ConcPHYTO

            # Compute RadDistr_ratio
            RadDistr_ratio = np.where(
                np.abs(Depth_chunk) <= np.abs(MLDepth_chunk), RadDistr / RadDistr0_ml, RadDistr / RadDistr0_bml
            )

            # Compute ScreeningFactor for nonzero depths
            valid_depth = Depth_chunk > 0
            ScreeningFactor[i:end][valid_depth] = RadDistr_ratio[valid_depth] * (
                (1 - np.exp(-Extinct[valid_depth] * Depth_chunk[valid_depth])) /
                (Extinct[valid_depth] * Depth_chunk[valid_depth])
            )

            if np.any((ScreeningFactor[i:end] < 0) | (ScreeningFactor[i:end] > 1)):
                raise ValueError("ScreeningFactor is not between 0 and 1")
            else:
                pass

        return ScreeningFactor

    def calc_LightFactor(self, AveSolar, Solar_radiation, Conc_CO2_asC, TW, Depth, MLDepth):
        ''' Light Factor for photolisis attenuation with depth
        '''
        # TW = Water temperature in °C
        # TO DO Check here the conversion from input to Ly/day
        N = len(MLDepth)  # Total number of elements
        LightFactor = np.empty_like(MLDepth)  # Initialize LightFactor

        # Precompute constant conversions
        Solar = Solar_radiation / 0.4843  # 1 Ly/day =  41868 J/m2 / 86400 s = 0.4843 W/m2
        Conc_CO2 = ((Conc_CO2_asC * 12.01) * 1000) * 22.73 * 1000 # from mol_C/m3, *12.01 g_C/mol = g_C/m3, *1000 = mg/m3, * 22.73 ueq/mg = ueq/m3, *1000 = ueq/L
        chunk_size = int(1e5)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)  # Ensure last chunk fits correctly

            # Slice chunks
            Depth_chunk = Depth[i:end]
            MLDepth_chunk = MLDepth[i:end]
            TW_chunk = TW[i:end]
            Solar_chunk = Solar[i:end]
            Conc_CO2_chunk = Conc_CO2[i:end]
            # Compute HyphoCorr (used only where Depth > MLDepth)
            HyphoCorr_chunk = ((10 ** (-(6.57 - 0.0118 * TW_chunk + 0.00012 * (TW_chunk ** 2)))) * Conc_CO2_chunk) * 10**-14

            # Compute Solar0
            Solar0_chunk = np.where(
                Depth_chunk <= MLDepth_chunk,
                Solar_chunk,
                Solar_chunk * np.exp(-HyphoCorr_chunk * MLDepth_chunk)
            )

            # Compute LightFactor
            LightFactor[i:end] = Solar0_chunk / AveSolar

        return LightFactor

    def init_transfer_rates(self):
        ''' Initialization of background values in the transfer rates 2D array.
        '''

        transfer_setup=self.get_config('chemical:transfer_setup')

#        logger.info( 'transfer setup: %s' % transfer_setup)


        self.transfer_rates = np.zeros([self.nspecies,self.nspecies])
        self.ntransformations = np.zeros([self.nspecies,self.nspecies])

        if transfer_setup == 'organics':

            self.num_lmm    = self.specie_name2num('LMM')
            self.num_humcol = self.specie_name2num('Humic colloid')
            self.num_prev   = self.specie_name2num('Particle reversible')
            self.num_srev   = self.specie_name2num('Sediment reversible')
            #self.num_psrev  = self.specie_name2num('Particle slowly reversible')
            self.num_ssrev  = self.specie_name2num('Sediment slowly reversible')

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
            if pKa_acid < 0 and diss in ['acid', 'amphoter']:
                raise ValueError("pKa_acid must be positive")
            else:
                pass

            pKa_base   = self.get_config('chemical:transformations:pKa_base')
            if pKa_base < 0 and diss in ['base', 'amphoter']:
                raise ValueError("pKa_base must be positive")
            else:
                pass

            if diss == 'amphoter' and abs(pKa_acid - pKa_base) < 2:
                raise ValueError("pKa_base and pKa_acid must differ of at least two units")
            else:
                pass

            # Read water pH to calculate dissociation
            # pH_water = self.environment.sea_water_ph_reported_on_total_scale
            pH_water   = 8.1 # 8.1

            pH_sed     = 6.9 # 6.9

            fOC_SPM    = self.get_config('chemical:transformations:fOC_SPM')       # typical values from 0.01 to 0.1 gOC/g
            fOC_sed    = self.get_config('chemical:transformations:fOC_sed')       # typical values from 0.01 to 0.1 gOC/g

            concDOM   = 1.e-3 / Org2C    # concentration of available dissolved organic matter (kg/m3)
                                         # rough initial estimate for coastal waters, doi: 10.1002/lom3.10118
            #concDOM   = 50.e-3     # HIGHER VALUE FOR TESTING!!!!!!!!!!!!

            # Values from Simonsen et al (2019a)
            slow_coeff  = self.get_config('chemical:transformations:slow_coeff')
            concSPM     = 50.e-3                                                # available SPM (kg/m3)
            sed_L       = self.get_config('chemical:sediment:mixing_depth')     # sediment mixing depth (m)
            sed_dens    = self.get_config('chemical:sediment:density')          # default particle density (kg/m3)
            sed_phi     = self.get_config('chemical:sediment:corr_factor')      # sediment correction factor
            sed_poro    = self.get_config('chemical:sediment:porosity')         # sediment porosity
            sed_H       = self.get_config('chemical:sediment:layer_thickness')  # thickness of seabed interaction layer (m)
            sed_burial  = self.get_config('chemical:sediment:burial_rate')      # sediment burial rate (m/y)
            sed_leaking_rate = self.get_config( 'chemical:sediment:buried_leaking_rate')

            if diss=='nondiss':
                KOC_DOM = self.get_config('chemical:transformations:KOC_DOM')
                if KOC_DOM < 0:
                    KOC_DOM = 2.88 * KOW**0.67   # (L/KgOC), Park and Clough, 2014

                KOC_sed = self.get_config('chemical:transformations:KOC_sed')
                if KOC_sed < 0:
                    KOC_sed = 2.62 * KOW**0.82   # (L/KgOC), Park and Clough, 2014 (334)/Org2C
                    #KOC_Sed    = 1.26 * kOW**0.81   # (L/KgOC), Ragas et al., 2019

                KOC_SPM = KOC_sed

            else:
                if diss=='acid':
                    # Dissociation in water
                    Phi_n_water    = 1/(1 + 10**(pH_water-pKa_acid))
                    Phi_diss_water = 1-Phi_n_water

                    KOC_sed_n = self.get_config('chemical:transformations:KOC_sed')

                    if KOC_sed_n < 0:
                        # KOC_sed_n    = 2.62 * KOW**0.82   # (L/KgOC), Park and Clough, 2014 (334)/Org2C
                        KOC_sed_n   = 10**((0.54*np.log10(KOW)) + 1.11) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass
                    KOC_sed_acid = self.get_config('chemical:transformations:KOC_sed_acid')
                    if KOC_sed_acid < 0:
                        KOC_sed_acid = (10**(0.11*np.log10(KOW)+1.54)) # KOC for anionic acid specie (L/kg_OC), from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202.
                    else:
                        pass

                    KOC_SPM = (KOC_sed_n * Phi_n_water) + (Phi_diss_water * KOC_sed_acid)


                    KOC_DOM_n = self.get_config('chemical:transformations:KOC_DOM')
                    if KOC_DOM_n < 0:
                        # KOC_DOM_n   = 2.88 * KOW**0.67   # (L/KgOC), Park and Clough, 2014
                        KOC_DOM_n   = (0.08 * KOW)/0.526 # KOC_DOC/Org2C, from DOC to DOM http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass
                    KOC_DOM_acid = self.get_config('chemical:transformations:KOC_DOM_acid')
                    if KOC_DOM_acid < 0:
                        KOC_DOM_acid = (0.08 * 10**(np.log10(KOW)-3.5))/0.526 # KOC_DOC/Org2C, from DOC to DOM http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass

                    KOC_DOM = ((KOC_DOM_n * Phi_n_water) + (Phi_diss_water * KOC_DOM_acid))


                    # Dissociation in sediments
                    Phi_n_sed    = 1/(1 + 10**(pH_sed-pKa_acid))
                    Phi_diss_sed = 1-Phi_n_sed
                    KOC_sed = (KOC_sed_n * Phi_n_sed) + (Phi_diss_sed * KOC_sed_acid)

                elif diss=='base':
                    # Dissociation in water
                    # Undissociated form is positively charged
                    Phi_n_water    = 1/(1 + 10**(pH_water-pKa_base))
                    # Dissociated form is neutral
                    Phi_diss_water = 1-Phi_n_water

                    # Neutral form, is dissociated with respect to pKa_base of conjugated acid
                    KOC_sed_n = self.get_config('chemical:transformations:KOC_sed')
                    if KOC_sed_n <0:
                        # KOC_sed_n   = 2.62 * KOW**0.82   # (L/KgOC), Park and Clough, 2014 (334)/Org2C TO DO Add if choice between input and estimation
                        KOC_sed_n   = 10**((0.37*np.log10(KOW)) + 1.70) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass
                    KOC_sed_base = self.get_config('chemical:transformations:KOC_sed_base')
                    if KOC_sed_base < 0:
                        KOC_sed_base = 10**(pKa_base**(0.65*((KOW/(KOW+1))**0.14))) # KOC for ionized cationic form of base specie (L/kg_OC) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass

                    KOC_SPM = (KOC_sed_base * Phi_n_water) + (KOC_sed_n * Phi_diss_water)


                    KOC_DOM_n = self.get_config('chemical:transformations:KOC_DOM')
                    if KOC_DOM_n < 0:
                        # KOC_DOM_n   = 2.88 * KOW**0.67   # (L/KgOC), Park and Clough, 2014
                        KOC_DOM_n   = (0.08 * KOW)/0.526 # KOC_DOC/Org2C, from DOC to DOM http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    KOC_DOM_base = self.get_config('chemical:transformations:KOC_DOM_base')
                    if KOC_DOM_base < 0:
                        KOC_DOM_base = (0.08 * 10**(np.log10(KOW)-3.5)) /0.526 # KOC_DOC/Org2C, from DOC to DOM http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202

                    KOC_DOM = ((KOC_DOM_base * Phi_n_water) + (Phi_diss_water * KOC_DOM_n))


                    # Dissociation in sediments
                    Phi_n_sed    = 1/(1 + 10**(pH_sed-pKa_base))
                    Phi_diss_sed = 1-Phi_n_sed
                    KOC_sed = (KOC_sed_base * Phi_n_sed) + (KOC_sed_n * Phi_diss_sed)

                elif diss=='amphoter':

                    # Dissociation in water # This approach ignores the zwitterionic fraction. 10.1002/etc.115
                    Phi_n_water      = 1/(1 + 10**(pH_water-pKa_acid) + 10**(pKa_base))
                    Phi_anion_water  = Phi_n_water * 10**(pH_water-pKa_acid)
                    Phi_cation_water = Phi_n_water * 10**(pKa_base-pH_water)
                    Phi_diss_water   = 1 - Phi_n_water

                    KOC_sed_n = self.get_config('chemical:transformations:KOC_sed')
                    if KOC_sed_n < 0:
                        # KOC_sed_n =  KOC_sed_n = 2.62 * KOW**0.82   # (L/KgOC), Park and Clough, 2014 (334)/Org2C TO DO Add if choice between input and estimation
                        KOC_sed_n   = 10**((0.37*np.log10(KOW)) + 1.70) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass
                    KOC_sed_acid = self.get_config('chemical:transformations:KOC_sed_acid')
                    if KOC_sed_acid < 0:
                        KOC_sed_acid = (10**(0.11*np.log10(KOW)+1.54)) # KOC for anionic acid specie (L/kg_OC), from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202.
                    else:
                        pass
                    KOC_sed_base = self.get_config('chemical:transformations:KOC_sed_base')
                    if KOC_sed_base < 0:
                        KOC_sed_base = 10**(pKa_base**(0.65*((KOW/(KOW+1))**0.14))) # KOC for ionized cationic form of base specie (L/kg_OC) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass

                    KOC_SPM = (KOC_sed_n * Phi_n_water) + (Phi_anion_water * KOC_sed_acid) + (Phi_cation_water * KOC_sed_base)


                    KOC_DOM_n = self.get_config('chemical:transformations:KOC_DOM')
                    if KOC_DOM_n <0:
                        # KOC_DOM_n   = 2.88 * KOW**0.67   # (L/KgOC), Park and Clough, 2014
                        KOC_DOM_n   = 0.08 * KOW # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass
                    KOC_DOM_acid = self.get_config('chemical:transformations:KOC_DOM_acid')
                    if KOC_DOM_acid < 0:
                        KOC_DOM_acid = 0.08 * 10**(np.log10(KOW)-3.5) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass
                    KOC_DOM_base = self.get_config('chemical:transformations:KOC_DOM_base')
                    if KOC_DOM_base < 0:
                        KOC_DOM_base = 0.08 * 10**(np.log10(KOW)-3.5) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                    else:
                        pass
                    KOC_DOM = (KOC_DOM_n * Phi_n_water) + (Phi_anion_water * KOC_DOM_acid) + (Phi_cation_water * KOC_DOM_base)


                    # Dissociation in sediments
                    Phi_n_sed      = 1/(1 + 10**(pH_sed-pKa_acid) + 10**(pKa_base))
                    Phi_anion_sed  = Phi_n_sed * 10**(pH_sed-pKa_acid)
                    Phi_cation_sed = Phi_n_sed * 10**(pKa_base-pH_sed)

                    KOC_sed = (KOC_sed_n * Phi_n_sed) + (Phi_anion_sed * KOC_sed_acid) + (Phi_cation_sed * KOC_sed_base)

            logger.debug('Partitioning coefficients (Tref,freshwater)')
            logger.debug('KOC_sed: %s L/KgOC' % KOC_sed)
            logger.debug('KOC_SPM: %s L/KgOC' % KOC_SPM)
            logger.debug('KOC_DOM: %s L/KgOC' % KOC_DOM)

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
            # TODO Use setconfig() to store these?

            self.transfer_rates[self.num_lmm,self.num_humcol] = k_ads * concDOM             # k12
            self.transfer_rates[self.num_humcol,self.num_lmm] = k_des_DOM / TcorrDOM / Scorr# k21

            self.transfer_rates[self.num_lmm,self.num_prev] = k_ads * concSPM               # k13
            self.transfer_rates[self.num_prev,self.num_lmm] = k_des_SPM / TcorrSed / Scorr  # k31

            self.transfer_rates[self.num_lmm,self.num_srev] = \
                k_ads * sed_L * sed_dens * (1.-sed_poro) * sed_phi / sed_H                  # k14
                # TODO CHECK DIMENSIONS!!!!! L-m3 !!!!

            self.transfer_rates[self.num_srev,self.num_lmm] = \
                k_des_sed * sed_phi / TcorrSed / Scorr                                      # k41

            #self.transfer_rates[self.num_srev,self.num_ssrev] = slow_coeff                 # k46
            #self.transfer_rates[self.num_ssrev,self.num_srev] = slow_coeff*.1              # k64

            # Using slowly reversible specie for burial - TODO buried sediment should be a new specie
            self.transfer_rates[self.num_srev,self.num_ssrev] = sed_burial / sed_L / 31556926 # k46 (m/y) / m / (s/y) = s-1
            self.transfer_rates[self.num_ssrev,self.num_srev] = sed_leaking_rate              # k64


            self.transfer_rates[self.num_humcol,self.num_prev] = self.get_config('chemical:transformations:aggregation_rate')
            self.transfer_rates[self.num_prev,self.num_humcol] = 0          # TODO check if valid for organics

        elif transfer_setup == 'metals':                                # renamed from radionuclides Bokna_137Cs

            self.num_lmm    = self.specie_name2num('LMM')
            self.num_prev   = self.specie_name2num('Particle reversible')
            self.num_srev   = self.specie_name2num('Sediment reversible')
            self.num_psrev  = self.specie_name2num('Particle slowly reversible')
            self.num_ssrev  = self.specie_name2num('Sediment slowly reversible')


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
            # Using slowly reversible specie for burial - TODO buried sediment should be a new specie
            # self.transfer_rates[self.num_srev,self.num_ssrev] = slow_coeff
            # self.transfer_rates[self.num_ssrev,self.num_srev] = slow_coeff*.1
            logger.info( 'transfer setup: metals- Using slowly reversible specie for burial')
            sed_L       = self.get_config('chemical:sediment:mixing_depth')     # sediment mixing depth (m)
            sed_burial  = self.get_config('chemical:sediment:burial_rate')      # sediment burial rate (m/y)
            sed_leaking_rate = self.get_config( 'chemical:sediment:buried_leaking_rate')
            self.transfer_rates[self.num_srev,self.num_ssrev] = sed_burial / sed_L / 3155692
            self.transfer_rates[self.num_ssrev,self.num_srev] = sed_leaking_rate
            self.transfer_rates[self.num_prev,self.num_psrev] = slow_coeff
            self.transfer_rates[self.num_ssrev,self.num_srev] = slow_coeff*.1
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

            self.num_lmm   = self.specie_name2num('LMM')
            if self.get_config('chemical:species:Colloid'):
                self.num_col = self.specie_name2num('Colloid')
            if self.get_config('chemical:species:Particle_reversible'):
                self.num_prev  = self.specie_name2num('Particle reversible')
            if self.get_config('chemical:species:Sediment_reversible'):
                self.num_srev  = self.specie_name2num('Sediment reversible')
            if self.get_config('chemical:slowly_fraction'):
                self.num_psrev  = self.specie_name2num('Particle slowly reversible')
                self.num_ssrev  = self.specie_name2num('Sediment slowly reversible')
            if self.get_config('chemical:irreversible_fraction'):
                self.num_pirrev  = self.specie_name2num('Particle irreversible')
                self.num_sirrev  = self.specie_name2num('Sediment irreversible')

            if self.get_config('chemical:species:Particle_reversible'):
                self.transfer_rates[self.num_lmm,self.num_prev] = 5.e-6 #*0.
                self.transfer_rates[self.num_prev,self.num_lmm] = \
                    self.get_config('chemical:transformations:Dc')
            if self.get_config('chemical:species:Sediment_reversible'):
                self.transfer_rates[self.num_lmm,self.num_srev] = 1.e-5 #*0.
                self.transfer_rates[self.num_srev,self.num_lmm] = \
                    self.get_config('chemical:transformations:Dc') * self.get_config('chemical:sediment:corr_factor')
#                self.transfer_rates[self.num_srev,self.num_lmm] = 5.e-6

            if self.get_config('chemical:slowly_fraction'):
                self.transfer_rates[self.num_prev,self.num_psrev] = 2.e-6
                self.transfer_rates[self.num_srev,self.num_ssrev] = 2.e-6
                self.transfer_rates[self.num_psrev,self.num_prev] = 2.e-7
                self.transfer_rates[self.num_ssrev,self.num_srev] = 2.e-7

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
            logger.ERROR('No transfer setup available')

        # Set diagonal to 0. (not possible to transform to present specie)
        if len(self.transfer_rates.shape) == 3:
            for ii in range(self.transfer_rates.shape[0]):
                np.fill_diagonal(self.transfer_rates[ii,:,:],0.)
        else:
            np.fill_diagonal(self.transfer_rates,0.)

#         # HACK :
#         self.transfer_rates[:] = 0.
#         print ('\n ###### \n IMPORTANT:: \n transfer rates have been hacked! \n#### \n ')

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
                z_i = range(Tprofiles.shape[0])  # evtl. move out of loop
                # evtl. move out of loop
                z_index = np.interp1d(-self.environment_profiles['z'],
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
        my_w = 0.001*(1.7915 - 0.0538*T0 + 0.007*(T0**(2.0)) - 0.0023*S0)
        # ~0.0014 kg m-1 s-1

        # terminal velocity for low Reynolds numbers
        W = (1.0/my_w)*(1.0/18.0)*g*partsize**2 * dr

        #W=np.zeros_like(W) #Setting to zero for debugging

        self.elements.terminal_velocity = W

        self.elements.terminal_velocity = W * self.elements.moving

    def update_transfer_rates(self):
        '''Pick out the correct row from transfer_rates for each element. Modify the
        transfer rates according to local environmental conditions '''

        transfer_setup=self.get_config('chemical:transfer_setup')
        if transfer_setup == 'metals' or \
         transfer_setup=='custom' or \
         transfer_setup=='137Cs_rev'or \
         transfer_setup=='organics':
            self.elements.transfer_rates1D = self.transfer_rates[self.elements.specie,:]
            diss       = self.get_config('chemical:transformations:dissociation')

            # Updating desorption rates according to local temperature, salinity, pH

            if transfer_setup=='organics' and diss=='nondiss':
                # filtering out zero values from temperature and salinity
                # TODO: Find out if problem is in the reader or in the data
                temperature=self.environment.sea_water_temperature
                #temperature[temperature==0]=np.median(temperature)

                salinity=self.environment.sea_water_salinity
                #salinity[salinity==0]=np.median(salinity)

                KOWTref    = self.get_config('chemical:transformations:TrefKOW')
                DH_KOC_Sed = self.get_config('chemical:transformations:DeltaH_KOC_Sed')
                DH_KOC_DOM = self.get_config('chemical:transformations:DeltaH_KOC_DOM')
                Setchenow  = self.get_config('chemical:transformations:Setchenow')

                tempcorrSed = self.tempcorr("Arrhenius",DH_KOC_Sed,temperature,KOWTref)
                tempcorrDOM = self.tempcorr("Arrhenius",DH_KOC_DOM,temperature,KOWTref)
                salinitycorr = self.salinitycorr(Setchenow,temperature,salinity)

                # Temperature and salinity correction for desorption rates (inversely proportional to Kd)

                self.elements.transfer_rates1D[self.elements.specie==self.num_humcol,self.num_lmm] = \
                    self.k21_0 / tempcorrDOM[self.elements.specie==self.num_humcol] / salinitycorr[self.elements.specie==self.num_humcol]

                self.elements.transfer_rates1D[self.elements.specie==self.num_prev,self.num_lmm] = \
                    self.k31_0 / tempcorrSed[self.elements.specie==self.num_prev] / salinitycorr[self.elements.specie==self.num_prev]

                self.elements.transfer_rates1D[self.elements.specie==self.num_srev,self.num_lmm] = \
                    self.k41_0 / tempcorrSed[self.elements.specie==self.num_srev] / salinitycorr[self.elements.specie==self.num_srev]

            elif transfer_setup=='organics' and diss!='nondiss':
                # Select elements for updating trasfer rates in sediments, SPM, and DOM

                #Sediments
                S =   (self.elements.specie == self.num_srev)# \
                    # + (self.elements.specie == self.num_ssrev)

                SPM = (self.elements.specie == self.num_prev)

                DOM = (self.elements.specie == self.num_humcol)

                pH_sed = self.environment.pH_sediment[S]
                # pH_sed[pH_sed==0]=np.median(pH_sed)

                pH_water_SPM=self.environment.sea_water_ph_reported_on_total_scale[SPM]
                # pH_water_SPM[pH_water_SPM==0]=np.median(TW)

                pH_water_DOM=self.environment.sea_water_ph_reported_on_total_scale[DOM]
                # pH_water_DOM[pH_water_DOM==0]=np.median(pH_water_DOM)

                pKa_acid   = self.get_config('chemical:transformations:pKa_acid')
                if pKa_acid < 0 and diss in ['amphoter', 'acid']:
                    raise ValueError("pKa_acid must be positive")
                else:
                    pass

                pKa_base   = self.get_config('chemical:transformations:pKa_base')
                if pKa_base < 0 and diss in ['amphoter', 'base']:
                    raise ValueError("pKa_base must be positive")
                else:
                    pass

                KOW = 10**self.get_config('chemical:transformations:LogKOW')

                KOC_sed_n = self.get_config('chemical:transformations:KOC_sed')
                if KOC_sed_n < 0:
                    # KOC_sed_n =  KOC_sed_n = 2.62 * KOW**0.82   # (L/KgOC), Park and Clough, 2014 (334)/Org2C
                    KOC_sed_n   = 10**((0.37*np.log10(KOW)) + 1.70) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                else:
                    pass
                KOC_sed_acid = self.get_config('chemical:transformations:KOC_sed_acid')
                if KOC_sed_acid < 0:
                    KOC_sed_acid = (10**(0.11*np.log10(KOW)+1.54)) # KOC for anionic acid specie (L/kg_OC), from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202.
                else:
                    pass
                KOC_sed_base = self.get_config('chemical:transformations:KOC_sed_base')
                if KOC_sed_base < 0:
                    KOC_sed_base = 10**(pKa_base**(0.65*((KOW/(KOW+1))**0.14))) # KOC for ionized cationic form of base specie (L/kg_OC) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                else:
                    pass


                KOC_DOM_n = self.get_config('chemical:transformations:KOC_DOM')
                if KOC_DOM_n <0:
                    # KOC_DOM_n   = 2.88 * KOW**0.67   # (L/KgOC), Park and Clough, 2014
                    KOC_DOM_n   = 0.08 * KOW # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                else:
                    pass
                KOC_DOM_acid = self.get_config('chemical:transformations:KOC_DOM_acid')
                if KOC_DOM_acid < 0:
                    KOC_DOM_acid = 0.08 * 10**(np.log10(KOW)-3.5) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                else:
                    pass
                KOC_DOM_base = self.get_config('chemical:transformations:KOC_DOM_base')
                if KOC_DOM_base < 0:
                    KOC_DOM_base = 0.08 * 10**(np.log10(KOW)-3.5) # from  http://i-pie.org/wp-content/uploads/2019/12/ePiE_Technical_Manual-Final_Version_20191202
                else:
                    pass

                fOC_SPM    = self.get_config('chemical:transformations:fOC_SPM')       # typical values from 0.01 to 0.1 gOC/g
                fOC_sed    = self.get_config('chemical:transformations:fOC_sed')
                Org2C      = 0.526  # kgOC/KgOM

                # Calculate original KOC_Values
                # TO DO: Store directly KOC values

                KOC_sed_initial = (self.Kd_sed)/fOC_sed # L/Kg / KgOC/Kg = L/KgOC
                KOC_SPM_initial = (self.Kd_SPM)/fOC_SPM # L/Kg / KgOC/Kg = L/KgOC
                KOC_DOM_initial = (self.Kd_DOM)/Org2C

                # filtering out zero values from temperature and salinity
                # TODO: Find out if problem is in the reader or in the data
                temperature=self.environment.sea_water_temperature
                #temperature[temperature==0]=np.median(temperature)

                salinity=self.environment.sea_water_salinity
                #salinity[salinity==0]=np.median(salinity)

                KOWTref    = self.get_config('chemical:transformations:TrefKOW')
                DH_KOC_Sed = self.get_config('chemical:transformations:DeltaH_KOC_Sed')
                DH_KOC_DOM = self.get_config('chemical:transformations:DeltaH_KOC_DOM')
                Setchenow  = self.get_config('chemical:transformations:Setchenow')

                tempcorrSed = self.tempcorr("Arrhenius",DH_KOC_Sed,temperature,KOWTref)
                tempcorrDOM = self.tempcorr("Arrhenius",DH_KOC_DOM,temperature,KOWTref)
                salinitycorr = self.salinitycorr(Setchenow,temperature,salinity)

                KOC_sedcorr = self.calc_KOC_sedcorr(KOC_sed_initial, KOC_sed_n, pKa_acid, pKa_base, KOW, pH_sed, diss,
                                                    KOC_sed_acid, KOC_sed_base)
                KOC_watcorrSPM = self.calc_KOC_watcorrSPM(KOC_SPM_initial, KOC_sed_n, pKa_acid, pKa_base, KOW, pH_water_SPM, diss,
                                                          KOC_sed_acid, KOC_sed_base)
                KOC_watcorrDOM = self.calc_KOC_watcorrDOM(KOC_DOM_initial, KOC_DOM_n, pKa_acid, pKa_base, KOW, pH_water_DOM, diss,
                                                          KOC_DOM_acid, KOC_DOM_base)

                # Temperature and salinity correction for desorption rates (inversely proportional to Kd)

                ####

                self.elements.transfer_rates1D[self.elements.specie==self.num_humcol,self.num_lmm] = \
                    self.k21_0 * KOC_watcorrDOM / tempcorrDOM[self.elements.specie==self.num_humcol] / salinitycorr[self.elements.specie==self.num_humcol]

                self.elements.transfer_rates1D[self.elements.specie==self.num_prev,self.num_lmm] = \
                    self.k31_0 * KOC_watcorrSPM / tempcorrSed[self.elements.specie==self.num_prev] / salinitycorr[self.elements.specie==self.num_prev]

                self.elements.transfer_rates1D[self.elements.specie==self.num_srev,self.num_lmm] = \
                    self.k41_0 * KOC_sedcorr / tempcorrSed[self.elements.specie==self.num_srev] / salinitycorr[self.elements.specie==self.num_srev]

            # Updating sorption rates

            if transfer_setup=='organics':

                # Updating sorption rates according to local SPM concentration

                concSPM=self.environment.spm * 1e-6 # (Kg/L) from (g/m3)

                # Apply SPM concentration profile if SPM reader has not depth coordinate
                # SPM concentration is kept constant to surface value in the mixed layer
                # Exponentially decreasing with depth below the mixed layers
                if not self.SPM_vertical_levels_given:
                    lowerMLD = self.elements.z < -self.environment.ocean_mixed_layer_thickness
                    #concSPM[lowerMLD] = concSPM[lowerMLD]/2
                    concSPM[lowerMLD] = concSPM[lowerMLD] * np.exp(
                        -(self.elements.z[lowerMLD]+self.environment.ocean_mixed_layer_thickness[lowerMLD])
                        *np.log(0.5)/self.get_config('chemical:particle_concentration_half_depth')
                        )

                self.elements.transfer_rates1D[self.elements.specie==self.num_lmm,self.num_prev] = \
                    self.k_ads * concSPM[self.elements.specie==self.num_lmm]      # k13

            if transfer_setup == 'metals':

                # Updating sorption rates according to local SPM concentration and salinity

                concSPM=self.environment.spm * 1e-3 # (Kg/m3) from (g/m3)

                salinity=self.environment.sea_water_salinity

                # Apply SPM concentration profile if SPM reader has not depth coordinate
                # SPM concentration is kept constant to surface value in the mixed layer
                # Exponentially decreasing with depth below the mixed layers
                if not self.SPM_vertical_levels_given:
                    lowerMLD = self.elements.z < -self.environment.ocean_mixed_layer_thickness
                    #concSPM[lowerMLD] = concSPM[lowerMLD]/2
                    concSPM[lowerMLD] = concSPM[lowerMLD] * np.exp(
                        -(self.elements.z[lowerMLD]+self.environment.ocean_mixed_layer_thickness[lowerMLD])
                        *np.log(0.5)/self.get_config('chemical:particle_concentration_half_depth')
                        )

                Kd0         = self.get_config('chemical:transformations:Kd')        # (m3/Kg)
                S0          = self.get_config('chemical:transformations:S0')        # (PSU)
                Dc          = self.get_config('chemical:transformations:Dc')        # (1/s)
                sed_L       = self.get_config('chemical:sediment:mixing_depth')     # sediment mixing depth (m)
                sed_dens    = self.get_config('chemical:sediment:density')          # default particle density (kg/m3)
                sed_f       = self.get_config('chemical:sediment:effective_fraction') # fraction of effective sorbents
                sed_phi     = self.get_config('chemical:sediment:corr_factor')      # sediment correction factor
                sed_poro    = self.get_config('chemical:sediment:porosity')         # sediment porosity
                sed_H       = self.get_config('chemical:sediment:layer_thickness')  # thickness of seabed interaction layer (m)

                # Adjust Kd for salinity according to Perianez 2018 https://doi.org/10.1016/j.jenvrad.2018.02.014
                if S0>0:
                    Kd=Kd0*(S0+salinity[self.elements.specie==self.num_lmm])/S0

                self.elements.transfer_rates1D[self.elements.specie==self.num_lmm,self.num_prev] = \
                    Dc * Kd * concSPM[self.elements.specie==self.num_lmm]      # k13

                self.elements.transfer_rates1D[self.elements.specie==self.num_lmm,self.num_srev] = \
                    Dc * Kd * sed_L * sed_dens * (1.-sed_poro) * sed_f * sed_phi / sed_H


            if transfer_setup=='organics':

                # Updating sorption rates according to local DOC concentration

                concDOM = self.environment.doc * 12e-6 / 1.025 / 0.526 * 1e-3 # (Kg[OM]/L) from (umol[C]/Kg)

                # Apply DOC concentration profile if DOC reader has not depth coordinate
                # DOC concentration is kept constant to surface value in the mixed layer
                # Exponentially decreasing with depth below the mixed layers

                if not self.DOC_vertical_levels_given:
                    lowerMLD = self.elements.z < -self.environment.ocean_mixed_layer_thickness
                    #concDOM[lowerMLD] = concDOM[lowerMLD]/2
                    concDOM[lowerMLD] = concDOM[lowerMLD] * np.exp(
                        -(self.elements.z[lowerMLD]+self.environment.ocean_mixed_layer_thickness[lowerMLD])
                        *np.log(0.5)/self.get_config('chemical:doc_concentration_half_depth')
                        )

                self.elements.transfer_rates1D[self.elements.specie==self.num_lmm,self.num_humcol] = \
                    self.k_ads * concDOM[self.elements.specie==self.num_lmm]      # k12

            if self.get_config('chemical:species:Sediment_reversible'):
                # Only LMM chemicals close to seabed are allowed to interact with sediments
                # minimum height/maximum depth for each particle
                Zmin = -1.*self.environment.sea_floor_depth_below_sea_level
                interaction_thick = self.get_config('chemical:sediment:layer_thickness')      # thickness of seabed interaction layer (m)
                dist_to_seabed = self.elements.z - Zmin
                self.elements.transfer_rates1D[(self.elements.specie == self.num_lmm) &
                                 (dist_to_seabed > interaction_thick), self.num_srev] = 0.

        elif transfer_setup=='Sandnesfj_Al':
            sal = self.environment.sea_water_salinity
            sali = np.searchsorted(self.salinity_intervals, sal) - 1
            self.elements.transfer_rates1D = self.transfer_rates[sali,self.elements.specie,:]

    def update_partitioning(self):
        '''Check if transformation processes shall occur
        Do transformation (change value of self.elements.specie)
        Update element properties for the transformed elements
        '''

        specie_in  = self.elements.specie.copy()    # for storage of the initial partitioning
        specie_out = self.elements.specie.copy()    # for storage of the final partitioning
        deltat = self.time_step.total_seconds()     # length of a time step
        # phaseshift = np.array(self.num_elements_active()*[False]
        phaseshift = np.zeros(self.num_elements_active(), dtype=bool) # Denotes which trajectory that shall be transformed

        p = 1. - np.exp(-self.elements.transfer_rates1D*deltat)  # Probability for transformation
        psum = np.sum(p,axis=1)

        ran1=np.random.random(self.num_elements_active())

        # Transformation where ran1 < total probability for transformation
        # phaseshift[ ran1 < psum ] = True
        phaseshift = ran1 < psum
        num_transformed = np.count_nonzero(phaseshift)
        logger.info('Number of transformations: %s' % num_transformed)
        if num_transformed == 0:
            return

        ran4 = np.random.random(num_transformed) # New random number to decide which specie to end up in
        p_selected = p[phaseshift]  # Only rows where transformation happens
        psum_selected = psum[phaseshift]  # Corresponding sum of probabilities
        psum_selected = np.where(psum_selected == 0, 1, psum_selected)
        cumsum_p = np.cumsum(p_selected / psum_selected[:, np.newaxis], axis=1)
        ran4 = ran4[:cumsum_p.shape[0]]  # Matching the number of transformed elements
        specie_out[phaseshift] = np.array([np.searchsorted(cumsum_p[i], ran4[i]) for i in range(len(ran4))])

        # Set the new partitioning
        self.elements.specie = specie_out

        logger.debug('old species: %s' % specie_in[phaseshift])
        logger.debug('new species: %s' % specie_out[phaseshift])


        for iin in range(self.nspecies):
            for iout in range(self.nspecies):
                self.ntransformations[iin,iout]+=np.count_nonzero((specie_in[phaseshift]==iin) & (specie_out[phaseshift]==iout))

        logger.debug('Number of transformations total:\n %s' % self.ntransformations )

        # Update Chemical properties after transformations
        self.update_chemical_diameter(specie_in, specie_out)
        self.sorption_to_sediments(specie_in, specie_out)
        self.desorption_from_sediments(specie_in, specie_out)

    def sorption_to_sediments(self,sp_in=None,sp_out=None):
        '''Update Chemical properties  when sorption to sediments occurs'''

        # Set z to local sea depth
        if self.get_config('chemical:species:LMM'):
            self.elements.z[(sp_out==self.num_srev) & (sp_in==self.num_lmm)] = \
                -1.*self.environment.sea_floor_depth_below_sea_level[(sp_out==self.num_srev) & (sp_in==self.num_lmm)]
            self.elements.moving[(sp_out==self.num_srev) & (sp_in==self.num_lmm)] = 0
        if self.get_config('chemical:species:LMMcation'):
            self.elements.z[(sp_out==self.num_srev) & (sp_in==self.num_lmmcation)] = \
                -1.*self.environment.sea_floor_depth_below_sea_level[(sp_out==self.num_srev) & (sp_in==self.num_lmmcation)]
            self.elements.moving[(sp_out==self.num_srev) & (sp_in==self.num_lmmcation)] = 0
        # avoid setting positive z values
        if np.nansum(self.elements.z>0):
            logger.debug('Number of elements lowered down to sea surface: %s' % np.nansum(self.elements.z>0))
        self.elements.z[self.elements.z > 0] = 0

    def desorption_from_sediments(self,sp_in=None,sp_out=None):
        '''Update Chemical properties when desorption from sediments occurs'''

        desorption_depth = self.get_config('chemical:sediment:desorption_depth')
        std = self.get_config('chemical:sediment:desorption_depth_uncert')

        if self.get_config('chemical:species:LMM'):
            self.elements.z[(sp_out==self.num_lmm) & (sp_in==self.num_srev)] = \
                -1.*self.environment.sea_floor_depth_below_sea_level[(sp_out==self.num_lmm) & (sp_in==self.num_srev)] + desorption_depth
            self.elements.moving[(sp_out==self.num_lmm) & (sp_in==self.num_srev)] = 1
            if std > 0:
                logger.debug('Adding uncertainty for desorption from sediments: %s m' % std)
                self.elements.z[(sp_out==self.num_lmm) & (sp_in==self.num_srev)] += np.random.normal(
                        0, std, sum((sp_out==self.num_lmm) & (sp_in==self.num_srev)))
        if self.get_config('chemical:species:LMMcation'):
            self.elements.z[(sp_out==self.num_lmmcation) & (sp_in==self.num_srev)] = \
                -1.*self.environment.sea_floor_depth_below_sea_level[(sp_out==self.num_lmmcation) & (sp_in==self.num_srev)] + desorption_depth
            self.elements.moving[(sp_out==self.num_lmmcation) & (sp_in==self.num_srev)] = 1
            if std > 0:
                logger.debug('Adding uncertainty for desorption from sediments: %s m' % std)
                self.elements.z[(sp_out==self.num_lmmcation) & (sp_in==self.num_srev)] += np.random.normal(
                        0, std, sum((sp_out==self.num_lmmcation) & (sp_in==self.num_srev)))
        # avoid setting positive z values
        if np.nansum(self.elements.z>0):
            logger.debug('Number of elements lowered down to sea surface: %s' % np.nansum(self.elements.z>0))
        self.elements.z[self.elements.z > 0] = 0

    def update_chemical_diameter(self,sp_in=None,sp_out=None):
        '''Update the diameter of the chemicals when specie is changed'''

        dia_part=self.get_config('chemical:particle_diameter')
        dia_DOM_part = self.get_config('chemical:doc_particle_diameter')
        dia_diss=self.get_config('chemical:dissolved_diameter')


        # Transfer to reversible particles
        self.elements.diameter[(sp_out==self.num_prev) & (sp_in!=self.num_prev)] = dia_part

        if self.get_config('chemical:species:Humic_colloid'):
            self.elements.diameter[(sp_out==self.num_prev) & (sp_in==self.num_humcol)] = dia_DOM_part

        logger.debug('Updated particle diameter for %s elements' % len(self.elements.diameter[(sp_out==self.num_prev) & (sp_in!=self.num_prev)]))

        std = self.get_config('chemical:particle_diameter_uncertainty')
        if std > 0:
            logger.debug('Adding uncertainty for particle diameter: %s m' % std)
            self.elements.diameter[(sp_out==self.num_prev) & (sp_in!=self.num_prev)] += np.random.normal(
                    0, std, sum((sp_out==self.num_prev) & (sp_in!=self.num_prev)))
        # Transfer to slowly reversible particles
        if self.get_config('chemical:slowly_fraction'):
            self.elements.diameter[(sp_out==self.num_psrev) & (sp_in!=self.num_psrev)] = dia_part
            if std > 0:
                logger.debug('Adding uncertainty for slowly rev particle diameter: %s m' % std)
                self.elements.diameter[(sp_out==self.num_psrev) & (sp_in!=self.num_psrev)] += np.random.normal(
                    0, std, sum((sp_out==self.num_psrev) & (sp_in!=self.num_psrev)))

        # Transfer to irreversible particles
        if self.get_config('chemical:irreversible_fraction'):
            self.elements.diameter[(sp_out==self.num_pirrev) & (sp_in!=self.num_pirrev)] = dia_part
            if std > 0:
                logger.debug('Adding uncertainty for irrev particle diameter: %s m' % std)
                self.elements.diameter[(sp_out==self.num_pirrev) & (sp_in!=self.num_pirrev)] += np.random.normal(
                    0, std, sum((sp_out==self.num_pirrev) & (sp_in!=self.num_pirrev)))

        # Transfer to LMM
        if self.get_config('chemical:species:LMM'):
            self.elements.diameter[(sp_out==self.num_lmm) & (sp_in!=self.num_lmm)] = dia_diss
        if self.get_config('chemical:species:LMManion'):
            self.elements.diameter[(sp_out==self.num_lmmanion) & (sp_in!=self.num_lmmanion)] = dia_diss
        if self.get_config('chemical:species:LMMcation'):
            self.elements.diameter[(sp_out==self.num_lmmcation) & (sp_in!=self.num_lmmcation)] = dia_diss

        # Transfer to colloids
        if self.get_config('chemical:species:Colloid'):
            self.elements.diameter[(sp_out==self.num_col) & (sp_in!=self.num_col)] = dia_diss
        if self.get_config('chemical:species:Humic_colloid'):
            self.elements.diameter[(sp_out==self.num_humcol) & (sp_in!=self.num_humcol)] = dia_diss
        if self.get_config('chemical:species:Polymer'):
            self.elements.diameter[(sp_out==self.num_polymer) & (sp_in!=self.num_polymer)] = dia_diss

    def bottom_interaction(self,Zmin=None):
        ''' Change partitioning of chemicals that reach bottom due to settling.
        particle specie -> sediment specie '''
        if not  ((self.get_config('chemical:species:Particle_reversible')) &
                  (self.get_config('chemical:species:Sediment_reversible')) or
                  (self.get_config('chemical:slowly_fraction')) or
                  (self.get_config('chemical:irreversible_fraction'))):
            return

        bottom = np.array(np.where(self.elements.z <= Zmin)[0])
        kktmp = np.array(np.where(self.elements.specie[bottom] == self.num_prev)[0])
        self.elements.specie[bottom[kktmp]] = self.num_srev
        self.ntransformations[self.num_prev,self.num_srev]+=len(kktmp)
        self.elements.moving[bottom[kktmp]] = 0
        if self.get_config('chemical:slowly_fraction'):
            kktmp = np.array(np.where(self.elements.specie[bottom] == self.num_psrev)[0])
            self.elements.specie[bottom[kktmp]] = self.num_ssrev
            self.ntransformations[self.num_psrev,self.num_ssrev]+=len(kktmp)
            self.elements.moving[bottom[kktmp]] = 0
        if self.get_config('chemical:irreversible_fraction'):
            kktmp = np.array(np.where(self.elements.specie[bottom] == self.num_pirrev)[0])
            self.elements.specie[bottom[kktmp]] = self.num_sirrev
            self.ntransformations[self.num_pirrev,self.num_sirrev]+=len(kktmp)
            self.elements.moving[bottom[kktmp]] = 0

    def resuspension(self):
        """ Simple method to estimate the resuspension of sedimented particles,
        checking whether the current speed near the bottom is above a critical velocity
        Sediment species -> Particle specie
        """
        # Exit function if particles and sediments not are present
        if not  ((self.get_config('chemical:species:Particle_reversible')) &
                  (self.get_config('chemical:species:Sediment_reversible'))):
            return

        specie_in = self.elements.specie.copy()

        critvel = self.get_config('chemical:sediment:resuspension_critvel')
        resusp_depth = self.get_config('chemical:sediment:resuspension_depth')
        std = self.get_config('chemical:sediment:resuspension_depth_uncert')

        Zmin = -1.*self.environment.sea_floor_depth_below_sea_level
        x_vel = self.environment.x_sea_water_velocity
        y_vel = self.environment.y_sea_water_velocity
        speed = np.sqrt(x_vel*x_vel + y_vel*y_vel)
        bottom = (self.elements.z <= Zmin)

        resusp = ( (bottom) & (speed >= critvel) )
        if self.get_config('chemical:slowly_fraction'):
            resusp = ( resusp & (self.elements.specie!=self.num_ssrev) )    # Prevent ssrev (buried) to be resuspended
                                                                        # TODO buried sediment should be a new specie
        logger.info('Number of resuspended particles: {}'.format(np.sum(resusp)))
        self.elements.moving[resusp] = 1

        self.elements.z[resusp] = Zmin[resusp] + resusp_depth
        if std > 0:
            logger.debug('Adding uncertainty for resuspension from sediments: %s m' % std)
            self.elements.z[resusp] += np.random.normal(
                        0, std, sum(resusp))
        # avoid setting positive z values
        if np.nansum(self.elements.z>0):
            logger.debug('Number of elements lowered down to sea surface: %s' % np.nansum(self.elements.z>0))
        self.elements.z[self.elements.z > 0] = 0

        self.ntransformations[self.num_srev,self.num_prev]+=sum((resusp) & (self.elements.specie==self.num_srev))
        self.elements.specie[(resusp) & (self.elements.specie==self.num_srev)] = self.num_prev
        if self.get_config('chemical:slowly_fraction'):
            self.ntransformations[self.num_ssrev,self.num_psrev]+=sum((resusp) & (self.elements.specie==self.num_ssrev))
            self.elements.specie[(resusp) & (self.elements.specie==self.num_ssrev)] = self.num_psrev

        if self.get_config('chemical:irreversible_fraction'):
            self.ntransformations[self.num_sirrev,self.num_pirrev]+=sum((resusp) & (self.elements.specie==self.num_sirrev))
            self.elements.specie[(resusp) & (self.elements.specie==self.num_sirrev)] = self.num_pirrev

        specie_out = self.elements.specie.copy()
        self.update_chemical_diameter(specie_in, specie_out)

    def degradation(self):
        '''degradation.'''

        if self.get_config('chemical:transformations:degradation') is True:
            Save_degr_now = self.get_config('chemical:transformations:Save_degr_now')
            if self.get_config('chemical:transformations:degradation_mode')=='OverallRateConstants':

                logger.debug('Calculating overall degradation using overall rate constants')

                degraded_now = np.zeros(self.num_elements_active())

                # Degradation in the water
                k_W_tot = -np.log(0.5)/(self.get_config('chemical:transformations:t12_W_tot')*(60*60)) # (1/s)
                Tref_kWt = self.get_config('chemical:transformations:Tref_kWt')
                DH_kWt = self.get_config('chemical:transformations:DeltaH_kWt')

                W = (self.elements.specie == self.num_lmm) | (self.elements.specie == self.num_humcol)
                W_deg = np.any(W)

                if W_deg:
                    TW=self.environment.sea_water_temperature[W]
                    # if np.any(TW==0):
                    #     TW[TW==0]=np.median(TW)
                    #     logger.debug("Temperature in degradation was 0, set to median value")

                    k_W_fin = k_W_tot * self.tempcorr("Arrhenius",DH_kWt,TW,Tref_kWt)
                    k_W_fin = np.maximum(k_W_fin, 0.0)

                    degraded_now[W] = np.minimum(self.elements.mass[W],
                                    self.elements.mass[W] * (1-np.exp(-k_W_fin * self.time_step.total_seconds()))) # avoid degradation of more mass than is present in element

                # Degradation in the sediments

                k_S_tot = -np.log(0.5)/(self.get_config('chemical:transformations:t12_S_tot')*(60*60)) # (1/s)
                Tref_kSt = self.get_config('chemical:transformations:Tref_kSt')
                DH_kSt = self.get_config('chemical:transformations:DeltaH_kSt')

                S =   (self.elements.specie == self.num_srev) | (self.elements.specie == self.num_ssrev)
                S_deg = np.any(S)

                if S_deg:
                    TS=self.environment.sea_water_temperature[S]
                    #TS[TS==0]=np.median(TS)

                    k_S_fin = k_S_tot * self.tempcorr("Arrhenius",DH_kSt,TS,Tref_kSt)
                    k_S_fin = np.maximum(k_S_fin, 0.0)

                    degraded_now[S] = np.minimum(self.elements.mass[S],
                                    self.elements.mass[S] * (1-np.exp(-k_S_fin * self.time_step.total_seconds()))) # avoid degradation of more mass than is present in element

                if W_deg or S_deg:
                    if Save_degr_now:
                        SPM = (self.elements.specie == self.num_prev)
                        SPM_deg = np.any(SPM)

                        self.elements.mass_degraded_now[W] = degraded_now[W]
                        self.elements.mass_degraded_now[S] = degraded_now[S]
                        if SPM_deg:
                            self.elements.mass_degraded_now[SPM] = 0.

                        self.elements.mass_degraded_water[W] = degraded_now[W]
                        if SPM_deg:
                            self.elements.mass_degraded_water[SPM] = 0.
                        if S_deg:
                            self.elements.mass_degraded_water[S] = 0.

                        self.elements.mass_degraded_sediment[S] = degraded_now[S]
                        if SPM_deg:
                            self.elements.mass_degraded_sediment[SPM] = 0.
                        if W_deg:
                            self.elements.mass_degraded_sediment[W] = 0.
                    else:
                        self.elements.mass_degraded_water[W] = self.elements.mass_degraded_water[W] + degraded_now[W]
                        self.elements.mass_degraded_sediment[S] = self.elements.mass_degraded_sediment[S] + degraded_now[S]

                    self.elements.mass_degraded = self.elements.mass_degraded + degraded_now
                    self.elements.mass = self.elements.mass - degraded_now

                    self.elements.mass = np.maximum(self.elements.mass, 0.0)
                    self.deactivate_elements(self.elements.mass < (self.elements.mass + self.elements.mass_degraded + self.elements.mass_volatilized)/500,
                                             reason='removed')
                else:
                    if Save_degr_now:
                        self.elements.mass_degraded_now = np.zeros(self.num_elements_active())
                        self.elements.mass_degraded_water = np.zeros(self.num_elements_active())
                        self.elements.mass_degraded_sediment = np.zeros(self.num_elements_active())


                #to_deactivate = self.elements.mass < (self.elements.mass + self.elements.mass_degraded + self.elements.mass_volatilized)/100
                #vol_morethan_degr = self.elements.mass_degraded >= self.elements.mass_volatilized
                #
                #self.deactivate_elements(to_deactivate +  vol_morethan_degr, reason='volatilized')
                #self.deactivate_elements(to_deactivate + ~vol_morethan_degr, reason='degraded')

            elif self.get_config('chemical:transformations:degradation_mode')=='SingleRateConstants':
                logger.debug('Calculating single degradation rates in water')
                # print(self.steps_calculation)
                Photo_degr = self.get_config('chemical:transformations:Photodegradation')
                Bio_degr = self.get_config('chemical:transformations:Biodegradation')
                Hydro_degr = self.get_config('chemical:transformations:Hydrolysis')

                degraded_now = np.zeros(self.num_elements_active())

                # Calculations here are for single process degradation including
                # biodegradation, photodegradation, and hydrolysys

                # Only "dissolved" and "DOC" elements will degrade in the water column
                W = (self.elements.specie == self.num_lmm) | (self.elements.specie == self.num_humcol)
                # All elements in sediments will degrade
                S =   (self.elements.specie == self.num_srev) | (self.elements.specie == self.num_ssrev)
                W_deg = np.any(W)
                S_deg = np.any(S)
                if Save_degr_now:
                    SPM = (self.elements.specie == self.num_prev)
                    SPM_deg = np.any(SPM)

                k_Photo = self.get_config('chemical:transformations:k_Photo')
                k_DecayMax_water = self.get_config('chemical:transformations:k_DecayMax_water')
                k_Anaerobic_water = self.get_config('chemical:transformations:k_Anaerobic_water')
                if k_Photo == 0:
                    logger.debug("k_Photo is set to 0 1/h, therefore no photodegradation occurs")
                if k_DecayMax_water == 0:
                    logger.debug("k_DecayMax_water is set to 0 1/h, therefore  DOCorr = 0 and no biodegradation occurs")
                if k_Anaerobic_water == 0:
                    logger.debug("k_Anaerobic_water is set to 0 1/h, therefore no biodegradation occurs without oxigen")


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
                        # # Dissolved oxigen in g/m3 or mg/L
                        Ox_water=self.environment.mole_concentration_of_dissolved_molecular_oxygen_in_sea_water[W]
                    else:
                        pass

                    if Hydro_degr is True:
                        k_Acid = self.get_config('chemical:transformations:k_Acid')
                        k_Base = self.get_config('chemical:transformations:k_Base')
                        k_Hydr_Uncat = self.get_config('chemical:transformations:k_Hydr_Uncat')
                        if (k_Acid <= 0 and k_Base <= 0 and k_Hydr_Uncat == 0):
                            logger.debug("k_Acid, k_Base, and  k_Hydr_Uncat are set to 0 1/h, therefore no hydrolysis occurs")
                    else:
                        pass


                if W_deg:
                    # Temperature
                    TW=self.environment.sea_water_temperature[W]
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
                        # Concentration of C02 in the water column (mol_C/m3)
                        Conc_CO2_asC=self.environment.mole_concentration_of_dissolved_inorganic_carbon_in_sea_water[W]
                        # if np.any(Conc_CO2_asC==0):
                        #     Conc_CO2_asC[Conc_CO2_asC==0]=np.median(Conc_CO2_asC)
                        #     logger.debug("CO2_asC in degradation was 0, set to median value")

                        # Solar radiation (W/m2)
                        Solar_radiation=self.environment.solar_irradiance[W]
                        # if np.any(Solar_radiation==0):
                        #     Solar_radiation[Solar_radiation==0]=np.median(Solar_radiation)
                        #     logger.debug("Solar_radiation in degradation was 0, set to median value")

                        # Concentration of phytoplankton in the water column (mol_C/m3)
                        Conc_Phyto_water=self.environment.mole_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water[W]
                        # if np.any(Conc_Phyto_water==0):
                        #     Conc_Phyto_water[Conc_Phyto_water==0]=np.median(Conc_Phyto_water)
                        #     logger.debug("Conc_Phyto_water in degradation was 0, set to median value")

                        # Concentration of SPM (g/m3)
                        concSPM=self.environment.spm

                        # Mixed Layer depth (m)
                        MLDepth=self.environment.ocean_mixed_layer_thickness[W]
                        # if np.any(MLDepthr==0):
                        #     MLDepth[MLDepthr==0]=np.median(MLDepth)
                        #     logger.debug("MLDepth in degradation was 0, set to median value")

                        # Depth of element (m)
                        Depth=-self.elements.z[W]     # self.elements.z is negative

                        # Apply SPM concentration profile if SPM reader has not depth coordinate
                        # SPM concentration is kept constant to surface value in the mixed layer
                        # Exponentially decreasing with depth below the mixed layers

                        if not self.SPM_vertical_levels_given:
                            lowerMLD = self.elements.z < -self.environment.ocean_mixed_layer_thickness
                            #concSPM[lowerMLD] = concSPM[lowerMLD]/2
                            concSPM[lowerMLD] = concSPM[lowerMLD] * np.exp(
                                -(self.elements.z[lowerMLD]+self.environment.ocean_mixed_layer_thickness[lowerMLD])
                                *np.log(0.5)/self.get_config('chemical:particle_concentration_half_depth')
                                )
                        concSPM=concSPM[W] # (g/m3)

                        # Concentration of DOC (umol[C]/Kg)
                        concDOC = self.environment.doc

                        # Apply DOC concentration profile if DOC reader has not depth coordinate
                        # DOC concentration is kept constant to surface value in the mixed layer
                        # Exponentially decreasing with depth below the mixed layers

                        if not self.DOC_vertical_levels_given:
                            lowerMLD = self.elements.z < -self.environment.ocean_mixed_layer_thickness
                            #concDOM[lowerMLD] = concDOM[lowerMLD]/2
                            concDOC[lowerMLD] = concDOC[lowerMLD] * np.exp(
                                -(self.elements.z[lowerMLD]+self.environment.ocean_mixed_layer_thickness[lowerMLD])
                                *np.log(0.5)/self.get_config('chemical:doc_concentration_half_depth')
                                )
                        concDOC=concDOC[W] # in (umol[C]/Kg)
                    else:
                        pass


                    if (Bio_degr is True and k_DecayMax_water > 0) or Hydro_degr is True:
                        # pH water
                        pH_water=self.environment.sea_water_ph_reported_on_total_scale[W]
                        if np.any(pH_water==0):
                            pH_water[pH_water==0]=np.median(pH_water)
                            logger.debug("pH_water in degradation was 0, set to median value")
                    else:
                        pass

                    # Calculate correction factors for degradation rates

                    if Bio_degr is True and k_DecayMax_water > 0:
                        k_W_bio = k_DecayMax_water * self.calc_DOCorr(HalfSatO_w, k_Anaerobic_water, k_DecayMax_water, Ox_water)
                        k_W_bio = k_W_bio * self.calc_pHCorr(pH_min_bio, pH_max_bio, pH_water)
                        k_W_bio = k_W_bio * self.calc_TCorr(T_Max_bio, T_Opt_bio, T_Adp_bio, Max_Accl_bio,
                                                            Dec_Accl_bio, Q10_bio, TW)
                        k_W_bio = np.maximum(k_W_bio, 0.0)
                    else:
                        k_W_bio = np.zeros_like(TW)

                    if Photo_degr is True and k_Photo > 0:
                        k_W_photo = k_Photo * self.calc_LightFactor(AveSolar, Solar_radiation, Conc_CO2_asC, TW, Depth, MLDepth)
                        k_W_photo = k_W_photo * self.calc_ScreeningFactor(RadDistr, RadDistr0_ml, RadDistr0_bml, WaterExt,
                                                                          ExtCoeffDOM, ExtCoeffSPM, ExtCoeffPHY, C2PHYC, concDOC,
                                                                          concSPM, Conc_Phyto_water, Depth, MLDepth)
                        k_W_photo = k_W_photo * self.tempcorr("Arrhenius",DH_kWt,TW,Tref_kWt)
                        k_W_photo = np.maximum(k_W_photo, 0.0)
                    else:
                        k_W_photo = np.zeros_like(TW)

                    if Hydro_degr is True:
                        k_W_hydro = self.calc_k_hydro_water(k_Acid, k_Base, k_Hydr_Uncat, pH_water)
                        k_W_hydro = k_W_hydro * self.tempcorr("Arrhenius",DH_kWt,TW,Tref_kWt)
                        k_W_hydro = np.maximum(k_W_hydro, 0.0)
                    else:
                        k_W_hydro = np.zeros_like(TW)

                    k_W_fin = (k_W_bio + k_W_hydro + k_W_photo)/(60*60) # from 1/h to 1/s
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

                # Degradation in the sediments

                if S_deg:
                    TS=self.environment.sea_water_temperature[S]
                    # if np.any(TS==0):
                    #     TS[TS==0]=np.median(TS)
                    #     logger.debug("Temperature in degradation was 0, set to median value")

                    if (Bio_degr is True and k_DecayMax_water > 0) or Hydro_degr is True:
                        # pH sediments
                        pH_sed=self.environment.pH_sediment[S]
                        if np.any(pH_sed==0):
                            pH_sed[pH_sed==0]=np.median(pH_sed)
                            logger.debug("pH_sed in degradation was 0, set to median value")

                    if Bio_degr is True and k_DecayMax_water > 0:
                        k_S_bio = self.get_config('chemical:transformations:k_DecayMax_water')/4  # From AQUATOX   k_DecayMax_water is a rate (1/h), and k_S_bio is four times slower than k_DecayMax_water
                        k_S_bio = k_S_bio * self.calc_pHCorr(pH_min_bio, pH_max_bio, pH_sed)
                        k_S_bio = k_S_bio * self.tempcorr("Arrhenius",DH_kSt,TS,Tref_kSt)
                        # k_S_bio = k_S_bio * self.calc_TCorr(T_Max_bio, T_Opt_bio, T_Adp_bio, Max_Accl_bio,
                        #                                     Dec_Accl_bio, Q10_bio, TW)
                    else:
                        k_S_bio = np.zeros_like(TS)

                    if Hydro_degr is True:
                        k_S_hydro = self.calc_k_hydro_sed(k_Acid, k_Base, k_Hydr_Uncat, pH_sed)
                        k_S_hydro = k_S_hydro * self.tempcorr("Arrhenius",DH_kSt,TS,Tref_kSt)
                    else:
                        k_S_hydro = np.zeros_like(TS)

                    k_S_fin = (k_S_bio + k_S_hydro)/(60*60) # from 1/h to 1/s
                    k_S_fin_sum = k_S_fin.sum()

                    # Update mass of elements due to degradation
                    if k_S_fin_sum > 0:
                        degraded_now[S] = np.minimum(self.elements.mass[S],
                                        self.elements.mass[S] * (1 - np.exp(-k_S_fin * self.time_step.total_seconds())))
                else:
                    k_S_bio = 0
                    k_S_hydro = 0
                    k_S_fin = 0
                    k_S_fin_sum = 0


                if self.get_config('chemical:transformations:Save_single_degr_mass') is True:
                    k_W_photo_fraction = 0
                    k_W_bio_fraction = 0
                    k_S_bio_fraction = 0
                    k_W_hydro_fraction = 0
                    k_S_hydro_fraction = 0

                    if Photo_degr is True and k_Photo > 0:
                        if np.sum(k_W_photo) > 0:
                            photo_degraded_now = np.zeros(self.num_elements_active())
                            k_W_photo_fraction = np.minimum((k_W_photo / 3600) / np.maximum(k_W_fin, 1e-12), 1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation
                            photo_degraded_now[W] = degraded_now[W] * k_W_photo_fraction
                            if Save_degr_now is True:
                                self.elements.mass_photodegraded[W] = photo_degraded_now[W]
                                if S_deg:
                                    self.elements.mass_photodegraded[S] = 0.
                                if SPM_deg:
                                    self.elements.mass_photodegraded[SPM] = 0.
                            else:
                                self.elements.mass_photodegraded[W] = self.elements.mass_photodegraded[W] + photo_degraded_now[W]
                    else:
                        if Save_degr_now:
                            self.elements.mass_photodegraded = np.zeros(self.num_elements_active())

                    if Bio_degr is True and k_DecayMax_water > 0:
                        if np.sum(k_W_bio) > 0 or np.sum(k_S_bio) > 0:
                            bio_degraded_now = np.zeros(self.num_elements_active())
                            if np.sum(k_W_bio) > 0:
                                k_W_bio_fraction = np.minimum((k_W_bio / 3600) / np.maximum(k_W_fin, 1e-12), 1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation
                                bio_degraded_now[W] = degraded_now[W] * k_W_bio_fraction
                            if np.sum(k_S_bio) > 0:
                                k_S_bio_fraction = np.minimum((k_S_bio / 3600) / np.maximum(k_S_fin, 1e-12), 1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation
                                bio_degraded_now[S] = degraded_now[S] * k_S_bio_fraction
                            if Save_degr_now is True:
                                self.elements.mass_biodegraded[W] = bio_degraded_now[W]
                                self.elements.mass_biodegraded[S] = bio_degraded_now[S]
                                if SPM_deg:
                                    self.elements.mass_biodegraded[SPM] = 0.

                                self.elements.mass_biodegraded_water[W] = bio_degraded_now[W]
                                if S_deg:
                                    self.elements.mass_biodegraded_water[S] = 0.
                                if SPM_deg:
                                    self.elements.mass_biodegraded_water[SPM] = 0.

                                self.elements.mass_biodegraded_sediment[S] = bio_degraded_now[S]
                                if W_deg:
                                    self.elements.mass_biodegraded_sediment[W] = 0.
                                if SPM_deg:
                                    self.elements.mass_biodegraded_sediment[SPM] = 0.

                            else:
                                self.elements.mass_biodegraded = self.elements.mass_biodegraded + bio_degraded_now
                                self.elements.mass_biodegraded_water[W] = self.elements.mass_biodegraded_water[W] + bio_degraded_now[W]
                                self.elements.mass_biodegraded_sediment[S] = self.elements.mass_biodegraded_sediment[S] + bio_degraded_now[S]
                    else:
                        if Save_degr_now:
                            self.elements.mass_biodegraded = np.zeros(self.num_elements_active())
                            self.elements.mass_biodegraded_water = np.zeros(self.num_elements_active())
                            self.elements.mass_biodegraded_sediment = np.zeros(self.num_elements_active())

                    if Hydro_degr is True:
                        if np.sum(k_W_hydro) > 0 or np.sum(k_S_hydro) > 0:
                            hydro_degraded_now = np.zeros(self.num_elements_active())
                            if np.sum(k_W_hydro) > 0:
                                k_W_hydro_fraction = np.minimum((k_W_hydro / 3600) / np.maximum(k_W_fin, 1e-12), 1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation
                                hydro_degraded_now[W] = degraded_now[W] * k_W_hydro_fraction
                            if np.sum(k_S_hydro) > 0:
                                k_S_hydro_fraction = np.minimum((k_S_hydro / 3600) / np.maximum(k_S_fin, 1e-12), 1.0) # from 1/h to 1/s, clamp fraction to 1 to avoid breaking mass conservation
                                hydro_degraded_now[S] = degraded_now[S] * k_S_hydro_fraction

                            if Save_degr_now is True:
                                self.elements.mass_hydrolyzed[W] = hydro_degraded_now[W]
                                self.elements.mass_hydrolyzed[S] = hydro_degraded_now[S]
                                if SPM_deg:
                                    self.elements.mass_hydrolyzed[SPM] = 0.

                                self.elements.mass_hydrolyzed_water[W] = hydro_degraded_now[W]
                                if S_deg:
                                    self.elements.mass_hydrolyzed_water[S] = 0.
                                if SPM_deg:
                                    self.elements.mass_hydrolyzed_water[SPM] = 0.

                                self.elements.mass_hydrolyzed_sediment[S] = hydro_degraded_now[S]
                                if W_deg:
                                    self.elements.mass_hydrolyzed_sediment[W] = 0.
                                if SPM_deg:
                                    self.elements.mass_hydrolyzed_sediment[SPM] = 0.

                            else:
                                self.elements.mass_hydrolyzed[W] = self.elements.mass_hydrolyzed[W] + hydro_degraded_now[W]
                                self.elements.mass_hydrolyzed[S] = self.elements.mass_hydrolyzed[S] + hydro_degraded_now[S]
                                self.elements.mass_hydrolyzed_water[W] = self.elements.mass_hydrolyzed[W] + hydro_degraded_now[W]
                                self.elements.mass_hydrolyzed_sediment[S] = self.elements.mass_hydrolyzed_sediment[S] + hydro_degraded_now[S]
                    else:
                        if Save_degr_now:
                            self.elements.mass_hydrolyzed = np.zeros(self.num_elements_active())
                            self.elements.mass_hydrolyzed_water = np.zeros(self.num_elements_active())
                            self.elements.mass_hydrolyzed_sediment = np.zeros(self.num_elements_active())

                    total_fraction = (k_W_photo_fraction + k_W_bio_fraction + k_W_hydro_fraction)
                    assert np.all(total_fraction <= 1.0 + 1e-6), "Degradation fractions exceed 100%"

                if (k_S_fin_sum > 0) or (k_W_fin_sum > 0):
                    self.elements.mass_degraded = self.elements.mass_degraded + degraded_now

                    if Save_degr_now:
                        if W_deg:
                            self.elements.mass_degraded_now[W] = degraded_now[W]
                        if S_deg:
                            self.elements.mass_degraded_now[S] = degraded_now[S]
                        if SPM_deg:
                            self.elements.mass_degraded_now[SPM] = 0.

                        if W_deg:
                            self.elements.mass_degraded_water[W] = degraded_now[W]
                        if S_deg:
                            self.elements.mass_degraded_water[S] = 0.
                        if SPM_deg:
                            self.elements.mass_degraded_water[SPM] = 0.

                        if S_deg:
                            self.elements.mass_degraded_sediment[S] = degraded_now[S]
                        if W_deg:
                            self.elements.mass_degraded_sediment[W] = 0.
                        if SPM_deg:
                            self.elements.mass_degraded_sediment[SPM] = 0.

                    else:
                        self.elements.mass_degraded_water[W] = self.elements.mass_degraded_water[W] + degraded_now[W]
                        self.elements.mass_degraded_sediment[S] = self.elements.mass_degraded_sediment[S] + degraded_now[S]

                    self.elements.mass = self.elements.mass - degraded_now
                    self.elements.mass = np.maximum(self.elements.mass, 0.0)

                    self.deactivate_elements(self.elements.mass < (self.elements.mass + self.elements.mass_degraded + self.elements.mass_volatilized)/500,
                                             reason='removed')
                else:
                    if Save_degr_now:
                        self.elements.mass_degraded_now = np.zeros(self.num_elements_active())
                        self.elements.mass_degraded_water = np.zeros(self.num_elements_active())
                        self.elements.mass_degraded_sediment = np.zeros(self.num_elements_active())

                assert np.isclose(np.sum(self.elements.mass_hydrolyzed),
                                  np.sum(self.elements.mass_hydrolyzed_sediment) + np.sum(self.elements.mass_hydrolyzed_water),
                                  rtol=1e-5, atol=1e-8), "Inconsistent sum of mass hydrolized in wat and sed"

                assert np.isclose(np.sum(self.elements.mass_biodegraded),
                                  np.sum(self.elements.mass_biodegraded_sediment) + np.sum(self.elements.mass_biodegraded_water),
                                  rtol=1e-5, atol=1e-8), "Inconsistent sum of mass biodegraded in wat and sed"

                if Save_degr_now is True:
                    assert np.isclose(np.sum(self.elements.mass_degraded_now),
                                      np.sum(self.elements.mass_degraded_sediment) + np.sum(self.elements.mass_degraded_water),
                                      rtol=1e-5, atol=1e-8), "Inconsistent sum of mass degraded in water and sediment"
                    if self.get_config('chemical:transformations:Save_single_degr_mass') is True:
                        # print(f"mass_degraded_now sum: {np.sum(self.elements.mass_degraded_now)}")
                        # print(f"sum of mechanisms: {np.sum(self.elements.mass_hydrolyzed)} + {np.sum(self.elements.mass_biodegraded)} + {np.sum(self.elements.mass_photodegraded)}")
                        # print(f"total mechanisms sum: {np.sum(self.elements.mass_hydrolyzed) + np.sum(self.elements.mass_biodegraded) + np.sum(self.elements.mass_photodegraded)}")
                        assert np.isclose(np.sum(self.elements.mass_degraded_now),
                                          np.sum(self.elements.mass_hydrolyzed) + np.sum(self.elements.mass_biodegraded) + np.sum(self.elements.mass_photodegraded),
                                          rtol=1e-5, atol=1e-8), "Inconsistent sum of mass degraded now and single mechanism"
                else:
                    assert np.isclose(np.sum(self.elements.mass_degraded),
                                      np.sum(self.elements.mass_degraded_sediment) + np.sum(self.elements.mass_degraded_water),
                                      rtol=1e-5, atol=1e-8), "Inconsistent sum of mass degraded in water and sediment"
                    if self.get_config('chemical:transformations:Save_single_degr_mass') is True:
                        assert np.isclose(np.sum(self.elements.mass_degraded),
                                          np.sum(self.elements.mass_hydrolyzed) + np.sum(self.elements.mass_biodegraded) + np.sum(self.elements.mass_photodegraded),
                                          rtol=1e-5, atol=1e-8), "Inconsistent sum of mass degraded now and single mechanism"

                # print(f"mass_hydrolyzed_sediment: {np.sum(self.elements.mass_hydrolyzed_sediment)/np.sum(self.elements.mass_hydrolyzed)}")
                # print(f"mass_hydrolyzed_water: {np.sum(self.elements.mass_hydrolyzed_water)/np.sum(self.elements.mass_hydrolyzed)}")
                # print(f"mass_biodegraded_sediment: {np.sum(self.elements.mass_biodegraded_sediment)/np.sum(self.elements.mass_biodegraded)}")
                # print(f"mass_biodegraded_water: {np.sum(self.elements.mass_biodegraded_water)/np.sum(self.elements.mass_biodegraded)}")
                # print(f"mass_degraded_sediment: {np.sum(self.elements.mass_degraded_sediment)/np.sum(self.elements.mass_degraded_now)}")
                # print(f"mass_degraded_water: {np.sum(self.elements.mass_degraded_water)/np.sum(self.elements.mass_degraded_now)}")
        else:
            pass

    def volatilization(self):
        if self.get_config('chemical:transformations:volatilization') is True:
            Save_degr_now = self.get_config('chemical:transformations:Save_degr_now')
            logger.debug('Calculating: volatilization')
            volatilized_now = np.zeros(self.num_elements_active())

            MolWtCO2=44.009
            MolWtH2O=18.015
            MolWt=self.get_config('chemical:transformations:MolWt')
            wind=5                  # (m/s) (to read from atmosferic forcing)
            mixedlayerdepth=50      # m     (to read from ocean forcing)

            diss = self.get_config('chemical:transformations:dissociation')

            pKa_acid = self.get_config('chemical:transformations:pKa_acid')
            if pKa_acid < 0 and diss in ['amphoter', 'acid']:
                raise ValueError("pKa_acid must be positive")
            else:
                pass

            pKa_base = self.get_config('chemical:transformations:pKa_base')
            if pKa_base < 0 and diss in ['amphoter', 'base']:
                raise ValueError("pKa_base must be positive")
            else:
                pass

            if diss == 'amphoter' and abs(pKa_acid - pKa_base) < 2:
                raise ValueError("pKa_base and pKa_acid must differ of at least two units")
            else:
                pass

            # mask of dissolved elements within mixed layer
            W =     (self.elements.specie == self.num_lmm) \
                  * (-self.elements.z <= mixedlayerdepth)
                    # does volatilization apply only to num_lmm?
                    # check

            mixedlayerdepth = self.environment.ocean_mixed_layer_thickness
            mixedlayerdepth = mixedlayerdepth[W]

            T=self.environment.sea_water_temperature[W]
            #T[T==0]=np.median(T)                            # temporary fix for missing values

            S=self.environment.sea_water_salinity[W]

            wind=(self.environment.x_wind[W]**2 + self.environment.y_wind[W]**2)**.5

            Vp=self.get_config('chemical:transformations:Vpress')
            Tref_Vp=self.get_config('chemical:transformations:Tref_Vpress')
            DH_Vp=self.get_config('chemical:transformations:DeltaH_Vpress')

            Slb=self.get_config('chemical:transformations:Solub')
            Tref_Slb=self.get_config('chemical:transformations:Tref_Solub')
            DH_Slb=self.get_config('chemical:transformations:DeltaH_Solub')

            R=8.206e-05 #(atm m3)/(mol K)

            H0 = self.get_config('chemical:transformations:Henry') # (atm m3/mol)

            if H0 < 0:
                if Vp > 0 and Slb > 0:
                    logger.debug("Henry constant calculated from Vp and Slb")
                    Henry=(      (Vp * self.tempcorr("Arrhenius",DH_Vp,T,Tref_Vp)))   \
                               / (Slb *  self.tempcorr("Arrhenius",DH_Slb,T,Tref_Slb))  \
                               * MolWt / 101325.    # atm m3 mol-1
                else:
                    raise ValueError("Vp, Slb, and Henry not specified")
            else:
                logger.debug("Henry constant calculated from chemical:transformations:Henry")
                Henry = H0 * np.exp((DH_Slb - DH_Vp) / R * ((1.0 / T) - (1.0 / Tref_Slb)))

            # Calculate mass transfer coefficient water side
            # Schwarzenbach et al., 2016 Eq.(19-20)

            pH_water = self.environment.sea_water_ph_reported_on_total_scale[W]

            if diss == 'nondiss':
                Undiss_n = 1
            elif diss == 'acid':
                # Only undissociated chemicals volatilize
                Undiss_n = 1 / (1 + 10 ** (pH_water - pKa_acid))
            elif diss == 'base':
                # Dissociation in water of conjugated acid: dissociated form is neutral
                Undiss_n = 1- (1 / (1 + 10 ** (pH_water - pKa_base)))
            elif diss == 'amphoter':
                # Only undissociated chemicals volatilize # This approach ignores the zwitterionic fraction. 10.1002/etc.115
                Undiss_n = 1 / (1 + 10 ** (pH_water - pKa_acid) + 10 ** (pKa_base))

            MTCw = (((9e-4) + (7.2e-6 * wind ** 3)) * (MolWtCO2 / MolWt) ** 0.25) * Undiss_n
            # Calculate mass transfer coefficient air side
            # Schwarzenbach et al., 2016 Eq.(19-17)(19-18)(19-19)

            # Simple
            #MTCaH2O = 0.1 + 0.11 * wind

            # More complex
            Sca_H2O = 0.62                                  # 0.6 in the book. check
            MTCaH2O = 0.1 + wind*(6.1+0.63*wind)**0.5 \
                /(13.3*(Sca_H2O)**0.5 + (6.1e-4+(6.3e-5)*wind)**-0.5 -5 + 1.25*np.log(Sca_H2O) )

            MTCa = MTCaH2O * (MolWtH2O/MolWt)**(1/3)

            # Calculate overall volatilization mass tansfer coefficient

            HenryLaw = Henry * (1 + 0.01143 * S) / ( R * (T+273.15) )

            MTCvol = 1 / ( 1/MTCw + 1/(MTCa * HenryLaw))     # (cm/s)
            #mixedlayerdepth = self.environment.ocean_mixed_layer_thickness[W]
            #Thick = np.clip(self.environment.sea_floor_depth_below_sea_level[W],0,mixedlayerdepth) # (m)
            Thick = mixedlayerdepth

            # Degubbing information to screen
            #print('################### Volatilization-info ##################')
            #print('Mixed Layer   ',len(mixedlayerdepth),min(mixedlayerdepth),max(mixedlayerdepth),'m')
            #print('Temperature   ',len(T),min(T),max(T),'C')
            #print('Salinity      ',len(S),min(S),max(S))
            #print('Henry         ',len(Henry),min(Henry),max(Henry),'atm m3 / mol')
            #print('HenryLaw      ',len(HenryLaw),min(HenryLaw),max(HenryLaw))
            #print('wind          ',len(wind),min(wind),max(wind), 'm/s')
            #print('MTCa          ',len(MTCa),min(MTCa),max(MTCa),'cm/s')
            #print('MTCw          ',len(MTCw),min(MTCw),max(MTCw),'cm/s')
            #print('MTCa*HenryLaw ',len(MTCa*HenryLaw),min(MTCa*HenryLaw),max(MTCa*HenryLaw),'cm/s')
            #print('MTCvol        ',len(MTCvol),min(MTCvol),max(MTCvol),'cm/s')

            K_volatilization = 0.01 * MTCvol / Thick # (1/s)

            #logger.debug('MTCa: %s cm/s' % MTCa)
            #logger.debug('MTCw: %s cm/s' % MTCw)
            #logger.debug('Henry: %s ' % HenryLaw)
            #logger.debug('MTCvol: %s cm/s' % MTCvol)
            #logger.debug('T: %s C' % T)
            #logger.debug('S: %s ' % S)
            #logger.debug('Thick: %s ' % Thick)

            volatilized_now[W] = self.elements.mass[W] * (1-np.exp(-K_volatilization * self.time_step.total_seconds()))
            if Save_degr_now:
                self.elements.mass_volatilized_now[W] = volatilized_now[W]

                Sed =   (self.elements.specie == self.num_srev) | (self.elements.specie == self.num_ssrev)
                SPM = (self.elements.specie == self.num_prev)
                if np.any(Sed):
                    self.elements.mass_volatilized_now[Sed] = np.zeros(np.count_nonzero(Sed) )
                if np.any(SPM):
                    self.elements.mass_volatilized_now[SPM] = np.zeros(np.count_nonzero(SPM) )


            self.elements.mass_volatilized = self.elements.mass_volatilized + volatilized_now
            self.elements.mass = self.elements.mass - volatilized_now
            self.elements.mass = np.maximum(self.elements.mass, 0.0)

            self.deactivate_elements(self.elements.mass < (self.elements.mass + self.elements.mass_degraded + self.elements.mass_volatilized)/500,
                                     reason='removed')

        else:
            pass

    def update(self):
        """Update positions and properties of Chemical particles."""

        # Workaround due to conversion of datatype
        self.elements.specie = self.elements.specie.astype(np.int32)

        # Degradation and Volatilization
        if self.get_config('chemical:transfer_setup')=='organics':
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

        # Resuspension
        self.resuspension()
        logger.info('partitioning: {} {}'.format([sum(self.elements.specie==ii) for ii in range(self.nspecies)],self.name_species))

        # Horizontal advection
        self.advect_ocean_current()

        # Vertical advection
        if self.get_config('drift:vertical_advection') is True:
            self.vertical_advection()

        # Update transfer rates after last time step
        if      self.time == (self.expected_end_time - self.time_step) or \
                self.time == (self.expected_end_time) or \
                self.num_elements_active() == 0 :
            self.update_transfer_rates()



# ################
# POSTPROCESSING
    def simulation_summary(self, chemical_compound):
        '''Print a summary of the simulation: number of elements, number of transformations
        and final speciation
        '''

        print(chemical_compound)

        print('Final speciation:')
        for isp,sp in enumerate(self.name_species):
            print ('{:32}: {:>6}'.format(sp,sum(self.elements.specie==isp)))

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
                                              landmask_shapefile=None,
                                              landmask_bathymetry_thr=None,
                                              origin_marker=None,
                                              elements_density=False,
                                              active_status=False,
                                              weight=None,
                                              sim_description=None):
        '''Write netCDF file with map of Chemical species densities and concentrations
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
            mass_unit:             string, mass unit of output concentration (ug/mg/g/kg)
            time_avg_conc:         boolean, calculate concentration averaged each deltat
            time_start:            string, datetime64[ns] string for start time of concentration map
            time_end:              string, datetime64[ns] string for end time of concentration map
            time_chunk_size        int, number of timesteps computed per chunk
            horizontal_smoothing:  boolean, smooth concentration horizontally
            smoothing_cells:       int, number of cells for horizontal smoothing,
            reader_sea_depth:      string, path of bathimethy .nc file,
            landmask_shapefile:    string, path of bathimethylandmask .shp file
            landmask_bathymetry_thr:   float32, if set the value is the threshold used to extract the landmask from reader_sea_depth
            elements_density:      boolean, add number of elements present in each grid cell to output
            origin_marker:         int/list/tuple/np.ndarray, only elements with these values of "origin_marker" will be considered
            active_status:         boolean, only active elements will be considered
            weight:                string, elements property to be extracted to produce maps
            sim_description:       string, descrition of simulation to be included in netcdf attributes
        '''

        from netCDF4 import Dataset, date2num #, stringtochar
        import opendrift
        from pyproj import CRS, Proj, Transformer
        import pandas as pd
        import gc
        from datetime import timedelta


        def is_valid_proj4(density_proj):
            try:
                CRS.from_string(density_proj)  # Try to create a CRS object
                return density_proj
            except:
                try:
                    density_proj = (CRS.from_epsg(density_proj)).to_proj4() # Try to create a CRS object from EPSG number
                    return density_proj
                except:
                    raise ValueError(f"Invalid density_proj: {density_proj}")

        if sum(x is None for x in [lat_resol, lon_resol]) == 1:
            raise ValueError("Both lat/lon_resol must be specified")
        elif sum(x is None for x in [lat_resol, lon_resol]) == 0:
            if pixelsize_m is not None:
                raise ValueError("If lat/lon_resol are specified pixelsize_m must be None")

        if self.mode != opendrift.models.basemodel.Mode.Config:
            self.mode = opendrift.models.basemodel.Mode.Config
            logger.debug("Changed self.mode to Config")

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
                logger.warning(f"No timesteps fall within time_start: {time_start} and time_end: {time_end}.")
                return
            # Save for later use when writing 'time' coords and averaging
            filtered_times = np.array(all_times)[tmask]
        else:
            tmask = None
            filtered_times = all_times

        if landmask_shapefile is not None:
            if 'shape' in self.env.readers.keys():
                # removing previously stored landmask
                del self.env.readers['shape']
            # Adding new landmask
            from opendrift.readers import reader_shape
            custom_landmask = reader_shape.Reader.from_shpfiles(landmask_shapefile)
            self.add_reader(custom_landmask)
        elif 'global_landmask' not in self.env.readers.keys():
            from opendrift.readers import reader_global_landmask
            global_landmask = reader_global_landmask.Reader()
            self.add_reader(global_landmask)

        if reader_sea_depth is not None:
            from opendrift.readers import reader_netCDF_CF_generic
            import xarray as xr

            reader_sea_depth_res = xr.open_dataset(reader_sea_depth)
            reader_sea_depth_res = (reader_sea_depth_res[list(reader_sea_depth_res.data_vars)[0]])

            lat_names = ["latitude", "lat", "y"]
            lat_name = next((name for name in lat_names if name in reader_sea_depth_res.coords), None)
            lon_names = ["longitude","lon", "x", "long"]
            lon_name = next((name for name in lon_names if name in reader_sea_depth_res.coords), None)

            if any(x is None for x in [lat_name, lon_name]):
                raise ValueError("Latitude/Longitude coordinate names not found in bathimetry")

            # Check if corners are covered by bathimetry
            reader_sea_depth_lat_values = reader_sea_depth_res.coords[lat_name].values
            reader_sea_depth_lon_values = reader_sea_depth_res.coords[lon_name].values

            reader_sea_depth_lon_min = min(reader_sea_depth_lon_values)
            reader_sea_depth_lat_min = min(reader_sea_depth_lat_values)
            reader_sea_depth_lon_max = max(reader_sea_depth_lon_values)
            reader_sea_depth_lat_max = max(reader_sea_depth_lat_values)

            if ((llcrnrlat < reader_sea_depth_lat_min) or (urcrnrlat > reader_sea_depth_lat_max)\
            or (llcrnrlon < reader_sea_depth_lon_min) or  (urcrnrlon > reader_sea_depth_lon_max)):
                if llcrnrlat < reader_sea_depth_lat_min:
                    logger.warning(f"Changed llcrnrlat from {llcrnrlat} to {str(reader_sea_depth_lat_values[1])[:8]}")
                    llcrnrlat = reader_sea_depth_lat_values[1]
                if urcrnrlat > reader_sea_depth_lat_max:
                    logger.warning(f"Changed urcrnrlat from {urcrnrlat} to {str(reader_sea_depth_lat_values[-2])[:8]}")
                    urcrnrlat = reader_sea_depth_lat_values[-2]
                if llcrnrlon < reader_sea_depth_lon_min:
                    logger.warning(f"Changed llcrnrlon from {llcrnrlon} to {str(reader_sea_depth_lon_values[1])[:8]}")
                    llcrnrlon = reader_sea_depth_lon_values[1]
                if urcrnrlon > reader_sea_depth_lon_max:
                    logger.warning(f"Changed urcrnrlon from {urcrnrlon} to {str(reader_sea_depth_lon_values[-2])[:8]}")
                    urcrnrlon = reader_sea_depth_lon_values[-2]
            else:
                pass

            # Find resolution of bathimetry's selected section
            reader_sea_depth_res=reader_sea_depth_res.where(
                                            (reader_sea_depth_res[lon_name] > llcrnrlon) &
                                            (reader_sea_depth_res[lon_name] < urcrnrlon) &
                                            (reader_sea_depth_res[lat_name] > llcrnrlat) &
                                            (reader_sea_depth_res[lat_name] < urcrnrlat),
                                            drop=True)
            # Update lat/lon values of final selection
            reader_sea_depth_lat_values = reader_sea_depth_res.coords[lat_name].values
            reader_sea_depth_lon_values = reader_sea_depth_res.coords[lon_name].values

            num_x = None
            num_y = None

            num_x = reader_sea_depth_lon_values.size
            if num_x == 0:
                raise ValueError("No longitude coordinate found in bathimetry")
            num_y = reader_sea_depth_lat_values.size
            if num_y == 0:
                raise ValueError("No latitude coordinate found in bathimetry")

            del reader_sea_depth_res
            # Load bathimetry as reader for interpolation
            reader_sea_depth = reader_netCDF_CF_generic.Reader(reader_sea_depth)
        else:
            raise ValueError('A reader for ''sea_floor_depth_below_sea_level'' must be specified')


        if self.mode != opendrift.models.basemodel.Mode.Result:
            self.mode = opendrift.models.basemodel.Mode.Result
            logger.debug("Changed self.mode to Result")

        # Temporary workaround if self.nspecies and self.name_species are not defined
        # TODO Make sure that these are saved when the simulation data is saved to the ncdf file
        # Then this workaround can be removed
        if not hasattr(self,'nspecies'):
            self.nspecies=4
        if not hasattr(self,'name_species'):
            self.name_species = ['dissolved',
                                 'DOC',
                                 'SPM',
                                 'sediment']

        logger.info('Postprocessing: Write density and concentration to netcdf file')

        # Default bathymetry resolution 500x500. Can be increased (carefully) if high-res data is available and needed
        bathimetry_res = 500
        if num_x > 500 and num_y > 500:
            bathimetry_res = min(num_x, num_y)-1
            logger.warning(f"Changed bathymetry resolution to {bathimetry_res}")

        grid=np.meshgrid(np.linspace(llcrnrlon,urcrnrlon,bathimetry_res), np.linspace(llcrnrlat,urcrnrlat,bathimetry_res))
        # grid=np.meshgrid(np.linspace(llcrnrlon,urcrnrlon,500), np.linspace(llcrnrlat,urcrnrlat,500))
        self.conc_lon=grid[0]
        self.conc_lat=grid[1]
        self.conc_topo=reader_sea_depth.get_variables_interpolated_xy(['sea_floor_depth_below_sea_level'],
                x = self.conc_lon.flatten(),
                y = self.conc_lat.flatten(),
                time = reader_sea_depth.times[0] if reader_sea_depth.times is not None else None
                )[0]['sea_floor_depth_below_sea_level'].reshape(self.conc_lon.shape)


        if pixelsize_m == 'auto':
            # lon = self.result.lon
            lat = self.result.lat
            latspan = lat.max()-lat.min()
            pixelsize_m=30
            if latspan > .05:
                pixelsize_m = 50
            if latspan > .1:
                pixelsize_m = 300
            if latspan > .3:
                pixelsize_m = 500
            if latspan > .7:
                pixelsize_m = 1000
            if latspan > 2:
                pixelsize_m = 2000
            if latspan > 5:
                pixelsize_m = 4000


        if density_proj is None: # add default projection with equal-area property
            density_proj_str = ('+proj=moll +ellps=WGS84 +lon_0=0.0')
            density_proj = pyproj.Proj('+proj=moll +ellps=WGS84 +lon_0=0.0')
        else:
            density_proj_str = density_proj
            density_proj = pyproj.Proj(is_valid_proj4(density_proj))

        if sum(x is None for x in [lat_resol, lon_resol]) == 0:
            if density_proj != pyproj.Proj('+proj=moll +ellps=WGS84 +lon_0=0.0'):
                if density_proj != pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"):
                    # lat/lon_resol are calculated at the centre of the grid
                    # Unexpected behaviour may arise with large grids
                    source_proj = Proj("+proj=longlat +datum=WGS84 +no_defs")
                    transformer = Transformer.from_proj(source_proj, density_proj)
                    dummy_lon, dummy_lat = (llcrnrlon + urcrnrlon)/2, (llcrnrlat + urcrnrlat)/2
                    x1, y1 = transformer.transform(dummy_lon, dummy_lat)
                    x2, y2 = transformer.transform(dummy_lon + lon_resol, dummy_lat + lat_resol)
                    lon_resol = abs(x2 - x1)
                    lat_resol = abs(y2 - y1)
                    logger.info(f'Changed lon_resol, lat_resol to reference system: {density_proj}')


        if mass_unit==None:
            mass_unit='microgram'  # default unit for chemicals



        z = (self.result.z.T).values
        # Move elements above sea level below the surface (-1 mm)
        z_mask = np.isfinite(z) & (z >= 0)
        if z_mask.any():
            z_positive = int(z_mask.sum())
            z[z_mask] = -1e-4
            logger.warning(f'{z_positive} elements were above surface level and were moved to z = -0.0001')
        del z_mask
        if not zlevels==None:
            zlevels = np.sort(zlevels)
            z_array = np.append(np.append(-10000, zlevels) , max(0,np.nanmax(z)))
        else:
            z_array = [min(-10000,np.nanmin(z)), max(0,np.nanmax(z))]
        logger.info('vertical grid boundaries: {}'.format([str(item) for item in z_array]))

        # H is array containing the mass of chemical within each box defined by lon_array, lat_array and z_array
        # H_count is array containing the number of elements within each box defined by lon_array, lat_array and z_array
        if weight is None:
            weight = 'mass'
        H, lon_array, lat_array, H_count = \
            self.get_chemical_density_array(pixelsize_m=pixelsize_m,
                                           z_array=z_array, z=z,
                                           lat_resol=lat_resol,
                                           lon_resol=lon_resol,
                                           density_proj=density_proj,
                                           llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                                           urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                                           weight=weight, origin_marker=origin_marker,
                                           active_status=active_status,
                                           elements_density=elements_density,
                                           time_start=time_start,
                                           time_end=time_end,
                                           time_chunk_size = time_chunk_size)

        # calculating center point for each pixel
        lon_array = (lon_array[:-1,:-1] + lon_array[1:,1:])/2
        lat_array = (lat_array[:-1,:-1] + lat_array[1:,1:])/2

        landmask = np.zeros_like(H[0,0,0,:,:])
        if (landmask_bathymetry_thr is not None) and (reader_sea_depth is not None):
            landmask = reader_sea_depth.get_variables_interpolated_xy(['sea_floor_depth_below_sea_level'],
                x = np.clip(lon_array.flatten(),reader_sea_depth.xmin,reader_sea_depth.xmax),
                y = np.clip(lat_array.flatten(),reader_sea_depth.ymin,reader_sea_depth.ymax),
                time = reader_sea_depth.times[0] if reader_sea_depth.times is not None else None
                )[0]['sea_floor_depth_below_sea_level'] <= landmask_bathymetry_thr #.reshape(self.conc_lon.shape)
        elif landmask_shapefile is not None:
            landmask = self.env.readers['shape'].get_variables('land_binary_mask', x=lon_array,y=lat_array)['land_binary_mask']
        else:
            landmask = self.env.readers['global_landmask'].get_variables('land_binary_mask', x=lon_array,y=lat_array)['land_binary_mask']

        if landmask.shape !=  lon_array.shape:
            landmask = landmask.reshape(lon_array.shape)

        if horizontal_smoothing:
            # Compute horizontally smoother field
            logger.debug('H.shape: ' + str(H.shape))

            Hsm = np.array([
                            [[self.horizontal_smooth(H[ti, sp, zi, :, :], n=smoothing_cells)
                              for zi in range(len(z_array) - 1)]
                             for sp in range(self.nspecies)]
                            for ti in range(H.shape[0])
                        ])

        # Compute mean depth and volume in each pixel grid cell
        pixel_mean_depth, pixel_area  =  self.get_pixel_mean_depth(lon_array, lat_array,
                                                      density_proj_str,
                                                      lat_resol, lon_resol)

        def _remove_reader_by_object(self, rdr_obj):
            for key, rdr in list(self.env.readers.items()):
                if rdr is rdr_obj:
                    self.env.readers.pop(key, None)
                    if hasattr(rdr, "close"):
                        try: rdr.close()
                        except: pass
                    break
        _remove_reader_by_object(self, 'shape')
        _remove_reader_by_object(self, 'global_landmask')
        _remove_reader_by_object(self, 'reader_sea_depth')

        pixel_volume = np.zeros_like(H[0,0,:,:,:])

        for zi,zz in enumerate(z_array[:-1]):
            topotmp = -pixel_mean_depth.copy()
            topotmp[np.where(topotmp < zz)] = zz
            topotmp = z_array[zi+1] - topotmp
            topotmp[np.where(topotmp < .1)] = 0.
            if density_proj_str == ('+proj=moll +ellps=WGS84 +lon_0=0.0'):
                pixel_volume[zi,:,:] = topotmp * pixelsize_m**2
            else:
                pixel_volume[zi,:,:] = topotmp * pixel_area

        pixel_volume[np.where(pixel_volume==0.)] = np.nan

        # Compute mass of dry sediment in each pixel grid cell
        sed_L       = self.get_config('chemical:sediment:mixing_depth')
        sed_dens    = self.get_config('chemical:sediment:density')
        sed_poro    = self.get_config('chemical:sediment:porosity')
        if density_proj_str == ('+proj=moll +ellps=WGS84 +lon_0=0.0'):
            pixel_sed_mass = (pixelsize_m**2 *sed_L)*(1-sed_poro)*sed_dens      # mass in kg dry weight
        else:
            pixel_sed_mass = (pixel_area *sed_L)*(1-sed_poro)*sed_dens
            pixel_sed_mass = np.tile(pixel_sed_mass, (len(z_array[:-1]), 1, 1))

        # TODO this should be multiplied for the fraction of grid cell are that is not on land

        for ti in range(H.shape[0]):
            for sp in range(self.nspecies):
                if not self.name_species[sp].lower().startswith('sed'):
                    #print('divide by water volume')
                    H[ti,sp,:,:,:] = H[ti,sp,:,:,:] / pixel_volume
                    if horizontal_smoothing:
                        Hsm[ti,sp,:,:,:] = Hsm[ti,sp,:,:,:] / pixel_volume
                elif self.name_species[sp].lower().startswith('sed'):
                    #print('divide by sediment mass')
                    H[ti,sp,:,:,:] = H[ti,sp,:,:,:] / pixel_sed_mass
                    if horizontal_smoothing:
                        Hsm[ti,sp,:,:,:] = Hsm[ti,sp,:,:,:] / pixel_sed_mass

        times = filtered_times

        if time_avg_conc:
            conctmp = H[:-1,:,:,:,:]
            cshape = conctmp.shape
            mdt =    np.mean(times[1:] - times[:-1])    # output frequency in opendrift output file
            if deltat==None:
                ndt = 1
            else:
                if not isinstance(mdt, timedelta):
                    logger.warning(
                        "Mean timestep is not a datetime.timedelta object "
                        "(likely only one timestep available). Cannot calculate mean concentration.")
                    return
                else:
                    ndt = int( deltat / (mdt.total_seconds()/3600.))
            times2 = times[::ndt]
            times2 = times2[1:]
            odt = int(cshape[0]/ndt)
            logger.debug ('ndt '+ str(ndt))   # number of time steps over which to average in conc file
            logger.debug ('odt '+ str(odt))   # number of average slices

            try:
                mean_conc = np.mean(conctmp.reshape(odt, ndt, *cshape[1:]), axis=1)
            except: # If (times) is not a perfect multiple of deltat
                mean_conc = np.zeros([odt,cshape[1],cshape[2],cshape[3],cshape[4]])
                for ii in range(odt):
                    meantmp  = np.mean(conctmp[(ii*ndt):(ii+1)*ndt,:,:,:,:],axis=0)
                    mean_conc[ii,:,:,:,:] = meantmp

            if elements_density is True:
                denstmp = H_count[:-1,:,:,:,:]
                dshape = denstmp.shape
                try:
                    mean_dens = np.sum(denstmp.reshape(odt, ndt, *dshape[1:]), axis=1)
                except:
                    mean_dens = np.zeros([odt,dshape[1],dshape[2],dshape[3],dshape[4]])
                    for ii in range(odt):
                        meantmp  = np.mean(denstmp[(ii*ndt):(ii+1)*ndt,:,:,:,:],axis=0)
                        mean_dens[ii,:,:,:,:] = meantmp

            if horizontal_smoothing is True:
                Hsmtmp = Hsm[:-1,:,:,:,:]
                Hsmshape = Hsmtmp.shape
                try:
                    mean_Hsm = np.mean(Hsmtmp.reshape(odt, ndt, *Hsmshape[1:]), axis=1)
                except:
                    mean_Hsm = np.zeros([odt,Hsmshape[1],Hsmshape[2],Hsmshape[3],Hsmshape[4]])
                    for ii in range(odt):
                        meantmp  = np.mean(Hsmtmp[(ii*ndt):(ii+1)*ndt,:,:,:,:],axis=0)
                        Hsmtmp[ii,:,:,:,:] = meantmp


        # Save outputs to netCDF Dataset
        compound = self.get_config('chemical:compound')
        species_str = ' '.join([f"{isp}:{sp}" for isp, sp in enumerate(self.name_species)])

        nc = Dataset(filename, 'w')
        nc.createDimension('x', lon_array.shape[0])
        nc.createDimension('y', lon_array.shape[1])
        nc.createDimension('depth', len(z_array)-1)
        nc.createDimension('specie', self.nspecies)
        timestr = 'seconds since 1970-01-01 00:00:00'
        if time_avg_conc is False:
            nc.createDimension('time', H.shape[0])
            nc.createVariable('time', 'f8', ('time',))
            nc.variables['time'][:] = date2num(times, timestr)
            nc.variables['time'].units = timestr
            nc.variables['time'].standard_name = 'time'
        else:
            nc.createDimension('avg_time', odt)
            nc.createVariable('avg_time', 'f8', ('avg_time',))
            nc.variables['avg_time'][:] = date2num(times2, timestr)
            nc.variables['avg_time'].units = timestr


        # Projection
        nc.createVariable('projection', 'i8')
        nc.variables['projection'].proj4 = density_proj.definition_string()


        # Cell size
        if pixelsize_m is not None:
            nc.createVariable('cell_size','f8')
            nc.variables['cell_size'][:] = pixelsize_m
            nc.variables['cell_size'].long_name = 'Length of cell'
            nc.variables['cell_size'].unit = 'm'
        else:
            nc.createVariable('lat_resol','f8')
            nc.variables['lat_resol'][:] = lat_resol
            nc.variables['lat_resol'].long_name = 'Latitude resolution'
            nc.variables['lat_resol'].unit = 'degrees_north'
            nc.createVariable('lon_resol','f8')
            nc.variables['lon_resol'][:] = lon_resol
            nc.variables['lon_resol'].long_name = 'Longitude resolution'
            nc.variables['lon_resol'].unit = 'degrees_east'

        if horizontal_smoothing:
            # Horizontal smoothing cells
            nc.createVariable('smoothing_cells','i8')
            nc.variables['smoothing_cells'][:] = smoothing_cells
            nc.variables['smoothing_cells'].long_name = 'Number of cells in each direction for horizontal smoothing'
            nc.variables['smoothing_cells'].units = '1'


        # Coordinates
        nc.createVariable('lon', 'f8', ('y','x'))
        nc.createVariable('lat', 'f8', ('y','x'))
        nc.createVariable('depth', 'f8', ('depth',))
        nc.createVariable('specie', 'i4', ('specie',))
        nc.variables['lon'][:] = lon_array.T
        nc.variables['lon'].long_name = 'longitude'
        nc.variables['lon'].short_name = 'longitude'
        nc.variables['lon'].units = 'degrees_east'
        nc.variables['lat'][:] = lat_array.T
        nc.variables['lat'].long_name = 'latitude'
        nc.variables['lat'].short_name = 'latitude'
        nc.variables['lat'].units = 'degrees_north'
        nc.variables['depth'][:] = z_array[1:]
        nc.variables['specie'][:] = np.arange(self.nspecies)
        nc.variables['specie'].long_name = ' '.join(['{}:{}'.format(isp,sp) for isp,sp in enumerate(self.name_species)])

        # Create final landmask
        if time_avg_conc is False:
            Landmask = np.tile(landmask, (len(times), self.nspecies, len(z_array)-1, 1, 1))
        else:
            Landmask = np.tile(landmask, (odt, self.nspecies, len(z_array)-1, 1, 1))
        Landmask = np.swapaxes(Landmask, 3, 4)
        landmask_depth = np.tile(landmask[np.newaxis, :, :], (len(z_array)-1, 1, 1))
        landmask_depth = np.swapaxes(landmask_depth, 1, 2)

        # Density
        if elements_density is True:
            if time_avg_conc is False:
                nc.createVariable('density', 'i4',
                                  ('time','specie','depth','y', 'x'),fill_value=99999)
                H_count = np.swapaxes(H_count, 3, 4)
                H_count = np.nan_to_num(H_count, nan=0, posinf=0, neginf=0).astype('i4')
                H_count = np.ma.masked_where(Landmask==1, H_count)
                nc.variables['density'][:] = H_count
                nc.variables['density'].long_name = 'Number of elements in grid cell'
                nc.variables['density'].grid_mapping = density_proj_str
                nc.variables['density'].units = '1'
                if sim_description is not None:
                    nc.variables['density'].sim_description = str(sim_description)
            else:
                nc.createVariable('density_avg', 'i4',
                                  ('avg_time','specie','depth','y', 'x'),fill_value=99999)
                mean_dens = np.swapaxes(mean_dens, 3, 4)
                mean_dens = np.nan_to_num(mean_dens, nan=0, posinf=0, neginf=0).astype('i4')
                mean_dens = np.ma.masked_where(Landmask==1, mean_dens)
                nc.variables['density_avg'][:] = mean_dens
                nc.variables['density_avg'].long_name = "Number of elements in grid cell at avg_time"
                nc.variables['density_avg'].grid_mapping = density_proj_str
                nc.variables['density_avg'].units = '1'
                if sim_description is not None:
                    nc.variables['density_avg'].sim_description = str(sim_description)


        # Chemical concentration
        if time_avg_conc is False:
            nc.createVariable('concentration', 'f8',
                          ('time','specie','depth','y', 'x'),fill_value=1.e36)
            H = np.swapaxes(H, 3, 4)
            H = np.ma.masked_where(Landmask==1,H)
            nc.variables['concentration'][:] = H
            nc.variables['concentration'].long_name = (f"{compound} concentration of {weight}\n"
                                                           f"specie {species_str}")
            nc.variables['concentration'].grid_mapping = density_proj_str
            nc.variables['concentration'].units = mass_unit+'/m3'+' (sed '+mass_unit+'/Kg d.w.)'
            if sim_description is not None:
                nc.variables['concentration'].sim_description = str(sim_description)
        else:
        # Chemical concentration, time averaged
            nc.createVariable('concentration_avg', 'f8',
                              ('avg_time','specie','depth','y', 'x'),fill_value=+1.e36)
            mean_conc = np.swapaxes(mean_conc, 3, 4)
            mean_conc = np.ma.masked_where(Landmask==1, mean_conc)
            nc.variables['concentration_avg'][:] = mean_conc
            nc.variables['concentration_avg'].long_name = (f"{compound} time averaged concentration of {weight}\n"
                                                           f"specie {species_str}")
            nc.variables['concentration_avg'].grid_mapping = density_proj_str
            nc.variables['concentration_avg'].units = mass_unit+'/m3'+' (sed '+mass_unit+'/Kg)'
            if sim_description is not None:
                nc.variables['concentration_avg'].sim_description = str(sim_description)


        # Chemical concentration, horizontally smoothed
        if horizontal_smoothing is True:
            if time_avg_conc is False:
                nc.createVariable('concentration_smooth', 'f8',
                                  ('time','specie','depth','y', 'x'),fill_value=1.e36)
                Hsm = np.swapaxes(Hsm, 3, 4)
                Hsm = np.ma.masked_where(Landmask==1, Hsm)
                nc.variables['concentration_smooth'][:] = Hsm
                nc.variables['concentration_smooth'].long_name = (f"{compound} horizontally smoothed concentration of {weight}\n"
                                                               f"specie {species_str}")
                nc.variables['concentration_smooth'].grid_mapping = density_proj_str
                nc.variables['concentration_smooth'].units = mass_unit+'/m3'+' (sed '+mass_unit+'/Kg)'
                nc.variables['concentration_smooth'].comment = 'Smoothed over '+str(smoothing_cells)+' grid points in all horizontal directions'
                if sim_description is not None:
                    nc.variables['concentration_smooth'].sim_description = str(sim_description)
            else:
            # Chemical concentration, horizontally smoothed, time averaged
                nc.createVariable('concentration_smooth_avg', 'f8',
                                  ('avg_time','specie','depth','y', 'x'),fill_value=+1.e36)
                mean_Hsm = np.swapaxes(mean_Hsm, 3, 4)
                mean_Hsm = np.ma.masked_where(Landmask==1, mean_Hsm)
                nc.variables['concentration_smooth_avg'][:] = mean_Hsm
                nc.variables['concentration_smooth_avg'].long_name = (f"{compound} horizontally smoothed time averaged concentration of {weight}\n"
                                                               f"specie {species_str}")
                nc.variables['concentration_smooth_avg'].grid_mapping = density_proj_str
                nc.variables['concentration_smooth_avg'].units = mass_unit+'/m3'+' (sed '+mass_unit+'/Kg)'
                nc.variables['concentration_smooth_avg'].comment = 'Smoothed over '+str(smoothing_cells)+' grid points in all horizontal directions'
                if sim_description is not None:
                    nc.variables['concentration_smooth_avg'].sim_description = str(sim_description)


        # Volume of boxes
        nc.createVariable('volume', 'f8',
                          ('depth','y', 'x'),fill_value=0)
        pixel_volume = np.swapaxes(pixel_volume, 1, 2) #.astype('i4')
        pixel_volume = np.ma.masked_where(pixel_volume==0, pixel_volume)
        # pixel_volume = np.ma.masked_where(landmask_depth==1, pixel_volume)
        nc.variables['volume'][:] = pixel_volume
        if pixelsize_m is not None:
            nc.variables['volume'].long_name = f'Volume of grid cell ({str(pixelsize_m)} x {str(pixelsize_m)} m)'
        else:
            nc.variables['volume'].long_name = f'Volume of grid cell (lat_resol: {lat_resol} degrees, lon_resol: {lon_resol} degrees)'
        nc.variables['volume'].grid_mapping = density_proj_str
        nc.variables['volume'].units = 'm3'


        # Topography
        nc.createVariable('topo', 'f8', ('y', 'x'),fill_value=0)
        pixel_mean_depth = np.ma.masked_where(landmask==1, pixel_mean_depth)
        nc.variables['topo'][:] = pixel_mean_depth.T
        nc.variables['topo'].long_name = 'Depth of grid point'
        nc.variables['topo'].grid_mapping = density_proj_str
        nc.variables['topo'].units = 'm'
        if sim_description is not None:
            nc.variables['topo'].sim_description = str(sim_description)


        # Gridcell area
        if pixelsize_m is None:
            nc.createVariable('area', 'f8', ('y', 'x'),fill_value=0)
            pixel_area = np.ma.masked_where(landmask==1, pixel_area)
            nc.variables['area'][:] = pixel_area.T
            nc.variables['area'].long_name = 'Area of grid point'
            nc.variables['area'].grid_mapping = density_proj_str
            nc.variables['area'].units = 'm2'


        # Binary mask
        nc.createVariable('land', 'i4', ('y', 'x'),fill_value=-1)
        #landmask = np.ma.masked_where(landmask==0, landmask)
        nc.variables['land'][:] = np.swapaxes(landmask,0,1).astype('i4')
        nc.variables['land'].long_name = 'Binary land mask'
        nc.variables['land'].grid_mapping = density_proj_str
        nc.variables['land'].units = 'm'

        nc.close()
        logger.info('Wrote to '+filename)

        del H, pixel_volume, pixel_mean_depth
        if time_avg_conc is True:
            del mean_conc
        if elements_density is True:
            del H_count
            if time_avg_conc is True:
                del mean_dens
        if horizontal_smoothing is True:
            if time_avg_conc is True:
                del mean_Hsm
            else:
                del Hsm

        del lon_array, lat_array, landmask, Landmask
        gc.collect()


    def get_chemical_density_array(self, pixelsize_m, z_array, z,
                                   lat_resol=None, lon_resol=None,
                                   density_proj=None, llcrnrlon=None,llcrnrlat=None,
                                   urcrnrlon=None,urcrnrlat=None,
                                   weight=None, origin_marker=None,
                                   active_status = False,
                                   elements_density = False,
                                   time_start=None, time_end=None,
                                   time_chunk_size = 50):
        '''
        compute a particle concentration map from particle positions
        Use user defined projection (density_proj=<proj4_string>)
        or create a lon/lat grid (density_proj=None)
            Arguments are described at write_netcdf_chemical_density_map
        '''

        from pyproj import Proj, Transformer
        import pandas as  pd
        import gc

        # Time window (inclusive)
        times_1d = pd.to_datetime(self.result.time.values)  # (n_time,)
        if time_start is not None: time_start = pd.to_datetime(time_start)
        if time_end   is not None: time_end   = pd.to_datetime(time_end)

        if (time_start is not None) or (time_end is not None):
            tmask = np.ones(times_1d.shape, dtype=bool)
            if time_start is not None: tmask &= (times_1d >= time_start)
            if time_end   is not None: tmask &= (times_1d <= time_end)
            if not tmask.any():
                raise ValueError("No timesteps fall within [time_start, time_end].")
        else:
            tmask = slice(None)

        lon_2d    = (self.result.lon.T).values[tmask, :]
        lat_2d    = (self.result.lat.T).values[tmask, :]
        z_2d      = (self.result.z.T).values[tmask, :]
        specie_2d = (self.result.specie.T).values[tmask, :]

        # In-place validity mask (no extra array allocations)
        valid_lon = (lon_2d >= -180) & (lon_2d <= 180)
        valid_lat = (lat_2d >= -90)  & (lat_2d <= 90)
        valid_mask = valid_lon & valid_lat
        lon_2d[~valid_mask] = np.nan
        lat_2d[~valid_mask] = np.nan
        del valid_lon, valid_lat, valid_mask
        gc.collect()

        if weight is not None:
            weight_2d = (self.result[weight].T).values[tmask, :]
        if origin_marker is not None:
            originmarker_2d = (self.result.origin_marker.T).values[tmask, :]
        if active_status:
            status_2d = (self.result.status.T).values[tmask, :]

        # Keep mask in (n_timef, n_elem)
        n_timef, n_elem = lon_2d.shape
        keep_mask = np.ones((n_timef, n_elem), dtype=bool)

        if origin_marker is not None:
            if isinstance(origin_marker, (list, tuple, np.ndarray)):
                origin_mask = np.isin(originmarker_2d, origin_marker)
            else:
                origin_mask = (originmarker_2d == origin_marker)
            keep_mask &= origin_mask
            logger.warning(f'only active elements with origin_marker: {origin_marker} were considered for concentration')
            del origin_mask

        if active_status:
            status_categories = self.status_categories
            if 'active' not in status_categories:
                raise ValueError("No active elements in simulation")
            active_index = status_categories.index('active')
            keep_mask &= (status_2d == active_index)
            logger.warning(f'only active elements were considered for concentration, status: {active_index}')

        # Create a grid in the specified projection
        if llcrnrlon is not None:
            llcrnrx,llcrnry = density_proj(llcrnrlon,llcrnrlat)
            urcrnrx,urcrnry = density_proj(urcrnrlon,urcrnrlat)
        else:
            x,y = density_proj(lon_2d, lat_2d)
            if density_proj == pyproj.Proj('+proj=moll +ellps=WGS84 +lon_0=0.0'):
                llcrnrx,llcrnry = x.min()-pixelsize_m, y.min()-pixelsize_m
                urcrnrx,urcrnry = x.max()+pixelsize_m, y.max()+pixelsize_m
            else:
                llcrnrx,llcrnry = x.min()-lon_resol, y.min()-lat_resol
                urcrnrx,urcrnry = x.max()+lon_resol, y.max()+lat_resol
            del x,y
            gc.collect()

        if density_proj == pyproj.Proj('+proj=moll +ellps=WGS84 +lon_0=0.0'):
            if pixelsize_m == None:
                raise ValueError("If density_proj is '+proj=moll +ellps=WGS84 +lon_0=0.0', pixelsize_m must be specified")
            else:
                x_array = np.arange(llcrnrx,urcrnrx, pixelsize_m)
                y_array = np.arange(llcrnry,urcrnry, pixelsize_m)
        else:
            x_array = np.arange(llcrnrx,urcrnrx, lon_resol)
            y_array = np.arange(llcrnry,urcrnry, lat_resol)

        x_array = np.asarray(x_array, dtype=np.float32)
        y_array = np.asarray(y_array, dtype=np.float32)
        z_array = np.asarray(z_array, dtype=np.float32)

        # Prepare projection once
        need_proj = (density_proj != pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"))
        if need_proj:
            source_proj = Proj("+proj=longlat +datum=WGS84 +no_defs")
            transformer = Transformer.from_proj(source_proj, density_proj)

        # Allocate outputs
        Nspecies = self.nspecies
        Tx = n_timef
        Zx = len(z_array) - 1
        Xx = len(x_array) - 1
        Yx = len(y_array) - 1

        H = np.zeros((Tx, Nspecies, Zx, Xx, Yx), dtype=np.float32)
        H_count = None
        if elements_density is True:
            logger.info('Calculating element density')
            H_count = np.zeros_like(H, dtype=np.uint32)

        # Chunk over time
        for t0i in range(0, n_timef, time_chunk_size):
            t1i = min(t0i + time_chunk_size, n_timef)
            blk_T = t1i - t0i
            logger.debug(f"Computing time_chunk {t0i} - {t1i}")

            # views (no copies) for this time block
            lon_blk    = lon_2d[t0i:t1i, :]           # (blk_T, n_elem)
            lat_blk    = lat_2d[t0i:t1i, :]
            z_blk      = z_2d[t0i:t1i, :]
            specie_blk = specie_2d[t0i:t1i, :]
            keep_blk   = keep_mask[t0i:t1i, :]

            w_blk = None
            if weight is not None:
                w_blk = weight_2d[t0i:t1i, :]

            # prefilter NaNs early
            finite_blk = np.isfinite(lon_blk) #& np.isfinite(lat_blk) & np.isfinite(z_blk)
            keep_blk &= finite_blk
            del finite_blk

            # project the entire chunk once where needed
            if need_proj:
                # Flatten → project → reshape back to (blk_T, n_elem)
                lonp, latp = transformer.transform(lon_blk.reshape(-1), lat_blk.reshape(-1))
                lon_blk_p = lonp.reshape(blk_T, n_elem)
                lat_blk_p = latp.reshape(blk_T, n_elem)
                del lonp, latp
            else:
                lon_blk_p = lon_blk
                lat_blk_p = lat_blk

            # inner loops: small discrete sets (species) and timesteps in the block
            for sp in range(Nspecies):
                # time-slice loop inside the block (≤ 50 iterations)
                for t_rel in range(blk_T):
                    ti = t0i + t_rel  # absolute time index in H

                    # select surviving samples for this (time, species)
                    mask = keep_blk[t_rel, :] & (specie_blk[t_rel, :] == sp)
                    if not mask.any():
                        continue

                    xs = lon_blk_p[t_rel, mask]
                    ys = lat_blk_p[t_rel, mask]
                    zs = z_blk[t_rel, mask]
                    ws = None if w_blk is None else w_blk[t_rel, mask]

                    # histogram over (x, y, z) only
                    if ws is None:
                        H_xyz, _ = np.histogramdd((xs, ys, zs), bins=(x_array, y_array, z_array))
                        # histogramdd returns (X, Y, Z) → transpose to (Z, X, Y)
                        H[ti, sp, :, :, :] += H_xyz.transpose(2, 0, 1)

                        if H_count is not None:
                            H_count[ti, sp, :, :, :] += H_xyz.transpose(2, 0, 1).astype(H_count.dtype, copy=False)

                    else:
                        # weighted sum in H
                        H_xyz_w, _ = np.histogramdd((xs, ys, zs), bins=(x_array, y_array, z_array), weights=ws)
                        H[ti, sp, :, :, :] += H_xyz_w.transpose(2, 0, 1)

                        if H_count is not None:
                            # counts (unweighted pass) for the same slice
                            H_xyz_c, _ = np.histogramdd((xs, ys, zs), bins=(x_array, y_array, z_array))
                            H_count[ti, sp, :, :, :] += H_xyz_c.transpose(2, 0, 1).astype(H_count.dtype, copy=False)

        # Grid lon/lat arrays for output
        if density_proj is not None:
            Y, X = np.meshgrid(y_array, x_array)
            lon_array, lat_array = density_proj(X, Y, inverse=True)

        return H, lon_array, lat_array, H_count


    def get_pixel_mean_depth(self,lons,lats,density_proj_str,lat_resol,lon_resol):
        from scipy import interpolate
        import gc
        # Ocean model depth and lat/lon
        h_grd = self.conc_topo
        h_grd[np.isnan(h_grd)] = 0.
        nx = h_grd.shape[0]
        ny = h_grd.shape[1]

        lat_grd = self.conc_lat[:nx,:ny]
        lon_grd = self.conc_lon[:nx,:ny]

        for attr in ('conc_lon', 'conc_lat', 'conc_topo'):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()

        # Interpolate topography to new grid
        h = interpolate.griddata((lon_grd.flatten(),lat_grd.flatten()), h_grd.flatten(), (lons, lats), method='linear')



        if density_proj_str != ('+proj=moll +ellps=WGS84 +lon_0=0.0'):
        # Calculate the area of each grid cell in square meters (m²)
            Radius = 6.371e6  # Earth's radius in meters
            # Convert degrees to radians
            lat_resol_rad = np.radians(lat_resol)
            lon_resol_rad = np.radians(lon_resol)
            # Convert latitude centers to radians
            lat_array_rad = np.radians(lats)
            # Compute lat edges
            lat1 = lat_array_rad - (lat_resol_rad / 2)  # Lower latitude boundary
            lat2 = lat_array_rad + (lat_resol_rad / 2)  # Upper latitude boundary
            # Calculate area using the spherical formula
            area = (Radius**2) * lon_resol_rad * (np.sin(lat2) - np.sin(lat1))
            return h, area
        else:
            area = None
            return h, area


    def horizontal_smooth(self, a, n=0):
        if n==0:
            num_coarse=a
            return num_coarse

        # if np.isnan(a).any() or np.isinf(a).any():
        #     a = np.nan_to_num(a, nan=0, posinf=0, neginf=0)

        xdm=a.shape[1]
        ydm=a.shape[0]
        #msk = self.conc_mask
        b=np.zeros([ydm+2*n,xdm+2*n],dtype=int)
        b[n:-n,n:-n]=a

        num_coarse = np.zeros([ydm,xdm],dtype=float)
        smo_tmp1=np.zeros([ydm,xdm])
        #smo_msk1=np.zeros([ydm-2*n,xdm-2*n],dtype=float)
        nlayers = 0
        for ism in np.arange(-n,n+1):
            for jsm in np.arange(-n,n+1):
                smo_tmp = b[n+jsm:ydm+n+jsm, n+ism:xdm+n+ism]
                smo_tmp1+=smo_tmp
                # Must preferrably take care of land points
                # smo_msk = msk[n+jsm:ydm-n+jsm, n+ism:xdm-n+ism]
                # smo_msk1+=smo_msk
                nlayers+=1

        if n>0:
            # num_coarse[n:-n,n:-n] = smo_tmp1 / smo_msk1
            num_coarse[:,:] = smo_tmp1 / nlayers
        else:
            num_coarse = smo_tmp1
        # num_coarse = num_coarse*msk

        return num_coarse

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

    @staticmethod
    def _get_number_of_elements(
            g_mode,
            mass_element_ug=None,
            data_point=None,
            n_elements=None):

        if g_mode == "mass" and mass_element_ug is not None and data_point is not None:
            return int(np.ceil(np.array(data_point / mass_element_ug)))
        elif g_mode == "fixed" and n_elements is not None and n_elements > 0.:
            return n_elements
        else:
            raise ValueError("Incorrect combination of mode and input - undefined inputs")


    @staticmethod
    def _get_z(mode, number, NETCDF_data_dim_names, depth_seed=None, sed_mix_depth=None,
               depth_min = None, depth_max = None):
        if mode == "water_conc" and depth_seed is not None:
            if "depth" in NETCDF_data_dim_names:
                if depth_min is not None and depth_max is not None:
                    return -1 * np.random.uniform(depth_min, depth_max, number)
                else:
                    raise ValueError("depth_min or depth_max is None when depth dimention of NETCDF_data is specified")
            else:
                return -1 * np.random.uniform(0.0001, depth_seed - 0.0001, number)
        elif mode == "sed_conc" and depth_seed is not None and sed_mix_depth is not None:
            return  -1 * np.random.uniform(depth_seed + 0.0001, depth_seed + sed_mix_depth - 0.0001, number)
        elif mode == "emission":
            return -1 * np.random.uniform(0.0001, 1 - 0.0001, number)
        else:
            raise ValueError("Incorrect mode or depth")

    def seed_from_NETCDF(
            self,
            NETCDF_data,
            Bathimetry_data,
            Bathimetry_seed_data,
            mode='water_conc',
            lon_resol=None,
            lat_resol=None,
            lowerbound=0,
            higherbound=np.inf,
            radius=50,
            mass_element_ug=100e3,
            number_of_elements=None,
            origin_marker=1,
            gen_mode="mass",
            last_depth_until_bathimetry = True
    ):
        """Seed elements based on a dataarray with water/sediment concentration or direct emissions to water

            Arguments:
                NETCDF_data:          dataarray with concentration or emission data, with coordinates
                    * latitude        (latitude) float32
                    * longitude       (longitude) float32
                    * time            (time) datetime64[ns]
                Bathimetry_data:      dataarray with bathimetry data, MUST have the same grid of NETCDF_data, no time dimension, and positive values
                    * latitude        (latitude) float32
                    * longitude       (longitude) float32
                Bathimetry_seed_data: dataarray with bathimetry data, MUST be the same used for running the simulation, no time dimension, and positive values
                    * latitude        (latitude) float32
                    * longitude       (longitude) float32
                mode:                 "water_conc" (seed from concentration in water column, in ug/L),
                                      "sed_conc" (seed from sediment concentration, in ug/kg d.w.),
                                      "emission" (seed from direct discharge to water, in kg)
                radius:               float32, unit: meters, elements will be created in a circular area around coordinates
                lowerbound:           float32 elements with lower values are discarded
                higherbound:          float32, elements with higher values are discarded
                number_of_elements:   int, number of elements created for each vertical layer at each gridpoint
                mass_element_ug:      float32, maximum mass of elements if number_of_elements is not specificed
                lon_resol:            float32, longitude resolution of the NETCDF dataset
                lat_resol:            float32, latitude resolution of the NETCDF dataset
                gen_mode:             string, "mass" (elements generated from mass), "fixed" (fixed number of elements for each data point)
                origin_marker:        int, or string "single", assign a marker to seeded elements. If "single" a different origin_marker will be assigned to each datapoint
                last_depth_until_bathimetry: boolean, when depth is specified in NETCDF_data using "water_conc" mode
                                            the water column below the highest depth value is considered the same as the last
                                            available layer (True) or is consedered without chemical (False)

            """
        import opendrift

        # mass_element_ug=1e3     # 1e3 - 1 element is 1mg chemical
        # mass_element_ug=100e3   # 100e3 - 1 element is 100mg chemical
        # mass_element_ug=1e6     # 1e6 - 1 element is 1g chemical
        # mass_element_ug=1e9     # 1e9 - 1 element is 1kg chemical
        if mode not in ['water_conc', 'sed_conc', 'emission']:
            raise ValueError(f"Invalid mode: '{mode}', only 'water_conc', 'sed_conc', and 'emission' are permitted")

        sel = np.where((NETCDF_data > lowerbound) & (NETCDF_data < higherbound))
        time_check = (NETCDF_data.time).size
        NETCDF_data_dim_names = list(NETCDF_data.dims)

        if"latitude" in NETCDF_data_dim_names:
            la_name_index = NETCDF_data_dim_names.index("latitude")
        if"longitude" in NETCDF_data_dim_names:
            lo_name_index = NETCDF_data_dim_names.index("longitude")
        if "time" in NETCDF_data_dim_names:
            time_name_index = NETCDF_data_dim_names.index("time")
        elif time_check > 1:
            raise ValueError("Dimention [time] is not present in NETCDF_data_dim_names")
        else:
            pass

        depth_min = None
        depth_max = None

        if "depth" in NETCDF_data.dims:
            depth_name_index = NETCDF_data_dim_names.index("depth")
            all_depth_values = np.sort(np.absolute(np.unique(np.array(NETCDF_data.depth)))) # Change depth to positive values

        if (time_check) == 1:
        # fix for different encoding of single time step emissions
            try:
                t = np.datetime64(str(np.array(NETCDF_data.time.data)))
                t = np.array(t, dtype='datetime64[s]') # time truncaded to [s] to avoid datetime.utcfromtimestamp only integers error
            except:
                t = np.datetime64(str(np.array(NETCDF_data.time[0])))
                t = np.array(t, dtype='datetime64[s]')

        elif time_check > 1:
            t = NETCDF_data.time[sel[time_name_index]].data
            t = np.array(t, dtype='datetime64[s]')

        if "depth" in NETCDF_data.dims:
            depth = np.absolute(NETCDF_data.depth[sel[depth_name_index]].data) # Change depth to positive values to calculate pixel volume

        if"latitude" in NETCDF_data_dim_names:
            la = NETCDF_data.latitude[sel[la_name_index]].data
        else:
            la = np.array(NETCDF_data.latitude)
        if"longitude" in NETCDF_data_dim_names:
            lo = NETCDF_data.longitude[sel[lo_name_index]].data
        else:
            lo = np.array(NETCDF_data.longitude)

        if (lon_resol is None or lat_resol is None):
            raise ValueError("lat/lon_resol must be specified")

        lon_array = lo + lon_resol / 2  # find center of pixel for volume of water / sediments
        lat_array = la + lat_resol / 2  # find center of pixel for volume of water / sediments

        # Check bathimetry for inconsistent data
        if mode != 'emission':
            Check_bathimetry = []
            for i in range(0, max(t.size, lo.size, la.size)):
                Bathimetry_seed = np.array([(Bathimetry_seed_data.sel(latitude=lat_array[i],longitude=lon_array[i],method='nearest'))]) # m
                if np.isnan(Bathimetry_seed) or Bathimetry_seed <=0:
                    Check_bathimetry.append(i)

            if len(Check_bathimetry) > 0:
                # Remove datapoints with inconsistent bathimetry for seeding

                def remove_positions(arrays, positions):
                    '''
                    Remove positions specified in Check_bathimetry from each array of sel/la/lo/depth
                    '''
                    if len(arrays) == 1:
                        return (np.delete(arrays[0], positions))
                    else:
                        return tuple(np.delete(array, positions) for array in arrays)

                sel = remove_positions(sel, Check_bathimetry)
                la = np.array(remove_positions([la], Check_bathimetry))
                lo = np.array(remove_positions([lo], Check_bathimetry))
                lat_array = np.array(remove_positions([lat_array], Check_bathimetry))
                lon_array = np.array(remove_positions([lon_array], Check_bathimetry))
                if "depth" in NETCDF_data.dims:
                    depth = np.array(remove_positions([depth], Check_bathimetry))
                if t.size == 1:
                    pass
                elif t.size > 1:
                    t = np.array(remove_positions([t], Check_bathimetry))
                logger.info(f"{len(Check_bathimetry)} datapoints removed due to inconsistent bathimetry")
                del(Check_bathimetry)
            else:
                del(Check_bathimetry)

        data = np.array(NETCDF_data.data)
        print(f"Seeding {str(np.sum((~np.isnan(data)) & (data > 0)))} datapoints")
        list_index_print = self._print_progress_list(max(t.size, lo.size, la.size))

        sed_mixing_depth = np.array(self.get_config('chemical:sediment:mixing_depth')) # m

        if mode == 'sed_conc':
            # Compute mass of dry sediment in each pixel grid cell
            sed_mixing_depth = np.array(self.get_config('chemical:sediment:mixing_depth')) # m
            sed_density      = np.array(self.get_config('chemical:sediment:density')) # density of sediment particles, in kg/m3 d.w.
            sed_porosity     = np.array(self.get_config('chemical:sediment:porosity') ) # fraction of sediment volume made of water, adimentional (m3/m3)
            if self.mode != opendrift.models.basemodel.Mode.Config:
                self.mode = opendrift.models.basemodel.Mode.Config
            self.init_species()
            self.init_transfer_rates()


        lat_grid_m = np.array([6.371e6 * lat_resol * (2 * np.pi) / 360])

        if mode == 'emission':
            Bathimetry_seed = None

        if origin_marker == "single":
            origin_marker_np = np.arange(0, max(t.size, lo.size, la.size))

        for i in range(0, max(t.size, lo.size, la.size)):
            if i == 0:
                time_start_0 = datetime.now()
            if i == 1:
                time_start_1 = datetime.now()
                estimated_time = (time_start_1 - time_start_0)* (max(t.size, lo.size, la.size))
                print(f"Estimated time (h:min:s): {estimated_time}")
            if i in list_index_print:
            #     print(f"Seeding elem {i} out of {max(t.size, lo.size, la.size)}")
                print(".", end="")
            lon_grid_m = None
            depth_min = None
            depth_max = None

            if mode != 'emission':
                lon_grid_m =  np.array([(6.371e6 * (np.cos(2 * (np.pi) * la[i] / 360)) * lon_resol * (2 * np.pi) / 360)])  # 6.371e6: radius of Earth in m
                if "depth" in NETCDF_data.dims:
                    # depth start from 0 at surface layer, with positive values at higher depths
                    depth_datapoint = np.absolute(depth[i])
                    # depth_datapoint_index = list(all_depth_values).index(depth_datapoint)
                    Bathimetry_datapoint = np.array([(Bathimetry_data.sel(latitude=la[i],longitude=lo[i],method='nearest'))]) # m
                    # depth of seeding must be the same as the one considered for resuspention process
                    Bathimetry_seed = np.array([(Bathimetry_seed_data.sel(latitude=lat_array[i],longitude=lon_array[i],method='nearest'))]) # m
                    # array of depth values available for lat/lon position
                    depth_datapoint_np = np.sort(np.absolute(np.array(NETCDF_data.sel(latitude=la[i],longitude=lo[i],method='nearest')['depth']))) # depth is changed to positive values to calculate pixel volume
                    depth_datapoint_np_index = list(depth_datapoint_np).index(depth_datapoint)

                    if 0 in all_depth_values:
                        # use of ChemicalDrift output, where depth represents top of vertical layer
                        depth_min = depth_datapoint
                        # check if depth value of datapiont is the last avalable for lat/lon
                        if depth_datapoint == depth_datapoint_np[-1]:
                            depth_max = Bathimetry_datapoint
                        else:
                            depth_max = min(depth_datapoint_np[depth_datapoint_np_index + 1], Bathimetry_datapoint)

                        depth_layer_high = depth_max - depth_min
                        if depth_layer_high < 0:
                            Error_depth = "datapoint at lon: " + str(lo[i])[0:8] +\
                            " & lat: " + str(la[i])[0:8]+ " & depth: " + str(depth_datapoint)[0:8]+\
                            " is at depth higher than bathimetry (" + str(Bathimetry_datapoint)[1:8] + ")"
                            raise ValueError(Error_depth)
                    else:
                        # use of standard convention where depth represents bottom of vertical layer
                        if depth_datapoint_np_index > 0:
                            # datapoint is below the surface layer
                            depth_min = (depth_datapoint_np[depth_datapoint_np_index - 1])
                        else:
                            # datapoint is within the surface layer
                            depth_min = 0

                        # check if datapoint is the last depth available for lat/lon position
                        if depth_datapoint_np_index == len(depth_datapoint_np)-1 and last_depth_until_bathimetry is True:
                            depth_max = max(depth_datapoint, Bathimetry_datapoint)
                        else:
                            depth_max = depth_datapoint

                else:
                    depth_layer_high = np.array([(Bathimetry_data.sel(latitude=la[i],longitude=lo[i],method='nearest'))]) # m
                    # depth of seeding must be the same as the one considered for resuspention process
                    Bathimetry_seed = np.array([(Bathimetry_seed_data.sel(latitude=lat_array[i],longitude=lon_array[i],method='nearest'))]) # m

            if mode == 'water_conc':
                pixel_volume = depth_layer_high * lon_grid_m * lat_grid_m
                # concentration is ug/L, volume is m: m3 * 1e3 = L
                mass_ug = (data[sel][i] * (pixel_volume * 1e3))

            elif mode == 'sed_conc':
                # sed_conc_ug_kg is ug/kg d.w. (dry weight)
                pixel_volume =  sed_mixing_depth * (lon_grid_m * lat_grid_m) # m3
                pixel_sed_mass = (pixel_volume)*(1-sed_porosity)*sed_density # kg
                mass_ug = data[sel][i]*pixel_sed_mass

            elif mode == 'emission':
                mass_ug = data[sel][i]*1e9 # emissions is kg, 1 kg = 1e9 ug
            else:
                raise ValueError("Incorrect mode")

            if mass_ug == 0:
                continue

            number = self._get_number_of_elements(
                g_mode=gen_mode,
                mass_element_ug=mass_element_ug,
                data_point=mass_ug,
                n_elements=number_of_elements)

            if t.size == 1:
                time = datetime.utcfromtimestamp(int(
                    (np.array(t - np.datetime64('1970-01-01T00:00:00'))) / np.timedelta64(1, 's')))
            elif t.size > 1:
                time = datetime.utcfromtimestamp(int(
                    (np.array(t[i] - np.datetime64('1970-01-01T00:00:00'))) / np.timedelta64(1, 's')))

            # specie to be added to seed parameters for sediments and water
            if mode == 'sed_conc':
                specie_elements = 3 # 'num_srev' # Name of specie for sediment elements
                moving_emement = False # sediment particles will not move
            else:
                specie_elements = 0 # 'num_lmm' # Name of specie for dissolved elements
                moving_emement = True # dissolved particles will move

            if gen_mode == 'fixed':
                mass_element_seed_ug = mass_ug / number
            elif gen_mode == 'mass':
                mass_element_seed_ug = mass_element_ug

            if mass_element_seed_ug > 0:
                # print(i)
                z = self._get_z(mode = mode,
                                number = number,
                                NETCDF_data_dim_names = NETCDF_data_dim_names,
                                depth_min = depth_min,
                                depth_max = depth_max,
                                depth_seed = Bathimetry_seed, # depth must be the same as the one considered for resuspention process
                                sed_mix_depth = sed_mixing_depth)

                for k in range(len(z)):

                    if"latitude" in NETCDF_data_dim_names:
                        elem_lat = lat_array[i]
                    else:
                        # specify lat if all elements are seeded in the same place
                        elem_lat = lat_array
                    if"longitude" in NETCDF_data_dim_names:
                        elem_lon = lon_array[i]
                    else:
                        # specify lon if all elements are seeded in the same place
                        elem_lon = lon_array

                    if origin_marker == "single":
                        origin_marker_seed = origin_marker_np[i]
                    else:
                        origin_marker_seed = origin_marker

                    self.seed_elements(
                        lon=elem_lon,
                        lat=elem_lat,
                        radius=radius,
                        number=1,
                        time=time,
                        mass=mass_element_seed_ug,
                        mass_degraded=0,
                        mass_volatilized=0,
                        specie = specie_elements,
                        moving = moving_emement,
                        z=z[k],
                        origin_marker=origin_marker_seed)

                    if gen_mode != "fixed":
                        mass_residual = (mass_ug) - (number * mass_element_seed_ug)

                        if mass_residual > 0:
                            z = self._get_z(mode = mode,
                                            number = 1,
                                            NETCDF_data_dim_names = NETCDF_data_dim_names,
                                            depth_min = depth_min,
                                            depth_max = depth_max,
                                            depth_seed = Bathimetry_seed, # depth must be the same as the one considered for resuspention process
                                            sed_mix_depth = sed_mixing_depth)

                            self.seed_elements(
                                lon=lon_array[i],
                                lat=lat_array[i],
                                radius=radius,
                                number=1,
                                time=time,
                                mass=mass_residual,
                                mass_degraded=0,
                                mass_volatilized=0,
                                specie = specie_elements,
                                moving = moving_emement,
                                z=z,
                                origin_marker=origin_marker_seed)

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
        time_name:    string, name of time dimention (tiime or time_avg)

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
                raise ValueError("Unknown spatial lat coordinates")

        DataArray['latitude'] = DataArray['latitude'].assign_attrs(
                                standard_name="latitude",
                                long_name="latitude",
                                units="degrees_north",
                                axis="Y",
                            )
        DataArray['longitude'] = DataArray['longitude'].assign_attrs(
                                standard_name="longitude",
                                long_name="longitude",
                                units="degrees_east",
                                axis="X",
                            )
        return(DataArray)


    def correct_conc_coordinates(self, DC_Conc_array, lon_coord, lat_coord, time_coord, time_name,
                                 shift_time=False):
        """
        Add longitude, latitude, and time coordinates to water and sediments concentration xarray DataArray

        DC_Conc_array:     xarray DataArray for water or sediment concetration from sum of "species"
                           from "write_netcdf_chemical_density_map" output
        lon_coord:         np array of float64, with longitude of "write_netcdf_chemical_density_map" output
        lat_coord:         np array of float64, with latitude of "write_netcdf_chemical_density_map" output
        time_coord:        np array of datetime64[ns] with avg_time of "write_netcdf_chemical_density_map" output
        shift_time:        boolean, if True shifts back time of 1 timestep so that the timestamp corresponds to
                           the beginning of the first simulation timestep, not to the next one
        """

        if (("longitude" not in DC_Conc_array.dims) or ("latitude" not in DC_Conc_array.dims)):
            DC_Conc_array = self._rename_dimensions(DC_Conc_array)
            if all(x is not None for x in [lon_coord, lat_coord]):
                DC_Conc_array['latitude'] = ('latitude', lat_coord)
                DC_Conc_array['longitude'] = ('longitude', lon_coord)
            else:
                raise ValueError('lat/lon_coord not in DS')

        # Add latitude and longitude to the concentration dataset
        # Add attributes to latitude and longitude so that "remapcon" function from cdo can interpolate results
        DC_Conc_array = self._rename_dimensions(DC_Conc_array)

        if time_name is not None:
            if (time_name in DC_Conc_array.dims) and shift_time == True:
                # Shifts back time 1 timestep so that the timestamp corresponds to the beginning of the first simulation timestep, not the next one
                time_correction = time_coord[1] - time_coord[0]
                time_corrected = np.array(time_coord - time_correction)
                DC_Conc_array[time_name] = (time_name, time_corrected)
                print(f"Shifted {time_name} back of one timestep")

        if ("avg_time" in DC_Conc_array.dims):
            DC_Conc_array_corrected=DC_Conc_array.rename({'avg_time': 'time'})
        else:
            DC_Conc_array_corrected=DC_Conc_array

        return DC_Conc_array_corrected

    def calculate_water_sediment_conc(self,
                                      File_Path,
                                      File_Name,
                                      File_Path_out,
                                      Chemical_name,
                                      Origin_marker_name,
                                      File_Name_out = None,
                                      variables = None,
                                      Transfer_setup = "organics",
                                      Concentration_file = None,
                                      Shift_time = False,
                                      Conc_SPM = True,
                                      Sim_description=None):
        """
        Sum dissolved, DOC, and SPM concentration arrays to obtain total water concentration and save the resulting xarray as netCDF file
        Save sediment concentration DataArray as netDCF file
        Results can be used as inputs by "seed_from_NETCDF" function

        Concentration_file:    "write_netcdf_chemical_density_map" output if already loaded (original or after regrid_conc)
        File_Path:             string, path of "write_netcdf_chemical_density_map" output
        File_Name:             string, name of "write_netcdf_chemical_density_map" output
        File_Name_out:         string, suffix of wat/sed output files
        File_Path_out:         string, path where created concentration files will be saved, must end with "/"
        Chemical_name:         string, name of modelled chemical
        Transfer_setup:        string, transfer_setup used for the simulation, "organics" or "metals"
        Origin_marker_name:    string, name of source indicated by "origin_marker" parameter
        variables:             list, list of variables' name to be considered
        Shift_time:            boolean, if True shifts back time of 1 timestep so that the timestamp corresponds to
                               the beginning of the first simulation timestep, not to the next one
        Sim_description:       string, descrition of simulation to be included in netcdf attributes
        """
        from datetime import datetime
        import xarray as xr

        if ((Concentration_file is None) and (File_Path and File_Name is not None)):
            print("Loading Concentration_file from File_Path")
            DS = xr.open_dataset(File_Path + File_Name)
        elif Concentration_file is not None:
            DS = Concentration_file
        else:
            raise ValueError("Incorrect file or file/path not specified")

        if not any([var in DS.data_vars for var in  ['concentration', 'concentration_avg',
                       'concentration_smooth', 'concentration_smooth_avg',
                       'density', 'density_avg']]):
            raise ValueError("No valid variables")


        # Sum DataArray for specie 0, 1, and 2 (dissolved, DOC, and SPM) to obtain total water concentration
        print("Running sum of water concentration", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

        if "time" in DS.dims:
            time_name = "time"
        else:
            time_name = "avg_time"

        variable_ls = ['concentration', 'concentration_avg',
                       'concentration_smooth', 'concentration_smooth_avg',
                       'density', 'density_avg']

        if variables is not None:
            variable_ls = variables

        time_array = np.array(DS[time_name])

        if ('topo' in DS.data_vars):
            topo = DS.topo

        sum_vars_wat_dict = {}
        sum_vars_sed_dict = {}
        first_var = True
        for variable in variable_ls:
            # variable = variable_ls[1]
            if (variable in DS.data_vars) and (variable != 'topo'):
                print(variable)
                var_wat_name = variable + "_wat"
                var_sed_name = variable + "_sed"

                TOT_Conc = DS[variable]
                if Transfer_setup == "organics":
                    Dissolved_conc = TOT_Conc.sel(specie = 0)
                    SPM_conc = TOT_Conc.sel(specie = 2)
                    if 1 in DS.specie:
                        DOC_conc = TOT_Conc.sel(specie = 1)
                        # print("DOC was considered for partitioning of chemical")
                        if Conc_SPM == True:
                            DA_Conc_array_wat = Dissolved_conc + SPM_conc + DOC_conc
                        else:
                            DA_Conc_array_wat = Dissolved_conc + DOC_conc
                            print("SPM was not considered for water concentration")
                    else:
                        if Conc_SPM == True:
                            DA_Conc_array_wat = Dissolved_conc + SPM_conc
                        else:
                            DA_Conc_array_wat = Dissolved_conc
                            print("SPM was not considered for water concentration")
                elif Transfer_setup == "metals":
                    Dissolved_conc = TOT_Conc.sel(specie = 0)
                    SPM_conc = TOT_Conc.sel(specie = 1)
                    SPM_conc_sr = TOT_Conc.sel(specie = 2)
                    if Conc_SPM == True:
                        DA_Conc_array_wat = Dissolved_conc + SPM_conc + SPM_conc_sr
                    else:
                        DA_Conc_array_wat = Dissolved_conc
                        print("SPM was not considered for water concentration")

                print("Running sediment concentration", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

                DA_Conc_array_sed = DS[variable].sel(specie = 3)

                if "depth" in DA_Conc_array_sed.dims:
                    print("depth included in DA_Conc_array_sed")
                    # Mask to keep landmask when saving sediment concentration
                    mask = np.isnan(DS[variable])
                    # Sediments not buried are elements with specie = 3
                    DA_Conc_array_sed = DS[variable][:,3,:,:,:].sum(dim='depth')
                    # Add mask to DA_Conc_array_sed
                    DA_Conc_array_sed = xr.where(mask[:,0,-1,:,:],np.nan, DA_Conc_array_sed)

                print("Changing coordinates", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

                if "latitude" not in DS[variable].dims:
                    if "lat" in DS.data_vars:
                        lat = np.array(DS.lat[:,1])
                        latitude = np.array(DS.lat[:,1])
                        print("lat data_var used")
                    else:
                        raise ValueError("Latitude information not present in DS")
                else:
                    latitude = np.array(DS[variable].latitude)
                    lat = None

                if "longitude" not in DS[variable].dims:
                    if "lon" in DS.data_vars:
                        lon = np.array(DS.lon[1,:])
                        longitude = np.array(DS.lon[1,:])
                        print("lon data_var used")
                    else:
                        raise ValueError("Incorrect dimention lon/x")
                else:
                    longitude = np.array(DS[variable].longitude)
                    lon = None

                DA_Conc_array_wat = self.correct_conc_coordinates(DC_Conc_array = DA_Conc_array_wat,
                                                              lon_coord = lon,
                                                              lat_coord = lat,
                                                              time_coord = time_array,
                                                              shift_time = Shift_time,
                                                              time_name = time_name)

                DA_Conc_array_sed = self.correct_conc_coordinates(DC_Conc_array = DA_Conc_array_sed,
                                                              lon_coord = lon,
                                                              lat_coord = lat,
                                                              time_coord = time_array,
                                                              shift_time = Shift_time,
                                                              time_name = time_name)
                if ('topo' in DS.data_vars):
                    if first_var == True:
                        DC_topo = self.correct_conc_coordinates(DC_Conc_array = topo,
                                                                      lon_coord = lon,
                                                                      lat_coord = lat,
                                                                      time_coord = time_array,
                                                                      shift_time = Shift_time,
                                                                      time_name = time_name)
                        first_var = False
                        sum_vars_wat_dict['topo'] = DC_topo
                        sum_vars_sed_dict['topo'] = DC_topo


                DA_Conc_array_wat.name = var_wat_name
                if hasattr(DS[variable], 'sim_description'):
                    DA_Conc_array_wat.attrs['sim_description'] = (DS[variable].sim_description)
                elif Sim_description is not None:
                    DA_Conc_array_wat.attrs['sim_description'] = str(Sim_description)

                if hasattr(DS[variable], 'long_name'):
                    DA_Conc_array_wat.attrs['long_name'] = (DS[variable].long_name).split('specie')[0].strip() + " in water"
                else:
                    DA_Conc_array_wat.attrs['long_name'] = (Chemical_name or "") + f" {variable} in water (assumed from mass of elements)"

                if hasattr(DS[variable], 'units'):
                    if "concentration" in variable:
                        DA_Conc_array_wat.attrs['units'] = DS[variable].units[0:5]
                    else:
                        DA_Conc_array_wat.attrs['units'] = '1'
                else:
                    DA_Conc_array_wat.attrs['units'] = 'ug/m3 (assumed default)'

                if "projection" in DS.data_vars:
                    DA_Conc_array_wat.attrs['projection'] = str(DS.projection.proj4)

                if hasattr(DS[variable], 'grid_mapping'):
                    DA_Conc_array_wat.attrs['grid_mapping'] = DS[variable].grid_mapping

                DA_Conc_array_wat.attrs['lon_resol'] = str(np.around(abs(longitude[0]-longitude[1]), decimals = 8)) + " degrees E"
                DA_Conc_array_wat.attrs['lat_resol'] = str(np.around(abs(latitude[0]-latitude[1]), decimals = 8)) + " degrees N"

                sum_vars_wat_dict[var_wat_name] = DA_Conc_array_wat

                DA_Conc_array_sed.name = var_sed_name
                if hasattr(DS[variable], 'sim_description'):
                    DA_Conc_array_sed.attrs['sim_description'] = (DS[variable].sim_description)
                elif Sim_description is not None:
                    DA_Conc_array_sed.attrs['sim_description'] = str(Sim_description)

                if hasattr(DS[variable], 'long_name'):
                    DA_Conc_array_sed.attrs['long_name'] = (DS[variable].long_name).split('specie')[0].strip() + " in sediments"
                else:
                    DA_Conc_array_sed.attrs['long_name'] = ((Chemical_name or "") + f" {variable} in sediments (assumed from mass of elements)")

                if hasattr(DS[variable], 'units'):
                    if "concentration" in variable:
                        DA_Conc_array_sed.attrs['units'] = DS[variable].units[11:20]
                    else:
                        DA_Conc_array_sed.attrs['units'] = '1'
                else:
                    DA_Conc_array_sed.attrs['units'] = 'ug/Kg d.w (assumed default)'

                if "projection" in DS.data_vars:
                    DA_Conc_array_sed.attrs['projection'] = str(DS.projection.proj4)

                if hasattr(DS[variable], 'grid_mapping'):
                    DA_Conc_array_sed.attrs['grid_mapping'] = DS[variable].grid_mapping

                DA_Conc_array_sed.attrs['lon_resol'] = str(np.around(abs(longitude[0]-longitude[1]), decimals = 8)) + " degrees E"
                DA_Conc_array_sed.attrs['lat_resol'] = str(np.around(abs(latitude[0]-latitude[1]), decimals = 8)) + " degrees N"
                sum_vars_sed_dict[var_sed_name] = DA_Conc_array_sed

        DS_wat_fin = xr.Dataset(sum_vars_wat_dict)
        DS_sed_fin = xr.Dataset(sum_vars_sed_dict)

        if File_Name_out is not None:
            wat_file = File_Path_out + "wat_" + File_Name_out
            sed_file = File_Path_out + "sed_" + File_Name_out
            if not wat_file.endswith(".nc"):
                wat_file = wat_file + ".nc"
            if not sed_file.endswith(".nc"):
                sed_file = sed_file + ".nc"
        else:
            wat_file = File_Path_out + "water_conc_" + (Chemical_name or "") + "_" + (Origin_marker_name or "") + ".nc"
            sed_file = File_Path_out + "sediments_conc_" + (Chemical_name or "") + "_" + (Origin_marker_name or "")+ ".nc"

        print("Saving water concentration file", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        DS_wat_fin.to_netcdf(wat_file)
        print("Saving sediment concentration file", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        DS_sed_fin.to_netcdf(sed_file)

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
                    # Change "year" dimention to "time", at the January, 1st
                    Conc_DataArray['year'] = pd.to_datetime(np.char.add(np.array(Conc_DataArray['year']).astype(str), '-01-01'))
                    Conc_DataArray = Conc_DataArray.rename({'year': 'time'})
                    Conc_DataArray = Conc_DataArray.assign_coords(time=Conc_DataArray['time'])
                elif "season" in Conc_DataArray.dims and time_start is not None:
                    # Change "season" dimention to "time", at the first day of each season
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
                    Conc_DataArray_selected = Conc_DataArray.isel(time = timestep, depth = selected_depth_index)
                elif Conc_DataArray_time_size > 1 and "depth" not in Conc_DataArray.dims:
                    Conc_DataArray_selected = Conc_DataArray.isel(time = timestep)
                elif Conc_DataArray_time_size <= 1 and "depth" in Conc_DataArray.dims:
                    Conc_DataArray_selected = Conc_DataArray.isel(depth = selected_depth_index)
                elif Conc_DataArray_time_size <= 1 and "depth" not in Conc_DataArray.dims:
                    Conc_DataArray_selected = Conc_DataArray

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
    def _plot_emission_data_frequency(emissions, title, n_bins = 100, zoom_max = 100, zoom_min = 0):
        '''
        Plot distribuion of emissions dataset values, mass, and cumulative mass

        emissions:  masked array of float32 with selected data points to plot
        title:      string, title of main plot
        n_bins:     int, number of bins to group datapoints
        zoom_max:   int, % of dataset lenght where the zoomed area stops
        zoom_min:   int, % of dataset lenght where the zoomed area starts
        '''
        import matplotlib.pyplot as plt
        import numpy as np

        values,base=np.histogram(emissions,n_bins)
        cumulative = np.cumsum(values*base[0:-1])
        fig,(ax1,ax2)=plt.subplots(nrows=2, ncols=1)
        fig.tight_layout(pad=1.5)
        ax1.plot(base[:-1], 100*values*base[0:-1]/max(values*base[0:-1]), c='red')
        ax1.plot(base[:-1], 100*cumulative/cumulative[-1], c='blue')
        ax1.plot(base[:-1], 100*values/max(values), c='black')
        ax1.set_title(title)
        ax1.set_ylabel("Frequency (%)")
        ax1.legend(['Mass','Cumulative mass','Data points'])
        ax2.plot(base[zoom_min:zoom_max], 100*values[zoom_min:zoom_max]*base[zoom_min:zoom_max]/max(values*base[0:-1]), c='red')
        ax2.plot(base[zoom_min:zoom_max], 100*cumulative[zoom_min:zoom_max]/cumulative[-1], c='blue')
        ax2.plot(base[zoom_min:zoom_max], 100*values[zoom_min:zoom_max]/max(values), c='black')
        ax2.set_title("Zoom from " + str(zoom_min) +" to "+str(zoom_max)+"% of dataset")
        ax2.set_ylabel("Frequency (%)")
        ax2.set_xlabel("Value")
        plt.show()

    def summary_created_elements(self,
                                 file_folder,
                                 file_name,
                                 variable_name,
                                 emiss_factor,
                                 upper_limit,
                                 lower_limit,
                                 name_dataset,
                                 long_min, long_max,
                                 lat_min, lat_max,
                                 time_start, time_end,
                                 range_max = None,
                                 range_min = None,
                                 n_bins = 100,
                                 zoom_max=100,
                                 zoom_min=0,
                                 print_results = False
                                 ):
        '''
        Calculate the maxium number of elements in a simulation created by seed_from_NETCDF from xarray DataArray.
        Produce histographs with frequency of datapoints values within the specified limits
        ----------
        file_folder:    string, path to file, must end with /
        file_name:      string, name of file, must end with .nc
        variable_name:  string, name of xarray DataArray variable
        emiss_factor:   float32, conversion factor between data and mass expressed as ug (e.g. ug/L, ug/kg), 1e9 if DataArray is in Kg
        upper_limit:    float32, limit under which datapoints in DataArray wll be ignored by seed_from_NETCDF
        lower_limit:    float32, limit over which datapoints in DataArray wll be ignored by seed_from_NETCDF
        name_dataset:   string, name of data to be reported in the title of figures
        time_start:     datetime64[ns], start time of dataset considered
        time_end:       datetime64[ns],  end time of dataset considered
        long_min:       float32, min longitude of dataset considered
        long_max:       float32, max longitude of dataset considered
        lat_min:        float32, min latitude of dataset considered
        lat_max:        float32, max latitude of dataset considered
        range_max:      float32, max value shown in the figure on data frequency for the whole dataset
        range_min:      float32, min value shown in the figure on data frequency for the whole dataset
        n_bins:         int, number of bins used for histograms
        zoom_max:       int, % of dataset lenght where the zoomed area stops
        zoom_min:       int, % of dataset lenght where the zoomed area starts
        print_results:  boolean, select if results are printed or returned as dictionaty
        '''
        import xarray as xr
        import matplotlib.pyplot as plt
        import numpy as np

        DS = xr.open_dataset(file_folder + file_name)
        if "lat" in DS.dims:
            DS = DS.rename({'lat': 'latitude','lon': 'longitude'})
        if "x" in DS.dims:
            DS = DS.rename({'y': 'latitude','x': 'longitude'})

        DS = DS[variable_name]
        if "time" in DS.dims:
            DS = DS.where((DS.longitude > long_min) & (DS.longitude < long_max) &
                                            (DS.latitude > lat_min) & (DS.latitude < lat_max) &
                                            (DS.time >= time_start) &
                                            (DS.time <= time_end), drop=True)
        else:
            DS = DS.where((DS.longitude > long_min) & (DS.longitude < long_max) &
                                            (DS.latitude > lat_min) & (DS.latitude < lat_max), drop=True)

        DS_ma = DS.to_masked_array() # Remove 0 and NA from dataArray, then change to np.array
        emissions = DS_ma[DS_ma>0]
        selected =np.all((emissions<upper_limit,emissions>lower_limit),axis=0)
        print ("##START "+ name_dataset + " ##")

        DS_max = np.array(emissions.max())
        DS_min = np.array(emissions.min())

        emissions_sum = np.sum(emissions)
        total_mass = (emissions_sum* emiss_factor)/1e9 # (L*ug/L)/10^9 -> Kg
        selected_mass = (sum((emissions[selected])* emiss_factor))/1e9

        Num_tot = len(emissions)
        Num_selected = len(emissions[selected])

        Perc_num_selected = (Num_selected/Num_tot)*100
        Perc_mass_selected = (selected_mass/total_mass)*100

        if print_results is False:
            results_dict = {}
            results_dict["name_dataset"] = name_dataset
            results_dict["upper_limit"] = upper_limit
            results_dict["lower_limit"] = lower_limit
            results_dict["DS_max"] = DS_max
            results_dict["DS_min"] = DS_min
            results_dict["Num_tot"] = Num_tot
            results_dict["Num_selected"] = Num_selected
            results_dict["Mass_tot"] = total_mass
            results_dict["Mass_selected"] = selected_mass
            results_dict["Perc_num_selected"] = Perc_num_selected
            results_dict["Perc_mass_selected"] = Perc_mass_selected
            return results_dict
        else:
            print(f"DS_max: {DS_max}")
            print(f"DS_min: {DS_min}", "\n")
            print(f"number of data-points without limits: {len(emissions)}")
            print(f"upper limit: {upper_limit}")
            print(f"lower limit: {lower_limit}", "\n")

            print(f"number of data-points selected within the limits: {Num_selected}", "\n")
            print(f'total mass of chemical: {total_mass} kg')
            print(f'selected mass of chemical: {selected_mass} kg')
            print(f'% of total mass selected: {Perc_mass_selected} %')

            # Print number and percentage of elements over upper limit
            num_upper_lim = np.count_nonzero(emissions > upper_limit)
            print(f"n° of data-points over upper limit: {num_upper_lim}")
            print(f"% of data-points over upper limit: \
                  {(num_upper_lim/np.prod(emissions.shape))*100} %")
            mass_over_limit = (sum((emissions[emissions > upper_limit])* emiss_factor))/1e9
            print(f'mass of chemical over upper limit: {mass_over_limit} kg')
                  # e.g. (L*ug/L)/10^9 -> kg
            print(f"% of total volume or mass of the elements over upper limit:\
                  {(mass_over_limit/emissions_sum)*100} %", "\n")

            # Print number and percentage of elements under lower limit
            num_lower_lim = np.count_nonzero(emissions < lower_limit)
            print(f"n° of data-points under lower limit: {num_lower_lim}")
            print(f"% of data-points under lower limit: \
                  {(num_lower_lim/np.prod(emissions.shape))*100} %")
            mass_below_limit = (sum((emissions[emissions < lower_limit])* emiss_factor))/1e9
            print(f'mass of chemical under limit: {mass_below_limit} kg')
                  # e.g. (L*ug/L)/10^9 -> Kg
            print(f"% of total volume or mass of the elements under lower limit: \
                  {(mass_below_limit/(emissions_sum))*100} %", "\n")

            print(f"% of total volume or mass of elements under lower limit considering also upper limit \
                  {((mass_below_limit)/(np.sum(emissions[emissions < upper_limit])))*100}", "\n")

            # Plot histograms for frequency of values

            self._plot_emission_data_frequency(emissions= emissions,
                                title = "Complete dataset",
                                n_bins = n_bins,
                                zoom_max = zoom_max,
                                zoom_min = zoom_min)

            self._plot_emission_data_frequency(emissions= emissions[selected],
                                title = "Selected dataset between lower and upper limit",
                                n_bins = n_bins,
                                zoom_max = zoom_max,
                                zoom_min = zoom_min)

            if range_max is not None and range_min is not None:
                zoom_max = (range_max/DS_max)*100
                zoom_min = (range_min/DS_max)*100

                self.plot_emission_data_frequency(emissions= emissions[selected],
                                                  title = "Selected dataset between range_min and range_max",
                                                  n_bins = n_bins,
                                                  zoom_max = zoom_max,
                                                  zoom_min = zoom_min)

            plt.hist(x=emissions, bins=n_bins, range=(0,lower_limit))
            plt.title("Datapoints between 0 and lower limit for "+ name_dataset)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()
            print ("##END##")

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

    @staticmethod
    def _change_datetime_format(datetime_str, new_format):
        '''
        Change format of datetime string to a specified format

        Parameters
        ----------
        datetime_str: str, datetime string to be reformatted
        new_format:  str, specified datetime format
        '''
        possible_formats = ["%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d %H", "%Y/%m/%d",
                            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%d",
                            "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y %H", "%d-%m-%Y",
                            "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %H", "%d/%m/%Y",
                            "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M", "%m-%d-%Y %H", "%m-%d%Y",
                            "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y %H", "%m/%d/%Y"
                            ]
        for fmt in possible_formats:
            try:
                # Attempt to parse the datetime string
                dt = datetime.strptime(datetime_str, fmt)
                # If parsing is successful, format the datetime object
                formatted_datetime = dt.strftime(new_format)
                # Return the formatted datetime string
                return formatted_datetime
            except ValueError:
                pass  # Continue to the next format if parsing fails
        # Return None if parsing fails for all formats
        raise ValueError("Unknown format of date_of_timestep")



    @staticmethod
    def _select_df_column(col_name, dataframe):
        '''
        Select a column of mass_fin_df based on name
        '''
        return dataframe[dataframe.columns[dataframe.columns.str.contains(col_name)]]



    def filter_dataframe(self, df, time_unit, start_date=None, end_date=None):
        '''
        Filter dataframe based on start and end date

        df:                     Pandas dataframe, mass_fin_df
        start_date:                   pd.Datetime, start of plotted timeserie
                                      (e.g., pd.to_datetime("2018-01-01 00:00:00", format = '%Y-%m-%d %H:%M:%S'))
        end_date:                     pd.Datetime, end of plotted timeserie
        '''
        import pandas as pd

        # Check if both start_date and end_date are None
        if start_date is None and end_date is None:
            return df

        date_of_timestep_ls = []
        for element in df['date_of_timestep']:
            if isinstance(element, pd.Timestamp):
                element = element.strftime('%Y-%m-%d %H:%M:%S')

            date_of_timestep_ls.append(self._change_datetime_format(datetime_str = element,
                                        new_format = "%Y-%m-%d %H:%M:%S"))
        df['date_of_timestep'] = date_of_timestep_ls
        del date_of_timestep_ls

        date_column = pd.to_datetime(df['date_of_timestep'], format='%Y-%m-%d %H:%M:%S')

        # Filter based on start_date and/or end_date
        if start_date is not None and end_date is not None:
            filtered_df = df[( date_column >= start_date) & (date_column <= end_date)]
        elif start_date is not None:
            filtered_df = df[(date_column >= start_date)]
        else:  # end_date is not None
            filtered_df = df[(date_column <= end_date)]
        # Correct 'time'+"-["+ time_unit +"]" column to start at new start_date
        time_delta_filter = filtered_df['time'+"-["+ time_unit +"]"].iloc[0] - df['time'+"-["+ time_unit +"]"].iloc[0]
        filtered_df['time'+"-["+ time_unit +"]"] = np.array(filtered_df['time'+"-["+ time_unit +"]"]) - time_delta_filter
        return filtered_df


    def extract_summary_timeseries(self,
            file_out_path,
            sim_name,
            chemical_compound= None,
            mass_unit='g',
            time_unit='hours',
            shp_file_path = None,
            load_timeseries_from_file = False,
            timeseries_file_path = None,
            plot_figures = False,
            title_fig_tserie = None,
            title_fig_mass = None,
            title_fig_deg = None,
            subplot_width = 4,
            subplot_height = 3,
            lon_min = None,
            lon_max = None,
            lat_min = None,
            lat_max = None,
            start_date = None,
            end_date = None,
            check_mass = False):
        '''
        Extract, plot to .png, and export to a .csv file aggregated mass of all elements at each timestep of the simulation
        Plot directly data from a produced .cvs file.
        Use deactivate_north_of/south_of/_east_of/_west_of parameters to define simulation domain and avoid double counting
        advection out of the system for elements that are not deactivated when crossing the simulation border

        Parameters
        ----------
        chemical_compound:            string, name of chemical to be included into plot titles
        file_out_path:                string, file path where .csv and .png files will be saved. Must end with /
        sim_name:                     string, name of simulation that will be included in .csv and .png file names
        mass_unit:                    string, 'kg' 'g','mg','ug'
        time_unit:                    string, 'seconds', 'minutes', 'hours' , 'days'
        shp_file_path:                string, file path of shapefile. Must end with /
        load_timeseries_from_file:    boolean,select if timeseries are loaded from .csv (True) or not (False)
        timeseries_file_path:         string, file path of timeseries .csv file. Must end with /
        plot_figures:                 boolean,select if data is plotted (True) or not (False)
        title_fig_tserie:             string, title of timeseries figure
        title_fig_mass:               string, title of mass among dissolved/DOC/SPM/sed figure
        title_fig_deg:                string, title of removal mechanisms figure
        subplot_width:                float32, width (inches) of each subplot in timeseries figure
        subplot_height                float32, height (inches) of each subplot in timeseries figure
        lon_min:                      float32, min longitude of domain to consider advection out of the simulation domain
        lon_max:                      float32, max longitude of domain to consider advection out of the simulation domain
        lat_min:                      float32, min latitude of domain to consider advection out of the simulation domain
        lat_max:                      float32, max latitude of domain to consider advection out of the simulation domain
        deactivate_coords:            boolean, indicate if deactivate_north_of/south_of/_east_of/_west_of were used in simulation when plotting from .csv file
        start_date:                   pd.Datetime, start of plotted timeserie
                                      (e.g., pd.to_datetime("2018-01-01 00:00:00", format = '%Y-%m-%d %H:%M:%S'))
        end_date:                     pd.Datetime, end of plotted timeserie

        Returns a Pandas Dataframe:
            ----
        date_of_timestep:       calendar date of timestep as datetime.datetime object
        time:                   time from start of simulation expressed as time_unit
        time_conversion_factor
        mass_conversion_factor
            ---
        [considers only active elements inside the simulation domain]
        dissolved:              mass associated to dissolved elements for each timestep
        DOC:                    mass associated to DOC elements for each timestep
        SPM:                    mass associated to SPM elements for each timestep
        sed_rev:                mass associated to superficial sediments elements for each timestep
        sed_buried:             mass associated to buried sediments elements for each timestep
        mass_current:           mass present in the simulation at each timestep
        mass_wat:               mass present in the water column at each timestep
        mass_sed:               mass present in the active sediments at each timestep
            ---
        mass_emitted:           mass emitted into the simulation until the end of each timestep (ONLY ACTIVE ELEMENTS)
            ---
        mass_removed:           mass removed from the system until the end of each timestep
            ---
        [does not consider the specie of each element at each timestep]
        mass_degr:              mass that has been degraded up to the end of each timestep
        mass_degr_sed:          mass that has been degraded in sediments up to the end of each timestep
        mass_degr_wat:          mass that has been degraded in water up to the end of each timestep
        mass_vol:               mass that has been volatilized in water up to the end of each timestep
        mass_sed_buried:        mass that has been buried in sediments up to the end of each timestep
            ---
        adv_out_cumul:          mass that has been advected out up to the end of each timestep
        adv_out_ts:             mass that has been advected at each timestep
        --- IF deactivate_north_of/south_of/_east_of/_west_of ARE NOT USED IN SIMULATION
            An element is considered advected out only if it is outside the domain of
            the ou and vo velocity fields (i.e. deactivated) AND out of the shp/ lat-lon specified.
            Comment "adv_out = out_of_bounds & missing_var" to consider only shp/ lat-lon limits
            but this  will double count elements moving along the border at different timesteps.
            IF shp/lat-lon limits are not the same area covered by ou and vo velocity fields mass
            advected may be overestimated.
        --- IF deactivate_north_of/south_of/_east_of/_west_of ARE USED IN SIMULATION only elements
            deactivated with "outside" as reason will be counted
        '''
        import opendrift
        import matplotlib.pyplot as plt
        import pandas as pd
        import xarray as xr

        if shp_file_path is not None:
            import geopandas as gpd
            from shapely.geometry import Point

        if timeseries_file_path is not None:
            if timeseries_file_path.endswith(".csv"):
                csv_file_name = ""
        else:
            csv_file_name = sim_name + "_extracted_timeseries.csv"

        if load_timeseries_from_file is False:
            # Initialize init_species() and init_transfer_rates() if they were not stored in self.result
            if not np.all([(hasattr(self.result, attr)) for attr in ['nspecies', 'name_species', 'transfer_rates', 'ntransformations',
                                                                     'num_srev', 'num_ssrev', 'num_prev', 'num_lmm', 'num_col']]):
                # Init species and transfer rates
                if self.mode != opendrift.models.basemodel.Mode.Config:
                    self.mode = opendrift.models.basemodel.Mode.Config
                self.init_species()
                self.init_transfer_rates()
                if self.mode != opendrift.models.basemodel.Mode.Result:
                    self.mode = opendrift.models.basemodel.Mode.Result

            # Check if deactivate_N/S/E/W_of where specified
            deactivate_coords = []
            deactivate_coords.append(self.get_config('drift:deactivate_north_of'))
            deactivate_coords.append(self.get_config('drift:deactivate_south_of'))
            deactivate_coords.append(self.get_config('drift:deactivate_east_of'))
            deactivate_coords.append(self.get_config('drift:deactivate_west_of'))
            deactivate_coords = any(bool(dic) for dic in deactivate_coords)

            # Define elimination processes
            specie_ids_num = {"dissolved" : self.num_lmm,
                              "DOC" : self.num_humcol,
                              "SPM" : self.num_prev,
                              "sed_rev" : self.num_srev,
                              "buried": self.num_ssrev}
            specie_ids_name = {self.num_lmm: "dissolved",
                              self.num_humcol: "DOC",
                              self.num_prev : "SPM" ,
                              self.num_srev : "sed_rev" ,
                              self.num_ssrev : "buried"}

            Save_degr_now = self.get_config('chemical:transformations:Save_degr_now')

            status_categories = self.status_categories
            active_idx        = status_categories.index('active')          if 'active'   in status_categories else None
            outside_idx       = status_categories.index('outside')         if 'outside'  in status_categories else None
            stranded_idx      = status_categories.index('stranded')        if 'stranded' in status_categories else None
            removed_idx       = status_categories.index('removed')         if 'removed'  in status_categories else None
            missing_data_idx  = status_categories.index('missing_data')    if 'missing_data'   in status_categories else None
            seeded_on_land_idx = status_categories.index('seeded_on_land') if 'seeded_on_land' in status_categories else None



            # active_index = status_categories.index('active') if 'active' in status_categories else None
            # removed_index = status_categories.index('removed') if 'removed' in status_categories else None
            # missing_data_index = status_categories.index('missing_data') if 'missing_data' in status_categories else None
            # out_of_bounds_index = status_categories.index('outside') if 'outside' in status_categories else None
            # stranded_index = status_categories.index('stranded') if 'stranded' in status_categories else None
            # Define time and mass convertions

            # age_seconds = np.arange(self.time_step.total_seconds(), self.steps_calculation * self.time_step.total_seconds(), self.time_step.total_seconds())

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
            print("Extracting data from simulation")
            # Extract properties from simulation

            ds = self.result
            steps=len(self.result.time)
            vars_time = [v for v in ds.data_vars if ds[v].dims == ("trajectory", "time")]
            if not vars_time:
                raise ValueError("No (trajectory, time) variables found.")

            # Check only lat/lon
            valid_traj = (
                ds["lon"].notnull().any("time") | ds["lat"].notnull().any("time")
                )
            removed = ds.trajectory.where(~valid_traj, drop=True).values
            if len(removed) > 0:
                print(f"Removed IDs {removed} from self.result  as lat/lon were NaN")

            # keep only valid trajectories
            ds_clean = ds.isel(trajectory=valid_traj)
            self.result = ds_clean
            del ds_clean, ds


            # vars_time = [v for v in ds.data_vars if ds[v].dims == ("trajectory", "time")]
            # tr = ds[vars_time].sel(trajectory=141)   # xarray DataSet for trajectory 141
            # tr = self.result[["lon","lat","z","status","specie"]].sel(trajectory=141)
            # df = tr.to_dataframe().reset_index().drop(columns=["trajectory"])


            attrs_ls = ['lat', 'lon', 'mass',
                        'mass_degraded','mass_degraded_water', 'mass_degraded_sediment',
                        'mass_volatilized', 'mass_photodegraded', 'mass_biodegraded',
                        'mass_biodegraded_water', 'mass_biodegraded_sediment',
                        'mass_hydrolyzed', 'mass_hydrolyzed_water', 'mass_hydrolyzed_sediment'
                        ]
            # Dynamically create variables and assign values
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
            if hasattr(self, 'num_lmm'):
                in_water_column_ls.append(self.num_lmm)
            if hasattr(self, 'num_humcol'):
                in_water_column_ls.append(self.num_humcol)
            if hasattr(self, 'num_prev'):
                in_water_column_ls.append(self.num_prev)

            in_water_column = np.any([specie == value for value in in_water_column_ls], axis=0)
            in_sediment_layer = (specie==self.num_srev)
            in_buried_sed = (specie==self.num_ssrev)
            # mask for elements active until the end of each timestep
            active = (status == active_idx) if active_idx is not None else np.zeros_like(status, dtype=bool)

            # Find which elements are advected out of the domain
            print("Calculating mass advected out of the simulation")
            shp_adv_out = True
            if shp_file_path is not None:
                # Calculate mass of elemements outside of shapefile at each timestep.
                # Cumulative advection will double count elements that are not deactivated
                # when they exit the domain
                print("shp used to define simulation domain")
                gdf_shapefile = gpd.read_file(shp_file_path)
                out_of_bounds = []
                for t_step in range(0, steps):
                    lon_ = extracted_attrs_dict_2d['lon'][t_step]
                    lat_ = extracted_attrs_dict_2d['lat'][t_step]
                    # Create Point objects from latitude and longitude
                    geometry = [Point(long, latit) for long, latit in zip(lon_, lat_)]
                    gdf_points = gpd.GeoDataFrame(geometry=geometry, crs=gdf_shapefile.crs)
                    # Spatial join to check if points are within the shapefile geometry
                    # Use 'within' operation to check if points are within the shapefile
                    joined = gpd.sjoin(gdf_points, gdf_shapefile, predicate='within')
                    # Extract the indices of points that are within the shapefile
                    indices_within = joined.index
                    # Create a boolean array indicating whether each point is within the shapefile
                    out_of_bounds.append([index in indices_within for index in range(len(geometry))])
                # Convert the list of boolean arrays to a NumPy array
                adv_out = np.array([np.array(arr) for arr in out_of_bounds])

            elif deactivate_coords is True:
                shp_adv_out = False
                print("deactivate_north_of/south_of/_east_of/_west_of parameters used")
                if ('outside' in status_categories):
                    adv_out = (status == outside_idx)
                else:
                    adv_out = np.zeros_like(status)
                    print("No elements advected out of domain")
            elif any([element is None for element in [lat_min, lat_max, lon_max, lon_min]]) is False:
                shp_adv_out = False
                # advection out of the domain is determined by the extent of uo/vo velocity field
                if ('missing_data' in status_categories):
                    print("missing_data used: : adv_out elements taken from self.elements_deactivated")
                    adv_out = (status == missing_data_idx)
                else:
                    adv_out = np.zeros_like(status)
                    print("No elements advected out of domain")
            else:
                error = ("shp_file_path and lat/lon limits not specified when " +
                         "deactivate_north_of/south_of/_east_of/_west_of parameters " +
                         "were not used")
                raise ValueError(error)

            age_seconds_output = np.array(((result_ds.time - result_ds.time[0]) / np.timedelta64(1, 's')) + (result_ds.time[1] - result_ds.time[0]) / np.timedelta64(1, 's'))

            time_steps =np.array(range(0, steps))*time_conversion_factor
            start_date = pd.Timestamp(self.start_time.year,
                                      self.start_time.month,
                                      self.start_time.day,
                                      self.start_time.hour,
                                      self.start_time.minute)
            time_date_serie = pd.date_range(start=start_date, periods=steps, freq=self.time_step_output)


            adv_out=adv_out.astype(bool)
            in_water_column=in_water_column.astype(bool)
            in_sediment_layer=in_sediment_layer.astype(bool)
            in_buried_sed=in_buried_sed.astype(bool)


            def masked_nansum(arr, mask, axis):
                """Sum arr over axis, only where mask==True; ignore NaNs inside mask."""
                return np.nansum(np.where(mask, arr, np.nan), axis=axis)


            # Dynamically create variables and assign values
            # ===== 1) Inside-system mass (active & not outside elements) =====
            inside_mask = active & (~adv_out)
            mass        = np.nan_to_num(extracted_attrs_dict_2d['mass'])

            extracted_active_dict_1d = {}
            extracted_active_dict_1d["mass_water_ts"] = masked_nansum(extracted_attrs_dict_2d['mass'], in_water_column  & inside_mask, axis=1) * mass_conversion_factor
            extracted_active_dict_1d["mass_sed_ts"] = masked_nansum(extracted_attrs_dict_2d['mass'], in_sediment_layer  & inside_mask, axis=1) * mass_conversion_factor
            extracted_active_dict_1d["mass_buried_ts"] = masked_nansum(extracted_attrs_dict_2d['mass'], in_buried_sed   & inside_mask, axis=1) * mass_conversion_factor
            extracted_active_dict_1d["mass_actual_ts"] = extracted_active_dict_1d["mass_water_ts"] + extracted_active_dict_1d["mass_sed_ts"]

            # ===== 2) Emitted mass (first timestep each element is finite) =====
            valid = np.isfinite(extracted_attrs_dict_2d['mass'])
            first_seen_idx = valid.argmax(axis=0)                 # 0 if never seen; guard with has_seen
            has_seen = valid.any(axis=0)
            # weight by mass at first_seen
            rows = first_seen_idx[has_seen]
            cols = np.arange(N_elem)[has_seen]
            first_seen_mass = np.zeros(N_elem); first_seen_mass[has_seen] = mass[rows, cols]
            emitted_mass = np.bincount(first_seen_idx[has_seen], weights=first_seen_mass[has_seen], minlength=steps)
            extracted_active_dict_1d["emitted_mass_ts"] = emitted_mass * mass_conversion_factor
            extracted_active_dict_1d["emitted_mass_cumulative"] = np.cumsum(extracted_active_dict_1d["emitted_mass_ts"])

            # ===== 3) Degraded mass (all active) ===== CONTROLLARE COME GESTIRE MASS DI ELEMENTI ADV OUT/STRADED ECC
            attrs_ls_1d_general = ['mass_degraded','mass_degraded_water', 'mass_degraded_sediment']
            attrs_ls_1d_general_now = ['mass_degraded_now', 'mass_degraded_water_now', 'mass_degraded_sediment_now']
            attrs_ls_1d_single_process = ['mass_volatilized', 'mass_photodegraded', 'mass_biodegraded',
                        'mass_biodegraded_water', 'mass_biodegraded_sediment',
                        'mass_hydrolyzed', 'mass_hydrolyzed_water', 'mass_hydrolyzed_sediment'
                        ]


            deactivated_mask = np.nan_to_num(status) != active_idx
            for attr in  attrs_ls_1d_general:
                attr = 'mass_degraded'
                extracted_active_dict_1d[f"{attr}_cumulative"] = (
                # Degraded mass logged for active elements (always cumulative)
                ((masked_nansum(extracted_attrs_dict_2d.get(attr), active, axis=1) * mass_conversion_factor )
                if extracted_attrs_dict_2d.get(attr) is not None else np.zeros_like(extracted_active_dict_1d["mass_actual"]))
                + # Account for degraded mass that is removed from self.result after deactivation
                (np.cumsum((masked_nansum(extracted_attrs_dict_2d.get(attr), deactivated_mask, axis=1) * mass_conversion_factor )
                if extracted_attrs_dict_2d.get(attr) is not None else np.zeros_like(extracted_active_dict_1d["mass_actual"])))
                )



            for attr in  attrs_ls_1d_general_now:
                extracted_active_dict_1d[f"{attr.replace('_now', '_ts')}"] = (
                    (masked_nansum(extracted_attrs_dict_2d.get(attr), active, axis=1) * mass_conversion_factor )
                if extracted_attrs_dict_2d.get(attr) is not None else np.zeros_like(extracted_active_dict_1d["mass_actual"]))


                if Save_degr_now:
                    extracted_active_dict_1d[attr] = (
                        (masked_nansum(extracted_attrs_dict_2d.get(attr), inside_mask, axis=1) * mass_conversion_factor )
                    if extracted_attrs_dict_2d.get(attr) is not None else np.zeros_like(extracted_active_dict_1d["mass_actual"]))

                    extracted_active_dict_1d[f"{attr}_cumulative"] = np.cumsum(extracted_active_dict_1d[attr])


            extracted_deactivated_dict_1d = {}

            deactivated_mask = np.nan_to_num(status) != active_idx
            for attr in  attrs_ls_1d:
                extracted_deactivated_dict_1d[attr] = (
                    (masked_nansum(extracted_attrs_dict_2d.get(attr), deactivated_mask, axis=1) * mass_conversion_factor )
                if extracted_attrs_dict_2d.get(attr) is not None else np.zeros_like(extracted_active_dict_1d["mass_actual"]))

                extracted_deactivated_dict_1d[f"{attr}_cumulative"] = np.cumsum(extracted_deactivated_dict_1d[attr])
            if not Save_degr_now:
                pass


            # ===== 4) Mass by specie (active & not outside elements) =====
            for sp in specie_ids_name.keys():
                if specie_ids_name.get(sp) in ["buried", "sed_rev"]:
                    continue
                extracted_active_dict_1d[f"mass_{specie_ids_name.get(sp)}"] = \
                masked_nansum(extracted_attrs_dict_2d['mass'], ((specie == sp)  & (inside_mask)), axis=1) * mass_conversion_factor

            # ===== 5) Outside/advection  =====
            # Either explicit status or adv_out mask when it flips False->True while active
            exit_outside_at_t = np.zeros(steps)
            if outside_idx is not None:
                # Use status if present: first time status==outside for each element
                outside_first_idx = np.full(N_elem, -1, dtype=int)
                hit_outside = (status == outside_idx).any(axis=0)
                if N_elem > 0:
                    outside_first_idx[hit_outside] = (status == outside_idx).argmax(axis=0)[hit_outside]
                # mass at that instant (use last finite before that instant if needed)
                w = np.zeros(N_elem)
                for j in np.where(hit_outside)[0]:
                    t0 = outside_first_idx[j]
                    k = t0
                    while k >= 0 and not np.isfinite(mass[k, j]):
                        k -= 1
                    if k >= 0:
                        w[j] = mass[k, j]
                exit_outside_at_t = np.bincount(outside_first_idx[hit_outside], weights=w[hit_outside], minlength=steps)
            else:
                # Fall back to adv_out mask: count first time it becomes True while previously inside
                became_out = (adv_out.astype(int).diff(axis=0, prepend=0) == 1) & active
                # time-local mass when it becomes outside
                exit_outside_at_t = masked_nansum(mass, became_out, axis=1)

            extracted_active_dict_1d["adv_out_mass"] = exit_outside_at_t * mass_conversion_factor
            extracted_active_dict_1d["adv_out_mass_cumulative"] = np.cumsum(extracted_active_dict_1d["adv_out_mass"])

            # ===== 6) Stranding  =====#
            exit_stranding_at_t = np.zeros(steps)
            if stranded_idx is not None:
                stranded_first_idx = np.full(N_elem, -1, dtype=int)
                hit_strand = (status == stranded_idx).any(axis=0)
                if N_elem > 0:
                    stranded_first_idx[hit_strand] = (status == stranded_idx).argmax(axis=0)[hit_strand]
                w = np.zeros(N_elem)
                for j in np.where(hit_strand)[0]:
                    t0 = stranded_first_idx[j]
                    k = t0
                    while k >= 0 and not np.isfinite(mass[k, j]):
                        k -= 1
                    if k >= 0:
                        w[j] = mass[k, j]
                exit_stranding_at_t = np.bincount(stranded_first_idx[hit_strand], weights=w[hit_strand], minlength=steps)
            extracted_active_dict_1d["stranded_mass"] = exit_stranding_at_t * mass_conversion_factor
            extracted_active_dict_1d["stranded_mass_cumulative"] = np.cumsum(extracted_active_dict_1d["stranded_mass"])

            # ===== 6) Burial  =====#
            # First time specie transitions to "buried" specie id (num_ssrev)
            exit_burial_at_t = np.zeros(steps)
            buried_id = specie_ids_num.get('buried', None)
            if buried_id is not None:
                is_buried = (specie == buried_id)
                hit_burial = is_buried.any(axis=0)
                burial_first_idx = np.full(N_elem, -1, dtype=int)
                if N_elem > 0:
                    burial_first_idx[hit_burial] = is_buried.argmax(axis=0)[hit_burial]
                w = np.zeros(N_elem)
                for j in np.where(hit_burial)[0]:
                    t0 = burial_first_idx[j]
                    k = t0
                    while k >= 0 and not np.isfinite(mass[k, j]):
                        k -= 1
                    if k >= 0:
                        w[j] = mass[k, j]
                exit_burial_at_t = np.bincount(burial_first_idx[hit_burial], weights=w[hit_burial], minlength=steps)
            extracted_active_dict_1d["buried_mass"] = exit_burial_at_t * mass_conversion_factor
            extracted_active_dict_1d["buried_mass_cumulative"] = np.cumsum( extracted_active_dict_1d["buried_mass"])




            # ===== 5) Assemble DataFrame =====
            df = pd.DataFrame({
                f'time_[{getattr(time_steps, "units", "user")}]' if hasattr(time_steps, 'units') else f'time_[user]': time_steps,
                'date_of_timestep': time_date_serie,
                'emitted_mass': emitted_mass,
                'mass_water': mass_water,
                'mass_sed': mass_sed,
                'mass_sed_buried': mass_bur,
                'mass_actual': mass_actual,
                'exit_degradation': exit_deg_at_t,
                'exit_outside': exit_outside_at_t,
                'exit_stranding': exit_stranding_at_t,
                'exit_burial': exit_burial_at_t,
                'cum_degradation_deactivated_only': cum_degradation_deactivated_only,
                'cum_outside': cum_outside,
                'cum_stranding': cum_strand,
                'cum_burial': cum_burial,
                'cum_total_exit': cum_total_exit,
            })
            return df

































    #     # #############

    #     # # Get mass associated to each specie for each timestep, considering only active elements that are inside
    #     # # the simulation domain (not stranded or advected out)
    #     # print("Calculating timeseries from simulation")
    #     # mass_by_specie=np.zeros((steps,5))

    #     # for i in range(steps):
    #     #     mass_by_specie[i] = [
    #     #             np.sum(extracted_attrs_dict_2d['mass'][i, :] * (sp[i, :] == j) * (~adv_out[i, :])*(active_elements[i, :])) * mass_conversion_factor
    #     #             for j in range(5)
    #     #             ]

    #     # time_steps =np.array(range(0, steps))*time_conversion_factor
    #     # mass_by_specie_df = pd.DataFrame(mass_by_specie)

    #     # for chem_specie in range(0,5):
    #     #     if (chem_specie not in mass_by_specie_df.columns):
    #     #         mass_by_specie_df[chem_specie] = np.zeros_like(time_steps)

    #     # # Insert time as new columns
    #     # mass_by_specie_df.insert(0,('time'+"-["+ time_unit +"]"), time_steps)
    #     # start_date = pd.Timestamp(self.start_time.year,
    #     #                           self.start_time.month,
    #     #                           self.start_time.day)
    #     # time_date_serie = pd.date_range(start=start_date, periods=steps, freq=self.time_step_output)
    #     # mass_by_specie_df.insert(0,('date_of_timestep'), time_date_serie)
    #     # del mass_by_specie

    #     # # Change name of dataframe columns
    #     # Col_names_res = list(mass_by_specie_df.columns)
    #     # names_to_replace = ({0: "dissolved" + "-["+ mass_unit +"]",
    #     #                     1: "DOC" + "-["+ mass_unit +"]",
    #     #                     2: "SPM" + "-["+ mass_unit +"]",
    #     #                     3: "sed_rev" + "-["+ mass_unit +"]",
    #     #                     4: "sed_buried" + "-["+ mass_unit +"]"
    #     #                     })
    #     # Col_names_res = [names_to_replace.get(e, e) for e in Col_names_res]
    #     # mass_by_specie_df.columns = Col_names_res

    #     # mass_results_timestep = {}
    #     # # Mass present in the system at each timestep considering only active elements
    #     # # that are inside the simulation domain

    #     # mass_results_timestep["mass_water"] = (np.sum(extracted_attrs_dict['mass']*in_water_column * (~adv_out)*active_elements,1) * mass_conversion_factor)
    #     # mass_results_timestep["mass_sed"] = (np.sum(extracted_attrs_dict['mass']*in_sediment_layer * (~adv_out)*active_elements,1) * mass_conversion_factor)
    #     # # mass_sed_buried is outside the domain
    #     # mass_results_timestep["mass_sed_buried"] = (np.sum(extracted_attrs_dict['mass']*in_buried_sed * (~adv_out)*active_elements,1) * mass_conversion_factor)
    #     # mass_results_timestep["mass_actual"] = mass_results_timestep["mass_water"] + mass_results_timestep["mass_sed"]


    #     # # Mass of all elements present in simulation during each timestep

    #     # timestep_attributes = ['mass', 'mass_degraded','mass_degraded_water', 'mass_degraded_sediment',
    #     #         'mass_volatilized', 'mass_photodegraded', 'mass_biodegraded',
    #     #         'mass_biodegraded_water', 'mass_biodegraded_sediment',
    #     #         'mass_hydrolyzed', 'mass_hydrolyzed_water', 'mass_hydrolyzed_sediment'
    #     #     ]

    #     # for attr in timestep_attributes:
    #     #     if extracted_attrs_dict[attr] is not None:
    #     #         mass_results_timestep[attr] = np.sum(extracted_attrs_dict[attr] * active_elements, axis=1) * mass_conversion_factor
    #     #     else:
    #     #         mass_results_timestep[attr] = np.zeros_like(mass_results_timestep["mass_actual"])



    #     # mass_tot_timestep = (np.sum(extracted_attrs_dict['mass'] * elements,1)*mass_conversion_factor)
    #     # # Mass that has been degraded up to the end of each timestep considering all elements
    #     # mass_degraded = (np.sum(extracted_attrs_dict['mass_degraded']*elements,1)*mass_conversion_factor)
    #     # # Mass that has been degraded (wat and sed) and volatilized up to the end of each timestep
    #     # # considering considering all elements
    #     # mass_volatilized = (np.sum((extracted_attrs_dict['mass_volatilized'])*elements,1)*mass_conversion_factor)
    #     # mass_degraded_w = (np.sum(extracted_attrs_dict['mass_degraded_water']*elements,1)*mass_conversion_factor)
    #     # mass_degraded_s = (np.sum(extracted_attrs_dict['mass_degraded_sediment']*elements,1)*mass_conversion_factor)

    #     # 'mass'
    #     # 'mass_degraded','mass_degraded_water', 'mass_degraded_sediment',
    #     # 'mass_volatilized', 'mass_photodegraded', 'mass_biodegraded',
    #     # 'mass_biodegraded_water', 'mass_biodegraded_sediments',
    #     # 'mass_hydrolyzed', 'mass_hydrolyzed_water', 'mass_hydrolyzed_sediments'

    #     # controllo = (active_elements * (~adv_out) * (~stranded_elements)) == elements


    #     # Calculate mass of  elements that are advected out of the domain
    #     if shp_adv_out == False:
    #         if ('missing_data' in status_categories) or ('outside' in status_categories):
    #             # adv_out = out_of_bounds
    #             # Mass advected out of the system during each time step
    #             mass_adv_out = (np.sum((extracted_attrs_dict['mass']*adv_out),1)*mass_conversion_factor)
    #             # Mass advected out of the system from start of simulation, until that time step
    #             mass_adv_out_cumulative = np.cumsum(mass_adv_out)
    #     else:
    #         mass_adv_out = np.zeros_like(mass_water)
    #         mass_adv_out_cumulative = np.zeros_like(mass_water)

    #         # Mass emitted into the system up to the end of each timestep
    #         mass_emitted = (mass_results_timestep['mass'] + mass_results_timestep['mass_degraded'] + mass_results_timestep['mass_volatilized'])

    #         # Mass that has been degraded (wat and sed) and volatilized of elements
    #         # advected out of the system
    #         mass_volatilized_adv = (np.sum((extracted_attrs_dict['mass_volatilized'])* adv_out,1)*mass_conversion_factor)
    #         mass_degraded_w_adv = (np.sum(extracted_attrs_dict['mass_degraded_water']* adv_out,1)*mass_conversion_factor)
    #         mass_degraded_s_adv = (np.sum(extracted_attrs_dict['mass_degraded_sediment']* adv_out,1)*mass_conversion_factor)
    #         mass_photodegr_adv = (np.sum(extracted_attrs_dict['mass_photodegraded']* adv_out,1)*mass_conversion_factor) if extracted_attrs_dict['mass_photodegraded'] is not None else np.zeros_like(mass_emitted)
    #         mass_degraded_bio_adv = (np.sum(extracted_attrs_dict['mass_biodegraded']* adv_out,1)*mass_conversion_factor) if extracted_attrs_dict['mass_biodegraded'] is not None else np.zeros_like(mass_emitted)
    #         mass_degraded_bio_w_adv = (np.sum(extracted_attrs_dict['mass_biodegraded_water']* adv_out,1)*mass_conversion_factor) if extracted_attrs_dict['mass_biodegraded_water'] is not None else np.zeros_like(mass_emitted)
    #         mass_degraded_bio_s_adv = (np.sum(extracted_attrs_dict['mass_biodegraded_sediments']* adv_out,1)*mass_conversion_factor) if extracted_attrs_dict['mass_biodegraded_sediments'] is not None else np.zeros_like(mass_emitted)
    #         mass_degraded_hydro_adv = (np.sum(extracted_attrs_dict['mass_hydrolyzed']* adv_out,1)*mass_conversion_factor) if extracted_attrs_dict['mass_hydrolyzed'] is not None else np.zeros_like(mass_emitted)
    #         mass_degraded_hydro_w_adv = (np.sum(extracted_attrs_dict['mass_hydrolyzed_water']* adv_out,1)*mass_conversion_factor) if extracted_attrs_dict['mass_hydrolyzed_water'] is not None else np.zeros_like(mass_emitted)
    #         mass_degraded_hydro_s_adv = (np.sum(extracted_attrs_dict['mass_hydrolyzed_sediments']* adv_out,1)*mass_conversion_factor) if extracted_attrs_dict['mass_hydrolyzed_sediments'] is not None else np.zeros_like(mass_emitted)


    #         # Mass removed from the system until the end of each timestep
    #         mass_removed = (mass_results_timestep['mass_volatilized'] - mass_volatilized_adv) + \
    #                        (mass_results_timestep['mass_degraded_water'] - mass_degraded_w_adv) +\
    #                        (mass_results_timestep['mass_degraded_sediment'] - mass_degraded_s_adv) + \
    #                        (mass_adv_out_cumulative + mass_sed_buried)
    #         # Contribtion of each mechanism to the elimination of the chemical in the simulation
    #         perc_volatilized = ((mass_results_timestep['mass_volatilized'] - mass_volatilized_adv)/mass_removed)*100
    #         perc_degraded_w = ((mass_results_timestep['mass_degraded_water'] - mass_degraded_w_adv)/mass_removed)*100
    #         perc_degraded_s = ((mass_results_timestep['mass_degraded_sediment'] - mass_degraded_s_adv)/mass_removed)*100
    #         perc_photodegr = ((mass_results_timestep['mass_photodegraded'] - mass_photodegr_adv)/mass_removed)*100
    #         perc_degraded_bio = ((mass_results_timestep['mass_biodegraded'] - mass_degraded_bio_adv)/mass_removed)*100
    #         perc_degraded_bio_w = ((mass_results_timestep['mass_biodegraded_water'] - mass_degraded_bio_w_adv)/mass_removed)*100
    #         perc_degraded_bio_s = ((mass_results_timestep['mass_biodegraded_sediments'] - mass_degraded_bio_s_adv)/mass_removed)*100
    #         perc_mass_degraded_hydro = ((mass_results_timestep['mass_hydrolyzed'] - mass_degraded_hydro_adv)/mass_removed)*100
    #         perc_mass_degraded_hydro_w = ((mass_results_timestep['mass_hydrolyzed_water'] - mass_degraded_hydro_w_adv)/mass_removed)*100
    #         perc_mass_degraded_hydro_s = ((mass_results_timestep['mass_hydrolyzed_sediments'] - mass_degraded_hydro_s_adv)/mass_removed)*100
    #         perc_adv = ((mass_adv_out_cumulative)/mass_removed)*100
    #         perc_sed_buried = ((mass_sed_buried)/mass_removed)*100

    #     mass_df = pd.DataFrame({
    #         'mass_wat'+ "-["+ mass_unit +"]": mass_water,
    #         'mass_sed_rev'+ "-["+ mass_unit +"]": mass_sed,
    #         'mass_sed_buried'+ "-["+ mass_unit +"]": mass_sed_buried,
    #         'mass_current'+ "-["+ mass_unit +"]": mass_actual,
    #         'mass_emitted'+ "-["+ mass_unit +"]": mass_emitted,
    #         'mass_degr_tot'+ "-["+ mass_unit +"]": mass_results_timestep['mass_degraded'],
    #         'mass_degr_wat'+ "-["+ mass_unit +"]": mass_results_timestep['mass_degraded_water'],
    #         'mass_degr_sed'+ "-["+ mass_unit +"]": mass_results_timestep['mass_degraded_sediment'],
    #         'mass_vol'+ "-["+ mass_unit +"]": mass_results_timestep['mass_volatilized'],
    #         'adv_out_ts'+"-["+ mass_unit +"]": mass_adv_out,
    #         'adv_out_cumul'+"-["+ mass_unit +"]": mass_adv_out_cumulative,
    #         'perc_vol-["%"]': perc_volatilized,
    #         'perc_deg_w-["%"]': perc_degraded_w,
    #         'perc_deg_s-["%"]': perc_degraded_s,
    #         'perc_adv-["%"]': perc_adv,
    #         'perc_sed_buried-["%"]': perc_sed_buried,
    #         'mass_removed'+ "-["+ mass_unit +"]": mass_removed,
    #         'mass_vol_adv'+ "-["+ mass_unit +"]": mass_volatilized_adv,
    #         'mass_degr_wat_adv'+ "-["+ mass_unit +"]": mass_degraded_w_adv,
    #         'mass_degr_sed_adv'+ "-["+ mass_unit +"]": mass_degraded_s_adv
    #         }).astype(np.float64)

    #     mass_fin_df = pd.concat([mass_by_specie_df, mass_df], axis=1)
    #     mass_fin_df[('convertion_factors-[time_mass]')] = np.zeros_like(mass_emitted)
    #     mass_fin_df.loc[0, ('convertion_factors-[time_mass]')]= time_conversion_factor
    #     mass_fin_df.loc[1, ('convertion_factors-[time_mass]')]= np.float32(mass_conversion_factor)
    #     print(f"save .csv file to {file_out_path + csv_file_name}")
    #     mass_fin_df.to_csv(file_out_path + csv_file_name)

    # elif load_timeseries_from_file is True and timeseries_file_path is not None:
    #     mass_fin_df = pd.read_csv(timeseries_file_path + csv_file_name)
    #     # Specify mass_unit and time_unit from loaded dataframe
    #     mass_col_ind = int(np.where(mass_fin_df.columns.str.contains("dissolved-"))[0])
    #     mass_unit = mass_fin_df.columns[mass_col_ind][11:-1]
    #     time_col_ind = int(np.where(mass_fin_df.columns.str.contains("time-"))[0])
    #     time_unit = mass_fin_df.columns[time_col_ind][6:-1]
    #     mass_fin_df = mass_fin_df.dropna(subset = ['date_of_timestep'])

    # else:
    #     raise ValueError("timeseries_file_path must be specified")

    # time_conversion_factor = np.float32(mass_fin_df.loc[0, ('convertion_factors-[time_mass]')])
    # mass_conversion_factor = np.float32(mass_fin_df.loc[1, ('convertion_factors-[time_mass]')])

    # # Filter dataframe based on date
    # mass_fin_df = self.filter_dataframe(df = mass_fin_df,
    #                                time_unit = time_unit,
    #                                start_date = start_date, end_date = end_date)

    # # Create timeseries to plot
    # time_steps = self._select_df_column(col_name = "time-", dataframe = mass_fin_df)
    # mass_emitted = self._select_df_column(col_name = 'mass_emitted-', dataframe = mass_fin_df)
    # mass_actual = self._select_df_column(col_name = 'mass_current-', dataframe = mass_fin_df)
    # mass_water = self._select_df_column(col_name = 'mass_wat-', dataframe = mass_fin_df)
    # mass_sed = self._select_df_column(col_name = 'mass_sed_rev-', dataframe = mass_fin_df)
    # mass_degraded = self._select_df_column(col_name = 'mass_degr_tot-', dataframe = mass_fin_df)
    # mass_degraded_s = self._select_df_column(col_name = 'mass_degr_sed-', dataframe = mass_fin_df)
    # mass_degraded_w = self._select_df_column(col_name = 'mass_degr_wat-', dataframe = mass_fin_df)
    # mass_adv_out_cumulative = self._select_df_column(col_name = 'adv_out_cumul-', dataframe = mass_fin_df)
    # mass_adv_out = self._select_df_column(col_name = 'adv_out_ts-', dataframe = mass_fin_df)
    # mass_volatilized = self._select_df_column(col_name = 'mass_vol-', dataframe = mass_fin_df)
    # mass_sed_buried = self._select_df_column(col_name = 'mass_sed_buried-', dataframe = mass_fin_df)

    # perc_adv = self._select_df_column(col_name = 'perc_adv-', dataframe = mass_fin_df)
    # perc_degraded_s = self._select_df_column(col_name = 'perc_deg_s-', dataframe = mass_fin_df)
    # perc_degraded_w = self._select_df_column(col_name = 'perc_deg_w-', dataframe = mass_fin_df)
    # perc_volatilized = self._select_df_column(col_name = 'perc_vol-', dataframe = mass_fin_df)
    # perc_sed_buried = self._select_df_column(col_name ='perc_sed_buried-', dataframe = mass_fin_df)

    # if plot_figures is True:
    #     print("Plotting figures")
    #     if title_fig_tserie is None and chemical_compound is not None:
    #         title_fig_tserie = (chemical_compound + ' mass')
    #     if title_fig_mass is None and chemical_compound is not None:
    #         title_fig_mass = (chemical_compound + ' mass')
    #     if title_fig_deg is None and chemical_compound is not None:
    #         title_fig_deg = (chemical_compound + ' removal')

    #     mass_dissolved = (mass_fin_df[mass_fin_df.columns[mass_fin_df.columns.str.contains("dissolved-")]])
    #     mass_DOC = (mass_fin_df[mass_fin_df.columns[mass_fin_df.columns.str.contains("DOC-")]])
    #     mass_SPM = (mass_fin_df[mass_fin_df.columns[mass_fin_df.columns.str.contains("SPM-")]])
    #     mass_sed_rev = (mass_fin_df[mass_fin_df.columns[mass_fin_df.columns.str.contains("mass_sed_rev-")]])
    #     mass_sed_buried = (mass_fin_df[mass_fin_df.columns[mass_fin_df.columns.str.contains("mass_sed_buried-")]])

    #     # Plot mass present/degraded/volatilized in the system
    #     xlabel = "time" + " ["+ time_unit +"]"
    #     fig_width = 2.5 * subplot_width
    #     fig_height = 3.7 * subplot_height

    #     fig1,((ax1,ax2),(ax3,ax4), (ax5, ax6), (ax7, ax8))=plt.subplots(nrows=4, ncols=2, figsize=(fig_width, fig_height))
    #     fig1.suptitle(title_fig_tserie)

    #     ax1.plot(time_steps, mass_emitted)
    #     ax1.set_xlabel(xlabel)
    #     ax1.set_ylabel('emitted mass'+ " ["+ mass_unit +"]")

    #     ax2.plot(time_steps, mass_actual, 'k',
    #              time_steps, mass_water, 'b',
    #              time_steps, mass_sed, 'y')
    #     ax2.set_xlabel(xlabel)
    #     ax2.set_ylabel('current mass'+ " ["+ mass_unit +"]")
    #     ax2.legend(['total mass','mass in wat','mass in sed'],prop={'size': 8})

    #     ax3.plot(time_steps, mass_degraded,'k',
    #              time_steps, mass_degraded_s,'y',
    #              time_steps, mass_degraded_w,'b')

    #     ax3.set_xlabel(xlabel)
    #     ax3.set_ylabel('degraded mass'+ " ["+ mass_unit +"]")
    #     ax3.legend(['degr. total','degr. in sed','degr. in wat'],prop={'size': 8})

    #     ax4.plot(time_steps, mass_adv_out_cumulative, 'lime')
    #     if deactivate_coords is False:
    #         ax4.plot(time_steps, mass_adv_out, 'm')
    #         ax4.legend(['adv. out cumulative','adv. out timestep'],prop={'size': 8})
    #     else:
    #         ax4.legend(['adv. out'],prop={'size': 8})
    #     ax4.set_xlabel(xlabel)
    #     ax4.set_ylabel('advected out mass' + " ["+ mass_unit +"]")

    #     ax5.plot(time_steps, mass_volatilized, 'orange')
    #     ax5.set_xlabel(xlabel)
    #     ax5.set_ylabel('volatilized mass' + " ["+ mass_unit +"]")

    #     ax6.plot(time_steps, mass_sed_buried, 'purple')
    #     ax6.set_xlabel(xlabel)
    #     ax6.set_ylabel('buried in sed mass' + " ["+ mass_unit +"]")

    #     mass_dissolved_arr = mass_dissolved.to_numpy().flatten()
    #     mass_DOC_arr = mass_DOC.to_numpy().flatten()
    #     mass_SPM_arr = mass_SPM.to_numpy().flatten()
    #     mass_sed_rev_arr = mass_sed_rev.to_numpy().flatten()
    #     mass_actual_arr = mass_actual.to_numpy().flatten()

    #     ax7.plot(time_steps, (mass_dissolved_arr/mass_actual_arr)*100, 'midnightblue',
    #              time_steps, (mass_DOC_arr/mass_actual_arr)*100, 'royalblue',
    #              time_steps, (mass_SPM_arr/mass_actual_arr)*100, 'palegreen',
    #              time_steps,(mass_sed_rev_arr/mass_actual_arr)*100, 'orange')
    #     ax7.set_xlabel(xlabel)
    #     ax7.set_ylabel('mass distribution [%]')
    #     ax7.legend(['dissolved','DOC', 'SPM', 'sed'],prop={'size': 8})

    #     ax8.plot(time_steps, perc_volatilized, 'orange',
    #              time_steps, perc_degraded_w, 'b',
    #              time_steps, perc_degraded_s, 'y',
    #              time_steps, perc_adv, 'm',
    #              time_steps, perc_sed_buried, 'saddlebrown',
    #              )
    #     ax8.set_xlabel(xlabel)
    #     ax8.set_ylabel('contribution to elimination [%]')
    #     ax8.legend(['vol','deg_w', 'deg_s', 'adv', 'sed_burial'],prop={'size': 8})

    #     fig1.tight_layout()
    #     fig1.savefig(file_out_path+"/"+sim_name+"-time_series"+".png")

    #     # Plot mass distribution among environmental media
    #     bars=np.zeros((len(time_steps),5))

    #     for i in range(0, len(time_steps)):
    #         bars[i]=[mass_dissolved.iloc[i, 0],
    #                  mass_DOC.iloc[i, 0],
    #                  mass_SPM.iloc[i, 0],
    #                  mass_sed_rev.iloc[i, 0],
    #                  mass_sed_buried.iloc[i, 0]]

    #     bottom=np.zeros_like(bars[:,0])

    #     fig2, ax = plt.subplots()
    #     fig2.suptitle(title_fig_mass)
    #     ax.bar(np.arange((len(time_steps))),bars[:,0],width=1.25,color='midnightblue')
    #     bottom=bars[:,0]
    #     ax.bar(np.arange((len(time_steps))),bars[:,1],bottom=bottom,width=1.25,color='royalblue')
    #     bottom=bottom+bars[:,1]
    #     ax.bar(np.arange((len(time_steps))),bars[:,2],bottom=bottom,width=1.25,color='palegreen')
    #     bottom=bottom+bars[:,2]
    #     ax.bar(np.arange((len(time_steps))),bars[:,3],bottom=bottom,width=1.25,color='orange')
    #     bottom=bottom+bars[:,3]

    #     print(f'dissolved: {str(bars[-1,0])} {mass_unit} ({str(100*bars[-1,0]/np.sum(bars[-1,:]))} %)')
    #     print(f'DOC: {str(bars[-1,1])} {mass_unit} ({str(100*bars[-1,1]/np.sum(bars[-1,:]))} %)')
    #     print(f'SPM: {str(bars[-1,2])} {mass_unit} ({str(100*bars[-1,2]/np.sum(bars[-1,:]))} %)')
    #     print(f'sediment: {str(bars[-1,3])} {mass_unit} ({str(100*bars[-1,3]/np.sum(bars[-1,:]))} %)')


    #     ax.legend(['dissolved', 'DOC', 'SPM','sediment'])
    #     ax.set_ylabel('mass (' + mass_unit + ')')
    #     # Get current tick positions
    #     xticks = ax.get_xticks()
    #     # Set the ticks explicitly before setting labels
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(np.round(xticks * time_conversion_factor))
    #     ax.set_xlabel('time (' + time_unit + ')')

    #     plt.savefig(file_out_path+"/"+sim_name+"-mass_specie"+".png", bbox_inches="tight", dpi=300)

    #     # Plot degradatin mecanisms during simulations
    #     bars_deg=np.zeros((len(time_steps),5))
    #     for i in range(0, len(time_steps)):
    #         bars_deg[i]=[perc_volatilized.iloc[i, 0],
    #                  perc_degraded_w.iloc[i, 0],
    #                  perc_degraded_s.iloc[i, 0],
    #                  perc_adv.iloc[i, 0],
    #                  perc_sed_buried.iloc[i, 0]]

    #     bottom_deg=np.zeros_like(bars_deg[:,0])

    #     fig3, ax = plt.subplots()
    #     fig3.suptitle(title_fig_deg)
    #     ax.bar(np.arange((len(time_steps))),bars_deg[:,0],width=1.25,color='orange')
    #     bottom_deg=bars_deg[:,0]
    #     ax.bar(np.arange((len(time_steps))),bars_deg[:,1],bottom=bottom_deg,width=1.25,color='b')
    #     bottom_deg = bottom_deg+bars_deg[:,1]
    #     ax.bar(np.arange((len(time_steps))),bars_deg[:,2],bottom=bottom_deg,width=1.25,color='y')
    #     bottom_deg = bottom_deg + bars_deg[:,2]
    #     ax.bar(np.arange((len(time_steps))),bars_deg[:,3],bottom=bottom_deg,width=1.25,color='royalblue')
    #     bottom_deg = bottom_deg+bars_deg[:,3]
    #     ax.bar(np.arange((len(time_steps))),bars_deg[:,4],bottom=bottom_deg,width=1.25,color='saddlebrown')

    #     ax.legend(['vol','deg_w', 'deg_s', 'adv', 'sed_burial'])
    #     ax.set_ylabel('percentage (%)')
    #     # Get current tick positions
    #     xticks = ax.get_xticks()
    #     # Set the ticks explicitly before setting labels
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(np.round(xticks * time_conversion_factor))
    #     ax.set_xlabel('time (' + time_unit + ')')

    #     plt.savefig(file_out_path+"/"+sim_name+"-degrad"+".png", bbox_inches="tight", dpi=300)
    #     print("save figures to ", file_out_path)

    #     if check_mass is True:
    #         print(f"mass_vol extracted: {mass_volatilized.to_numpy()[-1]}")
    #         vol_active = (sum(self.elements.mass_volatilized)*mass_conversion_factor)
    #         vol_inactive = (sum(self.elements_deactivated.mass_volatilized)*mass_conversion_factor)
    #         print(f"mass_vol active: {vol_active}")
    #         print(f"mass_vol inactive: {vol_inactive}")
    #         print(f"mass_vol active + inactive : {vol_inactive + vol_active}")
    #         print("####")

    #         print(f"mass_degraded extracted {mass_degraded.to_numpy()[-1]}")
    #         degraded_active = (sum(self.elements.mass_degraded)*mass_conversion_factor)
    #         degraded_inactive = (sum(self.elements_deactivated.mass_degraded)*mass_conversion_factor)
    #         print(f"mass_degraded active: {degraded_active}")
    #         print(f"mass_degraded inactive: {degraded_inactive}")
    #         print(f"mass_degraded active + inactive: {degraded_active + degraded_inactive}")
    #         print("####")

    #         print(f"mass_tot_ts extracted: {np.array(mass_tot_timestep[-1]).sum()}")
    #         print("####")
    #         mass_active = (sum(self.elements.mass)*mass_conversion_factor)
    #         mass_inactive = (sum(self.elements_deactivated.mass)*mass_conversion_factor)
    #         print(f"mass_tot_ts active: {mass_active}")
    #         print(f"mass_tot_ts inactive: {mass_inactive}")
    #         print(f"mass_tot_ts active + inactive: {mass_active + mass_inactive}")
    #         print("####")

    #         print(f"mass emitted extracted: {mass_emitted.to_numpy()[-1]}")
    #         print("mass emitted check: ",
    #               f"{(mass_active + mass_inactive)+ (degraded_active + degraded_inactive)+ (vol_inactive + vol_active)}")
    #         print("####")

    @staticmethod
    def _find_date_row(data_frame,
                      specified_date):
        '''
        Select the row of each dataframe nearest to the specified date. If the date is before the first
        date in the dataframe returns a row of 0

        Parameters
        ----------
        data_frame:            pandas dataframe
        specified_date:        Timestamp('%Y-%m-%d %H:%M:%S')
        '''
        import pandas as pd

        date_column = pd.to_datetime(data_frame['date_of_timestep'], format='%Y-%m-%d %H:%M:%S')
        # 'date_of_timestep' column must be ordered increasingly
        if specified_date < date_column.min():
            date_row = pd.DataFrame([[0] * len(data_frame.columns)], columns=data_frame.columns)
        else:
            row_index = (date_column <= specified_date)[::-1].idxmax()
            date_row =data_frame.loc[[row_index]]
        return date_row

    @staticmethod
    def _correct_percentages_ts(merged_df):
        '''
        Recalculate percentages of chemical eliminated by each precess considered

        Parameters
        ----------
        merged_df : pandas dataframe within sum_summary_timeseries function

        '''

        def find_col_name(col_name, data_frame = merged_df):
            return [col for col in data_frame.columns if col.startswith(col_name)]

        col_index_dict = {
            "dissolved": find_col_name(col_name = "dissolved-["),
            "DOC": find_col_name(col_name = "DOC-["),
            "SPM": find_col_name(col_name = "SPM-["),
            "sed_rev": find_col_name(col_name = "sed_rev-["),
            "sed_buried": find_col_name(col_name = "sed_buried-["),
            "mass_wat": find_col_name(col_name = "mass_wat-["),
            "mass_sed_rev": find_col_name(col_name = "mass_sed_rev-["),
            "mass_sed_buried": find_col_name(col_name = "mass_sed_buried-["),
            "mass_current": find_col_name(col_name = "mass_current-["),
            "mass_emitted": find_col_name(col_name = "mass_emitted-["),
            "mass_degr_tot": find_col_name(col_name = "mass_degr_tot-["),
            "mass_degr_wat": find_col_name(col_name = "mass_degr_wat-["),
            "mass_degr_sed": find_col_name(col_name = "mass_degr_sed-["),
            "mass_vol": find_col_name(col_name = "mass_vol-["),
            "adv_out_ts": find_col_name(col_name = "adv_out_ts-["),
            "adv_out_cumul": find_col_name(col_name = "adv_out_cumul-["),
            "perc_vol": find_col_name(col_name = "perc_vol-["),
            "perc_deg_w": find_col_name(col_name = "perc_deg_w-["),
            "perc_deg_s": find_col_name(col_name = "perc_deg_s-["),
            "perc_adv": find_col_name(col_name = "perc_adv-["),
            "perc_sed_buried": find_col_name(col_name = "perc_sed_buried-["),
            "mass_removed": find_col_name(col_name = "mass_removed-["),
            "mass_vol_adv": find_col_name(col_name = "mass_vol_adv-["),
            "mass_degr_wat_adv": find_col_name(col_name = "mass_degr_wat_adv-["),
            "mass_degr_sed_adv": find_col_name(col_name = "mass_degr_sed_adv-[")}

        mass_removed = np.array(merged_df.loc[:, col_index_dict["mass_removed"]])
        mass_volatilized = np.array(merged_df.loc[:, col_index_dict["mass_vol"]])
        mass_volatilized_adv = np.array(merged_df.loc[:, col_index_dict["mass_vol_adv"]])
        mass_degraded_w = np.array(merged_df.loc[:, col_index_dict["mass_degr_wat"]])
        mass_degraded_w_adv = np.array(merged_df.loc[:, col_index_dict["mass_degr_wat_adv"]])
        mass_degraded_s = np.array(merged_df.loc[:, col_index_dict["mass_degr_sed"]])
        mass_degraded_s_adv = np.array(merged_df.loc[:, col_index_dict["mass_degr_sed_adv"]])
        mass_adv_out_cumulative = np.array(merged_df.loc[:, col_index_dict["adv_out_cumul"]])
        mass_sed_buried = np.array(merged_df.loc[:, col_index_dict["mass_sed_buried"]])

        # Contribtion of each mechanism to the elimination of the chemical in the simulation
        perc_volatilized = ((mass_volatilized - mass_volatilized_adv)/mass_removed)*100
        perc_degraded_w = ((mass_degraded_w - mass_degraded_w_adv)/mass_removed)*100
        perc_degraded_s = ((mass_degraded_s - mass_degraded_s_adv)/mass_removed)*100
        perc_adv = ((mass_adv_out_cumulative)/mass_removed)*100
        perc_sed_buried = ((mass_sed_buried)/mass_removed)*100

        merged_df[col_index_dict["perc_vol"]] = perc_volatilized
        merged_df[col_index_dict["perc_deg_w"]] = perc_degraded_w
        merged_df[col_index_dict["perc_deg_s"]] = perc_degraded_s
        merged_df[col_index_dict["perc_adv"]] = perc_adv
        merged_df[col_index_dict["perc_sed_buried"]] = perc_sed_buried
        merged_df = merged_df.fillna(0)
        return merged_df


    def sum_summary_timeseries(self,
                               ts_file_path,
                               sim_name = None,
                               csv_suffix = None,
                               start_date = None,
                               end_date = None,
                               freq_time = None,
                               mass_unit_out = "g"):
        '''
        Sum all .csv files produced by extract_summary_timeseries with the same suffix present in a folder and return a
        .csv file

        Parameters
        ----------
        timeseries_file_path:         string, folder path .csv files produced by extract_summary_timeseries. Must end with /
        sim_name:                     string, name of simulation that will be included in sum .csv file name
        csv_suffix:                   string, suffix of .csv file names that will be summed
        start_date:                   pd.Datetime, start of reconstructed timeseries when .csv have different timesteps
                                      (e.g., ("2018-01-01 00:00:00", format = '%Y-%m-%d %H:%M:%S'))
        end_date:                     pd.Datetime, end of reconstructed timeseries when .csv have different timesteps
        freq_time:                    string, frequency of reconstructed timeseries when .csv have different timesteps expressed as hours (e.g., "6H")
        mass_unit_out:                string, mass unit of uotput .csv file ('kg' 'g','mg','ug')

        Returns a Pandas Dataframe equal to extract_summary_timeseries
        '''

        import pandas as pd
        from datetime import datetime
        import os

        if csv_suffix is None:
            csv_suffix = "_extracted_timeseries.csv"

        # list of timeseries filenames
        ts_file_name_ls = []
        for filename in os.listdir(ts_file_path):
            if filename.endswith(csv_suffix):
                ts_file_name_ls.append(filename)

        if len(ts_file_name_ls) == 0:
            raise ValueError("No .csv files were loaded, check csv_suffix")

        if sim_name is None:
            sim_name = ts_file_name_ls[0][:-3]
        csv_file_name_fin = sim_name + "_extracted_sum_timeseries.csv"

        # list of timeseries Pandas Dataframe imported
        mass_fin_df_ls = []
        for csv_file_name in ts_file_name_ls:
            df = pd.read_csv(ts_file_path + csv_file_name).fillna(0)
            # Correction for naming mistake. to be removed
            if any(name == "mass_degr_sed" for name in df.columns):
                df = df.rename(columns={"mass_degr_sed":"mass_degr_sed_adv-[g]"})
            if any("Unnamed" in column_name for column_name in df.columns):
                df = df.filter(regex='^(?!Unnamed).*$')
            mass_fin_df_ls.append(df)

        check_time_conv_factor = []
        check_mass_conv_factor = []
        check_time_step = []
        for df_mass in mass_fin_df_ls:
            check_time_conv_factor.append(df_mass.loc[0, ('convertion_factors-[time_mass]')])
            check_mass_conv_factor.append(df_mass.loc[1, ('convertion_factors-[time_mass]')])
            time_col_ind = int(np.where(df_mass.columns.str.contains("time-"))[0])
            check_time_step.append(df_mass.iloc[1, time_col_ind] - df_mass.iloc[0, time_col_ind])

        check_time_convertion_factor = all(element == check_time_conv_factor[0] for element in check_time_conv_factor)
        check_mass_convertion_factor = all(element == check_mass_conv_factor[0] for element in check_mass_conv_factor)
        check_time_step = all(element == check_time_step[0] for element in check_time_step)

        if check_time_step is True:
            if (check_time_convertion_factor and check_mass_convertion_factor) is False:
                raise ValueError("time/mass_conversion_factor is not equal for all extracted timeseries")
            time_conv_factor = mass_fin_df_ls[0].loc[0, ('convertion_factors-[time_mass]')]
            mass_conv_factor = mass_fin_df_ls[0].loc[1, ('convertion_factors-[time_mass]')]

            # Merge pd.Dataframes and sort by time
            merged_df = pd.concat(mass_fin_df_ls, axis = 0)
            merged_df.reset_index(drop = True, inplace = True)
            merged_df["date_of_timestep"] = pd.to_datetime(merged_df["date_of_timestep"])
            merged_df = merged_df.sort_values(by = ["date_of_timestep"])
            # Sum rows of the same timestep
            merged_df = pd.DataFrame(merged_df.groupby("date_of_timestep").sum())
            merged_df.insert(0,('date_of_timestep'), merged_df.index)
            merged_df["date_of_timestep"] = merged_df.index

            # Reconstruct ["date_of_timestep"] column
            time_start = (pd.to_datetime(merged_df["date_of_timestep"][0])).to_pydatetime()
            time_delta_ls = []
            for time_ind in range(1, len(merged_df["date_of_timestep"])):
                time_delta_ls.append(merged_df["date_of_timestep"][time_ind] - time_start)

            list_dates_fin = []
            list_dates_fin.append(time_start)
            for time_ind in time_delta_ls:
                list_dates_fin.append((time_start) + time_ind)
            merged_df["date_of_timestep"] = (list_dates_fin)

            # Reconstruct ["time-[]"] column
            new_time  = time_conv_factor * (range(0, len(merged_df["date_of_timestep"])))
            time_col = [col for col in merged_df.columns if "time-[" in col]
            merged_df.loc[:, time_col] = new_time
            merged_df.reset_index(drop = True, inplace = True)

            # Correct time/mass convertion factors
            merged_df.loc[0, ('convertion_factors-[time_mass]')] = time_conv_factor
            merged_df.loc[1, ('convertion_factors-[time_mass]')] = mass_conv_factor
            # Recalculate percentages
            merged_df_fin = self._correct_percentages_ts(merged_df)

        elif ((check_time_step is False) and (start_date is not None) and\
            (end_date is not None) and (freq_time is not None)):

            # Set mass_conv_factor to NaN if df have different mass units
            if all (element == check_mass_conv_factor[0] for element in check_mass_conv_factor):
                mass_conv_factor = check_mass_conv_factor[0]
            else:
                mass_conv_factor = np.nan

            # final row for each timestep that will be concatenated
            ts_sum_ls = []
            time_date_serie = pd.date_range(start=start_date, end = end_date, freq=freq_time)

            for df_index in range(0, len(mass_fin_df_ls)):
                df_mass = mass_fin_df_ls[df_index]

                # Check format of ['date_of_timestep']
                date_of_timestep_ls = []
                for element in df_mass['date_of_timestep']:
                    date_of_timestep_ls.append(self._change_datetime_format(datetime_str = element,
                                                new_format = "%Y-%m-%d %H:%M:%S"))
                df_mass['date_of_timestep'] = date_of_timestep_ls

                # Remove mass degraded or removed before start_date
                df_mass_date_column = pd.to_datetime(df_mass['date_of_timestep'], format='%Y-%m-%d %H:%M:%S')
                # 'date_of_timestep' column must be ordered increasingly
                df_mass_date_column = df_mass_date_column.sort_values(ascending=True)
                if start_date < df_mass_date_column.min():
                    start_date_index = df_mass.index[0]
                else:
                    start_date_index = (df_mass_date_column <= start_date)[::-1].idxmax()

                filtered_df = df_mass[df_mass.index >= start_date_index]
                # Select mass to remove
                first_ts_to_remove = pd.DataFrame(filtered_df.iloc[0]).T
                mass_col_ind = int(np.where(filtered_df.columns.str.contains("dissolved-"))[0])
                mass_unit =  filtered_df.columns[mass_col_ind][11:-1]

                degr_columns_names_ls = ['mass_sed_buried'+ "-["+ mass_unit +"]", 'mass_degr_tot'+ "-["+ mass_unit +"]",
                'mass_degr_wat'+ "-["+ mass_unit +"]", 'mass_degr_sed'+ "-["+ mass_unit +"]", 'mass_vol'+ "-["+ mass_unit +"]",
                'adv_out_ts'+"-["+ mass_unit +"]", 'adv_out_cumul'+"-["+ mass_unit +"]", 'mass_removed'+ "-["+ mass_unit +"]",
                'mass_vol_adv'+ "-["+ mass_unit +"]", 'mass_degr_wat_adv'+ "-["+ mass_unit +"]",'mass_degr_sed_adv'+ "-["+ mass_unit +"]"]

                for degr_columns_name in degr_columns_names_ls:
                    filtered_df[degr_columns_name] = np.array(filtered_df[degr_columns_name]) - np.array(first_ts_to_remove[degr_columns_name])

                mass_emitted_delta = (filtered_df["mass_emitted"+ "-["+ mass_unit +"]"].iloc[0] -
                                      filtered_df["mass_current" + "-["+ mass_unit +"]"].iloc[0])
                filtered_df["mass_emitted"+ "-["+ mass_unit +"]"] = np.array(filtered_df["mass_emitted"+ "-["+ mass_unit +"]"]) - mass_emitted_delta

                time_col_ind = int(np.where(filtered_df.columns.str.contains("time-"))[0])
                time_unit = filtered_df.columns[time_col_ind][6:-1]
                time_delta = (filtered_df["time"+ "-["+ time_unit +"]"].iloc[0])
                filtered_df["time"+ "-["+ time_unit +"]"] = np.array(filtered_df["time"+ "-["+ time_unit +"]"]) - time_delta

                if mass_unit_out != mass_unit:
                    print(f"converted mass from {mass_unit} to {mass_unit_out}")
                    if mass_unit_out == "ug":
                        if mass_unit=='kg':
                            mass_conversion_factor= 1e-9
                        if mass_unit=='g':
                            mass_conversion_factor= 1e-6
                        if mass_unit=='mg':
                            mass_conversion_factor= 1e-3
                        if mass_unit=='ug':
                            mass_conversion_factor= 1
                    elif mass_unit_out == "mg":
                        if mass_unit=='kg':
                            mass_conversion_factor= 1e-6
                        if mass_unit=='g':
                            mass_conversion_factor= 1e3
                        if mass_unit=='mg':
                            mass_conversion_factor= 1
                        if mass_unit=='ug':
                            mass_conversion_factor= 1e-3
                    elif mass_unit_out == "g":
                        if mass_unit=='kg':
                            mass_conversion_factor= 1e3
                        if mass_unit=='g':
                            mass_conversion_factor= 1
                        if mass_unit=='mg':
                            mass_conversion_factor=1e-3
                        if mass_unit=='ug':
                            mass_conversion_factor= 1e-6
                    elif mass_unit_out == "kg":
                        if mass_unit=='kg':
                            mass_conversion_factor= 1
                        if mass_unit=='g':
                            mass_conversion_factor= 1e-3
                        if mass_unit=='mg':
                            mass_conversion_factor=1e-6
                        if mass_unit=='ug':
                            mass_conversion_factor= 1e-9
                    else:
                        raise ValueError("wrong mass_unit_out specified")

                    # Update column names with mass_unit_out
                    new_column_names = list(filtered_df.columns)
                    for i in range(len(new_column_names)):
                       new_column_names[i] = new_column_names[i].replace(("-["+ mass_unit +"]"), "-["+ mass_unit_out +"]")
                    filtered_df.columns =  new_column_names

                    for col_name in list(filtered_df.columns):
                        if col_name.endswith ("-["+ mass_unit_out +"]"):
                            # Multiply mass columns by mass_conversion_factor
                            filtered_df[col_name] = np.array(filtered_df[col_name]) * mass_conversion_factor
                        else:
                            pass
                else:
                    pass

                # Update df in mass_fin_df_ls
                # print(df_index)
                mass_fin_df_ls[df_index] = filtered_df

            for time_step in time_date_serie:
                # rows for each timestep
                mass_row_ls = []
                for df_mass in mass_fin_df_ls:
                    # df_mass = mass_fin_df_ls[2]
                    date_row = self._find_date_row(data_frame=df_mass,
                                                specified_date = pd.to_datetime(time_step))
                    mass_row_ls.append(date_row)
                # Concatenate the row of each df and sum
                concat_df = pd.concat(mass_row_ls)
                concat_df['date_of_timestep'] = 0
                concat_df['convertion_factors-[time_mass]'] = np.nan
                concat_df_sum = concat_df.sum(axis=0)
                concat_df_sum = pd.DataFrame(np.array(concat_df_sum)).T
                concat_df_sum.columns = concat_df.columns
                concat_df_sum.loc[0, 'date_of_timestep'] = time_step

                ts_sum_ls.append(concat_df_sum)

            # Concatenate all timesteps
            mass_df_fin = pd.concat(ts_sum_ls)
            # Correct time-[] column
            time_col_ind = int(np.where(mass_df_fin.columns.str.contains("time-"))[0])
            time_unit = mass_df_fin.columns[time_col_ind][6:-1]

            if time_unit=='seconds':
                time_conversion_factor_ls = [(60*60*24), (1)]
                time_conv_factor = pd.Timedelta(freq_time).seconds
            if time_unit=='minutes':
                time_conversion_factor_ls = [(60*24), (60)]
                time_conv_factor = pd.Timedelta(freq_time).seconds / (60)
            if time_unit=='hours':
                time_conversion_factor_ls = [(24), (60*60)]
                time_conv_factor = pd.Timedelta(freq_time).seconds / (60*60)
            if time_unit=='days':
                time_conversion_factor_ls = [(1), (60*60*24)]
                time_conv_factor = pd.Timedelta(freq_time).seconds / (60*60*24)

            time_series = mass_df_fin['date_of_timestep']
            time_series_diff = np.array(time_series) - np.array(time_series.iloc[0])

            time_unit_col_ls = []
            for time_d in time_series_diff:
                n_days = time_d.days*time_conversion_factor_ls[0]
                n_seconds = time_d.seconds/time_conversion_factor_ls[1]
                time_unit_col_ls.append(n_days + n_seconds)

            time_unit_col = np.array(time_unit_col_ls)

            mass_df_fin.iloc[:, time_col_ind] = time_unit_col
            # Recalculate percentages
            merged_df_fin = self._correct_percentages_ts(mass_df_fin)
            merged_df_fin.loc[0, ('convertion_factors-[time_mass]')] = time_conv_factor
            merged_df_fin.loc[1, ('convertion_factors-[time_mass]')] = mass_conv_factor
        else:
            raise ValueError("start_date, end_date, and freq_time must be specified")

        merged_df_fin.to_csv(ts_file_path + csv_file_name_fin)
        print(f"saved sum file to {ts_file_path + csv_file_name_fin}")


    @staticmethod
    def _check_dims_values(dims_values):
        '''
        Check if each dimention within DataArrays_ls has the same values for all DataArrays in the list

        dims_values:        list of dict, [{dimention name:dimention values}, ...]
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
                raise ValueError(f' Dimention "{key}" has different values across DataArray_ls')
            else:
                pass

    ### Helpers for sum_DataArray_list
    @staticmethod
    def _reindex_da(DataArray,
                  time_name, target_time,
                  nearest_tol_time= None,
                  align_mode = "pad",
                  clip_to_original=True):
        """
        Reindex xr.DataArray along "time_name" dimention using "target_time" array

        DataArray:        xarray DataArray
        time_name:        string, name of time dimention of all DataArray present in DataArray_ls
        target_time:      np.array of np.datetime64[ns]
        nearest_tol_time: np.timedelta64, max distance for "nearest" (e.g., np.timedelta64(3,'h'))
        align_mode:       string, mode of selecting timestamp in reconstructed sum ("pad"|"nearest"|"exact")
        clip_to_original: bool, select if contribution of DataArray before the start and after the end of that array is set to 0

        """
        # sort for methods that need monotonic time
        if align_mode in {"pad", "nearest"} and not DataArray[time_name].to_index().is_monotonic_increasing:
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
            t_min  = t_orig.min()
            t_max  = t_orig.max()
            valid  = (out[time_name] >= t_min) & (out[time_name] <= t_max)
            out    = out.where(valid)

        return out


    @staticmethod
    def _infer_decimals_from_res(res,
                              extra = 1):
        """
        Infer decimals from the resolution: number of visible decimal places in |res| + `extra`.
        Examples: 0.005 -> 4, 0.001 -> 4, 0.2 -> 2, 0.25 -> 3
        """
        r = abs(float(res))
        s = f"{r:.12f}".rstrip("0").rstrip(".")
        d = len(s.split(".")[1]) if "." in s else 0
        return max(0, d + extra)


    def build_regular_series(self, *,
                          first_value,
                          res,
                          length,
                          direction,  # "asc" | "desc" | None -> use sign of res
                          decimals) -> np.ndarray:
        """
        Round the first_value to the desired decimals (derived from |res| if None),
        then create: first_rounded + step * i  for i = 0..length-1,
        where step = +|res| for ascending, -|res| for descending.
        """
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
    def _canonical_nontime_dims(DataArray_ls, nontime_dims,
                            dim_res_dict=None, coord_tolerance_dict=None,
                            snap_mode=None):  # "ref" recommended
        """
        If coord_tolerance is None -> strict equality (np.array_equal).
        If coord_tolerance is float -> allow per-index |diff| <= tolerance across arrays

        Return canonical 1-D coord arrays for each dim in nontime_dims.
        Does *not* relabel inputs; it only computes the canonical vectors.
        """
        import numpy as np

        def _is_regular(vec, tol):
            """
            Check if a 1D coordinate vector is regularly spaced within tolerance.
            """
            v = np.asarray(vec, dtype=np.float64)
            if v.size < 2:
                return True, 0.0
            diffs = np.diff(v)
            step = np.median(np.abs(diffs))
            err  = np.max(np.abs(diffs - np.sign(diffs.mean() if diffs.size else 1.0)*step))
            return (err <= (tol if tol is not None else 1e-12)), step

        def _min_decimals(x, max_dec=12):
            """
            Return minimum number of decimals to represent x as a clean decimal within tolerance.
            """
            x = float(abs(x));
            for d in range(max_dec+1):
                if np.isclose(x*(10**d), round(x*(10**d)), atol=1e-12):
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
                    raise ValueError(f'Dimension "{dim}" length differs: {da.sizes[dim]} vs {ref.size} for array {idx}')

            tol = (coord_tolerance_dict or {}).get(dim) if coord_tolerance_dict is not None else None

            # tolerance check
            identical_coord = []
            if tol is not None:
                ok = True
                for da in DataArray_ls[1:]:
                    vals = np.asarray(da[dim].values)
                    diff = np.abs(vals - ref)
                    identical_coord.append(diff==0)
                    if np.nanmax(diff) > tol:
                        ok = False; break
                if not ok:
                    raise ValueError(f'Dimension "{dim}" values differ by > tolerance ({tol}).')
            else:
                # strict
                for idx, da in enumerate(DataArray_ls[1:], start=1):
                    equal = np.array_equal(np.asarray(da[dim].values), ref)
                    if not equal:
                        raise ValueError(f'Dimension "{dim}" values differ between arrays')
                    else:
                        identical_coord.append(equal)

            if np.all(np.array(identical_coord)):
                # dim is identical across all DataArray_ls
                print(f"Identical dim '{dim}' among DataArray_ls: no interpolation needed")
                canonical[dim] = ref
                need_interpolation[dim] = False
                continue
            else:
                need_interpolation[dim] = True

            # resolution
            dim_res = (dim_res_dict or {}).get(dim) if dim_res_dict is not None else None
            if dim_res is None:
                regular, dim_res = _is_regular(ref, tol)
                dim_res = float(np.round(dim_res, _min_decimals(dim_res)))
                print(f"dim_res: {dim_res} inferred for dim: {dim}")
                if not regular:
                    raise ValueError(f'dim_res for "{dim}" not specified or inferred.')

            # build canonical
            if snap_mode == "ref":
                can = ref.astype(np.float64)
            else:  # "median"
                stack = np.stack([np.asarray(da[dim].values, dtype=np.float64) for da in DataArray_ls], axis=0)
                can0 = np.nanmedian(stack, axis=0)
                inc  = (can0[-1] >= can0[0])
                first = can0[0]
                step  = abs(dim_res)
                series = first + (np.arange(can0.size) * (step if inc else -step))
                can = np.round(series, decimals=_min_decimals(step))
                print(f"dim '{dim}' will be interpolated within tolerance ({tol}): {can0[0]} -> {can[0]}")

            # normalize longitude system to match first array’s convention if present
            if dim.lower() in ("lon","longitude"):
                lon = can
                if lon.min() < -10 and lon.max() <= 180:
                    # already [-180,180]; leave
                    pass
                elif lon.min() >= 0 and lon.max() > 180:
                    # [0,360]; leave
                    pass
                else:
                    # snap to either system based on first array
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

        Returns
        -------
        DataArray_ls : list[xr.DataArray]
            The list of DataArrays extracted or passed in.
        index_keys : list or None
            If DataArray_dict was provided, list index → dict key mapping.
            Otherwise None.
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

        Rules:
          - If no DataArray has 'depth': leave depth untouched (i.e. absent), just
            drop 'specie' dim/coord where present.
          - If some DataArrays have 'depth':
              * All 'depth' coordinates among those must have the same length and values
                (scalar vs [0.] is treated as equal).
              * If depth is single-valued (length 1), every DataArray is converted to have
                a 'depth' dimension of length 1 with that value and matching coord.
              * If depth is multi-valued (length > 1), every DataArray must already have
                'depth' present, and all must match; they are left multi-depth but their
                depth coord is set to the common reference.
          - In all cases, remove any 'specie' dimension and/or coordinate.
        """
        import numpy as np
        import xarray as xr
        depth_ref = None
        depth_ref_idx = None
        has_depth_flags = []

        # Inspect & verify depth where present
        for i, da in enumerate(DataArray_ls):
            has_depth = ("depth" in da.coords) or ("depth" in da.dims)
            has_depth_flags.append(has_depth)

            if not has_depth:
                continue

            # Get depth values irrespective of dim vs coord
            if "depth" in da.coords:
                depth_vals = da.coords["depth"].values
            else:
                depth_vals = da["depth"].values

            depth_arr_norm = np.atleast_1d(np.asarray(depth_vals))

            if depth_ref is None:
                depth_ref = depth_arr_norm
                depth_ref_idx = i
            else:
                depth_ref_norm = np.atleast_1d(np.asarray(depth_ref))

                # Require same shape and values for all with depth
                if depth_arr_norm.shape != depth_ref_norm.shape or not np.allclose(
                    depth_arr_norm, depth_ref_norm
                ):
                    raise ValueError(
                        f"Inconsistent depth coordinates between inputs {i} and {depth_ref_idx}:\n"
                        f"  depth[{i}] = {depth_arr_norm}\n"
                        f"  depth[{depth_ref_idx}] = {depth_ref_norm}\n"
                        "All depth coordinate values must be identical."
                    )

        # No depth anywhere
        if depth_ref is None:
            # Just drop 'specie' and return
            normalized = []
            for da in DataArray_ls:
                aa = da
                # Drop specie dimension if present
                if "specie" in aa.dims:
                    aa = aa.squeeze("specie", drop=True)
                # Drop specie coord if present
                if "specie" in aa.coords:
                    aa = aa.drop_vars("specie")
                normalized.append(aa)
            return normalized

        # There is a common depth
        depth_ref_norm = np.atleast_1d(np.asarray(depth_ref))
        n_depth = depth_ref_norm.size
        any_missing_depth = not all(has_depth_flags)

        # If multi-depth and some arrays have no depth at all, raise error.
        if n_depth > 1 and any_missing_depth:
            raise ValueError(
                "Some DataArrays have multi-layer 'depth' while others have no 'depth'. "
                "Cannot safely normalize in this case."
            )

        normalized = []

        for i, da in enumerate(DataArray_ls):
            aa = da

            # Normalize depth to be a dim + coord with the common values
            if n_depth == 1:
                depth_value = float(depth_ref_norm[0])

                if "depth" in aa.dims:
                    # Ensure size==1
                    if aa.sizes["depth"] != 1:
                        raise ValueError(
                            f"DataArray index {i} has 'depth' dim of size {aa.sizes['depth']} "
                            "but reference depth is single-valued."
                        )
                    # Enforce coord
                    aa = aa.assign_coords(depth=[depth_value])
                    print(f"Normalized depth coord for array {i} → [{depth_value}]")
                else:
                    # Remove any scalar coord if present
                    if "depth" in aa.coords:
                        aa = aa.drop_vars("depth")
                    # Add depth as a dimension with single coordinate
                    aa = aa.expand_dims(dim={"depth": [depth_value]})
                    print(f"Added depth=[{depth_value}] as dimension to array {i}")

            else:
                # Multi-depth: all must already have depth, and values were alredy checked
                if "depth" not in aa.dims:
                    raise ValueError(
                        f"DataArray index {i} has no 'depth' dimension, but reference depth "
                        "is multi-layer."
                    )
                # Enforce identical depth coord
                aa = aa.assign_coords(depth=depth_ref_norm)
                print(f"Normalized depth coord for array {i} → {depth_ref_norm.tolist()}")

            # -Drop 'specie' dim & coord in all cases
            if "specie" in aa.dims:
                aa = aa.squeeze("specie", drop=True)
                print(f"Removed 'specie' dimension from array {i}")
            if "specie" in aa.coords:
                aa = aa.drop_vars("specie")
                print(f"Removed 'specie' coordinate from array {i}")

            normalized.append(aa)

        return normalized


    def sum_DataArray_list(self,
                         DataArray_dict,
                         DataArray_ls = None,
                         variable = None,
                         start_date = None,
                         end_date = None,
                         freq_time = None,
                         time_name = "time",
                         align_mode ="pad",
                         nearest_tol_time = None,
                         dim_res_dict = None,
                         coord_tolerance_dict = {},
                         snap_mode ="median",
                         mask_mode = "input0",
                         sim_description = None):
        '''
        Sum a list of xarray DataArrays, with the same or different time step
        If start_date, end_date, or freq_time are specified time dimention is reconstructed

        DataArray_dict:      dictionary of {idx: {"topo": ..., "variable": ...}, ...}
        variable:            string, name of variable to be used as key for DataArray_dict
        DataArray_ls:        list of xarray DataArray
        start_date:          np.datetime64, start of reconstructed time dimention
        end_date:            np.datetime64, start of reconstructed time dimention
        freq_time:           np.timedelta64, frequency of reconstructed time dimention
        time_name:           string, name of time dimention of all DataArray present in DataArray_ls
        align_mode:          string, mode of selecting timestamp in reconstructed sum ("pad"|"nearest"|"exact")
        nearest_tol_time:    np.timedelta64, max distance for "nearest" (e.g., np.timedelta64(3,'h'))
        dim_res_dict:        dict {"dim" : float32} resolution of each dimention. Will be inferred if None or {}
        mask_mode:           string, mode of masking finla sum ("input0"|"union"|"intersection")
        sim_description:     string, descrition of simulation to be included in netcdf attributes
        '''
        import xarray as xr
        from datetime import datetime
        import numpy as np

        ### Convert time_name to string to allow indexing
        if not isinstance(time_name, str):
            time_name = str(time_name)
        if DataArray_dict is None and DataArray_ls is None:
            raise ValueError("Either DataArray_dict or DataArray_ls must be provided.")

        ### Build DataArray_ls from DataArray_dict if needed
        DataArray_ls, index_keys = self._normalize_inputs(DataArray_dict, DataArray_ls, variable)
        ### Ensure all DataArrays have consistent 'depth' semantics and Drop 'specie' dim/coord whenever it appears (size 1 or scalar).
        DataArray_ls = self._normalize_depth_and_specie(DataArray_ls)

        ### NaN counts  on cleaned arrays
        if DataArray_dict is not None:
            nan_counts_inputs = {
                idx: int(inner[variable].isnull().sum().item())
                for idx, inner in DataArray_dict.items()
                if variable in inner
            }
        else:
            nan_counts_inputs = {
                i: int(da.isnull().sum().item())
                for i, da in enumerate(DataArray_ls)
            }

        if len(DataArray_ls) < 1:
            raise ValueError("Empty DataArray_ls")
        if len(DataArray_ls) == 1:
            print("len(DataArray_ls) is 1, returning DataArray_ls[0]")
            return DataArray_ls[0]

        ### Check that input DataArrays have the same dimensions
        print("Checking input DataArray dimensions")
        all_dims_per_da = [set(da.dims) for da in DataArray_ls]
        common_dims = set.intersection(*all_dims_per_da)
        extra_dims = set.union(*all_dims_per_da) - common_dims
        if extra_dims:
            for dim in sorted(extra_dims):
                for idx, da in enumerate(DataArray_ls):
                    if dim in da.dims:
                        print(f'Extra dimension "{dim}" in array {idx}')
            raise ValueError("Uncommon dimensions are present in DataArray_ls")

        ### Check that all DataArrays have the same variable name
        names = [da.name for da in DataArray_ls]  # may contain None
        if any(n is None for n in names):
            raise ValueError(f"All DataArrays must have a name. Got: {names}")
        ref_name = next((n for n in names if n is not None), "data")

        ### Fail if any non-None name differs from ref_name
        mismatch = [n for n in names if (n is not None and n != ref_name)]
        if mismatch:
            raise ValueError(f"Mismatched DataArray names: {names}. Expected all '{ref_name}'.")
        print(f"var_name: {ref_name}")

        ### non-time dims must have identical coordinates
        nontime_dims = set.union(*all_dims_per_da)
        if time_name in nontime_dims:
            nontime_dims.remove(time_name)

        sum_conc_DataArray_ls_input = float(sum(da.fillna(0).sum().item() for da in DataArray_ls))

        ### check equality of dimension within tolerance
        canonical_dims_dict, need_interpolation = self._canonical_nontime_dims(
            DataArray_ls = DataArray_ls,
            nontime_dims = nontime_dims,
            coord_tolerance_dict=coord_tolerance_dict,
            snap_mode=snap_mode,
            dim_res_dict=dim_res_dict)

        ### Find common attributes to be added in Final_sum
        common_attrs = DataArray_ls[0].attrs.copy()
        for da in DataArray_ls[1:]:
            common_attrs = {key: value for key, value in common_attrs.items() if da.attrs.get(key) == value}

        H_can_out = None
        nan_counts_interp = None

        if not any(need_interpolation.values()):
            # Remove heavy DataArrays to free memory safely when interpolation is not needed
            if DataArray_dict is not None:
                DataArray_dict.clear()
            DataArray_dict = None

        if any(need_interpolation.values()):
            if DataArray_dict is None:
                raise ValueError("DataArray_dict {idx{'var','topo'}, ...} must be specified")

            import xesmf as xe
            import hashlib
            import tempfile
            import os

            if "latitude" in canonical_dims_dict.keys():
                lat_can = canonical_dims_dict["latitude"]
            elif "lat" in canonical_dims_dict.keys():
                lat_can = canonical_dims_dict["lat"]
            else:
                raise ValueError("No latitude/lat dimention present")

            if "longitude" in canonical_dims_dict.keys():
                lon_can = canonical_dims_dict["longitude"]
            elif "lon" in canonical_dims_dict.keys():
                lon_can = canonical_dims_dict["lon"]
            elif "long" in canonical_dims_dict.keys():
                lon_can = canonical_dims_dict["long"]
            else:
                raise ValueError("No longitude/lon/long dimention present")

            interp_keys = ("latitude", "lat", "longitude", "lon", "long")
            need_horizontal_interp = any(bool(need_interpolation.get(k)) for k in interp_keys)

            depth_top_can = canonical_dims_dict["depth"] if "depth" in canonical_dims_dict.keys() else None
            if depth_top_can is not None:
                need_vertical_interp = any(bool(need_interpolation.get(k)) for k in ["depth"])
            else:
                need_vertical_interp = False

            ### Helpers for "regrid_topography_conservative" ###
            def _maybe_depth_dim(da, candidates=("depth","z","lev","level")):
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
                    # fallback tiny cell
                    half = 0.5 * (c[0] if c[0] != 0 else 1.0)
                    return np.array([c[0]-half, c[0]+half], dtype=np.float64)
                mid = 0.5*(c[:-1] + c[1:])
                first = c[0]  - (mid[0] - c[0])
                last  = c[-1] + (c[-1] - mid[-1])
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
                        lon   = (["lon"],   lon),
                        lat   = (["lat"],   lat),
                        lon_b = (["lon_b"], lon_b),
                        lat_b = (["lat_b"], lat_b),
                    )
                )

            def get_regridder(src_lon, src_lat, dst_lon, dst_lat, *, method="conservative",
                              periodic=True, weights_dir=None, prefix="weights"):
                """
                Create (or reuse cached) xESMF Regridder for given src/dst grids and method.
                Optionally cache weights on disk under weights_dir.
                """
                src_grid = _make_rect_grid(src_lon, src_lat)
                dst_grid = _make_rect_grid(dst_lon, dst_lat)
                cache_id = _hash_grid(src_lon, src_lat, dst_lon, dst_lat)
                weights_path = None
                reuse = False
                if weights_dir:
                    os.makedirs(weights_dir, exist_ok=True)
                    weights_path = os.path.join(weights_dir, f"{prefix}_{method}_{cache_id}.nc")
                    reuse = os.path.exists(weights_path)
                R = xe.Regridder(src_grid, dst_grid, method=method,
                                 periodic=periodic,
                                 filename=weights_path,
                                 reuse_weights=reuse)
                # If we just created weights in-memory and have a path, persist them
                if (weights_path is not None) and (not reuse):
                    R.to_netcdf(weights_path)
                return R, weights_path

            def regrid_topography_conservative(
                H_src: xr.DataArray,
                lat_out,
                lon_out,
                *,
                periodic=False,
                unmapped_to_nan=True,
                weights_dir=None,
                weights_path=None,
            ):
                """
                Conservative remap of topography to (lat_out, lon_out).
                H_src must have dims named ('latitude','longitude') (any order). Extra dims broadcast.
                - Caches weights under weights_dir (default: system temp) unless weights_path is given.
                - Returns DataArray on the same non-horizontal dims as H_src, but with
                  coords named ('latitude','longitude') on output.
                """
                import os
                # ensure output coords are 1D numpy arrays
                lat_out = np.asarray(lat_out, dtype=np.float64)
                lon_out = np.asarray(lon_out, dtype=np.float64)

                # move horizontal dims to the end and standardize names to ('lat','lon')
                assert ("latitude" in H_src.dims) and ("longitude" in H_src.dims), \
                    "H_src must have dims 'latitude' and 'longitude'"
                other = [d for d in H_src.dims if d not in ("latitude","longitude")]
                H_ord = H_src.transpose(*other, "latitude", "longitude", missing_dims="ignore")

                # choose/cache weights path

                topo_R, topo_wpath = get_regridder(
                src_lon=H_src["longitude"].values,
                src_lat=H_src["latitude"].values,
                dst_lon=lon_out,
                dst_lat=lat_out,
                method="conservative", periodic=True,
                weights_dir=weights_dir, prefix="topo_conservative"
                )

                H_regridded = topo_R(H_ord.rename({"latitude":"lat","longitude":"lon"})).rename({"lat":"latitude","lon":"longitude"})

                H_back = H_regridded.transpose(*other, "latitude", "longitude", missing_dims="ignore")
                return H_back.assign_coords(latitude=("latitude", lat_out),
                                            longitude=("longitude", lon_out))

            # -------- geometry helpers --------
            def normalize_depth_tops(zt1d, *, ensure_zero=True, tol=1e-12):
                """
                Normalize layer TOPS to positive-down, strictly increasing depths.

                Accepts inputs like:
                  - positive-down: [0, 1, 50, 100]
                  - negative-up:   [-2, -1, 0]  -> [0, 1, 2]
                  - mixed/unsorted/duplicates/noisy floats

                Steps:
                  1) Flip sign for negatives (positive-down convention).
                  2) Insert 0 if missing (optional).
                  3) Sort ascending.
                  4) Remove near-duplicates using tolerance.
                  5) Return strictly increasing array (diff > 0).
                """
                z = np.asarray(zt1d, dtype=np.float64).ravel()
                if z.size == 0:
                    raise ValueError("depth_top array is empty")

                # finite only
                z = z[np.isfinite(z)]

                if z.size == 0:
                    raise ValueError("depth_top array has no finite values")

                # 1) flip sign for negatives (don't shift)
                z = np.where(z < 0, -z, z)

                # 2) ensure surface 0 present if requested
                if ensure_zero and not np.any(np.isclose(z, 0.0, atol=tol)):
                    z = np.concatenate(([0.0], z))

                # 3) sort ascending
                z = np.sort(z)

                # 4) dedupe with tolerance -> strictly increasing afterwards
                if z.size > 1:
                    keep = np.empty_like(z, dtype=bool)
                    keep[0] = True
                    for i in range(1, z.size):
                        keep[i] = (z[i] - z[np.where(keep)[0][-1]]) > tol
                    z = z[keep]

                # Final sanity
                if z.size == 0 or z[0] < -tol:
                    raise ValueError("Normalization failed to produce non-negative tops.")
                if z.size > 1 and not np.all(np.diff(z) > 0):
                    raise ValueError("Normalized tops are not strictly increasing.")

                return z


            def _safe_assign_depth(da, depth_dim, coord_vals):
                """
                Assign depth coordinates to a DataArray, enforcing size consistency
                between data along depth_dim and coord_vals.
                """
                coord_vals = np.asarray(coord_vals)
                if da.sizes[depth_dim] != coord_vals.size:
                    raise ValueError(
                        f"conflicting sizes for '{depth_dim}': data={da.sizes[depth_dim]} vs coord={coord_vals.size}"
                    )
                return da.assign_coords({depth_dim: (depth_dim, coord_vals)})


            def build_target_edges_from_tops(
                depth_tops_1d,
                H_can,
                *,
                depth_dim: str = "depth",
                edge_dim: str = "depth_edge",
                tol: float = 1e-12
            ):
                """

                Build target vertical interfaces and layer thicknesses from 1D tops and canonical bottom.

                depth_tops_1d : 1-D array of layer TOPS (positive down; will be cleaned/sorted by normalize_depth_tops)
                H_can         : 2-D bottom depth (lat,lon), positive down

                Returns
                -------
                z_edges : (edge, lat, lon) float64
                    Target vertical interfaces; tops clamped to local bathymetry, last edge == H_can.
                dz      : (depth, lat, lon) float64
                    Target layer thicknesses (non-negative). Depth coords set to the (clean) target tops.
                """
                # Normalize user-provided tops (assumed available in your codebase)
                zt = normalize_depth_tops(depth_tops_1d)   # 1-D, positive-down, sorted, unique, includes 0
                if zt.size > 1 and not np.all(np.diff(zt) > -tol):
                    raise ValueError("Target tops are not strictly increasing after normalization.")
                nz = int(zt.size)

                # 1) Make 1-D DA for tops and broadcast over H_can
                zt1d = xr.DataArray(zt, dims=(depth_dim,), coords={depth_dim: np.arange(nz)})
                zt_b = zt1d.broadcast_like(H_can).transpose(depth_dim, *H_can.dims)  # (depth, lat, lon)

                # 2) Prepare bathymetry as float64 (clip tiny negatives just in case)
                H = xr.where(H_can < 0, 0.0, H_can.astype(np.float64))

                # 3) Clamp any top that is deeper than local bathymetry
                zt_b = xr.where(zt_b > H, H, zt_b)

                # 4) Enforce non-decreasing tops with depth (fix sequences like [0, 5, 3] -> [0, 5, 5])
                zt_b = xr.apply_ufunc(
                    np.maximum.accumulate,
                    zt_b,
                    input_core_dims=[[depth_dim]],
                    output_core_dims=[[depth_dim]],
                    vectorize=True,
                    dask="parallelized",
                )

                # 5) Append exact bottom as last edge and rename depth->edge
                bottom = H.expand_dims({depth_dim: [nz]})  # (1, lat, lon)
                z_edges = xr.concat([zt_b, bottom], dim=depth_dim).rename({depth_dim: edge_dim}).astype(np.float64)

                # Monotonicity check on edges: diffs must be >= -tol
                diffs = z_edges.diff(edge_dim)
                if (diffs < -tol).any():
                    # report how many columns are problematic
                    bad_cols = (diffs < -tol).any(dim=edge_dim)
                    nbad = int(bad_cols.sum().item())
                    raise ValueError(f"Target edges are not monotone non-decreasing in {nbad} column(s).")

                # 6) Thickness (non-negative), stamp nice depth coords (the target tops)
                dz = z_edges.diff(edge_dim).rename({edge_dim: depth_dim})
                dz = dz.clip(min=0.0)
                dz = _safe_assign_depth(dz, depth_dim, zt)  # keeps depth coords equal to the cleaned tops

                return z_edges, dz

            def build_source_edges_on_canonical(
                depth_top_src_1d,
                H_can,
                *,
                depth_dim: str = "depth",
                edge_dim: str = "depth_edge",
            ):
                """
                Build source column edges (edge, lat, lon) from 1-D layer tops and bathymetry.

                Normalization:
                  - Negative inputs are sign-flipped with positive-down convention
                  (0 = surface) (e.g., [-2, -1, 0] -> [0, 1, 2])
                  - Ensure 0 is present (insert if missing)
                  - Remove NaN/inf, sort ascending, de-duplicate

                Post-processing:
                  - Clamp all tops to local bottom H_can
                  - Enforce non-decreasing interfaces downward (cumulative max)
                  - Append exact bottom as final edge
                  - Keep zero-thickness layers (size along edge = nz+1)
                """
                def _is_monotone_nondec_1d(a, tol=0.0):
                    """Return True if 1-D array `a` is monotone non-decreasing within tolerance `tol`.

                    Checks that each difference a[i+1] - a[i] ≥ −tol. Used to validate layer-top
                    or interface sequences before constructing vertical edges.
                    """
                    a = np.asarray(a, float)
                    return np.all(np.diff(a) >= -tol)

                def _check_monotone_edges3d(z_edges, edge_dim, tol=0.0):
                    """Verify that vertical interfaces in a 3-D array are monotone non-decreasing.

                    Ensures that diff(z_edges, dim=edge_dim) ≥ −tol for every (lat, lon) column.
                    Raises ValueError if any column violates monotonicity. Used after constructing
                    vertical edges to guarantee physically meaningful layering.
                    """
                    dec = (z_edges.diff(edge_dim) < -tol)
                    if dec.any():
                        # Optional: show where (lat, lon) fail
                        where_bad = dec.any(edge_dim)
                        nbad = int(where_bad.sum().item())
                        raise ValueError(f"Edges not monotone along '{edge_dim}' in {nbad} column(s).")

                # ---- Normalize 1-D tops (positive-down)
                tops_raw = np.asarray(depth_top_src_1d, dtype=np.float64).ravel()
                if tops_raw.size == 0:
                    raise ValueError("depth_top_src_1d must have at least one value.")

                tops_clean = tops_raw[np.isfinite(tops_raw)]
                # Flip sign for negatives (do NOT clamp to 0)
                tops_clean = np.where(tops_clean < 0, -tops_clean, tops_clean)
                if not np.any(np.isclose(tops_clean, 0.0)):
                    tops_clean = np.concatenate(([0.0], tops_clean))

                # sort + dedupe AFTER sign flip / zero insert
                tops_clean = np.unique(np.sort(tops_clean))

                # Pre-check (1-D)
                if not _is_monotone_nondec_1d(tops_clean, tol=0.0):
                    raise ValueError("Input tops are not non-decreasing after normalization.")

                # Ensure surface (0) present, sort ascending, unique
                if not np.any(np.isclose(tops_clean, 0.0)):
                    tops_clean = np.concatenate(([0.0], tops_clean))
                tops_clean = np.unique(np.sort(tops_clean))

                # 1-D DataArray with canonical indices 0..nz-1
                nz = int(tops_clean.size)
                zt1d = xr.DataArray(tops_clean, dims=(depth_dim,), coords={depth_dim: np.arange(nz)})

                # ---- Broadcast to H_can grid
                tops = zt1d.broadcast_like(H_can).transpose(depth_dim, *H_can.dims)

                # Bathymetry as float64; clip tiny negatives
                H = xr.where(H_can < 0, 0.0, H_can.astype(np.float64))

                # Clamp any top deeper than local bottom
                tops = xr.where(tops > H, H, tops)

                # Enforce monotone non-decreasing with depth (fixes [0,1,0] -> [0,1,1])
                tops = xr.apply_ufunc(
                    np.maximum.accumulate,
                    tops,
                    input_core_dims=[[depth_dim]],
                    output_core_dims=[[depth_dim]],
                    vectorize=True,
                    dask="parallelized",
                )

                # Append exact bottom as last edge (edge count = nz+1)
                bottom = H.expand_dims({depth_dim: [nz]})
                z_edges_src = xr.concat([tops, bottom], dim=depth_dim)

                # Safety & finalize
                z_edges_src = xr.where(z_edges_src < 0, 0.0, z_edges_src)
                z_edges_src[{depth_dim: nz}] = H  # force exact bottom
                z_edges_src = z_edges_src.rename({depth_dim: edge_dim}).astype(np.float64)

                # Post-check (3-D)
                _check_monotone_edges3d(z_edges_src, edge_dim=edge_dim, tol=0.0)

                return z_edges_src

            # -------- area helper (simple spherical rectangle) --------
            def _cell_areas_from_1d(lat, lon, R=6371000.0):
                """
                Compute spherical cell areas from 1D lat/lon centers using a simple rectangle formula.
                Returns DataArray with dims (latitude, longitude).
                """
                lat = np.asarray(lat, dtype=np.float64)
                lon = np.asarray(lon, dtype=np.float64)
                # edges (assume midpoints are provided)
                def centers_to_edges(v):
                    """Convert 1-D cell-center coordinates to cell-edge coordinates.

                    Produces edges by midpoints between consecutive centers and extrapolating the
                    first/last edges symmetrically. Used to build spherical cell boundaries for
                    conservative area computations.
                    """
                    v = np.asarray(v, dtype=np.float64)
                    e = np.empty(v.size + 1, dtype=np.float64)
                    e[1:-1] = 0.5 * (v[:-1] + v[1:])
                    e[0]  = v[0]  - (e[1] - v[0])
                    e[-1] = v[-1] + (v[-1] - e[-2])
                    return e
                lat_e = np.deg2rad(centers_to_edges(lat))
                lon_e = np.deg2rad(centers_to_edges(lon))
                # areas on sphere
                dlambda = np.diff(lon_e)[None, :]
                # area = R^2 * |sin(phi2)-sin(phi1)| * |dlambda|
                A = R**2 * np.abs(np.sin(lat_e[1:, None]) - np.sin(lat_e[:-1, None])) * np.abs(dlambda)
                return xr.DataArray(A, dims=("latitude", "longitude"),
                                    coords={"latitude": lat, "longitude": lon})

            # -------- horizontal conservative mass regrid (keeps depth) --------
            def _regrid_horizontal_conservative_mass(mass_src, lon_out, lat_out, *,
                                         periodic=True, weights_dir=None,
                                         depth_dim=None):
                """
                Conservatively regrid mass (and volume) horizontally from src grid to (lat_out, lon_out),
                preserving total mass; works for 2D or 3D (with depth) fields.
                (..., depth, latitude, longitude)  OR  (..., latitude, longitude)

                Returns same rank/shape (with lat/lon replaced by target coords).
                """
                lon_name, lat_name = "longitude", "latitude"

                # ---- detect depth if not specified
                if depth_dim is None:
                    depth_dim = _maybe_depth_dim(mass_src)

                dims = list(mass_src.dims)
                front = [d for d in dims if d not in (lat_name, lon_name)]
                if depth_dim in front:
                    front.remove(depth_dim)
                    lead = front + [depth_dim]
                else:
                    lead = front

                # Standardize order so last two are (lat,lon)
                M = mass_src.transpose(*front, *( [depth_dim] if depth_dim else [] ), lat_name, lon_name, missing_dims="ignore")

                # xESMF rect grid
                def _edges_1d(x, periodic=False):
                    """
                    Convert 1D centers to edges, optionally periodic in longitude.
                    """
                    x = np.asarray(x, dtype=np.float64)
                    dx = np.diff(x); mids = x[:-1] + 0.5*dx
                    first = x[0]-0.5*dx[0]; last = x[-1]+0.5*dx[-1]
                    e = np.concatenate(([first], mids, [last]))
                    if periodic:
                        span = x[-1] - x[0] + dx[-1]
                        e[0] = x[0] - 0.5*dx[0]
                        e[-1]= e[0] + span
                    return e

                def make_rect_grid(lon_1d, lat_1d, *, periodic=False):
                    """
                    Create xESMF rectilinear grid Dataset from 1D lon/lat.
                    """
                    lon = np.asarray(lon_1d, dtype=np.float64)
                    lat = np.asarray(lat_1d, dtype=np.float64)
                    return xr.Dataset(coords=dict(
                        lon   = (("lon",), lon),
                        lat   = (("lat",), lat),
                        lon_b = (("lon_b",), _edges_1d(lon, periodic=periodic)),
                        lat_b = (("lat_b",), _edges_1d(lat, periodic=False)),
                    ))

                src_grid = make_rect_grid(np.asarray(M[lon_name]), np.asarray(M[lat_name]), periodic=True)
                dst_grid = make_rect_grid(np.asarray(lon_out),      np.asarray(lat_out),   periodic=True)

                method = "conservative"
                weights_path, reuse = None, False
                if weights_dir is not None:
                    gid = _hash_grid(
                        np.asarray(M[lon_name].values),
                        np.asarray(M[lat_name].values),
                        np.asarray(lon_out),
                        np.asarray(lat_out),
                        method
                    )
                    os.makedirs(weights_dir, exist_ok=True)
                    weights_path = os.path.join(weights_dir, f"mass_{method}_{gid}.nc")
                    reuse = os.path.exists(weights_path)

                R = xe.Regridder(src_grid, dst_grid, method=method,
                                 periodic=periodic, filename=weights_path,
                                 reuse_weights=reuse)
                if (weights_path is not None) and (not reuse):
                    R.to_netcdf(weights_path)

                # stack all leading dims (including depth if present)
                if lead:
                    M_stacked = (M
                        .transpose(*lead, lat_name, lon_name)
                        .stack(_lead=lead)
                        .transpose("_lead", lat_name, lon_name))
                    out_stacked = R(M_stacked)
                    if "lat" in out_stacked.dims or "lon" in out_stacked.dims:
                        out_stacked = out_stacked.rename({"lat": lat_name, "lon": lon_name})
                    out = (out_stacked
                           .transpose("_lead", lat_name, lon_name)
                           .unstack("_lead")
                           .transpose(*front, *( [depth_dim] if depth_dim else [] ), lat_name, lon_name))
                else:
                    # scalar over (lat,lon) only
                    out = R(M)
                    if "lat" in out.dims or "lon" in out.dims:
                        out = out.rename({"lat": lat_name, "lon": lon_name})

                # stamp canonical coords and attrs
                out = out.assign_coords({lat_name: (lat_name, lat_out),
                                         lon_name: (lon_name, lon_out)})
                out.attrs.update(mass_src.attrs)
                out.attrs["regrid_method"] = "conservative"
                return out


            ### Horizontal and vertical interpolation ###

            def regrid3d_conservative_with_topo(
                da,                    # concentration [mass/volume], dims (..., depth, latitude, longitude)
                topo_src,              # 2-D topography [m] (bottom on source), positve downword, dims include ('latitude','longitude')
                lat_can, lon_can,      # 1-D canonical lat/lon
                depth_top_can,         # 1-D canonical TOPS (m, positive down; include 0 if surface)
                need_horizontal_interp,
                need_vertical_interp,
                *,
                periodic_lon=True,
                depth_dim="depth",
                weights_dir=None,
                weights_path=None,
                idx=None,
                idx_tot=None
            ):
                """
                Apply mass-conservative regrid to 3D/2D concentration field using topography:
                conservative xESMF horizontally, thickness-based vertically (not yet supported).
                """
                print(f"Interpolating array {idx} out of {idx_tot-1}")
                H_can = regrid_topography_conservative( # 2-D canonical bottom (lat,lon) in meters (positive down)
                    H_src=topo_src,
                    lat_out=lat_can, lon_out=lon_can,
                    periodic=True,                      # True for global; False for regional
                    # weights_dir="/media/admin-lc/Disk4/tmp_work", #   # or omit to use the system temp
                    weights_dir=weights_dir,
                    weights_path=weights_path
                )

                H_can = H_can.where(H_can>0, 0.0)
                dd = depth_dim if depth_dim in da.dims else None
                has_depth = dd is not None

                # ---- source geometry
                lat_src = np.asarray(da["latitude"].values)
                lon_src = np.asarray(da["longitude"].values)

                if topo_src is None:
                    raise ValueError("topo_src not specified")

                # Areas on source grid
                A_src = xr.DataArray(
                    _cell_areas_from_1d(lat_src, lon_src),
                    dims=("latitude", "longitude"),
                    coords={"latitude": da["latitude"], "longitude": da["longitude"]},
                )

                if has_depth:
                    # build dz_src from tops+bottom (your original code)
                    depth_top_src = np.asarray(da[dd].values, dtype=np.float64)
                    z_edges_src_col = xr.concat(
                        [
                            xr.DataArray(depth_top_src, dims=(dd,), coords={dd: np.arange(depth_top_src.size)}).broadcast_like(topo_src),
                            topo_src.astype(np.float64).expand_dims({dd: [depth_top_src.size]}),
                        ],
                        dim=dd
                    ).rename({dd: "depth_edge"}).transpose("depth_edge", "latitude", "longitude")

                    dz_src = z_edges_src_col.diff("depth_edge").rename({"depth_edge": dd}).clip(min=0.0).astype(np.float64)
                    dz_src = _safe_assign_depth(dz_src, dd, da[dd].values)

                    vol_src  = dz_src * A_src
                    mass_src = da.fillna(0).astype(np.float64) * vol_src

                else:
                    # 2-D case: “volume” is area*topo
                    vol_src  = A_src * topo_src
                    mass_src = da.fillna(0).astype(np.float64) * vol_src


                if not ((need_horizontal_interp is True) or (need_vertical_interp is True)):
                    print("No interpolation necessary")
                    return da

                if need_horizontal_interp:
                    # ---- horizontal conservative regrid of MASS to canonical (lat,lon)
                    mass_h = _regrid_horizontal_conservative_mass(
                        mass_src=mass_src,
                        lon_out=np.asarray(lon_can),
                        lat_out=np.asarray(lat_can),
                        periodic=periodic_lon,
                        weights_dir=weights_dir,
                        depth_dim=dd,
                    )
                    vol_h  = _regrid_horizontal_conservative_mass(
                        mass_src=vol_src,
                        lon_out=np.asarray(lon_can),
                        lat_out=np.asarray(lat_can),
                        periodic=periodic_lon,
                        weights_dir=weights_dir,
                        depth_dim=dd,
                    )

                    # Keep mass_h and vol_h with the original depth labels
                    if has_depth:
                        vol_h = _safe_assign_depth(vol_h, depth_dim, da[depth_dim].values)
                        mass_h = _safe_assign_depth(mass_h, depth_dim, da[depth_dim].values)

                    check_mass_original = np.array(mass_src.sum())
                    print(f"check_mass_original: {check_mass_original}")
                    check_mass_mass_h = np.array(mass_h.sum())
                    print(f"check_mass_mass_h: {check_mass_mass_h}")
                else:
                    mass_h = None

                check_conc_original = np.array(da.sum())
                print(f"check_conc_original: {check_conc_original}")


                if has_depth and need_vertical_interp:
                    raise ValueError("Vertical interpolation not yet supported")
                    # ---- build source & target edges on canonical columns (use canonical bottom H_can)
                    depth_top_can = normalize_depth_tops(depth_top_can)
                    z_edges_tgt, dz_tgt = build_target_edges_from_tops(
                        depth_top_can, H_can, depth_dim=depth_dim, edge_dim="depth_edge"
                    )

                    dz_tgt = _safe_assign_depth(dz_tgt, depth_dim, depth_top_can)

                # ---- back to concentration: C = M / Volume

                if need_horizontal_interp and not need_vertical_interp:
                    mass_fin = mass_h                 # on TARGET grid
                    vol_fin  = vol_h                  # on TARGET grid
                # elif (not need_horizontal_interp) and need_vertical_interp:
                #     mass_fin = mass_v                 # still on SOURCE grid horizontally
                #     vol_fin  = dz_tgt * A_src         # dz_tgt is on SOURCE grid (since no horiz. interp)
                # elif need_horizontal_interp and need_vertical_interp:
                #     mass_fin = mass_v                 # on TARGET grid
                #     vol_fin  = dz_tgt * A_tgt         # matches mass_fin


                conc_out = (mass_fin / vol_fin.where(vol_fin > 0)).where(vol_fin > 0)
                check_conc_out = np.array(conc_out.sum())
                print(f"check_conc_out: {check_conc_out}")

                Cvw_in  = float((da * dz_src * A_src).sum() / (dz_src * A_src).sum())
                Cvw_out = float(mass_fin.sum() / vol_fin.sum())
                print(f"Cvw_in: {Cvw_in}")
                print(f"Cvw_out: {Cvw_out}")

                # tidy dims & attrs
                lead = [d for d in da.dims if d not in ( "latitude", "longitude") and (dd is None or d != dd)]
                if has_depth:
                    conc_out = conc_out.transpose(*lead, dd, "latitude", "longitude", missing_dims="ignore")
                else:
                    conc_out = conc_out.transpose(*lead, "latitude", "longitude", missing_dims="ignore")

                conc_out.name = getattr(da, "name", "concentration")
                conc_out.attrs.update(da.attrs)
                conc_out.attrs["regrid_note"] = (
                    "mass-conserving: xESMF conservative (horizontal) + 3D thickness-overlap (vertical, columnwise); "
                    "bottom from H_can"
                )
                return conc_out

            with tempfile.TemporaryDirectory() as tmpdir:
                # Conservative regrid of MASS

                DataArray_ls = [
                    regrid3d_conservative_with_topo(
                        da=DataArray_dict[idx][variable],       # concentration [mass/volume], dims (..., depth, latitude, longitude)
                        topo_src=DataArray_dict[idx]["topo"],   # 2-D topography [m] (bottom on source), positve downword, dims include ('latitude','longitude')
                        lat_can=lat_can,                        # 1-D canonical lat
                        lon_can=lon_can,                        # 1-D canonical lon
                        depth_top_can=depth_top_can,            # 1-D canonical TOPS (m, positive down; include 0 if surface) e.g., [-1, 0] will be normalized to [0, 1]
                        need_horizontal_interp=need_horizontal_interp,
                        need_vertical_interp=need_vertical_interp,
                        periodic_lon=True,
                        depth_dim="depth",
                        weights_dir=tmpdir,                     # files are written here and auto-removed
                        weights_path=None,                      # let functions compute hashed paths
                        idx=idx, idx_tot=len(DataArray_dict)
                    )
                    for idx in DataArray_dict.keys()
                ]

                nan_counts_interp = {
                    i: int(np.count_nonzero(np.isnan(da.values)))
                    for i, da in enumerate(DataArray_ls)
                }

                first_idx = next(iter(DataArray_dict))
                H_can_out = regrid_topography_conservative(
                    H_src=DataArray_dict[first_idx]["topo"],
                    lat_out=lat_can, lon_out=lon_can,
                    periodic=True,
                    weights_dir=tmpdir,
                    weights_path=None
                ).where(lambda x: x > 0, 0.0)

                if DataArray_dict is not None:
                    DataArray_dict.clear()
                DataArray_dict = None

        else:
            print("No Interpolation needed")


        if nan_counts_interp is not None:
            import math

            def report_nan_changes(nan_counts_inputs, nan_counts_interp, threshold_pct=10.0):
                print("Index | NaN before → after | Δ (after-before) | % change")
                print("-" * 70)
                offenders = []

                all_keys = sorted(set(nan_counts_inputs) | set(nan_counts_interp))
                for k in all_keys:
                    before = nan_counts_inputs.get(k, 0)
                    after  = nan_counts_interp.get(k, 0)
                    delta  = after - before

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

                    print(f"{k:5} | {before:10} → {after:<10} | {delta:16} | {pct_str:>8}")

                    if (math.isinf(pct) or abs(pct) > threshold_pct):
                        offenders.append((k, before, after, pct))

                if offenders:
                    details = ", ".join(
                        f"idx={k} (before={b}, after={a}, change={'+' if (math.isinf(p) or p>=0) else ''}{'inf' if math.isinf(p) else f'{p:.2f}%'})"
                        for k, b, a, p in offenders
                    )
                    raise ValueError(f"NaN count changed by more than {threshold_pct:.1f}% for: {details}")

            report_nan_changes(nan_counts_inputs, nan_counts_interp)

        ### check equality of dimensions after interpolation
        canonical_dims_dict, need_interpolation = self._canonical_nontime_dims(
            DataArray_ls = DataArray_ls,
            nontime_dims = nontime_dims,
            coord_tolerance_dict=None,
            snap_mode=snap_mode,
            dim_res_dict=None)

        time_equal = False
        if all(time_name in da.dims for da in DataArray_ls):
            tvals = [da[time_name].values for da in DataArray_ls]
            time_equal = all(np.array_equal(tv, tvals[0]) for tv in tvals)

        time_start = datetime.now()

        Final_sum = None
        if time_equal and (start_date is None and end_date is None and freq_time is None):
            print("Time dimensions are all equal in DataArray_ls, skipping set-up of target_time to sum directly")
            print("Run sum of DataArray_ls")
            aligned = DataArray_ls
            Final_sum = DataArray_ls[0].fillna(0)
            for da in DataArray_ls[1:]:
                Final_sum = Final_sum + da.fillna(0)
        else:
            print("Time dimensions not all equal (or reconstruction requested)")

            ### Infer start_date/end_date if missing
            if start_date is None:
                start_date = np.min([da[time_name].values.min() for da in DataArray_ls])
                print("start_date set from DataArray_ls")
            if end_date is None:
                end_date = np.max([da[time_name].values.max() for da in DataArray_ls])
                print("end_date set from DataArray_ls")

            ### Prefer a regular grid if a minimal step can be inferred; otherwise use union of times
            target_time = None
            if freq_time is None:
                steps = []
                for da in DataArray_ls:
                    t = da[time_name].values
                    if t.size > 1:
                        # np.diff keeps dtype (datetime64[...]); take minimal positive
                        d = np.diff(t)
                        # filter out non-positive diffs just in case of duplicates
                        d = d[d > np.timedelta64(0, 'ns')]
                        if d.size:
                            steps.append(d.min())
                if steps:
                    freq_time = np.min(steps)
                    print("freq_time set from DataArray_ls")
                else:
                    # fallback: non-regular union of all timestamps
                    target_time = np.unique(
                        np.concatenate([da[time_name].values for da in DataArray_ls])
                    )
                    print("freq_time could not be inferred; using union of timestamps")

            if target_time is None:
                #  inclusive of final timestamp adding a timestep at the end
                target_time = np.arange(start_date, end_date + freq_time, freq_time)

            ### Print start_date, end_date, and freq_time
            try:
                print(f"start_date: {np.datetime_as_string(start_date)}")
                print(f"end_date:   {np.datetime_as_string(end_date)}")
            except Exception:
                pass
            if freq_time is not None:
                ns = (freq_time / np.timedelta64(1, 'ns')).astype('int64')
                if ns >= 3_600_000_000_000:
                    print(f"freq_time:  {ns/3.6e12:.0f} hours")
                else:
                    print(f"freq_time:  {ns/6e10:.0f} min")

            ### Reindex DataArrays using target_time
            print("Reindex and align DataArrays using target_time")
            reindexed = [self._reindex_da(DataArray = da,
                          time_name = time_name,
                          target_time = target_time,
                          nearest_tol_time = nearest_tol_time,
                          align_mode = align_mode) for da in DataArray_ls]

            # Align reindexed DataArrays for sum
            aligned = xr.align(*reindexed, join="exact", copy=False)

            print("Run sum of DataArray_ls")
            Final_sum = aligned[0].fillna(0)
            for da in aligned[1:]:
                Final_sum = Final_sum + da.fillna(0)

        if Final_sum is not None:
            ### Select mask to conserve landmask at different depths
            print("Create landmask")
            all_valid = xr.concat([~a.isnull() for a in aligned], dim="part").any("part").any(dim=time_name)
            Final_sum = Final_sum.where(all_valid)
            Final_sum.attrs["mask_source"] = "any-over-time, union across inputs"

            sum_conc_DataArray_ls_output = float(sum(da.fillna(0).sum().item() for da in aligned))
            sum_conc_Final_sum = float(Final_sum.fillna(0).sum().item())

            print(f"Sum of input DataArray_ls: {sum_conc_DataArray_ls_input}")
            print(f"Sum of aligned DataArray_ls: {sum_conc_DataArray_ls_output}")
            print(f"Sum of Final_sum: {sum_conc_Final_sum}")

            tol_pct = 5.0  # percent

            if sum_conc_DataArray_ls_input == 0:
                # if there was no “mass” in the input, any nonzero output is an issue
                if not np.isclose(sum_conc_Final_sum, 0.0):
                    raise AssertionError(
                        f"Sum check failed: input sum is 0, but Final_sum is {sum_conc_Final_sum} "
                        f"(> {tol_pct:.1f}% change)."
                    )
            else:
                rel_change = (sum_conc_Final_sum - sum_conc_DataArray_ls_input) / sum_conc_DataArray_ls_input
                pct_change = rel_change * 100.0

                print(f"Sum relative change: {pct_change:+.2f}%")

                if abs(rel_change) > tol_pct / 100.0:
                    raise AssertionError(
                        f"Sum check failed: Final_sum={sum_conc_Final_sum}, "
                        f"input={sum_conc_DataArray_ls_input}, "
                        f"change={pct_change:+.2f}% (exceeds ±{tol_pct:.1f}%)."
                    )

            ### Add common_attrs to Final_sum and add "sim_description" if not present
            Final_sum.attrs.update(common_attrs)
            if ("sim_description" not in Final_sum.attrs) and (sim_description is not None):
                Final_sum.attrs["sim_description"] = str(sim_description)
            time_end = datetime.now()
            sum_time = (time_end - time_start)
            print(f"Sum_time (h:min:s): {sum_time}")

            if nan_counts_interp is not None:
                vals = [v for v in nan_counts_interp.values() if isinstance(v, (int, float))]
                avg_nans = float(np.mean(vals)) if vals else 0.0

                ### NaNs in the final result
                Final_sum_nan = int(np.count_nonzero(np.isnan(Final_sum.values)))

                ### Compare with ±10% tolerance and raise if violated
                if avg_nans == 0.0:
                    # If average is zero, any NaNs in Final_sum is an immediate violation
                    if Final_sum_nan != 0:
                        raise ValueError(
                            f"NaN count check failed: avg_nans=0, Final_sum has {Final_sum_nan} NaNs (>10% change)."
                        )
                else:
                    rel_change = (Final_sum_nan - avg_nans) / avg_nans  # signed
                    pct_change = 100.0 * rel_change
                    if abs(rel_change) > 0.05:
                        raise ValueError(
                            f"NaN count check failed: Final_sum_nans={Final_sum_nan}, "
                            f"avg_nans={avg_nans:.2f}, change={pct_change:+.2f}% (exceeds ±5%)."
                        )
                    else:
                        print(
                            f"NaN count check OK: Final_sum_nans={Final_sum_nan}, "
                            f"avg_nans={avg_nans:.2f}, change={pct_change:+.2f}% (within ±5%)."
                        )

            if H_can_out is not None:

                H_can_out.assign_coords(
                    latitude = Final_sum["latitude"].values,
                    longitude= Final_sum["longitude"].values,
                )
                xr.align(Final_sum, H_can_out, join="exact")

            return Final_sum, H_can_out
        else:
            raise ValueError("Final_sum is None")



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