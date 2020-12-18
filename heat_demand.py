import sys
import os

import dask.dataframe as dd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from rc_buildingsimulator.building_physics import Zone  # Importing Zone Class
from rc_buildingsimulator import supply_system
from rc_buildingsimulator import emission_system
from rc_buildingsimulator.radiation import Location
from rc_buildingsimulator.radiation import Window

matplotlib.style.use('ggplot')

def calculate_individual_hh_demand(
    City,
    window_area=4.0,
    walls_area=11.0,
    floor_area=35.0,
    room_vol=105,
    total_internal_area=142.0,
    lighting_load=11.7,
    lighting_control=300.0,
    lighting_utilisation_factor=0.45,
    lighting_maintenance_factor=0.9,
    u_walls=0.2,
    u_windows=1.1,
    ach_vent=1.5,
    ach_infl=0.5,
    ventilation_efficiency=0.6,
    thermal_capacitance_per_floor_area=165000,
    t_set_heating=20.0,
    t_set_cooling=26.0,
    max_cooling_energy_per_floor_area=-np.inf,
    max_heating_energy_per_floor_area=np.inf,
    heating_supply_system=supply_system.OilBoilerMed,
    cooling_supply_system=supply_system.HeatPumpAir,
    heating_emission_system=emission_system.NewRadiators,
    cooling_emission_system=emission_system.AirConditioning,
):

    # Initialise an instance of the Zone. Empty spaces take on the default
    # parameters. See ZonePhysics.py to see the default values
    Household = Zone(
        window_area=window_area,
        walls_area=walls_area,
        floor_area=floor_area,
        room_vol=room_vol,
        total_internal_area=total_internal_area,
        lighting_load=lighting_load,
        lighting_control=lighting_control,
        lighting_utilisation_factor=lighting_utilisation_factor,
        lighting_maintenance_factor=lighting_maintenance_factor,
        u_walls=u_walls,
        u_windows=u_windows,
        ach_vent=ach_vent,
        ach_infl=ach_infl,
        ventilation_efficiency=ventilation_efficiency,
        thermal_capacitance_per_floor_area=thermal_capacitance_per_floor_area,
        t_set_heating=t_set_heating,
        t_set_cooling=t_set_cooling,
        max_cooling_energy_per_floor_area=max_cooling_energy_per_floor_area,
        max_heating_energy_per_floor_area=max_heating_energy_per_floor_area,
        heating_supply_system=heating_supply_system,
        cooling_supply_system=cooling_supply_system,
        heating_emission_system=heating_emission_system,
        cooling_emission_system=cooling_emission_system,
    )

    # Define Windows
    SouthWindow = Window(
        azimuth_tilt=0,
        alititude_tilt=90,
        glass_solar_transmittance=0.7,
        glass_light_transmittance=0.8,
        area=4
    )

    # A catch statement to prevent future coding bugs when modifying window area
    if SouthWindow.area != Household.window_area:
        raise ValueError('Window area defined in radiation file doesnt match area defined in zone')

    # Define constants for the Zone
    gain_per_person = 100  # W per person
    appliance_gains = 14  # W per sqm
    max_occupancy = 3.0

    # Read Occupancy Profile
    occupancyProfile = pd.read_csv('schedules_el_HOUSEHOLD.csv')

    # Starting temperature of the builidng
    t_m_prev = 20

    # Empty Lists for Storing Data to Plot
    ElectricityOut = []
    HeatingDemand = []  # Energy required by the zone
    HeatingEnergy = []  # Energy required by the supply system to provide HeatingDemand
    CoolingDemand = []  # Energy surplus of the zone
    CoolingEnergy = []  # Energy required by the supply system to get rid of CoolingDemand
    IndoorAir = []
    OutsideTemp = []
    SolarGains = []
    COP = []

    # Loop through all 8760 hours of the year
    for hour in range(8760):

        # Occupancy for the time step
        occupancy = occupancyProfile.loc[hour, 'People'] * max_occupancy
        
        # Gains from occupancy and appliances
        internal_gains = (
            occupancy * gain_per_person + appliance_gains * Household.floor_area
        )

        # Extract the outdoor temperature in Zurich for that hour
        t_out = City.weather_data['drybulb_C'][hour]

        Altitude, Azimuth = City.calc_sun_position(
            latitude_deg=47.480,
            longitude_deg=8.536,
            year=2015,
            hoy=hour
        )
        SouthWindow.calc_solar_gains(
            sun_altitude=Altitude,
            sun_azimuth=Azimuth,
            normal_direct_radiation=City.weather_data['dirnorrad_Whm2'][hour],
            horizontal_diffuse_radiation=City.weather_data['difhorrad_Whm2'][hour],
        )
        SouthWindow.calc_illuminance(
            sun_altitude=Altitude,
            sun_azimuth=Azimuth,
            normal_direct_illuminance=City.weather_data['dirnorillum_lux'][hour],
            horizontal_diffuse_illuminance=City.weather_data['difhorillum_lux'][hour],
        )
        Household.solve_energy(
            internal_gains=internal_gains,
            solar_gains=SouthWindow.solar_gains,
            t_out=t_out,
            t_m_prev=t_m_prev
        )
        Household.solve_lighting(
            illuminance=SouthWindow.transmitted_illuminance,
            occupancy=occupancy
        )

        # Set the previous temperature for the next time step
        t_m_prev = Household.t_m_next

        HeatingDemand.append(Household.heating_demand)
        HeatingEnergy.append(Household.heating_energy)
        CoolingDemand.append(Household.cooling_demand)
        CoolingEnergy.append(Household.cooling_energy)
        ElectricityOut.append(Household.electricity_out)
        IndoorAir.append(Household.t_air)
        OutsideTemp.append(t_out)
        SolarGains.append(SouthWindow.solar_gains)
        COP.append(Household.cop)

    return pd.DataFrame({
        'HeatingDemand': HeatingDemand,
        'HeatingEnergy': HeatingEnergy,
        'CoolingDemand': CoolingDemand,
        'CoolingEnergy': CoolingEnergy,
        'IndoorAir': IndoorAir,
        'OutsideTemp':  OutsideTemp,
        'SolarGains': SolarGains,
        'COP': COP
    })

def calculate_annual_heat_demand(berpublicsearch: pd.DataFrame = None) -> pd.DataFrame:
    
    # Initialise the Location with a weather file
    Dublin = Location(epwfile_path='IRL_Dublin.039690_IWEC.epw')

    return  calculate_individual_hh_demand(Dublin)
    