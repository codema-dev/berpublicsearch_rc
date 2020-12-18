from icontract import require
import numpy as np
import pandas as pd

from rc_buildingsimulator.building_physics import Zone  # Importing Zone Class
from rc_buildingsimulator import supply_system
from rc_buildingsimulator import emission_system
from rc_buildingsimulator.radiation import Location
from rc_buildingsimulator.radiation import Window

COLUMNS = [
    "WallArea",
    "WindowArea",
    "GroundFloorArea",
    "GroundFloorHeight",
    "FirstFloorHeight",
    "SecondFloorHeight",
    "ThirdFloorHeight",
    "HSMainSystemEfficiency",
    "UValueWall",
    "UValueWindow",
    "HeatExchangerEff",
    "ThermalMassCategory",
]


def calculate_annual_household_heat_demand(
    City,
    zone,
    window,
    gain_per_person,
    appliance_gains,
    max_occupancy,
    path_to_occupancy_profile,
):
    # A catch statement to prevent future coding bugs when modifying window area
    if window.area != zone.window_area:
        raise ValueError(
            "Window area defined in radiation file doesnt match area defined in zone"
        )

    # Read Occupancy Profile
    occupancyProfile = pd.read_csv(path_to_occupancy_profile)

    # Starting temperature of the builidng
    t_m_prev = 20

    # Empty Lists for Storing Data to Plot
    HeatingDemand = []  # Energy required by the zone

    # Loop through all 8760 hours of the year
    for hour in range(8760):

        # Occupancy for the time step
        occupancy = occupancyProfile.loc[hour, "People"] * max_occupancy

        # Gains from occupancy and appliances
        internal_gains = occupancy * gain_per_person + appliance_gains * zone.floor_area

        # Extract the outdoor temperature in City for that hour
        t_out = City.weather_data["drybulb_C"][hour]

        Altitude, Azimuth = City.calc_sun_position(
            latitude_deg=47.480,
            longitude_deg=8.536,
            year=2015,
            hoy=hour,
        )
        window.calc_solar_gains(
            sun_altitude=Altitude,
            sun_azimuth=Azimuth,
            normal_direct_radiation=City.weather_data["dirnorrad_Whm2"][hour],
            horizontal_diffuse_radiation=City.weather_data["difhorrad_Whm2"][hour],
        )
        zone.solve_energy(
            internal_gains=internal_gains,
            solar_gains=window.solar_gains,
            t_out=t_out,
            t_m_prev=t_m_prev,
        )

        # Set the previous temperature for the next time step
        t_m_prev = zone.t_m_next

        HeatingDemand.append(zone.heating_demand)

    return sum(HeatingDemand)


@require(
    lambda stock: set(COLUMNS).issubset(set(stock.columns)),
    "'stock' is missing required columns",
)
def calculate_annual_stock_heat_demand(stock: pd.DataFrame = None) -> pd.DataFrame:

    # Initialise the Location with a weather file
    Dublin = Location(epwfile_path="IRL_Dublin.039690_IWEC.epw")

    # Section 12.3.1.2 of ISO 13790
    thermal_capacitance = {
        "Low                 ": 80000,
        "Medium-low          ": 110000,
        "Medium              ": 165000,
        "Medium-high         ": 260000,
        "High                ": 370000,
    }

    heat_demands = []
    for row in stock.itertuples():

        total_internal_area = row.WallArea + row.WindowArea + row.GroundFloorArea
        building_height = (
            row.GroundFloorHeight
            + row.FirstFloorHeight
            + row.SecondFloorHeight
            + row.ThirdFloorHeight
        )
        building_volume = building_height * row.GroundFloorArea
        thermal_capacitance_per_floor_area = thermal_capacitance[
            row.ThermalMassCategory
        ]

        # Initialise an instance of the Zone. Empty spaces take on the default
        # parameters. See ZonePhysics.py to see the default values
        zone = Zone(
            window_area=row.WindowArea,
            walls_area=row.WallArea,
            floor_area=row.GroundFloorArea,
            room_vol=building_volume,
            total_internal_area=total_internal_area,
            u_walls=row.UValueWall,
            u_windows=row.UValueWindow,
            ach_vent=1.5,
            ach_infl=0.5,
            ventilation_efficiency=row.HeatExchangerEff,
            thermal_capacitance_per_floor_area=thermal_capacitance_per_floor_area,
            t_set_heating=20.0,
            max_cooling_energy_per_floor_area=-np.inf,
            max_heating_energy_per_floor_area=np.inf,
            heating_supply_system=supply_system.OilBoilerMed,
            heating_emission_system=emission_system.NewRadiators,
        )
        # Define Windows
        south_window = Window(
            azimuth_tilt=0,
            alititude_tilt=90,
            glass_solar_transmittance=0.7,
            glass_light_transmittance=0.8,
            area=row.WindowArea,
        )
        heat_demand = calculate_annual_household_heat_demand(
            Dublin,
            zone,
            south_window,
            gain_per_person=100,  # W per person
            appliance_gains=14,  # W per sqm
            max_occupancy=3.0,
            path_to_occupancy_profile="schedules_el_HOUSEHOLD.csv",
        )
        heat_demands.append(heat_demand)

    return pd.Series(heat_demands)
