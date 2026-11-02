"""The fixed Location vocabulary: every Location in Experiment 01's Starting save, in save order.

IDs never change once training starts, so append new names rather than reordering.
"""

UNKNOWN = 0
"""ID for Locations not in the vocabulary, e.g. mine levels and farm buildings' interiors."""

LOCATIONS = (
    "Farm",
    "FarmHouse",
    "FarmCave",
    "Town",
    "JoshHouse",
    "HaleyHouse",
    "SamHouse",
    "Blacksmith",
    "ManorHouse",
    "SeedShop",
    "Saloon",
    "Trailer",
    "Hospital",
    "HarveyRoom",
    "Beach",
    "BeachNightMarket",
    "ElliottHouse",
    "Mountain",
    "ScienceHouse",
    "SebastianRoom",
    "Tent",
    "Forest",
    "WizardHouse",
    "AnimalShop",
    "LeahHouse",
    "BusStop",
    "Mine",
    "Sewer",
    "BugLand",
    "Desert",
    "Club",
    "SandyHouse",
    "ArchaeologyHouse",
    "WizardHouseBasement",
    "AdventureGuild",
    "Woods",
    "Railroad",
    "WitchSwamp",
    "WitchHut",
    "WitchWarpCave",
    "Summit",
    "FishShop",
    "BathHouse_Entry",
    "BathHouse_MensLocker",
    "BathHouse_WomensLocker",
    "BathHouse_Pool",
    "CommunityCenter",
    "JojaMart",
    "Greenhouse",
    "SkullCave",
    "Backwoods",
    "Tunnel",
    "Trailer_Big",
    "Cellar",
    "MermaidHouse",
    "Submarine",
    "AbandonedJojaMart",
    "MovieTheater",
    "Sunroom",
    "BoatTunnel",
    "IslandSouth",
    "IslandSouthEast",
    "IslandSouthEastCave",
    "IslandEast",
    "IslandWest",
    "IslandNorth",
    "IslandHut",
    "IslandWestCave1",
    "IslandNorthCave1",
    "IslandFieldOffice",
    "IslandFarmHouse",
    "CaptainRoom",
    "IslandShrine",
    "IslandFarmCave",
    "Caldera",
    "LeoTreeHouse",
    "QiNutRoom",
    "MasteryCave",
    "DesertFestival",
    "LewisBasement",
)

_IDS = {name: index + 1 for index, name in enumerate(LOCATIONS)}

LOCATION_COUNT = len(LOCATIONS) + 1
"""Number of IDs, `UNKNOWN` included."""


def location_id(name: str) -> int:
    return _IDS.get(name, UNKNOWN)
