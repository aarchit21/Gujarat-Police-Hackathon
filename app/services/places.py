"""City-level Gujarat map positions inferred from camera names.

cameras.json is currently id+name only. These points are public city/locality
centroids, not surveyed camera GPS. GIS links stay inferred, never proven.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Obvious water over a naive Gujarat bounding box. Centroids below stay outside.
# Gulf of Kutch, Gulf of Khambhat, Arabian Sea west of Saurashtra.
WATER_BOXES = (
    (22.28, 22.98, 68.90, 69.95),  # Gulf of Kutch
    (21.32, 22.22, 72.22, 72.68),  # Gulf of Khambhat
    (20.00, 22.08, 68.00, 69.38),  # Arabian Sea west of Porbandar
    (19.50, 20.72, 68.00, 71.40),  # Open sea south of the coast
)

# ~1–2 km rings so cameras that share a city do not stack. Centroids are inland.
_OFFSETS = (
    (0.012, 0.010),
    (0.010, -0.012),
    (-0.011, 0.009),
    (-0.009, -0.011),
    (0.016, 0.004),
    (0.004, 0.016),
    (-0.014, 0.006),
    (0.006, -0.015),
    (0.008, 0.014),
    (-0.006, 0.013),
    (0.014, -0.007),
    (-0.013, -0.005),
)


@dataclass(frozen=True)
class Place:
    label: str
    lat: float
    lng: float
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MapPosition:
    lat: float
    lng: float
    source: str
    label: str


# Specific aliases first after length-sort. Public city centroids only.
PLACES: tuple[Place, ...] = (
    Place("Ahmedabad", 23.0263, 72.5755, ("chiman bhai bridge", "chimanbhai", "chiman bhai")),
    Place("Ahmedabad", 23.0116, 72.5632, ("paldi circle", "paldi",)),
    Place("Ahmedabad", 23.0754, 72.5878, ("visat teen rasta", "visat p2", "visat",)),
    Place("Ahmedabad", 23.0228, 72.5465, ("cn vidhyalaya", "cn vidyalaya", "c n vidhyalaya")),
    Place("Ahmedabad", 23.0225, 72.5714, ("janpath", "o n g c", "ongc office", "ongc", "ahmedabad", "amdavad")),
    Place("Adalaj", 23.1645, 72.5810, ("adalaj", "tri mandir")),
    Place("Junagadh", 21.5222, 70.4579, ("timbavadi", "majewadi", "majevadi", "dolatpara", "junagadh")),
    Place("Gir Somnath", 20.9159, 70.3629, ("gir somnath", "gir-somnath", "veraval", "somnath")),
    Place("Rajkot", 22.3039, 70.8022, ("rajkot bus port", "rajkot", "mohanpura")),
    Place("Gandevi", 20.8128, 73.0026, ("khaparia", "gandevi")),
    Place("Navsari", 20.9510, 72.9320, ("navsari",)),
    Place("Bilimora", 20.7696, 72.9613, ("bilimora",)),
    Place("Patan", 23.8493, 72.1266, ("patan dethali", "dethali", "patan")),
    Place("Dehgam", 23.1680, 72.8210, ("dehgam",)),
    Place("Gandhidham", 23.0753, 70.1337, ("gandhidham", "rambaugh", "rambagh")),
    Place("Surat", 21.1702, 72.8311, ("surat",)),
    Place("Gandhinagar", 23.2156, 72.6369, ("gandhinagar",)),
    Place("Vadodara", 22.3072, 73.1812, ("vadodara", "baroda")),
    Place("Bhavnagar", 21.7645, 72.1519, ("bhavnagar",)),
    Place("Jamnagar", 22.4707, 70.0577, ("jamnagar",)),
    Place("Bhuj", 23.2420, 69.6669, ("bhuj",)),
    Place("Anjar", 23.1132, 70.0269, ("anjar",)),
    Place("Porbandar", 21.6417, 69.6293, ("porbandar",)),
    Place("Dwarka", 22.2394, 68.9678, ("dwarka", "okha")),
    Place("Morbi", 22.8170, 70.8370, ("morbi", "morvi")),
    Place("Surendranagar", 22.7289, 71.6370, ("surendranagar", "wadhwan")),
    Place("Mehsana", 23.5880, 72.3693, ("mehsana",)),
    Place("Palanpur", 24.1725, 72.4345, ("palanpur",)),
    Place("Bharuch", 21.7051, 73.0059, ("bharuch", "broach")),
    Place("Anand", 22.5645, 72.9289, ("anand", "vallabh vidyanagar")),
    Place("Nadiad", 22.6916, 72.8634, ("nadiad",)),
    Place("Godhra", 22.7788, 73.6143, ("godhra",)),
    Place("Amreli", 21.6032, 71.2221, ("amreli",)),
    Place("Botad", 22.1696, 71.6684, ("botad",)),
    Place("Himmatnagar", 23.5989, 72.9660, ("himmatnagar", "himatnagar")),
    Place("Dahod", 22.8390, 74.2578, ("dahod",)),
    Place("Valsad", 20.6100, 72.9260, ("valsad", "bulsar")),
    Place("Vapi", 20.3893, 72.9106, ("vapi",)),
    Place("Gondal", 21.9607, 70.8026, ("gondal",)),
    Place("Jetpur", 21.7542, 70.6234, ("jetpur",)),
    Place("Palitana", 21.5174, 71.8235, ("palitana",)),
    Place("Bardoli", 21.1232, 73.1116, ("bardoli",)),
    Place("Modasa", 23.4625, 73.2986, ("modasa",)),
)

# Inland district towns used only when the name has no recognisable place.
INLAND_CITIES: tuple[tuple[str, float, float], ...] = (
    ("Ahmedabad", 23.0225, 72.5714),
    ("Surat", 21.1702, 72.8311),
    ("Vadodara", 22.3072, 73.1812),
    ("Rajkot", 22.3039, 70.8022),
    ("Gandhinagar", 23.2156, 72.6369),
    ("Junagadh", 21.5222, 70.4579),
    ("Bhavnagar", 21.7645, 72.1519),
    ("Jamnagar", 22.4707, 70.0577),
    ("Anand", 22.5645, 72.9289),
    ("Bharuch", 21.7051, 73.0059),
    ("Mehsana", 23.5880, 72.3693),
    ("Patan", 23.8493, 72.1266),
    ("Palanpur", 24.1725, 72.4345),
    ("Himmatnagar", 23.5989, 72.9660),
    ("Nadiad", 22.6916, 72.8634),
    ("Godhra", 22.7788, 73.6143),
    ("Surendranagar", 22.7289, 71.6370),
    ("Morbi", 22.8170, 70.8370),
    ("Amreli", 21.6032, 71.2221),
    ("Botad", 22.1696, 71.6684),
    ("Dahod", 22.8390, 74.2578),
    ("Navsari", 20.9510, 72.9320),
    ("Valsad", 20.6100, 72.9260),
    ("Bhuj", 23.2420, 69.6669),
    ("Anjar", 23.1132, 70.0269),
    ("Modasa", 23.4625, 73.2986),
    ("Bardoli", 21.1232, 73.1116),
    ("Palitana", 21.5174, 71.8235),
    ("Gondal", 21.9607, 70.8026),
    ("Jetpur", 21.7542, 70.6234),
)

_ALIAS_INDEX: list[tuple[str, Place]] = sorted(
    ((alias, place) for place in PLACES for alias in place.aliases),
    key=lambda item: len(item[0]),
    reverse=True,
)


def _norm(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return f" {cleaned.strip()} "


def _stable_n(camera_id: str) -> int:
    digits = "".join(ch for ch in (camera_id or "") if ch.isdigit())
    if digits:
        return int(digits)
    return int(hashlib.md5((camera_id or "cam").encode("utf-8")).hexdigest()[:8], 16)


def in_known_water(lat: float, lng: float) -> bool:
    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except (TypeError, ValueError):
        return True
    for south, north, west, east in WATER_BOXES:
        if south <= lat_f <= north and west <= lng_f <= east:
            return True
    return False


def offset_on_land(camera_id: str, lat: float, lng: float) -> tuple[float, float]:
    n = _stable_n(camera_id)
    dlat, dlng = _OFFSETS[n % len(_OFFSETS)]
    scale = 0.7 + (n % 5) * 0.12
    lat2 = round(lat + dlat * scale, 6)
    lng2 = round(lng + dlng * scale, 6)
    if in_known_water(lat2, lng2):
        return round(lat, 6), round(lng, 6)
    return lat2, lng2


def match_place(name: str, city: str = "") -> Place | None:
    blob = _norm(f"{name} {city}")
    if blob.strip() == "":
        return None
    for alias, place in _ALIAS_INDEX:
        if f" {alias} " in blob:
            return place
    return None


def inland_placeholder(camera_id: str) -> MapPosition:
    n = _stable_n(camera_id)
    label, lat, lng = INLAND_CITIES[n % len(INLAND_CITIES)]
    lat, lng = offset_on_land(camera_id, lat, lng)
    return MapPosition(lat=lat, lng=lng, source="placeholder", label=label)


def infer_map_position(camera_id: str, name: str = "", city: str = "") -> MapPosition:
    """Return a land position. Never a surveyed camera site."""
    place = match_place(name, city)
    if place is not None:
        lat, lng = offset_on_land(camera_id, place.lat, place.lng)
        return MapPosition(lat=lat, lng=lng, source="inferred_place", label=place.label)
    return inland_placeholder(camera_id)
