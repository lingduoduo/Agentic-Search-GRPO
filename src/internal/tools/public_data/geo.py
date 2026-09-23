"""Location tools: current weather, geocoding, and nearby points of interest.

All three answer with a flat JSON object of facts, so none is citeable.
"""

from __future__ import annotations

import re

from ..base import FunctionTool, ToolEffect
from ._http import PublicDataError, get_json, guarded, post_json

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass queries are slow; give them their own budget.
OVERPASS_TIMEOUT_SECONDS = 30.0

# The place type is interpolated into an Overpass QL regex. Validate rather
# than escape: a stray `"]` would otherwise close the tag filter and let the
# caller append arbitrary statements to the query.
_PLACE_TYPE_RE = re.compile(r"^[A-Za-z0-9 _-]{1,40}$")

# WMO weather interpretation codes.
_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ------------------------------------------------------------------ weather

_WEATHER_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "minLength": 1,
            "description": "City or place name, e.g. 'Berlin'.",
        },
        "latitude": {
            "type": "number",
            "minimum": -90,
            "maximum": 90,
            "description": "Latitude; skips the place-name lookup when given.",
        },
        "longitude": {
            "type": "number",
            "minimum": -180,
            "maximum": 180,
            "description": "Longitude; skips the place-name lookup when given.",
        },
    },
    "required": ["location"],
    "additionalProperties": False,
}


async def _get_weather(
    location: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict:
    country = ""
    if latitude is None or longitude is None:
        found = await get_json(
            GEOCODE_URL,
            params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )
        results = found.get("results") or []
        if not results:
            raise PublicDataError(f"could not find a place called {location!r}")
        first = results[0]
        latitude = first["latitude"]
        longitude = first["longitude"]
        location = first.get("name", location)
        country = first.get("country", "")

    payload = await get_json(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m,wind_direction_10m"
            ),
            "timezone": "auto",
        },
    )
    current = payload.get("current") or {}
    return {
        "location": location,
        "country": country,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "precipitation": current.get("precipitation"),
        "description": _WEATHER_CODES.get(current.get("weather_code"), "Unknown"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_direction": current.get("wind_direction_10m"),
        "units": "metric",
        "observed_at": current.get("time"),
    }


def build_weather_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_get_weather),
        name="get_weather",
        description=(
            "Get the current weather — temperature, conditions, wind, humidity "
            "— for a city or place."
        ),
        parameters=_WEATHER_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


# ----------------------------------------------------------------- geocoding

_LOCATION_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "Place, address, or landmark to find.",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": "How many matches to return (1-20).",
            "default": 5,
        },
        "country_code": {
            "type": "string",
            "pattern": "^[A-Za-z]{2}$",
            "description": "Two-letter country filter, e.g. 'fr'.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


async def _search_location(
    query: str, limit: int = 5, country_code: str | None = None
) -> dict:
    params = {
        "q": query,
        "format": "json",
        "limit": max(1, min(int(limit), 20)),
        "addressdetails": 1,
    }
    if country_code:
        params["countrycodes"] = country_code.strip().lower()

    rows = await get_json(NOMINATIM_URL, params=params)
    locations = []
    for item in rows or []:
        address = item.get("address") or {}
        locations.append(
            {
                "display_name": item.get("display_name"),
                "latitude": float(item["lat"]),
                "longitude": float(item["lon"]),
                "type": item.get("type"),
                "category": item.get("class"),
                "country": address.get("country"),
                "city": (
                    address.get("city") or address.get("town") or address.get("village")
                ),
                "postcode": address.get("postcode"),
            }
        )
    return {"query": query, "locations": locations, "count": len(locations)}


def build_location_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_location),
        name="search_location",
        description=(
            "Find the coordinates and address of a place, landmark, or address "
            "by name. Use this before search_nearby_places."
        ),
        parameters=_LOCATION_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )


# -------------------------------------------------------------------- places

_PLACES_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "Place type, e.g. 'cafe', 'pharmacy', 'hotel'.",
        },
        "latitude": {
            "type": "number",
            "minimum": -90,
            "maximum": 90,
            "description": "Centre latitude.",
        },
        "longitude": {
            "type": "number",
            "minimum": -180,
            "maximum": 180,
            "description": "Centre longitude.",
        },
        "radius_meters": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10000,
            "description": "Search radius in metres (max 10000).",
            "default": 1000,
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "description": "How many places to return (1-50).",
            "default": 10,
        },
    },
    "required": ["query", "latitude", "longitude"],
    "additionalProperties": False,
}


async def _search_nearby_places(
    query: str,
    latitude: float,
    longitude: float,
    radius_meters: int = 1000,
    limit: int = 10,
) -> dict:
    place_type = query.strip()
    if not _PLACE_TYPE_RE.match(place_type):
        raise PublicDataError(
            "place type must be 1-40 characters of letters, digits, spaces, "
            "hyphens, or underscores (for example 'cafe' or 'fire_station')"
        )
    lat = float(latitude)
    lon = float(longitude)
    radius = max(1, min(int(radius_meters), 10000))
    limit = max(1, min(int(limit), 50))

    # Anchored so "cafe" matches amenity=cafe and not every value containing
    # it. The name-substring clause the reference used is dropped: it makes
    # Overpass scan far more elements and routinely times out.
    around = f"(around:{radius},{lat},{lon})"
    overpass_ql = (
        "[out:json][timeout:25];\n"
        "(\n"
        f'  node["amenity"~"^{place_type}$",i]{around};\n'
        f'  node["shop"~"^{place_type}$",i]{around};\n'
        f'  node["tourism"~"^{place_type}$",i]{around};\n'
        ");\n"
        f"out body {limit};"
    )

    payload = await post_json(
        OVERPASS_URL,
        data={"data": overpass_ql},
        timeout_seconds=OVERPASS_TIMEOUT_SECONDS,
    )
    places = []
    for element in (payload.get("elements") or [])[:limit]:
        tags = element.get("tags") or {}
        places.append(
            {
                "name": tags.get("name", "Unnamed"),
                "type": tags.get("amenity") or tags.get("shop") or tags.get("tourism"),
                "latitude": element.get("lat"),
                "longitude": element.get("lon"),
                "street": tags.get("addr:street"),
                "city": tags.get("addr:city"),
                "phone": tags.get("phone"),
                "website": tags.get("website"),
                "opening_hours": tags.get("opening_hours"),
            }
        )
    return {
        "query": place_type,
        "centre": {"latitude": lat, "longitude": lon},
        "radius_meters": radius,
        "places": places,
        "count": len(places),
    }


def build_nearby_places_tool() -> FunctionTool:
    return FunctionTool(
        fn=guarded(_search_nearby_places),
        name="search_nearby_places",
        description=(
            "List points of interest of a given type — cafes, pharmacies, "
            "hotels — near a latitude and longitude."
        ),
        parameters=_PLACES_PARAMS,
        effect=ToolEffect.READ_ONLY,
    )
