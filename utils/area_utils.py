# utils/area_utils.py
"""
Offline-capable geospatial area calculations.

Uses pyproj and shapely to calculate the area of a
polygon defined by latitude and longitude coordinates.
"""

from typing import List, Dict
from pyproj import Geod
from shapely.geometry import Polygon

# Initialize a Geod object with the WGS-84 ellipsoid.
# This is the standard for GPS.
# This object is created once and reused, making it efficient.
geod = Geod(ellps="WGS84")

def calculate_area_acres(polygon_latlon: List[Dict[str, float]]) -> float:
    """
    Calculates the area of a polygon in acres, given a list of
    latitude/longitude dictionaries.

    Args:
        polygon_latlon: A list of dicts, e.g.,
                        [{"lat": 13.0, "lng": 80.0}, ...]
                        (Note: The original prompt used 'lat' and 'lng')
                        
    Returns:
        The calculated area in acres, or 0.0 if input is invalid.
    """
    if not polygon_latlon or len(polygon_latlon) < 3:
        # A polygon needs at least 3 points.
        return 0.0

    try:
        # 1. Extract coordinates into a list of (lon, lat) tuples.
        # Shapely and many geo-tools expect (longitude, latitude) order.
        # The prompt's ESP32 example used 'lng' for longitude.
        
        # Check if keys are 'lat'/'lng' or 'latitude'/'longitude'
        key_lat = 'lat' if 'lat' in polygon_latlon[0] else 'latitude'
        key_lon = 'lng' if 'lng' in polygon_latlon[0] else 'longitude'
        
        # Create the list of (lon, lat) tuples
        coords = [
            (point[key_lon], point[key_lat]) for point in polygon_latlon
        ]

        # 2. Create a Shapely Polygon
        # The coordinates are assumed to be in WGS-84 (lat/lon).
        polygon = Polygon(coords)

        # 3. Use pyproj.Geod.geometry_area_perimeter
        # This calculates the area on the WGS-84 ellipsoid.
        # It returns area (in square meters) and perimeter (in meters).
        area_sq_meters, perimeter = geod.geometry_area_perimeter(polygon)

        # 4. Convert square meters to acres
        # 1 acre = 4046.86 square meters
        area_acres = abs(area_sq_meters) / 4046.86

        return area_acres

    except Exception as e:
        print(f"[ERROR] Could not calculate polygon area: {e}")
        # This could be due to malformed input, self-intersecting polygon, etc.
        return 0.0

# --- Optional: Area from radius (as requested) ---
def area_from_point_radius(lat: float, lon: float, radius_meters: float) -> float:
    """
    Calculates the area (in acres) of a circle defined by a
    center point and a radius in meters.
    """
    try:
        # Calculate the area of a circle in square meters
        area_sq_meters = 3.14159 * (radius_meters ** 2)
        
        # Convert to acres
        area_acres = area_sq_meters / 4046.86
        return area_acres
        
    except Exception as e:
        print(f"[ERROR] Could not calculate area from radius: {e}")
        return 0.0