"""Basic geospatial point-data example."""
import pandas as pd

sites = pd.DataFrame({
    "site": ["A", "B", "C"],
    "latitude": [38.6270, 38.6100, 38.6500],
    "longitude": [-90.1994, -90.2200, -90.1800],
    "temperature": [72.0, 74.5, 70.2]
})

print(sites)
print("\nLatitude range:", sites["latitude"].min(), "to", sites["latitude"].max())
print("Longitude range:", sites["longitude"].min(), "to", sites["longitude"].max())
