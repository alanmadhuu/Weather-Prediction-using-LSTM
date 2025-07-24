import xarray as xr
import pandas as pd

# File path
grib_file = r"C:/Users/Aaron/Downloads/angamaly_2015_2007.grib"
csv_file = r"C:/Users/Aaron/Downloads/angamaly_2015_2007.csv"

try:
    # Open GRIB file using cfgrib without caching issues
    ds = xr.open_dataset(grib_file, engine="cfgrib")

    # Convert to DataFrame
    df = ds.to_dataframe().reset_index()

    # Save as CSV
    df.to_csv(csv_file, index=False)

    print(f"GRIB file '{grib_file}' successfully converted to CSV '{csv_file}'")

except Exception as e:
    print(f"Error: {e}")