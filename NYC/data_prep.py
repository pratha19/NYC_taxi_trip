# Function for preparing the data

"""
This module contains user defined functions for importing and cleaning the data and also for adding new features.
"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import io
import requests
import json
import pyproj
import geopandas as gpd
from shapely.geometry import Point


def extract_url(raw_url: str, query: str, months: list, limit: int):
    """
    This function takes the raw_url, query (SQL), months and limit to define the html query to request data from the city of 
    new york's site.
    Args:
        raw_url: the base URL where the data resides
        query: SQL query to extract the data
        months: a list of months e.g. [1,2,3,4] if you want to extract data for Jan, Feb, Mar, April
        limit: the number of records to be returned via the query
    Returns:
        url_content: the content of the URL+query

    NOTE: Check the data_multithreading.py implementation for a multithreading approach for this function. 
    """

    url_content = []
    for month in months:
        url = raw_url+query+" AND date_extract_m(tpep_pickup_datetime) = "+str(month)+" LIMIT "+str(limit)
        r = requests.get(url)  #total rows = 131165043
        url_content.append(r.content)
    return url_content


def generate_df(url_content: str, date_cols: list = ['tpep_pickup_datetime', 'tpep_dropoff_datetime'], 
                cols_to_use: str | list = 'default'):
    """
    This function creates the raw dataframe by extracting 'limit' number of rides randomly from each month in months and outputs
    a single dataframe with the NYC rides data.
    Args:
        url_content: output of the extract_utl function. List of responses from the URL. 
        raw_url: the base URL where the data resides
        query: SQL query to extract the data
        months: a list of months e.g. [1,2,3,4] if you want to extract data for Jan, Feb, Mar, April
        limit: the number of rides that should be extracted randomly from each month. Note: the Socrata API ensures that the limit is 
                executed randomly. Check https://dev.socrata.com/foundry/data.cityofnewyork.us/uacg-pexx for more details.
        date_cols: list of datetime columns
        cols_to_use: only these columns will be imported from the combined URL. If 'default' then the pre-selected 
                columns will be selected.
    Returns:
        A pandas dataframe with all the NYC taxi records extracted from the url
    """
    
    if cols_to_use == 'default':
        cols = ['vendorid', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
               'passenger_count', 'trip_distance', 'pickup_longitude',
               'pickup_latitude', 'store_and_fwd_flag',
               'dropoff_longitude', 'dropoff_latitude'] 
               # 'pulocationid', 'dolocationid' -- these cols don't have any data for Jan to June. 
               # the exact pickup and dropoff lat/lons were replaced by these locationid cols for the July to Dec period. 
    else:
        cols = cols_to_use
        
    df_list= []
    #url_content = extract_url(raw_url, query, months, limit = limit)

    for i, j in enumerate(url_content):
        df_list_item = pd.read_csv(io.StringIO(j.decode('utf-8')), 
                                usecols = cols, parse_dates = date_cols)
        df_list.append(df_list_item)

    df = pd.concat(df_list, axis='rows', ignore_index=True) 
    df = df.rename(columns={'tpep_pickup_datetime': "pickup_datetime", 
                                          'tpep_dropoff_datetime': "dropoff_datetime"})
    return df


def haversine(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    Calculates the haversine distance between pairs of (lat, lon) points using the haversine distance formula
    Refer to https://janakiev.com/blog/gps-points-distance-python/.
    For more accurate distance we should use the geopy.geodesic distance which calculates the shortest distance between two 
    points on the earth's surface taking into account the ellipsoid nature of the earth's shape. But the geopy.geodesic calculations
    take time and for a small area like NYC where maximum of the taxi trip durations are below 30 miles we can use the haversine
    function to save considerable amount of time and not lose much of the accuracy. I compared the values from both the functions
    and 99.8% of the times the difference in those values is less than 0.37 miles. So, using haversine here.
    
    If you want to use the more accurate geopy.geodesic function then simply import the function as, 
    -- from geopy.distance import geodesic
    and use it in the prepare_dataframe function below instead of the haversine function.

    Args:
        lat1: pickup latitude -- can be single value or an array of values
        lon1: pickup longitude -- can be single value or an array of values
        lat2: dropoff latitude -- can be single value or an array of values
        lon2: dropoff longitude -- can be single value or an array of values
    Returns:
        Haversine distance between pickup (lat, lon) and dropoff (lat, lon) locations -- single value or array, based on inputs
    """

    R = 6372800  # Earth radius in meters
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2) 
    dphi       = np.radians(lat2 - lat1)
    dlambda    = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2) ** 2

    # returning distance in miles (so converting meters to miles)
    return 0.000621371 * 2* R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def prepare_dataframe(df: pd.DataFrame, nyc_long_limits: tuple = (-74.257159, -73.699215), 
                        nyc_lat_limits: tuple = (40.471021, 40.987326)):
    """
    Prepare the raw dataframe for further analysis. Specific use case: NYC trip analysis
    e.g. removes outliers, adds datetime helper columns such as weekdays, holidays, etc., and adds distance column.
    Args:
        df: raw input dataframe to be worked on
        nyc_long_limits: max and min longitudes defining the city limits of NYC wihtin which to restrict the data, passed as a tuple (min, max)
        nyc_lat_limits: max and min latitudes defining the city limits of NYC wihtin which to restrict the data, passed as a tuple (min, max)
    Returns:
        dataframe with added columns, removed outliers
    """

    # Verifying the correct pickup and dropoff datetime columns
    if 'pickup_datetime' and 'dropoff_datetime' not in df:
        print("renaming pickup and dropoff datetime cols")
        df.rename(columns = {'tpep_pickup_datetime': "pickup_datetime", 
                                          'tpep_dropoff_datetime': "dropoff_datetime"}, inplace = True)
     
    # Calculating the y variable which is trip duration in seconds
    if 'trip_duration' not in df:
        print("Adding the trip duration col (unit: seconds)")
        df.loc[:, 'trip_duration'] = (df['dropoff_datetime'] 
                                            - df['pickup_datetime'])/np.timedelta64(1,'s')
        
    cols = ['vendorid', 'pickup_datetime', 'dropoff_datetime',
       'passenger_count', 'pickup_longitude', 'pickup_latitude', 'store_and_fwd_flag',
       'dropoff_longitude', 'dropoff_latitude', 'trip_duration']
    
    df = df.loc[:, cols]
    
    # Check for NULL values
    if df.isnull().sum().sum() !=0:
        print("There are NULL values in the dataset. You'll have take of the null values separately, this function doesn't deal \
              with Null value")

    # Adding extra datetime columns 
    print("Adding datetime cols, holidays, weekdays, etc. ")
    df.loc[:, 'pickup_date'] = pd.to_datetime(df['pickup_datetime'].dt.date)
    df.loc[:,'pickup_month'] = df['pickup_datetime'].dt.month
    df.loc[:,'pickup_day'] = df['pickup_datetime'].dt.day  
    df.loc[:,'pickup_hour'] = df['pickup_datetime'].dt.hour
    df.loc[:,'pickup_weekday'] = df['pickup_datetime'].dt.day_name()
    df["vendorid"] = df["vendorid"].astype('category')
    
    # Adding weekday variable 
    df['pickup_weekday'] = pd.Categorical(df['pickup_weekday'], 
                                                categories= ['Monday','Tuesday','Wednesday','Thursday',
                                                        'Friday','Saturday', 'Sunday'], ordered = True)   

    # Adding holidays column to indicate whether a day was a holiday as per the US calendar or not
    cal = calendar()
    holidays = cal.holidays(start =  df['pickup_datetime'].dt.date.min(), 
                            end =  df['pickup_datetime'].dt.date.max())
    df.loc[:,'holiday'] = 1 * pd.to_datetime(df.loc[:, 'pickup_datetime'].dt.date).isin(holidays) 

    # Add haversine distance
    print("adding the haversine distance")
    df.loc[:,'distance_hav'] = haversine(df['pickup_latitude'], df['pickup_longitude'], 
                                    df['dropoff_latitude'], df['dropoff_longitude']) 
    
    # Add trip direction (compass bearing)
    ## Let's add the bearing of each trip which simply means the overall direction in which the taxi 
    ##.. travelled from the pickup point to the dropoff point.
    ## The convention followed here is such the North is denoted as 0 degrees, East as 90 degrees, 
    ##.. South as 180 degrees, _West as 270 degrees and circle back to North as 360 degrees.

    print("adding the bearing direction")
    df.loc[:,'bearing'] = bearing(df['pickup_latitude'], df['pickup_longitude'], 
                            df['dropoff_latitude'], df['dropoff_longitude'])

    # REMOVING OUTLIERS
    print("removing outliers")
    # Since passenger count cannot be 0, assume the most common value (which is 1 for the NYC taxi dataset)
    df.loc[df.passenger_count == 0, 'passenger_count'] = df.passenger_count.value_counts().idxmax()
    df = df.loc[(df.trip_duration > 60) & (df.trip_duration <= np.percentile(df.trip_duration, 99.8))
                   & (df.pickup_longitude.between(nyc_long_limits[0], nyc_long_limits[1]))
                   & (df.dropoff_longitude.between(nyc_long_limits[0], nyc_long_limits[1]))
                   & (df.pickup_latitude.between(nyc_lat_limits[0], nyc_lat_limits[1]))
                   & (df.dropoff_latitude.between(nyc_lat_limits[0], nyc_lat_limits[1]))
                   & (df.distance_hav > 0)
                   & (df.distance_hav <= np.percentile(df.distance_hav, 99.8))]

    print("returning the prepared df")

    return df


def bearing(lat1: float, lon1: float, lat2: float, lon2: float):
    """
    This function calculates the direction of the trip from the pickup point towards the dropoff point. 
    Args:
        lat1: pickup latitude -- can be single value or an array of values
        lon1: pickup longitude -- can be single value or an array of values
        lat2: dropoff latitude -- can be single value or an array of values
        lon2: dropoff longitude -- can be single value or an array of values
    Returns:
        compass_bearing: the radial direction such that N is 0, E is 90, S is 180 and N after a complete circle is 360. 
    """

    lat_p = np.radians(lat1)
    lat_d = np.radians(lat2)
    
    long_diff = np.radians(lon1 - lon2)
    
    x = np.sin(long_diff) * np.cos(lat_d)
    y = np.cos(lat_p) * np.sin(lat_d) - (np.sin(lat_p) * np.cos(lat_d) * np.cos(long_diff))

    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (-initial_bearing - 360)%360
    
    return compass_bearing


def assign_taxi_zones(df: pd.DataFrame, lon_var: str, lat_var: str, locid_var: str, 
                        taxi_zones_file: str = '../data/external/taxi_zones_shape/taxi_zones.shp') -> gpd.GeoDataFrame:
    """
    Given a latitude and longitude, assign that point to a taxi zone. 
    Args:
        df: input pandas dataframe to be used for the latitude and longitude data
        lon_var: longitude column in the df
        lat_var: latitude column in the df
        locid_var: name of the column to be assigned to the LocatioId column in the returned geopandas df
        taxi_zones_file: taxi_zones.shp file location to be read into the geopanads df
    Returns:
        GeoPandas dataframe with locationIds
    """
    
    # make a copy since we will modify lats and lons
    localdf = df[[lon_var, lat_var]].copy()
    
    # missing lat lon info is indicated by nan. Fill with zero
    # which is outside New York shapefile. 
    localdf[lon_var] = localdf[lon_var].fillna(value=0.)
    localdf[lat_var] = localdf[lat_var].fillna(value=0.)
    
    shape_df = gpd.read_file(taxi_zones_file)
    shape_df.drop(['OBJECTID', "Shape_Area", "Shape_Leng"], axis=1, inplace=True)
    shape_df = shape_df.to_crs(pyproj.CRS('epsg:4326') )

    try:
        local_gdf = gpd.GeoDataFrame(
                localdf, crs = pyproj.CRS('epsg:4326') ,
                geometry = [Point(xy) for xy in
                            zip(localdf[lon_var], localdf[lat_var])
                            ]
                        )

        local_gdf = gpd.sjoin(local_gdf, shape_df, 
                                how = 'left', op = 'within')

        return local_gdf.LocationID.rename(locid_var)

    except ValueError as ve:
        print(ve)
        print(ve.stacktrace())
        series = localdf[lon_var]
        series = np.nan
        return series

    
def add_borough_names(taxi_zones_file: str, gdf: pd.DataFrame) -> None:
    """
    Add the nyc borough names to the gdf
    Args:
        taxi_zones_file: taxi_zones.shp file location to be used to map the borough info
        gdf: the Geopandas dataframe which has pickup and dropoff taxizone ids
    Returns:
        None. Adds the borough names inplace in the gdf
    """

    # Adding borough information
    shape_df = gpd.read_file(taxi_zones_file)
    shape_df.drop(['OBJECTID', "Shape_Area", "Shape_Leng"], axis=1, inplace=True)
    shape_df = shape_df.to_crs(pyproj.CRS('epsg:4326') )

    zones_ids = shape_df['LocationID'].to_list()
    borough_nyc_shp = shape_df['borough'].to_list()
    borough_zone_dict = dict(zip(zones_ids, borough_nyc_shp))

    gdf['pickup_borough'] = gdf['pickup_taxizone_id'].map(borough_zone_dict)
    gdf['dropoff_borough'] = gdf['dropoff_taxizone_id'].map(borough_zone_dict)