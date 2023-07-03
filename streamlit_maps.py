# Author: pratha19/Github

"""
Creates Interactive Visulaization of NYC taxi trips (pickup and dropoffs) using Bokeh. 
"""

import sys
sys.path.append('NYC')

# Importing the required packages
import pandas as pd
from NYC.data_fetch_n_save import NycUrlInfo, FetchNYCRaw, read_parquet_file
from NYC.data_fetch_n_save import AddFeatWrangle
from NYC.taxi_zone_async import AddShapeFileZones
from NYC.streamlit_general_maps import StreamlitMaps
import streamlit as st


# ===========================================================================================================


st.set_page_config(page_title = "NYC taxi trips maps", layout = 'wide', 
                menu_items = {'About': "# Interactive visualization of NYC taxi trips..!"})


@st.cache_data
def read_file(fname):
    """ 
    Given a filename, return the contents of that file
    """
    try:
        with open(fname, 'r') as f:
            # It's assumed our file contains a single line, with our API key only
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found. Please upload a file named apikey.txt which contains your Google API key" % fname)


@st.cache_data
def get_raw_data(raw_url: str, query: str, months: list[int], limit: int, nyc_sql_file_loc: str, fetch_if_exists: bool = True):
    '''
    Read raw data from the raw_url using query. Check the documentation for FetchNYCRaw.fetch_data_from_url() for more details. 
    '''
    nyc_url_info = NycUrlInfo(raw_url, query, months, limit)
    return FetchNYCRaw(nyc_url_info).fetch_data_from_url(nyc_sql_file_loc, fetch_if_exists)


@st.cache_data
def prepare_data(nyc: pd.DataFrame, nyc_raw_file_loc: str, fetch_if_exists: bool = True):
    '''
    Clean and add new features to the raw nyc trips dataframe. Check the documentation for AddFeatWrangle.fetch_prepared_data() for more details. 
    '''
    out_df = AddFeatWrangle(nyc).fetch_prepared_data(nyc_raw_file_loc, fetch_if_exists)
    out_df['trip_duration_minutes'] = out_df['trip_duration'] / 60 # Adding trip duration in minutes. The 'trip_duration' col is in seconds. 
    return out_df


@st.cache_data
def process_data(nyc: pd.DataFrame, nyc_zone_file_loc: str, fetch_if_exists: bool = True,
                taxi_zones_file: str = 'data/external/taxi_zones_shape/taxi_zones.shp', 
                transform_merc = True):
    '''
    Add taxi zone shape (zone ID, zone name) by joining the nyx taxi trip data to nyc's taxi zone shapefile. 
    Check the documentation for AddShapeFileZones.add_zone_info() for more details. 
    '''
    return AddShapeFileZones(raw_df = nyc).add_zone_info(nyc_zone_file_loc = nyc_zone_file_loc, 
                                                        fetch_if_exists = fetch_if_exists, 
                                                        taxi_zones_file = taxi_zones_file,
                                                        transform_merc = transform_merc)


@st.cache_data
def loc_read_parquet_file(file_location: str) -> pd.DataFrame:
    '''
    Read a parquet file and return as pandas dataframe. Check the read_parquet_file() for more details. 
    '''
    return read_parquet_file(file_location)


READ_SAVED_FILE_DIRECTLY = False # Instead of getting the data from the URL and preparing and processing it, get the stored file. Use True if not changing any raw data fetch query params. 
FETCH_IF_EXISTS = not READ_SAVED_FILE_DIRECTLY
FINAL_FILE_LOCATION = 'data/processed/nyc_with_zones_streamlit.parquet' # Location to store to or read from the final processed file

RAW_URL = "https://data.cityofnewyork.us/resource/uacg-pexx.csv?" # NYC TLC URL to fetch the yello taxi trips data
QUERY = "$query= SELECT * WHERE pickup_longitude IS NOT NULL AND pickup_latitude IS NOT NULL" # Query for filtering the data
MONTHS = [1, 2, 3, 4, 5, 6] # fetch taxi trips for these months in 2016. The exact pickup and dropoff lat/lon info is available only for these months
LIMIT = 300_000 # number of trips to fetch randomly within each month

NYC_SQL_FILE_LOC = 'data/raw/nyc_streamlit_2016_raw_sql.parquet' # Location to store or read from the fetched raw data
NYC_RAW_FILE_LOC = 'data/interim/1_nyc_streamlit_raw_cony.parquet' # Location to store or read from the prepared data (some processing and adding cols to raw data)
TAXI_ZONES_FILE = 'data/external/taxi_zones_shape/taxi_zones.shp' # Location to read the NYC taxi zone shape data

IS_GMAPS = False # If True, then plot on Google maps using Google Maps API. You need to add your API key in a file apikey.text. Else use tile_providers from bokeh.


# ===========================================================================================================


if __name__ == '__main__':

    api_key = None
    if IS_GMAPS:
        # Reading in Google API key from text file "apikey"
        fname = 'apikey.txt'
        api_key = read_file(fname)
        
    st.title(f":blue[NYC taxi trips in 2016 (Jan to June)]")

    if READ_SAVED_FILE_DIRECTLY:
        print('Reading pre-saved processed file..')
        nyc = loc_read_parquet_file(FINAL_FILE_LOCATION)
    else:
        # Getting raw NYC taxi data
        nyc = get_raw_data(RAW_URL, QUERY, MONTHS, LIMIT, NYC_SQL_FILE_LOC, fetch_if_exists = FETCH_IF_EXISTS)

        # Processing and adding new feats
        nyc = prepare_data(nyc, NYC_RAW_FILE_LOC, fetch_if_exists = FETCH_IF_EXISTS)

        # Adding taxi zones and mercator transformed lat/lons
        nyc = process_data(nyc, FINAL_FILE_LOCATION , fetch_if_exists = FETCH_IF_EXISTS, 
                        taxi_zones_file = TAXI_ZONES_FILE, transform_merc = True)

    print(f"Number of trips in NYC data: {nyc.shape[0]}")
    nyc.dropna(inplace = True)

    # ---------------------------------------------------------------------------------------------------------------------

    # Select how you want to visualize the trips
    map_options = ["Pickups and Dropoffs as circles on the same chart with lines between them", 
                "Pickups on left chart and Dropoffs on right chart as circles"]

    selected_map = st.sidebar.radio(label = f":blue[Select an option for displaying the trips:]", 
                    options = map_options, index = 0)

    streamlit_maps = StreamlitMaps(nyc)

    if selected_map == map_options[0]:
        # Lines bokeh chart
        layout = streamlit_maps.streamlit_lines()
        st.bokeh_chart(layout, use_container_width = False)
        streamlit_maps.add_notes_lines()
    else:
        # Points bokeh chart
        intrc_trips_loc_p = streamlit_maps.streamlit_points(is_gmaps = IS_GMAPS, api_key = api_key)
        st.bokeh_chart(intrc_trips_loc_p, use_container_width = False)
        streamlit_maps.add_notes_points()