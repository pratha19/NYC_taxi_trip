import data_prep
import pathlib
import pandas as pd
import time
import data_multithreading


def time_to_run(func):
    def wrapper_function(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args,  **kwargs)
        print(f"time to run function {func.__name__}: {time.perf_counter() - t0 :.5f} seconds")
        return result
    return wrapper_function
  

def read_parquet_file(file_loc: str) -> pd.DataFrame:
    """
    Reads parquet file into a pandas dataframe.

    Arguments:
        file_loc (str): location of the parquet file
    Returns:
        pandas dataframe
    """
    out_df = pd.read_parquet(file_loc) #, parse_dates = ['pickup_datetime', 'dropoff_datetime']) -- if file was .csv
    out_df = out_df.loc[:, ~out_df.columns.str.match('Unnamed')]
    return out_df


def save_parquet_file(in_df: pd.DataFrame, file_loc: str):
    """
    Saves a pandas dataframe into a parquet file in a local directory.

    Arguments:
        in_df (pandas dataframe): dataframe to be saved
        file_loc (str): location in the local directory where to save the parquet file (e.g. 'repo_dir/folder1/folder2/file.parquet')
    """
    print(f"Saving the downloaded file to {file_loc}")
    in_df.to_parquet(file_loc, index = False) 


class NycUrlInfo:
    """
    Stores information for FetchNYCRaw class. 
    """

    def __init__(self, raw_url: str, query: str, months: list[int], limit: int):
        self.raw_url = raw_url
        self.query = query
        self.months = months
        self.limit = limit


class FetchNYCRaw:
    """
    An API can be used to access the data from city of new york's website; the data can be filtered using 
    SQL queries. More information on how to extract the data from city of new york's site can be found 
    [here](https://dev.socrata.com/foundry/data.cityofnewyork.us/uacg-pexx)

    Fetch query parameters information from the NycUrlInfo class. 
    """

    def __init__(self, nyc_url_info: NycUrlInfo) -> None:
        self.nyc_url_info = nyc_url_info
        
    def loc_read_parquet_file(self, nyc_sql_file_loc) -> pd.DataFrame:
        return read_parquet_file(nyc_sql_file_loc)

    def loc_save_parquet_file(self, nyc_sql, nyc_sql_file_loc) -> None:
        save_parquet_file(nyc_sql, nyc_sql_file_loc)
    
    @time_to_run
    def fetch_data_from_url(self, nyc_sql_file_loc: str, fetch_if_exists: bool = False) -> pd.DataFrame:
        """
        Fetch content from URL and return the data in a dataframe. 

        Arguments:
            nyc_sql_file_loc (str): location where to store the fetched data as a parquet file. 
            fetch_if_exists (bool): if True, then the data will be fetched even if the nyc_sql_file_loc already exists. 
                Otherwise, the existing file at nyc_sql_file_loc will be returned instead of reading new content from the URL.
        
        Returns:
            pandas dataframe
        """

        nyc_sql_path = pathlib.Path(nyc_sql_file_loc)

        if nyc_sql_path.exists() and not fetch_if_exists :
            print(f"Loading the pre-existing file {nyc_sql_file_loc}")
            return self.loc_read_parquet_file(nyc_sql_file_loc) 
        else: 
            print(f"{nyc_sql_file_loc} file doesn't exist, so we need to download it from the city of new york's website first.")
            url_content = data_multithreading.get_content(self.nyc_url_info.raw_url, self.nyc_url_info.query, 
                                                          self.nyc_url_info.months, self.nyc_url_info.limit)
            
            return self.generate_df_from_content(url_content, nyc_sql_file_loc) 
            
    @time_to_run
    def generate_df_from_content(self, url_content: str, nyc_sql_file_loc: str = None, 
                                save_file: bool = True) -> pd.DataFrame: 
        """
        Generates pandas dataframe from URL content

        Arguments:
            url_content (str): the URL content to generate pandas dataframe from. 
            nyc_sql_file_loc (str, [optiona], default = None): location to save the parquet file to.
            save_file (bool, [optiona], default = True): if True, save the dataframe as a parquet file to nyc_sql_file_loc.

        Returns:
            pandas dataframe
        """
        nyc_sql = data_prep.generate_df(url_content)
        if save_file: self.loc_save_parquet_file(nyc_sql, nyc_sql_file_loc)
        return nyc_sql


class AddFeatWrangle:
    """Prepares data for further analysis. Adds new features, cleans the data, etc."""

    def __init__(self, raw_df: pd.DataFrame) -> None:
        self.nyc_sql = raw_df

    def loc_read_parquet_file(self, nyc_raw_file_loc: str) -> pd.DataFrame:
        return read_parquet_file(nyc_raw_file_loc)

    def loc_save_parquet_file(self, nyc_raw: pd.DataFrame, nyc_raw_file_loc: str) -> None:
        save_parquet_file(nyc_raw, nyc_raw_file_loc)
    
    @time_to_run
    def fetch_prepared_data(self, nyc_raw_file_loc: str, fetch_if_exists: bool = False, save_file: bool = True):
        """
        Fetch and return the trip dataframe after doing some pre-processing, adding features, etc.  

        Arguments:
            nyc_raw_file_loc (str): location where to store the pre-processed dataframe as a parquet file. 
            fetch_if_exists (bool, [optional], default = False): if True, then the data will be fetched 
                even if the nyc_raw_file_loc already exists. Otherwise, the existing file at nyc_raw_file_loc will 
                be returned instead of reading new content from the URL.
            save_file (bool, [optional], default = True): If True, save the pre-processed dataframe as a parquet file at nyc_raw_file_loc.
        Returns:
            Pre-processed data as pandas dataframe
        """

        nyc_raw_path = pathlib.Path(nyc_raw_file_loc)

        if nyc_raw_path.exists() and not fetch_if_exists :
            print(f"Loading the pre-existing file {nyc_raw_file_loc}")

            return self.loc_read_parquet_file(nyc_raw_file_loc)
            
        else: 
            print(f" {nyc_raw_file_loc} file doesn't exist; we need to carry out the wrangling and cleaning ops")
            nyc_raw = data_prep.prepare_dataframe(self.nyc_sql)

            # saving prepared data to interim datasets
            if save_file: self.loc_save_parquet_file(nyc_raw, nyc_raw_file_loc)
            
            return nyc_raw


class AddShapeFileZones:
    """
    Joins the zone geometry file with the nyc trips file, mapping each pickup and dropoff lat/lon into a nyc tazi zone id. 
    """

    def __init__(self, raw_df: pd.DataFrame) -> None:
        self.nyc_raw = raw_df

    def loc_read_parquet_file(self, nyc_zone_file_loc: str) -> pd.DataFrame:
        return read_parquet_file(nyc_zone_file_loc)

    def loc_save_parquet_file(self, nyc_raw: pd.DataFrame, nyc_zone_file_loc: str) -> None:
        save_parquet_file(nyc_raw, nyc_zone_file_loc)
    
    @time_to_run
    def add_zone_info(self, nyc_zone_file_loc: str, fetch_if_exists: bool = False, save_file: bool = True,
                     taxi_zones_file: str = '../data/external/taxi_zones_shape/taxi_zones.shp') -> pd.DataFrame:
        """
        Adds taxi zone id name, and NYC borough name, for the nyc trip pickup and dropoff lat lon points. 
        
        Arguments:
            nyc_zone_file_loc (str): location to save the end result as a parquet file.
            fetch_if_exists (bool, [optional], default = False): if False, fetch `nyc_zone_file_loc` if already exists. 
                Or else, if True, add the zone information on the nyc_raw df and save in `nyc_zone_file_loc`
            save_file (bool, [optional], default = True): If True, save the end result dataframe as a parquet file.
            taxi_zones_file (str, [optional], default = '../data/external/taxi_zones_shape/taxi_zones.shp'): 
                location of the file which includes the zone geometry information for NYC. 

        Returns:
            Dataframe with added information for NYC taxi zone ids for pickup and dropoff locations, as well as NYC borough names. 
        """

        nyc_zone_path = pathlib.Path(nyc_zone_file_loc)

        if nyc_zone_path.exists() and not fetch_if_exists :
            print(f"Loading the pre-existing file {nyc_zone_file_loc}")

            return self.loc_read_parquet_file(nyc_zone_file_loc)
            
        else: 
            print(f" {nyc_zone_file_loc} file doesn't exist; we need to carry out the lat/lon to zone mapping ops")
            nyc = self.nyc_raw.copy() # changing dataframe to indicate a change

            # mapping pickup locations 
            print('Mapping the pickup locations')
            nyc['pickup_taxizone_id'] = data_prep.assign_taxi_zones(df = nyc, 
                                        lon_var = "pickup_longitude", lat_var = "pickup_latitude", 
                                        locid_var = "pickup_taxizone_id", 
                                        taxi_zones_file = taxi_zones_file)

            # mapping dropoff locations
            print('Mapping the dropoff locations')
            nyc['dropoff_taxizone_id'] = data_prep.assign_taxi_zones(df = nyc, 
                                        lon_var = "dropoff_longitude", lat_var = "dropoff_latitude", 
                                        locid_var = "dropoff_taxizone_id",
                                        taxi_zones_file = taxi_zones_file)
            
            # Adding borough names
            print("Adding borough names")
            data_prep.add_borough_names(taxi_zones_file, nyc)

            # saving prepared data to interim datasets
            if save_file: self.loc_save_parquet_file(nyc, nyc_zone_file_loc)
            
            return nyc


from shapely.geometry import Point
import scipy.cluster.hierarchy as shc
import pyproj
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt


class AddClusterLabels:
    """
    Clusters the taxi zones into different groups based on proximity to each other as well as other attributes, such as 
    outwards minus inwards traffic count, whether or not it is an airport, etc. 
    """

    def __init__(self, raw_df: pd.DataFrame) -> None:
        self.nyc_raw = raw_df

    def loc_read_parquet_file(self, nyc_zone_file_loc: str) -> pd.DataFrame:
        return read_parquet_file(nyc_zone_file_loc)

    def loc_save_parquet_file(self, nyc_raw: pd.DataFrame, nyc_zone_file_loc: str) -> None:
        save_parquet_file(nyc_raw, nyc_zone_file_loc)

    @staticmethod
    def read_nyc_zone_shape(taxi_zones_file: str) -> gpd.GeoDataFrame:
        shape_df = gpd.read_file(taxi_zones_file).to_crs(pyproj.CRS('epsg:4326') )
        shape_df.drop(['OBJECTID', "Shape_Area", "Shape_Leng"], axis=1, inplace=True)
        
        return shape_df

    def add_zones_geometry_centroids(self, shape_df: gpd.GeoDataFrame) -> None:
        centroidseries = shape_df['geometry'].to_crs('epsg:4326').centroid.to_crs(shape_df.crs)
        zone_lat, zone_lon = [list(t) for t in zip(*centroidseries.map(self.get_lat_lon))]

        shape_df['zone_lat'] = zone_lat
        shape_df['zone_lon'] = zone_lon
    
    @staticmethod
    def get_lat_lon(pt: Point) -> tuple:
        """
        Returns the latitude and longitude of a shapely.geometry.Point. 
        Args:
            pt: shapely.geometry.Point
        Returns:
            (y, x) where y = latitude, x = longitude
        """

        return (pt.y, pt.x) 

    @staticmethod
    def add_out_in_traffic_diff(trips_df, shape_df):

        # Calculating the average count of pickups each hour per zone over the entire period of data
        df_pickup_count = trips_df.groupby(['pickup_taxizone_id', 'pickup_date', 'pickup_hour']).count()['vendorid'].groupby(\
                                    ['pickup_taxizone_id', 'pickup_hour']).mean()

        # Calculating the average count of dropoffs each hour per zone over the entire period of data
        df_dropoff_count = trips_df.groupby(['dropoff_taxizone_id', 'pickup_date', 'pickup_hour']).count()['vendorid'].groupby(\
                                    ['dropoff_taxizone_id', 'pickup_hour']).mean()

        # Transforming the above dataframes in a row vs column format
        df_pickup_count = df_pickup_count.unstack(fill_value = 0).reset_index().rename_axis(None, axis=1)
        df_dropoff_count = df_dropoff_count.unstack(fill_value = 0).reset_index().rename_axis(None, axis=1)
        df_pickup_count.set_index('pickup_taxizone_id', inplace = True)
        df_dropoff_count.set_index('dropoff_taxizone_id', inplace = True)

        # Calculating the difference between the dropoffs and pickups in each each hour...
        df_p_d = pd.DataFrame(range(1, len(shape_df)+1), columns = ['zone_id'])

        df_pickup_count_all = df_p_d.merge(df_pickup_count, how = 'left', left_on = 'zone_id', 
                                        right_index = True).fillna(0).set_index('zone_id')

        df_dropoff_count_all = df_p_d.merge(df_dropoff_count, how='left', left_on = 'zone_id', 
                                        right_index = True).fillna(0).set_index('zone_id')

        zone_diff = df_dropoff_count_all - df_pickup_count_all

        # ... and merging the dataframes together with the nyc_shp dataframe
        nyc_zones = shape_df.merge(zone_diff, how = 'left', left_on = 'LocationID', right_index = True).fillna(0)
        nyc_zones.drop(['zone', 'borough', 'geometry'], axis = 1, inplace = True)
        nyc_zones.set_index('LocationID', inplace = True)

        return nyc_zones

    @staticmethod
    def add_is_airport(nyc_zones: gpd.GeoDataFrame, airport_zones: list[int] = [1 , 132, 138]):
        """
        Adding whether a particular zone is an airport or not (1: Newark airport, 132: JFK), 138: La Guarida)
        """

        df_temp = nyc_zones.reset_index()
        df_temp['is_airport'] = 1 * df_temp.LocationID.isin(airport_zones)
        nyc_zones['is_airport'] = df_temp['is_airport'].values

    @staticmethod
    def normalize_zones_df(nyc_zones: gpd.GeoDataFrame):
        """
        Normalizing the values before clustering
        """

        zones_scaled = normalize(nyc_zones)
        zones_scaled = pd.DataFrame(zones_scaled, columns = nyc_zones.columns)

        # Converting all column names to str
        zones_scaled.columns = zones_scaled.columns.astype(str)

        return zones_scaled

    @staticmethod
    def clustering_zones(zones_scaled, n_clusters = 50, 
                        metric = 'euclidean', linkage = 'ward'):
        """
        Doing the clustering
        """

        cluster_zones = AgglomerativeClustering(n_clusters = n_clusters, metric = metric, 
                                                linkage = linkage)  

        _ = cluster_zones.fit_predict(zones_scaled)
        print(f"Unique cluster labels generated: {len(np.unique(cluster_zones.labels_))}")

        return cluster_zones.labels_

    def assign_cluster_labels(self, nyc_zones, raw_df):
        """
        Adding cluster info back to the df
        """

        nyc_Z = nyc_zones[['cluster_label']].reset_index()
        zone_id = nyc_Z['LocationID'].to_list()
        zone_cluster_labels = nyc_Z['cluster_label'].to_list()
        zone_cluster_dict = dict(zip(zone_id, zone_cluster_labels))

        raw_df['pickup_zone_cluster'] = raw_df['pickup_taxizone_id'].map(zone_cluster_dict)
        raw_df['dropoff_zone_cluster'] = raw_df['dropoff_taxizone_id'].map(zone_cluster_dict)
        print(f"Null values in the df: {raw_df.isna().sum().sum() == 0}")

    def draw_dendogram(self, figsize = (15, 4)):
        """
        Draw the dendogram
        """

        _ = plt.figure(figsize = figsize)  
        _ = plt.title("NYC taxi zones Dendrograms")  

        _ = shc.dendrogram(shc.linkage(self.zones_scaled, method='ward'))

    @time_to_run
    def add_cluster_labels(self, nyc_cluster_file_loc: str, fetch_if_exists: bool = False, save_file = True,
                     taxi_zones_file = '../data/external/taxi_zones_shape/taxi_zones.shp') -> pd.DataFrame:
        """
        Add cluster labels to the NYC taxi trip dataframe. The clustering is based on following features:
            - Zone centroid lat long
            - The net zone traffic (outbound - inbound) per hour of the day
            - Whether a zone is an airport or not
        
        Arguments:
            nyc_cluster_file_loc (str): the location to where save the cluster label added dataframe as parquet file. 
            fetch_if_exists (bool, [optional], default = False): If False, then try to fetch `nyc_cluster_file_loc` file if exists.
                If True, then carry out the clustering ops even if `nyc_cluster_file_loc` exists. 
            save_file (bool, [optional], default = True): Save the cluster labels added end result dataframe as a parquet file as `nyc_cluster_file_loc`
            taxi_zones_file (str, [optional], default = '../data/external/taxi_zones_shape/taxi_zones.shp'):
                The shape file with all zone geometry information for NYC. 

        Returns:
            Cluster labels added pandas dataframe
        """

        nyc_zone_path = pathlib.Path(nyc_cluster_file_loc)

        if nyc_zone_path.exists() and not fetch_if_exists :
            print(f"Loading the pre-existing file {nyc_cluster_file_loc}")
            
            return self.loc_read_parquet_file(nyc_cluster_file_loc)

        else:
            print("Reading the nyc zone shape file")
            nyc_shp = self.read_nyc_zone_shape(taxi_zones_file)

            print("Adding centroid lat and lon for each zone's geometry")
            self.add_zones_geometry_centroids(nyc_shp)

            print("Adding the difference in outgoing and incoming trips per hour each zone")
            nyc_zones = self.add_out_in_traffic_diff(self.nyc_raw, nyc_shp)

            print("Adding whether a particular zone is an airport or not")
            self.add_is_airport(nyc_zones, airport_zones = [1 , 132, 138])

            print("Normalizing the values before clustering")
            self.zones_scaled = self.normalize_zones_df(nyc_zones)

            print("Doing the clustering")
            nyc_zones['cluster_label'] = self.clustering_zones(self.zones_scaled, n_clusters = 50, 
                                                                metric = 'euclidean', linkage = 'ward')

            print("Adding cluster labels back to the original df")
            self.assign_cluster_labels(nyc_zones, self.nyc_raw)

            # saving prepared data to interim datasets
            if save_file: self.loc_save_parquet_file(self.nyc_raw, nyc_cluster_file_loc)

            return self.nyc_raw
