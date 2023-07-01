import pathlib
import pandas as pd
import numpy as np
import time
import pyproj
import geopandas as gpd
from shapely.geometry import Point
import multiprocessing as mp


def time_to_run(func):
    def wrapper_function(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args,  **kwargs)
        print(f"time to run function {func.__name__}: {time.perf_counter() - t0 :.5f} seconds")
        return result
    return wrapper_function


def read_parquet_file(file_loc: str) -> pd.DataFrame:
   out_df = pd.read_parquet(file_loc) #, parse_dates = ['pickup_datetime', 'dropoff_datetime']) -- if file was .csv
   out_df = out_df.loc[:, ~out_df.columns.str.match('Unnamed')]
   return out_df


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
    zone_names_nyc_shp = shape_df['zone'].to_list()

    borough_zone_dict = dict(zip(zones_ids, borough_nyc_shp))
    zone_name_id_dict = dict(zip(zones_ids, zone_names_nyc_shp))

    gdf['pickup_borough'] = gdf['pickup_taxizone_id'].map(borough_zone_dict)
    gdf['dropoff_borough'] = gdf['dropoff_taxizone_id'].map(borough_zone_dict)
    gdf['pickup_zone_name'] = gdf['pickup_taxizone_id'].map(zone_name_id_dict)
    gdf['dropoff_zone_name'] = gdf['dropoff_taxizone_id'].map(zone_name_id_dict)


def save_parquet_file(in_df: pd.DataFrame, file_loc: str):
    print(f"Saving the downloaded file to {file_loc}")
    in_df.to_parquet(file_loc, index = False) 


def assign_taxi_zones(df: pd.DataFrame, chunk: int | None = None, 
                    lon_var: str = 'pickup_longitude', lat_var: str = 'pickup_latitude', 
                    locid_var: str = 'pickup_taxizone_id', 
                    taxi_zones_file: str = '../data/external/taxi_zones_shape/taxi_zones.shp',
                    ) -> gpd.GeoDataFrame:
    
    # make a copy since we will modify lats and lons
    localdf = df[[lon_var, lat_var]].copy()
    
    # missing lat lon info is indicated by nan. Fill with zero
    # which is outside New York shapefile. 
    localdf[lon_var] = localdf[lon_var].fillna(value=0.)
    localdf[lat_var] = localdf[lat_var].fillna(value=0.)
    
    shape_df = gpd.read_file(taxi_zones_file)
    shape_df.drop(['OBJECTID', "Shape_Area", "Shape_Leng"], axis=1, inplace=True)
    shape_df = shape_df.to_crs(pyproj.CRS('epsg:4326'))

    try:
        print(f"assigning taxi zones to each location: {lon_var}, {lat_var}, chunk = {chunk}")
        local_gdf = gpd.GeoDataFrame(
                localdf, crs = pyproj.CRS('epsg:4326'),
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


def transform_to_mercator(df: pd.DataFrame, chunk: int | None = None, 
                    lon_var: str = 'pickup_longitude', lat_var: str = 'pickup_latitude', 
                    inProj: pyproj.CRS = pyproj.CRS('epsg:4326'), outProj: pyproj.CRS = pyproj.CRS('epsg:3857'), 
                    ) -> tuple[pd.Series]:
    
    # make a copy since we will modify lats and lons
    localdf = df[[lon_var, lat_var]].copy()
    
    # missing lat lon info is indicated by nan. Fill with zero
    # which is outside New York shapefile. 
    localdf[lon_var] = localdf[lon_var].fillna(value=0.)
    localdf[lat_var] = localdf[lat_var].fillna(value=0.)

    try:
        print(f"transforming from {inProj} to {outProj} coords: {lon_var}, {lat_var}, chunk = {chunk}")
        transform_to_lat_lon = pyproj.Transformer.from_crs(inProj, outProj, always_xy = True)
        ret_df = pd.DataFrame(transform_to_lat_lon.transform(localdf[lon_var], localdf[lat_var])).T
        ret_df.index = localdf.index

        return ret_df

    except ValueError as ve:
        print("--------------ERROR--------------")
        print(ve)
        print(ve.stacktrace())
        series = localdf[lon_var]
        series = np.nan
        return series


class AddShapeFileZones:
    """
    Joins the zone geometry file with the nyc trips file, mapping each pickup and dropoff lat/lon into a nyc tazi zone id. 
    Also transforms the lat lon pairs to Mercator projection. 
    """

    def __init__(self, raw_df: pd.DataFrame) -> None:
        self.nyc_raw = raw_df

    def loc_read_parquet_file(self, nyc_zone_file_loc: str) -> pd.DataFrame:
        return read_parquet_file(nyc_zone_file_loc)

    def loc_save_parquet_file(self, nyc_raw: pd.DataFrame, nyc_zone_file_loc: str) -> None:
        save_parquet_file(nyc_raw, nyc_zone_file_loc)
    
    @staticmethod
    def multi_process_chunks_zone(nyc: pd.DataFrame, apply_func, ignore_index, **kwargs):

        print(kwargs)
        # set the number of processes
        n_proc = mp.cpu_count()

        # this often can't be devided evenly (handle this in the for-loop below)
        chunksize = nyc.shape[0] // n_proc

        # devide into chunks
        proc_chunks = []

        for i_proc in range(n_proc):
            chunkstart = i_proc * chunksize
            # make sure to include the division remainder for the last process
            chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

            proc_chunks.append(nyc.iloc[slice(chunkstart, chunkend)])

        assert sum(map(len, proc_chunks)) == nyc.shape[0], "make sure all data is in the chunks"

        # distribute work to the worker processes
        with mp.Pool(processes = n_proc) as pool:
            # starts the sub-processes without blocking. Pass the chunk to each worker process
            proc_results = [pool.apply_async(apply_func, 
                                            args = (df_chunk, chunk_num),
                                            kwds = kwargs
                                            )
                                            for chunk_num, df_chunk in enumerate(proc_chunks)
                                            ]

            # blocks until all results are fetched
            result_chunks = [r.get() for r in proc_results]

        # concatenate results from worker processes
        results = pd.concat(result_chunks, ignore_index = ignore_index).sort_index()

        assert len(results) == nyc.shape[0] , \
                f"{len(results)}!={nyc.shape[0]}. Make sure we got a result for each coordinate pair. "

        return results 

    @time_to_run
    def add_zone_info(self, nyc_zone_file_loc: str, 
                    fetch_if_exists: bool = False, save_file = True,
                    taxi_zones_file = '../data/external/taxi_zones_shape/taxi_zones.shp', 
                    transform_merc = False):

        nyc_zone_path = pathlib.Path(nyc_zone_file_loc)

        if nyc_zone_path.exists() and not fetch_if_exists:
            print(f"Loading the pre-existing file {nyc_zone_file_loc}")

            return self.loc_read_parquet_file(nyc_zone_file_loc)
            
        else: 
            print(f" {nyc_zone_file_loc} file doesn't exist; we need to carry out the lat/lon to zone mapping ops")
            nyc = self.nyc_raw.copy() # changing dataframe to indicate a change

            # mapping pickup locations 
            print('Mapping the pickup locations')
            nyc['pickup_taxizone_id'] = self.multi_process_chunks_zone(
                                                                    nyc, assign_taxi_zones, ignore_index = False,
                                                                    lon_var = "pickup_longitude", lat_var = "pickup_latitude", 
                                                                    locid_var = "pickup_taxizone_id", taxi_zones_file = taxi_zones_file, 
                                                                    )

            # mapping dropoff locations
            print('Mapping the dropoff locations')
            nyc['dropoff_taxizone_id'] = self.multi_process_chunks_zone(
                                                                    nyc, assign_taxi_zones, ignore_index = False,
                                                                    lon_var = "dropoff_longitude", lat_var = "dropoff_latitude", 
                                                                    locid_var = "dropoff_taxizone_id", taxi_zones_file = taxi_zones_file,
                                                                    )
            
            # Transforming to Mercator projections
            if transform_merc:
                print(f"Transforming the pickup lon/lats....")
                nyc[['pickup_MercatorX', 'pickup_MercatorY']] = self.multi_process_chunks_zone(
                                                nyc, transform_to_mercator, ignore_index = False,
                                                lon_var = "pickup_longitude", lat_var = "pickup_latitude", 
                                                inProj = pyproj.CRS('epsg:4326'), outProj = pyproj.CRS('epsg:3857'),
                                                )

                print(f"Transforming the dropoff lon/lats....")
                nyc[['dropoff_MercatorX', 'dropoff_MercatorY']] = self.multi_process_chunks_zone(
                                                nyc, transform_to_mercator, ignore_index = False,
                                                lon_var = "dropoff_longitude", lat_var = "dropoff_latitude", 
                                                inProj = pyproj.CRS('epsg:4326'), outProj = pyproj.CRS('epsg:3857'),
                                                )

            # Adding borough names
            print("Adding borough names")
            add_borough_names(taxi_zones_file, nyc)

            # saving prepared data to interim datasets
            if save_file: self.loc_save_parquet_file(nyc, nyc_zone_file_loc)
            
            return nyc