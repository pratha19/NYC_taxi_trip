This project involves predicting the yellow taxi trip duration between any two locations X and Y within NYC. 

### Instructions on creating the virtual environment for this project
* Create Conda environment
    * Assuming you already have [VSCode](https://code.visualstudio.com/download) and [anaconda](https://www.anaconda.com/download) installed. If not, please find the links to install those first. If you don’t worry about storage space and do not want to install the basic dependencies yourself, you can download the Anaconda distribution, otherwise, just go with Miniconda. 
    * See the `environment.yml' file for a list of all dependencies. If you need an additional package to do additional work, please add it here. 
    * Create a new Conda environment using the `environment.yml` file. Add it to Jupyter kernel. And install Jupyter lab (you can also replace this with Jupyter notebook if you prefer that)
    ```> conda env create -f environment.yml
    > conda activate <env-name>
    > python -m ipykernel install --user --name=<custom-kernel-name>
    > jupyter lab 
    ```
    * Open jupyter notebook and select the above Python interpreter
    * If you need to update the environment at any stage, say you need to add a new dependency or change an existing one, edit the `environment.yml` file and do;  
    ```
    > conda env update --file environment.yml --prune
    ```
* For my dev work, I prefer to edit the docs, scripts, etc. files in VSCode, and also run the .py files in the terminal. I use Jupyter lab only for jupyter notebooks.

### The project covers the following topics:
[EDA](https://nbviewer.org/github/pratha19/NYC_taxi_trip/blob/pp_nyc_trip_blog_1_1/notebooks/NYC_EDA.ipynb#2)     
- Raw Data import    
- External data imports     
- Data cleaning    
- Visualizations
- Data wranging based on EDA observations
- Inferences  
- Preparing data attributes for modeling

[ML] -- to be added in the next blog post

### Introduction: 
This project is inspired by the [NYC taxi trip duration challenge](https://www.kaggle.com/c/nyc-taxi-trip-duration/overview) hosted on Kaggle. But instead of using the train and test data from Kaggle taxi data was queried directly from the city of NY's website. Only Yellow taxi trips were considered for this project. More information about different taxi options available in NYC can be found [here](https://www1.nyc.gov/site/tlc/vehicles/get-a-vehicle-license.page).

The motive of the project is to identify the main factors influencing the daily taxi trips of New Yorkers. The taxi trips data is taken from the NYC Taxi and Limousine Commission (TLC), and it includes pickup time, pickup and dropoff geo-coordinates, number of passengers, and several other variables. The taxi trips considered in this project are only for the year 2016 and only those trips will be considered whose exact pickup and dropoff geo-coordinates are available. 

Policy researchers at the TLC and NYC can use the project observations and ML models to observe changing trends in the industry and make informed decisions regarding transportation planning within the city.

### File descriptions:

-- data/raw_data/nyc_2016_raw_sql.csv. Contains around 4.2M trips within New York City (NYC) taken in 2016. 
NOTE: None of the data files were uploaded in this repo because of their large size but if you run the EDA notebook by cloning this repositories all the data files will be populated in the correct sub-folders.

-- The data includes the first 6 months of the year 2016 only because the exact pickup and dropoff
locations were available for the first 6 months only. It was pulled from the new york city's website using a query.

### Data fields:

- vendor_id - a code indicating the provider associated with the trip record

- pickup_datetime - date and time when the meter was engaged

- dropoff_datetime - date and time when the meter was disengaged

- passenger_count - the number of passengers in the vehicle (driver entered value)

- pickup_longitude - the longitude where the meter was engaged

- pickup_latitude - the latitude where the meter was engaged

- dropoff_longitude - the longitude where the meter was disengaged

- dropoff_latitude - the latitude where the meter was disengaged

- store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before
sending to the vendor because the vehicle did not have a connection to the server
- Y=store and forward; N=not a store and forward trip

- trip_distance - distance of the trip recorded in miles (was removed because this field won't be
available for unseen trips/test data)
