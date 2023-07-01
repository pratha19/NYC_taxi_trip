import requests
import random
import multiprocessing
from timeit import default_timer as timer


NUM_PROC = 4


def extract_url(raw_url, query, months, limit, url_content = []):
    """
    This function takes the raw_url, query (SQL), months and limit to define the html query to request data from the city of
    new york's site.
    """

    for month in months:
        url = raw_url+query+" AND date_extract_m(tpep_pickup_datetime) = "+str(month)+" LIMIT "+str(limit)
        r = requests.get(url)  
        url_content.append(r.content)


if __name__ == "__main__":

    raw_url = "https://data.cityofnewyork.us/resource/uacg-pexx.csv?"
    query = "$query= SELECT * WHERE pickup_longitude IS NOT NULL AND pickup_latitude IS NOT NULL"
    months = [1, 2, 3, 4]
    limit = 100_000
    url_content = []

    jobs = []
   
    start = timer()
   
    for i in range(NUM_PROC):
        process = multiprocessing.Process(
            target = extract_url,
            args=(raw_url, query, months, limit, url_content)
        )
        jobs.append(process)
       
    for j in jobs:
        j.start()
       
    for j in jobs:
        j.join()

    end = timer()
    print(f'time in seconds {end - start}')

    print(f"length of url content {len(url_content)}")
