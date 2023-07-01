import requests
from threading import Thread
import time


def extract_url(url_content, index, raw_url: str, query: str, month: int, limit: int) -> str:
    """
    This function takes the raw_url, query (SQL), months and limit to define the html query to request data from the city of
    new york's site.
    """

    print(f"Month {month}: Running")
    url = raw_url+query+" AND date_extract_m(tpep_pickup_datetime) = "+str(month)+" LIMIT "+str(limit)
    r = requests.get(url)  
    url_content[index] = r.content
    print(f"Task {month}: Request successful. Status: {r.status_code}")


def get_content(raw_url: str, query: str, months: list, limit: int):
    tasks = []
    url_content = [None]*len(months)

    for index, month in enumerate(months):
        tasks.append(Thread(target = extract_url, args = (url_content, index, raw_url, query, month, limit)))
        tasks[-1].start()

    for task in tasks:
        task.join()

    return url_content


if __name__ == '__main__':

    raw_url = "https://data.cityofnewyork.us/resource/uacg-pexx.csv?"
    query = "$query= SELECT * WHERE pickup_longitude IS NOT NULL AND pickup_latitude IS NOT NULL"
    months = [1, 2, 3, 4, 5, 6]
    limit = 700_000

    start = time.perf_counter()
    url_content = get_content(raw_url, query, months, limit)
    end = time.perf_counter()

    print(f'time in seconds {end - start}')
    print(f"length of url content {len(url_content)}")

# Threading thread does 6 months 700_000 rows in ~80s
# the concurrent threadpool exe,cutor takes around 100s
# 55, 55, 61, 55 => mean => 56.5s