import requests
from queue import Queue
from threading import Thread
from timeit import default_timer as timer


NUM_THREADS = 6
q = Queue()


def extract_url(raw_url, query, limit):
    """
    This function takes the raw_url, query (SQL), months and limit to define the html query to request data from the city of
    new york's site.
    """

    global q
    global url_content

    while True:
        month = q.get()
        print(f"Month {month}: Running")
        url = raw_url+query+" AND date_extract_m(tpep_pickup_datetime) = "+str(month)+" LIMIT "+str(limit)
        r = requests.get(url)
        url_content.append(r.content)
        print(f"Task {month}: Request successful. Status: {r.status_code}")
   
        q.task_done()


def main(raw_url: str, query: str, months: list, limit: int):
    for month in months:
        q.put(month)

    for t in range(NUM_THREADS):
        worker = Thread(target=extract_url, kwargs= {'raw_url': raw_url, 'query': query, 'limit': limit})
        worker.daemon = True
        worker.start()

    q.join()


if __name__ == '__main__':

    raw_url = "https://data.cityofnewyork.us/resource/uacg-pexx.csv?"
    query = "$query= SELECT * WHERE pickup_longitude IS NOT NULL AND pickup_latitude IS NOT NULL"
    months = [1, 2, 3, 4, 5, 6]
    limit = 700_000

    url_content = []
    start = timer()
    main(raw_url, query, months, limit)
    end = timer()
    print(f'time in seconds {end - start}')
    print(f"length of url content {len(url_content)}")

# does 6 months 700_000 rows in ~85s