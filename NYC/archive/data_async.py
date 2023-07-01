import asyncio
import time
import aiohttp


async def extract_url(s: aiohttp.ClientSession, url: str, month: int) -> str:
    print(f"Month {month}: Running")

    async with s.get(url) as response:
        if response.status != 200:
            response.raise_for_status()
        content = await response.text() # can directly return await response.text() if print 'Done' is not needed
        print(f"Month {month}: Done")
        return content
        

async def get_content(raw_url: str, query: str, months: list[int], limit: int):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for month in months:
            url = raw_url+query+" AND date_extract_m(tpep_pickup_datetime) = "+str(month)+" LIMIT "+str(limit)
            task = asyncio.create_task(extract_url(session, url, month))
            tasks.append(task)

        url_content = await asyncio.gather(*tasks)
        return url_content


if __name__ == '__main__':
    raw_url = "https://data.cityofnewyork.us/resource/uacg-pexx.csv?"
    query = "$query= SELECT * WHERE pickup_longitude IS NOT NULL AND pickup_latitude IS NOT NULL"
    months = [1, 2, 3, 4, 5, 6]
    limit = 700_000

    start = time.perf_counter()
    url_content = asyncio.run(get_content(raw_url, query, months, limit))
    end = time.perf_counter()
    print(f'time in seconds {end - start}')
    print(f"length of url content {len(url_content)}")

# does 6 months 700_000 rows in ~ 44 to 85s
# 54, 48, 53, 54 => mean => 52.25s