import aiohttp
import aiofiles
import asyncio

lines = [x.strip() for x in open("shared1000").readlines()]
urls = []
for x in lines:
    s3_url = x.strip().split(" ")[-1]
    # print(s3_url)
    parts = s3_url.split("s3://natural-scenes-dataset/")
    # print(parts)
    url = "https://natural-scenes-dataset.s3.amazonaws.com/" + parts[-1]
    # print(url)
    urls.append(url)
    # break
# print(lines[:5])
print("got urls!")



async def download_image(url, session):
    filename = "images/{}".format(url.split("/")[-1])
    async with session.get(url) as response:
        if response.status == 200:
            f = await aiofiles.open(filename, mode='wb')
            await f.write(await response.read())
            await f.close()
        else:
            raise Exception(f'Failed to download {url} due to status: {response.status}')

async def main(url_list):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in url_list:
            task = asyncio.ensure_future(download_image(url, session))
            tasks.append(task)
        await asyncio.gather(*tasks)

asyncio.run(main(urls))

