import ray
import asyncio
import time

@ray.remote
def my_sleep(seconds: float):
    # Simulate a long-running task, blocking the thread
    time.sleep(seconds)
    return f"Finished sleeping for {seconds} seconds"


async def submit_task(seconds: float = 5):
    # Submit a task to the Ray cluster
    result = await my_sleep.remote(seconds)
    return result

async def main():
    print("Starting...")
    t0 = time.time()
    r1 = asyncio.create_task(submit_task())
    r2 = asyncio.create_task(submit_task(10))
    r3 = asyncio.create_task(submit_task(7))
    # Wait for all tasks to complete
    r1 = await r1
    print(f"Result 1: {r1}")
    r2 = await r2
    print(f"Result 2: {r2}")
    r3 = await r3
    print(f"Result 3: {r3}")
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")



if __name__ == "__main__":
    ray.init()
    asyncio.run(main())