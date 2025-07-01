import argparse
import asyncio
import datetime
import json
import os
import re
import time
from io import BytesIO
from typing import Tuple, Any, AsyncGenerator, List, Union

import numpy as np
import rasterio
import requests
import torch

from vllm import AsyncEngineArgs, PoolingRequestOutput, PoolingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001

def read_geotiff(file_path: str) -> Tuple[np.ndarray, dict, Tuple[float, float]]:
    """Read all bands from *file_path* and return image + meta info.

    Args:
        file_path: path to image file.

    Returns:
        np.ndarray with shape (bands, height, width)
        meta info dict
    """

    if file_path.startswith("http"):
        response = requests.get(file_path)
        response.raise_for_status()  # Raise an error for bad responses
        file = BytesIO(response.content)
    else:
        file = file_path

    with rasterio.open(file) as src:
        img = src.read()
        meta = src.meta
        try:
            coords = src.lnglat()
        except:
            # Cannot read coords
            coords = None
    return img, meta, coords

def load_example(
        file_paths: List[str],
        mean: List[float] = None,
        std: List[float] = None,
        indices: Union[list[int], None] = None,
):
    """Build an input example by loading images in *file_paths*.

    Args:
        file_paths: list of file paths .
        mean: list containing mean values for each band in the images in *file_paths*.
        std: list containing std values for each band in the images in *file_paths*.
        indices: list of indices to select bands from the images in *file_paths*.

    Returns:
        np.array containing created example
        list of meta info for each image in *file_paths*
    """

    imgs = []
    metas = []
    temporal_coords = []
    location_coords = []

    for file in file_paths[:1]:
        img, meta, coords = read_geotiff(file)

        # Rescaling (don't normalize on nodata)
        img = np.moveaxis(img, 0, -1)  # channels last for rescaling
        if indices is not None:
            img = img[..., indices]
        if mean is not None and std is not None:
            img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean) / std)

        imgs.append(img)
        metas.append(meta)
        if coords is not None:
            location_coords.append(coords)

        try:
            match = re.search(r'(\d{7,8}T\d{6})', file)
            if match:
                year = int(match.group(1)[:4])
                julian_day = match.group(1).split('T')[0][4:]
                if len(julian_day) == 3:
                    julian_day = int(julian_day)
                else:
                    julian_day = datetime.datetime.strptime(julian_day, '%m%d').timetuple().tm_yday
                temporal_coords.append([year, julian_day])
        except Exception as e:
            print(f'Could not extract timestamp for {file} ({e})')

    imgs = np.stack(imgs, axis=0)  # num_frames, H, W, C
    imgs = np.moveaxis(imgs, -1, 0).astype("float32")  # C, num_frames, H, W
    imgs = np.expand_dims(imgs, axis=0)

    return imgs, temporal_coords, location_coords, metas

async def encode(engine: AsyncLLMEngine, geotiff_file: str, req_id, extra_data: int = 0, input_data =None, location_coords=None) -> tuple[
    Any, int, AsyncGenerator[PoolingRequestOutput, None]]:
    mm_data = {"pixel_values": None, "location_coords": location_coords, "temporal_coords": torch.empty(0),
               "geotiff_file": geotiff_file, "extra_data": extra_data, "input_data": input_data}

    prompt = {
        "prompt_token_ids": [1],
        "multi_modal_data": mm_data
    }

    pooling_params = PoolingParams()
    start_time = time.time_ns()
    outputs = engine.encode(prompt, pooling_params, req_id)
    return req_id, start_time, outputs

async def uniform_throughput(engine: AsyncLLMEngine, queue: asyncio.Queue, geotiff_file: str, num_req: int, rps: int, extra_data: int = 0, send_np_array: bool = False) -> None:
    """
    Send requests to the queue at a uniform rate of rps (requests per second).

    Args:
        engine: AsyncLLMEngine instance to handle requests.
        queue: Queue to send requests to.
        geotiff_file: Path or URL to the GeoTIFF file.
        num_req: Number of requests to send.
        rps: Requests per second.
    """
    if send_np_array:
        input_data, _, location_coords, _ = load_example(file_paths=[geotiff_file],
                                                     indices=[1, 2, 3, 8, 11, 12])
        geotiff_file = None
    else:
        input_data = None
        location_coords = None

    for req_id in range(num_req):
        queue_element = asyncio.create_task(encode(engine, geotiff_file, req_id, extra_data, input_data=input_data, location_coords=location_coords))
        await queue.put(queue_element)
        await asyncio.sleep(1.0 / rps)


async def benchmark(num_req: int, engine: AsyncLLMEngine, data_size: int, rps: int, geotiff_file: str, results_dir: str, args: argparse.Namespace):
    """
    Benchmark the geospatial deployment by sending requests and measuring latency.

    Args:
        num_req: Number of inferences to run.
        engine: AsyncLLMEngine instance to handle requests.
        data_size: Size of data to be passed (in bytes).
        rps: Requests per second.
        geotiff_file: Path or URL to the GeoTIFF file.
        results_dir: Directory to save results.
        args: Configuration used for the benchmark.
    """
    queue = asyncio.Queue()
    results_task = asyncio.create_task(process_outputs(queue, args.num_req))

    print("Starting benchmarking...")
    await uniform_throughput(engine, queue, geotiff_file, num_req, rps, data_size, args.send_np_array)

    results = await results_task
    print("Benchmarking completed.")
    save_results(args, results, results_dir)
    summary_results(results)


async def process_outputs(queue: asyncio.Queue[Tuple[int, float, AsyncGenerator[PoolingRequestOutput, None]]], num_req: int):
    """
    Process the outputs from the queue.
    Args:
        queue: Queue containing the results of the requests.
        num_req: Number of prompts to process.
    """
    results = []
    for _ in range(num_req):
        # Wait for the next output
        a = await queue.get()
        print(type(a))
        req_id, send_timestamp, results_generator = await a
        encoded = None
        async for request_output in results_generator:
            encoded = request_output
        receive_timestamp = time.time_ns()
        queue.task_done()
        result = {"send_time_ns": send_timestamp,
                  "receive_time_ns": receive_timestamp,
                  "latency_ns": receive_timestamp - send_timestamp,
                  "id": req_id}
        results.append(result)

    return results

def save_results(args, results, results_dir):
    """
    Save the results to a file in the specified directory.

    Args:
        args: Configuration used for the benchmark.
        results: List of results to save.
        results_dir: Directory to save the results file.
    """
    # Create a directory for results if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    config = vars(args)

    to_save = {"config": config, "results": results}
    timestamp = time.time_ns()
    results_file = os.path.join(results_dir, f"benchmark_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(to_save, f, indent=4)

    print(f"Results saved to {results_file}")


def summary_results(results):
    """
    Summarize the results of the benchmark.

    Args:
        results: List of results to summarize.
    """
    latencies = np.array([result["latency_ns"] for result in results])
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    std_latency = np.std(latencies)
    print("-" * 50)
    print(f"Mean Latency: {mean_latency / 1e6:.2f} ms")
    print(f"Standard Deviation Latency: {std_latency / 1e6:.2f} ms")
    print(f"95th Percentile Latency: {p95_latency / 1e6:.2f} ms")
    print(f"99th Percentile Latency: {p99_latency / 1e6:.2f} ms")
    print(f"Total Requests: {len(results)}")
    print("-" * 50)

def print_config(args):
    """
    Print the configuration used for the benchmark.

    Args:
        args: Configuration used for the benchmark.
    """
    print("-" * 50)
    print("Configuration:")
    print("-" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-" * 50)

async def main(arg: argparse.Namespace):
    """
    Main function to run the benchmarking script.
    """

    print_config(arg)

    # Initialize Ray
    # ray.init(address=arg.ray_address, runtime_env={"pip": arg.ray_pip_requirements, "env_vars": {"HF_HOME": "/data/gfinol/huggingface"}})
    # ray.init(address=args.ray_address, runtime_env={"image_uri": "icr.io/drl-nextgen/vllm/ad-orchestrator-gpu-0.8.1-vllm", "env_vars": {"HF_HOME": "/data/gfinol/huggingface"}})

    # Create a Ray Serve deployment
    # engine_args = AsyncEngineArgs(model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM", skip_tokenizer_init=True, dtype="float32")
    engine_args = AsyncEngineArgs(model="./model", skip_tokenizer_init=True, dtype="float32")
    async_engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Run the benchmark
    await benchmark(arg.num_req, async_engine, arg.data_size, arg.rps, arg.geotiff_file, arg.results_dir, arg)

    print("Exiting...")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument("--ray-address", type=str, help="Ray cluster address", default="ray://localhost:10001")
    parser.add_argument("--num-req", type=int, help="Number of inferences to run", default=10)
    parser.add_argument("--data-size", type=int, help="Size of data to be passed (in bytes)", default=1 * 1024 * 1024)
    parser.add_argument("--sleep-time", type=float, help="Sleep time in preprocessing (seconds)", default=0.0)
    parser.add_argument("--rps", type=int, help="Requests per second", default=1)
    parser.add_argument("--geotiff-file", type=str, help="Path or URL to the GeoTIFF file", required=True)
    parser.add_argument("--results-dir", type=str, help="Directory to save results", default="results")
    parser.add_argument("--ray-pip-requirements", type=str, help="Path to the Ray Pip requirements file", default="requirements.txt")
    parser.add_argument("--ray-deployment-name", type=str, help="Name of the Ray deployment", default="geoserve-benchmark")
    parser.add_argument("--extra-data-size", type=int, help="Extra data size to be passed (in bytes)", default=0)
    parser.add_argument("--send-np-array", action="store_true", help="Send numpy array instead of bytes")

    # Arguments that modify the behavior of the preprocessor
    parser.add_argument("--sleep_distribution", type=str, help="Sleep time distribution", choices=["fixed", "uniform"], default="fixed")

    # Fixed sleep time
    parser.add_argument("--sleep_fixed", type=float, help="Fixed sleep time in preprocessing (seconds), requires sleep_distribution=fixed", default=0.1)

    # Uniform sleep time
    parser.add_argument("--sleep_min", type=float, help="Minimum sleep time in preprocessing (seconds), requires sleep_distribution=uniform", default=0.05)
    parser.add_argument("--sleep_max", type=float, help="Maximum sleep time in preprocessing (seconds), requires sleep_distribution=uniform", default=0.1)



    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
