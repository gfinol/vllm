from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

import asyncio
import time
import uuid

st = time.time()


example_inputs = [
    {
        "prompt": "About 200 words, please give me some tourist information about Tokyo.",
        "temperature": 0.9,
    },
    {
        "prompt": "About 200 words, please give me some tourist information about Osaka.",
        "temperature": 0.9,
    },
]


async def gen(engine, example_input, id):
    results_generator = engine.generate(
        example_input["prompt"],
        SamplingParams(temperature=example_input["temperature"],max_tokens=300, min_tokens=200,),
        id,
    )

    final_output = None

    async for request_output in results_generator:
        final_output = request_output

    prompt = final_output.prompt
    text_output = [output.text for output in final_output.outputs]
    return text_output[0]


async def main():

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model="/kaggle/input/llama-2/pytorch/7b-chat-hf/1",
            dtype="half",
            enforce_eager=True,
            gpu_memory_utilization=0.99,
            swap_space=3,
            max_model_len=1024,
            kv_cache_dtype="fp8_e5m2",
            tensor_parallel_size=2,
            disable_log_requests=True
        )
    )

    results = []

    for example_input in example_inputs:
        tasks = []
        for i in range(100):
            tasks.append(asyncio.create_task(gen(engine, example_input, uuid.uuid4())))

        res = [await task for task in tasks]

        results.append(res)

    with open("async_res.txt", "w") as f:
        for r in results:
            f.writelines(r)


if __name__ == "__main__":
    asyncio.run(main())

    print("Async vLLM inference time: ", time.time() - st)