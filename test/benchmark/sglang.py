"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""

import argparse
import asyncio
import json
import os
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: AutoTokenizer,
) -> List[Tuple[str, int, int]]:
    def load_dataset():
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]

        # Tokenize the prompts and completions.
        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_dataset: List[Tuple[str, int, int]] = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 or output_len < 4:
                # Prune too short sequences.
                # This is because TGI causes errors when the input or output length
                # is too short.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_dataset.append((prompt, prompt_len, output_len))

        return filtered_dataset

    try:
        from diskcache import Cache

        home_dir = os.path.expanduser("~")
        cache = Cache(f"{home_dir}/.cache/sglang")
        with Cache(cache.directory) as reference:
            reference_key = f"{dataset_path}_{tokenizer.name_or_path}"
            if reference_key in reference:
                print("Reading dataset from cache...")
                dataset = reference[reference_key]
            else:
                dataset = load_dataset()
                reference[reference_key] = dataset
    except ImportError:
        dataset = load_dataset()

    # Sample the requests.
    sampled_requests = random.sample(dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    elif backend == "srt":
        assert not use_beam_search
        params = {
            "ignore_eos": True,
            "max_new_tokens": output_len,
        }
        pload = {
            "text": prompt,
            "sampling_params": params,
        }
    elif backend == "lightllm":
        assert not use_beam_search
        params = {
            "ignore_eos": True,
            "max_new_tokens": output_len,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    elif backend == "ginfer":
        pass
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if backend != "ginfer":
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(
                    api_url, headers=headers, json=pload
                ) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        chunks.append(chunk)
                output = b"".join(chunks).decode("utf-8")
                output = json.loads(output)

                # Re-send the request if it failed.
                if "error" not in output:
                    break
                else:
                    print(output)
    else:
        import grpc
        from ginfer import sampler_pb2, sampler_pb2_grpc

        api_url = api_url.replace("http://", "").replace("/generate", "")
        sampler_channel = grpc.aio.insecure_channel(api_url)
        sampler = sampler_pb2_grpc.SamplerStub(sampler_channel)

        request_end_time = time.perf_counter()
        sample_request = sampler_pb2.SampleTextRequest(
            prompt=prompt,
            settings=sampler_pb2.SampleSettings(
                max_len=output_len,
                rng_seed=0,
                temperature=0,
                nucleus_p=1,
            ),
        )
        stream = sampler.SampleText(sample_request)
        response = "".join([x.text async for x in stream])

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                prompt,
                prompt_len,
                output_len,
                best_of,
                use_beam_search,
            )
        )
        tasks.append(task)
    await tqdm_asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"{args.host}:{args.port}/generate"
    if args.tokenizer.endswith(".json") or args.tokenizer.endswith(".model"):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code
        )

    if args.dataset:
        input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
    else:
        input_lens = np.random.randint(
            int(args.input_len * args.range_ratio),
            args.input_len + 1,
            size=args.num_prompts,
        )
        output_lens = np.random.randint(
            int(args.output_len * args.range_ratio),
            args.output_len + 1,
            size=args.num_prompts,
        )
        offsets = np.random.randint(0, tokenizer.vocab_size, size=args.num_prompts)
        input_requests = []
        for i in range(args.num_prompts):
            prompt = tokenizer.decode(
                [
                    (offsets[i] + i + j) % (tokenizer.vocab_size - 129) + 128
                    for j in range(input_lens[i])
                ]
            )
            input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
        )
    )
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time

    # Compute the statistics.
    latencies = [latency for _, _, latency in REQUEST_LATENCY]
    avg_latency = np.mean(latencies)
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    decoding_throughput = (
        np.sum([output_len for _, output_len, _ in REQUEST_LATENCY]) / benchmark_time
    )

    # latencies = [round(latency, 2) for _, _, latency in REQUEST_LATENCY]
    # print(latencies)

    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Request throughput: {args.num_prompts / benchmark_time:.2f} requests/s")
    print(f"Decoding throughput: {decoding_throughput:.2f} token/s")
    print(f"Average latency: {avg_latency:.2f} s")
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    print(f"Average latency per output token: {avg_per_output_token_latency:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="srt",
        choices=["vllm", "tgi", "srt", "lightllm", "ginfer"],
    )
    parser.add_argument("--host", type=str, default="http://localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--dataset", type=str, help="Path to the dataset.")
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--range-ratio", type=float, default=1.0)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="NousResearch/Meta-Llama-3-8B",
        help="Name or path of the tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    args = parser.parse_args()
    main(args)