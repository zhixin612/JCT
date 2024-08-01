import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import queue
import threading
import time
import json
from workload import loader
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams


# max_seqs = 256
max_model_len = 4096
# max_num_batched_tokens = 2048
model_path = "Qwen/Qwen1.5-7B-Chat"
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
engine_args = EngineArgs(model=model_path,
                         # max_num_seqs=max_seqs,
                         # max_model_len=max_model_len,
                         # max_num_batched_tokens=max_num_batched_tokens,
                         enable_chunked_prefill=False,
                         # gpu_memory_utilization=0.2,
                         enforce_eager=True,
                         # num_gpu_blocks_override=5,
                         )

engine = LLMEngine.from_engine_args(engine_args)


def step(logs):
    seq_group_metadata_list, scheduler_outputs = engine.scheduler.schedule()
    now = time.time()
    for seqGroup in seq_group_metadata_list:
        # print(logs)
        logs[int(seqGroup.request_id)]["waiting"] += now - logs[int(seqGroup.request_id)]["lastTokenTime"]

    if not scheduler_outputs.is_empty():
        output = engine.model_executor.execute_model(
            seq_group_metadata_list, scheduler_outputs.blocks_to_swap_in,
            scheduler_outputs.blocks_to_swap_out,
            scheduler_outputs.blocks_to_copy)
        ed = time.time()
        for seqGroup in seq_group_metadata_list:
            if len(logs[int(seqGroup.request_id)]["tokenTime"]) == 0:
                logs[int(seqGroup.request_id)]["TFOT"] = ed - logs[int(seqGroup.request_id)]["arrivalTime"]
            logs[int(seqGroup.request_id)]["tokenTime"].append(ed - now)
            logs[int(seqGroup.request_id)]["lastTokenTime"] = ed
    else:
        output = []

    res = engine._process_model_outputs(output, scheduler_outputs)

    return res


def run(q: queue.Queue, logs, max_requests):
    job_id = 0
    cnt = 0
    while job_id < max_requests or engine.has_unfinished_requests():
        while not q.empty():
            engine.add_request(str(job_id), inputs=q.get(), params=sampling_params)
            job_id += 1
        res = step(logs)
        for output in res:
            if output.finished:
                print(output.request_id, output.metrics.finished_time - output.metrics.arrival_time,
                      # output.outputs[0].text,
                      end=", ")
                cnt += 1
                print("total finish " + str(cnt))


def test():
    max_request = 200000
    generator = loader.gen_workload('dolly', 'wiki', max_request, 0.1)
    logs = {}
    q = queue.Queue()

    _t = threading.Thread(target=run, args=(q, logs, max_request))
    _t.start()

    rid = 0
    for data in generator:
        q.put(data['input'])
        now = time.time()
        logs[rid] = {"arrivalTime": now, "tokenTime": [],
                     "lastTokenTime": now, "waiting": 0, "TTFT": 0}
        rid += 1

    _t.join()
    if not os.path.isdir('log'):
        os.mkdir('log')
    with open('log/[Qwen1.5-7B-Chat][tp1][dolly][200k][wiki][0.1].json', 'w') as log_file:
        json.dump(logs, log_file, indent=4)
    # print(logs)


if __name__ == '__main__':
    test()
