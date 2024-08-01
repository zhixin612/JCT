import logging
import os
import time

import numpy as np
import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import Dataset

logging.basicConfig(level=logging.INFO)


def load_input(dataset) -> Dataset:
    ds = None
    if dataset.lower() == 'mmlu':           # cais/mmlu (QA)
        # avg input = 323.5, avg output = 1.0
        # https://huggingface.co/datasets/cais/mmlu
        # Dataset({features: ['question', 'subject', 'choices', 'answer'], num_rows: 99842})
        ds = load_dataset("cais/mmlu", "all")['auxiliary_train']    # abstract_algebra, college_computer_science, ...
        ds = ds.map(lambda it: {
            'input': it['question'] + " " + "; ".join(it['choices']),
            'output': str(it['answer'])
        })

    if dataset.lower() == 'dolly':          # databricks/databricks-dolly-15k (QA)
        # avg input = 80, avg output = 69
        # https://huggingface.co/datasets/databricks/databricks-dolly-15k
        # Dataset({features: ['instruction', 'context', 'response', 'category'], num_rows: 15011})
        ds = load_dataset("databricks/databricks-dolly-15k")['train']
        ds = ds.map(lambda it: {
            'input': it['instruction'] + "\n" + it['context'],
            'output': it['response']
        })

    if dataset.lower() == 'alpaca':         # tatsu-lab/alpaca (QA)
        # avg input = 117, avg output = 51
        # https://huggingface.co/datasets/tatsu-lab/alpaca
        # Dataset({features: ['instruction', 'input', 'output', 'text'], num_rows: 52002})
        ds = load_dataset("tatsu-lab/alpaca")['train']
        ds = ds.map(lambda it: {
            'input': it['text'] + "\n" + it['instruction'] + "\n" + it['input'],
            'output': it['output']
        })

    if dataset.lower() == 'alpaca_python':  # iamtarun/python_code_instructions_18k_alpaca  (code generation)
        # avg input = 188, avg output = 99
        # https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca
        # Dataset({features: ['instruction', 'input', 'output', 'prompt'], num_rows: 18612})
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")['train']
        ds = ds.map(lambda it: {
            'input': it['prompt'] + "\n" + it['instruction'] + "\n" + it['input'],
            'output': it['output']
        })

    if dataset.lower() == 'dialogsum':      # knkarthick/dialogsum (dialogue summarization)
        # avg input = 193, avg output = 31
        # https://huggingface.co/datasets/knkarthick/dialogsum
        # train: Dataset({features: ['id', 'dialogue', 'summary', 'topic'], num_rows: 12460})
        ds = load_dataset("knkarthick/dialogsum")['train']
        ds = ds.map(lambda it: {
            'input': 'Summarize the following dialogue:\n' + it['dialogue'],
            'output': it['summary']
        })

    if dataset.lower() == 'openorca':       # Open-Orca/OpenOrca (QA)
        # https://huggingface.co/datasets/Open-Orca/OpenOrca
        # Dataset({features: ['id', 'system_prompt', 'question', 'response'], num_rows: 4233923})
        ds = load_dataset("Open-Orca/OpenOrca")['train']
        ds = ds.map(lambda it: {
            'input': it['system_prompt'] + "\n" + it['question'],
            'output': it['response']
        })

    # "wikimedia/wikipedia (wiki for summarization)
    # https://huggingface.co/datasets/wikimedia/wikipedia
    # ds = load_dataset("wikimedia/wikipedia", "20231101.en")
    # print(ds, ds.keys())
    # ds = load_dataset("wikimedia/wikipedia", "20231101.zh")
    # print(ds, ds.keys())

    if ds is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    ds = ds.remove_columns([col for col in ds.column_names if col not in ['input', 'output']])
    return ds


def load_trace(trace) -> np.ndarray:
    """
    :param trace: ['wiki', 'tweet']
    :return: List(Float) list of request interval in milliseconds
    """
    # TRACE     SIZE    AVG(qps)    MIN(qps)    MAX(qps)
    # wiki      417908  203         118         350
    # tweet     368300  298         203         546
    # burst0    331407  69          2.8         500
    # burst1    249229  70          1.0         498
    ts = None
    path = os.path.dirname(os.path.abspath(__file__))
    if trace in ['tweet', 'wiki', 'burstgpt_0', 'burstgpt_1']:
        ts = pd.read_csv(os.path.join(path, f'trace/{trace}.csv'), header=None).to_numpy().flatten()

    if ts is None:
        raise ValueError(f"Unknown trace: {trace}")

    mean = np.convolve(ts, np.ones(100) / 100, mode='valid')
    # print(1000/np.average(ts), 1000/np.min(mean), 1000/np.max(mean))
    # print(len(ts), type(ts))
    return ts


def gen_workload(dataset, trace, max_request=None, rate_scale=1.0):  # generator
    # request rate = trace rate * rate scale
    ds = load_input(dataset)
    ts = load_trace(trace)
    ddl, cnt = time.time(), 0
    logging.info(f"Generating workload for dataset {dataset} and trace {trace}")
    for prompt, gap in zip(ds, ts):
        yield prompt
        cnt += 1
        ddl += gap / rate_scale / 1000
        if max_request and cnt >= max_request:
            break
        time.sleep(max(0, ddl - time.time()))
    logging.info(f"Workload generation for dataset {dataset} and trace {trace} completed")


if __name__ == '__main__':
    # trace_names = ['tweet', 'wiki', 'burstgpt_0', 'burstgpt_1']
    # for name in trace_names:
    #     load_trace(name)

    dataset_names = ['mmlu', 'dolly', 'alpaca', 'alpaca_python', 'dialogsum', 'openorca']
    for name in dataset_names:
        _ds = load_input(name)
        print(_ds, type(_ds))

    # for p in gen_workload('alpaca_python', 'wiki', 100, 0.01):
    #     print(p['input'])
