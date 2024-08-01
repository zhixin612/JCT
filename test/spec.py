import time

from vllm import LLM, SamplingParams


sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen1.5-7B-Chat",
    tensor_parallel_size=1,
    # speculative_model="Qwen/Qwen1.5-0.5B-Chat",
    # num_speculative_tokens=5,
    # use_v2_block_manager=True,
)

while True:
    prompt = input("INPUT: ")
    print('Prompt: ', prompt)
    _start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    _duration = time.time() - _start
    tokens = len(outputs[0].outputs[0].token_ids)
    print(outputs[0])
    print(f"Tokens = {tokens}  |  Speed = {tokens / _duration:.2f} tokens/s")

    for output in outputs:  # batch
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
