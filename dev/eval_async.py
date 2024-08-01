import os
import pprint
import threading
import requests
import json

url = "http://localhost:9999/v1/chat/completions"

data = {
    "model": "Qwen1.5-7B-Chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "introduce yourself please"}
    ]}
headers = {"Content-Type": "application/json"}


def send_request(task_id):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            # {'choices', 'created', 'id', 'model', 'object', 'usage'}
            #   choices: [{'finish_reason', 'index', 'logprobs', 'message', 'stop_reason'}]
            #       message: {'content', 'role', 'tool_calls': List}
            #   usage: {'completion_tokens', 'prompt_tokens', 'total_tokens'}
            print(f"Request {task_id} done: {response.json()}")
            pprint.pprint(response.json())
    except requests.RequestException as e:
        print(f"Request {task_id} failed: {e}")


def main():
    request_num = 100000
    threads = []

    for i in range(request_num):
        thread = threading.Thread(target=send_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    main()
