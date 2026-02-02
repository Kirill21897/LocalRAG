import requests
import json 

response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'qwen2.5:14b',
        'prompt': 'Привет!',
        'stream': True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        print(chunk.get("response", ""), end="", flush=True)
print()