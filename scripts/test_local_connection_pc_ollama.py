import requests

response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'qwen2.5:14b',
        'prompt': 'Привет!',
        'stream': False
    }
)
print(response.json()["response"])