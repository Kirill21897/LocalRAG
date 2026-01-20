import requests

response = requests.post(
    "http://192.168.88.21:91/api/generate",
    json={
        "model": "qwen3-vl:8b",
        "prompt": "What is AI?",
        "stream": False
    }
)

print(response.json()["response"])