import requests

url = "http://127.0.0.1:8000/predict"
headers = {"Content-Type": "application/json"}
data = {"text": "Hello, how are you?"}

try:
    response = requests.post(url, headers=headers, json=data, timeout=5)
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
