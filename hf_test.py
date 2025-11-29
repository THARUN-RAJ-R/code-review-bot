import os
import requests

api_url = "https://router.huggingface.co/v1/chat/completions"

token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or os.environ.get("HF_API_TOKEN")
if not token:
    print("NO_TOKEN")
    raise SystemExit(0)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

payload = {
    "model": "Qwen/Qwen2.5-7B-Instruct:together",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
}

resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
print("STATUS", resp.status_code)
print("BODY", resp.text[:1000])
