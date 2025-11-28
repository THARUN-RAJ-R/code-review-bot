import os
import json
from textwrap import dedent

import requests
from requests.exceptions import RequestException

HF_BASE = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"


def build_prompt(code: str, language: str = "unknown") -> str:
    prompt = dedent(
        f"""
        You are an expert code reviewer. Return ONLY valid JSON (no extra text) with these fields:
        - issues: array of objects with keys: line (int or null), severity (critical|warning|info), issue (short string), suggestion (short string)
        - summary: a one-line summary of the main problems
        - explanation: plain-language explanation of the problems and recommended fixes
        - optimized_code: (optional) an improved or fixed code snippet when appropriate
        - learning_links: array of strings with useful docs/links (optional)

        LANGUAGE: {language}
        CODE:
        ```{code}
        ```
        Make the JSON concise and machine-parseable.
        """.strip()
    )
    return prompt


def call_qwen_inference(prompt: str, timeout: int = 120) -> str:
    """Call the Qwen2.5-7B-Instruct model via Hugging Face Inference API.

    Reads HF_API_TOKEN at call time so that .env / environment is already loaded
    by the Flask app.
    """
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN not set in environment")

    headers = {"Authorization": f"Bearer {hf_token}"}

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.0},
    }

    try:
        resp = requests.post(HF_BASE, headers=headers, json=payload, timeout=timeout)
    except RequestException as e:
        # Network or request-level error: return a JSON string so the caller can log it
        return json.dumps({"error": "request_failed", "detail": str(e)})

    if resp.status_code != 200:
        return json.dumps(
            {"error": f"HF returned {resp.status_code}", "detail": resp.text}
        )

    try:
        data = resp.json()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return resp.text
    except ValueError:
        return resp.text
