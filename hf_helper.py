import os
import json
from textwrap import dedent

import requests
from requests.exceptions import RequestException

# Use the Groq OpenAI-compatible chat completions endpoint
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def build_prompt(code: str, language: str = "unknown") -> str:
    prompt = dedent(
        f"""
        You are an expert code reviewer. Return ONLY valid JSON (no extra text) with these fields:
        - issues: array of objects with keys: line (int or null), severity (critical|warning|info), issue (short string), suggestion (short string)
        - summary: a one-line summary of the main problems
        - explanation: plain-language explanation of the problems and recommended fixes
        - time_complexity: a short string describing overall time complexity (e.g., "O(n)")
        - space_complexity: a short string describing overall space complexity (e.g., "O(1)")
        - optimized_code: (optional) an improved or fixed code snippet when appropriate
        - corrected_code: a complete, corrected version of the code that incorporates all recommended fixes and best practices
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
    """Call Groq via OpenAI-compatible chat API.

    We expect the model to return a string that is *JSON text* according to
    the schema described in `build_prompt`. `app.py` will then json.loads it.
    """
    # Read Groq API key from env
    groq_token = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_TOKEN") or os.getenv("GROQ_API_KEY_DEV")
    if not groq_token:
        raise RuntimeError("GROQ_API_KEY/GROQ_API_TOKEN not set in environment")

    headers = {
        "Authorization": f"Bearer {groq_token}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are an expert senior software engineer acting as a strict code review bot. "
        "You must ALWAYS respond with STRICT, valid JSON only (no explanations, no markdown). "
        "The JSON must match the schema described in the user message."
    )

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)
    except RequestException as e:
        # Network or request-level error: return a JSON string so the caller can log it
        return json.dumps({"error": "request_failed", "detail": str(e)})

    if resp.status_code != 200:
        return json.dumps(
            {"error": f"Groq returned {resp.status_code}", "detail": resp.text}
        )

    try:
        data = resp.json()
        # OpenAI-compatible schema: choices[0].message.content
        content = data["choices"][0]["message"]["content"]
        # This should already be JSON text according to our prompt
        return content
    except Exception:
        # If format is unexpected, fall back to raw body so caller can log it
        return resp.text
