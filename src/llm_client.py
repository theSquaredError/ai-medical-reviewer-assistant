import json
import re
import time

from openai import OpenAI

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen2.5-72B-instruct"

_client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")

SYSTEM_PROMPT = (
    "You are an expert FDA pharmacovigilance reviewer trained in MedDRA coding "
    "and WHO-UMC causality assessment. Output ONLY valid JSON. "
    "No markdown, no code fences, no explanation outside the JSON object."
)


def call_llm(prompt: str, retries: int = 3) -> dict | None:
    """Call Qwen 2.5 72B via vLLM and return parsed JSON dict, or None on failure."""
    for attempt in range(retries):
        try:
            resp = _client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = _parse_json(raw)
            if parsed is not None:
                return parsed
            # If parse failed and we have retries left, ask again more strictly
            if attempt < retries - 1:
                prompt = _strict_json_retry_prompt(prompt, raw)
        except Exception:
            if attempt < retries - 1:
                time.sleep(2**attempt)

    return None


def _parse_json(text: str) -> dict | None:
    """Strip markdown fences then parse JSON. Falls back to regex extraction."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _strict_json_retry_prompt(original_prompt: str, bad_output: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Your previous response was not valid JSON:\n{bad_output}\n\n"
        "Return ONLY the JSON object. Start with {{ and end with }}. Nothing else."
    )
