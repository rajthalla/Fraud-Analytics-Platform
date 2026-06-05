import os
import time
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

MODEL = "models/gemini-2.5-flash-lite"

_MAX_RETRIES = 4
_BACKOFF_BASE = 15


def generate(user_prompt: str, system_prompt: str = "") -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        max_output_tokens=1024,
        temperature=0.2,
    )

    delay = _BACKOFF_BASE
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_prompt,
                config=config,
            )
            return response.text
        except (ClientError, ServerError) as e:
            retryable = "429" in str(e) or "503" in str(e)
            if retryable and attempt < _MAX_RETRIES - 1:
                print(f"    [rate limit / unavailable] waiting {delay}s before retry {attempt + 2}/{_MAX_RETRIES}...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
