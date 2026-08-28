"""OpenAI API key and model-call diagnostic. Does not print the API key."""

import os
from pathlib import Path

from dotenv import dotenv_values
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_VALUES = dotenv_values(ENV_PATH)
API_KEY = ENV_VALUES.get("OPENAI_API_KEY") or ENV_VALUES.get("OPENAI_KEY")
MODEL = ENV_VALUES.get("OPENAI_MODEL") or "gpt-4o-mini"


def fail(message: str, exit_code: int) -> None:
    print(message)
    raise SystemExit(exit_code)


if not ENV_PATH.is_file():
    fail(f"FAIL: .env file not found: {ENV_PATH}", 2)

if not API_KEY:
    fail("FAIL: OPENAI_API_KEY (or OPENAI_KEY) is missing in .env.", 2)

print(f"Using .env file: {ENV_PATH}")
print(f"Using model: {MODEL}")
print("Sending a minimal chat completion request...")

try:
    client = OpenAI(api_key=API_KEY)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: API_KEY_VALID"}],
        max_tokens=10,
    )
except AuthenticationError as error:
    fail(f"AUTHENTICATION_FAILED (HTTP {error.status_code}): {error.message}", 10)
except PermissionDeniedError as error:
    fail(f"AUTHENTICATED_BUT_FORBIDDEN (HTTP {error.status_code}): {error.message}", 11)
except RateLimitError as error:
    fail(f"AUTHENTICATED_BUT_QUOTA_OR_RATE_LIMIT (HTTP {error.status_code}): {error.message}", 12)
except APIConnectionError as error:
    fail(f"CONNECTION_FAILED: {error}", 13)
except APIStatusError as error:
    fail(f"API_REQUEST_FAILED (HTTP {error.status_code}): {error.message}", 14)

answer = completion.choices[0].message.content or ""
print("API_KEY_AND_MODEL_CALL_VALID")
print(f"Response: {answer}")

# Guard against accidentally relying on a different process-level key.
process_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if process_key and process_key != API_KEY:
    print("WARNING: a different process environment key is also set.")
