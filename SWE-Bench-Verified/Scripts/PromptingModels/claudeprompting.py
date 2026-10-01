#!/usr/bin/env python3

import asyncio
import math
import os
import random
from pathlib import Path

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"

# Claude Haiku 4.5 output-token ceiling.
MAX_MODEL_OUTPUT_TOKENS = 64000

# Temperatures from 0.0 through 1.0 without using NumPy.
temperatures = [round(i / 10, 1) for i in range(11)]

MODEL_DIR_NAME = MODEL.replace("/", "_").replace(":", "_")

prompts_root = Path("prompts")

output_root = Path("llm_outputs")
output_root.mkdir(parents=True, exist_ok=True)

model_dir = output_root / MODEL_DIR_NAME
model_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------

api_key = ""


client = AsyncAnthropic(
    api_key=api_key,
    timeout=120.0,
    max_retries=0,
)


# ---------------------------------------------------------------------
# Token calculation
# ---------------------------------------------------------------------

def calculate_max_output_tokens(prompt: str) -> int:
    estimated_input_tokens = len(prompt) / 4
    calculated_output_tokens = math.ceil(estimated_input_tokens * 0.10)

    return max(2048, min(calculated_output_tokens, MAX_MODEL_OUTPUT_TOKENS))


# ---------------------------------------------------------------------
# Response helper
# ---------------------------------------------------------------------

def extract_text(response) -> str:
    """
    Extract and combine all text blocks from a Claude response.
    """
    text_parts = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    return "\n".join(text_parts).strip()


# ---------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------

def is_retryable_exception(error: Exception) -> bool:
    """
    Return True for temporary API, connection, timeout, rate-limit,
    and server errors.
    """
    if isinstance(
        error,
        (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        ),
    ):
        return True

    if isinstance(error, APIStatusError):
        return error.status_code in {
            408,
            409,
            429,
            500,
            502,
            503,
            504,
            529,
        }

    lowered = str(error).lower()

    retryable_patterns = [
        "rate limit",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "overloaded",
        "internal server error",
    ]

    return any(
        pattern in lowered
        for pattern in retryable_patterns
    )


# ---------------------------------------------------------------------
# One Claude API call
# ---------------------------------------------------------------------

async def run_one_temperature(
    prompt: str,
    temperature: float,
    max_retries: int = 3,
):
    max_output_tokens = calculate_max_output_tokens(prompt)

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=max_output_tokens,
                temperature=temperature,
                system="You are a code patching assistant. Return only the final code fix. Do not explain the issue. Do not add commentary. Do not include bullet points or reasoning.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            output_text = extract_text(response)

            if not output_text:
                return (
                    temperature,
                    "",
                    (
                        "Claude returned no text output. "
                        f"Stop reason: {response.stop_reason}"
                    ),
                )

            if response.stop_reason == "max_tokens":
                print(
                    f"Warning: output reached max_tokens "
                    f"for temp={temperature:.1f}. "
                    f"Limit={max_output_tokens}"
                )

            return (
                temperature,
                output_text,
                "",
            )

        except Exception as error:
            error_text = (
                f"{type(error).__name__}: {error}"
            )

            if not is_retryable_exception(error):
                return temperature, "", error_text

            if attempt == max_retries:
                return temperature, "", error_text

            wait_seconds = (
                min(30, 2 ** attempt)
                + random.uniform(0, 2)
            )

            print(
                f"Retryable error "
                f"temp={temperature:.1f}, "
                f"attempt={attempt}/{max_retries}: "
                f"{error_text}"
            )

            print(
                f"Retrying after "
                f"{wait_seconds:.1f} seconds..."
            )

            await asyncio.sleep(wait_seconds)

    return temperature, "", "Unknown error"


# ---------------------------------------------------------------------
# Run all temperatures for one prompt
# ---------------------------------------------------------------------

async def chat_complete(
    prompt: str,
    concurrency: int = 3,
):
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_run(temperature: float):
        async with semaphore:
            return await run_one_temperature(
                prompt=prompt,
                temperature=temperature,
            )

    tasks = [
        guarded_run(temperature)
        for temperature in temperatures
    ]

    results = await asyncio.gather(*tasks)

    by_temperature = {}

    for temperature, output_text, error_text in results:
        by_temperature[temperature] = {
            "output": output_text,
            "error": error_text,
        }

    return by_temperature


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

async def main():
    if not prompts_root.exists():
        raise FileNotFoundError(
            f"Prompts directory does not exist: "
            f"{prompts_root.resolve()}"
        )

    issue_folders = sorted(
        folder
        for folder in prompts_root.iterdir()
        if folder.is_dir()
    )

    print(
        f"Found {len(issue_folders)} issue folders "
        f"in {prompts_root.resolve()}"
    )

    for issue_folder in issue_folders:
        instance_id = issue_folder.name

        oracle_prompt_path = (
            issue_folder
            / "oracle_file_prompt.txt"
        )

        if not oracle_prompt_path.exists():
            print(
                f"Skipping {instance_id}: "
                f"missing {oracle_prompt_path.name}"
            )
            continue

        oracle_prompt = oracle_prompt_path.read_text(
            encoding="utf-8"
        ).strip()

        if not oracle_prompt:
            print(
                f"Skipping {instance_id}: "
                "oracle prompt is empty"
            )
            continue

        instance_output = model_dir / instance_id

        if instance_output.exists():
            print(
                f"Skipping {instance_id}: "
                f"{instance_output} already exists"
            )
            continue

        instance_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        estimated_input_tokens = math.ceil(
            len(oracle_prompt) / 4
        )

        max_output_tokens = calculate_max_output_tokens(
            oracle_prompt
        )

        print()
        print("=" * 80)
        print(f"Issue: {instance_id}")
        print(f"Prompt file: {oracle_prompt_path}")
        print(f"Prompt characters: {len(oracle_prompt)}")
        print(
            f"Estimated input tokens: "
            f"{estimated_input_tokens}"
        )
        print(
            f"Maximum output tokens: "
            f"{max_output_tokens}"
        )
        print(
            f"Running {len(temperatures)} temperatures "
            f"with {MODEL}"
        )
        print("=" * 80)

        results_by_temp = await chat_complete(
            prompt=oracle_prompt,
            concurrency=3,
        )

        for temperature in temperatures:
            temp_s = f"{temperature:.1f}"

            result = results_by_temp[temperature]

            output_path = (
                instance_output
                / f"Replacement_Temp_{temp_s}_oracle.txt"
            )

            error_path = (
                instance_output
                / (
                    f"Replacement_Temp_"
                    f"{temp_s}_oracle_ERROR.txt"
                )
            )

            if result["error"]:
                error_path.write_text(
                    result["error"],
                    encoding="utf-8",
                )

                print(
                    f"Error "
                    f"issue={instance_id} "
                    f"temp={temp_s}: "
                    f"{result['error']}"
                )

            else:
                output_path.write_text(
                    result["output"],
                    encoding="utf-8",
                )

                if error_path.exists():
                    error_path.unlink()

                print(f"Wrote {output_path}")

    await client.close()

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())