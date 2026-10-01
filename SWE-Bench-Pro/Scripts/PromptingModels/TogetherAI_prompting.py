#!/usr/bin/env python3

import asyncio
import os
import random
from pathlib import Path
import numpy as np
from together import AsyncTogether



#MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
MODEL ="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"


temperatures = [round(t, 1) for t in np.arange(0, 1.01, 0.1)]
MODEL_DIR_NAME = MODEL.replace("/", "_").replace(":", "_")
prompts_root = Path("prompts")
output_root = Path("llm_outputs")
output_root.mkdir(parents=True, exist_ok=True)



model_dir = output_root / MODEL_DIR_NAME
model_dir.mkdir(parents=True, exist_ok=True)

api_key = "t"
client = AsyncTogether(api_key=api_key)


def is_retryable_error(error_text: str) -> bool:
    """
    Return True for temporary API errors that should be retried.

    503 usually means the Together server is overloaded or not ready.
    429 usually means rate limit.
    Timeout/network errors can also be retried.
    """
    lowered = error_text.lower()

    retryable_patterns = [
        "503",
        "server is overloaded",
        "not ready",
        "overloaded",
        "429",
        "rate limit",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
    ]

    return any(pattern in lowered for pattern in retryable_patterns)


def is_credit_limit_error(error_text: str) -> bool:
    """
    Credit-limit errors should not be retried because retrying will not help.
    """
    lowered = error_text.lower()

    return (
        "402" in lowered
        or "credit limit" in lowered
        or "credit_limit" in lowered
        or "add credits" in lowered
    )


async def run_one_temperature(
    prompt: str,
    temperature: float,
    max_retries: int = 3,
) -> tuple[float, str, str]:
    """
    Run one Together AI request for one prompt at one temperature.

    If the request gets a retryable error such as 503, retry the same
    temperature. If a retry succeeds, return the successful output.

    Returns:
        temperature, output_text, error_text

    If successful, error_text is empty.
    If failed after all retries, output_text is empty.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
            )

            output_text = response.choices[0].message.content.strip()
            return temperature, output_text, ""

        except Exception as e:
            error_text = str(e)

            if is_credit_limit_error(error_text):
                return temperature, "", error_text

            if not is_retryable_error(error_text):
                return temperature, "", error_text

            if attempt == max_retries:
                return temperature, "", error_text

            wait_seconds = min(30, 2 ** attempt) + random.uniform(0, 2)

            print(
                f"Retryable error for temp={temperature:.1f}, "
                f"attempt {attempt}/{max_retries}: {error_text}"
            )
            print(f"Retrying temp={temperature:.1f} after {wait_seconds:.1f} seconds...")

            await asyncio.sleep(wait_seconds)

    return temperature, "", "Unknown error after retries"


async def chat_complete(
    prompt: str,
    concurrency: int = 5,
) -> dict[float, dict[str, str]]:
    """
    Send the same prompt to Together AI at all temperatures.

    Only `concurrency` requests run at the same time. This is safer than
    sending all 11 requests at once, because Together may return 503 if the
    model server is overloaded.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_run(temp):
        async with semaphore:
            return await run_one_temperature(prompt, temp)

    tasks = [
        guarded_run(temp)
        for temp in temperatures
    ]

    results = await asyncio.gather(*tasks)

    by_temperature = {}

    for temp, output_text, error_text in results:
        by_temperature[temp] = {
            "output": output_text,
            "error": error_text,
        }

    return by_temperature


async def main():
    if not prompts_root.exists():
        raise FileNotFoundError(f"Prompts directory does not exist: {prompts_root}")

    for instance_folder in sorted(prompts_root.iterdir()):
        if not instance_folder.is_dir():
            continue

        instance_id = instance_folder.name
        instance_output = model_dir / instance_id
        
        
        if instance_output.exists():
            print(f"Skipping {instance_id}: output folder already exists at {instance_output}")
            continue



        instance_output.mkdir(parents=True, exist_ok=True)

        oracle_prompt_path = instance_folder / "oracle_file_prompt.txt"

        if not oracle_prompt_path.exists():
            print(f"Missing oracle prompt for {instance_id}")
            continue

        oracle_prompt = oracle_prompt_path.read_text(encoding="utf-8")

        print("=" * 80)
        print(f"Running {instance_id} at {len(temperatures)} temperatures")
        print("=" * 80)

        results_by_temp = await chat_complete(
            oracle_prompt,
            concurrency=3,
        )

        stop_due_to_credit = False

        for temp in temperatures:
            temp_s = f"{temp:.1f}"
            result = results_by_temp[temp]

            oracle_output_path = (
                instance_output
                / f"Replacement_Temp_{temp_s}_oracle.txt"
            )

            error_path = (
                instance_output
                / f"Replacement_Temp_{temp_s}_oracle_ERROR.txt"
            )

            if result["error"]:
                error_path.write_text(
                    result["error"],
                    encoding="utf-8",
                )

                print(
                    f"Error oracle, instance={instance_id}, temp={temp_s}: "
                    f"{result['error']}"
                )

                if is_credit_limit_error(result["error"]):
                    stop_due_to_credit = True

            else:
                # Overwrite output txt file if it already exists.
                oracle_output_path.write_text(
                    result["output"],
                    encoding="utf-8",
                )

                # If there was an old error file for this temperature, remove it.
                if error_path.exists():
                    error_path.unlink()

                print(f"Wrote {oracle_output_path}")

        if stop_due_to_credit:
            print()
            print("=" * 80)
            print("STOPPING: Together AI credit limit exceeded.")
            print("Add credits, wait a few minutes, then rerun the script.")
            print("=" * 80)
            return

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())