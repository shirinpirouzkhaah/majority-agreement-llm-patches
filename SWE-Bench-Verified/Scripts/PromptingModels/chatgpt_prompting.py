#!/usr/bin/env python3

import asyncio
import os
import random
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI

MODEL = "gpt-4.1-mini"

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

temperatures = [round(t, 1) for t in np.arange(0, 1.01, 0.1)]

MODEL_DIR_NAME = MODEL.replace("/", "_").replace(":", "_")

prompts_root = Path("prompts")

output_root = Path("llm_outputs")
output_root.mkdir(parents=True, exist_ok=True)

model_dir = output_root / MODEL_DIR_NAME
model_dir.mkdir(parents=True, exist_ok=True)

client = AsyncOpenAI(
    api_key=""
)


# ---------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------

def is_retryable_error(error_text: str) -> bool:
    lowered = error_text.lower()

    retryable_patterns = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "overloaded",
        "internal server error",
    ]

    return any(p in lowered for p in retryable_patterns)


# ---------------------------------------------------------------------
# One API call
# ---------------------------------------------------------------------

async def run_one_temperature(
    prompt: str,
    temperature: float,
    max_retries: int = 3,
):


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
            
            return (
                temperature,
                output_text,
                "",
            )

        except Exception as e:

            error_text = str(e)

            if not is_retryable_error(error_text):
                return temperature, "", error_text

            if attempt == max_retries:
                return temperature, "", error_text

            wait_seconds = min(30, 2 ** attempt) + random.uniform(0, 2)

            print(
                f"Retryable error for temp={temperature:.1f}, "
                f"attempt {attempt}/{max_retries}: {error_text}"
            )

            print(
                f"Retrying after {wait_seconds:.1f} seconds..."
            )

            await asyncio.sleep(wait_seconds)

    return temperature, "", "Unknown error"


# ---------------------------------------------------------------------
# Prompt all temperatures
# ---------------------------------------------------------------------

async def chat_complete(
    prompt: str,
    concurrency: int = 5,
):

    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_run(temp):
        async with semaphore:
            return await run_one_temperature(
                prompt,
                temp,
            )

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

async def main():

    if not prompts_root.exists():
        raise FileNotFoundError(
            f"Prompts directory does not exist: {prompts_root}"
        )

    for instance_folder in sorted(prompts_root.iterdir()):

        if not instance_folder.is_dir():
            continue

        instance_id = instance_folder.name

        instance_output = model_dir / instance_id

        if instance_output.exists():
            print(
                f"Skipping {instance_id}: "
                f"{instance_output} already exists."
            )
            continue

        instance_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        oracle_prompt_path = (
            instance_folder /
            "oracle_file_prompt.txt"
        )

        if not oracle_prompt_path.exists():
            print(
                f"Missing oracle prompt for {instance_id}"
            )
            continue

        oracle_prompt = oracle_prompt_path.read_text(
            encoding="utf-8"
        )

        print("=" * 80)
        print(
            f"Running {instance_id} "
            f"at {len(temperatures)} temperatures"
        )
        print("=" * 80)

        results_by_temp = await chat_complete(
            oracle_prompt,
            concurrency=3,
        )

        for temp in temperatures:

            temp_s = f"{temp:.1f}"

            result = results_by_temp[temp]

            oracle_output_path = (
                instance_output /
                f"Replacement_Temp_{temp_s}_oracle.txt"
            )

            error_path = (
                instance_output /
                f"Replacement_Temp_{temp_s}_oracle_ERROR.txt"
            )

            if result["error"]:

                error_path.write_text(
                    result["error"],
                    encoding="utf-8",
                )

                print(
                    f"Error "
                    f"instance={instance_id} "
                    f"temp={temp_s}: "
                    f"{result['error']}"
                )

            else:

                oracle_output_path.write_text(
                    result["output"],
                    encoding="utf-8",
                )

                if error_path.exists():
                    error_path.unlink()

                print(
                    f"Wrote {oracle_output_path}"
                )

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())