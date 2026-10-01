#!/usr/bin/env python3

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

"""
This script is designed for “one repository per shard.” So if you have 32 repositories, the ideal run is:

python llm_fix_evaluation_swebench_repo_shard.py --shard_id 0 --num_shards 32
python llm_fix_evaluation_swebench_repo_shard.py --shard_id 1 --num_shards 32
python llm_fix_evaluation_swebench_repo_shard.py --shard_id 2 --num_shards 32
....
python llm_fix_evaluation_swebench_repo_shard.py --shard_id 31 --num_shards 32


"""



DATASET_NAME = "ScaleAI/SWE-bench_Pro"
SPLIT = "test"
MODEL = "meta-llama_Llama-3.3-70B-Instruct-Turbo"

PRED_ROOT = Path("predictions_from_replacements") / MODEL
REPORT_ROOT = Path("swebench_reports") / MODEL
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_RESULTS_DIR = Path("swebench_csv_results")
CSV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_RESULTS_DIR_PARTIAL = Path("swebench_csv_results_partial")
CSV_RESULTS_DIR_PARTIAL.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [round(i / 10, 1) for i in range(11)]

# You said you removed non_oracle.
PROMPT_TYPES = ["oracle"]

# For repeated issues from the same repo, env cache is usually the useful level.
CACHE_LEVEL = "env"

# Because each script process handles one repo shard, keep this conservative.
MAX_WORKERS = "4"


def temp_to_str(temp: float) -> str:
    return f"{temp:.1f}"


def make_run_id(instance_id: str, temp_s: str, prompt_type: str) -> str:
    safe_instance = instance_id.replace("/", "__").replace(".", "_")
    safe_temp = temp_s.replace(".", "_")
    return f"{safe_instance}_{prompt_type}_temp_{safe_temp}"


def bool_to_swebench_str(value: bool) -> str:
    return "true" if value else "false"


def load_instances_by_id():
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    return {row["instance_id"]: row for row in dataset}


def group_instance_dirs_by_repo(instance_dirs, instances_by_id):
    repo_to_dirs = defaultdict(list)

    for instance_dir in instance_dirs:
        instance_id = instance_dir.name

        if instance_id not in instances_by_id:
            print(f"WARNING: instance not found in dataset, skipping: {instance_id}")
            continue

        repo_name = instances_by_id[instance_id]["repo"]
        repo_to_dirs[repo_name].append(instance_dir)

    return dict(repo_to_dirs)


def run_swebench(
    prediction_path: Path,
    instance_id: str,
    run_id: str,
    clean: bool,
):
    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASET_NAME,
        "--split",
        SPLIT,
        "--predictions_path",
        str(prediction_path),
        "--max_workers",
        MAX_WORKERS,
        "--instance_ids",
        instance_id,
        "--run_id",
        run_id,
        "--cache_level",
        CACHE_LEVEL,
        "--clean",
        bool_to_swebench_str(clean),
    ]

    print()
    print("=" * 80)
    print(f"Running instance: {instance_id}")
    print(f"Prediction file: {prediction_path}")
    print(f"Run ID: {run_id}")
    print(f"cache_level: {CACHE_LEVEL}")
    print(f"clean: {clean}")
    print("=" * 80)

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    

    log_path = REPORT_ROOT / f"{run_id}_swebench_stdout_stderr.log"
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(cmd)
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\n\nSTDERR:\n"
        + result.stderr,
        encoding="utf-8",
    )

    matches = sorted(Path(".").glob(f"*.{run_id}.json"))

    if not matches:
        print(f"No SWE-bench report created for run_id={run_id}")
        print(f"Wrote harness log: {log_path}")
        return None

    generated_report = matches[-1]
    final_report_path = REPORT_ROOT / generated_report.name

    subprocess.run(
        ["mv", str(generated_report), str(final_report_path)],
        check=True,
    )

    return final_report_path


def extract_result_from_report(report_path: Path, instance_id: str):
    if report_path is None:
        return "NO_REPORT"

    if not report_path.exists():
        return "NO_REPORT"

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    resolved = set(report.get("resolved_ids", []))
    unresolved = set(report.get("unresolved_ids", []))
    error_ids = set(report.get("error_ids", []))
    incomplete = set(report.get("incomplete_ids", []))
    empty_patch = set(report.get("empty_patch_ids", []))

    if instance_id in resolved:
        return 1

    if instance_id in unresolved:
        return 0

    if instance_id in error_ids:
        return "ERROR"

    if instance_id in incomplete:
        return "INCOMPLETE"

    if instance_id in empty_patch:
        return "EMPTY_PATCH"

    return "UNKNOWN"




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"Invalid shard_id={args.shard_id}. "
            f"With num_shards={args.num_shards}, shard_id must be between "
            f"0 and {args.num_shards - 1}."
        )

    if not PRED_ROOT.exists():
        raise FileNotFoundError(f"Cannot find prediction root: {PRED_ROOT}")

    print(f"Loading dataset: {DATASET_NAME}, split={SPLIT}")
    instances_by_id = load_instances_by_id()

    instance_dirs = sorted([p for p in PRED_ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(instance_dirs)} total instance prediction folders.")

    repo_to_dirs = group_instance_dirs_by_repo(instance_dirs, instances_by_id)
    repo_names = sorted(repo_to_dirs.keys())
    
    """
    {
        "astropy/astropy": [
            Path(".../astropy__astropy-14598"),
            Path(".../astropy__astropy-12907"),
        ],
        "django/django": [
            Path(".../django__django-10554"),
            Path(".../django__django-12304"),
        ],
    }
    """

    print(f"Found {len(repo_names)} repositories.")

    if args.num_shards != len(repo_names):
        print()
        print("WARNING:")
        print(f"num_shards={args.num_shards}, but number of repositories={len(repo_names)}.")
        print("This script is designed for one repository per shard.")
        print("For exact repo-per-shard behavior, run with:")
        print(f"    --num_shards {len(repo_names)}")
        print()

    if args.shard_id >= len(repo_names):
        print(
            f"Shard {args.shard_id} has no repository because there are only "
            f"{len(repo_names)} repositories."
        )
        return

    #Shard 0 gets repo_names[0], Shard 1 gets repo_names[1], Shard 2 gets repo_names[2]
    repo_name = repo_names[args.shard_id]
    repo_instance_dirs = sorted(repo_to_dirs[repo_name])
    


    print()
    print("#" * 100)
    print(f"Shard {args.shard_id}/{args.num_shards}")
    print(f"Assigned repository: {repo_name}")
    print(f"Issue folders in this repository: {len(repo_instance_dirs)}")
    print("#" * 100)
    print()

    results_csv = CSV_RESULTS_DIR / f"repo_shard_{args.shard_id}_of_{args.num_shards}.csv"
    partial_csv = CSV_RESULTS_DIR_PARTIAL / f"repo_shard_{args.shard_id}_of_{args.num_shards}.csv"

    rows = []

    # Clean only once at the first actual SWE-bench run for this repo.
    # After that, cache is kept for the rest of this repo's issues.
    first_eval_call_for_repo = True

    for instance_dir in repo_instance_dirs:
        instance_id = instance_dir.name

        row = {
            "instance_id": instance_id,
            "repo": repo_name,
        }

        for temp in TEMPERATURES:
            temp_s = temp_to_str(temp)

            for prompt_type in PROMPT_TYPES:
                col_name = f"Temp_{temp_s}_{prompt_type}"
                file_name = f"Replacement_Temp_{temp_s}_{prompt_type}"
                prediction_path = instance_dir / f"{file_name}.jsonl"

                if not prediction_path.exists():
                    print(f"Missing prediction file: {prediction_path}")
                    row[col_name] = "MISSING"

                    partial_df = pd.DataFrame(rows + [row])
                    partial_df.to_csv(partial_csv, index=False)
                    continue

                run_id = make_run_id(instance_id, temp_s, prompt_type)

                existing_reports = sorted(REPORT_ROOT.glob(f"*.{run_id}.json"))

                if existing_reports:
                    report_path = existing_reports[-1]
                    print(f"Found existing report, skipping SWE-bench run: {report_path}")
                else:
                    clean_for_this_call = first_eval_call_for_repo

                    report_path = run_swebench(
                        prediction_path=prediction_path,
                        instance_id=instance_id,
                        run_id=run_id,
                        clean=clean_for_this_call,
                    )

                    first_eval_call_for_repo = False

                result = extract_result_from_report(report_path, instance_id)
                row[col_name] = result

                print(f"Result for {instance_id} {col_name}: {result}")

                partial_df = pd.DataFrame(rows + [row])
                partial_df.to_csv(partial_csv, index=False)

        rows.append(row)

        partial_df = pd.DataFrame(rows)
        partial_df.to_csv(partial_csv, index=False)

    final_df = pd.DataFrame(rows)
    final_df.to_csv(results_csv, index=False)

    print()
    print(f"Wrote {results_csv}")
    print("Done.")


if __name__ == "__main__":
    main()