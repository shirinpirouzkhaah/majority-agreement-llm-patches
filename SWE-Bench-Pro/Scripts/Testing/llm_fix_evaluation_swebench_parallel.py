#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datasets import load_dataset


"""
SWE-bench Pro repo-sharded evaluator.

Run from the SWE_pro directory:

    cd /srv/scratch/shirin/swebench-work/SWE_pro

    python llm_fix_evaluation_swebench_repo_shard.py --shard_id 0 --num_shards 8

This script is designed for repo-based sharding. If your Pro dataset/prediction
set has 8 repositories, use --num_shards 8.

For each instance assigned to the shard, and for each oracle temperature:

1. Reads:
       predictions_from_replacements/<MODEL>/<instance_id>/Replacement_Temp_0.0_oracle.jsonl

2. Extracts:
       model_patch

3. Creates a .pred file under:
       predfiles/<instance_id>/temp0_0/<instance_id>/Replacement_Temp_0.0_oracle.pred

   The extra <instance_id> subfolder is intentional because gather_patches.py
   expects a directory containing instance folders.

4. Runs:
       SWE-bench_Pro-os/helper_code/gather_patches.py

   and writes:
       jsonfiles/<instance_id>/Replacement_Temp_0.0_oracle.json

5. Looks up dockerhub_tag from ScaleAI/SWE-bench_Pro test split.

6. Pulls:
       jefzda/sweap-images:<dockerhub_tag>

7. Runs, from inside SWE-bench_Pro-os:
       python swe_bench_pro_eval.py ...

   with:
       --use_local_docker
       --num_workers 4
       --output_dir on /local/scratch

8. Reads:
       /local/scratch/shirin/sweap_outputs/.../eval_results.json

9. Writes partial CSV after every temperature evaluation:
       swebench_csv_results_partial/repo_shard_<id>_of_<n>.csv

10. Writes final shard CSV:
       swebench_csv_results/repo_shard_<id>_of_<n>.csv

11. Removes the instance Docker image after all 11 temperatures are evaluated.

CSV values:

    1              patch resolved the issue
    0              patch did not resolve the issue
    MISSING        prediction JSONL file missing
    EMPTY_PATCH    model_patch empty
    ERROR          evaluator/gather/docker failed
    NO_REPORT      eval_results.json missing
    UNKNOWN        eval_results.json exists but instance_id missing
"""


# ============================================================
# Required settings from your request
# ============================================================

DATASET_NAME = "ScaleAI/SWE-bench_Pro"
SPLIT = "test"
MODEL = "Qwen_Qwen3-235B-A22B-Instruct-2507-tput"

PRED_ROOT = Path("predictions_from_replacements") / MODEL

CSV_RESULTS_DIR = Path("swebench_csv_results") / MODEL
CSV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_RESULTS_DIR_PARTIAL = Path("swebench_csv_results_partial") / MODEL
CSV_RESULTS_DIR_PARTIAL.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [round(i / 10, 1) for i in range(11)]

# You said you removed non_oracle.
PROMPT_TYPES = ["oracle"]

# You requested 4 workers per shard.
MAX_WORKERS = "4"


# ============================================================
# SWE-bench Pro evaluator settings
# ============================================================

SWE_BENCH_PRO_DIR = Path("SWE-bench_Pro-os")
SWE_BENCH_PRO_EVAL = SWE_BENCH_PRO_DIR / "swe_bench_pro_eval.py"
GATHER_PATCHES = SWE_BENCH_PRO_DIR / "helper_code" / "gather_patches.py"

RAW_SAMPLE_CSV = Path("swe_bench_pro_full.csv")

PREDFILES_ROOT = Path("predfiles")
JSONFILES_ROOT = Path("jsonfiles")

# Important: Docker cannot bind-mount your /srv/scratch path on Kyle.
# Use /local/scratch for evaluator output/workspaces.
LOCAL_OUTPUT_ROOT = Path(
    os.environ.get("SWEAP_OUTPUT_ROOT", "/local/scratch/shirin/sweap_outputs")
)

EVAL_LOG_ROOT = Path("sweap_eval_logs")

DOCKERHUB_USERNAME = "jefzda"

# The evaluator already uses --redo. Set False if you want to reuse old outputs.
REDO_EVAL = True

# Run docker sanity checks once per shard.
RUN_DOCKER_SANITY_CHECK = True

# Remove Docker image after all 11 temperatures for an instance are done.
REMOVE_IMAGE_AFTER_INSTANCE = True


# ============================================================
# Basic helpers
# ============================================================

def temp_to_str(temp: float) -> str:
    return f"{temp:.1f}"


def temp_to_safe(temp_s: str) -> str:
    return temp_s.replace(".", "_")


def prediction_base_name(temp_s: str, prompt_type: str) -> str:
    return f"Replacement_Temp_{temp_s}_{prompt_type}"


def csv_col_name(temp_s: str, prompt_type: str) -> str:
    return f"Temp_{temp_s}_{prompt_type}"


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    cwd_str = str(cwd) if cwd is not None else None

    print()
    print("=" * 100)
    print("COMMAND:")
    print(" ".join(cmd))
    if cwd_str:
        print(f"CWD: {cwd_str}")
    print("=" * 100)

    result = subprocess.run(
        cmd,
        cwd=cwd_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND:\n"
            + " ".join(cmd)
            + "\n\nCWD:\n"
            + str(cwd_str)
            + "\n\nRETURN CODE:\n"
            + str(result.returncode)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr,
            encoding="utf-8",
        )
        print(f"Wrote log: {log_path}")

    if result.stdout.strip():
        print("STDOUT:")
        print(result.stdout[-4000:])

    if result.stderr.strip():
        print("STDERR:")
        print(result.stderr[-4000:])

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")

    return result


def save_partial(rows: List[dict], partial_csv: Path) -> None:
    partial_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(partial_csv, index=False)
    print(f"Wrote partial CSV: {partial_csv}")


def safe_path_component(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


# ============================================================
# Dataset helpers
# ============================================================

def load_dataset_rows() -> Dict[str, dict]:
    print(f"Loading dataset: {DATASET_NAME}, split={SPLIT}")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    rows = {}
    for row in dataset:
        rows[row["instance_id"]] = dict(row)

    if not RAW_SAMPLE_CSV.exists():
        tmp_path = RAW_SAMPLE_CSV.with_suffix(f".tmp.{os.getpid()}.csv")
        print(f"Writing raw sample CSV: {RAW_SAMPLE_CSV}")
        dataset.to_csv(str(tmp_path))
        os.replace(tmp_path, RAW_SAMPLE_CSV)
    else:
        print(f"Raw sample CSV already exists: {RAW_SAMPLE_CSV}")

    return rows


def normalize_raw_sample_csv_if_needed() -> None:
    """
    Some versions/comments mention FAIL_TO_PASS/PASS_TO_PASS,
    but swe_bench_pro_eval.py reads fail_to_pass/pass_to_pass.
    This keeps both if needed.
    """
    if not RAW_SAMPLE_CSV.exists():
        return

    df = pd.read_csv(RAW_SAMPLE_CSV)
    changed = False

    if "fail_to_pass" not in df.columns and "FAIL_TO_PASS" in df.columns:
        df["fail_to_pass"] = df["FAIL_TO_PASS"]
        changed = True

    if "pass_to_pass" not in df.columns and "PASS_TO_PASS" in df.columns:
        df["pass_to_pass"] = df["PASS_TO_PASS"]
        changed = True

    if changed:
        df.to_csv(RAW_SAMPLE_CSV, index=False)
        print(f"Normalized raw sample CSV columns in: {RAW_SAMPLE_CSV}")


def parse_repo_from_instance_id(instance_id: str) -> str:
    x = instance_id.strip()
    if x.startswith("instance_"):
        x = x[len("instance_"):]
    repo_part = x.split("-", 1)[0]
    if "__" not in repo_part:
        return "UNKNOWN_REPO"
    owner, repo = repo_part.split("__", 1)
    return f"{owner}/{repo}"


def group_instance_dirs_by_repo(
    instance_dirs: List[Path],
    rows_by_instance_id: Dict[str, dict],
) -> Dict[str, List[Path]]:
    repo_to_dirs = defaultdict(list)

    for instance_dir in instance_dirs:
        instance_id = instance_dir.name

        if instance_id not in rows_by_instance_id:
            print(f"WARNING: instance not found in dataset, skipping: {instance_id}")
            continue

        row = rows_by_instance_id[instance_id]
        repo = row.get("repo") or parse_repo_from_instance_id(instance_id)
        repo_to_dirs[repo].append(instance_dir)

    return dict(repo_to_dirs)


def get_assigned_repos(repo_names: List[str], shard_id: int, num_shards: int) -> List[str]:
    """
    If num_shards == number of repos, this gives one repo per shard.
    If not, it distributes repos by modulo.
    """
    return [
        repo
        for i, repo in enumerate(repo_names)
        if i % num_shards == shard_id
    ]


# ============================================================
# Prediction / patch conversion
# ============================================================

def read_first_jsonl_object(path: Path) -> Optional[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def read_model_patch(jsonl_path: Path) -> Tuple[str, str]:
    """
    Returns:
        status, patch

    status is OK, EMPTY_PATCH, or ERROR.
    """
    try:
        obj = read_first_jsonl_object(jsonl_path)
    except Exception as e:
        print(f"ERROR reading JSONL {jsonl_path}: {e}")
        return "ERROR", ""

    if not isinstance(obj, dict):
        print(f"ERROR: first JSONL object is not a dict: {jsonl_path}")
        return "ERROR", ""

    patch = obj.get("model_patch", "")

    if patch is None:
        patch = ""

    if not isinstance(patch, str):
        print(f"ERROR: model_patch is not a string: {jsonl_path}")
        return "ERROR", ""

    if not patch.strip():
        return "EMPTY_PATCH", ""

    return "OK", patch


def create_pred_file_for_gather(
    instance_id: str,
    temp_s: str,
    prompt_type: str,
    patch: str,
) -> Path:
    """
    Creates a temp-specific gather root:

        predfiles/<instance_id>/temp0_0/<instance_id>/Replacement_Temp_0.0_oracle.pred

    gather_patches.py should be called with:

        --directory predfiles/<instance_id>/temp0_0

    The extra instance_id subfolder is intentional because gather_patches.py
    expects directories of instance folders.
    """
    file_base = prediction_base_name(temp_s, prompt_type)
    temp_dir = f"temp{temp_to_safe(temp_s)}"

    pred_path = (
        PREDFILES_ROOT
        / instance_id
        / temp_dir
        / instance_id
        / f"{file_base}.pred"
    )

    pred_path.parent.mkdir(parents=True, exist_ok=True)

    pred_path.write_text(
        patch if patch.endswith("\n") else patch + "\n",
        encoding="utf-8",
    )

    print(f"Wrote .pred file: {pred_path}")
    return pred_path


def run_gather_patches(
    instance_id: str,
    temp_s: str,
    prompt_type: str,
) -> Tuple[str, Optional[Path]]:
    file_base = prediction_base_name(temp_s, prompt_type)
    temp_dir = f"temp{temp_to_safe(temp_s)}"

    gather_directory = PREDFILES_ROOT / instance_id / temp_dir
    output_json = JSONFILES_ROOT / instance_id / f"{file_base}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    log_path = (
        EVAL_LOG_ROOT
        / "gather_patches"
        / instance_id
        / f"{file_base}.log"
    )

    cmd = [
        sys.executable,
        str(GATHER_PATCHES),
        "--directory",
        str(gather_directory),
        "--prefix",
        MODEL,
        "--output",
        str(output_json),
    ]

    result = run_cmd(cmd, cwd=Path.cwd(), log_path=log_path, check=False)

    if result.returncode != 0:
        return "ERROR", None

    if not output_json.exists():
        print(f"ERROR: gather_patches did not create output JSON: {output_json}")
        return "ERROR", None

    try:
        with output_json.open("r", encoding="utf-8") as f:
            patches = json.load(f)
    except Exception as e:
        print(f"ERROR: could not read gathered patch JSON {output_json}: {e}")
        return "ERROR", None

    if not isinstance(patches, list) or len(patches) == 0:
        print(f"ERROR: gathered patch JSON is empty or invalid: {output_json}")
        return "ERROR", None

    # Strong sanity check: this file should only contain this one instance.
    ids = {p.get("instance_id") for p in patches if isinstance(p, dict)}
    if ids != {instance_id}:
        print(f"WARNING: gathered patch JSON has unexpected instance IDs: {ids}")

    print(f"Wrote gathered patch JSON: {output_json}")
    return "OK", output_json


# ============================================================
# Docker helpers
# ============================================================

def docker_sanity_check() -> None:
    if not RUN_DOCKER_SANITY_CHECK:
        return

    run_cmd(["docker", "--version"], check=True)
    run_cmd(["docker", "info"], check=True)
    run_cmd(["docker", "run", "hello-world"], check=True)
    run_cmd(["docker", "images"], check=False)


def get_full_image(row: dict) -> str:
    docker_tag = row.get("dockerhub_tag")
    if not docker_tag:
        raise ValueError(f"No dockerhub_tag in dataset row for {row.get('instance_id')}")
    return f"{DOCKERHUB_USERNAME}/sweap-images:{docker_tag}"


def docker_pull_image(full_image: str, instance_id: str) -> str:
    log_path = EVAL_LOG_ROOT / "docker_pull" / f"{safe_path_component(instance_id)}.log"
    result = run_cmd(["docker", "pull", full_image], log_path=log_path, check=False)

    if result.returncode != 0:
        return "ERROR"

    return "OK"


def docker_remove_image(full_image: str, instance_id: str) -> None:
    if not REMOVE_IMAGE_AFTER_INSTANCE:
        return

    print()
    print("=" * 100)
    print(f"Removing Docker image for instance: {instance_id}")
    print(full_image)
    print("=" * 100)

    # Remove containers using this image, if any remain.
    ps_result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"ancestor={full_image}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    container_ids = [x.strip() for x in ps_result.stdout.splitlines() if x.strip()]

    if container_ids:
        run_cmd(["docker", "rm", "-f"] + container_ids, check=False)

    run_cmd(["docker", "rmi", "-f", full_image], check=False)

    # Remove dangling layers only, not all images.
    run_cmd(["docker", "image", "prune", "-f"], check=False)


# ============================================================
# Evaluator
# ============================================================

def run_swe_bench_pro_eval(
    instance_id: str,
    temp_s: str,
    prompt_type: str,
    patch_json: Path,
) -> Tuple[str, Optional[bool]]:
    """
    Returns:
        status, value

    status:
        OK
        ERROR
        NO_REPORT
        UNKNOWN

    value:
        True / False if status == OK
    """
    file_base = prediction_base_name(temp_s, prompt_type)

    shard_output_dir = (
        LOCAL_OUTPUT_ROOT
        / f"{instance_id}"
        / file_base
    )
    shard_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        EVAL_LOG_ROOT
        / "swe_bench_pro_eval"
        / instance_id
        / f"{file_base}.log"
    )

    cmd = [
        sys.executable,
        "swe_bench_pro_eval.py",
        "--raw_sample_path",
        str((Path.cwd() / RAW_SAMPLE_CSV).resolve()),
        "--patch_path",
        str((Path.cwd() / patch_json).resolve()),
        "--output_dir",
        str(shard_output_dir.resolve()),
        "--scripts_dir",
        "run_scripts",
        "--num_workers",
        MAX_WORKERS,
        "--dockerhub_username",
        DOCKERHUB_USERNAME,
        "--use_local_docker",
    ]

    if REDO_EVAL:
        cmd.append("--redo")

    result = run_cmd(
        cmd,
        cwd=SWE_BENCH_PRO_DIR,
        log_path=log_path,
        check=False,
    )

    eval_results_path = shard_output_dir / "eval_results.json"

    if not eval_results_path.exists():
        if result.returncode != 0:
            return "ERROR", None
        return "NO_REPORT", None

    try:
        with eval_results_path.open("r", encoding="utf-8") as f:
            eval_results = json.load(f)
    except Exception as e:
        print(f"ERROR: could not read eval_results.json: {eval_results_path}: {e}")
        return "ERROR", None

    if instance_id not in eval_results:
        print(f"UNKNOWN: instance_id not found in eval_results.json: {instance_id}")
        return "UNKNOWN", None

    value = bool(eval_results[instance_id])

    # Distinguish real test failure from evaluator failure when possible.
    # If the value is False but no per-instance parsed output exists, the evaluator likely failed.
    instance_output_dir = shard_output_dir / instance_id
    expected_output_json = instance_output_dir / f"{MODEL}_output.json"

    if value is False and not expected_output_json.exists():
        print(
            "WARNING: eval_results.json says false, but per-instance output JSON "
            f"does not exist: {expected_output_json}. Marking as ERROR."
        )
        return "ERROR", None

    return "OK", value


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"Invalid shard_id={args.shard_id}. "
            f"With num_shards={args.num_shards}, shard_id must be 0..{args.num_shards - 1}."
        )

    root = Path.cwd()
    print(f"Running from: {root}")

    if not PRED_ROOT.exists():
        raise FileNotFoundError(f"Prediction root not found: {PRED_ROOT}")

    if not SWE_BENCH_PRO_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find {SWE_BENCH_PRO_DIR}. Clone it first:\n"
            f"git clone https://github.com/scaleapi/SWE-bench_Pro-os.git"
        )

    if not SWE_BENCH_PRO_EVAL.exists():
        raise FileNotFoundError(f"Cannot find evaluator: {SWE_BENCH_PRO_EVAL}")

    if not GATHER_PATCHES.exists():
        raise FileNotFoundError(f"Cannot find gather_patches.py: {GATHER_PATCHES}")

    CSV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_RESULTS_DIR_PARTIAL.mkdir(parents=True, exist_ok=True)
    PREDFILES_ROOT.mkdir(parents=True, exist_ok=True)
    JSONFILES_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_LOG_ROOT.mkdir(parents=True, exist_ok=True)

    rows_by_instance = {}

    results_csv = CSV_RESULTS_DIR / f"repo_shard_{args.shard_id}_of_{args.num_shards}.csv"
    partial_csv = CSV_RESULTS_DIR_PARTIAL / f"repo_shard_{args.shard_id}_of_{args.num_shards}.csv"

    rows_by_id = load_dataset_rows()
    normalize_raw_sample_csv_if_needed()

    docker_sanity_check()

    instance_dirs = sorted([p for p in PRED_ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(instance_dirs)} total instance prediction folders.")

    repo_to_dirs = group_instance_dirs_by_repo(instance_dirs, rows_by_id)
    repo_names = sorted(repo_to_dirs.keys())

    print(f"Found {len(repo_names)} repositories:")
    for i, repo in enumerate(repo_names):
        print(f"  [{i}] {repo}")

    assigned_repos = get_assigned_repos(repo_names, args.shard_id, args.num_shards)

    print()
    print("#" * 100)
    print(f"Shard {args.shard_id}/{args.num_shards}")
    print(f"Assigned repos: {len(assigned_repos)}")
    for repo in assigned_repos:
        print(f"  - {repo} ({len(repo_to_dirs[repo])} instances)")
    print("#" * 100)
    print()

    assigned_instance_dirs = []
    for repo in assigned_repos:
        assigned_instance_dirs.extend(sorted(repo_to_dirs[repo]))

    if not assigned_instance_dirs:
        print("This shard has no assigned instances.")
        pd.DataFrame([]).to_csv(results_csv, index=False)
        return

    for instance_dir in assigned_instance_dirs:
        instance_id = instance_dir.name

        row_data = rows_by_id.get(instance_id)
        if row_data is None:
            print(f"WARNING: instance not in dataset, skipping: {instance_id}")
            continue

        repo_name = row_data.get("repo") or parse_repo_from_instance_id(instance_id)
        full_image = get_full_image(row_data)

        row = {
            "instance_id": instance_id,
            "repo": repo_name,
        }

        print()
        print("=" * 100)
        print(f"Starting instance: {instance_id}")
        print(f"Repo: {repo_name}")
        print(f"Docker image: {full_image}")
        print("=" * 100)

        pull_status = docker_pull_image(full_image, instance_id)

        if pull_status != "OK":
            print(f"ERROR: Docker pull failed for {instance_id}. Marking all temps as ERROR.")
            for temp in TEMPERATURES:
                temp_s = temp_to_str(temp)
                for prompt_type in PROMPT_TYPES:
                    row[csv_col_name(temp_s, prompt_type)] = "ERROR"

            rows_by_instance[instance_id] = row
            save_partial(list(rows_by_instance.values()), partial_csv)
            continue

        try:
            for temp in TEMPERATURES:
                temp_s = temp_to_str(temp)

                for prompt_type in PROMPT_TYPES:
                    file_base = prediction_base_name(temp_s, prompt_type)
                    col_name = csv_col_name(temp_s, prompt_type)

                    prediction_path = instance_dir / f"{file_base}.jsonl"

                    print()
                    print("-" * 100)
                    print(f"Instance: {instance_id}")
                    print(f"Temperature: {temp_s}")
                    print(f"Prompt type: {prompt_type}")
                    print(f"Prediction path: {prediction_path}")
                    print("-" * 100)

                    if not prediction_path.exists():
                        print(f"MISSING: {prediction_path}")
                        row[col_name] = "MISSING"
                        rows_by_instance[instance_id] = row
                        save_partial(list(rows_by_instance.values()), partial_csv)
                        continue

                    patch_status, patch = read_model_patch(prediction_path)

                    if patch_status == "EMPTY_PATCH":
                        print(f"EMPTY_PATCH: {prediction_path}")
                        row[col_name] = "EMPTY_PATCH"
                        rows_by_instance[instance_id] = row
                        save_partial(list(rows_by_instance.values()), partial_csv)
                        continue

                    if patch_status != "OK":
                        print(f"ERROR reading patch: {prediction_path}")
                        row[col_name] = "ERROR"
                        rows_by_instance[instance_id] = row
                        save_partial(list(rows_by_instance.values()), partial_csv)
                        continue

                    create_pred_file_for_gather(
                        instance_id=instance_id,
                        temp_s=temp_s,
                        prompt_type=prompt_type,
                        patch=patch,
                    )

                    gather_status, patch_json = run_gather_patches(
                        instance_id=instance_id,
                        temp_s=temp_s,
                        prompt_type=prompt_type,
                    )

                    if gather_status != "OK" or patch_json is None:
                        print(f"ERROR: gather_patches failed for {instance_id} {file_base}")
                        row[col_name] = "ERROR"
                        rows_by_instance[instance_id] = row
                        save_partial(list(rows_by_instance.values()), partial_csv)
                        continue

                    eval_status, eval_value = run_swe_bench_pro_eval(
                        instance_id=instance_id,
                        temp_s=temp_s,
                        prompt_type=prompt_type,
                        patch_json=patch_json,
                    )

                    if eval_status == "OK":
                        row[col_name] = 1 if eval_value else 0
                    else:
                        row[col_name] = eval_status

                    print(f"Result for {instance_id} {col_name}: {row[col_name]}")

                    rows_by_instance[instance_id] = row
                    save_partial(list(rows_by_instance.values()), partial_csv)

        finally:
            docker_remove_image(full_image, instance_id)

    final_rows = list(rows_by_instance.values())
    final_df = pd.DataFrame(final_rows)

    ordered_cols = ["instance_id", "repo"]
    for temp in TEMPERATURES:
        temp_s = temp_to_str(temp)
        for prompt_type in PROMPT_TYPES:
            ordered_cols.append(csv_col_name(temp_s, prompt_type))

    for col in ordered_cols:
        if col not in final_df.columns:
            final_df[col] = ""

    final_df = final_df[ordered_cols].sort_values(["repo", "instance_id"]).reset_index(drop=True)
    final_df.to_csv(results_csv, index=False)

    print()
    print("=" * 100)
    print(f"Wrote final shard CSV: {results_csv}")
    print(f"Wrote partial shard CSV: {partial_csv}")
    print(f"Rows: {len(final_df)}")
    print("=" * 100)


if __name__ == "__main__":
    main()