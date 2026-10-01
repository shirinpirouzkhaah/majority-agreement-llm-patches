#!/usr/bin/env python3#

#maincode 

import argparse
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets import load_dataset
import numpy as np
from scipy.stats import mannwhitneyu



METRIC_CHOICES = ("bleu", "codebleu","codebert", "voyage")
MODEL_ALIASES = {
    "qwen": "Qwen_Qwen3-235B-A22B-Instruct-2507-tput",
    "gpt": "gpt-4.1-mini",
    "claude": "claude-haiku-4-5-20251001",
}


def parse_runtime_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clustering analysis for one model/similarity-metric configuration."
        )
    )
    parser.add_argument(
        "--metric",
        required=True,
        choices=METRIC_CHOICES,
        help="Similarity metric to use.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODEL_ALIASES.keys()),
        help="Model alias: qwen, gpt, or claude.",
    )
    return parser.parse_args()


# Runtime configuration must be resolved before the metric-specific imports and
# path constants below are created. Direct script execution therefore requires
# --metric and --model. Importing the module keeps the former defaults so helper
# functions can still be imported without consuming the caller's command line.
if __name__ == "__main__":
    _RUNTIME_ARGS = parse_runtime_args()
    metric = _RUNTIME_ARGS.metric
    MODEL = MODEL_ALIASES[_RUNTIME_ARGS.model]
else:
    metric = "voyage"
    MODEL = MODEL_ALIASES["claude"]


# DATASET_NAME = "ScaleAI/SWE-bench_Pro"
# SET = "Pro"


DATASET_NAME = "SWE-bench/SWE-bench_Verified"
SET = "Verified"


#RESULTS_CSV = Path("swebench_csv_resultss") / MODEL / "summary"/ "all_swebench_results.csv"
RESULTS_CSV = Path("swebench_csv_results") / MODEL / "final_swebench_results.csv"


# Every similarity module exposes the same three generic functions.
if metric == "codebert":
    from CodeBERT_similarity import (
        get_lang_for_repo,
        build_and_save_similarity_matrix,
        build_and_save_ground_truth_similarity_vector,
    )

elif metric == "codebleu":
    from codebleu_similarity import (
        get_lang_for_repo,
        build_and_save_similarity_matrix,
        build_and_save_ground_truth_similarity_vector,
    )
    
elif metric == "codebleu2":
    from codebleu2_similarity import (
        get_lang_for_repo,
        build_and_save_similarity_matrix,
        build_and_save_ground_truth_similarity_vector,
    )

elif metric == "voyage":
    from voyage_similarity import (
        get_lang_for_repo,
        build_and_save_similarity_matrix,
        build_and_save_ground_truth_similarity_vector,
    )
    
elif metric == "bleu":
    from bleu_similarity import (
        get_lang_for_repo,
        build_and_save_similarity_matrix,
        build_and_save_ground_truth_similarity_vector,
    )
else:
    raise ValueError(f"Unsupported similarity metric: {metric}")


SPLIT = "test"
dataset = load_dataset(DATASET_NAME, split=SPLIT)

INSTANCE_ID_TO_GROUND_TRUTH_PATCH = {
    str(row["instance_id"]).strip(): str(row["patch"])
    for row in dataset
    if row.get("instance_id") is not None
}

INSTANCE_ID_TO_REPO = {
    str(row["instance_id"]).strip(): str(row["repo"]).strip()
    for row in dataset
    if row.get("instance_id") is not None and row.get("repo") is not None
}

PRED_ROOT = Path("predictions_from_replacements") / MODEL
SYNTAX_STATUS_CSV = PRED_ROOT / "replacement_syntax_status.csv"
PROMPT_TYPE = "oracle"

BASE_OUTPUT_DIR = Path("clustering_results") / metric / MODEL / SET
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared matrix cache. Both nonsolvable and solvable subsets use this.
MATRIX_ROOT = Path("matrix") / metric / MODEL / SET

TEMPERATURES = [f"{i / 10:.1f}" for i in range(11)]
SIMILARITY_THRESHOLDS = [round(i / 10, 1) for i in range(1, 10)]
PAPER_FIXED_THRESHOLD = 0.9
CLOSE_TEMPERATURE_DISTANCE_LIMIT = 0.4

# Cluster-selection strategies used for RQ2 plots and threshold summaries.
# The current pipeline keeps singleton_furthest because it is already defined
# as a strategy in this code version.
CLUSTER_SELECTION_STRATEGIES = [
    "oracle",
    "largest",
    "smallest",
    "singleton_union",
    "singleton_furthest",
]

# The four deployable/oracle-free strategies used when selecting the highest F1.
ORACLE_FREE_CLUSTER_SELECTION_STRATEGIES = [
    "largest",
    "smallest",
    "singleton_union",
    "singleton_furthest",
]

STRATEGY_DISPLAY_NAMES = {
    "oracle": "oracle",
    "largest": "largest",
    "smallest": "smallest",
    "singleton_union": "singleton union",
    "singleton_furthest": "singleton furthest",
}

EXCLUDED_RESULT_STATUSES = {"MISSING", "EMPTY_PATCH"}



def make_output_paths(subset_name: str) -> dict:
    output_dir = BASE_OUTPUT_DIR / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "syntax_status_summary": output_dir / "syntax_status_summary.csv",
        "passing_status_summary": output_dir / "passing_status_summary.csv",
        "threshold_range": output_dir / "threshold_range_per_instance.csv",
        "range_averaged_f1": output_dir / "range_averaged_f1_per_instance.csv",
        "range_averaged_roc_auc": output_dir / "range_averaged_roc_auc_per_instance.csv",
        "oracle_per_instance": output_dir / "oracle_threshold_per_instance.csv",
        "global_optimal_thresholds": output_dir / "global_optimal_thresholds.csv",
        "threshold_summary": output_dir / "threshold_cluster_summary.csv",
        "threshold_average_metrics": output_dir / "threshold_average_metrics.csv",
        "rq2_average_f1_by_threshold": output_dir / "rq2_average_f1_by_threshold.csv",
        "rq2_average_cluster_size_by_threshold": output_dir / "rq2_average_cluster_size_by_threshold.csv",
        "best_configuration": output_dir / "best_configuration.csv",
        "issue_f1_matrix": output_dir / "issue_f1_by_threshold_and_strategy.csv",
        "issue_f1_by_threshold_dir": output_dir / "issue_f1_by_threshold",
        # Cross-threshold issue tables are written inside issue_f1_by_threshold after
        # the nine threshold_0.x.csv files have been created.
        "strategy_availability": output_dir / "strategy_availability_by_threshold.csv",
        "strategy_vs_matched_oracle": output_dir / "strategy_vs_matched_oracle_by_threshold.csv",
        "cluster_composition_summary": output_dir / "cluster_composition_summary.csv",
        "precision_plot": output_dir / "threshold_average_precision.pdf",
        "recall_plot": output_dir / "threshold_average_recall.pdf",
        "f1_plot": output_dir / "threshold_average_f1.pdf",
        "cluster_size_plot": output_dir / "threshold_average_cluster_size.pdf",
        "roc_auc_plot": output_dir / "threshold_average_roc_auc.pdf",
    }


def get_ground_truth_patch_for_instance(instance_id: str) -> str:
    instance_id = str(instance_id).strip()

    if instance_id not in INSTANCE_ID_TO_GROUND_TRUTH_PATCH:
        raise KeyError(f"No ground-truth patch found for instance_id={instance_id}")

    return INSTANCE_ID_TO_GROUND_TRUTH_PATCH[instance_id]


def get_repo_for_instance(instance_id: str) -> str:
    instance_id = str(instance_id).strip()

    if instance_id not in INSTANCE_ID_TO_REPO:
        raise KeyError(f"No repo found for instance_id={instance_id}")

    return INSTANCE_ID_TO_REPO[instance_id]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def read_prediction_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return {}

    if path.suffix == ".jsonl":
        first_line = text.splitlines()[0]
        return json.loads(first_line)

    return json.loads(text)


def extract_diff(prediction: dict) -> str:
    patch = prediction.get("model_patch")

    if patch is None:
        return ""

    return str(patch)


def find_prediction_file(
    instance_dir: Path,
    temp_s: str,
    prompt_type: str,
) -> Optional[Path]:
    base_name = f"Replacement_Temp_{temp_s}_{prompt_type}"
    candidate = instance_dir / f"{base_name}.jsonl"

    if candidate.exists():
        return candidate

    return None


def cluster_from_similarity_matrix(
    similarity_df: pd.DataFrame,
    similarity_threshold: float,
) -> dict:
    temps = list(similarity_df.index)

    graph = defaultdict(list)

    for temp_a in temps:
        for temp_b in temps:
            if temp_a == temp_b:
                continue

            score = float(similarity_df.loc[temp_a, temp_b])

            if score >= similarity_threshold:
                graph[temp_a].append(temp_b)

    visited = set()
    labels = {}
    current_cluster = 0

    for temp in temps:
        if temp in visited:
            continue

        queue = deque([temp])
        visited.add(temp)
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        for item in component:
            labels[item] = current_cluster

        current_cluster += 1

    return labels


def normalize_raw_result(value) -> str:
    if pd.isna(value):
        return "NaN"

    value_s = str(value).strip()

    if value_s in {"1", "1.0"}:
        return "1"

    if value_s in {"0", "0.0"}:
        return "0"

    return value_s


def normalize_test_result(value) -> int:
    value_s = normalize_raw_result(value)

    if value_s == "1":
        return 1

    return 0


def normalize_syntax_status(value) -> int:
    value_s = normalize_raw_result(value)

    if value_s == "1":
        return 1

    return 0


def get_syntax_column_for_temperature(
    syntax_df: pd.DataFrame,
    temp_s: str,
) -> Optional[str]:
    possible_col_names = [
        f"temp_{temp_s}",
        f"Temp_{temp_s}",
        f"Replacement_Temp_{temp_s}",
        f"Temp_{temp_s}_{PROMPT_TYPE}",
        f"Replacement_Temp_{temp_s}_{PROMPT_TYPE}",
        temp_s,
    ]

    for col_name in possible_col_names:
        if col_name in syntax_df.columns:
            return col_name

    return None


def get_correct_syntax_temperatures_for_instance(
    syntax_df: pd.DataFrame,
    instance_id: str,
) -> list:
    matching = syntax_df[syntax_df["instance_id"] == instance_id]

    if matching.empty:
        return []

    row = matching.iloc[0]
    correct_syntax_temperatures = []

    for temp_s in TEMPERATURES:
        col_name = get_syntax_column_for_temperature(
            syntax_df=syntax_df,
            temp_s=temp_s,
        )

        if col_name is None:
            continue

        if normalize_syntax_status(row[col_name]) == 1:
            correct_syntax_temperatures.append(temp_s)

    return correct_syntax_temperatures


def report_syntax_status_summary(
    syntax_df: pd.DataFrame,
    paths_by_subset: dict,
) -> pd.DataFrame:
    rows = []

    for _, row in syntax_df.iterrows():
        correct_count = 0

        for temp_s in TEMPERATURES:
            col_name = get_syntax_column_for_temperature(
                syntax_df=syntax_df,
                temp_s=temp_s,
            )

            if col_name is None:
                continue

            correct_count += normalize_syntax_status(row[col_name])

        rows.append({
            "instance_id": str(row["instance_id"]).strip(),
            "correct_syntax_count": int(correct_count),
        })

    per_issue_df = pd.DataFrame(rows)

    summary_rows = []
    for correct_count in range(len(TEMPERATURES), -1, -1):
        num_issues = int((per_issue_df["correct_syntax_count"] == correct_count).sum())
        summary_rows.append({
            "correct_syntax_temperatures": int(correct_count),
            "num_issues": num_issues,
        })

    summary_df = pd.DataFrame(summary_rows)
    average_correct_syntax_per_issue = float(
        per_issue_df["correct_syntax_count"].mean()
    ) if not per_issue_df.empty else 0.0

    for paths in paths_by_subset.values():
        summary_df.to_csv(paths["syntax_status_summary"], index=False)

    print()
    print("=" * 80)
    print("SYNTAX STATUS SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"Average correct syntax per issue: {average_correct_syntax_per_issue:.4f}")

    return summary_df



def report_passing_status_summary(
    results_df: pd.DataFrame,
    result_columns: list,
    paths_by_subset: dict,
) -> pd.DataFrame:
    rows = []

    for _, row in results_df.iterrows():
        passing_count = int(
            sum(
                normalize_test_result(row[col_name])
                for col_name in result_columns
            )
        )

        rows.append({
            "instance_id": str(row["instance_id"]).strip(),
            "passing_count": passing_count,
        })

    per_issue_df = pd.DataFrame(rows)

    summary_rows = []
    for passing_count in range(len(TEMPERATURES), -1, -1):
        num_issues = int((per_issue_df["passing_count"] == passing_count).sum())
        summary_rows.append({
            "passing_temperatures": int(passing_count),
            "num_issues": num_issues,
        })

    summary_df = pd.DataFrame(summary_rows)
    average_passing_per_issue = float(
        per_issue_df["passing_count"].mean()
    ) if not per_issue_df.empty else 0.0

    for paths in paths_by_subset.values():
        summary_df.to_csv(paths["passing_status_summary"], index=False)

    print()
    print("=" * 80)
    print("PASSING STATUS SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"Average passing candidates per issue: {average_passing_per_issue:.4f}")

    return summary_df


def is_temperature_usable_for_clustering(value) -> bool:
    value_s = normalize_raw_result(value)

    if value_s in EXCLUDED_RESULT_STATUSES:
        return False

    if value_s in {"1", "0"}:
        return True

    return False


def get_result_columns_for_prompt_type(
    results_df: pd.DataFrame,
    prompt_type: str,
) -> list:
    result_columns = []

    for temp_s in TEMPERATURES:
        possible_col_names = [
            f"Temp_{temp_s}_{prompt_type}",
            f"Replacement_Temp_{temp_s}_{prompt_type}",
        ]

        for col_name in possible_col_names:
            if col_name in results_df.columns:
                result_columns.append(col_name)
                break

    return result_columns


def get_result_column_for_temperature(
    results_df: pd.DataFrame,
    temp_s: str,
    prompt_type: str,
) -> Optional[str]:
    possible_col_names = [
        f"Temp_{temp_s}_{prompt_type}",
        f"Replacement_Temp_{temp_s}_{prompt_type}",
    ]

    for col_name in possible_col_names:
        if col_name in results_df.columns:
            return col_name

    return None


def row_has_at_least_one_pass(row, result_columns: list) -> bool:
    for col_name in result_columns:
        if normalize_test_result(row[col_name]) == 1:
            return True

    return False


def get_test_pass_for_temperature(
    results_df: pd.DataFrame,
    instance_id: str,
    temp_s: str,
    prompt_type: str,
) -> int:
    matching = results_df[results_df["instance_id"] == instance_id]

    if matching.empty:
        return 0

    col_name = get_result_column_for_temperature(
        results_df=results_df,
        temp_s=temp_s,
        prompt_type=prompt_type,
    )

    if col_name is None:
        return 0

    value = matching.iloc[0][col_name]
    return normalize_test_result(value)


def get_usable_temperatures_for_instance(
    results_df: pd.DataFrame,
    syntax_df: pd.DataFrame,
    instance_id: str,
    prompt_type: str,
) -> list:
    matching = results_df[results_df["instance_id"] == instance_id]

    if matching.empty:
        return []

    row = matching.iloc[0]
    correct_syntax_temperatures = set(
        get_correct_syntax_temperatures_for_instance(
            syntax_df=syntax_df,
            instance_id=instance_id,
        )
    )
    usable_temperatures = []

    for temp_s in TEMPERATURES:
        if temp_s not in correct_syntax_temperatures:
            continue

        col_name = get_result_column_for_temperature(
            results_df=results_df,
            temp_s=temp_s,
            prompt_type=prompt_type,
        )

        if col_name is None:
            continue

        value = row[col_name]

        if is_temperature_usable_for_clustering(value):
            usable_temperatures.append(temp_s)

    return usable_temperatures


def load_valid_candidates_for_instance(
    instance_dir: Path,
    prompt_type: str,
    allowed_temperatures: list,
) -> dict:
    temperature_to_diff = {}

    for temp_s in allowed_temperatures:
        pred_path = find_prediction_file(
            instance_dir=instance_dir,
            temp_s=temp_s,
            prompt_type=prompt_type,
        )

        if pred_path is None:
            continue

        try:
            prediction = read_prediction_file(pred_path)
            diff = extract_diff(prediction)
        except Exception as e:
            print(f"WARNING: failed to read {pred_path}: {e}")
            continue

        if not diff.strip():
            continue

        temperature_to_diff[temp_s] = diff

    return temperature_to_diff


def load_all_candidates_for_similarity(
    instance_dir: Path,
    prompt_type: str,
) -> dict:
    """Load candidate patches before syntax/result/solvability filtering.

    Every configured temperature is attempted. A candidate is not removed because
    of syntax status, SWE-bench result, EMPTY_PATCH status, or solvability. If the
    prediction file exists, its model_patch value is included even when it is an
    empty string. Missing/unreadable prediction files are reported because no
    similarity can be computed for data that are not physically available.
    """
    temperature_to_diff = {}
    unavailable_temperatures = []

    for temp_s in TEMPERATURES:
        pred_path = find_prediction_file(
            instance_dir=instance_dir,
            temp_s=temp_s,
            prompt_type=prompt_type,
        )

        if pred_path is None:
            unavailable_temperatures.append(temp_s)
            continue

        try:
            prediction = read_prediction_file(pred_path)
            diff = extract_diff(prediction)
        except Exception as exc:
            print(
                f"WARNING: failed to read {pred_path} before filtering: {exc}"
            )
            unavailable_temperatures.append(temp_s)
            continue

        # Deliberately keep empty model_patch values here. Filtering happens only
        # later, when the valid clustering temperatures are determined.
        temperature_to_diff[temp_s] = diff

    if unavailable_temperatures:
        print(
            f"WARNING: {instance_dir.name} has no readable prediction for "
            f"temperature(s) {unavailable_temperatures}. The raw similarity CSV "
            "will contain every physically available temperature; no syntax or "
            "test-result filtering was applied."
        )

    return temperature_to_diff


def report_prediction_file_availability_before_similarity(
    instance_dirs: list,
    prompt_type: str,
):
    """Print prediction-file availability across all issue folders.

    This diagnostic is independent of the matrix cache. Every prediction issue
    folder and every configured temperature is inspected so cached issues are
    included in the availability totals. A readable file counts as available even
    when model_patch is empty, matching the raw-similarity precomputation policy.
    """
    missing_prediction_files = 0
    unreadable_prediction_files = 0
    missing_only_histogram = {
        missing_count: 0
        for missing_count in range(len(TEMPERATURES) + 1)
    }
    unavailable_histogram = {
        missing_count: 0
        for missing_count in range(len(TEMPERATURES) + 1)
    }

    for instance_dir in instance_dirs:
        missing_for_issue = 0
        unavailable_for_issue = 0

        for temp_s in TEMPERATURES:
            pred_path = find_prediction_file(
                instance_dir=instance_dir,
                temp_s=temp_s,
                prompt_type=prompt_type,
            )

            if pred_path is None:
                missing_prediction_files += 1
                missing_for_issue += 1
                unavailable_for_issue += 1
                continue

            try:
                read_prediction_file(pred_path)
            except Exception:
                unreadable_prediction_files += 1
                unavailable_for_issue += 1

        missing_only_histogram[missing_for_issue] += 1
        unavailable_histogram[unavailable_for_issue] += 1

    total_unavailable_prediction_files = (
        missing_prediction_files + unreadable_prediction_files
    )
    total_expected_prediction_files = len(instance_dirs) * len(TEMPERATURES)

    print()
    print("=" * 80)
    print("PREDICTION FILE AVAILABILITY BEFORE SIMILARITY PRECOMPUTATION")
    print("=" * 80)
    print(f"Total issue folders under prediction root: {len(instance_dirs)}")
    print(f"Configured temperatures per issue: {len(TEMPERATURES)}")
    print(f"Expected prediction files: {total_expected_prediction_files}")
    print(f"Missing prediction files: {missing_prediction_files}")
    print(f"Unreadable prediction files: {unreadable_prediction_files}")
    print(
        "Total missing/unreadable prediction files: "
        f"{total_unavailable_prediction_files}"
    )
    print("Issues by number of missing prediction files:")
    for missing_count in range(len(TEMPERATURES) + 1):
        print(
            f"  {missing_count} missing: "
            f"{missing_only_histogram[missing_count]} issue(s)"
        )
    print("Issues by number of missing/unreadable prediction files:")
    for missing_count in range(len(TEMPERATURES) + 1):
        print(
            f"  {missing_count} missing/unreadable: "
            f"{unavailable_histogram[missing_count]} issue(s)"
        )
    print("=" * 80)


def precompute_similarity_files_before_filtering():
    """Create raw pairwise and ground-truth similarity CSVs before filtering.

    This phase intentionally does not read syntax status or SWE-bench results.
    The only cache-based skip condition is the existence of
    MATRIX_ROOT/<instance_id>. If that directory exists, the issue is left
    untouched. Otherwise, similarities are computed from all physically
    available temperature predictions for that issue.
    """
    MATRIX_ROOT.mkdir(parents=True, exist_ok=True)

    instance_dirs = sorted(path for path in PRED_ROOT.iterdir() if path.is_dir())

    print()
    print("=" * 80)
    print("PRECOMPUTING RAW SIMILARITIES BEFORE ANY CLUSTERING FILTER")
    print("=" * 80)
    print(f"Prediction folders found: {len(instance_dirs)}")
    print(f"Matrix root: {MATRIX_ROOT}")

    report_prediction_file_availability_before_similarity(
        instance_dirs=instance_dirs,
        prompt_type=PROMPT_TYPE,
    )

    built_count = 0
    skipped_existing_count = 0
    unavailable_count = 0

    for instance_dir in instance_dirs:
        instance_id = instance_dir.name
        instance_matrix_dir = MATRIX_ROOT / str(instance_id)

        # User-requested cache rule: existence of the instance directory is the
        # only filtering/skip decision in this precomputation phase.
        if instance_matrix_dir.exists():
            skipped_existing_count += 1
            continue

        try:
            ground_truth_patch = get_ground_truth_patch_for_instance(instance_id)
            repo = get_repo_for_instance(instance_id)
        except KeyError as exc:
            # Similarity cannot be produced without the benchmark reference/repo.
            # This is treated as unavailable source data, not as a validity filter.
            print(f"WARNING: cannot precompute similarity for {instance_id}: {exc}")
            unavailable_count += 1
            continue

        temperature_to_diff = load_all_candidates_for_similarity(
            instance_dir=instance_dir,
            prompt_type=PROMPT_TYPE,
        )

        if not temperature_to_diff:
            print(
                f"WARNING: cannot precompute similarity for {instance_id}: "
                "no readable prediction files were found."
            )
            unavailable_count += 1
            continue

        similarity_lang = get_lang_for_repo(repo)

        # These module functions save the raw similarity values used directly
        # by clustering after the valid-temperature subset is selected.
        build_and_save_similarity_matrix(
            temperature_to_diff=temperature_to_diff,
            matrix_root=MATRIX_ROOT,
            instance_id=instance_id,
            lang=similarity_lang,
            metric_name=metric,
        )
        build_and_save_ground_truth_similarity_vector(
            ground_truth_patch=ground_truth_patch,
            temperature_to_diff=temperature_to_diff,
            matrix_root=MATRIX_ROOT,
            instance_id=instance_id,
            lang=similarity_lang,
            metric_name=metric,
        )
        built_count += 1

    print(f"New instance similarity folders built: {built_count}")
    print(f"Existing instance folders skipped: {skipped_existing_count}")
    print(f"Instances unavailable for precomputation: {unavailable_count}")
    print("=" * 80)


def get_or_build_similarity_matrix(
    temperature_to_diff: dict,
    matrix_root: Path,
    instance_id: str,
    lang: str,
    metric_name: str = metric,
) -> tuple[pd.DataFrame, Path]:
    """Load the raw precomputed matrix and subset it for clustering.

    No similarity CSV is created or rewritten during clustering. Similarity files
    must already have been produced by precompute_similarity_files_before_filtering().
    The raw stored similarities are used directly for every metric.
    """
    del lang

    matrix_path = (
        Path(matrix_root)
        / str(instance_id)
        / metric_name
        / "similarity_matrix.csv"
    )

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Precomputed similarity matrix is missing for {instance_id}: "
            f"{matrix_path}. Delete {Path(matrix_root) / str(instance_id)} if it "
            "is a stale/incomplete cache directory, then rerun so the full raw "
            "similarity files are created before filtering."
        )

    raw_similarity_df = pd.read_csv(matrix_path, index_col=0)
    raw_similarity_df.index = raw_similarity_df.index.astype(str)
    raw_similarity_df.columns = raw_similarity_df.columns.astype(str)

    valid_temperatures = [str(temp) for temp in temperature_to_diff.keys()]
    missing_temperatures = [
        temp
        for temp in valid_temperatures
        if temp not in raw_similarity_df.index or temp not in raw_similarity_df.columns
    ]

    if missing_temperatures:
        raise ValueError(
            f"Precomputed raw matrix for {instance_id} is missing clustering "
            f"temperature(s) {missing_temperatures}. This usually means the cache "
            "was created by the older filtered pipeline. Delete the instance "
            "folder under MATRIX_ROOT and rerun."
        )

    raw_similarity_df = raw_similarity_df.loc[
        valid_temperatures,
        valid_temperatures,
    ]

    return raw_similarity_df, matrix_path


def compute_cluster_structure_counts(
    labels: dict,
    num_total_patches: int,
) -> dict:
    """
    Counts cluster structure for one issue at one threshold.

    non_full_issue_cluster_count counts clusters whose size is smaller than
    the number of valid temperatures for that issue. For an issue with 11 valid
    temperatures, a cluster of size 11 is a full-issue cluster; every smaller
    cluster is counted as non-full.
    """
    cluster_to_temps = defaultdict(list)

    for temp, cluster_id in labels.items():
        cluster_to_temps[cluster_id].append(temp)

    cluster_sizes = [
        len(cluster_temps)
        for cluster_temps in cluster_to_temps.values()
    ]

    singleton_cluster_count = sum(
        1 for size in cluster_sizes
        if size == 1
    )

    non_singleton_cluster_count = sum(
        1 for size in cluster_sizes
        if size > 1
    )

    full_issue_cluster_count = sum(
        1 for size in cluster_sizes
        if size == num_total_patches
    )

    non_full_issue_cluster_count = sum(
        1 for size in cluster_sizes
        if size < num_total_patches
    )

    return {
        "singleton_cluster_count": int(singleton_cluster_count),
        "non_singleton_cluster_count": int(non_singleton_cluster_count),
        "full_issue_cluster_count": int(full_issue_cluster_count),
        "non_full_issue_cluster_count": int(non_full_issue_cluster_count),
    }



def compute_fragmentation_g(labels: dict) -> float:
    """Return issue-level cluster fragmentation G_i(t) for one threshold.

    G_i(t) = 1 - [sum_k C(n_ik, 2)] / C(n_i, 2),
    where n_ik is the size of connected-component cluster k and n_i is the
    number of valid candidate patches for the issue.

    Interpretation: for n_i >= 2, G_i(t) is the probability that two distinct
    candidates chosen uniformly without replacement belong to different
    clusters. A single valid candidate is assigned G=0 because no fragmentation
    is possible.
    """
    num_candidates = len(labels)
    if num_candidates <= 1:
        return 0.0

    cluster_to_size = defaultdict(int)
    for cluster_id in labels.values():
        cluster_to_size[cluster_id] += 1

    total_pairs = math.comb(num_candidates, 2)
    same_cluster_pairs = sum(
        math.comb(cluster_size, 2)
        for cluster_size in cluster_to_size.values()
    )

    fragmentation_g = 1.0 - (same_cluster_pairs / total_pairs)
    # Protect against tiny floating-point excursions outside [0, 1].
    return float(min(1.0, max(0.0, fragmentation_g)))


def compute_cluster_density(
    cluster_temps: list,
    similarity_df: pd.DataFrame,
) -> float:
    if len(cluster_temps) == 1:
        return 1.0

    if len(cluster_temps) == 0:
        return 0.0

    pairwise_scores = []

    for i in range(len(cluster_temps)):
        for j in range(i + 1, len(cluster_temps)):
            temp_i = cluster_temps[i]
            temp_j = cluster_temps[j]

            score = float(similarity_df.loc[temp_i, temp_j])
            pairwise_scores.append(score)

    return safe_divide(sum(pairwise_scores), len(pairwise_scores))



def compute_temperature_coclustering_rates_for_issue(
    labels: dict,
    close_distance_limit: float = CLOSE_TEMPERATURE_DISTANCE_LIMIT,
) -> dict:
    """Compute close/far co-clustering rates for one issue at one threshold.

    Candidate pairs are split by decoding-temperature distance:
      close: |T_i - T_j| < close_distance_limit
      far:   |T_i - T_j| >= close_distance_limit

    The returned rates are issue-level quantities. If an issue has no eligible
    pairs for a rate, that rate is NaN so the issue is excluded from the
    corresponding macro average instead of being treated as a zero.
    """
    valid_temperatures = sorted(labels.keys(), key=float)

    close_total_pairs = 0
    close_same_cluster_pairs = 0
    far_total_pairs = 0
    far_same_cluster_pairs = 0

    for i in range(len(valid_temperatures)):
        for j in range(i + 1, len(valid_temperatures)):
            temp_i = valid_temperatures[i]
            temp_j = valid_temperatures[j]
            temperature_distance = round(
                abs(float(temp_i) - float(temp_j)),
                1,
            )

            same_cluster = labels[temp_i] == labels[temp_j]

            if temperature_distance < close_distance_limit:
                close_total_pairs += 1
                if same_cluster:
                    close_same_cluster_pairs += 1
            else:
                # The boundary belongs to the far group: distance >= limit.
                far_total_pairs += 1
                if same_cluster:
                    far_same_cluster_pairs += 1

    close_rate = (
        close_same_cluster_pairs / close_total_pairs
        if close_total_pairs > 0
        else np.nan
    )
    far_rate = (
        far_same_cluster_pairs / far_total_pairs
        if far_total_pairs > 0
        else np.nan
    )
    delta_rate = (
        close_rate - far_rate
        if not pd.isna(close_rate) and not pd.isna(far_rate)
        else np.nan
    )

    return {
        "close_temperature_coclustering_rate": float(close_rate)
        if not pd.isna(close_rate)
        else np.nan,
        "far_temperature_coclustering_rate": float(far_rate)
        if not pd.isna(far_rate)
        else np.nan,
        "close_minus_far_coclustering_rate": float(delta_rate)
        if not pd.isna(delta_rate)
        else np.nan,
    }


def summarize_issue_average_temperature_coclustering(
    issue_rate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Macro-average issue-level close, far, and close-minus-far rates.

    Every issue contributes equally. Close and far rates are averaged over the
    issues for which that rate is defined. The delta is first computed within
    each issue and is then averaged, so the delta uses only issues that have
    both close and far candidate pairs.
    """
    output_columns = [
        "threshold",
        "issue_average_close_temperature_coclustering_rate",
        "issue_average_far_temperature_coclustering_rate",
        "issue_average_close_minus_far_coclustering_rate",
    ]

    if issue_rate_df.empty:
        return pd.DataFrame(
            {
                "threshold": SIMILARITY_THRESHOLDS,
                "issue_average_close_temperature_coclustering_rate": np.nan,
                "issue_average_far_temperature_coclustering_rate": np.nan,
                "issue_average_close_minus_far_coclustering_rate": np.nan,
            },
            columns=output_columns,
        )

    required_columns = {
        "instance_id",
        "threshold",
        "close_temperature_coclustering_rate",
        "far_temperature_coclustering_rate",
        "close_minus_far_coclustering_rate",
    }
    missing_columns = required_columns - set(issue_rate_df.columns)
    if missing_columns:
        raise ValueError(
            "Cannot summarize issue-level temperature co-clustering because "
            f"columns are missing: {sorted(missing_columns)}"
        )

    rows = []
    work_df = issue_rate_df.copy()
    work_df["threshold"] = work_df["threshold"].astype(float)

    for threshold in SIMILARITY_THRESHOLDS:
        threshold_df = work_df[
            np.isclose(work_df["threshold"], float(threshold))
        ]

        rows.append({
            "threshold": float(threshold),
            "issue_average_close_temperature_coclustering_rate": float(
                threshold_df["close_temperature_coclustering_rate"].mean()
            ) if not threshold_df.empty else np.nan,
            "issue_average_far_temperature_coclustering_rate": float(
                threshold_df["far_temperature_coclustering_rate"].mean()
            ) if not threshold_df.empty else np.nan,
            "issue_average_close_minus_far_coclustering_rate": float(
                threshold_df["close_minus_far_coclustering_rate"].mean()
            ) if not threshold_df.empty else np.nan,
        })

    return pd.DataFrame(rows, columns=output_columns)

def compute_similarity_scores_to_cluster(
    cluster_temps: list,
    valid_temperatures: list,
    similarity_df: pd.DataFrame,
) -> list:
    if not cluster_temps:
        return [0.0 for _ in valid_temperatures]

    y_score = []

    for temp in valid_temperatures:
        similarities = [
            float(similarity_df.loc[temp, cluster_temp])
            for cluster_temp in cluster_temps
        ]

        score = safe_divide(sum(similarities), len(similarities))
        y_score.append(score)

    return y_score


def compute_auc_metrics(
    cluster_temps: list,
    valid_temperatures: list,
    temperature_to_pass: dict,
    similarity_df: pd.DataFrame,
) -> tuple:
    y_true = [
        int(temperature_to_pass[temp])
        for temp in valid_temperatures
    ]

    if len(set(y_true)) < 2:
        return 0.0, 0.0

    y_score = compute_similarity_scores_to_cluster(
        cluster_temps=cluster_temps,
        valid_temperatures=valid_temperatures,
        similarity_df=similarity_df,
    )

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    return float(roc_auc), float(pr_auc)


def compute_metrics_for_cluster(
    cluster_temps: list,
    valid_temperatures: list,
    temperature_to_pass: dict,
    similarity_df: pd.DataFrame,
) -> dict:
    cluster_temp_set = set(cluster_temps)

    tp = sum(
        1 for temp in valid_temperatures
        if temp in cluster_temp_set and temperature_to_pass[temp] == 1
    )

    fp = sum(
        1 for temp in valid_temperatures
        if temp in cluster_temp_set and temperature_to_pass[temp] == 0
    )

    fn = sum(
        1 for temp in valid_temperatures
        if temp not in cluster_temp_set and temperature_to_pass[temp] == 1
    )

    tn = sum(
        1 for temp in valid_temperatures
        if temp not in cluster_temp_set and temperature_to_pass[temp] == 0
    )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    specificity = safe_divide(tn, tn + fp)

    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )

    mcc = safe_divide(
        (tp * tn) - (fp * fn),
        mcc_denominator,
    )

    roc_auc, pr_auc = compute_auc_metrics(
        cluster_temps=cluster_temps,
        valid_temperatures=valid_temperatures,
        temperature_to_pass=temperature_to_pass,
        similarity_df=similarity_df,
    )

    cluster_density = compute_cluster_density(
        cluster_temps=cluster_temps,
        similarity_df=similarity_df,
    )

    return {
        "size": len(cluster_temps),
        "temperatures": cluster_temps,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "cluster_density": float(cluster_density),
    }


def empty_cluster_summary(
    total_passes: int,
    total_fails: int,
    prefix: str,
) -> dict:
    """Return NaN metrics when a selection strategy is structurally unavailable."""
    del total_passes, total_fails
    return {
        f"{prefix}_cluster_id": None,
        f"{prefix}_cluster_size": np.nan,
        f"{prefix}_cluster_tp": np.nan,
        f"{prefix}_cluster_fp": np.nan,
        f"{prefix}_cluster_fn": np.nan,
        f"{prefix}_cluster_tn": np.nan,
        f"{prefix}_cluster_precision": np.nan,
        f"{prefix}_cluster_recall": np.nan,
        f"{prefix}_cluster_f1": np.nan,
        f"{prefix}_cluster_specificity": np.nan,
        f"{prefix}_cluster_roc_auc": np.nan,
        f"{prefix}_cluster_pr_auc": np.nan,
        f"{prefix}_cluster_mcc": np.nan,
        f"{prefix}_cluster_density": np.nan,
    }



def cluster_data_to_summary(
    cluster_id,
    cluster_data: dict,
    prefix: str,
) -> dict:
    """Convert one selected cluster/set to the common summary schema.

    Numeric connected-component IDs remain integers. Composite selections such as
    ``singleton_union`` retain a descriptive string ID.
    """
    if cluster_id is None:
        serializable_id = None
    elif isinstance(cluster_id, (int, np.integer)):
        serializable_id = int(cluster_id)
    else:
        serializable_id = str(cluster_id)

    return {
        f"{prefix}_cluster_id": serializable_id,
        f"{prefix}_cluster_size": int(cluster_data["size"]),
        f"{prefix}_cluster_tp": int(cluster_data["tp"]),
        f"{prefix}_cluster_fp": int(cluster_data["fp"]),
        f"{prefix}_cluster_fn": int(cluster_data["fn"]),
        f"{prefix}_cluster_tn": int(cluster_data["tn"]),
        f"{prefix}_cluster_precision": float(cluster_data["precision"]),
        f"{prefix}_cluster_recall": float(cluster_data["recall"]),
        f"{prefix}_cluster_f1": float(cluster_data["f1"]),
        f"{prefix}_cluster_specificity": float(cluster_data["specificity"]),
        f"{prefix}_cluster_roc_auc": float(cluster_data["roc_auc"]),
        f"{prefix}_cluster_pr_auc": float(cluster_data["pr_auc"]),
        f"{prefix}_cluster_mcc": float(cluster_data["mcc"]),
        f"{prefix}_cluster_density": float(cluster_data["cluster_density"]),
    }



def compute_singleton_average_distance_to_others(
    singleton_temp: str,
    valid_temperatures: list,
    similarity_df: pd.DataFrame,
) -> float:
    """
    Computes how far one singleton candidate is from the rest of the valid
    generated candidates for the same issue.

    Distance is defined as 1 - similarity. The score is averaged over all
    other valid temperatures. The singleton-furthest strategy selects the
    singleton with the largest value of this score.
    """
    other_temperatures = [
        temp for temp in valid_temperatures
        if temp != singleton_temp
    ]

    if not other_temperatures:
        return 0.0

    distances = []

    for other_temp in other_temperatures:
        similarity = float(similarity_df.loc[singleton_temp, other_temp])
        distances.append(1.0 - similarity)

    return float(safe_divide(sum(distances), len(distances)))


def compute_cluster_metrics(
    labels: dict,
    temperature_to_pass: dict,
    similarity_df: pd.DataFrame,
) -> tuple:
    """Compute metrics and apply the revised cluster-selection design.

    Selection rules:
      * largest: largest non-singleton; equal size -> denser cluster.
      * smallest: exists only with >=2 non-singletons and cannot equal largest.
        Remove the selected largest cluster, then choose the smallest remaining
        non-singleton; equal size -> denser cluster.
      * singleton_furthest: exists with >=1 singleton.
      * singleton_union: exists only with >=2 singletons.

    Oracle rule:
      * The oracle is always defined for every issue/threshold with at least one
        valid temperature.
      * The oracle is the highest-F1 ACTUAL connected-component cluster using test
        outcomes. Composite strategy selections such as singleton_union are not
        oracle-cluster candidates because they are not one formed cluster.
      * Consequently singleton_union can legitimately have F1 above oracle-cluster
        F1; the oracle remains an upper bound for strategies that select one actual
        formed cluster (largest, smallest, singleton_furthest).
      * If an issue has zero passing candidates, every formed cluster has F1=0; the
        oracle therefore reports a real 0 rather than NaN.
    """
    valid_temperatures = list(labels.keys())
    num_total_patches = len(valid_temperatures)
    if num_total_patches == 0:
        raise ValueError("compute_cluster_metrics requires at least one valid temperature")

    cluster_ids = sorted(
        cluster_id for cluster_id in set(labels.values())
        if cluster_id != -1
    )
    cluster_to_temps = {
        cluster_id: [
            temp for temp in valid_temperatures
            if labels[temp] == cluster_id
        ]
        for cluster_id in cluster_ids
    }

    total_passes = sum(temperature_to_pass[temp] for temp in valid_temperatures)
    total_fails = num_total_patches - total_passes

    clusters = {}
    formed_cluster_records = []
    singleton_records = []
    non_singleton_records = []

    densest_cluster_id = None
    densest_cluster_data = None

    for cluster_id in cluster_ids:
        cluster_temps = cluster_to_temps[cluster_id]
        cluster_data = compute_metrics_for_cluster(
            cluster_temps=cluster_temps,
            valid_temperatures=valid_temperatures,
            temperature_to_pass=temperature_to_pass,
            similarity_df=similarity_df,
        )
        clusters[str(cluster_id)] = cluster_data

        record = {
            "selection_id": cluster_id,
            "selection_type": "formed_cluster",
            "temps": cluster_temps,
            "data": cluster_data,
        }
        formed_cluster_records.append(record)
        if cluster_data["size"] == 1:
            singleton_records.append(record)
        else:
            non_singleton_records.append(record)

        if densest_cluster_data is None:
            densest_cluster_id = cluster_id
            densest_cluster_data = cluster_data
        elif cluster_data["cluster_density"] > densest_cluster_data["cluster_density"]:
            densest_cluster_id = cluster_id
            densest_cluster_data = cluster_data

    # Largest non-singleton: size first, density second.
    largest_record = None
    if non_singleton_records:
        largest_record = sorted(
            non_singleton_records,
            key=lambda record: (
                -record["data"]["size"],
                -record["data"]["cluster_density"],
                record["selection_id"],
            ),
        )[0]

    # Smallest exists only when a second non-singleton exists. Remove largest
    # first so equal-size cases select the densest as largest and the next-densest
    # as smallest rather than selecting the same component twice.
    smallest_record = None
    if len(non_singleton_records) >= 2:
        remaining = [
            record for record in non_singleton_records
            if record["selection_id"] != largest_record["selection_id"]
        ]
        smallest_record = sorted(
            remaining,
            key=lambda record: (
                record["data"]["size"],
                -record["data"]["cluster_density"],
                record["selection_id"],
            ),
        )[0]

    # Singleton-furthest: one singleton is sufficient.
    singleton_furthest_record = None
    singleton_furthest_average_distance = np.nan
    if singleton_records:
        ranked_singletons = []
        for record in singleton_records:
            singleton_temp = record["temps"][0]
            average_distance = compute_singleton_average_distance_to_others(
                singleton_temp=singleton_temp,
                valid_temperatures=valid_temperatures,
                similarity_df=similarity_df,
            )
            ranked_singletons.append(
                (average_distance, float(singleton_temp), record)
            )
        ranked_singletons.sort(key=lambda item: (-item[0], item[1]))
        singleton_furthest_average_distance, _, singleton_furthest_record = (
            ranked_singletons[0]
        )

    singleton_union_temps = [
        record["temps"][0]
        for record in singleton_records
    ]
    singleton_union_record = None
    if len(singleton_union_temps) >= 2:
        singleton_union_data = compute_metrics_for_cluster(
            cluster_temps=singleton_union_temps,
            valid_temperatures=valid_temperatures,
            temperature_to_pass=temperature_to_pass,
            similarity_df=similarity_df,
        )
        singleton_union_record = {
            "selection_id": "singleton_union",
            "selection_type": "singleton_union",
            "temps": singleton_union_temps,
            "data": singleton_union_data,
        }

    # Oracle cluster = best ACTUAL connected component only. singleton_union is a
    # constructed strategy selection and is intentionally excluded from the oracle
    # candidate set. This matches the paper interpretation of "oracle cluster".
    oracle_candidates = list(formed_cluster_records)

    def oracle_rank_key(record: dict) -> tuple:
        data = record["data"]
        return (
            -float(data["f1"]),
            -float(data["precision"]),
            -float(data["recall"]),
            -float(data["cluster_density"]),
            -int(data["size"]),
            str(record["selection_id"]),
        )

    oracle_record = sorted(oracle_candidates, key=oracle_rank_key)[0]
    oracle_summary = cluster_data_to_summary(
        cluster_id=oracle_record["selection_id"],
        cluster_data=oracle_record["data"],
        prefix="oracle",
    )
    oracle_summary["oracle_selection_type"] = oracle_record["selection_type"]

    # Keep the separate ROC-AUC oracle over the same formed-cluster-only universe
    # used by the F1 oracle.
    oracle_roc_auc_record = sorted(
        oracle_candidates,
        key=lambda record: (
            -float(record["data"]["roc_auc"]),
            -float(record["data"]["f1"]),
            -float(record["data"]["precision"]),
            -float(record["data"]["recall"]),
            str(record["selection_id"]),
        ),
    )[0]
    oracle_roc_auc_summary = cluster_data_to_summary(
        cluster_id=oracle_roc_auc_record["selection_id"],
        cluster_data=oracle_roc_auc_record["data"],
        prefix="oracle_roc_auc",
    )
    oracle_roc_auc_summary["oracle_roc_auc_selection_type"] = (
        oracle_roc_auc_record["selection_type"]
    )

    if largest_record is None:
        largest_summary = empty_cluster_summary(
            total_passes=total_passes,
            total_fails=total_fails,
            prefix="largest",
        )
    else:
        largest_summary = cluster_data_to_summary(
            cluster_id=largest_record["selection_id"],
            cluster_data=largest_record["data"],
            prefix="largest",
        )

    if smallest_record is None:
        smallest_summary = empty_cluster_summary(
            total_passes=total_passes,
            total_fails=total_fails,
            prefix="smallest",
        )
    else:
        smallest_summary = cluster_data_to_summary(
            cluster_id=smallest_record["selection_id"],
            cluster_data=smallest_record["data"],
            prefix="smallest",
        )

    if densest_cluster_data is None:
        densest_summary = empty_cluster_summary(
            total_passes=total_passes,
            total_fails=total_fails,
            prefix="densest",
        )
    else:
        densest_summary = cluster_data_to_summary(
            cluster_id=densest_cluster_id,
            cluster_data=densest_cluster_data,
            prefix="densest",
        )

    if singleton_union_record is None:
        singleton_union_summary = empty_cluster_summary(
            total_passes=total_passes,
            total_fails=total_fails,
            prefix="singleton_union",
        )
    else:
        singleton_union_summary = cluster_data_to_summary(
            cluster_id="singleton_union",
            cluster_data=singleton_union_record["data"],
            prefix="singleton_union",
        )

    if singleton_furthest_record is None:
        singleton_furthest_summary = empty_cluster_summary(
            total_passes=total_passes,
            total_fails=total_fails,
            prefix="singleton_furthest",
        )
        singleton_furthest_summary[
            "singleton_furthest_average_distance_to_others"
        ] = np.nan
    else:
        singleton_furthest_summary = cluster_data_to_summary(
            cluster_id=singleton_furthest_record["selection_id"],
            cluster_data=singleton_furthest_record["data"],
            prefix="singleton_furthest",
        )
        singleton_furthest_summary[
            "singleton_furthest_average_distance_to_others"
        ] = float(singleton_furthest_average_distance)

    # The oracle is an upper bound for strategies that select one actual formed
    # cluster. singleton_union is excluded from this invariant because it combines
    # multiple singleton components and can legitimately outperform every one
    # connected component, as in the motivating examples.
    oracle_f1 = float(oracle_summary["oracle_cluster_f1"])
    formed_cluster_strategy_records = {
        "largest": largest_record,
        "smallest": smallest_record,
        "singleton_furthest": singleton_furthest_record,
    }
    for strategy_name, record in formed_cluster_strategy_records.items():
        if record is None:
            continue
        strategy_f1 = float(record["data"]["f1"])
        if strategy_f1 > oracle_f1 + 1e-12:
            raise AssertionError(
                f"Oracle formed-cluster F1 invariant failed for {strategy_name}: "
                f"strategy={strategy_f1}, oracle={oracle_f1}"
            )

    summary = {}
    summary.update(oracle_summary)
    summary.update(oracle_roc_auc_summary)
    summary.update(largest_summary)
    summary.update(smallest_summary)
    summary.update(densest_summary)
    summary.update(singleton_union_summary)
    summary.update(singleton_furthest_summary)
    return clusters, summary



def select_oracle_threshold_per_instance(
    threshold_summary_df: pd.DataFrame,
    strategy_prefix: str,
) -> pd.DataFrame:
    if threshold_summary_df.empty:
        return pd.DataFrame()

    f1_col = f"{strategy_prefix}_cluster_f1"
    precision_col = f"{strategy_prefix}_cluster_precision"
    recall_col = f"{strategy_prefix}_cluster_recall"
    if f1_col not in threshold_summary_df.columns:
        return pd.DataFrame()

    available_df = threshold_summary_df[threshold_summary_df[f1_col].notna()].copy()
    if available_df.empty:
        return pd.DataFrame()

    sorted_df = available_df.sort_values(
        by=["instance_id", f1_col, precision_col, recall_col, "threshold"],
        ascending=[True, False, False, False, True],
    )
    per_instance_df = (
        sorted_df
        .groupby("instance_id", group_keys=False)
        .head(1)
        .copy()
    )
    per_instance_df = per_instance_df.rename(
        columns={"threshold": "oracle_optimal_threshold"}
    )

    keep_columns = [
        "subset", "instance_id", "oracle_optimal_threshold",
        "num_total_patches", "num_passes_for_issue", "num_fails_for_issue",
        "num_clusters", "num_outliers",
        f"{strategy_prefix}_cluster_id", f"{strategy_prefix}_cluster_size",
        f"{strategy_prefix}_cluster_tp", f"{strategy_prefix}_cluster_fp",
        f"{strategy_prefix}_cluster_fn", f"{strategy_prefix}_cluster_tn",
        f"{strategy_prefix}_cluster_precision", f"{strategy_prefix}_cluster_recall",
        f"{strategy_prefix}_cluster_f1", f"{strategy_prefix}_cluster_specificity",
        f"{strategy_prefix}_cluster_roc_auc", f"{strategy_prefix}_cluster_pr_auc",
        f"{strategy_prefix}_cluster_mcc", f"{strategy_prefix}_cluster_density",
    ]
    keep_columns = [col for col in keep_columns if col in per_instance_df.columns]
    return per_instance_df[keep_columns]



def compute_threshold_average_metrics(
    threshold_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each threshold with explicit, non-mixed denominators.

    Oracle F1 is averaged over every clustered issue. All-failing issues retain a real oracle F1 of zero.

    Non-oracle strategy metrics are conditional means: NaN rows, where the
    strategy is structurally unavailable, are skipped. Availability counts/rates
    are reported separately. For valid comparisons, a matched oracle mean is also
    computed on exactly the issue rows where each strategy is available.
    """
    if threshold_summary_df.empty:
        return pd.DataFrame()

    rows = []
    work_df = threshold_summary_df.copy()
    work_df["threshold"] = work_df["threshold"].astype(float)

    for threshold in sorted(work_df["threshold"].unique()):
        threshold_df = work_df[
            np.isclose(work_df["threshold"], float(threshold))
        ].copy()
        num_instances = int(threshold_df["instance_id"].nunique())

        row = {
            "threshold": float(threshold),
            "num_instances": num_instances,
        }
        if "num_total_patches" in threshold_df.columns:
            row["average_num_total_patches"] = float(
                threshold_df["num_total_patches"].mean()
            )

        if "num_passes_for_issue" in threshold_df.columns:
            zero_pass_mask = threshold_df["num_passes_for_issue"].astype(int) == 0
            row["num_zero_pass_issues"] = int(zero_pass_mask.sum())
            row["zero_pass_issue_frequency"] = safe_divide(
                int(zero_pass_mask.sum()), num_instances
            )
            row["num_issues_with_at_least_one_pass"] = int((~zero_pass_mask).sum())

        oracle_f1_col = "oracle_cluster_f1"
        if oracle_f1_col not in threshold_df.columns:
            raise ValueError("threshold summary is missing oracle_cluster_f1")
        if threshold_df[oracle_f1_col].isna().any():
            missing_ids = threshold_df.loc[
                threshold_df[oracle_f1_col].isna(), "instance_id"
            ].astype(str).tolist()
            raise AssertionError(
                "Oracle F1 must be defined for every clustered issue; missing for "
                f"{missing_ids[:10]}"
            )

        for strategy in CLUSTER_SELECTION_STRATEGIES:
            for metric_name in [
                "size", "precision", "recall", "f1", "specificity",
                "roc_auc", "pr_auc", "mcc", "density",
            ]:
                source_col = f"{strategy}_cluster_{metric_name}"
                if source_col in threshold_df.columns:
                    row[f"average_{strategy}_{metric_name}"] = float(
                        threshold_df[source_col].mean()
                    )

            f1_source_col = f"{strategy}_cluster_f1"
            if f1_source_col not in threshold_df.columns:
                continue

            available_mask = threshold_df[f1_source_col].notna()
            available_count = int(available_mask.sum())
            row[f"num_available_{strategy}"] = available_count
            row[f"availability_{strategy}"] = safe_divide(
                available_count, num_instances
            )

            if strategy in ORACLE_FREE_CLUSTER_SELECTION_STRATEGIES:
                matched_oracle_col = (
                    f"average_oracle_f1_on_{strategy}_available_issues"
                )
                gap_col = f"average_oracle_minus_{strategy}_f1"
                if available_count == 0:
                    row[matched_oracle_col] = np.nan
                    row[gap_col] = np.nan
                else:
                    strategy_values = threshold_df.loc[
                        available_mask, f1_source_col
                    ].astype(float)
                    oracle_values = threshold_df.loc[
                        available_mask, oracle_f1_col
                    ].astype(float)
                    # For selections that are themselves one formed component,
                    # oracle-cluster F1 must upper-bound the strategy issue by issue.
                    # singleton_union is composite and may legitimately exceed the
                    # best individual formed cluster.
                    if strategy != "singleton_union":
                        violations = strategy_values - oracle_values
                        if (violations > 1e-12).any():
                            bad_ids = threshold_df.loc[
                                available_mask
                                & (
                                    threshold_df[f1_source_col]
                                    > threshold_df[oracle_f1_col] + 1e-12
                                ),
                                "instance_id",
                            ].astype(str).tolist()
                            raise AssertionError(
                                f"Oracle formed cluster is below {strategy} for issue(s) "
                                f"{bad_ids[:10]} at threshold {threshold}"
                            )
                    matched_oracle = float(oracle_values.mean())
                    conditional_strategy = float(strategy_values.mean())
                    row[matched_oracle_col] = matched_oracle
                    row[gap_col] = matched_oracle - conditional_strategy

        if "singleton_furthest_average_distance_to_others" in threshold_df.columns:
            row["average_singleton_furthest_distance_to_others"] = float(
                threshold_df[
                    "singleton_furthest_average_distance_to_others"
                ].mean()
            )

        # Oracle must be available on every issue, including all-failing issues.
        if row.get("num_available_oracle", 0) != num_instances:
            raise AssertionError(
                f"Oracle availability must be 100% at threshold {threshold}: "
                f"{row.get('num_available_oracle', 0)}/{num_instances}"
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)



def save_rq2_average_f1_by_threshold(
    threshold_average_df: pd.DataFrame,
    subset_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """Save conditional mean F1, skipping unavailable issue-strategy rows."""
    output_columns = [
        "model",
        "dataset",
        "subset",
        "similarity_metric",
        "threshold",
        "average_oracle_f1",
        "average_largest_f1",
        "average_smallest_f1",
        "average_singleton_union_f1",
        "average_singleton_furthest_f1",
    ]

    if threshold_average_df.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(output_path, index=False)
        return empty_df

    rows = []
    df = threshold_average_df.copy()
    df["threshold"] = df["threshold"].astype(float)

    for threshold in SIMILARITY_THRESHOLDS:
        matching = df[np.isclose(df["threshold"], float(threshold))]
        row = matching.iloc[0] if not matching.empty else pd.Series(dtype=float)

        rows.append({
            "model": MODEL,
            "dataset": DATASET_NAME,
            "subset": subset_name,
            "similarity_metric": metric,
            "threshold": float(threshold),
            "average_oracle_f1": float(row.get("average_oracle_f1", np.nan)),
            "average_largest_f1": float(row.get("average_largest_f1", np.nan)),
            "average_smallest_f1": float(row.get("average_smallest_f1", np.nan)),
            "average_singleton_union_f1": float(
                row.get("average_singleton_union_f1", np.nan)
            ),
            "average_singleton_furthest_f1": float(
                row.get("average_singleton_furthest_f1", np.nan)
            ),
        })

    output_df = pd.DataFrame(rows, columns=output_columns)
    output_df.to_csv(output_path, index=False)
    return output_df


def save_rq2_average_cluster_size_by_threshold(
    threshold_average_df: pd.DataFrame,
    subset_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """Save average selected cluster/set size for every strategy and threshold.

    For singleton_union, size is the number of singleton candidates in the selected
    union. For the other strategies, size is the size of the selected cluster.
    Missing strategies are NaN and are excluded from this conditional mean.
    """
    output_columns = [
        "model",
        "dataset",
        "subset",
        "similarity_metric",
        "threshold",
        "average_oracle_size",
        "average_largest_size",
        "average_smallest_size",
        "average_singleton_union_size",
        "average_singleton_furthest_size",
    ]

    if threshold_average_df.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(output_path, index=False)
        return empty_df

    rows = []
    df = threshold_average_df.copy()
    df["threshold"] = df["threshold"].astype(float)

    for threshold in SIMILARITY_THRESHOLDS:
        matching = df[np.isclose(df["threshold"], float(threshold))]
        row = matching.iloc[0] if not matching.empty else pd.Series(dtype=float)

        rows.append({
            "model": MODEL,
            "dataset": DATASET_NAME,
            "subset": subset_name,
            "similarity_metric": metric,
            "threshold": float(threshold),
            "average_oracle_size": float(row.get("average_oracle_size", np.nan)),
            "average_largest_size": float(row.get("average_largest_size", np.nan)),
            "average_smallest_size": float(row.get("average_smallest_size", np.nan)),
            "average_singleton_union_size": float(
                row.get("average_singleton_union_size", np.nan)
            ),
            "average_singleton_furthest_size": float(
                row.get("average_singleton_furthest_size", np.nan)
            ),
        })

    output_df = pd.DataFrame(rows, columns=output_columns)
    output_df.to_csv(output_path, index=False)
    return output_df



def save_best_configuration(
    threshold_average_df: pd.DataFrame,
    subset_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """Save the highest conditional-F1 oracle-free strategy/threshold.

    ``optimal F1`` is the strategy's mean only over issue rows where that strategy
    exists. ``Oracle cluster F1`` uses the exact same issue rows and represents the
    best ACTUAL formed cluster. It upper-bounds largest/smallest/singleton_furthest,
    but singleton_union may exceed it because singleton_union is a composite set.
    The separate ``overall Oracle F1 at optimal T`` uses every issue, including
    zero-pass issues whose oracle F1 is zero.

    Availability is coverage, not performance:
      * strategy available issue count: number of issues where the winning
        strategy exists at the winning threshold.
      * strategy availability rate: that count divided by all clustered issues at
        the threshold.
    """
    output_columns = [
        "model",
        "dataset",
        "subset",
        "similarity_metric",
        "optimal cluster",
        "average size",
        "optimal T",
        "optimal F1",
        "optimal ROC-AUC",
        "strategy available issue count",
        "total issue count at optimal T",
        "strategy availability rate",
        "Oracle cluster F1",
        "oracle comparison basis",
        "overall Oracle F1 at optimal T",
        "zero-pass issue count at optimal T",
        "zero-pass issue rate at optimal T",
        "highest Oracle F1",
        "T at highest Oracle F1",
        "average_oracle_cluster_size_at_highest_oracle_free_threshold",
        "average_oracle_cluster_size_at_highest_oracle_f1",
    ]

    if threshold_average_df.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(output_path, index=False)
        return empty_df

    candidates = []
    for strategy in ORACLE_FREE_CLUSTER_SELECTION_STRATEGIES:
        f1_col = f"average_{strategy}_f1"
        roc_auc_col = f"average_{strategy}_roc_auc"
        size_col = f"average_{strategy}_size"
        matched_oracle_col = (
            f"average_oracle_f1_on_{strategy}_available_issues"
        )

        if f1_col not in threshold_average_df.columns:
            continue

        for _, row in threshold_average_df.iterrows():
            f1_value = row[f1_col]
            if pd.isna(f1_value):
                continue

            matched_oracle_f1 = row.get(matched_oracle_col, np.nan)
            if pd.isna(matched_oracle_f1):
                raise AssertionError(
                    f"Matched oracle F1 is missing for available strategy {strategy} "
                    f"at threshold {row['threshold']}"
                )
            if (
                strategy != "singleton_union"
                and float(matched_oracle_f1) + 1e-12 < float(f1_value)
            ):
                raise AssertionError(
                    f"Matched formed-cluster oracle F1 {matched_oracle_f1} is below "
                    f"{strategy} F1 {f1_value} at threshold {row['threshold']}"
                )

            candidates.append({
                "strategy": strategy,
                "threshold": float(row["threshold"]),
                "average_f1": float(f1_value),
                "average_roc_auc": (
                    float(row[roc_auc_col])
                    if roc_auc_col in threshold_average_df.columns
                    and not pd.isna(row[roc_auc_col])
                    else np.nan
                ),
                "average_cluster_size": (
                    float(row[size_col])
                    if size_col in threshold_average_df.columns
                    and not pd.isna(row[size_col])
                    else np.nan
                ),
                "num_available_issues": int(
                    row.get(f"num_available_{strategy}", 0)
                ),
                "num_instances": int(row.get("num_instances", 0)),
                "availability_rate": float(
                    row.get(f"availability_{strategy}", np.nan)
                ),
                "matched_oracle_f1": float(matched_oracle_f1),
                "overall_oracle_f1": float(row["average_oracle_f1"]),
                "average_oracle_size": (
                    float(row["average_oracle_size"])
                    if "average_oracle_size" in threshold_average_df.columns
                    and not pd.isna(row["average_oracle_size"])
                    else np.nan
                ),
                "num_zero_pass_issues": int(row.get("num_zero_pass_issues", 0)),
                "zero_pass_issue_frequency": float(
                    row.get("zero_pass_issue_frequency", np.nan)
                ),
            })

    if not candidates:
        result_df = pd.DataFrame(columns=output_columns)
        result_df.to_csv(output_path, index=False)
        return result_df

    strategy_order = {
        strategy: index
        for index, strategy in enumerate(ORACLE_FREE_CLUSTER_SELECTION_STRATEGIES)
    }
    winner = sorted(
        candidates,
        key=lambda item: (
            -item["average_f1"],
            -(
                item["average_roc_auc"]
                if not pd.isna(item["average_roc_auc"])
                else -np.inf
            ),
            item["threshold"],
            strategy_order[item["strategy"]],
        ),
    )[0]

    oracle_rows = []
    if "average_oracle_f1" in threshold_average_df.columns:
        for _, row in threshold_average_df.iterrows():
            oracle_f1 = row["average_oracle_f1"]
            if pd.isna(oracle_f1):
                continue
            oracle_rows.append({
                "threshold": float(row["threshold"]),
                "average_oracle_f1": float(oracle_f1),
                "average_oracle_size": (
                    float(row["average_oracle_size"])
                    if "average_oracle_size" in threshold_average_df.columns
                    and not pd.isna(row["average_oracle_size"])
                    else np.nan
                ),
            })

    if oracle_rows:
        oracle_winner = sorted(
            oracle_rows,
            key=lambda item: (-item["average_oracle_f1"], item["threshold"]),
        )[0]
    else:
        oracle_winner = {
            "threshold": np.nan,
            "average_oracle_f1": np.nan,
            "average_oracle_size": np.nan,
        }

    result_df = pd.DataFrame(
        [{
            "model": MODEL,
            "dataset": DATASET_NAME,
            "subset": subset_name,
            "similarity_metric": metric,
            "optimal cluster": winner["strategy"],
            "average size": winner["average_cluster_size"],
            "optimal T": winner["threshold"],
            "optimal F1": winner["average_f1"],
            "optimal ROC-AUC": winner["average_roc_auc"],
            "strategy available issue count": winner["num_available_issues"],
            "total issue count at optimal T": winner["num_instances"],
            "strategy availability rate": winner["availability_rate"],
            # Backward-compatible label, now correctly matched to the exact same
            # issue rows used by the winning strategy's conditional mean.
            "Oracle cluster F1": winner["matched_oracle_f1"],
            "oracle comparison basis": (
                "same issues where selected strategy exists"
            ),
            "overall Oracle F1 at optimal T": winner["overall_oracle_f1"],
            "zero-pass issue count at optimal T": winner["num_zero_pass_issues"],
            "zero-pass issue rate at optimal T": winner[
                "zero_pass_issue_frequency"
            ],
            "highest Oracle F1": oracle_winner["average_oracle_f1"],
            "T at highest Oracle F1": oracle_winner["threshold"],
            "average_oracle_cluster_size_at_highest_oracle_free_threshold": winner[
                "average_oracle_size"
            ],
            "average_oracle_cluster_size_at_highest_oracle_f1": oracle_winner[
                "average_oracle_size"
            ],
        }],
        columns=output_columns,
    )
    result_df.to_csv(output_path, index=False)
    return result_df




def build_and_save_issue_f1_matrix(
    threshold_summary_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Save one row per issue with 45 strategy-by-threshold F1 values.

    Columns are exactly:
      instance_id
      9 oracle-cluster F1 values
      9 largest-cluster F1 values
      9 smallest-cluster F1 values
      9 singleton-union F1 values
      9 singleton-furthest F1 values

    This 46-column file is intended for later paired statistical analyses across
    similarity metrics for the same model.
    """
    strategy_to_source_column = {
        "oracle": "oracle_cluster_f1",
        "largest": "largest_cluster_f1",
        "smallest": "smallest_cluster_f1",
        "singleton_union": "singleton_union_cluster_f1",
        "singleton_furthest": "singleton_furthest_cluster_f1",
    }

    output_columns = ["instance_id"]
    for strategy in strategy_to_source_column:
        output_columns.extend(
            f"{strategy}_f1_threshold_{threshold:.1f}"
            for threshold in SIMILARITY_THRESHOLDS
        )

    if threshold_summary_df.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(output_path, index=False)
        return empty_df

    rows = []
    for instance_id, issue_df in threshold_summary_df.groupby("instance_id"):
        issue_df = issue_df.copy()
        issue_df["threshold"] = issue_df["threshold"].astype(float)
        row = {"instance_id": str(instance_id)}

        for strategy, source_col in strategy_to_source_column.items():
            for threshold in SIMILARITY_THRESHOLDS:
                output_col = f"{strategy}_f1_threshold_{threshold:.1f}"
                matching = issue_df[
                    np.isclose(issue_df["threshold"], float(threshold))
                ]

                if matching.empty or source_col not in matching.columns:
                    row[output_col] = np.nan
                else:
                    row[output_col] = float(matching.iloc[0][source_col])

        rows.append(row)

    output_df = (
        pd.DataFrame(rows, columns=output_columns)
        .sort_values("instance_id")
        .reset_index(drop=True)
    )
    output_df.to_csv(output_path, index=False)
    return output_df



def save_issue_f1_tables_by_threshold(
    threshold_summary_df: pd.DataFrame,
    subset_name: str,
    output_dir: Path,
):
    """Save one issue-level F1 dataframe for each fixed threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_to_source = {
        "oracle": "oracle_cluster_f1",
        "largest": "largest_cluster_f1",
        "smallest": "smallest_cluster_f1",
        "singleton_union": "singleton_union_cluster_f1",
        "singleton_furthest": "singleton_furthest_cluster_f1",
    }
    base_columns = [
        "subset", "instance_id", "threshold", "num_total_patches",
        "num_passes_for_issue", "num_fails_for_issue", "num_clusters",
        "non_singleton_cluster_count", "singleton_cluster_count",
        "cluster_size_pattern", "fragmentation_G", "oracle_selection_type",
    ]
    f1_columns = [f"{strategy}_f1" for strategy in strategy_to_source]
    availability_columns = [f"{strategy}_available" for strategy in strategy_to_source]
    output_columns = base_columns + f1_columns + availability_columns

    for threshold in SIMILARITY_THRESHOLDS:
        if threshold_summary_df.empty:
            output_df = pd.DataFrame(columns=output_columns)
        else:
            threshold_df = threshold_summary_df[
                np.isclose(
                    threshold_summary_df["threshold"].astype(float),
                    float(threshold),
                )
            ].copy()
            output_df = pd.DataFrame(index=threshold_df.index)
            for col in base_columns:
                if col == "subset":
                    output_df[col] = subset_name
                elif col in threshold_df.columns:
                    output_df[col] = threshold_df[col]
                else:
                    output_df[col] = np.nan

            for strategy, source_col in strategy_to_source.items():
                f1_col = f"{strategy}_f1"
                output_df[f1_col] = (
                    threshold_df[source_col]
                    if source_col in threshold_df.columns
                    else np.nan
                )
                output_df[f"{strategy}_available"] = output_df[f1_col].notna()

            output_df = output_df[output_columns].sort_values("instance_id")

        output_df.to_csv(
            output_dir / f"threshold_{threshold:.1f}.csv",
            index=False,
        )



def save_cross_threshold_issue_tables_from_threshold_csvs(
    issue_f1_by_threshold_dir: Path,
) -> dict:
    """Build compact one-row-per-issue tables from threshold_0.1.csv ... 0.9.csv.

    Each output contains one ``instance_id`` column plus nine threshold columns
    named ``0.1`` ... ``0.9``:

      1. cluster_size_pattern_by_threshold.csv
         Connected-component size pattern at every threshold.

      2. fragmentation_G_by_threshold.csv
         Issue-level fragmentation G_i(t) at every threshold.

      3. oracle_f1_by_threshold.csv
         F1 of the best ACTUAL connected-component cluster at every threshold.

      4. winning_strategy_multicluster_by_threshold.csv
         Only defined when clustering produced at least two connected components.
         The maximum-F1 available strategy among largest, smallest,
         singleton_furthest, and singleton_union is recorded. Exact F1 ties are
         preserved as ``tie:strategy_a+strategy_b`` instead of being broken
         arbitrarily.

      5. winning_strategy_f1_multicluster_by_threshold.csv
         The corresponding maximum strategy F1 for the same multicluster rows.

    A one-component pattern such as ``11`` is intentionally treated as NOT having
    performed a separation, so the two winner tables contain NA for that threshold.
    Fragmentation remains defined and equals zero for a one-component partition.
    """
    issue_f1_by_threshold_dir = Path(issue_f1_by_threshold_dir)
    threshold_frames = {}
    all_instance_ids = set()

    required_columns = {
        "instance_id",
        "num_clusters",
        "cluster_size_pattern",
        "fragmentation_G",
        "oracle_f1",
        "largest_f1",
        "smallest_f1",
        "singleton_union_f1",
        "singleton_furthest_f1",
    }

    for threshold in SIMILARITY_THRESHOLDS:
        threshold_key = f"{threshold:.1f}"
        threshold_path = issue_f1_by_threshold_dir / f"threshold_{threshold_key}.csv"
        if not threshold_path.exists():
            raise FileNotFoundError(
                f"Cannot build cross-threshold issue tables; missing {threshold_path}"
            )

        threshold_df = pd.read_csv(threshold_path)
        missing_columns = required_columns - set(threshold_df.columns)
        if missing_columns:
            raise ValueError(
                f"{threshold_path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        threshold_df = threshold_df.copy()
        threshold_df["instance_id"] = threshold_df["instance_id"].astype(str)
        threshold_frames[threshold_key] = threshold_df.set_index("instance_id")
        all_instance_ids.update(threshold_df["instance_id"].tolist())

    instance_ids = sorted(all_instance_ids)
    threshold_columns = [f"{threshold:.1f}" for threshold in SIMILARITY_THRESHOLDS]

    cluster_pattern_rows = []
    fragmentation_rows = []
    oracle_f1_rows = []
    winner_strategy_rows = []
    winner_f1_rows = []

    strategy_f1_columns = {
        "largest": "largest_f1",
        "smallest": "smallest_f1",
        "singleton_union": "singleton_union_f1",
        "singleton_furthest": "singleton_furthest_f1",
    }

    for instance_id in instance_ids:
        cluster_pattern_row = {"instance_id": instance_id}
        fragmentation_row = {"instance_id": instance_id}
        oracle_f1_row = {"instance_id": instance_id}
        winner_strategy_row = {"instance_id": instance_id}
        winner_f1_row = {"instance_id": instance_id}

        for threshold_key in threshold_columns:
            threshold_df = threshold_frames[threshold_key]
            if instance_id not in threshold_df.index:
                cluster_pattern_row[threshold_key] = np.nan
                fragmentation_row[threshold_key] = np.nan
                oracle_f1_row[threshold_key] = np.nan
                winner_strategy_row[threshold_key] = np.nan
                winner_f1_row[threshold_key] = np.nan
                continue

            source_row = threshold_df.loc[instance_id]
            if isinstance(source_row, pd.DataFrame):
                if len(source_row) != 1:
                    raise ValueError(
                        f"Duplicate rows for instance_id={instance_id} at threshold "
                        f"{threshold_key}"
                    )
                source_row = source_row.iloc[0]

            cluster_pattern_row[threshold_key] = source_row["cluster_size_pattern"]
            fragmentation_row[threshold_key] = float(source_row["fragmentation_G"])
            oracle_f1_row[threshold_key] = float(source_row["oracle_f1"])

            num_clusters = int(source_row["num_clusters"])
            if num_clusters < 2:
                winner_strategy_row[threshold_key] = np.nan
                winner_f1_row[threshold_key] = np.nan
                continue

            available_scores = {}
            for strategy, f1_column in strategy_f1_columns.items():
                value = source_row[f1_column]
                if pd.isna(value):
                    continue
                available_scores[strategy] = float(value)

            if not available_scores:
                winner_strategy_row[threshold_key] = np.nan
                winner_f1_row[threshold_key] = np.nan
                continue

            max_f1 = max(available_scores.values())
            winners = [
                strategy
                for strategy, value in available_scores.items()
                if np.isclose(value, max_f1, rtol=0.0, atol=1e-12)
            ]
            if len(winners) == 1:
                winner_label = winners[0]
            else:
                winner_label = "tie:" + "+".join(winners)

            winner_strategy_row[threshold_key] = winner_label
            winner_f1_row[threshold_key] = float(max_f1)

        cluster_pattern_rows.append(cluster_pattern_row)
        fragmentation_rows.append(fragmentation_row)
        oracle_f1_rows.append(oracle_f1_row)
        winner_strategy_rows.append(winner_strategy_row)
        winner_f1_rows.append(winner_f1_row)

    output_columns = ["instance_id"] + threshold_columns
    outputs = {
        "cluster_size_pattern": pd.DataFrame(
            cluster_pattern_rows, columns=output_columns
        ),
        "fragmentation_G": pd.DataFrame(
            fragmentation_rows, columns=output_columns
        ),
        "oracle_f1": pd.DataFrame(
            oracle_f1_rows, columns=output_columns
        ),
        "winning_strategy_multicluster": pd.DataFrame(
            winner_strategy_rows, columns=output_columns
        ),
        "winning_strategy_f1_multicluster": pd.DataFrame(
            winner_f1_rows, columns=output_columns
        ),
    }

    output_paths = {
        "cluster_size_pattern": (
            issue_f1_by_threshold_dir / "cluster_size_pattern_by_threshold.csv"
        ),
        "fragmentation_G": (
            issue_f1_by_threshold_dir / "fragmentation_G_by_threshold.csv"
        ),
        "oracle_f1": issue_f1_by_threshold_dir / "oracle_f1_by_threshold.csv",
        "winning_strategy_multicluster": (
            issue_f1_by_threshold_dir
            / "winning_strategy_multicluster_by_threshold.csv"
        ),
        "winning_strategy_f1_multicluster": (
            issue_f1_by_threshold_dir
            / "winning_strategy_f1_multicluster_by_threshold.csv"
        ),
    }

    for key, output_df in outputs.items():
        output_df.to_csv(output_paths[key], index=False, na_rep="NA")

    return output_paths


def save_rq2_prioritization_utility_outputs(
    threshold_summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Save issue-level U plus conditional and coverage-adjusted RQ2 summaries.

    This analysis is intended for the solvable subset. For issue i, threshold t,
    and strategy s:

        H_i,s(t) = 1 if the selected group contains at least one passing fix,
                   0 otherwise,

        R_i,s(t) = 1 - k_i,s(t) / n_i,

        U_i,s(t) = H_i,s(t) * R_i,s(t),

    where k_i,s(t) is the number of candidates selected by the strategy and n_i
    is the total number of valid candidates for the issue. U is therefore in
    [0, 1].

    The existing strategy definitions are reused unchanged:
      * largest: largest non-singleton component;
      * smallest: smallest non-singleton after excluding the selected largest;
      * singleton_union: union of all singleton components when available;
      * singleton_furthest: singleton with greatest average distance to the others.

    Issue-level U is reported only when clustering produced at least two connected
    components and the strategy is structurally available. Otherwise U is saved
    as NA.

    Two threshold-level summaries are produced for every strategy:

      1. Conditional mean utility:

            U_cond_s(t) = sum_i I_i,s(t) U_i,s(t) / sum_i I_i,s(t)

         where I_i,s(t)=1 when the strategy is available and 0 otherwise.

      2. Coverage-adjusted mean utility:

            U_overall_s(t) = (1 / N_t) * sum_i I_i,s(t) U_i,s(t)
                           = availability_s(t) * U_cond_s(t).

         N_t is the total number of solvable issues represented at threshold t.
         Thus structurally unavailable cases contribute zero to the overall
         effectiveness score without being rewritten as zeros in the raw
         issue-level CSVs.

    Threshold-level averages are reported only when the strategy is available for
    more than three issues. Otherwise both conditional and coverage-adjusted
    values are left as NA for that strategy-threshold point.

    Plot and summary values are multiplied by 100, so they are shown on a 0--100
    scale. The four raw issue-level U CSVs remain on the theoretical 0--1 scale.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove only artifacts created by previous custom RQ2 analyses so rerunning
    # into the same directory cannot leave stale custom outputs. Existing pipeline
    # outputs outside this added RQ2 analysis are untouched.
    legacy_custom_paths = [
        output_dir / "no_prioritization_F1.csv",
        output_dir / "average_prioritization_gain_by_threshold.pdf",
        output_dir / "no_prioritization_probability.csv",
        output_dir / "average_correct_fix_enrichment_by_threshold.pdf",
        output_dir / "average_prioritization_utility_U_by_threshold.pdf",
        output_dir / "coverage_adjusted_prioritization_utility_U_by_threshold.pdf",
        output_dir / "prioritization_utility_U_summary_by_threshold.csv",
    ]
    for strategy in ORACLE_FREE_CLUSTER_SELECTION_STRATEGIES:
        legacy_custom_paths.extend([
            output_dir / f"{strategy}_F1_by_threshold.csv",
            output_dir / f"{strategy}_prioritization_gain_delta.csv",
            output_dir / f"{strategy}_correct_fix_probability_by_threshold.csv",
            output_dir / f"{strategy}_correct_fix_enrichment_delta.csv",
        ])
    for legacy_path in legacy_custom_paths:
        if legacy_path.exists():
            legacy_path.unlink()

    threshold_columns = [
        f"{threshold:.1f}"
        for threshold in SIMILARITY_THRESHOLDS
    ]
    output_columns = ["instance_id"] + threshold_columns

    strategy_to_source_columns = {
        "largest": {
            "size": "largest_cluster_size",
            "tp": "largest_cluster_tp",
        },
        "smallest": {
            "size": "smallest_cluster_size",
            "tp": "smallest_cluster_tp",
        },
        "singleton_union": {
            "size": "singleton_union_cluster_size",
            "tp": "singleton_union_cluster_tp",
        },
        "singleton_furthest": {
            "size": "singleton_furthest_cluster_size",
            "tp": "singleton_furthest_cluster_tp",
        },
    }

    utility_rows = {
        strategy: []
        for strategy in strategy_to_source_columns
    }

    if not threshold_summary_df.empty:
        work_df = threshold_summary_df.copy()
        work_df["threshold"] = work_df["threshold"].astype(float)

        for instance_id, issue_df in work_df.groupby("instance_id"):
            utility_rows_for_issue = {
                strategy: {"instance_id": str(instance_id)}
                for strategy in strategy_to_source_columns
            }

            for threshold in SIMILARITY_THRESHOLDS:
                threshold_key = f"{threshold:.1f}"
                matching = issue_df[
                    np.isclose(
                        issue_df["threshold"].astype(float),
                        float(threshold),
                    )
                ]

                if matching.empty:
                    for strategy in strategy_to_source_columns:
                        utility_rows_for_issue[strategy][threshold_key] = np.nan
                    continue

                if len(matching) != 1:
                    raise ValueError(
                        f"Expected one row for instance_id={instance_id} at "
                        f"threshold={threshold_key}, found {len(matching)}"
                    )

                source_row = matching.iloc[0]
                num_total_patches = int(source_row["num_total_patches"])
                num_clusters = int(source_row["num_clusters"])

                if num_total_patches <= 0:
                    for strategy in strategy_to_source_columns:
                        utility_rows_for_issue[strategy][threshold_key] = np.nan
                    continue

                for strategy, source_columns in strategy_to_source_columns.items():
                    utility_value = np.nan
                    size_column = source_columns["size"]
                    tp_column = source_columns["tp"]

                    # A one-component partition is not a meaningful prioritization.
                    # Structurally unavailable strategies remain NA in the raw CSV.
                    if (
                        num_clusters >= 2
                        and size_column in source_row.index
                        and tp_column in source_row.index
                    ):
                        selected_size = source_row[size_column]
                        selected_tp = source_row[tp_column]

                        if not pd.isna(selected_size) and not pd.isna(selected_tp):
                            selected_size = int(selected_size)
                            selected_tp = int(selected_tp)

                            if selected_size <= 0:
                                raise AssertionError(
                                    f"Available strategy {strategy} selected no "
                                    f"candidates for instance_id={instance_id}, "
                                    f"threshold={threshold_key}"
                                )
                            if selected_size > num_total_patches:
                                raise AssertionError(
                                    f"Strategy {strategy} selected {selected_size} "
                                    f"candidates from only {num_total_patches} valid "
                                    f"candidates for instance_id={instance_id}, "
                                    f"threshold={threshold_key}"
                                )
                            if selected_tp < 0 or selected_tp > selected_size:
                                raise AssertionError(
                                    f"Invalid TP count {selected_tp} for strategy "
                                    f"{strategy}, selected_size={selected_size}, "
                                    f"instance_id={instance_id}, "
                                    f"threshold={threshold_key}"
                                )

                            hit = 1.0 if selected_tp > 0 else 0.0
                            candidate_reduction = (
                                1.0 - (selected_size / num_total_patches)
                            )
                            utility_value = float(hit * candidate_reduction)

                            # Floating-point guard for the theoretical [0, 1] range.
                            utility_value = float(
                                min(1.0, max(0.0, utility_value))
                            )

                    utility_rows_for_issue[strategy][threshold_key] = utility_value

            for strategy in strategy_to_source_columns:
                utility_rows[strategy].append(
                    utility_rows_for_issue[strategy]
                )

    utility_paths = {}
    utility_dfs = {}

    for strategy in strategy_to_source_columns:
        utility_df = pd.DataFrame(
            utility_rows[strategy],
            columns=output_columns,
        )

        if not utility_df.empty:
            utility_df = utility_df.sort_values(
                "instance_id"
            ).reset_index(drop=True)

        utility_path = (
            output_dir / f"{strategy}_prioritization_utility_U_by_threshold.csv"
        )
        utility_df.to_csv(
            utility_path,
            index=False,
            na_rep="NA",
        )

        utility_paths[strategy] = utility_path
        utility_dfs[strategy] = utility_df

    # Build threshold-level conditional and coverage-adjusted summaries.
    summary_rows = []
    for threshold in SIMILARITY_THRESHOLDS:
        threshold_key = f"{threshold:.1f}"

        if threshold_summary_df.empty:
            total_issues = 0
        else:
            threshold_mask = np.isclose(
                threshold_summary_df["threshold"].astype(float),
                float(threshold),
            )
            total_issues = int(
                threshold_summary_df.loc[threshold_mask, "instance_id"].nunique()
            )

        row = {
            "threshold": float(threshold),
            "total_solvable_issues": int(total_issues),
        }

        for strategy in strategy_to_source_columns:
            utility_df = utility_dfs[strategy]
            if utility_df.empty or threshold_key not in utility_df.columns:
                values = pd.Series(dtype=float)
            else:
                values = pd.to_numeric(
                    utility_df[threshold_key],
                    errors="coerce",
                ).dropna()

            available_count = int(len(values))
            availability = (
                available_count / total_issues
                if total_issues > 0
                else np.nan
            )

            # Honor the minimum support rule used by this analysis: report
            # threshold-level strategy averages only when more than 10 issue
            # values are available.
            if available_count > 10 and total_issues > 0:
                conditional_mean = float(values.mean())
                coverage_adjusted_mean = float(
                    values.sum() / total_issues
                )

                # Algebraic cross-check: overall = availability * conditional.
                expected_overall = float(availability * conditional_mean)
                if not np.isclose(
                    coverage_adjusted_mean,
                    expected_overall,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise AssertionError(
                        f"Coverage-adjusted U identity failed for {strategy} at "
                        f"threshold={threshold_key}: direct={coverage_adjusted_mean}, "
                        f"availability*conditional={expected_overall}"
                    )

                conditional_percent = 100.0 * conditional_mean
                coverage_adjusted_percent = 100.0 * coverage_adjusted_mean
            else:
                conditional_percent = np.nan
                coverage_adjusted_percent = np.nan

            row[f"{strategy}_available_issue_count"] = available_count
            row[f"{strategy}_availability_percent"] = (
                100.0 * availability
                if not pd.isna(availability)
                else np.nan
            )
            row[f"{strategy}_conditional_mean_U_percent"] = conditional_percent
            row[f"{strategy}_coverage_adjusted_U_percent"] = (
                coverage_adjusted_percent
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "prioritization_utility_U_summary_by_threshold.csv"
    summary_df.to_csv(summary_path, index=False, na_rep="NA")

    # ------------------------------------------------------------------
    # RQ2 paper-ready one-row summary.
    #
    # For each oracle-free selection strategy, find the highest observed
    # coverage-adjusted utility across the nine similarity thresholds and
    # record the threshold at which that maximum occurs. If the exact same
    # maximum occurs at multiple thresholds, use the LOWEST threshold as a
    # deterministic tie-break (matching the threshold preference already used
    # elsewhere in this pipeline).
    # ------------------------------------------------------------------
    rq2_summary_row = {
        "Model": MODEL,
        "Set": SET,
        "Similarity": metric,
    }

    for strategy in strategy_to_source_columns:
        value_column = f"{strategy}_coverage_adjusted_U_percent"
        numeric_values = pd.to_numeric(
            summary_df[value_column],
            errors="coerce",
        )
        valid_mask = numeric_values.notna()

        if not valid_mask.any():
            max_utility = np.nan
            best_threshold = np.nan
        else:
            max_utility = float(numeric_values.loc[valid_mask].max())
            tied_mask = valid_mask & np.isclose(
                numeric_values.astype(float),
                max_utility,
                rtol=0.0,
                atol=1e-12,
            )
            best_threshold = float(
                summary_df.loc[tied_mask, "threshold"].astype(float).min()
            )

        rq2_summary_row[
            f"{strategy}_highest_coverage_adjusted_U_percent"
        ] = max_utility
        rq2_summary_row[
            f"{strategy}_threshold_at_highest_coverage_adjusted_U"
        ] = best_threshold

    rq2_summary_df = pd.DataFrame([rq2_summary_row])
    rq2_summary_path = (
        output_dir / "rq2_highest_coverage_adjusted_utility_by_strategy.csv"
    )
    rq2_summary_df.to_csv(rq2_summary_path, index=False, na_rep="NA")

    # Conditional plot: when the strategy exists, how effective is it?
    conditional_plot_path = (
        output_dir / "average_prioritization_utility_U_by_threshold.pdf"
    )
    plt.figure(figsize=(9.0, 5.5))
    for strategy in strategy_to_source_columns:
        values = summary_df[f"{strategy}_conditional_mean_U_percent"]
        plt.plot(
            summary_df["threshold"],
            values,
            marker="o",
            label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Average utility (%)")
    plt.title(
        ""
    )
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 100.0)
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(conditional_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Coverage-adjusted plot: availability * conditional effectiveness.
    coverage_adjusted_plot_path = (
        output_dir / "coverage_adjusted_prioritization_utility_U_by_threshold.pdf"
    )
    plt.figure(figsize=(9.0, 5.5))
    for strategy in strategy_to_source_columns:
        values = summary_df[f"{strategy}_coverage_adjusted_U_percent"]
        plt.plot(
            summary_df["threshold"],
            values,
            marker="o",
            label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Coverage-adjusted prioritization utility (%)")
    plt.title(
        "Coverage-Adjusted Successful Candidate Reduction by Similarity Threshold"
    )
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 100.0)
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(coverage_adjusted_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "strategy_utility": utility_paths,
        "threshold_summary": summary_path,
        "rq2_highest_coverage_adjusted_utility_summary": rq2_summary_path,
        "conditional_utility_plot": conditional_plot_path,
        "coverage_adjusted_utility_plot": coverage_adjusted_plot_path,
    }


def save_rq2_precision_weighted_utility_outputs(
    threshold_summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Save precision-weighted candidate-reduction utility for RQ2.

    This is an additional metric; it does not replace the existing successful
    candidate-reduction utility U.

    For issue i, threshold t, and strategy s:

        R_i,s(t) = 1 - k_i,s(t) / n_i

        U_precision_i,s(t) = Precision_i,s(t) * R_i,s(t)

    where Precision_i,s(t) is the precision of the selected cluster/set,
    k_i,s(t) is its selected size, and n_i is the number of valid candidates.

    As with the existing U analysis, a one-component partition is not treated as
    meaningful prioritization. Structurally unavailable strategies are stored as
    NA in the raw issue-level CSVs.

    Two threshold-level summaries are produced:
      * conditional mean: average over issues where the strategy is available;
      * coverage-adjusted mean: sum of available issue values divided by all
        solvable issues at that threshold, equivalently availability times the
        conditional mean.

    The same minimum-support rule currently used by the existing U implementation
    is preserved here: a threshold-level point is reported only when more than 10
    issue values are available. Raw issue-level values remain on [0, 1]; summary
    and plot values are multiplied by 100.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_columns = [
        f"{threshold:.1f}"
        for threshold in SIMILARITY_THRESHOLDS
    ]
    output_columns = ["instance_id"] + threshold_columns

    strategy_to_source_columns = {
        "largest": {
            "size": "largest_cluster_size",
            "precision": "largest_cluster_precision",
        },
        "smallest": {
            "size": "smallest_cluster_size",
            "precision": "smallest_cluster_precision",
        },
        "singleton_union": {
            "size": "singleton_union_cluster_size",
            "precision": "singleton_union_cluster_precision",
        },
        "singleton_furthest": {
            "size": "singleton_furthest_cluster_size",
            "precision": "singleton_furthest_cluster_precision",
        },
    }

    weighted_rows = {
        strategy: []
        for strategy in strategy_to_source_columns
    }

    if not threshold_summary_df.empty:
        work_df = threshold_summary_df.copy()
        work_df["threshold"] = work_df["threshold"].astype(float)

        for instance_id, issue_df in work_df.groupby("instance_id"):
            rows_for_issue = {
                strategy: {"instance_id": str(instance_id)}
                for strategy in strategy_to_source_columns
            }

            for threshold in SIMILARITY_THRESHOLDS:
                threshold_key = f"{threshold:.1f}"
                matching = issue_df[
                    np.isclose(
                        issue_df["threshold"].astype(float),
                        float(threshold),
                    )
                ]

                if matching.empty:
                    for strategy in strategy_to_source_columns:
                        rows_for_issue[strategy][threshold_key] = np.nan
                    continue

                if len(matching) != 1:
                    raise ValueError(
                        f"Expected one row for instance_id={instance_id} at "
                        f"threshold={threshold_key}, found {len(matching)}"
                    )

                source_row = matching.iloc[0]
                num_total_patches = int(source_row["num_total_patches"])
                num_clusters = int(source_row["num_clusters"])

                if num_total_patches <= 0:
                    for strategy in strategy_to_source_columns:
                        rows_for_issue[strategy][threshold_key] = np.nan
                    continue

                for strategy, source_columns in strategy_to_source_columns.items():
                    weighted_value = np.nan
                    size_column = source_columns["size"]
                    precision_column = source_columns["precision"]

                    if (
                        num_clusters >= 2
                        and size_column in source_row.index
                        and precision_column in source_row.index
                    ):
                        selected_size = source_row[size_column]
                        selected_precision = source_row[precision_column]

                        if (
                            not pd.isna(selected_size)
                            and not pd.isna(selected_precision)
                        ):
                            selected_size = int(selected_size)
                            selected_precision = float(selected_precision)

                            if selected_size <= 0:
                                raise AssertionError(
                                    f"Available strategy {strategy} selected no "
                                    f"candidates for instance_id={instance_id}, "
                                    f"threshold={threshold_key}"
                                )
                            if selected_size > num_total_patches:
                                raise AssertionError(
                                    f"Strategy {strategy} selected {selected_size} "
                                    f"candidates from only {num_total_patches} valid "
                                    f"candidates for instance_id={instance_id}, "
                                    f"threshold={threshold_key}"
                                )
                            if selected_precision < 0.0 or selected_precision > 1.0:
                                raise AssertionError(
                                    f"Invalid precision {selected_precision} for strategy "
                                    f"{strategy}, instance_id={instance_id}, "
                                    f"threshold={threshold_key}"
                                )

                            candidate_reduction = (
                                1.0 - (selected_size / num_total_patches)
                            )
                            weighted_value = float(
                                selected_precision * candidate_reduction
                            )
                            weighted_value = float(
                                min(1.0, max(0.0, weighted_value))
                            )

                    rows_for_issue[strategy][threshold_key] = weighted_value

            for strategy in strategy_to_source_columns:
                weighted_rows[strategy].append(rows_for_issue[strategy])

    weighted_paths = {}
    weighted_dfs = {}

    for strategy in strategy_to_source_columns:
        weighted_df = pd.DataFrame(
            weighted_rows[strategy],
            columns=output_columns,
        )
        if not weighted_df.empty:
            weighted_df = weighted_df.sort_values(
                "instance_id"
            ).reset_index(drop=True)

        weighted_path = (
            output_dir / f"{strategy}_precision_weighted_U_by_threshold.csv"
        )
        weighted_df.to_csv(
            weighted_path,
            index=False,
            na_rep="NA",
        )
        weighted_paths[strategy] = weighted_path
        weighted_dfs[strategy] = weighted_df

    summary_rows = []
    for threshold in SIMILARITY_THRESHOLDS:
        threshold_key = f"{threshold:.1f}"

        if threshold_summary_df.empty:
            total_issues = 0
        else:
            threshold_mask = np.isclose(
                threshold_summary_df["threshold"].astype(float),
                float(threshold),
            )
            total_issues = int(
                threshold_summary_df.loc[
                    threshold_mask,
                    "instance_id",
                ].nunique()
            )

        row = {
            "threshold": float(threshold),
            "total_solvable_issues": int(total_issues),
        }

        for strategy in strategy_to_source_columns:
            weighted_df = weighted_dfs[strategy]
            if weighted_df.empty or threshold_key not in weighted_df.columns:
                values = pd.Series(dtype=float)
            else:
                values = pd.to_numeric(
                    weighted_df[threshold_key],
                    errors="coerce",
                ).dropna()

            available_count = int(len(values))
            availability = (
                available_count / total_issues
                if total_issues > 0
                else np.nan
            )

            if available_count > 10 and total_issues > 0:
                conditional_mean = float(values.mean())
                coverage_adjusted_mean = float(values.sum() / total_issues)

                expected_overall = float(availability * conditional_mean)
                if not np.isclose(
                    coverage_adjusted_mean,
                    expected_overall,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise AssertionError(
                        "Coverage-adjusted precision-weighted U identity failed "
                        f"for {strategy} at threshold={threshold_key}: "
                        f"direct={coverage_adjusted_mean}, "
                        f"availability*conditional={expected_overall}"
                    )

                conditional_percent = 100.0 * conditional_mean
                coverage_adjusted_percent = 100.0 * coverage_adjusted_mean
            else:
                conditional_percent = np.nan
                coverage_adjusted_percent = np.nan

            row[f"{strategy}_available_issue_count"] = available_count
            row[f"{strategy}_availability_percent"] = (
                100.0 * availability
                if not pd.isna(availability)
                else np.nan
            )
            row[f"{strategy}_conditional_mean_precision_weighted_U_percent"] = (
                conditional_percent
            )
            row[f"{strategy}_coverage_adjusted_precision_weighted_U_percent"] = (
                coverage_adjusted_percent
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "precision_weighted_U_summary_by_threshold.csv"
    summary_df.to_csv(summary_path, index=False, na_rep="NA")

    conditional_plot_path = (
        output_dir / "average_precision_weighted_U_by_threshold.pdf"
    )
    plt.figure(figsize=(9.0, 5.5))
    for strategy in strategy_to_source_columns:
        values = summary_df[
            f"{strategy}_conditional_mean_precision_weighted_U_percent"
        ]
        plt.plot(
            summary_df["threshold"],
            values,
            marker="o",
            label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Average precision-weighted utility (%)")
    plt.title("")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 100.0)
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(conditional_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    coverage_adjusted_plot_path = (
        output_dir / "coverage_adjusted_precision_weighted_U_by_threshold.pdf"
    )
    plt.figure(figsize=(9.0, 5.5))
    for strategy in strategy_to_source_columns:
        values = summary_df[
            f"{strategy}_coverage_adjusted_precision_weighted_U_percent"
        ]
        plt.plot(
            summary_df["threshold"],
            values,
            marker="o",
            label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Coverage-adjusted precision-weighted utility (%)")
    plt.title("Coverage-Adjusted Precision-Weighted Candidate Reduction")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 100.0)
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(coverage_adjusted_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "strategy_precision_weighted_utility": weighted_paths,
        "threshold_summary": summary_path,
        "conditional_plot": conditional_plot_path,
        "coverage_adjusted_plot": coverage_adjusted_plot_path,
    }


def save_rq2_strategy_agreement_correctness_precision_summary(
    threshold_summary_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Save RQ2 strategy availability, correct-fix hits, and mean precision.

    The CSV is organized with thresholds as columns and 12 metric rows: three rows
    for each oracle-free strategy.

    For each strategy and threshold:
      * available_issue_count: number of solvable issues where clustering produced
        at least two components and the strategy is structurally available;
      * with_correct_fix_count: among those available selections, number whose
        selected cluster/set contains at least one passing candidate (TP > 0);
      * average_precision: mean selected-set precision over all available issues,
        including precision=0 when an available selected set contains no pass.

    This aligns the availability definition with the RQ2 utility analysis: a
    one-component partition is not counted as meaningful prioritization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_to_source_columns = {
        "largest": {
            "size": "largest_cluster_size",
            "tp": "largest_cluster_tp",
            "precision": "largest_cluster_precision",
        },
        "smallest": {
            "size": "smallest_cluster_size",
            "tp": "smallest_cluster_tp",
            "precision": "smallest_cluster_precision",
        },
        "singleton_union": {
            "size": "singleton_union_cluster_size",
            "tp": "singleton_union_cluster_tp",
            "precision": "singleton_union_cluster_precision",
        },
        "singleton_furthest": {
            "size": "singleton_furthest_cluster_size",
            "tp": "singleton_furthest_cluster_tp",
            "precision": "singleton_furthest_cluster_precision",
        },
    }

    threshold_columns = [
        f"{threshold:.1f}"
        for threshold in SIMILARITY_THRESHOLDS
    ]

    rows_by_metric = {}
    for strategy in strategy_to_source_columns:
        rows_by_metric[f"{strategy}_available_issue_count"] = {
            "metric": f"{strategy}_available_issue_count"
        }
        rows_by_metric[f"{strategy}_with_correct_fix_count"] = {
            "metric": f"{strategy}_with_correct_fix_count"
        }
        rows_by_metric[f"{strategy}_average_precision"] = {
            "metric": f"{strategy}_average_precision"
        }

    if threshold_summary_df.empty:
        for row in rows_by_metric.values():
            for threshold_key in threshold_columns:
                row[threshold_key] = np.nan
    else:
        work_df = threshold_summary_df.copy()
        work_df["threshold"] = work_df["threshold"].astype(float)

        for threshold in SIMILARITY_THRESHOLDS:
            threshold_key = f"{threshold:.1f}"
            threshold_df = work_df[
                np.isclose(work_df["threshold"], float(threshold))
            ].copy()

            if threshold_df.empty:
                for row in rows_by_metric.values():
                    row[threshold_key] = np.nan
                continue

            multicluster_mask = threshold_df["num_clusters"].astype(int) >= 2

            for strategy, source_columns in strategy_to_source_columns.items():
                size_column = source_columns["size"]
                tp_column = source_columns["tp"]
                precision_column = source_columns["precision"]

                required_columns = [size_column, tp_column, precision_column]
                if any(column not in threshold_df.columns for column in required_columns):
                    available_df = pd.DataFrame()
                else:
                    available_mask = (
                        multicluster_mask
                        & threshold_df[size_column].notna()
                        & threshold_df[tp_column].notna()
                        & threshold_df[precision_column].notna()
                    )
                    available_df = threshold_df.loc[available_mask].copy()

                available_count = int(len(available_df))
                if available_count == 0:
                    hit_count = 0
                    average_precision = np.nan
                else:
                    hit_count = int(
                        (available_df[tp_column].astype(float) > 0.0).sum()
                    )
                    average_precision = float(
                        available_df[precision_column].astype(float).mean()
                    )

                rows_by_metric[
                    f"{strategy}_available_issue_count"
                ][threshold_key] = available_count
                rows_by_metric[
                    f"{strategy}_with_correct_fix_count"
                ][threshold_key] = hit_count
                rows_by_metric[
                    f"{strategy}_average_precision"
                ][threshold_key] = average_precision

    row_order = []
    for strategy in strategy_to_source_columns:
        row_order.extend([
            f"{strategy}_available_issue_count",
            f"{strategy}_with_correct_fix_count",
            f"{strategy}_average_precision",
        ])

    output_df = pd.DataFrame(
        [rows_by_metric[row_name] for row_name in row_order],
        columns=["metric"] + threshold_columns,
    )

    output_path = (
        output_dir / "strategy_agreement_correctness_precision_by_threshold.csv"
    )
    output_df.to_csv(output_path, index=False, na_rep="NA")

    # Build the two RQ2 agreement-quality plots directly from the saved CSV so the
    # plotted values are exactly the values exposed in the analysis artifact.
    plot_df = pd.read_csv(output_path, index_col="metric")
    strategies = list(strategy_to_source_columns.keys())

    # 1. At-least-one-correct-fix rate among issues where the strategy is available.
    #    rate_s(t) = with_correct_fix_count_s(t) / available_issue_count_s(t).
    correct_fix_rate_plot_path = (
        output_dir / "strategy_at_least_one_correct_fix_rate_by_threshold.pdf"
    )
    plt.figure(figsize=(9.0, 5.5))
    for strategy in strategies:
        available = pd.to_numeric(
            plot_df.loc[f"{strategy}_available_issue_count", threshold_columns],
            errors="coerce",
        ).astype(float)
        with_correct_fix = pd.to_numeric(
            plot_df.loc[f"{strategy}_with_correct_fix_count", threshold_columns],
            errors="coerce",
        ).astype(float)

        rate_percent = pd.Series(
            np.where(
                available.to_numpy() > 0.0,
                100.0 * with_correct_fix.to_numpy() / available.to_numpy(),
                np.nan,
            ),
            index=threshold_columns,
            dtype=float,
        )

        plt.plot(
            SIMILARITY_THRESHOLDS,
            rate_percent.to_numpy(),
            marker="o",
            label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("At-least-one-correct-fix rate (%)")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 100.0)
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(correct_fix_rate_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Mean precision of the selected group among issues where the strategy exists.
    average_precision_plot_path = (
        output_dir / "strategy_average_precision_by_threshold.pdf"
    )
    plt.figure(figsize=(9.0, 5.5))
    for strategy in strategies:
        average_precision = pd.to_numeric(
            plot_df.loc[f"{strategy}_average_precision", threshold_columns],
            errors="coerce",
        ).astype(float)

        plt.plot(
            SIMILARITY_THRESHOLDS,
            100.0 * average_precision.to_numpy(),
            marker="o",
            label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Average precision (%)")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 100.0)
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(average_precision_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path






def save_rq2_all_thresholds_group_coverage(
    cluster_hit_advantage_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    results_dir: Path,
    agreement_definition: str,
) -> dict:
    """Save RQ2 correct-fix coverage and group counts for all thresholds.

    For a solvable issue i and an available candidate group G:

        correct-fix coverage(i, G)
            = (# correct fixes in G) / (# correct fixes generated for issue i).

    Coverage is averaged only over solvable issues where the group exists;
    unavailable groups are excluded rather than assigned zero.

    The group-count CSV contains two rows per agreement category:
      * the number of issues where the group exists; and
      * the number of those issues where the group contains at least one correct fix.
    """
    if agreement_definition not in {"majority", "largest"}:
        raise ValueError(
            "agreement_definition must be either 'majority' or 'largest', got "
            f"{agreement_definition!r}"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # RQ2 no longer produces precision outputs or a separate FP main-results file.
    # Remove stale files from earlier runs so the directory reflects the current design.
    for stale_path in [
        results_dir / "rq2_all_thresholds_precision.csv",
        results_dir / "rq2_all_thresholds_precision.png",
        results_dir / "rq2_main_results_FP.csv",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    threshold_keys = [f"{threshold:.1f}" for threshold in SIMILARITY_THRESHOLDS]
    group_order = [
        "majority agreement",
        "minority agreement",
        "disagreement",
    ]

    coverage_values = {
        group: {threshold_key: [] for threshold_key in threshold_keys}
        for group in group_order
    }
    available_counts = {
        group: {threshold_key: 0 for threshold_key in threshold_keys}
        for group in group_order
    }
    correct_group_counts = {
        group: {threshold_key: 0 for threshold_key in threshold_keys}
        for group in group_order
    }

    required_summary_columns = {
        "instance_id",
        "threshold",
        "num_total_patches",
        "num_passes_for_issue",
        "non_singleton_cluster_count",
        "largest_cluster_size",
        "largest_cluster_tp",
        "smallest_cluster_size",
        "smallest_cluster_tp",
    }
    missing_summary_columns = required_summary_columns - set(threshold_summary_df.columns)
    if missing_summary_columns:
        raise ValueError(
            "Cannot compute RQ2 correct-fix coverage because threshold-summary "
            f"columns are missing: {sorted(missing_summary_columns)}"
        )

    required_cluster_columns = {
        "Issue",
        "Threshold",
        "Cluster size",
        "Actual hit H",
    }
    missing_cluster_columns = required_cluster_columns - set(cluster_hit_advantage_df.columns)
    if missing_cluster_columns:
        raise ValueError(
            "Cannot compute RQ2 disagreement correct-fix coverage because "
            f"cluster-level columns are missing: {sorted(missing_cluster_columns)}"
        )

    summary_df = threshold_summary_df.copy()
    summary_df["instance_id"] = summary_df["instance_id"].astype(str)
    summary_df["threshold"] = pd.to_numeric(summary_df["threshold"], errors="raise")

    cluster_df = cluster_hit_advantage_df.copy()
    cluster_df["Issue"] = cluster_df["Issue"].astype(str)
    cluster_df["Threshold"] = pd.to_numeric(cluster_df["Threshold"], errors="raise")
    cluster_df["Cluster size"] = pd.to_numeric(
        cluster_df["Cluster size"], errors="raise"
    ).astype(int)
    cluster_df["Actual hit H"] = pd.to_numeric(
        cluster_df["Actual hit H"], errors="raise"
    )

    cluster_lookup = {
        (str(issue), f"{float(threshold):.1f}"): group.copy()
        for (issue, threshold), group in cluster_df.groupby(
            ["Issue", "Threshold"], sort=False
        )
    }

    for threshold in SIMILARITY_THRESHOLDS:
        threshold_key = f"{threshold:.1f}"
        threshold_df = summary_df[
            np.isclose(summary_df["threshold"].astype(float), float(threshold))
        ].copy()

        duplicate_mask = threshold_df.duplicated(subset=["instance_id"], keep=False)
        if duplicate_mask.any():
            duplicate_ids = sorted(
                threshold_df.loc[duplicate_mask, "instance_id"].astype(str).unique()
            )
            raise ValueError(
                f"RQ2 coverage expected one solvable row per issue at threshold "
                f"{threshold_key}; duplicates found for {duplicate_ids[:10]}"
            )

        for _, row in threshold_df.iterrows():
            issue = str(row["instance_id"])
            n_i = int(row["num_total_patches"])
            total_correct = int(row["num_passes_for_issue"])

            if n_i <= 0:
                continue
            if total_correct <= 0:
                raise ValueError(
                    "RQ2 correct-fix coverage is defined on solvable issues only, but "
                    f"issue {issue} has num_passes_for_issue={total_correct}"
                )

            non_singleton_count = int(row["non_singleton_cluster_count"])
            largest_size_raw = row["largest_cluster_size"]
            largest_tp_raw = row["largest_cluster_tp"]
            largest_available = (
                non_singleton_count >= 1
                and not pd.isna(largest_size_raw)
                and not pd.isna(largest_tp_raw)
            )

            majority_available = False
            largest_size = None
            if largest_available:
                largest_size = int(largest_size_raw)
                largest_tp = int(largest_tp_raw)

                proper_non_singleton = bool(2 <= largest_size < n_i)
                if agreement_definition == "majority":
                    majority_available = bool(
                        proper_non_singleton and largest_size > (n_i / 2.0)
                    )
                else:
                    majority_available = proper_non_singleton

                if majority_available:
                    available_counts["majority agreement"][threshold_key] += 1
                    if largest_tp > 0:
                        correct_group_counts["majority agreement"][threshold_key] += 1
                    coverage_values["majority agreement"][threshold_key].append(
                        float(largest_tp / total_correct)
                    )

            # Minority agreement is the smallest agreeing group distinct from the
            # selected majority/largest group. If there is only one proper
            # non-singleton and no majority, that group is the minority agreement.
            if non_singleton_count >= 2:
                smallest_size_raw = row["smallest_cluster_size"]
                smallest_tp_raw = row["smallest_cluster_tp"]
                if not pd.isna(smallest_size_raw) and not pd.isna(smallest_tp_raw):
                    smallest_size = int(smallest_size_raw)
                    smallest_tp = int(smallest_tp_raw)
                    if 2 <= smallest_size < n_i:
                        available_counts["minority agreement"][threshold_key] += 1
                        if smallest_tp > 0:
                            correct_group_counts["minority agreement"][threshold_key] += 1
                        coverage_values["minority agreement"][threshold_key].append(
                            float(smallest_tp / total_correct)
                        )
            elif (
                non_singleton_count == 1
                and largest_available
                and largest_size is not None
                and 2 <= largest_size < n_i
                and not majority_available
            ):
                largest_tp = int(largest_tp_raw)
                available_counts["minority agreement"][threshold_key] += 1
                if largest_tp > 0:
                    correct_group_counts["minority agreement"][threshold_key] += 1
                coverage_values["minority agreement"][threshold_key].append(
                    float(largest_tp / total_correct)
                )

            # Disagreement is the union of all singleton components.
            issue_clusters = cluster_lookup.get((issue, threshold_key))
            if issue_clusters is not None and not issue_clusters.empty:
                singleton_rows = issue_clusters[
                    issue_clusters["Cluster size"].astype(int) == 1
                ]
                if not singleton_rows.empty:
                    disagreement_correct = float(
                        (
                            singleton_rows["Actual hit H"].astype(float)
                            / 100.0
                            * singleton_rows["Cluster size"].astype(float)
                        ).sum()
                    )
                    disagreement_coverage = float(
                        disagreement_correct / total_correct
                    )
                    disagreement_coverage = float(
                        min(1.0, max(0.0, disagreement_coverage))
                    )
                    available_counts["disagreement"][threshold_key] += 1
                    if disagreement_correct > 1e-12:
                        correct_group_counts["disagreement"][threshold_key] += 1
                    coverage_values["disagreement"][threshold_key].append(
                        disagreement_coverage
                    )

    coverage_rows = []
    for group in group_order:
        coverage_row = {"group": group}
        for threshold_key in threshold_keys:
            values = coverage_values[group][threshold_key]
            if len(values) != available_counts[group][threshold_key]:
                raise AssertionError(
                    "RQ2 coverage support does not match group availability for "
                    f"{group} at threshold {threshold_key}: "
                    f"coverage={len(values)}, available={available_counts[group][threshold_key]}"
                )
            coverage_row[threshold_key] = (
                float(np.mean(values)) if values else np.nan
            )
        coverage_rows.append(coverage_row)

    output_columns = ["group"] + threshold_keys
    coverage_df = pd.DataFrame(coverage_rows, columns=output_columns)

    count_rows = []
    for group in group_order:
        available_row = {"group": f"{group} count"}
        correct_row = {"group": f"{group} with at least one correct fix"}
        for threshold_key in threshold_keys:
            available_count = int(available_counts[group][threshold_key])
            correct_count = int(correct_group_counts[group][threshold_key])
            if correct_count > available_count:
                raise AssertionError(
                    f"RQ2 correct-group count exceeds availability for {group} at "
                    f"threshold {threshold_key}: {correct_count}>{available_count}"
                )
            available_row[threshold_key] = available_count
            correct_row[threshold_key] = correct_count
        count_rows.extend([available_row, correct_row])
    count_df = pd.DataFrame(count_rows, columns=output_columns)

    coverage_path = results_dir / "rq2_all_thresholds_coverage.csv"
    count_path = results_dir / "rq2_all_thresholds_group_count.csv"
    plot_path = results_dir / "rq2_all_thresholds_coverage.png"
    coverage_df.to_csv(coverage_path, index=False, na_rep="NA")
    count_df.to_csv(count_path, index=False)

    # Plot exactly the conditional mean correct-fix coverage values saved in the CSV.
    plot_df = pd.read_csv(coverage_path, index_col="group")
    plt.figure(figsize=(9.0, 5.5))
    for group in group_order:
        y_values = pd.to_numeric(
            plot_df.loc[group, threshold_keys], errors="coerce"
        ).to_numpy(dtype=float)
        plt.plot(
            SIMILARITY_THRESHOLDS,
            y_values,
            marker="o",
            label=group,
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Average correct-fix coverage")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "coverage_csv": coverage_path,
        "group_count_csv": count_path,
        "coverage_plot": plot_path,
    }


def save_rq2_all_thresholds_group_false_positives(
    cluster_hit_advantage_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    results_dir: Path,
    agreement_definition: str,
) -> dict:
    """Save RQ2 conditional average false-positive counts for all thresholds.

    A false positive is an incorrect candidate patch contained in an agreement
    group. For each solvable issue i and candidate group G:

        false positives(i, G) = # incorrect candidate patches in G.

    The average is conditional on BOTH requirements below:
      * the group exists for the issue at the threshold; and
      * the group contains at least one correct candidate patch.

    Issues where the group is unavailable or contains no correct candidate patch
    are excluded rather than assigned zero.
    """
    if agreement_definition not in {"majority", "largest"}:
        raise ValueError(
            "agreement_definition must be either 'majority' or 'largest', got "
            f"{agreement_definition!r}"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    threshold_keys = [f"{threshold:.1f}" for threshold in SIMILARITY_THRESHOLDS]
    group_order = [
        "majority agreement",
        "minority agreement",
        "disagreement",
    ]

    false_positive_values = {
        group: {threshold_key: [] for threshold_key in threshold_keys}
        for group in group_order
    }

    required_summary_columns = {
        "instance_id",
        "threshold",
        "num_total_patches",
        "non_singleton_cluster_count",
        "largest_cluster_size",
        "largest_cluster_tp",
        "largest_cluster_fp",
        "smallest_cluster_size",
        "smallest_cluster_tp",
        "smallest_cluster_fp",
    }
    missing_summary_columns = required_summary_columns - set(threshold_summary_df.columns)
    if missing_summary_columns:
        raise ValueError(
            "Cannot compute RQ2 false-positive counts because threshold-summary "
            f"columns are missing: {sorted(missing_summary_columns)}"
        )

    required_cluster_columns = {
        "Issue",
        "Threshold",
        "Cluster size",
        "Actual hit H",
    }
    missing_cluster_columns = required_cluster_columns - set(cluster_hit_advantage_df.columns)
    if missing_cluster_columns:
        raise ValueError(
            "Cannot compute RQ2 disagreement false-positive counts because "
            f"cluster-level columns are missing: {sorted(missing_cluster_columns)}"
        )

    summary_df = threshold_summary_df.copy()
    summary_df["instance_id"] = summary_df["instance_id"].astype(str)
    summary_df["threshold"] = pd.to_numeric(summary_df["threshold"], errors="raise")

    cluster_df = cluster_hit_advantage_df.copy()
    cluster_df["Issue"] = cluster_df["Issue"].astype(str)
    cluster_df["Threshold"] = pd.to_numeric(cluster_df["Threshold"], errors="raise")
    cluster_df["Cluster size"] = pd.to_numeric(
        cluster_df["Cluster size"], errors="raise"
    ).astype(int)
    cluster_df["Actual hit H"] = pd.to_numeric(
        cluster_df["Actual hit H"], errors="raise"
    )

    cluster_lookup = {
        (str(issue), f"{float(threshold):.1f}"): group.copy()
        for (issue, threshold), group in cluster_df.groupby(
            ["Issue", "Threshold"], sort=False
        )
    }

    for threshold in SIMILARITY_THRESHOLDS:
        threshold_key = f"{threshold:.1f}"
        threshold_df = summary_df[
            np.isclose(summary_df["threshold"].astype(float), float(threshold))
        ].copy()

        duplicate_mask = threshold_df.duplicated(subset=["instance_id"], keep=False)
        if duplicate_mask.any():
            duplicate_ids = sorted(
                threshold_df.loc[duplicate_mask, "instance_id"].astype(str).unique()
            )
            raise ValueError(
                f"RQ2 false positives expected one solvable row per issue at threshold "
                f"{threshold_key}; duplicates found for {duplicate_ids[:10]}"
            )

        for _, row in threshold_df.iterrows():
            issue = str(row["instance_id"])
            n_i = int(row["num_total_patches"])
            if n_i <= 0:
                continue

            non_singleton_count = int(row["non_singleton_cluster_count"])
            largest_size_raw = row["largest_cluster_size"]
            largest_tp_raw = row["largest_cluster_tp"]
            largest_fp_raw = row["largest_cluster_fp"]
            largest_available = (
                non_singleton_count >= 1
                and not pd.isna(largest_size_raw)
                and not pd.isna(largest_tp_raw)
                and not pd.isna(largest_fp_raw)
            )

            majority_available = False
            largest_size = None
            if largest_available:
                largest_size = int(largest_size_raw)
                largest_tp = int(largest_tp_raw)
                largest_fp = int(largest_fp_raw)

                proper_non_singleton = bool(2 <= largest_size < n_i)
                if agreement_definition == "majority":
                    majority_available = bool(
                        proper_non_singleton and largest_size > (n_i / 2.0)
                    )
                else:
                    majority_available = proper_non_singleton

                # FP is averaged only where the group contains a correct patch.
                if majority_available and largest_tp > 0:
                    false_positive_values["majority agreement"][threshold_key].append(
                        largest_fp
                    )

            if non_singleton_count >= 2:
                smallest_size_raw = row["smallest_cluster_size"]
                smallest_tp_raw = row["smallest_cluster_tp"]
                smallest_fp_raw = row["smallest_cluster_fp"]
                if (
                    not pd.isna(smallest_size_raw)
                    and not pd.isna(smallest_tp_raw)
                    and not pd.isna(smallest_fp_raw)
                ):
                    smallest_size = int(smallest_size_raw)
                    smallest_tp = int(smallest_tp_raw)
                    if 2 <= smallest_size < n_i and smallest_tp > 0:
                        false_positive_values["minority agreement"][threshold_key].append(
                            int(smallest_fp_raw)
                        )
            elif (
                non_singleton_count == 1
                and largest_available
                and largest_size is not None
                and 2 <= largest_size < n_i
                and not majority_available
            ):
                largest_tp = int(largest_tp_raw)
                if largest_tp > 0:
                    false_positive_values["minority agreement"][threshold_key].append(
                        int(largest_fp_raw)
                    )

            # Disagreement is the union of all singleton components. Only issues
            # where at least one singleton is correct contribute to the FP average.
            issue_clusters = cluster_lookup.get((issue, threshold_key))
            if issue_clusters is not None and not issue_clusters.empty:
                singleton_rows = issue_clusters[
                    issue_clusters["Cluster size"].astype(int) == 1
                ]
                if not singleton_rows.empty:
                    disagreement_correct = float(
                        (
                            singleton_rows["Actual hit H"].astype(float)
                            / 100.0
                            * singleton_rows["Cluster size"].astype(float)
                        ).sum()
                    )
                    disagreement_fp_float = float(
                        (
                            singleton_rows["Cluster size"].astype(float)
                            * (
                                1.0
                                - singleton_rows["Actual hit H"].astype(float) / 100.0
                            )
                        ).sum()
                    )
                    disagreement_fp = int(round(disagreement_fp_float))
                    if not np.isclose(
                        disagreement_fp_float,
                        float(disagreement_fp),
                        rtol=0.0,
                        atol=1e-9,
                    ):
                        raise AssertionError(
                            "Disagreement false-positive count is not integral for "
                            f"instance_id={issue}, threshold={threshold_key}: "
                            f"{disagreement_fp_float}"
                        )
                    if disagreement_correct > 1e-12:
                        false_positive_values["disagreement"][threshold_key].append(
                            disagreement_fp
                        )

    false_positive_rows = []
    false_positive_counts = {
        group: {} for group in group_order
    }
    for group in group_order:
        false_positive_row = {"group": group}
        for threshold_key in threshold_keys:
            values = false_positive_values[group][threshold_key]
            false_positive_counts[group][threshold_key] = int(len(values))
            false_positive_row[threshold_key] = (
                float(np.mean(values)) if values else np.nan
            )
        false_positive_rows.append(false_positive_row)

    output_columns = ["group"] + threshold_keys
    false_positive_df = pd.DataFrame(
        false_positive_rows,
        columns=output_columns,
    )

    false_positive_path = results_dir / "rq2_all_thresholds_FP.csv"
    false_positive_plot_path = results_dir / "rq2_all_thresholds_FP.png"
    false_positive_df.to_csv(false_positive_path, index=False, na_rep="NA")

    # The FP support must exactly match the count of available groups containing
    # at least one correct candidate patch.
    count_path = results_dir / "rq2_all_thresholds_group_count.csv"
    if not count_path.exists():
        raise FileNotFoundError(
            "Cannot validate RQ2 false-positive support because group-count file "
            f"is missing: {count_path}"
        )
    count_df = pd.read_csv(count_path).set_index("group")
    for group in group_order:
        correct_count_row = f"{group} with at least one correct fix"
        if correct_count_row not in count_df.index:
            raise KeyError(
                f"Row {correct_count_row!r} is missing from {count_path}"
            )
        for threshold_key in threshold_keys:
            expected_count = int(
                pd.to_numeric(
                    pd.Series([count_df.loc[correct_count_row, threshold_key]]),
                    errors="raise",
                ).iloc[0]
            )
            actual_count = false_positive_counts[group][threshold_key]
            if actual_count != expected_count:
                raise AssertionError(
                    "RQ2 FP support mismatch for "
                    f"{group} at threshold {threshold_key}: "
                    f"correct-containing count={expected_count}, FP count={actual_count}"
                )

    # Plot exactly the conditional mean false-positive counts saved in the CSV.
    plot_df = pd.read_csv(false_positive_path, index_col="group")
    plt.figure(figsize=(9.0, 5.5))
    for group in group_order:
        y_values = pd.to_numeric(
            plot_df.loc[group, threshold_keys], errors="coerce"
        ).to_numpy(dtype=float)
        plt.plot(
            SIMILARITY_THRESHOLDS,
            y_values,
            marker="o",
            label=group,
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Average number of incorrect candidate patches")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(false_positive_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "false_positive_csv": false_positive_path,
        "false_positive_plot": false_positive_plot_path,
    }


def save_rq2_main_results_at_rq1_threshold(
    results_dir: Path,
) -> pd.DataFrame:
    """Save the combined RQ2 coverage/count/FP results at fixed threshold 0.9.

    Coverage is averaged over issues where the corresponding group exists.
    False positives are averaged over the stricter subset where the group exists
    and contains at least one correct candidate patch.

    The function name is retained for backward compatibility with the existing
    pipeline call site; RQ1's selected t* is not read or used here.
    """
    results_dir = Path(results_dir)

    coverage_path = results_dir / "rq2_all_thresholds_coverage.csv"
    count_path = results_dir / "rq2_all_thresholds_group_count.csv"
    false_positive_path = results_dir / "rq2_all_thresholds_FP.csv"
    output_path = results_dir / "rq2_main_results.csv"

    # Clean obsolete RQ2 outputs from earlier versions.
    for legacy_output_path in [
        results_dir / "rq2_main_results.2csv",
        results_dir / "rq2_main_results_FP.csv",
        results_dir / "rq2_all_thresholds_precision.csv",
        results_dir / "rq2_all_thresholds_precision.png",
    ]:
        if legacy_output_path.exists():
            legacy_output_path.unlink()

    for required_path in [coverage_path, count_path, false_positive_path]:
        if not required_path.exists():
            raise FileNotFoundError(
                "Cannot build RQ2 main results; required file is missing: "
                f"{required_path}"
            )

    fixed_threshold = float(PAPER_FIXED_THRESHOLD)
    threshold_key = f"{fixed_threshold:.1f}"

    output_columns = [
        "Dataset",
        "Model",
        "Similarity",
        "t*",
        "Majority n",
        "Majority correct n",
        "Majority Coverage",
        "Majority FP",
        "Minority n",
        "Minority correct n",
        "Minority Coverage",
        "Minority FP",
        "Disagreement n",
        "Disagreement correct n",
        "Disagreement Coverage",
        "Disagreement FP",
    ]

    result_row = {
        "Dataset": SET,
        "Model": MODEL,
        "Similarity": metric,
        "t*": fixed_threshold,
    }
    for column in output_columns[4:]:
        result_row[column] = np.nan

    coverage_df = pd.read_csv(coverage_path).set_index("group")
    count_df = pd.read_csv(count_path).set_index("group")
    false_positive_df = pd.read_csv(false_positive_path).set_index("group")

    for input_path, input_df in [
        (coverage_path, coverage_df),
        (count_path, count_df),
        (false_positive_path, false_positive_df),
    ]:
        if threshold_key not in input_df.columns:
            raise KeyError(
                f"Fixed RQ2 threshold {threshold_key} is missing from {input_path}"
            )

    group_specs = [
        (
            "majority agreement",
            "Majority n",
            "Majority correct n",
            "Majority Coverage",
            "Majority FP",
        ),
        (
            "minority agreement",
            "Minority n",
            "Minority correct n",
            "Minority Coverage",
            "Minority FP",
        ),
        (
            "disagreement",
            "Disagreement n",
            "Disagreement correct n",
            "Disagreement Coverage",
            "Disagreement FP",
        ),
    ]

    for group, n_column, correct_n_column, coverage_column, fp_column in group_specs:
        available_count_row = f"{group} count"
        correct_count_row = f"{group} with at least one correct fix"

        if group not in coverage_df.index or group not in false_positive_df.index:
            raise KeyError(
                f"Group {group!r} is missing from RQ2 coverage/FP outputs"
            )
        if available_count_row not in count_df.index:
            raise KeyError(
                f"Row {available_count_row!r} is missing from {count_path}"
            )
        if correct_count_row not in count_df.index:
            raise KeyError(
                f"Row {correct_count_row!r} is missing from {count_path}"
            )

        n_value = pd.to_numeric(
            pd.Series([count_df.loc[available_count_row, threshold_key]]),
            errors="coerce",
        ).iloc[0]
        correct_n_value = pd.to_numeric(
            pd.Series([count_df.loc[correct_count_row, threshold_key]]),
            errors="coerce",
        ).iloc[0]
        coverage_value = pd.to_numeric(
            pd.Series([coverage_df.loc[group, threshold_key]]), errors="coerce"
        ).iloc[0]
        fp_value = pd.to_numeric(
            pd.Series([false_positive_df.loc[group, threshold_key]]), errors="coerce"
        ).iloc[0]

        result_row[n_column] = int(n_value) if not pd.isna(n_value) else np.nan
        result_row[correct_n_column] = (
            int(correct_n_value) if not pd.isna(correct_n_value) else np.nan
        )
        result_row[coverage_column] = (
            float(coverage_value) if not pd.isna(coverage_value) else np.nan
        )
        result_row[fp_column] = (
            float(fp_value) if not pd.isna(fp_value) else np.nan
        )

        if (
            not pd.isna(result_row[n_column])
            and not pd.isna(result_row[correct_n_column])
            and result_row[correct_n_column] > result_row[n_column]
        ):
            raise AssertionError(
                f"RQ2 correct count exceeds availability for {group}: "
                f"{result_row[correct_n_column]}>{result_row[n_column]}"
            )

    result_df = pd.DataFrame([result_row], columns=output_columns)
    result_df.to_csv(output_path, index=False, na_rep="NA")

    print()
    print("=" * 80)
    print(f"RQ2 MAIN RESULTS AT FIXED T={fixed_threshold:.1f}")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print(f"RQ2 main-results CSV: {output_path}")
    print("=" * 80)

    return result_df





def compute_random_precision_baseline(
    num_valid_candidates: int,
    num_passing_candidates: int,
    cluster_size: int,
) -> float:
    """Return the RQ2 random precision baseline q_i = m_i / n_i.

    H and q must measure the same quantity. H is the precision of the actual
    agreement/disagreement group, while q is the precision expected from random
    candidate selection for the same issue:

        H_C = (# passing candidates in C) / |C|
        q_i = m_i / n_i
        A_C = H_C - q_i

    q_i does not depend on group size k. ``cluster_size`` is kept only to validate
    that the evaluated group is a valid subset of the issue.
    """
    n = int(num_valid_candidates)
    m = int(num_passing_candidates)
    k = int(cluster_size)

    if n <= 0:
        raise ValueError(f"num_valid_candidates must be positive, got {n}")
    if m < 0 or m > n:
        raise ValueError(f"num_passing_candidates must be in [0, n], got m={m}, n={n}")
    if k <= 0 or k > n:
        raise ValueError(f"cluster_size must be in [1, n], got k={k}, n={n}")

    return float(m / n)


def save_cluster_hit_advantage_outputs(
    cluster_hit_advantage_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Save cluster-level random-baseline hit advantage summaries and plot.

    This analysis intentionally ignores cluster-selection strategies. Every
    actual connected-component cluster is evaluated.

    For issue i, threshold t, and an actual cluster C of size k:

        H_C = 100 * (# passing candidates in C / k)
        q_i = 100 * E[# passing candidates in a random size-k subset / k]
            = 100 * (m_i / n_i)
        A_C = H_C - q_i

    Thus H_C is the probability that one candidate picked from the specific
    cluster is correct (the cluster precision), while q_i is the corresponding
    expected probability for a uniformly random candidate from the issue
    (equivalently, the expected precision of a random same-size subset). Negative
    gains are retained: A_C < 0 means the cluster is less precise than the
    issue-level random baseline. H, q, and A are stored in
    percentage / percentage-point units. Legacy output filenames/column names
    containing the word "hit" are retained so the rest of the pipeline stays
    unchanged; their H values now represent precision.

    Outputs:
      * cluster_hit_advantage_long.csv
          One row per actual cluster. This is the numerically convenient raw form.
      * cluster_hit_advantage_by_issue_threshold.csv
          Exactly one row per issue-threshold (nine rows per clustered issue), with
          aligned pipe-separated cluster sizes, hits, baselines, and advantages.
      * cluster_size_availability_by_threshold.csv
          Number of actual clusters of each size 1..11 at each threshold.
      * cluster_size_issue_support_by_threshold.csv
          Number of issues contributing at least one cluster of each size.
      * cluster_size_average_hit_advantage_by_threshold.csv
          Issue-macro average advantage. If an issue has multiple clusters of the
          same size (e.g. several singletons), those cluster advantages are first
          averaged within the issue, then issues are averaged equally.
      * cluster_size_average_hit_advantage_by_threshold.pdf
          Threshold curves for the issue-macro average advantage by cluster size.
      * agreement_average_advantage_by_issue_threshold.csv
          One row per issue and one column per threshold. A cell is the mean
          signed precision gain across all non-singleton clusters, but only when the
          same issue-threshold also contains at least one singleton; otherwise NA.
      * singleton_union_advantage_by_issue_threshold.csv
          One row per issue and one column per threshold. All singleton components
          are treated as one composite singleton union. Its H value is the
          percentage of singleton candidates that pass, and its random baseline is
          the expected precision of a random same-size subset.
      * singleton_union_hit_gain_by_issue_threshold.csv
          Transparent long-form singleton-union details: union size, actual hit,
          random baseline, and signed gain for each applicable issue-threshold.
      * agreement_minus_singleton_union_delta_by_issue_threshold.csv
          Paired difference A_agree(i,t) - A_singleton_union(i,t).
      * agreement_vs_singleton_union_issue_summary.csv
          Threshold-averaged agreement, singleton-union, and delta scores per issue.
      * agreement_vs_singleton_union_configuration_summary.csv
          One row per exact passing-candidate subgroup (1 pass, 2 passes, ...),
          with A_bar_agree, A_bar_disagreement, Delta_A, a 95% issue-bootstrap CI
          for Delta_A, Agreement better (%), and N.
      * agreemen_vs_disagreement_winpercentage.csv
          Compact pass-count-by-threshold table. Rows are exact positive passing
          counts (1 pass, 2 passes, 3 passes, ...), columns are similarity
          thresholds, and each cell is the percentage of comparable issues where
          agreement gain is strictly greater than disagreement gain at that threshold.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = [
        "Issue",
        "Threshold",
        "Cluster size",
        "Actual hit H",
        "Random baseline q",
        "Advantage A",
    ]

    if cluster_hit_advantage_df.empty:
        long_df = pd.DataFrame(columns=required_columns)
    else:
        missing = set(required_columns) - set(cluster_hit_advantage_df.columns)
        if missing:
            raise ValueError(
                "Cluster hit-advantage dataframe is missing columns: "
                f"{sorted(missing)}"
            )
        long_df = cluster_hit_advantage_df[required_columns].copy()
        long_df["Issue"] = long_df["Issue"].astype(str)
        long_df["Threshold"] = long_df["Threshold"].astype(float)
        long_df["Cluster size"] = long_df["Cluster size"].astype(int)
        for column in ["Actual hit H", "Random baseline q", "Advantage A"]:
            long_df[column] = long_df[column].astype(float)

    long_path = output_dir / "cluster_hit_advantage_long.csv"
    long_df.to_csv(long_path, index=False, na_rep="NA")

    # Requested compact form: exactly one row per issue and threshold. Because a
    # threshold can contain several clusters, the four cluster-specific fields are
    # aligned pipe-separated sequences (e.g. 8|2|1 and 100|0|0).
    compact_rows = []
    if not long_df.empty:
        for (issue, threshold), group in long_df.groupby(
            ["Issue", "Threshold"], sort=True
        ):
            group = group.copy()

            def format_number(value: float) -> str:
                value = float(value)
                if np.isclose(value, round(value), rtol=0.0, atol=1e-12):
                    return str(int(round(value)))
                return f"{value:.6f}".rstrip("0").rstrip(".")

            compact_rows.append({
                "Issue": str(issue),
                "Threshold": float(threshold),
                "Cluster size": "|".join(
                    str(int(value)) for value in group["Cluster size"].tolist()
                ),
                "Actual hit H": "|".join(
                    format_number(value) for value in group["Actual hit H"].tolist()
                ),
                "Random baseline q": "|".join(
                    format_number(value) for value in group["Random baseline q"].tolist()
                ),
                "Advantage A": "|".join(
                    format_number(value) for value in group["Advantage A"].tolist()
                ),
            })

    compact_df = pd.DataFrame(compact_rows, columns=required_columns)
    compact_path = output_dir / "cluster_hit_advantage_by_issue_threshold.csv"
    compact_df.to_csv(compact_path, index=False, na_rep="NA")

    threshold_columns = [f"{threshold:.1f}" for threshold in SIMILARITY_THRESHOLDS]
    cluster_sizes = list(range(1, len(TEMPERATURES) + 1))

    availability_rows = []
    issue_support_rows = []
    average_gain_rows = []

    if long_df.empty:
        issue_size_df = pd.DataFrame(
            columns=["Issue", "Threshold", "Cluster size", "Advantage A"]
        )
    else:
        # Prevent highly fragmented issues from receiving extra weight in the mean:
        # if the same issue has several size-k clusters, average them inside that
        # issue first, then macro-average across issues.
        issue_size_df = (
            long_df.groupby(
                ["Issue", "Threshold", "Cluster size"],
                as_index=False,
            )["Advantage A"]
            .mean()
        )

    for cluster_size in cluster_sizes:
        availability_row = {"cluster_size": int(cluster_size)}
        issue_support_row = {"cluster_size": int(cluster_size)}
        average_gain_row = {"cluster_size": int(cluster_size)}

        for threshold in SIMILARITY_THRESHOLDS:
            threshold_key = f"{threshold:.1f}"

            if long_df.empty:
                cluster_count = 0
            else:
                cluster_count = int((
                    (long_df["Cluster size"] == int(cluster_size))
                    & np.isclose(long_df["Threshold"], float(threshold))
                ).sum())
            availability_row[threshold_key] = cluster_count

            if issue_size_df.empty:
                issue_values = pd.Series(dtype=float)
            else:
                mask = (
                    (issue_size_df["Cluster size"] == int(cluster_size))
                    & np.isclose(issue_size_df["Threshold"], float(threshold))
                )
                issue_values = issue_size_df.loc[mask, "Advantage A"].astype(float)

            issue_support_row[threshold_key] = int(len(issue_values))
            average_gain_row[threshold_key] = (
                float(issue_values.mean()) if len(issue_values) > 0 else np.nan
            )

        availability_rows.append(availability_row)
        issue_support_rows.append(issue_support_row)
        average_gain_rows.append(average_gain_row)

    matrix_columns = ["cluster_size"] + threshold_columns
    availability_df = pd.DataFrame(availability_rows, columns=matrix_columns)
    issue_support_df = pd.DataFrame(issue_support_rows, columns=matrix_columns)
    average_gain_df = pd.DataFrame(average_gain_rows, columns=matrix_columns)

    availability_path = output_dir / "cluster_size_availability_by_threshold.csv"
    issue_support_path = output_dir / "cluster_size_issue_support_by_threshold.csv"
    average_gain_path = output_dir / "cluster_size_average_hit_advantage_by_threshold.csv"

    availability_df.to_csv(availability_path, index=False)
    issue_support_df.to_csv(issue_support_path, index=False)
    average_gain_df.to_csv(average_gain_path, index=False, na_rep="NA")

    # ------------------------------------------------------------------
    # RQ2 paired agreement-vs-singleton-UNION comparison.
    #
    # A comparison is defined for issue i at threshold t only when the
    # clustering contains BOTH:
    #   * at least one singleton cluster (k = 1), and
    #   * at least one non-singleton/agreement cluster (k >= 2).
    #
    # Non-singleton clusters remain individual agreement groups. Their signed
    # precision gains are averaged within the issue-threshold:
    #
    #   A_agree(i,t) = mean_C A_C, over all actual clusters with |C| >= 2.
    #
    # All singleton components are instead treated as ONE composite candidate
    # set. If s singletons exist, the singleton-union size is k_union=s.
    # H is the precision of that composite singleton group, while the random
    # baseline is the issue-level probability that one random valid candidate
    # is correct:
    #
    #   H_union(i,t) = 100 * (# passing singleton candidates / k_union)
    #   q_i = 100 * (m_i / n_i)
    #   A_union(i,t) = H_union(i,t) - q_i
    #
    # Finally:
    #
    #   DeltaA(i,t) = A_agree(i,t) - A_union(i,t).
    #
    # If either cluster type is absent, all paired matrices receive NA.
    # ------------------------------------------------------------------
    issue_ids = sorted(long_df["Issue"].astype(str).unique().tolist())
    matrix_output_columns = ["instance_id"] + threshold_columns

    agreement_matrix = pd.DataFrame(
        {"instance_id": issue_ids},
        columns=matrix_output_columns,
    )
    singleton_union_matrix = pd.DataFrame(
        {"instance_id": issue_ids},
        columns=matrix_output_columns,
    )
    delta_matrix = pd.DataFrame(
        {"instance_id": issue_ids},
        columns=matrix_output_columns,
    )

    # Ensure every threshold column exists and starts as NA.
    for threshold_key in threshold_columns:
        agreement_matrix[threshold_key] = np.nan
        singleton_union_matrix[threshold_key] = np.nan
        delta_matrix[threshold_key] = np.nan

    # Metadata lookup supplies the exact issue-level n_i and m_i used when the
    # clusters were evaluated. This avoids inferring pass counts from q values.
    metadata_lookup = {}
    if not threshold_summary_df.empty:
        required_summary_columns = {
            "instance_id",
            "threshold",
            "num_total_patches",
            "num_passes_for_issue",
        }
        missing_summary_columns = (
            required_summary_columns - set(threshold_summary_df.columns)
        )
        if missing_summary_columns:
            raise ValueError(
                "Threshold summary is missing columns required for singleton-union "
                f"gain: {sorted(missing_summary_columns)}"
            )

        for _, summary_row in threshold_summary_df.iterrows():
            metadata_key = (
                str(summary_row["instance_id"]),
                round(float(summary_row["threshold"]), 1),
            )
            if metadata_key in metadata_lookup:
                raise ValueError(
                    "Duplicate threshold-summary row for singleton-union gain: "
                    f"instance_id={metadata_key[0]}, threshold={metadata_key[1]:.1f}"
                )
            metadata_lookup[metadata_key] = (
                int(summary_row["num_total_patches"]),
                int(summary_row["num_passes_for_issue"]),
            )

    singleton_union_detail_rows = []

    if not long_df.empty:
        issue_to_row = {
            str(instance_id): row_index
            for row_index, instance_id in enumerate(agreement_matrix["instance_id"])
        }

        for (issue, threshold), group in long_df.groupby(
            ["Issue", "Threshold"],
            sort=True,
        ):
            issue = str(issue)
            threshold = float(threshold)
            threshold_key = f"{threshold:.1f}"
            if threshold_key not in threshold_columns:
                continue

            singleton_rows = group.loc[
                group["Cluster size"].astype(int) == 1
            ].copy()
            agreement_values = group.loc[
                group["Cluster size"].astype(int) >= 2,
                "Advantage A",
            ].astype(float)

            # The comparison is meaningful only when disagreement and agreement
            # structures coexist at the same issue-threshold.
            if singleton_rows.empty or agreement_values.empty:
                continue

            metadata_key = (issue, round(threshold, 1))
            if metadata_key not in metadata_lookup:
                raise KeyError(
                    "Missing n_i/m_i metadata for singleton-union gain: "
                    f"instance_id={issue}, threshold={threshold_key}"
                )

            num_valid_candidates, num_passing_candidates = metadata_lookup[metadata_key]

            # The actual connected components must partition the valid candidates.
            partition_size = int(group["Cluster size"].astype(int).sum())
            if partition_size != num_valid_candidates:
                raise AssertionError(
                    "Cluster sizes do not sum to num_total_patches for "
                    f"instance_id={issue}, threshold={threshold_key}: "
                    f"sum={partition_size}, n={num_valid_candidates}"
                )

            # Every singleton contributes exactly one candidate to the composite
            # singleton union. This also naturally handles an 8|2|1 pattern, where
            # the union contains one candidate and k_union=1.
            singleton_union_size = int(len(singleton_rows))
            # Each singleton has size 1, so its H value is either 0 or 100.
            # Their mean is therefore exactly the precision of the full singleton
            # union: 100 * (# passing singleton candidates / # singleton candidates).
            singleton_union_actual_hit = float(
                singleton_rows["Actual hit H"].astype(float).mean()
            )
            singleton_union_random_probability = (
                compute_random_precision_baseline(
                    num_valid_candidates=num_valid_candidates,
                    num_passing_candidates=num_passing_candidates,
                    cluster_size=singleton_union_size,
                )
            )
            singleton_union_random_baseline = float(
                100.0 * singleton_union_random_probability
            )
            # Signed gain: do not clip negative disagreement gains to zero.
            singleton_union_advantage = float(
                singleton_union_actual_hit - singleton_union_random_baseline
            )

            agreement_value = float(agreement_values.mean())
            delta_value = float(agreement_value - singleton_union_advantage)

            row_index = issue_to_row[issue]
            agreement_matrix.at[row_index, threshold_key] = agreement_value
            singleton_union_matrix.at[row_index, threshold_key] = (
                singleton_union_advantage
            )
            delta_matrix.at[row_index, threshold_key] = delta_value

            singleton_union_detail_rows.append({
                "Issue": issue,
                "Threshold": threshold,
                "Singleton union size": singleton_union_size,
                "Actual hit H": singleton_union_actual_hit,
                "Random baseline q": singleton_union_random_baseline,
                "Advantage A": singleton_union_advantage,
            })

    agreement_matrix_path = (
        output_dir / "agreement_average_advantage_by_issue_threshold.csv"
    )
    singleton_union_matrix_path = (
        output_dir / "singleton_union_advantage_by_issue_threshold.csv"
    )
    delta_matrix_path = (
        output_dir / "agreement_minus_singleton_union_delta_by_issue_threshold.csv"
    )
    singleton_union_detail_path = (
        output_dir / "singleton_union_hit_gain_by_issue_threshold.csv"
    )

    agreement_matrix.to_csv(agreement_matrix_path, index=False, na_rep="NA")
    singleton_union_matrix.to_csv(
        singleton_union_matrix_path,
        index=False,
        na_rep="NA",
    )
    delta_matrix.to_csv(delta_matrix_path, index=False, na_rep="NA")
    pd.DataFrame(
        singleton_union_detail_rows,
        columns=[
            "Issue",
            "Threshold",
            "Singleton union size",
            "Actual hit H",
            "Random baseline q",
            "Advantage A",
        ],
    ).to_csv(singleton_union_detail_path, index=False, na_rep="NA")

    # Remove stale files from the previous per-singleton-average implementation so
    # a rerun cannot leave misleading artifacts beside the singleton-union outputs.
    obsolete_paired_paths = [
        output_dir / "singleton_average_advantage_by_issue_threshold.csv",
        output_dir / "agreement_minus_singleton_delta_by_issue_threshold.csv",
        output_dir / "agreement_vs_singleton_issue_summary.csv",
        output_dir / "agreement_vs_singleton_configuration_summary.csv",
    ]
    for obsolete_path in obsolete_paired_paths:
        if obsolete_path.exists():
            obsolete_path.unlink()

    # Cross-check that the applicability mask is identical in all three matrices.
    agreement_mask = agreement_matrix[threshold_columns].notna()
    singleton_union_mask = singleton_union_matrix[threshold_columns].notna()
    delta_mask = delta_matrix[threshold_columns].notna()
    if (
        not agreement_mask.equals(singleton_union_mask)
        or not agreement_mask.equals(delta_mask)
    ):
        raise AssertionError(
            "Agreement, singleton-union, and delta matrices must have identical NA masks."
        )

    # Build one stable metadata record per issue. num_passes_for_issue is computed
    # after all candidate-validity filters, so the pass-count subgroups below are
    # based on the exact candidates that entered clustering.
    issue_metadata_lookup = {}
    if not threshold_summary_df.empty:
        for metadata_issue, metadata_issue_df in threshold_summary_df.groupby(
            "instance_id", sort=False
        ):
            n_values = sorted(
                set(metadata_issue_df["num_total_patches"].astype(int).tolist())
            )
            m_values = sorted(
                set(metadata_issue_df["num_passes_for_issue"].astype(int).tolist())
            )
            if len(n_values) != 1 or len(m_values) != 1:
                raise AssertionError(
                    "Issue-level n_i/m_i must be constant across thresholds for "
                    f"instance_id={metadata_issue}: n={n_values}, m={m_values}"
                )
            issue_metadata_lookup[str(metadata_issue)] = (n_values[0], m_values[0])

    # First average over all applicable thresholds WITHIN each issue. This makes
    # the software issue, rather than an issue-threshold or individual cluster,
    # the final independent observational unit.
    issue_summary_rows = []
    for row_index, instance_id in enumerate(issue_ids):
        agree_values = pd.to_numeric(
            agreement_matrix.loc[row_index, threshold_columns],
            errors="coerce",
        )
        singleton_union_values = pd.to_numeric(
            singleton_union_matrix.loc[row_index, threshold_columns],
            errors="coerce",
        )
        delta_values = pd.to_numeric(
            delta_matrix.loc[row_index, threshold_columns],
            errors="coerce",
        )

        applicable_mask = delta_values.notna()
        applicable_threshold_count = int(applicable_mask.sum())

        if applicable_threshold_count == 0:
            issue_agree = np.nan
            issue_singleton_union = np.nan
            issue_delta = np.nan
            agreement_better = np.nan
        else:
            issue_agree = float(agree_values[applicable_mask].mean())
            issue_singleton_union = float(
                singleton_union_values[applicable_mask].mean()
            )
            issue_delta = float(delta_values[applicable_mask].mean())

            # With identical threshold support, mean(delta) must equal
            # mean(agreement) - mean(singleton union).
            expected_issue_delta = float(issue_agree - issue_singleton_union)
            if not np.isclose(
                issue_delta,
                expected_issue_delta,
                rtol=0.0,
                atol=1e-10,
            ):
                raise AssertionError(
                    "Issue-level paired delta identity failed for "
                    f"instance_id={instance_id}: direct={issue_delta}, "
                    f"agree-singleton_union={expected_issue_delta}"
                )

            # Strictly positive means average agreement gain is higher than the
            # singleton-union gain. Zero is a tie.
            agreement_better = int(issue_delta > 1e-12)

        metadata_key = str(instance_id)
        if metadata_key not in issue_metadata_lookup:
            raise KeyError(
                "Missing issue metadata while building pass-count RQ2 summary: "
                f"instance_id={metadata_key}"
            )
        issue_n, issue_m = issue_metadata_lookup[metadata_key]

        issue_summary_rows.append({
            "instance_id": str(instance_id),
            "num_valid_candidates": int(issue_n),
            "num_passing_candidates": int(issue_m),
            "solvable_pass_subset": f"{int(issue_m)}_passing",
            "applicable_threshold_count": applicable_threshold_count,
            "A_bar_agree_issue": issue_agree,
            "A_bar_singleton_union_issue": issue_singleton_union,
            "A_bar_disagreement_issue": issue_singleton_union,
            "Delta_A_issue": issue_delta,
            "agreement_better": agreement_better,
        })

    issue_summary_df = pd.DataFrame(issue_summary_rows)
    issue_summary_path = (
        output_dir / "agreement_vs_singleton_union_issue_summary.csv"
    )
    issue_summary_df.to_csv(issue_summary_path, index=False, na_rep="NA")

    # ------------------------------------------------------------------
    # Configuration summary stratified by the EXACT number of passing candidates.
    #
    # For the solvable subset this produces one row per exact m_i group:
    # solvable_1_passing, solvable_2_passing, solvable_3_passing, ... .
    # No positive-only filtering is applied: negative A and negative Delta_A values
    # are retained. Within each subgroup, every issue is first averaged across its
    # applicable thresholds, then issues are macro-averaged. This preserves the
    # issue as the independent observational unit.
    # ------------------------------------------------------------------
    subset_name_for_summary = output_dir.parent.name
    pass_counts = sorted(
        issue_summary_df["num_passing_candidates"].dropna().astype(int).unique().tolist()
    )

    def summarize_configuration_group(
        subgroup_df: pd.DataFrame,
        pass_count: int,
        seed_offset: int,
    ) -> dict:
        comparable_df = subgroup_df[subgroup_df["Delta_A_issue"].notna()].copy()
        comparable_count = int(len(comparable_df))
        total_group_issues = int(len(subgroup_df))

        if comparable_count == 0:
            group_A_bar_agree = np.nan
            group_A_bar_disagreement = np.nan
            group_Delta_A = np.nan
            group_ci_lower = np.nan
            group_ci_upper = np.nan
            group_agreement_better_percent = np.nan
        else:
            group_A_bar_agree = float(comparable_df["A_bar_agree_issue"].mean())
            group_A_bar_disagreement = float(
                comparable_df["A_bar_singleton_union_issue"].mean()
            )
            group_Delta_A = float(comparable_df["Delta_A_issue"].mean())

            expected_group_delta = float(
                group_A_bar_agree - group_A_bar_disagreement
            )
            if not np.isclose(
                group_Delta_A,
                expected_group_delta,
                rtol=0.0,
                atol=1e-10,
            ):
                raise AssertionError(
                    "Pass-count configuration delta identity failed for "
                    f"m={pass_count}: direct={group_Delta_A}, "
                    f"agree-disagreement={expected_group_delta}"
                )

            group_agreement_better_percent = float(
                100.0 * comparable_df["agreement_better"].astype(float).mean()
            )

            # Bootstrap the mean paired Delta_A at the ISSUE level inside this
            # exact-m subgroup.
            if comparable_count < 2:
                group_ci_lower = np.nan
                group_ci_upper = np.nan
            else:
                delta_issue_values = comparable_df["Delta_A_issue"].to_numpy(
                    dtype=float
                )
                bootstrap_rng = np.random.default_rng(20260819 + seed_offset)
                bootstrap_repetitions = 10000
                bootstrap_means = np.empty(bootstrap_repetitions, dtype=float)
                for bootstrap_index in range(bootstrap_repetitions):
                    sampled_indices = bootstrap_rng.integers(
                        0,
                        comparable_count,
                        size=comparable_count,
                    )
                    bootstrap_means[bootstrap_index] = float(
                        delta_issue_values[sampled_indices].mean()
                    )
                group_ci_lower, group_ci_upper = np.percentile(
                    bootstrap_means,
                    [2.5, 97.5],
                )
                group_ci_lower = float(group_ci_lower)
                group_ci_upper = float(group_ci_upper)

        if pd.isna(group_ci_lower) or pd.isna(group_ci_upper):
            group_ci_text = "NA"
        else:
            group_ci_text = f"[{group_ci_lower:.4f}, {group_ci_upper:.4f}]"

        subgroup_label = f"{subset_name_for_summary}_{int(pass_count)}_passing"
        return {
            "Dataset": SET,
            "Model": MODEL,
            "Similarity": metric,
            "Subset": subgroup_label,
            "Passing candidates": int(pass_count),
            "Total issues in subgroup": total_group_issues,
            "A_bar_agree": group_A_bar_agree,
            "A_bar_disagreement": group_A_bar_disagreement,
            # Backward-compatible alias for older aggregation scripts.
            "A_bar_singleton_union": group_A_bar_disagreement,
            "Delta_A": group_Delta_A,
            "95% CI": group_ci_text,
            "Agreement better": group_agreement_better_percent,
            # N remains the number of comparable issues (both agreement and
            # disagreement structures present at >=1 threshold).
            "N": comparable_count,
        }

    comparison_summary_rows = []
    for pass_count_index, pass_count in enumerate(pass_counts):
        pass_group_df = issue_summary_df[
            issue_summary_df["num_passing_candidates"].astype(int) == int(pass_count)
        ].copy()
        comparison_summary_rows.append(
            summarize_configuration_group(
                subgroup_df=pass_group_df,
                pass_count=int(pass_count),
                seed_offset=1000 * (pass_count_index + 1) + int(pass_count),
            )
        )

    comparison_summary_columns = [
        "Dataset",
        "Model",
        "Similarity",
        "Subset",
        "Passing candidates",
        "Total issues in subgroup",
        "A_bar_agree",
        "A_bar_disagreement",
        "A_bar_singleton_union",
        "Delta_A",
        "95% CI",
        "Agreement better",
        "N",
    ]
    comparison_summary_df = pd.DataFrame(
        comparison_summary_rows,
        columns=comparison_summary_columns,
    )
    comparison_summary_path = (
        output_dir / "agreement_vs_singleton_union_configuration_summary.csv"
    )
    comparison_summary_df.to_csv(
        comparison_summary_path,
        index=False,
        na_rep="NA",
    )

    # ------------------------------------------------------------------
    # Additional RQ2 threshold-specific configuration summary.
    #
    # Unlike agreement_vs_singleton_union_configuration_summary.csv, which first
    # averages applicable thresholds within each issue and then averages issues
    # inside each exact pass-count subgroup, this table keeps thresholds separate.
    # Each row is one pass-count subgroup at one threshold. N is the number of
    # issues in that subgroup for which BOTH agreement and disagreement structures
    # existed at the threshold.
    #
    # The requested file name deliberately preserves the spelling "threashold".
    # ------------------------------------------------------------------
    threshold_comparison_rows = []
    matrix_pass_counts = agreement_matrix["instance_id"].astype(str).map(
        lambda issue: issue_metadata_lookup[str(issue)][1]
    ).astype(int)

    for pass_count_index, pass_count in enumerate(pass_counts):
        pass_count_mask = matrix_pass_counts == int(pass_count)
        pass_group_total_issues = int(pass_count_mask.sum())

        for threshold_index, threshold in enumerate(SIMILARITY_THRESHOLDS):
            threshold_key = f"{threshold:.1f}"

            threshold_agree = pd.to_numeric(
                agreement_matrix[threshold_key],
                errors="coerce",
            )
            threshold_singleton_union = pd.to_numeric(
                singleton_union_matrix[threshold_key],
                errors="coerce",
            )
            threshold_delta = pd.to_numeric(
                delta_matrix[threshold_key],
                errors="coerce",
            )

            comparable_mask = threshold_delta.notna() & pass_count_mask
            threshold_n = int(comparable_mask.sum())

            if threshold_n == 0:
                threshold_A_bar_agree = np.nan
                threshold_A_bar_disagreement = np.nan
                threshold_Delta_A = np.nan
                threshold_ci_lower = np.nan
                threshold_ci_upper = np.nan
                threshold_agreement_better_percent = np.nan
            else:
                agree_values_at_threshold = threshold_agree[comparable_mask].astype(float)
                disagreement_values_at_threshold = threshold_singleton_union[
                    comparable_mask
                ].astype(float)
                delta_values_at_threshold = threshold_delta[comparable_mask].astype(float)

                threshold_A_bar_agree = float(agree_values_at_threshold.mean())
                threshold_A_bar_disagreement = float(
                    disagreement_values_at_threshold.mean()
                )
                threshold_Delta_A = float(delta_values_at_threshold.mean())

                expected_threshold_delta = float(
                    threshold_A_bar_agree - threshold_A_bar_disagreement
                )
                if not np.isclose(
                    threshold_Delta_A,
                    expected_threshold_delta,
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise AssertionError(
                        "Threshold/pass-count paired delta identity failed for "
                        f"m={pass_count}, threshold={threshold_key}: "
                        f"direct={threshold_Delta_A}, "
                        f"agree-disagreement={expected_threshold_delta}"
                    )

                threshold_agreement_better_percent = float(
                    100.0 * (delta_values_at_threshold > 1e-12).mean()
                )

                if threshold_n < 2:
                    threshold_ci_lower = np.nan
                    threshold_ci_upper = np.nan
                else:
                    threshold_bootstrap_rng = np.random.default_rng(
                        20260819
                        + 10000 * (pass_count_index + 1)
                        + threshold_index
                    )
                    threshold_bootstrap_repetitions = 10000
                    threshold_bootstrap_means = np.empty(
                        threshold_bootstrap_repetitions,
                        dtype=float,
                    )
                    threshold_delta_array = delta_values_at_threshold.to_numpy(
                        dtype=float
                    )
                    for bootstrap_index in range(threshold_bootstrap_repetitions):
                        sampled_indices = threshold_bootstrap_rng.integers(
                            0,
                            threshold_n,
                            size=threshold_n,
                        )
                        threshold_bootstrap_means[bootstrap_index] = float(
                            threshold_delta_array[sampled_indices].mean()
                        )
                    threshold_ci_lower, threshold_ci_upper = np.percentile(
                        threshold_bootstrap_means,
                        [2.5, 97.5],
                    )
                    threshold_ci_lower = float(threshold_ci_lower)
                    threshold_ci_upper = float(threshold_ci_upper)

            if pd.isna(threshold_ci_lower) or pd.isna(threshold_ci_upper):
                threshold_ci_text = "NA"
            else:
                threshold_ci_text = (
                    f"[{threshold_ci_lower:.4f}, {threshold_ci_upper:.4f}]"
                )

            threshold_comparison_rows.append({
                "Dataset": SET,
                "Model": MODEL,
                "Similarity": metric,
                "Subset": f"{subset_name_for_summary}_{int(pass_count)}_passing",
                "Passing candidates": int(pass_count),
                "Total issues in subgroup": pass_group_total_issues,
                "Threshold": float(threshold),
                "A_bar_agree": threshold_A_bar_agree,
                "A_bar_disagreement": threshold_A_bar_disagreement,
                "A_bar_singleton_union": threshold_A_bar_disagreement,
                "Delta_A": threshold_Delta_A,
                "95% CI": threshold_ci_text,
                "Agreement better": threshold_agreement_better_percent,
                "N": threshold_n,
            })

    threshold_comparison_summary_columns = [
        "Dataset",
        "Model",
        "Similarity",
        "Subset",
        "Passing candidates",
        "Total issues in subgroup",
        "Threshold",
        "A_bar_agree",
        "A_bar_disagreement",
        "A_bar_singleton_union",
        "Delta_A",
        "95% CI",
        "Agreement better",
        "N",
    ]
    threshold_comparison_summary_df = pd.DataFrame(
        threshold_comparison_rows,
        columns=threshold_comparison_summary_columns,
    )
    threshold_comparison_summary_path = (
        output_dir
        / "agreement_vs_singleton_union_configuration_summary_threashold.csv"
    )
    threshold_comparison_summary_df.to_csv(
        threshold_comparison_summary_path,
        index=False,
        na_rep="NA",
    )

    # ------------------------------------------------------------------
    # Compact RQ2 agreement-win percentage table requested for presentation.
    #
    # Rows: exact positive number of passing candidates for the issue
    #       (1 pass, 2 passes, 3 passes, ...).
    # Columns: similarity thresholds 0.1 ... 0.9.
    # Cell: percentage of COMPARABLE issues in that pass-count subgroup at that
    #       threshold for which Delta_A = A_agreement - A_disagreement > 0.
    #
    # Ties (Delta_A == 0) are not agreement wins, matching the existing
    # "Agreement better" definition. If no issue in a subgroup is comparable at
    # a threshold, the cell is NA.
    # ------------------------------------------------------------------
    win_percentage_rows = []
    positive_pass_counts = [int(value) for value in pass_counts if int(value) > 0]

    for pass_count in positive_pass_counts:
        win_row = {"Passing candidates": int(pass_count)}

        for threshold in SIMILARITY_THRESHOLDS:
            threshold_key = f"{threshold:.1f}"
            matching = threshold_comparison_summary_df[
                (threshold_comparison_summary_df["Passing candidates"].astype(int)
                 == int(pass_count))
                & np.isclose(
                    threshold_comparison_summary_df["Threshold"].astype(float),
                    float(threshold),
                )
            ]

            if matching.empty:
                win_row[threshold_key] = np.nan
            else:
                if len(matching) != 1:
                    raise ValueError(
                        "Expected one agreement-win summary row for "
                        f"pass_count={pass_count}, threshold={threshold_key}; "
                        f"found {len(matching)}"
                    )
                win_row[threshold_key] = float(
                    matching.iloc[0]["Agreement better"]
                ) if not pd.isna(matching.iloc[0]["Agreement better"]) else np.nan

        win_percentage_rows.append(win_row)

    win_percentage_columns = ["Passing candidates"] + threshold_columns
    win_percentage_df = pd.DataFrame(
        win_percentage_rows,
        columns=win_percentage_columns,
    )
    win_percentage_path = (
        output_dir / "agreemen_vs_disagreement_winpercentage.csv"
    )
    win_percentage_df.to_csv(
        win_percentage_path,
        index=False,
        na_rep="NA",
    )

    plot_path = output_dir / "cluster_size_average_hit_advantage_by_threshold.pdf"
    plt.figure(figsize=(10.0, 6.0))
    plotted_any = False
    for _, row in average_gain_df.iterrows():
        cluster_size = int(row["cluster_size"])
        values = pd.to_numeric(row[threshold_columns], errors="coerce").astype(float)
        if values.notna().any():
            plotted_any = True
            plt.plot(
                SIMILARITY_THRESHOLDS,
                values.to_numpy(),
                marker="o",
                label=f"cluster size {cluster_size}",
            )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Average correctness gain A (percentage points)")
    plt.xticks(SIMILARITY_THRESHOLDS)

    # Signed gains can be negative or positive. Use a symmetric y-axis around
    # zero so depletion below the random baseline is visible as clearly as
    # enrichment above it.
    plotted_values = pd.to_numeric(
        average_gain_df[threshold_columns].stack(),
        errors="coerce",
    ).dropna()
    if not plotted_values.empty:
        max_abs_gain = float(np.abs(plotted_values.to_numpy(dtype=float)).max())
        padded_limit = 1.15 * max_abs_gain
        y_limit = max(5.0, 5.0 * math.ceil(padded_limit / 5.0))
        plt.ylim(-y_limit, y_limit)

    plt.axhline(0.0, linewidth=1.0)
    plt.grid(True)
    if plotted_any:
        plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "cluster_level_long": long_path,
        "issue_threshold_compact": compact_path,
        "availability": availability_path,
        "issue_support": issue_support_path,
        "average_gain": average_gain_path,
        "agreement_average_by_issue_threshold": agreement_matrix_path,
        "singleton_union_by_issue_threshold": singleton_union_matrix_path,
        "singleton_union_detail": singleton_union_detail_path,
        "agreement_minus_singleton_union_delta_by_issue_threshold": delta_matrix_path,
        "agreement_vs_singleton_union_issue_summary": issue_summary_path,
        "agreement_vs_singleton_union_configuration_summary": comparison_summary_path,
        "agreement_vs_singleton_union_configuration_summary_threashold": (
            threshold_comparison_summary_path
        ),
        "agreemen_vs_disagreement_winpercentage": win_percentage_path,
        "average_gain_plot": plot_path,
    }



def save_rq3_main_results_at_rq1_threshold(results_dir: Path) -> pd.DataFrame:
    """Save RQ3 positive-gain ratios at the common fixed threshold 0.9.

    RQ3 is intentionally decoupled from the configuration-specific RQ1 threshold.
    It is evaluated on solvable issues at ``PAPER_FIXED_THRESHOLD``, separately
    for issues with exactly N correct fixes. For each candidate group and subgroup,
    the reported ratio is

        (# issues where the group is available and its gain over randomness is > 0)
        / (# issues where the group is available).

    ``rq3_all_results.csv`` contains majority agreement, minority agreement, and
    disagreement. ``rq3_main_results.csv`` contains only the majority-agreement row.
    If the group is unavailable for every issue in a subgroup, the ratio is NA.

    The function name is retained for backward compatibility with the existing
    pipeline call site, but RQ1's selected t* is not read or used here.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    subgroups_root = results_dir / "subgroups"
    all_results_path = results_dir / "rq3_all_results.csv"
    main_results_path = results_dir / "rq3_main_results.csv"
    main_plot_path = results_dir / "rq3_main_results.png"

    if not subgroups_root.exists():
        raise FileNotFoundError(
            f"Cannot build RQ3 results; subgroup directory is missing: {subgroups_root}"
        )

    fixed_threshold = float(PAPER_FIXED_THRESHOLD)
    threshold_key = f"{fixed_threshold:.1f}"

    correct_fix_counts = list(range(1, len(TEMPERATURES) + 1))
    subgroup_columns = [str(pass_count) for pass_count in correct_fix_counts]
    all_output_columns = [
        "Dataset",
        "Model",
        "Similarity",
        "T",
        "Group",
    ] + subgroup_columns

    group_specs = [
        (
            "majority agreement",
            "majority_agreement_gain_by_issue_threshold.csv",
        ),
        (
            "minority agreement",
            "minority_agreement_gain_by_issue_threshold.csv",
        ),
        (
            "disagreement",
            "disagreement_gain_by_issue_threshold.csv",
        ),
    ]

    result_rows = []
    for group_label, gain_filename in group_specs:
        result_row = {
            "Dataset": SET,
            "Model": MODEL,
            "Similarity": metric,
            "T": fixed_threshold,
            "Group": group_label,
        }
        for pass_count in correct_fix_counts:
            result_row[str(pass_count)] = np.nan

        for pass_count in correct_fix_counts:
            subgroup_path = (
                subgroups_root
                / f"{pass_count}CORRECT"
                / gain_filename
            )
            if not subgroup_path.exists():
                raise FileNotFoundError(
                    "Cannot build RQ3 results; required subgroup gain file "
                    f"is missing: {subgroup_path}"
                )

            subgroup_df = pd.read_csv(subgroup_path)
            if threshold_key not in subgroup_df.columns:
                raise KeyError(
                    f"Fixed RQ3 threshold {threshold_key} is missing from "
                    f"{subgroup_path}"
                )

            gain_values = pd.to_numeric(
                subgroup_df[threshold_key], errors="coerce"
            ).dropna().astype(float)

            if gain_values.empty:
                positive_gain_ratio = np.nan
            else:
                positive_gain_ratio = float(
                    (gain_values > 1e-12).sum() / len(gain_values)
                )

            result_row[str(pass_count)] = positive_gain_ratio

        result_rows.append(result_row)

    all_results_df = pd.DataFrame(result_rows, columns=all_output_columns)
    all_results_df.to_csv(all_results_path, index=False, na_rep="NA")

    majority_df = all_results_df[
        all_results_df["Group"] == "majority agreement"
    ].copy()
    if len(majority_df) != 1:
        raise AssertionError(
            "RQ3 main results require exactly one majority-agreement row; "
            f"found {len(majority_df)}"
        )

    main_output_columns = [
        "Dataset",
        "Model",
        "Similarity",
        "T",
    ] + subgroup_columns
    main_results_df = majority_df.drop(columns=["Group"])[main_output_columns]
    main_results_df.to_csv(main_results_path, index=False, na_rep="NA")

    # Plot only the majority-agreement main result. The x-axis subgroups are issues
    # with exactly N correct fixes; the y-axis is the positive-gain ratio.
    x_values = correct_fix_counts
    y_values = pd.to_numeric(
        main_results_df.iloc[0][subgroup_columns], errors="coerce"
    ).to_numpy(dtype=float)

    plt.figure(figsize=(9.0, 5.5))
    finite_mask = np.isfinite(y_values)
    if finite_mask.any():
        x_array = np.asarray(x_values, dtype=float)
        plt.plot(
            x_array[finite_mask],
            y_values[finite_mask],
            marker="o",
        )
    plt.xlabel("Number of correct fixes in issue")
    plt.ylabel("Ratio of issues with positive gain over randomness")
    plt.xticks(x_values)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(main_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print()
    print("=" * 80)
    print(f"RQ3 RESULTS AT FIXED T={fixed_threshold:.1f}")
    print("=" * 80)
    print(all_results_df.to_string(index=False))
    print(f"RQ3 all-results CSV: {all_results_path}")
    print(f"RQ3 main-results CSV: {main_results_path}")
    print(f"RQ3 main-results plot: {main_plot_path}")
    print("=" * 80)

    return main_results_df

def save_rq3_correct_fix_subgroup_scenarios(
    cluster_hit_advantage_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    solvable_output_dir: Path,
    results_dir: Path,
    agreement_definition: str,
) -> dict:
    """Save the revised RQ3 majority/minority/disagreement analysis.

    RQ3 asks: does agreement indicate correctness?

    For issue i, let n_i be the number of valid candidate fixes and m_i the number
    of correct/passing valid fixes. The random candidate-level correctness baseline is

        q_i = m_i / n_i.

    Every scenario uses signed correctness gain

        A = 100 * (precision - q_i),

    in percentage points. Negative gains are retained.

    A threshold is considered structurally valid for this RQ3 analysis only when it
    produces a real separation of the valid candidates:

        1 < number_of_components < n_i.

    Thus a single full-size component (for example 11) is discarded, and an
    all-singleton partition (for example 1|1|...|1) is also discarded.

    On a valid issue-threshold case, the three RQ3 structures are:

      * Agreement cluster: under ``agreement_definition="majority"``, the selected
        largest non-singleton component is available only when its size is strictly
        greater than n_i/2 (for n_i=11 the minimum is 6; for n_i=10 the minimum is 6).
        Under ``agreement_definition="largest"``, the selected largest non-singleton
        component is available on every structurally valid threshold regardless of
        whether it contains more than half of the candidates.

      * Minority agreement: the smallest non-singleton component that is distinct
        from the selected agreement component when that agreement component is
        available. Under the strict-majority definition only, if no majority exists
        and there is exactly one non-singleton component, that component retains the
        existing minority-agreement interpretation. Existing size/density tie-breaking
        is reused where multiple non-singleton components are available.

      * Disagreement: the union of all singleton/outlier components. One singleton is
        sufficient for disagreement to be available.

    Five issue-by-threshold matrices are saved inside every exact-correct-fix subgroup:

      1. majority_agreement_gain_by_issue_threshold.csv
      2. minority_agreement_gain_by_issue_threshold.csv
      3. disagreement_gain_by_issue_threshold.csv
      4. delta_majority_agreement_minus_disagreement_by_issue_threshold.csv
      5. delta_minority_agreement_minus_disagreement_by_issue_threshold.csv

    A delta is defined only when both of its two structures exist in the same valid
    issue-threshold case.

    Directly under the definition-specific Results/<definition>/subgroups/ directory,
    one summary CSV is also saved. Its rows are
    the 1CORRECT..11CORRECT subgroups. For majority agreement, minority agreement, and
    disagreement it reports: number of available issue-threshold cases, number of
    those cases containing at least one correct fix, and the corresponding percentage.
    A case is an issue at one threshold, not a unique issue across all thresholds.

    Every subgroup keeps delta_positive_percentage_by_threshold.csv as a secondary
    agreement-vs-disagreement comparison. It also receives
    positive_gain_percentage_by_threshold.csv. For each threshold, the latter reports
    the percentage of AVAILABLE cases whose own signed gain is strictly positive for:

        majority agreement
        minority agreement
        disagreement

    Each percentage uses that structure's own availability as its denominator. Thus,
    for example, majority positive-gain percentage is

        100 * (# issues with majority gain > 0) / (# issues where majority exists).

    Finally, every definition-specific subgroups/RQ3plots/<threshold>/ folder receives one plot
    with these three positive-gain percentages as separate lines across correct-fix
    counts 1..11. Delta is not used in this plot. No minimum-support (>10) rule is
    applied to these descriptive RQ3 subgroup outputs.
    """
    if agreement_definition not in {"majority", "largest"}:
        raise ValueError(
            "agreement_definition must be either 'majority' or 'largest', got "
            f"{agreement_definition!r}"
        )

    agreement_display_name = (
        "Majority agreement"
        if agreement_definition == "majority"
        else "Largest-cluster agreement"
    )

    solvable_output_dir = Path(solvable_output_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    subgroups_root = results_dir / "subgroups"
    plots_dir = subgroups_root / "RQ3plots"
    subgroups_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    threshold_columns = [
        f"{threshold:.1f}" for threshold in SIMILARITY_THRESHOLDS
    ]
    correct_fix_counts = list(range(1, len(TEMPERATURES) + 1))

    required_long_columns = {
        "Issue",
        "Threshold",
        "Cluster size",
        "Actual hit H",
        "Random baseline q",
        "Advantage A",
    }
    missing_long_columns = required_long_columns - set(cluster_hit_advantage_df.columns)
    if missing_long_columns:
        raise ValueError(
            "Cannot build revised RQ3 subgroup analysis because cluster-level columns "
            f"are missing: {sorted(missing_long_columns)}"
        )

    required_summary_columns = {
        "instance_id",
        "threshold",
        "num_total_patches",
        "num_passes_for_issue",
        "num_clusters",
        "singleton_cluster_count",
        "non_singleton_cluster_count",
        "largest_cluster_size",
        "largest_cluster_tp",
        "largest_cluster_precision",
        "smallest_cluster_size",
        "smallest_cluster_tp",
        "smallest_cluster_precision",
    }
    missing_summary_columns = required_summary_columns - set(threshold_summary_df.columns)
    if missing_summary_columns:
        raise ValueError(
            "Cannot build revised RQ3 subgroup analysis because threshold-summary "
            f"columns are missing: {sorted(missing_summary_columns)}"
        )

    long_df = cluster_hit_advantage_df.copy()
    long_df["Issue"] = long_df["Issue"].astype(str)
    long_df["Threshold"] = long_df["Threshold"].astype(float)
    long_df["threshold_key"] = long_df["Threshold"].map(lambda value: f"{value:.1f}")
    long_df["Cluster size"] = long_df["Cluster size"].astype(int)
    for column in ["Actual hit H", "Random baseline q", "Advantage A"]:
        long_df[column] = pd.to_numeric(long_df[column], errors="raise").astype(float)

    summary_df = threshold_summary_df.copy()
    summary_df["instance_id"] = summary_df["instance_id"].astype(str)
    summary_df["threshold"] = summary_df["threshold"].astype(float)
    summary_df["threshold_key"] = summary_df["threshold"].map(
        lambda value: f"{value:.1f}"
    )

    # Build one stable n_i/m_i record per issue and verify that filtering metadata is
    # constant across thresholds.
    issue_metadata = {}
    for instance_id, issue_rows in summary_df.groupby("instance_id", sort=True):
        n_values = issue_rows["num_total_patches"].dropna().astype(int).unique().tolist()
        m_values = issue_rows["num_passes_for_issue"].dropna().astype(int).unique().tolist()
        if len(n_values) != 1 or len(m_values) != 1:
            raise ValueError(
                "RQ3 subgroup metadata must be constant across thresholds for "
                f"instance_id={instance_id}; n={n_values}, m={m_values}"
            )
        n_i = int(n_values[0])
        m_i = int(m_values[0])
        if n_i <= 0 or m_i < 0 or m_i > n_i:
            raise ValueError(
                "Invalid RQ3 subgroup metadata for "
                f"instance_id={instance_id}: n={n_i}, m={m_i}"
            )
        issue_metadata[str(instance_id)] = (n_i, m_i)

    issue_ids = sorted(issue_metadata.keys())
    matrix_columns = ["instance_id"] + threshold_columns

    def empty_issue_threshold_matrix() -> pd.DataFrame:
        matrix = pd.DataFrame({"instance_id": issue_ids})
        for threshold_key in threshold_columns:
            matrix[threshold_key] = np.nan
        return matrix[matrix_columns]

    scenario_matrices = {
        "majority_agreement": empty_issue_threshold_matrix(),
        "minority_agreement": empty_issue_threshold_matrix(),
        "disagreement": empty_issue_threshold_matrix(),
        "delta_majority": empty_issue_threshold_matrix(),
        "delta_minority": empty_issue_threshold_matrix(),
    }
    issue_to_row = {
        str(instance_id): row_index
        for row_index, instance_id in enumerate(issue_ids)
    }

    # Cluster-level rows are needed for the singleton union, because one singleton is
    # a valid disagreement group even though the generic singleton_union strategy
    # elsewhere in the pipeline requires at least two singletons.
    long_group_lookup = {
        (str(issue), str(threshold_key)): group.copy()
        for (issue, threshold_key), group in long_df.groupby(
            ["Issue", "threshold_key"], sort=True
        )
    }

    case_rows = []

    for _, summary_row in summary_df.iterrows():
        issue = str(summary_row["instance_id"])
        threshold = float(summary_row["threshold"])
        threshold_key = f"{threshold:.1f}"
        if issue not in issue_to_row or threshold_key not in threshold_columns:
            continue

        n_i, m_i = issue_metadata[issue]
        q_percent = 100.0 * (m_i / n_i)
        num_clusters = int(summary_row["num_clusters"])
        non_singleton_count = int(summary_row["non_singleton_cluster_count"])

        # A valid RQ3 threshold must separate the valid candidates, but cannot fragment
        # them completely into singletons.
        valid_threshold = bool(1 < num_clusters < n_i)

        majority_available = False
        majority_has_correct = False
        majority_gain = np.nan

        minority_available = False
        minority_has_correct = False
        minority_gain = np.nan

        disagreement_available = False
        disagreement_has_correct = False
        disagreement_gain = np.nan

        cluster_group = long_group_lookup.get((issue, threshold_key))
        if cluster_group is not None and not cluster_group.empty:
            observed_q = cluster_group["Random baseline q"].astype(float).to_numpy()
            if observed_q.size > 0 and not np.allclose(
                observed_q,
                q_percent,
                rtol=0.0,
                atol=1e-6,
            ):
                raise AssertionError(
                    "RQ3 subgroup random baseline does not equal m/n for "
                    f"instance_id={issue}, threshold={threshold_key}: "
                    f"expected={q_percent}, observed={observed_q.tolist()}"
                )

        if valid_threshold:
            largest_size_raw = summary_row["largest_cluster_size"]
            largest_tp_raw = summary_row["largest_cluster_tp"]
            largest_precision_raw = summary_row["largest_cluster_precision"]

            largest_available = (
                non_singleton_count >= 1
                and not pd.isna(largest_size_raw)
                and not pd.isna(largest_tp_raw)
                and not pd.isna(largest_precision_raw)
            )

            largest_size = None
            largest_tp = None
            largest_precision = None
            if largest_available:
                largest_size = int(largest_size_raw)
                largest_tp = int(largest_tp_raw)
                largest_precision = float(largest_precision_raw)

                if largest_size < 2 or largest_size >= n_i:
                    raise AssertionError(
                        "Largest non-singleton has invalid size in a valid RQ3 case: "
                        f"instance_id={issue}, threshold={threshold_key}, "
                        f"size={largest_size}, n={n_i}"
                    )
                if largest_tp < 0 or largest_tp > largest_size:
                    raise AssertionError(
                        "Invalid largest-cluster TP count for revised RQ3: "
                        f"instance_id={issue}, threshold={threshold_key}, "
                        f"tp={largest_tp}, size={largest_size}"
                    )
                if largest_precision < 0.0 or largest_precision > 1.0:
                    raise AssertionError(
                        "Invalid largest-cluster precision for revised RQ3: "
                        f"instance_id={issue}, threshold={threshold_key}, "
                        f"precision={largest_precision}"
                    )

                if agreement_definition == "majority":
                    # Strict majority definition requested for the 1-Majority branch:
                    # cluster size must be MORE THAN half of the valid candidates.
                    # Examples: n=11 -> minimum 6; n=10 -> minimum 6.
                    majority_available = bool(largest_size > (n_i / 2.0))
                else:
                    # 2-Largest branch: the selected largest non-singleton component
                    # is treated as the agreement cluster on every structurally valid
                    # issue-threshold case, regardless of whether it exceeds n_i/2.
                    majority_available = True

                if majority_available:
                    majority_has_correct = bool(largest_tp > 0)
                    majority_gain = float(
                        (100.0 * largest_precision) - q_percent
                    )

            # Minority agreement is the smallest non-singleton distinct from the
            # majority when a majority exists. The pipeline's existing 'smallest'
            # selection already excludes the selected largest cluster and uses density
            # to resolve equal-size ties.
            if non_singleton_count >= 2:
                smallest_size_raw = summary_row["smallest_cluster_size"]
                smallest_tp_raw = summary_row["smallest_cluster_tp"]
                smallest_precision_raw = summary_row["smallest_cluster_precision"]

                if (
                    pd.isna(smallest_size_raw)
                    or pd.isna(smallest_tp_raw)
                    or pd.isna(smallest_precision_raw)
                ):
                    raise AssertionError(
                        "Expected a smallest non-singleton cluster when at least two "
                        "non-singletons exist: "
                        f"instance_id={issue}, threshold={threshold_key}"
                    )

                minority_size = int(smallest_size_raw)
                minority_tp = int(smallest_tp_raw)
                minority_precision = float(smallest_precision_raw)
                if minority_size < 2 or minority_size >= n_i:
                    raise AssertionError(
                        "Minority agreement has invalid size: "
                        f"instance_id={issue}, threshold={threshold_key}, "
                        f"size={minority_size}, n={n_i}"
                    )
                if minority_tp < 0 or minority_tp > minority_size:
                    raise AssertionError(
                        "Invalid minority-agreement TP count: "
                        f"instance_id={issue}, threshold={threshold_key}, "
                        f"tp={minority_tp}, size={minority_size}"
                    )
                if minority_precision < 0.0 or minority_precision > 1.0:
                    raise AssertionError(
                        "Invalid minority-agreement precision: "
                        f"instance_id={issue}, threshold={threshold_key}, "
                        f"precision={minority_precision}"
                    )

                minority_available = True
                minority_has_correct = bool(minority_tp > 0)
                minority_gain = float(
                    (100.0 * minority_precision) - q_percent
                )

            elif (
                non_singleton_count == 1
                and largest_available
                and not majority_available
            ):
                # With exactly one non-singleton and no majority, that cluster is the
                # only (and therefore smallest) minority agreement structure.
                minority_available = True
                minority_has_correct = bool(largest_tp > 0)
                minority_gain = float(
                    (100.0 * largest_precision) - q_percent
                )

            # Disagreement is the union of all singleton components. A single singleton
            # is valid. All-singleton thresholds have already been excluded above.
            if cluster_group is not None and not cluster_group.empty:
                singleton_rows = cluster_group[
                    cluster_group["Cluster size"].astype(int) == 1
                ]
                if not singleton_rows.empty:
                    singleton_h = pd.to_numeric(
                        singleton_rows["Actual hit H"], errors="raise"
                    ).astype(float)
                    disagreement_precision_percent = float(singleton_h.mean())
                    disagreement_correct_count = int((singleton_h > 1e-12).sum())

                    disagreement_available = True
                    disagreement_has_correct = bool(disagreement_correct_count > 0)
                    disagreement_gain = float(
                        disagreement_precision_percent - q_percent
                    )

        majority_delta = (
            float(majority_gain - disagreement_gain)
            if majority_available and disagreement_available
            else np.nan
        )
        minority_delta = (
            float(minority_gain - disagreement_gain)
            if minority_available and disagreement_available
            else np.nan
        )

        row_index = issue_to_row[issue]
        if majority_available:
            scenario_matrices["majority_agreement"].at[
                row_index, threshold_key
            ] = majority_gain
        if minority_available:
            scenario_matrices["minority_agreement"].at[
                row_index, threshold_key
            ] = minority_gain
        if disagreement_available:
            scenario_matrices["disagreement"].at[
                row_index, threshold_key
            ] = disagreement_gain
        if not pd.isna(majority_delta):
            scenario_matrices["delta_majority"].at[
                row_index, threshold_key
            ] = majority_delta
        if not pd.isna(minority_delta):
            scenario_matrices["delta_minority"].at[
                row_index, threshold_key
            ] = minority_delta

        case_rows.append({
            "instance_id": issue,
            "threshold": threshold,
            "threshold_key": threshold_key,
            "valid_candidates": int(n_i),
            "correct_fixes": int(m_i),
            "valid_threshold": bool(valid_threshold),
            "majority_available": bool(majority_available),
            "majority_has_correct": bool(majority_has_correct),
            "majority_gain": majority_gain,
            "minority_available": bool(minority_available),
            "minority_has_correct": bool(minority_has_correct),
            "minority_gain": minority_gain,
            "disagreement_available": bool(disagreement_available),
            "disagreement_has_correct": bool(disagreement_has_correct),
            "disagreement_gain": disagreement_gain,
            "delta_majority_minus_disagreement": majority_delta,
            "delta_minority_minus_disagreement": minority_delta,
        })

    case_df = pd.DataFrame(case_rows)

    scenario_file_names = {
        "majority_agreement": "majority_agreement_gain_by_issue_threshold.csv",
        "minority_agreement": "minority_agreement_gain_by_issue_threshold.csv",
        "disagreement": "disagreement_gain_by_issue_threshold.csv",
        "delta_majority": (
            "delta_majority_agreement_minus_disagreement_by_issue_threshold.csv"
        ),
        "delta_minority": (
            "delta_minority_agreement_minus_disagreement_by_issue_threshold.csv"
        ),
    }

    # Remove only stale files produced by previous versions of this custom subgroup
    # RQ3 analysis. Unrelated pipeline outputs are not touched.
    stale_subgroup_filenames = {
        "average_agreement_gain_by_issue_threshold.csv",
        "largest_agreement_gain_by_issue_threshold.csv",
        "delta_average_agreement_minus_disagreement_by_issue_threshold.csv",
        "delta_largest_agreement_minus_disagreement_by_issue_threshold.csv",
        "agreement_better_percentage_by_threshold.csv",
    }

    subgroup_paths = {}
    delta_positive_series = {
        "majority": {threshold_key: [] for threshold_key in threshold_columns},
        "minority": {threshold_key: [] for threshold_key in threshold_columns},
        "any_agreement": {threshold_key: [] for threshold_key in threshold_columns},
    }
    gain_positive_series = {
        "majority": {threshold_key: [] for threshold_key in threshold_columns},
        "minority": {threshold_key: [] for threshold_key in threshold_columns},
        "disagreement": {threshold_key: [] for threshold_key in threshold_columns},
    }
    case_summary_rows = []

    for pass_count in correct_fix_counts:
        subgroup_dir = subgroups_root / f"{pass_count}CORRECT"
        subgroup_dir.mkdir(parents=True, exist_ok=True)

        for stale_name in stale_subgroup_filenames:
            stale_path = subgroup_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

        subgroup_issue_ids = {
            issue
            for issue, (_, m_i) in issue_metadata.items()
            if int(m_i) == int(pass_count)
        }

        subgroup_paths[pass_count] = {}
        subgroup_scenario_dfs = {}
        for scenario_name, matrix_df in scenario_matrices.items():
            subgroup_df = matrix_df[
                matrix_df["instance_id"].astype(str).isin(subgroup_issue_ids)
            ].copy()
            subgroup_df = subgroup_df[matrix_columns].sort_values("instance_id")
            output_path = subgroup_dir / scenario_file_names[scenario_name]
            subgroup_df.to_csv(output_path, index=False, na_rep="NA")
            subgroup_paths[pass_count][scenario_name] = output_path
            subgroup_scenario_dfs[scenario_name] = subgroup_df

        if case_df.empty:
            subgroup_case_df = pd.DataFrame()
        else:
            subgroup_case_df = case_df[
                case_df["correct_fixes"].astype(int) == int(pass_count)
            ].copy()

        summary_row = {"correct_fixes": int(pass_count)}
        for scenario_name in ["majority", "minority", "disagreement"]:
            available_column = f"{scenario_name}_available"
            correct_column = f"{scenario_name}_has_correct"

            if subgroup_case_df.empty:
                available_count = 0
                correct_count = 0
            else:
                available_mask = subgroup_case_df[available_column].astype(bool)
                available_count = int(available_mask.sum())
                correct_count = int(
                    subgroup_case_df.loc[available_mask, correct_column]
                    .astype(bool)
                    .sum()
                )

            correct_ratio_percent = (
                float(100.0 * correct_count / available_count)
                if available_count > 0
                else np.nan
            )

            summary_row[f"{scenario_name}_available_cases"] = available_count
            summary_row[f"{scenario_name}_cases_with_correct_fix"] = correct_count
            summary_row[f"{scenario_name}_correct_fix_ratio_percent"] = (
                correct_ratio_percent
            )

        case_summary_rows.append(summary_row)

        # Save and collect positive-delta percentages. For majority and minority,
        # the denominator is the number of issues where that agreement structure and
        # disagreement are both available. Ties (delta == 0) are not wins.
        delta_percentage_rows = []
        for row_label, delta_scenario, series_name in [
            (
                "majority agreement > disagreement",
                "delta_majority",
                "majority",
            ),
            (
                "minority agreement > disagreement",
                "delta_minority",
                "minority",
            ),
        ]:
            delta_row = {"comparison": row_label}
            delta_df = subgroup_scenario_dfs[delta_scenario]

            for threshold_key in threshold_columns:
                delta_values = pd.to_numeric(
                    delta_df[threshold_key], errors="coerce"
                ).dropna()
                if len(delta_values) == 0:
                    positive_percent = np.nan
                else:
                    positive_percent = float(
                        100.0 * (delta_values.astype(float) > 1e-12).mean()
                    )

                delta_row[threshold_key] = positive_percent
                delta_positive_series[series_name][threshold_key].append(
                    positive_percent
                )

            delta_percentage_rows.append(delta_row)

        # Combined agreement-win percentage. Count each issue-threshold once. It is
        # comparable when disagreement and at least one agreement type are available,
        # and is a win when EITHER available agreement gain exceeds disagreement gain.
        any_agreement_row = {"comparison": "any agreement > disagreement"}
        majority_delta_df = subgroup_scenario_dfs["delta_majority"].set_index(
            "instance_id"
        )
        minority_delta_df = subgroup_scenario_dfs["delta_minority"].set_index(
            "instance_id"
        )
        combined_issue_ids = sorted(
            set(majority_delta_df.index.astype(str))
            | set(minority_delta_df.index.astype(str))
        )

        for threshold_key in threshold_columns:
            comparable_count = 0
            agreement_win_count = 0

            for instance_id in combined_issue_ids:
                majority_delta = (
                    pd.to_numeric(
                        pd.Series([majority_delta_df.at[instance_id, threshold_key]]),
                        errors="coerce",
                    ).iloc[0]
                    if instance_id in majority_delta_df.index
                    else np.nan
                )
                minority_delta = (
                    pd.to_numeric(
                        pd.Series([minority_delta_df.at[instance_id, threshold_key]]),
                        errors="coerce",
                    ).iloc[0]
                    if instance_id in minority_delta_df.index
                    else np.nan
                )

                available_deltas = [
                    float(value)
                    for value in [majority_delta, minority_delta]
                    if not pd.isna(value)
                ]
                if not available_deltas:
                    continue

                comparable_count += 1
                if any(value > 1e-12 for value in available_deltas):
                    agreement_win_count += 1

            any_agreement_percent = (
                float(100.0 * agreement_win_count / comparable_count)
                if comparable_count > 0
                else np.nan
            )
            any_agreement_row[threshold_key] = any_agreement_percent
            delta_positive_series["any_agreement"][threshold_key].append(
                any_agreement_percent
            )

        delta_percentage_rows.append(any_agreement_row)

        delta_percentage_df = pd.DataFrame(
            delta_percentage_rows,
            columns=["comparison"] + threshold_columns,
        )
        delta_percentage_path = (
            subgroup_dir / "delta_positive_percentage_by_threshold.csv"
        )
        delta_percentage_df.to_csv(
            delta_percentage_path,
            index=False,
            na_rep="NA",
        )
        subgroup_paths[pass_count]["delta_positive_percentage"] = (
            delta_percentage_path
        )

        # Primary RQ3 frequency view: how often is each structure's OWN signed gain
        # positive? The denominator is structure-specific availability at this threshold;
        # disagreement does not need to coexist with majority/minority for this measure.
        positive_gain_rows = []
        for row_label, gain_scenario, series_name in [
            (
                "majority agreement gain > 0",
                "majority_agreement",
                "majority",
            ),
            (
                "minority agreement gain > 0",
                "minority_agreement",
                "minority",
            ),
            (
                "disagreement gain > 0",
                "disagreement",
                "disagreement",
            ),
        ]:
            gain_row = {"comparison": row_label}
            gain_df = subgroup_scenario_dfs[gain_scenario]

            for threshold_key in threshold_columns:
                gain_values = pd.to_numeric(
                    gain_df[threshold_key], errors="coerce"
                ).dropna()
                if len(gain_values) == 0:
                    positive_percent = np.nan
                else:
                    positive_percent = float(
                        100.0 * (gain_values.astype(float) > 1e-12).mean()
                    )

                gain_row[threshold_key] = positive_percent
                gain_positive_series[series_name][threshold_key].append(
                    positive_percent
                )

            positive_gain_rows.append(gain_row)

        positive_gain_percentage_df = pd.DataFrame(
            positive_gain_rows,
            columns=["comparison"] + threshold_columns,
        )
        positive_gain_percentage_path = (
            subgroup_dir / "positive_gain_percentage_by_threshold.csv"
        )
        positive_gain_percentage_df.to_csv(
            positive_gain_percentage_path,
            index=False,
            na_rep="NA",
        )
        subgroup_paths[pass_count]["positive_gain_percentage"] = (
            positive_gain_percentage_path
        )

    # One 11-row case-count/correctness table directly under solvable/subgroups/.
    case_summary_columns = [
        "correct_fixes",
        "majority_available_cases",
        "majority_cases_with_correct_fix",
        "majority_correct_fix_ratio_percent",
        "minority_available_cases",
        "minority_cases_with_correct_fix",
        "minority_correct_fix_ratio_percent",
        "disagreement_available_cases",
        "disagreement_cases_with_correct_fix",
        "disagreement_correct_fix_ratio_percent",
    ]
    case_summary_df = pd.DataFrame(
        case_summary_rows,
        columns=case_summary_columns,
    )
    case_summary_path = subgroups_root / "rq3_agreement_correct_fix_case_summary.csv"
    case_summary_df.to_csv(case_summary_path, index=False, na_rep="NA")

    # Clean only known plot files from older custom RQ3 visualizations so reruns do
    # not leave obsolete figures beside the current positive-gain plot.
    stale_plot_filenames = {
        "01_average_agreement_gain_by_correct_fix_count.png",
        "02_largest_agreement_gain_by_correct_fix_count.png",
        "03_disagreement_gain_by_correct_fix_count.png",
        "04_delta_average_agreement_minus_disagreement_by_correct_fix_count.png",
        "05_delta_largest_agreement_minus_disagreement_by_correct_fix_count.png",
        "06_average_agreement_better_percentage_by_correct_fix_count.png",
        "07_largest_agreement_better_percentage_by_correct_fix_count.png",
        "08_positive_gain_ratio_largest_vs_disagreement_by_correct_fix_count.png",
        "01_positive_delta_ratio_majority_minority_vs_disagreement_by_correct_fix_count.png",
    }

    threshold_plot_paths = {}
    for threshold in SIMILARITY_THRESHOLDS:
        threshold_key = f"{threshold:.1f}"
        threshold_dir = plots_dir / threshold_key
        threshold_dir.mkdir(parents=True, exist_ok=True)

        for stale_name in stale_plot_filenames:
            stale_path = threshold_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

        majority_values = np.asarray(
            gain_positive_series["majority"][threshold_key],
            dtype=float,
        )
        minority_values = np.asarray(
            gain_positive_series["minority"][threshold_key],
            dtype=float,
        )
        disagreement_values = np.asarray(
            gain_positive_series["disagreement"][threshold_key],
            dtype=float,
        )

        plt.figure(figsize=(10.0, 6.0))
        plotted_any = False

        majority_finite = np.isfinite(majority_values)
        if majority_finite.any():
            plt.plot(
                correct_fix_counts,
                majority_values,
                marker="o",
                linewidth=1.8,
                label=f"{agreement_display_name} gain > 0",
            )
            plotted_any = True

        minority_finite = np.isfinite(minority_values)
        if minority_finite.any():
            plt.plot(
                correct_fix_counts,
                minority_values,
                marker="o",
                linewidth=1.8,
                label="Minority agreement gain > 0",
            )
            plotted_any = True

        disagreement_finite = np.isfinite(disagreement_values)
        if disagreement_finite.any():
            plt.plot(
                correct_fix_counts,
                disagreement_values,
                marker="o",
                linewidth=1.8,
                label="Disagreement gain > 0",
            )
            plotted_any = True

        plt.xlabel("Number of correct fixes among valid fixes")
        plt.ylabel("Available cases with positive gain (%)")
        plt.title(
            f"Positive Correctness Gain - Threshold {threshold_key} - "
            f"{agreement_display_name} definition"
        )
        plt.xticks(correct_fix_counts)
        plt.ylim(0.0, 100.0)
        plt.grid(True, alpha=0.3)

        if plotted_any:
            plt.legend(frameon=False)

        plt.tight_layout()
        plot_path = (
            threshold_dir
            / "01_positive_gain_ratio_majority_minority_disagreement_by_correct_fix_count.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        threshold_plot_paths[threshold_key] = plot_path

    return {
        "subgroups_root": subgroups_root,
        "subgroup_csvs": subgroup_paths,
        "agreement_correct_fix_case_summary": case_summary_path,
        "plots_dir": plots_dir,
        "threshold_plot_dirs": {
            threshold_key: plots_dir / threshold_key
            for threshold_key in threshold_columns
        },
        "plots_by_threshold": threshold_plot_paths,
    }

def save_strategy_availability_by_threshold(
    threshold_average_df: pd.DataFrame,
    subset_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """Save how often each selection strategy exists at each threshold."""
    rows = []
    for threshold in SIMILARITY_THRESHOLDS:
        matching = pd.DataFrame()
        if not threshold_average_df.empty and "threshold" in threshold_average_df.columns:
            matching = threshold_average_df[
                np.isclose(
                    threshold_average_df["threshold"].astype(float),
                    float(threshold),
                )
            ]
        source_row = matching.iloc[0] if not matching.empty else pd.Series(dtype=float)
        row = {
            "model": MODEL,
            "dataset": DATASET_NAME,
            "subset": subset_name,
            "similarity_metric": metric,
            "threshold": float(threshold),
            "num_instances": source_row.get("num_instances", np.nan),
        }
        for strategy in CLUSTER_SELECTION_STRATEGIES:
            row[f"num_available_{strategy}"] = source_row.get(
                f"num_available_{strategy}", np.nan
            )
            row[f"availability_{strategy}"] = source_row.get(
                f"availability_{strategy}", np.nan
            )
        rows.append(row)
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)
    return output_df



def save_strategy_vs_matched_oracle_by_threshold(
    threshold_average_df: pd.DataFrame,
    subset_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """Save paired comparisons between each strategy and the formed-cluster oracle.

    Each strategy and oracle value use exactly the same available issue rows.
    The oracle-minus-strategy gap must be non-negative for strategies that select
    one actual component. It may be negative for singleton_union because that
    strategy combines multiple singleton components.
    """
    rows = []
    for threshold in SIMILARITY_THRESHOLDS:
        matching = pd.DataFrame()
        if not threshold_average_df.empty and "threshold" in threshold_average_df.columns:
            matching = threshold_average_df[
                np.isclose(
                    threshold_average_df["threshold"].astype(float),
                    float(threshold),
                )
            ]
        source_row = matching.iloc[0] if not matching.empty else pd.Series(dtype=float)

        for strategy in ORACLE_FREE_CLUSTER_SELECTION_STRATEGIES:
            strategy_f1 = source_row.get(f"average_{strategy}_f1", np.nan)
            matched_oracle = source_row.get(
                f"average_oracle_f1_on_{strategy}_available_issues",
                np.nan,
            )
            gap = source_row.get(
                f"average_oracle_minus_{strategy}_f1",
                np.nan,
            )
            if (
                strategy != "singleton_union"
                and not pd.isna(strategy_f1)
                and not pd.isna(matched_oracle)
                and float(matched_oracle) + 1e-12 < float(strategy_f1)
            ):
                raise AssertionError(
                    f"Matched formed-cluster oracle is below {strategy} at threshold {threshold}"
                )

            rows.append({
                "model": MODEL,
                "dataset": DATASET_NAME,
                "subset": subset_name,
                "similarity_metric": metric,
                "threshold": float(threshold),
                "strategy": strategy,
                "strategy_average_f1_when_available": strategy_f1,
                "oracle_average_f1_on_same_issues": matched_oracle,
                "oracle_minus_strategy_f1": gap,
                "strategy_available_issue_count": source_row.get(
                    f"num_available_{strategy}", np.nan
                ),
                "total_issue_count": source_row.get("num_instances", np.nan),
                "strategy_availability_rate": source_row.get(
                    f"availability_{strategy}", np.nan
                ),
                "overall_oracle_average_f1_all_issues": source_row.get(
                    "average_oracle_f1", np.nan
                ),
                "zero_pass_issue_count": source_row.get(
                    "num_zero_pass_issues", np.nan
                ),
                "zero_pass_issue_rate": source_row.get(
                    "zero_pass_issue_frequency", np.nan
                ),
            })

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)
    return output_df


def save_cluster_composition_summary(
    threshold_summary_df: pd.DataFrame,
    paths: dict,
):
    """Save threshold-level cluster-composition counts only."""
    if threshold_summary_df.empty:
        print(
            "Skipping cluster composition output because "
            "threshold_summary_df is empty."
        )
        return

    required_cluster_cols = {
        "threshold",
        "instance_id",
        "num_total_patches",
        "num_clusters",
        "singleton_cluster_count",
        "non_singleton_cluster_count",
        "full_issue_cluster_count",
        "non_full_issue_cluster_count",
    }

    if not required_cluster_cols.issubset(set(threshold_summary_df.columns)):
        print(
            "Skipping cluster composition output because the required "
            "cluster-count columns are missing."
        )
        return

    cluster_summary_rows = []

    for threshold in sorted(threshold_summary_df["threshold"].unique()):
        threshold_df = threshold_summary_df[
            threshold_summary_df["threshold"] == threshold
        ]

        total_clusters = int(threshold_df["num_clusters"].sum())
        singleton_clusters = int(threshold_df["singleton_cluster_count"].sum())
        non_singleton_clusters = int(
            threshold_df["non_singleton_cluster_count"].sum()
        )
        full_issue_clusters = int(threshold_df["full_issue_cluster_count"].sum())
        non_full_issue_clusters = int(
            threshold_df["non_full_issue_cluster_count"].sum()
        )

        cluster_summary_rows.append({
            "threshold": float(threshold),
            "num_issue_threshold_rows": int(len(threshold_df)),
            "total_valid_candidates": int(threshold_df["num_total_patches"].sum()),
            "total_clusters": total_clusters,
            "singleton_clusters": singleton_clusters,
            "non_singleton_clusters": non_singleton_clusters,
            "full_issue_clusters": full_issue_clusters,
            "non_full_issue_clusters": non_full_issue_clusters,
            "non_singleton_cluster_rate": safe_divide(
                non_singleton_clusters,
                total_clusters,
            ),
            "singleton_cluster_rate": safe_divide(
                singleton_clusters,
                total_clusters,
            ),
        })

    total_clusters = int(threshold_summary_df["num_clusters"].sum())
    singleton_clusters = int(threshold_summary_df["singleton_cluster_count"].sum())
    non_singleton_clusters = int(
        threshold_summary_df["non_singleton_cluster_count"].sum()
    )
    full_issue_clusters = int(threshold_summary_df["full_issue_cluster_count"].sum())
    non_full_issue_clusters = int(
        threshold_summary_df["non_full_issue_cluster_count"].sum()
    )

    cluster_summary_rows.append({
        "threshold": "all_thresholds",
        "num_issue_threshold_rows": int(len(threshold_summary_df)),
        "total_valid_candidates": int(threshold_summary_df["num_total_patches"].sum()),
        "total_clusters": total_clusters,
        "singleton_clusters": singleton_clusters,
        "non_singleton_clusters": non_singleton_clusters,
        "full_issue_clusters": full_issue_clusters,
        "non_full_issue_clusters": non_full_issue_clusters,
        "non_singleton_cluster_rate": safe_divide(
            non_singleton_clusters,
            total_clusters,
        ),
        "singleton_cluster_rate": safe_divide(
            singleton_clusters,
            total_clusters,
        ),
    })

    pd.DataFrame(cluster_summary_rows).to_csv(
        paths["cluster_composition_summary"],
        index=False,
    )


def find_global_optimal_thresholds(
    threshold_average_df: pd.DataFrame,
    optimization_metric: str = "f1",
) -> pd.DataFrame:
    """Find each strategy's best threshold among thresholds where it exists."""
    rows = []

    for strategy in CLUSTER_SELECTION_STRATEGIES:
        metric_col = f"average_{strategy}_{optimization_metric}"
        precision_col = f"average_{strategy}_precision"
        recall_col = f"average_{strategy}_recall"
        f1_col = f"average_{strategy}_f1"
        roc_auc_col = f"average_{strategy}_roc_auc"
        available_count_col = f"num_available_{strategy}"
        availability_col = f"availability_{strategy}"

        if threshold_average_df.empty or metric_col not in threshold_average_df.columns:
            strategy_df = pd.DataFrame()
        else:
            strategy_df = threshold_average_df[
                threshold_average_df[metric_col].notna()
            ].copy()

        if strategy_df.empty:
            rows.append({
                "strategy": strategy,
                "optimization_metric": optimization_metric,
                "optimal_threshold": np.nan,
                f"average_{optimization_metric}_at_optimal_threshold": np.nan,
                "average_precision_at_optimal_threshold": np.nan,
                "average_recall_at_optimal_threshold": np.nan,
                "average_f1_at_optimal_threshold": np.nan,
                "average_roc_auc_at_optimal_threshold": np.nan,
                "num_instances": int(
                    threshold_average_df["num_instances"].max()
                ) if (
                    not threshold_average_df.empty
                    and "num_instances" in threshold_average_df.columns
                ) else 0,
                "num_available_instances": 0,
                "availability_frequency": 0.0,
            })
            continue

        sort_columns = [metric_col]
        ascending = [False]
        for tie_breaker_col in [precision_col, recall_col, f1_col, "threshold"]:
            if tie_breaker_col in strategy_df.columns:
                sort_columns.append(tie_breaker_col)
                ascending.append(tie_breaker_col == "threshold")

        optimal_row = strategy_df.sort_values(
            by=sort_columns, ascending=ascending
        ).iloc[0]

        rows.append({
            "strategy": strategy,
            "optimization_metric": optimization_metric,
            "optimal_threshold": float(optimal_row["threshold"]),
            f"average_{optimization_metric}_at_optimal_threshold": float(optimal_row[metric_col]),
            "average_precision_at_optimal_threshold": float(optimal_row[precision_col]),
            "average_recall_at_optimal_threshold": float(optimal_row[recall_col]),
            "average_f1_at_optimal_threshold": float(optimal_row[f1_col]),
            "average_roc_auc_at_optimal_threshold": float(optimal_row[roc_auc_col]),
            "num_instances": int(optimal_row["num_instances"]),
            "num_available_instances": int(optimal_row.get(available_count_col, 0)),
            "availability_frequency": float(optimal_row.get(availability_col, np.nan)),
        })

    return pd.DataFrame(rows)


def plot_threshold_metric(
    threshold_average_df: pd.DataFrame,
    metric_name: str,
    output_path: Path,
):
    if threshold_average_df.empty:
        print(f"Skipping plot because threshold_average_df is empty: {output_path}")
        return

    plt.figure(figsize=(9, 5.5))

    for strategy in CLUSTER_SELECTION_STRATEGIES:
        col = f"average_{strategy}_{metric_name}"

        if col not in threshold_average_df.columns:
            continue

        plt.plot(
            threshold_average_df["threshold"],
            threshold_average_df[col],
            marker="o",
            label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
        )

    plt.xlabel(f"{metric} similarity threshold")
    if metric_name == "size":
        plt.ylabel("Average selected cluster/set size (available issues only)")
        plt.title(f"Conditional Average Selected Size by {metric} Threshold")
    else:
        plt.ylabel(f"Average {metric_name} (available issues only)")
        plt.title(f"Conditional Average {metric_name.upper()} by {metric} Threshold")
    plt.xticks(threshold_average_df["threshold"])
    if metric_name != "size":
        plt.ylim(0.0, 1.05)
    else:
        plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(CLUSTER_SELECTION_STRATEGIES),
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def plot_nonsolvable_vs_solvable_f1(
    nonsolvable_threshold_average_df: pd.DataFrame,
    solvable_threshold_average_df: pd.DataFrame,
    output_path: Path,
):
    if nonsolvable_threshold_average_df.empty and solvable_threshold_average_df.empty:
        print(f"Skipping combined F1 plot because both dataframes are empty: {output_path}")
        return

    plt.figure(figsize=(11.5, 6.4))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    strategy_to_color = {
        strategy: color_cycle[index % len(color_cycle)]
        for index, strategy in enumerate(CLUSTER_SELECTION_STRATEGIES)
    } if color_cycle else {}

    for strategy in CLUSTER_SELECTION_STRATEGIES:
        col = f"average_{strategy}_f1"
        label_name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        color = strategy_to_color.get(strategy, None)

        if col in nonsolvable_threshold_average_df.columns:
            plt.plot(
                nonsolvable_threshold_average_df["threshold"],
                nonsolvable_threshold_average_df[col],
                marker="o",
                linestyle=":",
                color=color,
                label=f"nonsolvable {label_name}",
            )

        if col in solvable_threshold_average_df.columns:
            plt.plot(
                solvable_threshold_average_df["threshold"],
                solvable_threshold_average_df[col],
                marker="o",
                linestyle="-",
                color=color,
                label=f"solvable {label_name}",
            )

    all_thresholds = sorted(
        set(nonsolvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
        | set(solvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
    )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Average F1")
    if all_thresholds:
        plt.xticks(all_thresholds)
    plt.title("Nonsolvable vs Solvable Issues")
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=2,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_nonsolvable_vs_solvable_oracle_f1(
    nonsolvable_threshold_average_df: pd.DataFrame,
    solvable_threshold_average_df: pd.DataFrame,
    output_path: Path,
):
    if nonsolvable_threshold_average_df.empty and solvable_threshold_average_df.empty:
        print(f"Skipping oracle F1 plot because both dataframes are empty: {output_path}")
        return

    plt.figure(figsize=(9.0, 5.5))
    if "average_oracle_f1" in nonsolvable_threshold_average_df.columns:
        plt.plot(
            nonsolvable_threshold_average_df["threshold"],
            nonsolvable_threshold_average_df["average_oracle_f1"],
            marker="o",
            linestyle=":",
            label="nonsolvable oracle",
        )
    if "average_oracle_f1" in solvable_threshold_average_df.columns:
        plt.plot(
            solvable_threshold_average_df["threshold"],
            solvable_threshold_average_df["average_oracle_f1"],
            marker="o",
            linestyle="-",
            label="solvable oracle",
        )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Oracle average F1")
    plt.title("Oracle Clustering Capability: Nonsolvable vs Solvable Issues")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_nonsolvable_vs_solvable_strategy_availability(
    nonsolvable_threshold_average_df: pd.DataFrame,
    solvable_threshold_average_df: pd.DataFrame,
    output_path: Path,
):
    if nonsolvable_threshold_average_df.empty and solvable_threshold_average_df.empty:
        print(f"Skipping availability plot because both dataframes are empty: {output_path}")
        return

    plt.figure(figsize=(10.5, 6.0))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    strategy_to_color = {
        strategy: color_cycle[index % len(color_cycle)]
        for index, strategy in enumerate(CLUSTER_SELECTION_STRATEGIES)
    } if color_cycle else {}

    for strategy in CLUSTER_SELECTION_STRATEGIES:
        col = f"availability_{strategy}"
        label_name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        color = strategy_to_color.get(strategy, None)
        if col in nonsolvable_threshold_average_df.columns:
            plt.plot(
                nonsolvable_threshold_average_df["threshold"],
                nonsolvable_threshold_average_df[col],
                marker="o", linestyle=":", color=color,
                label=f"nonsolvable {label_name}",
            )
        if col in solvable_threshold_average_df.columns:
            plt.plot(
                solvable_threshold_average_df["threshold"],
                solvable_threshold_average_df[col],
                marker="o", linestyle="-", color=color,
                label=f"solvable {label_name}",
            )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Strategy availability frequency")
    plt.title("Strategy Availability by Similarity Threshold")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.25),
        ncol=5, frameon=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_nonsolvable_vs_solvable_cluster_size(
    nonsolvable_threshold_average_df: pd.DataFrame,
    solvable_threshold_average_df: pd.DataFrame,
    output_path: Path,
):
    if nonsolvable_threshold_average_df.empty and solvable_threshold_average_df.empty:
        print(
            "Skipping combined cluster-size plot because both dataframes are empty: "
            f"{output_path}"
        )
        return

    plt.figure(figsize=(10.5, 6.0))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    strategy_to_color = {
        strategy: color_cycle[index % len(color_cycle)]
        for index, strategy in enumerate(CLUSTER_SELECTION_STRATEGIES)
    } if color_cycle else {}

    for strategy in CLUSTER_SELECTION_STRATEGIES:
        col = f"average_{strategy}_size"
        label_name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        color = strategy_to_color.get(strategy, None)

        if col in nonsolvable_threshold_average_df.columns:
            plt.plot(
                nonsolvable_threshold_average_df["threshold"],
                nonsolvable_threshold_average_df[col],
                marker="o",
                linestyle=":",
                color=color,
                label=f"nonsolvable {label_name}",
            )

        if col in solvable_threshold_average_df.columns:
            plt.plot(
                solvable_threshold_average_df["threshold"],
                solvable_threshold_average_df[col],
                marker="o",
                linestyle="-",
                color=color,
                label=f"solvable {label_name}",
            )

    all_thresholds = sorted(
        set(nonsolvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
        | set(solvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
    )

    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Average selected cluster/set size")
    if all_thresholds:
        plt.xticks(all_thresholds)
    plt.ylim(bottom=0.0)
    plt.grid(True)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=5,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_nonsolvable_vs_solvable_temperature_coclustering_delta(
    nonsolvable_coclustering_df: pd.DataFrame,
    solvable_coclustering_df: pd.DataFrame,
    output_path: Path,
):
    value_column = "issue_average_close_minus_far_coclustering_rate"

    if nonsolvable_coclustering_df.empty and solvable_coclustering_df.empty:
        print(
            "Skipping temperature co-clustering delta plot because both "
            f"dataframes are empty: {output_path}"
        )
        return

    plt.figure(figsize=(9.0, 5.5))

    if value_column in nonsolvable_coclustering_df.columns:
        plt.plot(
            nonsolvable_coclustering_df["threshold"],
            nonsolvable_coclustering_df[value_column],
            marker="o",
            linestyle=":",
            label="nonsolvable",
        )

    if value_column in solvable_coclustering_df.columns:
        plt.plot(
            solvable_coclustering_df["threshold"],
            solvable_coclustering_df[value_column],
            marker="o",
            linestyle="-",
            label="solvable",
        )

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(f"{metric} similarity threshold")
    plt.ylabel("Close - far co-clustering rate")
    plt.xticks(SIMILARITY_THRESHOLDS)
    plt.ylim(-1.05, 1.05)
    plt.grid(True)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_two_row_threshold_table(
    nonsolvable_threshold_average_df: pd.DataFrame,
    solvable_threshold_average_df: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Build a two-row nonsolvable/solvable table over the nine thresholds."""
    rows = []
    for subset_name, subset_df in [
        ("nonsolvable", nonsolvable_threshold_average_df),
        ("solvable", solvable_threshold_average_df),
    ]:
        row = {"subset": subset_name}

        for threshold in SIMILARITY_THRESHOLDS:
            value = np.nan
            if value_column in subset_df.columns and "threshold" in subset_df.columns:
                matching = subset_df[
                    np.isclose(
                        subset_df["threshold"].astype(float),
                        float(threshold),
                    )
                ]
                if not matching.empty:
                    value = float(matching.iloc[0][value_column])
            row[f"{threshold:.1f}"] = value

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=["subset"] + [f"{threshold:.1f}" for threshold in SIMILARITY_THRESHOLDS],
    )


def build_average_f1_nonsolvable_solvable_table(
    nonsolvable_threshold_average_df: pd.DataFrame,
    solvable_threshold_average_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    all_thresholds = sorted(
        set(nonsolvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
        | set(solvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
    )

    for strategy in CLUSTER_SELECTION_STRATEGIES:
        row = {"strategy": STRATEGY_DISPLAY_NAMES.get(strategy, strategy)}
        col = f"average_{strategy}_f1"

        for threshold in all_thresholds:
            nonsolvable_value = np.nan
            solvable_value = np.nan

            if col in nonsolvable_threshold_average_df.columns:
                matching_nonsolvable = nonsolvable_threshold_average_df[
                    np.isclose(
                        nonsolvable_threshold_average_df["threshold"].astype(float),
                        float(threshold),
                    )
                ]
                if not matching_nonsolvable.empty:
                    nonsolvable_value = float(matching_nonsolvable.iloc[0][col])

            if col in solvable_threshold_average_df.columns:
                matching_solvable = solvable_threshold_average_df[
                    np.isclose(
                        solvable_threshold_average_df["threshold"].astype(float),
                        float(threshold),
                    )
                ]
                if not matching_solvable.empty:
                    solvable_value = float(matching_solvable.iloc[0][col])

            row[f"{threshold:.1f}_nonsolvable"] = nonsolvable_value
            row[f"{threshold:.1f}_solvable"] = solvable_value

        rows.append(row)

    return pd.DataFrame(rows)


def build_average_cluster_size_nonsolvable_solvable_table(
    nonsolvable_threshold_average_df: pd.DataFrame,
    solvable_threshold_average_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    all_thresholds = sorted(
        set(nonsolvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
        | set(solvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
    )

    for strategy in CLUSTER_SELECTION_STRATEGIES:
        row = {"strategy": STRATEGY_DISPLAY_NAMES.get(strategy, strategy)}
        col = f"average_{strategy}_size"

        for threshold in all_thresholds:
            nonsolvable_value = np.nan
            solvable_value = np.nan

            if col in nonsolvable_threshold_average_df.columns:
                matching_nonsolvable = nonsolvable_threshold_average_df[
                    np.isclose(
                        nonsolvable_threshold_average_df["threshold"].astype(float),
                        float(threshold),
                    )
                ]
                if not matching_nonsolvable.empty:
                    nonsolvable_value = float(matching_nonsolvable.iloc[0][col])

            if col in solvable_threshold_average_df.columns:
                matching_solvable = solvable_threshold_average_df[
                    np.isclose(
                        solvable_threshold_average_df["threshold"].astype(float),
                        float(threshold),
                    )
                ]
                if not matching_solvable.empty:
                    solvable_value = float(matching_solvable.iloc[0][col])

            row[f"{threshold:.1f}_nonsolvable"] = nonsolvable_value
            row[f"{threshold:.1f}_solvable"] = solvable_value

        rows.append(row)

    return pd.DataFrame(rows)

def format_latex_value(value) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def print_average_f1_latex_table(
    nonsolvable_threshold_average_df: pd.DataFrame,
    solvable_threshold_average_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> str:
    table_df = build_average_f1_nonsolvable_solvable_table(
        nonsolvable_threshold_average_df=nonsolvable_threshold_average_df,
        solvable_threshold_average_df=solvable_threshold_average_df,
    )

    thresholds = sorted(
        set(nonsolvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
        | set(solvable_threshold_average_df.get("threshold", pd.Series(dtype=float)).tolist())
    )

    column_spec = "l" + "cc" * len(thresholds)
    lines = []
    lines.append("\\begin{tabular}{" + column_spec + "}")
    lines.append("\\toprule")

    header_1 = ["Strategy"]
    for threshold in thresholds:
        header_1.append(f"\\multicolumn{{2}}{{c}}{{{threshold:.1f}}}")
    lines.append(" & ".join(header_1) + " \\\\")

    header_2 = [""]
    for _ in thresholds:
        header_2.extend(["Nonsolvable", "Solvable"])
    lines.append(" & ".join(header_2) + " \\\\")
    lines.append("\\midrule")

    for _, row in table_df.iterrows():
        values = [str(row["strategy"])]
        for threshold in thresholds:
            values.append(
                format_latex_value(
                    row.get(f"{threshold:.1f}_nonsolvable", np.nan)
                )
            )
            values.append(
                format_latex_value(
                    row.get(f"{threshold:.1f}_solvable", np.nan)
                )
            )
        lines.append(" & ".join(values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    latex_code = "\n".join(lines)

    print()
    print("=" * 80)
    print("LATEX TABLE: AVERAGE F1, NONSOLVABLE VS SOLVABLE")
    print("=" * 80)
    print(latex_code)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_code, encoding="utf-8")

    return latex_code

def print_solvable_paired_f1_latex_table(
    solvable_threshold_average_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> str:
    """Print the paper-ready solvable-only F1 table with matched oracle values.

    Every strategy F1 is paired with an oracle mean computed on exactly the same
    issue rows where that strategy exists. This avoids comparing conditional
    strategy means with an oracle mean that uses a different denominator.
    """
    thresholds = [float(value) for value in SIMILARITY_THRESHOLDS]
    column_spec = "l" + "cc" * len(thresholds)
    lines = ["\\begin{tabular}{" + column_spec + "}", "\\toprule"]

    header_1 = ["Strategy"]
    for threshold in thresholds:
        header_1.append(f"\\multicolumn{{2}}{{c}}{{{threshold:.1f}}}")
    lines.append(" & ".join(header_1) + " \\\\")

    header_2 = [""]
    for _ in thresholds:
        header_2.extend(["Strategy F1", "Matched Oracle F1"])
    lines.append(" & ".join(header_2) + " \\\\")
    lines.append("\\midrule")

    work_df = solvable_threshold_average_df.copy()
    if not work_df.empty and "threshold" in work_df.columns:
        work_df["threshold"] = work_df["threshold"].astype(float)

    for strategy in ORACLE_FREE_CLUSTER_SELECTION_STRATEGIES:
        values = [STRATEGY_DISPLAY_NAMES.get(strategy, strategy)]
        strategy_col = f"average_{strategy}_f1"
        oracle_col = f"average_oracle_f1_on_{strategy}_available_issues"

        for threshold in thresholds:
            matching = (
                work_df[np.isclose(work_df["threshold"], threshold)]
                if not work_df.empty and "threshold" in work_df.columns
                else pd.DataFrame()
            )
            row = matching.iloc[0] if not matching.empty else pd.Series(dtype=float)
            strategy_f1 = row.get(strategy_col, np.nan)
            matched_oracle_f1 = row.get(oracle_col, np.nan)
            if (
                strategy != "singleton_union"
                and not pd.isna(strategy_f1)
                and not pd.isna(matched_oracle_f1)
                and float(matched_oracle_f1) + 1e-12 < float(strategy_f1)
            ):
                raise AssertionError(
                    f"Matched formed-cluster oracle is below {strategy} at threshold {threshold}"
                )
            values.append(format_latex_value(strategy_f1))
            values.append(format_latex_value(matched_oracle_f1))

        lines.append(" & ".join(values) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    latex_code = "\n".join(lines)

    print()
    print("=" * 80)
    print("LATEX TABLE: SOLVABLE STRATEGY F1 WITH MATCHED ORACLE F1")
    print("=" * 80)
    print(latex_code)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_code, encoding="utf-8")

    return latex_code


def compute_threshold_range_per_instance(
    threshold_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if threshold_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "instance_id",
                "min_threshold",
                "max_threshold",
            ]
        )

    for instance_id, group in threshold_summary_df.groupby("instance_id"):
        group = group.sort_values("threshold").copy()

        num_total_patches = int(group["num_total_patches"].iloc[0])

        split_rows = group[group["num_clusters"] >= 2]

        if split_rows.empty:
            min_threshold = None
        else:
            min_threshold = float(split_rows["threshold"].min())

        before_all_singleton_rows = group[
            group["num_clusters"] < num_total_patches
        ]

        if before_all_singleton_rows.empty:
            max_threshold = None
        else:
            max_threshold = float(before_all_singleton_rows["threshold"].max())

        rows.append(
            {
                "instance_id": instance_id,
                "min_threshold": min_threshold,
                "max_threshold": max_threshold,
            }
        )

    return pd.DataFrame(rows)



def compute_range_averaged_f1_per_instance(
    threshold_summary_df: pd.DataFrame,
    threshold_range_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    output_columns = [
        "instance_id",
        "min_threshold",
        "max_threshold",
        "num_thresholds_in_range",
        "oracle_avg_f1",
        "largest_avg_f1",
        "smallest_avg_f1",
        "singleton_union_avg_f1",
        "singleton_furthest_avg_f1",
    ]

    if threshold_summary_df.empty or threshold_range_df.empty:
        return pd.DataFrame(columns=output_columns)

    range_lookup = {
        row["instance_id"]: row
        for _, row in threshold_range_df.iterrows()
    }

    for instance_id, group in threshold_summary_df.groupby("instance_id"):
        if instance_id not in range_lookup:
            continue

        min_threshold = range_lookup[instance_id]["min_threshold"]
        max_threshold = range_lookup[instance_id]["max_threshold"]

        if pd.isna(min_threshold) or pd.isna(max_threshold):
            continue

        group_in_range = group[
            (group["threshold"] >= float(min_threshold))
            & (group["threshold"] <= float(max_threshold))
        ].copy()

        if group_in_range.empty:
            continue

        rows.append(
            {
                "instance_id": instance_id,
                "min_threshold": float(min_threshold),
                "max_threshold": float(max_threshold),
                "num_thresholds_in_range": int(len(group_in_range)),
                "oracle_avg_f1": float(group_in_range["oracle_cluster_f1"].mean()),
                "largest_avg_f1": float(group_in_range["largest_cluster_f1"].mean()),
                "smallest_avg_f1": float(group_in_range["smallest_cluster_f1"].mean()),
                "singleton_union_avg_f1": float(group_in_range["singleton_union_cluster_f1"].mean()),
                "singleton_furthest_avg_f1": float(group_in_range["singleton_furthest_cluster_f1"].mean()),
            }
        )

    return pd.DataFrame(rows, columns=output_columns)

def compute_range_averaged_roc_auc_per_instance(
    threshold_summary_df: pd.DataFrame,
    threshold_range_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    output_columns = [
        "instance_id",
        "min_threshold",
        "max_threshold",
        "num_thresholds_in_range",
        "oracle_avg_roc_auc",
        "largest_avg_roc_auc",
        "smallest_avg_roc_auc",
        "singleton_union_avg_roc_auc",
        "singleton_furthest_avg_roc_auc",
    ]

    if threshold_summary_df.empty or threshold_range_df.empty:
        return pd.DataFrame(columns=output_columns)

    range_lookup = {
        row["instance_id"]: row
        for _, row in threshold_range_df.iterrows()
    }

    for instance_id, group in threshold_summary_df.groupby("instance_id"):
        if instance_id not in range_lookup:
            continue

        min_threshold = range_lookup[instance_id]["min_threshold"]
        max_threshold = range_lookup[instance_id]["max_threshold"]

        if pd.isna(min_threshold) or pd.isna(max_threshold):
            continue

        group_in_range = group[
            (group["threshold"] >= float(min_threshold))
            & (group["threshold"] <= float(max_threshold))
        ].copy()

        if group_in_range.empty:
            continue

        rows.append(
            {
                "instance_id": instance_id,
                "min_threshold": float(min_threshold),
                "max_threshold": float(max_threshold),
                "num_thresholds_in_range": int(len(group_in_range)),
                "oracle_avg_roc_auc": float(group_in_range["oracle_roc_auc_cluster_roc_auc"].mean()),
                "largest_avg_roc_auc": float(group_in_range["largest_cluster_roc_auc"].mean()),
                "smallest_avg_roc_auc": float(group_in_range["smallest_cluster_roc_auc"].mean()),
                "singleton_union_avg_roc_auc": float(group_in_range["singleton_union_cluster_roc_auc"].mean()),
                "singleton_furthest_avg_roc_auc": float(group_in_range["singleton_furthest_cluster_roc_auc"].mean()),
            }
        )

    return pd.DataFrame(rows, columns=output_columns)



def run_clustering_analysis_for_subset(
    subset_name: str,
    subset_results_df: pd.DataFrame,
    full_results_df: pd.DataFrame,
    syntax_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = make_output_paths(subset_name)

    allowed_instance_ids = set(
        subset_results_df["instance_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    threshold_summary_rows = []
    temperature_coclustering_issue_rows = []
    cluster_hit_advantage_rows = []

    instance_dirs = sorted(
        path for path in PRED_ROOT.iterdir()
        if path.is_dir() and path.name in allowed_instance_ids
    )

    num_subset_issues = len(allowed_instance_ids)
    num_prediction_folder_issues = len(instance_dirs)
    num_missing_prediction_folder_issues = (
        num_subset_issues - num_prediction_folder_issues
    )

    repo_available_issue_count = 0
    missing_syntax_row_issue_count = 0
    syntax_surviving_issue_count = 0
    syntax_valid_temperature_count = 0
    result_status_surviving_issue_count = 0
    result_status_valid_temperature_count = 0
    valid_patch_surviving_issue_count = 0
    valid_patch_temperature_count = 0
    zero_valid_candidate_issue_count = 0
    solvable_recheck_removed_issue_count = 0
    clustered_issue_count = 0

    print()
    print("=" * 80)
    print(f"RUNNING CLUSTERING ANALYSIS FOR SUBSET: {subset_name.upper()}")
    print("=" * 80)
    print(f"Instances in subset dataframe: {len(subset_results_df)}")
    print(f"Unique issues in subset: {num_subset_issues}")
    print(f"Matching prediction folders found: {num_prediction_folder_issues}")
    print(
        "Subset issues with no prediction folder: "
        f"{num_missing_prediction_folder_issues}"
    )
    print(f"Output directory: {paths['output_dir']}")
    print(f"Shared matrix cache: {MATRIX_ROOT}")

    for instance_dir in instance_dirs:
        instance_id = instance_dir.name

        try:
            repo = get_repo_for_instance(instance_id)
        except KeyError:
            continue

        repo_available_issue_count += 1

        syntax_matching = syntax_df[syntax_df["instance_id"] == instance_id]
        if syntax_matching.empty:
            missing_syntax_row_issue_count += 1

        correct_syntax_temperatures = get_correct_syntax_temperatures_for_instance(
            syntax_df=syntax_df,
            instance_id=instance_id,
        )
        syntax_valid_temperature_count += len(correct_syntax_temperatures)

        if correct_syntax_temperatures:
            syntax_surviving_issue_count += 1

        usable_temperatures = get_usable_temperatures_for_instance(
            results_df=full_results_df,
            syntax_df=syntax_df,
            instance_id=instance_id,
            prompt_type=PROMPT_TYPE,
        )
        result_status_valid_temperature_count += len(usable_temperatures)

        if usable_temperatures:
            result_status_surviving_issue_count += 1

        temperature_to_diff = load_valid_candidates_for_instance(
            instance_dir=instance_dir,
            prompt_type=PROMPT_TYPE,
            allowed_temperatures=usable_temperatures,
        )

        valid_temperatures = list(temperature_to_diff.keys())
        num_total_patches = len(valid_temperatures)
        valid_patch_temperature_count += num_total_patches

        if num_total_patches == 0:
            zero_valid_candidate_issue_count += 1
            continue

        valid_patch_surviving_issue_count += 1

        temperature_to_pass = {
            temp_s: get_test_pass_for_temperature(
                results_df=full_results_df,
                instance_id=instance_id,
                temp_s=temp_s,
                prompt_type=PROMPT_TYPE,
            )
            for temp_s in valid_temperatures
        }

        num_passes_for_issue = sum(temperature_to_pass.values())
        num_fails_for_issue = num_total_patches - num_passes_for_issue

        if subset_name == "solvable" and num_passes_for_issue <= 0:
            solvable_recheck_removed_issue_count += 1
            print(
                f"Skipping {instance_id} in solvable subset: "
                "valid loaded patches contain no passing candidate."
            )
            continue

        if subset_name == "nonsolvable" and num_passes_for_issue != 0:
            raise AssertionError(
                f"Nonsolvable subset invariant failed for {instance_id}: "
                f"found {num_passes_for_issue} passing valid candidate(s)."
            )

        clustered_issue_count += 1

        similarity_lang = get_lang_for_repo(repo)
        similarity_df, _ = get_or_build_similarity_matrix(
            temperature_to_diff=temperature_to_diff,
            matrix_root=MATRIX_ROOT,
            instance_id=instance_id,
            lang=similarity_lang,
            metric_name=metric,
        )

        for threshold in SIMILARITY_THRESHOLDS:
            labels = cluster_from_similarity_matrix(
                similarity_df=similarity_df,
                similarity_threshold=threshold,
            )

            issue_coclustering_rates = compute_temperature_coclustering_rates_for_issue(
                labels=labels,
            )
            temperature_coclustering_issue_rows.append({
                "subset": subset_name,
                "instance_id": instance_id,
                "threshold": float(threshold),
                **issue_coclustering_rates,
            })

            cluster_structure_counts = compute_cluster_structure_counts(
                labels=labels,
                num_total_patches=num_total_patches,
            )

            num_outliers = 0
            cluster_ids = sorted(set(labels.values()))
            num_clusters = len(cluster_ids)
            cluster_sizes = sorted(
                [
                    sum(1 for temp in labels if labels[temp] == cluster_id)
                    for cluster_id in cluster_ids
                ],
                reverse=True,
            )
            cluster_size_pattern = "|".join(str(size) for size in cluster_sizes)
            fragmentation_g = compute_fragmentation_g(labels=labels)

            clusters, summary = compute_cluster_metrics(
                labels=labels,
                temperature_to_pass=temperature_to_pass,
                similarity_df=similarity_df,
            )

            # RQ2 agreement/correctness analysis: evaluate EVERY actual formed
            # connected-component cluster, without applying a selection strategy.
            # Sort by size descending (then cluster id) so the compact output aligns
            # with cluster_size_pattern_by_threshold.csv.
            sorted_actual_clusters = sorted(
                clusters.items(),
                key=lambda item: (-int(item[1]["size"]), int(item[0])),
            )
            for _, cluster_data in sorted_actual_clusters:
                cluster_size = int(cluster_data["size"])
                cluster_tp = int(cluster_data["tp"])
                # H is now the probability that one candidate picked from this
                # specific cluster is correct: the cluster precision.
                actual_hit_percent = 100.0 * (cluster_tp / cluster_size)
                random_baseline_probability = (
                    compute_random_precision_baseline(
                        num_valid_candidates=num_total_patches,
                        num_passing_candidates=num_passes_for_issue,
                        cluster_size=cluster_size,
                    )
                )
                random_baseline_percent = 100.0 * random_baseline_probability
                # RQ2 signed correctness gain. IMPORTANT: do NOT clip at zero.
                # A < 0 is meaningful and means this cluster is less precise than
                # random candidate selection for the same issue.
                advantage_percent = float(
                    actual_hit_percent - random_baseline_percent
                )

                cluster_hit_advantage_rows.append({
                    "Issue": str(instance_id),
                    "Threshold": float(threshold),
                    "Cluster size": int(cluster_size),
                    "Actual hit H": float(actual_hit_percent),
                    "Random baseline q": float(random_baseline_percent),
                    "Advantage A": float(advantage_percent),
                })

            row = {
                "subset": subset_name,
                "instance_id": instance_id,
                "threshold": float(threshold),
                "num_total_patches": int(num_total_patches),
                "num_passes_for_issue": int(num_passes_for_issue),
                "num_fails_for_issue": int(num_fails_for_issue),
                "num_clusters": int(num_clusters),
                "num_outliers": int(num_outliers),
                "cluster_size_pattern": cluster_size_pattern,
                "fragmentation_G": float(fragmentation_g),
                "singleton_cluster_count": int(
                    cluster_structure_counts["singleton_cluster_count"]
                ),
                "non_singleton_cluster_count": int(
                    cluster_structure_counts["non_singleton_cluster_count"]
                ),
                "full_issue_cluster_count": int(
                    cluster_structure_counts["full_issue_cluster_count"]
                ),
                "non_full_issue_cluster_count": int(
                    cluster_structure_counts["non_full_issue_cluster_count"]
                ),
            }

            row.update(summary)
            threshold_summary_rows.append(row)

    print()
    print("=" * 80)
    print(f"FILTERING AUDIT SUMMARY: {subset_name.upper()}")
    print("=" * 80)
    print(f"Initial subset issues: {num_subset_issues}")
    print(
        "Issues remaining after prediction-folder filter: "
        f"{num_prediction_folder_issues}"
    )
    print(
        "Issues removed because prediction folder is missing: "
        f"{num_missing_prediction_folder_issues}"
    )
    print(
        "Issues remaining after repository metadata filter: "
        f"{repo_available_issue_count}"
    )
    print(
        "Issues removed because repository metadata is unavailable: "
        f"{num_prediction_folder_issues - repo_available_issue_count}"
    )
    print(f"Issues with no syntax-status row: {missing_syntax_row_issue_count}")
    print(
        "Issues remaining after syntax filter "
        "(>=1 syntax-valid temperature): "
        f"{syntax_surviving_issue_count}"
    )
    print(
        "Issues removed at syntax stage "
        "(zero syntax-valid temperatures): "
        f"{repo_available_issue_count - syntax_surviving_issue_count}"
    )
    print(
        "Syntax-valid temperatures across repository-matched issues: "
        f"{syntax_valid_temperature_count}"
    )
    print(
        "Issues remaining after SWE-bench result-status filter "
        "(>=1 syntax-valid temperature with result 0 or 1): "
        f"{result_status_surviving_issue_count}"
    )
    print(
        "Issues removed at SWE-bench result-status stage: "
        f"{syntax_surviving_issue_count - result_status_surviving_issue_count}"
    )
    print(
        "Temperatures remaining after syntax + SWE-bench result-status filters: "
        f"{result_status_valid_temperature_count}"
    )
    print(
        "Issues remaining after prediction-file/readability/non-empty-patch "
        f"filter: {valid_patch_surviving_issue_count}"
    )
    print(
        "Temperatures remaining after prediction-file/readability/non-empty-patch "
        f"filter: {valid_patch_temperature_count}"
    )
    print(
        "Issues removed at prediction-file/readability/non-empty-patch stage "
        "after surviving result-status filtering: "
        f"{result_status_surviving_issue_count - valid_patch_surviving_issue_count}"
    )
    print(
        "Issues skipped with zero valid candidates after all temperature-level "
        f"filters: {zero_valid_candidate_issue_count}"
    )

    if subset_name == "solvable":
        print(
            "Issues removed by second solvability check "
            "(valid patches no longer contain a passing candidate): "
            f"{solvable_recheck_removed_issue_count}"
        )
        print(
            "Final solvable issues remaining after second solvability check: "
            f"{clustered_issue_count}"
        )
    else:
        print(f"Final issues used for clustering: {clustered_issue_count}")

    print("=" * 80)

    threshold_summary_df = pd.DataFrame(threshold_summary_rows)
    threshold_summary_df.to_csv(paths["threshold_summary"], index=False)

    cluster_hit_advantage_df = pd.DataFrame(
        cluster_hit_advantage_rows,
        columns=[
            "Issue",
            "Threshold",
            "Cluster size",
            "Actual hit H",
            "Random baseline q",
            "Advantage A",
        ],
    )
    save_cluster_hit_advantage_outputs(
        cluster_hit_advantage_df=cluster_hit_advantage_df,
        threshold_summary_df=threshold_summary_df,
        output_dir=paths["issue_f1_by_threshold_dir"],
    )

    # RQ2 coverage/false-positive and RQ3 correctness-gain analyses for the solvable
    # subset under two alternative agreement definitions. Both branches use the same
    # agreement definitions so their results can be compared directly.
    if subset_name == "solvable":
        rq_results_root = paths["output_dir"].parent / "Results"
        rq_definition_specs = [
            ("majority", rq_results_root / "1-Majority"),
            ("largest", rq_results_root / "2-Largest"),
        ]
        for agreement_definition, definition_results_dir in rq_definition_specs:
            print()
            print("=" * 80)
            print(
                "RUNNING RQ2/RQ3 AGREEMENT DEFINITION: "
                f"{agreement_definition.upper()}"
            )
            print("=" * 80)
            save_rq3_correct_fix_subgroup_scenarios(
                cluster_hit_advantage_df=cluster_hit_advantage_df,
                threshold_summary_df=threshold_summary_df,
                solvable_output_dir=paths["output_dir"],
                results_dir=definition_results_dir,
                agreement_definition=agreement_definition,
            )
            save_rq2_all_thresholds_group_coverage(
                cluster_hit_advantage_df=cluster_hit_advantage_df,
                threshold_summary_df=threshold_summary_df,
                results_dir=definition_results_dir,
                agreement_definition=agreement_definition,
            )
            save_rq2_all_thresholds_group_false_positives(
                cluster_hit_advantage_df=cluster_hit_advantage_df,
                threshold_summary_df=threshold_summary_df,
                results_dir=definition_results_dir,
                agreement_definition=agreement_definition,
            )

    temperature_coclustering_issue_df = pd.DataFrame(
        temperature_coclustering_issue_rows
    )
    temperature_coclustering_df = summarize_issue_average_temperature_coclustering(
        issue_rate_df=temperature_coclustering_issue_df,
    )

    save_cluster_composition_summary(
        threshold_summary_df=threshold_summary_df,
        paths=paths,
    )

    threshold_range_df = compute_threshold_range_per_instance(
        threshold_summary_df=threshold_summary_df,
    )
    threshold_range_df.to_csv(paths["threshold_range"], index=False)

    range_averaged_f1_df = compute_range_averaged_f1_per_instance(
        threshold_summary_df=threshold_summary_df,
        threshold_range_df=threshold_range_df,
    )
    range_averaged_f1_df.to_csv(paths["range_averaged_f1"], index=False)

    range_averaged_roc_auc_df = compute_range_averaged_roc_auc_per_instance(
        threshold_summary_df=threshold_summary_df,
        threshold_range_df=threshold_range_df,
    )
    range_averaged_roc_auc_df.to_csv(paths["range_averaged_roc_auc"], index=False)

    oracle_per_instance_df = select_oracle_threshold_per_instance(
        threshold_summary_df=threshold_summary_df,
        strategy_prefix="oracle",
    )
    oracle_per_instance_df.to_csv(paths["oracle_per_instance"], index=False)

    threshold_average_df = compute_threshold_average_metrics(
        threshold_summary_df=threshold_summary_df,
    )
    threshold_average_df.to_csv(paths["threshold_average_metrics"], index=False)

    save_issue_f1_tables_by_threshold(
        threshold_summary_df=threshold_summary_df,
        subset_name=subset_name,
        output_dir=paths["issue_f1_by_threshold_dir"],
    )

    # Build four compact issue-by-threshold tables FROM the nine threshold CSVs.
    # These are created independently for both the nonsolvable and solvable subsets.
    save_cross_threshold_issue_tables_from_threshold_csvs(
        issue_f1_by_threshold_dir=paths["issue_f1_by_threshold_dir"],
    )

    # RQ2 prioritization-utility outputs are meaningful only for solvable issues.
    if subset_name == "solvable":
        save_rq2_prioritization_utility_outputs(
            threshold_summary_df=threshold_summary_df,
            output_dir=paths["issue_f1_by_threshold_dir"],
        )
        save_rq2_precision_weighted_utility_outputs(
            threshold_summary_df=threshold_summary_df,
            output_dir=paths["issue_f1_by_threshold_dir"],
        )
        save_rq2_strategy_agreement_correctness_precision_summary(
            threshold_summary_df=threshold_summary_df,
            output_dir=paths["issue_f1_by_threshold_dir"],
        )

    save_strategy_availability_by_threshold(
        threshold_average_df=threshold_average_df,
        subset_name=subset_name,
        output_path=paths["strategy_availability"],
    )

    save_strategy_vs_matched_oracle_by_threshold(
        threshold_average_df=threshold_average_df,
        subset_name=subset_name,
        output_path=paths["strategy_vs_matched_oracle"],
    )

    save_rq2_average_f1_by_threshold(
        threshold_average_df=threshold_average_df,
        subset_name=subset_name,
        output_path=paths["rq2_average_f1_by_threshold"],
    )

    save_rq2_average_cluster_size_by_threshold(
        threshold_average_df=threshold_average_df,
        subset_name=subset_name,
        output_path=paths["rq2_average_cluster_size_by_threshold"],
    )

    save_best_configuration(
        threshold_average_df=threshold_average_df,
        subset_name=subset_name,
        output_path=paths["best_configuration"],
    )

    build_and_save_issue_f1_matrix(
        threshold_summary_df=threshold_summary_df,
        output_path=paths["issue_f1_matrix"],
    )

    global_optimal_thresholds_df = find_global_optimal_thresholds(
        threshold_average_df=threshold_average_df,
        optimization_metric="f1",
    )
    global_optimal_thresholds_df.to_csv(
        paths["global_optimal_thresholds"],
        index=False,
    )

    plot_threshold_metric(
        threshold_average_df=threshold_average_df,
        metric_name="precision",
        output_path=paths["precision_plot"],
    )

    plot_threshold_metric(
        threshold_average_df=threshold_average_df,
        metric_name="recall",
        output_path=paths["recall_plot"],
    )

    plot_threshold_metric(
        threshold_average_df=threshold_average_df,
        metric_name="f1",
        output_path=paths["f1_plot"],
    )

    plot_threshold_metric(
        threshold_average_df=threshold_average_df,
        metric_name="size",
        output_path=paths["cluster_size_plot"],
    )

    plot_threshold_metric(
        threshold_average_df=threshold_average_df,
        metric_name="roc_auc",
        output_path=paths["roc_auc_plot"],
    )

    return threshold_summary_df, temperature_coclustering_df, cluster_hit_advantage_df


def remove_legacy_rq2_outputs():
    """Remove outputs created by older total-vs-solvable RQ2 implementations."""
    legacy_paths = [
        BASE_OUTPUT_DIR / "fragmentation_mann_whitney_by_threshold.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2_spearman_correct_fraction_vs_fragmentation.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2_spearman_correct_fraction_vs_fragmentation.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2_majority_positive_gain_ratio_t0.9.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2_majority_positive_gain_ratio_t0.9.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2_majority_positive_gain_ratio_t0.9.png",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2_majority_positive_gain_ratio_t0.9.png",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2_all_thresholds_positive_gain_percentage.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2_all_thresholds_positive_gain_percentage.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2_all_thresholds_positive_gain_percentage.png",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2_all_thresholds_positive_gain_percentage.png",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2_all_thresholds_positive_gain_fraction.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2_all_thresholds_positive_gain_fraction.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2_issue_averaged_positive_gain_percentage.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2_issue_averaged_positive_gain_percentage.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2main_results.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2main_results.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq3_all_thresholds_positive_gain_fraction.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq3_all_thresholds_positive_gain_fraction.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq3_all_thresholds_positive_gain_percentage.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq3_all_thresholds_positive_gain_percentage.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq3_all_thresholds_positive_gain_percentage.png",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq3_all_thresholds_positive_gain_percentage.png",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq3_issue_averaged_positive_gain_percentage.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq3_issue_averaged_positive_gain_percentage.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq3_majority_first_emergence_counts_by_threshold.txt",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq3_majority_first_emergence_counts_by_threshold.txt",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq3_majority_positive_gain_ratio_t0.9.csv",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq3_majority_positive_gain_ratio_t0.9.csv",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq3_majority_positive_gain_ratio_t0.9.png",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq3_majority_positive_gain_ratio_t0.9.png",
        BASE_OUTPUT_DIR / "Results" / "1-Majority" / "rq2_majority_first_emergence_counts_by_threshold.txt",
        BASE_OUTPUT_DIR / "Results" / "2-Largest" / "rq2_majority_first_emergence_counts_by_threshold.txt",
        BASE_OUTPUT_DIR / "threshold_close_temperature_coclustering_total_vs_solvable.pdf",
        BASE_OUTPUT_DIR / "threshold_close_temperature_coclustering_total_vs_solvable.csv",
        BASE_OUTPUT_DIR / "threshold_far_temperature_coclustering_total_vs_solvable.csv",
        BASE_OUTPUT_DIR / "threshold_close_minus_far_temperature_coclustering_total_vs_solvable.csv",
        BASE_OUTPUT_DIR / "threshold_close_minus_far_temperature_coclustering_total_vs_solvable.pdf",
    ]

    for subset_name in ["total", "nonsolvable", "solvable"]:
        subset_dir = BASE_OUTPUT_DIR / subset_name
        legacy_paths.extend([
            subset_dir / "close_temperature_coclustering_per_issue.csv",
            subset_dir / "close_temperature_coclustering_summary.csv",
        ])

    for legacy_path in legacy_paths:
        if legacy_path.exists():
            legacy_path.unlink()



# ==============================================================================
# ADDED ANALYSIS: RQ3 MAJORITY-AGREEMENT FIRST-EMERGENCE COUNTS
# ==============================================================================
def save_rq1_majority_agreement_prevalence(
    solvable_threshold_summary_df: pd.DataFrame,
    nonsolvable_threshold_summary_df: pd.DataFrame,
    results_dir: Path,
    agreement_definition: str,
) -> pd.DataFrame:
    """Save threshold-wise majority-agreement prevalence for solvable/nonsolvable issues.

    This RQ1 analysis reuses the two existing agreement definitions without
    changing the RQ2 calculations:

      * ``majority``: on a structurally valid threshold, the selected largest
        non-singleton component is agreement only when |C| > n_i / 2.
      * ``largest``: on a structurally valid threshold, the selected largest
        non-singleton component is agreement regardless of its size relative to n_i.

    The same structural-validity rule already used by the existing RQ2 analysis is
    retained here: 1 < number_of_components < n_i. Thus a single full component and
    an all-singleton partition do not count as majority agreement under either branch.

    At each threshold t:

        P(M=1|S) = (# solvable issues with majority agreement at t)
                   / (# solvable issues represented at t)

        P(M=1|N) = (# nonsolvable issues with majority agreement at t)
                   / (# nonsolvable issues represented at t)

        deltaP   = P(M=1|S) - P(M=1|N)

    A two-sided Fisher's exact test is also computed from the 2x2 table

        [[solvable majority,    solvable no-majority],
         [nonsolvable majority, nonsolvable no-majority]].

    The Fisher odds ratio and raw p-value are saved for every threshold.

    Prevalence and deltaP values are saved as proportions on [0, 1], not percentages.
    """
    if agreement_definition not in {"majority", "largest"}:
        raise ValueError(
            "agreement_definition must be either 'majority' or 'largest', got "
            f"{agreement_definition!r}"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "instance_id",
        "threshold",
        "num_total_patches",
        "num_clusters",
        "non_singleton_cluster_count",
        "largest_cluster_size",
    }

    def prepare_subset(df: pd.DataFrame, subset_label: str) -> pd.DataFrame:
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Cannot compute RQ1 for {subset_label}; threshold-summary columns "
                f"are missing: {sorted(missing_columns)}"
            )

        work_df = df.copy()
        work_df["instance_id"] = work_df["instance_id"].astype(str)
        work_df["threshold"] = pd.to_numeric(work_df["threshold"], errors="raise")
        work_df["num_total_patches"] = pd.to_numeric(
            work_df["num_total_patches"], errors="raise"
        )
        work_df["num_clusters"] = pd.to_numeric(
            work_df["num_clusters"], errors="raise"
        )
        work_df["non_singleton_cluster_count"] = pd.to_numeric(
            work_df["non_singleton_cluster_count"], errors="raise"
        )
        work_df["largest_cluster_size"] = pd.to_numeric(
            work_df["largest_cluster_size"], errors="coerce"
        )
        return work_df

    solvable_df = prepare_subset(solvable_threshold_summary_df, "solvable")
    nonsolvable_df = prepare_subset(nonsolvable_threshold_summary_df, "nonsolvable")

    def prevalence_at_threshold(
        work_df: pd.DataFrame,
        threshold: float,
        subset_label: str,
    ) -> tuple[float, int, int]:
        threshold_df = work_df[
            np.isclose(work_df["threshold"].astype(float), float(threshold))
        ].copy()

        if threshold_df.empty:
            return np.nan, 0, 0

        duplicate_mask = threshold_df.duplicated(subset=["instance_id"], keep=False)
        if duplicate_mask.any():
            duplicate_ids = sorted(
                threshold_df.loc[duplicate_mask, "instance_id"].astype(str).unique()
            )
            raise ValueError(
                f"RQ1 expected one {subset_label} row per issue at threshold "
                f"{threshold:.1f}; duplicates found for {duplicate_ids[:10]}"
            )

        total_issues = int(threshold_df["instance_id"].nunique())
        majority_count = 0

        for _, row in threshold_df.iterrows():
            n_i = int(row["num_total_patches"])
            num_clusters = int(row["num_clusters"])
            non_singleton_count = int(row["non_singleton_cluster_count"])

            if n_i <= 0:
                raise ValueError(
                    f"RQ1 found non-positive valid-candidate count for "
                    f"instance_id={row['instance_id']}, threshold={threshold:.1f}: "
                    f"n={n_i}"
                )

            # Match the existing RQ2 structural-validity rule exactly.
            valid_threshold = bool(1 < num_clusters < n_i)
            if not valid_threshold:
                continue

            largest_size_raw = row["largest_cluster_size"]
            largest_available = (
                non_singleton_count >= 1
                and not pd.isna(largest_size_raw)
            )
            if not largest_available:
                continue

            largest_size = int(largest_size_raw)
            if largest_size < 2 or largest_size >= n_i:
                raise AssertionError(
                    "RQ1 largest non-singleton has invalid size in a structurally "
                    f"valid case: instance_id={row['instance_id']}, "
                    f"threshold={threshold:.1f}, size={largest_size}, n={n_i}"
                )

            if agreement_definition == "majority":
                majority_exists = bool(largest_size > (n_i / 2.0))
            else:
                majority_exists = True

            if majority_exists:
                majority_count += 1

        ratio = (
            float(majority_count / total_issues)
            if total_issues > 0
            else np.nan
        )
        return ratio, majority_count, total_issues

    from scipy.stats import fisher_exact

    def compute_or_95_ci(fisher_table: list) -> tuple[float, float]:
        """Return a 95% CI for the sample odds ratio from one 2x2 table.

        The point OR reported by scipy.stats.fisher_exact is the sample cross-product
        odds ratio.  The CI therefore uses the standard log-OR (Wald) interval so it
        targets the same sample-OR definition.  When any cell is zero, a 0.5
        Haldane-Anscombe correction is applied to all four cells for the CI only;
        the reported OR and Fisher p-value remain unchanged.
        """
        table = np.asarray(fisher_table, dtype=float)
        if table.shape != (2, 2) or (table < 0).any():
            return np.nan, np.nan

        # A zero cell makes log(OR) and its standard error undefined.  The
        # Haldane-Anscombe correction gives a finite, conventional approximate CI.
        ci_table = table + 0.5 if (table == 0).any() else table
        a, b = ci_table[0]
        c, d = ci_table[1]

        if min(a, b, c, d) <= 0:
            return np.nan, np.nan

        ci_or = (a * d) / (b * c)
        log_or = math.log(ci_or)
        se_log_or = math.sqrt((1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d))
        z_975 = 1.959963984540054

        low_log = log_or - z_975 * se_log_or
        high_log = log_or + z_975 * se_log_or
        try:
            ci_low = math.exp(low_log)
        except OverflowError:
            ci_low = np.inf
        try:
            ci_high = math.exp(high_log)
        except OverflowError:
            ci_high = np.inf

        return float(ci_low), float(ci_high)

    threshold_keys = [f"{threshold:.1f}" for threshold in SIMILARITY_THRESHOLDS]
    solvable_ratios = {}
    nonsolvable_ratios = {}
    delta_ratios = {}
    odds_ratios = {}
    or_ci95_lows = {}
    or_ci95_highs = {}
    or_ci95_strings = {}
    fisher_p_values = {}
    audit_counts = []

    for threshold in SIMILARITY_THRESHOLDS:
        threshold_key = f"{threshold:.1f}"
        p_s, m_s, n_s = prevalence_at_threshold(
            solvable_df,
            threshold,
            "solvable",
        )
        p_n, m_n, n_n = prevalence_at_threshold(
            nonsolvable_df,
            threshold,
            "nonsolvable",
        )
        delta_p = (
            float(p_s - p_n)
            if not pd.isna(p_s) and not pd.isna(p_n)
            else np.nan
        )

        if n_s > 0 and n_n > 0:
            fisher_table = [
                [int(m_s), int(n_s - m_s)],
                [int(m_n), int(n_n - m_n)],
            ]
            fisher_result = fisher_exact(fisher_table, alternative="two-sided")
            odds_ratio = float(fisher_result.statistic)
            fisher_p = float(fisher_result.pvalue)
            or_ci95_low, or_ci95_high = compute_or_95_ci(fisher_table)
        else:
            odds_ratio = np.nan
            fisher_p = np.nan
            or_ci95_low = np.nan
            or_ci95_high = np.nan

        if pd.isna(or_ci95_low) or pd.isna(or_ci95_high):
            or_ci95_string = np.nan
        else:
            or_ci95_string = f"[{or_ci95_low:.6g}, {or_ci95_high:.6g}]"

        solvable_ratios[threshold_key] = p_s
        nonsolvable_ratios[threshold_key] = p_n
        delta_ratios[threshold_key] = delta_p
        odds_ratios[threshold_key] = odds_ratio
        or_ci95_lows[threshold_key] = or_ci95_low
        or_ci95_highs[threshold_key] = or_ci95_high
        or_ci95_strings[threshold_key] = or_ci95_string
        fisher_p_values[threshold_key] = fisher_p
        audit_counts.append((threshold_key, m_s, n_s, m_n, n_n))

    output_rows = [
        {"measure": "P(M=1|S)", **solvable_ratios},
        {"measure": "P(M=1|N)", **nonsolvable_ratios},
        {"measure": "deltaP", **delta_ratios},
        {"measure": "OR", **odds_ratios},
        {"measure": "OR 95% CI", **or_ci95_strings},
        {"measure": "pFisher", **fisher_p_values},
    ]
    output_columns = ["measure"] + threshold_keys
    output_df = pd.DataFrame(output_rows, columns=output_columns)

    output_path = (
        results_dir
        / "rq1_majority_agreement_prevalence_solvable_vs_nonsolvable.csv"
    )
    output_df.to_csv(output_path, index=False, na_rep="NA")

    # Paper-summary operating threshold t*.
    #
    # Selection rule:
    #   A. If one or more thresholds have raw two-sided pFisher < 0.05,
    #      select the threshold with the HIGHEST OR among all significant thresholds.
    #   B. If no threshold is significant, restrict to thresholds > 0.5 and select
    #      the threshold with the HIGHEST OR in that fallback set.
    #   C. Exact OR ties are broken by choosing the higher similarity threshold.
    #
    # For transparency, rq1_main_results.csv also records the significant thresholds
    # separately by OR direction (>1 versus <1).
    def choose_highest_or_candidate(candidates: list[dict]) -> Optional[dict]:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (item["odds_ratio"], item["threshold"]),
        )

    significant_candidates = []
    significant_thresholds_or_gt_1 = []
    significant_thresholds_or_lt_1 = []

    for threshold in SIMILARITY_THRESHOLDS:
        threshold_key = f"{threshold:.1f}"
        fisher_p = fisher_p_values.get(threshold_key, np.nan)
        odds_ratio = odds_ratios.get(threshold_key, np.nan)

        if pd.isna(fisher_p) or float(fisher_p) >= 0.05:
            continue
        if pd.isna(odds_ratio) or float(odds_ratio) < 0.0:
            continue

        odds_ratio = float(odds_ratio)
        threshold_value = float(threshold)
        significant_candidates.append({
            "threshold": threshold_value,
            "threshold_key": threshold_key,
            "odds_ratio": odds_ratio,
        })

        if odds_ratio > 1.0:
            significant_thresholds_or_gt_1.append(threshold_value)
        elif odds_ratio < 1.0:
            significant_thresholds_or_lt_1.append(threshold_value)

    selected_candidate = choose_highest_or_candidate(significant_candidates)

    if selected_candidate is None:
        fallback_candidates = []
        for threshold in SIMILARITY_THRESHOLDS:
            if float(threshold) <= 0.5:
                continue

            threshold_key = f"{threshold:.1f}"
            odds_ratio = odds_ratios.get(threshold_key, np.nan)
            if pd.isna(odds_ratio) or float(odds_ratio) < 0.0:
                continue

            fallback_candidates.append({
                "threshold": float(threshold),
                "threshold_key": threshold_key,
                "odds_ratio": float(odds_ratio),
            })

        selected_candidate = choose_highest_or_candidate(fallback_candidates)

    if selected_candidate is None:
        # Extremely defensive fallback: if every OR is undefined even among
        # thresholds > 0.5, still choose the highest allowed fallback threshold.
        fallback_thresholds = [
            float(t) for t in SIMILARITY_THRESHOLDS
            if float(t) > 0.5
        ]
        fallback_threshold = (
            max(fallback_thresholds)
            if fallback_thresholds
            else max(float(t) for t in SIMILARITY_THRESHOLDS)
        )
        selected_candidate = {
            "threshold": fallback_threshold,
            "threshold_key": f"{fallback_threshold:.1f}",
        }

    selected_threshold_key = selected_candidate["threshold_key"]
    selected_threshold = selected_candidate["threshold"]
    selected_summary_row = {
        "Dataset": SET,
        "Model": MODEL,
        "Similarity": metric,
        "t*": selected_threshold,
        "P(M=1|S)": solvable_ratios[selected_threshold_key],
        "P(M=1|N)": nonsolvable_ratios[selected_threshold_key],
        "deltaP": delta_ratios[selected_threshold_key],
        "OR": odds_ratios[selected_threshold_key],
        "OR_CI95_low": or_ci95_lows[selected_threshold_key],
        "OR_CI95_high": or_ci95_highs[selected_threshold_key],
        "OR 95% CI": or_ci95_strings[selected_threshold_key],
        "pFisher": fisher_p_values[selected_threshold_key],
        "Significant thresholds OR>1": json.dumps(significant_thresholds_or_gt_1),
        "Significant thresholds OR<1": json.dumps(significant_thresholds_or_lt_1),
    }

    selected_summary_columns = [
        "Dataset",
        "Model",
        "Similarity",
        "t*",
        "P(M=1|S)",
        "P(M=1|N)",
        "deltaP",
        "OR",
        "OR_CI95_low",
        "OR_CI95_high",
        "OR 95% CI",
        "pFisher",
        "Significant thresholds OR>1",
        "Significant thresholds OR<1",
    ]
    selected_summary_df = pd.DataFrame(
        [selected_summary_row],
        columns=selected_summary_columns,
    )
    # Paper-ready RQ1 main result for this dataset/model/similarity configuration.
    selected_summary_path = results_dir / "rq1_main_results.csv"
    legacy_selected_summary_path = results_dir / "rq1_selected_threshold_summary.csv"
    if legacy_selected_summary_path.exists():
        legacy_selected_summary_path.unlink()
    selected_summary_df.to_csv(
        selected_summary_path,
        index=False,
        na_rep="NA",
    )

    print()
    print("=" * 80)
    print("RQ1 MAJORITY-AGREEMENT PREVALENCE: SOLVABLE VS NONSOLVABLE")
    print("=" * 80)
    print(f"Agreement definition: {agreement_definition}")
    for threshold_key, m_s, n_s, m_n, n_n in audit_counts:
        print(
            f"  {threshold_key}: solvable={m_s}/{n_s}, "
            f"nonsolvable={m_n}/{n_n}"
        )
    print(output_df.to_string(index=False))
    print(f"RQ1 CSV: {output_path}")
    print("RQ1 main results:")
    print(selected_summary_df.to_string(index=False))
    print(f"RQ1 main-results CSV: {selected_summary_path}")
    print("=" * 80)

    return output_df

def main():
    remove_legacy_rq2_outputs()

    if not PRED_ROOT.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {PRED_ROOT}")

    # Phase 1: compute and save pairwise candidate similarity plus ground-truth
    # similarity before consulting syntax status, SWE-bench outcomes, subset class,
    # or clustering validity. Existing MATRIX_ROOT/<instance_id> folders are the
    # only cache skip.
    precompute_similarity_files_before_filtering()

    # Phase 2: load validity/test information and create the two disjoint RQ2 groups:
    #   solvable    = at least one passing generated candidate;
    #   nonsolvable = zero passing generated candidates.
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV does not exist: {RESULTS_CSV}")

    if not SYNTAX_STATUS_CSV.exists():
        raise FileNotFoundError(f"Syntax status CSV does not exist: {SYNTAX_STATUS_CSV}")

    results_df = pd.read_csv(RESULTS_CSV)
    syntax_df = pd.read_csv(SYNTAX_STATUS_CSV)

    if "instance_id" not in syntax_df.columns:
        raise ValueError(f"Missing instance_id column in {SYNTAX_STATUS_CSV}")
    syntax_df["instance_id"] = syntax_df["instance_id"].astype(str).str.strip()

    if "instance_id" not in results_df.columns:
        raise ValueError(f"Missing instance_id column in {RESULTS_CSV}")
    results_df["instance_id"] = results_df["instance_id"].astype(str).str.strip()

    result_columns = get_result_columns_for_prompt_type(
        results_df=results_df,
        prompt_type=PROMPT_TYPE,
    )

    if not result_columns:
        raise ValueError(
            f"No result columns found for prompt_type={PROMPT_TYPE}. "
            f"Expected columns like Replacement_Temp_0.0_{PROMPT_TYPE}"
        )

    solvable_mask = results_df.apply(
        lambda row: row_has_at_least_one_pass(row, result_columns),
        axis=1,
    )
    nonsolvable_mask = ~solvable_mask

    solvable_results_df = results_df[solvable_mask].copy()
    nonsolvable_results_df = results_df[nonsolvable_mask].copy()

    if len(solvable_results_df) + len(nonsolvable_results_df) != len(results_df):
        raise AssertionError("Solvable/nonsolvable partition does not cover all result rows")

    paths_by_subset = {
        "nonsolvable": make_output_paths("nonsolvable"),
        "solvable": make_output_paths("solvable"),
    }
    report_syntax_status_summary(
        syntax_df=syntax_df,
        paths_by_subset=paths_by_subset,
    )
    report_passing_status_summary(
        results_df=results_df,
        result_columns=result_columns,
        paths_by_subset=paths_by_subset,
    )

    num_issues = len(results_df)
    num_total_available_candidates = num_issues * len(TEMPERATURES)
    num_passing_candidates = int(
        results_df[result_columns]
        .applymap(normalize_test_result)
        .sum()
        .sum()
    )

    print("=" * 80)
    print("RESULTS CSV BASIC COUNTS")
    print("=" * 80)
    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Prompt type: {PROMPT_TYPE}")
    print(f"Number of issues in CSV: {num_issues}")
    print(f"Number of temperature columns found: {len(result_columns)}")
    print(f"Solvable issues (>=1 passing candidate): {len(solvable_results_df)}")
    print(f"Nonsolvable issues (0 passing candidates): {len(nonsolvable_results_df)}")
    print(f"Total available candidates, issues * 11: {num_total_available_candidates}")
    print(f"Number of passing candidates: {num_passing_candidates}")
    print("=" * 80)

    (
        nonsolvable_threshold_summary_df,
        nonsolvable_temperature_coclustering_df,
        nonsolvable_cluster_hit_advantage_df,
    ) = run_clustering_analysis_for_subset(
        subset_name="nonsolvable",
        subset_results_df=nonsolvable_results_df,
        full_results_df=results_df,
        syntax_df=syntax_df,
    )

    (
        solvable_threshold_summary_df,
        solvable_temperature_coclustering_df,
        solvable_cluster_hit_advantage_df,
    ) = run_clustering_analysis_for_subset(
        subset_name="solvable",
        subset_results_df=solvable_results_df,
        full_results_df=results_df,
        syntax_df=syntax_df,
    )

    # Also save an all-issues version for the agreement-as-correctness-signal RQ.
    # This preserves the existing solvable/nonsolvable outputs while giving the RQ
    # one combined population when desired.
    all_issues_cluster_hit_advantage_df = pd.concat(
        [
            nonsolvable_cluster_hit_advantage_df,
            solvable_cluster_hit_advantage_df,
        ],
        ignore_index=True,
    )
    all_issues_threshold_summary_df = pd.concat(
        [
            nonsolvable_threshold_summary_df,
            solvable_threshold_summary_df,
        ],
        ignore_index=True,
    )
    save_cluster_hit_advantage_outputs(
        cluster_hit_advantage_df=all_issues_cluster_hit_advantage_df,
        threshold_summary_df=all_issues_threshold_summary_df,
        output_dir=(BASE_OUTPUT_DIR / "all_issues" / "issue_f1_by_threshold"),
    )

    nonsolvable_threshold_average_df = compute_threshold_average_metrics(
        threshold_summary_df=nonsolvable_threshold_summary_df,
    )
    solvable_threshold_average_df = compute_threshold_average_metrics(
        threshold_summary_df=solvable_threshold_summary_df,
    )

    results_definition_dirs = [
        BASE_OUTPUT_DIR / "Results" / "1-Majority",
        BASE_OUTPUT_DIR / "Results" / "2-Largest",
    ]

    # RQ1: compare how often majority agreement occurs in candidate sets with
    # versus without a correct generated fix. Save the same table separately for
    # the two existing agreement definitions.
    rq1_definition_specs = [
        ("majority", results_definition_dirs[0]),
        ("largest", results_definition_dirs[1]),
    ]
    for agreement_definition, definition_results_dir in rq1_definition_specs:
        save_rq1_majority_agreement_prevalence(
            solvable_threshold_summary_df=solvable_threshold_summary_df,
            nonsolvable_threshold_summary_df=nonsolvable_threshold_summary_df,
            results_dir=definition_results_dir,
            agreement_definition=agreement_definition,
        )
        save_rq2_main_results_at_rq1_threshold(
            results_dir=definition_results_dir,
        )
        save_rq3_main_results_at_rq1_threshold(
            results_dir=definition_results_dir,
        )

    plot_nonsolvable_vs_solvable_f1(
        nonsolvable_threshold_average_df=nonsolvable_threshold_average_df,
        solvable_threshold_average_df=solvable_threshold_average_df,
        output_path=(
            BASE_OUTPUT_DIR / "threshold_average_f1_nonsolvable_vs_solvable.pdf"
        ),
    )

    plot_nonsolvable_vs_solvable_oracle_f1(
        nonsolvable_threshold_average_df=nonsolvable_threshold_average_df,
        solvable_threshold_average_df=solvable_threshold_average_df,
        output_path=(
            BASE_OUTPUT_DIR / "threshold_oracle_f1_nonsolvable_vs_solvable.pdf"
        ),
    )

    oracle_f1_nonsolvable_solvable_df = pd.DataFrame({
        "threshold": SIMILARITY_THRESHOLDS,
        "nonsolvable_oracle_average_f1": [
            float(
                nonsolvable_threshold_average_df.loc[
                    np.isclose(
                        nonsolvable_threshold_average_df["threshold"].astype(float),
                        float(threshold),
                    ),
                    "average_oracle_f1",
                ].iloc[0]
            )
            if (
                not nonsolvable_threshold_average_df.empty
                and "average_oracle_f1" in nonsolvable_threshold_average_df.columns
                and np.isclose(
                    nonsolvable_threshold_average_df["threshold"].astype(float),
                    float(threshold),
                ).any()
            )
            else np.nan
            for threshold in SIMILARITY_THRESHOLDS
        ],
        "solvable_oracle_average_f1": [
            float(
                solvable_threshold_average_df.loc[
                    np.isclose(
                        solvable_threshold_average_df["threshold"].astype(float),
                        float(threshold),
                    ),
                    "average_oracle_f1",
                ].iloc[0]
            )
            if (
                not solvable_threshold_average_df.empty
                and "average_oracle_f1" in solvable_threshold_average_df.columns
                and np.isclose(
                    solvable_threshold_average_df["threshold"].astype(float),
                    float(threshold),
                ).any()
            )
            else np.nan
            for threshold in SIMILARITY_THRESHOLDS
        ],
    })
    oracle_f1_nonsolvable_solvable_df.to_csv(
        BASE_OUTPUT_DIR / "oracle_f1_nonsolvable_vs_solvable_by_threshold.csv",
        index=False,
    )

    plot_nonsolvable_vs_solvable_strategy_availability(
        nonsolvable_threshold_average_df=nonsolvable_threshold_average_df,
        solvable_threshold_average_df=solvable_threshold_average_df,
        output_path=(
            BASE_OUTPUT_DIR
            / "threshold_strategy_availability_nonsolvable_vs_solvable.pdf"
        ),
    )

    average_cluster_size_nonsolvable_solvable_df = (
        build_average_cluster_size_nonsolvable_solvable_table(
            nonsolvable_threshold_average_df=nonsolvable_threshold_average_df,
            solvable_threshold_average_df=solvable_threshold_average_df,
        )
    )
    average_cluster_size_nonsolvable_solvable_df.to_csv(
        BASE_OUTPUT_DIR / "average_cluster_size_nonsolvable_solvable_table.csv",
        index=False,
    )

    plot_nonsolvable_vs_solvable_cluster_size(
        nonsolvable_threshold_average_df=nonsolvable_threshold_average_df,
        solvable_threshold_average_df=solvable_threshold_average_df,
        output_path=(
            BASE_OUTPUT_DIR
            / "threshold_average_cluster_size_nonsolvable_vs_solvable.pdf"
        ),
    )

    close_temperature_coclustering_table_df = build_two_row_threshold_table(
        nonsolvable_threshold_average_df=nonsolvable_temperature_coclustering_df,
        solvable_threshold_average_df=solvable_temperature_coclustering_df,
        value_column="issue_average_close_temperature_coclustering_rate",
    )
    close_temperature_coclustering_table_df.to_csv(
        BASE_OUTPUT_DIR
        / "threshold_close_temperature_coclustering_nonsolvable_vs_solvable.csv",
        index=False,
    )

    far_temperature_coclustering_table_df = build_two_row_threshold_table(
        nonsolvable_threshold_average_df=nonsolvable_temperature_coclustering_df,
        solvable_threshold_average_df=solvable_temperature_coclustering_df,
        value_column="issue_average_far_temperature_coclustering_rate",
    )
    far_temperature_coclustering_table_df.to_csv(
        BASE_OUTPUT_DIR
        / "threshold_far_temperature_coclustering_nonsolvable_vs_solvable.csv",
        index=False,
    )

    close_minus_far_coclustering_table_df = build_two_row_threshold_table(
        nonsolvable_threshold_average_df=nonsolvable_temperature_coclustering_df,
        solvable_threshold_average_df=solvable_temperature_coclustering_df,
        value_column="issue_average_close_minus_far_coclustering_rate",
    )
    close_minus_far_coclustering_table_df.to_csv(
        BASE_OUTPUT_DIR
        / "threshold_close_minus_far_temperature_coclustering_nonsolvable_vs_solvable.csv",
        index=False,
    )

    plot_nonsolvable_vs_solvable_temperature_coclustering_delta(
        nonsolvable_coclustering_df=nonsolvable_temperature_coclustering_df,
        solvable_coclustering_df=solvable_temperature_coclustering_df,
        output_path=(
            BASE_OUTPUT_DIR
            / "threshold_close_minus_far_temperature_coclustering_nonsolvable_vs_solvable.pdf"
        ),
    )

    print_average_f1_latex_table(
        nonsolvable_threshold_average_df=nonsolvable_threshold_average_df,
        solvable_threshold_average_df=solvable_threshold_average_df,
        output_path=(
            BASE_OUTPUT_DIR / "average_f1_nonsolvable_solvable_table.tex"
        ),
    )

    print_solvable_paired_f1_latex_table(
        solvable_threshold_average_df=solvable_threshold_average_df,
        output_path=(
            BASE_OUTPUT_DIR
            / "solvable_strategy_vs_matched_oracle_f1_table.tex"
        ),
    )


if __name__ == "__main__":
    main()
