#!/usr/bin/env python3

from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from codebleu import calc_codebleu


REPO_TO_CODEBLEU_LANG = {
    # SWE-bench Verified / Python SWE-bench repos
    "astropy/astropy": "python",
    "django/django": "python",
    "matplotlib/matplotlib": "python",
    "mwaskom/seaborn": "python",
    "pallets/flask": "python",
    "psf/requests": "python",
    "pydata/xarray": "python",
    "pylint-dev/pylint": "python",
    "pytest-dev/pytest": "python",
    "scikit-learn/scikit-learn": "python",
    "sphinx-doc/sphinx": "python",
    "sympy/sympy": "python",

    # SWE-bench Pro repos
    "ansible/ansible": "python",
    "element-hq/element-web": "javascript",
    "flipt-io/flipt": "go",
    "future-architect/vuls": "go",
    "gravitational/teleport": "go",
    "internetarchive/openlibrary": "python",
    "navidrome/navidrome": "go",
    "NodeBB/NodeBB": "javascript",
    "ProtonMail/WebClients": "javascript",
    "qutebrowser/qutebrowser": "python",
    "tutao/tutanota": "javascript",
}

DEFAULT_CODEBLEU_LANG = "python"

# Optional diagnostic transformation. The main CodeBLEU similarity used by
# clustering remains RAW/untransformed in similarity_matrix.csv. This scale is
# used only for transformed_similarity_matrix.csv and its ground-truth analogue.
CODEBLEU_DISTANCE_SCALE = 0.5

# Additional decomposition/diagnostic CSVs.
JACCARD_MATRIX_FILENAME = "jaccard_similarity_matrix.csv"
JACCARD_GROUND_TRUTH_FILENAME = "ground_truth_jaccard_similarity.csv"
SHARED_ONLY_MATRIX_FILENAME = "shared_files_only_similarity_matrix.csv"
SHARED_ONLY_GROUND_TRUTH_FILENAME = "ground_truth_shared_files_only_similarity.csv"
TRANSFORMED_MATRIX_FILENAME = "transformed_similarity_matrix.csv"
TRANSFORMED_GROUND_TRUTH_FILENAME = "ground_truth_transformed_similarity.csv"


def get_lang_for_repo(
    repo: Optional[str],
    default_lang: str = DEFAULT_CODEBLEU_LANG,
) -> str:
    """Return the CodeBLEU language for a SWE-bench repository."""
    if repo is None:
        return default_lang

    repo = str(repo).strip()
    return REPO_TO_CODEBLEU_LANG.get(repo, default_lang)


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------


def _normalize_diff_path(path_text: Optional[str]) -> Optional[str]:
    """Normalize a Git/unified-diff path to a repository-relative file path."""
    if path_text is None:
        return None

    path = str(path_text).strip()

    # Unified-diff headers can include timestamps after a tab.
    if "\t" in path:
        path = path.split("\t", 1)[0].strip()

    # Remove simple surrounding quotes if present.
    if len(path) >= 2 and path[0] == path[-1] == '"':
        path = path[1:-1]

    if path in {"/dev/null", "dev/null", ""}:
        return None

    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]

    return path or None


def _parse_diff_git_paths(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse source/target paths from a `diff --git a/... b/...` header."""
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()

    if len(tokens) >= 4 and tokens[0] == "diff" and tokens[1] == "--git":
        return _normalize_diff_path(tokens[2]), _normalize_diff_path(tokens[3])

    return None, None


def extract_file_changes_from_diff(diff_text: str) -> pd.DataFrame:
    """
    Parse a Git/unified diff into one row per changed file.

    Returned columns:
        file
        removed
        added

    All removed lines from all hunks of the same file are concatenated in their
    original order. The same is done for added lines. Unchanged context, Git
    metadata, index lines, and hunk-position metadata are not included.
    """
    text = str(diff_text or "")

    if not text.strip():
        return pd.DataFrame(columns=["file", "removed", "added"])

    lines = text.splitlines()

    # file -> {"removed": [...], "added": [...]}
    changes: Dict[str, Dict[str, list]] = {}

    current_source_path: Optional[str] = None
    current_target_path: Optional[str] = None
    current_file: Optional[str] = None
    in_hunk = False

    def ensure_current_file() -> Optional[str]:
        nonlocal current_file

        if current_file is None:
            current_file = current_target_path or current_source_path

        if current_file is not None and current_file not in changes:
            changes[current_file] = {
                "removed": [],
                "added": [],
            }

        return current_file

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("diff --git "):
            current_source_path, current_target_path = _parse_diff_git_paths(line)
            current_file = current_target_path or current_source_path
            in_hunk = False
            ensure_current_file()
            i += 1
            continue

        # Recognize standard unified-diff file headers. The look-ahead keeps this
        # from confusing most code lines that merely begin with dashes.
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            source_header = _normalize_diff_path(line[4:])
            target_header = _normalize_diff_path(lines[i + 1][4:])

            current_source_path = source_header or current_source_path
            current_target_path = target_header
            current_file = current_target_path or current_source_path
            in_hunk = False
            ensure_current_file()

            i += 2
            continue

        if line.startswith("@@"):
            in_hunk = True
            ensure_current_file()
            i += 1
            continue

        if in_hunk:
            # Metadata emitted by Git for a missing terminal newline.
            if line.startswith("\\ No newline at end of file"):
                i += 1
                continue

            file_name = ensure_current_file()

            if file_name is not None:
                if line.startswith("+") and not line.startswith("+++"):
                    changes[file_name]["added"].append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    changes[file_name]["removed"].append(line[1:])
                # Unchanged context lines intentionally contribute nothing.

        i += 1

    rows = []
    for file_name, data in changes.items():
        rows.append(
            {
                "file": file_name,
                "removed": "\n".join(data["removed"]),
                "added": "\n".join(data["added"]),
            }
        )

    return pd.DataFrame(rows, columns=["file", "removed", "added"])


def _file_change_mapping(diff_text: str) -> Dict[str, Dict[str, str]]:
    """Return parsed file changes as a dictionary keyed by file path."""
    df = extract_file_changes_from_diff(diff_text)

    if df.empty:
        return {}

    return {
        str(row["file"]): {
            "removed": str(row["removed"] or ""),
            "added": str(row["added"] or ""),
        }
        for _, row in df.iterrows()
    }


def clean_diff_for_codebleu(diff_text: str) -> str:
    """
    Backward-compatible helper that returns changed code without Git metadata.

    The new whole-patch metric does not use this flattened representation; it
    preserves file boundaries and keeps removed/added channels separate.
    """
    df = extract_file_changes_from_diff(diff_text)

    cleaned_parts = []
    for _, row in df.iterrows():
        removed = str(row["removed"] or "")
        added = str(row["added"] or "")

        if removed.strip():
            cleaned_parts.append(removed)
        if added.strip():
            cleaned_parts.append(added)

    return "\n".join(cleaned_parts)


# ---------------------------------------------------------------------------
# CodeBLEU on code fragments
# ---------------------------------------------------------------------------


def codebleu_score(
    candidate: str,
    reference: str,
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> float:
    """
    Calculate directional CodeBLEU between two already-extracted code fragments.

    This function intentionally does NOT parse or clean a Git diff. Whole-patch
    parsing is performed before this function is called so removed code is only
    compared with removed code and added code only with added code.
    """
    candidate = str(candidate or "")
    reference = str(reference or "")

    if not candidate.strip() and not reference.strip():
        return 1.0

    if not candidate.strip() or not reference.strip():
        return 0.0

    try:
        result = calc_codebleu(
            references=[[reference]],
            predictions=[candidate],
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25),
        )
        return float(result["codebleu"])

    except Exception as exc:
        print(
            "WARNING: CodeBLEU failed for one code-fragment pair "
            f"with lang={lang}: {exc}"
        )
        return 0.0


def _symmetric_codebleu_fragments(
    code_a: str,
    code_b: str,
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> float:
    """Calculate symmetric CodeBLEU between two extracted code fragments."""
    return (
        codebleu_score(code_a, code_b, lang=lang)
        + codebleu_score(code_b, code_a, lang=lang)
    ) / 2.0


def _channel_similarity(
    channel_a: str,
    channel_b: str,
    lang: str,
) -> Optional[float]:
    """
    Compare one change channel (removed or added).

    Rules:
      * nonempty vs nonempty -> symmetric CodeBLEU
      * nonempty vs empty    -> 0.0
      * empty vs nonempty    -> 0.0
      * empty vs empty       -> None (exclude this channel from the file mean)
    """
    text_a = str(channel_a or "")
    text_b = str(channel_b or "")

    has_a = bool(text_a.strip())
    has_b = bool(text_b.strip())

    if not has_a and not has_b:
        return None

    if has_a != has_b:
        return 0.0

    return float(_symmetric_codebleu_fragments(text_a, text_b, lang=lang))


def shared_file_similarity(
    file_change_a: Dict[str, str],
    file_change_b: Dict[str, str],
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> float:
    """
    Calculate similarity for one file that is present in both patches.

    The removed-code and added-code similarities are averaged over the channels
    that are defined. A channel that is empty in both patches is excluded. A
    channel that is present in only one patch contributes 0.
    """
    component_scores = []

    removed_score = _channel_similarity(
        file_change_a.get("removed", ""),
        file_change_b.get("removed", ""),
        lang=lang,
    )
    if removed_score is not None:
        component_scores.append(float(removed_score))

    added_score = _channel_similarity(
        file_change_a.get("added", ""),
        file_change_b.get("added", ""),
        lang=lang,
    )
    if added_score is not None:
        component_scores.append(float(added_score))

    if not component_scores:
        # No textual added/removed code was available for this shared file.
        # Do not reward the file merely because its path is shared.
        return 0.0

    return float(sum(component_scores) / len(component_scores))



def _transform_shared_file_score(
    raw_shared_file_score: float,
    scale: float = CODEBLEU_DISTANCE_SCALE,
) -> float:
    """Apply the requested exponential transform to one SHARED-file score only."""
    if scale <= 0:
        raise ValueError("CodeBLEU transform scale must be positive")

    return float(
        math.exp(-(1.0 - float(raw_shared_file_score)) / float(scale))
    )


def _patch_similarity_decomposition(
    diff_a: str,
    diff_b: str,
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> Tuple[float, float, float, float]:
    """Return raw union, transformed union, file Jaccard, and raw shared-only.

    Returns:
        raw_union_similarity:
            The EXISTING CodeBLEU score used by similarity_matrix.csv. For each
            shared file, removed/added CodeBLEU channels are averaged under the
            existing empty-channel rules. Unshared files contribute literal 0,
            then scores are averaged over the union of changed files.

        transformed_union_similarity:
            The same structure, except each RAW shared-file score c is first
            transformed with exp(-(1-c)/0.5). Unshared files remain literal 0.
            This is saved only as transformed_similarity_matrix.csv.

        jaccard_file_similarity:
            |shared changed files| / |union changed files|.

        raw_shared_only_similarity:
            Mean of the RAW, untransformed shared-file CodeBLEU scores over
            shared files only.

    For non-empty file unions:
        raw_union_similarity ==
        jaccard_file_similarity * raw_shared_only_similarity
    up to floating-point precision.
    """
    text_a = str(diff_a or "")
    text_b = str(diff_b or "")

    # Preserve the existing whole-patch empty behavior. For the diagnostic
    # decomposition, treat two empty patches as identical in all four outputs.
    if not text_a.strip() and not text_b.strip():
        return 1.0, 1.0, 1.0, 1.0

    if not text_a.strip() or not text_b.strip():
        return 0.0, 0.0, 0.0, 0.0

    files_a = _file_change_mapping(text_a)
    files_b = _file_change_mapping(text_b)

    union_files = sorted(set(files_a) | set(files_b))
    if not union_files:
        return 0.0, 0.0, 0.0, 0.0

    shared_files = sorted(set(files_a) & set(files_b))
    union_count = len(union_files)
    shared_count = len(shared_files)

    jaccard_file_similarity = float(shared_count / union_count)

    raw_shared_scores = []
    transformed_shared_scores = []

    for file_name in shared_files:
        raw_score = shared_file_similarity(
            file_change_a=files_a[file_name],
            file_change_b=files_b[file_name],
            lang=lang,
        )
        raw_shared_scores.append(float(raw_score))
        transformed_shared_scores.append(
            _transform_shared_file_score(raw_score)
        )

    if shared_count == 0:
        raw_shared_only_similarity = 0.0
        raw_union_similarity = 0.0
        transformed_union_similarity = 0.0
    else:
        # Raw/untransformed similarity averaged ONLY over shared files.
        raw_shared_only_similarity = float(
            sum(raw_shared_scores) / shared_count
        )

        # Main CodeBLEU score used by clustering: raw shared-file scores plus
        # literal zeros for unshared files, averaged over the file union.
        raw_union_similarity = float(
            sum(raw_shared_scores) / union_count
        )

        # Diagnostic transformed union: transform each shared-file score first,
        # keep unshared files at literal zero, then average over the file union.
        transformed_union_similarity = float(
            sum(transformed_shared_scores) / union_count
        )

    if not math.isclose(
        raw_union_similarity,
        jaccard_file_similarity * raw_shared_only_similarity,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise AssertionError(
            "CodeBLEU raw union decomposition failed: "
            "similarity_matrix != Jaccard * raw shared-files-only similarity"
        )

    return (
        raw_union_similarity,
        transformed_union_similarity,
        jaccard_file_similarity,
        raw_shared_only_similarity,
    )


def whole_patch_similarity(
    diff_a: str,
    diff_b: str,
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> float:
    """
    Calculate one CodeBLEU-based similarity score for two complete patches.

    Procedure:
      1. Parse each diff into files with removed and added code.
      2. Take the UNION of changed file paths.
      3. For a shared file, compare removed-to-removed and added-to-added and
         average the available channel scores.
      4. A file present in only one patch receives similarity 0.
      5. Average file-level similarities over the union of changed files.

    Therefore patches that make similar edits to the same files score highly,
    while patches that touch different files are penalized in the final score.
    """
    text_a = str(diff_a or "")
    text_b = str(diff_b or "")

    if not text_a.strip() and not text_b.strip():
        return 1.0

    if not text_a.strip() or not text_b.strip():
        return 0.0

    files_a = _file_change_mapping(text_a)
    files_b = _file_change_mapping(text_b)

    union_files = sorted(set(files_a) | set(files_b))

    if not union_files:
        print(
            "WARNING: could not extract any changed file from one patch pair; "
            "returning whole-patch similarity 0.0"
        )
        return 0.0

    file_scores = []

    for file_name in union_files:
        if file_name not in files_a or file_name not in files_b:
            # File-scope disagreement: this file is changed by only one patch.
            file_scores.append(0.0)
            continue

        file_scores.append(
            shared_file_similarity(
                file_change_a=files_a[file_name],
                file_change_b=files_b[file_name],
                lang=lang,
            )
        )

    return float(sum(file_scores) / len(file_scores))


# Alias with an explicit name for callers/tests that want a "two diffs" method.
def compare_two_diffs(
    diff_a: str,
    diff_b: str,
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> float:
    """Return the final whole-patch similarity between two diffs."""
    return whole_patch_similarity(diff_a, diff_b, lang=lang)


def symmetric_codebleu(
    diff_a: str,
    diff_b: str,
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> float:
    """
    Backward-compatible whole-diff entry point.

    Unlike the old implementation, this now uses the file-aware removed/added
    aggregation implemented by whole_patch_similarity().
    """
    return whole_patch_similarity(diff_a, diff_b, lang=lang)


# ---------------------------------------------------------------------------
# Pairwise matrix
# ---------------------------------------------------------------------------


def build_similarity_matrix(
    temperature_to_diff: dict,
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> pd.DataFrame:
    """Build the pairwise whole-patch CodeBLEU similarity matrix."""
    temperature_keys = list(temperature_to_diff.keys())
    valid_temperatures = [str(temp) for temp in temperature_keys]

    similarity_df = pd.DataFrame(
        0.0,
        index=valid_temperatures,
        columns=valid_temperatures,
        dtype=float,
    )

    for i, temp_key_i in enumerate(temperature_keys):
        temp_i = valid_temperatures[i]
        similarity_df.loc[temp_i, temp_i] = 1.0
        diff_i = temperature_to_diff[temp_key_i]

        for j in range(i + 1, len(temperature_keys)):
            temp_key_j = temperature_keys[j]
            temp_j = valid_temperatures[j]
            diff_j = temperature_to_diff[temp_key_j]

            score = whole_patch_similarity(
                diff_i,
                diff_j,
                lang=lang,
            )

            similarity_df.loc[temp_i, temp_j] = float(score)
            similarity_df.loc[temp_j, temp_i] = float(score)

    return similarity_df


def save_similarity_matrix(
    similarity_df: pd.DataFrame,
    matrix_root: Path,
    instance_id: str,
    metric_name: str = "codebleu",
    filename: str = "similarity_matrix.csv",
) -> Path:
    """Save the generated-patch similarity matrix."""
    output_dir = Path(matrix_root) / str(instance_id) / metric_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    similarity_df.to_csv(output_path)

    return output_path


def build_and_save_similarity_matrix(
    temperature_to_diff: dict,
    matrix_root: Path,
    instance_id: str,
    lang: str = DEFAULT_CODEBLEU_LANG,
    metric_name: str = "codebleu",
    filename: str = "similarity_matrix.csv",
) -> tuple[pd.DataFrame, Path]:
    """Build/save raw CodeBLEU plus requested decomposition diagnostics.

    IMPORTANT:
      * similarity_matrix.csv remains the RAW/untransformed union score used by
        the clustering pipeline.
      * transformed_similarity_matrix.csv contains the 0.5 exponential transform
        applied independently to each shared-file score before union averaging.
      * shared_files_only_similarity_matrix.csv contains RAW shared-file scores
        averaged only over shared files.
      * jaccard_similarity_matrix.csv contains shared-file-count / union-file-count.
    """
    if not temperature_to_diff:
        raise ValueError("temperature_to_diff is empty")

    temperature_keys = list(temperature_to_diff.keys())
    temperatures = [str(temp) for temp in temperature_keys]

    raw_union_df = pd.DataFrame(
        0.0,
        index=temperatures,
        columns=temperatures,
        dtype=float,
    )
    transformed_union_df = pd.DataFrame(
        0.0,
        index=temperatures,
        columns=temperatures,
        dtype=float,
    )
    jaccard_df = pd.DataFrame(
        0.0,
        index=temperatures,
        columns=temperatures,
        dtype=float,
    )
    shared_only_df = pd.DataFrame(
        0.0,
        index=temperatures,
        columns=temperatures,
        dtype=float,
    )

    for i, temp_key_i in enumerate(temperature_keys):
        temp_i = temperatures[i]
        diff_i = temperature_to_diff[temp_key_i]

        # A patch compared with itself is identical for every representation.
        raw_union_df.loc[temp_i, temp_i] = 1.0
        transformed_union_df.loc[temp_i, temp_i] = 1.0
        jaccard_df.loc[temp_i, temp_i] = 1.0
        shared_only_df.loc[temp_i, temp_i] = 1.0

        for j in range(i + 1, len(temperature_keys)):
            temp_key_j = temperature_keys[j]
            temp_j = temperatures[j]
            diff_j = temperature_to_diff[temp_key_j]

            (
                raw_union_score,
                transformed_union_score,
                jaccard_score,
                raw_shared_only_score,
            ) = _patch_similarity_decomposition(
                diff_a=diff_i,
                diff_b=diff_j,
                lang=lang,
            )

            for df, value in [
                (raw_union_df, raw_union_score),
                (transformed_union_df, transformed_union_score),
                (jaccard_df, jaccard_score),
                (shared_only_df, raw_shared_only_score),
            ]:
                df.loc[temp_i, temp_j] = float(value)
                df.loc[temp_j, temp_i] = float(value)

    output_dir = Path(matrix_root) / str(instance_id) / metric_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main clustering input: RAW/untransformed CodeBLEU union similarity.
    output_path = output_dir / filename
    raw_union_df.to_csv(output_path)

    # Requested additional diagnostics.
    transformed_union_df.to_csv(output_dir / TRANSFORMED_MATRIX_FILENAME)
    jaccard_df.to_csv(output_dir / JACCARD_MATRIX_FILENAME)
    shared_only_df.to_csv(output_dir / SHARED_ONLY_MATRIX_FILENAME)

    return raw_union_df, output_path


# ---------------------------------------------------------------------------
# Ground-truth similarity
# ---------------------------------------------------------------------------


def build_ground_truth_similarity_vector(
    ground_truth_patch: str,
    temperature_to_diff: dict,
    lang: str = DEFAULT_CODEBLEU_LANG,
) -> pd.DataFrame:
    """
    Build a one-row DataFrame of whole-patch similarities between the benchmark
    ground-truth patch and every generated patch.
    """
    temperature_keys = list(temperature_to_diff.keys())
    valid_temperatures = [str(temp) for temp in temperature_keys]

    scores = []
    for temp_key in temperature_keys:
        generated_patch = temperature_to_diff[temp_key]
        score = whole_patch_similarity(
            ground_truth_patch,
            generated_patch,
            lang=lang,
        )
        scores.append(float(score))

    return pd.DataFrame(
        [scores],
        index=["ground_truth"],
        columns=valid_temperatures,
    )


def save_ground_truth_similarity_vector(
    ground_truth_similarity_df: pd.DataFrame,
    matrix_root: Path,
    instance_id: str,
    metric_name: str = "codebleu",
    filename: str = "ground_truth_similarity.csv",
) -> Path:
    """Save the one-row ground-truth similarity CSV."""
    output_dir = Path(matrix_root) / str(instance_id) / metric_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    ground_truth_similarity_df.to_csv(output_path)

    return output_path


def build_and_save_ground_truth_similarity_vector(
    ground_truth_patch: str,
    temperature_to_diff: dict,
    matrix_root: Path,
    instance_id: str,
    lang: str = DEFAULT_CODEBLEU_LANG,
    metric_name: str = "codebleu",
    filename: str = "ground_truth_similarity.csv",
) -> tuple[pd.DataFrame, Path]:
    """Build/save raw ground-truth CodeBLEU plus requested diagnostics.

    ground_truth_similarity.csv remains RAW/untransformed so existing callers and
    clustering behavior stay unchanged. The transformed ground-truth analogue is
    saved separately as ground_truth_transformed_similarity.csv.
    """
    if not temperature_to_diff:
        raise ValueError("temperature_to_diff is empty")

    temperature_keys = list(temperature_to_diff.keys())
    temperatures = [str(temp) for temp in temperature_keys]

    raw_union_scores = []
    transformed_union_scores = []
    jaccard_scores = []
    raw_shared_only_scores = []

    for temp_key in temperature_keys:
        generated_patch = temperature_to_diff[temp_key]

        (
            raw_union_score,
            transformed_union_score,
            jaccard_score,
            raw_shared_only_score,
        ) = _patch_similarity_decomposition(
            diff_a=ground_truth_patch,
            diff_b=generated_patch,
            lang=lang,
        )

        raw_union_scores.append(float(raw_union_score))
        transformed_union_scores.append(float(transformed_union_score))
        jaccard_scores.append(float(jaccard_score))
        raw_shared_only_scores.append(float(raw_shared_only_score))

    raw_ground_truth_df = pd.DataFrame(
        [raw_union_scores],
        index=["ground_truth"],
        columns=temperatures,
    )
    transformed_ground_truth_df = pd.DataFrame(
        [transformed_union_scores],
        index=["ground_truth"],
        columns=temperatures,
    )
    jaccard_ground_truth_df = pd.DataFrame(
        [jaccard_scores],
        index=["ground_truth"],
        columns=temperatures,
    )
    shared_only_ground_truth_df = pd.DataFrame(
        [raw_shared_only_scores],
        index=["ground_truth"],
        columns=temperatures,
    )

    output_dir = Path(matrix_root) / str(instance_id) / metric_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main ground-truth file remains RAW/untransformed.
    output_path = output_dir / filename
    raw_ground_truth_df.to_csv(output_path)

    # Requested ground-truth diagnostics.
    transformed_ground_truth_df.to_csv(
        output_dir / TRANSFORMED_GROUND_TRUTH_FILENAME
    )
    jaccard_ground_truth_df.to_csv(
        output_dir / JACCARD_GROUND_TRUTH_FILENAME
    )
    shared_only_ground_truth_df.to_csv(
        output_dir / SHARED_ONLY_GROUND_TRUTH_FILENAME
    )

    return raw_ground_truth_df, output_path

