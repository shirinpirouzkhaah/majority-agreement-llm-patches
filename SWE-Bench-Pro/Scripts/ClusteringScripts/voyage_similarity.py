#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import voyageai


# ---------------------------------------------------------------------------
# Voyage configuration
# ---------------------------------------------------------------------------

VOYAGE_API_KEY = ""

VOYAGE_MODEL_NAME = "voyage-code-3"
VOYAGE_OUTPUT_DIMENSION = 1024


VOYAGE_MODEL_CONTEXT_TOKENS = 32_000
VOYAGE_CHUNK_TOKEN_LIMIT = 32_000

VOYAGE_REQUEST_TOKEN_LIMIT = 120_000
VOYAGE_REQUEST_TOKEN_BUDGET = 115_000
VOYAGE_MAX_TEXTS_PER_REQUEST = 1_000

DEFAULT_VOYAGE_LANG = "code"

PREPROCESSING_VERSION = "file_removed_added_union_v2_32000_fragment_chunks"

# Shared-file exponential calibration used for the main Voyage similarity.
VOYAGE_DISTANCE_SCALE = 0.5

# Additional diagnostic CSVs.
JACCARD_MATRIX_FILENAME = "jaccard_similarity_matrix.csv"
JACCARD_GROUND_TRUTH_FILENAME = "ground_truth_jaccard_similarity.csv"
SHARED_ONLY_MATRIX_FILENAME = "shared_files_only_similarity_matrix.csv"
SHARED_ONLY_GROUND_TRUTH_FILENAME = "ground_truth_shared_files_only_similarity.csv"
UNTRANSFORMED_UNION_MATRIX_FILENAME = "untransformed_union_similarity_matrix.csv"
UNTRANSFORMED_UNION_GROUND_TRUTH_FILENAME = (
    "ground_truth_untransformed_union_similarity.csv"
)

_VOYAGE_CLIENT: Optional[voyageai.Client] = None


# ---------------------------------------------------------------------------
# Voyage client
# ---------------------------------------------------------------------------

def _get_client() -> voyageai.Client:
    """Create the Voyage client lazily."""
    global _VOYAGE_CLIENT

    if _VOYAGE_CLIENT is not None:
        return _VOYAGE_CLIENT

    api_key = str(VOYAGE_API_KEY or "").strip()
    if not api_key:
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "Voyage API key is empty. Set VOYAGE_API_KEY in voyage_similarity.py "
            "or export the VOYAGE_API_KEY environment variable."
        )

    _VOYAGE_CLIENT = voyageai.Client(
        api_key=api_key,
        max_retries=3,
        timeout=120,
    )
    return _VOYAGE_CLIENT


# ---------------------------------------------------------------------------
# Compatibility with the clustering pipeline
# ---------------------------------------------------------------------------

def get_lang_for_repo(
    repo: Optional[str],
    default_lang: str = DEFAULT_VOYAGE_LANG,
) -> str:
    """Compatibility helper; voyage-code-3 does not need a language argument."""
    del repo
    return default_lang


# ---------------------------------------------------------------------------
# Diff parsing -- intentionally matches BLEU / CodeBLEU / CodeBERT
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
    original order. The same is done independently for added lines.

    Excluded:
        * diff --git metadata
        * index metadata
        * --- / +++ file headers
        * @@ hunk headers
        * unchanged context lines
        * "\\ No newline at end of file" metadata

    This preprocessing intentionally matches the BLEU, CodeBLEU, and CodeBERT
    modules so all four metrics compare the same file/change units.
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
        if (
            line.startswith("--- ")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("+++ ")
        ):
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


def clean_diff_for_voyage(diff_text: str) -> str:
    """
    Backward-compatible helper returning changed code without Git metadata.

    The file-aware whole-patch metric below does not use this flattened text for
    comparison. It preserves file boundaries and removed/added channels.
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
# Token counting and fragment-level chunking
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Count tokens using the voyage-code-3 tokenizer exposed by the Voyage client."""
    text = str(text or "")
    if not text:
        return 0

    return int(
        _get_client().count_tokens(
            [text],
            model=VOYAGE_MODEL_NAME,
        )
    )


def _largest_prefix_within_token_limit(
    text: str,
    token_limit: int,
) -> Tuple[str, int]:
    """
    Return a non-empty character prefix whose Voyage token count is <= token_limit.

    The search first finds a large safe prefix, then prefers a nearby newline so
    source-code lines are not split when possible. No characters are discarded:
    the returned prefix is removed verbatim from the remaining text.
    """
    text = str(text or "")
    if not text:
        return "", 0

    low = 1
    high = len(text)
    best_end = 0
    best_tokens = 0

    while low <= high:
        middle = (low + high) // 2
        prefix = text[:middle]
        token_count = _count_tokens(prefix)

        if token_count <= token_limit:
            best_end = middle
            best_tokens = token_count
            low = middle + 1
        else:
            high = middle - 1

    if best_end <= 0:
        raise ValueError(
            "Could not construct a non-empty Voyage chunk within the token limit."
        )

    # Prefer a newline near the end of the safe prefix, but only when it still
    # leaves a reasonably large chunk.
    prefix = text[:best_end]
    newline_pos = prefix.rfind("\n", int(len(prefix) * 0.80))

    if newline_pos > 0:
        newline_end = newline_pos + 1
        newline_prefix = text[:newline_end]
        newline_tokens = _count_tokens(newline_prefix)

        if 0 < newline_tokens <= token_limit:
            return newline_prefix, newline_tokens

    return prefix, best_tokens


def split_text_into_voyage_chunks(
    text: str,
    chunk_token_limit: int = VOYAGE_CHUNK_TOKEN_LIMIT,
) -> List[Tuple[str, int]]:
    """
    Split one extracted removed/added fragment into non-overlapping chunks.

    Each returned tuple is:
        (chunk_text, exact_voyage_token_count)

    Every chunk has at most 32,000 tokens by default. Chunk order is preserved,
    chunks do not overlap, and no fragment text is intentionally discarded.

    Important: this function is called on one removed or added fragment, not on
    a complete unified diff.
    """
    text = str(text or "")

    if not text.strip():
        return []

    if chunk_token_limit <= 0 or chunk_token_limit > VOYAGE_MODEL_CONTEXT_TOKENS:
        raise ValueError(
            f"chunk_token_limit must be in [1, {VOYAGE_MODEL_CONTEXT_TOKENS}]"
        )

    total_tokens = _count_tokens(text)

    if total_tokens <= chunk_token_limit:
        return [(text, total_tokens)]

    chunks: List[Tuple[str, int]] = []
    remaining = text

    while remaining:
        remaining_tokens = _count_tokens(remaining)

        if remaining_tokens <= chunk_token_limit:
            chunks.append((remaining, remaining_tokens))
            break

        prefix, prefix_tokens = _largest_prefix_within_token_limit(
            remaining,
            chunk_token_limit,
        )

        if not prefix:
            raise ValueError("Voyage chunking produced an empty prefix.")

        chunks.append((prefix, prefix_tokens))
        remaining = remaining[len(prefix):]

    return chunks


# ---------------------------------------------------------------------------
# Voyage fragment embeddings
# ---------------------------------------------------------------------------

def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize one embedding vector."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))

    if norm <= 0.0:
        return np.zeros_like(vector, dtype=np.float32)

    return (vector / norm).astype(np.float32)


def _embed_chunk_batch(texts: Sequence[str]) -> np.ndarray:
    """Embed one API-safe batch of already-sized Voyage chunks."""
    if not texts:
        return np.empty((0, VOYAGE_OUTPUT_DIMENSION), dtype=np.float32)

    result = _get_client().embed(
        list(texts),
        model=VOYAGE_MODEL_NAME,
        input_type=None,
        truncation=False,
        output_dimension=VOYAGE_OUTPUT_DIMENSION,
        output_dtype="float",
    )

    embeddings = np.asarray(result.embeddings, dtype=np.float32)
    expected_shape = (len(texts), VOYAGE_OUTPUT_DIMENSION)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Voyage returned embedding shape {embeddings.shape}; "
            f"expected {expected_shape}."
        )

    return embeddings


def embed_fragments(texts: Sequence[str]) -> np.ndarray:
    """
    Embed extracted removed/added fragments into one normalized vector each.

    For every non-empty fragment:
      1. Count Voyage tokens.
      2. If <= 32,000 tokens, embed directly.
      3. Otherwise split the fragment into ordered, non-overlapping chunks.
      4. Embed every chunk with truncation disabled.
      5. Combine chunk vectors using a token-count-weighted mean.
      6. L2-normalize the final fragment vector.

    Empty fragments are represented by an all-zero vector, although the
    file-aware similarity code normally avoids embedding empty channels.
    """
    texts = [str(text or "") for text in texts]

    if not texts:
        return np.empty((0, VOYAGE_OUTPUT_DIMENSION), dtype=np.float32)

    chunk_texts: List[str] = []
    chunk_token_counts: List[int] = []
    chunk_owner_indices: List[int] = []

    for fragment_index, text in enumerate(texts):
        for chunk_text, token_count in split_text_into_voyage_chunks(text):
            chunk_texts.append(chunk_text)
            chunk_token_counts.append(int(token_count))
            chunk_owner_indices.append(fragment_index)

    weighted_sums = np.zeros(
        (len(texts), VOYAGE_OUTPUT_DIMENSION),
        dtype=np.float64,
    )
    token_totals = np.zeros(len(texts), dtype=np.float64)

    batch_start = 0

    while batch_start < len(chunk_texts):
        batch_end = batch_start
        batch_tokens = 0

        while batch_end < len(chunk_texts):
            next_tokens = int(chunk_token_counts[batch_end])

            too_many_texts = (
                batch_end - batch_start >= VOYAGE_MAX_TEXTS_PER_REQUEST
            )
            too_many_tokens = (
                batch_tokens + next_tokens > VOYAGE_REQUEST_TOKEN_BUDGET
            )

            if batch_end > batch_start and (too_many_texts or too_many_tokens):
                break

            if next_tokens > VOYAGE_REQUEST_TOKEN_BUDGET:
                raise ValueError(
                    f"One Voyage chunk has {next_tokens} tokens, exceeding the "
                    f"request budget of {VOYAGE_REQUEST_TOKEN_BUDGET}."
                )

            batch_tokens += next_tokens
            batch_end += 1

        batch_embeddings = _embed_chunk_batch(
            chunk_texts[batch_start:batch_end]
        )

        for local_index, embedding in enumerate(batch_embeddings):
            global_index = batch_start + local_index
            owner = chunk_owner_indices[global_index]
            weight = max(1, int(chunk_token_counts[global_index]))

            weighted_sums[owner] += embedding.astype(np.float64) * weight
            token_totals[owner] += weight

        batch_start = batch_end

    fragment_embeddings = np.zeros(
        (len(texts), VOYAGE_OUTPUT_DIMENSION),
        dtype=np.float32,
    )

    for fragment_index in range(len(texts)):
        if token_totals[fragment_index] <= 0:
            continue

        mean_embedding = (
            weighted_sums[fragment_index] / token_totals[fragment_index]
        ).astype(np.float32)

        fragment_embeddings[fragment_index] = _normalize_vector(mean_embedding)

    return fragment_embeddings


def embed_fragment(text: str) -> np.ndarray:
    """Embed one extracted removed/added fragment."""
    return embed_fragments([str(text or "")])[0]


# Backward-compatible names from the previous Voyage module.
def embed_patches(texts: Sequence[str]) -> np.ndarray:
    """
    Backward-compatible alias.

    The revised similarity pipeline passes extracted code fragments here rather
    than complete diffs.
    """
    return embed_fragments(texts)


def embed_patch(text: str) -> np.ndarray:
    """Backward-compatible alias for embedding one extracted code fragment."""
    return embed_fragment(text)


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute a full pairwise cosine matrix from normalized embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")

    if embeddings.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)

    similarities = embeddings @ embeddings.T
    similarities = np.clip(similarities, -1.0, 1.0)
    np.fill_diagonal(similarities, 1.0)

    return similarities.astype(np.float32)


def cosine_similarity_vector(
    reference_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute reference-to-candidate cosine similarities."""
    reference_embedding = np.asarray(reference_embedding, dtype=np.float32)
    candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)

    if reference_embedding.ndim != 1:
        raise ValueError("reference_embedding must be one-dimensional")

    if candidate_embeddings.ndim != 2:
        raise ValueError("candidate_embeddings must be two-dimensional")

    if candidate_embeddings.shape[1] != reference_embedding.shape[0]:
        raise ValueError(
            "reference and candidate embedding dimensions do not match"
        )

    similarities = candidate_embeddings @ reference_embedding
    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


def _cosine_similarity_between_embeddings(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """Cosine similarity for two already L2-normalized Voyage vectors."""
    embedding_a = np.asarray(embedding_a, dtype=np.float32)
    embedding_b = np.asarray(embedding_b, dtype=np.float32)

    if embedding_a.ndim != 1 or embedding_b.ndim != 1:
        raise ValueError("Both embeddings must be one-dimensional")

    if embedding_a.shape[0] != embedding_b.shape[0]:
        raise ValueError("Embedding dimensions do not match")

    if np.linalg.norm(embedding_a) == 0.0 or np.linalg.norm(embedding_b) == 0.0:
        return 0.0

    return float(np.clip(np.dot(embedding_a, embedding_b), -1.0, 1.0))


# ---------------------------------------------------------------------------
# File-aware whole-patch Voyage similarity
# ---------------------------------------------------------------------------

def _embed_file_change_mapping(
    file_changes: Dict[str, Dict[str, str]],
) -> Dict[Tuple[str, str], np.ndarray]:
    """Embed every non-empty removed/added channel in one parsed patch."""
    keys: List[Tuple[str, str]] = []
    texts: List[str] = []

    for file_name in sorted(file_changes):
        for channel in ("removed", "added"):
            text = str(file_changes[file_name].get(channel, "") or "")

            if text.strip():
                keys.append((file_name, channel))
                texts.append(text)

    if not texts:
        return {}

    vectors = embed_fragments(texts)

    return {
        key: vectors[index]
        for index, key in enumerate(keys)
    }


def _channel_similarity_from_embeddings(
    channel_a: str,
    channel_b: str,
    embedding_a: Optional[np.ndarray],
    embedding_b: Optional[np.ndarray],
) -> Optional[float]:
    """
    Compare one removed/added channel.

    Rules:
      * nonempty vs nonempty -> Voyage cosine similarity
      * nonempty vs empty    -> 0.0
      * empty vs nonempty    -> 0.0
      * empty vs empty       -> None (exclude from the file-level mean)
    """
    text_a = str(channel_a or "")
    text_b = str(channel_b or "")

    has_a = bool(text_a.strip())
    has_b = bool(text_b.strip())

    if not has_a and not has_b:
        return None

    if has_a != has_b:
        return 0.0

    if embedding_a is None or embedding_b is None:
        raise ValueError("Missing Voyage embedding for a non-empty change channel")

    return _cosine_similarity_between_embeddings(
        embedding_a,
        embedding_b,
    )


def shared_file_similarity(
    file_name: str,
    file_change_a: Dict[str, str],
    file_change_b: Dict[str, str],
    embeddings_a: Dict[Tuple[str, str], np.ndarray],
    embeddings_b: Dict[Tuple[str, str], np.ndarray],
) -> float:
    """
    Calculate transformed Voyage similarity for one file present in both patches.

    Removed code is compared only with removed code and added code only with added
    code. Defined raw cosine channel scores are averaged first. A channel empty in
    both patches is excluded; a channel present in only one patch contributes 0.

    The resulting raw shared-file cosine score c is transformed as:
        exp(-(1 - c) / 0.5)

    Only shared files are transformed. Files present in only one patch remain
    literal zero when the patch-level score is averaged over the file union.
    """
    component_scores: List[float] = []

    removed_score = _channel_similarity_from_embeddings(
        file_change_a.get("removed", ""),
        file_change_b.get("removed", ""),
        embeddings_a.get((file_name, "removed")),
        embeddings_b.get((file_name, "removed")),
    )

    if removed_score is not None:
        component_scores.append(float(removed_score))

    added_score = _channel_similarity_from_embeddings(
        file_change_a.get("added", ""),
        file_change_b.get("added", ""),
        embeddings_a.get((file_name, "added")),
        embeddings_b.get((file_name, "added")),
    )

    if added_score is not None:
        component_scores.append(float(added_score))

    if not component_scores:
        # Do not reward a shared path when no textual changed code was extracted.
        return 0.0

    raw_file_score = float(sum(component_scores) / len(component_scores))
    transformed_file_score = math.exp(
        -(1.0 - raw_file_score) / VOYAGE_DISTANCE_SCALE
    )
    return float(transformed_file_score)


def _raw_shared_file_cosine_similarity(
    file_name: str,
    file_change_a: Dict[str, str],
    file_change_b: Dict[str, str],
    embeddings_a: Dict[Tuple[str, str], np.ndarray],
    embeddings_b: Dict[Tuple[str, str], np.ndarray],
) -> float:
    """Return the untransformed mean Voyage cosine for one shared file.

    Empty-channel behavior is identical to shared_file_similarity(): both empty is
    excluded; exactly one empty contributes zero. No exponential transformation is
    applied here.
    """
    component_scores: List[float] = []

    removed_score = _channel_similarity_from_embeddings(
        file_change_a.get("removed", ""),
        file_change_b.get("removed", ""),
        embeddings_a.get((file_name, "removed")),
        embeddings_b.get((file_name, "removed")),
    )
    if removed_score is not None:
        component_scores.append(float(removed_score))

    added_score = _channel_similarity_from_embeddings(
        file_change_a.get("added", ""),
        file_change_b.get("added", ""),
        embeddings_a.get((file_name, "added")),
        embeddings_b.get((file_name, "added")),
    )
    if added_score is not None:
        component_scores.append(float(added_score))

    if not component_scores:
        return 0.0

    return float(sum(component_scores) / len(component_scores))


def _patch_similarity_decomposition_from_precomputed(
    files_a: Dict[str, Dict[str, str]],
    files_b: Dict[str, Dict[str, str]],
    embeddings_a: Dict[Tuple[str, str], np.ndarray],
    embeddings_b: Dict[Tuple[str, str], np.ndarray],
) -> Tuple[float, float, float, float]:
    """Return main and diagnostic patch-level similarities.

    Returns:
        transformed_union_similarity:
            Main score written to similarity_matrix.csv. Each shared file uses the
            scale-0.5 exponential transformation; unshared files contribute zero;
            scores are averaged over the union of changed files.
        jaccard_file_similarity:
            |shared changed files| / |union changed files|.
        raw_shared_only_similarity:
            Mean raw (untransformed) shared-file cosine similarity over shared files
            only. Each shared-file raw score is the mean of the defined removed and
            added cosine channels.
        raw_union_similarity:
            Raw shared-file cosine scores plus literal zeros for unshared files,
            averaged over the union.

    For non-empty unions:
        raw_union_similarity ==
            jaccard_file_similarity * raw_shared_only_similarity
    up to floating-point precision.
    """
    union_files = sorted(set(files_a) | set(files_b))

    if not union_files:
        return 0.0, 0.0, 0.0, 0.0

    shared_files = sorted(set(files_a) & set(files_b))
    union_count = len(union_files)
    shared_count = len(shared_files)

    jaccard_file_similarity = float(shared_count / union_count)

    transformed_shared_scores: List[float] = []
    raw_shared_scores: List[float] = []

    for file_name in shared_files:
        transformed_shared_scores.append(
            shared_file_similarity(
                file_name=file_name,
                file_change_a=files_a[file_name],
                file_change_b=files_b[file_name],
                embeddings_a=embeddings_a,
                embeddings_b=embeddings_b,
            )
        )
        raw_shared_scores.append(
            _raw_shared_file_cosine_similarity(
                file_name=file_name,
                file_change_a=files_a[file_name],
                file_change_b=files_b[file_name],
                embeddings_a=embeddings_a,
                embeddings_b=embeddings_b,
            )
        )

    if shared_count == 0:
        transformed_union_similarity = 0.0
        raw_shared_only_similarity = 0.0
        raw_union_similarity = 0.0
    else:
        transformed_union_similarity = float(
            sum(transformed_shared_scores) / union_count
        )
        raw_shared_only_similarity = float(
            sum(raw_shared_scores) / shared_count
        )
        raw_union_similarity = float(
            sum(raw_shared_scores) / union_count
        )

    if not np.isclose(
        raw_union_similarity,
        jaccard_file_similarity * raw_shared_only_similarity,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError(
            "Voyage raw union similarity decomposition failed: "
            "raw union != Jaccard * raw shared-only similarity"
        )

    return (
        transformed_union_similarity,
        jaccard_file_similarity,
        raw_shared_only_similarity,
        raw_union_similarity,
    )


def _whole_patch_similarity_from_precomputed(
    files_a: Dict[str, Dict[str, str]],
    files_b: Dict[str, Dict[str, str]],
    embeddings_a: Dict[Tuple[str, str], np.ndarray],
    embeddings_b: Dict[Tuple[str, str], np.ndarray],
) -> float:
    """Return transformed shared-file scores averaged over the file union."""
    transformed_union_similarity, _, _, _ = (
        _patch_similarity_decomposition_from_precomputed(
            files_a=files_a,
            files_b=files_b,
            embeddings_a=embeddings_a,
            embeddings_b=embeddings_b,
        )
    )
    return float(transformed_union_similarity)

def whole_patch_similarity(
    diff_a: str,
    diff_b: str,
    lang: str = DEFAULT_VOYAGE_LANG,
) -> float:
    """
    Calculate one file-aware Voyage similarity score for two complete diffs.

    Procedure:
      1. Parse each unified diff into one entry per changed file.
      2. Concatenate all removed lines per file and all added lines per file.
      3. Chunk each non-empty removed/added fragment independently at 32K tokens.
      4. Produce one normalized Voyage vector per fragment.
      5. For a shared file:
           cosine(removed_a, removed_b)
           cosine(added_a, added_b)
         and average the defined channel scores.
      6. A file changed by only one patch receives similarity 0.
      7. Average file-level scores over the UNION of changed files.

    The returned value is the raw Voyage whole-patch similarity written to
    similarity_matrix.csv. No additional clustering-time transformation is
    applied by the supplied clustering code for metric="voyage".
    """
    del lang

    text_a = str(diff_a or "")
    text_b = str(diff_b or "")

    if not text_a.strip() and not text_b.strip():
        return 1.0

    if not text_a.strip() or not text_b.strip():
        return 0.0

    files_a = _file_change_mapping(text_a)
    files_b = _file_change_mapping(text_b)

    if not files_a and not files_b:
        print(
            "WARNING: could not extract changed files from either patch; "
            "returning whole-patch Voyage similarity 0.0"
        )
        return 0.0

    embeddings_a = _embed_file_change_mapping(files_a)
    embeddings_b = _embed_file_change_mapping(files_b)

    return _whole_patch_similarity_from_precomputed(
        files_a=files_a,
        files_b=files_b,
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
    )


def compare_two_diffs(
    diff_a: str,
    diff_b: str,
    lang: str = DEFAULT_VOYAGE_LANG,
) -> float:
    """Return the final file-aware whole-patch Voyage similarity."""
    return whole_patch_similarity(diff_a, diff_b, lang=lang)


def symmetric_voyage(
    diff_a: str,
    diff_b: str,
    lang: str = DEFAULT_VOYAGE_LANG,
) -> float:
    """
    Backward-compatible whole-diff entry point.

    Cosine similarity is symmetric, so the file-aware whole-patch result is
    already symmetric.
    """
    return whole_patch_similarity(diff_a, diff_b, lang=lang)


# ---------------------------------------------------------------------------
# Per-issue candidate file/channel embedding cache
# ---------------------------------------------------------------------------

def _matrix_output_dir(
    matrix_root: Path,
    instance_id: str,
    metric_name: str,
) -> Path:
    output_dir = Path(matrix_root) / str(instance_id) / str(metric_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _candidate_embedding_cache_paths(
    output_dir: Path,
) -> Tuple[Path, Path]:
    # Keep the same filenames used by the previous implementation so the
    # surrounding clustering code does not need any changes. PREPROCESSING_VERSION
    # prevents the old whole-patch cache from being reused.
    return (
        output_dir / "candidate_embeddings.npz",
        output_dir / "candidate_embeddings_metadata.json",
    )


def _serialize_embedding_key(
    temp: str,
    file_name: str,
    channel: str,
) -> str:
    return json.dumps(
        [str(temp), str(file_name), str(channel)],
        ensure_ascii=False,
    )


def _deserialize_embedding_key(
    value: str,
) -> Tuple[str, str, str]:
    temp, file_name, channel = json.loads(str(value))
    return str(temp), str(file_name), str(channel)


def _load_or_build_candidate_change_embeddings(
    temperature_to_diff: Dict[str, str],
    output_dir: Path,
) -> Tuple[
    List[str],
    Dict[str, str],
    Dict[str, Dict[str, Dict[str, str]]],
    Dict[Tuple[str, str, str], np.ndarray],
]:
    """
    Load/build per-file removed/added embeddings for all candidate patches.

    Returns:
        temperatures
        raw_patch_by_temperature
        parsed_changes_by_temperature
        embedding_by_(temperature, file, channel)
    """
    temperature_keys = list(temperature_to_diff.keys())
    temperatures = [str(temp) for temp in temperature_keys]
    patches = [
        str(temperature_to_diff[temp_key] or "")
        for temp_key in temperature_keys
    ]

    raw_patch_by_temperature = {
        temp: patch
        for temp, patch in zip(temperatures, patches)
    }

    expected_hashes = {
        temp: _text_sha256(patch)
        for temp, patch in zip(temperatures, patches)
    }

    parsed_changes_by_temperature: Dict[
        str,
        Dict[str, Dict[str, str]],
    ] = {
        temp: _file_change_mapping(patch)
        for temp, patch in zip(temperatures, patches)
    }

    cache_path, metadata_path = _candidate_embedding_cache_paths(output_dir)

    if cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            cached = np.load(cache_path, allow_pickle=False)

            cached_temperatures = [
                str(value)
                for value in cached["temperatures"]
            ]
            cached_keys = [
                str(value)
                for value in cached["embedding_keys"]
            ]
            cached_embeddings = cached["embeddings"].astype(np.float32)

            cache_matches = (
                metadata.get("model_name") == VOYAGE_MODEL_NAME
                and metadata.get("output_dimension") == VOYAGE_OUTPUT_DIMENSION
                and metadata.get("input_type") is None
                and metadata.get("preprocessing_version") == PREPROCESSING_VERSION
                and metadata.get("model_context_tokens")
                == VOYAGE_MODEL_CONTEXT_TOKENS
                and metadata.get("chunk_token_limit")
                == VOYAGE_CHUNK_TOKEN_LIMIT
                and metadata.get("request_token_budget")
                == VOYAGE_REQUEST_TOKEN_BUDGET
                and metadata.get("patch_sha256") == expected_hashes
                and cached_temperatures == temperatures
                and cached_embeddings.ndim == 2
                and cached_embeddings.shape[0] == len(cached_keys)
                and cached_embeddings.shape[1] == VOYAGE_OUTPUT_DIMENSION
            )

            if cache_matches:
                embedding_map: Dict[
                    Tuple[str, str, str],
                    np.ndarray,
                ] = {}

                for index, serialized_key in enumerate(cached_keys):
                    key = _deserialize_embedding_key(serialized_key)
                    embedding_map[key] = cached_embeddings[index]

                return (
                    temperatures,
                    raw_patch_by_temperature,
                    parsed_changes_by_temperature,
                    embedding_map,
                )

        except Exception as exc:
            print(
                "WARNING: failed to read Voyage file/channel embedding cache at "
                f"{cache_path}: {exc}. Recomputing embeddings."
            )

    embedding_keys: List[Tuple[str, str, str]] = []
    embedding_texts: List[str] = []

    for temp in temperatures:
        file_changes = parsed_changes_by_temperature[temp]

        for file_name in sorted(file_changes):
            for channel in ("removed", "added"):
                text = str(
                    file_changes[file_name].get(channel, "") or ""
                )

                if not text.strip():
                    continue

                embedding_keys.append((temp, file_name, channel))
                embedding_texts.append(text)

    if embedding_texts:
        embeddings = embed_fragments(embedding_texts)
    else:
        embeddings = np.empty(
            (0, VOYAGE_OUTPUT_DIMENSION),
            dtype=np.float32,
        )

    embedding_map = {
        key: embeddings[index]
        for index, key in enumerate(embedding_keys)
    }

    serialized_keys = [
        _serialize_embedding_key(*key)
        for key in embedding_keys
    ]

    np.savez_compressed(
        cache_path,
        temperatures=np.asarray(temperatures),
        embedding_keys=np.asarray(serialized_keys),
        embeddings=embeddings,
    )

    metadata_path.write_text(
        json.dumps(
            {
                "model_name": VOYAGE_MODEL_NAME,
                "output_dimension": VOYAGE_OUTPUT_DIMENSION,
                "input_type": None,
                "preprocessing_version": PREPROCESSING_VERSION,
                "model_context_tokens": VOYAGE_MODEL_CONTEXT_TOKENS,
                "chunk_token_limit": VOYAGE_CHUNK_TOKEN_LIMIT,
                "request_token_limit": VOYAGE_REQUEST_TOKEN_LIMIT,
                "request_token_budget": VOYAGE_REQUEST_TOKEN_BUDGET,
                "max_texts_per_request": VOYAGE_MAX_TEXTS_PER_REQUEST,
                "pooling": (
                    "per removed/added fragment: direct embedding when <=32000 "
                    "tokens; otherwise non-overlapping <=32000-token chunks; "
                    "token-count-weighted mean across chunk embeddings; "
                    "final fragment L2 normalization"
                ),
                "patch_preprocessing": (
                    "parse unified diff by file; concatenate all removed lines "
                    "and all added lines across hunks; exclude unchanged context "
                    "and diff metadata; compare removed-to-removed and "
                    "added-to-added; unmatched files=0; average defined channels "
                    "per shared file; then unweighted average over file union"
                ),
                "patch_sha256": expected_hashes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return (
        temperatures,
        raw_patch_by_temperature,
        parsed_changes_by_temperature,
        embedding_map,
    )


def _embeddings_for_temperature(
    temp: str,
    embedding_map: Dict[Tuple[str, str, str], np.ndarray],
) -> Dict[Tuple[str, str], np.ndarray]:
    """Extract one temperature's (file, channel) -> embedding mapping."""
    return {
        (file_name, channel): embedding
        for (cached_temp, file_name, channel), embedding in embedding_map.items()
        if cached_temp == temp
    }


def _candidate_patch_similarity_from_cache(
    temp_a: str,
    temp_b: str,
    raw_patch_by_temperature: Dict[str, str],
    parsed_changes_by_temperature: Dict[
        str,
        Dict[str, Dict[str, str]],
    ],
    embedding_map: Dict[Tuple[str, str, str], np.ndarray],
) -> float:
    """Compare two candidate patches using cached file/channel embeddings."""
    raw_a = str(raw_patch_by_temperature[temp_a] or "")
    raw_b = str(raw_patch_by_temperature[temp_b] or "")

    # Keep whole-patch empty behavior aligned with BLEU/CodeBLEU.
    if not raw_a.strip() and not raw_b.strip():
        return 1.0

    if not raw_a.strip() or not raw_b.strip():
        return 0.0

    files_a = parsed_changes_by_temperature[temp_a]
    files_b = parsed_changes_by_temperature[temp_b]

    embeddings_a = _embeddings_for_temperature(
        temp_a,
        embedding_map,
    )
    embeddings_b = _embeddings_for_temperature(
        temp_b,
        embedding_map,
    )

    return _whole_patch_similarity_from_precomputed(
        files_a=files_a,
        files_b=files_b,
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
    )


# ---------------------------------------------------------------------------
# Pairwise candidate similarity matrix
# ---------------------------------------------------------------------------

def build_similarity_matrix(
    temperature_to_diff: Dict[str, str],
    lang: str = DEFAULT_VOYAGE_LANG,
) -> pd.DataFrame:
    """
    Build the file-aware whole-patch Voyage similarity matrix in memory.

    This uncached helper is useful for direct testing. The clustering pipeline
    should normally call build_and_save_similarity_matrix(), which caches
    candidate file/channel embeddings.
    """
    del lang

    if not temperature_to_diff:
        raise ValueError("temperature_to_diff is empty")

    temperature_keys = list(temperature_to_diff.keys())
    valid_temperatures = [str(temp) for temp in temperature_keys]

    raw_patch_by_temperature = {
        str(temp): str(temperature_to_diff[temp] or "")
        for temp in temperature_keys
    }

    parsed_changes_by_temperature = {
        temp: _file_change_mapping(raw_patch_by_temperature[temp])
        for temp in valid_temperatures
    }

    embeddings_by_temp = {
        temp: _embed_file_change_mapping(
            parsed_changes_by_temperature[temp]
        )
        for temp in valid_temperatures
    }

    similarity_df = pd.DataFrame(
        0.0,
        index=valid_temperatures,
        columns=valid_temperatures,
        dtype=float,
    )

    for i, temp_i in enumerate(valid_temperatures):
        similarity_df.loc[temp_i, temp_i] = 1.0

        for j in range(i + 1, len(valid_temperatures)):
            temp_j = valid_temperatures[j]

            raw_i = raw_patch_by_temperature[temp_i]
            raw_j = raw_patch_by_temperature[temp_j]

            if not raw_i.strip() and not raw_j.strip():
                score = 1.0
            elif not raw_i.strip() or not raw_j.strip():
                score = 0.0
            else:
                score = _whole_patch_similarity_from_precomputed(
                    files_a=parsed_changes_by_temperature[temp_i],
                    files_b=parsed_changes_by_temperature[temp_j],
                    embeddings_a=embeddings_by_temp[temp_i],
                    embeddings_b=embeddings_by_temp[temp_j],
                )

            similarity_df.loc[temp_i, temp_j] = float(score)
            similarity_df.loc[temp_j, temp_i] = float(score)

    return similarity_df


def save_similarity_matrix(
    similarity_df: pd.DataFrame,
    matrix_root: Path,
    instance_id: str,
    metric_name: str = "voyage",
    filename: str = "similarity_matrix.csv",
) -> Path:
    """Save the generated-patch file-aware Voyage similarity matrix."""
    output_dir = _matrix_output_dir(
        matrix_root=matrix_root,
        instance_id=instance_id,
        metric_name=metric_name,
    )

    output_path = output_dir / filename
    similarity_df.to_csv(output_path)

    return output_path


def build_and_save_similarity_matrix(
    temperature_to_diff: Dict[str, str],
    matrix_root: Path,
    instance_id: str,
    lang: str,
    metric_name: str = "voyage",
    filename: str = "similarity_matrix.csv",
) -> Tuple[pd.DataFrame, Path]:
    """Build/save the main Voyage matrix plus decomposition diagnostics."""
    del lang

    if not temperature_to_diff:
        raise ValueError("temperature_to_diff is empty")

    output_dir = _matrix_output_dir(
        matrix_root=matrix_root,
        instance_id=instance_id,
        metric_name=metric_name,
    )

    (
        temperatures,
        raw_patch_by_temperature,
        parsed_changes_by_temperature,
        embedding_map,
    ) = _load_or_build_candidate_change_embeddings(
        temperature_to_diff=temperature_to_diff,
        output_dir=output_dir,
    )

    similarity_df = pd.DataFrame(
        0.0, index=temperatures, columns=temperatures, dtype=float
    )
    jaccard_df = pd.DataFrame(
        0.0, index=temperatures, columns=temperatures, dtype=float
    )
    shared_only_df = pd.DataFrame(
        0.0, index=temperatures, columns=temperatures, dtype=float
    )
    untransformed_union_df = pd.DataFrame(
        0.0, index=temperatures, columns=temperatures, dtype=float
    )

    for i, temp_i in enumerate(temperatures):
        similarity_df.loc[temp_i, temp_i] = 1.0
        jaccard_df.loc[temp_i, temp_i] = 1.0
        shared_only_df.loc[temp_i, temp_i] = 1.0
        untransformed_union_df.loc[temp_i, temp_i] = 1.0

        for j in range(i + 1, len(temperatures)):
            temp_j = temperatures[j]

            raw_i = str(raw_patch_by_temperature[temp_i] or "")
            raw_j = str(raw_patch_by_temperature[temp_j] or "")

            if not raw_i.strip() and not raw_j.strip():
                score = 1.0
                jaccard_score = 1.0
                shared_only_score = 1.0
                untransformed_union_score = 1.0
            elif not raw_i.strip() or not raw_j.strip():
                score = 0.0
                jaccard_score = 0.0
                shared_only_score = 0.0
                untransformed_union_score = 0.0
            else:
                embeddings_i = _embeddings_for_temperature(temp_i, embedding_map)
                embeddings_j = _embeddings_for_temperature(temp_j, embedding_map)

                (
                    score,
                    jaccard_score,
                    shared_only_score,
                    untransformed_union_score,
                ) = _patch_similarity_decomposition_from_precomputed(
                    files_a=parsed_changes_by_temperature[temp_i],
                    files_b=parsed_changes_by_temperature[temp_j],
                    embeddings_a=embeddings_i,
                    embeddings_b=embeddings_j,
                )

            for df, value in [
                (similarity_df, score),
                (jaccard_df, jaccard_score),
                (shared_only_df, shared_only_score),
                (untransformed_union_df, untransformed_union_score),
            ]:
                df.loc[temp_i, temp_j] = float(value)
                df.loc[temp_j, temp_i] = float(value)

    matrix_path = output_dir / filename
    similarity_df.to_csv(matrix_path)

    jaccard_df.to_csv(output_dir / JACCARD_MATRIX_FILENAME)
    shared_only_df.to_csv(output_dir / SHARED_ONLY_MATRIX_FILENAME)
    untransformed_union_df.to_csv(
        output_dir / UNTRANSFORMED_UNION_MATRIX_FILENAME
    )

    return similarity_df, matrix_path


# ---------------------------------------------------------------------------
# Ground-truth-to-candidate similarity
# ---------------------------------------------------------------------------

def build_ground_truth_similarity_vector(
    ground_truth_patch: str,
    temperature_to_diff: Dict[str, str],
    lang: str = DEFAULT_VOYAGE_LANG,
) -> pd.DataFrame:
    """
    Build ground-truth-to-candidate file-aware Voyage similarities in memory.
    """
    del lang

    if not temperature_to_diff:
        raise ValueError("temperature_to_diff is empty")

    temperature_keys = list(temperature_to_diff.keys())
    valid_temperatures = [str(temp) for temp in temperature_keys]

    ground_truth_text = str(ground_truth_patch or "")
    ground_truth_files = _file_change_mapping(ground_truth_text)
    ground_truth_embeddings = _embed_file_change_mapping(ground_truth_files)

    scores: List[float] = []

    for temp_key, temp in zip(temperature_keys, valid_temperatures):
        candidate_text = str(temperature_to_diff[temp_key] or "")

        if not ground_truth_text.strip() and not candidate_text.strip():
            score = 1.0
        elif not ground_truth_text.strip() or not candidate_text.strip():
            score = 0.0
        else:
            candidate_files = _file_change_mapping(candidate_text)
            candidate_embeddings = _embed_file_change_mapping(candidate_files)

            score = _whole_patch_similarity_from_precomputed(
                files_a=ground_truth_files,
                files_b=candidate_files,
                embeddings_a=ground_truth_embeddings,
                embeddings_b=candidate_embeddings,
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
    metric_name: str = "voyage",
    filename: str = "ground_truth_similarity.csv",
) -> Path:
    """Save the one-row ground-truth Voyage similarity CSV."""
    output_dir = _matrix_output_dir(
        matrix_root=matrix_root,
        instance_id=instance_id,
        metric_name=metric_name,
    )

    output_path = output_dir / filename
    ground_truth_similarity_df.to_csv(output_path)

    return output_path


def build_and_save_ground_truth_similarity_vector(
    ground_truth_patch: str,
    temperature_to_diff: Dict[str, str],
    matrix_root: Path,
    instance_id: str,
    lang: str,
    metric_name: str = "voyage",
    filename: str = "ground_truth_similarity.csv",
) -> Tuple[pd.DataFrame, Path]:
    """Build/save transformed ground-truth similarities plus diagnostics."""
    del lang

    if not temperature_to_diff:
        raise ValueError("temperature_to_diff is empty")

    output_dir = _matrix_output_dir(
        matrix_root=matrix_root,
        instance_id=instance_id,
        metric_name=metric_name,
    )

    (
        temperatures,
        raw_patch_by_temperature,
        parsed_changes_by_temperature,
        candidate_embedding_map,
    ) = _load_or_build_candidate_change_embeddings(
        temperature_to_diff=temperature_to_diff,
        output_dir=output_dir,
    )

    ground_truth_text = str(ground_truth_patch or "")
    ground_truth_files = _file_change_mapping(ground_truth_text)
    ground_truth_embeddings = _embed_file_change_mapping(ground_truth_files)

    scores: List[float] = []
    jaccard_scores: List[float] = []
    shared_only_scores: List[float] = []
    untransformed_union_scores: List[float] = []

    for temp in temperatures:
        candidate_text = str(raw_patch_by_temperature[temp] or "")

        if not ground_truth_text.strip() and not candidate_text.strip():
            score = 1.0
            jaccard_score = 1.0
            shared_only_score = 1.0
            untransformed_union_score = 1.0
        elif not ground_truth_text.strip() or not candidate_text.strip():
            score = 0.0
            jaccard_score = 0.0
            shared_only_score = 0.0
            untransformed_union_score = 0.0
        else:
            candidate_embeddings = _embeddings_for_temperature(
                temp,
                candidate_embedding_map,
            )

            (
                score,
                jaccard_score,
                shared_only_score,
                untransformed_union_score,
            ) = _patch_similarity_decomposition_from_precomputed(
                files_a=ground_truth_files,
                files_b=parsed_changes_by_temperature[temp],
                embeddings_a=ground_truth_embeddings,
                embeddings_b=candidate_embeddings,
            )

        scores.append(float(score))
        jaccard_scores.append(float(jaccard_score))
        shared_only_scores.append(float(shared_only_score))
        untransformed_union_scores.append(float(untransformed_union_score))

    ground_truth_similarity_df = pd.DataFrame(
        [scores], index=["ground_truth"], columns=temperatures
    )
    ground_truth_jaccard_df = pd.DataFrame(
        [jaccard_scores], index=["ground_truth"], columns=temperatures
    )
    ground_truth_shared_only_df = pd.DataFrame(
        [shared_only_scores], index=["ground_truth"], columns=temperatures
    )
    ground_truth_untransformed_union_df = pd.DataFrame(
        [untransformed_union_scores], index=["ground_truth"], columns=temperatures
    )

    ground_truth_similarity_path = output_dir / filename
    ground_truth_similarity_df.to_csv(ground_truth_similarity_path)

    ground_truth_jaccard_df.to_csv(
        output_dir / JACCARD_GROUND_TRUTH_FILENAME
    )
    ground_truth_shared_only_df.to_csv(
        output_dir / SHARED_ONLY_GROUND_TRUTH_FILENAME
    )
    ground_truth_untransformed_union_df.to_csv(
        output_dir / UNTRANSFORMED_UNION_GROUND_TRUTH_FILENAME
    )

    return ground_truth_similarity_df, ground_truth_similarity_path

