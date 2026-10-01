#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# CodeBERT configuration
# ---------------------------------------------------------------------------

CODEBERT_MODEL_NAME = "microsoft/codebert-base"
DEVICE = torch.device("cpu")

CODEBERT_TOKENIZER = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)

CODEBERT_MODEL = AutoModel.from_pretrained(
    CODEBERT_MODEL_NAME,
    use_safetensors=True,
).to(DEVICE)

CODEBERT_MODEL.eval()

# User-requested operational limit. CodeBERT itself supports up to 512 tokens,
# but every prepared chunk in this experiment is capped at 500 tokens including
# tokenizer-added special tokens.
MAX_CODEBERT_TOKENS = 500

# Preserve the existing CPU-friendly chunk batch size.
CODEBERT_BATCH_SIZE = 8

# Shared-file cosine calibration. The raw removed/added cosine-channel mean for
# each shared file is transformed before patch-level averaging over the file union.
CODEBERT_DISTANCE_SCALE = 0.5

# Used only to invalidate old embedding caches after changing preprocessing.
PREPROCESSING_VERSION = "file_removed_added_union_v2_actual_token_chunks_empty_patch_rules"

# Additional diagnostic CSVs. The existing similarity_matrix.csv and
# ground_truth_similarity.csv filenames remain unchanged; their shared-file scores
# use the 0.5 exponential transformation defined above. The shared-only and
# untransformed-union diagnostics keep raw CodeBERT cosine scores.
JACCARD_MATRIX_FILENAME = "jaccard_similarity_matrix.csv"
JACCARD_GROUND_TRUTH_FILENAME = "ground_truth_jaccard_similarity.csv"
SHARED_ONLY_MATRIX_FILENAME = "shared_files_only_similarity_matrix.csv"
SHARED_ONLY_GROUND_TRUTH_FILENAME = "ground_truth_shared_files_only_similarity.csv"
UNTRANSFORMED_UNION_MATRIX_FILENAME = "untransformed_union_similarity_matrix.csv"
UNTRANSFORMED_UNION_GROUND_TRUTH_FILENAME = (
    "ground_truth_untransformed_union_similarity.csv"
)


# ---------------------------------------------------------------------------
# Diff parsing -- intentionally matches codebleu_similarity.py
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

    This preprocessing intentionally matches codebleu_similarity.py so CodeBLEU
    and CodeBERT compare the same file/change units.
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


# ---------------------------------------------------------------------------
# CodeBERT embedding -- preserves the previous chunking/pooling design
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _mean_pool_content_tokens(
    last_hidden_state: torch.Tensor,
    content_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool non-special, non-padding token states for every chunk."""
    mask = content_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    pooled = (last_hidden_state * mask).sum(dim=1) / token_counts
    return pooled, token_counts.squeeze(-1)


def _max_content_tokens() -> int:
    """Return payload capacity after reserving tokenizer special tokens."""
    num_special_tokens = int(
        CODEBERT_TOKENIZER.num_special_tokens_to_add(pair=False)
    )
    capacity = MAX_CODEBERT_TOKENS - num_special_tokens
    if capacity <= 0:
        raise ValueError(
            "MAX_CODEBERT_TOKENS is too small after reserving special tokens."
        )
    return capacity


def split_text_into_codebert_token_chunks(text: str) -> List[List[int]]:
    """
    Split text into complete, non-overlapping CodeBERT token chunks.

    Actual CodeBERT tokenization alone determines whether chunking is required.
    Every prepared chunk is at most 500 tokens including tokenizer-added special
    tokens, and no tokenized content is discarded.
    """
    text = str(text or "")
    if not text:
        return [[]]

    content_token_ids: List[int] = CODEBERT_TOKENIZER.encode(
        text,
        add_special_tokens=False,
    )

    if not content_token_ids:
        return [[]]

    max_content_tokens = _max_content_tokens()

    num_chunks = max(
        1,
        int(math.ceil(len(content_token_ids) / max_content_tokens)),
    )

    # Split as evenly as possible while preserving order and every token.
    chunk_size = int(math.ceil(len(content_token_ids) / num_chunks))
    chunk_size = min(chunk_size, max_content_tokens)

    return [
        content_token_ids[start : start + chunk_size]
        for start in range(0, len(content_token_ids), chunk_size)
    ]


def _prepare_chunk_batch(
    token_chunks: Sequence[Sequence[int]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Add special tokens, pad a chunk batch, and build a content-token mask."""
    encoded_chunks = []
    content_masks: List[List[int]] = []

    for chunk in token_chunks:
        chunk_ids = list(chunk)
        input_ids = CODEBERT_TOKENIZER.build_inputs_with_special_tokens(chunk_ids)

        if len(input_ids) > MAX_CODEBERT_TOKENS:
            raise ValueError(
                f"Prepared CodeBERT chunk has {len(input_ids)} tokens; "
                f"maximum is {MAX_CODEBERT_TOKENS}."
            )

        special_tokens_mask = CODEBERT_TOKENIZER.get_special_tokens_mask(
            input_ids,
            already_has_special_tokens=True,
        )
        content_mask = [
            1 if is_special == 0 else 0
            for is_special in special_tokens_mask
        ]

        encoded_chunks.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
        )
        content_masks.append(content_mask)

    padded = CODEBERT_TOKENIZER.pad(
        encoded_chunks,
        padding=True,
        return_tensors="pt",
    )

    sequence_length = int(padded["input_ids"].shape[1])
    padded_content_masks = [
        mask + [0] * (sequence_length - len(mask))
        for mask in content_masks
    ]
    content_mask_tensor = torch.tensor(
        padded_content_masks,
        dtype=torch.float32,
        device=DEVICE,
    )

    model_inputs = {
        key: value.to(DEVICE)
        for key, value in padded.items()
    }
    return model_inputs, content_mask_tensor


@torch.inference_mode()
def embed_patch(text: str) -> np.ndarray:
    """
    Embed one removed/added code fragment into one normalized CodeBERT vector.

    The method name is kept for backward compatibility. Each fragment is split
    only when needed. Every chunk is mean-pooled over its content tokens, then
    chunk vectors are combined using the existing token-count-weighted mean.
    """
    text = str(text or "")

    if not text.strip():
        hidden_size = int(CODEBERT_MODEL.config.hidden_size)
        return np.zeros(hidden_size, dtype=np.float32)

    token_chunks = split_text_into_codebert_token_chunks(text)

    weighted_embedding_sum: Optional[torch.Tensor] = None
    total_content_tokens = 0.0

    for start in range(0, len(token_chunks), CODEBERT_BATCH_SIZE):
        batch_chunks = token_chunks[start : start + CODEBERT_BATCH_SIZE]
        model_inputs, content_mask = _prepare_chunk_batch(batch_chunks)

        outputs = CODEBERT_MODEL(**model_inputs)
        chunk_embeddings, chunk_token_counts = _mean_pool_content_tokens(
            last_hidden_state=outputs.last_hidden_state,
            content_mask=content_mask,
        )

        batch_weighted_sum = (
            chunk_embeddings * chunk_token_counts.unsqueeze(-1)
        ).sum(dim=0)
        batch_token_count = float(chunk_token_counts.sum().item())

        if weighted_embedding_sum is None:
            weighted_embedding_sum = batch_weighted_sum
        else:
            weighted_embedding_sum = weighted_embedding_sum + batch_weighted_sum

        total_content_tokens += batch_token_count

    if weighted_embedding_sum is None or total_content_tokens <= 0:
        hidden_size = int(CODEBERT_MODEL.config.hidden_size)
        return np.zeros(hidden_size, dtype=np.float32)

    fragment_embedding = weighted_embedding_sum / total_content_tokens
    fragment_embedding = torch.nn.functional.normalize(
        fragment_embedding,
        p=2,
        dim=0,
    )

    return fragment_embedding.detach().cpu().numpy().astype(np.float32)


def embed_patches(texts: Iterable[str]) -> np.ndarray:
    """Embed an iterable of code fragments into a two-dimensional NumPy array."""
    embeddings = [embed_patch(text) for text in texts]
    if not embeddings:
        return np.empty(
            (0, int(CODEBERT_MODEL.config.hidden_size)),
            dtype=np.float32,
        )
    return np.vstack(embeddings)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute a full pairwise cosine matrix from normalized embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")

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
    if reference_embedding.ndim != 1:
        raise ValueError("reference_embedding must be one-dimensional")
    if candidate_embeddings.ndim != 2:
        raise ValueError("candidate_embeddings must be two-dimensional")
    if candidate_embeddings.shape[1] != reference_embedding.shape[0]:
        raise ValueError("reference and candidate embedding sizes do not match")

    similarities = candidate_embeddings @ reference_embedding
    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


def _cosine_similarity_between_embeddings(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """Cosine similarity for two already L2-normalized CodeBERT vectors."""
    embedding_a = np.asarray(embedding_a, dtype=np.float32)
    embedding_b = np.asarray(embedding_b, dtype=np.float32)

    if embedding_a.ndim != 1 or embedding_b.ndim != 1:
        raise ValueError("Both embeddings must be one-dimensional")

    if embedding_a.shape[0] != embedding_b.shape[0]:
        raise ValueError("Embedding dimensions do not match")

    # Non-empty fragments should not produce zero vectors, but this guard keeps
    # the behavior well-defined if a model/tokenizer edge case does occur.
    if np.linalg.norm(embedding_a) == 0.0 or np.linalg.norm(embedding_b) == 0.0:
        return 0.0

    return float(np.clip(np.dot(embedding_a, embedding_b), -1.0, 1.0))


# ---------------------------------------------------------------------------
# File-aware whole-patch CodeBERT similarity
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

    vectors = embed_patches(texts)
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
    Compare one removed/added channel using the same empty-channel rules as CodeBLEU.

    Rules:
      * nonempty vs nonempty -> CodeBERT cosine similarity
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
        raise ValueError("Missing embedding for a non-empty change channel")

    return _cosine_similarity_between_embeddings(embedding_a, embedding_b)


def shared_file_similarity(
    file_name: str,
    file_change_a: Dict[str, str],
    file_change_b: Dict[str, str],
    embeddings_a: Dict[Tuple[str, str], np.ndarray],
    embeddings_b: Dict[Tuple[str, str], np.ndarray],
) -> float:
    """
    Calculate similarity for one file present in both patches.

    Removed code is compared only with removed code; added code only with added
    code. Defined channel cosine scores are averaged, then the resulting shared-file
    score is transformed as exp(-(1 - cosine_mean) / 0.5). A channel empty in both
    patches is excluded; a channel present in only one patch contributes zero.
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
    return float(
        math.exp(-(1.0 - raw_file_score) / CODEBERT_DISTANCE_SCALE)
    )


def _raw_shared_file_cosine_similarity(
    file_name: str,
    file_change_a: Dict[str, str],
    file_change_b: Dict[str, str],
    embeddings_a: Dict[Tuple[str, str], np.ndarray],
    embeddings_b: Dict[Tuple[str, str], np.ndarray],
) -> float:
    """Return the untransformed mean CodeBERT cosine for one shared file.

    This deliberately bypasses any current/future calibration that may be applied
    inside shared_file_similarity(). Empty-channel behavior remains identical:
    both empty is excluded; exactly one empty contributes zero.
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
    """Return four related patch-level quantities.

    Returns:
        current_union_similarity:
            The score used by similarity_matrix.csv. Each shared file first uses
            the 0.5 exponential transformation in shared_file_similarity();
            unshared files contribute zero; scores are averaged over the union.
        jaccard_file_similarity:
            |shared changed files| / |union changed files|.
        shared_only_similarity:
            Mean of the RAW, untransformed shared-file CodeBERT cosine scores over
            shared files only. Each raw shared-file score is the mean of its defined
            removed/added cosine channels under the existing empty-channel rules.
        untransformed_union_similarity:
            Mean over the union using those same RAW shared-file cosine scores, with
            unshared files contributing zero.

    For non-empty file unions, untransformed_union_similarity ==
    jaccard_file_similarity * shared_only_similarity (up to floating precision).
    """
    union_files = sorted(set(files_a) | set(files_b))

    if not union_files:
        return 0.0, 0.0, 0.0, 0.0

    shared_files = sorted(set(files_a) & set(files_b))
    union_count = len(union_files)
    shared_count = len(shared_files)

    jaccard_file_similarity = float(shared_count / union_count)

    current_shared_scores: List[float] = []
    raw_shared_scores: List[float] = []

    for file_name in shared_files:
        current_shared_scores.append(
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
        shared_only_similarity = 0.0
        current_union_similarity = 0.0
        untransformed_union_similarity = 0.0
    else:
        # Diagnostic shared-only CSV: raw/untransformed CodeBERT similarity,
        # averaged only over files changed by both patches.
        shared_only_similarity = float(
            sum(raw_shared_scores) / shared_count
        )

        # Main similarity CSV: transformed shared-file scores, unmatched files=0,
        # then averaged over the full union of changed files.
        current_union_similarity = float(
            sum(current_shared_scores) / union_count
        )

        # Raw union diagnostic: raw shared-file scores, unmatched files=0, then
        # averaged over the full union of changed files.
        untransformed_union_similarity = float(
            sum(raw_shared_scores) / union_count
        )

    # Requested raw-score decomposition:
    # raw union = file Jaccard * raw shared-files-only mean.
    if not np.isclose(
        untransformed_union_similarity,
        jaccard_file_similarity * shared_only_similarity,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError(
            "CodeBERT raw union similarity decomposition failed: "
            "untransformed union != Jaccard * raw shared-only similarity"
        )

    return (
        current_union_similarity,
        jaccard_file_similarity,
        shared_only_similarity,
        untransformed_union_similarity,
    )


def _whole_patch_similarity_from_precomputed(
    files_a: Dict[str, Dict[str, str]],
    files_b: Dict[str, Dict[str, str]],
    embeddings_a: Dict[Tuple[str, str], np.ndarray],
    embeddings_b: Dict[Tuple[str, str], np.ndarray],
) -> float:
    """Calculate final whole-patch similarity from parsed changes/embeddings."""
    current_union_similarity, _, _, _ = (
        _patch_similarity_decomposition_from_precomputed(
            files_a=files_a,
            files_b=files_b,
            embeddings_a=embeddings_a,
            embeddings_b=embeddings_b,
        )
    )
    return float(current_union_similarity)


def whole_patch_similarity(diff_a: str, diff_b: str) -> float:
    """
    Calculate one file-aware CodeBERT similarity score for two complete diffs.

    Procedure:
      1. Parse each diff into one entry per changed file.
      2. For each file, keep removed and added code separately.
      3. Use the union of file paths from both diffs.
      4. Shared files: cosine(removed, removed) and cosine(added, added), average
         the defined channel scores, then transform the shared-file score with
         exp(-(1 - score) / 0.5).
      5. Files changed by only one diff receive file similarity 0.
      6. Average transformed shared-file scores and literal zeros across the union
         of changed files.

    The returned value is the transformed-file, union-averaged CodeBERT similarity
    written to similarity_matrix.csv.
    """
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
            "returning whole-patch CodeBERT similarity 0.0"
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


# Explicit two-diff method for testing/direct use.
def compare_two_diffs(diff_a: str, diff_b: str) -> float:
    """Return the final file-aware whole-patch CodeBERT similarity."""
    return whole_patch_similarity(diff_a, diff_b)


# ---------------------------------------------------------------------------
# Clustering-pipeline compatibility
# ---------------------------------------------------------------------------


def get_lang_for_repo(repo: str) -> str:
    """Compatibility helper; CodeBERT does not require a language argument."""
    del repo
    return "code"


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


def _candidate_embedding_cache_paths(output_dir: Path) -> Tuple[Path, Path]:
    # Keep the existing cache filenames so the surrounding pipeline does not need
    # to know that the cache now stores per-file/per-channel embeddings.
    return (
        output_dir / "candidate_embeddings.npz",
        output_dir / "candidate_embeddings_metadata.json",
    )


def _serialize_embedding_key(temp: str, file_name: str, channel: str) -> str:
    return json.dumps([str(temp), str(file_name), str(channel)], ensure_ascii=False)


def _deserialize_embedding_key(value: str) -> Tuple[str, str, str]:
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
    temperatures = [str(temp) for temp in temperature_to_diff.keys()]
    patches = [str(temperature_to_diff[temp] or "") for temp in temperature_to_diff.keys()]

    raw_patch_by_temperature = {
        temp: patch
        for temp, patch in zip(temperatures, patches)
    }

    expected_hashes = {
        temp: _text_sha256(patch)
        for temp, patch in zip(temperatures, patches)
    }

    parsed_changes_by_temperature: Dict[str, Dict[str, Dict[str, str]]] = {
        temp: _file_change_mapping(patch)
        for temp, patch in zip(temperatures, patches)
    }

    cache_path, metadata_path = _candidate_embedding_cache_paths(output_dir)

    if cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached = np.load(cache_path, allow_pickle=False)

            cached_temperatures = [str(value) for value in cached["temperatures"]]
            cached_keys = [str(value) for value in cached["embedding_keys"]]
            cached_embeddings = cached["embeddings"].astype(np.float32)

            cache_matches = (
                metadata.get("model_name") == CODEBERT_MODEL_NAME
                and metadata.get("preprocessing_version") == PREPROCESSING_VERSION
                and metadata.get("max_codebert_tokens") == MAX_CODEBERT_TOKENS
                and metadata.get("patch_sha256") == expected_hashes
                and cached_temperatures == temperatures
                and cached_embeddings.ndim == 2
                and cached_embeddings.shape[0] == len(cached_keys)
                and cached_embeddings.shape[1] == int(CODEBERT_MODEL.config.hidden_size)
            )

            if cache_matches:
                embedding_map: Dict[Tuple[str, str, str], np.ndarray] = {}
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
                "WARNING: failed to read CodeBERT file/channel embedding cache at "
                f"{cache_path}: {exc}. Recomputing embeddings."
            )

    embedding_keys: List[Tuple[str, str, str]] = []
    embedding_texts: List[str] = []

    for temp in temperatures:
        file_changes = parsed_changes_by_temperature[temp]

        for file_name in sorted(file_changes):
            for channel in ("removed", "added"):
                text = str(file_changes[file_name].get(channel, "") or "")
                if not text.strip():
                    continue

                embedding_keys.append((temp, file_name, channel))
                embedding_texts.append(text)

    if embedding_texts:
        embeddings = embed_patches(embedding_texts)
    else:
        embeddings = np.empty(
            (0, int(CODEBERT_MODEL.config.hidden_size)),
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
                "model_name": CODEBERT_MODEL_NAME,
                "preprocessing_version": PREPROCESSING_VERSION,
                "max_codebert_tokens": MAX_CODEBERT_TOKENS,
                "chunking": (
                    "actual CodeBERT tokenizer count only; chunk when content tokens "
                    "exceed the <=500-prepared-token capacity after special tokens"
                ),
                "pooling": (
                    "content-token mean per <=500-token chunk; "
                    "token-count-weighted mean across chunks; final L2 normalization"
                ),
                "patch_preprocessing": (
                    "parse by file; concatenate all removed lines and all added "
                    "lines across hunks; remove unchanged context and diff metadata; "
                    "compare removed-to-removed and added-to-added; unmatched files=0; "
                    "average channels per shared file, then average over file union; "
                    "empty-empty patch=1 and exactly-one-empty patch=0"
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


def _candidate_patch_similarity_from_cache(
    temp_a: str,
    temp_b: str,
    raw_patch_by_temperature: Dict[str, str],
    parsed_changes_by_temperature: Dict[str, Dict[str, Dict[str, str]]],
    embedding_map: Dict[Tuple[str, str, str], np.ndarray],
) -> float:
    """Compare two candidate patches using cached file/channel embeddings."""
    raw_a = str(raw_patch_by_temperature[temp_a] or "")
    raw_b = str(raw_patch_by_temperature[temp_b] or "")

    if not raw_a.strip() and not raw_b.strip():
        return 1.0

    if not raw_a.strip() or not raw_b.strip():
        return 0.0

    files_a = parsed_changes_by_temperature[temp_a]
    files_b = parsed_changes_by_temperature[temp_b]

    embeddings_a = {
        (file_name, channel): embedding
        for (temp, file_name, channel), embedding in embedding_map.items()
        if temp == temp_a
    }
    embeddings_b = {
        (file_name, channel): embedding
        for (temp, file_name, channel), embedding in embedding_map.items()
        if temp == temp_b
    }

    return _whole_patch_similarity_from_precomputed(
        files_a=files_a,
        files_b=files_b,
        embeddings_a=embeddings_a,
        embeddings_b=embeddings_b,
    )


def build_similarity_matrix(
    temperature_to_diff: Dict[str, str],
    lang: str = "code",
) -> pd.DataFrame:
    """Build the file-aware whole-patch CodeBERT similarity matrix in memory."""
    del lang

    temperature_keys = list(temperature_to_diff.keys())
    valid_temperatures = [str(temp) for temp in temperature_keys]

    # This uncached public helper is useful for direct use/testing. The pipeline's
    # build_and_save_similarity_matrix() below uses the persistent cache instead.
    raw_patch_by_temperature = {
        str(temp): str(temperature_to_diff[temp] or "")
        for temp in temperature_keys
    }
    parsed = {
        temp: _file_change_mapping(raw_patch_by_temperature[temp])
        for temp in valid_temperatures
    }
    embeddings_by_temp = {
        temp: _embed_file_change_mapping(parsed[temp])
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
                    files_a=parsed[temp_i],
                    files_b=parsed[temp_j],
                    embeddings_a=embeddings_by_temp[temp_i],
                    embeddings_b=embeddings_by_temp[temp_j],
                )

            similarity_df.loc[temp_i, temp_j] = float(score)
            similarity_df.loc[temp_j, temp_i] = float(score)

    return similarity_df


def build_and_save_similarity_matrix(
    temperature_to_diff: Dict[str, str],
    matrix_root: Path,
    instance_id: str,
    lang: str,
    metric_name: str = "codebert",
    filename: str = "similarity_matrix.csv",
) -> Tuple[pd.DataFrame, Path]:
    """
    Build and save the candidate-to-candidate file-aware CodeBERT similarity matrix.

    This function name/signature remains compatible with the clustering code.
    """
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
        0.0,
        index=temperatures,
        columns=temperatures,
        dtype=float,
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
        # A patch compared with itself is identical for all four quantities.
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
                embeddings_i = {
                    (file_name, channel): embedding
                    for (temp, file_name, channel), embedding in embedding_map.items()
                    if temp == temp_i
                }
                embeddings_j = {
                    (file_name, channel): embedding
                    for (temp, file_name, channel), embedding in embedding_map.items()
                    if temp == temp_j
                }

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

    # Existing output remains unchanged.
    matrix_path = output_dir / filename
    similarity_df.to_csv(matrix_path)

    # Additional requested decompositions/diagnostics.
    jaccard_df.to_csv(output_dir / JACCARD_MATRIX_FILENAME)
    shared_only_df.to_csv(output_dir / SHARED_ONLY_MATRIX_FILENAME)
    untransformed_union_df.to_csv(
        output_dir / UNTRANSFORMED_UNION_MATRIX_FILENAME
    )

    return similarity_df, matrix_path


def build_ground_truth_similarity_vector(
    ground_truth_patch: str,
    temperature_to_diff: Dict[str, str],
    lang: str = "code",
) -> pd.DataFrame:
    """Build ground-truth-to-candidate file-aware CodeBERT similarities."""
    del lang

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


def build_and_save_ground_truth_similarity_vector(
    ground_truth_patch: str,
    temperature_to_diff: Dict[str, str],
    matrix_root: Path,
    instance_id: str,
    lang: str,
    metric_name: str = "codebert",
    filename: str = "ground_truth_similarity.csv",
) -> Tuple[pd.DataFrame, Path]:
    """
    Build and save ground-truth-to-candidate file-aware CodeBERT similarities.

    Candidate file/channel embeddings are loaded from the same cache created by
    build_and_save_similarity_matrix(), so candidate code is not re-embedded.
    """
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
            candidate_embeddings = {
                (file_name, channel): embedding
                for (cached_temp, file_name, channel), embedding in candidate_embedding_map.items()
                if cached_temp == temp
            }

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
        [scores],
        index=["ground_truth"],
        columns=temperatures,
    )
    ground_truth_jaccard_df = pd.DataFrame(
        [jaccard_scores],
        index=["ground_truth"],
        columns=temperatures,
    )
    ground_truth_shared_only_df = pd.DataFrame(
        [shared_only_scores],
        index=["ground_truth"],
        columns=temperatures,
    )
    ground_truth_untransformed_union_df = pd.DataFrame(
        [untransformed_union_scores],
        index=["ground_truth"],
        columns=temperatures,
    )

    # Existing output remains unchanged.
    ground_truth_similarity_path = output_dir / filename
    ground_truth_similarity_df.to_csv(ground_truth_similarity_path)

    # Additional requested decompositions/diagnostics.
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
