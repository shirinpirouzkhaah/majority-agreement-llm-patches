import re
from pathlib import Path
import subprocess
import pandas as pd
import shutil


def get_files_from_diff(diff_text):
    """
    Extract file names from a git diff.

    Example input line:
    diff --git a/src/sqlfluff/rules/L039.py b/src/sqlfluff/rules/L039.py

    Returns:
    ["src/sqlfluff/rules/L039.py"]
    """
    file_names = []

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()

            # Expected format:
            # ["diff", "--git", "a/path", "b/path"]
            if len(parts) >= 4:
                old_path = parts[2]  # a/src/...
                new_path = parts[3]  # b/src/...

                # Usually we want the new path from b/
                if new_path.startswith("b/"):
                    file_path = new_path[2:]
                elif old_path.startswith("a/"):
                    file_path = old_path[2:]
                else:
                    file_path = new_path

                if file_path not in file_names:
                    file_names.append(file_path)

    return file_names


def get_patch_files_by_index(ds, split, index):
    """
    Read row = ds[split][index],
    then read row["patch"],
    then return all file names changed in that patch.
    """
    row = ds[split][index]
    patch_text = row["patch"]

    return get_files_from_diff(patch_text)


def file_list_has_test_file(file_paths):
    """
    Check whether any file path contains the word 'test'.

    Example:
    ["src/a.py", "tests/test_parser.py"] -> True
    ["src/sqlfluff/rules/L039.py"] -> False
    """
    for file_path in file_paths:
        if "test" in file_path.lower():
            return True

    return False


def get_code_block_from_text(text):
    """
    Extract content between <code> and </code> from the text column.
    Returns an empty string if no code block is found.
    """
    match = re.search(r"<code>(.*?)</code>", text, flags=re.DOTALL)

    if match is None:
        return ""

    return match.group(1)


def get_file_names_from_code_block(code_block):
    """
    Extract file names from lines like:

    [start of src/sqlfluff/testing/rules.py]
    [end of src/sqlfluff/testing/rules.py]

    We only use the [start of ...] lines to avoid duplicates.
    """
    file_names = []

    pattern = r"\[start of (.*?)\]"

    matches = re.findall(pattern, code_block)

    for file_name in matches:
        file_name = file_name.strip()

        if file_name not in file_names:
            file_names.append(file_name)

    return file_names


def get_text_code_files_by_index(ds, split, index):
    """
    Read row = ds[split][index],
    then read row["text"],
    then extract file names from the <code>...</code> section.
    """
    row = ds[split][index]
    text = row["text"]

    code_block = get_code_block_from_text(text)
    file_names = get_file_names_from_code_block(code_block)

    return file_names


def keep_only_src_files(file_paths):
    """
    Remove README files and files inside docs/, test/, tests/,
    example/, or examples/.

    Keep all other files, including Python files that do not start with src/.

    Example:
    [
        'README.md',
        'src/a.py',
        'sqlfluff/core/parser.py',
        'test/test_a.py',
        'tests/test_b.py',
        'docs/index.rst',
        'examples/demo.py'
    ]

    returns:
    [
        'src/a.py',
        'sqlfluff/core/parser.py'
    ]
    """
    kept_files = []

    for file_path in file_paths:
        normalized = file_path.strip()
        lower_path = normalized.lower()

        file_name = lower_path.split("/")[-1]

        # Remove README files such as README, README.md, README.rst
        if file_name == "readme" or file_name.startswith("readme."):
            continue

        # Remove files in these top-level folders
        excluded_prefixes = (
            "docs/",
            "test/",
            "tests/",
            "example/",
            "examples/",
        )

        if lower_path.startswith(excluded_prefixes):
            continue

        kept_files.append(normalized)

    return kept_files


def percentage_gold_files_in_prompt(src_prompt_files, golden_fix_files):
    """
    Return the percentage of files in golden_fix_files
    that also appear in src_prompt_files.

    Example:
    golden_fix_files = ["src/a.py", "src/b.py"]
    src_prompt_files = ["src/a.py", "src/c.py"]

    Result:
    50.0
    """
    if len(golden_fix_files) == 0:
        return 0.0

    src_prompt_set = set(src_prompt_files)

    matched_count = 0

    for file_path in golden_fix_files:
        if file_path in src_prompt_set:
            matched_count += 1

    percentage = (matched_count / len(golden_fix_files)) * 100

    return percentage


def build_github_links_by_index(ds, split, index):
    """
    Build GitHub links for one SWT-Bench instance.

    Returns:
    {
        "commit_link": "...",
        "issue_link": "...",
        "pr_link": "..."
    }
    """
    row = ds[split][index]

    repo = row["repo"]
    base_commit = row["base_commit"]
    instance_id = row["instance_id"]

    # Example instance_id:
    # sqlfluff__sqlfluff-1625
    issue_or_pr_number = instance_id.split("-")[-1]

    commit_link = f"https://github.com/{repo}/commit/{base_commit}"
    issue_link = f"https://github.com/{repo}/issues"
    pr_link = f"https://github.com/{repo}/pull"

    return {
        "commit_link": commit_link,
        "issue_link": issue_link,
        "pr_link": pr_link,
    }


def clone_repo_and_checkout_base_commit(ds, split, index, repos_root="repos"):
    """
    Clone the repository once into a cached directory, then copy it to a fresh
    working directory and check out the instance base_commit.

    Returns the absolute path to the checked-out working repository.
    """
    row = ds[split][index]

    repo_name = row["repo"]
    base_commit = row["base_commit"]
    instance_id = row["instance_id"]

    repos_root = Path(repos_root)
    repos_root.mkdir(parents=True, exist_ok=True)

    repo_url = f"https://github.com/{repo_name}.git"

    # Cached clone shared across all instances from the same repository.
    cache_root = repos_root / "_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    cached_repo_name = repo_name.replace("/", "__")
    cached_repo_dir = cache_root / cached_repo_name

    # Fresh working copy for this specific instance.
    repo_dir = repos_root / instance_id

    if not cached_repo_dir.exists():
        print(f"Cloning repository once into cache: {repo_url}")
        subprocess.run(
            ["git", "clone", repo_url, str(cached_repo_dir)],
            check=True,
        )
    else:
        print(f"Using cached repository: {cached_repo_dir}")

    # Always create a fresh working copy for the instance.
    if repo_dir.exists():
        print(f"Removing old working directory: {repo_dir}")
        shutil.rmtree(repo_dir)

    print(f"Creating fresh working copy: {repo_dir}")
    shutil.copytree(
        cached_repo_dir,
        repo_dir,
        symlinks=True,
    )

    print(f"Checking out base_commit: {base_commit}")
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=repo_dir,
        check=True,
    )

    return str(repo_dir.resolve())


def get_file_path_mapping_in_repo(repo_dir, file_paths):
    """
    Given a checked-out repository and a list of relative file paths,
    return a dataframe mapping each original relative path to its absolute path.

    If the file does not exist, abs_path is None.

    Returns dataframe with columns:
        path
        abs_path
        exists
    """
    repo_dir = Path(repo_dir).resolve()

    rows = []

    for file_path in file_paths:
        abs_path = (repo_dir / file_path).resolve()

        if abs_path.exists():
            rows.append({
                "path": file_path,
                "abs_path": str(abs_path),
                "exists": True,
            })
        else:
            rows.append({
                "path": file_path,
                "abs_path": None,
                "exists": False,
            })

    return pd.DataFrame(rows)

REPO_TO_LANGUAGE = {
    "NodeBB/NodeBB": "js",
    "ansible/ansible": "python",
    "element-hq/element-web": "js",
    "flipt-io/flipt": "go",
    "future-architect/vuls": "go",
    "gravitational/teleport": "go",
    "internetarchive/openlibrary": "python",
    "navidrome/navidrome": "go",
    "protonmail/webclients": "js",
    "qutebrowser/qutebrowser": "python",
    "tutao/tutanota": "ts",
}

LANGUAGE_TO_EXAMPLE_FILE = {
    "python": "path/to/file.py",
    "js": "path/to/file.js",
    "ts": "path/to/file.ts",
    "go": "path/to/file.go",
}

IGNORE_GOLDEN_FIX_EXTENSIONS = {
    # docs / text
    "md", "mdx", "txt", "rst", "asciidoc",

    # images / media
    "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "icns",
    "mp3", "ogg", "wav", "mp4", "webm", "flac", "m4a", "aiff", "opus", "wma",

    # fonts
    "ttf", "otf", "woff", "woff2",

    # archives / binaries
    "zip", "tar", "gz", "rpm", "deb", "exe", "dll", "bin", "wasm", "jar", "aar",
    "o", "pdb", "sqlite", "db",

    # snapshots / generated expected outputs
    "snap", "golden", "expected", "output", "stdout", "stderr",

    # patches / diffs / logs
    "patch", "diff", "log",

    # certs / keys / signatures
    "pem", "crt", "key", "pub", "gpg", "pgp", "sig", "asc", "keytab",

    # raw design / metadata-ish files
    "psd1", "metadata", "thumbnail", "cache",
}

KEEP_EXTENSIONS = {
    "py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "kt", "kts",
    "swift", "c", "cpp", "h", "hpp", "cs", "rb", "pl",
    "json", "json5", "jsonc", "yaml", "yml", "toml", "ini", "cfg", "conf",
    "xml", "html", "css", "scss", "pcss",
    "sh", "bash", "bat", "cmd", "ps1", "psm1",
    "sql", "proto", "tf", "tfvars", "hcl", "nix", "rego",
    "vue", "svelte", "mjs", "cjs",
    "tpl", "tmpl", "j2", "ejs", "mustache",
    "dockerfile", "dockerignore", "gradle", "mk", "properties",
}


def normalize_extension(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if not suffix:
        return "[no extension]"

    return suffix.lstrip(".")


def filter_golden_fix_files(golden_fix_files: list[str]) -> list[str]:
    filtered = []

    for file_path in golden_fix_files:
        ext = normalize_extension(file_path)

        if ext in IGNORE_GOLDEN_FIX_EXTENSIONS:
            continue

        filtered.append(file_path)

    return filtered

def build_fix_prompt_from_files(repo_dir, path_df, problem_statement, repo):
    repo_dir = Path(repo_dir).resolve()

    if repo not in REPO_TO_LANGUAGE:
        raise ValueError(
            f"Repository language is unknown for repo={repo}. "
            f"Known repos: {sorted(REPO_TO_LANGUAGE.keys())}"
        )

    repo_language = REPO_TO_LANGUAGE[repo]
    example_file_path = LANGUAGE_TO_EXAMPLE_FILE.get(
        repo_language,
        "path/to/file.txt",
    )

    prompt_parts = []

    existing_files_df = path_df[
        (path_df["exists"] == True) &
        (path_df["abs_path"].notna())
    ]

    n = len(existing_files_df)
    file_word = "file" if n == 1 else "files"
    these_word = "This is" if n == 1 else "These are"
    each_file = "This provided source file" if n == 1 else "Each provided source file"
    there_are = "There is" if n == 1 else "There are"
    change_permission = "the only file that can be changed" if n == 1 else "the only files that can be changed"



    task_description = (
        "The following text contains an issue posted at a repository, provided inside <issue> and </issue> tags. "
        f"You are also provided with the content of {n} source code {file_word} from the repository inside <code> and </code> tags. "
        f"{these_word} {change_permission} to solve the issue. "
        f"{each_file} is relevant to the issue and must receive at least one change. "
        f"{each_file} is enclosed within file-boundary markers. "
        f"For example, the beginning of a file is marked as [start of {example_file_path}], and the end is marked as [end of {example_file_path}]. "
        "The file-boundary markers are not part of the source code. "
        "The path inside each marker is the repository-relative file path. "
        "When specifying which file to edit in your response, copy the file path exactly as it appears in the file-boundary markers. "
        "Do not add, remove, rename, shorten, extend, or modify any part of the path. "
        "You are not allowed to mention, edit, or propose changes to any file path or file content that is not shown in this prompt. "
        f"Your task is to fix the issue using only the issue description and the provided {file_word}. "
    )
    
    

    prompt_parts.append(task_description)

    prompt_parts.append("\n\n<issue>")
    prompt_parts.append(problem_statement)
    prompt_parts.append("</issue>")

    prompt_parts.append("\n<code>")

    for _, file_row in existing_files_df.iterrows():
        original_path = file_row["path"]
        abs_path = Path(file_row["abs_path"])

        file_content = abs_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        prompt_parts.append(f"\n[start of {original_path}]")
        prompt_parts.append(file_content)
        prompt_parts.append(f"[end of {original_path}]")

    prompt_parts.append("</code>")

    patch_instruction = (
        "\n\nReturn only file-edit instructions inside <edits> and </edits> tags. "
        "Do not return a unified diff. "
        "Do not include explanations, markdown, comments about the fix, or any text outside the <edits> block. "
        "Your output must start with <edits> and end with </edits>. "
        "Each code edit must be represented by its own <edit> block. "
        "Use separate <edit> blocks for each replacement. "
        "Multiple edits can target the same file or multiple files. "
        f"{there_are} {n} provided source code {file_word}. {each_file} must receive at least one <edit> block.\n\n"

        "Use this exact format:\n\n"
        "<edits>\n"
        "<edit>\n"
        "<file>\n"
        f"{example_file_path}\n"
        "</file>\n"
        "<replace_this>\n"
        "exact buggy code segment copied from the provided file, with exact indentation\n"
        "</replace_this>\n"
        "<with_this>\n"
        "fixed version of that exact code segment\n"
        "</with_this>\n"
        "</edit>\n"
        "<edit>\n"
        "<file>\n"
        f"path/to/anotherFile{Path(example_file_path).suffix}\n"
        "</file>\n"
        "<replace_this>\n"
        "another exact buggy code segment copied from the provided file, with exact indentation\n"
        "</replace_this>\n"
        "<with_this>\n"
        "fixed version of that exact code segment\n"
        "</with_this>\n"
        "</edit>\n"
        "</edits>\n\n"

        "Strict rules:\n"
        "1. Every path inside a <file> block must be copied exactly from one of the [start of ...] markers in the prompt.\n"
        "2. Do not invent, rename, shorten, extend, or modify file paths. Do not use any file path that is not shown in this prompt in a [start of ...] marker.\n"
        "3. Do not mention, edit, or propose changes to any file or file content that is not shown inside the file-boundary markers in this prompt.\n"
        "4. The <replace_this> block must be copied exactly from the corresponding provided source file.\n"
        "5. The <replace_this> block must preserve the exact indentation, spaces, blank lines, and line breaks from the provided source file.\n"
        "6. The <replace_this> block must be a continuous code segment from one single file.\n"
        "7. The <replace_this> block must be specific enough to occur exactly once in the corresponding provided file.\n"
        "8. The <replace_this> block must be directly findable in the original file using a normal Python string replacement operation.\n"
        "9. Do not paraphrase, reformat, summarize, reconstruct, or change indentation inside <replace_this>; copy it exactly as shown.\n"
        "10. The <with_this> block must contain the corrected replacement code for the exact <replace_this> block.\n"
        "11. The <with_this> block must use indentation that is valid for the same location as the original code.\n"
        "12. Do not include the [start of ...] or [end of ...] file-boundary markers in any <replace_this> or <with_this> block.\n"
        "13. Multiple edits can be made in the same file, but each separate replacement must use a separate <edit> block.\n\n"

        "Before writing the final answer, internally check that:\n"
        "- Every <file> path is exactly one of the provided [start of ...] paths.\n"
        "- Every provided file has at least one <edit> block.\n"
        "- Every <replace_this> block is copied exactly from the corresponding file content provided in the prompt.\n"
        "- No extra explanation, markdown, or text outside <edits> is included.\n"
    )

    prompt_parts.append(patch_instruction)

    return "\n".join(prompt_parts)


def delete_repo_dir(repo_dir):
    """
    Delete a cloned repository directory after it is no longer needed.
    """
    repo_dir = Path(repo_dir)

    if repo_dir.exists() and repo_dir.is_dir():
        print(f"Deleting repo directory: {repo_dir}")
        shutil.rmtree(repo_dir)
    else:
        print(f"Repo directory does not exist: {repo_dir}")
