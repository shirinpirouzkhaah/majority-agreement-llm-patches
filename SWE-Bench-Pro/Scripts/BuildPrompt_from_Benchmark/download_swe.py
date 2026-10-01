from pathlib import Path
from datasets import load_dataset
from utilities import *
import pandas as pd
import math
import shutil
import traceback


dataset_name = "ScaleAI/SWE-bench_Pro"
ds = load_dataset(dataset_name)
print("\nSplits:")
print(list(ds.keys()))

print("\nSplit sizes:")
for split_name in ds.keys():
    print(f"{split_name}: {len(ds[split_name])}")


split = "test"
rows = []


for index in range(len(ds[split])):
    # if row["repo"] is not "protonmail__webclients":
    #     continue
    row = ds[split][index]
    instance_id = row["instance_id"]
    repos_dirr = Path("repos")
    repo_dir = repos_dirr / instance_id
    out_dir = Path("prompts") / instance_id

    if out_dir.exists():
        print(f"Skipping {instance_id}: prompt directory already exists")
        continue

    try:
        golden_fix_files = get_patch_files_by_index(ds, split, index)
        golden_fix_files = filter_golden_fix_files(golden_fix_files)

        has_test = file_list_has_test_file(golden_fix_files)

        links = build_github_links_by_index(ds, split, index)

        repo_dir = clone_repo_and_checkout_base_commit(ds, split, index)

        gold_path_df = get_file_path_mapping_in_repo(
            repo_dir,
            golden_fix_files
        )

        oracle_prompt_text = build_fix_prompt_from_files(
            repo_dir=repo_dir,
            path_df=gold_path_df,
            problem_statement=row["problem_statement"],
            repo = row["repo"]
        )

        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "oracle_file_prompt.txt").write_text(
            oracle_prompt_text,
            encoding="utf-8"
        )

        rows.append({
            "index": index,
            "instance_id": instance_id,
            "golden_fix_files": golden_fix_files,
            "golden_fix_has_test_file": has_test,
            "commit_link": links["commit_link"],
            "issue_link": links["issue_link"],
            "pr_link": links["pr_link"],
            "status": "OK",
        })

    except Exception as e:
        print("=" * 80)
        print(f"FAILED issue: {instance_id}")
        print(f"Index: {index}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()

        if out_dir.exists():
            print(f"Deleting incomplete prompt folder: {out_dir}")
            shutil.rmtree(out_dir, ignore_errors=True)

        rows.append({
            "index": index,
            "instance_id": instance_id,
            "golden_fix_files": None,
            "golden_fix_has_test_file": None,
            "commit_link": None,
            "issue_link": None,
            "pr_link": None,
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
        })

        continue

    finally:
        if repo_dir is not None:
            delete_repo_dir(repo_dir)



df = pd.DataFrame(rows)
output_csv = f"swe_pro_summary.csv"
df.to_csv(output_csv, index=False)



# k = math.ceil(df["src_prompt_files_size"].mean())
# overlap_positive_percentage = (
#     (df["overlap_percentage"] > 0).mean() * 100
# )

# print("recall@k =", k)
# print(overlap_positive_percentage)












