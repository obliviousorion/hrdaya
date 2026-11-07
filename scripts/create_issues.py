"""Create task issues only when explicitly run; exact-title deduplication supports reruns."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def tasks(text):
    milestone = ""
    for block in re.split(r"(?=^## )", text, flags=re.M):
        if not block.startswith("## "):
            continue
        milestone = block.splitlines()[0][3:]
        for task in re.split(r"^### ", block, flags=re.M)[1:]:
            title, body = task.split("\n", 1)
            yield title, f"Milestone: {milestone}\n\n{body.strip()}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="Explicit OWNER/REPO required for writes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.repo:
        parser.error("supply --repo OWNER/REPO or --dry-run")
    existing = set()
    if not args.dry_run:
        existing = {i["title"] for i in json.loads(subprocess.check_output(
            ["gh", "issue", "list", "--repo", args.repo, "--state", "all",
             "--limit", "10000", "--json", "title"], text=True))}
    for title, body in tasks(Path("TASKS.md").read_text()):
        if args.dry_run:
            print(f"{title}\n{body}")
        elif title not in existing:
            subprocess.run(["gh", "issue", "create", "--repo", args.repo,
                            "--title", title, "--body", body], check=True)
