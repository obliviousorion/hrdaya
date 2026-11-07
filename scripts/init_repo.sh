#!/usr/bin/env bash
# Run locally when the workspace permits .git writes. Never commits or adds a remote.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e .git ]]; then
  echo 'A repository already exists; inspect it before initializing.' >&2
  exit 1
fi
git init -b main
git add --all
git status --short
