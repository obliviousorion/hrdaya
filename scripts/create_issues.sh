#!/usr/bin/env bash
# Manual only. Preview without network: scripts/create_issues.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/create_issues.py "$@"
