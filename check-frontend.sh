#!/usr/bin/env bash
# Run frontend code quality checks.
# Usage:
#   ./check-frontend.sh          # check formatting only (CI-safe, no writes)
#   ./check-frontend.sh --fix    # reformat files in place

set -euo pipefail

FRONTEND_DIR="$(dirname "$0")/frontend"

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
fi

if [[ "${1:-}" == "--fix" ]]; then
  echo "Formatting frontend files..."
  (cd "$FRONTEND_DIR" && npx prettier --write .)
  echo "Done."
else
  echo "Checking frontend formatting..."
  (cd "$FRONTEND_DIR" && npx prettier --check .)
  echo "All frontend files are correctly formatted."
fi
