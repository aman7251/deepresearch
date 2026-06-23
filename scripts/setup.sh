#!/usr/bin/env bash
# One-shot setup on macOS/Linux.
# Creates a venv, installs all deps, and creates .env from the template.
#
# Usage:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Creating virtual environment (.venv) ..."
python3 -m venv .venv

echo "Installing dependencies (this can take a few minutes on first run) ..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-dev.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from template - edit it to set DEMO_MODE / GROQ_API_KEY."
fi

echo
echo "Setup complete."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  pytest                                # offline tests, no keys"
echo "  streamlit run app/ui_streamlit.py     # UI (set DEMO_MODE=1 in .env first)"
