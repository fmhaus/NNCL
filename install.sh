#!/bin/bash
set -e

echo "=== Installing UV (Astral) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh

. "$HOME/.local/bin/env"

echo "=== Setting up Python project ==="
uv sync
uv pip install .[dali,umap,h5] --extra-index-url https://developer.download.nvidia.com/compute/redist
uv pip install ./solo-learn --no-build-isolation

echo "=== Done ==="
