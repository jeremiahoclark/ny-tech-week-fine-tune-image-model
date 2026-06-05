#!/bin/bash
set -euo pipefail

if [[ -f variables.env ]]; then
  set -a
  source variables.env
  set +a
fi

bash scripts/setup_brev.sh

if [[ "${DOWNLOAD_MODELS_ON_BUILD:-0}" == "1" ]]; then
  bash scripts/download_models.sh
else
  echo "Skipping model download during build. Run bash scripts/download_models.sh before the live workshop or set DOWNLOAD_MODELS_ON_BUILD=1."
fi
