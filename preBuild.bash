#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
if [[ -f apt.txt ]]; then
  xargs -r sudo apt-get install -y < apt.txt
fi

git lfs install || true

cat <<'EOM' >> ~/.bashrc

# NY Tech Week image fine-tuning workshop defaults
if [ -f /project/variables.env ]; then
  set -a
  source /project/variables.env
  set +a
fi

export PATH="$HOME/.local/bin:$PATH"
EOM

