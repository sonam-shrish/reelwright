#!/usr/bin/env bash
# Install Google's Colab CLI for Reelwright (once per container).
# Network needs: colab.research.google.com and *.colab.dev allowed in the cloud environment.
set -euo pipefail
VENV="$HOME/.reelwright/colab-venv"
if [ ! -x "$VENV/bin/colab" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q google-colab-cli
fi
# Containers have no OS keyring; without this the CLI forgets the sign-in between commands.
"$VENV/bin/pip" install -q keyrings.alt
mkdir -p "$HOME/.config/colab-cli"
SETTINGS="$HOME/.config/colab-cli/settings.json"
[ -f "$SETTINGS" ] || echo '{"enable_update_check": false}' > "$SETTINGS"
echo "Colab CLI ready: $VENV/bin/colab"
if [ -f "$HOME/.config/colab-cli/token.json" ] || [ -f "$HOME/.local/share/python_keyring/keyring_pass.cfg" ]; then
  echo "Already signed in."
else
  echo "Not signed in yet: run  bash scripts/colab_signin.sh"
fi
