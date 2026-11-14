#!/usr/bin/env bash
# Sign in to Google Colab (once per container). Copy-paste OAuth flow:
#   1) bash scripts/colab_signin.sh          -> prints a Google link
#   2) open it, approve, copy the code
#   3) bash scripts/colab_signin.sh CODE     -> finishes the sign-in
# The token is kept only in this container (~/.config/colab-cli/token.json).
# It grants broad Google access (incl. cloud-platform); revoke at myaccount.google.com/permissions.
set -euo pipefail
VENV="$HOME/.reelwright/colab-venv"
CLI="${REELWRIGHT_COLAB_CLI:-$VENV/bin/colab}"
DIR="$HOME/.reelwright/signin"
mkdir -p "$DIR"
FIFO="$DIR/code.fifo"; LOG="$DIR/signin.log"

if [ $# -ge 1 ]; then
  [ -p "$FIFO" ] || { echo "No sign-in is waiting. Run without a code first."; exit 1; }
  echo "$1" > "$FIFO"
  for _ in $(seq 1 60); do grep -q "EXIT:" "$LOG" && break; sleep 1; done
  if [ -f "$HOME/.config/colab-cli/token.json" ] || [ -f "$HOME/.local/share/python_keyring/keyring_pass.cfg" ]; then echo "Signed in."; else echo "Sign-in failed:"; tail -5 "$LOG"; exit 1; fi
  rm -f "$FIFO"
  exit 0
fi

if [ -f "$HOME/.config/colab-cli/token.json" ] || [ -f "$HOME/.local/share/python_keyring/keyring_pass.cfg" ]; then echo "Already signed in."; exit 0; fi
rm -f "$FIFO" "$LOG"; mkfifo "$FIFO"
( exec 3<>"$FIFO"; "$CLI" --auth oauth2 sessions <&3 > "$LOG" 2>&1; echo "EXIT:$?" >> "$LOG" ) &
for _ in $(seq 1 30); do grep -q "authorization code" "$LOG" 2>/dev/null && break; sleep 1; done
grep -o "https://accounts.google.com[^ ]*" "$LOG" | head -1
echo
echo "Open the link above, approve, then run:  bash scripts/colab_signin.sh <CODE>"
