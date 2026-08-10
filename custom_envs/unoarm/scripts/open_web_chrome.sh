#!/usr/bin/env bash
# Open Unoarm Web in Windows Chrome with WebGL forced on (fixes WSL / sandboxed preview).
set -euo pipefail

PORT="${1:-7860}"
URL="http://127.0.0.1:${PORT}"

CHROME=""
for candidate in \
  "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \
  "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
  "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe"
do
  if [[ -x "$candidate" ]]; then
    CHROME="$candidate"
    break
  fi
done

if [[ -z "$CHROME" ]]; then
  echo "Windows Chrome/Edge not found under /mnt/c/Program Files." >&2
  echo "Open ${URL} manually in Windows Chrome (not Cursor Simple Browser)." >&2
  exit 1
fi

# Ignore GPU blocklist + allow WebGL. disable-gpu-sandbox fixes BindToCurrentSequence
# failures seen when Chromium reports GL_VENDOR=Disabled / Sandboxed=yes.
USER_DATA="${TMPDIR:-/tmp}/unoarm-chrome-webgl"
mkdir -p "$USER_DATA"

echo "Opening ${URL} with: $CHROME"
"$CHROME" \
  --user-data-dir="$USER_DATA" \
  --ignore-gpu-blocklist \
  --enable-webgl \
  --enable-accelerated-2d-canvas \
  --disable-gpu-sandbox \
  --use-gl=angle \
  "$URL" >/dev/null 2>&1 &
disown || true
echo "If the 3D view is still blank, close other Chrome windows using a broken GPU profile and retry."
