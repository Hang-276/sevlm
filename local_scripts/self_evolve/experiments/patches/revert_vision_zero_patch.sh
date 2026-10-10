#!/usr/bin/env bash
# Revert the Vision-Zero paper-alignment patch, restoring the official checkout
# to the pinned upstream commit. The launcher re-applies the patch on its next
# run, so this is safe to run between experiments.
#
# Refuses to touch a checkout whose local changes are not exactly the recorded
# patch -- the same guarantee the launcher enforces.
set -euo pipefail

: "${VISION_ZERO_REPO:?Set VISION_ZERO_REPO to the official vision-zero checkout}"
PATCH_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vision_zero_paper_alignment.patch"
[ -f "$PATCH_FILE" ] || { echo "[ERROR] Missing patch file: $PATCH_FILE" >&2; exit 2; }

if git -C "$VISION_ZERO_REPO" diff --quiet HEAD -- src/open-r1-multimodal; then
  echo "[revert] checkout is already clean"
  exit 0
fi

APPLIED_SHA="$(git -C "$VISION_ZERO_REPO" diff -- src/open-r1-multimodal | sha256sum | cut -d' ' -f1)"
PATCH_SHA="$(sha256sum "$PATCH_FILE" | cut -d' ' -f1)"
if [ "$APPLIED_SHA" != "$PATCH_SHA" ]; then
  echo "[ERROR] Checkout has local changes that are not the recorded patch" >&2
  echo "        diff=$APPLIED_SHA  patch=$PATCH_SHA" >&2
  echo "        Inspect it by hand; nothing was reverted." >&2
  exit 2
fi

git -C "$VISION_ZERO_REPO" checkout -- src/open-r1-multimodal
echo "[revert] restored $VISION_ZERO_REPO to $(git -C "$VISION_ZERO_REPO" rev-parse --short HEAD)"
