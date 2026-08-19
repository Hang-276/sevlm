#!/usr/bin/env bash
# Pre-fetch VLMEvalKit Tier-1 benchmark TSVs into LMUData over PLAIN HTTP.
#
# Why http:// — on networks where openxlab's https is unreachable, plain
# http:// to the same CDN still works. VLMEvalKit's prepare_tsv skips its own
# (https) download when LMUData/{name}.tsv already exists, so pre-placing the
# files here means the framework never has to reach https at all. On a network
# where https works, you can skip this script entirely and let VLMEvalKit
# fetch the files itself.
#
# Usage: bash fetch_vlmeval_tsv.sh
set -u

LMUDATA="${LMUData:-${LMUDATA:-}}"
[ -n "$LMUDATA" ] || { echo "[ERROR] source paths.sh first, or set LMUData" >&2; exit 2; }
BASE="http://opencompass.openxlab.space/utils/VLMEval"
mkdir -p "$LMUDATA"

# Framework dataset key -> tsv file name (must match vlmeval config exactly).
FILES=(MMVP MMStar BLINK RealWorldQA AI2D_TEST ChartQA_TEST)

fetch() {
  local name="$1"
  local url="$BASE/${name}.tsv"
  local out="$LMUDATA/${name}.tsv"
  echo "[fetch] $name -> $out"
  # -C - resumes partial downloads; generous retries for the flaky CDN nodes
  # (BLINK/RealWorldQA in particular).
  curl -fsSL --http1.1 -C - --retry 8 --retry-delay 5 --connect-timeout 20 \
       --max-time 1800 -o "$out" "$url"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  [WARN] $name download failed (curl rc=$rc)"
    return 1
  fi
  # Validate: non-empty and NOT an HTML error page (squid returns <!DOCTYPE).
  if [ ! -s "$out" ]; then
    echo "  [WARN] $name is empty"; return 1
  fi
  local head1
  head1="$(head -c 20 "$out" | tr -d '\0')"
  case "$head1" in
    "<!DOCTYPE"*|"<html"*|"<HTML"*)
      echo "  [WARN] $name looks like an HTML error page, not a tsv"; return 1;;
  esac
  local sz
  sz="$(du -h "$out" | cut -f1)"
  local rows
  rows="$(wc -l < "$out")"
  echo "  [ok] $name size=$sz rows=$rows"
  return 0
}

declare -a OK=() FAIL=()
for f in "${FILES[@]}"; do
  if fetch "$f"; then OK+=("$f"); else FAIL+=("$f"); fi
done

echo ""
echo "==== fetch summary ===="
echo "  OK   (${#OK[@]}): ${OK[*]:-none}"
echo "  FAIL (${#FAIL[@]}): ${FAIL[*]:-none}"
echo "  LMUData=$LMUDATA"
[ ${#FAIL[@]} -eq 0 ]
