#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/ai-native-hedge-fund}"
HEARTBEAT_FILE="${PROJECT_DIR}/outputs/heartbeat.json"
MAX_AGE_SEC="${MAX_AGE_SEC:-900}"

if [[ ! -f "${HEARTBEAT_FILE}" ]]; then
  echo "Heartbeat missing: ${HEARTBEAT_FILE}"
  exit 2
fi

NOW="$(date +%s)"
MTIME="$(stat -c %Y "${HEARTBEAT_FILE}")"
AGE="$((NOW - MTIME))"

if (( AGE > MAX_AGE_SEC )); then
  echo "Heartbeat stale: age=${AGE}s > ${MAX_AGE_SEC}s"
  exit 3
fi

echo "Heartbeat healthy: age=${AGE}s"
