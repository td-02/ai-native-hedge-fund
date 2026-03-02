#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/ai-native-hedge-fund}"
REPO_URL="${REPO_URL:-https://github.com/td-02/ai-native-hedge-fund.git}"
BRANCH="${BRANCH:-main}"

echo "[1/7] Installing Docker + Git"
sudo dnf -y update
sudo dnf -y install git docker
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "[2/7] Cloning repo"
if [[ -d "${PROJECT_DIR}/.git" ]]; then
  git -C "${PROJECT_DIR}" fetch origin
  git -C "${PROJECT_DIR}" checkout "${BRANCH}"
  git -C "${PROJECT_DIR}" pull --ff-only origin "${BRANCH}"
else
  sudo mkdir -p "${PROJECT_DIR}"
  sudo chown -R "$USER":"$USER" "${PROJECT_DIR}"
  git clone -b "${BRANCH}" "${REPO_URL}" "${PROJECT_DIR}"
fi

echo "[3/7] Preparing env/config"
cd "${PROJECT_DIR}"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from template. Fill broker/API keys before live broker execution."
fi

echo "[4/7] Installing systemd units"
sudo cp deploy/oracle/ai-hedge-fund.service /etc/systemd/system/ai-hedge-fund.service
sudo cp deploy/oracle/ai-hedge-fund-watchdog.service /etc/systemd/system/ai-hedge-fund-watchdog.service
sudo cp deploy/oracle/ai-hedge-fund-watchdog.timer /etc/systemd/system/ai-hedge-fund-watchdog.timer
sudo chmod +x deploy/oracle/check_heartbeat.sh

echo "[5/7] Reloading daemon"
sudo systemctl daemon-reload

echo "[6/7] Enabling services"
sudo systemctl enable --now ai-hedge-fund.service
sudo systemctl enable --now ai-hedge-fund-watchdog.timer

echo "[7/7] Status"
sudo systemctl --no-pager status ai-hedge-fund.service || true
sudo systemctl --no-pager status ai-hedge-fund-watchdog.timer || true
echo "Deployment complete."
