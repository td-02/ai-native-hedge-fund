# Oracle Always Free Deployment (24/7 Paper Trading)

This setup runs the engine as a Docker Compose service managed by `systemd`, with a watchdog timer that restarts the service if heartbeat becomes stale.

## 1) Create VM
- Provision Oracle Always Free Linux VM (Oracle Linux 8/9 recommended).
- Open outbound internet and SSH.
- For Streamlit UI (optional), open inbound `8501/tcp`.

## 2) SSH and Run Installer
```bash
ssh opc@<YOUR_VM_PUBLIC_IP>
git clone https://github.com/td-02/ai-native-hedge-fund.git
cd ai-native-hedge-fund
chmod +x deploy/oracle/install.sh
./deploy/oracle/install.sh
```

## 3) Configure Environment
```bash
cd /opt/ai-native-hedge-fund
nano .env
```
- Fill broker/API credentials.
- Keep `execution.primary_broker: stub` for safe paper simulation.

## 4) Service Commands
```bash
sudo systemctl status ai-hedge-fund.service
sudo systemctl restart ai-hedge-fund.service
sudo journalctl -u ai-hedge-fund.service -f
sudo systemctl status ai-hedge-fund-watchdog.timer
```

## 5) Update to Latest Code
```bash
cd /opt/ai-native-hedge-fund
git pull --ff-only origin main
sudo systemctl reload ai-hedge-fund.service
```

## 6) Optional Streamlit UI
```bash
cd /opt/ai-native-hedge-fund
docker compose -f docker-compose.oracle.yml run --rm --service-ports hedge-fund streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## Notes
- `ai-hedge-fund.service` runs container in detached mode.
- Watchdog checks `outputs/heartbeat.json` every 5 minutes.
- If heartbeat is stale, watchdog restarts the live trading service automatically.
