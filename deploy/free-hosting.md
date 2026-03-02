# Free Hosting Setup (No VM)

This repo includes ready configs for:
- `render.yaml` (Render Blueprint worker)
- `railway.json` + `Procfile` (Railway worker)

## Render (Fastest No-VM)
1. Push repo to GitHub.
2. In Render, create service from Blueprint (`render.yaml` auto-detected).
3. Deploy.

Worker command used:
```bash
python scripts/live.py --config configs/live_stub.yaml --poll-seconds 300 --max-cycles 0
```

## Railway
1. New project from this GitHub repo.
2. Railway uses `railway.json` start command automatically.
3. Deploy.

## Important Free-Tier Limits
- Free tiers may sleep, pause, or limit monthly runtime.
- If service sleeps, paper loop pauses until platform wakes it.
- `live_stub.yaml` is used so deployment is safe and fully free (no paid broker requirement).

## Optional: Real Broker Later
- Switch command/config to `configs/default.yaml` after adding credentials.
- Keep this on non-free/always-on infra for reliable 24/7 execution.
