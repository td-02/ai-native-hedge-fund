from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from free_fund.mcp_server import build_research_mcp_server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "sse", "stdio"],
        default="streamable-http",
    )
    args = parser.parse_args()

    server = build_research_mcp_server(
        config_path=args.config,
        host=args.host,
        port=args.port,
    )
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
