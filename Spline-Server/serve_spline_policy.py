from __future__ import annotations

import argparse

from spline_server.config import load_config
from spline_server.runtime import SplineRuntime
from spline_server.server_policy import SplineRuntimePolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the LeHome spline runtime over websocket.")
    parser.add_argument("--config", required=True, help="Spline server YAML/JSON config.")
    parser.add_argument("--port", type=int, default=None, help="Optional websocket port override.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values with KEY=VALUE.")
    args = parser.parse_args()

    overrides = list(args.overrides)
    if args.port is not None:
        overrides.append(f"server.port={int(args.port)}")

    config = load_config(args.config, overrides)
    runtime = SplineRuntime(config)

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    server = WebsocketPolicyServer(
        policy=SplineRuntimePolicy(runtime),
        host=str(config["server"]["host"]),
        port=int(config["server"]["port"]),
        metadata=runtime.server_metadata(),
    )
    print(
        f"[serve_spline_policy] host={config['server']['host']} "
        f"port={config['server']['port']} device={runtime.server_metadata()['device']} "
        f"end_mode={runtime.server_metadata()['end_mode']}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
