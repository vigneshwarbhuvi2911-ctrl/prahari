"""Command line:

    python -m prahari build [--scale full|small]   simulate, train, evaluate and save the artifacts
    python -m prahari report                       print the evaluation summary of saved artifacts
"""
import argparse
import json

from prahari.config import Settings
from prahari.pipeline import Engine
from prahari.sim import SimConfig


def main() -> None:
    ap = argparse.ArgumentParser(prog="prahari")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--scale", choices=["full", "small"], default=None)
    sub.add_parser("report")
    args = ap.parse_args()
    s = Settings()
    if args.cmd == "build":
        scale = args.scale or s.scale
        e = Engine.build(SimConfig.small() if scale == "small" else SimConfig())
        print("saved", e.save(s.artifacts))
    else:
        m = Engine.load(s.artifacts).metrics
        print(json.dumps({k: v for k, v in m.items() if k not in ("fairness",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
