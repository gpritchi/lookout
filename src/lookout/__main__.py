"""Command line: `python -m lookout <command>`.

  check   validate a config file and print what it defines
  replay  run the engine offline on a JSONL event fixture with a scripted model
"""

from __future__ import annotations

import argparse
import logging
import sys

from lookout.config import ConfigError, load_config
from lookout.events import ReplaySource
from lookout.runtime import ScriptedModel, run_replay


def cmd_check(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.config}: ok")
    for name, spec in config.models.items():
        print(f"  model   {name:<16} {spec.model} @ {spec.endpoint}  accepts {', '.join(spec.capabilities)}")
    for name, spec in config.cameras.items():
        print(f"  camera  {name:<16} {spec.source}  buffer {config.buffer_seconds(name):.0f}s")
    for chain in config.chains:
        if chain.steps:
            steps = " -> ".join(f"{s.id}({s.payload},p{s.priority})" for s in chain.steps)
        else:
            steps = f"one-shot {chain.on_trigger.action.type}"  # type: ignore[union-attr]
        print(f"  chain   {chain.id:<16} {chain.camera}:{chain.trigger.label}  {steps}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    source = ReplaySource(args.events, fps=args.fps)
    model = ScriptedModel.from_file(args.answers)
    report = run_replay(config, source, model)
    print()
    print(f"stream ended at {report.stream_end_ts:.2f}s, settled at {report.final_ts:.2f}s, "
          f"{report.inference_calls} inference call(s)")
    print(f"{len(report.actions)} action(s) fired:")
    for line in report.actions:
        print(f"  {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lookout")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="validate a config file")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("replay", help="run the engine offline on a fixture")
    p.add_argument("--config", required=True)
    p.add_argument("--events", required=True, help="JSONL of detection events")
    p.add_argument("--answers", required=True, help="JSON script of model answers per step id")
    p.add_argument("--fps", type=float, default=2.0, help="synthetic frame rate for window sampling")
    p.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
