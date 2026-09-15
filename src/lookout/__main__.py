"""Command line: `python -m lookout <command>`.

  check   validate a config file and print what it defines
  replay  run the engine offline on a JSONL event fixture with a scripted model
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

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
        print(f"  chain   {chain.id:<16} {chain.camera}:{'|'.join(chain.trigger.labels)}  {steps}")
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


def cmd_tier1(args: argparse.Namespace) -> int:
    """Run only the detector over one camera's source and print the events it
    would raise, with detector latency. The verification tool for tier-1."""
    from prometheus_client import REGISTRY

    from lookout.events import DetectionEvent
    from lookout.tier1 import VideoSource, YoloDetector

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    camera = config.cameras[args.camera]
    uri = args.source or camera.source
    detector = YoloDetector(config.tier1, models_dir=args.models_dir)
    source = VideoSource(args.camera, uri, detector, config.tier1, loop=False)
    t0 = time.perf_counter()
    events = 0
    for item in source.stream():
        if isinstance(item, DetectionEvent):
            events += 1
            print(f"[{item.ts:7.2f}] {item.camera}: {item.label} ({item.confidence:.2f}) bbox={item.bbox}", flush=True)
        if args.max_seconds and time.perf_counter() - t0 >= args.max_seconds:
            break  # a live source never ends on its own
    wall = time.perf_counter() - t0
    labels = {"model": detector.name, "tier": "tier1"}
    count = REGISTRY.get_sample_value("lookout_inference_seconds_count", labels) or 0
    total = REGISTRY.get_sample_value("lookout_inference_seconds_sum", labels) or 0.0
    print()
    print(f"{source.frames_read} frames read, {source.frames_analysed} analysed, {events} event(s), "
          f"{wall:.1f}s wall")
    if count:
        print(f"detector {detector.name}: {int(count)} calls, mean {1000 * total / count:.1f} ms")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """The real thing: every camera in the config streams through the detector
    into the engine, escalations go to the models in the registry, actions go
    to the configured sink. Runs until Ctrl-C, --max-seconds, or
    --stop-after-actions."""
    from lookout.actions import ActionSink, HAWebhookSink, MockSink
    from lookout.runtime import run_live
    from lookout.tier1 import VideoSource, YoloDetector
    from lookout.vlm import OpenAICompatibleClient

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    detector = YoloDetector(config.tier1, models_dir=args.models_dir)
    sources = {}
    for name, camera in config.cameras.items():
        if args.camera and name not in args.camera:
            continue
        if camera.tier1 != "yolo":
            print(f"skipping camera {name}: tier1 source '{camera.tier1}' is not implemented", file=sys.stderr)
            continue
        sources[name] = VideoSource(name, camera.source, detector, config.tier1, loop=args.loop)
    if not sources:
        print("no cameras to run", file=sys.stderr)
        return 1
    if config.metrics.port:
        from lookout.metrics import serve

        serve(config.metrics.port, config.metrics.host)
    sink: ActionSink
    if config.actions.sink == "ha_webhook":
        sink = HAWebhookSink(config.actions.ha_webhook_url or "", timeout_s=config.actions.ha_timeout_s)
    else:
        sink = MockSink()
    client = OpenAICompatibleClient(config.models)
    try:
        report = run_live(
            config, sources, client, sink=sink,
            max_seconds=args.max_seconds, stop_after_actions=args.stop_after_actions,
            trace=lambda line: print(line, flush=True),
        )
    finally:
        client.close()
        if isinstance(sink, HAWebhookSink):
            sink.close()
            if sink.failed:
                print(f"{sink.failed} webhook call(s) failed; see the log", file=sys.stderr)
    print()
    print(f"ran {report.final_ts:.0f}s, {report.inference_calls} inference call(s), {len(report.actions)} action(s):")
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

    p = sub.add_parser("tier1", help="run only the detector over a camera and print its events")
    p.add_argument("--config", required=True)
    p.add_argument("--camera", required=True, help="camera name from the config")
    p.add_argument("--source", help="override the camera's source URI (file, RTSP, or device index)")
    p.add_argument("--models-dir", default=".", help="where <model>.pt and the OpenVINO export live")
    p.add_argument("--max-seconds", type=float, default=0, help="stop after this many seconds (live sources never end)")
    p.set_defaults(func=cmd_tier1)

    p = sub.add_parser("run", help="watch every camera live, escalate to the models, fire actions")
    p.add_argument("--config", required=True)
    p.add_argument("--camera", action="append", help="only these cameras (repeatable); default all")
    p.add_argument("--models-dir", default=".", help="where <model>.pt and the OpenVINO export live")
    p.add_argument("--loop", action="store_true", help="loop file sources instead of stopping at the end")
    p.add_argument("--max-seconds", type=float, default=0, help="stop after this many seconds")
    p.add_argument("--stop-after-actions", type=int, default=0, help="stop once this many actions have fired")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
