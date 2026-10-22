"""Exercise the shipped image, exported detector and both CA stores offline."""

from importlib import metadata, util
from pathlib import Path
import ssl
import sys

import certifi
import numpy as np

from lookout.config import Tier1Spec, load_config
from lookout.events import ReplaySource
from lookout.runtime import ScriptedModel, run_replay
from lookout.tier1 import OpenVinoDetector


expected_python = tuple(map(int, sys.argv[1].split(".")))
assert sys.version_info[:len(expected_python)] == expected_python, sys.version
print(f"Python {sys.version.split()[0]}")
assert tuple(map(int, metadata.version("certifi").split("."))) >= (2026, 10, 19)
# Build-only dependencies must not leak into the final image.
for package in ("torch", "torchvision", "ultralytics", "PIL", "urllib3"):
    assert util.find_spec(package) is None, f"Unexpected runtime package: {package}"

blocked_roots = {"emSign ECC Root CA - G3", "AAA Certificate Services"}
for bundle in (ssl.get_default_verify_paths().cafile, certifi.where()):
    assert bundle and Path(bundle).is_file(), f"Missing CA bundle: {bundle}"
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=bundle)
    roots = context.get_ca_certs()
    assert roots, f"Empty CA bundle: {bundle}"
    names = {value for root in roots for rdn in root["subject"]
             for key, value in rdn if key == "commonName"}
    assert not names & blocked_roots, f"Compromised trust path in {bundle}: {names & blocked_roots}"
    print(f"CA bundle {bundle}: {len(roots)} roots, compromised trust paths absent")

config = load_config("config/chains.example.json")
report = run_replay(
    config, ReplaySource("fixtures/sample/driveway-replay.jsonl"),
    ScriptedModel.from_file("fixtures/sample/fake-vlm-answers.json"),
)
assert report.inference_calls == 7, report
assert report.actions == [
    "ha_webhook(service=tv_netflix_on) <- tv-netflix on living-room",
    "notify(message=Package delivered) <- driveway-arrivals on driveway",
    "notify(message=Sister-in-law and her husband are at the door) <- driveway-arrivals on driveway",
], report.actions
print("Offline replay: 7 inference calls, 3 expected mock actions")

detector = OpenVinoDetector(Tier1Spec(device="CPU"), "/app/models/yolov8n_openvino_model")
detections = detector(np.zeros((640, 640, 3), dtype=np.uint8))
assert isinstance(detections, list), detections
print(f"Shipped OpenVINO detector: CPU inference passed ({len(detections)} detections)")
