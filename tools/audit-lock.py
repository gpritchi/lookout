"""Check every uv.lock registry pin against OSV, without installing the torch extra.

The lock spans platforms and build-only dependencies. A match is a reason to
update/review a package, not evidence that a production code path is exploitable.
Network failures fail the check rather than silently reporting a clean audit.
"""

import json
from pathlib import Path
import sys
import tomllib
import urllib.request


def main() -> int:
    lock = tomllib.loads(Path("uv.lock").read_text())
    packages = sorted({
        # CPU wheel local versions share the upstream package's security history.
        (p["name"], p["version"].split("+")[0])
        for p in lock["package"] if "registry" in p.get("source", {})
    })
    if not packages:
        raise RuntimeError("No registry packages found in uv.lock")
    request = urllib.request.Request(
        "https://api.osv.dev/v1/querybatch",
        data=json.dumps({"queries": [
            {"package": {"ecosystem": "PyPI", "name": name}, "version": version}
            for name, version in packages
        ]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        results = json.load(response)["results"]
    if len(results) != len(packages):
        raise RuntimeError("Incomplete OSV audit response")
    affected = 0
    for (name, version), result in zip(packages, results, strict=True):
        advisories = result.get("vulns", [])
        if advisories:
            affected += 1
            print(f"{name}=={version}: " + ", ".join(v["id"] for v in advisories))
    print(f"Audited {len(packages)} distinct PyPI pins; {affected} affected packages")
    return int(affected > 0)


if __name__ == "__main__":
    sys.exit(main())
