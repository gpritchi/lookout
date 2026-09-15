# Fixtures

- `private/` — **gitignored.** Real camera footage. Never committed. The
  development clips were two exports from a driveway camera: a delivery van
  arriving and a pickup with two visitors. Drop your own here and point
  `fixtures/sample/mediamtx.yml` at them to get a looping fake camera.
- `sample/` — committed, and contains nothing personal:
  - `driveway-replay.jsonl` — canned tier-1 events for `python -m lookout replay`.
  - `fake-vlm-answers.json` — the scripted model's answers per step for that replay.
  - `mediamtx.yml` — a MediaMTX config that loops a clip as an RTSP camera and
    records it, so lookout can be run against a "live" stream without a camera,
    and `video_clip` steps have a recorder to fetch from.

There is no public sample video in this repo. The replay fixture exercises the
whole engine without one; the live path needs a clip you supply.
