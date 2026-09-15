# Human README

This is basically an AI-powered IFTTT (If This Then That) I wanted to write to exist between [Frigate](https://github.com/blakeblackshear/frigate) and [Home Assistant](https://github.com/home-assistant), to bridge more advanced chains/checks using increasingly powerful self-hosted AI models as needed. No testing has been done with cloud models.

It's possible some of this functionality might already be natively included in those open source projects, I just wanted to build on it and use some spare computer hardware I have at home.

This is a work in progress. The Frigate side is a stub, the Home Assistant webhook works but I have not wired an automation to it yet. I was testing with 2 clips from my own camera footage (one of a family member arriving in their Tacoma, another of the amazon delivery guy dropping something off), looping and pretending to be a live RTSP stream like a Eufy/Wyze/Reolink Cam. Integration functionality is not guaranteed.

Use cases I'm working on:
- Hand gestures to control smart home stuff, like 2 fingers to turn on TV and open Netflix, 3 fingers for Prime Video, "ok" hand gesture to turn it all off (basic CPU or iGPU model can handle this)
- Using the specific make/model of vehicle to determine who has arrived. Facial detection at a distance with mediocre quality video can't tell my wife apart from her sister, but she has a black Tacoma and we don't. Escalating to a smarter model to determine the exact vehicle allows me to be notified her sister is at the door, without it erroneously triggering every time my wife walks up.
- Further escalation can be made to a model that can actually "watch videos", sound included, which is cheaper per second of footage and a lot simpler than frame by frame "seeing" plus some separate audio "hearing" model. For this I tested with Nemotron-3-Nano-Omni-30B-A3B-Reasoning. This part gets _much_ slower on my hardware. I use it for checking someone delivered a package versus stealing one, and the plan is for it to hear the doorbell too, which isn't working yet (see Limitations).

The AI written README below gets into much more detail on how this works.

# lookout

Tiered AI escalation for home camera feeds. A cheap local detector watches every
frame, a rules file says which detections deserve a closer look and what
question to ask, a priority scheduler decides who gets the scarce VLM time
first, and answers turn into Home Assistant actions.

```
camera ──► YOLO (CPU, ~20 ms/frame) ──► "a car arrived" ──► chain
                                                              │
        ┌─────────────────────────────────────────────────────┘
        ▼
  step 1  image          → small VLM   "grey van, black Tacoma, or other?"
  step 2  image_sequence → small VLM   "did someone in a vest take a package toward the house?"
  step 3  video_clip     → omni model  "did they set it down, or pick one up?"
        │
        ▼
  action: notify / HA webhook
```

Each tier only runs when the tier below genuinely can't answer. A COCO detector
can say `car`. It cannot say `black Tacoma`, cannot say which door the
passenger used, and cannot tell a delivery from a porch theft. Those are
prompts, not models: adding a condition is a config edit.

## Status

Working end to end against a live RTSP camera, with real models behind a
LiteLLM proxy. See *Limitations* for the honest list of rough edges.

## Run it

Everything below needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
```

### 1. Offline, no camera, no GPU, no network

The engine, scheduler, and chains on a virtual clock with canned events and a
scripted model. This is the fastest way to see the whole state machine work.

```
uv run python -m lookout replay \
  --config config/chains.example.json \
  --events fixtures/sample/driveway-replay.jsonl \
  --answers fixtures/sample/fake-vlm-answers.json
```

Seventy seconds of "footage" replays in milliseconds. The trace shows every
trigger, dispatch, answer, retry, and action.

The same thing in Docker:

```
docker build -t lookout .
docker run --rm lookout
```

### 2. A real camera, or a fake one

lookout reads anything OpenCV can open: a file path, an RTSP URL, a device
index. To develop without a camera, loop a clip through
[MediaMTX](https://github.com/bluenviron/mediamtx) with the config in
`fixtures/sample/mediamtx.yml`. That also gives you the recorder the video
steps fetch clips from.

Watch the detector alone, no models involved:

```
uv run python -m lookout tier1 --config config/chains.example.json \
  --camera driveway --source rtsp://<host>:8554/driveway --max-seconds 120
```

### 3. The whole thing

Point the models registry at an OpenAI-compatible endpoint that serves the
capabilities your chains use. The demo uses a LiteLLM proxy in front of two
llama.cpp servers; which cards, quants, and hosts sit behind it is the proxy's
business, not this repo's.

```
cp config/chains.example.json config/chains.local.json   # gitignored; edit endpoints and cameras
uv run python -m lookout check --config config/chains.local.json
uv run python -m lookout run --config config/chains.local.json --stop-after-actions 1
```

## Configuration

One JSON file. `python -m lookout check` validates it and prints what it
defines. Every cross-reference is checked at load time, so a bad model swap is
a startup error with a message naming the chain, step, and value, never a
runtime surprise.

| Section | What it holds |
|---|---|
| `models` | name → endpoint, model id, `capabilities` (`image`, `image_sequence`, `video_clip`), optional `api_key_env` |
| `cameras` | name → `source` URI, optional `clips` (a MediaMTX playback endpoint and path, required by any `video_clip` step) |
| `tier1` | detector model and backend, confidence thresholds, debounce, analysis and buffer frame rates |
| `actions` | `sink`: `mock` (logs what it would do) or `ha_webhook` with a URL |
| `chains` | trigger, steps, windows, outcomes; or `on_trigger` for a chain with no steps |

### A chain

```json
{
  "id": "driveway-arrivals",
  "camera": "driveway",
  "trigger": { "label": ["car", "truck"], "priority": 10 },
  "window":  { "before_s": 2, "after_s": 8, "frames": 6 },
  "max_age_s": 300,
  "steps": [ ... ]
}
```

- **trigger** matches a detector label (or any of several: COCO flips a pickup
  between `car` and `truck`). `priority` orders competing chain starts.
- **window** is how much stream a step looks at, relative to the trigger:
  `before_s` back, `after_s` forward, sampled into `frames`. A step cannot
  dispatch until `after_s` has elapsed, because that footage doesn't exist
  yet. An elderly visitor who takes thirty seconds to get out is
  `"after_s": 30`.
- **max_age_s** bounds a run. A step that keeps answering "not yet" would
  otherwise hold the chain forever and block the next real arrival.

### A step

```json
{
  "id": "check-driver",
  "model": "qwen3-vl",
  "payload": "image_sequence",
  "priority": 60,
  "timeout_s": 60,
  "prompt": "These frames follow a grey van parking. Does a person in a reflective vest take a package from the van and head toward the house? Answer exactly one of: DELIVERY_IN_PROGRESS, PERSON_NO_PACKAGE, NOBODY_YET.",
  "outcomes": {
    "DELIVERY_IN_PROGRESS": { "next": "confirm-delivery" },
    "PERSON_NO_PACKAGE":    { "action": { "type": "notify", "message": "Unknown grey van in driveway" } },
    "NOBODY_YET":           { "retry_in_s": 5 }
  }
}
```

- **payload** must be in the model's declared `capabilities`, or the config
  fails to load. That's the gate that makes swapping models safe.
- **priority** is the queue priority of the inference job. Higher wins. A
  gesture chain at 100 jumps ahead of a driveway classification at 50; nothing
  is preempted mid-call.
- **outcomes** map the model's answer to exactly one of: `next` step, `end`,
  fire an `action`, or `retry_in_s` seconds later on a fresh window. Answers
  are matched exactly, then case-insensitively, then as a whole word inside
  prose, since models like to explain themselves.

### A chain with no steps

```json
{
  "id": "tv-off",
  "camera": "living-room",
  "trigger": { "label": "gesture:ok_sign", "held_frames": 8, "priority": 100 },
  "steps": [],
  "on_trigger": { "action": { "type": "ha_webhook", "service": "tv_off" } }
}
```

Finger counting is cheap-model territory (MediaPipe on a CPU), so these chains
escalate to nothing: the action fires at tier one, with `held_frames` as the
intent filter. The tier boundary is empirical and movable in both directions.
(The gesture source itself is not built yet; the chains show the schema.)

## How it works

- **`tier1.py`** — YOLOv8n via OpenVINO (exported on first run; PyTorch as the
  fallback backend) over an OpenCV capture. Detections become events through
  count-rise hysteresis per label: fire when a label's count rises above its
  baseline and stays there, let the baseline follow the count back down. A
  parked car is scenery, not an arrival. Objects are counted down to a low
  confidence so they don't flicker, but only a confident newcomer fires.
- **`frames.py`** — a per-camera ring buffer of JPEG frames, sized from the
  config, and the window sampler.
- **`engine.py`** — the chain state machine. One active run per chain. Time
  is injected, so the same engine runs on a virtual clock for replay and the
  wall clock live.
- **`scheduler.py`** — one priority heap per model, a worker that sleeps on a
  condition and drains the queues for the models it serves. A second host
  serving one model is a second worker with a shorter list.
- **`vlm.py`** — the OpenAI-compatible client. Sequences interleave a text
  part between images because llama.cpp merges consecutive images into one
  video input. Clips go as llama.cpp's `input_video` part.
- **`clips.py`** — fetches a time range from a MediaMTX recorder as MP4,
  audio track included. Whether the server actually uses that track is the
  server's business; see *Limitations* for the one I tested against.
- **`metrics.py`** — every inference call, detector frames included, is timed
  into a Prometheus histogram labelled by model and tier, served at
  `/metrics` when `metrics.port` is set.
- **`actions.py`** — the mock sink logs what it would do; the Home Assistant
  sink POSTs each action as flat JSON to a webhook trigger, so an automation
  on the HA side decides what "notify" means in that house.

## Why not a cheap model all the way down

Three honest points.

1. The demo conditions need reasoning a detector doesn't have: make and colour
   of a vehicle, which door someone used, whether they carried a package
   toward the house, whether a package was set down or picked up. Those are
   relational, fine-grained, or temporal questions.
2. Where a specialised small model could encroach, gesture recognition being
   the honest example, the VLM buys open-vocabulary configuration. A new
   condition is a prompt edit, not a data collection and retraining cycle.
3. The tier boundary is a config decision, not an architectural one. When a
   cheap model proves sufficient for a condition, promoting it down to tier
   one is exactly what the steps-less chains do. The design treats that as
   the expected outcome, not a refutation.

## Limitations

- No object tracking. A vehicle that repositions in frame re-triggers; the
  VLM's "other, end" outcome absorbs the cost. Zone masks would be the proper
  fix for edge-of-frame noise and are not built.
- A looping fake camera has a seam where the scene cuts; the debouncer sees
  an arrival there. Real cameras don't do that.
- Capture-to-event lag is about two seconds at 1080p on a laptop CPU.
- Clip steps fetch the MP4 on the engine's tick thread. A few MB over a LAN;
  fine at this scale, not at fifty cameras.
- `video_clip` steps rely on a server that accepts llama.cpp's `input_video`
  part. Upstream llama.cpp feeds frames only; true audio and temporal video
  for Nemotron Omni need the `llama.cpp-omni` fork. A proxy that validates
  against the OpenAI schema (LiteLLM does) rejects that part, so the clip
  model's `endpoint` points at the server directly. Per-model endpoints exist
  for exactly this.
- The audio never made it. The fork's build I ran logged "failed to decode
  the video's audio track — audio track skipped, frames only" for every
  recorder clip, so every live Nemotron answer in this write-up was reached
  from frames alone. The cause is in the fork, not here: its audio pass
  demuxes a buffer over ffmpeg's `cache:pipe:0` with the video discarded,
  and the mov demuxer's seeks yield no samples for a well-formed MP4. A
  patched build that spools the buffer to a temp file exists in my cluster
  repo and had not been rolled out when this was written. The engine's
  "clip with audio" step is real; the model that heard it is not yet.
- Clips are slow. Thirty seconds of video at the fork's sampling is about
  5,000 prompt tokens; on a 32 GB MI50 over Vulkan that's two and a half
  minutes. The video step is the slow tier by design: low priority, long
  timeout, dispatched only after the cheap steps have earned it.
- The gesture tier-one source is not implemented; those chains are config only.
- Single worker, single process. Multi-host is a design property, not a
  tested feature.
