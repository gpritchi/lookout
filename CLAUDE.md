# lookout — agent working conventions

Tiered camera-event escalation engine: a cheap local detector watches video feeds,
a rules file defines condition chains, and a priority scheduler decides which frames
earn time on scarce VLM inference capacity. Actions fire into Home Assistant.

This is a timeboxed build (4–8 hours total) and a public repo. Read this whole file
before writing code. Treat it as intent, not law: when a rule here gets in the way of
something simpler, say so and ask.

## Method

- Requirements first: before implementing a component, restate what it must do and
  what is explicitly out of scope for v1. Ask George when a scope call is ambiguous.
- Verify before claiming done: every component gets run for real (or exercised by a
  test) before it is reported as working. Never report untested code as complete.
- Review your own diff against the patterns already in the repo before presenting it.
- Commit after each working milestone, descriptive messages, no giant end-of-day dump.
- Be able to explain any line on request. If you can't explain it, rewrite it.

## AI-usage journal — required, after every milestone

Append to `notes/ai-journal.md` (gitignored) after each milestone or notable moment:
what was built, what the agent got right, **where it went wrong and how that was
caught**, and anything that surprised. Entries are dated, terse, honest. This journal
is the raw material for the submission write-up ("how you used AI and what worked or
didn't", "where the AI got things wrong and how you dealt with it") — without it the
write-up becomes fiction. Do not skip it because the build is going well; "nothing
went wrong this hour" is itself an entry.

The journal and everything in `notes/` stays out of the repo. Do not weaken the
.gitignore.

## v1 scope — settled decisions, do not relitigate

- **Tier-1 is in-repo**: Ultralytics YOLO exported to OpenVINO, OpenCV VideoCapture
  reading a file/RTSP URI, simple debounce (same label, N consecutive frames).
  **Hard cap: 1 hour.** Fallback ladder if OpenVINO fights: plain CPU YOLO → the
  ReplaySource fixtures. No tracking, no multi-model zoo, no device selection UI.
- **Sources are swappable**: everything enters the engine as a `DetectionEvent` via
  the `Source` protocol (`src/lookout/events.py`). `OpenVinoVideoSource` now,
  `ReplaySource` for tests, `FrigateMqttSource` is a documented stub for later.
- **Config**: one JSON file (see `config/chains.example.json`) with a **models
  registry** (name → endpoint + declared input capabilities) and **chains** (trigger,
  steps, each step referencing a model by name, a payload type, a priority, a
  timeout_s). **Capability gating is load-time validation**: a step whose payload type
  is not in its model's capabilities fails config load with a clear error.
- **Scheduler**: one queue per model, priority-ordered (higher number wins), a single
  worker per model draining that model's queue (a slow clip on one card must not
  starve one-second classifications on the other; found live, not designed). No
  mid-inference preemption in v1 — priority is ordering and queue-jumping only.
  Chain state resets when a step's timeout_s expires.
- **Tier-2 endpoint**: any OpenAI-compatible chat endpoint. The demo uses a LiteLLM
  proxy in front of llama.cpp; the endpoint, model alias, and quant behind it are the
  proxy's business, never this repo's. Endpoint and model name live in config, never
  hardcoded. `image_sequence` is sent as multiple image parts with a short text part
  between each (llama.cpp merges strictly consecutive images as video frames).
  `video_clip` payloads are MP4s fetched from the recorder (MediaMTX playback API)
  for the step's window, audio track included, and sent as llama.cpp's `input_video`
  part. Only a model that genuinely understands sequence and sound earns that
  step: put-down vs pick-up, how many got out, what was said at the door.
- **Actions**: mock sink (prints/logs what it would have done) is the default; a real
  HA webhook URL is an optional config field. Demo runs on the mock sink.
- **Metrics**: wrap every inference call in a timing context manager from the start;
  `prometheus_client` Histogram labeled {model, tier}, `/metrics` endpoint. Dashboards,
  per-host labels, multi-worker are future work — but keep the model-keyed queue design
  so adding a second worker host later is config, not redesign.
- **Tests**: basics only, pytest. Config validation (valid loads; video payload on an
  image-only model rejected; unknown model rejected), scheduler ordering, debounce.
  See `tests/`. No mocking frameworks beyond what pytest gives you.

## Workflow

- One pull request per phase below, each self-contained and mergeable on its own,
  with a body that says what's in it, which choices were deliberate, and what to
  expect after merge. George reviews inline; nothing is pushed as final without him
  seeing it. Stacked branches are fine when a phase builds on an unmerged one.
- The app builds and runs as a Dockerfile (multi-stage `uv` build), so the same image
  runs on a laptop today and on a cluster node later.
- Notes: `notes/` and `unsanitized-notes/` are gitignored (AI journal, LAN specifics).
  If a note is worth publishing, it goes in `sanitized-notes/`, which is committed.

## Phases (v1)

Engine first, because it is the thesis and runs fully offline. Tier-1 and the VLM step
wait on a clip and a proxy respectively, so they come after.

| PR | Goal |
|---|---|
| 1 | Scaffold: conventions, example config, module contracts |
| 2 | Config loading and validation (the four rules, steps-less chains with `on_trigger`, tests) |
| 3 | Priority scheduler and worker (per-model queues, worker over a model subset, timing wrapper, tests) |
| 4 | Chain engine, ReplaySource, mock actions, Dockerfile; first offline end-to-end run |
| 5 | Tier-1 video source: YOLO export, OpenCV capture, debounce (HARD CAP 1h — take the fallback ladder rather than overrun) |
| 6 | VLM client against the proxy, driveway prompts, live end-to-end on an RTSP camera |
| 6b | Video clips from a recorder, the Omni step, a worker per model (stretch, taken) |
| 7 | `/metrics` endpoint, optional HA webhook sink |
| 8 | README a stranger can run, sample fixtures, write-up drafted from the journal |

All eight landed. Phase 6b was the stretch the plan allowed for "if everything above is
done and verified"; it was, and it produced the two most useful findings of the build
(a worker per model, and the age limit eating the slow tier).

Working-and-verified beats feature-complete. If behind at any checkpoint, cut from the
bottom of the current block, not from verification. The gesture chains (tv-* in the
example config) are a **config-only demonstration** in this build — they showcase
zero-escalation one-shot chains and require a `mediapipe-hands` tier-1 source that is
stretch-goal only. Do not build it unless everything above is done and verified; the
schema must still support steps-less chains with `on_trigger`, and that IS in scope
(it's a config-validation case, not a model integration).

## Escalation must be earned

The project's premise is that each tier does something the tier below genuinely
cannot. Every demo condition must survive the reviewer question "couldn't
OpenVINO/YOLO have done this?" — if it can't, the whole build looks like
ceremony. Rules:

- Tier-1 conditions stay at COCO-class level (car, truck, person). That is what
  the cheap detector actually offers off the shelf.
- Escalation-step conditions must require at least one of: **relational
  reasoning** (which door of the truck the person exited, package in hand,
  heading toward the house), **fine-grained identity attributes** (make/model/
  color combos, man vs woman, vest), **intent disambiguation**, or
  **open-vocabulary conditions** (anything expressible as a prompt edit rather
  than a retrained model).
- When writing new example chains or prompts, check them against this list.
  If a step's condition is solvable by a cheap specialized model, don't force
  it up — resolve it AT the cheap tier. The tv-gesture chains are the worked
  example, by George's own call: finger counting is MediaPipe-Hands-on-CPU
  territory, so those chains have **no escalation steps at all** — the action
  fires one-shot at tier-1, with `held_frames` as the intent filter. The system
  demonstrably escalates only when escalation is needed, in both directions.
  (George intends to actually use the gesture feature, so `mediapipe-hands` as a
  second tier-1 source type is a real future component; in this build it is
  stretch-goal only, and the chains stand in the example config regardless.)

The write-up must include a short "why not a cheap model all the way down"
paragraph making three points honestly: (1) the demo conditions need the
reasoning classes above; (2) where a specialized small model *could* encroach
(gesture recognition being the honest example), the VLM buys open-vocabulary
config — a new condition is a prompt edit, not a data-collection-and-retraining
cycle; (3) the architecture treats the tier boundary as empirical and movable —
promoting a condition down to tier-1 when a cheap model proves sufficient is a
config change the design explicitly supports, not a refutation of it.

## Privacy

Real footage (driveway, family, interior cams) lives only in `fixtures/private/`
(gitignored). Anything committed under `fixtures/sample/` must be synthetic, public,
or explicitly cleared by George. When in doubt, don't commit it.
