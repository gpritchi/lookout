"""Config validation contract — the basics, per CLAUDE.md.

Implement against src/lookout/config.py. Each test builds a minimal config dict
in-line (helper fixture) and asserts load/validation behavior:

- test_valid_example_config_loads: config/chains.example.json parses clean.
- test_unknown_model_rejected: step referencing a model absent from the registry
  fails with an error naming the step and the model.
- test_capability_gate_video_on_image_model: a step with payload "video_clip"
  whose model only declares ["image"] fails at load time. THE test that makes
  model swap-out safe.
- test_dangling_next_rejected: an outcome whose `next` names a nonexistent step.
- test_unknown_camera_rejected: chain bound to a camera not in cameras.

Scheduler basics live in test_scheduler.py once scheduler.py is real:
- higher priority number dequeues first; FIFO within equal priority.
- a worker configured for a subset of models never picks another model's job.

Debounce basics (test_tier1.py):
- N-1 consecutive frames of a label: no event. N: exactly one event.
"""
