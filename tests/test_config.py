"""Config validation contract — the basics, per CLAUDE.md.

Each test builds a minimal config dict inline and asserts load/validation
behaviour. The four cross-reference rules are THE tests that make model swap-out
safe: a bad config must fail at startup with a message naming the problem, never
at the moment a frame reaches the wrong model.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from lookout.config import Config, ConfigError, load_config

EXAMPLE = Path(__file__).parent.parent / "config" / "chains.example.json"


def minimal() -> dict:
    """The smallest valid config: one image-only model, one camera, one chain
    with a single step that ends."""
    return {
        "models": {
            "vlm": {
                "endpoint": "http://localhost:4000/v1",
                "model": "some-vlm",
                "capabilities": ["image"],
            }
        },
        "cameras": {"driveway": {"source": "clip.mp4"}},
        "chains": [
            {
                "id": "arrivals",
                "camera": "driveway",
                "trigger": {"label": "car", "priority": 10},
                "steps": [
                    {
                        "id": "classify",
                        "model": "vlm",
                        "payload": "image",
                        "priority": 50,
                        "timeout_s": 30,
                        "prompt": "What is it?",
                        "outcomes": {"ANY": {"end": True}},
                    }
                ],
            }
        ],
    }


def test_minimal_config_loads():
    config = Config.from_dict(minimal())
    assert config.chain("arrivals").steps[0].model == "vlm"
    assert config.tier1.debounce_frames == 3  # defaults fill in
    assert config.actions.sink == "mock"


def test_valid_example_config_loads():
    config = load_config(EXAMPLE)
    assert {c.id for c in config.chains} >= {"driveway-arrivals", "tv-netflix"}
    assert "video_clip" in config.models["nemotron-omni"].capabilities


def test_unknown_model_rejected():
    data = minimal()
    data["chains"][0]["steps"][0]["model"] = "ghost"
    with pytest.raises(ConfigError, match=r"step 'classify'.*model 'ghost'"):
        Config.from_dict(data)


def test_capability_gate_video_on_image_model():
    """THE test that makes model swap-out safe."""
    data = minimal()
    data["chains"][0]["steps"][0]["payload"] = "video_clip"
    data["chains"][0]["steps"][0]["window"] = {"before_s": 2, "after_s": 8, "frames": 6}
    with pytest.raises(ConfigError, match=r"payload 'video_clip'.*model 'vlm'.*\['image'\]"):
        Config.from_dict(data)


def test_capability_gate_passes_when_declared():
    data = minimal()
    data["models"]["vlm"]["capabilities"] = ["image", "video_clip"]
    data["chains"][0]["steps"][0]["payload"] = "video_clip"
    data["chains"][0]["window"] = {"before_s": 2, "after_s": 8, "frames": 6}  # chain-level default
    data["cameras"]["driveway"]["clips"] = {"playback": "http://rec:9996", "path": "driveway"}
    config = Config.from_dict(data)
    assert config.chain("arrivals").window_for(config.chain("arrivals").steps[0]).after_s == 8


def test_sequence_payload_requires_a_window():
    data = minimal()
    data["models"]["vlm"]["capabilities"] = ["image", "image_sequence"]
    data["chains"][0]["steps"][0]["payload"] = "image_sequence"
    with pytest.raises(ConfigError, match="no window"):
        Config.from_dict(data)


def test_buffer_seconds_is_deepest_lookback_per_camera():
    data = minimal()
    data["models"]["vlm"]["capabilities"] = ["image", "image_sequence"]
    data["chains"][0]["window"] = {"before_s": 5, "after_s": 8, "frames": 4}
    data["chains"][0]["steps"][0]["payload"] = "image_sequence"
    data["chains"].append({
        "id": "other", "camera": "driveway", "trigger": {"label": "truck"},
        "steps": [{"id": "s", "model": "vlm", "payload": "image_sequence", "timeout_s": 5, "prompt": "?",
                   "window": {"before_s": 12, "after_s": 1, "frames": 2}, "outcomes": {"X": {"end": True}}}],
    })
    config = Config.from_dict(data)
    assert config.buffer_seconds("driveway") == 12
    assert config.buffer_seconds("nonexistent") == 0


def test_dangling_next_rejected():
    data = minimal()
    data["chains"][0]["steps"][0]["outcomes"] = {"MORE": {"next": "nowhere"}}
    with pytest.raises(ConfigError, match=r"outcome 'MORE'.*'nowhere'"):
        Config.from_dict(data)


def test_unknown_camera_rejected():
    data = minimal()
    data["chains"][0]["camera"] = "attic"
    with pytest.raises(ConfigError, match=r"chain 'arrivals'.*camera 'attic'"):
        Config.from_dict(data)


def test_outcome_must_do_exactly_one_thing():
    data = minimal()
    data["chains"][0]["steps"][0]["outcomes"] = {"X": {"end": True, "retry_in_s": 5}}
    with pytest.raises(ConfigError, match="exactly one of"):
        Config.from_dict(data)

    data = minimal()
    data["chains"][0]["steps"][0]["outcomes"] = {"X": {}}
    with pytest.raises(ConfigError, match="exactly one of"):
        Config.from_dict(data)


def test_steps_less_chain_with_on_trigger_loads():
    """The one-shot case: tier-1 is enough, the action fires on the trigger."""
    data = minimal()
    data["chains"].append(
        {
            "id": "tv-off",
            "camera": "driveway",
            "trigger": {"label": "gesture:ok_sign", "held_frames": 8, "priority": 100},
            "steps": [],
            "on_trigger": {"action": {"type": "ha_webhook", "service": "tv_off"}},
        }
    )
    config = Config.from_dict(data)
    chain = config.chain("tv-off")
    assert chain.steps == []
    assert chain.on_trigger is not None
    assert chain.on_trigger.action.type == "ha_webhook"
    assert chain.trigger.held_frames == 8


def test_steps_less_chain_without_on_trigger_rejected():
    data = minimal()
    data["chains"][0]["steps"] = []
    with pytest.raises(ConfigError, match="no steps and no on_trigger"):
        Config.from_dict(data)


def test_webhook_sink_requires_url():
    data = minimal()
    data["actions"] = {"sink": "ha_webhook"}
    with pytest.raises(ConfigError, match="ha_webhook_url"):
        Config.from_dict(data)


def test_unknown_fields_rejected():
    """Typos in keys must not be silently ignored."""
    data = minimal()
    data["chains"][0]["steps"][0]["timeout"] = 30  # should be timeout_s
    with pytest.raises(ConfigError, match="timeout"):
        Config.from_dict(data)


def test_trigger_accepts_one_label_or_several():
    single = Config.from_dict(minimal()).chain("arrivals").trigger
    assert single.labels == ["car"] and single.matches("car") and not single.matches("truck")
    data = minimal()
    data["chains"][0]["trigger"]["label"] = ["car", "truck"]
    multi = Config.from_dict(data).chain("arrivals").trigger
    assert multi.matches("truck") and not multi.matches("person")


def test_original_dict_not_mutated():
    data = minimal()
    before = deepcopy(data)
    Config.from_dict(data)
    assert data == before
