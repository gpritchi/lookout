"""Config loading and validation (pydantic).

Load config/chains.json into typed models: ModelSpec (endpoint, capabilities),
CameraSpec, Tier1Spec, ChainSpec/StepSpec.

Validation rules — all enforced at load time, all with clear error messages:

1. Every step's `model` exists in the models registry.
2. Capability gate: every step's `payload` type is in its model's declared
   `capabilities`. Swapping a clip-watching model for an image-only one must fail
   here, at startup, not at runtime.
3. Every `next` in an outcome references a step id in the same chain.
4. Every chain's `camera` exists in cameras.

These four rules are the contract the tests in tests/test_config.py pin down.
"""
