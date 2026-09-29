# Three stages: build the runtime venv with uv, export the detector in a stage
# of its own, then copy both onto a slim runtime image. The same image runs the
# offline replay demo on a laptop and the live pipeline on a cluster node; only
# the command changes.
#
# The runtime needs neither Ultralytics nor torch: tier1.py runs the exported
# OpenVINO model with the openvino runtime directly. They are the `torch` extra
# in pyproject.toml, installed only in the export stage and pinned there to
# the CPU wheel index (the default resolution pulls the CUDA build and several
# GB of NVIDIA libraries). With them in the runtime venv, plus the GUI OpenCV
# build and the libGL it needs, the image was 1.99 GB on disk; without, 559 MB.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# The detector, baked in: pinned weights exported to OpenVINO at build time, so
# a pod needs no internet and no writable models dir to start. The URL is a
# fixed release asset and the checksum pins its bytes: the same weights every
# tuning run in this repo used. imgsz matches Tier1Spec's default (640); a
# config that changes it is refused at startup with a message saying so.
# Its own venv, from the lockfile alone, so a source edit does not redo the
# torch install and the export.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS detector
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /export
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra torch
ADD --checksum=sha256:f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt /models/yolov8n.pt
RUN YOLO_CONFIG_DIR=/tmp /export/.venv/bin/python -c \
    "from ultralytics import YOLO; YOLO('/models/yolov8n.pt').export(format='openvino', imgsz=640, half=False)"

# No apt packages: the headless OpenCV wheel does not link libGL.
FROM python:3.13-slim-bookworm
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY config ./config
COPY fixtures/sample ./fixtures/sample
# Only the export: the .pt weights stay in the export stage.
COPY --from=detector /models/yolov8n_openvino_model /app/models/yolov8n_openvino_model
# LOOKOUT_MODELS_DIR is the default for --models-dir.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    LOOKOUT_MODELS_DIR=/app/models
# /metrics, when config enables it.
EXPOSE 9108
ENTRYPOINT ["python", "-m", "lookout"]
CMD ["replay", "--config", "config/chains.example.json", \
     "--events", "fixtures/sample/driveway-replay.jsonl", \
     "--answers", "fixtures/sample/fake-vlm-answers.json"]
