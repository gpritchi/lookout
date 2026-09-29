# Two stages: build the venv with uv, then copy it onto a slim runtime image.
# The same image runs the offline replay demo on a laptop and, later, the live
# pipeline on a cluster node; only the command changes.
#
# Torch is pinned to the CPU wheel index for Linux in pyproject.toml. Without
# that, the default resolution pulls the CUDA build and several GB of NVIDIA
# libraries that nothing here uses (tier-1 is OpenVINO/CPU; tier-2 is remote).

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Dependencies first, so source edits do not invalidate the (large) dependency layer.
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
# config that changes it gets a mismatched export, as it would have before.
FROM builder AS detector
ADD --checksum=sha256:f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt /models/yolov8n.pt
# cv2 links libGL and glib, and ultralytics imports cv2 even to export.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN /app/.venv/bin/python -c \
    "from ultralytics import YOLO; YOLO('/models/yolov8n.pt').export(format='openvino', imgsz=640, half=False)"

FROM python:3.13-slim-bookworm
# OpenCV's wheel links against libGL and glib even when no window is ever opened.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY config ./config
COPY fixtures/sample ./fixtures/sample
COPY --from=detector /models /app/models
# LOOKOUT_MODELS_DIR is the default for --models-dir. YOLO_CONFIG_DIR puts
# ultralytics' settings file under /tmp, the one writable path a locked-down pod
# has; it must already exist, or ultralytics warns and falls back anyway.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    LOOKOUT_MODELS_DIR=/app/models YOLO_CONFIG_DIR=/tmp
# /metrics, when config enables it.
EXPOSE 9108
ENTRYPOINT ["python", "-m", "lookout"]
CMD ["replay", "--config", "config/chains.example.json", \
     "--events", "fixtures/sample/driveway-replay.jsonl", \
     "--answers", "fixtures/sample/fake-vlm-answers.json"]
