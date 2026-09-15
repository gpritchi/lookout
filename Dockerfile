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
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# /metrics, when config enables it.
EXPOSE 9108
ENTRYPOINT ["python", "-m", "lookout"]
CMD ["replay", "--config", "config/chains.example.json", \
     "--events", "fixtures/sample/driveway-replay.jsonl", \
     "--answers", "fixtures/sample/fake-vlm-answers.json"]
