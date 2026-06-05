# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 AS trainer

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.83.0
ARG NODE_VERSION=22.22.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/opt/node/bin:$PATH

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl build-essential cmake ninja-build pkg-config git \
      python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal \
    && rustup component add rustfmt clippy

RUN curl -fsSL https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz \
    | tar -xJ -C /opt \
    && ln -s /opt/node-v${NODE_VERSION}-linux-x64 /opt/node

WORKDIR /workspace
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

COPY dashboard/package*.json dashboard/
RUN --mount=type=cache,target=/root/.npm \
    cd dashboard && npm ci

COPY . .

CMD ["python3", "-u", "tools/v8_train.py", "--help"]
