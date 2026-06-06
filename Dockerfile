FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 AS trainer

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.83.0
ARG NODE_VERSION=22.22.0
ARG CUTLASS_VERSION=v3.7.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    CUTLASS_PATH=/opt/cutlass \
    CPLUS_INCLUDE_PATH=/opt/cutlass/include:/opt/cutlass/tools/util/include \
    PATH=/opt/cargo/bin:/opt/node/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl build-essential cmake ninja-build pkg-config git \
      python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal \
    && rustup component add rustfmt clippy

RUN curl -fsSL https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz \
    | tar -xJ -C /opt \
    && ln -s /opt/node-v${NODE_VERSION}-linux-x64 /opt/node

RUN git clone --depth 1 --branch ${CUTLASS_VERSION} https://github.com/NVIDIA/cutlass.git /opt/cutlass \
    && test -f /opt/cutlass/include/cutlass/cutlass.h \
    && test -d /opt/cutlass/include/cute

WORKDIR /workspace
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

COPY dashboard/package*.json dashboard/
RUN cd dashboard && npm ci

COPY . .

CMD ["python3", "-u", "tools/v8_train.py", "--help"]
