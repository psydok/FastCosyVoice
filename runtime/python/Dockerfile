FROM ghcr.io/astral-sh/uv:0.10.2 AS uv-builder

FROM nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3 AS builder

COPY --from=uv-builder /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y --no-install-recommends python3.10-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cosyvoice
RUN git clone https://github.com/psydok/FastCosyVoice . && \
    rm -fr .git

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project

FROM nvcr.io/nvidia/tensorrt:24.12-py3
RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt install -y --no-install-recommends python3.10-dev && \
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /opt/cosyvoice /opt/cosyvoice
WORKDIR /opt/cosyvoice

COPY --from=uv-builder /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN uv pip install "grpcio-tools<=1.76.0" "grpcio<=1.76.0"
RUN cd runtime/python/grpc && uv run -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
ENV VLLM_NO_USAGE_STATS=1
