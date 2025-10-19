FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git curl cmake libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

ENV UV_PYTHON=/usr/local/bin/python3.12
WORKDIR /workspaces/SafeGBPO
ENV PYTHONPATH="/workspaces/SafeGBPO/src:${PYTHONPATH}"
SHELL ["/bin/bash", "-lc"]
CMD ["bash"]