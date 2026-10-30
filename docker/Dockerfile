FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        gromacs \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# PyTorch with CUDA 12.1 runtime (shipped inside the wheel; host provides driver via --gpus all)
RUN pip install --no-cache-dir \
        torch==2.3.* --index-url https://download.pytorch.org/whl/cu121

COPY . /opt/asmsa_v2
RUN pip install --no-cache-dir /opt/asmsa_v2[dev] && \
    pip install --no-cache-dir jupyterlab mdtraj matplotlib

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
