# CPU author runtime. Build from an immutable shared benchmark image ID.
ARG BENCHMARK_IMAGE
FROM ${BENCHMARK_IMAGE}
USER root
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends git libxrender1 libxext6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/author-requirements.lock
# TorchDrug's Python upper bound predates 3.11; the launcher supplies import
# compatibility. Keep base dependencies intact and verify every added wheel.
RUN pip install --no-cache-dir --no-deps --require-hashes --ignore-requires-python \
    -r /tmp/author-requirements.lock
COPY torchdrug_compat.py /opt/dicee/benchmarks/cqa/training/torchdrug_compat.py
RUN python /opt/dicee/benchmarks/cqa/training/torchdrug_compat.py > /opt/dicee/benchmarks/cqa/training/torchdrug-compat.json
ENV MAX_JOBS=2 TORCH_EXTENSIONS_DIR=/cache/torch_extensions MPLCONFIGDIR=/cache/matplotlib
# TorchDrug also creates short-lived compiler locks in the working directory.
WORKDIR /tmp
