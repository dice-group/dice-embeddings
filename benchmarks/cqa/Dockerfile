FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS runtime
RUN rm /etc/apt/sources.list.d/debian.sources \
    && echo 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260901T000000Z bookworm main' > /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

FROM runtime AS dependencies

COPY benchmarks/cqa/requirements.lock /tmp/requirements.lock
COPY wheels /wheels
RUN sed 's|torch @ https://[^ ]*|torch==2.9.1+cu128|' /tmp/requirements.lock > /tmp/offline.lock \
    && pip install --no-cache-dir --no-index --find-links=/wheels --require-hashes -r /tmp/offline.lock \
    && pip check

FROM runtime
COPY --from=dependencies /usr/local /usr/local
WORKDIR /opt/dicee
COPY dicee dicee
COPY pyproject.toml pyproject.toml
COPY benchmarks/__init__.py benchmarks/__init__.py
COPY benchmarks/cqa benchmarks/cqa
COPY benchmarks/adapters benchmarks/adapters
COPY benchmarks/plus_h benchmarks/plus_h
COPY benchmarks/ultraquery benchmarks/ultraquery
ARG SOURCE_COMMIT=unknown
ARG SOURCE_DIRTY=unknown
LABEL org.opencontainers.image.revision=$SOURCE_COMMIT
ENV DICEE_SOURCE_COMMIT=$SOURCE_COMMIT \
    DICEE_SOURCE_DIRTY=$SOURCE_DIRTY \
    PYTHONPATH=/opt/dicee \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_ALLOC_CONF=expandable_segments:True
ENTRYPOINT ["python", "-m", "benchmarks.cqa"]
