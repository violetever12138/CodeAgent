# syntax=docker/dockerfile:1.7
# Supply a digest-pinned Python 3.12 Linux image, for example:
#   --build-arg BASE_IMAGE=python:3.12-slim@sha256:<64-hex-digest>
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG PYTEST_VERSION=8.4.2
ARG PYTEST_ASYNCIO_VERSION=1.2.0

RUN test -n "${BASE_IMAGE}" \
    && python -m pip install --no-cache-dir \
       "pytest==${PYTEST_VERSION}" \
       "pytest-asyncio==${PYTEST_ASYNCIO_VERSION}" \
    && python -m pip check

RUN groupadd --gid 10001 codeswarm \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin codeswarm \
    && mkdir -p /workspace /agent_tools \
    && chown 10001:10001 /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INDEX=1

WORKDIR /workspace
USER 10001:10001
CMD ["sleep", "infinity"]
