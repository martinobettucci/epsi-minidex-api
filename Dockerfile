FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && curl -LsSf https://astral.sh/uv/0.8.14/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY gen-server.py .
COPY env certs minimon ./

EXPOSE 22222
CMD ["/app/.venv/bin/python", "gen-server.py"]
