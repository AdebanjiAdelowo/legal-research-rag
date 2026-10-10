# API on the synthetic demonstration corpus. No model weights and no dataset are
# included in the image; mount a built data directory and set LEGALRAG_CONFIG to
# serve the real corpus.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
RUN useradd --create-home appuser && mkdir /app && chown appuser /app
USER appuser
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1

COPY --chown=appuser pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra api --no-install-project
COPY --chown=appuser src ./src
COPY --chown=appuser configs ./configs
RUN uv sync --frozen --no-dev --extra api

ENV PATH="/app/.venv/bin:$PATH" LEGALRAG_CONFIG=/app/configs/demo.yaml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["legalrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
