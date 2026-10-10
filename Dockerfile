# API on the synthetic demonstration corpus. No model weights and no dataset are
# included in the image; mount a built data directory and set LEGALRAG_CONFIG to
# serve the real corpus.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra app --no-install-project
COPY src ./src
COPY configs ./configs
COPY app ./app
RUN uv sync --frozen --no-dev --extra app

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
ENV PATH="/app/.venv/bin:$PATH" LEGALRAG_CONFIG=/app/configs/demo.yaml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["legalrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
