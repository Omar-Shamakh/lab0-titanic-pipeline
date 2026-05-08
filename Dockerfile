# ── Base image ───────────────────────────────────────────────────
FROM python:3.11-slim

# ── Set working directory ─────────────────────────────────────────
WORKDIR /app

# ── Install uv ────────────────────────────────────────────────────
RUN pip install uv

# ── Copy dependency files first (for Docker layer caching) ────────
COPY pyproject.toml uv.lock ./

# ── Install dependencies ──────────────────────────────────────────
RUN uv sync --frozen --no-dev

# ── Copy the rest of the application ─────────────────────────────
COPY app/ ./app/
COPY src/ ./src/

# ── Expose port ───────────────────────────────────────────────────
EXPOSE 8000

# ── Start the API ─────────────────────────────────────────────────
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]