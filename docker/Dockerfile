FROM python:3.15-slim

ARG VERSION
LABEL org.opencontainers.image.version="${VERSION}"

WORKDIR /app

# Install uv via official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y curl && apt-get clean

# Copy files needed for dependency resolution
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY cli/ cli/
COPY app/ app/

# Install dependencies into system Python (no venv in container)
ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --no-dev --group server --no-editable --python-preference only-system

# Copy remaining application files
COPY alembic.ini ./
COPY alembic/ alembic/
COPY docker/ docker/

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos "" app
USER app

# Expose the application port
EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
