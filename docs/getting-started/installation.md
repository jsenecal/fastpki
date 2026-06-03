# Installation

## Local Development

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Steps

```bash
# Clone the repository
git clone https://github.com/jsenecal/fastpki.git
cd fastpki

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (including dev tools)
uv pip install -e ".[dev]"

# Create a .env file from the example
cp .env.example .env

# Create the data directory for SQLite
mkdir -p data

# Run the application
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs` when `ENABLE_DOCS=true` is set in your environment; they are disabled by default to avoid exposing the OpenAPI schema in production.

## CLI Tool

The FastPKI CLI is published on PyPI as a standalone client. It does not require the server code — just install and point it at a running FastPKI instance:

```bash
# Install from PyPI
pip install fastpki
# or
uv tool install fastpki

# Configure the server
fastpki config set server.url http://localhost:8000

# Login
fastpki auth login -u admin -p securepassword

# Use it
fastpki ca list
fastpki cert list -o json
```

See the [CLI guide](../guides/cli.md) for full documentation.

## Docker — Development

Development mode mounts the source tree into the container so code changes reload automatically.

```bash
# Clone and enter the repo
git clone https://github.com/jsenecal/fastpki.git
cd fastpki

# Copy the example env file
cp .env.example .env

# Create the data directory
mkdir -p data

# Start in development mode (SQLite, live reload)
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
```

## Docker — Production

Production mode uses PostgreSQL and does **not** mount the source tree.

```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d
```

See [Docker deployment](../deployment/docker.md) for full details on volumes, environment variables, and building custom images.
