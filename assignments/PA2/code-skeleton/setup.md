# PA-2 Setup Guide

> Part 0 of PA-2. Stand up the pgvector container, install Python deps, and prepare the corpus inventory.

## Prerequisites

- **Python 3.12** (Part 1+ uses `mypy --strict` against 3.12 syntax).
- **Podman 5.x** (Docker is not used in this PA). Verify with `podman --version`.
- **~1 GB free disk** for the pgimage plus the 3 RAG research papers at `data/corpus/`.
- **macOS / Linux / Windows** host.

## Install Podman

| OS | Command |
|---|---|
| Debian / Ubuntu | `sudo apt-get update && sudo apt-get install -y podman` |
| Arch / Manjaro | `sudo pacman -S podman` |
| macOS | `brew install podman && podman machine init && podman machine start` |
| Windows | Recommended to get wsl ubuntu. Fallback: `winget install -e --id RedHat.Podman` (then enable the `podman` Windows provider) |

Confirm: `podman --version` should print `podman version 5.x`.

## Pull the pgvector image

```bash
podman pull docker.io/pgvector/pgvector:pg18-trixie
```

The `:pg18-trixie` tag ships PostgreSQL 18 + the `vector` extension pre-installed. Do **not** use the plain `postgres:18` image — it has no pgvector binary.

## Run the container

```bash
podman run -dt \
  --name northwind-pgvector \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD='langchain!' \
  -e POSTGRES_DB=cs4603_vectordb \
  -v northwind-pgdata:/var/lib/postgresql \
  -p 5432:5432 \
  docker.io/pgvector/pgvector:pg18-trixie
```

Then run:
```bash
podman exec -it northwind-pgvector psql -U langchain -d cs4603_vectordb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Note:
- Container name `northwind-pgvector` and port `5432` are referenced by every later Part.

## Verify the connection

```bash
podman exec northwind-pgvector psql -U langchain -d cs4603_vectordb -c "SELECT version();"
podman exec northwind-pgvector psql -U langchain -d cs4603_vectordb -c "\dx"
```

The first command returns a non-empty `PostgreSQL 18.x ...` line. The second lists installed extensions and should include `vector`.

## Python virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

`requirements.txt` pins the deps for Part 0 (PyPDFLoader, tiktoken). Later Parts add their own deps to the same file; uncomment them as you reach each Part.

## Connection string

Parts 1+ use this SQLAlchemy/psycopg connection string everywhere:

```
postgresql+psycopg://langchain:langchain!@localhost:5432/cs4603_vectordb
```

Treat the password as a configuration value — load it from an environment variable (e.g., `os.environ["NORTHWIND_PG_URL"]`) in your own code rather than hard-coding it in source.

## When to reach for `podman-compose` (note for Task 0.1)

`podman run` is enough for a single container. When you need multi-service orchestration (e.g., pgvector + a sidecar like pgAdmin or a separate worker), reach for `podman-compose`. Compose reads two files: a declarative `compose.yaml` (services, networks, volumes, healthchecks) and a `.env` (secrets), and adds one extra capability: dependency-ordered startup with `depends_on` so the app waits for the database to be ready before launching. We are not switching to compose for this PA — `podman run` is sufficient for the single-service local setup.

## Tear-down (optional)

```bash
podman stop northwind-pgvector
podman rm northwind-pgvector
podman volume rm northwind-pgdata
```

The named volume outlives the container; remove it explicitly to wipe the data dir.
