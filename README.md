# LLM Universal Adapter

![CI Pipeline](https://github.com/placeholder/repo/actions/workflows/ci.yml/badge.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 1. Title & Overview
**LLM Universal Adapter** is a resilient backend service designed to standardize and streamline interactions with various Large Language Models (LLMs) such as OpenAI and Google Gemini. The system includes an asynchronous web scraper and utilizes a PostgreSQL database with the `pgvector` extension for Retrieval-Augmented Generation (RAG) capabilities, supported by Redis for caching or background tasks.

## 2. Architecture & Tech Stack
- **Language**: Python 3.11+
- **Application Server**: uWSGI
- **Configuration**: Pydantic Settings
- **LLM SDKs**: `openai`, `google-genai`
- **Scraping**: `aiohttp`, `beautifulsoup4`
- **Resilience**: `tenacity`
- **Database**: PostgreSQL (with `pgvector` via `ankane/pgvector`)
- **Cache / Message Broker**: Redis

### High-Level Architecture
- **API**: Serves incoming requests utilizing a unified adapter pattern to interface with multiple LLM providers.
- **Adapters (`src/adapter/`)**: Implements specific logic for OpenAI and Gemini using a factory pattern.
- **Scraper (`src/scraper/`)**: Features an asynchronous crawler for data ingestion.
- **Data Storage**: A containerized PostgreSQL database configured for RAG using pgvector, and a Redis instance.

## 3. Prerequisites
To run and develop this application, ensure you have the following installed:
- **Python**: 3.11 or higher
- **Docker**: Latest version (for containerized deployment)
- **Docker Compose**: Latest version (for orchestrating DB, Redis, and API)

## 4. Environment Variables
Create a `.env` file in the root directory. Use `.env.example` as a template.

| Variable | Type | Required/Optional | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | String | Optional* | API key for authenticating with OpenAI. |
| `GEMINI_API_KEY` | String | Optional* | API key for authenticating with Google Gemini. |
| `DEFAULT_TEMPERATURE` | Float | Optional | Default temperature setting for LLM responses (default: `0.7`). |
| `DATABASE_URL` | String | Required | Connection string for PostgreSQL (e.g., `postgresql://user:pass@db:5432/rag_db`). |
| `REDIS_URL` | String | Required | Connection string for Redis (e.g., `redis://redis:6379/0`). |

*\*At least one LLM API key must be provided depending on your usage.*

## 5. Installation & Local Setup

### a) Bare-metal / Local Virtual Environment
1. Clone the repository and navigate into the directory.
2. Create and activate a Python virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your `.env` file manually.

### b) Containerized Setup (Docker Compose)
1. Ensure Docker and Docker Compose are installed and running.
2. Build and start the services:
   ```bash
   docker-compose up --build -d
   ```
This will spin up the `api`, `db` (PostgreSQL + pgvector), and `redis` containers.

## 6. Usage & Execution

### Starting the Core Application
If running bare-metal, use the following `uwsgi` command to start the application:
```bash
uwsgi --http :8000 --module src.main:app --processes 4 --threads 2
```

If using Docker Compose, the services will start automatically when you run:
```bash
docker-compose up
```

### Stopping the Containerized Application
```bash
docker-compose down
```

## 7. Testing & Linting
The project enforces code quality through tests, static type checking, and linting. Make sure to install the testing dependencies before running these commands:

```bash
pip install pytest pytest-mock ruff mypy tiktoken
```

### Unit & Integration Tests
Run tests using `pytest` with the configuration defined in `pytest.ini`:
```bash
pytest tests/
```

### Static Type Checking
Check typing with `mypy` using configurations mapped in `pyproject.toml`:
```bash
mypy src/
```

### Linting & Formatting
Ensure code styling and catch errors with `ruff`:
```bash
ruff check .
```

## 8. API / Core Interfaces
- **Application Entry Point**: The primary module to run the API is expected to be at `src.main:app`.
- **Adapter Factory**: The main interaction layer for models is routed through `src/adapter/factory.py`, instantiating logic from `openai_adapter.py` and `gemini_adapter.py`.
- **Asynchronous Crawler**: Used for fetching context data via `src/scraper/async_crawler.py`.

## 9. Deployment
- **CI/CD Pipeline**: GitHub Actions workflows are defined in `.github/workflows/ci.yml`. The pipeline automatically tests against Python 3.11 by installing dependencies, running `ruff`, `mypy`, and `pytest` on every push and pull request to the `main` branch.
- **Docker Deployment**: The `docker-compose.yml` configures an optimized multi-container environment ready for deployment to cloud providers supporting Docker Compose or as a starting point for Kubernetes orchestration.