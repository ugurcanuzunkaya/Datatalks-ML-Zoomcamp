
FROM python:3.12-slim-bookworm

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image.
ADD . /app

# Sync the project into a new environment, using the frozen lockfile.
WORKDIR /app
RUN uv sync --frozen

# Run the app
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
