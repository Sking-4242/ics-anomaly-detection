# Base image: stable Python with good manylinux support
FROM python:3.11-slim

# Prevent Python from writing pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only dependency files first (better Docker caching)
COPY pyproject.toml poetry.lock* /app/

# Configure Poetry to install into the container environment
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the project
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Default command (can be overridden by docker-compose)
CMD ["bash"]
