FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim-2022-11-25
ENV PYTHONUNBUFFERED=1
WORKDIR /code

# Install Poetry
RUN apt clean && apt update && apt install curl -y
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy poetry.lock* in case it doesn't exist in the repo
COPY app/pyproject.toml app/poetry.lock* /code/

# Ajout des dépendances pour OpenTelemetry
RUN poetry add opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-exporter-otlp \
    prometheus_client \
    email-validator

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --with dev --no-root ; else poetry install --no-root --without dev ; fi"

ENV PYTHONPATH=/code
EXPOSE 8000