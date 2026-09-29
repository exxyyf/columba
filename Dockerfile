# Сервис инференса columba (README, раздел «Сборка и запуск в Docker»;
# подробное руководство — docs/deployment.md).
#
# Один образ для CPU и GPU: стандартное PyPI-колесо torch само определяет
# CUDA в рантайме (`region_cnn.default_device`) — специальной CUDA-базы не
# нужно, но для GPU на хосте нужны драйверы NVIDIA + nvidia-container-toolkit,
# контейнер запускается с `--gpus all`. Без GPU падает на CPU честно (уже
# ~0.08 с/файл на CPU, см. stages/stage_7.md — лимит 3 мин/
# исследование выполняется с огромным запасом).
#
# Веса CNN-веток не запекаются в образ. Перед запуском контейнер проверяет
# смонтированные artifacts/models/ по weights_registry.json и завершает работу
# с ошибкой, если хотя бы одного файла нет или его SHA-256 не совпадает.
#
# --- CPU-вариант сборки (этап 9, TORCH_VARIANT) --------------------------- #
# По умолчанию `uv sync --frozen` ставит torch из `uv.lock` (на Linux — колесо
# с бандлом CUDA, несколько ГБ). Для CI/демо без GPU:
#   docker build --build-arg TORCH_VARIANT=cpu -t columba:cpu .
# внутри образа пересобирает lock-файл для Linux/Python 3.12 с явным CPU-индексом
# только для torch/torchvision. Файлы проекта на хосте не меняются.

FROM python:3.12-slim AS base

ARG TORCH_VARIANT=gpu

# uv — многостадийно из официального образа, без pip install в рантайм-слое.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Зависимости отдельным слоем — кэшируются, пока pyproject.toml/uv.lock не меняются.
COPY pyproject.toml uv.lock ./
COPY docker/prepare_cpu_lock.py /tmp/prepare_cpu_lock.py
RUN if [ "$TORCH_VARIANT" = "cpu" ]; then \
        python /tmp/prepare_cpu_lock.py && \
        uv lock --upgrade-package torch --upgrade-package torchvision; \
    elif [ "$TORCH_VARIANT" != "gpu" ]; then \
        echo "Unsupported TORCH_VARIANT: $TORCH_VARIANT" >&2; exit 1; \
    fi
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY main.py ./
COPY README.md weights_registry.json ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

# Веса читаются из artifacts/models/ (см. WEIGHTS.md), монтируются read-only.
#   docker run -p 8000:8000 --mount type=bind,source="$(pwd)/artifacts",target=/app/artifacts,readonly columba
CMD ["sh", "-c", "python -m columba.weights verify && exec uvicorn columba.service:app --host 0.0.0.0 --port 8000"]
