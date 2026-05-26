FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
COPY pyproject.toml ./
COPY hermes_trading ./hermes_trading
# Bundle state defaults separately — the real /app/state is a persistent volume.
# docker-entrypoint.sh seeds the volume from here on first boot.
COPY state ./state-defaults
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh
RUN uv sync
ENV HERMES_TRADING_MODE=paper
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uv", "run", "python", "-m", "hermes_trading.run"]
