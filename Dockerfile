FROM ghcr.io/sinead-k-doherty/base-ml-image:latest

RUN mkdir /mmr_model

COPY pyproject.toml poetry.lock \
     /mmr_model/

WORKDIR /mmr_model

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root

COPY app/ /mmr_model/app/
