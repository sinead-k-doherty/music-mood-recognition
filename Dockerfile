FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y build-essential cmake \
    ffmpeg libsm6 libxext6 pkg-config libx11-dev \
    libatlas-base-dev libgtk-3-dev \ 
    libboost-python-dev 

RUN pip install poetry==1.1.0rc1

RUN mkdir /mmr_model

COPY pyproject.toml poetry.toml \
     /mmr_model/

WORKDIR /mmr_model

RUN poetry config virtualenvs.create true \
    && poetry install

RUN poetry install --no-interaction --no-root

COPY app/ /mmr_model/app/

EXPOSE 5000
