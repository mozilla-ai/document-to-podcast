FROM python:3.10-slim

RUN pip3 install --no-cache-dir --upgrade \
    pip \
    virtualenv

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git

    ENV VIRTUAL_ENV=/home/venv

COPY . /home/appuser/document-to-podcast
WORKDIR /home/appuser/document-to-podcast

RUN virtualenv ${VIRTUAL_ENV}
RUN . ${VIRTUAL_ENV}/bin/activate && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN . ${VIRTUAL_ENV}/bin/activate && pip install /home/appuser/document-to-podcast
RUN . ${VIRTUAL_ENV}/bin/activate && python demo/download_models.py

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

USER appuser

EXPOSE 8501
ENTRYPOINT ["./demo/run.sh"]
