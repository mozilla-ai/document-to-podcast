FROM python:3.10-slim

RUN pip3 install --no-cache-dir --upgrade pip
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git


RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY ./.git /home/appuser/document-to-podcast/.git
COPY ./src /home/appuser/document-to-podcast/src
COPY ./pyproject.toml /home/appuser/document-to-podcast/pyproject.toml

WORKDIR /home/appuser/document-to-podcast

RUN pip3 install /home/appuser/document-to-podcast

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser \
    && chown -R appuser:appuser /home/appuser

USER appuser

COPY ./demo demo
RUN python3 demo/download_models.py
EXPOSE 8501
ENTRYPOINT ["./demo/run.sh"]
