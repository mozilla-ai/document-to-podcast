FROM python:3.10-slim

RUN pip3 install --no-cache-dir --upgrade pip
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git


COPY . /home/appuser/document-to-podcast
WORKDIR /home/appuser/document-to-podcast

RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install git+https://github.com/huggingface/parler-tts.git
RUN pip3 install /home/appuser/document-to-podcast
RUN python3 demo/download_models.py

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

USER appuser

EXPOSE 8501
ENTRYPOINT ["./demo/run.sh"]
