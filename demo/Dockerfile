FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG PYTHON_VERSION=3.13.15

RUN apt-get update && apt-get install --no-install-recommends -y \
  espeak-ng \
  build-essential \
  ca-certificates \
  curl \
  git \
  libbz2-dev \
  libffi-dev \
  liblzma-dev \
  libncursesw5-dev \
  libreadline-dev \
  libsqlite3-dev \
  libssl-dev \
  tk-dev \
  uuid-dev \
  wget \
  xz-utils \
  zlib1g-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o /tmp/Python.tgz \
  && tar -xzf /tmp/Python.tgz -C /tmp \
  && cd "/tmp/Python-${PYTHON_VERSION}" \
  && ./configure --enable-optimizations --with-ensurepip=install \
  && make -j"$(nproc)" \
  && make altinstall \
  && rm -rf /tmp/Python.tgz "/tmp/Python-${PYTHON_VERSION}"

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN python3.13 -m pip install --user --no-cache-dir --upgrade pip \
  && CMAKE_ARGS="-DGGML_CUDA=on" python3.13 -m pip install --user --no-cache-dir llama-cpp-python \
  && python3.13 -m pip install --user --no-cache-dir phonemizer document-to-podcast

COPY --chown=user . $HOME/app

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.enableXsrfProtection", "false"]
