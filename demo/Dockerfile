FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  python3.10 \
  python3-pip \
  git \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu122/llama_cpp_python-0.3.4-cp310-cp310-linux_x86_64.whl
RUN pip3 install document-to-podcast

EXPOSE 8501
ENTRYPOINT ["python3", "app.py"]
