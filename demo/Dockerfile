FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu122/llama_cpp_python-0.3.4-cp310-cp310-linux_x86_64.whl
RUN pip3 install document-to-podcast

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser \
    && chown -R appuser:appuser /home/appuser

USER appuser

EXPOSE 8501
ENTRYPOINT ["python3", "app.py"]
