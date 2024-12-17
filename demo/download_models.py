"""
Used when building the Dockerfile to download the models that are used in the hosted demo
"""

from document_to_podcast.inference.model_loaders import (
    load_llama_cpp_model,
    load_tts_model,
)

load_llama_cpp_model(
    "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf"
)
load_tts_model("OuteAI/OuteTTS-0.2-500M-GGUF/OuteTTS-0.2-500M-FP16.gguf")
