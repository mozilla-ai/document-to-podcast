"""
Used when building the Dockerfile to download the models that are used in the hosted demo
"""

from document_to_podcast.inference.model_loaders import (
    load_llama_cpp_model,
    load_parler_tts_model_and_tokenizer,
)

load_llama_cpp_model(
    model_id="allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf"
)
load_parler_tts_model_and_tokenizer("parler-tts/parler-tts-mini-v1")
