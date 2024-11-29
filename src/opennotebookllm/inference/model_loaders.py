from typing import Tuple

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from outetts import GGUFModelConfig_v1, InterfaceGGUF

def load_llama_cpp_model(
    model_id: str,
) -> Llama:
    """
    Loads the given model_id using Llama.from_pretrained.

    Examples:
        >>> model = load_llama_cpp_model(
            "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{org}/{repo}/{filename}`.

    Returns:
        Llama: The loaded model.
    """
    org, repo, filename = model_id.split("/")
    model = Llama.from_pretrained(
        repo_id=f"{org}/{repo}",
        filename=filename,
        # 0 means that the model limit will be used, instead of the default (512) or other hardcoded value
        n_ctx=0,
    )
    return model


def load_outetts_interface(
    model_id: str,
) -> InterfaceGGUF:
    local_path = hf_hub_download("OuteAI/OuteTTS-0.2-500M-GGUF", "OuteTTS-0.2-500M-FP16.gguf")

    model_config = GGUFModelConfig_v1(model_path=local_path,language="en")

    return InterfaceGGUF(model_version="0.2", cfg=model_config)
