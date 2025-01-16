import os

from huggingface_hub import HfApi

if __name__ == "__main__":
    api = HfApi()
    api.restart_space(
        repo_id="mozilla-ai/document-to-podcast",
        token=os.getenv("HF_TOKEN"),
        factory_reboot=True,
    )
