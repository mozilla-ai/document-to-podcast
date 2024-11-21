from llama_cpp import Llama

PROMPT = """
Convert this text into a podcast script.
The conversation should be between 2 speakers.
Use [SPEAKER1] and [SPEAKER2] to limit sections.
Do not include [INTRO], [OUTRO] or any other [SECTION].
Text:
"""


def load_model(
    model_id: str = "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf",
):
    org, repo, filename = model_id.split("/")
    model = Llama.from_pretrained(
        repo_id=f"{org}/{repo}",
        filename=filename,
        # 0 means that the model limit will be used, instead of the default (512) or other hardcoded value
        n_ctx=0,
    )
    return model


def text_to_podcast(
    input_text: str, model: Llama, system_prompt: str = PROMPT, stream: bool = False
):
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        stream=stream,
    )
    if stream:
        for item in response:
            if item["choices"][0].get("delta", {}).get("content", None):
                yield item["choices"][0].get("delta", {}).get("content", None)
    else:
        return response["choices"][0]["message"]["content"]
