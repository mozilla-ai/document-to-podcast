from typing import Iterator

from llama_cpp import Llama


def chat_completion(
    input_text: str,
    model: Llama,
    system_prompt: str,
    return_json: bool,
    stream: bool,
    stop: str | list[str] | None = None,
) -> str | Iterator[str]:
    # create_chat_completion uses an empty list as default
    stop = stop or []
    return model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        response_format={
            "type": "json_object",
        }
        if return_json
        else None,
        stream=stream,
        stop=stop,
    )


def text_to_text(
    input_text: str,
    model: Llama,
    system_prompt: str,
    return_json: bool = True,
    stop: str | list[str] | None = None,
) -> str:
    """
    Transforms input_text using the given model and system prompt.

    Args:
        input_text (str): The text to be transformed.
        model (Llama): The model to use for conversion.
        system_prompt (str): The system prompt to use for conversion.
        return_json (bool, optional): Whether to return the response as JSON.
            Defaults to True.
        stop (str | list[str] | None, optional): The stop token(s).

    Returns:
        str: The full transformed text.
    """
    response = chat_completion(
        input_text, model, system_prompt, return_json, stop=stop, stream=False
    )
    return response["choices"][0]["message"]["content"]


def text_to_text_stream(
    input_text: str,
    model: Llama,
    system_prompt: str,
    return_json: bool = True,
    stop: str | list[str] | None = None,
) -> Iterator[str]:
    """
    Transforms input_text using the given model and system prompt.

    Args:
        input_text (str): The text to be transformed.
        model (Llama): The model to use for conversion.
        system_prompt (str): The system prompt to use for conversion.
        return_json (bool, optional): Whether to return the response as JSON.
            Defaults to True.
        stop (str | list[str] | None, optional): The stop token(s).

    Yields:
        str: Chunks of the transformed text as they are available.
    """
    response = chat_completion(
        input_text, model, system_prompt, return_json, stop=stop, stream=True
    )
    for item in response:
        if item["choices"][0].get("delta", {}).get("content", None):
            yield item["choices"][0].get("delta", {}).get("content", None)
