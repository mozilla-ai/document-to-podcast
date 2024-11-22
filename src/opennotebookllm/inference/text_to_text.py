from llama_cpp import Llama


def text_to_text(
    input_text: str, model: Llama, system_prompt: str, stream: bool = False
):
    """
    Transforms input_text using the given model and system prompt.

    Args:
        input_text (str): The text to be transformed.
        model (Llama): The model to use for conversion.
        system_prompt (str): The system prompt to use for conversion.
        stream (bool, optional): Whether to stream the response. Defaults to False.

    Yields:
        str: Chunks of the transformed text as they are available. If stream=True

    Returns:
        str: The full transformed text. If stream=False.
    """
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
