from outetts import InterfaceGGUF


def text_to_speech(
    input_text: str,
    interface: InterfaceGGUF,
    speaker_profile: str,
    temperature: float = 0.3,
):
    return interface.generate(
        text=input_text,
        temperature=temperature,
        repetition_penalty=1.1,
        max_length=4096,
        speaker=interface.load_default_speaker(name=speaker_profile),
    )
