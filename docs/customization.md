# 🎨 **Customization Guide**

The Document-to-Podcast Blueprint is designed to be flexible and easily adaptable to your specific needs. This guide will walk you through some key areas you can customize to make the Blueprint your own.

---

## 🧠 **Changing the Text-to-Text Model**
You can swap the language model used for generating podcast scripts to suit your needs, such as using a smaller model for faster processing or a larger one for higher quality outputs.

Customizing the app:

1. Open the `app.py` file.
2. Locate the `load_text_to_text_model` function.
3. Replace the `model_id` with the ID of your desired model from a supported repository (e.g., Hugging Face). Note: The model repository must be in GGFUF format, for example: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`

Example:

```python
@st.cache_resource
def load_text_to_text_model():
    return load_llama_cpp_model(
        model_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf"
```


## 📝 **Modifying the Text Generation Prompt**
The system prompt defines the structure and tone of the generated script. Customizing this can allow you to generate conversations that align with your project’s needs.

Customizing the app:

1.	Open the `app.py` file.
2.	Locate the PODCAST_PROMPT variable.
3.	Edit the instructions to suit your desired conversation style.

Example:

```python
PODCAST_PROMPT = """
You are a radio show scriptwriter generating lively and humorous dialogues.
Speaker 1: A comedian who is interested in learning new things.
Speaker 2: A scientist explaining concepts in a fun way.
"""
```


## 🎙️ **Customizing Speaker Descriptions**
Adjusting the speaker profiles allows you to create distinct and engaging voices for your podcast.

Customizing the app:

1. Open the `app.py` file.
2.	Locate the SPEAKER_DESCRIPTIONS dictionary.
3.	Update the descriptions to define new voice characteristics for each speaker
Example:

```python
SPEAKER_DESCRIPTIONS_OUTE = {
    "1": "A cheerful and animated voice with a fast-paced delivery.",
    "2": "A calm and deep voice, speaking with authority and warmth."
}
"""
```


## 🧠 **Changing the Text-to-Speech Model**
You can use a different TTS model to achieve specific voice styles or improve performance.

Customizing the app:

1. Open the `app.py` file.
2. Locate the `load_text_to_speech_model_and_tokenizer` function.
3.	Replace the model_id with your preferred TTS model.

Example:
```python
@st.cache_resource
def load_text_to_speech_model_and_tokenizer():
    return load_parler_tts_model_and_tokenizer(
        "parler-tts/parler-tts-mini-expresso", "cpu")
```

## 💡 Other Customization Ideas

- Add Multiple Speakers: Modify `script_to_audio.py` to include additional speakers in your podcast.


## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!
