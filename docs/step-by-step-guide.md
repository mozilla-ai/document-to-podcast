# **Step-by-Step Guide: How the Document-to-Podcast Blueprint Works**

Transforming static documents into engaging podcast episodes involves an integration of pre-processing, LLM-powered transcript generation, and text-to-speech generation. Here's how it all works under the hood:

---

## **Overview**
This system has three core stages:


📄 **1. Document Pre-Processing**
   Prepare the input document by extracting and cleaning the text.

📜 **2. Podcast Script Generation**
   Use an LLM to transform the cleaned text into a conversational podcast script.

🎙️ **3. Audio Podcast Generation**
   Convert the script into an engaging audio podcast with distinct speaker voices.

We'll also look at how `app.py` brings all these steps together to build an end-to-end demo application.

First, let’s dive into each step to understand how this works in practice.


---

## **Step 1: Document Pre-Processing**

The process begins with preparing the input document for AI processing. The system handles various document types while ensuring the extracted content is clean and structured.

Cleaner input data ensures that the model works with reliable and consistent information, reducing the likelihood of confusing with unexpected tokens and therefore helping it to generate better outputs.

### ⚙️ **Key Components in this Step**

 **1 - File Loading**

   - Uses functions defined in [`data_loaders.py`](api.md/#document_to_podcast.preprocessing.data_loaders)

   - Supports `.html`, `.pdf`, `.txt`, and `.docx` formats.

   - Extracts readable text from uploaded files using specialized loaders.

 **2 - Text Cleaning**

   - Uses functions defined in [`data_cleaners.py`](api.md/#document_to_podcast.preprocessing.data_cleaners)

   - Removes unwanted elements like URLs, email addresses, and special characters using Python's `re` library, which leverages **Regular Expressions** (regex) to identify and manipulate specific patterns in text.

   - Ensures the document is clean and ready for the next step.

### 🔍 **API Example**

```py
from document_to_podcast.preprocessing import DATA_CLEANERS, DATA_LOADERS

input_file = "example_data/introducing-mozilla-ai-investing-in-trustworthy-ai.html"
data_loader = DATA_LOADERS[".html"]
data_cleaner = DATA_CLEANERS[".html"]

raw_data = data_loader(input_file)
print(raw_data[:200])
"""
<!doctype html>
<html class="no-js" lang="en-US">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="profile" href="https://gmpg.org/x
"""
clean_data = data_cleaner(raw_data)
print(clean_data[:200])
"""
Skip to content Mozilla Internet Culture Deep Dives Mozilla Explains Interviews Videos Privacy Security Products Firefox Pocket Mozilla VPN Mozilla News Internet Policy Leadership Mitchell Baker, CEO
"""
```

## **Step 2: Podcast Script Generation**

In this step, the pre-processed text is transformed into a conversational podcast transcript. Using a Language Model, the system generates a dialogue that’s both informative and engaging.

### ⚙️ **Key Components in this Step**

 **1 - Model Loading**

   - The [`model_loader.py`](api.md/#document_to_podcast.inference.model_loaders) module is responsible for loading the `text-to-text` models using the `llama_cpp` library.

   - The function `load_llama_cpp_model` takes a model ID in the format `{org}/{repo}/{filename}` and loads the specified model. This approach of using the `llama_cpp` library supports efficient CPU-based inference, making language models accessible even on machines without GPUs.

 **2 - Text-to-Text Generation**

   - The [`text_to_text.py`](api.md/#document_to_podcast.inference.text_to_text) script manages the interaction with the language model, converting input text into a structured conversational podcast script.

   - It uses the `chat_completion` function to process the input text and a customizable system prompt, guiding the language to generate a text output (e.g. a coherent podcast script between speakers).

   - The `return_json` parameter allows the output to be formatted as a JSON object style, which can make it easier to parse and integrate structured responses into applications.

   - Supports both single-pass outputs (`text_to_text`) and real-time streamed responses (`text_to_text_stream`), offering flexibility for different use cases.


### 🔍 **API Example**

```py
from document_to_podcast.inference.model_loaders import load_llama_cpp_model
from document_to_podcast.inference.text_to_text import text_to_text, text_to_text_stream

# Load the model
model = load_llama_cpp_model(
    "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf"
)

# Define your input and system prompt
input_text = (
    "Electric vehicles (EVs) have seen a significant rise in adoption over the past "
    "decade, driven by advancements in battery technology, government incentives, "
    "and growing consumer awareness of environmental issues."
)

system_prompt = (
    """
    You are a podcast scriptwriter generating engaging and natural-sounding conversations in JSON format.
    - Write dynamic, easy-to-follow dialogue.
    - Include natural interruptions and interjections.
    - Avoid repetitive phrasing between speakers.
    - Format output as a JSON conversation.
    Example:
    {
      "Speaker 1": "Welcome to our podcast! Today, we're exploring...",
      "Speaker 2": "Hi! I'm excited to hear about this. Can you explain...",
    }
    """
)

# Generate a podcast script from the input text
podcast_script = text_to_text(input_text, model, system_prompt)
print(podcast_script)

"""
{
  "Speaker 1": "Welcome to our podcast! Today, we're exploring the rise of electric vehicles (EVs) and what's driving this significant increase in adoption over the past decade.",
  "Speaker 2": "Absolutely, it's fascinating to see how the market has evolved and how consumers are becoming more environmentally conscious.",
  "Speaker 1": "Absolutely! Let's dive into the key factors driving this growth.",
  "Speaker 2": "Sure, here are a few key drivers: advancements in battery technology, government incentives, and growing consumer awareness of environmental issues.",
  ...
}
"""

# Example of real-time script generation with streaming
for chunk in text_to_text_stream(input_text, model, system_prompt):
    print(chunk, end="")

```


## **Step 3: Audio Podcast Generation**

In this final step, the generated podcast transcript is brought to life as an audio file. Using a Text-to-Speech (TTS) model, each speaker in the script is assigned a unique voice, creating an engaging and professional-sounding podcast.

### ⚙️ **Key Components in this Step**

 **1 - Model Loading**

   - The [`model_loader.py`](api.md/#document_to_podcast.inference.model_loaders) module is responsible for loading the `text-to-text` and `text-to-speech` models.

   - The function `load_outetts_model` takes a model ID in the format `{org}/{repo}/{filename}` and loads the specified model, either on CPU or GPU, based on the `device` parameter. The parameter `language` also enables to swap between the languages the Oute package supports (as of Dec 2024: `en, zh, ja, ko`)


**2 - Text-to-Speech Audio Generation**

   - The [`text_to_speech.py`](api.md/#document_to_podcast.inference.text_to_speech) script converts text into audio using a specified TTS model.

   - A **speaker profile** defines the voice characteristics (e.g., tone, speed, clarity) for each speaker. This is specific to each TTS package. Oute models require one of the IDs specified [here](https://github.com/edwko/OuteTTS/tree/main/outetts/version/v1/default_speakers).

   - The function `text_to_speech` takes the input text (e.g. podcast script) and speaker profile, generating a waveform (audio data in a numpy array) that represents the spoken version of the text.

### 🔍 **API Example**

```py
import soundfile as sf
from document_to_podcast.inference.model_loaders import load_outetts_model
from document_to_podcast.inference.text_to_speech import text_to_speech

# Load the TTS model
model = load_outetts_model(
    "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf"
)

# Generate the waveform
waveform = text_to_speech(
    input_text="Welcome to our amazing podcast",
    model=model,
    voice_profile="male_1"
)

# Save the audio file
sf.write(
    "podcast.wav",
    waveform,
    samplerate=model.sample_rate
)
```

## **Bringing It All Together in `app.py`**

The `app.py` demo app is shows you how all the components of the Document-to-Podcast Blueprint can come together. It demonstrates how you can take the individual steps—Document Pre-Processing, Podcast Script Generation, and Audio Podcast Generation—and integrate them into a functional application. This is the heart of the Blueprint in action, showing how you can build an app using the provided tools and components.

This demo uses [Streamlit](https://streamlit.io/), an open-source Python framework for interactive apps.

<div style="text-align: center;">
  <img src="../images/document-to-podcast-diagram.png" alt="Project Logo" style="width: 100%; margin-bottom: 1px; margin-top: 1px;">
</div>


---

### 🧠 **How `app.py` Applies Each Step**

**📄 Document Upload & Pre-Processing**

   - Users upload a file via the Streamlit interface (`st.file_uploader`), which supports `.pdf`, `.txt`, `.docx`, `.html`, and `.md` formats.

   - The uploaded file is passed to the **File Loading** and **Text Cleaning** modules.

   - Raw text is extracted using `DATA_LOADERS`, and the cleaned version is displayed alongside it using `DATA_CLEANERS`, and displayed to the end user.

**⚙️ Loading Models**

- The script uses `load_llama_cpp_model` from `model_loader.py` to load the LLM for generating the podcast script.

- Similarly, `load_outetts_model` is used to prepare the TTS model and tokenizer for audio generation.

- These models are cached using `@st.cache_resource` to ensure fast and efficient reuse during app interactions.

**📝 Podcast Script Generation**

 - The cleaned text and a system-defined podcast prompt are fed into the text_to_text_stream function.

 - The `DEFAULT_PROMPT` is loaded from `config.py`

 - The script is streamed back to the user in real-time, allowing them to see the generated conversation between speakers

**🎙️ Podcast Generation**

- For each speaker in the podcast script, audio is generated using the `text_to_speech` function with distinct speaker profiles

- The `DEFAULT_SPEAKERS` is loaded from `config.py`

- The generated audio is displayed with a player so users can listen directly in the app.


## 🎨 **Customizing the Blueprint**

To better understand how you can tailor this Blueprint to suit your specific needs, please visit the **[Customization Guide](customization.md)**.

## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!
