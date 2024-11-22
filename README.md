<img src="./images/Blueprints-logo.png" alt="Project Logo" style="width:25%;">

# OpenNotebookLLM - a Blueprint for generating podcasts from documents using AI

This blueprint shows you can use open-source models & tools to convert input documents into a podcast featuring two speakers. 
It is designed to work on most local setups or with [GitHub Codespaces](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=888426876&skip_quickstart=true), meaning no external API calls or GPU access is required. This makes it more accessible and privacy-friendly by keeping everything local.

![Blueprint Diagram](./images/blueprint-diagram.png)

## Quick-start

---

## How it Works

Architecture diagram to be added

1. **Document Pre-Processing**  
This step extracts and cleans text from input documents to prepare it for the language model.

- **File Loading (`data_loaders.py`)**: Supports PDF, .txt, and .docx formats, extracting readable text from these files.  
- **Text Cleaning (`data_cleaners.py`)**: Removes noise like URLs, email addresses, and special characters to ensure a clean input.

These steps ensure the document text is structured and ready for generating high-quality podcast transcripts.

2. **Podcast Transcript Generation**  

In this step, the cleaned text from the pre-processing stage is processed by an LLM to generate a podcast transcript in the form of a conversation between two speakers.

- **Model Loader (`model_loader.py`)**: Loads GGUF-type LLM models from repositories using the llama_cpp library. This enables the models to run efficiently on CPUs, making them more accessible and suitable for local setups.

- **Text-to-Text Interaction (`text_to_text.py`)**: Combines the text from the input doc and a user-defined 'system prompt' and feeds this to loaded LLM to generate the desired text output, such as conversational transcript.

Together, **Model Loading** and **Text-to-Text Interaction** can be combined, as demonstrated in `app.py`, to produce a podcast transcript:  
```json
{
    "Speaker 1": "Welcome to the podcast on AI advancements.",
    "Speaker 2": "Thank you! What are the latest trends?",
    ...
}
```

3. **Audio Podcast Creation**  
   - Converts the transcript into audio using a TTS (Text-to-Speech) model with distinct voices for each speaker.
   - Outputs an audio file in formats like MP3 or WAV.


## Pre-requisites

- **System requirements**:
  - OS: Windows, macOS, or Linux
  - Python 3.10 or higher
  - Minimum RAM: 4 GB
  - Disk space: 1 GB minimum

- **Dependencies**:
  - Dependencies listed in `requirements.txt`

## Installation

---


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
