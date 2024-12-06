[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Docs](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/)

<p align="center"><img src="./images/Blueprints-logo.png" width="35%" alt="Project logo"/></p>

# Document-to-podcast: a Blueprint by Mozilla.ai for generating podcasts from documents using local AI

This blueprint demonstrate how you can use open-source models & tools to convert input documents into a podcast featuring two speakers.
It is designed to work on most local setups or with [GitHub Codespaces](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=888426876&skip_quickstart=true&machine=standardLinux32gb), meaning no external API calls or GPU access is required. This makes it more accessible and privacy-friendly by keeping everything local.

### 👉 📖 For more detailed guidance on using this project, please visit our [Docs here](https://mozilla-ai.github.io/document-to-podcast/).

### Built with
- Python 3.10+
- [Llama-cpp](https://github.com/abetlen/llama-cpp-python) (text-to-text, i.e script generation)
- [Parler_tts](https://github.com/huggingface/parler-tts) (text-to-speech, i.e audio generation)
- [Streamlit](https://streamlit.io/) (UI demo)


## Quick-start

Get started with Document-to-Podcast using one of the two options below: **GitHub Codespaces** for a hassle-free setup or **Local Installation** for running on your own machine.

---

### **Option 1: GitHub Codespaces**

The fastest way to get started. Click the button below to launch the project directly in GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=888426876&skip_quickstart=true&machine=standardLinux32gb)

Once the Codespaces environment launches, inside the terminal, start the Streamlit demo by running:
   ```bash
   python -m streamlit run demo/app.py
   ```

### **Option 2: Local Installation**

1. **Clone the Repository**
   Inside the Codespaces terminal, run:
   ```bash
   git clone https://github.com/mozilla-ai/document-to-podcast.git
   cd document-to-podcast
   ```

2. **Install Dependencies**
   Inside the terminal, run:
   ```bash
   pip install -e .
3. **Run the Demo**
   Inside the terminal, start the Streamlit demo by running:
   ```bash
   python -m streamlit run demo/app.py
   ```

***NOTE***: The first time you run the demo app it might take a while to generate the script or the audio because it will download the models to the machine which are a few GBs in size.

## How it Works

<img src="./images/document-to-podcast-diagram.png" width="1200" />


1. **Document Upload**
   Start by uploading a document in a supported format (e.g., PDF, .txt, or .docx).

2. **Document Pre-Processing**
   The uploaded document is processed to extract and clean the text. This involves:
   - Extracting readable text from the document.
   - Removing noise such as URLs, email addresses, and special characters to ensure the text is clean and structured.

3. **Script Generation**
   The cleaned text is passed to a language model to generate a podcast transcript in the form of a conversation between two speakers.
   - **Model Loading**: The system selects and loads a pre-trained LLM optimized for running locally, using the llama_cpp library. This enables the model to run efficiently on CPUs, making them more accessible and suitable for local setups.
   - **Customizable Prompt**: A user-defined "system prompt" guides the LLM in shaping the conversation, specifying tone, content, speaker interaction, and format.
   - **Output Transcript**: The model generates a podcast script in structured format, with each speaker's dialogue clearly labeled.
     Example output:
     ```json
     {
         "Speaker 1": "Welcome to the podcast on AI advancements.",
         "Speaker 2": "Thank you! So what's new this week for the latest AI trends?",
         "Speaker 1": "Where should I start.. Lots has been happening!",
         ...
     }
     ```
   This step ensures that the podcast script is engaging, relevant, and ready for audio conversion.

4. **Audio Generation**
  - The generated transcript is converted into audio using a Text-to-Speech (TTS) model.
  -	Each speaker is assigned a distinct voice.
	- The final output is saved as an audio file in formats like MP3 or WAV.

## Pre-requisites

- **System requirements**:
  - OS: Windows, macOS, or Linux
  - Python 3.10 or higher
  - Minimum RAM: 16 GB
  - Disk space: 32 GB minimum

- **Dependencies**:
  - Dependencies listed in `pyproject.toml`

## Troubleshooting

> When starting up the codespace, I get the message `Oh no, it looks like you are offline!`

If you are on Firefox and have Enhanced Tracking Protection `On`, try turning it `Off` for the codespace webpage.

> During the installation of the package, it fails with `ERROR: Failed building wheel for llama-cpp-python`

You are probably missing the `GNU Make` package. A quick way to solve it is run on your terminal `sudo apt install build-essential`

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
