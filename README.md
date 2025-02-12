<p align="center">
  <picture>
    <!-- When the user prefers dark mode, show the white logo -->
    <source media="(prefers-color-scheme: dark)" srcset="./images/Blueprint-logo-white.png">
    <!-- When the user prefers light mode, show the black logo -->
    <source media="(prefers-color-scheme: light)" srcset="./images/Blueprint-logo-black.png">
    <!-- Fallback: default to the black logo -->
    <img src="./images/Blueprint-logo-black.png" width="35%" alt="Project logo"/>
  </picture>
</p>

# Document-to-podcast: a Blueprint by Mozilla.ai for generating podcasts from documents using local AI

[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Docs](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/)

This blueprint demonstrate how you can use open-source models & tools to convert input documents into a podcast featuring two speakers.
It is designed to work on most local setups, meaning no external API calls or GPU access is required.
This makes it more accessible and privacy-friendly by keeping everything local.

<img src="./images/document-to-podcast-diagram.png" width="1200" alt="document-to-podcast Diagram" />

📘 To explore this project further and discover other Blueprints, visit the [**Blueprints Hub**](https://developer-hub.mozilla.ai/blueprints/create-your-own-tailored-podcast-using-your-documents).

## Example Results

- [Introducing Blueprints](https://blog.mozilla.ai/introducing-blueprints-customizable-ai-workflows-for-developers/)

https://github.com/user-attachments/assets/0487640b-a800-4c60-96ae-f1b93632a87b

- [Attention is All You Need](https://arxiv.org/pdf/1706.03762)

https://github.com/user-attachments/assets/0d5364e7-a57b-4976-8cb6-4ebf1cbbd37c

---

### 👉 📖 For more detailed guidance on using this project, please visit our [Docs](https://mozilla-ai.github.io/document-to-podcast/).
### 👉 🔨 Built with
- Python 3.10+ (use Python 3.12 for Apple M1/2/3 chips)
- [Llama-cpp](https://github.com/abetlen/llama-cpp-python)
- [Streamlit](https://streamlit.io/) (UI demo)

### 👉 🧠 Check the [Supported Models](https://mozilla-ai.github.io/document-to-podcast/customization/#supported-models).

## Quick-start

Get started right away using one of the options below:

| Google Colab | HuggingFace Spaces  | GitHub Codespaces |
| -------------| ------------------- | ----------------- |
| [![Try on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/document-to-podcast/blob/main/demo/notebook.ipynb) | [![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/mozilla-ai/document-to-podcast) | [![Try on Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=888426876&skip_quickstart=true&machine=standardLinux32gb) |

You can also install and use the blueprint locally:

### Command Line Interface

```bash
pip install document-to-podcast
```

```bash
document-to-podcast \
--input_file "example_data/Mozilla-Trustworthy_AI.pdf" \
--output_folder "example_data"
--text_to_text_model "Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf"
```

### Graphical Interface App

```bash
git clone https://github.com/mozilla-ai/document-to-podcast.git
cd document-to-podcast
pip install -e .
```

```bash
python -m streamlit run demo/app.py
```

## System requirements
  - OS: Windows, macOS, or Linux
  - Python 3.10+ / 3.12+ for Apple M chips
  - Minimum RAM: 8 GB
  - Disk space: 20 GB minimum

## Troubleshooting

If you are having issues / bugs, check our [Troubleshooting](https://mozilla-ai.github.io/document-to-podcast/getting-started/#troubleshooting) section, before opening a new issue.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.
