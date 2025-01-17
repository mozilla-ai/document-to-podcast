<p align="center"><img src="./images/Blueprints-logo.png" width="35%" alt="Project logo"/></p>

# Document-to-podcast: a Blueprint by Mozilla.ai for generating podcasts from documents using local AI

[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Docs](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/)

This blueprint demonstrate how you can use open-source models & tools to convert input documents into a podcast featuring two speakers.
It is designed to work on most local setups, meaning no external API calls or GPU access is required.
This makes it more accessible and privacy-friendly by keeping everything local.

<img src="./images/document-to-podcast-diagram.png" width="1200" alt="document-to-podcast Diagram" />

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


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.
