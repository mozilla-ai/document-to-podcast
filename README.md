<p align="center"><img src="./images/Blueprints-logo.png" width="35%" alt="Project logo"/></p>

# Document-to-podcast: a Blueprint by Mozilla.ai for generating podcasts from documents using local AI

[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Docs](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/document-to-podcast/actions/workflows/lint.yaml/)

This blueprint demonstrate how you can use open-source models & tools to convert input documents into a podcast featuring two speakers.
It is designed to work on most local setups or with [GitHub Codespaces](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=888426876&skip_quickstart=true&machine=standardLinux32gb), meaning no external API calls or GPU access is required. This makes it more accessible and privacy-friendly by keeping everything local.

<img src="./images/document-to-podcast-diagram.png" width="1200" alt="document-to-podcast Diagram" />

### 👉 📖 For more detailed guidance on using this project, please visit our [Docs here](https://mozilla-ai.github.io/document-to-podcast/).


## Quick-start

Get started with Document-to-Podcast using one of the options below:

| HuggingFace Spaces  | GitHub Codespaces | Local Installation |
| ------------------- | ----------------- | ------------------ |
| [![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/mozilla-ai/document-to-podcast) | [![Try on Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=888426876&skip_quickstart=true&machine=standardLinux32gb) | `pip install document-to-podcast` |

## Troubleshooting

> When starting up the codespace, I get the message `Oh no, it looks like you are offline!`

If you are on Firefox and have Enhanced Tracking Protection `On`, try turning it `Off` for the codespace webpage.

> During the installation of the package, it fails with `ERROR: Failed building wheel for llama-cpp-python`

You are probably missing the `GNU Make` package. A quick way to solve it is run on your terminal `sudo apt install build-essential`

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.
