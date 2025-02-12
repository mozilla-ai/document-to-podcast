Get started with Document-to-Podcast using one of the options below:
---

## Setup options

=== "☁️ Google Colab (GPU)"

      The easiest way to play with the code on a GPU, for free.

      Click the button below to launch the project directly in Google Colab:

      <p align="center"><a href="https://colab.research.google.com/github/mozilla-ai/document-to-podcast/blob/main/demo/notebook.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a></p>

=== "☁️ GitHub Codespaces"

      Click the button below to launch the project directly in GitHub Codespaces:

      <p align="center"><a href="https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=888426876&skip_quickstart=true&machine=standardLinux32gb"><img src="https://github.com/codespaces/badge.svg" /></a></p>

      Once the Codespaces environment launches, inside the terminal, start the Streamlit demo by running:

      ```bash
      python -m streamlit run demo/app.py
      ```

=== "💻 Local Installation"

      You can install the project from Pypi:

      ```bash
      pip install document-to-podcast
      ```

      Check the [Command Line Interface](./cli.md) guide.

      ---

      Alternatively, you can clone and install it in editable mode:

      1. **Clone the Repository**

         ```bash
         git clone https://github.com/mozilla-ai/document-to-podcast.git
         cd document-to-podcast
         ```

      2. **Install the project and its Dependencies**

         ```bash
         pip install -e .
         ```

      3. **Run the Demo**

         ```bash
         python -m streamlit run demo/app.py
         ```


## Troubleshooting

> When starting up the codespace, I get the message `Oh no, it looks like you are offline!`

If you are on Firefox and have Enhanced Tracking Protection `On`, try turning it `Off` for the codespace webpage.

> During the installation of the package, it fails with `ERROR: Failed building wheel for llama-cpp-python`

You are probably missing the `GNU Make` package. A quick way to solve it is run on your terminal `sudo apt install build-essential`

> When running the Streamlit demo app I get: `RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_`

This seems to be an [issue](https://discuss.streamlit.io/t/message-error-about-torch/90886) with Streamlit, as of Feb 2025. If the app works fine, you can ignore it for now.
