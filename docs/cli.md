# Command Line Interface

Once you have [installed the blueprint](./getting-started.md), you can use it from the CLI.

You can either provide the path to a configuration file:

```bash
document-to-podcast --from_config "example_data/config.yaml"
```

Or provide values to the arguments directly:


```bash
document-to-podcast \
--input_file "example_data/Mozilla-Trustworthy_AI.pdf" \
--output_folder "example_data"
--text_to_text_model "Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf"
```

Note that you can also exit the podcast generation prematurely (before the whole podcast is created), by pressing Ctrl+C in the terminal. This will make the application stop the generation, but still save the result (script & audio) to disk up until that point.

---

::: document_to_podcast.cli.document_to_podcast

---

::: document_to_podcast.config.Config
::: document_to_podcast.config.Speaker
::: document_to_podcast.config.DEFAULT_PROMPT
::: document_to_podcast.config.DEFAULT_SPEAKERS
