# 🎨 **Customization Guide**

The Document-to-Podcast Blueprint is designed to be flexible and adaptable to your specific needs.
This guide outlines the key parameters you can customize and explains how to make these changes depending on whether you’re running the application via app.py or the CLI pipeline.

## 🖋️ **Key Parameters for Customization**

- **`input_file`**: The input file specifies the document to be processed. Supports the following formats: `pdf`, `html`, `txt`, `docx`, `md`.

- **`text_to_text_model`**: The language model used to generate the podcast script. Note: The model parameter must be in GGFUF format, for example: `Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf`.

- **`text_to_text_prompt`**: Defines the tone, structure, and instructions for generating the podcast script. This prompt is crucial for tailoring the conversation style to your project.

- **`text_to_speech_model`**: Specifies the model used for text-to-speech conversion. You can change this to achieve the desired voice style or improve performance. Check `config.py` to choose from supported models.

- **`speakers`**: Defines the podcast participants, including their names, roles, descriptions, and voice profiles. Customize this to create engaging personas and voices for your podcast.

## 🖥️ **Customizing When Running via `app.py`**

If you’re running the application using `app.py`, you can customize these parameters in the **`src/config.py`** file. This centralized configuration file simplifies the customization process.

Running app.py:
```python
python -m streamlit run demo/app.py
```

### Steps to Customize
1. Open the `config.py` file.
2. Locate the relevant parameter you want to change (e.g., `text_to_text_model`, `speakers`).
3. Update the value according to your needs.

#### Example: Updating the Prompt
In `config.py`, modify the `text_to_text_prompt` parameter:

```python
DEFAULT_PROMPT = """
You are a podcast scriptwriter generating engaging and humorous conversations in JSON format.
The script features the following speakers:
{SPEAKERS}
Instructions:
- Use a casual and fun tone.
- Include jokes and lighthearted banter.
- Format output as a JSON conversation.
  {
    "Speaker 1": "Well we a have a hilarious podcast in store for you today...",
    "Speaker 2": "I can't wait, I had the weirdest week - let me tell you all about it...",
"""
```

## ⌨️ **Customizing When Running via the CLI**

If you’re running the pipeline from the command line, you can customize the parameters by modifying the **`example_data/config.yaml`** file.

Running in the CLI:
```bash
document-to-podcast --from_config example_data/config.yaml
```

### Steps to Customize
1. Open the `config.yaml` file.
2. Locate the parameter you want to adjust.
3. Update the value and save the file.

#### Example: Changing the Text-to-Text Model
In `config.yaml`, modify the `text_to_text_model` entry:

```yaml
text_to_text_model: "Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q8_0.gguf"
```

## ✏️ **Customization Examples**

Looking for inspiration? Check out these examples of how others have customized the Document-to-Podcast Blueprint for their unique needs:

- **[Radio Drama Generator](https://github.com/stefanfrench/radio-drama-generator)**: A creative adaptation that generates radio dramas by customizing ng the Blueprint parameters.
- **[Readme-to-Podcast](https://github.com/alexmeckes/readme-to-podcast)**: This project transforms GitHub README files into podcast-style audio, showcasing the Blueprint’s ability to handle diverse text inputs.
- **[Multilingual Podcast](https://github.com/Kostis-S-Z/document-to-podcast/)**: A repo that showcases how to use this package in other languages, like Hindi, Polish, Korean and many more.

## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!
