from pathlib import Path

from llama_cpp import Llama
from opennotebookllm.preprocessing import DATA_LOADERS, DATA_CLEANERS

llm = Llama.from_pretrained(
	repo_id="allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
	filename="olmoe-1b-7b-0924-instruct-q8_0.gguf",
	n_ctx=2048
)
INPUT = "example_data/introducing-mozilla-ai-investing-in-trustworthy-ai.html"
print("Loading raw text")
loader = DATA_LOADERS[Path(INPUT).suffix]
raw_text = loader(INPUT)

print("Cleaning text")
cleaner = DATA_CLEANERS[Path(INPUT).suffix]
clean_text = cleaner(raw_text)
print(clean_text[:2000])

print("Inference")
response = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "Convert this into a podcast transcript using SPEAKER1 and SPEAKER2: {}".format(clean_text[:2000])
		}
	]
)
print(response)
