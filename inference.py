from llama_cpp import Llama
from opennotebookllm.preprocessing import DATA_LOADERS, DATA_CLEANERS

llm = Llama.from_pretrained(
	repo_id="allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
	filename="olmoe-1b-7b-0924-instruct-q8_0.gguf",
)

print("Loading raw text")
loader = DATA_LOADERS[".html"]
raw_text = loader("example_data/introducing-mozilla-ai-investing-in-trustworthy-ai.html")


print("Cleaning text")
cleaner = DATA_CLEANERS[".html"]
clean_text = cleaner(raw_text)

print("Inference")
response = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "Generate a podcast from this information: {}".format(clean_text)
		}
	]
)
print(response)