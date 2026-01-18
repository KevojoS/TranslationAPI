# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [
    {"role": "user", "content": "Translate this sentence to informal Chinese: I love you."},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
)

outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))