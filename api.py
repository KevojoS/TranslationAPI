from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

app = FastAPI()

@app.get("/translate")
def translate(prompt: str):
    messages = [
    {"role": "user", "content":  prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    outputs = model.generate(**inputs)
    output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    return {
        "translation": output,
    }