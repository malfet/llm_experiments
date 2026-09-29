import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.accelerator.current_accelerator()
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

prompt = '''def print_prime(n):
   """
   Print all primes between 1 and n
   """'''
inputs = tokenizer(prompt, return_tensors="pt").to(device)

start = time.perf_counter()
outputs = model.generate(**inputs, max_length=200)
elapsed = time.perf_counter() - start
new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
print(tokenizer.batch_decode(outputs)[0])
print(f"\n{new_tokens} tokens in {elapsed:.2f}s ({new_tokens / elapsed:.1f} tok/s) on {device}")
