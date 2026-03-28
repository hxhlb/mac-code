"""
THE DEFINITIVE TEST: Does Qwen3.5-122B-A10B work with from_pretrained?
Using OFFICIAL GPTQ-Int4 format — native HF, zero conversion risk.

Downloads from Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 (79 GB).
Runs ONE forward pass. Checks if "Paris" is the top prediction.
"""

import torch
import time

print("=" * 60)
print("  DEFINITIVE TEST: GPTQ 122B Unpatched")
print("=" * 60)

# Step 1: Download + load
print("\n[1] Loading Qwen3.5-122B-A10B-GPTQ-Int4...")
print("    This downloads 79 GB and loads with device_map=auto")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",
)
load_time = time.time() - t0
print(f"    Loaded in {load_time:.0f}s")

# Check device map
if hasattr(model, 'hf_device_map'):
    devices = set(model.hf_device_map.values())
    print(f"    Devices: {devices}")

vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
print(f"    VRAM: {vram:.2f} GB")

# Step 2: Tokenize
print("\n[2] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    trust_remote_code=True,
)

# Step 3: Single forward pass
print("\n[3] Running single forward pass...")
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
attention_mask = torch.ones_like(input_ids)

t0 = time.time()
with torch.no_grad():
    out = model(input_ids, attention_mask=attention_mask, use_cache=False)
elapsed = time.time() - t0

logits = out.logits if hasattr(out, 'logits') else out[0]
top10 = torch.topk(logits[0, -1].float(), 10)

print(f"    Forward pass: {elapsed:.1f}s")
print(f"    Top 10 predictions:")
for i, (val, idx) in enumerate(zip(top10.values, top10.indices)):
    print(f"      {i+1}. '{tokenizer.decode([idx.item()])}' (logit={val.item():.2f})")

paris_tokens = tokenizer.encode("Paris", add_special_tokens=False)
paris_in_top = any(idx.item() in paris_tokens for idx in top10.indices)

print(f"\n    'Paris' in top 10: {paris_in_top}")
if paris_in_top:
    print("    >>> MODEL WORKS!")
    print("    >>> The GPTQ model produces correct output.")
    print("    >>> Next: patch MoE forward with expert sniping.")
else:
    print("    >>> MODEL DOESN'T PREDICT PARIS")
    print("    >>> Check what it predicts — might be a synonym or different language.")

# Step 4: Quick generate
print("\n[4] Generating 10 tokens...")
t0 = time.time()
with torch.no_grad():
    gen_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
elapsed = time.time() - t0

new_tokens = gen_ids[0][input_ids.shape[1]:]
output = tokenizer.decode(new_tokens, skip_special_tokens=True)
tps = len(new_tokens) / elapsed if elapsed > 0 else 0

print(f"    Output: '{output}'")
print(f"    Speed: {tps:.2f} tok/s")
print(f"    Time: {elapsed:.1f}s")

print(f"\n{'='*60}")
print(f"  RESULT: {'SUCCESS' if paris_in_top else 'INVESTIGATE'}")
print(f"  Model: Qwen3.5-122B-A10B-GPTQ-Int4 (79 GB)")
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print(f"{'='*60}")
