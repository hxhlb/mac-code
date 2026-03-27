"""
Gate Weight Diagnostic — Run this FIRST to determine if routing is random.
If gate weight is zeros, that's the bug (random routing → structured garbage).
"""
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

print("=" * 60)
print("  GATE WEIGHT DIAGNOSTIC")
print("=" * 60)

config = AutoConfig.from_pretrained('/workspace/qwen35-122b-a10b-4bit', trust_remote_code=True)
text_cfg = config.text_config if hasattr(config, 'text_config') else config

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(text_cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    for i in range(text_cfg.num_hidden_layers):
        model.model.layers[i].mlp.experts = torch.nn.ModuleList()

# Materialize as zeros
for name, param in list(model.named_parameters()):
    if param.device == torch.device("meta"):
        set_module_tensor_to_device(model, name, device="cpu",
            value=torch.zeros(param.shape, dtype=torch.bfloat16))

gate0 = model.model.layers[0].mlp.gate
print(f"\nBEFORE injection:")
print(f"  Gate class: {gate0.__class__.__name__}")
print(f"  Gate weight shape: {gate0.weight.shape}")
print(f"  Gate weight mean: {gate0.weight.float().mean():.6f}")
print(f"  Gate weight std:  {gate0.weight.float().std():.6f}")

# Check pinned file keys
with safe_open('/workspace/qwen35-122b-stream/pinned.safetensors', framework='pt', device='cpu') as f:
    gate_keys = [k for k in f.keys() if 'mlp.gate.' in k and 'layers.0.' in k and 'shared' not in k and 'expert' not in k]
    print(f"\nGate keys in pinned (layer 0):")
    for k in sorted(gate_keys):
        t = f.get_tensor(k)
        print(f"  {k}: {t.shape} {t.dtype}")

    # Model param names
    model_params = dict(model.named_parameters())
    model_gate_keys = [n for n in model_params if 'mlp.gate' in n and 'layers.0.' in n and 'shared' not in n]
    print(f"\nModel gate params (layer 0):")
    for n in sorted(model_gate_keys):
        print(f"  {n}: {model_params[n].shape}")

    # Key mapping test
    print(f"\nKEY MAPPING:")
    for k in sorted(gate_keys):
        mapped = k.replace("language_model.", "", 1) if k.startswith("language_model.") else k
        found = mapped in model_params
        print(f"  {k}")
        print(f"    -> {mapped} -> {'FOUND' if found else 'MISSING'}")

    # Try manual injection of gate weight
    print(f"\nMANUAL INJECTION TEST:")
    gw = [k for k in gate_keys if k.endswith('.weight')][0]
    gs = [k for k in gate_keys if k.endswith('.scales')][0]
    gb = [k for k in gate_keys if k.endswith('.biases')][0]
    w = f.get_tensor(gw)
    s = f.get_tensor(gs)
    b = f.get_tensor(gb)
    print(f"  Raw weight: {w.shape} {w.dtype}")
    print(f"  Scales: {s.shape} {s.dtype}")
    print(f"  Biases: {b.shape} {b.dtype}")

    # Dequantize
    ww = w.to(torch.int32)
    shifts = torch.arange(0, 32, 4)
    unpacked = (ww.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0xF
    in_f = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(w.shape[0], in_f).float()
    ng = in_f // 64
    unpacked = unpacked.reshape(w.shape[0], ng, 64)
    dq = unpacked * s.float().unsqueeze(-1) + b.float().unsqueeze(-1)
    dq = dq.reshape(w.shape[0], in_f).to(torch.bfloat16)
    print(f"  Dequantized: {dq.shape} {dq.dtype}")
    print(f"  Dequant mean: {dq.float().mean():.6f}, std: {dq.float().std():.6f}")
    print(f"  Model expects: {gate0.weight.shape}")
    print(f"  Shapes match: {dq.shape == gate0.weight.shape}")

    # Inject and verify
    target = gw.replace("language_model.", "", 1) if gw.startswith("language_model.") else gw
    print(f"\n  Injecting into: {target}")
    try:
        set_module_tensor_to_device(model, target, device="cpu", value=dq)
        print(f"  AFTER injection:")
        print(f"  Gate weight mean: {gate0.weight.float().mean():.6f}")
        print(f"  Gate weight std:  {gate0.weight.float().std():.6f}")
        if gate0.weight.float().std() > 0.0001:
            print(f"\n  >>> GATE WEIGHT INJECTED SUCCESSFULLY")
        else:
            print(f"\n  >>> GATE WEIGHT STILL ZEROS — INJECTION FAILED")
    except Exception as e:
        print(f"  INJECTION FAILED: {e}")

print("\n" + "=" * 60)
