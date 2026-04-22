"""
Dumps T5-XXL encoder intermediate tensors for cross-runtime validation.
Saves per-layer outputs, relative position bias, and final embeddings.

Usage: python dump_t5_reference.py <model_dir> [output_dir]
  model_dir: path to T5 encoder safetensors directory (e.g., text_encoder_3/)
Requires: pip install transformers torch safetensors sentencepiece
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path


def tensor_stats(name: str, t: torch.Tensor) -> dict:
    """Compute stats for a tensor."""
    with torch.no_grad():
        flat = t.float().flatten()
        return {
            "name": name,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "abs_mean": float(flat.abs().mean()),
            "first_8": [float(x) for x in flat[:8]],
        }


def save_tensor(t: torch.Tensor, path: str):
    """Save tensor as raw float32 binary."""
    t.float().cpu().numpy().tofile(path)


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sd3_reference_tensors/t5"

    if model_dir is None:
        print("Usage: python dump_t5_reference.py <model_dir> [output_dir]")
        print("  model_dir: path to T5 encoder model directory (HuggingFace format)")
        print("  Example: python dump_t5_reference.py stabilityai/stable-diffusion-3-medium-diffusers")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    from transformers import T5EncoderModel, T5Tokenizer

    device = "cpu"
    dtype = torch.float32

    # Load T5 encoder
    print(f"Loading T5 encoder from {model_dir}...")
    # If model_dir points to the full SD3 model, use text_encoder_3 subfolder
    t5_path = model_dir
    if os.path.exists(os.path.join(model_dir, "text_encoder_3")):
        t5_path = os.path.join(model_dir, "text_encoder_3")

    tokenizer_path = model_dir
    if os.path.exists(os.path.join(model_dir, "tokenizer_3")):
        tokenizer_path = os.path.join(model_dir, "tokenizer_3")

    t5_model = T5EncoderModel.from_pretrained(t5_path, torch_dtype=dtype).to(device)
    t5_model.eval()
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, legacy=True)

    # Print model config for verification
    config = t5_model.config
    print(f"  d_model={config.d_model}, d_ff={config.d_ff}, d_kv={config.d_kv}")
    print(f"  num_heads={config.num_heads}, num_layers={config.num_layers}")
    print(f"  vocab_size={config.vocab_size}")
    print(f"  relative_attention_num_buckets={config.relative_attention_num_buckets}")
    print(f"  relative_attention_max_distance={config.relative_attention_max_distance}")

    stats = []
    stats.append({
        "name": "t5_config",
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "d_kv": config.d_kv,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "vocab_size": config.vocab_size,
    })

    # Test prompt — same as SD3 pipeline test
    test_text = "A photograph of an astronaut riding a horse"
    print(f"\nTokenizing: '{test_text}'")

    # Tokenize
    tokens = tokenizer(test_text, max_length=77, padding="max_length",
                       truncation=True, return_tensors="pt")
    input_ids = tokens["input_ids"]  # [1, 77]
    attention_mask = tokens["attention_mask"]  # [1, 77]

    token_ids_list = input_ids[0].tolist()
    mask_list = attention_mask[0].tolist()
    print(f"  Token IDs (first 20): {token_ids_list[:20]}")
    print(f"  Attention mask (first 20): {mask_list[:20]}")
    print(f"  Num real tokens: {sum(mask_list)}")

    stats.append({"name": "token_ids", "values": token_ids_list})
    stats.append({"name": "attention_mask", "values": mask_list})

    # Save token IDs for C# test
    np.array(token_ids_list, dtype=np.int32).tofile(os.path.join(output_dir, "token_ids.bin"))
    np.array(mask_list, dtype=np.int32).tofile(os.path.join(output_dir, "attention_mask.bin"))

    # --- Run encoder with hooks to capture per-layer outputs ---
    print("\nRunning T5 encoder with layer hooks...")

    layer_outputs = {}
    hooks = []

    # Hook each encoder block output
    for i, block in enumerate(t5_model.encoder.block):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # T5Block output is a tuple: (hidden_states, ...)
                hidden = output[0]
                layer_outputs[f"block_{layer_idx}"] = hidden.detach().clone()
            return hook_fn
        h = block.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Hook the embedding output
    def embed_hook(module, input, output):
        layer_outputs["embedding"] = output.detach().clone()
    hooks.append(t5_model.encoder.embed_tokens.register_forward_hook(embed_hook))

    # Hook the final layer norm
    def final_norm_hook(module, input, output):
        layer_outputs["final_norm"] = output.detach().clone()
    hooks.append(t5_model.encoder.final_layer_norm.register_forward_hook(final_norm_hook))

    # Hook relative position bias (first block, first attention layer)
    first_attn = t5_model.encoder.block[0].layer[0].SelfAttention
    def bias_hook(module, input, output):
        # output is (attn_output, position_bias, ...)
        if len(output) > 2 and output[2] is not None:
            layer_outputs["relative_position_bias"] = output[2].detach().clone()
        elif len(output) > 1 and output[1] is not None:
            layer_outputs["relative_position_bias"] = output[1].detach().clone()
    hooks.append(first_attn.register_forward_hook(bias_hook))

    # Forward pass
    with torch.no_grad():
        encoder_output = t5_model(input_ids=input_ids, attention_mask=attention_mask)

    # Remove hooks
    for h in hooks:
        h.remove()

    final_hidden = encoder_output.last_hidden_state  # [1, 77, 4096]
    print(f"\nFinal output shape: {final_hidden.shape}")

    # Save all layer outputs
    print("\nSaving layer outputs...")

    # Embedding
    if "embedding" in layer_outputs:
        emb = layer_outputs["embedding"]
        save_tensor(emb, os.path.join(output_dir, "embedding.bin"))
        stats.append(tensor_stats("embedding", emb))
        print(f"  embedding: mean={float(emb.mean()):.6f}, std={float(emb.std()):.6f}")

    # Relative position bias
    if "relative_position_bias" in layer_outputs:
        bias = layer_outputs["relative_position_bias"]
        save_tensor(bias, os.path.join(output_dir, "relative_position_bias.bin"))
        stats.append(tensor_stats("relative_position_bias", bias))
        print(f"  relative_position_bias: shape={list(bias.shape)}, mean={float(bias.mean()):.6f}")

    # Per-block outputs
    for i in range(config.num_layers):
        key = f"block_{i}"
        if key in layer_outputs:
            out = layer_outputs[key]
            save_tensor(out, os.path.join(output_dir, f"block_{i}_output.bin"))
            s = tensor_stats(f"block_{i}_output", out)
            stats.append(s)
            if i < 3 or i == config.num_layers - 1:
                print(f"  block_{i}: mean={s['mean']:.6f}, std={s['std']:.6f}")

    # Final norm output
    if "final_norm" in layer_outputs:
        fn = layer_outputs["final_norm"]
        save_tensor(fn, os.path.join(output_dir, "final_norm.bin"))
        stats.append(tensor_stats("final_norm", fn))
        print(f"  final_norm: mean={float(fn.mean()):.6f}, std={float(fn.std()):.6f}")

    # Final encoder output
    save_tensor(final_hidden, os.path.join(output_dir, "final_output.bin"))
    stats.append(tensor_stats("final_output", final_hidden))
    print(f"  final_output: mean={float(final_hidden.mean()):.6f}, std={float(final_hidden.std()):.6f}")

    # --- Also dump relative position bias computation details ---
    print("\nDumping relative position bias details...")

    # Compute bias table lookup
    bias_table = t5_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
    save_tensor(bias_table, os.path.join(output_dir, "bias_table.bin"))
    stats.append(tensor_stats("bias_table", bias_table))
    print(f"  bias_table shape: {list(bias_table.shape)}")  # Should be [32, 64]

    # Save stats
    stats_path = os.path.join(output_dir, "t5_reference_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAll stats saved to {stats_path}")
    print(f"All tensors saved to {output_dir}")
    print(f"Total stats entries: {len(stats)}")


if __name__ == "__main__":
    main()
