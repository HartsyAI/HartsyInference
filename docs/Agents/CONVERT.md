# Convert Agent

> Handle model format conversion and quantization — .ckpt→.safetensors, FP32→FP16, FP16→Q8_0, cross-format validation.

## Extra Reading
- `docs/Research/SAFETENSORS_FORMAT.md`, `docs/Research/GGUF_FORMAT.md`, `docs/Research/QUANTIZATION_DIFFUSION.md`
- `docs/Design/IMPLEMENTATION_DETAILS.md`
- Existing code in `src/HartsyInference.ModelHandler/Convert/`

## Workflow
1. Understand conversion (source→target, preserved/lost)
2. Load source model
3. Transform weights (dtype, quantization, restructuring)
4. Write output
5. Validate: compare outputs within tolerance

## Conversion Types

| Conversion | Steps |
|---|---|
| .ckpt → .safetensors | Safe-pickle load → extract state dict → write safetensors. Byte-for-byte round-trip check. |
| FP32 → FP16 | Load → convert → write. Validate within 1e-3. |
| FP16 → Q8_0 | Load → quantize eligible tensors (skip VAE, text encoders if needed) → write GGUF or quantized safetensors. Validate visual quality. |
| Mixed | UNet/DiT Q8_0; VAE FP16; text encoders FP16 (or Q8_0 for T5 if safe). Validate per component. |

## Safety Rules
- **Never execute pickle** — use safe subset parser only
- **Always validate** — compare output against original before shipping
- **Preserve metadata** — architecture info, config, tokenizer files
- **Report quality loss** — document any degradation
