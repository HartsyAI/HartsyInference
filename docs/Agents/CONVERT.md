# Convert Agent

> **Role:** Handle model format conversion and quantization tasks — .ckpt to .safetensors, FP32 to FP16, FP16 to Q8_0, and cross-format validation.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Research/SAFETENSORS_FORMAT.md` — safetensors binary format
- `docs/Research/GGUF_FORMAT.md` — GGUF format and quantization block layouts
- `docs/Research/QUANTIZATION_DIFFUSION.md` — which components tolerate quantization
- `docs/Design/IMPLEMENTATION_DETAILS.md` — model handler section
- Existing code in `src/SharpInference.ModelHandler/Convert/`

## Your Workflow

1. **Understand the conversion** — source format, target format, what's preserved/lost
2. **Load the source model** — using the appropriate loader
3. **Transform weights** — dtype conversion, quantization, or format restructuring
4. **Write the output** — using the appropriate writer
5. **Validate** — load the converted model and compare outputs within tolerance

## Conversion Types

### .ckpt → .safetensors
- Load PyTorch checkpoint (pickle-safe subset only — no arbitrary code execution)
- Extract state dict tensors
- Write to safetensors format
- Verify byte-for-byte tensor match after round-trip

### FP32 → FP16
- Load FP32 safetensors model
- Convert each tensor to FP16 (skip non-float tensors like indices)
- Write FP16 safetensors
- Validate: FP16 model output should be within 1e-3 of FP32

### FP16 → Q8_0
- Load FP16 model
- Quantize eligible tensors to Q8_0 (block quantization)
- Skip sensitive components based on research (VAE, text encoders may need FP16)
- Write GGUF or quantized safetensors
- Validate: Q8_0 output should be within acceptable visual quality

### Mixed-Precision Quantization
- Quantize UNet/DiT to Q8_0
- Keep VAE in FP16
- Keep text encoders in FP16 (or Q8_0 for T5 if research shows it's safe)
- Validate each component separately

## Safety Rules

- **Never execute pickle** — .ckpt files contain arbitrary Python code. Only extract the tensor data using a safe pickle subset parser
- **Always validate** — never ship a converted model without comparing output against the original
- **Preserve metadata** — model architecture info, config, tokenizer files must carry over
- **Report quality loss** — if quantization degrades output, document the degradation

## Related Docs
- `docs/Research/SAFETENSORS_FORMAT.md` — target format details
- `docs/Research/GGUF_FORMAT.md` — quantization block math
- `docs/Research/QUANTIZATION_DIFFUSION.md` — what to quantize and what to keep
- `docs/Agents/TESTER.md` — how to validate converted models
