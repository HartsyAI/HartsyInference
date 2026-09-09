# GGUF implementation constraints

Reader, writer, codecs, and key mappers live in
[ModelAssets/Gguf](../../src/HartsyInference.ModelAssets/Gguf/).
[GGUF_FORMAT.md](GGUF_FORMAT.md) preserves the upstream format; [quantizer usage](GGUF_QUANTIZER_USAGE.md)
describes writing. Current model support is in [status](../Checklists/MODEL_STATUS.md).

## Extending a format or architecture

Add the DType/block layout, GGUF type-id mapping, codec registration, and canonical-byte tests. Implement
quantization only when supported; decode support does not imply encode support. For a key mapper, register
it and test detection, fused tensors, scale companions, and shape/ownership conversions. A mapper alone
does not make a new architecture executable.

## Known traps

- GGUF magic is bytes GGUF (LE 0x46554747). Reader/writer round-trips alone once concealed a shared wrong constant.
- Quantized byte sizes use DType.ComputeByteCount, never ElementCount * SizeInBytes (which can be zero).
- Fused QKV/gate-up splits must preserve block alignment and scale metadata. Do not slice quantized buffers
  with F32 byte arithmetic.
- Q6_K subscale selection varies across 16-element halves; uniform-scale fixtures concealed a real bug.
- Relabeling a mmap tensor returns a borrowed view; an owned F32 copy is a different contract.
- GPU dequant/Linear support varies by format. Inspect the actual dispatch before estimating resident
  footprint: dequantized casts can dominate memory even when on-disk weights are small.
- Writer round-trip error on random data is not model perplexity or image quality, and does not establish
  bit-identical output to llama.cpp's quantizer. Validate real downstream output separately.

Historical Phase C/D/F completion logs and stale machine-specific OOM claims were removed; git preserves them.
