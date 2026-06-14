using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.Gguf.Codecs;

/// <summary>One ggml quantization codec. Each implementation knows the canonical block layout for one <see cref="DType"/> and provides forward (dequantize) and reverse (quantize) operations. Forward is required for inference; reverse is required only for the offline <c>GgufQuantizer</c> writer.</summary>
public unsafe interface IGgufCodec
{
    /// <summary>The DType this codec handles. Must be a quantized type (<see cref="DType.IsQuantized"/>).</summary>
    DType DType { get; }

    /// <summary>Whether the reverse direction (quantize) is implemented. <c>false</c> for codecs we only need to read (e.g. i-quants where we consume llama.cpp-produced GGUFs but don't author them).</summary>
    bool SupportsQuantize { get; }

    /// <summary>Dequantizes <paramref name="elementCount"/> elements from the canonical packed block layout to F32. Caller guarantees <paramref name="src"/> has at least <see cref="DType.ComputeByteCount"/> bytes and <paramref name="dst"/> has at least <c>elementCount</c> floats.</summary>
    void DequantizeToF32(byte* src, float* dst, long elementCount);

    /// <summary>Optional: dequantize directly to F16. Default impl in <see cref="GgufCodecBase"/> goes via an F32 temp buffer; override for codecs where direct F16 is faster (e.g. Q8_0 where the FP16 scale stays packed).</summary>
    void DequantizeToF16(byte* src, Half* dst, long elementCount);

    /// <summary>Quantize F32 input to the canonical packed block layout. Throws <see cref="NotSupportedException"/> if <see cref="SupportsQuantize"/> is false.</summary>
    void QuantizeFromF32(float* src, byte* dst, long elementCount);
}
