using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the load-time reconciliation between a checkpoint's quantized dtypes and the backend's packed-weight
/// kernels. Without it a Q2_K or IQ4_NL diffusion GGUF — both routinely published — loads happily and then dies inside
/// the first <c>Linear</c>, minutes into a generation, with a stack trace that names a kernel rather than a file.</summary>
public sealed unsafe class QuantizedWeightPolicyTests
{
    /// <summary>A Q8_0 tensor of <paramref name="blocks"/> blocks: each is an F16 scale followed by 32 int8 codes.</summary>
    private static Tensor Q8Weight(int blocks)
    {
        Tensor tensor = new Tensor(new TensorShape(blocks, 32), DType.Q8_0);
        Span<byte> bytes = tensor.AsSpan<byte>();
        for (int block = 0; block < blocks; block++)
        {
            int offset = block * 34;
            BitConverter.GetBytes((Half)0.5f).CopyTo(bytes[offset..]);
            for (int i = 0; i < 32; i++) bytes[offset + 2 + i] = (byte)(sbyte)(i - 16);
        }
        return tensor;
    }

    private static Dictionary<string, Tensor> Weights() => new()
    {
        ["blocks.0.attn.to_q.weight"] = Q8Weight(2),
        ["blocks.0.norm.weight"] = new Tensor(new TensorShape(8), DType.F32),
    };

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values) tensor.Dispose();
    }

    [Fact]
    public void PrepareForBackend_LeavesAWeightPackedWhenTheBackendCanReadIt()
    {
        Dictionary<string, Tensor> weights = Weights();
        Tensor original = weights["blocks.0.attn.to_q.weight"];
        try
        {
            using QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareFor(weights, dtype => dtype == DType.Q8_0, "packed-quant backend");

            Assert.Equal(0, prepared.WidenedCount);
            // Widening a format the backend reads natively would quadruple a 20 GB checkpoint for nothing.
            Assert.Same(original, weights["blocks.0.attn.to_q.weight"]);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void PrepareForBackend_WidensAWeightTheBackendHasNoKernelFor()
    {
        Dictionary<string, Tensor> weights = Weights();
        try
        {
            using QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareFor(weights, _ => false, "dense-only backend");

            Assert.Equal(1, prepared.WidenedCount);
            Tensor widened = weights["blocks.0.attn.to_q.weight"];
            Assert.Equal(DType.F16, widened.DType);
            Assert.Equal(new TensorShape(2, 32), widened.Shape);
            // 0.5 * (0 - 16) for the first code; a widened weight that lost its scale is the whole risk here.
            Assert.Equal(-8f, (float)widened.AsReadOnlySpan<Half>()[0], 3);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void PrepareForBackend_NeverTouchesADenseWeight()
    {
        Dictionary<string, Tensor> weights = Weights();
        Tensor norm = weights["blocks.0.norm.weight"];
        try
        {
            using QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareFor(weights, _ => false, "dense-only backend");
            Assert.Same(norm, weights["blocks.0.norm.weight"]);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void PreparedWeights_DisposesOnlyWhatItCreated()
    {
        Dictionary<string, Tensor> weights = Weights();
        try
        {
            QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareFor(weights, _ => false, "dense-only backend");
            Tensor widened = weights["blocks.0.attn.to_q.weight"];
            prepared.Dispose();
            prepared.Dispose();

            Assert.Throws<ObjectDisposedException>(() => _ = widened.DataPointer);
            // The originals borrow the checkpoint's mmap and are the loader's to free, not this handle's.
            _ = weights["blocks.0.norm.weight"].DataPointer;
        }
        finally
        {
            DisposeAll(weights);
        }
    }
}
