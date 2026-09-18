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
    public void PrepareForBackend_WidensAQuantizedWeightThatIsNotAMatrix()
    {
        // Only a GEMM reads a packed weight. A quantized SD1.5 UNet's rank-4 conv kernels would otherwise load
        // happily and die in Conv2D, which has no dequant path at all — so the rank decides this, not the backend.
        Dictionary<string, Tensor> weights = new()
        {
            ["down_blocks.0.resnets.0.conv1.weight"] = Q8Weight(4).Reshape(new TensorShape(2, 2, 1, 32)),
            ["blocks.0.attn.to_q.weight"] = Q8Weight(2),
        };
        try
        {
            using QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareFor(weights, _ => true, "packed-quant backend");

            Assert.Equal(1, prepared.WidenedCount);
            Assert.Equal(DType.F16, weights["down_blocks.0.resnets.0.conv1.weight"].DType);
            Assert.Equal(new TensorShape(2, 2, 1, 32), weights["down_blocks.0.resnets.0.conv1.weight"].Shape);
            // The matrix the backend can read stays packed.
            Assert.Equal(DType.Q8_0, weights["blocks.0.attn.to_q.weight"].DType);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void PrepareForBackend_WidensACompanionBackedInt8WeightOnABackendWithoutAnInt8Path()
    {
        // int8_tensorwise arrives as plain I8 — the dtype does not report as quantized — so a policy that asked only
        // DType.IsQuantized left these untouched on CPU and Vulkan and let them reach an op that cannot read them.
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.to_q.weight"] = Int8Weight(4, 256, out Tensor rowScale),
        };
        try
        {
            using QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareFor(weights, _ => false, "dense-only backend");

            Assert.Equal(1, prepared.WidenedCount);
            Tensor widened = weights["blocks.0.attn.to_q.weight"];
            Assert.Equal(DType.F16, widened.DType);
            Assert.Null(widened.QuantInfo);
            Assert.Equal(new TensorShape(4, 256), widened.Shape);
        }
        finally
        {
            rowScale.Dispose();
            DisposeAll(weights);
        }
    }

    [Fact]
    public void PrepareForBackend_LeavesACompanionBackedInt8WeightPackedWhenTheBackendReadsIt()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.to_q.weight"] = Int8Weight(4, 256, out Tensor rowScale),
        };
        Tensor original = weights["blocks.0.attn.to_q.weight"];
        try
        {
            using QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareFor(weights, dtype => dtype == DType.I8, "int8 backend");

            Assert.Equal(0, prepared.WidenedCount);
            // Widening it would turn LTX 2.5's 21.5 GB DiT back into 42 GB, which is the whole reason for the format.
            Assert.Same(original, weights["blocks.0.attn.to_q.weight"]);
        }
        finally
        {
            rowScale.Dispose();
            DisposeAll(weights);
        }
    }

    /// <summary>A ComfyUI <c>int8_tensorwise</c> weight as the container hands it over: packed I8 with its per-row scale on QuantInfo.</summary>
    private static Tensor Int8Weight(int rows, int columns, out Tensor rowScale)
    {
        rowScale = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> scales = rowScale.AsSpan<float>();
        for (int i = 0; i < rows; i++) scales[i] = 0.01f * (i + 1);

        Tensor weight = new Tensor(new TensorShape(rows, columns), DType.I8);
        Span<sbyte> values = weight.AsSpan<sbyte>();
        for (int i = 0; i < values.Length; i++) values[i] = (sbyte)(i % 127 - 63);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = rowScale };
        return weight;
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
