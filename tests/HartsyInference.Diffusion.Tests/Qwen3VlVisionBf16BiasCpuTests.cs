using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Qwen3-VL vision tower from a tiny int8 checkpoint whose biases are BF16, run on the CPU backend. The int8 matrices are
/// widened by <see cref="QuantizedWeightPolicy"/> as the Fibo 2.0 encoder loader does; the BF16 biases must then give the same output as
/// the same biases in F32.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class Qwen3VlVisionBf16BiasCpuTests
{
    private static readonly Qwen3VlVisionConfig Tiny = new()
    {
        Depth = 1,
        HiddenSize = 8,
        NumHeads = 2,
        IntermediateSize = 16,
        InChannels = 3,
        PatchSize = 2,
        SpatialMergeSize = 2,
        TemporalPatchSize = 1,
        OutHiddenSize = 6,
        NumPositionEmbeddings = 4,
        DeepstackVisualIndexes = [],
    };

    [Fact]
    public void ForwardOnCpu_WithInt8WidenedMatricesAndBf16Biases_MatchesTheF32Biases()
    {
        List<Tensor> owned = [];
        try
        {
            // One checkpoint, built once: the BF16 biases, and the same tensors with every bias widened to F32 by hand. BF16 to F32 is
            // exact and the bias values are multiples of 1/4, so the two forwards see equal values.
            (Dictionary<string, Tensor> bf16Biases, QuantizedWeightPolicy.PreparedWeights widened) = Checkpoint(owned);
            using (widened)
            {
                Dictionary<string, Tensor> f32Biases = new(bf16Biases, StringComparer.Ordinal);
                foreach ((string key, Tensor tensor) in bf16Biases)
                {
                    if (tensor.DType != DType.BF16) continue;
                    Tensor f32 = Widen(tensor);
                    owned.Add(f32);
                    f32Biases[key] = f32;
                }

                using IBackend backend = new CpuBackend();
                using Tensor pixels = Pixels();
                float[] expected = Run(backend, f32Biases, pixels);
                float[] actual = Run(backend, bf16Biases, pixels); // threw "the CPU backend is F32-only" before LinearTransB widened the bias

                Assert.Equal(expected, actual);
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
            }
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }

    /// <summary>Runs the vision tower on one 2×2 patch grid and returns the merged tokens.</summary>
    private static float[] Run(IBackend backend, Dictionary<string, Tensor> weights, Tensor pixels)
    {
        using Qwen3VlVisionEncoder encoder = new(Tiny);
        encoder.LoadWeights(weights);
        Qwen3VlVisionEncoder.VisionOutput output = encoder.Forward(backend, pixels, 1, 2, 2);
        try
        {
            return output.MergedTokens.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            output.MergedTokens.Dispose();
            foreach (Tensor tap in output.DeepstackFeatures) tap.Dispose();
        }
    }

    /// <summary>A tiny vision checkpoint. The matrices are int8 with one scale per row, widened to BF16 by the policy; every bias is BF16;
    /// the norm gains are F32. <paramref name="owned"/> collects the tensors this disposes itself; the widened matrices belong to the
    /// returned handle.</summary>
    private static (Dictionary<string, Tensor>, QuantizedWeightPolicy.PreparedWeights) Checkpoint(List<Tensor> owned)
    {
        Random random = new(7);
        int h = Tiny.HiddenSize, i = Tiny.IntermediateSize, merged = h * Tiny.SpatialMergeSize * Tiny.SpatialMergeSize;
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        void Add(string key, Tensor t)
        {
            w[key] = t;
            owned.Add(t);
        }

        Add("patch_embed.proj.weight", Dense(random, new TensorShape([h, Tiny.InChannels, 1, Tiny.PatchSize, Tiny.PatchSize])));
        Add("patch_embed.proj.bias", Bias(random, h));
        Add("pos_embed.weight", Dense(random, new TensorShape(Tiny.NumPositionEmbeddings, h)));
        Add("blocks.0.norm1.weight", Dense(random, new TensorShape(h)));
        Add("blocks.0.norm1.bias", Dense(random, new TensorShape(h)));
        Add("blocks.0.norm2.weight", Dense(random, new TensorShape(h)));
        Add("blocks.0.norm2.bias", Dense(random, new TensorShape(h)));
        Add("blocks.0.attn.qkv.weight", Int8(random, 3 * h, h));
        Add("blocks.0.attn.qkv.bias", Bias(random, 3 * h));
        Add("blocks.0.attn.proj.weight", Int8(random, h, h));
        Add("blocks.0.attn.proj.bias", Bias(random, h));
        Add("blocks.0.mlp.linear_fc1.weight", Int8(random, i, h));
        Add("blocks.0.mlp.linear_fc1.bias", Bias(random, i));
        Add("blocks.0.mlp.linear_fc2.weight", Int8(random, h, i));
        Add("blocks.0.mlp.linear_fc2.bias", Bias(random, h));
        Add("merger.norm.weight", Dense(random, new TensorShape(h)));
        Add("merger.norm.bias", Dense(random, new TensorShape(h)));
        Add("merger.linear_fc1.weight", Int8(random, merged, merged));
        Add("merger.linear_fc1.bias", Bias(random, merged));
        Add("merger.linear_fc2.weight", Int8(random, Tiny.OutHiddenSize, merged));
        Add("merger.linear_fc2.bias", Bias(random, Tiny.OutHiddenSize));

        // The packed int8 matrices become BF16 on the host, as the Fibo 2.0 encoder loader does for a CPU consumer.
        QuantizedWeightPolicy.PreparedWeights widened = QuantizedWeightPolicy.PrepareForBackend(w, new CpuBackend(), DType.BF16);
        Assert.Equal(6, widened.WidenedCount); // qkv, proj, fc1, fc2 of the block, and the two merger matrices
        return (w, widened);
    }

    private static Tensor Pixels()
    {
        Tensor t = new(new TensorShape(Tiny.NumPositionEmbeddings, Tiny.PatchEmbedInDim), DType.F32);
        Random random = new(11);
        foreach (ref float value in t.AsSpan<float>())
            value = (float)(random.NextDouble() * 2 - 1);
        return t;
    }

    /// <summary>An int8 matrix with a per-row F32 scale; the codes are seeded.</summary>
    private static Tensor Int8(Random random, int rows, int cols)
    {
        Tensor codes = new(new TensorShape(rows, cols), DType.I8);
        foreach (ref sbyte code in codes.AsSpan<sbyte>())
            code = (sbyte)random.Next(-100, 101);
        Tensor scale = new(new TensorShape(rows), DType.F32);
        for (int r = 0; r < rows; r++)
            scale.AsSpan<float>()[r] = random.Next(1, 9) / 64f;
        codes.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale };
        return codes;
    }

    private static Tensor Dense(Random random, TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        foreach (ref float value in t.AsSpan<float>())
            value = (float)(random.NextDouble() * 0.5 - 0.25);
        return t;
    }

    /// <summary>A BF16 bias whose values are multiples of 1/4, which BF16 holds exactly.</summary>
    private static Tensor Bias(Random random, int n)
    {
        Tensor t = new(new TensorShape(n), DType.BF16);
        for (int k = 0; k < n; k++)
            t.AsSpan<ushort>()[k] = TensorCasts.F32ToBf16Bits(random.Next(-8, 9) / 4f);
        return t;
    }

    private static Tensor Widen(Tensor bf16)
    {
        Tensor f32 = new(bf16.Shape, DType.F32);
        ReadOnlySpan<ushort> bits = bf16.AsReadOnlySpan<ushort>();
        Span<float> values = f32.AsSpan<float>();
        for (int k = 0; k < bits.Length; k++)
            values[k] = BitConverter.UInt32BitsToSingle((uint)bits[k] << 16);
        return f32;
    }
}
