using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.LLM.Tests;

public sealed class LlamaStyleEncoderTests
{
    private const int Hidden = 16, Layers = 2, QueryHeads = 2, KvHeads = 1, HeadDim = 8, Intermediate = 24, Vocab = 32;
    private const float Tolerance = 1e-4f;
    private static readonly int[] Taps = [1, Layers];

    [Fact]
    public void FlashSoftcapCapsLogitsBeforeSoftmax()
    {
        IBackend backend = new CpuBackend();
        const float cap = 50f;
        using Tensor q = FromArray(new TensorShape(1, 1, 1, 1), 1f);
        using Tensor k = FromArray(new TensorShape(1, 1, 2, 1), 100f, 90f);
        using Tensor v = FromArray(new TensorShape(1, 1, 2, 1), 1f, 0f);
        using Tensor capped = new(new TensorShape(1, 1, 1, 1), DType.F32);
        using Tensor uncapped = new(new TensorShape(1, 1, 1, 1), DType.F32);
        backend.FlashAttention(capped, q, k, v, kvLen: 2, kvGroup: 1, causal: false, qOffset: 0, scale: 1f, softcap: cap);
        backend.FlashAttention(uncapped, q, k, v, kvLen: 2, kvGroup: 1, causal: false, qOffset: 0, scale: 1f, softcap: 0f);

        // 100 and 90 cap to 50·tanh(2) and 50·tanh(1.8); the first key's weight is the sigmoid of their gap.
        float c1 = cap * MathF.Tanh(100f / cap);
        float c2 = cap * MathF.Tanh(90f / cap);
        float expected = MathF.Exp(c1) / (MathF.Exp(c1) + MathF.Exp(c2));
        Assert.InRange(capped.AsSpan<float>()[0], expected - Tolerance, expected + Tolerance);
        Assert.True(uncapped.AsSpan<float>()[0] > 0.9999f);
    }

    [Fact]
    public void RightPaddedRowsMatchUnpaddedEncodes()
    {
        IBackend backend = new CpuBackend();
        using LlamaStyleEncoder encoder = BuildEncoder(TinyConfig(gemmaLike: true), seed: 3);
        int[] rowA = [5, 9, 12, 0, 0];
        int[] rowB = [7, 1, 4, 20, 13];
        using Tensor padded = encoder.EncodeMultiLayerRightPadded(backend, [rowA, rowB], Taps, [3, 5]);
        using Tensor unpaddedA = encoder.EncodeMultiLayer(backend, [[5, 9, 12]], Taps);
        using Tensor paddedA = encoder.EncodeMultiLayer(backend, [rowA], Taps, false, TextEncoderPadding.Transformers(3));
        using Tensor unpaddedB = encoder.EncodeMultiLayer(backend, [rowB], Taps);

        // Valid tokens match the unpadded encode; padding rows match the single-prompt padded path, which masks the same keys.
        AssertClose(Slice(unpaddedA, 0, 3), Slice(padded, 0, 3));
        AssertClose(Slice(paddedA, 0, 5), Slice(padded, 0, 5));
        AssertClose(Slice(unpaddedB, 0, 5), Slice(padded, 1, 5));
    }

    [Fact]
    public void NormalizedFinalOutputMatchesManualFinalNorm()
    {
        IBackend backend = new CpuBackend();
        LlamaStyleEncoderConfig config = TinyConfig(gemmaLike: true);
        Dictionary<string, Tensor> weights = TinyWeights(config, seed: 4);
        // LoadWeights folds the Gemma (1 + w) offset into the tensors in place, so capture the folded scale first.
        float[] foldedFinalNorm = Values(weights["norm.weight"]).Select(x => x + 1f).ToArray();
        using LlamaStyleEncoder encoder = new(config);
        encoder.LoadWeights(weights);
        int[][] batch = [[5, 9, 12, 0, 0], [7, 1, 4, 20, 13]];
        int[] validLengths = [3, 5];

        using Tensor raw = encoder.EncodeMultiLayerRightPadded(backend, batch, [Layers], validLengths);
        using Tensor normalized = encoder.EncodeMultiLayerRightPadded(backend, batch, [Layers], validLengths, normalizeFinalOutput: true);
        using Tensor scale = FromArray(new TensorShape(Hidden), foldedFinalNorm);
        using Tensor expected = new(raw.Shape, DType.F32);
        backend.RmsNorm(expected, raw, scale, config.RmsNormEps);

        AssertClose(Values(expected), Values(normalized));
        Assert.NotEqual(Values(raw)[0], Values(normalized)[0]);

        // With an earlier tap packed alongside, only the final channel block is normalized.
        using Tensor earlier = encoder.EncodeMultiLayerRightPadded(backend, batch, [1], validLengths);
        using Tensor packed = encoder.EncodeMultiLayerRightPadded(backend, batch, [1, Layers], validLengths, normalizeFinalOutput: true);
        AssertClose(Values(earlier), Channels(packed, 0, Hidden));
        AssertClose(Values(normalized), Channels(packed, Hidden, Hidden));
    }

    [Fact]
    public void NormalizedOutputRequiresTheFinalLayerLast()
    {
        IBackend backend = new CpuBackend();
        using LlamaStyleEncoder encoder = BuildEncoder(TinyConfig(gemmaLike: true), seed: 6);
        int[][] batch = [[5, 9, 12, 0, 0], [7, 1, 4, 20, 13]];
        Assert.Throws<ArgumentException>(() => { encoder.EncodeMultiLayerRightPadded(backend, batch, [1], [3, 5], normalizeFinalOutput: true); });
    }

    [Fact]
    public void SoftcapPathMatchesDenseAttentionForLargeCap()
    {
        IBackend backend = new CpuBackend();
        int[][] batch = [[5, 9, 12, 3, 17], [7, 1, 4, 20, 13]];
        int[] validLengths = [3, 3];
        using LlamaStyleEncoder dense = BuildEncoder(TinyConfig(gemmaLike: true, softcap: 0f), seed: 5);
        using LlamaStyleEncoder flash = BuildEncoder(TinyConfig(gemmaLike: true, softcap: 1e6f), seed: 5);
        using Tensor expected = dense.EncodeMultiLayerRightPadded(backend, batch, Taps, validLengths);
        using Tensor actual = flash.EncodeMultiLayerRightPadded(backend, batch, Taps, validLengths);

        AssertClose(Values(expected), Values(actual));
    }

    /// <summary>CUDA's fused kernel refuses causal rows past its key length, so right-padded soft-capped rows take a
    /// separate non-causal call over the real prefix; the result must match the CPU encode on every row.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void SoftcapRightPaddedRowsMatchCpuOnCuda()
    {
        if (!HartsyInference.Cuda.CudaContext.IsAvailable()) return;
        int[][] batch = [[5, 9, 12, 0, 0]];
        int[] validLengths = [3];
        using LlamaStyleEncoder cpuEncoder = BuildEncoder(TinyConfig(gemmaLike: true, softcap: 50f), seed: 9);
        using LlamaStyleEncoder cudaEncoder = BuildEncoder(TinyConfig(gemmaLike: true, softcap: 50f), seed: 9);
        using Tensor expected = cpuEncoder.EncodeMultiLayerRightPadded(new CpuBackend(), batch, [Layers], validLengths, normalizeFinalOutput: true);
        using HartsyInference.Cuda.CudaBackend cuda = new(0, CudaPtxDirectory.Resolve());
        using Tensor actual = cudaEncoder.EncodeMultiLayerRightPadded(cuda, batch, [Layers], validLengths, normalizeFinalOutput: true);
        float[] e = Values(expected);
        float[] a = actual.CastTo(DType.F32).AsSpan<float>().ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) <= 2e-3f, $"element {i}: cpu {e[i]} cuda {a[i]}");
    }

    [Fact]
    public void SoftcapChangesEncoderOutput()
    {
        IBackend backend = new CpuBackend();
        int[][] tokens = [[5, 9, 12, 3, 17]];
        using LlamaStyleEncoder dense = BuildEncoder(TinyConfig(gemmaLike: true, softcap: 0f), seed: 7);
        using LlamaStyleEncoder capped = BuildEncoder(TinyConfig(gemmaLike: true, softcap: 0.01f), seed: 7);
        using Tensor expected = dense.EncodeMultiLayer(backend, tokens, Taps);
        using Tensor actual = capped.EncodeMultiLayer(backend, tokens, Taps);

        float[] e = Values(expected);
        float[] a = Values(actual);
        float maxDiff = 0f;
        for (int i = 0; i < e.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(e[i] - a[i]));
        Assert.True(maxDiff > 1e-3f, $"Soft-cap left the output unchanged (max diff {maxDiff}).");
    }

    [Fact]
    public void SoftcapRejectsRowsWithDifferentValidLengths()
    {
        IBackend backend = new CpuBackend();
        using LlamaStyleEncoder encoder = BuildEncoder(TinyConfig(gemmaLike: true, softcap: 50f), seed: 8);
        int[][] batch = [[5, 9, 12, 0, 0], [7, 1, 4, 20, 13]];
        Assert.Throws<NotSupportedException>(() => { encoder.EncodeMultiLayerRightPadded(backend, batch, Taps, [3, 5]); });
    }

    [Fact]
    public void OnlyGemma2PresetEnablesSoftcap()
    {
        List<(string Name, LlamaStyleEncoderConfig Config)> presets = [];
        foreach (PropertyInfo property in typeof(LlamaStyleEncoderConfig).GetProperties(BindingFlags.Public | BindingFlags.Static))
        {
            if (property.PropertyType == typeof(LlamaStyleEncoderConfig))
                presets.Add((property.Name, (LlamaStyleEncoderConfig)property.GetValue(null)!));
        }

        Assert.Contains(presets, preset => preset.Name == nameof(LlamaStyleEncoderConfig.Gemma2_2B));
        foreach ((string name, LlamaStyleEncoderConfig config) in presets)
        {
            float expected = name == nameof(LlamaStyleEncoderConfig.Gemma2_2B) ? 50f : 0f;
            Assert.True(config.AttentionLogitSoftcap == expected, $"{name} soft-cap is {config.AttentionLogitSoftcap}, expected {expected}.");
        }
    }

    private static LlamaStyleEncoderConfig TinyConfig(bool gemmaLike = false, float softcap = 0f) => new()
    {
        HiddenSize = Hidden,
        NumLayers = Layers,
        NumQueryHeads = QueryHeads,
        NumKvHeads = KvHeads,
        HeadDim = HeadDim,
        IntermediateSize = Intermediate,
        VocabSize = Vocab,
        MaxPositionEmbeddings = 64,
        QkHeadNorm = false,
        AttentionBias = false,
        Activation = gemmaLike ? MlpActivation.GeluTanh : MlpActivation.Silu,
        RmsNormScalePlusOne = gemmaLike,
        HasFfnSandwichNorms = gemmaLike,
        EmbeddingScale = gemmaLike ? 4f : 1f,
        AttentionLogitSoftcap = softcap,
    };

    private static LlamaStyleEncoder BuildEncoder(LlamaStyleEncoderConfig config, int seed)
    {
        LlamaStyleEncoder encoder = new(config);
        encoder.LoadWeights(TinyWeights(config, seed));
        return encoder;
    }

    /// <summary>Random F32 weights keyed as a bare Gemma/Llama checkpoint names them.</summary>
    private static Dictionary<string, Tensor> TinyWeights(LlamaStyleEncoderConfig config, int seed)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        float normOffset = config.RmsNormScalePlusOne ? 0f : 1f;
        void Add(string key, float offset, float spread, params long[] dims)
        {
            Tensor tensor = new(new TensorShape(dims), DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = offset + (float)(rng.NextDouble() * 2 - 1) * spread;
            weights[key] = tensor;
        }

        int h = config.HiddenSize;
        Add("embed_tokens.weight", 0f, 0.5f, config.VocabSize, h);
        Add("norm.weight", normOffset, 0.2f, h);
        for (int i = 0; i < config.NumLayers; i++)
        {
            string p = $"layers.{i}";
            Add($"{p}.input_layernorm.weight", normOffset, 0.2f, h);
            Add($"{p}.post_attention_layernorm.weight", normOffset, 0.2f, h);
            if (config.HasFfnSandwichNorms)
            {
                Add($"{p}.pre_feedforward_layernorm.weight", normOffset, 0.2f, h);
                Add($"{p}.post_feedforward_layernorm.weight", normOffset, 0.2f, h);
            }
            Add($"{p}.self_attn.q_proj.weight", 0f, 0.25f, config.QDim, h);
            Add($"{p}.self_attn.k_proj.weight", 0f, 0.25f, config.KvDim, h);
            Add($"{p}.self_attn.v_proj.weight", 0f, 0.25f, config.KvDim, h);
            Add($"{p}.self_attn.o_proj.weight", 0f, 0.25f, h, config.QDim);
            Add($"{p}.mlp.gate_proj.weight", 0f, 0.25f, config.IntermediateSize, h);
            Add($"{p}.mlp.up_proj.weight", 0f, 0.25f, config.IntermediateSize, h);
            Add($"{p}.mlp.down_proj.weight", 0f, 0.25f, h, config.IntermediateSize);
        }
        return weights;
    }

    private static Tensor FromArray(TensorShape shape, params float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] Values(Tensor tensor) => tensor.AsSpan<float>().ToArray();

    /// <summary>The first <paramref name="rows"/> tokens of batch row <paramref name="batch"/> from a [B, S, C] tensor.</summary>
    private static float[] Slice(Tensor tensor, int batch, int rows)
    {
        int seqLen = (int)tensor.Shape[1];
        int channels = (int)tensor.Shape[2];
        return tensor.AsSpan<float>().Slice(batch * seqLen * channels, rows * channels).ToArray();
    }

    /// <summary>The <paramref name="count"/> channels from <paramref name="start"/> of every token in a [B, S, C] tensor.</summary>
    private static float[] Channels(Tensor tensor, int start, int count)
    {
        int batch = (int)tensor.Shape[0];
        int seqLen = (int)tensor.Shape[1];
        int channels = (int)tensor.Shape[2];
        ReadOnlySpan<float> source = tensor.AsSpan<float>();
        float[] result = new float[batch * seqLen * count];
        for (int token = 0; token < batch * seqLen; token++)
            source.Slice(token * channels + start, count).CopyTo(result.AsSpan(token * count, count));
        return result;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxDiff = 0f;
        for (int i = 0; i < expected.Length; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(expected[i] - actual[i]));
        Assert.True(maxDiff <= Tolerance, $"Outputs differ by {maxDiff} (tolerance {Tolerance}).");
    }
}
