using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>Tiny TRELLIS.2 flow architectures, random weights and the shipped config texts for CPU unit tests.</summary>
internal static unsafe class Trellis2TestData
{
    /// <summary>The <c>args</c> of the shipped structure flow sidecar.</summary>
    internal const string ShippedStructureFlowJson = """
        {"name": "SparseStructureFlowModel", "args": {"resolution": 16, "in_channels": 8, "out_channels": 8, "model_channels": 1536,
        "cond_channels": 1024, "num_blocks": 30, "num_heads": 12, "mlp_ratio": 5.3334, "pe_mode": "rope", "share_mod": true,
        "initialization": "scaled", "qk_rms_norm": true, "qk_rms_norm_cross": true, "dtype": "bfloat16"}}
        """;

    /// <summary>The shipped 512 shape flow sidecar.</summary>
    internal const string ShippedShapeFlow512Json = """
        {"name": "SLatFlowModel", "args": {"resolution": 32, "in_channels": 32, "out_channels": 32, "model_channels": 1536,
        "cond_channels": 1024, "num_blocks": 30, "num_heads": 12, "mlp_ratio": 5.3334, "pe_mode": "rope", "share_mod": true,
        "initialization": "scaled", "qk_rms_norm": true, "qk_rms_norm_cross": true, "dtype": "bfloat16"}}
        """;

    /// <summary>Width 16 over 2 heads gives head width 8: one RoPE pair per axis (6 rotated channels) and 2 that pass through.</summary>
    internal static Trellis2FlowConfig Tiny(bool dense) => new()
    {
        Dense = dense,
        Resolution = dense ? 4 : 8,
        InChannels = 4,
        OutChannels = 4,
        ModelChannels = 16,
        CondChannels = 8,
        NumBlocks = 2,
        NumHeads = 2,
        MlpRatio = 2.0,
    };

    internal static Dictionary<string, Tensor> RandomWeights(Trellis2FlowConfig config, int seed)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> weights = [];
        foreach (ShapeVaeKeyMap.TensorSpec spec in Trellis2FlowKeyMap.Specs(config))
        {
            Tensor t = new(new TensorShape(spec.Shape), DType.F32);
            Span<float> values = t.AsSpan<float>();
            bool gamma = spec.CheckpointKey.EndsWith(".gamma", StringComparison.Ordinal);
            bool norm = spec.CheckpointKey.Contains("norm2", StringComparison.Ordinal);
            for (int i = 0; i < values.Length; i++)
                values[i] = gamma || (norm && spec.CheckpointKey.EndsWith("weight", StringComparison.Ordinal))
                    ? (float)(0.5 + rng.NextDouble())
                    : (float)(rng.NextDouble() * 0.6 - 0.3);
            weights[spec.CheckpointKey] = t;
        }

        return weights;
    }

    internal static Tensor Random(Random rng, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    internal static void AssertDisposed(Tensor t) => Assert.Throws<ObjectDisposedException>(() => Touch(t));

    private static void Touch(Tensor t) => _ = t.AsSpan<float>().Length;

    internal static float[] ToArray(Tensor t) => t.AsReadOnlySpan<float>().ToArray();

    internal static void DisposeAll(IEnumerable<Tensor> tensors)
    {
        foreach (Tensor t in tensors)
            t.Dispose();
    }
}
