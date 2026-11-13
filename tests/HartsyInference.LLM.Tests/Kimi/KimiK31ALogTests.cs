using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Transformer.Kda;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>A_log is one value per head, not per head-channel: the loaded KDA forget gate folds each key channel with the value of the head
/// that channel belongs to. Checked against the shard's own values for the synthetic checkpoint's KDA layer.</summary>
public sealed class KimiK31ALogTests : IDisposable
{
    private const float Tol = 1e-6f;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-kimi-alog-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void ALog_IsReadPerHead_AndEachChannelFoldsItsHeadsValue()
    {
        KimiSyntheticCheckpoint.Write(_dir);
        KimiK31Config c = KimiK31HfConfig.Parse(File.ReadAllText(Path.Combine(_dir, "config.json")));
        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(_dir);
        KimiK31Weights weights = KimiK31WeightLoader.Load(c, source);
        try
        {
            // Layer 0 is a KDA layer in the synthetic checkpoint.
            const string p = "language_model.model.layers.0.self_attn.";
            float[] aLog = ReadValues(source, p + "A_log");
            float[] dtBias = ReadValues(source, p + "dt_bias");
            float[] forgetUp = ReadValues(source, p + "f_b_proj.weight");
            Assert.Equal(c.Heads, aLog.Length);

            KdaWeights kda = weights.Layers[0].Kda!;
            float[] bias = kda.ForgetBias.AsReadOnlySpan<float>().ToArray();
            float[] up = kda.ForgetUp.AsReadOnlySpan<float>().ToArray();
            int rank = c.KdaForgetRank;
            for (int ch = 0; ch < c.KeyDim; ch++)
            {
                float scale = MathF.Exp(aLog[ch / c.KdaHeadDim]);
                Assert.Equal(scale * dtBias[ch], bias[ch], Tol);
                for (int r = 0; r < rank; r++) Assert.Equal(scale * forgetUp[ch * rank + r], up[ch * rank + r], Tol);
            }
        }
        finally
        {
            foreach (Tensor t in weights.Owned) t.Dispose();
        }
    }

    [Fact]
    public void ALog_OfExactlyOneValuePerHead_LoadsAndRunsThroughTheDecoder()
    {
        KimiSyntheticCheckpoint.Write(_dir);
        KimiK31Config c = KimiK31HfConfig.Parse(File.ReadAllText(Path.Combine(_dir, "config.json")));
        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(_dir);
        KimiK31Weights weights = KimiK31WeightLoader.Load(c, source);
        try
        {
            using KimiK31Model model = new(new CpuBackend(), c, weights, maxPositions: 16);
            using KimiK31State state = new(c, capacity: 4);
            float[] logits = model.Forward([1, 2], state);
            Assert.Equal(c.Vocab, logits.Length);
            foreach (float v in logits) Assert.True(float.IsFinite(v));
        }
        finally
        {
            foreach (Tensor t in weights.Owned) t.Dispose();
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    public void ALog_OfAnyLengthOtherThanOnePerHead_IsRefusedWithItsShape(int length)
    {
        KimiSyntheticCheckpoint.Write(_dir);
        KimiK31Config c = KimiK31HfConfig.Parse(File.ReadAllText(Path.Combine(_dir, "config.json")));
        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(_dir);

        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => KimiK31WeightLoader.Load(c, new WithALogShape(source, length)));
        Assert.Contains($"[{length}]", ex.Message);
        Assert.Contains("open", ex.Message);
    }

    [Fact]
    public void ALog_InAMultiDimensionalExport_IsRefusedWithItsShape()
    {
        // The HF or NeMo export may store A_log as [1, 1, Heads, 1]; the loader names that shape rather than reading it.
        KimiSyntheticCheckpoint.Write(_dir);
        KimiK31Config c = KimiK31HfConfig.Parse(File.ReadAllText(Path.Combine(_dir, "config.json")));
        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(_dir);

        WithALogShape exported = new(source, 1, 1, c.Heads, 1);
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => KimiK31WeightLoader.Load(c, exported));
        Assert.Contains("[1, 1, 2, 1]", ex.Message);
    }

    // Serves the checkpoint as it is except A_log, which comes back zero-filled with the given shape.
    private sealed class WithALogShape(IKimiK31WeightSource inner, params long[] shape) : IKimiK31WeightSource
    {
        public bool Has(string name) => inner.Has(name);

        public Tensor Read(string name)
        {
            if (!name.EndsWith("A_log", StringComparison.Ordinal)) return inner.Read(name);
            Tensor t = new(new TensorShape(shape), DType.F32);
            t.AsSpan<float>().Clear();
            return t;
        }
    }

    private static float[] ReadValues(IKimiK31WeightSource source, string name)
    {
        Tensor t = source.Read(name);
        try
        {
            return t.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            t.Dispose();
        }
    }
}
