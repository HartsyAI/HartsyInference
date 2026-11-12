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
