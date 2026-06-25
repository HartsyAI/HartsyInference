using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numeric parity for the GPT-SoVITS s1 (text→semantic AR decoder) against the real
/// <c>s1bert25hz</c> checkpoint: the first-step prefill logits over the 1025 semantic vocab must match the
/// upstream <c>Text2SemanticDecoder</c> forward. Gated on <c>HARTSYINFERENCE_S1_CKPT</c>; skips when absent.</summary>
public sealed unsafe class Text2SemanticParityTests
{
    [Fact]
    public void FirstStepLogits_MatchUpstream()
    {
        string? pt = Environment.GetEnvironmentVariable("HARTSYINFERENCE_S1_CKPT");
        if (string.IsNullOrEmpty(pt) || !File.Exists(pt)) return;

        using CpuBackend backend = new();
        using PytorchPickleLoader loader = new();
        loader.Load(pt);

        using Text2Semantic s1 = new(new Text2SemanticConfig());
        s1.LoadWeights(loader.GetAllTensors(), "model");

        const int tp = 10, tpr = 8, bertDim = 1024;
        int[] phon = new int[tp]; for (int i = 0; i < tp; i++) phon[i] = (i * 37) % 512;
        int[] prefix = new int[tpr]; for (int i = 0; i < tpr; i++) prefix[i] = (i * 97) % 1024;
        Tensor bert = new(new TensorShape(bertDim, tp), DType.F32);
        float* bp = (float*)bert.DataPointer;
        for (int c = 0; c < bertDim; c++)
            for (int t = 0; t < tp; t++) bp[c * tp + t] = ((c * 3 + t * 5) % 89) / 89f - 0.5f;

        float[] logits = s1.FirstStepLogits(backend, phon, bert, prefix);
        bert.Dispose();

        Assert.Equal(1025, logits.Length);
        // argmax + first logits, re-derived from the REAL upstream module (AR.modules.transformer
        // TransformerEncoder + patched MultiheadAttention) loaded with the s1bert25hz v2 checkpoint, taken at
        // the last (audio) position — what inference samples from. The previous golden values
        // (-4.26 / -3.14 / -2.34 / -2.77) were stale/wrong; an independent from-scratch reimplementation AND
        // the real upstream module both produce these values, matching the C# Text2Semantic forward.
        int argmax = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[argmax]) argmax = i;
        Assert.Equal(1024, argmax);
        Assert.Equal(-6.1648f, logits[0], precision: 1);
        Assert.Equal(-4.2708f, logits[1], precision: 1);
        Assert.Equal(-4.1612f, logits[2], precision: 1);
        Assert.Equal(-4.0448f, logits[3], precision: 1);
    }
}
