using HartsyInference.Audio.Models.OpenVoice;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numeric parity for the OpenVoice V2 tone-color <c>ReferenceEncoder</c> against the real
/// <c>myshell-ai/OpenVoiceV2/converter/checkpoint.pth</c>. Golden values come from the upstream
/// <c>ReferenceEncoder</c> (torch) run on a deterministic spectrogram with the real weights. Gated on the
/// checkpoint path via <c>HARTSYINFERENCE_OPENVOICE_CKPT</c>; skips cleanly when absent.</summary>
public sealed unsafe class OpenVoiceSpeakerEncoderTests
{
    private const int SpecCh = 513, T = 20;

    // se = ReferenceEncoder(real ref_enc weights)(deterministic spec). See the test body for the spec formula.
    private static readonly float[] ExpectedFirst8 =
        [-0.090418f, -0.380032f, 0.263137f, -0.798096f, -0.237468f, 0.269168f, 0.612303f, -0.200696f];
    private static readonly float[] ExpectedLast4 = [-0.264693f, 0.481366f, -0.58718f, -0.565772f];
    private const float ExpectedSum = 3.042841f;
    private const float ExpectedNorm = 8.523634f;

    [Fact]
    public void RefEncoder_MatchesUpstream_OnRealCheckpoint()
    {
        string? ckpt = Environment.GetEnvironmentVariable("HARTSYINFERENCE_OPENVOICE_CKPT");
        if (string.IsNullOrEmpty(ckpt) || !File.Exists(ckpt)) return;     // skip without weights

        using CpuBackend backend = new();
        using PytorchPickleLoader loader = new();
        loader.Load(ckpt);

        OpenVoiceSpeakerEncoder enc = new(OpenVoiceSpeakerConfig.Default);
        enc.LoadWeights(loader.GetAllTensors(), "ref_enc");

        Tensor spec = new(new TensorShape(1, SpecCh, T), DType.F32);
        float* sp = (float*)spec.DataPointer;
        for (int c = 0; c < SpecCh; c++)
            for (int t = 0; t < T; t++)
                sp[c * T + t] = ((c * 31 + t * 17) % 1000) * 0.001f;

        Tensor g = enc.Extract(backend, spec, T);
        try
        {
            Assert.Equal(1, (int)g.Shape[0]);
            Assert.Equal(256, (int)g.Shape[1]);
            Assert.Equal(1, (int)g.Shape[2]);
            float* gp = (float*)g.DataPointer;
            float sum = 0f, sq = 0f;
            for (int i = 0; i < 256; i++) { sum += gp[i]; sq += gp[i] * gp[i]; }

            for (int i = 0; i < ExpectedFirst8.Length; i++) Assert.Equal(ExpectedFirst8[i], gp[i], precision: 3);
            for (int i = 0; i < ExpectedLast4.Length; i++) Assert.Equal(ExpectedLast4[i], gp[256 - 4 + i], precision: 3);
            Assert.Equal(ExpectedSum, sum, precision: 2);
            Assert.Equal(ExpectedNorm, MathF.Sqrt(sq), precision: 2);
        }
        finally
        {
            g.Dispose();
            spec.Dispose();
            enc.Dispose();
        }
    }
}
