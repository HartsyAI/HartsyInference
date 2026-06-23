using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numeric parity for the CAM++ speaker encoder against the real <c>campplus_cn_common.bin</c>
/// (3D-Speaker), validated to the ONNX-Runtime output of CosyVoice2's <c>campplus.onnx</c> — the same weights.
/// Two lengths exercise the short path (single seg-pool segment) and the multi-segment seg-pool. Gated on
/// <c>HARTSYINFERENCE_CAMPPLUS_PT</c>; skips cleanly when absent.</summary>
public sealed unsafe class CamPlusSpeakerEncoderTests
{
    // Golden = ORT(campplus.onnx) on the deterministic fbank below.
    [Theory]
    [InlineData(40, new[] { 0.418665f, -0.095306f, 1.33638f, 1.638615f, -0.60447f, -2.381926f }, 11.8102f, -4.20949f)]
    [InlineData(240, new[] { 1.118069f, 0.929883f, 0.674f, 2.112268f, -1.509852f, -2.326308f }, 19.22358f, -6.13308f)]
    public void Forward_MatchesOnnxRuntime_OnRealCheckpoint(int t, float[] first6, float norm, float sum)
    {
        string? pt = Environment.GetEnvironmentVariable("HARTSYINFERENCE_CAMPPLUS_PT");
        if (string.IsNullOrEmpty(pt) || !File.Exists(pt)) return;

        using CpuBackend backend = new();
        using PytorchPickleLoader loader = new();
        loader.Load(pt);

        using CamPlusSpeakerEncoder enc = new(embedDim: 192);
        enc.LoadWeights(loader.GetAllTensors());

        Tensor fbank = new(new TensorShape(1, t, 80), DType.F32);
        float* fp = (float*)fbank.DataPointer;
        for (int i = 0; i < t; i++)
            for (int c = 0; c < 80; c++)
                fp[i * 80 + c] = ((i * 7 + c * 13) % 100) * 0.01f;

        Tensor emb = enc.Forward(backend, fbank);
        try
        {
            Assert.Equal(192, (int)emb.Shape[1]);
            float* ep = (float*)emb.DataPointer;
            for (int i = 0; i < first6.Length; i++) Assert.Equal(first6[i], ep[i], precision: 2);
            float total = 0f, sq = 0f;
            for (int i = 0; i < 192; i++) { total += ep[i]; sq += ep[i] * ep[i]; }
            Assert.Equal(sum, total, precision: 1);
            Assert.Equal(norm, MathF.Sqrt(sq), precision: 1);
        }
        finally
        {
            emb.Dispose();
            fbank.Dispose();
        }
    }
}
