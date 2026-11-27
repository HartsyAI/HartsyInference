using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The avatar audio cross-attention (<c>SingleStreamAttention</c>, @ 58eb336) against the pinned reference class on a tiny geometry, for a
/// text-to-video call and an image-to-video call with one conditioning frame (tests/python-reference/longcat_avatar/dump_longcat_avatar_audio_attn.py),
/// CPU F32. Skipped, and so not evidence, until HARTSY_LONGCAT_AVATAR_ATTN_DIR names the dump directory.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAvatarAudioAttnParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_ATTN_DIR";
    private const double Tolerance = 1e-5;

    [LongCatAvatarAttnFact]
    public void TextToVideoAndImageToVideoMatchTheReference()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        JsonElement dims = document.RootElement.GetProperty("dims");
        int dim = dims.GetProperty("dim").GetInt32(), heads = dims.GetProperty("heads").GetInt32(), enc = dims.GetProperty("enc").GetInt32();
        int frames = dims.GetProperty("frames").GetInt32();

        Dictionary<string, Tensor> weights = [];
        try
        {
            foreach (JsonProperty tensor in document.RootElement.GetProperty("tensors").EnumerateObject())
                if (tensor.Name.StartsWith("attn.", StringComparison.Ordinal))
                    weights[tensor.Name] = LongCatReferenceDump.Load(dir, document, tensor.Name);

            using LongCatAudioCrossAttention layer = new(dim, enc, heads);
            layer.LoadWeights(weights);
            using Tensor x = LongCatReferenceDump.Load(dir, document, "inputs.x");
            using Tensor audio = LongCatReferenceDump.Load(dir, document, "inputs.audio");
            using CpuBackend backend = new();

            using Tensor t2v = layer.Forward(backend, x, audio, frames, numCond: 0);
            Assert.True(RelRms(t2v.AsReadOnlySpan<float>(), LongCatReferenceDump.Read(dir, document, "output.t2v")) <= Tolerance,
                "text-to-video audio attention exceeds relRMS 1e-5.");

            // The reference's image-to-video output is the zero block for the conditioning frame, then the noisy frames.
            float[] expected = [.. LongCatReferenceDump.Read(dir, document, "output.i2v_cond"), .. LongCatReferenceDump.Read(dir, document, "output.i2v")];
            using Tensor i2v = layer.Forward(backend, x, audio, frames, numCond: 1);
            Assert.True(RelRms(i2v.AsReadOnlySpan<float>(), expected) <= Tolerance, "image-to-video audio attention exceeds relRMS 1e-5.");
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    private static double RelRms(ReadOnlySpan<float> actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double diff = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            diff += d * d;
            norm += (double)expected[i] * expected[i];
        }
        return norm == 0 ? Math.Sqrt(diff) : Math.Sqrt(diff / norm);
    }

    private sealed class LongCatAvatarAttnFactAttribute : FactAttribute
    {
        public LongCatAvatarAttnFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat avatar audio attention reference comparison did not run.";
        }
    }
}
