using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The LongCat-Video-Avatar audio projection (<c>AudioProjModel</c>, @ 58eb336) against the pinned reference class on a tiny geometry
/// (tests/python-reference/longcat_avatar/dump_longcat_audio_proj.py), CPU F32. Skipped, and so not evidence, until
/// HARTSY_LONGCAT_AVATAR_DIR names the dump directory.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAudioProjectionParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_DIR";
    private const double Tolerance = 1e-5;

    [LongCatAvatarFact]
    public void ProjectionMatchesTheReferenceClass()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        JsonElement dims = document.RootElement.GetProperty("dims");

        Dictionary<string, Tensor> weights = [];
        try
        {
            foreach (JsonProperty tensor in document.RootElement.GetProperty("tensors").EnumerateObject())
            {
                if (tensor.Name.StartsWith("audio_proj.", StringComparison.Ordinal))
                    weights[tensor.Name] = LongCatReferenceDump.Load(dir, document, tensor.Name);
            }

            using LongCatAudioProjection projection = new(
                dims.GetProperty("seq_len").GetInt32(), dims.GetProperty("seq_len_vf").GetInt32(), dims.GetProperty("blocks").GetInt32(),
                dims.GetProperty("channels").GetInt32(), dims.GetProperty("intermediate_dim").GetInt32(),
                dims.GetProperty("output_dim").GetInt32(), dims.GetProperty("context_tokens").GetInt32());
            projection.LoadWeights(weights);

            using Tensor first = LongCatReferenceDump.Load(dir, document, "inputs.first");
            using Tensor latter = LongCatReferenceDump.Load(dir, document, "inputs.latter");
            using CpuBackend backend = new();
            using Tensor got = projection.Forward(backend, first, latter);
            float[] expected = LongCatReferenceDump.Read(dir, document, "output");

            Assert.Equal(expected.Length, (int)got.Shape.ElementCount);
            double relRms = RelRms(got.AsReadOnlySpan<float>(), expected);
            Assert.True(relRms <= Tolerance, $"audio projection relRMS {relRms:E3} exceeds {Tolerance:E0}.");
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    private static double RelRms(ReadOnlySpan<float> actual, float[] expected)
    {
        double diff = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            diff += d * d;
            norm += (double)expected[i] * expected[i];
        }
        return norm == 0 ? Math.Sqrt(diff) : Math.Sqrt(diff / norm);
    }

    private sealed class LongCatAvatarFactAttribute : FactAttribute
    {
        public LongCatAvatarFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat avatar audio reference comparison did not run.";
        }
    }
}
