using System.Text.Json;
using HartsyInference.Benchmarks.Shared;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Engine;
using HartsyInference.GpuBenchmarks.Attention;

namespace HartsyInference.OpParity;

/// <summary>Checks the same serialized attention operands consumed by Kitchen and the operation benchmark.</summary>
internal static class SharedAttentionCheck
{
    /// <summary>Rejects unsupported oracle cases and unreviewed mandatory acceptance instead of skipping them.</summary>
    public static int Run(Dictionary<string, string> options)
    {
        using AttentionFixture fixture = AttentionFixture.Load(options["fixture"]);
        if (options.GetValueOrDefault("mandatory", "false") == "true" && !fixture.AcceptanceReviewed)
            throw new InvalidOperationException("Mandatory evidence needs reviewed numerical acceptance.");
        using IDisposable settings = BenchmarkKnobProfiles.Capture(false, out _).Push();
        using IBackend backend = BackendFactory.Create(options.GetValueOrDefault("backend", "cpu"));
        AttentionOracle oracle = fixture.Oracle();
        AttentionCall call = fixture.Call;
        backend.Attention(in call);
        backend.Sync();
        (double relative, double cosine, double maximum, int rows) = oracle.Compare(fixture.OutputAt, 64);
        double sum = 0;
        long count = 0;
        foreach ((int b, int h, int row) in oracle.Sample(64))
        {
            double[] expected = oracle.Row(b, h, row);
            for (int channel = 0; channel < expected.Length; channel++)
            {
                sum += Math.Abs(expected[channel] - fixture.OutputAt(b, h, row, channel));
                count++;
            }
        }
        bool passed = double.IsFinite(relative) && relative <= fixture.RelativeTolerance;
        if (fixture.MeanAbsoluteTolerance is { } limit) passed &= sum / count <= limit;
        using FileStream file = new(options["out"], FileMode.CreateNew);
        using Utf8JsonWriter writer = new(file, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();
        writer.WriteString("fixture_sha256", fixture.ManifestHash);
        writer.WriteString("output_sha256", fixture.OutputHash());
        writer.WriteBoolean("quality_passed", passed);
        writer.WriteBoolean("acceptance_reviewed", fixture.AcceptanceReviewed);
        writer.WriteNumber("oracle_rows", rows);
        if (double.IsFinite(relative)) writer.WriteNumber("rel_l2", relative); else writer.WriteNull("rel_l2");
        if (double.IsFinite(maximum)) writer.WriteNumber("max_abs", maximum); else writer.WriteNull("max_abs");
        writer.WriteEndObject();
        return passed ? 0 : 1;
    }
}
