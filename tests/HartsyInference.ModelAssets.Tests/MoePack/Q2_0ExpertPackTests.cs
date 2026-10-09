using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.MoePack;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.MoePack;

/// <summary>Q2_0 expert packs: a write, read and verify round trip within the codec's measured error, and the refusals
/// that keep a pack from holding a dimension the 64-value blocks cannot split.</summary>
public sealed class Q2_0ExpertPackTests : IDisposable
{
    private const int Hidden = 256;
    private const int Intermediate = 256;
    private const string Fingerprint = "q2-0-test-fingerprint";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-q20-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private static (float[] Gate, float[] Up, float[] Down) Expert(Random rng)
    {
        return (Values(Intermediate * Hidden, rng), Values(Intermediate * Hidden, rng), Values(Hidden * Intermediate, rng));

        static float[] Values(int count, Random rng)
        {
            float[] values = new float[count];
            for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
            return values;
        }
    }

    private static Dictionary<ExpertKey, (float[] Gate, float[] Up, float[] Down)> WritePack(string directory, int experts)
    {
        Random rng = new(4);
        Dictionary<ExpertKey, (float[] Gate, float[] Up, float[] Down)> sources = [];
        using ExpertPackWriter writer = new(directory, Fingerprint, Hidden, Intermediate, DType.Q2_0,
            Enumerable.Range(0, experts).Select(static e => new ExpertKey(e / 2, e % 2)));
        for (int e = 0; e < experts; e++)
        {
            (float[] Gate, float[] Up, float[] Down) source = Expert(rng);
            ExpertKey key = new(e / 2, e % 2);
            writer.AddExpert(key.Layer, key.Expert, source.Gate, source.Up, source.Down);
            sources[key] = source;
        }
        writer.Finish();
        return sources;
    }

    [Fact]
    public void Q2_0Pack_RoundTripsWithinCodecToleranceAndHasExactPayloadSize()
    {
        string dir = Path.Combine(_root, "q20");
        Dictionary<ExpertKey, (float[] Gate, float[] Up, float[] Down)> sources = WritePack(dir, experts: 4);

        using ExpertPackReader reader = ExpertPackReader.Open(dir, Fingerprint);
        ExpertPackVerification report = ExpertPackVerifier.Verify(reader, key => sources[key], reader.Keys);

        Assert.Equal(4, reader.Count);
        Assert.Equal(4, report.Checked);
        Assert.Empty(report.Failures);
        Assert.True(report.RelativeRmse < Q2_0CodecTests.MaxRelativeRmse,
            $"Q2_0 pack relative RMSE {report.RelativeRmse:F4} exceeds {Q2_0CodecTests.MaxRelativeRmse}.");
        Assert.Equal(4L * 3 * Hidden * Intermediate / 64 * 18, reader.PayloadBytes);
    }

    [Fact]
    public void Q2_0Pack_ManifestNamesTheDType()
    {
        string dir = Path.Combine(_root, "named");
        WritePack(dir, experts: 2);

        string manifest = File.ReadAllText(Path.Combine(dir, "manifest.json"));

        Assert.Contains("\"Q2_0\"", manifest);
    }

    [Fact]
    public void Writer_Refuses_A_Dimension_That_Is_Not_A_Multiple_Of_64()
    {
        string dir = Path.Combine(_root, "refused");

        Assert.Throws<ArgumentException>(() => new ExpertPackWriter(dir, Fingerprint, 100, Intermediate, DType.Q2_0, [new ExpertKey(0, 0)]));
        Assert.Throws<ArgumentException>(() => new ExpertPackWriter(dir, Fingerprint, Hidden, 100, DType.Q2_0, [new ExpertKey(0, 0)]));
    }
}
