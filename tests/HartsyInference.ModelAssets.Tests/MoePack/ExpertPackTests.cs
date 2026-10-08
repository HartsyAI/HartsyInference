using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.MoePack;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.MoePack;

/// <summary>Expert pack round trips against their F32 sources, plus the refusals and corruption checks that make a pack trustworthy.</summary>
public sealed class ExpertPackTests : IDisposable
{
    private const int Hidden = 256;
    private const int Intermediate = 256;
    private const string Fingerprint = "test-fingerprint";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-pack-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private static uint _seed = 0xC0FFEEu;

    private static float Next()
    {
        _seed ^= _seed << 13;
        _seed ^= _seed >> 17;
        _seed ^= _seed << 5;
        return ((_seed & 0xFFFF) / 65535f - 0.5f) * 1.0f;
    }

    private static float[] Random(int n)
    {
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = Next();
        return r;
    }

    private static (float[] Gate, float[] Up, float[] Down) Expert() =>
        (Random(Intermediate * Hidden), Random(Intermediate * Hidden), Random(Hidden * Intermediate));

    private static Dictionary<ExpertKey, (float[] Gate, float[] Up, float[] Down)> WritePack(string directory, DType dtype, int experts,
            string fingerprint = Fingerprint)
    {
        Dictionary<ExpertKey, (float[] Gate, float[] Up, float[] Down)> sources = [];
        using ExpertPackWriter writer = new(directory, fingerprint, Hidden, Intermediate, dtype);
        for (int e = 0; e < experts; e++)
        {
            (float[] Gate, float[] Up, float[] Down) source = Expert();
            ExpertKey key = new(e / 2, e % 2);
            writer.AddExpert(key.Layer, key.Expert, source.Gate, source.Up, source.Down);
            sources[key] = source;
        }
        writer.Finish();
        return sources;
    }

    [Fact]
    public void Q8_0Pack_RoundTripsWithinQuantizationErrorAndHasExactPayloadSize()
    {
        string dir = Path.Combine(_root, "q8");
        Dictionary<ExpertKey, (float[] Gate, float[] Up, float[] Down)> sources = WritePack(dir, DType.Q8_0, experts: 4);

        using ExpertPackReader reader = ExpertPackReader.Open(dir, Fingerprint);
        ExpertPackVerification report = ExpertPackVerifier.Verify(reader, key => sources[key], reader.Keys);

        Assert.Equal(4, reader.Count);
        Assert.Equal(4, report.Checked);
        Assert.Empty(report.Failures);
        Assert.True(report.RelativeRmse < 0.01, $"Q8_0 relative RMSE {report.RelativeRmse:E3} exceeds 1%.");
        Assert.Equal(4L * 3 * Hidden * Intermediate / 32 * 34, reader.PayloadBytes);
    }

    [Fact]
    public void Q4_KPack_RoundTripsWithinQuantizationErrorAndHasExactPayloadSize()
    {
        string dir = Path.Combine(_root, "q4k");
        Dictionary<ExpertKey, (float[] Gate, float[] Up, float[] Down)> sources = WritePack(dir, DType.Q4_K, experts: 4);

        using ExpertPackReader reader = ExpertPackReader.Open(dir, Fingerprint);
        ExpertPackVerification report = ExpertPackVerifier.Verify(reader, key => sources[key], reader.Keys);

        Assert.Empty(report.Failures);
        Assert.True(report.RelativeRmse < 0.1, $"Q4_K relative RMSE {report.RelativeRmse:E3} exceeds 10%.");
        Assert.Equal(4L * 3 * Hidden * Intermediate / 256 * 144, reader.PayloadBytes);
    }

    [Fact]
    public void Records_AreAlignedAndTheManifestNamesTheTopology()
    {
        string dir = Path.Combine(_root, "aligned");
        WritePack(dir, DType.Q8_0, experts: 3);

        using ExpertPackReader reader = ExpertPackReader.Open(dir, Fingerprint);
        Assert.Equal(Fingerprint, reader.TopologyFingerprint);
        Assert.Equal(Hidden, reader.Hidden);
        Assert.Equal(Intermediate, reader.Intermediate);
        Assert.Equal(ExpertBacking.Pack, reader.Backing);
        Assert.All(reader.Records, record => Assert.Equal(0, record.Offset % ExpertPackWriter.RecordAlignment));
        Assert.Equal(0, reader.Records.Min(static record => record.Offset));
    }

    [Fact]
    public void Incomplete_Pack_IsRefused()
    {
        string dir = Path.Combine(_root, "incomplete");
        using (ExpertPackWriter writer = new(dir, Fingerprint, Hidden, Intermediate, DType.Q8_0))
        {
            (float[] Gate, float[] Up, float[] Down) source = Expert();
            writer.AddExpert(0, 0, source.Gate, source.Up, source.Down);
        }

        Assert.Throws<InvalidDataException>(() => ExpertPackReader.Open(dir));
    }

    [Fact]
    public void FingerprintMismatch_IsRefused()
    {
        string dir = Path.Combine(_root, "fingerprint");
        WritePack(dir, DType.Q8_0, experts: 2);

        Assert.Throws<InvalidDataException>(() => ExpertPackReader.Open(dir, "a-different-topology"));
    }

    [Fact]
    public void CompletedPack_IsNeverOverwritten()
    {
        string dir = Path.Combine(_root, "complete");
        WritePack(dir, DType.Q8_0, experts: 2);

        Assert.Throws<InvalidOperationException>(() => new ExpertPackWriter(dir, Fingerprint, Hidden, Intermediate, DType.Q8_0));
    }

    [Fact]
    public void Corruption_IsDetectedOnReadAndReportedByVerification()
    {
        string dir = Path.Combine(_root, "corrupt");
        Dictionary<ExpertKey, (float[] Gate, float[] Up, float[] Down)> sources = WritePack(dir, DType.Q8_0, experts: 2);
        string data = Path.Combine(dir, "experts.bin");
        byte[] bytes = File.ReadAllBytes(data);
        bytes[100] ^= 0xFF;
        File.WriteAllBytes(data, bytes);

        using ExpertPackReader reader = ExpertPackReader.Open(dir, Fingerprint);
        ExpertKey first = new(0, 0);
        Assert.Throws<InvalidDataException>(() => reader.Resolve(first));

        ExpertPackVerification report = ExpertPackVerifier.Verify(reader, key => sources[key], reader.Keys);
        Assert.Contains(first, report.Failures);
        Assert.Equal(1, report.Checked);
    }

    [Fact]
    public void DuplicateExpert_AndUnknownKey_AreRejected()
    {
        string dir = Path.Combine(_root, "dup");
        using ExpertPackWriter writer = new(dir, Fingerprint, Hidden, Intermediate, DType.Q8_0);
        (float[] Gate, float[] Up, float[] Down) source = Expert();
        writer.AddExpert(0, 1, source.Gate, source.Up, source.Down);
        Assert.Throws<ArgumentException>(() => writer.AddExpert(0, 1, source.Gate, source.Up, source.Down));
        writer.Finish();

        using ExpertPackReader reader = ExpertPackReader.Open(dir, Fingerprint);
        Assert.Throws<KeyNotFoundException>(() => reader.Resolve(new ExpertKey(9, 9)));
    }

    [Fact]
    public void Dimensions_MustMatchTheDtypeBlockSize()
    {
        Assert.Throws<ArgumentException>(() => new ExpertPackWriter(Path.Combine(_root, "bad"), Fingerprint, 100, Intermediate, DType.Q4_K));
    }
}
