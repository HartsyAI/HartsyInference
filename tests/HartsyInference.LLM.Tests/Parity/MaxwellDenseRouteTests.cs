using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Maxwell;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The device-cache dense route against the host route on the tiny checkpoint. The tiny geometry puts the dense limit at 24 tokens (6 blocks of
/// 4), so a prefill of 10 tokens and the decode steps up to position 24 run dense, and the later steps run the sparse route on the host in both models.
/// The two models differ only in <see cref="MaxwellLoadOptions.DeviceDenseAttention"/>.</summary>
public sealed class MaxwellDenseRouteTests
{
    private static readonly string Directory = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Maxwell");

    // Largest difference relative to the largest reference value (floored at 1e-6, so an all-zero row does not divide by zero).
    private static double Relative(ReadOnlySpan<float> actual, ReadOnlySpan<float> reference)
    {
        Assert.Equal(reference.Length, actual.Length);
        double diff = 0, scale = 1e-6;
        for (int i = 0; i < reference.Length; i++)
        {
            diff = Math.Max(diff, Math.Abs(actual[i] - reference[i]));
            scale = Math.Max(scale, Math.Abs(reference[i]));
        }
        return diff / scale;
    }

    private static MaxwellModel Load(CpuBackend backend, bool deviceDense) =>
        MaxwellModel.Load(Directory, backend, new MaxwellLoadOptions { MaxSequenceTokens = 64, DeviceDenseAttention = deviceDense });

    [Fact]
    public void DeviceDenseRoute_MatchesTheHostRoute_ThroughTheSparseLimit()
    {
        using CpuBackend backend = new();
        using MaxwellModel dense = Load(backend, deviceDense: true);
        using MaxwellModel host = Load(backend, deviceDense: false);
        int[] ids = NpyReader.ReadInt32(Path.Combine(Directory, "input_ids.npy")).Data;
        int limit = dense.Config.Sparse.DenseCausalLimit;
        Assert.Equal(24, limit);
        using MaxwellSequenceState a = dense.CreateState(64);
        using MaxwellSequenceState b = host.CreateState(64);

        const int Prefill = 10, Total = 40;
        float[] fromDense = dense.Forward(ids.AsSpan(0, Prefill), a);
        float[] fromHost = host.Forward(ids.AsSpan(0, Prefill), b);
        double worst = Relative(fromDense, fromHost);
        for (int pos = Prefill; pos < Total; pos++)
        {
            fromDense = dense.Forward(ids.AsSpan(pos, 1), a);
            fromHost = host.Forward(ids.AsSpan(pos, 1), b);
            worst = Math.Max(worst, Relative(fromDense, fromHost));
        }

        // The device cache holds the dense positions only: the first step at or past the limit is sparse, and every later step follows it.
        Assert.Equal(limit, a.DenseLength);
        Assert.Equal(Total, a.Length);
        Assert.True(worst < 1e-4, $"dense and host routes differ by {worst:E2}");
    }

    [Fact]
    public void Truncate_ClearsTheDeviceCache_AndTheHostRouteTakesOver()
    {
        using CpuBackend backend = new();
        using MaxwellModel dense = Load(backend, deviceDense: true);
        using MaxwellModel host = Load(backend, deviceDense: false);
        int[] ids = NpyReader.ReadInt32(Path.Combine(Directory, "input_ids.npy")).Data;
        using MaxwellSequenceState a = dense.CreateState(64);
        using MaxwellSequenceState b = host.CreateState(64);
        dense.Forward(ids.AsSpan(0, 10), a);
        host.Forward(ids.AsSpan(0, 10), b);
        Assert.Equal(10, a.DenseLength);

        a.Truncate(6);
        b.Truncate(6);
        Assert.Equal(0, a.DenseLength);
        float[] fromDense = dense.Forward(ids.AsSpan(6, 4), a);
        float[] fromHost = host.Forward(ids.AsSpan(6, 4), b);

        Assert.True(Relative(fromDense, fromHost) < 1e-4);
        Assert.Equal(0, a.DenseLength);
    }
}
