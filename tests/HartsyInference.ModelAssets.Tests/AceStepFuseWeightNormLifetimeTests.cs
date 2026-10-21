using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Weight-norm fusion reads its F32 casts of bf16 pairs through raw pointers. A cast the GC finalizes before the
/// loop ends is freed under it: an access violation that aborts the test host, or wrong weights.</summary>
[Collection(ForcedGcCollection.Name)]
public sealed unsafe class AceStepFuseWeightNormLifetimeTests
{
    [Fact]
    public void FuseWeightNorm_UnderForcedCollections_KeepsItsCastsAlive()
    {
        // Past glibc's 32 MB mmap ceiling, so a freed cast is unmapped rather than recycled.
        const int outChannels = 4096, perChannel = 4096;
        using Tensor gFull = new Tensor(new TensorShape(outChannels, 1, 1), DType.F32);
        using Tensor vFull = new Tensor(new TensorShape(outChannels, perChannel, 1), DType.F32);
        gFull.AsSpan<float>().Fill(2f);
        vFull.AsSpan<float>().Fill(1f);
        using Tensor g = gFull.CastTo(DType.BF16);
        using Tensor v = vFull.CastTo(DType.BF16);
        float expected = 2f / MathF.Sqrt(perChannel);
        using CancellationTokenSource stop = new CancellationTokenSource();
        Thread collector = new Thread(() =>
        {
            while (!stop.IsCancellationRequested)
            {
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
                GC.WaitForPendingFinalizers();
            }
        });
        collector.Start();
        try
        {
            for (int i = 0; i < 3; i++)
            {
                Dictionary<string, Tensor> fused = AceStepCheckpointConverter.FuseWeightNorm(
                    new Dictionary<string, Tensor> { ["x.weight_g"] = g, ["x.weight_v"] = v });
                using Tensor weight = fused["x.weight"];
                float* p = (float*)weight.DataPointer;
                for (long e = 0; e < (long)outChannels * perChannel; e += 4099)
                {
                    Assert.Equal(expected, p[e], 6);
                }
            }
        }
        finally
        {
            stop.Cancel();
            collector.Join();
        }
    }
}
