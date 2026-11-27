using System.Diagnostics;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Diagnostic: the host memory the streamed MovieRoll DiT load holds, read from the released checkpoint on the CPU. Loads the shared
/// weights, then blocks 0 and 1, and writes the process's private bytes after each step, so the one-block residency and its release can be
/// read off the log. Diagnostic only (the bytes are the process's, not a bound); skipped unless <c>MOVIEROLL_DIT_PATH</c> names the file.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollHostMemoryProbeTests(ITestOutputHelper output)
{
    [RealWeightsFact("MOVIEROLL_DIT_PATH")]
    public void StreamedLoad_HoldsOneBlockOfHostMemory()
    {
        string dit = Environment.GetEnvironmentVariable("MOVIEROLL_DIT_PATH")!;
        HunyuanVideo15Config config = HunyuanVideo15Config.Full;
        using CheckpointSource checkpoint = CheckpointSource.Open(dit);

        void Report(string step)
        {
            GC.Collect();
            long priv = Process.GetCurrentProcess().PrivateMemorySize64;
            output.WriteLine($"PROBE {step}: private {priv / (1L << 20)} MiB, managed {GC.GetTotalMemory(false) / (1L << 20)} MiB");
        }

        Report("mapped checkpoint");
        using HunyuanVideo15Dit dit15 = new(config);
        using HunyuanVideo15BlockLoader loader = HunyuanVideo15BlockLoader.FromCheckpoint(dit15, config, checkpoint, [DType.F32]);
        loader.LoadShared();
        Report("shared weights loaded");
        loader.LoadBlock(0);
        Report("block 0 loaded");
        loader.LoadBlock(1);
        Report("block 1 loaded (block 0 released)");
        Assert.True(true);
    }
}
