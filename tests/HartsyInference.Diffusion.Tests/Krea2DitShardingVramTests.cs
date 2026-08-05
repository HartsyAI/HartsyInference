using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Phase 8 (ROADMAP.md §1): real-weight verification that <see cref="Krea2Transformer.ForwardSharded"/>
/// actually POOLS VRAM rather than replicating the DiT on both cards — the whole point of block-range sharding
/// over CFG-branch parallelism's weight-replication approach. Loads the real Krea2 Turbo fp8 checkpoint
/// (~13 GB), splits its 28 blocks across the 3060 (ordinal 0, 12 GB total) and the 4090 (ordinal 1, 24 GB
/// total), and asserts each card's resident footprint stays near its OWN half rather than the whole DiT —
/// the 3060 in particular could not hold the whole ~13 GB checkpoint resident alongside its own activations
/// (11-12 GB free in practice), so a passing run is itself evidence the split is real, not just "it ran."
/// Skips cleanly if the checkpoint is absent or either card doesn't have enough free VRAM at run time.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class Krea2DitShardingVramTests
{
    private readonly ITestOutputHelper _output;
    public Krea2DitShardingVramTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ForwardSharded_RealTurboCheckpoint_PoolsVramAcrossTwoGpus()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string rootDir = TestPaths.Krea2.TurboDir;
        if (!RealWeightGate.Require(_output.WriteLine, rootDir)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Krea2DitShardingVramTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading Krea2 Turbo transformer from {rootDir}...");
        (Dictionary<string, Tensor> txWeights, IReadOnlyList<SafeTensorsLoader> txLoaders) =
            Krea2CheckpointConverter.LoadTransformer(rootDir);
        _output.WriteLine($"  transformer={txWeights.Count} keys, {sw.Elapsed.TotalSeconds:F1}s");

        try
        {
            Krea2Config config = Krea2Config.Turbo;
            Krea2Transformer transformer = new(config);
            transformer.LoadWeights(txWeights);
            int splitBlock = config.NumLayers / 2;
            _output.WriteLine($"[2/5] {config.NumLayers} blocks, split at {splitBlock} (A: 0-{splitBlock - 1}, B: {splitBlock}-{config.NumLayers - 1}).");

            using CudaBackend backendA = new(deviceOrdinal: 0, ptxDir: ptxDir);   // 3060, 12 GB
            using CudaBackend backendB = new(deviceOrdinal: 1, ptxDir: ptxDir);   // 4090, 24 GB
            backendA.CacheWeightCasts = false;   // matches Krea2GenerationTests: transient fp8 dequant, not cached
            backendB.CacheWeightCasts = false;

            (nuint freeA0, nuint totalA) = backendA.Context.GetMemoryInfo();
            (nuint freeB0, nuint totalB) = backendB.Context.GetMemoryInfo();
            double freeAGb = freeA0 / (1024.0 * 1024.0 * 1024.0), freeBGb = freeB0 / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[3/5] Free VRAM right now — A (ordinal 0): {freeAGb:F2} GB / {totalA / (1024.0 * 1024.0 * 1024.0):F1} GB total; " +
                $"B (ordinal 1): {freeBGb:F2} GB / {totalB / (1024.0 * 1024.0 * 1024.0):F1} GB total.");
            if (freeAGb < 6.0 || freeBGb < 8.0)
            {
                _output.WriteLine("SKIPPED: not enough free VRAM on one or both cards for this run.");
                transformer.Dispose();
                return;
            }

            List<Tensor> aWeights = new(transformer.EnumerateSharedWeights());
            aWeights.AddRange(transformer.EnumerateBlockRangeWeights(0, splitBlock));
            List<Tensor> bWeights = new(transformer.EnumerateBlockRangeWeights(splitBlock, config.NumLayers));
            _output.WriteLine($"[4/5] Preloading — A: {aWeights.Count} tensors (shared + blocks 0-{splitBlock - 1}); " +
                $"B: {bWeights.Count} tensors (blocks {splitBlock}-{config.NumLayers - 1} ONLY, no shared weights).");
            backendA.PreloadWeights(aWeights);
            backendB.PreloadWeights(bWeights);

            (nuint freeA1, _) = backendA.Context.GetMemoryInfo();
            (nuint freeB1, _) = backendB.Context.GetMemoryInfo();
            double residentAGb = (freeA0 - freeA1) / (1024.0 * 1024.0 * 1024.0);
            double residentBGb = (freeB0 - freeB1) / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[5/5] Resident after preload — A: {residentAGb:F2} GB, B: {residentBGb:F2} GB, total: {residentAGb + residentBGb:F2} GB.");

            // 1024x1024 image → 128x128 latent (VaeChannels=16, matches real Krea2 generation shapes).
            using Tensor latent = RandF32(new TensorShape(1, config.VaeChannels, 128, 128), seed: 1, scale: 0.5f);
            const int txtSeq = 32;
            using Tensor encoderHidden = RandF32(
                new TensorShape(1, txtSeq, config.NumTextLayers * config.TextHiddenDim), seed: 2, scale: 0.2f);

            using Tensor velocity = transformer.ForwardSharded(backendA, backendB, latent, 0.5f, encoderHidden, splitBlock);
            AssertFinite(velocity);
            Assert.Equal(config.VaeChannels, (int)velocity.Shape[1]);
            Assert.Equal(128, (int)velocity.Shape[2]);
            Assert.Equal(128, (int)velocity.Shape[3]);

            _output.WriteLine($"Sharded forward produced finite {velocity.Shape} velocity in {sw.Elapsed.TotalSeconds:F1}s total. " +
                $"A resident ~{residentAGb:F1} GB (< 3060's 12 GB budget, and < the ~13 GB whole checkpoint it could not " +
                $"hold alone) — split confirmed as real VRAM pooling, not replication.");

            // The key assertion: A holds meaningfully less than the whole checkpoint would require. It doesn't need
            // to equal exactly half (norm/scale-shift tensors are tiny; block byte counts vary slightly), just
            // clearly less than what "replicate everything on both" (CFG-parallel's approach) would cost.
            Assert.True(residentAGb < 10.0, $"backend A (3060) holds {residentAGb:F1} GB resident — expected roughly half the DiT, not the whole thing.");

            backendA.FreeWeights(aWeights);
            backendB.FreeWeights(bWeights);
            transformer.Dispose();
        }
        finally
        {
            foreach (SafeTensorsLoader l in txLoaders) l.Dispose();
        }
    }

    private static unsafe Tensor RandF32(TensorShape s, int seed, float scale)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static unsafe void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }
}
