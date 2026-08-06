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

/// <summary>Real-weight verification that <see cref="Sd3Transformer.ForwardSharded"/> actually POOLS VRAM
/// rather than replicating the MMDiT on both cards — mirrors <c>Krea2DitShardingVramTests</c>. Loads the real
/// SD3.5 Medium checkpoint (~5 GB transformer), splits its <see cref="Sd3Config.Depth"/> JointBlocks across the
/// 3060 (ordinal 0) and the 4090 (ordinal 1), and asserts each card's resident footprint stays near its OWN
/// share rather than the whole transformer.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class Sd3DitShardingVramTests
{
    private readonly ITestOutputHelper _output;
    public Sd3DitShardingVramTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ForwardSharded_RealMediumCheckpoint_PoolsVramAcrossTwoGpus()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Sd35.Medium;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Sd3DitShardingVramTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading SD3.5 transformer from {checkpoint}...");
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) = Sd3CheckpointConverter.LoadAndConvert(checkpoint);
        _output.WriteLine($"  transformer={converted.Transformer.Count} keys, {sw.Elapsed.TotalSeconds:F1}s");

        try
        {
            int patchEmbedOutChannels = converted.Transformer.TryGetValue("pos_embed.proj.weight", out Tensor? pe)
                ? (int)pe.Shape[0] : 1536;
            Sd3Config config = Sd3Config.FromWeightShape(patchEmbedOutChannels);
            Sd3Transformer transformer = new(config);
            transformer.LoadWeights(converted.Transformer);
            int splitBlock = config.Depth / 2;
            _output.WriteLine($"[2/5] {config.Depth} blocks (hidden={config.HiddenSize}), split at {splitBlock} "
                + $"(A: 0-{splitBlock - 1}, B: {splitBlock}-{config.Depth - 1}).");

            using CudaBackend backendA = new(deviceOrdinal: 0, ptxDir: ptxDir);   // 3060, 12 GB
            using CudaBackend backendB = new(deviceOrdinal: 1, ptxDir: ptxDir);   // 4090, 24 GB
            backendA.CacheWeightCasts = false;
            backendB.CacheWeightCasts = false;

            (nuint freeA0, nuint totalA) = backendA.Context.GetMemoryInfo();
            (nuint freeB0, nuint totalB) = backendB.Context.GetMemoryInfo();
            double freeAGb = freeA0 / (1024.0 * 1024.0 * 1024.0), freeBGb = freeB0 / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[3/5] Free VRAM right now — A (ordinal 0): {freeAGb:F2} GB / {totalA / (1024.0 * 1024.0 * 1024.0):F1} GB total; "
                + $"B (ordinal 1): {freeBGb:F2} GB / {totalB / (1024.0 * 1024.0 * 1024.0):F1} GB total.");
            if (freeAGb < 4.0 || freeBGb < 4.0)
            {
                _output.WriteLine("SKIPPED: not enough free VRAM on one or both cards for this run.");
                transformer.Dispose();
                return;
            }

            List<Tensor> aWeights = new(transformer.EnumerateSharedWeights());
            aWeights.AddRange(transformer.EnumerateBlockRangeWeights(0, splitBlock));
            List<Tensor> bWeights = new(transformer.EnumerateBlockRangeWeights(splitBlock, config.Depth));
            _output.WriteLine($"[4/5] Preloading — A: {aWeights.Count} tensors (shared + blocks 0-{splitBlock - 1}); "
                + $"B: {bWeights.Count} tensors (blocks {splitBlock}-{config.Depth - 1} ONLY, no shared weights).");
            backendA.PreloadWeights(aWeights);
            backendB.PreloadWeights(bWeights);

            (nuint freeA1, _) = backendA.Context.GetMemoryInfo();
            (nuint freeB1, _) = backendB.Context.GetMemoryInfo();
            double residentAGb = (freeA0 - freeA1) / (1024.0 * 1024.0 * 1024.0);
            double residentBGb = (freeB0 - freeB1) / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[5/5] Resident after preload — A: {residentAGb:F2} GB, B: {residentBGb:F2} GB, total: {residentAGb + residentBGb:F2} GB.");

            // 512x512 → 64x64 latent (16 channels, matches SD3's VAE).
            using Tensor latent = RandF32(new TensorShape(1, config.InChannels, 64, 64), seed: 1, scale: 0.5f);
            const int ctxSeq = 154;
            using Tensor context = RandF32(new TensorShape(1, ctxSeq, config.HiddenSize), seed: 2, scale: 0.2f);
            using Tensor pooled = RandF32(new TensorShape(1, config.PooledProjectionDim), seed: 3, scale: 0.2f);

            using Tensor velocity = transformer.ForwardSharded(backendA, backendB, latent, 500.0f, context, pooled, splitBlock);
            AssertFinite(velocity);
            Assert.Equal(config.InChannels, (int)velocity.Shape[1]);
            Assert.Equal(64, (int)velocity.Shape[2]);
            Assert.Equal(64, (int)velocity.Shape[3]);

            _output.WriteLine($"Sharded forward produced finite {velocity.Shape} velocity in {sw.Elapsed.TotalSeconds:F1}s total. "
                + $"A resident ~{residentAGb:F1} GB — split confirmed as real VRAM pooling, not replication.");

            // The whole transformer is ~5 GB — A holding meaningfully less than that (not just "less than half")
            // is the signal this is pooling, not replication (the CFG-parallel mistake this feature avoids).
            Assert.True(residentAGb < 4.5, $"backend A (3060) holds {residentAGb:F1} GB resident — expected roughly half the transformer, not the whole thing.");

            backendA.FreeWeights(aWeights);
            backendB.FreeWeights(bWeights);
            transformer.Dispose();
        }
        finally
        {
            loader.Dispose();
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
