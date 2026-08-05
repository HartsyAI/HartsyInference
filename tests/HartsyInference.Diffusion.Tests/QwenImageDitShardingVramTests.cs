using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Placement;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight proof that Qwen-Image DiT sharding POOLS VRAM: the 20B Edit-2511 fp8mixed checkpoint
/// (~19 GB of transformer, loaded as T2I — the recipe defers edit vision conditioning, and the transformer
/// architecture is identical) is split across both cards by the live byte-proportional plan and a real 1024²-shaped
/// forward runs with NEITHER card holding the whole DiT. This is the headline "a model that cannot fit the 4090
/// alone now fits" case: ~19 GB resident + activations does not coexist on a 24 GB card without streaming, and
/// certainly not on the 12 GB card that here holds a genuine share. Skips cleanly when the checkpoint is absent
/// (or fails under HARTSY_REQUIRE_REAL_WEIGHTS=1) or free VRAM is short.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class QwenImageDitShardingVramTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageDitShardingVramTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ForwardSharded_RealEditFp8Checkpoint_PoolsVramAcrossTwoGpus()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageDitShardingVramTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading Qwen-Image Edit fp8 transformer from {checkpoint}...");
        (QwenImageCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            QwenImageCheckpointConverter.LoadAndConvert(checkpoint);
        _output.WriteLine($"  transformer={converted.Transformer.Count} keys, {sw.Elapsed.TotalSeconds:F1}s");

        try
        {
            QwenImageConfig config = QwenImageConfig.V1;
            QwenImageTransformer transformer = new(config);
            transformer.LoadWeights(converted.Transformer);

            using CudaBackend backendA = new(deviceOrdinal: 0, ptxDir: ptxDir);
            using CudaBackend backendB = new(deviceOrdinal: 1, ptxDir: ptxDir);
            // Transient fp8 dequant on both — cast-caching would roughly double each card's share.
            backendA.CacheWeightCasts = false;
            backendB.CacheWeightCasts = false;

            (nuint freeA0, nuint totalA) = backendA.Context.GetMemoryInfo();
            (nuint freeB0, nuint totalB) = backendB.Context.GetMemoryInfo();
            double freeAGb = freeA0 / (1024.0 * 1024.0 * 1024.0), freeBGb = freeB0 / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[2/5] Free VRAM — A (ordinal 0): {freeAGb:F2}/{totalA >> 30} GB; B (ordinal 1): {freeBGb:F2}/{totalB >> 30} GB.");
            if (freeAGb + freeBGb < 25.0 || Math.Min(freeAGb, freeBGb) < 6.0)
            {
                _output.WriteLine("SKIPPED: not enough free VRAM for the ~19 GB pooled DiT plus activations.");
                transformer.Dispose();
                return;
            }

            long sharedBytes = 0;
            foreach (Tensor t in transformer.EnumerateSharedWeights())
                sharedBytes += t.DType.ComputeByteCount(t.ElementCount);
            int splitBlock = PlacementPlanner.DitSplitPlan((long)freeA0, (long)freeB0, config.Depth, sharedBytes);
            _output.WriteLine($"[3/5] {config.Depth} blocks, live-VRAM split at {splitBlock} (A: 0-{splitBlock - 1} + shared, B: {splitBlock}-{config.Depth - 1}).");

            List<Tensor> aWeights = new(transformer.EnumerateSharedWeights());
            aWeights.AddRange(transformer.EnumerateBlockRangeWeights(0, splitBlock));
            List<Tensor> bWeights = new(transformer.EnumerateBlockRangeWeights(splitBlock, config.Depth));
            backendA.PreloadWeights(aWeights);
            backendB.PreloadWeights(bWeights);

            (nuint freeA1, _) = backendA.Context.GetMemoryInfo();
            (nuint freeB1, _) = backendB.Context.GetMemoryInfo();
            double residentAGb = (freeA0 - freeA1) / (1024.0 * 1024.0 * 1024.0);
            double residentBGb = (freeB0 - freeB1) / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[4/5] Resident after preload — A: {residentAGb:F2} GB, B: {residentBGb:F2} GB, total: {residentAGb + residentBGb:F2} GB.");

            // 1024² image → 128×128 latent → 64×64 packed grid → 4096 tokens of patchDim 64.
            const int hPacked = 64, wPacked = 64;
            int patchDim = config.PatchSize * config.PatchSize * config.InChannels;
            using Tensor packedLatent = RandF32(new TensorShape(1, hPacked * wPacked, patchDim), seed: 1, scale: 0.5f);
            using Tensor encoderHidden = RandF32(new TensorShape(1, 32, config.ContextDim), seed: 2, scale: 0.2f);

            using Tensor velocity = transformer.ForwardSharded(
                backendA, backendB, packedLatent, encoderHidden, 0.5f, hPacked, wPacked, splitBlock);
            AssertFinite(velocity);
            Assert.Equal(hPacked * wPacked, (int)velocity.Shape[1]);
            Assert.Equal(patchDim, (int)velocity.Shape[2]);

            _output.WriteLine($"[5/5] Sharded forward produced finite {velocity.Shape} velocity in {sw.Elapsed.TotalSeconds:F1}s total.");

            // The pooling proof: neither card holds the whole ~19 GB DiT, and the 12 GB-class card holds a
            // genuine multi-GB share it could never hold alongside the rest.
            Assert.True(residentAGb < 17.0, $"backend A holds {residentAGb:F1} GB — expected its split share, not the whole ~19 GB DiT.");
            Assert.True(residentBGb > 2.0, $"backend B holds only {residentBGb:F1} GB — the split share never landed there.");
            Assert.True(residentAGb + residentBGb > 15.0, $"combined resident {residentAGb + residentBGb:F1} GB is implausibly small for the ~19 GB checkpoint.");

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
