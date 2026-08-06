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

/// <summary>Flux DiT sharding v1 on the REAL dev fp8 checkpoint, one load, two phases: (1) same-device split vs
/// unsharded <see cref="FluxTransformer.Forward"/> must be BIT-EXACT (identical kernels — any drift is a
/// split/concat-seam bug across the 19 double + 38 single block boundary); (2) cross-device split must POOL
/// (neither card holds the whole ~11-12 GB fp8 DiT; the 12 GB card holds a genuine share it could not hold
/// beside activations alone) and produce finite velocity. No cross-device numeric gate here — the 4090 takes
/// native fp8 GEMM and the 3060 dequant fallback, a legitimately different numeric path; the engine e2e's SSIM
/// covers that comparison. There is no synthetic Flux harness in this repo (ROADMAP-acknowledged), so real
/// weights ARE the parity vehicle.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class FluxDitShardingVramTests
{
    private readonly ITestOutputHelper _output;
    public FluxDitShardingVramTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ForwardSharded_RealDevFp8_SameDeviceBitParity_AndCrossDevicePooling()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.Flux.Dev;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(FluxDitShardingVramTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/6] Loading Flux dev fp8 from {checkpoint}...");
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(checkpoint);
        try
        {
            (int doubles, int singles, bool hasGuidance) = FluxCheckpointConverter.DetectArchitecture(converted.Transformer);
            FluxConfig config = hasGuidance ? FluxConfig.Dev : FluxConfig.Schnell;
            _output.WriteLine($"  {doubles} double + {singles} single blocks, guidance={hasGuidance}, {sw.Elapsed.TotalSeconds:F1}s");
            using FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(converted.Transformer);
            int blockCount = transformer.BlockCount;

            // 1024² → 64×64 packed grid → 4096 img tokens; txt 512 (Dev's T5 length).
            const int hPacked = 64, wPacked = 64, txtSeq = 128;
            using Tensor packedLatent = RandF32(new TensorShape(1, hPacked * wPacked, 64), seed: 1, scale: 0.5f);
            using Tensor t5 = RandF32(new TensorShape(1, txtSeq, 4096), seed: 2, scale: 0.2f);
            using Tensor clipPooled = RandF32(new TensorShape(1, 768), seed: 3, scale: 0.2f);

            using CudaBackend backendA = new(deviceOrdinal: 0, ptxDir: ptxDir);
            (nuint freeA0, _) = backendA.Context.GetMemoryInfo();

            _output.WriteLine("[2/6] Unsharded reference forward (whole DiT resident on ordinal 0)...");
            backendA.PreloadWeights(transformer.EnumerateWeights());
            float[] reference;
            using (Tensor velocityRef = transformer.Forward(backendA, packedLatent, t5, 0.5f, clipPooled, 3.5f, txtSeq, hPacked, wPacked))
            {
                reference = ToArray(velocityRef);
            }
            backendA.FreeWeights(transformer.EnumerateWeights());

            _output.WriteLine("[3/6] Same-device sharded forward (both ranges on ordinal 0) — bit-parity gate...");
            using (CudaBackend backendB0 = new(deviceOrdinal: 0, ptxDir: ptxDir))
            {
                int split = blockCount / 2;
                List<Tensor> aW = new(transformer.EnumerateSharedWeights());
                aW.AddRange(transformer.EnumerateBlockRangeWeights(0, split));
                List<Tensor> bW = new(transformer.EnumerateBlockRangeWeights(split, blockCount));
                backendA.PreloadWeights(aW);
                backendB0.PreloadWeights(bW);
                using Tensor velocitySharded = transformer.ForwardSharded(
                    backendA, backendB0, packedLatent, t5, 0.5f, clipPooled, 3.5f, txtSeq, hPacked, wPacked, split);
                float[] sameDevice = ToArray(velocitySharded);
                backendA.FreeWeights(aW);
                backendB0.FreeWeights(bW);

                int mismatches = 0;
                for (int i = 0; i < reference.Length; i++)
                {
                    if (reference[i] != sameDevice[i]) mismatches++;
                }
                Assert.True(mismatches == 0, $"{mismatches}/{reference.Length} values differ between unsharded and same-device sharded forward.");
                _output.WriteLine($"  bit-exact over {reference.Length} velocity values.");
            }

            _output.WriteLine("[4/6] Cross-device sharded forward (ordinal 0 + ordinal 1) — pooling gate...");
            using CudaBackend backendB = new(deviceOrdinal: 1, ptxDir: ptxDir);
            backendB.CacheWeightCasts = false;   // SM 8.6 dequant fallback must not double its share
            (nuint freeB0, _) = backendB.Context.GetMemoryInfo();
            double freeAGb = freeA0 / (1024.0 * 1024.0 * 1024.0), freeBGb = freeB0 / (1024.0 * 1024.0 * 1024.0);
            if (freeBGb < 5.0)
            {
                _output.WriteLine("SKIPPED (cross-device phase): second card too full.");
                return;
            }
            long sharedBytes = 0;
            foreach (Tensor t in transformer.EnumerateSharedWeights())
                sharedBytes += t.DType.ComputeByteCount(t.ElementCount);
            long[] perBlock = new long[blockCount];
            for (int i = 0; i < blockCount; i++)
                foreach (Tensor t in transformer.EnumerateBlockRangeWeights(i, i + 1))
                    perBlock[i] += t.DType.ComputeByteCount(t.ElementCount);
            (nuint freeANow, _) = backendA.Context.GetMemoryInfo();
            int crossSplit = PlacementPlanner.DitSplitPlan([(long)freeANow, (long)freeB0], perBlock, sharedBytes)[0];
            _output.WriteLine($"[5/6] byte-weighted split at {crossSplit} of {blockCount}.");

            List<Tensor> aWeights = new(transformer.EnumerateSharedWeights());
            aWeights.AddRange(transformer.EnumerateBlockRangeWeights(0, crossSplit));
            List<Tensor> bWeights = new(transformer.EnumerateBlockRangeWeights(crossSplit, blockCount));
            (nuint fa0, _) = backendA.Context.GetMemoryInfo();
            (nuint fb0, _) = backendB.Context.GetMemoryInfo();
            backendA.PreloadWeights(aWeights);
            backendB.PreloadWeights(bWeights);
            (nuint fa1, _) = backendA.Context.GetMemoryInfo();
            (nuint fb1, _) = backendB.Context.GetMemoryInfo();
            double residentAGb = (fa0 - fa1) / (1024.0 * 1024.0 * 1024.0);
            double residentBGb = (fb0 - fb1) / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"  resident — A: {residentAGb:F2} GB, B: {residentBGb:F2} GB, total {residentAGb + residentBGb:F2} GB.");

            using Tensor velocityCross = transformer.ForwardSharded(
                backendA, backendB, packedLatent, t5, 0.5f, clipPooled, 3.5f, txtSeq, hPacked, wPacked, crossSplit);
            AssertFinite(velocityCross);
            _output.WriteLine($"[6/6] cross-device sharded forward finite in {sw.Elapsed.TotalSeconds:F1}s total.");

            Assert.True(residentAGb < 10.0, $"backend A holds {residentAGb:F1} GB — expected its split share, not the whole fp8 DiT.");
            Assert.True(residentBGb > 2.0, $"backend B holds only {residentBGb:F1} GB — the split share never landed there.");

            backendA.FreeWeights(aWeights);
            backendB.FreeWeights(bWeights);
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

    private static unsafe float[] ToArray(Tensor t)
    {
        float[] arr = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) arr[i] = p[i];
        return arr;
    }

    private static unsafe void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }
}
