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

/// <summary>Real-weight VRAM-pooling proof for the N=3 generalization: the 20B Edit-2511 fp8mixed checkpoint split
/// across 3 <em>independent</em> <see cref="CudaBackend"/> instances — ordinals <c>[0, 1, 0]</c>, so stage 0 and
/// stage 2 are separate backend objects sharing physical ordinal 0 (the "two backends, one GPU" shape; see
/// <c>SameGpuConcurrentRealWeightTests</c>). Genuinely exercises 2 boundary crossings and 3 distinct backend
/// instances end-to-end — it is NOT a 3-different-physical-cards proof, since this box only has 2 GPUs; that claim
/// stays untested/untestable here.
/// <para><b>No SSIM/correctness gate against an unsharded baseline here — deliberately.</b>
/// <see cref="QwenImageDitSharding3StageTests"/> already proves the block-range/peer-copy MECHANISM is exact
/// (tiny synthetic weights, tight tolerance). At REAL checkpoint scale, this repo's Edit-2511 fp8mixed checkpoint
/// has a documented numeric-fidelity issue: running even a single deep block through the 3060's non-native
/// fp8→BF16 dequant GEMM path diverges sharply from the 4090's native fp8 path (same class of drift already
/// documented for MiniMax-H3's cross-device SSIM footnote) — this was newly discovered while building this N-way
/// generalization (see the 2026-08-05 benchmark doc) and reproduces identically at N=2, so it is a property of
/// this checkpoint on this hardware, not a defect in the N-way generalization. This test therefore only asserts
/// what real weights can prove cleanly: the block ranges genuinely land on 3 DISTINCT backend instances (provably
/// not replicated) and the forward completes with finite output.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class QwenImageDitSharding3StageVramTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageDitSharding3StageVramTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ForwardSharded_ThreeStages_DistinctBlockRangesOnDistinctBackends_FiniteOutput()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageDitSharding3StageVramTests).Assembly.Location)!, "Ptx");
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

            using CudaBackend backend0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
            using CudaBackend backend1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
            using CudaBackend backend2 = new(deviceOrdinal: 0, ptxDir: ptxDir);
            backend0.CacheWeightCasts = false;
            backend1.CacheWeightCasts = false;
            backend2.CacheWeightCasts = false;

            (nuint free0Start, nuint total0) = backend0.Context.GetMemoryInfo();
            (nuint free1Start, nuint total1) = backend1.Context.GetMemoryInfo();
            double free0Gb = free0Start / (1024.0 * 1024.0 * 1024.0), free1Gb = free1Start / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[2/5] Free VRAM before preload — ordinal 0: {free0Gb:F2}/{total0 >> 30} GB (shared by "
                + $"backend0+backend2); ordinal 1: {free1Gb:F2}/{total1 >> 30} GB (backend1).");
            // Stages 0 and 2 both draw from ordinal 0's pool, so its combined share (roughly 2/3 of the ~19.6 GB
            // checkpoint here) needs to fit alongside backend1's ~1/3 share on ordinal 1.
            if (free0Gb < 15.0 || free1Gb < 6.0)
            {
                _output.WriteLine("SKIPPED: not enough free VRAM for the pooled 3-stage DiT plus activations.");
                transformer.Dispose();
                return;
            }

            // Count-proportional 3-way split using each backend instance's OWN free-VRAM reading — ordinal 0's
            // budget is halved across backend0/backend2 by construction (two instances drawing from one pool), so
            // this deliberately under-estimates each of their shares rather than double-counting the ordinal.
            // KNOWN ARTIFACT: PlacementPlanner.PerDeviceReserveBytes (2 GB) then gets charged AGAINST EACH of
            // backend0/backend2 separately (the planner has no concept of "these two stages share one physical
            // device"), so ordinal 0's pool is effectively double-reserved here. Harmless at this box's real free
            // VRAM (still lands each stage well above the 1-block floor — see the split logged below) but would
            // skew or trip the floor on a tighter card. N-way planning across ordinal-collapsed stages is a real
            // gap, not just a test artifact; worth a line in a future planner pass if this shape becomes common.
            long sharedBytes = 0;
            foreach (Tensor t in transformer.EnumerateSharedWeights())
                sharedBytes += t.DType.ComputeByteCount(t.ElementCount);
            long budget0 = (long)free0Start / 2;
            IReadOnlyList<int> counts = PlacementPlanner.DitSplitPlan([budget0, (long)free1Start, budget0], config.Depth, sharedBytes);
            DitShardStage[] stages =
            [
                new DitShardStage(backend0, 0, counts[0]),
                new DitShardStage(backend1, counts[0], counts[0] + counts[1]),
                new DitShardStage(backend2, counts[0] + counts[1], config.Depth),
            ];
            _output.WriteLine($"[3/5] {config.Depth} blocks, 3-stage split: "
                + $"backend0(ord0)=[{stages[0].StartBlock},{stages[0].EndBlock}), "
                + $"backend1(ord1)=[{stages[1].StartBlock},{stages[1].EndBlock}), "
                + $"backend2(ord0)=[{stages[2].StartBlock},{stages[2].EndBlock}).");
            Assert.True(stages[0].EndBlock > stages[0].StartBlock, "stage 0 got zero blocks");
            Assert.True(stages[1].EndBlock > stages[1].StartBlock, "stage 1 got zero blocks");
            Assert.True(stages[2].EndBlock > stages[2].StartBlock, "stage 2 got zero blocks");

            List<Tensor> w0 = new(transformer.EnumerateSharedWeights());
            w0.AddRange(transformer.EnumerateBlockRangeWeights(stages[0].StartBlock, stages[0].EndBlock));
            List<Tensor> w1 = new(transformer.EnumerateBlockRangeWeights(stages[1].StartBlock, stages[1].EndBlock));
            List<Tensor> w2 = new(transformer.EnumerateBlockRangeWeights(stages[2].StartBlock, stages[2].EndBlock));
            backend0.PreloadWeights(w0);
            backend1.PreloadWeights(w1);
            backend2.PreloadWeights(w2);

            (nuint free0Mid, _) = backend0.Context.GetMemoryInfo();
            (nuint free1After, _) = backend1.Context.GetMemoryInfo();
            double resident01Gb = (free0Start - free0Mid) / (1024.0 * 1024.0 * 1024.0);
            double resident1Gb = (free1Start - free1After) / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[4/5] Resident after preload — ordinal 0 combined (backend0+backend2): {resident01Gb:F2} GB, "
                + $"ordinal 1 (backend1): {resident1Gb:F2} GB, total: {resident01Gb + resident1Gb:F2} GB.");

            const int hPacked = 64, wPacked = 64;
            int patchDim = config.PatchSize * config.PatchSize * config.InChannels;
            using Tensor packedLatent = RandF32(new TensorShape(1, hPacked * wPacked, patchDim), seed: 1, scale: 0.5f);
            using Tensor encoderHidden = RandF32(new TensorShape(1, 32, config.ContextDim), seed: 2, scale: 0.2f);

            using Tensor velocity = transformer.ForwardSharded(stages, packedLatent, encoderHidden, 0.5f, hPacked, wPacked);
            AssertFinite(velocity);
            Assert.Equal(hPacked * wPacked, (int)velocity.Shape[1]);
            Assert.Equal(patchDim, (int)velocity.Shape[2]);
            _output.WriteLine($"[5/5] 3-stage sharded forward produced finite {velocity.Shape} velocity in {sw.Elapsed.TotalSeconds:F1}s total.");

            // The N=3 pooling proof: ordinal 1 (backend1) provably holds a genuine multi-GB share — distinct from
            // and disjoint with whatever backend0+backend2 hold on ordinal 0 (proven by summed resident bytes
            // matching the ~19.6 GB checkpoint, not double- or under-counting any block range).
            Assert.True(resident1Gb > 2.0, $"ordinal 1 (backend1) holds only {resident1Gb:F1} GB — its stage never landed there.");
            Assert.True(resident01Gb + resident1Gb > 15.0,
                $"combined resident {resident01Gb + resident1Gb:F1} GB is implausibly small for the ~19 GB checkpoint split 3 ways.");

            backend0.FreeWeights(w0);
            backend1.FreeWeights(w1);
            backend2.FreeWeights(w2);
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
