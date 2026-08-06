using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>N-stage generalization proof for <see cref="QwenImageTransformer.ForwardSharded(System.Collections.Generic.IReadOnlyList{DitShardStage},Tensor,Tensor,float,int,int,ValueTuple{int,int}[],bool)"/>:
/// a 3-stage block-range split — <c>[cuda:0, cuda:1, cuda:0]</c> — run through 3 <em>independent</em>
/// <see cref="CudaBackend"/> instances, matching <see cref="QwenImageDitShardingTests"/>'s tiny-synthetic-weight
/// parity pattern (that class is the N=2 proof this generalizes).
/// <para><b>Hardware-honesty note (this box has exactly 2 physical GPUs):</b> this test genuinely exercises TWO
/// boundary crossings and THREE distinct backend instances end-to-end (stage 0 and stage 2 are separate
/// <see cref="CudaBackend"/> objects that happen to share physical ordinal 0 — the same "two backends, one GPU"
/// shape as <c>SameGpuConcurrentRealWeightTests</c>). It does NOT exercise 3 DIFFERENT physical cards — that claim
/// is untested/untestable here and is not implied by this test passing. Real-weight checkpoint scale is
/// deliberately NOT used here (see <see cref="QwenImageDitSharding3StageVramTests"/> for that): this repo's
/// Edit-2511 fp8mixed checkpoint has a documented numeric-fidelity issue running (even a single block of) its
/// deep-block activations through the 3060's non-native fp8→BF16 dequant GEMM path (same class of drift already
/// documented for MiniMax-H3), which swamps any N=2-vs-N=3 signal at real scale — see the 2026-08-05 benchmark
/// doc entry. Tiny random weights avoid that confound entirely and isolate the thing this test is actually
/// proving: the block-range tiling / peer-copy boundary math, not fp8 numeric fidelity.</para></summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed class QwenImageDitSharding3StageTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageDitSharding3StageTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    // 6 blocks so a 3-way split gives each stage real work (2 blocks each); HeadDim must stay 128 (rope axis split).
    private static QwenImageConfig TinyConfig => new()
    {
        HiddenSize = 128,
        NumHeads = 1,
        HeadDim = 128,
        Depth = 6,
        MlpRatio = 2.0f,
        ContextDim = 32,
        InChannels = 16,
        PatchSize = 2,
    };

    [Fact]
    public void ForwardSharded_ThreeStages_TwoBoundaryCrossings_OneGpuHostingTwoBackends_MatchesUnsharded()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        int secondOrdinal = CudaContext.GetDeviceCount() >= 2 ? 1 : 0;
        if (secondOrdinal == 0)
            _output.WriteLine("Only 1 physical GPU visible — stages 1 collapses onto ordinal 0 too (still 3 distinct backend instances, 0 cross-device boundaries).");

        QwenImageConfig cfg = TinyConfig;
        const int hPacked = 4, wPacked = 4;
        int patchDim = cfg.PatchSize * cfg.PatchSize * cfg.InChannels;
        int mainSeq = hPacked * wPacked;
        const int txtSeq = 5;

        using Tensor packedLatent = QwenImageWeightBuilder.Rand(new TensorShape(1, mainSeq, patchDim), 100, 0.5f);
        using Tensor encoderHidden = QwenImageWeightBuilder.Rand(new TensorShape(1, txtSeq, cfg.ContextDim), 200, 0.2f);

        // ── Reference: one backend, whole DiT resident, unsharded Forward. ──
        Dictionary<string, Tensor> wRef = QwenImageWeightBuilder.Build(cfg);
        float[] refValues;
        using (CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir()))
        using (QwenImageTransformer refTransformer = new(cfg))
        {
            refTransformer.LoadWeights(wRef);
            refBackend.PreloadWeights(refTransformer.EnumerateWeights());
            using Tensor velocityRef = refTransformer.Forward(refBackend, packedLatent, encoderHidden, 0.7f, hPacked, wPacked);
            refValues = ToArray(velocityRef);
            refBackend.FreeWeights(refTransformer.EnumerateWeights());
        }
        QwenImageWeightBuilder.DisposeAll(wRef);

        // ── Sharded: 3 stages [0,2) [2,4) [4,6) on 3 INDEPENDENT backend instances, ordinals [0, secondOrdinal, 0].
        // Stage 0 and stage 2 are separate CudaBackend objects sharing ordinal 0 — two genuine boundary crossings. ──
        Dictionary<string, Tensor> wSharded = QwenImageWeightBuilder.Build(cfg);
        float[] shardedValues;
        long peerCopiesBefore, peerCopiesAfter;
        using (CudaBackend backend0 = new(deviceOrdinal: 0, PtxDir()))
        using (CudaBackend backend1 = new(deviceOrdinal: secondOrdinal, PtxDir()))
        using (CudaBackend backend2 = new(deviceOrdinal: 0, PtxDir()))
        using (QwenImageTransformer shardedTransformer = new(cfg))
        {
            shardedTransformer.LoadWeights(wSharded);
            int third = cfg.Depth / 3;
            DitShardStage[] stages =
            [
                new DitShardStage(backend0, 0, third),
                new DitShardStage(backend1, third, 2 * third),
                new DitShardStage(backend2, 2 * third, cfg.Depth),
            ];
            List<Tensor> w0 = new(shardedTransformer.EnumerateSharedWeights());
            w0.AddRange(shardedTransformer.EnumerateBlockRangeWeights(stages[0].StartBlock, stages[0].EndBlock));
            backend0.PreloadWeights(w0);
            List<Tensor> w1 = new(shardedTransformer.EnumerateBlockRangeWeights(stages[1].StartBlock, stages[1].EndBlock));
            backend1.PreloadWeights(w1);
            List<Tensor> w2 = new(shardedTransformer.EnumerateBlockRangeWeights(stages[2].StartBlock, stages[2].EndBlock));
            backend2.PreloadWeights(w2);

            peerCopiesBefore = backend0.GetPeerCopyCount() + backend1.GetPeerCopyCount() + backend2.GetPeerCopyCount();
            using Tensor velocitySharded = shardedTransformer.ForwardSharded(stages, packedLatent, encoderHidden, 0.7f, hPacked, wPacked);
            peerCopiesAfter = backend0.GetPeerCopyCount() + backend1.GetPeerCopyCount() + backend2.GetPeerCopyCount();
            shardedValues = ToArray(velocitySharded);

            backend0.FreeWeights(w0);
            backend1.FreeWeights(w1);
            backend2.FreeWeights(w2);
        }
        QwenImageWeightBuilder.DisposeAll(wSharded);
        _output.WriteLine($"3-stage [0,{secondOrdinal},0]: P2P peer copies observed: {peerCopiesAfter - peerCopiesBefore} "
            + "(0 = host-staged fallback, expected on this non-P2P box).");

        Assert.Equal(refValues.Length, shardedValues.Length);
        double scale = 0;
        foreach (float v in refValues) scale = Math.Max(scale, Math.Abs(v));
        double maxErr = 0;
        for (int i = 0; i < refValues.Length; i++)
            maxErr = Math.Max(maxErr, Math.Abs(refValues[i] - shardedValues[i]));
        double relToScale = maxErr / Math.Max(scale, 1e-6);
        _output.WriteLine($"3-stage vs unsharded: max abs err {maxErr:E3}, output scale {scale:E3}, err/scale {relToScale:E3}");
        Assert.True(relToScale < 1e-3, $"3-stage sharded forward diverged (err/scale {relToScale:E3}).");
    }

    private static unsafe float[] ToArray(Tensor t)
    {
        float[] arr = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) arr[i] = p[i];
        return arr;
    }
}
