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

/// <summary>The headline "doesn't fit one card" proof: MiniMax-H3's fp8 pruned DiT (~19.5 GB, 50 blocks) split
/// across the two cards by the live VRAM plan, with a real sharded forward at a small video geometry producing
/// finite video AND audio velocities. Unsharded, this DiT is marginal-to-impossible resident on the 24 GB card
/// beside its F32 activations (the pipeline degrades to streaming ~19 GB of weights per step); pooled across
/// 24+12 GB it sits fully resident with headroom. Loads through the exact recipe sequence (converter, norm/rope
/// F32 promotion, config detect) so what's tested is what ships.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed unsafe class MiniMaxH3DitShardingVramTests
{
    private readonly ITestOutputHelper _output;
    public MiniMaxH3DitShardingVramTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ForwardSharded_RealFp8Checkpoint_PoolsVramAcrossTwoGpus()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.MiniMaxH3.DitFp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(MiniMaxH3DitShardingVramTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading MiniMax-H3 fp8 DiT from {checkpoint} (recipe-identical sequence)...");
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(checkpoint);
        Dictionary<string, Tensor> raw = new Dictionary<string, Tensor>(loader.GetAllTensors());
        MiniMaxH3CheckpointConverter.ThrowIfInt8Convrot(raw);
        MiniMaxH3CheckpointConverter.ConvertedWeights converted = MiniMaxH3CheckpointConverter.Convert(raw, castToF32: false);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(converted.Transformer);
        foreach (string key in weights.Keys.ToList())
        {
            bool isNorm = key.EndsWith("norm.weight", StringComparison.Ordinal)
                || key.EndsWith("norm1.weight", StringComparison.Ordinal)
                || key.EndsWith("norm2.weight", StringComparison.Ordinal)
                || key.Equals("rope.inv_freq", StringComparison.Ordinal);
            if (isNorm && weights[key].DType != DType.F32)
            {
                weights[key] = weights[key].CastTo(DType.F32);
            }
        }
        MiniMaxH3Config config = MiniMaxH3Config.Detect(weights);
        _output.WriteLine($"  {weights.Count} tensors, {config.NumLayers} blocks, hidden {config.HiddenSize}, "
            + $"curves={config.UseAdalnCurves}, {sw.Elapsed.TotalSeconds:F1}s");

        MiniMaxH3Transformer transformer = new MiniMaxH3Transformer(config);
        transformer.LoadWeights(weights);
        try
        {
            using CudaBackend backendA = new(deviceOrdinal: 0, ptxDir: ptxDir);
            using CudaBackend backendB = new(deviceOrdinal: 1, ptxDir: ptxDir);

            (nuint freeA0, nuint totalA) = backendA.Context.GetMemoryInfo();
            (nuint freeB0, nuint totalB) = backendB.Context.GetMemoryInfo();
            double freeAGb = freeA0 / (1024.0 * 1024.0 * 1024.0), freeBGb = freeB0 / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[2/5] Free VRAM — A (ordinal 0): {freeAGb:F2}/{totalA >> 30} GB; B (ordinal 1): {freeBGb:F2}/{totalB >> 30} GB.");
            if (freeAGb + freeBGb < 26.0 || Math.Min(freeAGb, freeBGb) < 6.0)
            {
                _output.WriteLine("SKIPPED: not enough free VRAM for the ~19.5 GB pooled DiT plus F32 activations.");
                return;
            }

            long sharedBytes = 0;
            foreach (Tensor t in transformer.EnumerateSharedWeights())
                sharedBytes += t.DType.ComputeByteCount(t.ElementCount);
            int splitBlock = PlacementPlanner.DitSplitPlan([(long)freeA0, (long)freeB0], config.NumLayers, sharedBytes)[0];
            _output.WriteLine($"[3/5] {config.NumLayers} blocks, live-VRAM split at {splitBlock}; shared {sharedBytes >> 20} MB on A.");

            List<Tensor> aWeights = new(transformer.EnumerateSharedWeights());
            aWeights.AddRange(transformer.EnumerateBlockRangeWeights(0, splitBlock));
            List<Tensor> bWeights = new(transformer.EnumerateBlockRangeWeights(splitBlock, config.NumLayers));
            backendA.PreloadWeights(aWeights);
            backendB.PreloadWeights(bWeights);

            (nuint freeA1, _) = backendA.Context.GetMemoryInfo();
            (nuint freeB1, _) = backendB.Context.GetMemoryInfo();
            double residentAGb = (freeA0 - freeA1) / (1024.0 * 1024.0 * 1024.0);
            double residentBGb = (freeB0 - freeB1) / (1024.0 * 1024.0 * 1024.0);
            _output.WriteLine($"[4/5] Resident after preload — A: {residentAGb:F2} GB, B: {residentBGb:F2} GB, total: {residentAGb + residentBGb:F2} GB.");

            // Small real geometry: 256×160, 9 latent frames, 8 audio frames — the layout/rope machinery is the
            // real thing; only the token count is scaled down to keep the F32 activation footprint test-sized.
            const int latentT = 3, latentH = 16, latentW = 10, audioT = 8, textLen = 16;
            MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, latentT, latentH, latentW, audioT);
            int videoRowCount = latentT * (latentH / 2) * (latentW / 2);
            int audioRowCount = audioT * 2;
            using Tensor videoRows = RandF32(new TensorShape(videoRowCount, config.VideoPatchDim), seed: 1, scale: 0.5f);
            using Tensor audioRows = RandF32(new TensorShape(audioRowCount, config.AudioLatentsDim), seed: 2, scale: 0.5f);
            using Tensor text = RandF32(new TensorShape(textLen, config.TextDim), seed: 3, scale: 0.2f);
            (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(
                layout.PositionIds, transformer.RopeInvFreq(), config.AttentionHeadDim);
            float[] uniqueT = [0.3f, 0.55f];
            Dictionary<MiniMaxH3SegmentKind, int> rowOf = new()
            {
                [MiniMaxH3SegmentKind.Text] = 0,
                [MiniMaxH3SegmentKind.Video] = 0,
                [MiniMaxH3SegmentKind.Cond] = 0,
                [MiniMaxH3SegmentKind.RefImage] = 0,
                [MiniMaxH3SegmentKind.Audio] = 1,
                [MiniMaxH3SegmentKind.RefAudio] = 1,
            };

            (Tensor video, Tensor audio) = transformer.ForwardSharded(
                backendA, backendB, layout, videoRows, audioRows, text, cos, sin, uniqueT, rowOf, splitBlock);
            try
            {
                AssertFinite(video, "video");
                AssertFinite(audio, "audio");
                Assert.Equal(videoRowCount, (int)video.Shape[0]);
                Assert.Equal(audioRowCount, (int)audio.Shape[0]);
                _output.WriteLine($"[5/5] Sharded forward produced finite video {video.Shape} + audio {audio.Shape} in {sw.Elapsed.TotalSeconds:F1}s total.");
            }
            finally
            {
                video.Dispose(); audio.Dispose(); cos.Dispose(); sin.Dispose();
            }

            // Pooling proof: neither card holds the whole ~19.5 GB DiT; the 12 GB card holds a genuine share.
            Assert.True(residentAGb < 17.0, $"backend A holds {residentAGb:F1} GB — expected its split share, not the whole ~19.5 GB DiT.");
            Assert.True(residentBGb > 2.0, $"backend B holds only {residentBGb:F1} GB — the split share never landed there.");
            Assert.True(residentAGb + residentBGb > 15.0, $"combined resident {residentAGb + residentBGb:F1} GB is implausibly small for the ~19.5 GB checkpoint.");

            backendA.FreeWeights(aWeights);
            backendB.FreeWeights(bWeights);
        }
        finally
        {
            transformer.Dispose();
        }
    }

    private static Tensor RandF32(TensorShape s, int seed, float scale)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private void AssertFinite(Tensor t, string label)
    {
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float* p = (float*)f.DataPointer;
        for (long i = 0; i < f.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"{label}: non-finite at {i}");
        if (!ReferenceEquals(f, t)) f.Dispose();
    }
}
