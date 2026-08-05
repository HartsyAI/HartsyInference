using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DiT-sharding parity for MiniMax-H3: <see cref="MiniMaxH3Transformer.ForwardSharded"/> — a block-range
/// split of the single-stream packed loop across two <see cref="CudaBackend"/>s — vs
/// <see cref="MiniMaxH3Transformer.Forward"/> on the tiny synthetic config <c>MiniMaxH3ForwardTests</c> uses
/// (bumped to 4 blocks so the split point has real work on both sides). Same two bars as the Qwen-Image twin:
/// same-ordinal split is BIT-EXACT; cross-device gets a scale-relative tolerance for legitimate
/// cross-architecture GEMM reduction-order drift. Covers both checkpoint families — the curve-table variant
/// (the shippable fp8 pruned build) and the sinusoidal time embedder (whose tEmb is device-built on A and must
/// cross via CopyFromPeer).</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed unsafe class MiniMaxH3DitShardingTests
{
    private readonly ITestOutputHelper _output;
    public MiniMaxH3DitShardingTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static int _seed = 11;

    private static Tensor Rand(params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            _seed = _seed * 1103515245 + 12345;
            p[i] = ((_seed >> 16) & 0x7fff) / 32768f * 0.04f - 0.02f;
        }
        return t;
    }

    private static Tensor Ones(params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = 1f;
        return t;
    }

    // MiniMaxH3ForwardTests' tiny config with 4 blocks instead of 2 (the split at 2 needs work on both sides).
    private static MiniMaxH3Config TinyConfig(bool curves) => new MiniMaxH3Config
    {
        HiddenSize = 32,
        NumLayers = 4,
        TokenRefinerNumLayers = 1,
        NumAttentionHeads = 2,
        AttentionHeadDim = 16,
        FfnHiddenSize = 48,
        LatentsDim = 24,
        AudioLatentsDim = 32,
        TextDim = 20,
        TimestepInputDim = 8,
        TimeEmbedHiddenSize = 16,
        TimeEmbedDim = curves ? 4 : 12,
        RopeInvFreqLen = 2,
        AdalnCurveGrid = curves ? 17 : null,
    };

    private static Dictionary<string, Tensor> BuildWeights(MiniMaxH3Config c)
    {
        int h = c.HiddenSize, inner = c.NumAttentionHeads * c.AttentionHeadDim;
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>
        {
            ["video_patch_proj.weight"] = Rand(h, c.VideoPatchDim),
            ["video_patch_proj.bias"] = Rand(h),
            ["audio_patch_proj.weight"] = Rand(h, c.AudioLatentsDim),
            ["audio_patch_proj.bias"] = Rand(h),
            ["condition_proj.weight"] = Rand(h, c.TextDim),
            ["condition_proj.bias"] = Rand(h),
            ["rope.inv_freq"] = Rand(c.RopeInvFreqLen),
            ["token_refiner.final_norm.weight"] = Ones(h),
            ["final_layer.norm.weight"] = Ones(h),
            ["final_layer.adaln_proj.linear.weight"] = Rand(2 * h * 1, c.TimeEmbedDim),
            ["final_layer.adaln_proj.linear.bias"] = Rand(2 * h * 1),
            ["final_layer.video_out.weight"] = Rand(c.VideoPatchDim, h),
            ["final_layer.video_out.bias"] = Rand(c.VideoPatchDim),
            ["final_layer.audio_out.weight"] = Rand(c.AudioLatentsDim, h),
            ["final_layer.audio_out.bias"] = Rand(c.AudioLatentsDim),
        };
        if (c.UseAdalnCurves)
        {
            w["adaln_t_table"] = Rand(c.AdalnCurveGrid!.Value, c.TimeEmbedDim);
        }
        else
        {
            w["time_embedder.proj_in.weight"] = Rand(c.TimeEmbedHiddenSize, c.TimestepInputDim);
            w["time_embedder.proj_in.bias"] = Rand(c.TimeEmbedHiddenSize);
            w["time_embedder.proj_out.weight"] = Rand(c.TimeEmbedDim, c.TimeEmbedHiddenSize);
            w["time_embedder.proj_out.bias"] = Rand(c.TimeEmbedDim);
        }
        for (int i = 0; i < c.TokenRefinerNumLayers; i++)
        {
            AddBlock(w, $"token_refiner.blocks.{i}", c, h, inner, adaln: false);
        }
        for (int i = 0; i < c.NumLayers; i++)
        {
            AddBlock(w, $"blocks.{i}", c, h, inner, adaln: true);
        }
        return w;
    }

    private static void AddBlock(Dictionary<string, Tensor> w, string p, MiniMaxH3Config c, int h, int inner, bool adaln)
    {
        w[$"{p}.norm1.weight"] = Ones(h);
        w[$"{p}.norm2.weight"] = Ones(h);
        w[$"{p}.attn.qkv_proj.weight"] = Rand(inner * 3, h);
        w[$"{p}.attn.q_norm.weight"] = Ones(c.AttentionHeadDim);
        w[$"{p}.attn.k_norm.weight"] = Ones(c.AttentionHeadDim);
        w[$"{p}.attn.out_proj.weight"] = Rand(h, inner);
        w[$"{p}.mlp.fc1.weight"] = Rand(c.FfnHiddenSize * 2, h);
        w[$"{p}.mlp.fc2.weight"] = Rand(h, c.FfnHiddenSize);
        if (adaln)
        {
            w[$"{p}.adaln_proj.linear.weight"] = Rand(6 * h * 3, c.TimeEmbedDim);
            w[$"{p}.adaln_proj.linear.bias"] = Rand(6 * h * 3);
        }
    }

    [Theory]
    [InlineData(true)]    // pruned/curve variant — the fp8 build sharding actually targets
    [InlineData(false)]   // sinusoidal variant — tEmb is device-built on A and crosses via CopyFromPeer
    public void ForwardSharded_SameDevice_MatchesUnsharded_BitParity(bool curves)
    {
        RunParityCase(curves, secondOrdinal: 0, exact: true);
    }

    [Fact]
    public void ForwardSharded_CrossDevice_MatchesUnsharded_WithinTolerance()
    {
        if (CudaContext.IsAvailable() && CudaContext.GetDeviceCount() < 2)
        {
            _output.WriteLine("SKIPPED: needs 2 physical GPUs for the cross-device case.");
            return;
        }
        RunParityCase(curves: true, secondOrdinal: 1, exact: false);
    }

    private void RunParityCase(bool curves, int secondOrdinal, bool exact)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        _output.WriteLine($"Sharded backend B on ordinal {secondOrdinal}; curves={curves}; exact={exact}.");

        MiniMaxH3Config cfg = TinyConfig(curves);
        const int splitBlock = 2;
        const int textLen = 3, latentT = 2, latentH = 4, latentW = 6, audioT = 2;
        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, latentT, latentH, latentW, audioT);
        int videoRowCount = latentT * (latentH / 2) * (latentW / 2);
        int audioRowCount = audioT * 2;

        _seed = 11;
        using Tensor videoRows = Rand(videoRowCount, cfg.VideoPatchDim);
        using Tensor audioRows = Rand(audioRowCount, cfg.AudioLatentsDim);
        using Tensor text = Rand(textLen, cfg.TextDim);
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

        // ── Reference: one backend, everything resident, unsharded Forward. ──
        _seed = 101;
        float[] refVideo, refAudio;
        {
            using CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir());
            using MiniMaxH3Transformer refDit = new(cfg);
            refDit.LoadWeights(BuildWeights(cfg));
            (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(
                layout.PositionIds, refDit.RopeInvFreq(), cfg.AttentionHeadDim);
            refBackend.PreloadWeights(refDit.EnumerateWeights());
            (Tensor video, Tensor audio) = refDit.Forward(
                refBackend, layout, videoRows, audioRows, text, cos, sin, uniqueT, rowOf);
            refVideo = ToArray(video);
            refAudio = ToArray(audio);
            video.Dispose(); audio.Dispose(); cos.Dispose(); sin.Dispose();
            refBackend.FreeWeights(refDit.EnumerateWeights());
        }

        // ── Sharded: SAME weight values (same seed), asymmetric preload across two backends. ──
        _seed = 101;
        float[] shVideo, shAudio;
        {
            using CudaBackend backendA = new(deviceOrdinal: 0, PtxDir());
            using CudaBackend backendB = new(deviceOrdinal: secondOrdinal, PtxDir());
            using MiniMaxH3Transformer dit = new(cfg);
            dit.LoadWeights(BuildWeights(cfg));
            (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(
                layout.PositionIds, dit.RopeInvFreq(), cfg.AttentionHeadDim);
            List<Tensor> aWeights = new(dit.EnumerateSharedWeights());
            aWeights.AddRange(dit.EnumerateBlockRangeWeights(0, splitBlock));
            backendA.PreloadWeights(aWeights);
            List<Tensor> bWeights = new(dit.EnumerateBlockRangeWeights(splitBlock, cfg.NumLayers));
            backendB.PreloadWeights(bWeights);

            (Tensor video, Tensor audio) = dit.ForwardSharded(
                backendA, backendB, layout, videoRows, audioRows, text, cos, sin, uniqueT, rowOf, splitBlock);
            shVideo = ToArray(video);
            shAudio = ToArray(audio);
            video.Dispose(); audio.Dispose(); cos.Dispose(); sin.Dispose();
            backendA.FreeWeights(aWeights);
            backendB.FreeWeights(bWeights);
        }

        Compare("video", refVideo, shVideo, exact);
        Compare("audio", refAudio, shAudio, exact);
    }

    private void Compare(string label, float[] reference, float[] sharded, bool exact)
    {
        Assert.Equal(reference.Length, sharded.Length);
        if (exact)
        {
            int mismatches = 0;
            for (int i = 0; i < reference.Length; i++)
            {
                if (reference[i] != sharded[i])
                {
                    if (mismatches < 5)
                        _output.WriteLine($"{label} mismatch at {i}: ref={reference[i]} sharded={sharded[i]}");
                    mismatches++;
                }
            }
            Assert.True(mismatches == 0, $"{label}: {mismatches}/{reference.Length} values differ on the same-device split.");
        }
        else
        {
            double scale = 0;
            foreach (float v in reference) scale = Math.Max(scale, Math.Abs(v));
            double maxErr = 0;
            for (int i = 0; i < reference.Length; i++)
                maxErr = Math.Max(maxErr, Math.Abs(reference[i] - sharded[i]));
            double relToScale = maxErr / Math.Max(scale, 1e-6);
            _output.WriteLine($"{label} cross-device: max abs err {maxErr:E3}, scale {scale:E3}, err/scale {relToScale:E3}");
            Assert.True(relToScale < 1e-3, $"{label}: cross-device sharded forward diverged (err/scale {relToScale:E3}).");
        }
    }

    private static float[] ToArray(Tensor t)
    {
        float[] arr = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) arr[i] = p[i];
        return arr;
    }
}
