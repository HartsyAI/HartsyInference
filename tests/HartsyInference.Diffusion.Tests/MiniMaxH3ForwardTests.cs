using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Executes the MiniMax-H3 DiT end to end on a tiny synthetic config: packed layout, rotary tables, token
/// refiner, blocks, dual output heads. Gates wiring and shapes, not numerics — real-weight parity comes later.</summary>
[Trait("Category", "SyntheticSmoke")]
public unsafe class MiniMaxH3ForwardTests
{
    private readonly ITestOutputHelper _output;

    public MiniMaxH3ForwardTests(ITestOutputHelper output) => _output = output;

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

    /// <summary>A miniature H3 that keeps every structural relationship (patch volume, gated MLP, 3-modality adaln,
    /// 3-axis rotary) while staying small enough to run on CPU.</summary>
    private static MiniMaxH3Config TinyConfig(bool curves) => new MiniMaxH3Config
    {
        HiddenSize = 32,
        NumLayers = 2,
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
        RopeInvFreqLen = 2,   // 3 axes x 2 freqs x 2 halves = 12 of 16 head dims rotate
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
    [InlineData(false)]   // full variant: sinusoid + time embedder
    [InlineData(true)]    // pruned variant: adaln curve table
    public void ForwardProducesFiniteVideoAndAudioOfTheExpectedShape(bool curves)
    {
        MiniMaxH3Config c = TinyConfig(curves);
        IBackend backend = new CpuBackend();
        using MiniMaxH3Transformer dit = new MiniMaxH3Transformer(c);
        dit.LoadWeights(BuildWeights(c));

        const int textLen = 3, latentT = 2, latentH = 4, latentW = 6, audioT = 2;
        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, latentT, latentH, latentW, audioT);
        int frameRows = (latentH / 2) * (latentW / 2);
        int videoRowCount = latentT * frameRows;
        int audioRowCount = audioT * 2;
        _output.WriteLine($"seq={layout.SequenceLength} video rows={videoRowCount} audio rows={audioRowCount}");

        using Tensor videoRows = Rand(videoRowCount, c.VideoPatchDim);
        using Tensor audioRows = Rand(audioRowCount, c.AudioLatentsDim);
        using Tensor text = Rand(textLen, c.TextDim);
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(
            layout.PositionIds, MiniMaxH3Rope.DefaultInvFreq(c.RopeInvFreqLen), c.AttentionHeadDim);

        // Video and audio ride distinct timestep rows, matching the dual-schedule design.
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

        (Tensor video, Tensor audio) = dit.Forward(backend, layout, videoRows, audioRows, text, cos, sin, uniqueT, rowOf);
        try
        {
            Assert.Equal(videoRowCount, (int)video.Shape[0]);
            Assert.Equal(c.VideoPatchDim, (int)video.Shape[1]);
            Assert.Equal(audioRowCount, (int)audio.Shape[0]);
            Assert.Equal(c.AudioLatentsDim, (int)audio.Shape[1]);

            float* vp = (float*)video.DataPointer;
            for (long i = 0; i < video.ElementCount; i++) Assert.True(float.IsFinite(vp[i]), $"video non-finite at {i}");
            float* ap = (float*)audio.DataPointer;
            for (long i = 0; i < audio.ElementCount; i++) Assert.True(float.IsFinite(ap[i]), $"audio non-finite at {i}");

            double vAbs = 0;
            for (long i = 0; i < video.ElementCount; i++) vAbs = Math.Max(vAbs, Math.Abs(vp[i]));
            _output.WriteLine($"video absmax={vAbs:F5}  (curves={curves})");
            Assert.True(vAbs > 0, "video head produced all zeros");
        }
        finally
        {
            video.Dispose(); audio.Dispose(); cos.Dispose(); sin.Dispose();
        }
    }
}
