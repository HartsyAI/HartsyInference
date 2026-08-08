using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Validates the chunked attention/MLP path (Phase 1 of the MiniMax-H3 memory overhaul) against the exact
/// same synthetic weights run unchunked — same shapes as <see cref="MiniMaxH3ForwardTests"/>, but with enough rows
/// to actually split into several chunks under <c>HARTSY_H3_CHUNK_ROWS</c>. The two paths use different GEMM calls
/// (many small matmuls vs one large one), so they are compared by relative L2 distance, not bitwise equality.</summary>
[Trait("Category", "SyntheticSmoke")]
public unsafe class MiniMaxH3ChunkedAttentionTests
{
    private readonly ITestOutputHelper _output;

    public MiniMaxH3ChunkedAttentionTests(ITestOutputHelper output) => _output = output;

    private static int _seed = 29;

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

    private static MiniMaxH3Config TinyConfig() => new MiniMaxH3Config
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
        TimeEmbedDim = 12,
        RopeInvFreqLen = 2,
        AdalnCurveGrid = null,
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
            ["time_embedder.proj_in.weight"] = Rand(c.TimeEmbedHiddenSize, c.TimestepInputDim),
            ["time_embedder.proj_in.bias"] = Rand(c.TimeEmbedHiddenSize),
            ["time_embedder.proj_out.weight"] = Rand(c.TimeEmbedDim, c.TimeEmbedHiddenSize),
            ["time_embedder.proj_out.bias"] = Rand(c.TimeEmbedDim),
        };
        for (int i = 0; i < c.TokenRefinerNumLayers; i++) AddBlock(w, $"token_refiner.blocks.{i}", c, h, inner, adaln: false);
        for (int i = 0; i < c.NumLayers; i++) AddBlock(w, $"blocks.{i}", c, h, inner, adaln: true);
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

    private static double RelL2(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        double num = 0, den = 0;
        for (long i = 0; i < a.ElementCount; i++)
        {
            double d = pa[i] - pb[i];
            num += d * d;
            den += (double)pa[i] * pa[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    /// <summary>Same weights, same inputs, same seed-derived noise-free deterministic forward — only the chunk
    /// size differs (forced via <c>HARTSY_H3_CHUNK_ROWS</c>). A large mismatch here means the chunk loop dropped,
    /// duplicated, or misaligned rows; small floating-point drift from different GEMM call shapes is expected.</summary>
    [Fact]
    public void ChunkedForward_MatchesUnchunked_WithinFloatingPointTolerance()
    {
        MiniMaxH3Config c = TinyConfig();
        Dictionary<string, Tensor> weights = BuildWeights(c);
        IBackend backend = new CpuBackend();

        const int textLen = 5, latentT = 6, latentH = 8, latentW = 12, audioT = 6;
        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, latentT, latentH, latentW, audioT);
        int frameRows = (latentH / 2) * (latentW / 2);
        int videoRowCount = latentT * frameRows;
        int audioRowCount = audioT * 2;
        _output.WriteLine($"seq={layout.SequenceLength} video rows={videoRowCount} audio rows={audioRowCount}");
        Assert.True(layout.SequenceLength > 32, "geometry must be large enough for multiple 16-row chunks");

        using Tensor videoRows = Rand(videoRowCount, c.VideoPatchDim);
        using Tensor audioRows = Rand(audioRowCount, c.AudioLatentsDim);
        using Tensor text = Rand(textLen, c.TextDim);
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(
            layout.PositionIds, MiniMaxH3Rope.DefaultInvFreq(c.RopeInvFreqLen), c.AttentionHeadDim);
        float[] uniqueT = [0.3f, 0.55f];
        Dictionary<MiniMaxH3SegmentKind, int> rowOf = new()
        {
            [MiniMaxH3SegmentKind.Text] = 0, [MiniMaxH3SegmentKind.Video] = 0,
            [MiniMaxH3SegmentKind.Cond] = 0, [MiniMaxH3SegmentKind.RefImage] = 0,
            [MiniMaxH3SegmentKind.Audio] = 1, [MiniMaxH3SegmentKind.RefAudio] = 1,
        };

        Environment.SetEnvironmentVariable("HARTSY_H3_CHUNK_ROWS", null);
        using MiniMaxH3Transformer legacyDit = new MiniMaxH3Transformer(c);
        legacyDit.LoadWeights(CloneWeights(weights));
        (Tensor legacyVideo, Tensor legacyAudio) =
            legacyDit.Forward(backend, layout, videoRows, audioRows, text, cos, sin, uniqueT, rowOf);

        Environment.SetEnvironmentVariable("HARTSY_H3_CHUNK_ROWS", "16");
        try
        {
            using MiniMaxH3Transformer chunkedDit = new MiniMaxH3Transformer(c);
            chunkedDit.LoadWeights(CloneWeights(weights));
            (Tensor chunkedVideo, Tensor chunkedAudio) =
                chunkedDit.Forward(backend, layout, videoRows, audioRows, text, cos, sin, uniqueT, rowOf);
            try
            {
                double videoRelL2 = RelL2(legacyVideo, chunkedVideo);
                double audioRelL2 = RelL2(legacyAudio, chunkedAudio);
                _output.WriteLine($"video relL2={videoRelL2:E3} audio relL2={audioRelL2:E3}");
                Assert.True(videoRelL2 < 1e-4, $"video relL2 {videoRelL2:E3} exceeds tolerance");
                Assert.True(audioRelL2 < 1e-4, $"audio relL2 {audioRelL2:E3} exceeds tolerance");
            }
            finally
            {
                chunkedVideo.Dispose();
                chunkedAudio.Dispose();
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_H3_CHUNK_ROWS", null);
            legacyVideo.Dispose();
            legacyAudio.Dispose();
            cos.Dispose();
            sin.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    /// <summary>Each transformer instance disposes its own loaded weights, so the two forwards in the test above
    /// need independent tensor copies of the same values rather than sharing one dictionary.</summary>
    private static Dictionary<string, Tensor> CloneWeights(Dictionary<string, Tensor> source)
    {
        Dictionary<string, Tensor> clone = new Dictionary<string, Tensor>(source.Count);
        foreach (KeyValuePair<string, Tensor> kv in source)
        {
            Tensor copy = new Tensor(kv.Value.Shape, kv.Value.DType);
            Buffer.MemoryCopy(kv.Value.DataPointer, copy.DataPointer,
                (long)kv.Value.ElementCount * kv.Value.DType.SizeInBytes, (long)kv.Value.ElementCount * kv.Value.DType.SizeInBytes);
            clone[kv.Key] = copy;
        }
        return clone;
    }
}
