using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runs the whole MiniMax-H3 DiT on a tiny synthetic config through CpuBackend and CudaBackend and requires
/// the two to agree. The CPU path is the numerical reference (verified against a PyTorch transcription of Kijai's
/// <c>ldm/minimax/model.py</c> at relL2 1.6e-7), so this pins the GPU path to it.
///
/// <para>This exists because the DiT drives rope, the adaLN modulation and the gated residual as <b>in-place ops
/// whose output tensor is a borrowed pointer view</b> of a larger buffer. A device backend binds the result to the
/// view object; disposing the view unread frees the device buffer WITHOUT the device-to-host copy, so all three
/// silently became no-ops on CUDA while CPU stayed correct — the residual stream never advanced past the patch
/// embedding, for all 50 blocks. Nothing CPU-only can catch that.</para></summary>
[Trait("Category", "Integration")]
public unsafe class MiniMaxH3BackendParityTests
{
    private readonly ITestOutputHelper _output;

    public MiniMaxH3BackendParityTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
        {
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        }
        return dir;
    }

    /// <summary>Deterministic Gaussian source; weights are scaled by fan-in so the tiny model's activations stay
    /// O(1) and a dropped modulation/residual shows up as a large relative error instead of rounding.</summary>
    private sealed class Lcg
    {
        private ulong _state;

        public Lcg(ulong seed) => _state = seed;

        public double Normal()
        {
            double u1 = Math.Max(Next(), 1e-12), u2 = Next();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        private double Next()
        {
            _state = (_state * 6364136223846793005UL) + 1442695040888963407UL;
            return ((_state >> 11) & ((1UL << 53) - 1)) / (double)(1UL << 53);
        }
    }

    private Lcg _rng = new Lcg(20260803);

    private Tensor Rand(double scale, params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(_rng.Normal() * scale);
        return t;
    }

    /// <summary>RMS-norm weights sit near 1 in a real checkpoint.</summary>
    private Tensor RandNorm(long dim)
    {
        Tensor t = new Tensor(new TensorShape(dim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(1.0 + (_rng.Normal() * 0.1));
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
        RopeInvFreqLen = 2,   // 3 axes x 2 freqs x 2 halves = 12 of 16 head dims rotate
    };

    private Dictionary<string, Tensor> BuildWeights(MiniMaxH3Config c)
    {
        int h = c.HiddenSize, inner = c.NumAttentionHeads * c.AttentionHeadDim;
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>
        {
            ["video_patch_proj.weight"] = Rand(1.0 / Math.Sqrt(c.VideoPatchDim), h, c.VideoPatchDim),
            ["video_patch_proj.bias"] = Rand(0.05, h),
            ["audio_patch_proj.weight"] = Rand(1.0 / Math.Sqrt(c.AudioLatentsDim), h, c.AudioLatentsDim),
            ["audio_patch_proj.bias"] = Rand(0.05, h),
            ["condition_proj.weight"] = Rand(1.0 / Math.Sqrt(c.TextDim), h, c.TextDim),
            ["condition_proj.bias"] = Rand(0.05, h),
            ["rope.inv_freq"] = InvFreq(c.RopeInvFreqLen),
            ["token_refiner.final_norm.weight"] = RandNorm(h),
            ["final_layer.norm.weight"] = RandNorm(h),
            ["final_layer.adaln_proj.linear.weight"] = Rand(1.0 / Math.Sqrt(c.TimeEmbedDim), 2 * h, c.TimeEmbedDim),
            ["final_layer.adaln_proj.linear.bias"] = Rand(0.05, 2 * h),
            ["final_layer.video_out.weight"] = Rand(1.0 / Math.Sqrt(h), c.VideoPatchDim, h),
            ["final_layer.video_out.bias"] = Rand(0.05, c.VideoPatchDim),
            ["final_layer.audio_out.weight"] = Rand(1.0 / Math.Sqrt(h), c.AudioLatentsDim, h),
            ["final_layer.audio_out.bias"] = Rand(0.05, c.AudioLatentsDim),
            ["time_embedder.proj_in.weight"] = Rand(1.0 / Math.Sqrt(c.TimestepInputDim), c.TimeEmbedHiddenSize, c.TimestepInputDim),
            ["time_embedder.proj_in.bias"] = Rand(0.05, c.TimeEmbedHiddenSize),
            ["time_embedder.proj_out.weight"] = Rand(1.0 / Math.Sqrt(c.TimeEmbedHiddenSize), c.TimeEmbedDim, c.TimeEmbedHiddenSize),
            ["time_embedder.proj_out.bias"] = Rand(0.05, c.TimeEmbedDim),
        };
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

    private void AddBlock(Dictionary<string, Tensor> w, string p, MiniMaxH3Config c, int h, int inner, bool adaln)
    {
        w[$"{p}.norm1.weight"] = RandNorm(h);
        w[$"{p}.norm2.weight"] = RandNorm(h);
        w[$"{p}.attn.qkv_proj.weight"] = Rand(1.0 / Math.Sqrt(h), inner * 3, h);
        w[$"{p}.attn.q_norm.weight"] = RandNorm(c.AttentionHeadDim);
        w[$"{p}.attn.k_norm.weight"] = RandNorm(c.AttentionHeadDim);
        w[$"{p}.attn.out_proj.weight"] = Rand(1.0 / Math.Sqrt(inner), h, inner);
        w[$"{p}.mlp.fc1.weight"] = Rand(1.0 / Math.Sqrt(h), c.FfnHiddenSize * 2, h);
        w[$"{p}.mlp.fc2.weight"] = Rand(1.0 / Math.Sqrt(c.FfnHiddenSize), h, c.FfnHiddenSize);
        if (adaln)
        {
            w[$"{p}.adaln_proj.linear.weight"] = Rand(1.0 / Math.Sqrt(c.TimeEmbedDim), 6 * h * 3, c.TimeEmbedDim);
            w[$"{p}.adaln_proj.linear.bias"] = Rand(0.05, 6 * h * 3);
        }
    }

    private static Tensor InvFreq(int n)
    {
        Tensor t = new Tensor(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = (float)(1.0 / Math.Pow(10000.0, (2.0 * i) / (2.0 * n)));
        return t;
    }

    private static float[] InvFreqArray(int n)
    {
        float[] inv = new float[n];
        for (int i = 0; i < n; i++) inv[i] = (float)(1.0 / Math.Pow(10000.0, (2.0 * i) / (2.0 * n)));
        return inv;
    }

    /// <summary>Each <c>LoadWeights</c> takes ownership, so every run needs its own copy.</summary>
    private static Dictionary<string, Tensor> CopyWeights(Dictionary<string, Tensor> src)
    {
        Dictionary<string, Tensor> copy = new Dictionary<string, Tensor>(src.Count);
        foreach (KeyValuePair<string, Tensor> kv in src)
        {
            Tensor t = new Tensor(kv.Value.Shape, kv.Value.DType);
            Buffer.MemoryCopy((void*)kv.Value.DataPointer, (void*)t.DataPointer,
                t.ElementCount * sizeof(float), kv.Value.ElementCount * sizeof(float));
            copy[kv.Key] = t;
        }
        return copy;
    }

    /// <summary>Host copy of a possibly device-resident activation.</summary>
    private static Tensor Materialize(Tensor t)
    {
        Tensor copy = new Tensor(t.Shape, t.DType);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)copy.DataPointer,
            copy.ElementCount * sizeof(float), t.ElementCount * sizeof(float));
        return copy;
    }

    private static double RelL2(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        double num = 0, den = 0;
        for (long i = 0; i < a.ElementCount; i++)
        {
            double d = pa[i] - pb[i];
            num += d * d;
            den += (double)pb[i] * pb[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static double Rms(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        double s = 0;
        for (long i = 0; i < t.ElementCount; i++) s += (double)p[i] * p[i];
        return Math.Sqrt(s / t.ElementCount);
    }

    /// <summary>The CPU reference is always F32 (CpuBackend is F32-only), so the bf16 case measures the residual
    /// stream's own rounding, not backend disagreement.</summary>
    [Theory]
    [InlineData(false, 5e-3)]
    [InlineData(true, 1.5e-2)]
    public void CudaForwardMatchesTheCpuReferencePath(bool bf16Body, double tolerance)
    {
        if (!Directory.Exists(PtxDir()))
        {
            return;
        }

        MiniMaxH3Config c = TinyConfig();
        Dictionary<string, Tensor> weights = BuildWeights(c);
        const int textLen = 3, latentT = 2, latentH = 4, latentW = 6, audioT = 2;
        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, latentT, latentH, latentW, audioT);
        int videoRowCount = latentT * (latentH / 2) * (latentW / 2), audioRowCount = audioT * 2;

        using Tensor videoRows = Rand(1.0, videoRowCount, c.VideoPatchDim);
        using Tensor audioRows = Rand(1.0, audioRowCount, c.AudioLatentsDim);
        using Tensor text = Rand(1.0, textLen, c.TextDim);
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(
            layout.PositionIds, InvFreqArray(c.RopeInvFreqLen), c.AttentionHeadDim);
        float[] uniqueT = [0.30f, 0.55f];
        Dictionary<MiniMaxH3SegmentKind, int> rowOf = new Dictionary<MiniMaxH3SegmentKind, int>
        {
            [MiniMaxH3SegmentKind.Text] = 0, [MiniMaxH3SegmentKind.Video] = 0,
            [MiniMaxH3SegmentKind.Cond] = 0, [MiniMaxH3SegmentKind.RefImage] = 0,
            [MiniMaxH3SegmentKind.Audio] = 1, [MiniMaxH3SegmentKind.RefAudio] = 1,
        };
        // Vision pads inside the text span modulate as video, so the tag-run split is exercised too.
        (int, int, int)[] tagRuns = [(0, 1, 1), (1, 2, 0), (2, 3, 1)];

        Tensor videoCpu, audioCpu;
        using (IBackend cpu = new CpuBackend())
        using (MiniMaxH3Transformer dit = new MiniMaxH3Transformer(c))
        {
            dit.LoadWeights(CopyWeights(weights));
            (videoCpu, audioCpu) = dit.Forward(cpu, layout, videoRows, audioRows, text, cos, sin, uniqueT, rowOf, tagRuns);
        }

        Tensor videoGpu, audioGpu;
        long forwardSyncs;
        using (IBackend gpu = new CudaBackend(deviceOrdinal: 0, ptxDir: PtxDir()))
        using (MiniMaxH3Transformer dit = new MiniMaxH3Transformer(c) { BodyDType = bf16Body ? DType.BF16 : DType.F32 })
        {
            dit.LoadWeights(CopyWeights(weights));
            gpu.ResetD2hSyncCount();
            (Tensor v, Tensor a) = dit.Forward(gpu, layout, videoRows, audioRows, text, cos, sin, uniqueT, rowOf, tagRuns);
            forwardSyncs = gpu.GetD2hSyncCount();
            gpu.Sync();
            // Device activations must be pulled to the host before the backend (and its allocations) go away.
            try
            {
                videoGpu = Materialize(v);
                audioGpu = Materialize(a);
            }
            finally
            {
                v.Dispose();
                a.Dispose();
            }
        }

        try
        {
            double video = RelL2(videoGpu, videoCpu), audio = RelL2(audioGpu, audioCpu);
            _output.WriteLine($"video relL2={video:E3} (cpu rms={Rms(videoCpu):F5}, gpu rms={Rms(videoGpu):F5})");
            _output.WriteLine($"audio relL2={audio:E3} (cpu rms={Rms(audioCpu):F5}, gpu rms={Rms(audioGpu):F5})");
            _output.WriteLine($"D2H syncs during the forward: {forwardSyncs}");
            // Floors: TF32 GEMM difference (~5e-4) at F32, one bf16 rounding (~4e-3) with the bf16 stream.
            // A dropped in-place write lands at O(1) either way.
            Assert.True(video < tolerance, $"CUDA video head diverged from the CPU reference: relL2 {video:E3}");
            Assert.True(audio < tolerance, $"CUDA audio head diverged from the CPU reference: relL2 {audio:E3}");
            // Each sync is a full stream drain plus a device-to-host copy that also evicts the tensor from the
            // activation cache; the forward must stay entirely on-device.
            Assert.True(forwardSyncs == 0, $"MiniMax-H3 forward pulled {forwardSyncs} activations back to the host.");
        }
        finally
        {
            videoCpu.Dispose(); audioCpu.Dispose(); videoGpu.Dispose(); audioGpu.Dispose();
            cos.Dispose(); sin.Dispose();
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}
