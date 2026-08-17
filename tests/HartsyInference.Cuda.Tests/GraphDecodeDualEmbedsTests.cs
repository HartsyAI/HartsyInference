using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validates <see cref="GenericTransformer.ForwardGraphDecodeStepDualEmbeds"/> — the two-stream
/// (classifier-free) graph-decode step CSM/HeartMuLa use, and the one MiniMax Music 3's language model wants.
/// The single-stream twin is covered by <see cref="GraphDecodeEmbedsTests"/>; this exists because the dual step's
/// per-row attention has its own shape contract that <see cref="CpuBackend"/> does not validate, so a violation
/// is invisible until a CUDA run throws. Both projection branches are exercised: fused QKV (QkNorm off) and
/// composed (QkNorm on, which is what Qwen3 backbones take).</summary>
[Collection("CudaSerial")]
public sealed unsafe class GraphDecodeDualEmbedsTests
{
    private readonly ITestOutputHelper _output;
    public GraphDecodeDualEmbedsTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x5EEDBEEFu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Embeds(int t, int h) => Fill(new Tensor(new TensorShape(1, t, h), DType.F32));

    [Fact]
    public void DualGraphDecodeStepEmbeds_FusedQkv_MatchesTwoEagerStreams()
        => RunConfig(qkNorm: false);

    [Fact]
    public void DualGraphDecodeStepEmbeds_ComposedQkNorm_MatchesTwoEagerStreams()
        => RunConfig(qkNorm: true);

    private void RunConfig(bool qkNorm)
    {
        if (!CudaContext.IsAvailable()) { Console.Error.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        TransformerConfig cfg = new()
        {
            HiddenSize = 32, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 8,
            IntermediateSize = 64, VocabSize = 48, MaxPositionEmbeddings = 128,
            AttentionBias = !qkNorm, QkNorm = qkNorm, TieWordEmbeddings = true,
        };
        Dictionary<string, Tensor> w = Weights(cfg);

        CudaBackend backend = new(0, ptxDir);
        try { RunBody(backend, cfg, w); }
        finally { try { backend.Dispose(); } catch (Exception ex) { Console.Error.WriteLine($"[teardown-ignored] {ex.GetType().Name}: {ex.Message}"); } }
    }

    private void RunBody(CudaBackend backend, TransformerConfig cfg, Dictionary<string, Tensor> w)
    {
        int h = cfg.HiddenSize;
        IBackend b = backend;
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        // The dual step batches its projections into one 2-row GEMM while the eager streams do two 1-row ones.
        // Under the default TF32 compute type those are not the same arithmetic: cuBLAS picks tensor cores per
        // shape and per architecture, and a 10-bit mantissa costs ~1e-4 relative — which is 1e-3 by the end of
        // two layers. This test is about the dual path's plumbing, not cuBLAS's heuristics, so pin full precision
        // and let BatchedGemmPrecisionProbe own the precision question. Without this the test passed on a 3060
        // (where cuBLAS declines tensor cores at 32 wide) and failed on a 4090, which is luck, not coverage.
        backend.HighPrecisionGemm = true;
        Assert.True(model.SupportsDualGraphDecode(b), "small dense backbone must be dual-graph-decode eligible");

        const int prefill = 5, steps = 6, maxLen = 64;
        using FixedKvCache eagerA = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);
        using FixedKvCache eagerB = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);
        using FixedKvCache dualA = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);
        using FixedKvCache dualB = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);

        // The two streams carry DIFFERENT prefixes (that is the whole point of classifier-free decode), so a row
        // mix-up shows up as a mismatch rather than cancelling out.
        using (Tensor pa = Embeds(prefill, h))
        using (Tensor pb = Embeds(prefill, h))
        {
            using Tensor oa = model.ForwardEmbeds(b, pa, prefill, 0, eagerA);
            using Tensor ob = model.ForwardEmbeds(b, pb, prefill, 0, eagerB);
            using Tensor da = model.ForwardEmbeds(b, pa, prefill, 0, dualA);
            using Tensor db = model.ForwardEmbeds(b, pb, prefill, 0, dualB);
        }

        if (!b.StepGraphSupported) { Console.Error.WriteLine("SKIPPED: StepGraph unsupported"); foreach (Tensor t in w.Values) t.Dispose(); return; }

        (Tensor cos, Tensor sin) = model.EnsureRopeTableForGraphDecode(b, maxLen);
        ulong devicePos = b.AllocDevicePos();
        using Tensor inEmbed = new(new TensorShape(1, 2, h), DType.F32);
        using Tensor outHidden = new(new TensorShape(1, 2, h), DType.F32);
        float[] scratch = new float[2 * h];
        const int captureAt = 2;
        float maxWarm = 0f, maxCap = 0f, maxReplay = 0f;
        b.StepGraphOwner = this;
        try
        {
            for (int s = 0; s < steps; s++)
            {
                int pos = prefill + s;
                using Tensor ea = Embeds(1, h);
                using Tensor eb = Embeds(1, h);
                using Tensor refA = model.ForwardEmbeds(b, ea, 1, pos, eagerA);
                using Tensor refB = model.ForwardEmbeds(b, eb, 1, pos, eagerB);

                using (Tensor stacked = new(new TensorShape(1, 2, h), DType.F32))
                {
                    Buffer.MemoryCopy((void*)ea.DataPointer, (void*)stacked.DataPointer, (long)h * 4, (long)h * 4);
                    Buffer.MemoryCopy((void*)eb.DataPointer, (float*)stacked.DataPointer + h, (long)h * 4, (long)h * 4);
                    b.CopyInto(inEmbed, stacked);
                }
                b.WriteDevicePos(devicePos, pos + 1, pos);

                bool replay = s > captureAt && b.StepGraphReady;
                if (replay)
                {
                    b.StepGraphLaunch();
                }
                else if (s == captureAt)
                {
                    b.StepGraphBegin();
                    model.ForwardGraphDecodeStepDualEmbeds(b, inEmbed, dualA, dualB, cos, sin, devicePos, outHidden);
                    b.StepGraphEndAndLaunch();
                }
                else
                {
                    model.ForwardGraphDecodeStepDualEmbeds(b, inEmbed, dualA, dualB, cos, sin, devicePos, outHidden);
                }
                dualA.AdvanceLength(1);
                dualB.AdvanceLength(1);
                backend.Sync();
                b.ReadResidentInto(outHidden, scratch);

                float d = MathF.Max(
                    MaxAbsDiff(new Span<float>((float*)refA.DataPointer, h), scratch.AsSpan(0, h)),
                    MaxAbsDiff(new Span<float>((float*)refB.DataPointer, h), scratch.AsSpan(h, h)));
                if (s < captureAt) maxWarm = MathF.Max(maxWarm, d);
                else if (s == captureAt) maxCap = d;
                else maxReplay = MathF.Max(maxReplay, d);
                Console.Error.WriteLine($"pos={pos} {(replay ? "replay " : s == captureAt ? "capture" : "warmup ")} max|Δ|={d:E3}");
            }
        }
        finally
        {
            b.StepGraphReset();
            if (ReferenceEquals(b.StepGraphOwner, this)) b.StepGraphOwner = null;
            b.FreeDevicePos(devicePos);
            foreach (Tensor t in w.Values) t.Dispose();
        }

        Assert.True(maxWarm <= 1e-6f, $"eager dual step diverges from two eager streams by {maxWarm:E3}");
        Assert.True(maxCap <= 1e-6f, $"captured dual step diverges by {maxCap:E3}");
        Assert.True(maxReplay <= 1e-6f, $"dual graph replay diverges by {maxReplay:E3} (one capture must be valid for every position)");
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = F2(c.VocabSize, h),
            ["model.norm.weight"] = Ones(h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            if (c.AttentionBias)
            {
                w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
                w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
                w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
            }
            if (c.QkNorm)
            {
                w[$"{p}.self_attn.q_norm.weight"] = Ones(c.HeadDim);
                w[$"{p}.self_attn.k_norm.weight"] = Ones(c.HeadDim);
            }
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }
}
