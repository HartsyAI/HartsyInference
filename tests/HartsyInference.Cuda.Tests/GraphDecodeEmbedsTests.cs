using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validates <see cref="GenericTransformer.ForwardGraphDecodeStepEmbeds"/> — the embedding-driven,
/// fixed-buffer graph-decode step that Sesame CSM / HeartMuLa use (host-built embedding in, post-norm hidden out,
/// no token gather, no argmax). Proves two things against the plain <c>ForwardEmbeds(t=1)</c> reference on a small
/// synthetic Qwen2 config (small = no multi-GB load, safe in the IDE test host): (a) the device-position step math
/// matches eager decode, and (b) a captured graph replayed across advancing positions stays byte-close to eager —
/// i.e. one capture is valid for every frame.</summary>
[Collection("CudaSerial")]
public sealed unsafe class GraphDecodeEmbedsTests
{
    private readonly ITestOutputHelper _output;
    public GraphDecodeEmbedsTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0xC0FFEEu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Embeds(int t, int h) => Fill(new Tensor(new TensorShape(1, t, h), DType.F32));

    private static TensorShape LastShape(int h) => new(1, 1, h);

    [Fact]
    public void GraphDecodeStepEmbeds_MatchesEagerDecode_DirectAndCaptured()
    {
        if (!CudaContext.IsAvailable()) { Console.Error.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        TransformerConfig cfg = new()
        {
            HiddenSize = 32, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 8,
            IntermediateSize = 64, VocabSize = 48, MaxPositionEmbeddings = 128, AttentionBias = true, QkNorm = false,
            TieWordEmbeddings = true,
        };
        Dictionary<string, Tensor> w = Qwen2Weights(cfg);

        CudaBackend backend = new(0, ptxDir);
        try { RunBody(backend, cfg, w, ptxDir); }
        finally { try { backend.Dispose(); } catch (Exception ex) { Console.Error.WriteLine($"[teardown-ignored] {ex.GetType().Name}: {ex.Message}"); } }
    }

    private void RunBody(CudaBackend backend, TransformerConfig cfg, Dictionary<string, Tensor> w, string ptxDir)
    {
        int h = cfg.HiddenSize;
        IBackend b = backend;
        Assert.True(b.GraphDecodeSupported, "backend must support graph decode");
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        Assert.True(model.SupportsGraphDecode(b), "small dense Qwen2 must be graph-decode eligible");

        const int prefill = 5, steps = 6, maxLen = 64;
        // Two identical caches: one driven by eager ForwardEmbeds, one by the graph step.
        using FixedKvCache eager = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);
        using FixedKvCache graph = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxLen);

        // Identical prefill through both caches (eager path; establishes matching KV state).
        using (Tensor pe = Embeds(prefill, h))
        {
            using Tensor oe = model.ForwardEmbeds(b, pe, prefill, 0, eager); AssertFinite(oe, "eager-prefill");
            using Tensor og = model.ForwardEmbeds(b, pe, prefill, 0, graph); AssertFinite(og, "graph-prefill");
        }

        if (!b.StepGraphSupported) { Console.Error.WriteLine("SKIPPED: StepGraph unsupported"); foreach (Tensor t in w.Values) t.Dispose(); return; }

        (Tensor cos, Tensor sin) = model.EnsureRopeTableForGraphDecode(b, maxLen);
        ulong devicePos = b.AllocDevicePos();
        using Tensor inEmbed = new(LastShape(h), DType.F32);     // fixed input buffer (refreshed per step, OUTSIDE capture)
        using Tensor outHidden = new(LastShape(h), DType.F32);   // fixed output buffer (read per step via ReadResidentInto)
        float[] scratch = new float[h];
        // Mirror MusicGen's lifecycle: eager warmup THROUGH the fixed buffers (so weights/RoPE/buffers are resident and
        // nothing allocates during capture), then capture at CaptureAt, then replay. All steps compared to eager decode.
        const int captureAt = 2;
        float maxWarm = 0f, maxCap = 0f, maxReplay = 0f;
        b.StepGraphOwner = this;
        try
        {
            for (int s = 0; s < steps; s++)
            {
                int pos = prefill + s;
                using Tensor e = Embeds(1, h);

                // Eager reference in its own cache at `pos`.
                using Tensor eagerOut = model.ForwardEmbeds(b, e, 1, pos, eager);
                float[] refv = ReadLast(eagerOut, h);

                // Refresh fixed input + device position OUTSIDE any capture region.
                b.CopyInto(inEmbed, e);
                b.WriteDevicePos(devicePos, pos + 1, pos);

                bool replay = s > captureAt && b.StepGraphReady;
                if (replay)
                {
                    b.StepGraphLaunch();
                }
                else if (s == captureAt)
                {
                    b.StepGraphBegin();
                    model.ForwardGraphDecodeStepEmbeds(b, inEmbed, graph, cos, sin, devicePos, outHidden);
                    b.StepGraphEndAndLaunch();   // ends capture and RUNS this step (capture alone only records)
                }
                else
                {
                    model.ForwardGraphDecodeStepEmbeds(b, inEmbed, graph, cos, sin, devicePos, outHidden);
                }
                backend.Sync();
                b.ReadResidentInto(outHidden, scratch);   // non-freeing DtoH; keeps the baked address

                float d = MaxAbsDiff(refv, scratch);
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

        // Eager warmup proves the primitive's math; the captured step and every replay across advancing positions
        // must stay byte-close to eager decode — one capture valid for every frame.
        Assert.True(maxWarm <= 1e-6f, $"eager primitive diverges from ForwardEmbeds by {maxWarm:E3}");
        Assert.True(maxCap <= 1e-6f, $"captured step diverges from eager by {maxCap:E3}");
        Assert.True(maxReplay <= 1e-6f, $"graph replay diverges from eager by {maxReplay:E3} (one capture must be valid for every position)");
    }

    private static float[] ReadLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        float[] outv = new float[h];
        new Span<float>((float*)hidden.DataPointer + (long)(t - 1) * h, h).CopyTo(outv);
        return outv;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static void AssertFinite(Tensor t, string tag)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
            Assert.True(float.IsFinite(p[i]), $"{tag}: non-finite at {i}");
    }

    private static Dictionary<string, Tensor> Qwen2Weights(TransformerConfig c)
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
            w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
            w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
            w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }
}
