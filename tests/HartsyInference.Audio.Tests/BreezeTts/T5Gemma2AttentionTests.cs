using HartsyInference.Audio.Models.BreezeTts;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.BreezeTts;

/// <summary>The Breeze TTS 2 text encoder ran its attention as a scalar loop on the host, one head, query and key at a
/// time, for every one of its 26 layers. It now goes through the shared attention layer. The loop and the whole former
/// <c>Encode</c> are kept here, verbatim in their arithmetic, as the oracle.
/// <para>The tolerances were fixed before the first run: 1e-4 relative to the larger of 1 and the value on the CPU
/// backend and on Vulkan, and 2e-3 on CUDA, whose F32 dense row multiplies in TF32 (the contract's bound for it, and
/// what the other audio sites on that row are held to).</para></summary>
public sealed unsafe class T5Gemma2AttentionTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-4f, Tf32Tolerance = 2e-3f;

    /// <summary>Wall time of each call's former loop and of the shared layer, reported and never asserted. The tests of
    /// one class run one after another, so one list serves them.</summary>
    private static readonly List<string> Timings = [];

    private void ReportTimings()
    {
        foreach (string line in Timings) output.WriteLine(line);
        Timings.Clear();
    }

    private static float[] Random(int count, int seed, float scale = 1f)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (random.NextSingle() * 2f - 1f) * scale;
        return values;
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(expected[i] - actual[i]) / Math.Max(1f, Math.Abs(expected[i]));
            if (!(error <= worst)) (worst, at) = (error, i);
        }
        Assert.True(worst <= tolerance, $"{what}: element {at} is {(at < 0 ? 0f : actual[at])}, expected "
            + $"{(at < 0 ? 0f : expected[at])} (relative error {worst:G3}, bound {tolerance})");
    }

    /// <summary>The encoder's former attention loop: <paramref name="q"/> <c>[n, heads · hd]</c>, <paramref name="k"/> and
    /// <paramref name="v"/> <c>[n, kvHeads · hd]</c>.</summary>
    private static float[] FormerLoop(float[] q, float[] k, float[] v, int n, int heads, int kvHeads, int hd, float scale, bool full,
        int left, int right)
    {
        float[] ctx = new float[n * heads * hd];
        float[] sc = new float[n];
        int group = heads / kvHeads;
        for (int head = 0; head < heads; head++)
        {
            int kv = head / group;
            for (int i = 0; i < n; i++)
            {
                int lo = full ? 0 : Math.Max(0, i - left + 1), hi = full ? n - 1 : Math.Min(n - 1, i + right - 1);
                float max = float.NegativeInfinity;
                for (int j = lo; j <= hi; j++)
                {
                    float s = 0f;
                    for (int e = 0; e < hd; e++) s += q[(i * heads + head) * hd + e] * k[(j * kvHeads + kv) * hd + e];
                    sc[j] = s * scale; max = MathF.Max(max, sc[j]);
                }
                float sum = 0f;
                for (int j = lo; j <= hi; j++) { sc[j] = MathF.Exp(sc[j] - max); sum += sc[j]; }
                for (int e = 0; e < hd; e++)
                {
                    float o = 0f;
                    for (int j = lo; j <= hi; j++) o += sc[j] * v[(j * kvHeads + kv) * hd + e];
                    ctx[(i * heads + head) * hd + e] = o / sum;
                }
            }
        }
        return ctx;
    }

    /// <summary>One attention call the way <c>Encode</c> now makes it, against the former loop.</summary>
    private static void AttentionMatches(IBackend backend, float tolerance, int n, int heads, int kvHeads, int hd, bool full, int window,
        int seed, float scalar = 0f)
    {
        // The checkpoint's scalar is its head dim, which is also the layer's default scale; a case with another scalar
        // is what shows the scale is the configured one.
        if (scalar == 0f) scalar = hd;
        T5Gemma2TextEncoderConfig cfg = new() { NumHeads = heads, NumKvHeads = kvHeads, HeadDim = hd, QueryPreAttnScalar = scalar, SlidingWindow = window };
        int left = (window + 1) / 2, right = window / 2 + 1;
        float[] q = Random(n * heads * hd, seed), k = Random(n * kvHeads * hd, seed + 1), v = Random(n * kvHeads * hd, seed + 2);
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        float[] expected = FormerLoop(q, k, v, n, heads, kvHeads, hd, 1f / MathF.Sqrt(scalar), full, left, right);
        double formerMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        using Tensor? bias = full ? null : T5Gemma2TextEncoder.WindowBias(n, left, right);
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        float[] actual = T5Gemma2TextEncoder.Attend(backend, T5Gemma2TextEncoder.AttentionFor(cfg), q, k, v, n, heads, kvHeads, hd, bias);
        backend.Sync();
        Timings.Add($"n {n}, heads {heads}/{kvHeads}, D {hd}, scalar {scalar}, {(full ? "full" : $"window {window}")}: former loop {formerMs:F1} ms, shared layer {System.Diagnostics.Stopwatch.GetElapsedTime(t1).TotalMilliseconds:F1} ms");
        AssertClose(expected, actual, tolerance, $"n {n}, heads {heads}/{kvHeads}, D {hd}, scalar {scalar}, {(full ? "full" : $"window {window}")}");
    }

    private static void EveryCallMatches(IBackend backend, float tolerance)
    {
        // The checkpoint's geometry: 4 query heads over 1 key/value head of 256.
        AttentionMatches(backend, tolerance, 40, 4, 1, 256, full: true, 512, 11);
        AttentionMatches(backend, tolerance, 40, 4, 1, 256, full: false, 6, 21);       // a window that binds: 3 back, 4 ahead
        AttentionMatches(backend, tolerance, 300, 4, 1, 256, full: false, 512, 31);    // the real window over a segment it covers: no mask
        AttentionMatches(backend, tolerance, 700, 4, 1, 256, full: false, 512, 41);    // the real window where it binds
        AttentionMatches(backend, tolerance, 700, 4, 1, 256, full: true, 512, 51);
        // Two key/value heads, an odd window and one token.
        AttentionMatches(backend, tolerance, 57, 4, 2, 64, full: false, 9, 61);
        AttentionMatches(backend, tolerance, 1, 4, 1, 256, full: false, 512, 71);
        // A scalar that is not the head dim, so the scores are scaled by 1/12 and not by the layer's default of 1/16.
        AttentionMatches(backend, tolerance, 90, 4, 1, 256, full: true, 512, 81, scalar: 144f);
        AttentionMatches(backend, tolerance, 90, 4, 2, 64, full: false, 20, 91, scalar: 25f);
    }

    [Fact]
    public void TheWindowBiasIsTheFormerLoopsRange()
    {
        Assert.Null(T5Gemma2TextEncoder.WindowBias(256, 256, 257));
        Assert.Null(T5Gemma2TextEncoder.WindowBias(1, 1, 1));
        using Tensor bias = T5Gemma2TextEncoder.WindowBias(10, 3, 4)!;
        ReadOnlySpan<float> rows = bias.AsReadOnlySpan<float>();
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++)
            {
                bool seen = j >= Math.Max(0, i - 3 + 1) && j <= Math.Min(9, i + 4 - 1);
                Assert.Equal(seen ? 0f : float.NegativeInfinity, rows[i * 10 + j]);
            }
    }

    [Fact]
    public void OnTheCpuBackendTheAttentionMatchesTheFormerLoop()
    {
        using CpuBackend backend = new();
        try { EveryCallMatches(backend, Tolerance); }
        finally { ReportTimings(); }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnCudaTheAttentionMatchesTheFormerLoop()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDirectory());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        try { EveryCallMatches(backend, Tf32Tolerance); }
        finally
        {
            ReportRows(evidence.Snapshot());
            ReportTimings();
        }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnVulkanTheAttentionMatchesTheFormerLoop()
    {
        VulkanBackend backend;
        try { backend = new VulkanBackend(); }
        catch (Exception ex) { output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})"); return; }
        using (backend)
        {
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
            try { EveryCallMatches(backend, Tolerance); }
            finally
            {
                ReportRows(evidence.Snapshot());
                ReportTimings();
            }
        }
    }

    private void ReportRows(AttentionEvidenceEvent[] events)
    {
        foreach (IGrouping<string, AttentionEvidenceEvent> row in events.Where(e => e.Kind == AttentionEvidenceKind.Execution)
                     .GroupBy(e => $"{e.Implementation} heads {e.Heads}, seq {e.QueryTokens}"))
            output.WriteLine($"{row.Key}: {row.Count()} call(s)");
        foreach (IGrouping<AttentionEvidenceKind, AttentionEvidenceEvent> kind in events.Where(e => e.Kind != AttentionEvidenceKind.Execution).GroupBy(e => e.Kind))
            output.WriteLine($"{kind.Key}: {kind.Count()} record(s)");
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx was not found above the test directory.");
    }

    // ---- The whole encoder, so that the wiring (which layers slide, which table rotates them) is covered too. ----

    private static readonly T5Gemma2TextEncoderConfig Small = new()
    {
        VocabSize = 60, HiddenSize = 32, IntermediateSize = 64, NumLayers = 4, NumHeads = 4, NumKvHeads = 2, HeadDim = 8,
        QueryPreAttnScalar = 12f, SlidingWindow = 6, SlidingPatternPeriod = 2, EoiTokenIndex = 59,
    };

    private static Tensor Weight(int seed, float scale, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        Random(checked((int)tensor.ElementCount), seed, scale).CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Dictionary<string, Tensor> SmallWeights()
    {
        T5Gemma2TextEncoderConfig c = Small;
        int h = c.HiddenSize, inner = c.NumHeads * c.HeadDim, kv = c.NumKvHeads * c.HeadDim, seed = 500;
        Dictionary<string, Tensor> w = new()
        {
            ["text_encoder.embed_tokens.weight"] = Weight(seed++, 0.5f, c.VocabSize, h),
            ["text_encoder.norm.weight"] = Weight(seed++, 0.2f, h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"text_encoder.layers.{i}";
            foreach (string norm in (string[])["pre_self_attn_layernorm", "post_self_attn_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"])
                w[$"{p}.{norm}.weight"] = Weight(seed++, 0.2f, h);
            w[$"{p}.self_attn.q_norm.weight"] = Weight(seed++, 0.2f, c.HeadDim);
            w[$"{p}.self_attn.k_norm.weight"] = Weight(seed++, 0.2f, c.HeadDim);
            w[$"{p}.self_attn.q_proj.weight"] = Weight(seed++, 0.3f, inner, h);
            w[$"{p}.self_attn.k_proj.weight"] = Weight(seed++, 0.3f, kv, h);
            w[$"{p}.self_attn.v_proj.weight"] = Weight(seed++, 0.3f, kv, h);
            w[$"{p}.self_attn.o_proj.weight"] = Weight(seed++, 0.3f, h, inner);
            w[$"{p}.mlp.gate_proj.weight"] = Weight(seed++, 0.3f, c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = Weight(seed++, 0.3f, c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = Weight(seed++, 0.3f, h, c.IntermediateSize);
        }
        return w;
    }

    /// <summary>The former <c>Encode</c>, with the attention loop in place, over the same weights.</summary>
    private static float[] FormerEncode(IBackend backend, T5Gemma2TextEncoderConfig cfg, Dictionary<string, Tensor> w, ReadOnlySpan<int> tokens)
    {
        int n = tokens.Length, h = cfg.HiddenSize, heads = cfg.NumHeads, kvHeads = cfg.NumKvHeads, hd = cfg.HeadDim;
        float embedScale = MathF.Sqrt(h);
        float[] x = new float[n * h];
        ReadOnlySpan<float> emb = w["text_encoder.embed_tokens.weight"].AsReadOnlySpan<float>();
        for (int i = 0; i < n; i++)
            for (int c = 0; c < h; c++) x[i * h + c] = emb[tokens[i] * h + c] * embedScale;
        float[,] localRope = T5Gemma2TextEncoder.RopeTable(n, hd, cfg.LocalRopeTheta, 1f);
        float[,] globalRope = T5Gemma2TextEncoder.RopeTable(n, hd, cfg.GlobalRopeTheta, cfg.GlobalRopeLinearFactor);
        float scale = 1f / MathF.Sqrt(cfg.QueryPreAttnScalar);
        int left = (cfg.SlidingWindow + 1) / 2, right = cfg.SlidingWindow / 2 + 1;
        List<Tensor> owned = [];
        Tensor OnePlus(string name)
        {
            Tensor t = NormWeights.OnePlus(w[name]);
            owned.Add(t);
            return t;
        }
        float[] Norm(float[] values, string name, int rows, int dim) => DacOps.RmsNorm(backend, values, OnePlus(name), rows, dim, cfg.RmsNormEps);
        try
        {
            for (int li = 0; li < cfg.NumLayers; li++)
            {
                string p = $"text_encoder.layers.{li}";
                bool full = cfg.IsFullAttention(li);
                float[,] rope = full ? globalRope : localRope;
                float[] a = Norm(x, $"{p}.pre_self_attn_layernorm.weight", n, h);
                float[] q = DacOps.Linear(backend, a, w[$"{p}.self_attn.q_proj.weight"], n, h, heads * hd);
                float[] k = DacOps.Linear(backend, a, w[$"{p}.self_attn.k_proj.weight"], n, h, kvHeads * hd);
                float[] v = DacOps.Linear(backend, a, w[$"{p}.self_attn.v_proj.weight"], n, h, kvHeads * hd);
                q = Norm(q, $"{p}.self_attn.q_norm.weight", n * heads, hd);
                k = Norm(k, $"{p}.self_attn.k_norm.weight", n * kvHeads, hd);
                T5Gemma2TextEncoder.ApplyRope(q, rope, n, heads, hd);
                T5Gemma2TextEncoder.ApplyRope(k, rope, n, kvHeads, hd);
                float[] ctx = FormerLoop(q, k, v, n, heads, kvHeads, hd, scale, full, left, right);
                float[] attn = DacOps.Linear(backend, ctx, w[$"{p}.self_attn.o_proj.weight"], n, heads * hd, h);
                attn = Norm(attn, $"{p}.post_self_attn_layernorm.weight", n, h);
                for (int i = 0; i < x.Length; i++) x[i] += attn[i];
                float[] f = Norm(x, $"{p}.pre_feedforward_layernorm.weight", n, h);
                float[] gate = DacOps.Linear(backend, f, w[$"{p}.mlp.gate_proj.weight"], n, h, cfg.IntermediateSize);
                float[] up = DacOps.Linear(backend, f, w[$"{p}.mlp.up_proj.weight"], n, h, cfg.IntermediateSize);
                for (int i = 0; i < gate.Length; i++) gate[i] = T5Gemma2TextEncoder.GeluTanh(gate[i]) * up[i];
                float[] ff = DacOps.Linear(backend, gate, w[$"{p}.mlp.down_proj.weight"], n, cfg.IntermediateSize, h);
                ff = Norm(ff, $"{p}.post_feedforward_layernorm.weight", n, h);
                for (int i = 0; i < x.Length; i++) x[i] += ff[i];
            }
            return Norm(x, "text_encoder.norm.weight", n, h);
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }

    /// <summary>Four layers, two that slide over a window of six and two that see everything, on a segment of 23 tokens: the
    /// encoder's output is the former encoder's. A wrong mask on either kind of layer, or none, moves it.</summary>
    [Fact]
    public void OnTheCpuBackendTheWholeEncoderMatchesTheFormerEncoder()
    {
        Dictionary<string, Tensor> weights = SmallWeights();
        try
        {
            int[] tokens = new int[23];
            Random random = new(7);
            for (int i = 0; i < tokens.Length; i++) tokens[i] = random.Next(0, 59);
            using CpuBackend backend = new();
            float[] expected = FormerEncode(backend, Small, weights, tokens);
            using T5Gemma2TextEncoder encoder = new(Small);
            encoder.LoadWeights(weights);
            float[] actual = encoder.Encode(backend, tokens);
            AssertClose(expected, actual, Tolerance, "the whole encoder");
            // The oracle has teeth: with every layer taken as full attention it is not the encoder's output.
            float[] unmasked = FormerEncode(backend, Small with { SlidingPatternPeriod = 1 }, weights, tokens);
            double apart = 0;
            for (int i = 0; i < expected.Length; i++) apart = Math.Max(apart, Math.Abs(expected[i] - unmasked[i]));
            Assert.True(apart > 100 * Tolerance, $"the window changes the output by only {apart}");
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}
