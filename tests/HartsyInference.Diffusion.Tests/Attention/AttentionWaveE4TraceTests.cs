using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.Diamond;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using static HartsyInference.Tests.Common.Attention.AttentionTraceReport;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>The op sequence of each Wave E4 Diffusion-package attention site on the recording stub (no compute): every backend op in order, the
/// attention call's scale, F16 assertion and mask, the operand and output shapes and dtypes, and which of those tensors are freed when the site
/// returns. The pins were recorded from the code before the sites moved onto the shared layer and must not change. Ops the stub leaves to the
/// interface's host defaults are not recorded.</summary>
public sealed unsafe class AttentionWaveE4TraceTests
{
    private static void Pin(string site, string[] actual)
    {
        if (!Pins.TryGetValue(site, out string[]? expected))
            Assert.Fail($"No pin for '{site}'. Recorded:\n[\n    {string.Join(",\n    ", actual.Select(a => $"\"{a}\""))},\n]");
        Assert.Equal(expected, actual);
    }

    private static void PinDigest(string site, string actual)
    {
        if (!Digests.TryGetValue(site, out string? expected))
            Assert.Fail($"No digest for '{site}'. Recorded: {actual}");
        Assert.Equal(expected, actual);
    }

    private static Tensor T(params long[] dims) => new(new TensorShape(dims), DType.F32);

    private static Tensor T16(params long[] dims) => new(new TensorShape(dims), DType.F16);

    private static void LayerNormW(Weights w, string name, int n) { w.AddNorm($"{name}.weight", n); w.AddBias($"{name}.bias", n); }

    private static object NewPrivate(Type owner, string nested, params object[] ctorArgs) =>
        Activator.CreateInstance(owner.GetNestedType(nested, BindingFlags.NonPublic | BindingFlags.Public)!,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null, ctorArgs, null)!;

    private static object Invoke(object target, string method, params object?[] args) =>
        target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(target, args)!;

    // ── T5 block ───────────────────────────────────────────────────────────────────────────────────────────────────

    private static void T5(AttentionTraceBackend b, int batch, bool padded, bool gated, float? scale)
    {
        const int d = 16, heads = 2, hd = 8, ff = 24, seq = 7;
        Weights w = new(400);
        w.AddNorm("t.layer.0.layer_norm.weight", d); w.AddNorm("t.layer.1.layer_norm.weight", d);
        foreach (string n in new[] { "q", "k", "v" }) w.Add($"t.layer.0.SelfAttention.{n}.weight", heads * hd, d);
        w.Add("t.layer.0.SelfAttention.o.weight", d, heads * hd);
        if (gated) { w.Add("t.layer.1.DenseReluDense.wi_0.weight", ff, d); w.Add("t.layer.1.DenseReluDense.wi_1.weight", ff, d); }
        else w.Add("t.layer.1.DenseReluDense.wi.weight", ff, d);
        w.Add("t.layer.1.DenseReluDense.wo.weight", d, ff);
        T5Block block = new(d, heads, hd, 1e-6f, gatedFeedForward: gated, useReluActivation: !gated, attentionScale: scale ?? 1f / MathF.Sqrt(hd));
        block.LoadWeights(w, "t");
        Tensor x = T(batch, seq, d), bias = T(1, heads, seq, seq);
        Tensor? mask = padded ? T(batch, seq) : null;
        if (mask is not null) new Span<float>((float*)mask.DataPointer, (int)mask.ElementCount).Fill(1f);
        block.Forward(b, x, bias, mask);
        Assert.False(IsDisposed(bias));
    }

    [Fact]
    public void T5Block_SingleBatchWithoutPaddingAttendsWithTheSharedBiasItself() =>
        Pin("t5_b1", Run(b => T5(b, 1, padded: false, gated: true, scale: null)));

    [Fact]
    public void T5Block_PaddedBatchAttendsWithItsOwnCombinedMaskAndFreesIt() =>
        Pin("t5_b2_padded", Run(b => T5(b, 2, padded: true, gated: true, scale: null)));

    [Fact]
    public void T5Block_V10WithUnitScaleAndReluAttendsAtScaleOne() =>
        Pin("t5_v10_scale1", Run(b => T5(b, 1, padded: true, gated: false, scale: 1f)));

    // ── CLIP ───────────────────────────────────────────────────────────────────────────────────────────────────────

    private static void ClipWeights(Weights w, int h, int inter)
    {
        LayerNormW(w, "c.layer_norm1", h); LayerNormW(w, "c.layer_norm2", h);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Linear($"c.self_attn.{n}", h, h, true);
        w.Linear("c.mlp.fc1", inter, h, true); w.Linear("c.mlp.fc2", h, inter, true);
    }

    [Fact]
    public void ClipTextLayer_AttendsCausalWithTheMaskCopyAndFreesIt() => Pin("clip_text", Run(b =>
    {
        const int h = 16, inter = 24, seq = 7;
        Weights w = new(410);
        ClipWeights(w, h, inter);
        ClipTransformerLayer layer = new(h, 2, inter, 1e-5f, useQuickGelu: true);
        layer.LoadWeights(w, "c");
        Tensor causal = T(seq, seq);
        layer.Forward(b, T(2, seq, h), causal);
        Assert.False(IsDisposed(causal));
    }));

    [Fact]
    public void ClipVisionLayer_AttendsBidirectionalWithoutAMask() => Pin("clip_vision", Run(b =>
    {
        const int h = 16, inter = 24, seq = 7;
        Weights w = new(411);
        ClipWeights(w, h, inter);
        ClipVisionTransformerLayer layer = new(h, 2, inter, 1e-5f, useQuickGelu: true);
        layer.LoadWeights(w, "c");
        layer.Forward(b, T(2, seq, h));
    }));

    // ── ACE-Step 1.5 attention ─────────────────────────────────────────────────────────────────────────────────────

    private static AceStep15Attention AceStep()
    {
        const int hidden = 32, heads = 4, kvHeads = 2, hd = 8;
        AceStep15Config cfg = new() { HiddenSize = hidden, NumHeads = heads, NumKvHeads = kvHeads, HeadDim = hd, RmsNormEps = 1e-6f };
        Weights w = new(420);
        w.Add("a.q_proj.weight", heads * hd, hidden); w.Add("a.k_proj.weight", kvHeads * hd, hidden);
        w.Add("a.v_proj.weight", kvHeads * hd, hidden); w.Add("a.o_proj.weight", hidden, heads * hd);
        w.AddNorm("a.q_norm.weight", hd); w.AddNorm("a.k_norm.weight", hd);
        AceStep15Attention attn = new(cfg);
        attn.LoadWeights(w, "a");
        return attn;
    }

    [Fact]
    public void AceStep15Attention_SelfSlidingRopesRepeatsKvAndAttendsWithTheMask() => Pin("acestep_self_sliding", Run(b =>
    {
        const int s = 9, hd = 8;
        (float[] cos, float[] sin) = AceStep15Attention.BuildRopeTables(s, hd, 10_000.0);
        Tensor mask = AceStep15Attention.BuildSlidingMask(s, 2);
        AceStep().Forward(b, T(1, s, 32), null, cos, sin, mask);
        Assert.False(IsDisposed(mask));
    }));

    [Fact]
    public void AceStep15Attention_CrossAttendsOverTheConditionWithoutRopeOrMask() =>
        Pin("acestep_cross", Run(b => AceStep().Forward(b, T(1, 9, 32), T(1, 6, 32), null, null, null)));

    [Fact]
    public void AceStep15Attention_F16ActivationsAssertTheBoundedPath() =>
        Pin("acestep_cross_f16", Run(b => AceStep().Forward(b, T16(1, 9, 32), T16(1, 6, 32), null, null, null)));

    // ── Leftovers inside larger models ─────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void DiamondSelfAttention_SplitsHeadsOnTheHostAttendsAndMergesThemBack() => Pin("diamond", Run(b =>
    {
        const int c = 16;
        DiamondConfig cfg = new() { AttnHeadDim = 8, GnGroupSize = 8 };
        DiamondOps.SelfAttention(b, T(1, c, 4, 4), T(c), T(c), T(3 * c, c, 1, 1), T(3 * c), T(c, c, 1, 1), T(c), cfg);
    }));

    private static void QwenImage(AttentionTraceBackend b, int batch)
    {
        const int dim = 16;
        Weights wt = new(430);
        wt.AddTensor("a.norm.gamma", 0.5f, dim, 1, 1);
        wt.AddTensor("a.to_qkv.weight", 0.3f, 3 * dim, dim, 1, 1); wt.AddBias("a.to_qkv.bias", 3 * dim);
        wt.AddTensor("a.proj.weight", 0.3f, dim, dim, 1, 1); wt.AddBias("a.proj.bias", dim);
        QwenImageAttentionBlock block = new(dim);
        block.LoadWeights(wt, "a");
        block.Forward(b, T(batch, dim, 4, 3));
    }

    [Fact]
    public void QwenImageVaeAttention_SingleBatchSlicesOnTheDeviceAndAttendsOneWideHead() =>
        Pin("qwenimage_b1", Run(b => QwenImage(b, 1)));

    [Fact]
    public void QwenImageVaeAttention_BatchOfTwoSplitsOnTheHostAndAttendsOneWideHead() =>
        Pin("qwenimage_b2", Run(b => QwenImage(b, 2)));

    [Fact]
    public void H3VideoVaeVitBlock_PermutesAttendsAndMergesThroughTheOutputProjection() => Pin("h3_vit", Run(b =>
    {
        const int seq = 9;
        MiniMaxH3VideoVaeConfig cfg = new() { Heads = 2, DimHead = 8 };
        int dim = cfg.Dim, hd = cfg.DimHead, inner = cfg.FfnInner;
        Weights w = new(440);
        w.AddNorm("v.norm1.weight", dim); w.AddNorm("v.norm2.weight", dim);
        w.AddTensor("v.scale1", 0.5f, dim); w.AddTensor("v.scale2", 0.5f, dim);
        w.Linear("v.attn.to_qkv", 3 * dim, dim, true); w.Linear("v.attn.to_out", dim, dim, true);
        w.Linear("v.ff.w1", 2 * inner, dim, true); w.Linear("v.ff.w2", dim, inner, true);
        MiniMaxH3VideoVaeVitBlock block = new(cfg);
        block.LoadWeights(w, "v", []);
        block.Forward(b, T(seq, dim), T(seq, hd), T(seq, hd), T(hd));
    }));

    [Fact]
    public void Qwen25VlVisionBlock_AttendsWithTheCallersMaskAndLeavesItAlive() => Pin("qwen25vl", Run(b =>
    {
        const int hidden = 16, heads = 2, hd = 8, inter = 24, np = 12;
        Qwen25VlVisionConfig cfg = new() { HiddenSize = hidden, NumHeads = heads, IntermediateSize = inter };
        Weights w = new(450);
        w.AddNorm("b.norm1.weight", hidden); w.AddNorm("b.norm2.weight", hidden);
        w.Linear("b.attn.qkv", 3 * hidden, hidden, true); w.Linear("b.attn.proj", hidden, hidden, true);
        w.Linear("b.mlp.gate_proj", inter, hidden, true); w.Linear("b.mlp.up_proj", inter, hidden, true); w.Linear("b.mlp.down_proj", hidden, inter, true);
        object block = NewPrivate(typeof(Qwen25VlVisionEncoder), "Block", cfg);
        Invoke(block, "LoadWeights", w, "b");
        Tensor mask = T(1, 1, np, np);
        Invoke(block, "Forward", b, T(1, np, hidden), new float[np * hd], new float[np * hd], np, mask);
        Assert.False(IsDisposed(mask));
    }));

    [Fact]
    public void Qwen3VlVisionBlock_RopesThePermutedHeadsAndAttendsWithoutAMask() => Pin("qwen3vl", Run(b =>
    {
        const int hidden = 16, heads = 2, hd = 8, half = hd / 2, inter = 24, seq = 12;
        Qwen3VlVisionConfig cfg = new() { HiddenSize = hidden, NumHeads = heads, IntermediateSize = inter };
        Weights w = new(452);
        LayerNormW(w, "b.norm1", hidden); LayerNormW(w, "b.norm2", hidden);
        w.Linear("b.attn.qkv", 3 * hidden, hidden, true); w.Linear("b.attn.proj", hidden, hidden, true);
        w.Linear("b.mlp.linear_fc1", inter, hidden, true); w.Linear("b.mlp.linear_fc2", hidden, inter, true);
        object block = NewPrivate(typeof(Qwen3VlVisionEncoder), "VisionBlock", cfg);
        Invoke(block, "LoadWeights", w, "b");
        Invoke(block, "Forward", b, T(1, seq, hidden), new float[seq * half], new float[seq * half], seq);
    }));

    [Fact]
    public void StableAudioBlock_AttendsSelfThenCrossWithGroupedKvRepeated() => Pin("stableaudio", Run(b =>
    {
        const int dim = 32, heads = 4, hd = 8, kvHeads = 2, inner = 48, rows = 9, condLen = 5;
        StableAudioDitConfig cfg = new() { HiddenSize = dim, NumHeads = heads, HeadDim = hd, CrossKvHeads = kvHeads, FfInner = inner, RopeDim = 4 };
        Weights w = new(460);
        LayerNormW(w, "b.pre_norm", dim); LayerNormW(w, "b.cross_attend_norm", dim); LayerNormW(w, "b.ff_norm", dim);
        w.Add("b.self_attn.to_qkv.weight", 3 * dim, dim); w.Add("b.self_attn.to_out.weight", dim, dim);
        LayerNormW(w, "b.self_attn.q_norm", hd); LayerNormW(w, "b.self_attn.k_norm", hd);
        w.Add("b.cross_attn.to_q.weight", dim, dim); w.Add("b.cross_attn.to_kv.weight", 2 * kvHeads * hd, dim); w.Add("b.cross_attn.to_out.weight", dim, dim);
        LayerNormW(w, "b.cross_attn.q_norm", hd); LayerNormW(w, "b.cross_attn.k_norm", hd);
        w.Linear("b.ff.ff.0.proj", 2 * inner, dim, true); w.Linear("b.ff.ff.2", dim, inner, true);
        object block = NewPrivate(typeof(StableAudioDit), "Block", cfg);
        Invoke(block, "LoadWeights", w, "b");
        Invoke(block, "Forward", b, T(1, rows, dim), T(1, condLen, dim), rows, condLen, T(rows, hd), T(rows, hd));
    }));

    // ── Sites with no golden fixture: traced, and digested on the CPU backend ──────────────────────────────────────

    private static void Fill(Tensor t, uint seed, float scale)
    {
        Tensor s = Seeded(t.Shape, seed, scale);
        new ReadOnlySpan<float>((float*)s.DataPointer, (int)t.ElementCount).CopyTo(new Span<float>((float*)t.DataPointer, (int)t.ElementCount));
        s.Dispose();
    }

    private static MiniMaxH3AudioVaeEncoder AudioVaeHead(int width, MiniMaxH3AudioVaeConfig cfg, bool seeded)
    {
        const int latent = 6;
        int mlp = latent * cfg.MlpRatio;
        MiniMaxH3AudioVaeEncoder enc = new(cfg);
        uint seed = 7000;
        void Set(string field, params long[] dims)
        {
            Tensor t = T(dims);
            if (seeded) Fill(t, seed++, field.Contains("Norm") && field.EndsWith('W') ? 0.1f : 0.3f);
            typeof(MiniMaxH3AudioVaeEncoder).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(enc, t);
        }
        foreach (string n in new[] { "_norm1", "_norm2", "_norm3" }) { Set(n + "W", width); Set(n + "B", width); }
        Set("_qkvW", 3 * width, width); Set("_qkvBias", 3 * width);
        Set("_attnProjW", latent, latent); Set("_attnProjB", latent);
        Set("_projW", latent, width); Set("_projB", latent);
        Set("_mlpNormW", latent); Set("_mlpNormB", latent);
        Set("_w0W", mlp, latent); Set("_w0B", mlp); Set("_w1W", mlp, latent); Set("_w1B", mlp); Set("_w2W", latent, mlp); Set("_w2B", latent);
        Set("_meanProjW", latent, latent);
        return enc;
    }

    private static Tensor PosteriorHead(IBackend backend, int batch, bool seeded)
    {
        const int width = 16, frames = 5;
        MiniMaxH3AudioVaeEncoder enc = AudioVaeHead(width, new MiniMaxH3AudioVaeConfig { AttnHeads = 2 }, seeded);
        Tensor x = T(batch, width, frames);
        if (seeded) Fill(x, 7100, 1f);
        return (Tensor)typeof(MiniMaxH3AudioVaeEncoder).GetMethod("PosteriorHead", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(enc, [backend, x, batch])!;
    }

    [Fact]
    public void H3AudioVaePosteriorHead_AttendsCausalOverSplitHeadsAndFreesEverythingOnReturn() =>
        Pin("h3_audio_posterior", Run(b => PosteriorHead(b, 2, seeded: false)));

    [Fact]
    public void H3AudioVaePosteriorHead_ReproducesThePreMigrationOutputBitForBit()
    {
        using CpuBackend cpu = new();
        using Tensor y = PosteriorHead(cpu, 2, seeded: true);
        PinDigest("h3_audio_posterior", Digest("out", y));
    }

    private static (object Dit, object Block, object Scratch, MiniMaxMusic3DitConfig Cfg) MusicBlock(int rows, int batch, bool seeded)
    {
        MiniMaxMusic3DitConfig cfg = new()
        {
            InChannels = 4, ConditionDim = 8, NumLayers = 1, NumAttentionHeads = 2, AttentionHeadDim = 8, FfInnerDim = 24, RotaryDim = 4,
            FourierEmbeddingDim = 8,
        };
        MiniMaxMusic3Dit dit = new(cfg);
        int dim = cfg.InnerDim;
        Weights w = new(470);
        LayerNormW(w, "b.norm1", dim); LayerNormW(w, "b.norm2", dim);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "to_out.0" }) w.Add($"b.attn.{n}.weight", dim, dim);
        w.Linear("b.ff_in", 2 * cfg.FfInnerDim, dim, true); w.Linear("b.ff_out", dim, cfg.FfInnerDim, true);
        Type dt = typeof(MiniMaxMusic3Dit);
        object block = NewPrivate(dt, "Block", cfg);
        Invoke(block, "Load", w, "b");
        object scratch = Activator.CreateInstance(dt.GetNestedType("BlockScratch", BindingFlags.NonPublic)!,
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public, null, [rows, batch, cfg], null)!;
        return (dit, block, scratch, cfg);
    }

    private static Tensor MusicForward(IBackend backend, int rows, int batch, bool seeded, out object scratch)
    {
        (_, object block, object sc, MiniMaxMusic3DitConfig cfg) = MusicBlock(rows, batch, seeded);
        scratch = sc;
        Tensor hidden = T(batch * rows, cfg.InnerDim), cos = T(batch, rows, cfg.AttentionHeadDim), sin = T(batch, rows, cfg.AttentionHeadDim);
        if (seeded) { Fill(hidden, 7200, 1f); Fill(cos, 7201, 1f); Fill(sin, 7202, 1f); }
        return (Tensor)Invoke(block, "Forward", backend, hidden, cfg, cos, sin, sc);
    }

    [Fact]
    public void MiniMaxMusic3Block_AttendsIntoTheSharedScratchAndFreesNothingOfIt()
    {
        string[] trace = Run(b =>
        {
            Tensor y = MusicForward(b, 6, 2, seeded: false, out object scratch);
            Tensor head = (Tensor)scratch.GetType().GetProperty("HeadAttention", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(scratch)!;
            Assert.False(IsDisposed(head));
            Assert.False(IsDisposed(y));
        });
        Pin("minimaxmusic3_block", trace);
    }

    [Fact]
    public void MiniMaxMusic3Block_ReproducesThePreMigrationOutputBitForBit()
    {
        using CpuBackend cpu = new();
        using Tensor y = MusicForward(cpu, 6, 2, seeded: true, out _);
        PinDigest("minimaxmusic3_block", Digest("out", y));
    }

    private static readonly Dictionary<string, string> Digests = new()
    {
        ["h3_audio_posterior"] = "out=e3fcc23b08c118fd3ebf1971eec5a294845b2f3764eb21c303c4c5c3479b52e9",
        ["minimaxmusic3_block"] = "out=d9c4a80477cd32a799f362e41bf76f7b0d4ce46f9eb48c047d242bc3a413b849",
    };

    private static readonly Dictionary<string, string[]> Pins = new()
    {
        ["h3_vit"] =
        [
            "RmsNorm w16",
            "Linear w48",
            "SliceLastDim @0",
            "SliceLastDim @8",
            "SliceLastDim @16",
            "RmsNorm w8",
            "RmsNorm w8",
            "ApplyRopeSingle",
            "ApplyRopeSingle",
            "Permute0213 9,2,8",
            "Permute0213 9,2,8",
            "Permute0213 9,2,8",
            "Attn Dense scale=0.353553 f16=False mask=False",
            "Permute0213 2,9,8",
            "Linear w16",
            "RmsNorm w16",
            "Linear w128",
            "SliceLastDim @0",
            "SliceLastDim @64",
            "Silu",
            "Mul",
            "Linear w16",
            "out [1, 2, 9, 8]F32 q [1, 2, 9, 8]F32 k [1, 2, 9, 8]F32 v [1, 2, 9, 8]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["qwen3vl"] =
        [
            "LayerNorm w16",
            "Linear w48",
            "Attn Dense scale=0.353553 f16=False mask=False",
            "Linear w16",
            "Add",
            "LayerNorm w16",
            "Linear w24",
            "Gelu",
            "Linear w16",
            "Add",
            "out [1, 2, 12, 8]F32 q [1, 2, 12, 8]F32 k [1, 2, 12, 8]F32 v [1, 2, 12, 8]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["minimaxmusic3_block"] =
        [
            "LayerNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "ApplyRopeSingle",
            "ApplyRopeSingle",
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "Attn Dense scale=0.353553 f16=False mask=False",
            "Permute0213 2,6,8",
            "Linear w16",
            "Add",
            "LayerNorm w16",
            "Linear w48",
            "Split 2 dim=1",
            "Silu",
            "Mul",
            "Linear w16",
            "Add",
            "out [2, 2, 6, 8]F32 q [2, 2, 6, 8]F32 k [2, 2, 6, 8]F32 v [2, 2, 6, 8]F32 mask none",
            "freed out=False q=False k=False v=False mask=False",
        ],
        ["qwenimage_b2"] =
        [
            "Conv2D w48",
            "Attn Dense scale=0.25 f16=False mask=False",
            "Conv2D w16",
            "Add",
            "out [2, 1, 12, 16]F32 q [2, 1, 12, 16]F32 k [2, 1, 12, 16]F32 v [2, 1, 12, 16]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["t5_v10_scale1"] =
        [
            "RmsNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Attn Dense scale=1 f16=False mask=True",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "Linear w24",
            "out [1, 2, 7, 8]F32 q [1, 2, 7, 8]F32 k [1, 2, 7, 8]F32 v [1, 2, 7, 8]F32 mask [1, 2, 7, 7]F32",
            "freed out=True q=True k=True v=True mask=True",
        ],
        ["qwenimage_b1"] =
        [
            "Conv2D w48",
            "SliceRows 16@0",
            "Transpose2D 16,12",
            "SliceRows 16@16",
            "Transpose2D 16,12",
            "SliceRows 16@32",
            "Transpose2D 16,12",
            "Attn Dense scale=0.25 f16=False mask=False",
            "Transpose2D 12,16",
            "Conv2D w16",
            "Add",
            "out [1, 1, 12, 16]F32 q [1, 1, 12, 16]F32 k [1, 1, 12, 16]F32 v [1, 1, 12, 16]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["qwen25vl"] =
        [
            "RmsNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 12,2,8",
            "Permute0213 12,2,8",
            "Permute0213 12,2,8",
            "Attn Dense scale=0.353553 f16=False mask=True",
            "Permute0213 2,12,8",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "Linear w24",
            "Linear w24",
            "Silu",
            "Mul",
            "Linear w16",
            "Add",
            "out [1, 2, 12, 8]F32 q [1, 2, 12, 8]F32 k [1, 2, 12, 8]F32 v [1, 2, 12, 8]F32 mask [1, 1, 12, 12]F32",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["clip_text"] =
        [
            "LayerNorm w16",
            "BatchedMatMul [2, 7, 16] x [16, 16]",
            "BatchedMatMul [2, 7, 16] x [16, 16]",
            "BatchedMatMul [2, 7, 16] x [16, 16]",
            "Attn Dense scale=0.353553 f16=False mask=True",
            "BatchedMatMul [2, 7, 16] x [16, 16]",
            "Add",
            "LayerNorm w16",
            "BatchedMatMul [2, 7, 16] x [16, 24]",
            "BatchedMatMul [2, 7, 24] x [24, 16]",
            "Add",
            "out [2, 2, 7, 8]F32 q [2, 2, 7, 8]F32 k [2, 2, 7, 8]F32 v [2, 2, 7, 8]F32 mask [1, 1, 7, 7]F32",
            "freed out=True q=True k=True v=True mask=True",
        ],
        ["acestep_cross"] =
        [
            "Linear w32",
            "Linear w16",
            "Linear w16",
            "RmsNorm w8",
            "RmsNorm w8",
            "Permute0213 9,4,8",
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "RepeatKvHeads 2x2",
            "RepeatKvHeads 2x2",
            "Attn Dense scale=0.353553 f16=False mask=False",
            "Permute0213 4,9,8",
            "Linear w32",
            "out [1, 4, 9, 8]F32 q [1, 4, 9, 8]F32 k [1, 4, 6, 8]F32 v [1, 4, 6, 8]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["acestep_cross_f16"] =
        [
            "Linear w32",
            "Linear w16",
            "Linear w16",
            "RmsNorm w8",
            "RmsNorm w8",
            "Permute0213 9,4,8",
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "RepeatKvHeads 2x2",
            "RepeatKvHeads 2x2",
            "Attn Dense scale=0.353553 f16=True mask=False",
            "Permute0213 4,9,8",
            "Linear w32",
            "out [1, 4, 9, 8]F16 q [1, 4, 9, 8]F16 k [1, 4, 6, 8]F16 v [1, 4, 6, 8]F16 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["h3_audio_posterior"] =
        [
            "LayerNorm w16",
            "Linear w48",
            "Attn Dense scale=0.353553 f16=False mask=True",
            "Linear w6",
            "LayerNorm w16",
            "Linear w6",
            "Add",
            "LayerNorm w16",
            "LayerNorm w6",
            "Linear w12",
            "Gelu",
            "Linear w12",
            "Mul",
            "Linear w6",
            "Add",
            "out [2, 2, 5, 8]F32 q [2, 2, 5, 8]F32 k [2, 2, 5, 8]F32 v [2, 2, 5, 8]F32 mask [1, 1, 5, 5]F32",
            "freed out=True q=True k=True v=True mask=True",
        ],
        ["acestep_self_sliding"] =
        [
            "Linear w32",
            "Linear w16",
            "Linear w16",
            "RmsNorm w8",
            "RmsNorm w8",
            "ApplyRopeSingle",
            "ApplyRopeSingle",
            "Permute0213 9,4,8",
            "Permute0213 9,2,8",
            "Permute0213 9,2,8",
            "RepeatKvHeads 2x2",
            "RepeatKvHeads 2x2",
            "Attn Dense scale=0.353553 f16=False mask=True",
            "Permute0213 4,9,8",
            "Linear w32",
            "out [1, 4, 9, 8]F32 q [1, 4, 9, 8]F32 k [1, 4, 9, 8]F32 v [1, 4, 9, 8]F32 mask [1, 1, 9, 9]F32",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["t5_b1"] =
        [
            "RmsNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Attn Dense scale=0.353553 f16=False mask=True",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "Linear w24",
            "Linear w24",
            "Gelu",
            "Mul",
            "Linear w16",
            "Add",
            "out [1, 2, 7, 8]F32 q [1, 2, 7, 8]F32 k [1, 2, 7, 8]F32 v [1, 2, 7, 8]F32 mask [1, 2, 7, 7]F32",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["clip_vision"] =
        [
            "LayerNorm w16",
            "BatchedMatMul [2, 7, 16] x [16, 16]",
            "BatchedMatMul [2, 7, 16] x [16, 16]",
            "BatchedMatMul [2, 7, 16] x [16, 16]",
            "Attn Dense scale=0.353553 f16=False mask=False",
            "BatchedMatMul [2, 7, 16] x [16, 16]",
            "Add",
            "LayerNorm w16",
            "BatchedMatMul [2, 7, 16] x [16, 24]",
            "BatchedMatMul [2, 7, 24] x [24, 16]",
            "Add",
            "out [2, 2, 7, 8]F32 q [2, 2, 7, 8]F32 k [2, 2, 7, 8]F32 v [2, 2, 7, 8]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["t5_b2_padded"] =
        [
            "RmsNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Attn Dense scale=0.353553 f16=False mask=True",
            "Linear w16",
            "Add",
            "RmsNorm w16",
            "Linear w24",
            "Linear w24",
            "Gelu",
            "Mul",
            "Linear w16",
            "Add",
            "out [2, 2, 7, 8]F32 q [2, 2, 7, 8]F32 k [2, 2, 7, 8]F32 v [2, 2, 7, 8]F32 mask [2, 2, 7, 7]F32",
            "freed out=True q=True k=True v=True mask=True",
        ],
        ["stableaudio"] =
        [
            "LayerNorm w32",
            "Linear w96",
            "SliceLastDim @0",
            "SliceLastDim @32",
            "SliceLastDim @64",
            "LayerNorm w8",
            "ApplyRopeSingle",
            "Permute0213 9,4,8",
            "LayerNorm w8",
            "ApplyRopeSingle",
            "Permute0213 9,4,8",
            "Permute0213 9,4,8",
            "Attn Dense scale=0.353553 f16=False mask=False",
            "Permute0213 4,9,8",
            "Linear w32",
            "Add",
            "LayerNorm w32",
            "Linear w32",
            "LayerNorm w8",
            "Permute0213 9,4,8",
            "Linear w32",
            "SliceLastDim @0",
            "SliceLastDim @16",
            "LayerNorm w8",
            "Permute0213 5,2,8",
            "Permute0213 5,2,8",
            "RepeatKvHeads 2x2",
            "RepeatKvHeads 2x2",
            "Attn Dense scale=0.353553 f16=False mask=False",
            "Permute0213 4,9,8",
            "Linear w32",
            "Add",
            "LayerNorm w32",
            "Linear w96",
            "SliceLastDim @0",
            "SliceLastDim @48",
            "Silu",
            "Mul",
            "Linear w32",
            "Add",
            "out [1, 4, 9, 8]F32 q [1, 4, 9, 8]F32 k [1, 4, 9, 8]F32 v [1, 4, 9, 8]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
            "out [1, 4, 9, 8]F32 q [1, 4, 9, 8]F32 k [1, 4, 5, 8]F32 v [1, 4, 5, 8]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
        ["diamond"] =
        [
            "GroupNorm 2",
            "Conv2D w48",
            "Attn Dense scale=0.353553 f16=False mask=False",
            "Conv2D w16",
            "Add",
            "out [1, 2, 16, 8]F32 q [1, 2, 16, 8]F32 k [1, 2, 16, 8]F32 v [1, 2, 16, 8]F32 mask none",
            "freed out=True q=True k=True v=True mask=False",
        ],
    };
}
