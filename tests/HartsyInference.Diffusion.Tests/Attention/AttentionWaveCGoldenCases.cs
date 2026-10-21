using System.Security.Cryptography;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.LLM.Transformer;
using HartsyInference.Video.Models.Cosmos;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>Forward passes of the Wave C attention code (Wan video block and its variants, S2V audio injector, Matrix-Game 3 action
/// module, Oasis, VAE attention, UNet cross-attention, ControlNet-Union, IP-Adapter resampler, SeedVR2, MiniMax-H3, Cosmos AR) on tiny
/// pseudo-random weights and inputs, each reduced to SHA-256 digests of its output bits. Everything runs on the CPU backend through the
/// public/internal constructors and Forward methods, so the same cases produce the golden fixture from the pre-migration source and must
/// reproduce it after the attention refactor. Weights come from the per-case-salted <c>Weights</c> harness, never
/// <c>WanSyntheticWeights</c> (its static seed counter makes the weights order-dependent).</summary>
internal static unsafe class AttentionWaveCGoldenCases
{
    private const float NegInf = -1e9f;
    private const int WanHeads = 2, WanHeadDim = 16, WanDim = WanHeads * WanHeadDim, WanFfn = 64;
    private const int WanFrames = 2, WanGridH = 3, WanGridW = 2, WanHw = WanGridH * WanGridW, WanS = WanFrames * WanHw, WanCtx = 7;

    /// <summary>Every case, in fixture order.</summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "wan_self_t2v", "wan_t2v_nobias_nocrossnorm", "wan_i2v_dual_kv", "wan_i2v_dual_kv_nonorm2bias", "wan_mg2_block_causal",
        "wan_sigma_theta_per_head", "wan_post_hooks", "wan_animate2_prepass_capture",
        "wan_animate2_splice_plain", "wan_animate2_splice_logscale_pose_ref", "wan_cp_rank0", "wan_cp_rank1",
        "wan_s2v_injector_exact", "wan_s2v_injector_with_passthrough",
        "mg3_action_mouse", "mg3_action_keyboard",
        "oasis_block", "oasis_vitvae_encode", "oasis_vitvae_decode",
        "vae_attn_2d", "vae_attn_3d_causal", "vae_attn_3d_noncausal", "wan22_attn_f32", "wan22_attn_bf16",
        "unet_self_cross_b1", "unet_self_cross_b2", "unet_cross_cache", "unet_ipa_scale_0p6", "unet_ipa_scale_0",
        "controlnet_union_b1", "controlnet_union_b2", "ipa_resampler_layer_b1", "ipa_resampler_layer_b2",
        "seedvr2_v2_tiny", "seedvr2_v1_tiny", "h3_dense", "h3_chunked", "h3_refiner",
        "cosmos_ar_prefill", "cosmos_ar_decode", "cosmos_ar_cross",
    ];

    /// <summary>Runs <paramref name="name"/> and returns its digests as space-separated <c>label=hex</c> pairs.</summary>
    public static string Run(string name)
    {
        using CpuBackend cpu = new();
        return name switch
        {
            _ when name.StartsWith("wan_cp_", StringComparison.Ordinal) => WanContextParallel(name),
            _ when name.StartsWith("wan_animate2_splice_", StringComparison.Ordinal) => WanAnimate2Splice(cpu, name),
            _ when name.StartsWith("wan_s2v_", StringComparison.Ordinal) => S2VInjector(cpu, name),
            _ when name.StartsWith("wan22_", StringComparison.Ordinal) => Wan22Attn(cpu, name),
            _ when name.StartsWith("wan_", StringComparison.Ordinal) => Wan(cpu, name),
            _ when name.StartsWith("mg3_", StringComparison.Ordinal) => Mg3Action(cpu, name),
            _ when name == "oasis_block" => OasisBlock(cpu),
            _ when name.StartsWith("oasis_vitvae_", StringComparison.Ordinal) => OasisVit(cpu, name),
            _ when name.StartsWith("vae_attn_", StringComparison.Ordinal) => VaeAttn(cpu, name),
            _ when name.StartsWith("unet_", StringComparison.Ordinal) => UNet(cpu, name),
            _ when name.StartsWith("controlnet_union_", StringComparison.Ordinal) => ControlNetUnion(cpu, name),
            _ when name.StartsWith("ipa_resampler_layer_", StringComparison.Ordinal) => IpaResamplerLayer(cpu, name),
            _ when name.StartsWith("seedvr2_", StringComparison.Ordinal) => SeedVr2(cpu, name),
            _ when name.StartsWith("h3_", StringComparison.Ordinal) => H3(cpu, name),
            _ when name.StartsWith("cosmos_ar_", StringComparison.Ordinal) => CosmosAr(cpu, name),
            _ => throw new ArgumentException($"Unknown attention golden case '{name}'.", nameof(name)),
        };
    }

    // ── Wan video block ────────────────────────────────────────────────────────────────────────────────────────────

    private static WanVideoConfig WanConfig() => new() { NumHeads = WanHeads, HeadDim = WanHeadDim, FfnDim = WanFfn };

    /// <summary>Block weights under <c>b.*</c>; <paramref name="biases"/> drops every projection bias, <paramref name="image"/> adds the I2V K/V.</summary>
    private static WanVideoBlock WanBlock(uint salt, bool crossNorm, bool biases = true, bool image = false, bool norm2Bias = true)
    {
        Weights w = new(salt);
        w.AddShape("b.scale_shift_table", [1, 6, WanDim], 0.3f);
        if (crossNorm)
        {
            w.AddNorm("b.norm2.weight", WanDim);
            if (norm2Bias) w.AddBias("b.norm2.bias", WanDim);
        }
        foreach (string a in new[] { "attn1", "attn2" })
        {
            foreach (string p in new[] { "to_q", "to_k", "to_v", "to_out.0" }) w.Linear($"b.{a}.{p}", WanDim, WanDim, biases);
            w.AddNorm($"b.{a}.norm_q.weight", WanDim);
            w.AddNorm($"b.{a}.norm_k.weight", WanDim);
        }
        if (image)
        {
            w.Linear("b.attn2.add_k_proj", WanDim, WanDim, biases);
            w.Linear("b.attn2.add_v_proj", WanDim, WanDim, biases);
            w.AddNorm("b.attn2.norm_added_k.weight", WanDim);
        }
        w.Linear("b.ffn.net.0.proj", WanFfn, WanDim, biases);
        w.Linear("b.ffn.net.2", WanDim, WanFfn, biases);
        WanVideoBlock block = new(WanConfig(), crossNorm);
        block.LoadWeights(w, "b");
        return block;
    }

    private static WanRope WanRopeFor(bool perHead = false) =>
        perHead ? new WanRope(WanHeadDim, 10000, 64, sigmaTheta: 0.2, numHeads: WanHeads) : new WanRope(WanHeadDim, 10000, 64);

    private static string Wan(CpuBackend cpu, string name)
    {
        Owner o = new();
        bool image = name.StartsWith("wan_i2v", StringComparison.Ordinal);
        bool noBias = name == "wan_t2v_nobias_nocrossnorm";
        WanVideoBlock block = WanBlock(201, crossNorm: !noBias, biases: !noBias, image: image, norm2Bias: name != "wan_i2v_dual_kv_nonorm2bias");
        WanRope rope = WanRopeFor(perHead: name == "wan_sigma_theta_per_head");
        (Tensor cos, Tensor sin) = rope.BuildCosSin(WanFrames, WanGridH, WanGridW);
        o.Own(cos); o.Own(sin);
        Tensor hidden = o.Own(Seeded(new TensorShape(WanS, WanDim), 2001));
        Tensor temb = o.Own(Seeded(new TensorShape(1, 6, WanDim), 2003, 0.5f));
        int ctxRows = image ? 4 + WanCtx : WanCtx;
        Tensor ctx = o.Own(Seeded(new TensorShape(ctxRows, WanDim), 2002));
        int imageLen = image ? 4 : 0;
        switch (name)
        {
            case "wan_self_t2v" or "wan_sigma_theta_per_head" or "wan_t2v_nobias_nocrossnorm" or "wan_i2v_dual_kv" or "wan_i2v_dual_kv_nonorm2bias":
            {
                Tensor outT = o.Own(block.Forward(cpu, hidden, ctx, temb, rope, cos, sin, WanS, imageContextLen: imageLen));
                return Finish(o, ("out", outT));
            }
            case "wan_mg2_block_causal":
            {
                Tensor mask = o.Own(MatrixGame2Transformer.BuildBlockCausalMask([0, 1, 3, 4], 4, 3, 2));
                Tensor h = o.Own(Seeded(new TensorShape(16, WanDim), 2004));
                (Tensor mcos, Tensor msin) = rope.BuildCosSin([0, 1, 3, 4], 2, 2);
                o.Own(mcos); o.Own(msin);
                Tensor outT = o.Own(block.Forward(cpu, h, ctx, temb, rope, mcos, msin, (int)h.Shape[0], selfAttnMask: mask));
                return Finish(o, ("out", outT));
            }
            case "wan_post_hooks":
            {
                Tensor outT = o.Own(block.Forward(cpu, hidden, ctx, temb, rope, cos, sin, WanS,
                    postCrossAttnHook: t => cpu.Scale(t, t, 1.25f), postSelfAttnHook: t => cpu.Scale(t, t, 0.8f)));
                return Finish(o, ("out", outT));
            }
            case "wan_animate2_prepass_capture":
            {
                WanAnimate2KvCapture capture = new() { CaptureProjected = true };
                Tensor outT = o.Own(block.Forward(cpu, hidden, ctx, temb, rope, cos, sin, WanS, selfAttnKvCapture: capture));
                o.Own(capture.Input!); o.Own(capture.K!); o.Own(capture.V!);
                return Finish(o, ("out", outT), ("input", capture.Input!), ("k", capture.K!), ("v", capture.V!));
            }
            default: throw new ArgumentException($"Unknown Wan case '{name}'.", nameof(name));
        }
    }

    /// <summary>Animate-2 frame-local splice over 3 generation frames of 3×2 tokens, one driving frame fewer than the generation.</summary>
    private static string WanAnimate2Splice(CpuBackend cpu, string name)
    {
        const int frames = 3, gh = 3, gw = 2, hw = gh * gw, s = frames * hw, refSeq = (frames - 1) * hw;
        bool scaled = name.EndsWith("logscale_pose_ref", StringComparison.Ordinal);
        Owner o = new();
        WanVideoBlock block = WanBlock(211, crossNorm: true);
        WanRope rope = WanRopeFor();
        (Tensor cos, Tensor sin) = rope.BuildCosSin(frames, gh, gw);
        (Tensor refCos, Tensor refSin) = rope.BuildCosSin(frames - 1, gh, gw, 1, 0, gw);
        o.Own(cos); o.Own(sin); o.Own(refCos); o.Own(refSin);
        Tensor hidden = o.Own(Seeded(new TensorShape(s, WanDim), 2101));
        Tensor ctx = o.Own(Seeded(new TensorShape(WanCtx, WanDim), 2102));
        Tensor temb = o.Own(Seeded(new TensorShape(1, 6, WanDim), 2103, 0.5f));
        Tensor driving = o.Own(Seeded(new TensorShape(refSeq, WanDim), 2104));
        Tensor? biasGen = scaled ? o.Own(WanAnimate2Transformer.BuildLogScaleBias(hw, s, 1.5f)) : null;
        Tensor? biasSpliced = scaled ? o.Own(WanAnimate2Transformer.BuildLogScaleBias(hw, s + hw, 1.5f)) : null;
        using WanAnimate2SelfAttnContext a2 = new()
        {
            RefCos = refCos, RefSin = refSin, GenFrames = frames, TokensPerFrame = hw,
            LogScaleBiasGen = biasGen, LogScaleBiasSpliced = biasSpliced,
            PoseStrength = scaled ? 0.7f : 1.0f, ReferenceImageStrength = scaled ? 1.3f : 1.0f,
            KeyBuffer = Zeroed(new TensorShape(1, WanHeads, s + hw, WanHeadDim)),
            ValueBuffer = Zeroed(new TensorShape(1, WanHeads, s + hw, WanHeadDim)),
            DrivingInput = driving,
        };
        Tensor outT = o.Own(block.Forward(cpu, hidden, ctx, temb, rope, cos, sin, s, animate2: a2));
        return Finish(o, ("out", outT));
    }

    /// <summary>Two ranks on two threads, each with its own backend and the real exchange; the requested rank's local output is digested.</summary>
    private static string WanContextParallel(string name)
    {
        int rankWanted = name.EndsWith("rank1", StringComparison.Ordinal) ? 1 : 0;
        Owner o = new();
        WanVideoBlock block = WanBlock(221, crossNorm: true);
        WanRope rope = WanRopeFor();
        (Tensor cos, Tensor sin) = rope.BuildCosSin(WanFrames, WanGridH, WanGridW);
        o.Own(cos); o.Own(sin);
        Tensor hidden = o.Own(Seeded(new TensorShape(WanS, WanDim), 2201));
        Tensor ctx = o.Own(Seeded(new TensorShape(WanCtx, WanDim), 2202));
        Tensor temb = o.Own(Seeded(new TensorShape(1, 6, WanDim), 2203, 0.5f));
        CpSequencePlan plan = CpSequencePlan.Create(WanFrames, WanHw, [1, 1]);
        using CpKvExchange exchange = new(plan);
        Tensor?[] outs = new Tensor?[2];
        Exception?[] errors = new Exception?[2];
        Thread[] threads = new Thread[2];
        for (int r = 0; r < 2; r++)
        {
            int rank = r;
            CpRankRange range = plan.Ranks[rank];
            Tensor localHidden = o.Own(HostRows(hidden, range.TokenStart, range.TokenCount));
            Tensor localCos = o.Own(HostRows(cos, range.TokenStart, range.TokenCount));
            Tensor localSin = o.Own(HostRows(sin, range.TokenStart, range.TokenCount));
            threads[rank] = new Thread(() =>
            {
                try
                {
                    using CpuBackend backend = new();
                    outs[rank] = block.Forward(backend, localHidden, ctx, temb, rope, localCos, localSin, range.TokenCount,
                        selfAttnKvRewrite: new CpExchangeRewrite(exchange, rank, 0));
                }
                catch (Exception e)
                {
                    errors[rank] = e;
                    exchange.Abort();
                }
            });
        }
        foreach (Thread t in threads) t.Start();
        foreach (Thread t in threads) t.Join();
        foreach (Exception? e in errors)
            if (e is not null) throw new InvalidOperationException("Context-parallel rank failed.", e);
        foreach (Tensor? t in outs) o.Own(t!);
        return Finish(o, ("out", outs[rankWanted]!));
    }

    // ── Wan S2V audio injector ─────────────────────────────────────────────────────────────────────────────────────

    private static string S2VInjector(CpuBackend cpu, string name)
    {
        const int frames = 3, tokensPerFrame = 4, nTok = 3, seqLen = frames * tokensPerFrame;
        bool passthrough = name.EndsWith("with_passthrough", StringComparison.Ordinal);
        Owner o = new();
        Weights w = new(231);
        const string p = "audio_injector.injector.0";
        foreach (string n in new[] { "q", "k", "v", "o" }) w.Linear($"{p}.{n}", WanDim, WanDim, true);
        w.AddNorm($"{p}.norm_q.weight", WanDim);
        w.AddNorm($"{p}.norm_k.weight", WanDim);
        w.Linear("audio_injector.injector_adain_layers.0.linear", 2 * WanDim, WanDim, true);
        WanS2VAudioInjector injector = new(1, WanDim, WanHeads, WanHeadDim);
        injector.LoadWeights(w);
        int totalS = passthrough ? seqLen + 5 : seqLen;
        Tensor hidden = o.Own(Seeded(new TensorShape(totalS, WanDim), 2301));
        Tensor local = o.Own(Seeded(new TensorShape(frames, nTok, WanDim), 2302));
        Tensor global = o.Own(Seeded(new TensorShape(frames, 1, WanDim), 2303));
        Tensor outT = o.Own(injector.Forward(cpu, hidden, 0, local, global, seqLen));
        return Finish(o, ("out", outT));
    }

    // ── Matrix-Game 3 action module ────────────────────────────────────────────────────────────────────────────────

    private static string Mg3Action(CpuBackend cpu, string name)
    {
        const int img = 16, stream = 32, heads = 4, headDim = 8, hiddenSize = 8, frames = 3, gh = 2, gw = 2, sp = gh * gw, window = 12;
        bool mouse = name.EndsWith("mouse", StringComparison.Ordinal);
        Owner o = new();
        Weights w = new(241);
        if (mouse)
        {
            w.Linear("m.mouse_mlp.0", stream, img + window * 2, true);
            w.Linear("m.mouse_mlp.2", stream, stream, true);
            w.AddNorm("m.mouse_mlp.3.weight", stream);
            w.AddBias("m.mouse_mlp.3.bias", stream);
            w.Linear("m.t_qkv", 3 * stream, stream, true);
            w.Linear("m.proj_mouse", img, stream, true);
            w.AddNorm("m.img_attn_q_norm.weight", headDim);
            w.AddNorm("m.img_attn_k_norm.weight", headDim);
        }
        else
        {
            w.Linear("m.keyboard_embed.0", hiddenSize, 6, true);
            w.Linear("m.keyboard_embed.2", hiddenSize, hiddenSize, true);
            w.Linear("m.keyboard_attn_kv", 2 * stream, hiddenSize * window, true);
            w.Linear("m.mouse_attn_q", stream, img, true);
            w.Linear("m.proj_keyboard", img, stream, true);
            w.AddNorm("m.key_attn_q_norm.weight", headDim);
            w.AddNorm("m.key_attn_k_norm.weight", headDim);
        }
        MatrixGame3ActionModule module = new(img, 2, 6, hiddenSize, stream, heads, 4, 3, 256, 4, 2, 2, enableMouse: mouse, enableKeyboard: !mouse);
        module.LoadWeights(w, "m");
        int memory = mouse ? 0 : 1;
        Tensor hidden = o.Own(Seeded(new TensorShape(frames * sp, img), 2401));
        Tensor mouseRaw = o.Own(Seeded(new TensorShape(8, 2), 2402));
        Tensor keyboardRaw = o.Own(Seeded(new TensorShape(8, 6), 2403));
        module.Forward(cpu, hidden, (frames, gh, gw), [0, 1, 2], memory, mouseRaw, keyboardRaw);
        return Finish(o, ("hidden", hidden));
    }

    // ── VAE attention ──────────────────────────────────────────────────────────────────────────────────────────────

    private static string VaeAttn(CpuBackend cpu, string name)
    {
        Owner o = new();
        Weights w = new(251);
        w.AddNorm("v.group_norm.weight", 32);
        w.AddBias("v.group_norm.bias", 32);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "to_out.0" }) w.Linear($"v.{n}", 32, 32, true);
        VaeAttention attn = new(32, 8);
        attn.LoadWeights(w, "v");
        Tensor outT = name switch
        {
            "vae_attn_2d" => attn.Forward(cpu, o.Own(Seeded(new TensorShape(2, 32, 4, 4), 2501))),
            "vae_attn_3d_causal" => attn.Forward3D(cpu, o.Own(Seeded(new TensorShape([1L, 32, 3, 2, 2]), 2502)), frameCausal: true),
            "vae_attn_3d_noncausal" => attn.Forward3D(cpu, o.Own(Seeded(new TensorShape([1L, 32, 3, 2, 2]), 2503)), frameCausal: false),
            _ => throw new ArgumentException($"Unknown VAE case '{name}'.", nameof(name)),
        };
        return Finish(o, ("out", o.Own(outT)));
    }

    private static string Wan22Attn(CpuBackend cpu, string name)
    {
        const int c = 16;
        Owner o = new();
        Weights w = new(261);
        w.AddNorm("a.norm.gamma", c);
        w.AddShape("a.to_qkv.weight", [3 * c, c, 1, 1], 0.3f);
        w.AddBias("a.to_qkv.bias", 3 * c);
        w.AddShape("a.proj.weight", [c, c, 1, 1], 0.3f);
        w.AddBias("a.proj.bias", c);
        Wan22AttentionBlock block = new(c);
        block.LoadWeights(w, "a");
        Tensor x = o.Own(Seeded(new TensorShape([1L, c, 2, 3, 3]), 2601));
        if (name.EndsWith("bf16", StringComparison.Ordinal)) x = o.Own(x.CastTo(DType.BF16));
        return Finish(o, ("out", o.Own(block.Forward(cpu, x))));
    }

    // ── Oasis ──────────────────────────────────────────────────────────────────────────────────────────────────────

    private static string OasisBlock(CpuBackend cpu)
    {
        const int dim = 32, heads = 4, headDim = dim / heads, frames = 3, sp = 4, rot = 4, mlp = 64;
        Owner o = new();
        Weights w = new(271);
        foreach (string h in new[] { "s", "t" })
        {
            w.Linear($"b.{h}_adaLN_modulation.1", 6 * dim, dim, true);
            w.Add($"b.{h}_attn.to_qkv.weight", 3 * dim, dim);
            w.Linear($"b.{h}_attn.to_out", dim, dim, true);
            w.Linear($"b.{h}_mlp.fc1", mlp, dim, true);
            w.Linear($"b.{h}_mlp.fc2", dim, mlp, true);
        }
        OasisSpatioTemporalBlock block = new(new OasisDitConfig { HiddenSize = dim, NumHeads = heads, MlpRatio = mlp / (float)dim });
        block.LoadWeights(w, "b");
        Tensor x = o.Own(Seeded(new TensorShape(frames * sp, dim), 2701));
        Tensor cond = o.Own(Seeded(new TensorShape(frames, dim), 2702, 0.5f));
        Tensor sCos = o.Own(RopeTable(sp, rot, true)), sSin = o.Own(RopeTable(sp, rot, false));
        Tensor tCos = o.Own(RopeTable(frames, headDim, true)), tSin = o.Own(RopeTable(frames, headDim, false));
        Tensor mask = o.Own(AdditiveMask(frames, frames, (i, j) => j <= i));
        Tensor outT = o.Own(block.Forward(cpu, x, cond, frames, sp, sCos, sSin, tCos, tSin, mask));
        return Finish(o, ("out", outT));
    }

    /// <summary>Interleaved-pair rotary table <c>[rows, width]</c>: each pair shares one angle.</summary>
    private static Tensor RopeTable(int rows, int width, bool cos)
    {
        Tensor t = new(new TensorShape(rows, width), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < width; i++)
            {
                double angle = (r + 1) * 0.37 / (1 + i / 2);
                p[r * width + i] = (float)(cos ? Math.Cos(angle) : Math.Sin(angle));
            }
        return t;
    }

    private static string OasisVit(CpuBackend cpu, string name)
    {
        const int latent = 4, patch = 2, dim = 32, heads = 4, mlp = 64;
        Owner o = new();
        Weights w = new(281);
        w.AddShape("patch_embed.proj.weight", [dim, 3, patch, patch], 0.3f);
        w.AddBias("patch_embed.proj.bias", dim);
        foreach (string p in new[] { "encoder.0", "decoder.0" })
        {
            w.AddNorm($"{p}.norm1.weight", dim); w.AddBias($"{p}.norm1.bias", dim);
            w.AddNorm($"{p}.norm2.weight", dim); w.AddBias($"{p}.norm2.bias", dim);
            w.Linear($"{p}.attn.qkv", 3 * dim, dim, true);
            w.Linear($"{p}.attn.proj", dim, dim, true);
            w.Linear($"{p}.mlp.fc1", mlp, dim, true);
            w.Linear($"{p}.mlp.fc2", dim, mlp, true);
        }
        w.AddNorm("enc_norm.weight", dim); w.AddBias("enc_norm.bias", dim);
        w.AddNorm("dec_norm.weight", dim); w.AddBias("dec_norm.bias", dim);
        w.Linear("quant_conv", 2 * latent, dim, true);
        w.Linear("post_quant_conv", dim, latent, true);
        w.Linear("predictor", 3 * patch * patch, dim, true);
        OasisVitVae vae = new(latent, patch, dim, heads, encDepth: 1, decDepth: 1, mlpRatio: mlp / (float)dim);
        vae.LoadWeights(w);
        Tensor outT = name.EndsWith("decode", StringComparison.Ordinal)
            ? vae.Decode(cpu, o.Own(Seeded(new TensorShape([1L, latent, 3, 4]), 2801)))
            : vae.Encode(cpu, o.Own(Seeded(new TensorShape([1L, 3, 6, 8]), 2802)));
        return Finish(o, ("out", o.Own(outT)));
    }

    // ── UNet cross-attention ───────────────────────────────────────────────────────────────────────────────────────

    private const int UChannels = 32, UCtxDim = 24, UCtxRows = 7, UBlocks = 2;

    private static CrossAttentionBlock UNetBlock(uint salt)
    {
        Weights w = new(salt);
        w.AddNorm("u.norm.weight", UChannels); w.AddBias("u.norm.bias", UChannels);
        w.Linear("u.proj_in", UChannels, UChannels, true);
        for (int i = 0; i < UBlocks; i++)
        {
            string p = $"u.transformer_blocks.{i}";
            foreach (string n in new[] { "norm1", "norm2", "norm3" }) { w.AddNorm($"{p}.{n}.weight", UChannels); w.AddBias($"{p}.{n}.bias", UChannels); }
            foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"{p}.attn1.{n}", UChannels, UChannels);   // self-attn: no qkv biases
            w.Linear($"{p}.attn1.to_out.0", UChannels, UChannels, true);
            w.Linear($"{p}.attn2.to_q", UChannels, UChannels, true);
            w.Linear($"{p}.attn2.to_k", UChannels, UCtxDim, true);
            w.Linear($"{p}.attn2.to_v", UChannels, UCtxDim, true);
            w.Linear($"{p}.attn2.to_out.0", UChannels, UChannels, true);
            w.Linear($"{p}.ff.net.0.proj", 8 * UChannels, UChannels, true);
            w.Linear($"{p}.ff.net.2", UChannels, 4 * UChannels, true);
        }
        w.Linear("u.proj_out", UChannels, UChannels, true);
        CrossAttentionBlock block = new(UChannels, 2, UCtxDim, UBlocks, normGroups: 8);
        block.LoadWeights(w, "u");
        return block;
    }

    private static string UNet(CpuBackend cpu, string name)
    {
        Owner o = new();
        CrossAttentionBlock block = UNetBlock(291);
        switch (name)
        {
            case "unet_self_cross_b1" or "unet_self_cross_b2":
            {
                int b = name.EndsWith("b2", StringComparison.Ordinal) ? 2 : 1;
                Tensor x = o.Own(Seeded(new TensorShape(b, UChannels, 4, 4), 2901));
                Tensor ctx = o.Own(Seeded(new TensorShape(b, UCtxRows, UCtxDim), 2902));
                return Finish(o, ("out", o.Own(block.Forward(cpu, x, ctx))));
            }
            case "unet_cross_cache":
            {
                using CpuBackend other = new();
                Tensor ctxA = o.Own(Seeded(new TensorShape(1, UCtxRows, UCtxDim), 2911));
                Tensor ctxB = o.Own(Seeded(new TensorShape(1, UCtxRows, UCtxDim), 2912));
                (IBackend Backend, Tensor Ctx)[] calls =
                [
                    (cpu, ctxA), (cpu, ctxA), (cpu, ctxB), (cpu, ctxA),   // miss, hit, replace, replace
                    (cpu, ctxA), (other, ctxB), (cpu, ctxA), (other, ctxB),   // hit, second slot, two hits
                ];
                List<(string, Tensor)> outs = [];
                for (int i = 0; i < calls.Length; i++)
                {
                    Tensor x = o.Own(Seeded(new TensorShape(1, UChannels, 4, 4), (uint)(2920 + i)));
                    Tensor r = o.Own(block.Forward(calls[i].Backend, x, calls[i].Ctx));
                    outs.Add(($"call{i}", r));
                }
                return Finish(o, [.. outs]);
            }
            case "unet_ipa_scale_0p6" or "unet_ipa_scale_0":
            {
                float scale = name.EndsWith("0p6", StringComparison.Ordinal) ? 0.6f : 0f;
                Tensor x = o.Own(Seeded(new TensorShape(1, UChannels, 4, 4), 2931));
                Tensor ctx = o.Own(Seeded(new TensorShape(1, UCtxRows, UCtxDim), 2932));
                Tensor tokens = o.Own(Seeded(new TensorShape(1, 5, UCtxDim), 2933));
                List<Tensor> toK = [], toV = [];
                for (int i = 0; i < UBlocks; i++)
                {
                    toK.Add(o.Own(Seeded(new TensorShape(UChannels, UCtxDim), (uint)(2940 + i), 0.3f)));
                    toV.Add(o.Own(Seeded(new TensorShape(UChannels, UCtxDim), (uint)(2950 + i), 0.3f)));
                }
                float[] scales = [scale, scale];
                return Finish(o, ("out", o.Own(block.Forward(cpu, x, ctx, tokens, toK, toV, 0, scales))));
            }
            default: throw new ArgumentException($"Unknown UNet case '{name}'.", nameof(name));
        }
    }

    // ── ControlNet-Union fusion ────────────────────────────────────────────────────────────────────────────────────

    private static string ControlNetUnion(CpuBackend cpu, string name)
    {
        const int types = 2, c = 32, timeDim = 64, addDim = 8, layers = 2, mlp = 4 * c;
        int batch = name.EndsWith("b2", StringComparison.Ordinal) ? 2 : 1;
        Owner o = new();
        Weights w = new(301);
        w.Add("task_embedding", types, c);
        for (int i = 0; i < layers; i++)
        {
            string p = $"transformer_layes.{i}";
            w.Add($"{p}.attn.in_proj_weight", 3 * c, c); w.AddBias($"{p}.attn.in_proj_bias", 3 * c);
            w.Linear($"{p}.attn.out_proj", c, c, true);
            foreach (string n in new[] { "ln_1", "ln_2" }) { w.AddNorm($"{p}.{n}.weight", c); w.AddBias($"{p}.{n}.bias", c); }
            w.Linear($"{p}.mlp.c_fc", mlp, c, true);
            w.Linear($"{p}.mlp.c_proj", c, mlp, true);
        }
        w.Linear("spatial_ch_projs", c, c, true);
        w.Linear("control_add_embedding.linear_1", timeDim, addDim * types, true);
        w.Linear("control_add_embedding.linear_2", timeDim, timeDim, true);
        using ControlNetUnionFusion fusion = new(types, c, timeDim, addDim, numHeads: 4);
        fusion.LoadWeights(w);
        Tensor convIn = o.Own(Seeded(new TensorShape(batch, c, 4, 4), 3001));
        Tensor hint = o.Own(Seeded(new TensorShape(batch, c, 4, 4), 3002));
        return Finish(o, ("out", o.Own(fusion.FuseConditionHint(cpu, convIn, hint, 1))));
    }

    // ── IP-Adapter resampler layer ─────────────────────────────────────────────────────────────────────────────────

    private static string IpaResamplerLayer(CpuBackend cpu, string name)
    {
        const int hidden = 32, heads = 4, headDim = 8, inner = heads * headDim, ffMult = 2, latents = 4, tokens = 6;
        int batch = name.EndsWith("b2", StringComparison.Ordinal) ? 2 : 1;
        Owner o = new();
        Weights w = new(311);
        foreach (string n in new[] { "norm1", "norm2" }) { w.AddNorm($"l.0.{n}.weight", hidden); w.AddBias($"l.0.{n}.bias", hidden); }
        w.Add("l.0.to_q.weight", inner, hidden);
        w.Add("l.0.to_kv.weight", 2 * inner, hidden);
        w.Add("l.0.to_out.weight", hidden, inner);
        w.AddNorm("l.1.0.weight", hidden); w.AddBias("l.1.0.bias", hidden);
        w.Add("l.1.1.weight", hidden * ffMult, hidden);
        w.Add("l.1.3.weight", hidden, hidden * ffMult);
        ResamplerLayer layer = new(hidden, heads, headDim, inner, ffMult);
        layer.LoadWeights(w, "l");
        Tensor x = o.Own(Seeded(new TensorShape(batch, tokens, hidden), 3101));
        Tensor lat = o.Own(Seeded(new TensorShape(batch, latents, hidden), 3102));
        return Finish(o, ("out", o.Own(layer.Forward(cpu, x, lat, batch, tokens))));
    }

    // ── SeedVR2 ────────────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>The tiny NaDiT: head dim 128 is fixed by the rope, so one head; a 5×2×2 patch grid with window counts (2,2,2) tiles
    /// time into windows of 3+2 (regular) and 1+3+1 (shifted), so several same-shape groups form in every layer.</summary>
    private static string SeedVr2(CpuBackend cpu, string name)
    {
        const int dim = 128, txtIn = 24, layers = 4, mmLayers = 2, inCh = 6, outCh = 4, mlp = 64, lt = 5, t = 5, h = 4, wd = 4;
        bool v1 = name == "seedvr2_v1_tiny";
        SeedVr2Config cfg = new()
        {
            VidDim = dim, TxtInDim = txtIn, EmbDim = 6 * dim, Heads = 1, HeadDim = 128, MlpDim = mlp, NumLayers = layers,
            MmLayers = v1 ? layers : mmLayers, InChannels = inCh, OutChannels = outCh, WindowCounts = (2, 2, 2),
            SwiGluMlp = !v1, HasTailNorm = !v1, PixelRope = v1, LastLayerVidOnly = !v1,
        };
        Owner o = new();
        Weights w = new(321);
        w.Linear("vid_in.proj", dim, inCh * 4, true);
        w.Linear("txt_in", dim, txtIn, true);
        w.Linear("emb_in.proj_in", dim, 256, true);
        w.Linear("emb_in.proj_hid", dim, dim, true);
        w.Linear("emb_in.proj_out", 6 * dim, dim, true);
        w.Linear("vid_out.proj", outCh * 4, dim, true);
        if (cfg.HasTailNorm)
        {
            w.AddNorm("vid_out_norm.weight", dim);
            w.AddBias("vid_out_ada.out_shift", dim);
            w.AddBias("vid_out_ada.out_scale", dim);
        }
        for (int i = 0; i < layers; i++)
        {
            string p = $"blocks.{i}";
            foreach (string b in i >= cfg.MmLayers ? new[] { "all" } : new[] { "vid", "txt" })
            {
                w.Add($"{p}.attn.proj_qkv.{b}.weight", 3 * dim, dim);
                w.Linear($"{p}.attn.proj_out.{b}", dim, dim, true);
                w.AddNorm($"{p}.attn.norm_q.{b}.weight", 128);
                w.AddNorm($"{p}.attn.norm_k.{b}.weight", 128);
                if (cfg.SwiGluMlp)
                {
                    w.Add($"{p}.mlp.{b}.proj_in_gate.weight", mlp, dim);
                    w.Add($"{p}.mlp.{b}.proj_in.weight", mlp, dim);
                    w.Add($"{p}.mlp.{b}.proj_out.weight", dim, mlp);
                }
                else
                {
                    w.Linear($"{p}.mlp.{b}.proj_in", mlp, dim, true);
                    w.Linear($"{p}.mlp.{b}.proj_out", dim, mlp, true);
                }
                foreach (string a in new[] { "attn_shift", "attn_scale", "attn_gate", "mlp_shift", "mlp_scale", "mlp_gate" })
                    w.AddBias($"{p}.ada.{b}.{a}", dim);
            }
        }
        using SeedVr2Dit dit = new(cfg);
        dit.LoadWeights(w);
        List<string> digests = [];
        dit.OnBlockOutput = (i, vid, txt) =>
        {
            if (i < 0) return;
            digests.Add(Digest($"b{i}_vid", vid));
            digests.Add(Digest($"b{i}_txt", txt));
        };
        Tensor latent = o.Own(Seeded(new TensorShape(t, h, wd, inCh), 3201));
        Tensor text = o.Own(Seeded(new TensorShape(lt, txtIn), 3202));
        Tensor outT = o.Own(dit.Forward(cpu, latent, text, 500f));
        digests.Add(Digest("out", outT));
        foreach (Tensor x in o) x.Dispose();
        return string.Join(' ', digests);
    }

    // ── MiniMax-H3 ─────────────────────────────────────────────────────────────────────────────────────────────────

    private static MiniMaxH3Config H3Config(int layers, int refinerLayers) => new()
    {
        HiddenSize = 32, NumLayers = layers, TokenRefinerNumLayers = refinerLayers, NumAttentionHeads = 2, AttentionHeadDim = 16,
        FfnHiddenSize = 48, LatentsDim = 24, AudioLatentsDim = 32, TextDim = 20, TimestepInputDim = 8, TimeEmbedHiddenSize = 16,
        TimeEmbedDim = 12, RopeInvFreqLen = 2, AdalnCurveGrid = null,
    };

    private static Dictionary<string, Tensor> H3Weights(MiniMaxH3Config c, uint salt)
    {
        int h = c.HiddenSize, inner = c.NumAttentionHeads * c.AttentionHeadDim;
        Weights w = new(salt);
        w.Linear("video_patch_proj", h, c.VideoPatchDim, true);
        w.Linear("audio_patch_proj", h, c.AudioLatentsDim, true);
        w.Linear("condition_proj", h, c.TextDim, true);
        w.AddBias("rope.inv_freq", c.RopeInvFreqLen);
        w.AddNorm("token_refiner.final_norm.weight", h);
        w.AddNorm("final_layer.norm.weight", h);
        w.Linear("final_layer.adaln_proj.linear", 2 * h, c.TimeEmbedDim, true);
        w.Linear("final_layer.video_out", c.VideoPatchDim, h, true);
        w.Linear("final_layer.audio_out", c.AudioLatentsDim, h, true);
        w.Linear("time_embedder.proj_in", c.TimeEmbedHiddenSize, c.TimestepInputDim, true);
        w.Linear("time_embedder.proj_out", c.TimeEmbedDim, c.TimeEmbedHiddenSize, true);
        for (int i = 0; i < c.TokenRefinerNumLayers + c.NumLayers; i++)
        {
            bool refiner = i < c.TokenRefinerNumLayers;
            string p = refiner ? $"token_refiner.blocks.{i}" : $"blocks.{i - c.TokenRefinerNumLayers}";
            w.AddNorm($"{p}.norm1.weight", h);
            w.AddNorm($"{p}.norm2.weight", h);
            w.Add($"{p}.attn.qkv_proj.weight", inner * 3, h);
            w.AddNorm($"{p}.attn.q_norm.weight", c.AttentionHeadDim);
            w.AddNorm($"{p}.attn.k_norm.weight", c.AttentionHeadDim);
            w.Add($"{p}.attn.out_proj.weight", h, inner);
            w.Add($"{p}.mlp.fc1.weight", c.FfnHiddenSize * 2, h);
            w.Add($"{p}.mlp.fc2.weight", h, c.FfnHiddenSize);
            if (!refiner) w.Linear($"{p}.adaln_proj.linear", 6 * h * 3, c.TimeEmbedDim, true);
        }
        return w;
    }

    /// <summary>Dense, chunked (<c>vram.h3ChunkRows</c> forced to 16: the CPU reports no VRAM so that knob is the only way to pick the path)
    /// and refiner-heavy forwards; each digests the video and audio heads.</summary>
    private static string H3(CpuBackend cpu, string name)
    {
        bool chunked = name == "h3_chunked", refiner = name == "h3_refiner";
        MiniMaxH3Config c = H3Config(layers: refiner ? 1 : 2, refinerLayers: refiner ? 2 : 1);
        int textLen = chunked ? 5 : refiner ? 9 : 3, latentT = chunked ? 6 : 2, latentH = chunked ? 8 : 4, latentW = chunked ? 12 : 6;
        int audioT = chunked ? 6 : 2;
        Owner o = new();
        MiniMaxH3PackedLayout layout = new(textLen, latentT, latentH, latentW, audioT);
        using MiniMaxH3Transformer dit = new(c);
        dit.LoadWeights(H3Weights(c, 331));
        Tensor videoRows = o.Own(Seeded(new TensorShape(latentT * (latentH / 2) * (latentW / 2), c.VideoPatchDim), 3301));
        Tensor audioRows = o.Own(Seeded(new TensorShape(audioT * 2, c.AudioLatentsDim), 3302));
        Tensor text = o.Own(Seeded(new TensorShape(textLen, c.TextDim), 3303));
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(layout.PositionIds, MiniMaxH3Rope.DefaultInvFreq(c.RopeInvFreqLen), c.AttentionHeadDim);
        o.Own(cos); o.Own(sin);
        Dictionary<MiniMaxH3SegmentKind, int> rowOf = new()
        {
            [MiniMaxH3SegmentKind.Text] = 0, [MiniMaxH3SegmentKind.Video] = 0, [MiniMaxH3SegmentKind.Cond] = 0,
            [MiniMaxH3SegmentKind.RefImage] = 0, [MiniMaxH3SegmentKind.Audio] = 1, [MiniMaxH3SegmentKind.RefAudio] = 1,
        };
        KnobStore.Clear(EngineKnobs.H3ChunkRows);
        if (chunked) KnobStore.Set(EngineKnobs.H3ChunkRows, 16);
        Tensor video, audio;
        try
        {
            (video, audio) = dit.Forward(cpu, layout, videoRows, audioRows, text, cos, sin, [0.3f, 0.55f], rowOf);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.H3ChunkRows);
        }
        o.Own(video); o.Own(audio);
        return Finish(o, ("video", video), ("audio", audio));
    }

    // ── Cosmos AR ──────────────────────────────────────────────────────────────────────────────────────────────────

    private static string CosmosAr(CpuBackend cpu, string name)
    {
        const int dim = 32, heads = 4, kvHeads = 2, headDim = 8, ffn = 48, ctxDim = 24, ctxLen = 6, prefill = 5, maxSeq = 16;
        bool cross = name == "cosmos_ar_cross", decode = name == "cosmos_ar_decode";
        CosmosArConfig cfg = new()
        {
            NumLayers = 1, Dim = dim, NumHeads = heads, NumKvHeads = kvHeads, HeadDim = headDim, FfnHiddenSize = ffn, ContextDim = ctxDim,
            QkNorm = true, InsertCrossAttn = cross,
        };
        Owner o = new();
        Weights w = new(341);
        const string p = "layers.0";
        w.AddNorm($"{p}.attention_norm.weight", dim);
        w.Add($"{p}.attention.wq.weight", heads * headDim, dim);
        w.Add($"{p}.attention.wk.weight", kvHeads * headDim, dim);
        w.Add($"{p}.attention.wv.weight", kvHeads * headDim, dim);
        w.Add($"{p}.attention.wo.weight", dim, heads * headDim);
        w.AddNorm($"{p}.attention.q_norm.weight", headDim);
        w.AddNorm($"{p}.attention.k_norm.weight", headDim);
        if (cross)
        {
            w.AddNorm($"{p}.cross_attention_norm.weight", dim);
            w.Add($"{p}.cross_attention.wq.weight", heads * headDim, dim);
            w.Add($"{p}.cross_attention.wk.weight", kvHeads * headDim, ctxDim);
            w.Add($"{p}.cross_attention.wv.weight", kvHeads * headDim, ctxDim);
            w.Add($"{p}.cross_attention.wo.weight", dim, heads * headDim);
        }
        w.AddNorm($"{p}.ffn_norm.weight", dim);
        w.Add($"{p}.feed_forward.w1.weight", ffn, dim);
        w.Add($"{p}.feed_forward.w2.weight", dim, ffn);
        w.Add($"{p}.feed_forward.w3.weight", ffn, dim);
        CosmosArBlock block = new(cfg, 0);
        block.LoadWeights(w, p);
        using FixedKvCache cache = new(1, 1, kvHeads, headDim, maxSeq);
        Tensor? context = cross ? o.Own(Seeded(new TensorShape(1, ctxLen, ctxDim), 3401)) : null;

        Tensor Step(int t, int posStart, uint seed)
        {
            Tensor hidden = o.Own(Seeded(new TensorShape(1, t, dim), seed));
            Tensor cos = o.Own(RopeTable3(t, headDim, posStart, true)), sin = o.Own(RopeTable3(t, headDim, posStart, false));
            Tensor r = o.Own(block.Forward(cpu, hidden, t, posStart, cache, 0, cos, sin, context, ctxLen));
            cache.AdvanceLength(t);
            return r;
        }

        Tensor first = Step(prefill, 0, 3402);
        if (!decode && !cross)
            return Finish(o, ("out", first), ("k", cache.KeyPrefix(0)), ("v", cache.ValuePrefix(0)));
        Tensor second = Step(1, prefill, 3403);
        return Finish(o, ("prefill", first), ("out", second), ("k", cache.KeyPrefix(0)), ("v", cache.ValuePrefix(0)));
    }

    /// <summary>Rotary table <c>[1, t, headDim]</c> for absolute positions <c>posStart..</c> (NEOX halves share an angle).</summary>
    private static Tensor RopeTable3(int t, int headDim, int posStart, bool cos)
    {
        Tensor r = new(new TensorShape(1, t, headDim), DType.F32);
        float* p = (float*)r.DataPointer;
        for (int s = 0; s < t; s++)
            for (int i = 0; i < headDim; i++)
            {
                double angle = (posStart + s) * 0.21 / (1 + i % (headDim / 2));
                p[s * headDim + i] = (float)(cos ? Math.Cos(angle) : Math.Sin(angle));
            }
        return r;
    }

    // ── Harness ────────────────────────────────────────────────────────────────────────────────────────────────────

    private sealed class Owner : List<Tensor>
    {
        public Tensor Own(Tensor t) { Add(t); return t; }
    }

    /// <summary>Digests the outputs, then releases every tensor the case allocated.</summary>
    private static string Finish(List<Tensor> owned, params (string Label, Tensor Tensor)[] outputs)
    {
        string digests = string.Join(' ', outputs.Select(entry => Digest(entry.Label, entry.Tensor)));
        foreach (Tensor t in owned) t.Dispose();
        return digests;
    }

    /// <summary>SHA-256 of the tensor's bits; rejects non-finite values.</summary>
    private static string Digest(string label, Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        for (long i = 0; i < f32.ElementCount; i++)
            if (!float.IsFinite(((float*)f32.DataPointer)[i]))
                throw new InvalidOperationException($"Non-finite value in '{label}' at {i}.");
        if (!ReferenceEquals(f32, t)) f32.Dispose();
        ReadOnlySpan<byte> bytes = new(t.DataPointer, checked((int)(t.ElementCount * t.DType.SizeInBytes)));
        return $"{label}={Convert.ToHexStringLower(SHA256.HashData(bytes))}";
    }

    /// <summary>Additive F32 mask <c>[1, 1, rows, cols]</c>: 0 where <paramref name="allowed"/>, a large negative elsewhere.</summary>
    private static Tensor AdditiveMask(int rows, int cols, Func<int, int, bool> allowed)
    {
        Tensor mask = new(new TensorShape(1, 1, rows, cols), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                p[i * cols + j] = allowed(i, j) ? 0f : NegInf;
        return mask;
    }

    private static Tensor Zeroed(TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        new Span<float>((float*)t.DataPointer, checked((int)t.ElementCount)).Clear();
        return t;
    }

    /// <summary>Rows <c>[start, start + count)</c> of a rank-2 F32 tensor as a fresh tensor.</summary>
    private static Tensor HostRows(Tensor src, int start, int count)
    {
        long width = src.Shape[1];
        Tensor t = new(new TensorShape(count, width), DType.F32);
        new ReadOnlySpan<float>((float*)src.DataPointer + start * width, checked((int)(count * width)))
            .CopyTo(new Span<float>((float*)t.DataPointer, checked((int)(count * width))));
        return t;
    }

    private static Tensor Seeded(TensorShape shape, uint seed, float scale = 1f)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        uint state = seed * 2654435761u + 0x9E3779B9u;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * scale;
        }
        return tensor;
    }

    /// <summary>A checkpoint stand-in: each tensor is seeded from its own name (FNV-1a) plus a per-case salt.</summary>
    private sealed class Weights(uint salt) : Dictionary<string, Tensor>
    {
        public void Add(string name, long rows, long cols) => base.Add(name, Seeded(new TensorShape(rows, cols), Seed(name), 0.3f));

        public void AddShape(string name, long[] shape, float scale) => base.Add(name, Seeded(new TensorShape(shape), Seed(name), scale));

        public void AddBias(string name, long n) => base.Add(name, Seeded(new TensorShape(n), Seed(name), 0.1f));

        /// <summary>A weight <c>[out, in]</c> under <c>prefix.weight</c>, plus <c>prefix.bias</c> when asked.</summary>
        public void Linear(string prefix, long outFeatures, long inFeatures, bool bias = false)
        {
            Add($"{prefix}.weight", outFeatures, inFeatures);
            if (bias) AddBias($"{prefix}.bias", outFeatures);
        }

        /// <summary>Norm scale near 1.</summary>
        public void AddNorm(string name, long n)
        {
            Tensor t = Seeded(new TensorShape(n), Seed(name), 0.25f);
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < n; i++) p[i] += 1f;
            base.Add(name, t);
        }

        private uint Seed(string name)
        {
            uint h = 2166136261u ^ salt;
            foreach (char c in name) h = (h ^ c) * 16777619u;
            return h;
        }
    }
}
