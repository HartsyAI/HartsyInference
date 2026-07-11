using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Self-parity for the CSM/HeartMuLa incremental decode fast path: the persistent-KV-cache
/// <see cref="CsmModel.StepFrame"/> loop (feed the prefix once, then one audio-embedding row per frame) must
/// produce <b>bit-identical</b> frames to the stateless full-context <see cref="CsmModel.GenerateFrame"/> loop
/// (which re-scans the whole growing context every frame). This is the correctness gate for the O(n²)→O(n)
/// rework — checkpoint-free (tiny random-weight dual-transformer), so it runs anywhere. Covers both the plain
/// (CSM) path and the CFG dual-cache path (HeartMuLa <c>cfg_scale</c>), including the frame-0 uncond zero row.</summary>
public sealed unsafe class CsmIncrementalDecodeTests
{
    private readonly ITestOutputHelper _out;
    public CsmIncrementalDecodeTests(ITestOutputHelper o) => _out = o;

    [Theory]
    [InlineData(1.0f)]   // no CFG (CSM path)
    [InlineData(1.5f)]   // CFG on (HeartMuLa path): dual persistent cache + standalone frame-0 uncond row
    public void StepFrame_MatchesGenerateFrame_FrameForFrame(float cfgScale)
    {
        CsmConfig cfg = TinyConfig();
        CsmModel model = new(cfg);
        model.LoadWeights(BuildRandomWeights(cfg, seed: 1234));
        using CpuBackend backend = new();

        int bh = cfg.Backbone.HiddenSize;
        int[] lyrics = { 1, 2, 3, 4, 5 };
        const int maxFrames = 8;
        const int seed = 777;
        bool useCfg = cfgScale != 1f;

        // ── Path A: stateless full-context recompute (the old hot path) ──
        uint rngA = DeterministicRng.Seed(seed);
        List<int[]> framesA = new();
        for (int f = 0; f < maxFrames; f++)
        {
            Tensor ctx = BuildContext(model, lyrics, framesA, bh, includePrefix: true);
            Tensor? uctx = useCfg ? BuildContext(model, lyrics, framesA, bh, includePrefix: false) : null;
            int[] codes = model.GenerateFrame(backend, ctx, ref rngA, null, null, null, useCfg ? cfgScale : 1f, uctx);
            ctx.Dispose();
            uctx?.Dispose();
            framesA.Add(codes);
        }

        // ── Path B: incremental persistent-cache decode (the new fast path) ──
        uint rngB = DeterministicRng.Seed(seed);
        List<int[]> framesB = new();
        List<int[]> noFrames = new(0);
        using CsmModel.DecodeSession session = model.CreateSession(lyrics.Length + maxFrames + 4, maxFrames + 4, useCfg);
        for (int f = 0; f < maxFrames; f++)
        {
            Tensor condNew;
            Tensor? uncondNew;
            bool standalone;
            if (f == 0)
            {
                condNew = BuildContext(model, lyrics, noFrames, bh, includePrefix: true);
                uncondNew = useCfg ? new Tensor(new TensorShape(1, 1, bh), DType.F32) : null;
                standalone = true;
            }
            else
            {
                condNew = model.EmbedAudioFrame(framesB[f - 1]);
                uncondNew = useCfg ? model.EmbedAudioFrame(framesB[f - 1]) : null;
                standalone = false;
            }
            int[] codes = model.StepFrame(backend, session, condNew, ref rngB, null, null, null, useCfg ? cfgScale : 1f, uncondNew, standalone);
            condNew.Dispose();
            uncondNew?.Dispose();
            framesB.Add(codes);
        }

        model.Dispose();

        // Frames must be identical token-for-token.
        int mismatches = 0;
        for (int f = 0; f < maxFrames; f++)
            for (int c = 0; c < cfg.NumCodebooks; c++)
                if (framesA[f][c] != framesB[f][c]) mismatches++;
        _out.WriteLine($"cfg={cfgScale}: {maxFrames} frames × {cfg.NumCodebooks} codebooks, mismatches={mismatches}");
        _out.WriteLine($"  A[0]=[{string.Join(",", framesA[0])}]  A[last]=[{string.Join(",", framesA[maxFrames - 1])}]");
        Assert.Equal(0, mismatches);
    }

    /// <summary>CUDA graph-decode parity: the backbone CUDA-graph decode path (HARTSY_CSM_GRAPH=1 — one captured
    /// single-frame step replayed per frame, cond + uncond as separate graphs) must produce <b>bit-identical</b>
    /// frames to the eager path (HARTSY_CSM_GRAPH=0) on the same weights/seed. The device-position graph kernels are
    /// numerically identical to eager decode (proven at &lt;1e-6 in GraphDecodeEmbedsTests), so identical logits →
    /// identical nucleus samples. Runs enough frames to pass warmup and exercise capture + replay. CFG on and off.</summary>
    [Theory]
    [InlineData(1.0f)]   // no CFG: single backbone graph
    [InlineData(1.5f)]   // CFG: cond + uncond backbone graphs (two concurrent captured graphs)
    public void StepFrame_GraphDecode_MatchesEager(float cfgScale)
    {
        if (!CudaContext.IsAvailable()) { _out.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string? ptxDir = ResolvePtxDir();
        if (ptxDir is null) { _out.WriteLine("SKIPPED: PTX dir not found"); return; }

        CsmConfig cfg = TinyConfig();
        int[] eagerFrames = RunGraphDecode(cfg, ptxDir, cfgScale, graph: false);
        int[] graphFrames = RunGraphDecode(cfg, ptxDir, cfgScale, graph: true);

        int mismatches = 0;
        for (int i = 0; i < eagerFrames.Length; i++) if (eagerFrames[i] != graphFrames[i]) mismatches++;
        _out.WriteLine($"cfg={cfgScale}: {eagerFrames.Length} codes, graph-vs-eager mismatches={mismatches}");
        Assert.Equal(0, mismatches);
    }

    // Finds the compiled PTX kernel dir: the test output's Ptx (copied from the Cuda project), else walk up to source.
    private static string? ResolvePtxDir()
    {
        string local = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(local)) return local;
        for (DirectoryInfo? d = new(AppContext.BaseDirectory); d is not null; d = d.Parent)
        {
            string cand = Path.Combine(d.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(cand)) return cand;
        }
        return null;
    }

    // One StepFrame loop on CUDA with the graph flag forced on/off; returns the flattened frame codes.
    private static int[] RunGraphDecode(CsmConfig cfg, string ptxDir, float cfgScale, bool graph)
    {
        string? prev = Environment.GetEnvironmentVariable("HARTSY_CSM_GRAPH");
        Environment.SetEnvironmentVariable("HARTSY_CSM_GRAPH", graph ? "1" : "0");
        try
        {
            using CudaBackend backend = new(0, ptxDir);
            CsmModel model = new(cfg);
            model.LoadWeights(BuildRandomWeights(cfg, seed: 1234));
            int bh = cfg.Backbone.HiddenSize;
            int[] lyrics = { 1, 2, 3, 4, 5 };
            const int maxFrames = 8;
            bool useCfg = cfgScale != 1f;

            uint rng = DeterministicRng.Seed(777);
            List<int[]> frames = new();
            List<int[]> noFrames = new(0);
            using CsmModel.DecodeSession session = model.CreateSession(lyrics.Length + maxFrames + 4, maxFrames + 4, useCfg);
            for (int f = 0; f < maxFrames; f++)
            {
                Tensor condNew;
                Tensor? uncondNew;
                bool standalone;
                if (f == 0)
                {
                    condNew = BuildContext(model, lyrics, noFrames, bh, includePrefix: true);
                    uncondNew = useCfg ? new Tensor(new TensorShape(1, 1, bh), DType.F32) : null;
                    standalone = true;
                }
                else
                {
                    condNew = model.EmbedAudioFrame(frames[f - 1]);
                    uncondNew = useCfg ? model.EmbedAudioFrame(frames[f - 1]) : null;
                    standalone = false;
                }
                int[] codes = model.StepFrame(backend, session, condNew, ref rng, null, null, null, useCfg ? cfgScale : 1f, uncondNew, standalone);
                condNew.Dispose();
                uncondNew?.Dispose();
                frames.Add(codes);
            }
            model.Dispose();

            int[] flat = new int[maxFrames * cfg.NumCodebooks];
            for (int f = 0; f < maxFrames; f++)
                for (int c = 0; c < cfg.NumCodebooks; c++) flat[f * cfg.NumCodebooks + c] = frames[f][c];
            return flat;
        }
        finally { Environment.SetEnvironmentVariable("HARTSY_CSM_GRAPH", prev); }
    }

    /// <summary>Builds the running context <c>[1, T, bh]</c> exactly like the pipelines: optional prefix (lyric text
    /// embeddings) followed by prior audio frames. With no prefix and no frames it is a single zeroed row.</summary>
    private static Tensor BuildContext(CsmModel model, int[] lyrics, List<int[]> frames, int bh, bool includePrefix)
    {
        int rows = (includePrefix ? lyrics.Length : 0) + frames.Count;
        Tensor ctx = new(new TensorShape(1, Math.Max(1, rows), bh), DType.F32);
        float* cp = (float*)ctx.DataPointer;
        int row = 0;
        if (includePrefix)
            foreach (int tok in lyrics)
            {
                using Tensor e = model.EmbedText(tok);
                Buffer.MemoryCopy((void*)e.DataPointer, cp + (long)row++ * bh, bh * 4, bh * 4);
            }
        foreach (int[] fr in frames)
        {
            using Tensor e = model.EmbedAudioFrame(fr);
            Buffer.MemoryCopy((void*)e.DataPointer, cp + (long)row++ * bh, bh * 4, bh * 4);
        }
        return ctx;
    }

    private static CsmConfig TinyConfig() => new()
    {
        Backbone = new Qwen2Config
        {
            HiddenSize = 64, NumHiddenLayers = 2, NumAttentionHeads = 4, NumKeyValueHeads = 2,
            IntermediateSize = 128, VocabSize = 40, MaxPositionEmbeddings = 512,
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, AttentionBias = false, TieWordEmbeddings = false,
        },
        Decoder = new Qwen2Config
        {
            HiddenSize = 32, NumHiddenLayers = 2, NumAttentionHeads = 2, NumKeyValueHeads = 1,
            IntermediateSize = 64, VocabSize = 40, MaxPositionEmbeddings = 512,
            RopeTheta = 500_000f, RmsNormEps = 1e-5f, AttentionBias = false, TieWordEmbeddings = false,
        },
        NumCodebooks = 4,
        AudioVocab = 32,
        TextVocab = 40,
        AudioEosToken = 1_000,   // out of the tiny vocab range → never triggers, so all frames generate.
    };

    private static Dictionary<string, Tensor> BuildRandomWeights(CsmConfig cfg, int seed)
    {
        Random r = new(seed);
        Dictionary<string, Tensor> w = new();
        AddBody(w, r, "backbone", cfg.Backbone);
        AddBody(w, r, "decoder", cfg.Decoder);

        int bh = cfg.Backbone.HiddenSize, dh = cfg.Decoder.HiddenSize;
        w["text_embeddings.weight"] = Rand(r, cfg.TextVocab, bh);
        for (int i = 0; i < cfg.NumCodebooks; i++) w[$"audio_embeddings.{i}.weight"] = Rand(r, cfg.AudioVocab, bh);
        w["codebook0_head.weight"] = Rand(r, cfg.AudioVocab, bh);
        w["projection.weight"] = Rand(r, dh, bh);
        for (int i = 0; i < cfg.NumCodebooks - 1; i++) w[$"audio_head.{i}.weight"] = Rand(r, cfg.AudioVocab, dh);
        return w;
    }

    private static void AddBody(Dictionary<string, Tensor> w, Random r, string p, Qwen2Config c)
    {
        int h = c.HiddenSize, qDim = c.NumAttentionHeads * c.HeadDim, kvDim = c.NumKeyValueHeads * c.HeadDim, inter = c.IntermediateSize;
        for (int i = 0; i < c.NumHiddenLayers; i++)
        {
            string lp = $"{p}.layers.{i}";
            w[$"{lp}.self_attn.q_proj.weight"] = Rand(r, qDim, h);
            w[$"{lp}.self_attn.k_proj.weight"] = Rand(r, kvDim, h);
            w[$"{lp}.self_attn.v_proj.weight"] = Rand(r, kvDim, h);
            w[$"{lp}.self_attn.o_proj.weight"] = Rand(r, h, qDim);
            w[$"{lp}.mlp.gate_proj.weight"] = Rand(r, inter, h);
            w[$"{lp}.mlp.up_proj.weight"] = Rand(r, inter, h);
            w[$"{lp}.mlp.down_proj.weight"] = Rand(r, h, inter);
            w[$"{lp}.input_layernorm.weight"] = Ones(r, h);
            w[$"{lp}.post_attention_layernorm.weight"] = Ones(r, h);
        }
        w[$"{p}.norm.weight"] = Ones(r, h);
    }

    private static Tensor Rand(Random r, int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        long n = (long)rows * cols;
        for (long i = 0; i < n; i++) p[i] = (float)((r.NextDouble() - 0.5) * 0.08);
        return t;
    }

    private static Tensor Ones(Random r, int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f + (float)((r.NextDouble() - 0.5) * 0.02);
        return t;
    }
}
