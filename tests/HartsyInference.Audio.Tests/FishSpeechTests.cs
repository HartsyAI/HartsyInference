using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Streaming;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for Fish-Speech 1.5: config, a DualAR synthetic-weights frame generation (slow semantic +
/// fast 8-codebook depth AR → valid tokens), and a firefly decoder forward (codes → finite audio).</summary>
public sealed unsafe class FishSpeechTests
{
    private static uint _rng = 0xF15Au;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));

    /// <summary>Real-weight Fish-Speech 1.5 DualAR LM: loads <c>model.pth</c> (fused-key Llama, adapted to the
    /// Qwen2 split layout) and runs the slow + fast forwards, asserting finite logits of the right shape.
    /// Gated on <c>FISH_CKPT</c>. The slow LM is bit-exact (corr 1.0) and the fast LM corr 0.9999 vs the
    /// upstream `DualARTransformer` — see PARITY_VERIFICATION; this guards the key adapter + interleaved RoPE.</summary>
    [Fact]
    public void RealWeights_DualAr_SlowAndFast_AreFinite()
    {
        string? path = Environment.GetEnvironmentVariable("FISH_CKPT");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

        FishSpeechConfig cfg = FishSpeechConfig.V1_5;
        using PytorchPickleLoader ld = new(); ld.Load(path);
        using FishSpeechDualAr dual = new(cfg); dual.LoadWeights(ld.GetAllTensors());
        using CpuBackend backend = new();

        int t = 6;
        int[] sems = new int[t];
        int[][] codes = new int[t][];
        for (int i = 0; i < t; i++) { codes[i] = new int[cfg.NumCodebooks]; for (int c = 0; c < cfg.NumCodebooks; c++) codes[i][c] = (i * 13 + c) % cfg.CodebookSize; }
        using Tensor slow = dual.DebugSlowLogits(backend, sems, codes);
        Assert.Equal(new TensorShape(1, t, cfg.TextVocab), slow.Shape);

        float[] hid = new float[cfg.Backbone.HiddenSize];
        for (int i = 0; i < hid.Length; i++) hid[i] = MathF.Sin(i * 0.01f);
        int[] prev = new int[cfg.NumCodebooks - 1];
        for (int i = 0; i < prev.Length; i++) prev[i] = (i * 37) % cfg.CodebookSize;
        using Tensor fast = dual.DebugFastLogits(backend, hid, prev);
        Assert.Equal(new TensorShape(1, cfg.NumCodebooks, cfg.CodebookSize), fast.Shape);

        float* sp = (float*)slow.DataPointer;
        for (long i = 0; i < slow.ElementCount; i++) Assert.True(float.IsFinite(sp[i]));
        float* fp = (float*)fast.DataPointer;
        for (long i = 0; i < fast.ElementCount; i++) Assert.True(float.IsFinite(fp[i]));
    }
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void V1_5_Config()
    {
        FishSpeechConfig c = FishSpeechConfig.V1_5;
        Assert.Equal(1_024, c.Backbone.HiddenSize);
        Assert.Equal(24, c.Backbone.NumHiddenLayers);
        Assert.Equal(2, c.Backbone.NumKeyValueHeads);
        Assert.Equal(8, c.NumCodebooks);
        Assert.Equal(1_024, c.CodebookSize);
        Assert.Equal(4, c.Fast.NumHiddenLayers);
    }

    [Fact]
    public void DualAr_SyntheticForward_GeneratesSemanticAndCodebooks()
    {
        FishSpeechConfig c = new()
        {
            Backbone = TinyQwen(h: 16, layers: 2, vocab: 12),
            Fast = TinyQwen(h: 16, layers: 2, vocab: 6),
            NumCodebooks = 4, CodebookSize = 6, TextVocab = 12,
        };
        using CpuBackend backend = new();
        using FishSpeechDualAr m = new(c);
        m.LoadWeights(Weights(c));
        using StreamingKvCache slow = new(c.Backbone.NumHiddenLayers, 1, c.Backbone.NumKeyValueHeads, 8, c.Backbone.HeadDim);
        uint rng = 7;
        Span<int> zero = stackalloc int[c.NumCodebooks];
        using Tensor e = m.EmbedFrame(3, zero);
        (int sem, int[] codes) = m.GenerateFrame(backend, e, 0, slow, ref rng);
        Assert.InRange(sem, 0, c.TextVocab - 1);
        Assert.Equal(c.NumCodebooks, codes.Length);
        foreach (int code in codes) Assert.InRange(code, 0, c.CodebookSize - 1);
    }

    /// <summary>Upstream <c>BaseTransformer.embed</c> zeroes the codebook sum for frames whose row-0 is NOT a
    /// <c>&lt;|semantic:i|&gt;</c> token: text positions must embed identically regardless of codebook values,
    /// while semantic positions must include the codebook contribution.</summary>
    [Fact]
    public void EmbedFrame_ZeroesCodebooksForNonSemanticRow0()
    {
        FishSpeechConfig c = new()
        {
            Backbone = TinyQwen(h: 16, layers: 2, vocab: 12),
            Fast = TinyQwen(h: 16, layers: 2, vocab: 6),
            NumCodebooks = 4, CodebookSize = 6, TextVocab = 12, SemanticBeginId = 4,   // semantic range [4, 10)
        };
        using FishSpeechDualAr m = new(c);
        m.LoadWeights(Weights(c));
        int[] codesA = [1, 2, 3, 4], codesB = [5, 0, 2, 1];

        using Tensor textA = m.EmbedFrame(2, codesA);   // 2 < 4 → text token
        using Tensor textB = m.EmbedFrame(2, codesB);
        float* ta = (float*)textA.DataPointer, tb = (float*)textB.DataPointer;
        for (int i = 0; i < 16; i++) Assert.Equal(ta[i], tb[i]);

        using Tensor semA = m.EmbedFrame(5, codesA);    // 5 ∈ [4, 10) → semantic token
        using Tensor semB = m.EmbedFrame(5, codesB);
        float* sa = (float*)semA.DataPointer, sb = (float*)semB.DataPointer;
        bool differs = false;
        for (int i = 0; i < 16; i++) differs |= sa[i] != sb[i];
        Assert.True(differs);
    }

    /// <summary>Upstream <c>decode_one_token_ar</c> derives codebook 0 from the slow semantic token
    /// (<c>a = semantic − semantic_begin_id</c>, clamped) instead of sampling it from the fast model.</summary>
    [Fact]
    public void SampleFrame_DerivesCodebook0FromSemanticToken()
    {
        FishSpeechConfig c = new()
        {
            Backbone = TinyQwen(h: 16, layers: 2, vocab: 12),
            Fast = TinyQwen(h: 16, layers: 2, vocab: 6),
            NumCodebooks = 4, CodebookSize = 6, TextVocab = 12, SemanticBeginId = 4,
        };
        using CpuBackend backend = new();
        using FishSpeechDualAr m = new(c);
        m.LoadWeights(Weights(c));
        using StreamingKvCache slow = new(c.Backbone.NumHiddenLayers, 1, c.Backbone.NumKeyValueHeads, 8, c.Backbone.HeadDim);
        uint rng = 21;
        Span<int> zero = stackalloc int[c.NumCodebooks];
        using Tensor e = m.EmbedFrame(3, zero);
        (int sem, int[] codes) = m.GenerateFrame(backend, e, 0, slow, ref rng);
        Assert.InRange(sem, 0, c.TextVocab - 1);
        Assert.Equal(Math.Clamp(sem - c.SemanticBeginId, 0, c.CodebookSize - 1), codes[0]);
        for (int i = 1; i < codes.Length; i++) Assert.InRange(codes[i], 0, c.CodebookSize - 1);
    }

    /// <summary>Batched prompt prefill (one <c>ForwardHidden</c> over <c>[1,T,h]</c>) must match the per-token
    /// sequential path: same last-position hidden, so the same sampled first frame for a fixed RNG.</summary>
    [Fact]
    public void PrefillBatched_MatchesSequential()
    {
        FishSpeechConfig c = new()
        {
            Backbone = TinyQwen(h: 16, layers: 2, vocab: 12),
            Fast = TinyQwen(h: 16, layers: 2, vocab: 6),
            NumCodebooks = 4, CodebookSize = 6, TextVocab = 12, SemanticBeginId = 4,
        };
        using CpuBackend backend = new();
        using FishSpeechDualAr m = new(c);
        m.LoadWeights(Weights(c));
        int[] prompt = [1, 3, 0, 2, 11];

        using StreamingKvCache cacheA = new(c.Backbone.NumHiddenLayers, 1, c.Backbone.NumKeyValueHeads, 16, c.Backbone.HeadDim);
        using Tensor batched = m.EmbedPrompt(prompt);
        using Tensor hidA = m.ForwardHidden(backend, batched, prompt.Length, 0, cacheA);

        using StreamingKvCache cacheB = new(c.Backbone.NumHiddenLayers, 1, c.Backbone.NumKeyValueHeads, 16, c.Backbone.HeadDim);
        Span<int> zero = stackalloc int[c.NumCodebooks];
        Tensor hidB = null!;
        for (int i = 0; i < prompt.Length; i++)
        {
            using Tensor e = m.EmbedFrame(prompt[i], zero);
            Tensor h = m.ForwardHidden(backend, e, 1, i, cacheB);
            if (i == prompt.Length - 1) hidB = h; else h.Dispose();
        }

        float* pa = (float*)hidA.DataPointer, pb = (float*)hidB.DataPointer;
        for (int i = 0; i < 16; i++) Assert.Equal(pa[i], pb[i], 4);
        hidB.Dispose();
    }

    [Fact]
    public void FireflyDecoder_SyntheticForward_CodesToFiniteAudio()
    {
        // Tiny firefly: grouped-residual FSQ (levels [2,2] → vocab 4) with 2 groups over an 8-dim latent (groupDim 4),
        // quantizer upsample [2], then the SiLU HiFi-GAN generator. The quantizer output channels (QuantizerInputDim)
        // must equal the generator input channels (NumMels); nGroups (= numCodebooks) must divide QuantizerInputDim.
        int[] levels = [2, 2];
        int nGroups = 2;   // FireflyDecoder passes numCodebooks as the quantizer's nGroups.
        using CpuBackend backend = new();
        FireflyConfig cfg = new()
        {
            FsqLevels = levels, QuantizerUpsampleFactors = [2], QuantizerInputDim = 8, NumMels = 8,
            UpsampleInitialChannel = 16, UpsampleRates = [2, 2], UpsampleKernelSizes = [4, 4],
            ResBlockKernelSizes = [3], ResBlockDilations = [[1]],
            PreConvKernelSize = 3, PostConvKernelSize = 3, SampleRate = 44_100,
        };
        FireflyDecoder dec = new(nGroups, FishSpeechCodecTests.FsqVocab(levels), config: cfg);
        dec.LoadWeights(FishSpeechCodecTests.FireflyWeights(cfg, nGroups));
        int t = 3;
        int cbVocab = FishSpeechCodecTests.FsqVocab(levels);
        int[,] codes = new int[nGroups, t];
        for (int i = 0; i < nGroups; i++) for (int j = 0; j < t; j++) codes[i, j] = (i + j) % cbVocab;
        float[] audio = dec.Decode(backend, codes, t);
        Assert.True(audio.Length > 0);
        foreach (float s in audio) Assert.True(float.IsFinite(s));
    }

    private static Qwen2Config TinyQwen(int h, int layers, int vocab) => new()
    {
        HiddenSize = h, NumHiddenLayers = layers, NumAttentionHeads = 2, NumKeyValueHeads = 1,
        IntermediateSize = 32, VocabSize = vocab, MaxPositionEmbeddings = 64,
        AttentionBias = false, TieWordEmbeddings = false,
    };

    // Synthetic weights in the REAL fish-speech checkpoint layout that FishSpeechDualAr.LoadWeights consumes: fused
    // `layers.{i}.attention.wqkv` (q|k|v) + `.wo`, `feed_forward.w1/w3/w2` (gate/up/down), `attention_norm`/`ffn_norm`,
    // a bare `norm.weight` (slow final) and `fast_norm.weight` (fast final). LoadWeights remaps these to the split
    // Qwen2 layout internally.
    private static Dictionary<string, Tensor> Weights(FishSpeechConfig c)
    {
        int h = c.Backbone.HiddenSize, fh = c.Fast.HiddenSize;
        Dictionary<string, Tensor> w = new()
        {
            ["embeddings.weight"] = F2(c.TextVocab, h),
            ["codebook_embeddings.weight"] = F2(c.NumCodebooks * c.CodebookSize, h),
            ["output.weight"] = F2(c.TextVocab, h),
            ["fast_embeddings.weight"] = F2(c.CodebookSize, fh),
            ["fast_output.weight"] = F2(c.CodebookSize, fh),
            ["fast_norm.weight"] = Ones(fh),   // fast final RMSNorm
            ["norm.weight"] = Ones(h),         // slow final RMSNorm
        };
        AddFishLlama(w, "layers", c.Backbone);
        AddFishLlama(w, "fast_layers", c.Fast);
        return w;
    }

    private static void AddFishLlama(Dictionary<string, Tensor> w, string prefix, Qwen2Config c)
    {
        int h = c.HiddenSize, q = c.NumAttentionHeads * c.HeadDim, kv = c.NumKeyValueHeads * c.HeadDim;
        for (int i = 0; i < c.NumHiddenLayers; i++)
        {
            string p = $"{prefix}.{i}";
            w[$"{p}.attention.wqkv.weight"] = F2(q + 2 * kv, h);   // fused q|k|v
            w[$"{p}.attention.wo.weight"] = F2(h, q);
            w[$"{p}.feed_forward.w1.weight"] = F2(c.IntermediateSize, h);   // gate
            w[$"{p}.feed_forward.w3.weight"] = F2(c.IntermediateSize, h);   // up
            w[$"{p}.feed_forward.w2.weight"] = F2(h, c.IntermediateSize);   // down
            w[$"{p}.attention_norm.weight"] = Ones(h);
            w[$"{p}.ffn_norm.weight"] = Ones(h);
        }
    }

}
