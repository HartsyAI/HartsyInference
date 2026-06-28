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

    [Fact]
    public void FireflyDecoder_SyntheticForward_CodesToFiniteAudio()
    {
        // Tiny firefly: FSQ levels [2,2] (vocab 4), 3 residual quantizers, quantizer upsample [2], SiLU gen
        // upsample [2,2]. Channel bookkeeping is checkpoint-driven, so the weights below define the shapes.
        int nq = 3;
        int[] levels = [2, 2];
        using CpuBackend backend = new();
        FireflyConfig cfg = new()
        {
            FsqLevels = levels, QuantizerUpsampleFactors = [2],
            UpsampleInitialChannel = 16, UpsampleRates = [2, 2], UpsampleKernelSizes = [4, 4],
            ResBlockKernelSizes = [3], ResBlockDilations = [[1]], SampleRate = 44_100,
        };
        FireflyDecoder dec = new(nq, FishSpeechCodecTests.FsqVocab(levels), config: cfg);
        dec.LoadWeights(FishSpeechCodecTests.FireflyWeights(levels, qUp: [2], genInit: 16, genRates: [2, 2], genKernels: [4, 4]));
        int t = 3;
        int cbVocab = FishSpeechCodecTests.FsqVocab(levels);
        int[,] codes = new int[nq, t];
        for (int i = 0; i < nq; i++) for (int j = 0; j < t; j++) codes[i, j] = (i + j) % cbVocab;
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

    private static Dictionary<string, Tensor> Weights(FishSpeechConfig c)
    {
        int h = c.Backbone.HiddenSize;
        Dictionary<string, Tensor> w = new()
        {
            ["embeddings.weight"] = F2(c.TextVocab, h),
            ["codebook_embeddings.weight"] = F2(c.NumCodebooks * c.CodebookSize, h),
            ["output.weight"] = F2(c.TextVocab, h),
            ["fast_embeddings.weight"] = F2(c.CodebookSize, c.Fast.HiddenSize),
            ["fast_output.weight"] = F2(c.CodebookSize, c.Fast.HiddenSize),
            ["fast_norm.weight"] = Ones(c.Fast.HiddenSize),
        };
        AddQwen(w, "model", c.Backbone);
        AddQwen(w, "fast_model", c.Fast);
        return w;
    }

    private static void AddQwen(Dictionary<string, Tensor> w, string prefix, Qwen2Config c)
    {
        int h = c.HiddenSize, kv = c.NumKeyValueHeads * c.HeadDim;
        w[$"{prefix}.norm.weight"] = Ones(h);
        for (int i = 0; i < c.NumHiddenLayers; i++)
        {
            string p = $"{prefix}.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(h, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kv, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kv, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, h);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
    }

}
