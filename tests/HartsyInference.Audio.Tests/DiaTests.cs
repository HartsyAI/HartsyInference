using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for Dia: the encoder-decoder presets, the BOS/PAD delay staging (the index
/// bookkeeping that must be exact), clean construction, and a tiny synthetic-weight forward that exercises
/// the net-new non-causal encoder + GQA self-attn + cross-attn + multi-channel head end-to-end.</summary>
public sealed unsafe class DiaTests
{
    private static uint _rng = 0x9e3779b1u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    /// <summary>Real-weight Dia-1.6B transformer: byte text → encoder → decoder step (cross-attn + 9-channel
    /// fused head). Gated on <c>DIA_CKPT</c> (the nari-native <c>model.safetensors</c>). The full encoder +
    /// decoder are bit-exact (corr 1.0) vs the upstream <c>DiaModel</c> — see PARITY_VERIFICATION; this guards
    /// the <see cref="DiaWeights"/> DenseGeneral adapter + KV-cache wiring against regressions.</summary>
    [Fact]
    public void RealWeights_EncoderDecoderForward_IsFinite()
    {
        string? path = Environment.GetEnvironmentVariable("DIA_CKPT");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

        DiaConfig cfg = DiaConfig.Dia1_6B;
        using SafeTensorsLoader ld = new(); ld.Load(path);
        Dictionary<string, Tensor> w = DiaWeights.Adapt(ld.GetAllTensors());
        using CpuBackend backend = new();

        using DiaEncoder enc = new(cfg); enc.LoadWeights(w, "encoder");
        int[] text = [1, 72, 101, 108, 108, 111, 2];
        using Tensor encOut = enc.Forward(backend, text);
        Assert.Equal(new TensorShape(1, text.Length, cfg.EncoderDim), encOut.Shape);

        using DiaDecoder dec = new(cfg); dec.LoadWeights(w, "decoder", "decoder.logits_dense.weight");
        dec.PrecomputeCrossKv(backend, encOut, text.Length);
        using StreamingKvCache cache = new(cfg.DecoderLayers, 1, cfg.DecoderSelfKvHeads, 8, cfg.HeadDim);
        int[] bos = new int[cfg.Channels];
        for (int c = 0; c < cfg.Channels; c++) bos[c] = 1026;   // BOS in every channel
        for (int step = 0; step < 3; step++)
        {
            using Tensor logits = dec.StepLogits(backend, bos, step, cache);
            Assert.Equal(new TensorShape(1, 1, cfg.Channels * cfg.AudioVocab), logits.Shape);
            float* lp = (float*)logits.DataPointer;
            for (long i = 0; i < logits.ElementCount; i++) Assert.True(float.IsFinite(lp[i]));
        }
    }

    [Fact]
    public void Dia1_6B_EncoderDecoderPresets()
    {
        DiaConfig c = DiaConfig.Dia1_6B;
        Assert.Equal(12, c.EncoderLayers);
        Assert.Equal(1_024, c.EncoderDim);
        Assert.Equal(16, c.EncoderKvHeads);             // encoder is full MHA
        Assert.Equal(18, c.DecoderLayers);
        Assert.Equal(2_048, c.DecoderDim);
        Assert.Equal(16, c.DecoderSelfQHeads);
        Assert.Equal(4, c.DecoderSelfKvHeads);          // decoder self-attn is GQA
        Assert.Equal(16, c.DecoderCrossKvHeads);        // cross-attn is MHA
        Assert.Equal(128, c.HeadDim);
        Assert.Equal(9, c.Channels);
        Assert.Equal(1_028, c.AudioVocab);              // 1024 + EOS/PAD/BOS
        Assert.Equal(1_024, c.AudioEos);
        Assert.Equal(1_026, c.AudioBos);
    }

    [Fact]
    public void DelayPattern_IsZeroThenEightToFifteen()
    {
        DiaConfig c = DiaConfig.Dia1_6B;
        Assert.Equal(new[] { 0, 8, 9, 10, 11, 12, 13, 14, 15 }, c.DelayPattern.ToArray());
        Assert.Equal(15, c.MaxDelay);
        Assert.Equal(9, c.DelayPattern.Count);          // one per codebook
    }

    [Fact]
    public void Delay_BosPadOverload_FillsLeadInWithBos_TailWithPad()
    {
        int[] delay = [0, 2, 3];                        // 3 channels
        int bos = 1026, pad = 1025;
        int t = 4, k = 3;
        int[,] real = new int[t, k];
        for (int s = 0; s < t; s++)
            for (int c = 0; c < k; c++) real[s, c] = s * 10 + c + 100;   // distinct non-special

        int[,] delayed = MusicGenDelay.Apply(real, delay, bos, pad);
        Assert.Equal(t + 3, delayed.GetLength(0));      // grows by max delay (3)
        Assert.Equal(bos, delayed[0, 1]);               // channel 1 lead-in = BOS
        Assert.Equal(bos, delayed[2, 2]);               // channel 2 still in lead-in at row 2
        Assert.Equal(real[0, 2], delayed[3, 2]);        // first real content shifted by delay 3
        Assert.Equal(pad, delayed[t + 2, 1]);           // channel 1 tail = PAD

        int[,] back = MusicGenDelay.Revert(delayed, delay, t);
        for (int s = 0; s < t; s++)
            for (int c = 0; c < k; c++) Assert.Equal(real[s, c], back[s, c]);
    }

    [Fact]
    public void Dac44kHz_HasNineCodebooks_MatchingDiaChannels()
    {
        DiaConfig c = DiaConfig.Dia1_6B;
        Assert.Equal(c.Channels, DacConfig.Dac44kHz.NCodebooks);
    }

    [Fact]
    public void EncoderDecoderPipeline_ConstructCleanly()
    {
        DiaConfig c = DiaConfig.Dia1_6B;
        using DiaEncoder enc = new(c);
        using DiaDecoder dec = new(c);
        using DiaPipeline pipe = new(c);
        Assert.Equal(44_100, pipe.SampleRate);
        Assert.Equal(18, dec.NumLayers);
        Assert.Equal(4, dec.KvHeads);
    }

    [Fact]
    public void SyntheticForward_EncoderAndDecoderStep_AreFinite()
    {
        // Tiny config: enc 2L/dim16/2 heads, dec 2L/dim24/GQA 4:2, head_dim 8, 9 channels, small audio vocab.
        DiaConfig c = new()
        {
            EncoderLayers = 2, EncoderDim = 16, EncoderHeads = 2, EncoderKvHeads = 2, EncoderFfn = 32,
            DecoderLayers = 2, DecoderDim = 24, DecoderSelfQHeads = 4, DecoderSelfKvHeads = 2,
            DecoderCrossQHeads = 4, DecoderCrossKvHeads = 2, DecoderFfn = 48, HeadDim = 8,
            AudioVocab = 16, AudioEos = 13, AudioPad = 14, AudioBos = 15,
        };
        Dictionary<string, Tensor> w = SyntheticWeights(c);
        using CpuBackend backend = new();
        using DiaEncoder enc = new(c);
        using DiaDecoder dec = new(c);
        enc.LoadWeights(w);
        dec.LoadWeights(w);

        int[] text = [3, 7, 1, 9, 2];
        using Tensor encOut = enc.Forward(backend, text);
        Assert.Equal(new TensorShape(1, text.Length, c.EncoderDim), encOut.Shape);
        AssertFinite(encOut);

        dec.PrecomputeCrossKv(backend, encOut, text.Length);
        using StreamingKvCache cache = new(c.DecoderLayers, 1, c.DecoderSelfKvHeads, 8, c.HeadDim);
        Span<int> frame = stackalloc int[c.Channels];
        for (int ci = 0; ci < c.Channels; ci++) frame[ci] = c.AudioBos;
        using Tensor logits = dec.StepLogits(backend, frame, 0, cache);
        Assert.Equal(c.Channels * c.AudioVocab, (int)logits.Shape[2]);
        AssertFinite(logits);
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> SyntheticWeights(DiaConfig c)
    {
        int eh = c.HeadDim, encQ = c.EncoderHeads * eh, encKv = c.EncoderKvHeads * eh;
        int decQ = c.DecoderSelfQHeads * eh, decKv = c.DecoderSelfKvHeads * eh;
        int crQ = c.DecoderCrossQHeads * eh, crKv = c.DecoderCrossKvHeads * eh;
        Dictionary<string, Tensor> w = new()
        {
            ["model.encoder.embedding.weight"] = F(c.TextVocab, c.EncoderDim),
            ["model.encoder.norm.weight"] = Ones(c.EncoderDim),
            ["model.decoder.norm.weight"] = Ones(c.DecoderDim),
            ["logits_dense.weight"] = F(c.Channels * c.AudioVocab, c.DecoderDim),
        };
        for (int i = 0; i < c.EncoderLayers; i++)
        {
            string p = $"model.encoder.layers.{i}";
            w[$"{p}.pre_sa_norm.weight"] = Ones(c.EncoderDim);
            w[$"{p}.post_sa_norm.weight"] = Ones(c.EncoderDim);
            w[$"{p}.self_attention.q_proj.weight"] = F(encQ, c.EncoderDim);
            w[$"{p}.self_attention.k_proj.weight"] = F(encKv, c.EncoderDim);
            w[$"{p}.self_attention.v_proj.weight"] = F(encKv, c.EncoderDim);
            w[$"{p}.self_attention.o_proj.weight"] = F(c.EncoderDim, encQ);
            w[$"{p}.mlp.gate_up_proj.weight"] = F(2 * c.EncoderFfn, c.EncoderDim);
            w[$"{p}.mlp.down_proj.weight"] = F(c.EncoderDim, c.EncoderFfn);
        }
        for (int ci = 0; ci < c.Channels; ci++)
            w[$"model.decoder.embeddings.{ci}.weight"] = F(c.AudioVocab, c.DecoderDim);
        for (int i = 0; i < c.DecoderLayers; i++)
        {
            string p = $"model.decoder.layers.{i}";
            w[$"{p}.pre_sa_norm.weight"] = Ones(c.DecoderDim);
            w[$"{p}.pre_ca_norm.weight"] = Ones(c.DecoderDim);
            w[$"{p}.pre_mlp_norm.weight"] = Ones(c.DecoderDim);
            w[$"{p}.self_attention.q_proj.weight"] = F(decQ, c.DecoderDim);
            w[$"{p}.self_attention.k_proj.weight"] = F(decKv, c.DecoderDim);
            w[$"{p}.self_attention.v_proj.weight"] = F(decKv, c.DecoderDim);
            w[$"{p}.self_attention.o_proj.weight"] = F(c.DecoderDim, decQ);
            w[$"{p}.cross_attention.q_proj.weight"] = F(crQ, c.DecoderDim);
            w[$"{p}.cross_attention.k_proj.weight"] = F(crKv, c.EncoderDim);
            w[$"{p}.cross_attention.v_proj.weight"] = F(crKv, c.EncoderDim);
            w[$"{p}.cross_attention.o_proj.weight"] = F(c.DecoderDim, crQ);
            w[$"{p}.mlp.gate_up_proj.weight"] = F(2 * c.DecoderFfn, c.DecoderDim);
            w[$"{p}.mlp.down_proj.weight"] = F(c.DecoderDim, c.DecoderFfn);
        }
        return w;
    }
}
