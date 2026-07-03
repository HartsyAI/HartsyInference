using HartsyInference.Audio.Models.Music;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for MusicGen: the model presets, the delay-pattern staging math
/// (the one piece of MusicGen that is pure index bookkeeping and must be exactly right), and the
/// KV-cached incremental decode on a tiny synthetic-weight decoder.</summary>
public sealed unsafe class MusicGenTests
{
    private static uint _rng = 0x9E377u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }

    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor Filled(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Filled(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Filled(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f;
        return t;
    }

    private static Dictionary<string, Tensor> SyntheticDecoderWeights(MusicGenConfig c)
    {
        int h = c.Hidden;
        Dictionary<string, Tensor> w = new()
        {
            ["enc_to_dec_proj.weight"] = Filled(h, c.TextDim),
            ["layer_norm.weight"] = Ones(h),
            ["layer_norm.bias"] = Filled(h),
        };
        for (int i = 0; i < c.NumCodebooks; i++)
        {
            w[$"embed_tokens.{i}.weight"] = Filled(c.CodebookSize + 1, h);
            w[$"lm_heads.{i}.weight"] = Filled(c.CodebookSize, h);
        }
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"layers.{i}";
            w[$"{p}.self_attn_layer_norm.weight"] = Ones(h);
            w[$"{p}.self_attn_layer_norm.bias"] = Filled(h);
            w[$"{p}.self_attn.q_proj.weight"] = Filled(h, h);
            w[$"{p}.self_attn.k_proj.weight"] = Filled(h, h);
            w[$"{p}.self_attn.v_proj.weight"] = Filled(h, h);
            w[$"{p}.self_attn.out_proj.weight"] = Filled(h, h);
            w[$"{p}.encoder_attn_layer_norm.weight"] = Ones(h);
            w[$"{p}.encoder_attn_layer_norm.bias"] = Filled(h);
            w[$"{p}.encoder_attn.q_proj.weight"] = Filled(h, h);
            w[$"{p}.encoder_attn.k_proj.weight"] = Filled(h, h);
            w[$"{p}.encoder_attn.v_proj.weight"] = Filled(h, h);
            w[$"{p}.encoder_attn.out_proj.weight"] = Filled(h, h);
            w[$"{p}.final_layer_norm.weight"] = Ones(h);
            w[$"{p}.final_layer_norm.bias"] = Filled(h);
            w[$"{p}.fc1.weight"] = Filled(c.FfnDim, h);
            w[$"{p}.fc2.weight"] = Filled(h, c.FfnDim);
        }
        return w;
    }
    [Fact]
    public void Presets_MatchAudioCraftShapes()
    {
        Assert.Equal((1_024, 24, 16), Dims(MusicGenConfig.Small));
        Assert.Equal((1_536, 48, 24), Dims(MusicGenConfig.Medium));
        Assert.Equal((2_048, 48, 32), Dims(MusicGenConfig.Large));
        Assert.Equal(32_000, MusicGenConfig.Small.CodecSampleRate);
        Assert.Equal(16_000, MusicGenConfig.AudioGen.CodecSampleRate);
        Assert.Equal(4, MusicGenConfig.Small.NumCodebooks);
        Assert.Equal(2_048, MusicGenConfig.Small.SpecialToken);   // == codebook size
        Assert.Equal(64, MusicGenConfig.Small.HeadDim);           // 1024 / 16
    }

    [Fact]
    public void Delay_RevertUndoesApply()
    {
        int[] delay = [0, 1, 2, 3];
        int special = 2_048;
        int t = 5, k = 4;
        int[,] real = new int[t, k];
        for (int j = 0; j < t; j++)
            for (int c = 0; c < k; c++) real[j, c] = j * 10 + c;   // distinct, non-special values

        int[,] delayed = MusicGenDelay.Apply(real, delay, special);
        Assert.Equal(t + 3, delayed.GetLength(0));                 // grid grows by maxDelay
        int[,] back = MusicGenDelay.Revert(delayed, delay, t);

        for (int j = 0; j < t; j++)
            for (int c = 0; c < k; c++) Assert.Equal(real[j, c], back[j, c]);
    }

    [Fact]
    public void Delay_StaggersCodebooksWithSpecialLeadIn()
    {
        int[] delay = [0, 1, 2, 3];
        int special = 2_048;
        int[,] real = { { 7, 7, 7, 7 } };                          // single frame, all 7s
        int[,] delayed = MusicGenDelay.Apply(real, delay, special);

        // Codebook 0 is active at step 0; codebooks 1/2/3 are still in their special lead-in.
        Assert.Equal(7, delayed[0, 0]);
        Assert.Equal(special, delayed[0, 1]);
        Assert.Equal(special, delayed[0, 2]);
        Assert.Equal(special, delayed[0, 3]);
        // Each codebook's single real token lands on its delayed row.
        Assert.Equal(7, delayed[1, 1]);
        Assert.Equal(7, delayed[2, 2]);
        Assert.Equal(7, delayed[3, 3]);

        Assert.True(MusicGenDelay.IsActive(0, 0, delay));
        Assert.False(MusicGenDelay.IsActive(0, 3, delay));
        Assert.True(MusicGenDelay.IsActive(3, 3, delay));
    }

    [Fact]
    public void Decoder_ForwardStep_MatchesFullSequenceForward()
    {
        MusicGenConfig cfg = new() { Hidden = 16, NumLayers = 2, NumHeads = 4, NumCodebooks = 2, CodebookSize = 8, SpecialToken = 8, TextDim = 12, DelayPattern = [0, 1] };
        using CpuBackend backend = new();
        using MusicGenDecoder dec = new(cfg);
        dec.LoadWeights(SyntheticDecoderWeights(cfg), prefix: "");

        const int prefillLen = 3, totalLen = 7, textLen = 5;
        using Tensor t5 = Filled(1, textLen, cfg.TextDim);
        using Tensor cross = dec.ProjectText(backend, t5);
        int[,] frames = new int[totalLen, cfg.NumCodebooks];
        for (int s = 0; s < totalLen; s++)
            for (int c = 0; c < cfg.NumCodebooks; c++) frames[s, c] = (s * 3 + c) % cfg.CodebookSize;

        // Incremental: prefill the first 3 frames (cross K/V primed once), then step frames 3..6 one at a time.
        MusicGenKvCache cache = dec.CreateCache(backend, cross, capacity: totalLen);
        using Tensor prefillEmbeds = dec.EmbedFrames(Rows(frames, prefillLen));
        dec.Forward(backend, prefillEmbeds, cross, cache);
        Assert.Equal(prefillLen, cache.Length);

        for (int pos = prefillLen; pos < totalLen; pos++)
        {
            // Probe a placeholder first (advance: false) — the real step must overwrite its K/V row cleanly.
            dec.ForwardStep(backend, new int[cfg.NumCodebooks], cache, advance: false);
            int[] frame = new int[cfg.NumCodebooks];
            for (int c = 0; c < cfg.NumCodebooks; c++) frame[c] = frames[pos, c];
            float[][] step = dec.ForwardStep(backend, frame, cache);

            // Reference: one full causal forward over frames [0..pos] → last-position logits.
            using Tensor embeds = dec.EmbedFrames(Rows(frames, pos + 1));
            float[][] full = dec.Forward(backend, embeds, cross);
            for (int c = 0; c < cfg.NumCodebooks; c++)
                for (int v = 0; v < cfg.CodebookSize; v++)
                    Assert.True(MathF.Abs(full[c][v] - step[c][v]) < 1e-4f,
                        $"pos {pos} cb {c} tok {v}: full={full[c][v]} step={step[c][v]}");
        }
        Assert.Equal(totalLen, cache.Length);
    }

    private static int[,] Rows(int[,] src, int count)
    {
        int k = src.GetLength(1);
        int[,] slice = new int[count, k];
        for (int s = 0; s < count; s++)
            for (int c = 0; c < k; c++) slice[s, c] = src[s, c];
        return slice;
    }

    private static (int, int, int) Dims(MusicGenConfig c) => (c.Hidden, c.NumLayers, c.NumHeads);
}
