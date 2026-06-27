using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.MeloTts;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for MeloTTS: the config, a synthetic-weights forward through the extended text encoder
/// (phoneme + tone + language + BERT summed embedding → reused VITS encoder layers) asserting finite prior, and
/// (gated) the full <see cref="MeloTts"/> text→audio facade against the deterministic reference.</summary>
public sealed unsafe class MeloTtsTests
{
    private readonly ITestOutputHelper _out;
    public MeloTtsTests(ITestOutputHelper o) => _out = o;

    private static uint _rng = 0xBADF00Du;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));

    [Fact]
    public void EnglishV3_ConfigReusesVitsCore()
    {
        MeloTtsConfig c = MeloTtsConfig.EnglishV3;
        Assert.Equal(192, c.Core.InterChannels);
        Assert.Equal(256, c.Core.GinChannels);        // multispeaker
        Assert.Equal(1_024, c.BertDim);
        Assert.Equal(768, c.JaBertDim);
        Assert.Equal(0.2f, c.SdpRatio);
    }

    [Fact]
    public void TextEncoder_SyntheticForward_IsFinite()
    {
        MeloTtsConfig c = new()
        {
            Core = new VitsConfig
            {
                NumVocab = 16, InterChannels = 8, HiddenChannels = 8, FilterChannels = 16, NumHeads = 2,
                NumEncoderLayers = 2, WindowSize = 4,
            },
            NumTones = 4, NumLanguages = 3, BertDim = 6, JaBertDim = 5,
        };
        using CpuBackend backend = new();
        MeloTtsTextEncoder enc = new(c);
        enc.LoadWeights(Weights(c));

        int t = 3;
        int[] phon = [2, 5, 1], tones = [1, 0, 3], langs = [0, 2, 1];
        using Tensor bert = F3(1, c.BertDim, t);
        using Tensor jaBert = F3(1, c.JaBertDim, t);
        (Tensor hidden, Tensor mP, Tensor logsP) = enc.Forward(backend, phon, tones, langs, bert, jaBert);
        Assert.Equal(new TensorShape(1, c.Core.InterChannels, t), mP.Shape);
        AssertFinite(mP); AssertFinite(logsP); AssertFinite(hidden);
        hidden.Dispose(); mP.Dispose(); logsP.Dispose();
    }

    /// <summary>Full facade end-to-end: text → CMUdict g2p + bert-base-uncased prosody → VITS → waveform. Gated on
    /// <c>MELO_CHECKPOINT</c> (the <c>.pth</c>/<c>.safetensors</c>), <c>MELO_BERT</c> (bert-base-uncased weights), and
    /// <c>MELO_VOCAB</c> (its <c>vocab.txt</c>). With <c>MELO_REF_AUDIO0</c> set (the noise-0 reference <c>.npy</c>) it
    /// also asserts numeric parity; otherwise it just asserts finite, non-silent audio and writes a WAV for QA.</summary>
    [Fact]
    public void Facade_TextToAudio_EndToEnd()
    {
        string? ckpt = Environment.GetEnvironmentVariable("MELO_CHECKPOINT");
        string? bert = Environment.GetEnvironmentVariable("MELO_BERT");
        string? vocab = Environment.GetEnvironmentVariable("MELO_VOCAB");
        if (string.IsNullOrEmpty(ckpt) || !File.Exists(ckpt)) return;
        if (string.IsNullOrEmpty(bert) || !File.Exists(bert)) return;
        if (string.IsNullOrEmpty(vocab) || !File.Exists(vocab)) return;

        // NoiseScaleW = 0 makes the stochastic duration predictor deterministic so the run reproduces the reference.
        MeloTtsConfig cfg = MeloTtsConfig.EnglishV3 with { NoiseScaleW = 0f };
        using MeloTts melo = MeloTts.LoadFromFiles(ckpt, bert, vocab, cfg);
        using CpuBackend backend = new();

        const string text = "This is a test of the speech synthesizer.";
        float[] audio = melo.SynthesizeText(backend, text, lengthScale: 1.0f, noiseScale: 0.0f, seed: 0);

        _out.WriteLine($"{audio.Length} samples ({audio.Length / (double)melo.SampleRate:F2}s) @ {melo.SampleRate}Hz");
        Assert.NotEmpty(audio);
        double sumSq = 0;
        foreach (float a in audio) { Assert.True(float.IsFinite(a)); sumSq += (double)a * a; }
        double rms = Math.Sqrt(sumSq / audio.Length);
        _out.WriteLine($"rms={rms:F4}");
        Assert.True(rms > 0.01, $"audio is near-silent (rms {rms:F4})");

        string? refPath = Environment.GetEnvironmentVariable("MELO_REF_AUDIO0");
        if (!string.IsNullOrEmpty(refPath) && File.Exists(refPath))
        {
            float[] reference = LoadNpyF32(refPath);
            Assert.Equal(reference.Length, audio.Length);
            float corr = Correlation(audio, reference);
            _out.WriteLine($"parity vs reference: corr={corr:F4}");
            Assert.True(corr > 0.99f, $"audio parity too low (corr {corr:F4})");
        }

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_melotts");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "melotts_e2e.wav");
        WavFile.WriteMono16(outPath, audio, melo.SampleRate);
        _out.WriteLine($"wrote {outPath}");
    }

    // Minimal little-endian float32 .npy reader (header line then raw data).
    private static float[] LoadNpyF32(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        int headerLen = b[8] | (b[9] << 8);
        int start = 10 + headerLen;
        int n = (b.Length - start) / 4;
        float[] d = new float[n];
        Buffer.BlockCopy(b, start, d, 0, n * 4);
        return d;
    }

    private static float Correlation(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double ma = 0, mb = 0;
        for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        double sa = 0, sb = 0, sab = 0;
        for (int i = 0; i < n; i++) { double da = a[i] - ma, db = b[i] - mb; sa += da * da; sb += db * db; sab += da * db; }
        return (float)(sab / (Math.Sqrt(sa * sb) + 1e-12));
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> Weights(MeloTtsConfig c)
    {
        int h = c.Core.HiddenChannels, inter = c.Core.InterChannels, kc = h / c.Core.NumHeads;
        Dictionary<string, Tensor> w = new()
        {
            ["enc_p.emb.weight"] = F2(c.Core.NumVocab, h),
            ["enc_p.tone_emb.weight"] = F2(c.NumTones, h),
            ["enc_p.language_emb.weight"] = F2(c.NumLanguages, h),
            ["enc_p.bert_proj.weight"] = F3(h, c.BertDim, 1), ["enc_p.bert_proj.bias"] = F1(h),
            ["enc_p.ja_bert_proj.weight"] = F3(h, c.JaBertDim, 1), ["enc_p.ja_bert_proj.bias"] = F1(h),
            ["enc_p.proj.weight"] = F3(2 * inter, h, 1), ["enc_p.proj.bias"] = F1(2 * inter),
        };
        for (int i = 0; i < c.Core.NumEncoderLayers; i++)
        {
            string e = "enc_p.encoder";
            foreach (string p in new[] { "conv_q", "conv_k", "conv_v", "conv_o" })
            { w[$"{e}.attn_layers.{i}.{p}.weight"] = F3(h, h, 1); w[$"{e}.attn_layers.{i}.{p}.bias"] = F1(h); }
            w[$"{e}.attn_layers.{i}.emb_rel_k"] = F3(1, 2 * c.Core.WindowSize + 1, kc);
            w[$"{e}.attn_layers.{i}.emb_rel_v"] = F3(1, 2 * c.Core.WindowSize + 1, kc);
            w[$"{e}.norm_layers_1.{i}.gamma"] = F1(h); w[$"{e}.norm_layers_1.{i}.beta"] = F1(h);
            w[$"{e}.norm_layers_2.{i}.gamma"] = F1(h); w[$"{e}.norm_layers_2.{i}.beta"] = F1(h);
            w[$"{e}.ffn_layers.{i}.conv_1.weight"] = F3(c.Core.FilterChannels, h, 3); w[$"{e}.ffn_layers.{i}.conv_1.bias"] = F1(c.Core.FilterChannels);
            w[$"{e}.ffn_layers.{i}.conv_2.weight"] = F3(h, c.Core.FilterChannels, 3); w[$"{e}.ffn_layers.{i}.conv_2.bias"] = F1(h);
        }
        return w;
    }
}
