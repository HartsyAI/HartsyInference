using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Audio.Models.OpenVoice;
using HartsyInference.Audio.Models.Rvc;
using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weight forward tests for the net-new neural encoders: Zonos speaker encoder + prefix
/// conditioner, RVC RMVPE pitch extractor, OpenVoice tone-color extractor + linear spectrogram, and the
/// NeuCodec FSQ encoder. Each test loads random weights into a tiny config and asserts finite output /
/// valid token ranges. Mirrors the synthetic-weight pattern in <see cref="HubertTests"/>.</summary>
public sealed unsafe class AudioEncodersTests
{
    private static uint _rng = 0x51A3u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F3(int a, int b, int c) => Fill(new Tensor(new TensorShape(a, b, c), DType.F32));
    private static Tensor F4(int a, int b, int c, int d) => Fill(new Tensor(new TensorShape(a, b, c, d), DType.F32));

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    [Fact]
    public void ZonosSpeakerEncoder_SyntheticForward_Produces128dEmbedding()
    {
        ZonosSpeakerConfig c = new()
        {
            BaseWidth = 8,
            StageWidths = [8, 16],
            StageBlocks = [1, 1],
            PooledFreq = 40,        // 80-mel, strides [1,2] → 40
            AspAttentionDim = 16,
            BottleneckDim = 32,
            EmbedDim = 128,
        };
        using CpuBackend backend = new();
        using ZonosSpeakerEncoder enc = new(c);
        (Dictionary<string, Tensor> w, Dictionary<string, Tensor> lda) = ZonosSpeakerWeights(c);
        enc.LoadWeights(w, lda);

        int t = 48;
        using Tensor mel = F3(1, c.NumMels, t);
        using Tensor emb = enc.Embed(backend, mel, t);
        Assert.Equal(1, (int)emb.Shape[0]);
        Assert.Equal(128, (int)emb.Shape[1]);
        AssertFinite(emb);
    }

    [Fact]
    public void ZonosConditioning_SyntheticForward_BuildsPrefix()
    {
        using CpuBackend backend = new();
        using ZonosConditioning cond = new();
        cond.LoadWeights(ZonosCondWeights(phonemeVocab: 64, numLanguages: 8));

        int[] phonemes = [4, 5, 6, 7, 8];
        using Tensor speaker = F2(1, ZonosConditioning.SpeakerDim);
        float[] emotion = new float[ZonosConditioning.EmotionDim];
        for (int i = 0; i < emotion.Length; i++) emotion[i] = 0.1f * i;

        using Tensor prefix = cond.BuildPrefix(backend, phonemes, speaker, emotion,
            fmax: 22050f, pitchStd: 45f, speakingRate: 15f, languageId: 3);

        // Channels-last [1, P, DModel] — the [B, seq, hidden] layout the backbone/pipeline consume.
        Assert.Equal(phonemes.Length + 6, (int)prefix.Shape[1]);
        Assert.Equal(ZonosConditioning.DModel, (int)prefix.Shape[2]);
        AssertFinite(prefix);
    }

    [Fact]
    public void OpenVoiceSpeakerEncoder_SyntheticForward_ProducesG()
    {
        OpenVoiceSpeakerConfig c = new()
        {
            SpecChannels = 64,
            Channels = [16, 32],
            KernelSize = 3,
            Stride = 2,
            Padding = 1,
            GruHidden = 8,
            Gin = 256,
        };
        using CpuBackend backend = new();
        using OpenVoiceSpeakerEncoder enc = new(c);
        enc.LoadWeights(OpenVoiceWeights(c));

        int t = 60;
        using Tensor spec = F3(1, c.SpecChannels, t);
        using Tensor g = enc.Extract(backend, spec, t);
        Assert.Equal(1, (int)g.Shape[0]);
        Assert.Equal(256, (int)g.Shape[1]);
        Assert.Equal(1, (int)g.Shape[2]);
        AssertFinite(g);
    }

    [Fact]
    public void LinearSpectrogram_ProducesCorrectShape()
    {
        float[] pcm = new float[8_000];
        for (int i = 0; i < pcm.Length; i++) pcm[i] = 0.2f * MathF.Sin(2f * MathF.PI * 200f * i / 22_050f);
        int nFft = 512, hop = 128;
        using Tensor spec = LinearSpectrogram.Extract(pcm, nFft, hop);
        Assert.Equal(1, (int)spec.Shape[0]);
        Assert.Equal(nFft / 2 + 1, (int)spec.Shape[1]);
        Assert.Equal(pcm.Length / hop + 1, (int)spec.Shape[2]);
        AssertFinite(spec);
    }

    // NOTE: the former NeuCodecEncoder_SyntheticForward test (and its NeuCodecEncoderWeights helper) were
    // removed — they targeted the obsolete single-branch encoder (keys encoder.stem.* / encoder.stages.* /
    // fc_prior.*, config props Ngf/DownRatios/StemKernel/FinalKernel/FcInDim). The current NeuCodecEncoder is a
    // two-branch (acoustic + Wav2Vec2-BERT semantic) design whose real-weight validation lives in
    // NeuCodecEncoderDumpTest (diffed against tests/python-reference/neucodec_ref.py).

    // ── synthetic weight dictionaries ──

    /// <summary>Builds synthetic ResNet293-scheme weights (folded-conv keys) + the separate LDA dict.
    /// running_var is filled positive so the folded BatchNorm has a real square root.</summary>
    private static (Dictionary<string, Tensor>, Dictionary<string, Tensor>) ZonosSpeakerWeights(ZonosSpeakerConfig c)
    {
        Dictionary<string, Tensor> w = new()
        {
            ["front.conv1.weight"] = F4(c.BaseWidth, 1, 3, 3),
        };
        Bn(w, "front.bn1", c.BaseWidth);
        int inC = c.BaseWidth;
        for (int s = 0; s < c.StageBlocks.Count; s++)
        {
            int outC = c.StageWidths[s];
            int stride = s == 0 ? 1 : 2;
            for (int b = 0; b < c.StageBlocks[s]; b++)
            {
                string bp = $"front.layer{s + 1}.{b}";
                int blockStride = b == 0 ? stride : 1;
                int blockIn = b == 0 ? inC : outC;
                w[$"{bp}.conv1.weight"] = F4(outC, blockIn, 3, 3);
                Bn(w, $"{bp}.bn1", outC);
                w[$"{bp}.conv2.weight"] = F4(outC, outC, 3, 3);
                Bn(w, $"{bp}.bn2", outC);
                if (blockStride != 1 || blockIn != outC)
                {
                    w[$"{bp}.downsample.0.weight"] = F4(outC, blockIn, 1, 1);
                    Bn(w, $"{bp}.downsample.1", outC);
                }
            }
            inC = outC;
        }
        int poolIn = c.StageWidths[^1] * c.PooledFreq;
        w["pooling.attention.0.weight"] = F3(c.AspAttentionDim, poolIn, 1);
        w["pooling.attention.0.bias"] = F1(c.AspAttentionDim);
        Bn(w, "pooling.attention.2", c.AspAttentionDim);
        w["pooling.attention.3.weight"] = F3(poolIn, c.AspAttentionDim, 1);
        w["pooling.attention.3.bias"] = F1(poolIn);
        w["bottleneck.weight"] = F2(c.BottleneckDim, 2 * poolIn);
        w["bottleneck.bias"] = F1(c.BottleneckDim);
        Dictionary<string, Tensor> lda = new()
        {
            ["weight"] = F2(c.EmbedDim, c.BottleneckDim),
            ["bias"] = F1(c.EmbedDim),
        };
        return (w, lda);
    }

    /// <summary>Adds a synthetic BatchNorm parameter set (running_var forced positive) under <paramref name="p"/>.</summary>
    private static void Bn(Dictionary<string, Tensor> w, string p, int c)
    {
        w[$"{p}.weight"] = F1(c);
        w[$"{p}.bias"] = F1(c);
        w[$"{p}.running_mean"] = F1(c);
        Tensor v = new(new TensorShape(c), DType.F32);
        float* vp = (float*)v.DataPointer;
        for (int i = 0; i < c; i++) vp[i] = 0.5f + 0.5f * (i % 3);
        w[$"{p}.running_var"] = v;
    }

    private static Dictionary<string, Tensor> ZonosCondWeights(int phonemeVocab, int numLanguages)
    {
        int d = ZonosConditioning.DModel;
        string p = "prefix_conditioner";
        Dictionary<string, Tensor> w = new()
        {
            [$"{p}.conditioners.0.phoneme_embedder.weight"] = F2(phonemeVocab, d),
            [$"{p}.conditioners.1.project.weight"] = F2(d, ZonosConditioning.SpeakerDim),
            [$"{p}.conditioners.1.project.bias"] = F1(d),
            [$"{p}.conditioners.2.weight"] = F2(d / 2, ZonosConditioning.EmotionDim),
            [$"{p}.conditioners.3.weight"] = F2(d / 2, 1),
            [$"{p}.conditioners.4.weight"] = F2(d / 2, 1),
            [$"{p}.conditioners.5.weight"] = F2(d / 2, 1),
            [$"{p}.conditioners.6.int_embedder.weight"] = F2(numLanguages, d),
            [$"{p}.project.weight"] = F2(d, d),
            [$"{p}.project.bias"] = F1(d),
            [$"{p}.norm.weight"] = F1(d),
            [$"{p}.norm.bias"] = F1(d),
        };
        return w;
    }

    private static Dictionary<string, Tensor> OpenVoiceWeights(OpenVoiceSpeakerConfig c)
    {
        string p = "ref_enc";
        Dictionary<string, Tensor> w = new()
        {
            [$"{p}.layernorm.weight"] = F1(c.SpecChannels),
            [$"{p}.layernorm.bias"] = F1(c.SpecChannels),
        };
        int inC = 1, freq = c.SpecChannels;
        for (int i = 0; i < c.Channels.Count; i++)
        {
            int outC = c.Channels[i];
            w[$"{p}.convs.{i}.weight_g"] = F4(outC, 1, 1, 1);
            w[$"{p}.convs.{i}.weight_v"] = F4(outC, inC, c.KernelSize, c.KernelSize);
            w[$"{p}.convs.{i}.bias"] = F1(outC);
            inC = outC;
            freq = (freq + 2 * c.Padding - c.KernelSize) / c.Stride + 1;
        }
        int gruIn = c.Channels[^1] * freq;
        int g3 = 3 * c.GruHidden;
        w[$"{p}.gru.weight_ih_l0"] = F2(g3, gruIn);
        w[$"{p}.gru.weight_hh_l0"] = F2(g3, c.GruHidden);
        w[$"{p}.gru.bias_ih_l0"] = F1(g3);
        w[$"{p}.gru.bias_hh_l0"] = F1(g3);
        w[$"{p}.proj.weight"] = F2(c.Gin, c.GruHidden);
        w[$"{p}.proj.bias"] = F1(c.Gin);
        return w;
    }

}
