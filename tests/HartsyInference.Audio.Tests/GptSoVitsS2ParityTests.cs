using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numeric parity for the GPT-SoVITS v2 stage-2 stack against the real <c>s2G2333k.pth</c>. Validates
/// the reference speaker encoder (<c>ge</c>) and the full deterministic decode (dequant → enc_p → flow → HiFiGAN,
/// noise=0) against golden stats from the upstream <c>SynthesizerTrn.decode</c>. Gated on
/// <c>HARTSYINFERENCE_S2G_V2</c>; skips cleanly when absent.</summary>
public sealed unsafe class GptSoVitsS2ParityTests
{
    private const int N = 20, Tp = 12, Tr = 60, SpecCh = 1025, RefCh = 704;

    private static VitsConfig V2Config() => new()
    {
        InterChannels = 192, HiddenChannels = 192, FilterChannels = 768, NumHeads = 2, NumEncoderLayers = 6,
        EncoderKernelSize = 3, WindowSize = 4, GinChannels = 512,
        FlowLayers = 4, FlowFlows = 4, FlowKernelSize = 5, FlowDilationRate = 1,
        ResBlock = "1", ResBlockKernelSizes = [3, 7, 11],
        ResBlockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        UpsampleRates = [10, 8, 2, 2, 2], UpsampleInitialChannel = 512, UpsampleKernelSizes = [16, 16, 8, 2, 2],
        SampleRate = 32_000,
    };

    [Fact]
    public void S2Decode_MatchesUpstream_OnRealWeights()
    {
        string? pt = Environment.GetEnvironmentVariable("HARTSYINFERENCE_S2G_V2");
        if (string.IsNullOrEmpty(pt) || !File.Exists(pt)) return;

        using CpuBackend backend = new();
        using PytorchPickleLoader loader = new();
        loader.Load(pt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        // ge = ref_enc(refer[:, :704]). refer[c,t] = ((c*5+t*11)%97)/97 - 0.5.
        SoVitsRefEnc refEnc = new();
        refEnc.LoadWeights(w, "ref_enc");
        Tensor refer = new(new TensorShape(1, RefCh, Tr), DType.F32);
        float* rp = (float*)refer.DataPointer;
        for (int c = 0; c < RefCh; c++)
            for (int t = 0; t < Tr; t++)
                rp[c * Tr + t] = ((c * 5 + t * 11) % 97) / 97f - 0.5f;
        Tensor ge = refEnc.Forward(backend, refer, Tr);
        refer.Dispose();

        float* gp = (float*)ge.DataPointer;
        Assert.Equal(8.46714f, gp[0], precision: 1);
        Assert.Equal(12.24901f, gp[1], precision: 1);
        Assert.Equal(-2.52898f, gp[2], precision: 1);
        Assert.Equal(0.51629f, Mean(gp, 512), precision: 1);

        // Full decode (noise=0): codes (i*53)%1024, phonemes (i*61)%732.
        using SoVitsSynthesizer sv = new(V2Config(), sslDim: 768, sslLayers: 3, textLayers: 6, enc2Layers: 3,
            mrteHidden: 512, mrteHeads: 4);
        sv.LoadWeights(w);
        int[] codes = new int[N]; for (int i = 0; i < N; i++) codes[i] = (i * 53) % 1024;
        int[] phon = new int[Tp]; for (int i = 0; i < Tp; i++) phon[i] = (i * 61) % 732;

        float[] audio = sv.Forward(backend, codes, phon, ge, seed: 0, noiseScale: 0f);
        ge.Dispose();

        Assert.Equal(N * 2 * 640, audio.Length);     // t=40 frames × hop 640 = 25600
        float mean = 0f, sq = 0f;
        foreach (float s in audio) { mean += s; sq += s * s; }
        mean /= audio.Length;
        float std = MathF.Sqrt(sq / audio.Length - mean * mean);
        Assert.Equal(-0.00009f, mean, precision: 3);
        Assert.Equal(0.00491f, std, precision: 3);
        Assert.Equal(-0.01276f, audio[0], precision: 2);
        Assert.Equal(-0.01301f, audio[1], precision: 2);
    }

    private static float Mean(float* p, int n)
    {
        float s = 0f;
        for (int i = 0; i < n; i++) s += p[i];
        return s / n;
    }

    [Fact]
    public void SslEncode_MatchesUpstream_Codes()
    {
        string? pt = Environment.GetEnvironmentVariable("HARTSYINFERENCE_S2G_V2");
        if (string.IsNullOrEmpty(pt) || !File.Exists(pt)) return;

        using CpuBackend backend = new();
        using PytorchPickleLoader loader = new();
        loader.Load(pt);
        using SoVitsSynthesizer sv = new(V2Config(), sslDim: 768, sslLayers: 3, textLayers: 6, enc2Layers: 3,
            mrteHidden: 512, mrteHeads: 4);
        sv.LoadWeights(loader.GetAllTensors());

        const int hubT = 80;
        Tensor ssl = new(new TensorShape(1, 768, hubT), DType.F32);
        float* sp = (float*)ssl.DataPointer;
        for (int c = 0; c < 768; c++)
            for (int t = 0; t < hubT; t++) sp[c * hubT + t] = (((c * 3 + t * 7) % 101) / 101f - 0.5f) * 2f;

        int[] codes = sv.EncodeSsl(backend, ssl);
        ssl.Dispose();

        int[] golden = [14, 664, 563, 1016, 684, 619, 263, 242, 854, 330, 201, 348, 689, 749, 451, 541];
        Assert.Equal(hubT / 2, codes.Length);
        for (int i = 0; i < golden.Length; i++) Assert.Equal(golden[i], codes[i]);
    }

    [Fact]
    public void EncP_MatchesUpstream_MpLogsP()
    {
        string? pt = Environment.GetEnvironmentVariable("HARTSYINFERENCE_S2G_V2");
        if (string.IsNullOrEmpty(pt) || !File.Exists(pt)) return;

        using CpuBackend backend = new();
        using PytorchPickleLoader loader = new();
        loader.Load(pt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        SoVitsRefEnc refEnc = new();
        refEnc.LoadWeights(w, "ref_enc");
        Tensor refer = new(new TensorShape(1, RefCh, Tr), DType.F32);
        float* rp = (float*)refer.DataPointer;
        for (int c = 0; c < RefCh; c++)
            for (int t = 0; t < Tr; t++) rp[c * Tr + t] = ((c * 5 + t * 11) % 97) / 97f - 0.5f;
        Tensor ge = refEnc.Forward(backend, refer, Tr); refer.Dispose();

        // Dequantize codes → [1,768,2N] (nearest x2). Codebook is F16 → cast.
        using Tensor cb = HartsyInference.Audio.Models.Whisper.WhisperOps.EnsureF32(w["quantizer.vq.layers.0._codebook.embed"]);
        int t2 = N * 2;
        Tensor latent = new(new TensorShape(1, 768, t2), DType.F32);
        float* lp = (float*)latent.DataPointer, cbp = (float*)cb.DataPointer;
        for (int i = 0; i < N; i++)
        {
            int code = (i * 53) % 1024;
            for (int c = 0; c < 768; c++)
            {
                float v = cbp[(long)code * 768 + c];
                lp[(long)c * t2 + 2 * i] = v; lp[(long)c * t2 + 2 * i + 1] = v;
            }
        }
        int[] phon = new int[Tp]; for (int i = 0; i < Tp; i++) phon[i] = (i * 61) % 732;

        SoVitsEncP encP = new(V2Config(), sslDim: 768, sslLayers: 3, textLayers: 6, enc2Layers: 3,
            mrteHidden: 512, mrteHeads: 4);
        encP.LoadWeights(w, "enc_p");
        (Tensor mP, Tensor logsP) = encP.Forward(backend, latent, t2, phon, ge);
        latent.Dispose(); ge.Dispose();
        try
        {
            float* mp = (float*)mP.DataPointer; float* lsp = (float*)logsP.DataPointer;
            // m_p/logs_p feed flow+dec; tolerance ~0.01 absolute (deep enc_p over fp16-origin weights).
            Assert.Equal(0.03955f, mp[0], tolerance: 0.012f);
            Assert.Equal(-0.00792f, mp[1], tolerance: 0.012f);
            Assert.Equal(-0.08396f, mp[2], tolerance: 0.012f);
            Assert.Equal(0.2103f, lsp[0], tolerance: 0.012f);
            Assert.Equal(0.10628f, lsp[1], tolerance: 0.012f);
            Assert.Equal(0.00093f, Mean(mp, 192 * t2), tolerance: 0.003f);
        }
        finally { mP.Dispose(); logsP.Dispose(); }
    }
}
