using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Audio.Models.OpenVoice;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Full GPT-SoVITS v2 ENGLISH end-to-end on real weights + a real reference clip (JFK): G2P → phoneme
/// ids, zero BERT (English path), HuBERT(ref 16 kHz) → SSL → prompt codes, s1 AR → semantic tokens,
/// ref_enc(ref 32 kHz spec) → speaker, s2 SoVITS → 32 kHz waveform. Confirms the whole chain wires together and
/// produces finite, non-silent audio (each component is already numerically validated separately).
///
/// <para>Gated on <c>GSV_S2G</c> (s2G2333k.pth), <c>GSV_S1</c> (s1 ckpt), <c>GSV_HUBERT</c>
/// (chinese-hubert-base pytorch_model.bin), and a cached <c>jfk.wav</c>. Optional <c>GSV_CUDA=1</c> +
/// <c>GSV_PTX</c>; <c>GSV_TEXT</c>, <c>GSV_OUT_WAV</c>, <c>GSV_MAXNEW</c> (default 60).</para></summary>
public sealed unsafe class GptSoVitsEnglishEndToEndTests
{
    private readonly ITestOutputHelper _out;
    public GptSoVitsEnglishEndToEndTests(ITestOutputHelper o) => _out = o;

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
    [Trait("Category", "Integration")]
    public void English_EndToEnd_OnRealWeights()
    {
        string? s2p = Environment.GetEnvironmentVariable("GSV_S2G");
        string? s1p = Environment.GetEnvironmentVariable("GSV_S1");
        string? hubp = Environment.GetEnvironmentVariable("GSV_HUBERT");
        string clip = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        if (Missing(s2p) || Missing(s1p) || Missing(hubp) || !File.Exists(clip)) return; // gated

        // --- models ---
        PytorchPickleLoader s2l = new(); s2l.Load(s2p!);
        IReadOnlyDictionary<string, Tensor> s2w = s2l.GetAllTensors();
        SoVitsSynthesizer s2 = new(V2Config(), sslDim: 768, sslLayers: 3, textLayers: 6, enc2Layers: 3, mrteHidden: 512, mrteHeads: 4);
        s2.LoadWeights(s2w);
        SoVitsRefEnc refEnc = new(); refEnc.LoadWeights(s2w, "ref_enc");
        PytorchPickleLoader s1l = new(); s1l.Load(s1p!);
        Text2Semantic s1 = new(new Text2SemanticConfig()); s1.LoadWeights(s1l.GetAllTensors(), "model");
        PytorchPickleLoader hl = new(); hl.Load(hubp!);
        Hubert hubert = new(new HubertConfig()); hubert.LoadWeights(hl.GetAllTensors());
        using GptSoVitsPipeline pipe = new(hubert, s1, s2, refEnc);

        // --- reference audio: JFK clip → 16 kHz (HuBERT) + 32 kHz linear spec (speaker) ---
        WavFile.DecodedAudio dec = WavFile.Read(clip);
        float[] mono = dec.ToMono();
        float[] ref16 = dec.SampleRate == 16_000 ? mono : Resampler.Create(dec.SampleRate, 16_000).Resample(mono);
        float[] ref32 = dec.SampleRate == 32_000 ? mono : Resampler.Create(dec.SampleRate, 32_000).Resample(mono);
        Tensor refPcm = new(new TensorShape(1, 1, ref16.Length), DType.F32);
        new Span<float>(ref16).CopyTo(new Span<float>((void*)refPcm.DataPointer, ref16.Length));
        Tensor refSpec = LinearSpectrogram.Extract(ref32, 2048, 640);   // [1, 1025, tSpec]
        int tSpec = (int)refSpec.Shape[2];

        // --- text → phoneme ids (English G2P), zero BERT ---
        string refText = "ask not what your country can do for you ask what you can do for your country";
        string targetText = Environment.GetEnvironmentVariable("GSV_TEXT") ?? "Hello there, this is a voice cloning test.";
        int[] refIds = GptSoVitsSymbols.ToSequence(GptSoVitsFrontend.CleanText(refText, "en").Phones);
        int[] tgtIds = GptSoVitsSymbols.ToSequence(GptSoVitsFrontend.CleanText(targetText, "en").Phones);
        int[] allIds = refIds.Concat(tgtIds).ToArray();
        Tensor zeroBert = new(new TensorShape(1024, allIds.Length), DType.F32);   // English → zero BERT
        _out.WriteLine($"ref {refIds.Length} + target {tgtIds.Length} phonemes; ref clip {dec.SampleRate}Hz → spec [1,1025,{tSpec}].");

        IBackend backend;
        if (Environment.GetEnvironmentVariable("GSV_CUDA") == "1")
            backend = new CudaBackend(0, Environment.GetEnvironmentVariable("GSV_PTX") ?? throw new InvalidOperationException("GSV_CUDA=1 needs GSV_PTX."));
        else backend = new CpuBackend();

        int maxNew = int.TryParse(Environment.GetEnvironmentVariable("GSV_MAXNEW"), out int mn) ? mn : 60;
        long t0 = Environment.TickCount64;
        float[] audio = pipe.Generate(backend, refPcm, ref16.Length, refSpec, tSpec, allIds, zeroBert, tgtIds, maxNewTokens: maxNew, seed: 0);
        double secs = (Environment.TickCount64 - t0) / 1000.0;

        refPcm.Dispose(); refSpec.Dispose(); zeroBert.Dispose();
        (backend as IDisposable)?.Dispose();
        hubert.Dispose(); s1.Dispose(); s2.Dispose();

        Assert.NotEmpty(audio);
        double sumSq = 0; float peak = 0;
        foreach (float v in audio) { Assert.True(float.IsFinite(v)); sumSq += (double)v * v; peak = Math.Max(peak, Math.Abs(v)); }
        double rms = Math.Sqrt(sumSq / audio.Length);
        string outWav = Environment.GetEnvironmentVariable("GSV_OUT_WAV") ?? Path.Combine(Path.GetTempPath(), "gptsovits_en.wav");
        WavFile.WriteMono16(outWav, audio, 32_000);
        _out.WriteLine($"Generated {audio.Length} samples ({audio.Length / 32000.0:0.00}s) in {secs:0.0}s. RMS={rms:0.0000} peak={peak:0.0000}. WAV: {outWav}");
        Assert.True(rms > 1e-4, "output should not be silent");
    }

    private static bool Missing(string? p) => string.IsNullOrEmpty(p) || !File.Exists(p);
}
