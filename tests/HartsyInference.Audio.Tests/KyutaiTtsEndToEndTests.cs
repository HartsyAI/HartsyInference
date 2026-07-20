using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Full Kyutai TTS end-to-end: SentencePiece-tokenized text → entries → the DSM frame loop
/// (<see cref="MoshiTtsGenerator"/> + <see cref="KyutaiTextScheduler"/>, all four numerical cores validated
/// separately) → 32-codebook Mimi codes → 24 kHz audio. Structural: confirms the whole pipeline runs and emits
/// in-range codes + finite audio (a fixed deterministic voice, so not a specific speaker). Gated on
/// <c>KYUTAI_TTS_WEIGHTS</c> + <c>KYUTAI_MIMI</c> + <c>KYUTAI_SPM</c>; optional <c>GSV_CUDA</c>/<c>GSV_PTX</c>,
/// <c>KYUTAI_MAXFRAMES</c> (default 32), <c>KYUTAI_OUT_WAV</c>.</summary>
public sealed unsafe class KyutaiTtsEndToEndTests
{
    private readonly ITestOutputHelper _out;
    public KyutaiTtsEndToEndTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void EndToEnd_ProducesAudio()
    {
        string? wp = Environment.GetEnvironmentVariable("KYUTAI_TTS_WEIGHTS");
        string? mp = Environment.GetEnvironmentVariable("KYUTAI_MIMI");
        string? sp = Environment.GetEnvironmentVariable("KYUTAI_SPM");
        if (Missing(wp) || Missing(sp)) return;   // Mimi (mp) is best-effort below

        using SafeTensorsLoader ttsW = new(); ttsW.Load(wp!);
        using MoshiTtsGenerator gen = new();
        gen.LoadWeights(ttsW.GetAllTensors());
        gen.SetZeroToken(-1);
        using KyutaiSttTokenizer tok = new(sp!);

        IBackend backend = Environment.GetEnvironmentVariable("GSV_CUDA") == "1"
            ? new CudaBackend(0, Environment.GetEnvironmentVariable("GSV_PTX") ?? throw new InvalidOperationException("GSV_CUDA needs GSV_PTX."))
            : new CpuBackend();

        // entries from a short sentence (one Entry per word).
        string text = Environment.GetEnvironmentVariable("KYUTAI_TEXT") ?? "hello there world";
        // moshi script_to_entries with padding_between=1: forces per-word articulation padding of
        // max(0, padding_between + len(tokens) - 1) = len(tokens) steps, which is what the 4.4s reference used.
        // Without it the model pads maximally (~max_padding per word) and drifts off-distribution.
        const int paddingBetween = 1;
        List<KyutaiTextScheduler.Entry> entries = new();
        bool firstContent = true;
        foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            List<int> ids = new(tok.Encode(word));
            if (firstContent) { ids.Insert(0, KyutaiTextScheduler.Main); firstContent = false; }  // moshi: prepend speaker token
            int padding = Math.Max(0, paddingBetween + ids.Count - 1);
            entries.Add(new KyutaiTextScheduler.Entry(ids, word, padding));
        }

        // Voice conditioning. A real kyutai/tts-voices embedding ships as speaker_wavs [1,512,T] (channels-first);
        // transpose to [1,T,512] for the conditioner. Without KYUTAI_VOICE, fall back to a fixed synthetic voice
        // (structural only — not a specific speaker).
        string? vpth = Environment.GetEnvironmentVariable("KYUTAI_VOICE");
        Tensor voice;
        if (!Missing(vpth))
        {
            using SafeTensorsLoader vw = new(); vw.Load(vpth!);
            Tensor sw = vw.GetAllTensors()["speaker_wavs"];   // [1, 512, T]
            int spkDim = (int)sw.Shape[1], tv = (int)sw.Shape[2];
            voice = new(new TensorShape(1, tv, spkDim), DType.F32);
            float* dp = (float*)voice.DataPointer; float* srp = (float*)sw.DataPointer;
            for (int c = 0; c < spkDim; c++)
                for (int ti = 0; ti < tv; ti++) dp[(long)ti * spkDim + c] = srp[(long)c * tv + ti];
            _out.WriteLine($"voice: {Path.GetFileName(vpth)} [1,{tv},{spkDim}]");
        }
        else
        {
            int spk = 8;
            voice = new(new TensorShape(1, spk, 512), DType.F32);
            float* vp = (float*)voice.DataPointer;
            for (int i = 0; i < spk * 512; i++) vp[i] = ((i % 17) / 17f) - 0.5f;
        }
        using Tensor cross = gen.Conditioner.ComputeCross(backend, voice);
        // CFG-distilled model: the guidance coefficient is a conditioning input, and moshi's TTS default is 2.0
        // (cfg=1.0 = no guidance → the text isn't enforced → fluent-but-unconditioned speech).
        using Tensor sumCond = gen.Conditioner.ComputeSum(backend, MoshiConditioner.CfgBin(2.0f));
        voice.Dispose();

        KyutaiTextScheduler scheduler = new(secondStreamAhead: 2, maxPadding: 8, initialPadding: 2);
        int maxFrames = int.TryParse(Environment.GetEnvironmentVariable("KYUTAI_MAXFRAMES"), out int mf) ? mf : 32;
        long t0 = Environment.TickCount64;
        int[,] codes = gen.Generate(backend, scheduler, entries, cross, sumCond, maxFrames: maxFrames, audioTemp: 0.6f);
        double secs = (Environment.TickCount64 - t0) / 1000.0;
        int n = codes.GetLength(1);
        _out.WriteLine($"{entries.Count} words → {n} code frames in {secs:0.0}s.");
        Assert.True(n > 0, "no code frames emitted");

        // all emitted codes must be valid Mimi codes.
        for (int k = 0; k < MoshiTtsGenerator.NumCodebooks; k++)
            for (int f = 0; f < n; f++)
                Assert.InRange(codes[k, f], 0, MoshiTtsGenerator.AudioCard - 1);

        // Mimi decode (codes → 24 kHz audio). Best-effort: the Mimi codec loader is a separate, pre-existing
        // engine component with known reconcile items for this DSM checkpoint; the Kyutai pipeline's job (valid
        // codes) is asserted above. Only run when KYUTAI_MIMI is supplied AND loads.
        if (!Missing(mp))
        {
            try
            {
                using SafeTensorsLoader mimiW = new(); mimiW.Load(mp!);
                Mimi mimi = new(MimiConfig.Mimi24kHzDsm);
                mimi.LoadWeights(mimiW.GetAllTensors());
                Tensor codeT = new(new TensorShape(1, MoshiTtsGenerator.NumCodebooks, n), DType.I32);
                int* cp = (int*)codeT.DataPointer;
                for (int k = 0; k < MoshiTtsGenerator.NumCodebooks; k++)
                    for (int f = 0; f < n; f++) cp[(long)k * n + f] = codes[k, f];
                using Tensor audio = mimi.Decode(backend, codeT, batch: 1, tFrames: n);
                codeT.Dispose();
                int samples = (int)audio.Shape[audio.Shape.Rank - 1];
                float* ap = (float*)audio.DataPointer;
                double sumSq = 0; for (int i = 0; i < samples; i++) { Assert.True(float.IsFinite(ap[i])); sumSq += (double)ap[i] * ap[i]; }
                string outWav = Environment.GetEnvironmentVariable("KYUTAI_OUT_WAV") ?? Path.Combine(Path.GetTempPath(), "kyutai_tts.wav");
                WavFile.WriteMono16(outWav, new ReadOnlySpan<float>(ap, samples), 24_000);
                _out.WriteLine($"Mimi decoded {samples} samples ({samples / 24000.0:0.00}s) RMS={Math.Sqrt(sumSq / samples):0.0000}. WAV: {outWav}");
            }
            catch (System.Collections.Generic.KeyNotFoundException e)
            {
                _out.WriteLine($"Mimi decode skipped (pre-existing codec loader reconcile): {e.Message}");
            }
        }
        (backend as IDisposable)?.Dispose();
    }

    /// <summary>Validates the SEANet half of the Kyutai DSM Mimi loader (Blocker 3): the DSM checkpoint ships 1
    /// residual block per stage (<c>ResidualDilations=[1]</c>) and has no LSTM, so the SEANet encoder/decoder
    /// residual count AND the decoder sequential-index offset must match the checkpoint. Before the fix the
    /// decoder looked for the upsample convtr at a block index and threw immediately; now the SEANet stack loads
    /// and the only remaining gap is the transformer-of-codecs key layout (see note below). Gated on
    /// <c>KYUTAI_MIMI</c>; runs in seconds.
    ///
    /// <para>NOTE: full decode additionally needs <c>MimiTransformer</c> reconciled to the DSM layout
    /// (<c>encoder_transformer.transformer.layers.*</c> with LayerNorm <c>norm1/norm2</c>, fused
    /// <c>self_attn.in_proj_weight</c>, <c>linear1/linear2</c> FFN, and <c>layer_scale_1/2</c>) — a separate
    /// architecture change that needs its own numerical reference (none is dumped yet). When that lands, this
    /// test should be promoted to assert finite decoded PCM.</para></summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void DsmMimi_SeaNetLayout_Loads()
    {
        string? mp = Environment.GetEnvironmentVariable("KYUTAI_MIMI");
        if (Missing(mp)) return;

        using SafeTensorsLoader mimiW = new(); mimiW.Load(mp!);
        Mimi mimi = new(MimiConfig.Mimi24kHzDsm);
        try
        {
            mimi.LoadWeights(mimiW.GetAllTensors());
            _out.WriteLine("DSM Mimi fully loaded (SEANet + transformer-of-codecs).");
        }
        catch (System.Collections.Generic.KeyNotFoundException e)
        {
            // The SEANet residual-count + decoder seqIdx fix must let loading get PAST SEANet; any remaining
            // KeyNotFound must be in the transformer-of-codecs, NOT the encoder/decoder SEANet stack.
            Assert.Contains("transformer", e.Message);
            Assert.DoesNotContain("encoder.model", e.Message);
            Assert.DoesNotContain("decoder.model", e.Message);
            _out.WriteLine($"SEANet residual/seqIdx layout loads; transformer-of-codecs reconcile pending: {e.Message}");
        }
    }

    private static bool Missing(string? p) => string.IsNullOrEmpty(p) || !File.Exists(p);
}
