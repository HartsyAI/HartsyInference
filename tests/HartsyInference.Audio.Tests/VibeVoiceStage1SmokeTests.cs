using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Stage-1 smoke tests for the VibeVoice port — verifies the acoustic VAE,
/// semantic VAE, and diffusion head load all of their published weights and produce
/// finite outputs with the expected shapes. The Qwen2.5 LM and full inference pipeline
/// are deferred to Stage 2 (needs the dotLLM autoregressive transformer dependency).
///
/// <para>These tests skip cleanly when the 5.4 GB <c>vibevoice/VibeVoice-1.5B</c> shards
/// are not in the HartsyInference cache. With the cache present they:
/// <list type="bullet">
///   <item>Load all 1204 published tensors via <see cref="SafeTensorsShardLoader"/>.</item>
///   <item>Confirm <see cref="VibeVoiceConfig.V15B"/> matches the HF <c>config.json</c>.</item>
///   <item>Encode 1 s of synthetic 24 kHz audio to a <c>[1, 8, 64]</c> acoustic latent
///         (8 frames at 7.5 Hz × 64-d).</item>
///   <item>Decode a <c>[1, 8, 64]</c> latent back to <c>[1, 1, 25600]</c> audio.</item>
///   <item>Round-trip encode → sample → decode and confirm the output is non-pathological.</item>
///   <item>Encode through the 128-d semantic VAE.</item>
///   <item>Run a single diffusion-head v-prediction step.</item>
///   <item>Forward through both <c>SpeechConnector</c>s.</item>
/// </list></para></summary>
[Trait("Category", "Integration")]
public sealed class VibeVoiceStage1SmokeTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly SafeTensorsShardLoader? _loader;
    private readonly bool _cacheReady;

    public VibeVoiceStage1SmokeTests(ITestOutputHelper output)
    {
        _out = output;
        string repoDir = AudioModelCache.GetRepoDirectory("vibevoice/VibeVoice-1.5B");
        string shard1 = Path.Combine(repoDir, "model-00001-of-00003.safetensors");
        _cacheReady = File.Exists(shard1)
            && File.Exists(Path.Combine(repoDir, "model-00002-of-00003.safetensors"))
            && File.Exists(Path.Combine(repoDir, "model-00003-of-00003.safetensors"));
        if (_cacheReady)
        {
            _loader = new SafeTensorsShardLoader();
            _loader.LoadDirectory(repoDir);
        }
    }

    public void Dispose() => _loader?.Dispose();

    private bool Skip()
    {
        if (!_cacheReady) _out.WriteLine("VibeVoice-1.5B cache missing (expected at ~/.cache/hartsyinference/models/vibevoice--VibeVoice-1.5B/) — skipping.");
        return !_cacheReady;
    }

    [Fact]
    public void Config_V15B_MatchesHfDefaults()
    {
        VibeVoiceConfig cfg = VibeVoiceConfig.V15B;
        Assert.Equal("vibevoice", cfg.ModelType);
        Assert.Equal(64, cfg.AcousticVaeDim);
        Assert.Equal(128, cfg.SemanticVaeDim);
        Assert.NotNull(cfg.SemanticTokenizer);
        Assert.Equal(1_536, cfg.Decoder.HiddenSize);
        Assert.Equal(28, cfg.Decoder.NumHiddenLayers);
        Assert.Equal(12, cfg.Decoder.NumAttentionHeads);
        Assert.Equal(2, cfg.Decoder.NumKeyValueHeads);
        Assert.Equal(151_936, cfg.Decoder.VocabSize);
        Assert.Equal(65_536, cfg.Decoder.MaxPositionEmbeddings);
        Assert.True(cfg.Decoder.TieWordEmbeddings);
        Assert.Equal(4, cfg.DiffusionHead.HeadLayers);
        Assert.Equal(3.0f, cfg.DiffusionHead.HeadFfnRatio);
        Assert.Equal("v_prediction", cfg.DiffusionHead.PredictionType);
        Assert.Equal("cosine", cfg.DiffusionHead.DdpmBetaSchedule);
        Assert.Equal(20, cfg.DiffusionHead.DdpmNumInferenceSteps);
    }

    [Fact]
    public void ShardLoader_LoadsAll1204Tensors()
    {
        if (Skip()) return;
        Assert.NotNull(_loader);
        int count = _loader.TensorNames.Count;
        _out.WriteLine($"Loaded {count} tensors across {_loader.ShardCount} shards.");
        Assert.Equal(1_204, count);
        // Spot-check a tensor from each major module.
        Assert.Contains("model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight", _loader.TensorNames);
        Assert.Contains("model.semantic_tokenizer.encoder.head.conv.conv.weight", _loader.TensorNames);
        Assert.Contains("model.prediction_head.layers.0.ffn.gate_proj.weight", _loader.TensorNames);
        Assert.Contains("model.acoustic_connector.fc1.weight", _loader.TensorNames);
        Assert.Contains("model.semantic_connector.fc2.weight", _loader.TensorNames);
        Assert.Contains("model.speech_scaling_factor", _loader.TensorNames);
        Assert.Contains("model.speech_bias_factor", _loader.TensorNames);
    }

    [Fact]
    public unsafe void SpeechConnector_LoadsAndForwards()
    {
        if (Skip()) return;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> w = _loader!.GetAllTensors();

        // Acoustic connector: 64 → 1536
        SpeechConnector ac = new(inputDim: 64, outputDim: 1_536);
        ac.LoadWeights(w, "model.acoustic_connector");

        int n = 5;
        Tensor latent = MakeRandomCL(1, n, 64, seed: 1);
        Tensor ah = ac.Forward(backend, latent, batch: 1, seqLen: n);
        Assert.Equal(new TensorShape(1, n, 1_536), ah.Shape);
        AssertAllFinite(ah, "acoustic_connector(latent)");
        latent.Dispose();
        ah.Dispose();

        // Semantic connector: 128 → 1536
        SpeechConnector sc = new(inputDim: 128, outputDim: 1_536);
        sc.LoadWeights(w, "model.semantic_connector");
        Tensor sem = MakeRandomCL(1, n, 128, seed: 2);
        Tensor sh = sc.Forward(backend, sem, batch: 1, seqLen: n);
        Assert.Equal(new TensorShape(1, n, 1_536), sh.Shape);
        AssertAllFinite(sh, "semantic_connector(sem)");
        _out.WriteLine("SpeechConnector: acoustic + semantic forward shapes verified.");
        sem.Dispose();
        sh.Dispose();
    }

    [Fact]
    public unsafe void AcousticVae_EncodeDecodeRoundTrip()
    {
        if (Skip()) return;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> w = _loader!.GetAllTensors();

        VibeVoiceTokenizerConfig tokCfg = VibeVoiceConfig.V15B.AcousticTokenizer;
        VibeVoiceAcousticTokenizerModel acoustic = new(tokCfg, "model.acoustic_tokenizer");
        acoustic.LoadWeights(w);
        _out.WriteLine($"Acoustic VAE weights loaded ({acoustic.EnumerateWeights().Count()} tensors).");

        // 1 second of 24 kHz audio padded to a multiple of 3200 (= 7.5 frames). Round up
        // to the next multiple to avoid stride-rounding boundary effects.
        int sr = 24_000;
        int frames = 8;     // 8 latent frames ≈ 1.07 s
        int tPcm = frames * 3_200;     // 25 600 samples
        Tensor pcm = new(new TensorShape(1, 1, tPcm), DType.F32);
        Random rng = new(42);
        float* pp = (float*)pcm.DataPointer;
        for (int i = 0; i < tPcm; i++)
            pp[i] = 0.1f * MathF.Sin(2 * MathF.PI * 220f * i / sr) + 0.02f * (float)(rng.NextDouble() * 2 - 1);

        // Encode → [1, T_latent, 64] channels-last mean.
        Tensor mean = acoustic.Encode(backend, pcm, batch: 1, tPcm: tPcm);
        Assert.Equal(new TensorShape(1, frames, 64), mean.Shape);
        AssertAllFinite(mean, "acoustic_vae.encode(audio).mean");
        _out.WriteLine($"Encode: {tPcm} pcm samples → latent {mean.Shape}.");

        // Decode the mean back (skip sampling for determinism).
        Tensor recon = acoustic.Decode(backend, mean, batch: 1);
        Assert.Equal(3, recon.Shape.Rank);
        Assert.Equal(1, (int)recon.Shape[0]);
        Assert.Equal(1, (int)recon.Shape[1]);
        _out.WriteLine($"Decode: latent → {recon.Shape} pcm samples.");
        AssertAllFinite(recon, "acoustic_vae.decode(mean)");
        // Decoded length should be very close to T_latent * 3200; allow a small ± few
        // samples wobble from the SConv1d/SConvTranspose1d causal-padding accounting.
        int tOut = (int)recon.Shape[2];
        Assert.InRange(tOut, tPcm - 64, tPcm + 64);

        // Reconstruction shouldn't be silent or clipping — a sane VAE will preserve some energy.
        double inSq = 0d, outSq = 0d;
        float* rp = (float*)recon.DataPointer;
        for (int i = 0; i < tPcm; i++) inSq += (double)pp[i] * pp[i];
        for (int i = 0; i < tOut; i++) outSq += (double)rp[i] * rp[i];
        double inRms = Math.Sqrt(inSq / tPcm);
        double outRms = Math.Sqrt(outSq / tOut);
        _out.WriteLine($"Reconstruction RMS: in={inRms:F4}, out={outRms:F4} (ratio {outRms / inRms:F3}).");
        // Even an untrained-on-this-clip output should not be dead silence or hugely clipping —
        // we're checking sanity here, not perceptual quality (no Python reference yet).
        Assert.True(outRms > 1e-5, $"reconstruction is essentially silent (RMS {outRms}).");
        Assert.True(outRms < 10.0, $"reconstruction is wildly clipping (RMS {outRms}).");

        pcm.Dispose();
        mean.Dispose();
        recon.Dispose();
    }

    [Fact]
    public unsafe void SemanticVae_Encode_Produces128dLatent()
    {
        if (Skip()) return;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> w = _loader!.GetAllTensors();

        VibeVoiceTokenizerConfig semCfg = VibeVoiceConfig.V15B.SemanticTokenizer!;
        VibeVoiceSemanticTokenizerModel semantic = new(semCfg, "model.semantic_tokenizer");
        semantic.LoadWeights(w);
        _out.WriteLine($"Semantic VAE weights loaded ({semantic.EnumerateWeights().Count()} tensors).");

        int frames = 4;
        int tPcm = frames * 3_200;
        Tensor pcm = new(new TensorShape(1, 1, tPcm), DType.F32);
        float* pp = (float*)pcm.DataPointer;
        for (int i = 0; i < tPcm; i++) pp[i] = 0.05f * MathF.Sin(2 * MathF.PI * 330f * i / 24_000f);

        Tensor mean = semantic.Encode(backend, pcm, batch: 1, tPcm: tPcm);
        Assert.Equal(new TensorShape(1, frames, 128), mean.Shape);
        AssertAllFinite(mean, "semantic_vae.encode(audio).mean");
        _out.WriteLine($"Semantic encode: {tPcm} pcm samples → {mean.Shape}.");

        pcm.Dispose();
        mean.Dispose();
    }

    [Fact]
    public unsafe void DiffusionHead_SingleStepForward_ProducesVTarget()
    {
        if (Skip()) return;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> w = _loader!.GetAllTensors();

        VibeVoiceDiffusionHeadConfig dhCfg = VibeVoiceConfig.V15B.DiffusionHead;
        VibeVoiceDiffusionHead head = new(dhCfg, "model.prediction_head");
        head.LoadWeights(w);
        _out.WriteLine($"Diffusion head weights loaded ({head.EnumerateWeights().Count()} tensors).");

        // The head denoises a [1, N, 64] noisy latent given an [1, N, 1536] LM condition
        // and an N-vector of timesteps. N=2 covers a CFG batch (cond + uncond).
        int n = 2;
        Tensor noisy = MakeRandomCL(1, n, dhCfg.LatentSize, seed: 7);
        Tensor cond = MakeRandomCL(1, n, dhCfg.HiddenSize, seed: 8);
        float[] timesteps = [999f, 999f];     // start of the 1000-step DDPM schedule

        Tensor v = head.Forward(backend, noisy, timesteps, cond);
        Assert.Equal(new TensorShape(1, n, dhCfg.LatentSize), v.Shape);
        AssertAllFinite(v, "diffusion_head.forward(noisy, t, cond)");
        _out.WriteLine($"Diffusion head: 1 step → v-target {v.Shape}.");

        noisy.Dispose();
        cond.Dispose();
        v.Dispose();
    }

    [Fact]
    public unsafe void Stage1FullLoad_AllSubmodulesAcceptPublishedWeights()
    {
        if (Skip()) return;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> w = _loader!.GetAllTensors();

        VibeVoiceConfig cfg = VibeVoiceConfig.V15B;
        VibeVoiceAcousticTokenizerModel acoustic = new(cfg.AcousticTokenizer, "model.acoustic_tokenizer");
        VibeVoiceSemanticTokenizerModel semantic = new(cfg.SemanticTokenizer!, "model.semantic_tokenizer");
        SpeechConnector acConn = new(cfg.AcousticVaeDim, cfg.Decoder.HiddenSize);
        SpeechConnector seConn = new(cfg.SemanticVaeDim, cfg.Decoder.HiddenSize);
        VibeVoiceDiffusionHead head = new(cfg.DiffusionHead, "model.prediction_head");

        acoustic.LoadWeights(w);
        semantic.LoadWeights(w);
        acConn.LoadWeights(w, "model.acoustic_connector");
        seConn.LoadWeights(w, "model.semantic_connector");
        head.LoadWeights(w);

        int totalWeights = acoustic.EnumerateWeights().Count()
            + semantic.EnumerateWeights().Count()
            + acConn.EnumerateWeights().Count()
            + seConn.EnumerateWeights().Count()
            + head.EnumerateWeights().Count();
        _out.WriteLine($"Stage-1 modules bound {totalWeights} tensors total.");
        // 552 (acoustic) + 276 (semantic) + 5+5 (connectors) + 26 (head) = 864 of the 1204
        // tensors are consumed here. The remaining ~340 belong to the Qwen2.5 LM, which
        // Stage 2 will pick up.
        Assert.True(totalWeights >= 860, $"Bound only {totalWeights} weights — expected ≥860 for Stage 1.");
        Assert.True(totalWeights <= 880, $"Bound {totalWeights} weights — more than the Stage-1 budget.");
    }

    /// <summary>Streaming-equivalence: feeding audio to the VAE encoder one 3200-sample
    /// frame at a time through a persistent <see cref="VibeVoiceTokenizerStreamingCache"/>
    /// must reproduce the full-sequence (non-streaming) encode. This is the exact per-frame
    /// path the AR loop's semantic re-encode depends on — if it drifts, the LM's next-step
    /// conditioning drifts and it never emits speech_end (the over-generation bug). With
    /// input length an exact multiple of the 3200 hop, every conv stage adds zero
    /// right-padding, so the two paths are purely causal and must match to FP tolerance.</summary>
    [Fact]
    public unsafe void StreamingEncode_MatchesFullNonStreamingEncode()
    {
        if (Skip()) return;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> w = _loader!.GetAllTensors();

        VibeVoiceConfig cfg = VibeVoiceConfig.V15B;
        VibeVoiceSemanticTokenizerModel semantic = new(cfg.SemanticTokenizer!, "model.semantic_tokenizer");
        semantic.LoadWeights(w);

        const int frames = 4;
        const int hop = 3_200;
        int tPcm = frames * hop;
        int dim = cfg.SemanticVaeDim;     // 128

        // Deterministic, non-silent PCM.
        Tensor full = new(new TensorShape(1, 1, tPcm), DType.F32);
        float* fp = (float*)full.DataPointer;
        Random rng = new(1234);
        for (int i = 0; i < tPcm; i++)
            fp[i] = 0.08f * MathF.Sin(2 * MathF.PI * 180f * i / 24_000f) + 0.01f * (float)(rng.NextDouble() * 2 - 1);

        // Non-streaming: whole sequence at once → [1, frames, 128].
        Tensor nonStream = semantic.Encode(backend, full, batch: 1, tPcm: tPcm);
        Assert.Equal(new TensorShape(1, frames, dim), nonStream.Shape);

        // Streaming: one 3200-sample chunk at a time through a persistent cache.
        using VibeVoiceTokenizerStreamingCache cache = new();
        int[] sampleIndices = [0];
        float[] streamed = new float[frames * dim];
        int frameCursor = 0;
        for (int c = 0; c < frames; c++)
        {
            Tensor chunk = new(new TensorShape(1, 1, hop), DType.F32);
            Buffer.MemoryCopy(fp + (long)c * hop, (void*)chunk.DataPointer, hop * 4, hop * 4);
            Tensor outc = semantic.Encode(backend, chunk, batch: 1, tPcm: hop, cache, sampleIndices);
            int m = (int)outc.Shape[1];
            float* op = (float*)outc.DataPointer;
            for (int f = 0; f < m && frameCursor < frames; f++, frameCursor++)
                for (int k = 0; k < dim; k++)
                    streamed[frameCursor * dim + k] = op[f * dim + k];
            chunk.Dispose();
            outc.Dispose();
        }
        Assert.Equal(frames, frameCursor);     // one latent frame per 3200-sample chunk

        float* np = (float*)nonStream.DataPointer;
        double maxAbs = 0, sumSq = 0;
        for (int i = 0; i < frames * dim; i++)
        {
            double d = Math.Abs(streamed[i] - np[i]);
            if (d > maxAbs) maxAbs = d;
            sumSq += d * d;
        }
        double rmsDiff = Math.Sqrt(sumSq / (frames * dim));
        _out.WriteLine($"Streaming-equivalence: max|Δ|={maxAbs:E3}, rmsΔ={rmsDiff:E3} over {frames}×{dim} latents.");
        Assert.True(maxAbs < 1e-2, $"streaming encode diverged from full encode (max|Δ|={maxAbs:E3}).");

        full.Dispose();
        nonStream.Dispose();
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private static unsafe Tensor MakeRandomCL(int batch, int seq, int dim, int seed)
    {
        Tensor t = new(new TensorShape(batch, seq, dim), DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static unsafe void AssertAllFinite(Tensor t, string ctx)
    {
        long n = t.ElementCount;
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            if (float.IsNaN(v) || float.IsInfinity(v))
                Assert.Fail($"[{ctx}] non-finite value {v} at index {i}.");
        }
    }
}
