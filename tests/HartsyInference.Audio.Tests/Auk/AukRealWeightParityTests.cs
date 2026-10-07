using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.QwenOmni;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Stage-by-stage real-weight parity of AuK / AuK-Flash against the upstream F32 reference dumped by
/// <c>tests/python-reference/auk_ref.py</c>. Each stage is fed the reference's own inputs so a defect shows up at the
/// stage that owns it; the last step chains the engine stages end to end with the same injected noise.</summary>
/// <remarks><c>HARTSY_AUK_REFDIR</c> names the dump root (one folder per case); <c>HARTSY_AUK_DIR</c>,
/// <c>HARTSY_AUK_FLASH_DIR</c> and <c>HARTSY_AUK_OMNI_DIR</c> override the checkpoint folders and
/// <c>HARTSY_AUK_BACKEND</c> (default cuda) picks the backend. <c>HARTSY_AUK_STRICT=1</c> runs CUDA without TF32, cuDNN
/// attention, cuDNN audio convs (TF32 engines, not governed by the TF32 knob) or reduced-precision GEMM, separating
/// precision drift from logic defects. Without the dumps the tests do nothing.</remarks>
[Trait("Category", "RealWeights")]
public sealed class AukRealWeightParityTests(ITestOutputHelper output)
{
    private readonly List<string> _failures = [];
    private string _lastLabel = "";

    private const string ModelsRoot = "/mnt/model-storage/Models/audio/tts";

    private static string? RefDir => Environment.GetEnvironmentVariable("HARTSY_AUK_REFDIR");

    private static string Setting(string name, string fallback) =>
        Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : fallback;

    [Fact]
    public void Clone_MatchesReference() => RunCase("clone");

    [Fact]
    public void InstructTts_MatchesReference() => RunCase("instruct");

    private void RunCase(string name)
    {
        if (RefDir is not { Length: > 0 } root) return;
        string dir = Path.Combine(root, name);
        string aukDir = Setting("HARTSY_AUK_DIR", Path.Combine(ModelsRoot, "tencent--AuK"));
        string flashDir = Setting("HARTSY_AUK_FLASH_DIR", Path.Combine(ModelsRoot, "tencent--AuK-Flash"));
        string omniDir = Setting("HARTSY_AUK_OMNI_DIR", Path.Combine(ModelsRoot, "Qwen--Qwen2.5-Omni-3B"));
        if (!RealWeightGate.Require(output.WriteLine, Path.Combine(dir, "meta.json"), Path.Combine(aukDir, "auk_base.safetensors"),
                Path.Combine(aukDir, "vae.safetensors"), Path.Combine(flashDir, "auk_flash.safetensors"),
                Path.Combine(omniDir, "tokenizer.json")))
        {
            return;
        }
        bool strict = Environment.GetEnvironmentVariable("HARTSY_AUK_STRICT") == "1";
        if (strict)
        {
            KnobStore.Set(EngineKnobs.NoTf32, true);
            KnobStore.Set(EngineKnobs.SdpaCudnn, false);
            KnobStore.Set(EngineKnobs.HighPrecisionGemm, true);
            KnobStore.Set(EngineKnobs.AudioConvCudnn, false);
            KnobStore.Set(EngineKnobs.SageAttn, false);
            KnobStore.Set(EngineKnobs.SdpaNoF16, true);
        }
        try
        {
            string kind = Setting("HARTSY_AUK_BACKEND", "cuda");
            if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? opened)) return;
            using IBackend backend = opened!;
            output.WriteLine($"[{name}] backend {backend.GetType().Name}{(strict ? " (strict precision)" : "")}");
            // The CPU backend is F32 throughout, so it is held to the strict tolerances too.
            RunCase(backend, name, dir, aukDir, flashDir, omniDir, strict || kind == "cpu");
            Assert.True(_failures.Count == 0, string.Join("; ", _failures));
        }
        finally
        {
            if (strict)
            {
                KnobStore.Clear(EngineKnobs.NoTf32);
                KnobStore.Clear(EngineKnobs.SdpaCudnn);
                KnobStore.Clear(EngineKnobs.HighPrecisionGemm);
                KnobStore.Clear(EngineKnobs.AudioConvCudnn);
                KnobStore.Clear(EngineKnobs.SageAttn);
                KnobStore.Clear(EngineKnobs.SdpaNoF16);
            }
        }
    }

    private void RunCase(IBackend backend, string name, string dir, string aukDir, string flashDir, string omniDir, bool strict)
    {
        // Strict runs keep reduced precision off and are held tighter; default runs keep TF32 GEMMs and F16 cuDNN
        // attention, whose drift through the 36-layer thinker stays far below upstream's own production bf16 thinker
        // (final-layer corr 0.975 against the F32 reference on the clone case).
        double thinkerMin = strict ? 0.99999 : 0.9995;
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        string instruction = meta.RootElement.GetProperty("instruction").GetString()!;
        bool hasAudio = meta.RootElement.GetProperty("audio").ValueKind == JsonValueKind.String;
        int frames = meta.RootElement.GetProperty("frames").GetInt32();
        double seconds = meta.RootElement.GetProperty("seconds").GetDouble();
        Assert.Equal(frames, AukDuration.Frames(seconds, 24_000, 480));

        List<IDisposable> owned = [];
        try
        {
            Dictionary<string, Tensor> omni = LoadShards(omniDir, owned, "thinker.model.", "thinker.audio_tower.");
            Dictionary<string, Tensor> vaeWeights = Load(Path.Combine(aukDir, "vae.safetensors"), owned);
            QwenOmniConfig omniCfg = QwenOmniConfig.Default;
            GgufTokenizer tokenizer;
            using (FileStream stream = File.OpenRead(Path.Combine(omniDir, "tokenizer.json")))
            {
                tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
            }

            // Stage 0: prompt token ids.
            int[] refIds = ReadI32(dir, "input_ids");
            int audioTokens = refIds.Count(id => id == omniCfg.AudioTokenId);
            int[] ids = AukPrompt.BuildIds(instruction, audioTokens, tokenizer.EncodeOrdinary, omniCfg);
            Assert.Equal(refIds, ids);
            output.WriteLine($"[{name}] ids: {ids.Length} tokens, {audioTokens} audio, identical");

            // Stages 1-2: mel and audio tower.
            Tensor? refTower = null;
            Tensor? chainTower = null;
            if (hasAudio)
            {
                QwenOmniProcessor processor = new(omniCfg);
                float[] pcm16k = Read(dir, "pcm16k").Data;
                Assert.Equal(audioTokens, processor.AudioTokensForSamples(pcm16k.Length));
                using Tensor mel = new(new TensorShape(omniCfg.NumMelBins, processor.PaddedFrames), DType.F32);
                int melFrames = processor.ComputeMel(pcm16k, mel);
                (float[] refMel, long[] melShape) = Read(dir, "mel");
                Assert.Equal(melShape[1], melFrames);
                float[] ourMel = Columns(mel.AsSpan<float>(), omniCfg.NumMelBins, processor.PaddedFrames, melFrames);
                Report($"[{name}] mel", ourMel, refMel);
                using Tensor chainMel = new(mel.Shape, DType.F32);
                mel.AsSpan<float>().CopyTo(chainMel.AsSpan<float>());

                // Feed the reference mel to the tower so its error is the tower's own.
                Span<float> melSpan = mel.AsSpan<float>();
                for (int b = 0; b < omniCfg.NumMelBins; b++)
                {
                    refMel.AsSpan(b * melFrames, melFrames).CopyTo(melSpan.Slice(b * processor.PaddedFrames, melFrames));
                }
                QwenOmniAudioEncoder tower = new(omniCfg);
                owned.Add(tower);
                tower.LoadWeights(omni);
                refTower = ToTensor(Read(dir, "audio_tower"));
                owned.Add(refTower);
                backend.PreloadWeights([.. tower.EnumerateWeights()]);
                try
                {
                    using Tensor rows = tower.Forward(backend, mel, melFrames);
                    Expect(Report($"[{name}] audio tower", rows.AsSpan<float>(), refTower.AsSpan<float>()), 0.9999);
                    chainTower = tower.Forward(backend, chainMel, melFrames);
                    owned.Add(chainTower);
                }
                finally
                {
                    backend.FreeWeights([.. tower.EnumerateWeights()]);
                }
            }

            // Stage 3: thinker hidden states, with the reference tower rows spliced in.
            Qwen2Config lmCfg = Qwen2Config.Qwen25Omni_3B_Thinker;
            Qwen2Model lm = new(lmCfg);
            owned.Add(lm);
            lm.LoadWeightsNoLmHead(omni, "thinker.model");
            List<float[]> ourHidden;
            List<float[]> chainHidden;
            backend.PreloadWeights([.. lm.EnumerateWeights()]);
            try
            {
                ourHidden = RunThinker(backend, lm, lmCfg, omniCfg, ids, refTower);
                chainHidden = RunThinker(backend, lm, lmCfg, omniCfg, ids, chainTower);
            }
            finally
            {
                backend.FreeWeights([.. lm.EnumerateWeights()]);
            }
            Assert.Equal(lmCfg.NumHiddenLayers, ourHidden.Count);
            for (int i = 0; i < ourHidden.Count; i++)
            {
                Expect(Report($"[{name}] hidden {i:00}", ourHidden[i], Read(dir, $"hidden_{i:00}").Data), thinkerMin);
            }

            foreach (string variant in new[] { "flash", "base" })
            {
                bool flash = variant == "flash";
                string vdir = Path.Combine(dir, variant);
                string checkpoint = flash ? Path.Combine(flashDir, "auk_flash.safetensors") : Path.Combine(aukDir, "auk_base.safetensors");
                Dictionary<string, Tensor> auk = Load(checkpoint, owned);
                RunVariant(backend, name, variant, flash, dir, vdir, auk, vaeWeights, hasAudio, frames, lmCfg.HiddenSize,
                    chainHidden, strict);
            }
        }
        finally
        {
            for (int i = owned.Count - 1; i >= 0; i--) owned[i].Dispose();
        }
    }

    private void RunVariant(IBackend backend, string name, string variant, bool flash, string dir, string vdir,
        Dictionary<string, Tensor> auk, Dictionary<string, Tensor> vaeWeights, bool hasAudio, int frames, int hidden,
        List<float[]> chainHidden, bool strict)
    {
        string tag = $"[{name}/{variant}]";
        AukConfig cfg = AukConfig.Default;

        // Stage 4: layer fusion over the reference hidden states.
        using AukLayerFusion fusion = AukLayerFusion.FromCheckpoint(backend, auk, hidden);
        Report($"{tag} fusion coeffs", fusion.Coefficients.ToArray().Select(c => (float)c).ToArray(), Read(vdir, "coeffs").Data);
        int t = (int)Read(dir, "hidden_00").Shape[0];
        fusion.Begin(t);
        for (int i = 0; i < cfg.FusionLayers - 1; i++)
        {
            using Tensor h = ToTensor(Read(dir, $"hidden_{i:00}"));
            fusion.Accumulate(i, h);
        }
        using Tensor last = ToTensor(Read(dir, $"hidden_{cfg.FusionLayers - 1:00}"));
        using (Tensor fused = fusion.Complete(last))
        {
            Expect(Report($"{tag} fused text", fused.AsSpan<float>(), Read(vdir, "fused").Data), 0.99999);
        }

        using AukDit dit = new(cfg);
        dit.LoadWeights(auk);
        Report($"{tag} inv_freq", dit.InvFreq.ToArray(), Read(vdir, "inv_freq").Data);
        Assert.Equal(Read(vdir, "inv_freq").Data, dit.InvFreq.ToArray());

        AukSchedule schedule = flash ? AukSchedule.Flash() : AukSchedule.Base();
        float[] refGrid = Read(vdir, "t_grid").Data;
        Report($"{tag} t grid", schedule.Timesteps, refGrid);

        // Stage 5: VAE encode with the reference eps.
        using AukVaeStats stats = AukVaeStats.Load(vaeWeights);
        Tensor? refLatent = null;
        Tensor? chainRef = null;
        if (hasAudio)
        {
            using AukVaeEncoder encoder = new(new AukVaeConfig(), stats);
            encoder.LoadWeights(vaeWeights);
            using Tensor pcm = ToTensor(Read(dir, "pcm24k"), 1, 1, -1);
            using Tensor eps = ToTensor(Read(dir, "vae_eps"));
            chainRef = Stage(backend, encoder.EnumerateWeights(), () => encoder.Encode(backend, pcm, eps));
            refLatent = ToTensor(Read(dir, "ref_latent"));
            Expect(Report($"{tag} ref latent", chainRef.AsSpan<float>(), refLatent.AsSpan<float>()), 0.9999);
        }

        using AukVae vae = new(new AukVaeConfig());
        vae.LoadWeights(vaeWeights);
        try
        {
            // Stages 6-7: DiT velocity at every step from the reference state, then the engine's own loop.
            using Tensor text = ToTensor(Read(vdir, "fused"));
            using Tensor noise = ToTensor(Read(dir, "noise"));
            Assert.Equal(frames, (int)noise.Shape[1]);
            Tensor chainLatent;
            backend.PreloadWeights([.. dit.EnumerateWeights()]);
            try
            {
                using (Tensor condText = dit.ProjectText(backend, text))
                using (Tensor? uncondText = schedule.UsesCfg ? dit.ProjectText(backend, text, drop: true) : null)
                {
                    for (int step = 0; step < schedule.Steps; step++)
                    {
                        using Tensor xRef = ToTensor(step == 0 ? Read(dir, "noise") : Read(vdir, $"x_{step:00}"));
                        using Tensor vCond = dit.Forward(backend, xRef, refLatent, condText, refGrid[step]);
                        if (uncondText is not null)
                        {
                            using Tensor vUncond = dit.Forward(backend, xRef, refLatent, uncondText, refGrid[step], dropAudioCond: true);
                            if (step == 0)
                            {
                                Expect(Report($"{tag} v_cond step 0", vCond.AsSpan<float>(), Read(vdir, "v_cond_00").Data), 0.9999);
                                Expect(Report($"{tag} v_uncond step 0", vUncond.AsSpan<float>(), Read(vdir, "v_uncond_00").Data), 0.9999);
                            }
                            AukSchedule.CfgCombine(vCond.AsSpan<float>(), vUncond.AsSpan<float>(), schedule.Cfg);
                        }
                        Expect(Report($"{tag} v step {step:00}", vCond.AsSpan<float>(), Read(vdir, $"v_{step:00}").Data), 0.9999);
                    }
                }
                using (Tensor loop = Sample(backend, dit, schedule, text, refLatent, noise))
                {
                    Expect(Report($"{tag} final latent (engine loop)", loop.AsSpan<float>(), Read(vdir, $"x_{schedule.Steps:00}").Data), 0.999);
                }

                // Stage 9 input: the chained engine conditioning (own mel, tower, thinker, fusion and VAE encode).
                using Tensor chainText = Fuse(backend, auk, chainHidden, hidden);
                Expect(Report($"{tag} chained fused text", chainText.AsSpan<float>(), text.AsSpan<float>()), 0.9999);
                chainLatent = Sample(backend, dit, schedule, chainText, chainRef, noise);
            }
            finally
            {
                backend.FreeWeights([.. dit.EnumerateWeights()]);
            }

            float[] refPcm = Read(vdir, "pcm_out").Data;
            using (chainLatent)
            using (Tensor latent = ToTensor(Read(vdir, $"x_{schedule.Steps:00}")))
            {
                Expect(Report($"{tag} chained final latent", chainLatent.AsSpan<float>(), latent.AsSpan<float>()), 0.999);
                backend.PreloadWeights([.. vae.EnumerateWeights()]);
                try
                {
                    // Stage 8: VAE decode of the reference final latent.
                    float[] pcmOut = DecodePcm(backend, vae, stats, latent);
                    Assert.Equal(refPcm.Length, pcmOut.Length);
                    Expect(Report($"{tag} VAE decode", pcmOut, refPcm), 0.999);

                    // Stage 9: every stage chained in the engine, only the noise taken from the reference.
                    float[] chainPcm = DecodePcm(backend, vae, stats, chainLatent);
                    // Flash text-only amplifies its first-step error through four large steps (measured 0.99934
                    // strict, 0.995 default, with word-identical transcripts), so it sets these floors.
                    Expect(Report($"{tag} chained end to end pcm", chainPcm, refPcm), strict ? 0.998 : 0.99);
                    WavFile.WriteMono16(Path.Combine(vdir, $"engine_{backend.GetType().Name}.wav"), chainPcm, 24_000);
                }
                finally
                {
                    backend.FreeWeights([.. vae.EnumerateWeights()]);
                }
            }
        }
        finally
        {
            refLatent?.Dispose();
            chainRef?.Dispose();
        }
    }

    /// <summary>The engine's Euler loop (with CFG on base) from <paramref name="noise"/>; the caller owns the result.</summary>
    private static Tensor Sample(IBackend backend, AukDit dit, AukSchedule schedule, Tensor text, Tensor? refLatent, Tensor noise)
    {
        using Tensor condText = dit.ProjectText(backend, text);
        using Tensor? uncondText = schedule.UsesCfg ? dit.ProjectText(backend, text, drop: true) : null;
        Tensor x = new(noise.Shape, DType.F32);
        noise.AsSpan<float>().CopyTo(x.AsSpan<float>());
        for (int step = 0; step < schedule.Steps; step++)
        {
            using Tensor vCond = dit.Forward(backend, x, refLatent, condText, schedule.Timesteps[step]);
            if (uncondText is not null)
            {
                using Tensor vUncond = dit.Forward(backend, x, refLatent, uncondText, schedule.Timesteps[step], dropAudioCond: true);
                AukSchedule.CfgCombine(vCond.AsSpan<float>(), vUncond.AsSpan<float>(), schedule.Cfg);
            }
            AukSchedule.EulerStep(x.AsSpan<float>(), vCond.AsSpan<float>(), schedule.Deltas[step]);
        }
        return x;
    }

    private static float[] DecodePcm(IBackend backend, AukVae vae, AukVaeStats stats, Tensor latent)
    {
        using Tensor raw = stats.Denormalize(backend, latent);
        using Tensor decoded = vae.DecodeTimeMajor(backend, raw);
        return decoded.AsSpan<float>().ToArray();
    }

    /// <summary>Fuses host-side hidden states <c>[t, hidden]</c> (the last one final-normed) with the checkpoint's layer weights.</summary>
    private static Tensor Fuse(IBackend backend, IReadOnlyDictionary<string, Tensor> auk, List<float[]> layers, int hidden)
    {
        using AukLayerFusion fusion = AukLayerFusion.FromCheckpoint(backend, auk, hidden);
        int t = layers[0].Length / hidden;
        fusion.Begin(t);
        for (int i = 0; i < layers.Count - 1; i++)
        {
            using Tensor h = ToTensor((layers[i], [t, hidden]));
            fusion.Accumulate(i, h);
        }
        using Tensor last = ToTensor((layers[^1], [t, hidden]));
        return fusion.Complete(last);
    }

    /// <summary>The thinker's 36 hidden states (HF <c>output_hidden_states[1:]</c>) with <paramref name="audioRows"/>
    /// spliced at the audio placeholders.</summary>
    private static List<float[]> RunThinker(IBackend backend, Qwen2Model lm, Qwen2Config lmCfg, QwenOmniConfig omniCfg, int[] ids, Tensor? audioRows)
    {
        int t = ids.Length;
        List<float[]> layers = [];
        using Tensor embeds = new(new TensorShape(1, t, lmCfg.HiddenSize), DType.F32);
        lm.EmbedLookup(embeds, ids, 1, t);
        if (audioRows is not null)
        {
            QwenOmniProcessor.SpliceAudioRows(ids, omniCfg.AudioTokenId, embeds.AsSpan<float>(), audioRows.AsSpan<float>(), lmCfg.HiddenSize);
        }
        using IKvCache cache = lm.CreateDecodeCache(t);
        using Tensor finalNormed = lm.ForwardEmbeds(backend, embeds, 1, t, 0, cache, layerTap: (_, h) => layers.Add(h.AsSpan<float>().ToArray()));
        layers[^1] = finalNormed.AsSpan<float>().ToArray();
        return layers;
    }

    /// <summary>Records a stage whose correlation falls below <paramref name="minCorr"/>, so one run reports every stage.</summary>
    private void Expect(double corr, double minCorr)
    {
        if (!(corr > minCorr)) _failures.Add($"{_lastLabel} corr {corr:F8} <= {minCorr}");
    }

    private static T Stage<T>(IBackend backend, IEnumerable<Tensor> weights, Func<T> run)
    {
        List<Tensor> held = [.. weights];
        backend.PreloadWeights(held);
        try
        {
            return run();
        }
        finally
        {
            backend.FreeWeights(held);
        }
    }

    /// <summary>Logs corr / relative L2 / max-abs of <paramref name="ours"/> against <paramref name="expected"/> and
    /// returns the correlation.</summary>
    private double Report(string label, ReadOnlySpan<float> ours, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, ours.Length);
        double dot = 0, oo = 0, ee = 0, diff = 0, maxAbs = 0, so = 0, se = 0;
        for (int i = 0; i < ours.Length; i++)
        {
            so += ours[i];
            se += expected[i];
        }
        double mo = so / ours.Length, me = se / ours.Length;
        for (int i = 0; i < ours.Length; i++)
        {
            double a = ours[i] - mo, b = expected[i] - me;
            dot += a * b;
            oo += a * a;
            ee += b * b;
            double d = ours[i] - expected[i];
            diff += d * d;
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
        }
        double norm = 0;
        foreach (float v in expected) norm += (double)v * v;
        double corr = oo == 0 && ee == 0 ? 1 : dot / Math.Sqrt(oo * ee);
        double rel = norm == 0 ? Math.Sqrt(diff) : Math.Sqrt(diff / norm);
        _lastLabel = label;
        output.WriteLine($"{label,-40} corr={corr:F8} relL2={rel:E2} maxAbs={maxAbs:E2}");
        return corr;
    }

    private static float[] Columns(ReadOnlySpan<float> rows, int bins, int stride, int frames)
    {
        float[] result = new float[bins * frames];
        for (int b = 0; b < bins; b++) rows.Slice(b * stride, frames).CopyTo(result.AsSpan(b * frames, frames));
        return result;
    }

    private static (float[] Data, long[] Shape) Read(string dir, string name)
    {
        long[] shape = ReadShape(dir, name);
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        float[] data = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return (data, shape);
    }

    private static int[] ReadI32(string dir, string name)
    {
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        int[] data = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return data;
    }

    private static long[] ReadShape(string dir, string name)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, name + ".json")));
        return doc.RootElement.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
    }

    private static Tensor ToTensor((float[] Data, long[] Shape) value, params long[] shape)
    {
        long[] dims = shape.Length > 0 ? shape.Select(d => d < 0 ? value.Data.Length : d).ToArray() : value.Shape;
        if (dims.Length == 2) dims = [1, .. dims];
        Tensor result = new(new TensorShape(dims), DType.F32);
        value.Data.CopyTo(result.AsSpan<float>());
        return result;
    }

    private static Dictionary<string, Tensor> Load(string path, List<IDisposable> owned)
    {
        SafeTensorsLoader loader = new();
        owned.Add(loader);
        loader.Load(path);
        return new Dictionary<string, Tensor>(loader.GetAllTensors());
    }

    private static Dictionary<string, Tensor> LoadShards(string dir, List<IDisposable> owned, params string[] prefixes)
    {
        Dictionary<string, Tensor> result = [];
        foreach (string shard in Directory.GetFiles(dir, "model-*-of-*.safetensors").Order())
        {
            SafeTensorsLoader loader = new();
            owned.Add(loader);
            loader.Load(shard);
            foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
            {
                if (prefixes.Any(p => kv.Key.StartsWith(p, StringComparison.Ordinal))) result[kv.Key] = kv.Value;
            }
        }
        return result;
    }
}
