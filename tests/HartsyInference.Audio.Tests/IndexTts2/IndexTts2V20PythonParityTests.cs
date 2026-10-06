using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Stage-by-stage numeric parity of the IndexTTS-2.0 C# pipeline against the reference PyTorch implementation.
/// The reference side is a safetensors dump written by the Python harness (every intermediate of
/// <c>infer_generator</c> for one fixed prompt: w2v-bert feature, quantized codes, reference mel, CAM++ style, prompt
/// condition, speaker latents, emotion vector, a sampled code sequence and everything derived from it, plus the S2Mel
/// estimator's inputs/outputs at three diffusion steps). Each C# stage is fed the reference's own input and its output
/// compared with the reference's, so one stage's error never hides or inflates another's. Gated on
/// <c>INDEXTTS2_PARITY_DIR</c> (holding <c>case1_b3.safetensors</c>, optionally <c>greedy1.safetensors</c> and the
/// original <c>alex_ref.wav</c> as <c>ref.wav</c>) plus the same real-weight environment as
/// <see cref="IndexTts2V20PipelineRealWeightTests"/>.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2V20PythonParityTests(ITestOutputHelper output)
{
    private readonly List<string> _failures = [];

    private sealed record Dump(Dictionary<string, (float[] Data, long[] Shape)> Items)
    {
        public float[] this[string name] => Items[name].Data;
        public long[] Shape(string name) => Items[name].Shape;
        public bool Has(string name) => Items.ContainsKey(name);
    }

    private static Dump LoadDump(string path)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, (float[], long[])> items = [];
        foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
        {
            Tensor t = kv.Value;
            float[] data = new float[t.ElementCount];
            t.AsSpan<float>().CopyTo(data);
            long[] shape = new long[t.Shape.Rank];
            for (int i = 0; i < shape.Length; i++) shape[i] = t.Shape[i];
            items[kv.Key] = (data, shape);
        }
        return new Dump(items);
    }

    private static Tensor ToTensor(float[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        data.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }

    private static float[] ToArray(Tensor t)
    {
        float[] a = new float[t.ElementCount];
        t.AsSpan<float>().CopyTo(a);
        return a;
    }

    private (double RelRms, double MaxAbs, double Cos) Compare(string stage, float[] got, float[] want, double maxRelRms)
    {
        Assert.Equal(want.Length, got.Length);
        double num = 0, den = 0, maxAbs = 0, dot = 0, ng = 0, nw = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double d = got[i] - want[i];
            num += d * d;
            den += (double)want[i] * want[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
            dot += (double)got[i] * want[i];
            ng += (double)got[i] * got[i];
            nw += (double)want[i] * want[i];
        }
        double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
        double cos = dot / Math.Max(Math.Sqrt(ng * nw), 1e-30);
        output.WriteLine($"{stage,-34} n={want.Length,8}  relRMS={rel:E3}  maxAbs={maxAbs:E3}  cos={cos:F6}");
        if (!(rel <= maxRelRms)) _failures.Add($"{stage}: relative RMS error {rel:E3} exceeds {maxRelRms:E1} (maxAbs {maxAbs:E3}, cos {cos:F6}).");
        return (rel, maxAbs, cos);
    }

    [Fact]
    public async Task EveryStage_MatchesTheReferenceImplementation()
    {
        string? parityDir = Environment.GetEnvironmentVariable("INDEXTTS2_PARITY_DIR");
        string? v20Dir = Environment.GetEnvironmentVariable("INDEXTTS2_V20_DIR");
        string? codecPath = Environment.GetEnvironmentVariable("INDEXTTS2_MASKGCT_CODEC_PATH");
        string? w2vBertPath = Environment.GetEnvironmentVariable("INDEXTTS2_W2VBERT_SAFETENSORS_PATH");
        string? campplusPath = Environment.GetEnvironmentVariable("INDEXTTS2_CAMPPLUS_PATH");
        string? bigVganPath = Environment.GetEnvironmentVariable("INDEXTTS2_BIGVGAN_PT_PATH");
        string? refWavPath = Environment.GetEnvironmentVariable("INDEXTTS2_REF_WAV");
        if (string.IsNullOrEmpty(parityDir) || !File.Exists(Path.Combine(parityDir, "case1_b3.safetensors"))) { output.WriteLine("INDEXTTS2_PARITY_DIR missing — skip."); return; }
        if (string.IsNullOrEmpty(v20Dir) || !Directory.Exists(v20Dir)) { output.WriteLine("INDEXTTS2_V20_DIR missing — skip."); return; }
        foreach (string? p in new[] { codecPath, w2vBertPath, campplusPath, bigVganPath, refWavPath })
            if (string.IsNullOrEmpty(p) || !File.Exists(p)) { output.WriteLine("auxiliary checkpoint missing — skip."); return; }

        Dump d = LoadDump(Path.Combine(parityDir, "case1_b3.safetensors"));
        using CpuBackend backend = new();
        using IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
            Path.Combine(v20Dir, "bpe.model"), Path.Combine(v20Dir, "gpt.pth"), Path.Combine(v20Dir, "s2mel.pth"), codecPath!,
            w2vBertPath!, Path.Combine(v20Dir, "wav2vec2bert_stats.pt"), campplusPath!, bigVganPath!,
            feat1Path: Path.Combine(v20Dir, "feat1.pt"), feat2Path: Path.Combine(v20Dir, "feat2.pt"), cfg: IndexTts2Config.V2_0);

        // 0. Resampling: the C# resamplers against librosa(22.05k)+torchaudio(16k) on the same clip.
        WavFile.DecodedAudio wav = WavFile.Read(refWavPath!);
        float[] clip = wav.Channels[0];
        output.WriteLine($"reference clip: {clip.Length / (double)wav.SampleRate:F2}s @ {wav.SampleRate} Hz");
        ReadOnlySpan<float> capped = clip.AsSpan(0, Math.Min(clip.Length, 15 * wav.SampleRate));
        float[] r22 = SincResampler.Resample(capped, wav.SampleRate, 22_050, 64);
        float[] r16 = SincResampler.Resample(r22, 22_050, 16_000);
        float[] py16 = d["audio16k"];
        Compare("resample 22.05k", r22.AsSpan(0, Math.Min(r22.Length, d["audio22k"].Length)).ToArray(), d["audio22k"].AsSpan(0, Math.Min(r22.Length, d["audio22k"].Length)).ToArray(), 0.02);
        int n16 = Math.Min(r16.Length, py16.Length);
        output.WriteLine($"audio16k length: C# {r16.Length} vs reference {py16.Length}");
        Compare("resample 16k (overlap)", r16[..n16], py16[..n16], 0.02);

        // 1a. SeamlessM4T-style front end (Kaldi fbank, per-bin normalization, frame-pair stacking).
        float[] a16 = py16;
        using (Tensor feats = pipeline.SemanticFeatures.Extractor.ExtractStackedFeatures(a16))
        {
            Assert.Equal(d.Shape("input_features")[0], feats.Shape[1]);
            Compare("w2v-bert input features", ToArray(feats), d["input_features"], 5e-3);
        }

        // 1. w2v-bert feature from the reference's own 16 kHz audio.
        using Tensor spk = pipeline.SemanticFeatures.Forward(backend, a16);
        long tSpk = spk.Shape[1];
        Assert.Equal(d.Shape("spk_cond_emb")[0], tSpk);
        Compare("w2v-bert hidden[17] normalized", ToArray(spk), d["spk_cond_emb"], 2e-3);

        // 2. CAM++ style.
        using Tensor style = S3GenReference.SpeakerEmbedding(backend, GetCamplus(pipeline), a16);
        Compare("CAM++ style", ToArray(style), d["style"], 1e-2);

        // 3. Reference mel (22.05k).
        using Tensor refMel = pipeline.ComputeRefMel(d["audio22k"]);
        Assert.Equal(d.Shape("ref_mel")[1], refMel.Shape[2]);
        Compare("reference mel", ToArray(refMel), d["ref_mel"], 2e-3);

        // 4. Semantic-codec quantization of the reference feature (codes must agree almost everywhere).
        using Tensor spkPy = ToTensor(d["spk_cond_emb"], 1, tSpk, 1024);
        (int[] codesRef, Tensor cont) = pipeline.SemanticCodec.Quantize(backend, spkPy, (int)tSpk);
        using (cont)
        {
            // Older dumps stored the continuous S_ref (what `_, S_ref = quantize(...)` returns) under codes_ref.
            float[] sRef = d.Has("s_ref") ? d["s_ref"] : d["codes_ref"];
            Compare("S_ref (quantized embedding)", ToArray(cont), sRef, 2e-3);
            if (d.Has("s_ref"))
            {
                float[] pyCodesRef = d["codes_ref"];
                int same = 0;
                for (int i = 0; i < codesRef.Length; i++) if (codesRef[i] == (int)pyCodesRef[i]) same++;
                output.WriteLine($"{"reference codes",-34} {same}/{codesRef.Length} identical");
                Assert.True(same >= codesRef.Length * 0.99, $"reference quantization disagrees on {codesRef.Length - same}/{codesRef.Length} frames.");
            }
        }

        // 5. Prompt condition (length regulator over the reference feature).
        int tRef = (int)d.Shape("ref_mel")[1];
        using Tensor sRefPy = ToTensor(d.Has("s_ref") ? d["s_ref"] : d["codes_ref"], 1, tSpk, 1024);
        using Tensor prompt = pipeline.LengthRegulator.Forward(backend, sRefPy, (int)tSpk, tRef);
        Compare("prompt condition (length reg.)", ToArray(prompt), d["prompt_condition"], 2e-3);

        // 6. Speaker Conformer+Perceiver conditioning and the base emotion vector.
        using Tensor latents = pipeline.Gpt.ComputeSpeakerConditioningConformerPerceiver(backend, spkPy, (int)tSpk);
        Compare("speaker latents (conformer+perceiver)", ToArray(latents), d["spk_latent"], 5e-3);
        using Tensor baseEmo = pipeline.Gpt.ComputeEmoVec(backend, spkPy, (int)tSpk);
        Compare("base emotion vector", ToArray(baseEmo), d["emovec_base"], 5e-3);

        // 7. Second GPT pass + gpt_layer on the reference's sampled codes, and the codebook embedding.
        int[] codes = [.. d["seg0_codes"].Select(static c => (int)c)];
        int[] textIds = [.. d["seg0_text_ids"].Select(static c => (int)c)];
        using Tensor latentsPy = ToTensor(d["spk_latent"], 1, 32, 1280);
        using Tensor emovecPy = ToTensor(d["emovec"], 1, 1280);
        using Tensor second = pipeline.Gpt.ComputeSecondPassLatent(backend, latentsPy, emovecPy, textIds, codes);
        Compare("second pass + gpt_layer", ToArray(second), d["seg0_latent"], 5e-3);
        using Tensor vq = pipeline.SemanticCodec.VqToEmbedding(backend, codes);
        Compare("vq2emb(codes)", ToArray(vq), d["seg0_vq2emb"], 1e-3);

        // 8. Length regulator over S_infer.
        using Tensor sInferPy = ToTensor(d["seg0_vq2emb"].Zip(d["seg0_latent"], static (a, b) => a + b).ToArray(), 1, codes.Length, 1024);
        int targetLen = (int)(codes.Length * 1.72);
        using Tensor cond = pipeline.LengthRegulator.Forward(backend, sInferPy, codes.Length, targetLen);
        Compare("content condition (length reg.)", ToArray(cond), d["seg0_cond"], 2e-3);

        // 9. The S2Mel estimator on the reference's own stacked step inputs (conditional + CFG-null halves).
        foreach (int step in new[] { 1, 13, 25 })
        {
            if (!d.Has($"est{step}_x")) continue;
            long[] xs = d.Shape($"est{step}_x");           // [2, 80, T]
            int t = (int)xs[2];
            float[] x = d[$"est{step}_x"], promptX = d[$"est{step}_prompt_x"], mu = d[$"est{step}_mu"], sty = d[$"est{step}_style"], outRef = d[$"est{step}_out"];
            float tv = d[$"est{step}_t"][0];
            int xChunk = 80 * t, muChunk = t * 512;
            for (int half = 0; half < 2; half++)
            {
                using Tensor xT = ToTensor(x[(half * xChunk)..((half + 1) * xChunk)], 1, 80, t);
                using Tensor pT = ToTensor(promptX[(half * xChunk)..((half + 1) * xChunk)], 1, 80, t);
                using Tensor muT = ToTensor(mu[(half * muChunk)..((half + 1) * muChunk)], 1, t, 512);
                using Tensor sT = ToTensor(sty[(half * 192)..((half + 1) * 192)], 1, 192);
                using Tensor est = pipeline.Dit.Estimate(backend, xT, muT, tv, sT, pT);
                Compare($"DiT estimator step {step} ({(half == 0 ? "cond" : "null")})", ToArray(est), outRef[(half * xChunk)..((half + 1) * xChunk)], 5e-3);
            }
        }

        // 10. BigVGAN on the reference's own final mel.
        int melLen = (int)d.Shape("seg0_vc_target")[1];
        using Tensor mel = ToTensor(d["seg0_vc_target"], 1, 80, melLen);
        float[] pcm = pipeline.Vocode(backend, mel, melLen);
        float[] pyWav = d["seg0_wav"];
        output.WriteLine($"vocoder length: C# {pcm.Length} vs reference {pyWav.Length}");
        Compare("BigVGAN waveform", pcm, pyWav.Length == pcm.Length ? pyWav : pyWav[..pcm.Length], 5e-3);

        Assert.True(_failures.Count == 0, string.Join(Environment.NewLine, _failures));
    }

    /// <summary>The whole autoregressive path (conditioning assembly, KV-cached stepping, repetition penalty, sampling
    /// head) against the reference's greedy decode — with temperature 0 both pick the arg-max, so the code sequences
    /// must agree token for token (up to a floating-point near-tie flipping a late token).</summary>
    [Fact]
    public async Task GreedyDecode_ReproducesTheReferenceCodes_WithAndWithoutRepetitionPenalty()
    {
        string? parityDir = Environment.GetEnvironmentVariable("INDEXTTS2_PARITY_DIR");
        string? v20Dir = Environment.GetEnvironmentVariable("INDEXTTS2_V20_DIR");
        string? codecPath = Environment.GetEnvironmentVariable("INDEXTTS2_MASKGCT_CODEC_PATH");
        string? w2vBertPath = Environment.GetEnvironmentVariable("INDEXTTS2_W2VBERT_SAFETENSORS_PATH");
        string? campplusPath = Environment.GetEnvironmentVariable("INDEXTTS2_CAMPPLUS_PATH");
        string? bigVganPath = Environment.GetEnvironmentVariable("INDEXTTS2_BIGVGAN_PT_PATH");
        if (string.IsNullOrEmpty(parityDir) || !File.Exists(Path.Combine(parityDir, "greedy1.safetensors"))) { output.WriteLine("greedy1.safetensors missing — skip."); return; }
        if (string.IsNullOrEmpty(v20Dir) || !Directory.Exists(v20Dir)) { output.WriteLine("INDEXTTS2_V20_DIR missing — skip."); return; }
        foreach (string? p in new[] { codecPath, w2vBertPath, campplusPath, bigVganPath })
            if (string.IsNullOrEmpty(p) || !File.Exists(p)) { output.WriteLine("auxiliary checkpoint missing — skip."); return; }

        Dump d = LoadDump(Path.Combine(parityDir, "case1_b3.safetensors"));
        Dump g = LoadDump(Path.Combine(parityDir, "greedy1.safetensors"));
        using CpuBackend backend = new();
        using IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
            Path.Combine(v20Dir, "bpe.model"), Path.Combine(v20Dir, "gpt.pth"), Path.Combine(v20Dir, "s2mel.pth"), codecPath!,
            w2vBertPath!, Path.Combine(v20Dir, "wav2vec2bert_stats.pt"), campplusPath!, bigVganPath!, cfg: IndexTts2Config.V2_0);

        int[] textIds = [.. d["seg0_text_ids"].Select(static c => (int)c)];
        using Tensor latents = ToTensor(d["spk_latent"], 1, 32, 1280);
        using Tensor emovec = ToTensor(g["emovec"], 1, 1280);
        foreach ((string key, float penalty) in new[] { ("codes_greedy_rp1", 1f), ("codes_greedy_rp10", 10f) })
        {
            int[] want = [.. g[key].Select(static c => (int)c)];
            // The reference list stops at the stop token (excluded) or at the 400-token cap.
            uint rng = 1;
            int[] got = pipeline.Gpt.Generate(backend, latents, emovec, textIds, null,
                new HartsyInference.Audio.Pipelines.IndexTtsOptions { Temperature = 0f, RepetitionPenalty = penalty, MaxMelTokens = 400 }, ref rng);
            int common = 0;
            while (common < Math.Min(got.Length, want.Length) && got[common] == want[common]) common++;
            output.WriteLine($"{key}: C# {got.Length} codes, reference {want.Length} codes, identical prefix {common}");
            Assert.True(common >= Math.Min(got.Length, want.Length) * 0.97 || common == got.Length,
                $"{key}: greedy decode diverges from the reference at token {common} (C# {got.Length}, reference {want.Length}).");
        }
    }

    private static CamPlusSpeakerEncoder GetCamplus(IndexTts2Pipeline pipeline) => pipeline.CamPlus;
}
