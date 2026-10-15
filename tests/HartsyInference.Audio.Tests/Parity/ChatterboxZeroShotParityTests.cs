using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Chatterbox zero-shot conditioning on the CPU backend against upstream <c>prepare_conditionals</c>, from
/// tools/parity/chatterbox_zeroshot_dump.py (directory in <c>CHATTERBOX_ZS_REF_DIR</c>). Upstream makes the 16 kHz copy
/// the voice encoder and the T3 prompt tokens see with librosa's soxr_hq, which has no port, so that path is checked
/// exactly on upstream's own 16 kHz samples and to a 95% token floor through the engine's long-sinc stand-in. S3Gen's
/// <c>embed_ref</c> path runs from the 24 kHz clip through the engine's resampler. The S3Gen flow runs from upstream's
/// own noise, and teacher-forced T3 logits with the engine-derived conditioning close the loop.</summary>
public sealed class ChatterboxZeroShotParityTests
{
    private const float LogitTolerance = 2e-3f;
    private const float PromptMelTolerance = 1e-3f;
    private const float FlowMelTolerance = 1e-3f;
    private const double MinCosine = 0.9999;
    private const double MinStandInTokenShare = 0.95;

    private readonly ITestOutputHelper _out;

    public ChatterboxZeroShotParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void ZeroShotConditioning_MatchesUpstream()
    {
        string refDir = Environment.GetEnvironmentVariable("CHATTERBOX_ZS_REF_DIR") ?? "";
        string ckpt = AudioModelCache.GetRepoDirectory("ResembleAI/chatterbox", "tts");
        string meta = Path.Combine(refDir, "meta.json");
        string t3Path = Path.Combine(ckpt, "t3_cfg.safetensors");
        string s3Path = Path.Combine(ckpt, "s3gen.safetensors");
        string vePath = Path.Combine(ckpt, "ve.safetensors");
        if (!RealWeightGate.Require(_out.WriteLine, meta, t3Path, s3Path, vePath)) return;

        ChatterboxConfig cfg = ChatterboxConfig.Default;
        CosyVoiceConfig cosy = CosyVoiceConfig.V2_0_5B;
        using CpuBackend backend = new();
        using SafeTensorsLoader t3Loader = new(), s3Loader = new(), veLoader = new();
        t3Loader.Load(t3Path);
        s3Loader.Load(s3Path);
        veLoader.Load(vePath);
        Dictionary<string, Tensor> merged = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kv in t3Loader.GetAllTensors()) merged["t3." + kv.Key] = kv.Value;
        foreach (KeyValuePair<string, Tensor> kv in s3Loader.GetAllTensors()) merged["s3gen." + kv.Key] = kv.Value;
        foreach (KeyValuePair<string, Tensor> kv in veLoader.GetAllTensors()) merged["ve." + kv.Key] = kv.Value;
        ChatterboxT3 t3 = new(cfg);
        ChatterboxVoiceEncoder voiceEncoder = new();
        CosyVoiceFlow flow = new(cosy);
        using ChatterboxPipeline pipeline = new(cfg, t3, flow, new HiFTNetVocoder(cosy.Hift),
            ChatterboxPipeline.CreateSpeakerEncoder(cosy.Flow.SpeakerEmbedDim), new S3Tokenizer(), voiceEncoder);
        pipeline.LoadWeights(merged);

        StringBuilder report = new();
        bool ok = true;

        // T3 path on upstream's 16 kHz samples: voice-encoder embedding and cond-prompt tokens.
        float[] ref16 = ReadF32(refDir, "ref16_librosa.f32");
        using Tensor ve = voiceEncoder.EmbedUtterance(backend, ref16);
        double veCos = Cosine(ve.AsSpan<float>(), ReadF32(refDir, "ve.f32"));
        int[] condTokens = pipeline.T3CondPromptTokens(backend, ref16);
        int[] wantCond = ReadI32(refDir, "t3_cond_tokens.i32");
        report.AppendLine($"voice encoder cos {veCos:F6}; T3 cond tokens {Matches(condTokens, wantCond)}");
        ok &= veCos >= MinCosine && condTokens.AsSpan().SequenceEqual(wantCond);

        // The same from the 24 kHz clip through the soxr stand-in. Upstream's torchaudio and soxr copies agree on 93/111
        // of these tokens, so the floor sits well above that.
        float[] ref24 = ReadF32(refDir, "ref24.f32");
        float[] ref16Engine = ChatterboxPipeline.ResampleFullReference(ref24);
        using (Tensor veEngine = voiceEncoder.EmbedUtterance(backend, ref16Engine))
        {
            int[] engineCond = pipeline.T3CondPromptTokens(backend, ref16Engine);
            double veEngineCos = Cosine(veEngine.AsSpan<float>(), ReadF32(refDir, "ve.f32"));
            int same = 0;
            for (int i = 0; i < Math.Min(engineCond.Length, wantCond.Length); i++) if (engineCond[i] == wantCond[i]) same++;
            report.AppendLine($"from the clip via the soxr stand-in: 16 kHz max|d| {MaxDiff(ref16Engine, ref16):G4}, voice encoder cos "
                + $"{veEngineCos:F6}, T3 cond tokens {Matches(engineCond, wantCond)}");
            ok &= veEngineCos >= MinCosine && engineCond.Length == wantCond.Length && same >= MinStandInTokenShare * wantCond.Length;
        }

        // embed_ref from the 24 kHz clip.
        (int[] promptTokens, Tensor promptMel, Tensor xvec) = pipeline.EmbedReference(backend, ref24);
        using (promptMel)
        using (xvec)
        {
            int[] wantPrompt = ReadI32(refDir, "gen_prompt_tokens.i32");
            float melDiff = MaxDiffChannelMajorVsTimeMajor(promptMel, ReadF32(refDir, "gen_prompt_feat_tm.f32"));
            double xvecCos = Cosine(xvec.AsSpan<float>(), ReadF32(refDir, "gen_xvec.f32"));
            report.AppendLine($"embed_ref: prompt tokens {Matches(promptTokens, wantPrompt)}, prompt mel max|d| {melDiff:G4}, x-vector cos {xvecCos:F6}");
            ok &= promptTokens.AsSpan().SequenceEqual(wantPrompt) && melDiff <= PromptMelTolerance && xvecCos >= MinCosine;

            // S3Gen flow from upstream's own noise: the solver's time grid, CFG and prompt trim end to end.
            using JsonDocument flowMeta = JsonDocument.Parse(File.ReadAllText(meta));
            int[] flowTokens = flowMeta.RootElement.GetProperty("flow_tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            float[] z = ReadF32(refDir, "flow_z.f32");
            float[] wantFeat = ReadF32(refDir, "flow_feat.f32");
            using Tensor x0 = new(new TensorShape(1, 80, z.Length / 80), DType.F32);
            z.CopyTo(x0.AsSpan<float>());
            using Tensor feat = flow.Inference(backend, flowTokens, promptTokens, promptMel, xvec, 0, null, x0);
            ReadOnlySpan<float> gotFeat = feat.AsSpan<float>();
            float featDiff = gotFeat.Length == wantFeat.Length ? 0f : float.PositiveInfinity;
            for (int i = 0; i < Math.Min(gotFeat.Length, wantFeat.Length); i++) featDiff = MathF.Max(featDiff, MathF.Abs(gotFeat[i] - wantFeat[i]));
            report.AppendLine($"flow from upstream noise: {feat.Shape[2]} frames (upstream {wantFeat.Length / 80}), mel max|d| {featDiff:G4}");
            ok &= featDiff <= FlowMelTolerance;
        }

        // Teacher-forced T3 logits with the engine-derived conditioning.
        ChatterboxEnTokenizer tokenizer = new();
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(meta));
        foreach (JsonProperty tf in doc.RootElement.GetProperty("tf").EnumerateObject())
        {
            string text = tf.Value.GetProperty("text").GetString()!;
            int[] ids = tokenizer.EncodeWithStartStop(ChatterboxTextNormalizer.Normalize(text));
            int[] forced = tf.Value.GetProperty("forced").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            float[] want = ReadF32(refDir, tf.Name + ".f32");
            float[] got = t3.TeacherForcedLogits(backend, ids, ve, 0.5f, condTokens, forced);
            Assert.Equal(want.Length, got.Length);
            float maxAbs = 0f;
            for (int i = 0; i < want.Length; i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(want[i] - got[i]));
            report.AppendLine($"{tf.Name} \"{text}\": {forced.Length} forced tokens, max|d| {maxAbs:G4}");
            ok &= maxAbs <= LogitTolerance;
        }

        _out.WriteLine(report.ToString());
        Assert.True(ok, report.ToString());
    }

    private static float MaxDiffChannelMajorVsTimeMajor(Tensor channelMajor, float[] timeMajor)
    {
        int bins = (int)channelMajor.Shape[1], frames = (int)channelMajor.Shape[2];
        Assert.Equal(timeMajor.Length, bins * frames);
        ReadOnlySpan<float> got = channelMajor.AsSpan<float>();
        float maxAbs = 0f;
        for (int f = 0; f < frames; f++)
            for (int b = 0; b < bins; b++) maxAbs = MathF.Max(maxAbs, MathF.Abs(got[b * frames + f] - timeMajor[f * bins + b]));
        return maxAbs;
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        float maxAbs = a.Length == b.Length ? 0f : float.PositiveInfinity;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(a[i] - b[i]));
        return maxAbs;
    }

    private static string Matches(int[] got, int[] want)
    {
        int same = 0;
        for (int i = 0; i < Math.Min(got.Length, want.Length); i++) if (got[i] == want[i]) same++;
        return $"{same}/{want.Length} (engine {got.Length})";
    }

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return dot / Math.Sqrt(na * nb);
    }

    private static float[] ReadF32(string dir, string name)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(dir, name));
        float[] values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * 4);
        return values;
    }

    private static int[] ReadI32(string dir, string name)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(dir, name));
        int[] values = new int[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * 4);
        return values;
    }
}
