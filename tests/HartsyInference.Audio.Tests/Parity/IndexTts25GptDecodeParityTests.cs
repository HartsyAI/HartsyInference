using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Tests.IndexTts2;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>IndexTTS-2.5 GPT decode against upstream on real weights, CPU backend, F32. Reference dumps come from
/// tools/parity/indextts25_gpt_decode_dump.py (<c>indextts25_gpt_decode_l*.safetensors</c> in
/// <c>INDEXTTS25_GPT_DECODE_DUMP_DIR</c>), recorded on the clip in <c>INDEXTTS2_BEAM_REF_WAV</c>; weights come from
/// <c>HARTSY_INDEXTTS_MODELS_ROOT</c>.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts25GptDecodeParityTests(ITestOutputHelper output)
{
    private const string DumpDirVariable = "INDEXTTS25_GPT_DECODE_DUMP_DIR";

    /// <summary>Upstream's own conditioning, text and greedy codes through the C# GPT one cached step at a time: every
    /// block output, the <c>ln_f</c> output and the mel-head logits at every step, and the penalized argmax.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void TeacherForcedDecode_MatchesUpstream(int line)
    {
        string gptPath = Path.Combine(IndexTtsFrontendFixture.ModelsRoot, "IndexTeam--IndexTTS-2.5", "gpt.pth");
        string dumpPath = DumpPath(line);
        if (!RealWeightGate.Require(output.WriteLine, gptPath, dumpPath)) return;

        using SafeTensorsLoader dump = new();
        dump.Load(dumpPath);
        Dictionary<string, Tensor> d = dump.GetAllTensors();
        int[] codes = Ids(d["codes"]);
        int[] textIds = [.. Ids(d["text_ids"]).Where(static id => id is not (IndexTts2T2sDecoder.StartTextToken or IndexTts2T2sDecoder.StopTextToken))];
        int langId = Ids(d["lang_id"])[0];

        using PytorchPickleLoader loader = new();
        loader.Load(gptPath, recursiveFlatten: true);
        List<Tensor> converted = [];
        IndexTts2Config cfg = IndexTts2Config.V2_5;
        using IndexTts2T2sDecoder gpt = new(cfg.Gpt, cfg.NumberTextTokens, cfg.MaxMelTokens);
        try
        {
            gpt.LoadWeights(IndexTtsPipeline.ToF32(loader.GetAllTensors(), converted));
            using CpuBackend backend = new();
            using Tensor spk = gpt.ComputeSpeakerConditioning(backend, d["style"]);
            double spkErr = RelRms(spk.AsSpan<float>(), d["spk_proj"].AsSpan<float>());
            output.WriteLine($"spk_emb_proj relRMS {spkErr:E2}");

            int layers = cfg.Gpt.NumLayers;
            double[,] err = new double[codes.Length, layers + 1];
            float[][] logits = gpt.TeacherForce(backend, d["spk_proj"], d["emovec"], textIds, langId, codes, (step, layer, x) =>
                err[step, layer] = RelRms(x.AsSpan<float>(), d[layer == layers ? $"step{step}_lnf" : $"step{step}_block{layer}"].AsSpan<float>()));

            HashSet<int> seen = [1, IndexTts2T2sDecoder.StartMelToken];
            int firstFlip = -1;
            double worstHidden = 0, worstLogits = 0;
            for (int s = 0; s < codes.Length; s++)
            {
                ReadOnlySpan<float> want = d[$"step{s}_logits"].AsSpan<float>();
                double logitErr = RelRms(logits[s], want);
                int pick = PenalizedArgMax(logits[s], seen);
                if (pick != codes[s] && firstFlip < 0) firstFlip = s;
                seen.Add(codes[s]);
                for (int l = 0; l <= layers; l++) worstHidden = Math.Max(worstHidden, err[s, l]);
                worstLogits = Math.Max(worstLogits, logitErr);
                output.WriteLine($"step {s,2}: block0 {err[s, 0]:E2} block{layers - 1} {err[s, layers - 1]:E2} ln_f {err[s, layers]:E2} " +
                    $"logits {logitErr:E2} pick {pick} upstream {codes[s]}");
            }
            output.WriteLine($"worst hidden relRMS {worstHidden:E2}, worst logits relRMS {worstLogits:E2}, first argmax flip {firstFlip}");
            Assert.True(spkErr < 1e-5, $"spk_emb_proj relRMS {spkErr:E2}");
            Assert.True(worstHidden < 1e-4, $"a block or ln_f output reaches relRMS {worstHidden:E2}");
            Assert.True(worstLogits < 1e-4, $"logits relRMS reaches {worstLogits:E2}");
            Assert.Equal(-1, firstFlip);
        }
        finally
        {
            foreach (Tensor t in converted) t.Dispose();
        }
    }

    /// <summary>The GPT's conditioning inputs from the reference clip: CAM++ style and <c>spk_emb_proj</c>, the
    /// w2v-bert feature (frame count included) and the emotion vector, first from upstream's own resampled audio (CAM++,
    /// w2v-bert and emotion encoder numerics only) and then from <see cref="IndexTts2Pipeline.PrepareReference"/> on the
    /// clip itself (adding the C# resamplers, which approximate librosa's soxr).</summary>
    [Fact]
    public async Task ReferenceConditioning_MatchesUpstream()
    {
        string root = IndexTtsFrontendFixture.ModelsRoot;
        string v25 = Path.Combine(root, "IndexTeam--IndexTTS-2.5");
        string w2v = Path.Combine(root, "facebook--w2v-bert-2.0", "model.safetensors");
        string campplus = Path.Combine(root, "funasr--campplus", "campplus_cn_common.bin");
        string bigVgan = Path.Combine(root, "nvidia--bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt");
        string refWav = Environment.GetEnvironmentVariable("INDEXTTS2_BEAM_REF_WAV") ?? "";
        string[] files = ["multilingual_zh_ja_yue_char_del.tiktoken", "gpt.pth", "s2mel.pth", "codec.pth", "wav2vec2bert_stats.pt"];
        if (!RealWeightGate.Require(output.WriteLine, [.. files.Select(f => Path.Combine(v25, f)), w2v, campplus, bigVgan, refWav, DumpPath(0)])) return;

        using SafeTensorsLoader dump = new();
        dump.Load(DumpPath(0));
        Dictionary<string, Tensor> d = dump.GetAllTensors();
        using IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
            Path.Combine(v25, files[0]), Path.Combine(v25, files[1]), Path.Combine(v25, files[2]), Path.Combine(v25, files[3]),
            w2v, Path.Combine(v25, files[4]), campplus, bigVgan, IndexTtsTestNormalizers.Resolve());
        using CpuBackend backend = new();
        List<string> failures = [];
        void Check(string stage, Tensor got, string key, double tolerance)
        {
            Tensor want = d[key];
            if (got.ElementCount != want.ElementCount)
            {
                failures.Add($"{stage}: {got.ElementCount} values, upstream {want.ElementCount}");
                output.WriteLine($"{stage,-40} {got.ElementCount} values vs upstream {want.ElementCount}");
                return;
            }
            double err = RelRms(got.AsSpan<float>(), want.AsSpan<float>());
            output.WriteLine($"{stage,-40} relRMS {err:E2} (limit {tolerance:E0})");
            if (!(err < tolerance)) failures.Add($"{stage}: relRMS {err:E2} exceeds {tolerance:E0}");
        }

        // Upstream's own 16 kHz signals: speaker chain (file → 22.05 kHz → 16 kHz) and emotion load (file → 16 kHz).
        float[] up16 = d["audio16k"].AsSpan<float>().ToArray(), upEmo16 = d["emo_audio16k"].AsSpan<float>().ToArray();
        using (Tensor style = S3GenReference.SpeakerEmbedding(backend, pipeline.CamPlus, up16)) Check("CAM++ style (upstream audio)", style, "style", 1e-4);
        using (Tensor spkFeat = pipeline.SemanticFeatures.Forward(backend, up16)) Check("w2v-bert spk_cond_emb (upstream audio)", spkFeat, "spk_cond_emb", 1e-4);
        using (Tensor emoFeat = pipeline.SemanticFeatures.Forward(backend, upEmo16))
        {
            Check("w2v-bert emo_cond_emb (upstream audio)", emoFeat, "emo_cond_emb", 1e-4);
            using Tensor emoVec = pipeline.Gpt.ComputeEmoVec(backend, emoFeat, (int)emoFeat.Shape[1]);
            Check("emovec (upstream audio)", emoVec, "emovec", 1e-3);
        }

        // The pipeline's own reference preparation from the clip.
        WavFile.DecodedAudio audio = WavFile.Read(refWav);
        using IndexTts2Reference reference = pipeline.PrepareReference(backend, audio.Channels[0], audio.SampleRate);
        Check("CAM++ style (PrepareReference)", reference.Style, "style", 1e-4);
        Check("spk_emb_proj (PrepareReference)", reference.SpeakerConditioning, "spk_proj", 1e-4);
        Check("spk_cond_emb (PrepareReference)", reference.SpkCondEmb, "spk_cond_emb", 1e-4);
        Check("emovec (PrepareReference)", reference.GetEmoVec(backend), "emovec", 5e-3);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private static string DumpPath(int line) =>
        Path.Combine(Environment.GetEnvironmentVariable(DumpDirVariable) ?? "", $"indextts25_gpt_decode_l{line}.safetensors");

    private static int[] Ids(Tensor t)
    {
        ReadOnlySpan<float> s = t.AsSpan<float>();
        int[] ids = new int[s.Length];
        for (int i = 0; i < s.Length; i++) ids[i] = (int)s[i];
        return ids;
    }

    /// <summary>HF greedy with <c>repetition_penalty=10</c> over the ids fed so far (filler 1, start token, codes).</summary>
    private static int PenalizedArgMax(float[] logits, HashSet<int> seen)
    {
        float[] l = (float[])logits.Clone();
        foreach (int id in seen) l[id] = l[id] > 0 ? l[id] / 10f : l[id] * 10f;
        int best = 0;
        for (int i = 1; i < l.Length; i++) if (l[i] > l[best]) best = i;
        return best;
    }

    private static double RelRms(ReadOnlySpan<float> got, ReadOnlySpan<float> want)
    {
        Assert.Equal(want.Length, got.Length);
        double num = 0, den = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double diff = got[i] - want[i];
            num += diff * diff;
            den += (double)want[i] * want[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
