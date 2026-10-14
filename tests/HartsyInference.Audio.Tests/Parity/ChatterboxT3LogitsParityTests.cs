using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Teacher-forced Chatterbox T3 speech logits on the CPU backend against upstream <c>T3.inference</c>, from
/// tools/parity/chatterbox_t3_logits_dump.py (default voice, CFG rows, forced decode tokens). The reference directory
/// comes from <c>CHATTERBOX_T3_REF_DIR</c>; the checkpoint from the audio model cache.</summary>
public sealed class ChatterboxT3LogitsParityTests
{
    private const float MaxAbsTolerance = 2e-3f;

    private readonly ITestOutputHelper _out;

    public ChatterboxT3LogitsParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void TeacherForcedLogits_MatchUpstream()
    {
        string refDir = Environment.GetEnvironmentVariable("CHATTERBOX_T3_REF_DIR") ?? "";
        string ckpt = AudioModelCache.GetRepoDirectory("ResembleAI/chatterbox", "tts");
        string meta = Path.Combine(refDir, "t3_logits_meta.json");
        string t3Path = Path.Combine(ckpt, "t3_cfg.safetensors");
        string condsPath = Path.Combine(ckpt, "conds.pt");
        if (!RealWeightGate.Require(_out.WriteLine, meta, t3Path, condsPath)) return;

        ChatterboxConfig cfg = ChatterboxConfig.Default;
        using CpuBackend backend = new();
        using SafeTensorsLoader t3Loader = new();
        t3Loader.Load(t3Path);
        using ChatterboxT3 t3 = new(cfg);
        t3.LoadWeights(t3Loader.GetAllTensors());
        using AnyFormatCheckpointLoader condsLoader = new();
        condsLoader.Load(condsPath, recursiveFlatten: true);
        IReadOnlyDictionary<string, Tensor> conds = condsLoader.GetAllTensors();
        using Tensor speaker = new(new TensorShape(cfg.SpeakerEmbedDim), DType.F32);
        conds["t3.speaker_emb"].AsSpan<float>()[..cfg.SpeakerEmbedDim].CopyTo(speaker.AsSpan<float>());
        ReadOnlySpan<long> promptLong = conds["t3.cond_prompt_speech_tokens"].AsSpan<long>();
        int[] prompt = new int[promptLong.Length];
        for (int i = 0; i < prompt.Length; i++) prompt[i] = (int)promptLong[i];

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(meta));
        int[] forced = Ints(doc.RootElement.GetProperty("forced"));
        float exaggeration = doc.RootElement.GetProperty("exaggeration").GetSingle();
        StringBuilder report = new();
        bool ok = true;
        foreach (JsonProperty text in doc.RootElement.GetProperty("texts").EnumerateObject())
        {
            int[] ids = Ints(text.Value.GetProperty("ids"));
            float[] want = ReadF32(Path.Combine(refDir, $"t3_logits_{text.Name}.f32"));
            foreach (int bos in new[] { 2, 1 })
            {
                float[] got = t3.TeacherForcedLogits(backend, ids, speaker, exaggeration, prompt, forced, bos);
                Assert.Equal(want.Length, got.Length);
                (float maxAbs, int argmaxAgree, int steps) = Compare(want, got, cfg.SpeechVocab);
                report.AppendLine($"{text.Name} bos={bos}: max|d|={maxAbs:G4}, cond argmax agrees {argmaxAgree}/{steps} steps");
                if (bos == 2) ok &= maxAbs <= MaxAbsTolerance && argmaxAgree == steps;
            }
        }
        _out.WriteLine(report.ToString());
        Assert.True(ok, report.ToString());
    }

    private static (float MaxAbs, int ArgmaxAgree, int Steps) Compare(float[] want, float[] got, int vocab)
    {
        float maxAbs = 0f;
        for (int i = 0; i < want.Length; i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(want[i] - got[i]));
        int steps = want.Length / (2 * vocab), agree = 0;
        for (int s = 0; s < steps; s++)
        {
            ReadOnlySpan<float> w = want.AsSpan(s * 2 * vocab, vocab), g = got.AsSpan(s * 2 * vocab, vocab);
            if (ArgMax(w) == ArgMax(g)) agree++;
        }
        return (maxAbs, agree, steps);
    }

    private static int ArgMax(ReadOnlySpan<float> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static int[] Ints(JsonElement array) => array.EnumerateArray().Select(e => e.GetInt32()).ToArray();

    private static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * 4);
        return values;
    }
}
