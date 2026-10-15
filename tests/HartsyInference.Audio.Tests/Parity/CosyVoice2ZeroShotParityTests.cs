using System.Text;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>CosyVoice 2 zero-shot prompt conditioning on the CPU backend against upstream <c>frontend_zero_shot</c>, from
/// tools/parity/cosyvoice2_zeroshot_dump.py (directory in <c>COSYVOICE2_ZS_REF_DIR</c>): speech tokens, the prompt mel
/// after the <c>min(frames / 2, tokens)</c> cut, and the CAM++ embedding, all from the 24 kHz clip through the engine's
/// resampling. The frozen S3 tokenizer and CAM++ load from Chatterbox's <c>s3gen.safetensors</c>, as the engine does.</summary>
public sealed class CosyVoice2ZeroShotParityTests
{
    private const float PromptMelTolerance = 1e-3f;
    private const double MinCosine = 0.9999;

    private readonly ITestOutputHelper _out;

    public CosyVoice2ZeroShotParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void ZeroShotPrompt_MatchesUpstreamFrontend()
    {
        string refDir = Environment.GetEnvironmentVariable("COSYVOICE2_ZS_REF_DIR") ?? "";
        string s3Path = Path.Combine(AudioModelCache.GetRepoDirectory("ResembleAI/chatterbox", "tts"), "s3gen.safetensors");
        if (!RealWeightGate.Require(_out.WriteLine, Path.Combine(refDir, "meta.json"), s3Path)) return;

        using CpuBackend backend = new();
        using SafeTensorsLoader loader = new();
        loader.Load(s3Path);
        IReadOnlyDictionary<string, Tensor> s3gen = loader.GetAllTensors();
        using CamPlusSpeakerEncoder speakerEncoder = new(CosyVoiceConfig.V2_0_5B.Flow.SpeakerEmbedDim);
        speakerEncoder.LoadWeights(s3gen, "speaker_encoder");
        using S3Tokenizer s3 = new();
        s3.LoadWeights(s3gen, "tokenizer");

        float[] ref24 = ReadF32(refDir, "ref24.f32");
        (int[] tokens, Tensor mel, Tensor speaker) = CosyVoicePipeline.ZeroShotPrompt(backend, s3, speakerEncoder, ref24, 24_000);
        using (mel)
        using (speaker)
        {
            int[] wantTokens = ReadI32(refDir, "prompt_tokens.i32");
            float[] wantMel = ReadF32(refDir, "prompt_feat_tm.f32");
            int bins = (int)mel.Shape[1], frames = (int)mel.Shape[2];
            float melDiff = float.PositiveInfinity;
            if (wantMel.Length == bins * frames)
            {
                ReadOnlySpan<float> got = mel.AsSpan<float>();
                melDiff = 0f;
                for (int f = 0; f < frames; f++)
                    for (int b = 0; b < bins; b++) melDiff = MathF.Max(melDiff, MathF.Abs(got[b * frames + f] - wantMel[f * bins + b]));
            }
            double cos = Cosine(speaker.AsSpan<float>(), ReadF32(refDir, "embedding.f32"));
            int same = 0;
            for (int i = 0; i < Math.Min(tokens.Length, wantTokens.Length); i++) if (tokens[i] == wantTokens[i]) same++;
            string report = new StringBuilder()
                .AppendLine($"prompt tokens {same}/{wantTokens.Length} (engine {tokens.Length})")
                .AppendLine($"prompt mel frames {frames} vs {wantMel.Length / bins}, max|d| {melDiff:G4}")
                .AppendLine($"embedding cos {cos:F6}").ToString();
            _out.WriteLine(report);
            Assert.True(tokens.AsSpan().SequenceEqual(wantTokens) && melDiff <= PromptMelTolerance && cos >= MinCosine, report);
        }
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
