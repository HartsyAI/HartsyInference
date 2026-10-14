using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>IndexTTS-2.5 greedy decoding (3 beams and 1, no sampling) against upstream's codes for the same clip and
/// text (<c>Fixtures/indextts25_beam_reference.json</c>, from tools/parity/indextts25_beam_reference.py). Needs the
/// 2.5 checkpoint and aux models under <c>HARTSY_INDEXTTS_MODELS_ROOT</c> and the recorded clip in
/// <c>INDEXTTS2_BEAM_REF_WAV</c> (checked against the fixture's SHA-256). Runs on the CPU backend.</summary>
/// <remarks>The whole code sequence is asserted. The C# resamplers approximate librosa's soxr, so another clip can still
/// flip a near-tie late in a sequence; <see cref="Parity.IndexTts25GptDecodeParityTests"/> localizes such a drift.
/// The search itself is checked exactly against transformers in <see cref="BeamSearchTests"/>.</remarks>
[Trait("Category", "Integration")]
public sealed class IndexTts25BeamParityTests(ITestOutputHelper output)
{
    [Fact]
    public async Task GreedyBeamCodes_MatchUpstream()
    {
        string root = IndexTtsFrontendFixture.ModelsRoot;
        string v25 = Path.Combine(root, "IndexTeam--IndexTTS-2.5");
        string w2v = Path.Combine(root, "facebook--w2v-bert-2.0", "model.safetensors");
        string campplus = Path.Combine(root, "funasr--campplus", "campplus_cn_common.bin");
        string bigVgan = Path.Combine(root, "nvidia--bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt");
        string refWav = Environment.GetEnvironmentVariable("INDEXTTS2_BEAM_REF_WAV") ?? "";
        string[] files = ["multilingual_zh_ja_yue_char_del.tiktoken", "gpt.pth", "s2mel.pth", "codec.pth", "wav2vec2bert_stats.pt"];
        if (!RealWeightGate.Require(output.WriteLine, [.. files.Select(f => Path.Combine(v25, f)), w2v, campplus, bigVgan, refWav])) return;

        using JsonDocument fixture = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "indextts25_beam_reference.json")));
        string sha = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(refWav)));
        Assert.True(sha == fixture.RootElement.GetProperty("reference_wav_sha256").GetString(), "INDEXTTS2_BEAM_REF_WAV is not the recorded clip.");

        using IndexTts2Pipeline pipeline = await IndexTts2Pipeline.LoadAsync(
            Path.Combine(v25, files[0]), Path.Combine(v25, files[1]), Path.Combine(v25, files[2]), Path.Combine(v25, files[3]),
            w2v, Path.Combine(v25, files[4]), campplus, bigVgan, IndexTtsTestNormalizers.Resolve());
        WavFile.DecodedAudio audio = WavFile.Read(refWav);
        using CpuBackend backend = new();
        using IndexTts2Reference reference = pipeline.PrepareReference(backend, audio.Channels[0], audio.SampleRate);

        foreach (JsonElement c in fixture.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            int beams = c.GetProperty("num_beams").GetInt32();
            IndexTts2Options opts = new() { NumBeams = beams, DoSample = false, Language = c.GetProperty("lang").GetString() };
            List<int[]> segments = pipeline.SegmentText(text, opts);
            Assert.Single(segments);
            (Tensor emoVec, bool owned) = pipeline.ResolveEmotion(backend, opts, text, reference, new IndexTts2Timings());
            try
            {
                int[] codes = pipeline.GenerateCodes(backend, reference, emoVec, segments[0], opts, seed: 0);
                int[] expected = [.. c.GetProperty("codes").EnumerateArray().Select(static e => e.GetInt32())];
                if (expected.Length > 0 && expected[^1] == 8193) expected = expected[..^1];
                int same = 0;
                for (int i = 0; i < Math.Min(codes.Length, expected.Length); i++) if (codes[i] == expected[i]) same++;
                output.WriteLine($"{text} beams={beams}: {codes.Length} codes (upstream {expected.Length}), {same} equal by position");
                output.WriteLine("  engine:   " + string.Join(',', codes));
                output.WriteLine("  upstream: " + string.Join(',', expected));
                Assert.Equal(expected, codes);
            }
            finally
            {
                if (owned) emoVec.Dispose();
            }
        }
    }
}
