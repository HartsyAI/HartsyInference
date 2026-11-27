using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>The Hy4 image route below the TextService: the vision tower and the expanded placeholder run prefill the tiny model to the HF dump's image
/// logits, and the generator samples a reply from them. The dump's image case feeds the dumped patch values straight to the tower.</summary>
public sealed class Hy4FlashVisionGeneratorTests
{
    private readonly ITestOutputHelper _output;

    public Hy4FlashVisionGeneratorTests(ITestOutputHelper output) => _output = output;

    private static string DataDir
    {
        get
        {
            string? dir = AppContext.BaseDirectory;
            while (dir is not null && !Directory.Exists(Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data"))) dir = Path.GetDirectoryName(dir);
            return Path.Combine(dir ?? throw new DirectoryNotFoundException("tests/python-reference/hy4_flash/data not found."), "tests", "python-reference", "hy4_flash", "data");
        }
    }

    private static float[] Npy(string name) => NpyReader.ReadFloat32(Path.Combine(DataDir, name + ".npy")).Data;

    private static int[] ImageIds()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDir, "inputs_image.json")));
        return [.. document.RootElement.GetProperty("input_ids").EnumerateArray().Select(static e => e.GetInt32())];
    }

    private static Hy4FlashPatchedImage DumpedImage() => new(Npy("image__pixel_values"), 4, 4, 16, 16);

    [Fact]
    public void PrefillOfTheExpandedImagePrompt_MatchesTheHfDumpLogits()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using CpuBackend backend = new();
        using Hy4FlashCheckpoint checkpoint = Hy4FlashCheckpoint.Open(DataDir);
        using Hy4FlashModel model = new(backend, config, checkpoint, 64);
        using Hy4FlashVisionTower tower = new(backend, config.Vision, checkpoint);
        Hy4FlashVisionGenerator generator = new(model, tower, new FakeTokenizer());

        // The dump's last position is the last prompt token, and its logits are the vocabulary row the generator returns.
        float[] expected = Npy("image__logits");
        int vocab = config.VocabSize;
        float[] lastRow = expected.AsSpan(expected.Length - vocab, vocab).ToArray();
        float[] actual = generator.PrefillLogits(ImageIds(), [DumpedImage()]);

        float maxRelative = 0;
        float scale = lastRow.Select(MathF.Abs).Max();
        for (int i = 0; i < vocab; i++) maxRelative = MathF.Max(maxRelative, MathF.Abs(lastRow[i] - actual[i]) / scale);
        _output.WriteLine($"image prefill logits: max relative difference {maxRelative:E2}");
        Assert.True(maxRelative < 1e-5f, $"the prefill logits differ from the HF dump by {maxRelative:E2}");
    }

    [Fact]
    public void Generate_SamplesAReplyWithinTheTokenLimit_AndIsDeterministicGreedy()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using CpuBackend backend = new();
        using Hy4FlashCheckpoint checkpoint = Hy4FlashCheckpoint.Open(DataDir);
        using Hy4FlashModel model = new(backend, config, checkpoint, 64);
        using Hy4FlashVisionTower tower = new(backend, config.Vision, checkpoint);
        Hy4FlashVisionGenerator generator = new(model, tower, new FakeTokenizer());

        Hy4FlashVisionReply first = generator.Generate(ImageIds(), [DumpedImage()], SamplingOptions.GreedyPreset, 4, null, CancellationToken.None);
        Hy4FlashVisionReply second = generator.Generate(ImageIds(), [DumpedImage()], SamplingOptions.GreedyPreset, 4, null, CancellationToken.None);
        Assert.Equal(ImageIds().Length, first.PromptTokens);
        Assert.True(first.Tokens <= 4);
        Assert.Equal(first.Text, second.Text);
        Assert.Equal(first.Tokens, second.Tokens);
    }

    /// <summary>Only the ids the generator reads matter here: the stop ids (none) and the decode of the sampled ids.</summary>
    private sealed class FakeTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => EncodeOrdinary(text);
        public int[] EncodeOrdinary(string text) => text.Select(c => (int)c).ToArray();
        public string Decode(IReadOnlyList<int> ids) => string.Concat(ids.Select(i => (char)(i % 0xD7FF + 1)));
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => null;
        public IReadOnlyList<int> StopIds => Array.Empty<int>();
        public string? BosToken => null;
        public string? EosToken => null;
    }
}
