using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The Drama 3 pipeline on the tiny reference model and the tiny ModifiedDAC: streamed audio is exactly the audio of a
/// one-shot run (the codec is causal, so a decode of the frames so far ends where the full decode does), long text runs as chunks
/// that each come back as a turn, and the same seed reproduces the same take.</summary>
public sealed class FishAudioDrama3PipelineTests
{
    private static readonly string Drama3Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "FishAudioDrama3");
    private static readonly string CodecDir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "FishAudioS2");

    private static ModifiedDacConfig TinyCodec() => new()
    {
        LatentDim = 64, CodebookDim = 8, NumResidualCodebooks = 3, SemanticCodebookSize = 32, ResidualCodebookSize = 16,
        DecoderDim = 32, DecoderRates = [2, 2, 2, 2], TransformerLayers = 2, TransformerHeads = 4, TransformerHeadDim = 16,
        TransformerIntermediate = 96, TransformerWindow = 8,
        EncoderDim = 4, EncoderRates = [2, 2, 2, 2], EncoderTransformerLayers = [0, 0, 0, 2],
    };

    [Fact]
    public void StreamedAudio_EqualsTheOneShotAudio_AcrossChunks()
    {
        string text = string.Join(" ", Enumerable.Range(0, 14).Select(i => $"Sentence {i} goes on for a little while."));
        using CpuBackend backend = new();
        using Loaded loaded = new();
        FishAudioDrama3Pipeline.Request request = new() { Text = text, MaxFrames = 40, Seed = 11 };
        FishAudioDrama3Pipeline.Result oneShot = loaded.Pipeline.Synthesize(backend, request);
        List<float> streamed = [];
        int emissions = 0;
        FishAudioDrama3Pipeline.Result withStream = loaded.Pipeline.Synthesize(backend, request, pcm =>
        {
            emissions++;
            streamed.AddRange(pcm);
        });
        Assert.True(oneShot.Turns.Count > 1, "the text should run as several chunks");
        Assert.Equal(oneShot.Audio, withStream.Audio);
        Assert.Equal(oneShot.Audio, streamed);
        Assert.True(emissions >= withStream.Turns.Count);
        Assert.Equal(string.Join(" ", oneShot.Turns.Select(t => t.Text)), text);
    }

    [Fact]
    public void CallerHistory_AndReferences_ReachThePrompt()
    {
        using CpuBackend backend = new();
        using Loaded loaded = new();
        int[,] codes = { { 1, 2, 3 }, { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 0 } };
        // A random tiny model may end a take at once; take the first seed whose plain take carries audio.
        for (int seed = 0; seed < 32; seed++)
        {
            FishAudioDrama3Pipeline.Request plain = new() { Text = "Say it again.", MaxFrames = 24, Seed = seed, TopP = 1f };
            FishAudioDrama3Pipeline.Result take = loaded.Pipeline.Synthesize(backend, plain);
            if (take.Turns.Count == 0 || take.Turns[0].Frames < 4) continue;
            Assert.Equal(take.Audio, loaded.Pipeline.Synthesize(backend, plain).Audio);
            FishAudioDrama3Pipeline.Request revised = plain with { History = [new FishAudioDrama3Turn("Say it once.", codes)] };
            FishAudioDrama3Pipeline.Request cloned = plain with { References = [new FishAudioDrama3Turn("A voice.", codes)] };
            Assert.NotEqual(take.Audio, loaded.Pipeline.Synthesize(backend, revised).Audio);
            Assert.NotEqual(take.Audio, loaded.Pipeline.Synthesize(backend, cloned).Audio);
            return;
        }
        Assert.Fail("No seed produced a take with audio.");
    }

    [Theory]
    [InlineData(FishAudioDrama3Config.MaxPromptRows + FishAudioDrama3Config.DefaultMaxNewFrames, 600, FishAudioDrama3Config.MaxPromptRows)]
    [InlineData(FishAudioDrama3Config.MaxPromptRows + FishAudioDrama3Config.DefaultMaxNewFrames, 2048, FishAudioDrama3Config.MaxPromptRows)]
    [InlineData(2048, 40, 2008)]
    [InlineData(2048, 5000, 1024)]
    public void PromptRowLimit_LeavesTheRequestedFramesRoom(int capacity, int requested, int expected) =>
        Assert.Equal(expected, FishAudioDrama3Pipeline.PromptRowLimit(capacity, requested));

    // One id in 3..32 per word: text rows never fall in the semantic band (40..47) or on eos (2), and prompts stay within the tiny
    // model's 256 positions.
    private static int[] Encode(string literal) =>
        [.. literal.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(word => 3 + word.Sum(c => c) % 30)];

    private sealed class Loaded : IDisposable
    {
        private readonly HfQuantCheckpoint _checkpoint;
        private readonly SafeTensorsLoader _codec = new();

        public Loaded()
        {
            FishAudioDrama3Config config = FishAudioDrama3Config.FromJson(File.ReadAllText(Path.Combine(Drama3Dir, "config.json")));
            _checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(Drama3Dir, "drama3_tiny.safetensors")]);
            _codec.Load(Path.Combine(CodecDir, "modded_dac_tiny.safetensors"));
            Dictionary<string, Tensor> codec = new(StringComparer.Ordinal);
            foreach (string name in _codec.Descriptors.Keys)
            {
                if (!name.StartsWith("ref.", StringComparison.Ordinal) && !name.StartsWith("tap.", StringComparison.Ordinal)) codec[name] = _codec.GetTensor(name);
            }
            Pipeline = new FishAudioDrama3Pipeline(config, TinyCodec(), Encode);
            Pipeline.LoadWeights(FishAudioDrama3Weights.Borrow(_checkpoint), codec, capacity: 2048);
        }

        public FishAudioDrama3Pipeline Pipeline { get; }

        public void Dispose()
        {
            Pipeline.Dispose();
            _checkpoint.Dispose();
            _codec.Dispose();
        }
    }
}
