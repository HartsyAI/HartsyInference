using HartsyInference.Audio.Models.Haru;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>Haru's language model (stream-summed input, Qwen3.5 trunk, text logits, F32 PAD/EOP logits) against the upstream PyTorch
/// model (<c>tools/haru/haru_reference.py</c>, CPU F32) on a tiny random checkpoint: the prompt in one call, then single positions.</summary>
public sealed unsafe class HaruLmReferenceTests(ITestOutputHelper output)
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Haru");

    [Fact]
    public void Prompt_AndSingleSteps_MatchUpstream()
    {
        using HaruReferenceModel model = new(Path.Combine(Dir, "haru_tiny.safetensors"), Path.Combine(Dir, "config.json"), capacity: 64);
        (float hidden, float logits, float stream) = model.CompareLanguageModel(Path.Combine(Dir, "haru_tiny_expected.safetensors"));
        output.WriteLine($"hidden states {hidden:E3}, text logits {logits:E3}, PAD/EOP logits {stream:E3}");
        Assert.True(hidden < 5e-5f, $"hidden states: relative error {hidden:E3}");
        Assert.True(logits < 5e-5f, $"text logits: relative error {logits:E3}");
        Assert.True(stream < 5e-5f, $"PAD/EOP logits: relative error {stream:E3}");
    }

    [Fact]
    public void Reset_RestartsTheConversation()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        using HaruLm lm = new(config, tensors, capacity: 64);
        using CpuBackend backend = new();
        int[] tokens = [3, 4, 5];
        using Tensor embeds = lm.Embed(backend, tokens, [], [], [], []);
        using Tensor first = lm.Forward(backend, embeds, 3);
        float[] before = HaruReferenceModel.Read(first);
        Assert.Equal(3, lm.Position);
        lm.Reset();
        Assert.Equal(0, lm.Position);
        using Tensor again = lm.Forward(backend, embeds, 3);
        Assert.Equal(before, HaruReferenceModel.Read(again));
    }

    [Fact]
    public void Embed_RefusesLatentsOfTheWrongLength()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        using HaruLm lm = new(config, tensors, capacity: 8);
        using CpuBackend backend = new();
        Assert.Throws<ArgumentException>(() => lm.Embed(backend, [1, 2], new float[3], [], [], []));
        Assert.Throws<ArgumentException>(() => lm.Embed(backend, [], [], [], [], []));
        Assert.Throws<ArgumentException>(() => lm.Embed(backend, [1, 2], new float[2 * config.LatentDim], new bool[3], [], []));
    }

    [Fact]
    public void Forward_RefusesAnInputWhoseRowsAreNotTheClaimedCount()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using HaruLm lm = new(config, weights.GetAllTensors(), capacity: 8);
        using CpuBackend backend = new();
        using Tensor embeds = lm.Embed(backend, [1, 2, 3], [], [], [], []);
        Assert.Throws<ArgumentException>(() => lm.Forward(backend, embeds, 2));
    }

    [Fact]
    public void Logits_OfTheLastRowsOfAPrefill_MatchTheFullProjection()
    {
        using SafeTensorsLoader weights = new();
        weights.Load(Path.Combine(Dir, "haru_tiny.safetensors"));
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using HaruLm lm = new(config, weights.GetAllTensors(), capacity: 16);
        using CpuBackend backend = new();
        int vocab = config.Lm.VocabSize;
        using Tensor embeds = lm.Embed(backend, [3, 4, 5, 6], [], [], [], []);
        using Tensor states = lm.Forward(backend, embeds, 4);
        float[] full = HaruReferenceModel.Read(lm.Logits(backend, states, 4));
        Assert.Equal(full[(3 * vocab)..], HaruReferenceModel.Read(lm.Logits(backend, states, 1)));
        Assert.Equal(full[(2 * vocab)..], HaruReferenceModel.Read(lm.Logits(backend, states, 2)));
        Assert.Throws<ArgumentOutOfRangeException>(() => lm.Logits(backend, states, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => lm.Logits(backend, states, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => lm.StreamLogits(states, 4, [config.PadTokenId], new float[1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => lm.StreamLogits(states, 0, [vocab], new float[1]));
    }
}
