using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>ERNIE-5.5 with its layers split over several backends (<see cref="Ernie5LoadOptions.Placement"/>): the same logits as one backend, and
/// the placements the model refuses. The device run is in <see cref="Ernie5ModelGpuParityTests"/>.</summary>
public sealed class Ernie5LayerSplitTests : IDisposable
{
    private static readonly int[] Tokens = [1, 2, 3, 4, 5];
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-ernie5-split-" + Guid.NewGuid().ToString("N"));

    public Ernie5LayerSplitTests() => Ernie5SyntheticCheckpoint.Write(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void Two_cpu_stages_give_the_single_backend_logits_exactly()
    {
        using CpuBackend single = new();
        float[] want = Run(single, placement: null);

        using CpuBackend first = new();
        using CpuBackend second = new();
        int layers = Ernie5Config.Load(Path.Combine(_dir, "config.json")).NumHiddenLayers;
        Assert.True(layers >= 2, "the synthetic checkpoint needs two layers to split");
        LlmPlacement split = new([new LlmStage(first, 0, 1), new LlmStage(second, 1, layers)]);
        Assert.Equal(want, Run(second, split));
    }

    [Fact]
    public void A_placement_that_does_not_cover_the_layers_is_refused()
    {
        using CpuBackend cpu = new();
        LlmPlacement one = LlmPlacement.Single(cpu, 1);
        ArgumentException ex = Assert.Throws<ArgumentException>(() => Run(cpu, one));
        Assert.Contains("covers 1 layers", ex.Message);
    }

    [Fact]
    public void The_model_backend_must_be_the_last_stage()
    {
        using CpuBackend first = new();
        using CpuBackend second = new();
        int layers = Ernie5Config.Load(Path.Combine(_dir, "config.json")).NumHiddenLayers;
        LlmPlacement split = new([new LlmStage(first, 0, 1), new LlmStage(second, 1, layers)]);
        ArgumentException ex = Assert.Throws<ArgumentException>(() => Run(first, split));
        Assert.Contains("last stage", ex.Message);
    }

    [Fact]
    public void A_split_without_resident_weights_is_refused()
    {
        using CpuBackend first = new();
        using CpuBackend second = new();
        int layers = Ernie5Config.Load(Path.Combine(_dir, "config.json")).NumHiddenLayers;
        LlmPlacement split = new([new LlmStage(first, 0, 1), new LlmStage(second, 1, layers)]);
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => Run(second, split, resident: false));
        Assert.Contains("needs ResidentWeights", ex.Message);
    }

    // A prefill with every row's logits, then one decode step, so both the chunk and the cache handoff are compared.
    private float[] Run(IBackend backend, LlmPlacement? placement, bool resident = true)
    {
        using Ernie5Model model = Ernie5Model.Load(_dir, backend, new Ernie5LoadOptions
        {
            MaxSequenceTokens = Tokens.Length + 1, ResidentWeights = resident, Placement = placement,
        });
        using Ernie5SequenceState state = model.CreateState(Tokens.Length + 1);
        float[] prefill = model.Forward(Tokens, state, allRows: true);
        float[] decode = model.Forward([Tokens[^1] + 1], state);
        return [.. prefill, .. decode];
    }
}
