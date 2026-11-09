using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

public sealed class Ernie5TextPathTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-ernie5-text-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Synthetic_checkpoint_loads_through_the_host_reference_path()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("loads"));
        Assert.Equal(Ernie5Config.ModelType, info.ModelType);

        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);

        Assert.IsType<Ernie5TextModel>(model);
        Assert.Equal(Ernie5SyntheticCheckpoint.Vocab, model.Generation.Info.VocabSize);
        Assert.Equal(Ernie5SyntheticCheckpoint.BosId, model.Tokenizer.BosId);
        Assert.Equal(Ernie5SyntheticCheckpoint.EosId, model.Tokenizer.EosId);
    }

    [Fact]
    public void Text_pipeline_generates_tokens_on_the_cpu_and_is_deterministic()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("generates"));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        TextGenerationPipeline pipeline = new(model.Generation, model.Tokenizer, model.Template);
        GenerationRequest request = new()
        {
            RawTokenIds = [1, 2, 3],
            MaxTokens = 4,
            Sampling = new SamplingOptions { Temperature = 0f },
        };

        GenerationResult first = pipeline.Generate(request);
        GenerationResult second = pipeline.Generate(request);

        Assert.True(first.TokenIds.Count <= 4);
        Assert.True(first.TokenIds.Count == 4 || first.StoppedOnStopToken);
        Assert.All(first.TokenIds, id => Assert.InRange(id, 0, Ernie5SyntheticCheckpoint.Vocab - 1));
        Assert.Equal(first.TokenIds, second.TokenIds);
    }

    [Fact]
    public void Rolling_back_and_replaying_reproduces_the_same_hidden_row()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("replays"));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        int dim = Ernie5SyntheticCheckpoint.Hidden;
        using ISequenceState state = model.Generation.CreateSequenceState(new SequenceStateOptions(16));

        using Tensor full = model.Generation.Prefill(new PrefillChunk(new[] { 1, 2, 3 }, 0), state);
        float[] original = full.AsReadOnlySpan<float>().Slice(2 * dim, dim).ToArray();
        state.Truncate(2);
        using Tensor replayed = model.Generation.Prefill(new PrefillChunk(new[] { 3 }, 2, LastRowOnly: true), state);
        ReadOnlySpan<float> again = replayed.AsReadOnlySpan<float>();

        Assert.Equal(3, state.Length);
        for (int i = 0; i < dim; i++) Assert.InRange(Math.Abs(original[i] - again[i]), 0f, 1e-4f);
    }

    [Theory]
    [InlineData("num_nextn_predict_layers", "1", "multi-token-prediction layer(s) from layer 2")]
    [InlineData("hidden_act", "\"gelu\"", "FFN activation 'gelu'")]
    [InlineData("compress_ratios", "[4, -2]", "compress ratio 4")]
    public void Text_path_refuses_an_unrunnable_checkpoint_by_name(string field, string value, string named)
    {
        string dir = WriteCheckpoint("refuses-" + field);
        File.WriteAllText(Path.Combine(dir, "config.json"), Ernie5SyntheticCheckpoint.ConfigJson(new Dictionary<string, string> { [field] = value }));
        HfCheckpointInfo info = Probe(dir);

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.RequireSupported(info));

        Assert.Contains(named, ex.Message);
    }

    [Fact]
    public void Text_path_accepts_the_synthetic_checkpoint_at_the_support_gate()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("accepts"));

        HfTextDirectoryLoader.RequireSupported(info);
    }

    private string WriteCheckpoint(string name)
    {
        string dir = Path.Combine(_root, name);
        Ernie5SyntheticCheckpoint.Write(dir);
        return dir;
    }

    private static HfCheckpointInfo Probe(string dir) =>
        HfCheckpointDirectory.TryProbe(dir) ?? throw new InvalidOperationException($"'{dir}' is not a Hugging Face checkpoint directory.");
}
