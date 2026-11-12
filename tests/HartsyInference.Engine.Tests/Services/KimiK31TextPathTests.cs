using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Tests.Services;

public sealed class KimiK31TextPathTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-kimi-text-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Synthetic_checkpoint_loads_through_the_host_reference_route_on_the_cpu()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("loads"));
        Assert.True(HfTextDirectoryLoader.IsKimiK31(info));
        Assert.True(HfTextDirectoryLoader.IsHostReferenceFamily(info));

        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);

        Assert.IsType<KimiK31TextModel>(model);
        Assert.Equal(KimiSyntheticCheckpoint.Vocab, model.Generation.Info.VocabSize);
        Assert.Equal(KimiSyntheticCheckpoint.BosId, model.Tokenizer.BosId);
        Assert.Equal(KimiSyntheticCheckpoint.EosId, model.Tokenizer.EosId);
    }

    [Fact]
    public void Prefill_projects_finite_logits_for_the_last_token()
    {
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = Load(backend, "logits");
        using ISequenceState state = model.Generation.CreateSequenceState(new SequenceStateOptions(16));

        using Tensor hidden = model.Generation.Prefill(new PrefillChunk(new[] { 1, 2, 3 }, 0, LastRowOnly: true), state);
        using Tensor logits = model.Generation.ProjectLogits(hidden, 1);

        Assert.Equal((long)KimiSyntheticCheckpoint.Vocab, logits.ElementCount);
        ReadOnlySpan<float> values = logits.AsReadOnlySpan<float>();
        for (int i = 0; i < values.Length; i++) Assert.True(float.IsFinite(values[i]), $"logit[{i}] is not finite");
        Assert.Equal(3, state.Length);
    }

    [Fact]
    public void Text_pipeline_generates_deterministically_on_the_cpu()
    {
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = Load(backend, "generates");
        TextGenerationPipeline pipeline = new(model.Generation, model.Tokenizer, model.Template);
        GenerationRequest request = new()
        {
            RawTokenIds = [1, 2, 3],
            MaxTokens = 3,
            Sampling = new SamplingOptions { Temperature = 0f },
        };

        GenerationResult first = pipeline.Generate(request);
        GenerationResult second = pipeline.Generate(request);

        Assert.True(first.TokenIds.Count <= 3);
        Assert.True(first.TokenIds.Count == 3 || first.StoppedOnStopToken);
        Assert.All(first.TokenIds, id => Assert.InRange(id, 0, KimiSyntheticCheckpoint.Vocab - 1));
        Assert.Equal(first.TokenIds, second.TokenIds);
    }

    [Fact]
    public void Rolling_back_and_replaying_reproduces_the_same_hidden_row()
    {
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = Load(backend, "replays");
        int dim = KimiSyntheticCheckpoint.Hidden;
        using ISequenceState state = model.Generation.CreateSequenceState(new SequenceStateOptions(16));

        using Tensor full = model.Generation.Prefill(new PrefillChunk(new[] { 1, 2, 3 }, 0), state);
        float[] original = full.AsReadOnlySpan<float>().Slice(2 * dim, dim).ToArray();
        state.Truncate(2);
        using Tensor replayed = model.Generation.Prefill(new PrefillChunk(new[] { 3 }, 2, LastRowOnly: true), state);
        ReadOnlySpan<float> again = replayed.AsReadOnlySpan<float>();

        Assert.Equal(3, state.Length);
        for (int i = 0; i < dim; i++) Assert.InRange(Math.Abs(original[i] - again[i]), 0f, 1e-4f);
    }

    [Fact]
    public void Checkpoint_with_a_vision_tower_loads_text_only_and_refuses_image_input_by_name()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("vision", withVision: true));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => KimiK31Refusals.RefuseImage(model, hasImage: true));
        Assert.Contains("vision", refused.Message, StringComparison.OrdinalIgnoreCase);
        KimiK31Refusals.RefuseImage(model, hasImage: false);
    }

    [Theory]
    [InlineData("cuda:0+cuda:1", 0, 1)]
    [InlineData("cuda:0", 2, 1)]
    [InlineData("cuda:0", 0, 2)]
    public void Multi_gpu_placement_is_refused_by_name(string deviceKey, int shardDevices, int tensorParallelDegree)
    {
        NotSupportedException refused = Assert.Throws<NotSupportedException>(() =>
            KimiK31Refusals.RefuseMultiGpuPlacement(deviceKey, shardDevices, tensorParallelDegree));
        Assert.Contains("multi-GPU", refused.Message);
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("cuda:0")]
    [InlineData("vulkan:1")]
    public void A_single_device_request_is_not_refused(string deviceKey)
    {
        KimiK31Refusals.RefuseMultiGpuPlacement(deviceKey, shardDeviceCount: 0, tensorParallelDegree: 1);
    }

    [Fact]
    public void Missing_chat_template_is_refused_by_name_before_any_weight_loads()
    {
        string dir = WriteCheckpoint("no-template");
        File.Delete(Path.Combine(dir, "chat_template.jinja"));
        HfCheckpointInfo info = Probe(dir);

        HartsyInferenceException refused = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.RequireSupported(info));
        Assert.Contains("chat_template.jinja", refused.Message);
    }

    [Fact]
    public void A_moonshot_layout_without_tokenizer_json_is_refused_by_name()
    {
        string dir = WriteCheckpoint("tiktoken-only");
        File.Delete(Path.Combine(dir, "tokenizer.json"));
        HfCheckpointInfo info = Probe(dir);

        HartsyInferenceException refused = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.RequireSupported(info));
        Assert.Contains("tiktoken.model", refused.Message);
    }

    private IHostReferenceTextModel Load(IBackend backend, string name) =>
        HfTextDirectoryLoader.LoadHostReference(Probe(WriteCheckpoint(name)), backend);

    private string WriteCheckpoint(string name, bool withVision = false)
    {
        string dir = Path.Combine(_root, name);
        KimiSyntheticCheckpoint.Write(dir, withVision: withVision);
        return dir;
    }

    private static HfCheckpointInfo Probe(string dir) =>
        HfCheckpointDirectory.TryProbe(dir) ?? throw new InvalidOperationException($"'{dir}' is not a Hugging Face checkpoint directory.");
}
