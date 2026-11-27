using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Experts;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

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
    [InlineData("hidden_act", "\"gelu\"", "FFN activation 'gelu'")]
    [InlineData("compress_ratios", "[4, -2, -2]", "layer 0 attention with compress ratio 4")]
    [InlineData("compress_ratios", "[2]", "compress_ratios with 1 entries for 2 decoder layers")]
    public void Text_path_refuses_an_unrunnable_checkpoint_by_name(string field, string value, string named)
    {
        string dir = WriteCheckpoint("refuses-" + field);
        File.WriteAllText(Path.Combine(dir, "config.json"), Ernie5SyntheticCheckpoint.ConfigJson(new Dictionary<string, string> { [field] = value }));
        HfCheckpointInfo info = Probe(dir);

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.RequireSupported(info));

        Assert.Contains(named, ex.Message);
    }

    [Fact]
    public void Text_path_keeps_layers_resident_on_a_gpu_and_streams_them_on_the_cpu()
    {
        Assert.False(HfTextDirectoryLoader.Ernie5OptionsFor(DeviceKind.Cpu).ResidentWeights);
        Assert.True(HfTextDirectoryLoader.Ernie5OptionsFor(DeviceKind.Cuda(1)).ResidentWeights);
        Assert.True(HfTextDirectoryLoader.Ernie5OptionsFor(DeviceKind.Vulkan()).ResidentWeights);
        Assert.All(new[] { DeviceKind.Cpu, DeviceKind.Cuda() }, d =>
        {
            Ernie5LoadOptions options = HfTextDirectoryLoader.Ernie5OptionsFor(d);
            Assert.Equal(HfTextDirectoryLoader.MaxSequenceTokens, options.MaxSequenceTokens);
            Assert.False(options.Mtp);
        });
    }

    [Fact]
    public void Text_path_takes_a_gpu_layer_split_and_refuses_tensor_parallelism_or_a_cpu_stage_by_name()
    {
        TextService.RequireRunnableErnie5Placement([], tensorParallelDegree: 1);
        TextService.RequireRunnableErnie5Placement(["cuda:0", "cuda:1"], tensorParallelDegree: 1);

        NotSupportedException cpu = Assert.Throws<NotSupportedException>(() => TextService.RequireRunnableErnie5Placement(["cuda:0", "cpu"], 1));
        Assert.Contains("'cpu' is not (cuda:0 + cpu)", cpu.Message);
        NotSupportedException mixed = Assert.Throws<NotSupportedException>(() => TextService.RequireRunnableErnie5Placement(["cuda:0", "vulkan:0"], 1));
        Assert.Contains("mixed GPU kinds", mixed.Message);
        NotSupportedException tp = Assert.Throws<NotSupportedException>(() => TextService.RequireRunnableErnie5Placement([], tensorParallelDegree: 2));
        Assert.Contains("tensor-parallel degree is 2", tp.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Text_service_loads_on_the_requested_cpu_and_generates(bool continuousBatching)
    {
        ServiceRun run = await GenerateThroughService(WriteCheckpoint("service-" + continuousBatching), "cpu", continuousBatching);

        Assert.True(run.Device.IsCpu);
        Assert.Equal(continuousBatching, run.Scheduled);
        Assert.InRange(run.Result.CompletionTokens, 1, 4);
        // Both paths time the request: the scheduler reports the same prefill/decode split the pipeline does.
        Assert.True(run.Result.PrefillMilliseconds > 0, $"Prefill time was {run.Result.PrefillMilliseconds} ms.");
        // Every generated token is added in a decode round, so a request with a completion token has decode time.
        Assert.True(run.Result.DecodeMilliseconds > 0, $"Decode time was {run.Result.DecodeMilliseconds} ms.");
    }

    [Fact]
    public void Base_checkpoint_without_a_chat_template_prompts_with_the_post_processor_prefix()
    {
        string dir = Path.Combine(_root, "base-prefix");
        Ernie5SyntheticCheckpoint.Write(dir, baseLayout: true);
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(Probe(dir), backend);

        int[] ids = model.Template.Encode(model.Tokenizer, [new ChatMessage("user", "hi")], addGenerationPrompt: true);

        // tokenizer("hi") on the base layout: the post-processor's end token, then the merged "hi".
        Assert.IsType<RawCompletionTemplate>(model.Template);
        Assert.Equal([Ernie5SyntheticCheckpoint.EosId, 258], ids);
    }

    [Fact]
    public void Base_checkpoint_with_a_malformed_post_processor_names_the_tokenizer_file()
    {
        string dir = Path.Combine(_root, "base-malformed");
        Ernie5SyntheticCheckpoint.Write(dir, baseLayout: true);
        string noSpecials = "{\"type\":\"TemplateProcessing\",\"single\":[{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],\"pair\":[]}";
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), Ernie5SyntheticCheckpoint.TokenizerJson(noSpecials));
        using CpuBackend backend = new();

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.LoadHostReference(Probe(dir), backend));

        Assert.Contains("tokenizer.json", ex.Message);
        Assert.Contains("special_tokens", ex.Message);
    }

    [Fact]
    public async Task Base_checkpoint_without_a_chat_template_generates_through_the_text_service()
    {
        string dir = Path.Combine(_root, "base-service");
        Ernie5SyntheticCheckpoint.Write(dir, baseLayout: true);

        ServiceRun run = await GenerateThroughService(dir, "cpu", continuousBatching: false);

        // "Hello" is five byte tokens after the one prefix token.
        Assert.Equal(6, run.Result.PromptTokens);
        Assert.InRange(run.Result.CompletionTokens, 1, 4);
    }

    /// <summary>The scheduler prefills as the pipeline does (one projected row, the same first token), so a greedy request gives the same text on both.</summary>
    [Fact]
    public async Task Continuous_batching_generates_what_the_pipeline_generates()
    {
        string checkpoint = WriteCheckpoint("service-parity");
        ServiceRun pipeline = await GenerateThroughService(checkpoint, "cpu", continuousBatching: false);
        ServiceRun scheduled = await GenerateThroughService(checkpoint, "cpu", continuousBatching: true);

        Assert.False(pipeline.Scheduled);
        Assert.True(scheduled.Scheduled);
        Assert.Equal(pipeline.Result.Text, scheduled.Result.Text);
        Assert.Equal(pipeline.Result.CompletionTokens, scheduled.Result.CompletionTokens);
        Assert.Equal(pipeline.Result.PromptTokens, scheduled.Result.PromptTokens);
    }

    /// <summary>What one request through the engine's Text service did: its result, whether the slot built a batching scheduler, the device
    /// the slot's backend runs on, and, when an expert cache is attached, where the routed rows ran and what the cache holds.</summary>
    internal sealed record ServiceRun(TextResult Result, bool Scheduled, DeviceKind Device, MoeOffloadStats? Offload,
        ExpertCacheStats? ExpertCache);

    /// <summary>Loads <paramref name="directory"/> through the engine's Text service on <paramref name="device"/> and generates up to four tokens,
    /// with the continuous-batching knob pinned for the load and the request.</summary>
    internal static async Task<ServiceRun> GenerateThroughService(string directory, string device, bool continuousBatching)
    {
        using IDisposable knobs = KnobProfileScope.Push(KnobProfile.Create("ernie5-text-service").With(EngineKnobs.ContinuousBatching, continuousBatching));
        using InferenceEngine engine = new(device);
        ModelSpec spec = new() { Requested = directory, Modality = Modality.Text, LocalPath = directory };
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "Hello" }],
            Device = device,
            Temperature = 0,
            MaxTokens = 4,
        };
        TextResult result = await engine.Text.GenerateAsync(spec, request);
        TextDeviceSlot slot = ((TextService)engine.Text).SlotFor(device) ?? throw new InvalidOperationException($"No slot ran on '{device}'.");
        // Read here: disposing the engine unloads the slot, which drops the offload.
        return new ServiceRun(result, slot.Scheduler is not null, slot.Backend!.Device, slot.ExpertOffload?.Stats,
            slot.ExpertOffload?.CacheBase.Stats);
    }

    [Fact]
    public void Text_path_accepts_the_synthetic_checkpoint_at_the_support_gate()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("accepts"));

        HfTextDirectoryLoader.RequireSupported(info);
    }

    [Fact]
    public void Forward_matches_the_hidden_state_and_logits_split()
    {
        string dir = WriteCheckpoint("split");
        using CpuBackend backend = new();
        using Ernie5Model model = Ernie5Model.Load(dir, backend, new Ernie5LoadOptions { MaxSequenceTokens = 16 });
        int[] ids = [1, 2, 3];

        float[] direct;
        using (Ernie5SequenceState a = model.CreateState(16)) direct = model.Forward(ids, a, allRows: true);
        float[] split;
        using (Ernie5SequenceState b = model.CreateState(16)) split = model.Logits(model.ForwardHidden(ids, b, allRows: true), ids.Length);

        Assert.Equal(direct, split);
    }

    [Fact]
    public void Rejected_token_id_leaves_the_sequence_intact()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("rejects"));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        using ISequenceState state = model.Generation.CreateSequenceState(new SequenceStateOptions(16));
        using (Tensor first = model.Generation.Prefill(new PrefillChunk(new[] { 1, 2 }, 0), state)) { }

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Generation.Prefill(new PrefillChunk(new[] { 3, Ernie5SyntheticCheckpoint.Vocab }, 2), state));

        Assert.Equal(2, state.Length);
        using Tensor next = model.Generation.Prefill(new PrefillChunk(new[] { 3 }, 2, LastRowOnly: true), state);
        Assert.Equal(3, state.Length);
    }

    [Fact]
    public void Rolling_back_across_two_appends_reproduces_the_second_append()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("segments"));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        int dim = Ernie5SyntheticCheckpoint.Hidden;
        using ISequenceState state = model.Generation.CreateSequenceState(new SequenceStateOptions(16));
        using (Tensor head = model.Generation.Prefill(new PrefillChunk(new[] { 1, 2 }, 0), state)) { }
        using Tensor tail = model.Generation.Prefill(new PrefillChunk(new[] { 3 }, 2, LastRowOnly: true), state);
        float[] original = tail.AsReadOnlySpan<float>().ToArray();

        state.Truncate(2);
        using Tensor replayed = model.Generation.Prefill(new PrefillChunk(new[] { 3 }, 2, LastRowOnly: true), state);

        Assert.Equal(dim, original.Length);
        for (int i = 0; i < dim; i++) Assert.InRange(Math.Abs(original[i] - replayed.AsReadOnlySpan<float>()[i]), 0f, 1e-4f);
    }

    [Fact]
    public void Prompt_text_renders_through_the_template_and_generates()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("prompt"));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        TextGenerationPipeline pipeline = new(model.Generation, model.Tokenizer, model.Template);

        GenerationResult result = pipeline.Generate(new GenerationRequest
        {
            Prompt = "hi",
            MaxTokens = 2,
            Sampling = new SamplingOptions { Temperature = 0f },
        });

        Assert.True(result.TokenIds.Count <= 2);
        Assert.True(result.PromptTokens > 0);
    }

    [Fact]
    public void Chat_template_receives_the_bos_literal_as_its_special_id()
    {
        HfCheckpointInfo info = Probe(WriteCheckpoint("template-bos"));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);

        int[] ids = model.Template.Encode(model.Tokenizer, [new ChatMessage("user", "hi")], addGenerationPrompt: false);

        Assert.Equal(Ernie5SyntheticCheckpoint.BosId, ids[0]);
    }

    [Fact]
    public void Generation_config_stop_ids_reach_the_tokenizer_once_each()
    {
        string dir = WriteCheckpoint("stop-ids");
        File.WriteAllText(Path.Combine(dir, "generation_config.json"), "{\"eos_token_id\": [257, 100, 257]}");
        HfCheckpointInfo info = Probe(dir);
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);

        IReadOnlyList<int> stops = model.Tokenizer.StopIds;

        Assert.Contains(100, stops);
        Assert.Equal(1, stops.Count(id => id == Ernie5SyntheticCheckpoint.EosId));
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
