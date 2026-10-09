using System.Diagnostics;
using HartsyInference.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The CPU serving path on the official checkpoint: the oracle prompt's greedy continuation through the pipeline TextService builds for a
/// V4.1 directory, and through <see cref="InferenceEngine"/> itself. The oracle is the unmodified upstream model in float32, structural mode, 40
/// layers, 16 decode steps (<c>oracle_l40s16/meta.json</c>). Run alone under a memory cap, as <see cref="DeepSeekV41RealWeightsTests"/> describes.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DeepSeekV41CpuServingRealWeightsTests
{
    private static readonly int[] PromptIds = [0, 671, 6102, 294, 8760, 344];
    private static readonly int[] OracleIds = [11111, 16, 660, 270, 3584, 294, 3980, 14, 270, 5214, 6102, 769, 1047, 295, 1623, 12525, 915];

    private readonly ITestOutputHelper _output;

    public DeepSeekV41CpuServingRealWeightsTests(ITestOutputHelper output) => _output = output;

    private bool HaveWeights(out string dir)
    {
        dir = ModelResolver.Resolve(DeepSeekV41Catalog.Id, null, Modality.Text).LocalPath
            ?? Path.Combine(RepoPaths.ModelsRoot(), "llm", "deepseek-v4.1-flash");
        return RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), Path.Combine(dir, "model.safetensors.index.json"));
    }

    [Fact]
    public void PipelineGreedy_MatchesTheOracleIdsExactly()
    {
        if (!HaveWeights(out string dir)) return;
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(dir)
            ?? throw new InvalidOperationException($"'{dir}' is not a Hugging Face checkpoint directory.");
        using CpuBackend backend = new();
        using DeepSeekV41TextModel model = HfTextDirectoryLoader.Load(info, backend);
        Assert.Empty(model.Tokenizer.StopIds.Intersect(OracleIds));
        TextGenerationPipeline pipeline = new(model.Generation, model.Tokenizer, model.Template);
        Stopwatch clock = Stopwatch.StartNew();
        TimeSpan firstToken = TimeSpan.Zero;

        GenerationResult result = pipeline.Generate(new GenerationRequest
        {
            RawTokenIds = PromptIds,
            MaxTokens = OracleIds.Length,
            Sampling = SamplingOptions.GreedyPreset,
            SpeculativeDecode = false,
            GraphDecode = false,
            OnPrefillCompleted = _ => firstToken = clock.Elapsed,
        });
        TimeSpan total = clock.Elapsed;

        _output.WriteLine($"ids [{string.Join(",", result.TokenIds)}] text '{result.Text.Replace("\n", "\\n")}'");
        _output.WriteLine($"time to first token {firstToken.TotalSeconds:F1}s, {(total - firstToken).TotalSeconds / (OracleIds.Length - 1):F2}s per decode token, "
            + $"total {total.TotalSeconds:F1}s, peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / (1L << 30)} GiB");
        Assert.Equal(OracleIds, result.TokenIds);
        Assert.False(result.StoppedOnStopToken);
    }

    [Fact]
    public async Task EngineText_ReturnsTheOracleContinuationAndRepeatsIt()
    {
        if (!HaveWeights(out string dir)) return;
        using FileStream tokenizerStream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerStream, bosToken: DeepSeekV41TextModel.BosLiteral, eosToken: DeepSeekV41TextModel.EosLiteral);
        string expected = tokenizer.Decode(OracleIds);
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = DeepSeekV41Catalog.Id, Modality = Modality.Text, LocalPath = dir };
        TextRequest request = new()
        {
            Messages = [],
            RawTokenIds = PromptIds,
            MaxTokens = OracleIds.Length,
            Greedy = true,
            SpeculativeDecode = false,
            GraphDecode = false,
        };

        TextResult first = await engine.Text.GenerateAsync(spec, request);
        TextResult second = await engine.Text.GenerateAsync(spec, request);

        _output.WriteLine($"engine text '{first.Text.Replace("\n", "\\n")}' prompt {first.PromptTokens} completion {first.CompletionTokens}");
        Assert.Equal(expected, first.Text);
        Assert.Equal(first.Text, second.Text);
        Assert.Equal(OracleIds.Length, first.CompletionTokens);
    }
}
