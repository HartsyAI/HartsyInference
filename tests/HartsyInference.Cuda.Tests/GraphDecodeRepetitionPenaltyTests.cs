using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Correctness gate for on-device repetition penalty in graph decode (see
/// <see cref="TextGenerationPipeline.GenerateGraphDecode"/> and
/// <see cref="HartsyInference.LLM.Transformer.GenericTransformer.ForwardGraphDecodeStep"/>). Repetition penalty
/// is the only sampler stage graph decode replicates — temperature/top-k/top-p/min-p can never change which
/// token wins a greedy argmax, but repetition penalty can, and prior to this it was silently ignored by graph
/// decode (raw unpenalized argmax). Gated on <c>HARTSY_TEST_GGUF_MODELS</c> (needs a real checkpoint) and CUDA
/// availability; skips cleanly otherwise.</summary>
[Collection("CudaSerial")]
public sealed class GraphDecodeRepetitionPenaltyTests
{
    private readonly ITestOutputHelper _output;
    public GraphDecodeRepetitionPenaltyTests(ITestOutputHelper output) => _output = output;

    private const string Prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. Continue this pattern:";

    private static string[] ModelPaths()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSY_TEST_GGUF_MODELS");
        if (string.IsNullOrWhiteSpace(env)) return [];
        return [.. env
            .Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(File.Exists)];
    }

    [Fact]
    public void GraphDecode_WithRepetitionPenalty_MatchesEagerPath()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string[] models = ModelPaths();
        if (models.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_GGUF_MODELS not set"); return; }

        List<string> failures = [];
        foreach (string path in models)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: false);
            if (!model.Transformer.SupportsGraphDecode(backend))
            {
                _output.WriteLine($"SKIPPED (not graph-decode-eligible): {Path.GetFileName(path)}");
                continue;
            }

            TextGenerationPipeline pipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);
            SamplingOptions sampling = SamplingOptions.Default with { Greedy = true, RepetitionPenalty = 1.3f };

            GenerationResult eager = pipeline.Generate(new GenerationRequest
            {
                Prompt = Prompt,
                MaxTokens = 32,
                Sampling = sampling,
                GraphDecode = false,
            });
            GenerationResult graphed = pipeline.Generate(new GenerationRequest
            {
                Prompt = Prompt,
                MaxTokens = 32,
                Sampling = sampling,
                GraphDecode = true,
            });

            string name = $"{Path.GetFileName(path)} (arch={model.Architecture})";
            bool tokensMatch = eager.TokenIds.SequenceEqual(graphed.TokenIds);
            _output.WriteLine($"{name}: eager=[{string.Join(",", eager.TokenIds)}]");
            _output.WriteLine($"{name}: graph=[{string.Join(",", graphed.TokenIds)}]");
            if (!tokensMatch)
                failures.Add($"{name}: graph-decode token ids diverge from eager path with RepetitionPenalty=1.3");

            // Also confirm the penalty actually DID something vs penalty=1.0 graph decode (i.e. this isn't a
            // false-positive match because the model just didn't repeat on this prompt in the first place).
            GenerationResult graphedNoPenalty = pipeline.Generate(new GenerationRequest
            {
                Prompt = Prompt,
                MaxTokens = 32,
                Sampling = sampling with { RepetitionPenalty = 1.0f },
                GraphDecode = true,
            });
            if (graphedNoPenalty.TokenIds.SequenceEqual(graphed.TokenIds))
                failures.Add($"{name}: graph decode produced IDENTICAL output with and without repetition penalty — penalty kernel may not be engaging");
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
