using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.BenchmarkRunner.Contracts;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Frozen greedy text workload through the native text service.</summary>
public sealed class TextAdapter : IBenchmarkAdapter
{
    public Modality Modality => Modality.Text;

    public async Task<BenchmarkOutput> GenerateAsync(InferenceEngine engine, ModelSpec spec, CaseDefinition definition, string prompt,
        int input, string selector, CancellationToken cancel) => new()
    {
        Text = await engine.Text.GenerateAsync(spec, new TextRequest { Messages = [new TextMessage { Role = TextRole.User,
            Content = prompt }], Temperature = 0, Greedy = true, TopP = 1, TopK = 0, MinP = 0, RepetitionPenalty = 1, MaxTokens = definition
            .MaxTokens, Seed = definition.Seed + input, EnableThinking = false, Device = selector, GraphDecode = false,
            SpeculativeDecode = false, AlwaysFreeMemory = false, }, cancel),
    };
}
