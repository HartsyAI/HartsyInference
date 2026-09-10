using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.BenchmarkRunner.Contracts;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Compiled modality adapter; execution always goes through Engine and timing stays in the common worker.</summary>
public interface IBenchmarkAdapter
{
    Modality Modality { get; }

    Task<BenchmarkOutput> GenerateAsync(InferenceEngine engine, ModelSpec spec, CaseDefinition definition, string prompt, int input,
        string selector, CancellationToken cancel);
}
