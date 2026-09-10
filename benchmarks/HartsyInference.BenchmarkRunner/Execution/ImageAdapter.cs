using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.BenchmarkRunner.Contracts;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Frozen image workload through the native image service.</summary>
public sealed class ImageAdapter : IBenchmarkAdapter
{
    public Modality Modality => Modality.Image;

    public async Task<BenchmarkOutput> GenerateAsync(InferenceEngine engine, ModelSpec spec, CaseDefinition definition, string prompt,
        int input, string selector, CancellationToken cancel) => new()
    {
        Image = await engine.Images.GenerateAsync(spec, new ImageRequest { Prompt = prompt, Width = definition.Width, Height = definition
            .Height, Steps = definition.Steps, Seed = definition.Seed + input, CfgScale = 7.5f, Sampler = "euler", Scheduler = "normal",
            Batch = 1, }, cancel: cancel),
    };
}
