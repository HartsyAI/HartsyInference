using HartsyInference.Engine.Requests;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Native output retained until untimed evidence encoding and quality checks.</summary>
public sealed record BenchmarkOutput
{
    public TextResult? Text { get; init; }
    public ImageResult? Image { get; init; }
}
