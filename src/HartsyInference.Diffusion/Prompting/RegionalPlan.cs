using System.Collections.Generic;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>A complete regional-conditioning plan: the global/background conditioning plus an ordered list of regional overrides. Built once per generation and consumed every denoise step. Model pipelines turn this into a per-step attention bias.</summary>
public sealed record RegionalPlan
{
    /// <summary>Global conditioning applied everywhere the regions do not override.</summary>
    public required Tensor BaseCond { get; init; }

    /// <summary>Per-region conditioning overrides, in priority order.</summary>
    public IReadOnlyList<RegionConditioning> Regions { get; init; } = [];

    /// <summary>Writes the effective per-region weight at <paramref name="step"/> into <paramref name="outWeights"/> (length must equal <see cref="Regions"/>.Count): the region's weight inside its <c>[StartStep, EndStep)</c> window, else 0. Zero-alloc.</summary>
    public void ResolveStep(int step, Span<float> outWeights)
    {
        if (outWeights.Length != Regions.Count)
        {
            throw new HartsyInferenceException($"outWeights length {outWeights.Length} must equal region count {Regions.Count}.");
        }
        for (int i = 0; i < Regions.Count; i++)
        {
            RegionConditioning region = Regions[i];
            outWeights[i] = step >= region.StartStep && step < region.EndStep ? region.Weight : 0f;
        }
    }
}
