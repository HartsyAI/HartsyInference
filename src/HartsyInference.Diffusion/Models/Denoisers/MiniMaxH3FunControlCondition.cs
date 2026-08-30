using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One independently weighted/windowed control stream bound to a deduplicated branch model.</summary>
public sealed record MiniMaxH3FunControlCondition
{
    /// <summary>Index returned by <see cref="MiniMaxH3Transformer.RegisterFunControlNet"/>.</summary>
    public required int ModelIndex { get; init; }

    /// <summary>Target-video rows after 1x2x2 patchification, already padded to the branch's 49-channel input width.</summary>
    public required Tensor ControlRows { get; init; }

    /// <summary>Residual multiplier; zero is an exact branch bypass.</summary>
    public required float Strength { get; init; }

    /// <summary>Whether this control stream uses the 49-channel visibility/source inpaint contract. The sampler
    /// uses this marker only to reject an ambiguous combination with AV denoise masks.</summary>
    public bool IsInpaint { get; init; }

    /// <summary>Inclusive normalized denoise start.</summary>
    public float Start { get; init; }

    /// <summary>Inclusive normalized denoise end.</summary>
    public float End { get; init; } = 1f;

    /// <summary>Whether this stream participates in the named denoising evaluation.</summary>
    public bool IsActive(int step, int totalSteps)
    {
        if (step < 0 || totalSteps <= 0 || step >= totalSteps)
        {
            throw new ArgumentOutOfRangeException(nameof(step));
        }
        float position = totalSteps == 1 ? 0f : step / (float)(totalSteps - 1);
        return Strength != 0f && position >= Start && position <= End;
    }
}
