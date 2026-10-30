using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>MOSS-Image's guidance recipe beyond the CFG scale (OpenMOSS pipeline arguments, SGLang's
/// <c>MOSSImageSamplingParams</c>): the joint blocks the unconditional branch skips ("two-pass SLG", the reference
/// default of layer 22) and the optional three-pass skip-layer guidance, a conditional branch with
/// <see cref="SlgLayers"/> skipped that adds <c>slg_scale·(cond − skipped)</c>. Both apply on the steps whose
/// <c>index / steps</c> lies in <c>[Start, Stop)</c>.</summary>
public sealed record MossImageGuidance
{
    /// <summary>The reference pipeline's recipe: layer 22 skipped on the unconditional branch at every step, no extra branch.</summary>
    public static MossImageGuidance Reference { get; } = new() { UncondSkipLayers = [MossImageConfig.ReferenceUncondSkipLayer] };

    /// <summary>Joint blocks the unconditional branch skips; empty for plain CFG.</summary>
    public IReadOnlyList<int> UncondSkipLayers { get; init; } = [];

    /// <summary>Joint blocks the three-pass SLG branch skips; empty (or a zero <see cref="SlgScale"/>) for none.</summary>
    public IReadOnlyList<int> SlgLayers { get; init; } = [];

    /// <summary>Weight of <c>cond − skipped</c> in the three-pass SLG combine.</summary>
    public float SlgScale { get; init; }

    /// <summary>Start of the skip window as a fraction of the steps.</summary>
    public float Start { get; init; }

    /// <summary>End of the skip window (exclusive) as a fraction of the steps.</summary>
    public float Stop { get; init; } = 1f;

    /// <summary>Whether the three-pass SLG branch runs at all.</summary>
    public bool HasSlgBranch => SlgLayers.Count > 0 && SlgScale > 0f;

    /// <summary>Distinct skipped-layer sets a generation's branches run, which is how many reference key/value caches an
    /// edit builds per conditioning: the conditional set, the unconditional one inside the window (and outside it when
    /// the window is partial), and the SLG set.</summary>
    public int DistinctBranchCaches(bool guided)
    {
        bool partialWindow = Start > 0f || Stop < 1f;
        int caches = 1;
        if (guided)
            caches += UncondSkipLayers.Count == 0 ? 1 : partialWindow ? 2 : 1;
        if (HasSlgBranch)
            caches++;
        return caches;
    }

    /// <summary>Whether step <paramref name="stepIndex"/> of <paramref name="steps"/> is inside the skip window.</summary>
    public bool ActiveAt(int stepIndex, int steps) => Start <= (float)stepIndex / steps && (float)stepIndex / steps < Stop;

    /// <summary>The branches step <paramref name="stepIndex"/> runs: the conditional one, then the unconditional one when
    /// <paramref name="guided"/> (skipping <see cref="UncondSkipLayers"/> inside the window), then the SLG branch when it
    /// is active. <see cref="CombineSlg"/> and the sampler read the velocities in this order.</summary>
    public List<MossImageBranch> BranchesFor(MossImageConditioning cond, MossImageConditioning? uncond, bool guided, int stepIndex, int steps)
    {
        ArgumentNullException.ThrowIfNull(cond);
        if (guided && uncond is null)
            throw new ArgumentNullException(nameof(uncond), "A guided MOSS-Image step needs the unconditional conditioning.");
        bool window = ActiveAt(stepIndex, steps);
        List<MossImageBranch> branches = [new MossImageBranch(cond, [])];
        if (guided)
            branches.Add(new MossImageBranch(uncond!, window ? UncondSkipLayers : []));
        if (window && HasSlgBranch)
            branches.Add(new MossImageBranch(cond, SlgLayers));
        return branches;
    }

    /// <summary>The three-pass SLG combine in the reference's order: <c>base + SlgScale·(cond − skipped)</c>, where base is
    /// the CFG mix <c>uncond + cfg·(cond − uncond)</c>, or the skipped prediction itself when <paramref name="uncond"/> is
    /// null (no CFG). Returns a new tensor; the inputs stay the caller's.</summary>
    public Tensor CombineSlg(IBackend backend, Tensor cond, Tensor? uncond, Tensor skipped, float cfg)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(skipped);
        Tensor combined = new(cond.Shape, DType.F32);
        try
        {
            // uncond + cfg·(cond − uncond) + SlgScale·(cond − skipped). With no uncond, skipped stands in and the CFG weight is zero.
            // The helper takes (cond, imageCond, uncond). The skipped prediction goes in as cond, so its outer term
            // outer·(cond − imageCond) = −SlgScale·(skipped − cond) is the SLG term; imageCond carries the conditional prediction.
            CfgHelper.ApplyDualCfgDevice(backend, combined, skipped, cond, uncond ?? skipped, uncond is null ? 0f : cfg, -SlgScale);
            return combined;
        }
        catch
        {
            combined.Dispose();
            throw;
        }
    }

    /// <summary>Refuses a recipe the model cannot run.</summary>
    public void Validate(int numLayers)
    {
        foreach (int layer in UncondSkipLayers.Concat(SlgLayers))
        {
            if (layer < 0 || layer >= numLayers)
                throw new ArgumentOutOfRangeException(nameof(numLayers), $"MOSS-Image skip layers must be in [0, {numLayers}); got {layer}.");
        }
        if (!float.IsFinite(SlgScale) || SlgScale < 0f)
            throw new ArgumentOutOfRangeException(nameof(SlgScale), SlgScale, "MOSS-Image SLG scale must be finite and non-negative.");
        if (!(Start >= 0f && Start <= Stop && Stop <= 1f))
            throw new ArgumentOutOfRangeException(nameof(Start), $"MOSS-Image needs 0 <= start <= stop <= 1; got {Start}..{Stop}.");
    }
}
