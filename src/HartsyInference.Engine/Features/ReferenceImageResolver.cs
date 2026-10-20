using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Collects an edit request's reference images in picture order and prepares the two copies an in-context edit
/// model reads: one for its VAE and one for its text encoder's vision tower, each sized by a
/// <see cref="ReferenceImagePolicy"/>. Family-agnostic: Qwen-Image-Edit and MOSS-Image differ only in their policies.</summary>
public static class ReferenceImageResolver
{
    /// <summary>The references in picture order: <paramref name="initReference"/> first when the init image is a
    /// reference, then <paramref name="references"/>, cut to the policy's count (or refused, per its overflow rule).</summary>
    public static IReadOnlyList<ImageData> Collect(ImageData? initReference, IReadOnlyList<ImageData>? references,
        ReferenceImagePolicy policy, string family)
    {
        ArgumentNullException.ThrowIfNull(policy);
        List<ImageData> ordered = new(policy.MaxReferences);
        int offered = 0;
        if (initReference is not null)
        {
            ordered.Add(initReference);
            offered++;
        }
        if (references is not null)
        {
            foreach (ImageData reference in references)
            {
                if (reference is null)
                    continue;
                offered++;
                if (ordered.Count < policy.MaxReferences)
                    ordered.Add(reference);
            }
        }
        if (offered > ordered.Count)
        {
            if (policy.Overflow == ReferenceOverflow.Refuse)
                throw new ArgumentException($"{family} takes at most {policy.MaxReferences} reference images; {offered} were given.");
            Logs.Warning($"[{family}] {offered} reference images supplied; the model addresses {policy.MaxReferences}, "
                + $"so the last {offered - ordered.Count} were dropped.");
        }
        return ordered;
    }

    /// <summary>Prepares <paramref name="ordered"/>, or returns null when it is empty. <paramref name="visionSizer"/> maps
    /// the budgeted vision size to the size the vision tower itself resizes to, so each vision copy is resampled once,
    /// straight to it; null keeps the budgeted size. Caller disposes.</summary>
    public static ResolvedReferences? Resolve(IReadOnlyList<ImageData> ordered, ReferenceImagePolicy policy,
        Func<int, int, (int Width, int Height)>? visionSizer = null)
    {
        ArgumentNullException.ThrowIfNull(ordered);
        ArgumentNullException.ThrowIfNull(policy);
        if (ordered.Count == 0)
            return null;
        List<Tensor> latent = new(ordered.Count);
        List<Tensor> vision = new(ordered.Count);
        try
        {
            foreach (ImageData source in ordered)
            {
                ImageData reference = policy.FitTo is { } fit ? FeatureImaging.CoverCrop(source, fit.Width, fit.Height) : source;
                (int latentW, int latentH) = LatentSize(reference.Width, reference.Height, policy);
                ImageData latentSource = policy.FitTo is null ? reference : FeatureImaging.CoverCrop(reference, latentW, latentH);
                byte[] latentRgb = FeatureImaging.ResizeRgb24(latentSource, latentW, latentH);
                latent.Add(FeatureImaging.RgbToTensorMinusOneOne(latentRgb, latentW, latentH));
                if (policy.VisionTargetArea <= 0)
                    continue;
                (int visionW, int visionH) = VisionSize(reference.Width, reference.Height, policy);
                if (visionSizer is not null)
                    (visionW, visionH) = visionSizer(visionW, visionH);
                vision.Add(FeatureImaging.RgbToTensorZeroOne(FeatureImaging.ResizeRgb24(reference, visionW, visionH), visionW, visionH));
            }
        }
        catch
        {
            foreach (Tensor tensor in latent) tensor.Dispose();
            foreach (Tensor tensor in vision) tensor.Dispose();
            throw;
        }
        return new ResolvedReferences { Latent = latent, Vision = vision };
    }

    /// <summary>The VAE copy's size: the image at the policy's budget (or its own size), sides rounded to the multiple; with a
    /// <see cref="ReferenceImagePolicy.LatentSideDivisor"/>, each side divided by it instead.</summary>
    public static (int Width, int Height) LatentSize(int width, int height, ReferenceImagePolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        if (policy.LatentSideDivisor > 0)
            return (SideAtDivisor(width, policy), SideAtDivisor(height, policy));
        return FeatureImaging.ScaleToArea(width, height, Budget(width, height, policy.LatentTargetArea, policy.LatentUpscale), policy.LatentMultiple);
    }

    /// <summary>One side divided by the policy's factor and snapped to its multiple, never below one multiple.</summary>
    private static int SideAtDivisor(int side, ReferenceImagePolicy policy)
    {
        int multiple = policy.LatentMultiple;
        return Math.Max(multiple, (int)Math.Round(side / (double)policy.LatentSideDivisor / multiple) * multiple);
    }

    /// <summary>The vision copy's budgeted size, before the tower's own resize.</summary>
    public static (int Width, int Height) VisionSize(int width, int height, ReferenceImagePolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        return FeatureImaging.ScaleToArea(width, height, Budget(width, height, policy.VisionTargetArea, policy.VisionUpscale), multiple: 1);
    }

    private static int Budget(int width, int height, int targetArea, bool upscale)
    {
        long own = (long)width * height;
        if (targetArea <= 0 || (!upscale && own <= targetArea))
            return (int)Math.Min(own, int.MaxValue);
        return targetArea;
    }
}
