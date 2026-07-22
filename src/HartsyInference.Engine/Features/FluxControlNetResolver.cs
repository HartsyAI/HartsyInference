using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using RequestConditioning = HartsyInference.Engine.Requests.ControlNetConditioning;

namespace HartsyInference.Engine.Features;

/// <summary>Flux DiT counterpart of <see cref="ControlNetResolver"/>: resolves the request's ControlNet layers into
/// <see cref="FluxControlNetConditioning"/>s. Per-slot checkpoint load, hint packing to <c>[1, 3, H, W]</c> in
/// <c>[-1, 1]</c> (the Flux control convention), strength + start/end wiring. Hint preprocessing is host-side, so the
/// image arrives as the finished control map — union checkpoints therefore take it as-is.</summary>
public static class FluxControlNetResolver
{
    /// <summary>One generation's resolved Flux ControlNet state: owns the adapters and the control tensors.</summary>
    public sealed class ResolvedSpec : IDisposable
    {
        /// <summary>The per-layer conditionings to hand to the Flux pipeline.</summary>
        public required List<FluxControlNetConditioning> Conditionings { get; init; }

        /// <summary>The loaded adapters backing <see cref="Conditionings"/>.</summary>
        public required List<FluxControlNetCacheEntry> Adapters { get; init; }

        /// <summary>The packed control tensors backing <see cref="Conditionings"/>.</summary>
        public required List<Tensor> ControlImages { get; init; }

        /// <summary>Frees the control tensors and the adapters.</summary>
        public void Dispose()
        {
            foreach (Tensor img in ControlImages)
            {
                img.Dispose();
            }
            foreach (FluxControlNetCacheEntry a in Adapters)
            {
                a.Dispose();
            }
        }
    }

    /// <summary>Resolves every layer, or null when there are none. Throws when a slot names a UNet-family ControlNet.</summary>
    public static ResolvedSpec? Resolve(
        IReadOnlyList<RequestConditioning>? controlNets, int targetW, int targetH, Action<string> log)
    {
        ArgumentNullException.ThrowIfNull(log);
        if (controlNets is null || controlNets.Count == 0)
        {
            return null;
        }
        List<FluxControlNetConditioning> conditionings = [];
        List<FluxControlNetCacheEntry> adapters = [];
        List<Tensor> images = [];
        try
        {
            for (int i = 0; i < controlNets.Count; i++)
            {
                RequestConditioning slot = controlNets[i];
                if (slot.Image is null)
                {
                    throw new InvalidOperationException($"ControlNet[{i}] '{slot.Model}' is selected but carries no hint image.");
                }
                log($"Loading Flux ControlNet[{i}]: {slot.Model}");
                string path = ModelFileLocator.Require(slot.Model, "ControlNet model", ControlNetWeightLoader.Folders);
                ControlNetFile file = ControlNetLoader.Load(path);
                if (file.FluxConfig is null)
                {
                    file.Dispose();
                    throw new InvalidOperationException(
                        $"ControlNet '{slot.Model}' is a {file.BaseModel} (UNet) ControlNet, but the current generation uses a Flux model. "
                        + "Pick a Flux DiT ControlNet.");
                }
                FluxControlNet adapter = new FluxControlNet(file.FluxConfig);
                adapter.LoadWeights(file.Weights);
                adapters.Add(new FluxControlNetCacheEntry { File = file, Adapter = adapter });

                byte[] rgb = FeatureImaging.ResizeRgb24(slot.Image, targetW, targetH);
                Tensor control = FeatureImaging.RgbToTensorMinusOneOne(rgb, targetW, targetH);
                images.Add(control);

                conditionings.Add(new FluxControlNetConditioning
                {
                    Adapter = adapter,
                    ControlImage = control,
                    Scale = (float)slot.Strength,
                    StartFraction = (float)Math.Clamp(slot.Start, 0.0, 1.0),
                    EndFraction = (float)Math.Clamp(slot.End, 0.0, 1.0),
                });
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[Features][FluxControlNet] Resolution failed; rolling back partial state.", ex);
            foreach (Tensor img in images)
            {
                img.Dispose();
            }
            foreach (FluxControlNetCacheEntry a in adapters)
            {
                a.Dispose();
            }
            throw;
        }
        log($"Flux ControlNet enabled: {conditionings.Count} adapter(s).");
        return new ResolvedSpec { Conditionings = conditionings, Adapters = adapters, ControlImages = images };
    }
}
