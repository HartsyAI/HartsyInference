using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using EngineConditioning = HartsyInference.Diffusion.Adapters.ControlNetConditioning;
using RequestConditioning = HartsyInference.Engine.Requests.ControlNetConditioning;

namespace HartsyInference.Engine.Features;

/// <summary>Resolves the request's ControlNet layers into engine <see cref="EngineConditioning"/>s for the UNet families
/// (SDXL / SD 1.5). Each layer names a checkpoint and carries an ALREADY-PREPROCESSED hint image — Canny/Depth/OpenPose
/// annotation stays host-side — so this loads the adapter, packs the hint into a <c>[1, 3, H, W]</c> tensor in
/// <c>[0, 1]</c>, and wires strength + the step window. Pipelines stack multiple layers by summing residuals.
///
/// <para><b>Union checkpoints</b> (xinsir controlnet-union-sdxl, detected via
/// <see cref="ControlNetConfig.UnionControlTypeCount"/>) need a control type per slot. Since the request contract does
/// not carry one, supply <paramref name="unionTypeSelector"/> to map a slot index onto the host's union-type string;
/// without it, union slots default to the thin-line (canny) type with a log.</para></summary>
public static class ControlNetResolver
{
    /// <summary>One generation's resolved ControlNet state: owns both the loaded adapters and the hint tensors.</summary>
    public sealed class ResolvedSpec : IDisposable
    {
        /// <summary>The per-layer conditionings to hand to the pipeline.</summary>
        public required List<EngineConditioning> Conditionings { get; init; }

        /// <summary>The loaded adapters backing <see cref="Conditionings"/>.</summary>
        public required List<ControlNetCacheEntry> Adapters { get; init; }

        /// <summary>The packed hint tensors backing <see cref="Conditionings"/>.</summary>
        public required List<Tensor> ConditionImages { get; init; }

        /// <summary>Frees the hint tensors and the adapters.</summary>
        public void Dispose()
        {
            foreach (Tensor img in ConditionImages)
            {
                img.Dispose();
            }
            foreach (ControlNetCacheEntry a in Adapters)
            {
                a.Dispose();
            }
        }
    }

    /// <summary>Resolves every layer in <paramref name="controlNets"/>, or null when there are none. Caller disposes the spec.</summary>
    public static ResolvedSpec? Resolve(
        IReadOnlyList<RequestConditioning>? controlNets,
        UNetConfig baseConfig,
        int targetW,
        int targetH,
        Action<string> log,
        Func<int, string?>? unionTypeSelector = null)
    {
        ArgumentNullException.ThrowIfNull(log);
        if (controlNets is null || controlNets.Count == 0)
        {
            return null;
        }
        List<EngineConditioning> conditionings = [];
        List<ControlNetCacheEntry> adapters = [];
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
                log($"Loading ControlNet[{i}]: {slot.Model}");
                ControlNetCacheEntry entry = ControlNetWeightLoader.Load(slot.Model, baseConfig);
                adapters.Add(entry);

                // Union checkpoints (xinsir controlnet-union-sdxl): one file covers all modes, and the type must be
                // supplied out-of-band. Single-mode files must pass null.
                bool isUnion = entry.File.Config.UnionControlTypeCount > 0;
                SdxlUnionControlType? unionType = isUnion
                    ? ResolveUnionType(unionTypeSelector?.Invoke(i), i, entry.File.Config.UnionControlTypeCount, slot.Model, log)
                    : null;

                byte[] rgb = FeatureImaging.ResizeRgb24(slot.Image, targetW, targetH);
                Tensor condTensor = FeatureImaging.RgbToTensorZeroOne(rgb, targetW, targetH);
                images.Add(condTensor);

                conditionings.Add(new EngineConditioning
                {
                    Adapter = entry.Adapter,
                    ConditionImage = condTensor,
                    Scale = (float)slot.Strength,
                    StartFraction = (float)Math.Clamp(slot.Start, 0.0, 1.0),
                    EndFraction = (float)Math.Clamp(slot.End, 0.0, 1.0),
                    UnionControlType = unionType,
                });
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[Features][ControlNet] Resolution failed; rolling back partial state.", ex);
            foreach (Tensor img in images)
            {
                img.Dispose();
            }
            foreach (ControlNetCacheEntry a in adapters)
            {
                a.Dispose();
            }
            throw;
        }
        log($"ControlNet enabled: {conditionings.Count} adapter(s).");
        return new ResolvedSpec
        {
            Conditionings = conditionings,
            Adapters = adapters,
            ConditionImages = images,
        };
    }

    /// <summary>Maps a host union-type string (values follow the xinsir training list) onto the checkpoint's control-type
    /// index. Null / "auto" defaults to the thin-line (canny) type. Tile/Repaint need the 8-type ProMax revision; the
    /// 6-type standard union is rejected here with a clear message instead of an out-of-range engine error.</summary>
    private static SdxlUnionControlType ResolveUnionType(string? requested, int slot, int numControlTypes, string modelName, Action<string> log)
    {
        string typeStr = string.IsNullOrWhiteSpace(requested) ? "auto" : requested.Trim().ToLowerInvariant();
        SdxlUnionControlType type = typeStr switch
        {
            "openpose" => SdxlUnionControlType.OpenPose,
            "depth" => SdxlUnionControlType.Depth,
            "hed/pidi/scribble/ted" or "softedge" => SdxlUnionControlType.SoftEdge,
            "canny/lineart/anime_lineart/mlsd" or "canny" => SdxlUnionControlType.Canny,
            "normal" => SdxlUnionControlType.Normal,
            "segment" => SdxlUnionControlType.Segment,
            "tile" => SdxlUnionControlType.Tile,
            "repaint" => SdxlUnionControlType.Repaint,
            "auto" => SdxlUnionControlType.Canny,
            _ => throw new InvalidOperationException($"ControlNet[{slot}] Union Type '{typeStr}' is not a recognized union control type."),
        };
        if (typeStr == "auto")
        {
            log($"ControlNet[{slot}] '{modelName}' is a union checkpoint and no union type was supplied — defaulting to canny (thin line).");
        }
        if ((int)type >= numControlTypes)
        {
            throw new InvalidOperationException(
                $"ControlNet[{slot}] '{modelName}' has {numControlTypes} control types (standard union revision) — "
                + $"'{type}' needs the 8-type ProMax revision (controlnet-union-sdxl promax).");
        }
        return type;
    }
}
