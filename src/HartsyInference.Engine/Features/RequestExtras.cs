using System.Globalization;

namespace HartsyInference.Engine.Features;

/// <summary>Typed reads out of <c>ImageRequest.Extra</c> for the knobs the flat contract does not name. The contract is frozen, so features whose resolvers need extra parameters (ControlNet union type, IP-Adapter model/weight/window, Redux blending) source them here under documented keys and fall back to sensible defaults when absent. <para>Keys: <c>controlnet.union_type.{index}</c> (string, per-slot union control type, e.g. "canny"); <c>ipadapter.model</c> (string, adapter id/path — absent disables IP-Adapter); <c>ipadapter.clipvision</c> (string, CLIP-Vision override); <c>ipadapter.weight</c> (number, default 1.0); <c>ipadapter.start</c> (number, default 0.0); <c>ipadapter.end</c> (number, default 1.0); <c>ipadapter.weight_type</c> (string, default "standard"); <c>redux.stylemodel</c> (string, projector id/path — absent disables Redux); <c>redux.multiply</c> (number, default 1.0); <c>redux.merge</c> (number 0..1, default 1.0); <c>redux.apply_start</c> (number, default 0.0); <c>flux.tools_control_image</c> (<see cref="Requests.ImageData"/>, FLUX.1 Canny/Depth's already-preprocessed control hint — absent means the checkpoint is driven as vanilla Flux or FLUX.1 Fill instead).</para></summary>
public static class RequestExtras
{
    /// <summary>Per-slot ControlNet union control type key, e.g. <c>controlnet.union_type.0</c>.</summary>
    public const string ControlNetUnionTypePrefix = "controlnet.union_type.";

    /// <summary>IP-Adapter model id or path; absent/blank means the feature stays off even when prompt images are set.</summary>
    public const string IpAdapterModel = "ipadapter.model";

    /// <summary>CLIP-Vision model override for the IP-Adapter image encoder.</summary>
    public const string IpAdapterClipVision = "ipadapter.clipvision";

    /// <summary>IP-Adapter conditioning weight.</summary>
    public const string IpAdapterWeight = "ipadapter.weight";

    /// <summary>Fraction of the schedule at which IP-Adapter conditioning starts.</summary>
    public const string IpAdapterStart = "ipadapter.start";

    /// <summary>Fraction of the schedule at which IP-Adapter conditioning ends.</summary>
    public const string IpAdapterEnd = "ipadapter.end";

    /// <summary>IP-Adapter per-layer weight ramp name.</summary>
    public const string IpAdapterWeightType = "ipadapter.weight_type";

    /// <summary>Redux style-model (projector) id or path; absent/blank means Redux stays off.</summary>
    public const string ReduxStyleModel = "redux.stylemodel";

    /// <summary>Redux embedding multiplier.</summary>
    public const string ReduxMultiply = "redux.multiply";

    /// <summary>Redux merge strength 0..1 (ConditioningAverage weight; reduces to a scalar on the redux tokens).</summary>
    public const string ReduxMerge = "redux.merge";

    /// <summary>Fraction of the schedule at which Redux conditioning starts applying.</summary>
    public const string ReduxApplyStart = "redux.apply_start";

    /// <summary>FLUX.1 Canny/Depth's already-preprocessed control hint (<see cref="Requests.ImageData"/>) — the host runs the edge/depth annotator (it needs host-app image types) and hands back the finished map. Absent means no Tools control image was supplied; the recipe still resolves the checkpoint's actual variant from the loaded weights, so an absent hint on a genuine Tools checkpoint surfaces as <see cref="Diffusion.Pipelines.FluxPipeline"/>'s own clear "requires a control image" error rather than a silent fallback.</summary>
    public const string FluxToolsControlImage = "flux.tools_control_image";

    /// <summary>The string at <paramref name="key"/>, or <paramref name="fallback"/> when absent or blank.</summary>
    public static string? String(IReadOnlyDictionary<string, object>? extra, string key, string? fallback = null)
    {
        if (extra is null || !extra.TryGetValue(key, out object? value) || value is null)
        {
            return fallback;
        }
        string? text = value as string ?? Convert.ToString(value, CultureInfo.InvariantCulture);
        return string.IsNullOrWhiteSpace(text) ? fallback : text;
    }

    /// <summary>The number at <paramref name="key"/>, or <paramref name="fallback"/> when absent or unparseable.</summary>
    public static double Number(IReadOnlyDictionary<string, object>? extra, string key, double fallback)
    {
        if (extra is null || !extra.TryGetValue(key, out object? value) || value is null)
        {
            return fallback;
        }
        try
        {
            return Convert.ToDouble(value, CultureInfo.InvariantCulture);
        }
        catch (Exception ex) when (ex is FormatException or InvalidCastException or OverflowException)
        {
            HartsyInference.Core.Logging.Logs.Warning($"[Features][Extra] '{key}' is not a number ('{value}'); using {fallback}. {ex.Message}");
            return fallback;
        }
    }

    /// <summary>The ControlNet union control type declared for slot <paramref name="index"/>, or null for none.</summary>
    public static string? ControlNetUnionType(IReadOnlyDictionary<string, object>? extra, int index) =>
        String(extra, ControlNetUnionTypePrefix + index.ToString(CultureInfo.InvariantCulture));

    /// <summary>The <see cref="Requests.ImageData"/> at <paramref name="key"/>, or null when absent or not actually an <see cref="Requests.ImageData"/> (the bag is <c>object</c>-valued; a host bug that puts the wrong type under this key should surface as "no image", not an <see cref="InvalidCastException"/> mid-generation).</summary>
    public static Requests.ImageData? Image(IReadOnlyDictionary<string, object>? extra, string key)
    {
        if (extra is null || !extra.TryGetValue(key, out object? value))
        {
            return null;
        }
        return value as Requests.ImageData;
    }
}
