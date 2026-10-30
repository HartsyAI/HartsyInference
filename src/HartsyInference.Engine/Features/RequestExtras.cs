using System.Collections;
using System.Globalization;
using System.Text.Json;

namespace HartsyInference.Engine.Features;

/// <summary>Typed reads out of <c>ImageRequest.Extra</c> for the knobs the flat contract does not name. The contract is frozen, so features whose resolvers need extra parameters (ControlNet union type, IP-Adapter model/weight/window, Redux blending) source them here under documented keys and fall back to sensible defaults when absent. <para>Keys: <c>controlnet.union_type.{index}</c> (string, per-slot union control type, e.g. "canny"); <c>ipadapter.model</c> (string, adapter id/path — absent disables IP-Adapter); <c>ipadapter.clipvision</c> (string, CLIP-Vision override); <c>ipadapter.weight</c> (number, default 1.0); <c>ipadapter.start</c> (number, default 0.0); <c>ipadapter.end</c> (number, default 1.0); <c>ipadapter.weight_type</c> (string, default "standard"); <c>redux.stylemodel</c> (string, projector id/path — absent disables Redux); <c>redux.multiply</c> (number, default 1.0); <c>redux.merge</c> (number 0..1, default 1.0); <c>redux.apply_start</c> (number, default 0.0); <c>flux.tools_control_image</c> (<see cref="Requests.ImageData"/>, FLUX.1 Canny/Depth's already-preprocessed control hint — absent means the checkpoint is driven as vanilla Flux or FLUX.1 Fill instead); <c>slg.uncond_skip_layers</c>, <c>slg.layers</c> (integer lists, see <see cref="IntList"/>), <c>slg.scale</c>, <c>slg.start</c>, <c>slg.stop</c> (numbers) for skip-layer guidance, each defaulting to the family's own recipe.</para></summary>
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

    /// <summary>Blocks (0-indexed) the unconditional branch skips — "two-pass" skip-layer guidance. An empty list turns it
    /// off; absent keeps the family's default.</summary>
    public const string SlgUncondSkipLayers = "slg.uncond_skip_layers";

    /// <summary>Blocks (0-indexed) an extra conditional branch skips in "three-pass" skip-layer guidance, which adds
    /// <c>scale·(cond − skipped)</c>. Empty or a zero <see cref="SlgScale"/> runs no extra branch.</summary>
    public const string SlgLayers = "slg.layers";

    /// <summary>Weight of the three-pass skip-layer guidance term.</summary>
    public const string SlgScale = "slg.scale";

    /// <summary>Fraction of the steps at which skip-layer guidance starts.</summary>
    public const string SlgStart = "slg.start";

    /// <summary>Fraction of the steps at which skip-layer guidance stops (exclusive).</summary>
    public const string SlgStop = "slg.stop";

    /// <summary>Edit references attend through cached keys and values (true, ai-toolkit's <c>kv_cache</c> default) or ride
    /// in the live sequence (false, for an adapter trained with <c>kv_cache: false</c>).</summary>
    public const string ReferenceKvCache = "reference.kv_cache";

    /// <summary>Pixel budget of each reference's VAE copy (scaled down only); 0 keeps each image's own size.</summary>
    public const string ReferenceMaxPixels = "reference.max_pixels";

    /// <summary>Pixel budget of each reference's copy for the text encoder's vision tower (scaled down only).</summary>
    public const string ReferenceVisionPixels = "reference.vision_pixels";

    /// <summary>The edit layout a reference-trained adapter expects: <c>ostris</c> (default, ai-toolkit's edit LoRAs, kv cache
    /// or joint) or <c>colorize</c> (one page between the image and the text, <c>sorryhyun/moss-image-colorize</c>).</summary>
    public const string ReferenceLayout = "reference.layout";

    /// <summary>The colorize layout's reference scale: the page's VAE copy is the target divided by this per side (default 2).</summary>
    public const string ReferenceScale = "reference.ref_scale";

    /// <summary>The reference adapter a reference request uses: a catalog name or a file name in the LoRA folders. Empty or absent
    /// means the released adapter.</summary>
    public const string ReferenceAdapter = "reference.adapter";

    /// <summary>The adapter's strength, the gate's scale on it. 0 turns the reference off without a rebuild.</summary>
    public const string ReferenceStrength = "reference.strength";

    /// <summary>The reference guidance scale (the adapter's <c>ref_guidance</c>): 1 costs no extra call.</summary>
    public const string ReferenceGuidance = "reference.guidance";

    /// <summary>The correction after the reference combine: <c>apg</c>, <c>rescale</c> or <c>none</c>.</summary>
    public const string ReferenceCorrection = "reference.correction";

    /// <summary>The blend for the rescale correction, in [0, 1].</summary>
    public const string ReferenceRescalePhi = "reference.rescale_phi";

    /// <summary>The first denoising fraction with extra reference guidance.</summary>
    public const string ReferenceStart = "reference.start";

    /// <summary>The last denoising fraction with extra reference guidance.</summary>
    public const string ReferenceEnd = "reference.end";

    /// <summary>What steps outside the window do: <c>keep</c> (the reference stays in both branches) or <c>drop</c> (plain CFG on the base).</summary>
    public const string ReferenceOutside = "reference.outside";

    /// <summary>The boolean at <paramref name="key"/> (<c>true/false</c>, <c>on/off</c>, <c>1/0</c>, or a JSON boolean), or
    /// <paramref name="fallback"/> when absent or unparseable (logged).</summary>
    public static bool Bool(IReadOnlyDictionary<string, object>? extra, string key, bool fallback)
    {
        if (extra is null || !extra.TryGetValue(key, out object? value) || value is null)
            return fallback;
        switch (value)
        {
            case bool flag:
                return flag;
            case JsonElement { ValueKind: JsonValueKind.True }:
                return true;
            case JsonElement { ValueKind: JsonValueKind.False }:
                return false;
        }
        string text = (value is JsonElement json ? json.ToString() : Convert.ToString(value, CultureInfo.InvariantCulture) ?? "").Trim();
        if (text is "1" || text.Equals("true", StringComparison.OrdinalIgnoreCase) || text.Equals("on", StringComparison.OrdinalIgnoreCase))
            return true;
        if (text is "0" || text.Equals("false", StringComparison.OrdinalIgnoreCase) || text.Equals("off", StringComparison.OrdinalIgnoreCase))
            return false;
        HartsyInference.Core.Logging.Logs.Warning($"[Features][Extra] '{key}' is not a boolean ('{value}'); using {fallback}.");
        return fallback;
    }

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
            // A request body deserializes extras as JsonElement, which Convert cannot read: take the number directly.
            return value switch
            {
                JsonElement { ValueKind: JsonValueKind.Number } json => json.GetDouble(),
                JsonElement { ValueKind: JsonValueKind.String } json => double.Parse(json.GetString() ?? "", CultureInfo.InvariantCulture),
                JsonElement json => throw new FormatException($"expected a number, got {json.ValueKind}"),
                _ => Convert.ToDouble(value, CultureInfo.InvariantCulture),
            };
        }
        catch (Exception ex) when (ex is FormatException or InvalidCastException or OverflowException)
        {
            HartsyInference.Core.Logging.Logs.Warning($"[Features][Extra] '{key}' is not a number ('{value}'); using {fallback}. {ex.Message}");
            return fallback;
        }
    }

    /// <summary>The integer list at <paramref name="key"/>: a comma- or space-separated string (<c>"22, 24"</c>; blank or
    /// <c>none</c> is the empty list), one number, or a sequence of numbers (including a JSON array). Null when absent, or unparseable
    /// (logged), so the caller's default applies.</summary>
    public static IReadOnlyList<int>? IntList(IReadOnlyDictionary<string, object>? extra, string key)
    {
        if (extra is null || !extra.TryGetValue(key, out object? value) || value is null)
        {
            return null;
        }
        try
        {
            return value switch
            {
                string text => ParseIntList(text),
                JsonElement { ValueKind: JsonValueKind.String } json => ParseIntList(json.GetString() ?? ""),
                JsonElement { ValueKind: JsonValueKind.Array } json => [.. json.EnumerateArray().Select(item => item.GetInt32())],
                JsonElement json => [json.GetInt32()],
                IEnumerable sequence => [.. sequence.Cast<object>().Select(item => Convert.ToInt32(item, CultureInfo.InvariantCulture))],
                _ => [Convert.ToInt32(value, CultureInfo.InvariantCulture)],
            };
        }
        catch (Exception ex) when (ex is FormatException or InvalidCastException or OverflowException or InvalidOperationException)
        {
            HartsyInference.Core.Logging.Logs.Warning($"[Features][Extra] '{key}' is not a list of integers ('{value}'); using the default. {ex.Message}");
            return null;
        }
    }

    private static int[] ParseIntList(string text) => string.Equals(text.Trim(), "none", StringComparison.OrdinalIgnoreCase)
        ? []
        : [.. text.Split([',', ' ', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(item => int.Parse(item, NumberStyles.Integer, CultureInfo.InvariantCulture))];

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
