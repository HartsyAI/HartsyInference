using System.Text.Json;

namespace HartsyInference.Vision.Moge;

/// <summary>How the head's raw 3-channel output becomes a point (upstream <c>remap_output</c>).</summary>
public enum MogeRemap
{
    /// <summary>The output is the point.</summary>
    Linear,

    /// <summary><c>sinh</c> of every channel.</summary>
    Sinh,

    /// <summary><c>(x · e^z, y · e^z, e^z)</c>: the released <c>moge_vitl</c>.</summary>
    Exp,

    /// <summary><c>(sinh x, sinh y, e^z)</c>.</summary>
    SinhExp,
}

/// <summary>
/// The MoGe v1 (<c>microsoft/MoGe</c>, MIT) architecture knobs the checkpoint's config JSON carries: a DINOv2 backbone whose last <see cref="IntermediateLayers"/> blocks feed a
/// convolutional head. Defaults are upstream's constructor defaults; <see cref="FromJson"/> reads <c>moge_vitl_config.json</c>.
/// </summary>
public sealed record MogeConfig
{
    /// <summary>The released <c>moge_vitl</c> (<c>Ruicheng/moge-vitl</c>) as its <c>moge_vitl_config.json</c> states it, for a checkpoint that comes without the file.</summary>
    public static MogeConfig Released { get; } = new()
    {
        IntermediateLayers = 4, DimUpsample = [256, 128, 64], ResBlockHiddenMultiplier = 2, NumResBlocks = 2, Remap = MogeRemap.Exp,
        NumTokensRange = (250000 / (14 * 14), 500000 / (14 * 14)),
    };

    /// <summary>The backbone's last blocks that feed the head (<c>intermediate_layers</c>).</summary>
    public int IntermediateLayers { get; init; } = 4;

    /// <summary>Width the 1 x 1 projections bring every tap to (<c>dim_proj</c>).</summary>
    public int DimProj { get; init; } = 512;

    /// <summary>Output width of each of the upsample blocks (<c>dim_upsample</c>); every block doubles the resolution.</summary>
    public int[] DimUpsample { get; init; } = [256, 128, 128];

    /// <summary>Hidden width of a residual block as a multiple of its width (<c>dim_times_res_block_hidden</c>).</summary>
    public int ResBlockHiddenMultiplier { get; init; } = 1;

    /// <summary>Residual blocks after each upsampler (<c>num_res_blocks</c>).</summary>
    public int NumResBlocks { get; init; } = 1;

    /// <summary>Width of the output blocks' first convolution (<c>last_conv_channels</c>).</summary>
    public int LastConvChannels { get; init; } = 32;

    /// <summary>Kernel size of the output blocks' last convolution (<c>last_conv_size</c>).</summary>
    public int LastConvSize { get; init; } = 1;

    /// <summary>The point remap (<c>remap_output</c>).</summary>
    public MogeRemap Remap { get; init; } = MogeRemap.Linear;

    /// <summary>The token budget <c>infer</c> interpolates between by its resolution level (<c>num_tokens_range</c>, or <c>trained_area_range // 14²</c> in older configs).</summary>
    public (int Min, int Max) NumTokensRange { get; init; } = (1200, 2500);

    /// <summary>Raw mask output above which a pixel counts as valid (<c>mask_threshold</c>).</summary>
    public float MaskThreshold { get; init; } = 0.5f;

    /// <summary>Reads a checkpoint's config JSON.</summary>
    /// <exception cref="InvalidDataException">The JSON is not a MoGe v1 config this port supports.</exception>
    public static MogeConfig FromJson(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        MogeConfig config = new();
        if (root.TryGetProperty("encoder", out JsonElement encoder) && encoder.GetString() is string name && name != "dinov2_vitl14")
            throw new InvalidDataException($"MoGe encoder '{name}' is not supported; this port reads the ViT-L backbone ('dinov2_vitl14').");
        if (root.TryGetProperty("split_head", out JsonElement split) && split.ValueKind == JsonValueKind.False)
            throw new InvalidDataException("A MoGe checkpoint without a split head is not supported.");
        if (root.TryGetProperty("intermediate_layers", out JsonElement layers))
        {
            if (layers.ValueKind != JsonValueKind.Number)
                throw new InvalidDataException("A list of intermediate layers is not supported; the config must give the number of last blocks.");
            config = config with { IntermediateLayers = layers.GetInt32() };
        }

        if (root.TryGetProperty("dim_proj", out JsonElement proj))
            config = config with { DimProj = proj.GetInt32() };
        if (root.TryGetProperty("dim_upsample", out JsonElement upsample))
            config = config with { DimUpsample = [.. upsample.EnumerateArray().Select(e => e.GetInt32())] };
        if (root.TryGetProperty("dim_times_res_block_hidden", out JsonElement hidden))
            config = config with { ResBlockHiddenMultiplier = hidden.GetInt32() };
        if (root.TryGetProperty("num_res_blocks", out JsonElement blocks))
            config = config with { NumResBlocks = blocks.GetInt32() };
        if (root.TryGetProperty("last_conv_channels", out JsonElement lastChannels))
            config = config with { LastConvChannels = lastChannels.GetInt32() };
        if (root.TryGetProperty("last_conv_size", out JsonElement lastSize))
            config = config with { LastConvSize = lastSize.GetInt32() };
        if (root.TryGetProperty("mask_threshold", out JsonElement threshold))
            config = config with { MaskThreshold = threshold.GetSingle() };
        if (root.TryGetProperty("remap_output", out JsonElement remap))
        {
            config = config with
            {
                Remap = remap.ValueKind switch
                {
                    JsonValueKind.True or JsonValueKind.False => throw new InvalidDataException("A boolean remap_output is invalid upstream; the config must name the remap."),
                    _ => remap.GetString() switch
                    {
                        "linear" => MogeRemap.Linear,
                        "sinh" => MogeRemap.Sinh,
                        "exp" => MogeRemap.Exp,
                        "sinh_exp" => MogeRemap.SinhExp,
                        string other => throw new InvalidDataException($"Unknown MoGe remap_output '{other}'."),
                        null => MogeRemap.Linear,
                    },
                },
            };
        }

        if (root.TryGetProperty("num_tokens_range", out JsonElement tokens))
        {
            int[] range = [.. tokens.EnumerateArray().Select(e => e.GetInt32())];
            config = config with { NumTokensRange = (range[0], range[1]) };
        }
        else if (root.TryGetProperty("trained_area_range", out JsonElement area))
        {
            int[] range = [.. area.EnumerateArray().Select(e => e.GetInt32())];
            config = config with { NumTokensRange = (range[0] / (14 * 14), range[1] / (14 * 14)) };
        }

        if (config.DimUpsample.Length == 0 || config.IntermediateLayers <= 0)
            throw new InvalidDataException("A MoGe config needs at least one upsample block and one intermediate layer.");
        return config;
    }
}
