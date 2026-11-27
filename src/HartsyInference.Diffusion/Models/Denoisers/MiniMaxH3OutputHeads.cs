using System.Text.Json;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Stacked output heads a MiniMax-H3-Image checkpoint declares in its safetensors <c>config</c> metadata
/// (<c>transformer.pdd_steps</c> and <c>transformer.pdd_sigmas</c>). Validated before use: a wrong grid produces
/// plausible-looking output, so every mismatch is refused.</summary>
public sealed record MiniMaxH3OutputHeads
{
    /// <summary>Rows of one head block: latent channels (24) times the 1x2x2 patch volume (4).</summary>
    public const int RowsPerHead = 96;

    public required int Count { get; init; }

    /// <summary>Strictly decreasing grid from 1.0 to 0.0 with <see cref="Count"/> + 1 points.</summary>
    public required IReadOnlyList<double> Sigmas { get; init; }

    /// <summary>Head layout for a checkpoint with <paramref name="videoOutRows"/> output rows. Returns null for a
    /// single-head checkpoint without metadata; refuses any row count that does not match the declared heads.</summary>
    public static MiniMaxH3OutputHeads? Resolve(long videoOutRows, IReadOnlyDictionary<string, string>? metadata)
    {
        MiniMaxH3OutputHeads? heads = FromMetadata(metadata);
        if (heads is null)
        {
            if (videoOutRows != RowsPerHead)
            {
                throw new ArgumentException(
                    $"MiniMax-H3-Image 'final_layer.video_out.weight' has {videoOutRows} rows but the checkpoint declares no "
                    + $"output heads. Without a 'pdd_steps' header only a single head ({RowsPerHead} rows) is accepted.");
            }
            return null;
        }

        if (videoOutRows != (long)RowsPerHead * heads.Count)
        {
            throw new ArgumentException(
                $"MiniMax-H3-Image declares {heads.Count} output heads, so 'final_layer.video_out.weight' must have "
                + $"{(long)RowsPerHead * heads.Count} rows, but it has {videoOutRows}.");
        }
        return heads;
    }

    /// <summary>Parses the heads from the <c>config</c> metadata, or returns null when it declares none.</summary>
    public static MiniMaxH3OutputHeads? FromMetadata(IReadOnlyDictionary<string, string>? metadata)
    {
        if (metadata is null || !metadata.TryGetValue("config", out string? json))
        {
            return null;
        }

        using JsonDocument document = JsonDocument.Parse(json);
        if (!document.RootElement.TryGetProperty("transformer", out JsonElement transformer)
            || !transformer.TryGetProperty("pdd_steps", out JsonElement steps))
        {
            return null;
        }

        int count = steps.GetInt32();
        if (count < 1)
        {
            throw new ArgumentException($"MiniMax-H3-Image 'pdd_steps' must be at least 1, got {count}.");
        }

        if (!transformer.TryGetProperty("pdd_sigmas", out JsonElement grid) || grid.ValueKind != JsonValueKind.Array)
        {
            throw new ArgumentException("MiniMax-H3-Image declares 'pdd_steps' but no 'pdd_sigmas' array.");
        }

        List<double> sigmas = [];
        foreach (JsonElement element in grid.EnumerateArray())
        {
            sigmas.Add(element.GetDouble());
        }

        if (sigmas.Count != count + 1)
        {
            throw new ArgumentException(
                $"MiniMax-H3-Image 'pdd_sigmas' has {sigmas.Count} points; {count} heads need {count + 1}.");
        }

        if (sigmas[0] != 1.0 || sigmas[^1] != 0.0)
        {
            throw new ArgumentException("MiniMax-H3-Image 'pdd_sigmas' must run from 1.0 down to 0.0.");
        }

        for (int i = 0; i + 1 < sigmas.Count; i++)
        {
            if (!(sigmas[i] > sigmas[i + 1]))
            {
                throw new ArgumentException($"MiniMax-H3-Image 'pdd_sigmas' must be strictly decreasing (index {i}).");
            }
        }

        return new MiniMaxH3OutputHeads { Count = count, Sigmas = sigmas };
    }
}
