using System.Text.Json;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Chooses the cross-attention heads that carry word timing for a checkpoint.</summary>
public static class WhisperAlignmentHeads
{
    /// <summary>The checkpoint's own <c>alignment_heads</c> from its <c>generation_config.json</c>, or <see cref="Default"/> when the file is absent, lists none, or lists none that fit <paramref name="cfg"/>.</summary>
    public static IReadOnlyList<WhisperAlignmentHead> Resolve(string? generationConfigPath, WhisperConfig cfg)
    {
        if (generationConfigPath is null)
        {
            return Default(cfg);
        }
        using FileStream stream = File.OpenRead(generationConfigPath);
        using JsonDocument doc = JsonDocument.Parse(stream);
        return Parse(doc.RootElement, cfg);
    }

    /// <summary>Reads <c>alignment_heads</c> (a list of <c>[layer, head]</c> pairs) from a parsed <c>generation_config.json</c>; a pair outside the model's layer or head count is dropped.</summary>
    public static IReadOnlyList<WhisperAlignmentHead> Parse(JsonElement generationConfig, WhisperConfig cfg)
    {
        if (!generationConfig.TryGetProperty("alignment_heads", out JsonElement listed) || listed.ValueKind != JsonValueKind.Array)
        {
            return Default(cfg);
        }
        List<WhisperAlignmentHead> heads = new List<WhisperAlignmentHead>();
        foreach (JsonElement pair in listed.EnumerateArray())
        {
            if (pair.ValueKind != JsonValueKind.Array || pair.GetArrayLength() != 2
                || !pair[0].TryGetInt32(out int layer) || !pair[1].TryGetInt32(out int head))
            {
                continue;
            }
            if (layer >= 0 && layer < cfg.DecoderLayers && head >= 0 && head < cfg.NumHeads)
            {
                heads.Add(new WhisperAlignmentHead(layer, head));
            }
        }
        if (heads.Count == 0)
        {
            Logs.Warning("[Audio][STT] generation_config.json lists no usable alignment head; using every head of the upper decoder layers.");
            return Default(cfg);
        }
        return heads;
    }

    /// <summary>Every head of the upper half of the decoder, the heuristic OpenAI's Whisper falls back to for a checkpoint with no published alignment heads.</summary>
    public static IReadOnlyList<WhisperAlignmentHead> Default(WhisperConfig cfg)
    {
        List<WhisperAlignmentHead> heads = new List<WhisperAlignmentHead>();
        for (int layer = cfg.DecoderLayers / 2; layer < cfg.DecoderLayers; layer++)
        {
            for (int head = 0; head < cfg.NumHeads; head++)
            {
                heads.Add(new WhisperAlignmentHead(layer, head));
            }
        }
        return heads;
    }
}
