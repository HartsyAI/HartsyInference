using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Splits a TripoSR (<c>stabilityai/TripoSR</c>) checkpoint into the three component weight dicts the pipeline loads: the DINO ViT-B/16 image tokenizer (<c>Dino</c>, keys stripped of <c>image_tokenizer.model.</c>), the backbone (<c>Transformer</c>, retaining <c>tokenizer.</c> / <c>backbone.</c> / <c>post_processor.</c> prefixes), and the NeRF MLP (<c>Decoder</c>, stripped of <c>decoder.</c>). The published checkpoint key layout is verified against the real <c>model.ckpt</c>.</summary>
public static class TripoSrCheckpointConverter
{
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Dino { get; init; }
        public required Dictionary<string, Tensor> Transformer { get; init; }
        public required Dictionary<string, Tensor> Decoder { get; init; }
    }

    private const string DinoPrefix = "image_tokenizer.model.";
    private const string DecoderPrefix = "decoder.";

    public static ConvertedWeights Convert(IReadOnlyDictionary<string, Tensor> all)
    {
        ArgumentNullException.ThrowIfNull(all);
        Dictionary<string, Tensor> dino = new(), tr = new(), dec = new();
        foreach ((string key, Tensor t) in all)
        {
            if (key.StartsWith(DinoPrefix, StringComparison.Ordinal)) { dino[key[DinoPrefix.Length..]] = t; continue; }
            if (key.StartsWith(DecoderPrefix, StringComparison.Ordinal)) { dec[key[DecoderPrefix.Length..]] = t; continue; }
            // tokenizer.* / backbone.* / post_processor.* keep their prefixes (the backbone loads them directly).
            tr[key] = t;
        }
        return new ConvertedWeights { Dino = dino, Transformer = tr, Decoder = dec };
    }
}
