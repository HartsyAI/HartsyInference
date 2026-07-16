using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Adapts a moshi-native "DSM" Mimi checkpoint (the <c>tokenizer-*.safetensors</c> shipped with
/// <c>kyutai/tts-1.6b-en_fr</c> and the STT models) to the HF <c>transformers</c> MimiModel key layout that the
/// decode-path loaders (<see cref="MimiSplitRvq"/>, <see cref="MimiSeanetDecoder"/>, <see cref="Mimi"/>'s upsample)
/// expect. The moshi checkpoint wraps every conv in a streaming module (<c>.conv.conv</c> / <c>.convtr.convtr</c>),
/// indexes the SEANet stack as a flat <c>decoder.model.N</c> Sequential (same N as HF <c>decoder.layers.N</c>),
/// and names the split RVQ <c>rvq_first</c>/<c>rvq_rest</c> with <c>.vq.layers.N._codebook.embedding_sum</c>.
///
/// <para>The transformer-of-codecs keys (<c>decoder_transformer.transformer.layers.*</c>) are intentionally left
/// UNTOUCHED — <see cref="MimiTransformer"/> detects and loads that moshi layout (fused <c>in_proj_weight</c>)
/// directly. Encode-only keys (<c>encoder.*</c>, <c>downsample.*</c>, <c>encoder_transformer.*</c>, RVQ
/// <c>input_proj</c>) are passed through unchanged; the decode path just ignores them.</para></summary>
internal static class MimiDsmWeights
{
    /// <summary>True if <paramref name="w"/> is a moshi-native DSM Mimi checkpoint (vs an HF <c>transformers</c> one).</summary>
    public static bool IsDsm(IReadOnlyDictionary<string, Tensor> w) =>
        w.ContainsKey("quantizer.rvq_first.output_proj.weight");

    /// <summary>Returns a new dict with the DSM decode-path keys renamed to the HF layout. Non-matching keys are
    /// copied through. Tensor instances are shared (borrowed from the caller's loader).</summary>
    public static Dictionary<string, Tensor> Adapt(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> outw = new(raw.Count);
        foreach ((string key, Tensor t) in raw)
            outw[Remap(key)] = t;
        return outw;
    }

    private static string Remap(string k)
    {
        // Split RVQ: rvq_first/rvq_rest → semantic/acoustic_residual_vector_quantizer, .vq.layers → .layers,
        // ._codebook.embedding_sum → .codebook.embed_sum (cluster_usage keeps its leaf name).
        if (k.StartsWith("quantizer.rvq_first."))
            k = "quantizer.semantic_residual_vector_quantizer." + k["quantizer.rvq_first.".Length..];
        else if (k.StartsWith("quantizer.rvq_rest."))
            k = "quantizer.acoustic_residual_vector_quantizer." + k["quantizer.rvq_rest.".Length..];
        if (k.Contains("_residual_vector_quantizer."))
        {
            k = k.Replace(".vq.layers.", ".layers.")
                 .Replace("._codebook.embedding_sum", ".codebook.embed_sum")
                 .Replace("._codebook.cluster_usage", ".codebook.cluster_usage");
        }

        // SEANet decoder: decoder.model.N → decoder.layers.N; unwrap the streaming conv/convtr wrappers.
        if (k.StartsWith("decoder.model."))
            k = "decoder.layers." + k["decoder.model.".Length..];

        // Upsample: upsample.convtr.convtr.convtr.weight → upsample.conv.weight.
        if (k.StartsWith("upsample."))
            k = "upsample.conv" + k[(k.LastIndexOf('.'))..];

        // Collapse the moshi streaming-conv double-wrap so a single ".conv."/".convtr." remains, then normalise a
        // transposed-conv slot to ".conv." (the decode loaders key both regular and transposed convs off ".conv").
        k = k.Replace(".conv.conv.", ".conv.").Replace(".convtr.convtr.", ".conv.");
        return k;
    }
}
