using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Loads MusicGen / AudioGen checkpoints (<c>facebook/musicgen-{small,medium,large}</c>,
/// <c>facebook/encodec_32khz</c>) for the <c>MusicGenDecoder</c> / <c>EnCodec</c> classes.
///
/// <para><b>Decoder</b> — handles both HF exports by prefix probing: the combined
/// <c>MusicgenForConditionalGeneration</c> file (<c>decoder.model.decoder.*</c> / <c>decoder.lm_heads.*</c> /
/// top-level <c>enc_to_dec_proj.*</c>, with <c>text_encoder.*</c> + <c>audio_encoder.*</c> siblings dropped) and a
/// standalone <c>MusicgenForCausalLM</c> dump (<c>model.decoder.*</c> / <c>lm_heads.*</c>, pass-through). The
/// sinusoidal <c>embed_positions</c> buffer is dropped — the engine recomputes positions.</para>
///
/// <para><b>EnCodec</b> — maps the HF <c>transformers</c> <c>EncodecModel</c> naming (<c>encoder.layers.{i}.conv.*</c>,
/// <c>quantizer.layers.{q}.codebook.embed</c>) to the Meta AudioCraft naming the engine expects
/// (<c>encoder.model.{i}.conv.conv.*</c>, <c>quantizer.vq.layers.{q}._codebook.embed</c>,
/// <c>decoder.model.{i}.convtr.convtr.*</c>). Transpose convs are detected per-module from the <c>weight_g</c> shape
/// (PyTorch weight-norms <c>ConvTranspose1d</c> on dim=1 → <c>[1, C_out, 1]</c> vs <c>[C_out, 1, 1]</c> for
/// <c>Conv1d</c>). Weight-norm pairs stay RAW (no fusion) — <c>SeaNetEncoder</c>/<c>SeaNetDecoder</c> fuse them via
/// <c>WeightNormFusion</c> at <c>LoadWeights</c> time, unlike the ACE-Step vocoder path which fuses in the converter.
/// Meta-named dumps and the EnCodec embedded in a combined MusicGen file (<c>audio_encoder.*</c>) also load.</para></summary>
public sealed class MusicGenCheckpointConverter
{
    /// <summary>Pure decoder key mapping (testable without files). Returns <c>null</c> for keys the decoder never
    /// consumes (T5 text encoder, embedded EnCodec, sinusoidal position buffer).</summary>
    public static string? MapDecoderKey(string key)
    {
        if (key.StartsWith("text_encoder.", StringComparison.Ordinal) ||
            key.StartsWith("audio_encoder.", StringComparison.Ordinal))
            return null;
        if (key.Contains(".embed_positions.", StringComparison.Ordinal))
            return null;
        return CodecKeyUtils.StripPrefix(key, "decoder.");
    }

    /// <summary>Loads the MusicGen decoder safetensors (combined or standalone layout). Caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadDecoder(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = new();
        foreach (string key in loader.Descriptors.Keys)
        {
            string? mapped = MapDecoderKey(key);
            if (mapped is not null) weights[mapped] = CodecKeyUtils.MaybeCast(loader.GetTensor(key), castToF32);
        }
        return (weights, loader);
    }

    /// <summary>Loads an EnCodec safetensors — either <c>facebook/encodec_32khz</c> standalone or the
    /// <c>audio_encoder.*</c> subset of a combined MusicGen file. Caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadEnCodec(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        bool combined = false;
        foreach (string key in loader.Descriptors.Keys)
        {
            if (key.StartsWith("audio_encoder.", StringComparison.Ordinal))
            {
                combined = true;
                break;
            }
        }
        Dictionary<string, Tensor> raw = new();
        foreach (string key in loader.Descriptors.Keys)
        {
            if (combined && !key.StartsWith("audio_encoder.", StringComparison.Ordinal)) continue;
            string stripped = combined ? key["audio_encoder.".Length..] : key;
            raw[stripped] = CodecKeyUtils.MaybeCast(loader.GetTensor(key), castToF32);
        }
        return (ConvertEnCodec(raw), loader);
    }

    /// <summary>Converts an EnCodec weight dictionary from HF <c>transformers</c> naming to the Meta naming the engine
    /// <c>EnCodec</c> loads; Meta-named dictionaries pass through. Weight-norm pairs are kept raw — the engine fuses.</summary>
    public static Dictionary<string, Tensor> ConvertEnCodec(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> normalized = new(raw.Count);
        foreach ((string key, Tensor value) in raw)
            normalized[CodecKeyUtils.NormalizeWeightNormKey(key)] = value;

        HashSet<string> transposeModules = new(StringComparer.Ordinal);
        foreach ((string key, Tensor value) in normalized)
        {
            if (key.EndsWith(".conv.weight_g", StringComparison.Ordinal) &&
                value.Shape.Rank == 3 && value.Shape[0] == 1 && value.Shape[1] > 1)
                transposeModules.Add(key[..^".weight_g".Length]);
        }

        Dictionary<string, Tensor> result = new(normalized.Count);
        foreach ((string key, Tensor value) in normalized)
        {
            int convIdx = key.LastIndexOf(".conv.", StringComparison.Ordinal);
            bool transpose = convIdx >= 0 && transposeModules.Contains(key[..(convIdx + ".conv".Length)]);
            string? mapped = MapEnCodecKey(key, transpose);
            if (mapped is not null) result[mapped] = value;
        }
        return result;
    }

    /// <summary>Pure EnCodec key mapping (testable without files). <paramref name="transposeConv"/> marks the key's
    /// conv module as a <c>ConvTranspose1d</c> (decoder upsamplers) → <c>convtr.convtr</c> instead of <c>conv.conv</c>.
    /// Returns <c>null</c> for training-only quantizer buffers (<c>inited</c>/<c>cluster_size</c>/<c>embed_avg</c>);
    /// keys not in the HF layout pass through unchanged.</summary>
    public static string? MapEnCodecKey(string key, bool transposeConv = false)
    {
        key = CodecKeyUtils.NormalizeWeightNormKey(key);
        if (key.StartsWith("quantizer.layers.", StringComparison.Ordinal))
        {
            string rest = key["quantizer.layers.".Length..];
            int dot = rest.IndexOf('.');
            if (dot < 0) return null;
            return rest[dot..] == ".codebook.embed" ? $"quantizer.vq.layers.{rest[..dot]}._codebook.embed" : null;
        }

        bool encoder = key.StartsWith("encoder.layers.", StringComparison.Ordinal);
        if (!encoder && !key.StartsWith("decoder.layers.", StringComparison.Ordinal))
            return key;

        string section = encoder ? "encoder" : "decoder";
        string mapped = section + ".model." + key[(section.Length + ".layers.".Length)..];
        int convIdx = mapped.LastIndexOf(".conv.", StringComparison.Ordinal);
        if (convIdx < 0) return mapped;     // LSTM keys etc. need only the layers → model rename

        string head = mapped[..convIdx];
        string param = mapped[(convIdx + ".conv.".Length)..];
        return transposeConv ? $"{head}.convtr.convtr.{param}" : $"{head}.conv.conv.{param}";
    }
}
