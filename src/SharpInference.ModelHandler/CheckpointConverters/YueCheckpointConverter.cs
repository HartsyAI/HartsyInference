using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Loads YuE checkpoints (<c>m-a-p/YuE-s1-7B-*</c>, <c>m-a-p/YuE-s2-1B-general</c>,
/// <c>m-a-p/xcodec_mini_infer</c>) for the <c>YueStage1Lm</c> (Qwen2/LLaMA body) and <c>XCodec</c> classes.
///
/// <para><b>Stage-1 / Stage-2 LMs</b> — standard HF LLaMA layout (<c>model.embed_tokens.weight</c>,
/// <c>model.layers.{i}.self_attn.{q,k,v,o}_proj.weight</c>, <c>mlp.{gate,up,down}_proj.weight</c>,
/// <c>input_layernorm</c>/<c>post_attention_layernorm</c>, <c>model.norm.weight</c>, <c>lm_head.weight</c>), which is
/// exactly what <c>Qwen2Model.LoadWeights(w, "model")</c> consumes — pass-through, dropping the legacy
/// <c>rotary_emb.inv_freq</c> buffers (the engine computes RoPE itself). Sharded HF folders are supported via
/// <c>SafeTensorsShardLoader</c>; single files via <c>SafeTensorsLoader</c>.</para>
///
/// <para><b>X-Codec</b> — the engine <c>XCodec</c> expects DAC naming (<c>encoder.block.*</c>,
/// <c>quantizer.quantizers.{q}.{in_proj,out_proj,codebook}.*</c>, <c>decoder.model.*</c>) with RAW weight-norm pairs
/// (the engine fuses via <c>WeightNormFusion</c>, so the converter must NOT pre-fuse). The xcodec_mini_infer original
/// is a torch <c>.pth</c> — this loader takes a safetensors export of its <c>codec_model</c> state dict. The acoustic
/// path maps through; the training-only semantic branch (HuBERT extractor + semantic encoder/decoder + the
/// <c>fc_prior</c>/<c>fc_post*</c> projections) is dropped.</para></summary>
public sealed class YueCheckpointConverter
{
    // VALIDATION-GATED (pending a real xcodec_mini_infer key dump): the upstream SoundStream class names its
    // acoustic waveform decoder `decoder_2` (`decoder` is the semantic-reconstruction head); spellings here follow
    // the soundstream_hubert_new.py module names. If a dump shows different roots, extend these tables.
    private static readonly string[] XCodecWrapperPrefixes = ["codec_model.", "generator.", "model."];
    private static readonly string[] XCodecDropPrefixes =
    [
        "semantic_model.", "encoder_semantic.", "decoder_semantic.", "decoder_semantic_2.",
        "fc_prior.", "fc_post1.", "fc_post2.", "fc_post_a.", "fc_post_s.", "discriminator.",
    ];

    /// <summary>Loads a Stage-1 LM checkpoint (<c>m-a-p/YuE-s1-7B-anneal-en-cot</c> and siblings). <paramref name="path"/>
    /// may be a sharded HF folder or a single safetensors file. Caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) LoadStage1(string path, bool castToF32 = false)
        => LoadLm(path, castToF32);

    /// <summary>Loads a Stage-2 LM checkpoint (<c>m-a-p/YuE-s2-1B-general</c>) — same LLaMA layout as Stage-1.</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) LoadStage2(string path, bool castToF32 = false)
        => LoadLm(path, castToF32);

    /// <summary>Loads any YuE LLaMA-layout LM from a sharded folder or single safetensors file.</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) LoadLm(string path, bool castToF32 = false)
    {
        if (Directory.Exists(path))
        {
            SafeTensorsShardLoader shards = new();
            try
            {
                shards.LoadDirectory(path);
                return (MapLmTensors(shards.TensorNames, shards.GetTensor, castToF32), shards);
            }
            catch
            {
                shards.Dispose();
                throw;
            }
        }
        if (!File.Exists(path))
            throw new FileNotFoundException($"YuE LM checkpoint not found: {path}");
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return (MapLmTensors(loader.Descriptors.Keys, loader.GetTensor, castToF32), loader);
    }

    /// <summary>Pure LM key mapping (testable without files): pass-through, with recomputed RoPE buffers dropped.</summary>
    public static string? MapLmKey(string key) =>
        key.EndsWith(".rotary_emb.inv_freq", StringComparison.Ordinal) ? null : key;

    /// <summary>Loads an X-Codec safetensors export for the engine <c>XCodec</c> class. Caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadXCodec(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = new();
        foreach (string key in loader.Descriptors.Keys)
        {
            string? mapped = MapXCodecKey(key);
            if (mapped is not null) weights[mapped] = CodecKeyUtils.MaybeCast(loader.GetTensor(key), castToF32);
        }
        return (weights, loader);
    }

    /// <summary>Pure X-Codec key mapping (testable without files): strips wrapper prefixes, drops the semantic branch,
    /// renames the acoustic <c>decoder_2.*</c> root to <c>decoder.*</c>, and normalizes weight-norm key spellings.
    /// Weight-norm pairs stay raw — the engine DAC blocks fuse them at load time.</summary>
    public static string? MapXCodecKey(string key)
    {
        foreach (string prefix in XCodecWrapperPrefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                key = key[prefix.Length..];
                break;
            }
        }
        foreach (string prefix in XCodecDropPrefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
                return null;
        }
        if (key.StartsWith("decoder_2.", StringComparison.Ordinal))
            key = "decoder." + key["decoder_2.".Length..];
        return CodecKeyUtils.NormalizeWeightNormKey(key);
    }

    private static Dictionary<string, Tensor> MapLmTensors(IEnumerable<string> keys, Func<string, Tensor> getTensor, bool castToF32)
    {
        Dictionary<string, Tensor> weights = new();
        foreach (string key in keys)
        {
            string? mapped = MapLmKey(key);
            if (mapped is not null) weights[mapped] = CodecKeyUtils.MaybeCast(getTensor(key), castToF32);
        }
        return weights;
    }
}
