using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Gguf.Codecs;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

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
/// decode path maps through and the semantic-reconstruction head (<c>decoder_semantic</c> + <c>fc_post1</c>) is always
/// dropped; the encode branch (HuBERT <c>semantic_model</c> + <c>encoder_semantic</c> + acoustic <c>encoder</c> +
/// <c>fc_prior</c>, ~95M extra params) is dropped unless <c>forEncode</c> is set.</para></summary>
public sealed class YueCheckpointConverter
{
    // VALIDATION-GATED (pending a real xcodec_mini_infer key dump): the upstream SoundStream class names its
    // acoustic waveform decoder `decoder_2` (`decoder` is the semantic-reconstruction head); spellings here follow
    // the soundstream_hubert_new.py module names. If a dump shows different roots, extend these tables.
    private static readonly string[] _xCodecWrapperPrefixes = ["codec_model.", "generator.", "model."];
    // fc_post2 is KEPT — it is the 1024->256 acoustic projection on the YuE decode path (the wave decoder's
    // input). Only the semantic-reconstruction branch and the encoder/quantizer-train extras are dropped.
    private static readonly string[] _xCodecDropPrefixes =
    [
        "semantic_model.", "encoder_semantic.", "decoder_semantic.", "decoder_semantic_2.", "encoder.",
        "fc_prior.", "fc_post1.", "fc_post_a.", "fc_post_s.", "discriminator.",
    ];
    // Encode (YuE ICL audio prompt) additionally needs the four roots the decode path drops: the HuBERT semantic
    // model, the RepCodec `encoder_semantic`, the acoustic `encoder` (note the dot — it does NOT match
    // `encoder_semantic.`), and the `fc_prior` fusion projection.
    private static readonly string[] _xCodecEncodeDropPrefixes =
    [
        "decoder_semantic.", "decoder_semantic_2.", "fc_post1.", "fc_post_a.", "fc_post_s.", "discriminator.",
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

    // The 2D GEMM matrices worth quantizing: attention + MLP projections + the LM head. Embeddings (a gather,
    // not a matmul) and the 1D norms stay full-precision.
    private static readonly string[] _quantizableSuffixes =
    [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight", "lm_head.weight",
    ];

    /// <summary>Quantizes the LM's big 2D GEMM weights (attention/MLP projections + lm_head) to <paramref name="target"/>
    /// (a GGUF quant dtype) in place, so a 7B fits a 12 GB card resident instead of streaming per forward. Prefer
    /// <see cref="DType.Q4_K"/>: it fits the 7B in ~3.5 GB AND decode hits the llama.cpp-style dp4a GEMV
    /// (<c>mul_mat_vec_q4k_q8_1</c>), which is ~an order of magnitude faster than the naive Q8_0 F32 GEMV. Only rank-2
    /// tensors whose in-dim is a multiple of the target's block size (256 for Q4_K, so quant blocks never cross a row)
    /// are touched; embeddings/norms are left as-is. Replaced tensors are owned by the dict (the borrowed source stays
    /// valid until its loader disposes).</summary>
    public static unsafe void QuantizeLmWeights(Dictionary<string, Tensor> weights, DType target)
    {
        IGgufCodec codec = GgufCodecRegistry.Get(target);
        if (!codec.SupportsQuantize) throw new ArgumentException($"Codec {target} cannot quantize.", nameof(target));
        int blk = target.BlockElementCount;
        foreach (string key in new List<string>(weights.Keys))
        {
            bool eligible = false;
            foreach (string s in _quantizableSuffixes)
            {
                if (key.EndsWith(s, StringComparison.Ordinal)) { eligible = true; break; }
            }
            if (!eligible) continue;

            Tensor w = weights[key];
            if (w.Shape.Rank != 2 || w.Shape[1] % blk != 0) continue;   // quant blocks must not cross rows

            Tensor f32 = w.DType == DType.F32 ? w : w.CastTo(DType.F32);
            Tensor q = new(w.Shape, target);
            codec.QuantizeFromF32((float*)f32.DataPointer, (byte*)q.DataPointer, w.Shape.ElementCount);
            if (!ReferenceEquals(f32, w)) f32.Dispose();
            weights[key] = q;
        }
    }

    /// <summary>Loads an X-Codec safetensors export for the engine <c>XCodec</c> class. Caller owns the loader.
    /// <paramref name="forEncode"/> additionally keeps the encode branch (HuBERT semantic model, RepCodec
    /// <c>encoder_semantic</c>, acoustic <c>encoder</c>, <c>fc_prior</c>) — ~95M extra params, so leave it off for
    /// the pure-generation path.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadXCodec(string path, bool castToF32 = false, bool forEncode = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = new();
        foreach (string key in loader.Descriptors.Keys)
        {
            string? mapped = MapXCodecKey(key, forEncode);
            if (mapped is not null) weights[mapped] = CodecKeyUtils.MaybeCast(loader.GetTensor(key), castToF32);
        }
        return (weights, loader);
    }

    /// <summary>Loads a converted YuE Vocos vocoder (<c>decoder_131000/151000.pth</c> → safetensors) as-is: the keys
    /// (<c>backbone.embed/norm/convnext.*/final_layer_norm</c>, <c>head.out</c>) are already what <c>VocosDecoder</c>
    /// consumes, so no mapping — just F32 (small: ~18M params).</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadVocoder(string path, bool castToF32 = true)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = new();
        foreach (string key in loader.Descriptors.Keys)
            weights[key] = CodecKeyUtils.MaybeCast(loader.GetTensor(key), castToF32);
        return (weights, loader);
    }

    /// <summary>The mapped key that proves an x-codec export carries the encode branch. <c>fc_prior</c> is the fusion
    /// projection every encode call needs and nothing on the decode path touches, so its presence is exactly the
    /// engine <c>XCodec.CanEncode</c> condition.</summary>
    public const string XCodecEncodeProbeKey = "fc_prior.weight";

    /// <summary>Whether a repacked x-codec export carries the encode roots. Keys are routed through
    /// <see cref="MapXCodecKey"/> rather than compared literally because a repack preserves the source spelling,
    /// which may still carry the <c>codec_model.</c> wrapper prefix. An export written before the encode branch was
    /// kept answers false — the caller must re-repack rather than serve a silently encode-less codec.</summary>
    public static bool XCodecExportHasEncodeRoots(IEnumerable<string> keys)
    {
        foreach (string key in keys)
        {
            if (MapXCodecKey(key, forEncode: true) == XCodecEncodeProbeKey) return true;
        }
        return false;
    }

    /// <summary>Pure X-Codec key mapping (testable without files): strips wrapper prefixes, drops the semantic branch,
    /// renames the acoustic <c>decoder_2.*</c> root to <c>decoder.*</c>, and normalizes weight-norm key spellings.
    /// Weight-norm pairs stay raw — the engine DAC blocks fuse them at load time.</summary>
    public static string? MapXCodecKey(string key, bool forEncode = false)
    {
        foreach (string prefix in _xCodecWrapperPrefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                key = key[prefix.Length..];
                break;
            }
        }
        foreach (string prefix in forEncode ? _xCodecEncodeDropPrefixes : _xCodecDropPrefixes)
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
