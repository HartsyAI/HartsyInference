using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads MusicGen / AudioGen checkpoints (<c>facebook/musicgen-{small,medium,large}</c>, <c>facebook/encodec_32khz</c>) for the <c>MusicGenDecoder</c> / <c>EnCodec</c> classes.
///
/// <para><b>Decoder</b> — handles both HF exports by prefix probing: the combined <c>MusicgenForConditionalGeneration</c> file (<c>decoder.model.decoder.*</c> / <c>decoder.lm_heads.*</c> / top-level <c>enc_to_dec_proj.*</c>, with <c>text_encoder.*</c> + <c>audio_encoder.*</c> siblings dropped) and a standalone <c>MusicgenForCausalLM</c> dump (<c>model.decoder.*</c> / <c>lm_heads.*</c>, pass-through). The sinusoidal <c>embed_positions</c> buffer is dropped — the engine recomputes positions.</para>
///
/// <para><b>EnCodec</b> — maps the HF <c>transformers</c> <c>EncodecModel</c> naming (<c>encoder.layers.{i}.conv.*</c>, <c>quantizer.layers.{q}.codebook.embed</c>) to the Meta AudioCraft naming the engine expects (<c>encoder.model.{i}.conv.conv.*</c>, <c>quantizer.vq.layers.{q}._codebook.embed</c>, <c>decoder.model.{i}.convtr.convtr.*</c>). Transpose convs are detected per-module from the <c>weight_g</c> shape (PyTorch weight-norms <c>ConvTranspose1d</c> on dim=1 → <c>[1, C_out, 1]</c> vs <c>[C_out, 1, 1]</c> for <c>Conv1d</c>). Weight-norm pairs stay RAW (no fusion) — <c>SeaNetEncoder</c>/<c>SeaNetDecoder</c> fuse them via <c>WeightNormFusion</c> at <c>LoadWeights</c> time, unlike the ACE-Step vocoder path which fuses in the converter. Meta-named dumps and the EnCodec embedded in a combined MusicGen file (<c>audio_encoder.*</c>) also load.</para></summary>
public sealed unsafe class MusicGenCheckpointConverter
{
    /// <summary>Pure decoder key mapping (testable without files). Returns <c>null</c> for keys the decoder never consumes (T5 text encoder, embedded EnCodec, sinusoidal position buffer).</summary>
    public static string? MapDecoderKey(string key)
    {
        if (key.StartsWith("text_encoder.", StringComparison.Ordinal) ||
            key.StartsWith("audio_encoder.", StringComparison.Ordinal))
            return null;
        if (key.Contains(".embed_positions.", StringComparison.Ordinal))
            return null;
        return CodecKeyUtils.StripPrefix(key, "decoder.");
    }

    /// <summary>Loads the MusicGen decoder safetensors (combined or standalone layout). Caller owns the loader. For AudioGen (which ships PyTorch <c>.bin</c> pickles, not safetensors) use <see cref="LoadDecoderAny"/>.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadDecoder(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return (MapDecoderWeights(loader.GetAllTensors(), castToF32), loader);
    }

    /// <summary>Loads an EnCodec safetensors — either <c>facebook/encodec_32khz</c> standalone or the <c>audio_encoder.*</c> subset of a combined MusicGen file. Caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadEnCodec(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return (MapEnCodecWeights(loader.GetAllTensors(), castToF32), loader);
    }

    /// <summary>Loads the T5-base text encoder weights — either the <c>text_encoder.*</c> subset of a combined MusicGen / AudioGen file or a standalone HF T5 dump. Only the <c>text_encoder.</c> prefix is stripped; the inner HF T5 key names (<c>shared.weight</c>, <c>encoder.block.{i}.layer.*</c>, <c>encoder.final_layer_norm.*</c>) are what <c>T5TextEncoder.LoadWeights</c> already consumes for the standalone (Flux/SD3) path, so no further remapping is needed. Caller owns the loader. Pair with <see cref="T5TextEncoderConfig.T5Base"/>.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadTextEncoder(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return (MapTextEncoderWeights(loader.GetAllTensors(), castToF32), loader);
    }

    // ── Format-agnostic loaders (safetensors OR PyTorch .bin/.pt pickle) ──────────────────────────────────────
    // AudioGen (facebook/audiogen-medium) and the standalone google/t5-base it pairs with ship as PyTorch pickles,
    // not safetensors. These dispatch by file extension and apply the same key mapping as the safetensors loaders.

    /// <summary>Loads the MusicGen / AudioGen decoder from a <c>.safetensors</c> OR <c>.bin</c>/<c>.pt</c> pickle. Caller owns (and must dispose) the returned loader.</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) LoadDecoderAny(string path, bool castToF32 = false)
    {
        (Dictionary<string, Tensor> raw, IDisposable loader) = LoadAny(path);
        return (MapDecoderWeights(raw, castToF32), loader);
    }

    /// <summary>Loads EnCodec from a <c>.safetensors</c> OR <c>.bin</c>/<c>.pt</c> pickle (AudioGen pairs with the 16 kHz EnCodec). Caller owns (and must dispose) the returned loader.</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) LoadEnCodecAny(string path, bool castToF32 = false)
    {
        (Dictionary<string, Tensor> raw, IDisposable loader) = LoadAny(path);
        return (MapEnCodecWeights(raw, castToF32), loader);
    }

    /// <summary>Loads the T5-base text encoder from a <c>.safetensors</c> OR <c>.bin</c>/<c>.pt</c> pickle (the standalone <c>google/t5-base</c> AudioGen fetches separately ships a <c>pytorch_model.bin</c>). Caller owns (and must dispose) the returned loader. Pair with <see cref="T5TextEncoderConfig.T5Base"/>.</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) LoadTextEncoderAny(string path, bool castToF32 = false)
    {
        (Dictionary<string, Tensor> raw, IDisposable loader) = LoadAny(path);
        return (MapTextEncoderWeights(raw, castToF32), loader);
    }

    // ── One-entry load-once cache (combined-file dedup) ──────────────────────────────────────────────────────
    // The combined MusicGen small/medium checkpoint holds decoder + EnCodec + T5 in ONE file, and both AudioLab's
    // MusicModels loader and the CLI MusicHandler call LoadDecoderAny/LoadEnCodecAny/LoadTextEncoderAny back-to-back
    // on that same path. Without dedup each call re-opened and re-parsed the whole file — 3× the parsed-tensor host
    // RAM (which OOM-killed the server on medium) and 3× the load time. This cache parses a path once and shares the
    // parsed dict + one underlying loader across those calls; the loader is reference-counted and disposed exactly
    // once, when the last handle returned to a caller is disposed. Only the LAST path is remembered — a different
    // path (the AudioCraft 3-different-files branch) replaces it, and a full release clears it, so nothing leaks.
    private static readonly object _cacheLock = new();
    private static string? _cachedPath;
    private static Dictionary<string, Tensor>? _cachedRaw;
    private static RefCountedLoader? _cachedLoader;

    /// <summary>Reads every tensor from a checkpoint file, choosing the reader by extension: <c>.safetensors</c> → <see cref="SafeTensorsLoader"/>, otherwise (<c>.bin</c>/<c>.pt</c>/<c>.pth</c>) → <see cref="PytorchPickleLoader"/>. Parses the file once per path across the back-to-back decoder/EnCodec/T5 calls (see the cache above); each returned loader is an independent handle over one shared, reference-counted underlying loader.</summary>
    private static (Dictionary<string, Tensor> Raw, IDisposable Loader) LoadAny(string path)
    {
        lock (_cacheLock)
        {
            if (_cachedLoader is not null && _cachedRaw is not null && _cachedPath == path)
            {
                _cachedLoader.AddRef();
                return (_cachedRaw, new Handle(_cachedLoader));
            }
        }
        // Cache miss: parse the file once, outside the lock (parsing is I/O + allocation heavy).
        (Dictionary<string, Tensor> raw, IDisposable inner) = ReadRaw(path);
        lock (_cacheLock)
        {
            // A concurrent load may have published the same path while we parsed — prefer the cached copy, drop ours.
            if (_cachedLoader is not null && _cachedRaw is not null && _cachedPath == path)
            {
                inner.Dispose();
                _cachedLoader.AddRef();
                return (_cachedRaw, new Handle(_cachedLoader));
            }
            RefCountedLoader shared = new(inner);
            shared.AddRef();
            _cachedPath = path;
            _cachedRaw = raw;
            _cachedLoader = shared;
            return (raw, new Handle(shared));
        }
    }

    /// <summary>Extension-dispatched raw read (no caching) backing <see cref="LoadAny"/>.</summary>
    private static (Dictionary<string, Tensor> Raw, IDisposable Loader) ReadRaw(string path)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader st = new();
            st.Load(path);
            return (st.GetAllTensors(), st);
        }
        PytorchPickleLoader pt = new();
        pt.Load(path);
        return (pt.GetAllTensors(), pt);
    }

    /// <summary>Wraps the one underlying loader shared by the combined-file consumers; the underlying loader is disposed exactly once, when the reference count (one per live <see cref="Handle"/>) reaches zero. All refcount mutations run under <see cref="_cacheLock"/>.</summary>
    private sealed class RefCountedLoader(IDisposable inner)
    {
        private readonly IDisposable _inner = inner;
        private int _refs;
        public void AddRef() => _refs++;
        /// <summary>Drops one reference; returns true when this call freed the underlying loader.</summary>
        public bool Release()
        {
            if (--_refs > 0) return false;
            _inner.Dispose();
            return true;
        }
    }

    /// <summary>Per-consumer disposable over a shared <see cref="RefCountedLoader"/>. Idempotent: disposing more than once is a no-op, so a caller that hands the same handle to several owners can't over-release.</summary>
    private sealed class Handle(RefCountedLoader shared) : IDisposable
    {
        private RefCountedLoader? _shared = shared;
        public void Dispose()
        {
            RefCountedLoader? s = _shared;
            if (s is null) return;
            _shared = null;
            lock (_cacheLock)
            {
                // If this was the last reference and the cache still points at it, clear the (now-stale) entry.
                if (s.Release() && ReferenceEquals(_cachedLoader, s))
                {
                    _cachedPath = null;
                    _cachedRaw = null;
                    _cachedLoader = null;
                }
            }
        }
    }

    /// <summary>True for the big 2-D projection weights that the MusicGen decoder / T5 encoder consume ONLY via <c>backend.Linear</c> (q/k/v/out projections, FFN fc1/fc2 &amp; T5 wi/wo, the K <c>lm_heads</c>, and <c>enc_to_dec_proj</c>). These never touch host pointer-math, and the GPU GEMM casts a non-F32 weight itself, so we keep them in their native (fp16) checkpoint dtype even when <paramref name="castToF32"/> is requested — F32-upconverting them at load doubles host RAM and is what OOM'd AudioGen. Embeddings, norms, biases, the T5 relative-position table, and the EnCodec convs/codebooks are NOT matched here: they are read by host math and still honor <c>castToF32</c> (and their own <c>LoadWeights</c> re-<c>EnsureF32</c>s the host-touched ones).</summary>
    public static bool IsGpuMatmulWeight(string mappedKey)
    {
        if (mappedKey.EndsWith(".q_proj.weight", StringComparison.Ordinal) ||
            mappedKey.EndsWith(".k_proj.weight", StringComparison.Ordinal) ||
            mappedKey.EndsWith(".v_proj.weight", StringComparison.Ordinal) ||
            mappedKey.EndsWith(".out_proj.weight", StringComparison.Ordinal) ||
            mappedKey.EndsWith(".fc1.weight", StringComparison.Ordinal) ||
            mappedKey.EndsWith(".fc2.weight", StringComparison.Ordinal) ||
            mappedKey == "enc_to_dec_proj.weight" ||
            (mappedKey.StartsWith("lm_heads.", StringComparison.Ordinal) && mappedKey.EndsWith(".weight", StringComparison.Ordinal)))
            return true;
        // T5 encoder (MusicGen text tower): self-attention q/k/v/o + gated/non-gated FFN wi*/wo.
        if (mappedKey.Contains(".SelfAttention.", StringComparison.Ordinal) &&
            (mappedKey.EndsWith(".q.weight", StringComparison.Ordinal) || mappedKey.EndsWith(".k.weight", StringComparison.Ordinal) ||
             mappedKey.EndsWith(".v.weight", StringComparison.Ordinal) || mappedKey.EndsWith(".o.weight", StringComparison.Ordinal)))
            return true;
        if (mappedKey.Contains(".DenseReluDense.", StringComparison.Ordinal) &&
            (mappedKey.EndsWith(".wi.weight", StringComparison.Ordinal) || mappedKey.EndsWith(".wi_0.weight", StringComparison.Ordinal) ||
             mappedKey.EndsWith(".wi_1.weight", StringComparison.Ordinal) || mappedKey.EndsWith(".wo.weight", StringComparison.Ordinal)))
            return true;
        return false;
    }

    /// <summary>Honors <paramref name="castToF32"/> EXCEPT for the big <see cref="IsGpuMatmulWeight"/> projections, which stay in their native (fp16) checkpoint dtype to halve host RAM — the fix that lets AudioGen's F32 load fit. These weights are consumed only by <c>backend.Linear</c>, whose cuBLAS GEMM upconverts an fp16 weight and writes an F32 output tensor, so no F16 activation ever reaches a host/elementwise op. (An earlier "F16 reached T5's ReLU" crash was actually a stale-PTX CUDA→Vulkan fallback, not this path.)</summary>
    private static Tensor MaybeCastKeepMatmulNative(string mappedKey, Tensor value, bool castToF32) =>
        CodecKeyUtils.MaybeCast(value, castToF32 && !IsGpuMatmulWeight(mappedKey));

    /// <summary>Applies <see cref="MapDecoderKey"/> across a raw name→tensor dictionary (any source format).</summary>
    public static Dictionary<string, Tensor> MapDecoderWeights(IReadOnlyDictionary<string, Tensor> raw, bool castToF32 = false)
    {
        if (IsAudioCraftDecoder(raw)) return ConvertAudioCraftDecoder(raw, castToF32);
        Dictionary<string, Tensor> weights = new();
        foreach ((string key, Tensor value) in raw)
        {
            string? mapped = MapDecoderKey(key);
            if (mapped is not null) weights[mapped] = MaybeCastKeepMatmulNative(mapped, value, castToF32);
        }
        return weights;
    }

    /// <summary>True when <paramref name="raw"/> is a Meta AudioCraft single-file LM dump (AudioGen / MusicGen-large), keyed by <c>emb.k.weight</c> + <c>transformer.layers.i.*</c> rather than HF <c>model.decoder.*</c> names.</summary>
    public static bool IsAudioCraftDecoder(IReadOnlyDictionary<string, Tensor> raw) =>
        raw.ContainsKey("emb.0.weight") || raw.ContainsKey("transformer.layers.0.self_attn.in_proj_weight");

    /// <summary>Remaps a Meta AudioCraft LM dump to the HF <c>MusicgenForCausalLM</c> names the engine <c>MusicGenDecoder.LoadWeights(prefix:"model.decoder")</c> consumes. The fused <c>self_attn/cross_attention.in_proj_weight</c> <c>[3d,d]</c> is split into q/k/v <c>[d,d]</c> row-blocks (Q, K, V order — matching torch <c>nn.MultiheadAttention</c> and HF's <c>convert_musicgen_transformers.py</c>).</summary>
    public static Dictionary<string, Tensor> ConvertAudioCraftDecoder(IReadOnlyDictionary<string, Tensor> raw, bool castToF32 = false)
    {
        Dictionary<string, Tensor> o = new(raw.Count + 128);
        foreach ((string key, Tensor value) in raw)
        {
            if (key.StartsWith("transformer.layers.", StringComparison.Ordinal))
            {
                if (key.EndsWith(".self_attn.in_proj_weight", StringComparison.Ordinal))
                { SplitInProj(value, LayerModulePrefix(key, "self_attn"), o); continue; }
                if (key.EndsWith(".cross_attention.in_proj_weight", StringComparison.Ordinal))
                { SplitInProj(value, LayerModulePrefix(key, "encoder_attn"), o); continue; }
            }
            string? mapped = MapAudioCraftDecoderKey(key);
            if (mapped is not null) o[mapped] = MaybeCastKeepMatmulNative(mapped, value, castToF32);
        }
        return o;
    }

    /// <summary>Pure AudioCraft → HF decoder key mapping (testable without files). Returns <c>null</c> for the fused <c>in_proj_weight</c> keys (split separately by <see cref="ConvertAudioCraftDecoder"/>) and any key the decoder never consumes (optional <c>in_proj_bias</c>, stray metadata).</summary>
    public static string? MapAudioCraftDecoderKey(string key)
    {
        if (key.StartsWith("emb.", StringComparison.Ordinal) && key.EndsWith(".weight", StringComparison.Ordinal))
            return "model.decoder.embed_tokens." + key["emb.".Length..];
        if (key.StartsWith("linears.", StringComparison.Ordinal) && key.EndsWith(".weight", StringComparison.Ordinal))
            return "lm_heads." + key["linears.".Length..];
        if (key == "out_norm.weight") return "model.decoder.layer_norm.weight";
        if (key == "out_norm.bias") return "model.decoder.layer_norm.bias";
        const string op = "condition_provider.conditioners.description.output_proj.";
        if (key.StartsWith(op, StringComparison.Ordinal)) return "enc_to_dec_proj." + key[op.Length..];
        if (key.StartsWith("transformer.layers.", StringComparison.Ordinal))
        {
            // Fused qkv is split by the caller; the out_proj / norms / FFN map 1:1.
            if (key.EndsWith(".self_attn.in_proj_weight", StringComparison.Ordinal) ||
                key.EndsWith(".cross_attention.in_proj_weight", StringComparison.Ordinal))
                return null;
            string s = "model.decoder." + key["transformer.".Length..];
            s = s.Replace(".cross_attention.", ".encoder_attn.", StringComparison.Ordinal);
            s = s.Replace(".norm_cross.", ".encoder_attn_layer_norm.", StringComparison.Ordinal);
            s = s.Replace(".norm1.", ".self_attn_layer_norm.", StringComparison.Ordinal);
            s = s.Replace(".norm2.", ".final_layer_norm.", StringComparison.Ordinal);
            s = s.Replace(".linear1.", ".fc1.", StringComparison.Ordinal);
            s = s.Replace(".linear2.", ".fc2.", StringComparison.Ordinal);
            return s;
        }
        return null;
    }

    /// <summary>Builds <c>model.decoder.layers.{i}.{hfModule}</c> from a <c>transformer.layers.{i}.*</c> key.</summary>
    private static string LayerModulePrefix(string key, string hfModule)
    {
        int start = "transformer.layers.".Length;
        int dot = key.IndexOf('.', start);
        string idx = key[start..dot];
        return $"model.decoder.layers.{idx}.{hfModule}";
    }

    /// <summary>Splits a fused <c>in_proj_weight</c> <c>[3d,d]</c> into <c>{modulePrefix}.{q,k,v}_proj.weight</c> <c>[d,d]</c> row-blocks (Q, K, V order).</summary>
    private static void SplitInProj(Tensor fused, string modulePrefix, Dictionary<string, Tensor> o)
    {
        // The q/k/v projections are big backend.Linear-only weights (IsGpuMatmulWeight) — keep them native fp16
        // to save host RAM, matching the non-fused decoder/T5 path.
        int h = (int)fused.Shape[0] / 3;
        o[$"{modulePrefix}.q_proj.weight"] = RowSlice(fused, 0, h);
        o[$"{modulePrefix}.k_proj.weight"] = RowSlice(fused, h, h);
        o[$"{modulePrefix}.v_proj.weight"] = RowSlice(fused, 2 * h, h);
    }

    /// <summary>Copies rows <c>[startRow, startRow+numRows)</c> of <paramref name="src"/> into a new owned tensor (same dtype). A "row" is the product of all dims after dim-0.</summary>
    private static Tensor RowSlice(Tensor src, int startRow, int numRows)
    {
        long cols = src.Shape.Rank == 1 ? 1 : src.ElementCount / src.Shape[0];
        int esz = src.DType.SizeInBytes;
        long rowBytes = cols * esz;
        Tensor dst = new(new TensorShape(numRows, cols), src.DType);
        byte* sp = (byte*)src.DataPointer + startRow * rowBytes;
        Buffer.MemoryCopy(sp, (void*)dst.DataPointer, numRows * rowBytes, numRows * rowBytes);
        return dst;
    }

    /// <summary>Strips the combined <c>audio_encoder.</c> envelope (if present) then remaps EnCodec HF naming to Meta naming via <see cref="ConvertEnCodec"/>. Source-format agnostic.</summary>
    public static Dictionary<string, Tensor> MapEnCodecWeights(IReadOnlyDictionary<string, Tensor> raw, bool castToF32 = false)
    {
        bool combined = raw.Keys.Any(k => k.StartsWith("audio_encoder.", StringComparison.Ordinal));
        Dictionary<string, Tensor> stripped = new();
        foreach ((string key, Tensor value) in raw)
        {
            if (combined && !key.StartsWith("audio_encoder.", StringComparison.Ordinal)) continue;
            string name = combined ? key["audio_encoder.".Length..] : key;
            stripped[name] = CodecKeyUtils.MaybeCast(value, castToF32);
        }
        return ConvertEnCodec(stripped);
    }

    /// <summary>Strips the combined <c>text_encoder.</c> envelope (if present); the inner HF T5 names are what <c>T5TextEncoder.LoadWeights</c> consumes directly. Source-format agnostic.</summary>
    public static Dictionary<string, Tensor> MapTextEncoderWeights(IReadOnlyDictionary<string, Tensor> raw, bool castToF32 = false)
    {
        bool combined = raw.Keys.Any(k => k.StartsWith("text_encoder.", StringComparison.Ordinal));
        Dictionary<string, Tensor> weights = new();
        foreach ((string key, Tensor value) in raw)
        {
            if (combined && !key.StartsWith("text_encoder.", StringComparison.Ordinal)) continue;
            string name = combined ? key["text_encoder.".Length..] : key;
            weights[name] = MaybeCastKeepMatmulNative(name, value, castToF32);
        }
        return weights;
    }

    /// <summary>Converts an EnCodec weight dictionary from HF <c>transformers</c> naming to the Meta naming the engine <c>EnCodec</c> loads; Meta-named dictionaries pass through. Weight-norm pairs are kept raw — the engine fuses.</summary>
    public static Dictionary<string, Tensor> ConvertEnCodec(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> normalized = new(raw.Count);
        foreach ((string key, Tensor value) in raw)
            normalized[CodecKeyUtils.NormalizeWeightNormKey(key)] = value;

        // The decoder's upsampling convs are PyTorch ConvTranspose1d (EncodecConvTranspose1d). In the HF Sequential
        // they are every top-level `decoder.layers.{i}.conv.*` EXCEPT the first (index 0, the latent→bottleneck
        // stem) and the last (the bottleneck→audio projection). They CANNOT be told apart from regular Conv1d by
        // the `weight_g` shape: HF/AudioCraft weight-norm both convs with the default `dim=0`, giving
        // `weight_g = [out_or_in, 1, 1]` in either case. So we identify them structurally by their Sequential
        // index: the min and max top-level `decoder.layers.{i}.conv` indices are the Conv1d stem/head; all the
        // others are ConvTranspose1d. (`.block.` convs inside resnet blocks are always Conv1d.)
        HashSet<string> transposeModules = new(StringComparer.Ordinal);
        SortedSet<int> decoderTopConvIdx = new();
        foreach (string key in normalized.Keys)
        {
            if (TryParseDecoderTopConvIndex(key, out int idx)) decoderTopConvIdx.Add(idx);
        }
        if (decoderTopConvIdx.Count > 2)
        {
            int first = decoderTopConvIdx.Min;
            int last = decoderTopConvIdx.Max;
            foreach (int idx in decoderTopConvIdx)
                if (idx != first && idx != last)
                    transposeModules.Add($"decoder.layers.{idx}.conv");
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

    /// <summary>True when <paramref name="key"/> is a top-level decoder conv parameter (<c>decoder.layers.{i}.conv.*</c>, NOT a resnet-block <c>.block.</c> conv), parsing the Sequential index <paramref name="idx"/>. Used to classify which top-level decoder convs are <c>ConvTranspose1d</c>.</summary>
    private static bool TryParseDecoderTopConvIndex(string key, out int idx)
    {
        idx = -1;
        const string pre = "decoder.layers.";
        if (!key.StartsWith(pre, StringComparison.Ordinal)) return false;
        string rest = key[pre.Length..];
        int dot = rest.IndexOf('.');
        if (dot < 0) return false;
        if (!int.TryParse(rest[..dot], out idx)) return false;
        // Top-level conv only: the immediate next segment must be "conv" (a `.block.` resnet conv is nested deeper).
        return rest[(dot + 1)..].StartsWith("conv.", StringComparison.Ordinal);
    }

    /// <summary>Pure EnCodec key mapping (testable without files). <paramref name="transposeConv"/> marks the key's conv module as a <c>ConvTranspose1d</c> (decoder upsamplers) → <c>convtr.convtr</c> instead of <c>conv.conv</c>. Returns <c>null</c> for training-only quantizer buffers (<c>inited</c>/<c>cluster_size</c>/<c>embed_avg</c>); keys not in the HF layout pass through unchanged.</summary>
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
