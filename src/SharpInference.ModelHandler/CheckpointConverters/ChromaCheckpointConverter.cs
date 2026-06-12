using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.ModelHandler.CheckpointConverters;

/// <summary>Converts <c>lodestones/Chroma</c> single-file safetensors (BFL-style native naming) to
/// diffusers-format weight dictionaries for our C# <see cref="SharpInference.Diffusion.Models.Denoisers.ChromaTransformer"/>.
///
/// Mirrors <c>diffusers/loaders/single_file_utils.py:convert_chroma_transformer_checkpoint_to_diffusers</c>
/// (lines 3491-3657). Top-level changes vs Flux:
/// <list type="bullet">
///   <item>Distilled-guidance approximator (<c>distilled_guidance_layer.{in_proj,out_proj,layers,norms}.*</c>) —
///         BFL stores the inner PixArtAlphaTextProjection linears as <c>in_layer/out_layer</c>; we rename to
///         <c>linear_1/linear_2</c>. RMSNorm scales (<c>norms.{i}.scale</c>) become <c>norms.{i}.weight</c>.</item>
///   <item>Single-block <c>linear1</c> splits 4-way: Q + K + V + proj_mlp. The diffusers naming for the MLP half
///         is <c>proj_mlp</c> (NOT <c>ff.net.0.proj</c> like double blocks), and the attention out is <c>proj_out</c>
///         (NOT <c>attn.to_out.0</c>). The Chroma single block is FluxSingleBlock-shaped, so we reuse the
///         <c>FluxCheckpointConverter</c>-style 4-way split here.</item>
///   <item>NO per-block <c>img_mod</c> / <c>txt_mod</c> / <c>modulation</c> linears in the BFL checkpoint — Chroma's
///         architecture replaces them with the shared approximator, so there's nothing to rename.</item>
///   <item>NO <c>swap_scale_shift</c> for the final norm — Chroma's <c>ChromaAdaLayerNormContinuousPruned</c>
///         consumes its scale+shift from the runtime modulation table, not from a checkpoint linear.</item>
///   <item><c>final_layer.linear.{weight,bias}</c> → <c>proj_out.{weight,bias}</c>.</item>
/// </list>
/// The pipeline-side T5-XXL text encoder + Flux VAE weights are shipped as separate files alongside the
/// Chroma single-file checkpoint and are not handled by this converter (mirrors <see cref="AuraFlowCheckpointConverter"/>).</summary>
public sealed class ChromaCheckpointConverter
{
    /// <summary>Result of converting a single-file Chroma checkpoint. Transformer weights only; text encoder + VAE
    /// must be loaded separately from their own safetensors.</summary>
    public sealed class ConvertedWeights
    {
        /// <summary>Chroma transformer weights in diffusers format.</summary>
        public required Dictionary<string, Tensor> Transformer { get; init; }
    }

    private const int InnerDim = 3072;
    private const int MlpDim = 12288;

    /// <summary>Converts a flat Chroma single-file weight dictionary to diffusers naming. Folds any
    /// ComfyUI / BFL fp8_scaled companion tensors into <see cref="Tensor.Fp8ScaleFactor"/> first.</summary>
    public static ConvertedWeights Convert(Dictionary<string, Tensor> allWeights)
    {
        allWeights = CheckpointConvertUtils.ApplyFp8ScaledDequant(allWeights);

        Dictionary<string, Tensor> transformer = new(2000);

        // First pass: strip "model.diffusion_model." prefix on a per-key basis so the rest of the converter
        // sees the BFL keys directly. Mirrors the Python converter's first loop. Also strip the
        // "_orig_mod." torch.compile artifact prefix that Chroma Radiance checkpoints ship with.
        Dictionary<string, Tensor> bfl = new(allWeights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in allWeights)
        {
            if (kvp.Key.EndsWith(".scaled_fp8") || kvp.Key == "scaled_fp8")
                continue;
            string k = kvp.Key.StartsWith("model.diffusion_model.", StringComparison.Ordinal)
                ? kvp.Key["model.diffusion_model.".Length..]
                : kvp.Key;
            if (k.StartsWith("_orig_mod.", StringComparison.Ordinal))
                k = k["_orig_mod.".Length..];
            bfl[k] = kvp.Value;
        }

        // Detect block counts so we can iterate the indexed sections deterministically.
        int numDoubles = MaxIndex(bfl, "double_blocks.") + 1;
        int numSingles = MaxIndex(bfl, "single_blocks.") + 1;
        int numGuidanceLayers = MaxIndex(bfl, "distilled_guidance_layer.layers.") + 1;

        // ── Distilled-guidance approximator ─────────────────────────────
        TryMove(bfl, "distilled_guidance_layer.in_proj.weight", transformer);
        TryMove(bfl, "distilled_guidance_layer.in_proj.bias", transformer);
        TryMove(bfl, "distilled_guidance_layer.out_proj.weight", transformer);
        TryMove(bfl, "distilled_guidance_layer.out_proj.bias", transformer);

        for (int i = 0; i < numGuidanceLayers; i++)
        {
            // BFL uses in_layer/out_layer; diffusers uses linear_1/linear_2 (PixArtAlphaTextProjection).
            string srcPrefix = $"distilled_guidance_layer.layers.{i}.";
            string dstPrefix = $"distilled_guidance_layer.layers.{i}.";
            TryRename(bfl, $"{srcPrefix}in_layer.weight", $"{dstPrefix}linear_1.weight", transformer);
            TryRename(bfl, $"{srcPrefix}in_layer.bias", $"{dstPrefix}linear_1.bias", transformer);
            TryRename(bfl, $"{srcPrefix}out_layer.weight", $"{dstPrefix}linear_2.weight", transformer);
            TryRename(bfl, $"{srcPrefix}out_layer.bias", $"{dstPrefix}linear_2.bias", transformer);

            // RMSNorm scale: norms.{i}.scale → norms.{i}.weight
            TryRename(bfl, $"distilled_guidance_layer.norms.{i}.scale",
                $"distilled_guidance_layer.norms.{i}.weight", transformer);
        }

        // ── Top-level embedders ─────────────────────────────────────────
        TryRename(bfl, "txt_in.weight", "context_embedder.weight", transformer);
        TryRename(bfl, "txt_in.bias", "context_embedder.bias", transformer);
        TryRename(bfl, "img_in.weight", "x_embedder.weight", transformer);
        TryRename(bfl, "img_in.bias", "x_embedder.bias", transformer);

        // ── Double transformer blocks ───────────────────────────────────
        for (int i = 0; i < numDoubles; i++)
        {
            string srcBase = $"double_blocks.{i}";
            string dstBase = $"transformer_blocks.{i}";

            // Image stream: split fused img_attn.qkv → to_q / to_k / to_v
            SplitFusedQkv(bfl, $"{srcBase}.img_attn.qkv.weight", $"{srcBase}.img_attn.qkv.bias",
                dstBase, "attn.to_q", "attn.to_k", "attn.to_v", InnerDim, transformer);
            // Text stream: split fused txt_attn.qkv → add_q_proj / add_k_proj / add_v_proj
            SplitFusedQkv(bfl, $"{srcBase}.txt_attn.qkv.weight", $"{srcBase}.txt_attn.qkv.bias",
                dstBase, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", InnerDim, transformer);

            // QK-norm scales
            TryRename(bfl, $"{srcBase}.img_attn.norm.query_norm.scale", $"{dstBase}.attn.norm_q.weight", transformer);
            TryRename(bfl, $"{srcBase}.img_attn.norm.key_norm.scale", $"{dstBase}.attn.norm_k.weight", transformer);
            TryRename(bfl, $"{srcBase}.txt_attn.norm.query_norm.scale", $"{dstBase}.attn.norm_added_q.weight", transformer);
            TryRename(bfl, $"{srcBase}.txt_attn.norm.key_norm.scale", $"{dstBase}.attn.norm_added_k.weight", transformer);

            // Output projections
            TryRename(bfl, $"{srcBase}.img_attn.proj.weight", $"{dstBase}.attn.to_out.0.weight", transformer);
            TryRename(bfl, $"{srcBase}.img_attn.proj.bias", $"{dstBase}.attn.to_out.0.bias", transformer);
            TryRename(bfl, $"{srcBase}.txt_attn.proj.weight", $"{dstBase}.attn.to_add_out.weight", transformer);
            TryRename(bfl, $"{srcBase}.txt_attn.proj.bias", $"{dstBase}.attn.to_add_out.bias", transformer);

            // FFN (img_mlp.0/2 → ff.net.0.proj / ff.net.2; txt_mlp same with ff_context)
            TryRename(bfl, $"{srcBase}.img_mlp.0.weight", $"{dstBase}.ff.net.0.proj.weight", transformer);
            TryRename(bfl, $"{srcBase}.img_mlp.0.bias", $"{dstBase}.ff.net.0.proj.bias", transformer);
            TryRename(bfl, $"{srcBase}.img_mlp.2.weight", $"{dstBase}.ff.net.2.weight", transformer);
            TryRename(bfl, $"{srcBase}.img_mlp.2.bias", $"{dstBase}.ff.net.2.bias", transformer);
            TryRename(bfl, $"{srcBase}.txt_mlp.0.weight", $"{dstBase}.ff_context.net.0.proj.weight", transformer);
            TryRename(bfl, $"{srcBase}.txt_mlp.0.bias", $"{dstBase}.ff_context.net.0.proj.bias", transformer);
            TryRename(bfl, $"{srcBase}.txt_mlp.2.weight", $"{dstBase}.ff_context.net.2.weight", transformer);
            TryRename(bfl, $"{srcBase}.txt_mlp.2.bias", $"{dstBase}.ff_context.net.2.bias", transformer);
        }

        // ── Single transformer blocks ───────────────────────────────────
        for (int i = 0; i < numSingles; i++)
        {
            string srcBase = $"single_blocks.{i}";
            string dstBase = $"single_transformer_blocks.{i}";

            // 4-way split of fused linear1 → to_q / to_k / to_v / proj_mlp.
            SplitFusedLinear1Weight(bfl, $"{srcBase}.linear1.weight", dstBase, transformer);
            SplitFusedLinear1Bias(bfl, $"{srcBase}.linear1.bias", dstBase, transformer);

            // QK-norm scales
            TryRename(bfl, $"{srcBase}.norm.query_norm.scale", $"{dstBase}.attn.norm_q.weight", transformer);
            TryRename(bfl, $"{srcBase}.norm.key_norm.scale", $"{dstBase}.attn.norm_k.weight", transformer);

            // linear2 → proj_out (combined attn+mlp output projection)
            TryRename(bfl, $"{srcBase}.linear2.weight", $"{dstBase}.proj_out.weight", transformer);
            TryRename(bfl, $"{srcBase}.linear2.bias", $"{dstBase}.proj_out.bias", transformer);
        }

        // ── Final layer ────────────────────────────────────────────────
        // No `swap_scale_shift` — Chroma's norm_out has no checkpoint linear; its scale/shift come from
        // the runtime modulation table.
        TryRename(bfl, "final_layer.linear.weight", "proj_out.weight", transformer);
        TryRename(bfl, "final_layer.linear.bias", "proj_out.bias", transformer);

        // ── Chroma Radiance pixel-space IO (absent on classic Chroma) ──
        // No diffusers rename exists for these; the C# ChromaRadianceImagePatchifier / ChromaRadianceNerfHead
        // load them under their native names, so they pass through verbatim.
        foreach (KeyValuePair<string, Tensor> kvp in bfl)
        {
            if (kvp.Key.StartsWith("img_in_patch.", StringComparison.Ordinal) ||
                kvp.Key.StartsWith("nerf_", StringComparison.Ordinal))
            {
                transformer[kvp.Key] = kvp.Value;
            }
        }

        return new ConvertedWeights { Transformer = transformer };
    }

    /// <summary>True when a raw (pre- or post-conversion) Chroma-family dict is a Radiance checkpoint —
    /// the NeRF head replaces <c>final_layer</c>. Handles wrapped/compiled key prefixes.</summary>
    public static bool ContainsRadianceKeys(IReadOnlyDictionary<string, Tensor> weights)
    {
        foreach (string key in weights.Keys)
        {
            if (key.Contains("nerf_blocks.", StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    /// <summary>Loads from disk and converts in one shot. Returns the converted weights plus the loader
    /// (the caller is responsible for disposing the loader once weights are no longer needed).</summary>
    public static (ConvertedWeights weights, SafeTensorsLoader loader) LoadAndConvert(string checkpointPath)
    {
        SafeTensorsLoader loader = new();
        loader.Load(checkpointPath);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        ConvertedWeights converted = Convert(raw);
        return (converted, loader);
    }

    /// <summary>Returns the maximum index <c>i</c> seen for keys of the form <c>{prefix}{i}.*</c>, or -1 if absent.</summary>
    private static int MaxIndex(Dictionary<string, Tensor> source, string prefix)
    {
        int max = -1;
        foreach (string key in source.Keys)
        {
            if (!key.StartsWith(prefix, StringComparison.Ordinal)) continue;
            string rest = key[prefix.Length..];
            int dot = rest.IndexOf('.');
            if (dot < 0) continue;
            if (int.TryParse(rest[..dot], out int idx) && idx > max) max = idx;
        }
        return max;
    }

    private static void TryMove(Dictionary<string, Tensor> source, string key, Dictionary<string, Tensor> output)
    {
        if (source.TryGetValue(key, out Tensor? t)) output[key] = t;
    }

    private static void TryRename(Dictionary<string, Tensor> source, string srcKey, string dstKey,
        Dictionary<string, Tensor> output)
    {
        if (source.TryGetValue(srcKey, out Tensor? t)) output[dstKey] = t;
    }

    /// <summary>Splits a BFL fused QKV weight+bias along dim 0 into three diffusers-named tensors. Reuses the
    /// row-copy pattern from <see cref="FluxCheckpointConverter"/>; identical layout per dim 0 split.</summary>
    private static unsafe void SplitFusedQkv(Dictionary<string, Tensor> source,
        string weightKey, string biasKey, string dstPrefix,
        string qName, string kName, string vName, int innerDim, Dictionary<string, Tensor> output)
    {
        if (source.TryGetValue(weightKey, out Tensor? wFused))
        {
            int inDim = (int)wFused.Shape[1];
            long elemBytes = wFused.DType.SizeInBytes;
            long rowBytes = (long)inDim * elemBytes;
            long chunkBytes = (long)innerDim * rowBytes;
            TensorShape splitShape = new TensorShape(innerDim, inDim);

            Tensor qW = new Tensor(splitShape, wFused.DType);
            Tensor kW = new Tensor(splitShape, wFused.DType);
            Tensor vW = new Tensor(splitShape, wFused.DType);
            qW.Fp8ScaleFactor = wFused.Fp8ScaleFactor;
            kW.Fp8ScaleFactor = wFused.Fp8ScaleFactor;
            vW.Fp8ScaleFactor = wFused.Fp8ScaleFactor;

            byte* src = (byte*)wFused.DataPointer;
            Buffer.MemoryCopy(src, (void*)qW.DataPointer, chunkBytes, chunkBytes);
            Buffer.MemoryCopy(src + chunkBytes, (void*)kW.DataPointer, chunkBytes, chunkBytes);
            Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vW.DataPointer, chunkBytes, chunkBytes);

            output[$"{dstPrefix}.{qName}.weight"] = qW;
            output[$"{dstPrefix}.{kName}.weight"] = kW;
            output[$"{dstPrefix}.{vName}.weight"] = vW;
        }

        if (source.TryGetValue(biasKey, out Tensor? bFused))
        {
            long elemBytes = bFused.DType.SizeInBytes;
            long chunkBytes = (long)innerDim * elemBytes;
            TensorShape splitShape = new TensorShape(innerDim);

            Tensor qB = new Tensor(splitShape, bFused.DType);
            Tensor kB = new Tensor(splitShape, bFused.DType);
            Tensor vB = new Tensor(splitShape, bFused.DType);

            byte* src = (byte*)bFused.DataPointer;
            Buffer.MemoryCopy(src, (void*)qB.DataPointer, chunkBytes, chunkBytes);
            Buffer.MemoryCopy(src + chunkBytes, (void*)kB.DataPointer, chunkBytes, chunkBytes);
            Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)vB.DataPointer, chunkBytes, chunkBytes);

            output[$"{dstPrefix}.{qName}.bias"] = qB;
            output[$"{dstPrefix}.{kName}.bias"] = kB;
            output[$"{dstPrefix}.{vName}.bias"] = vB;
        }
    }

    /// <summary>Splits Chroma's single-block fused <c>linear1.weight</c> [3*inner + mlpDim, inner] along dim 0
    /// into <c>(to_q, to_k, to_v, proj_mlp)</c>. Same layout as <see cref="FluxCheckpointConverter"/>'s splitter.</summary>
    private static unsafe void SplitFusedLinear1Weight(Dictionary<string, Tensor> source, string srcKey,
        string dstPrefix, Dictionary<string, Tensor> output)
    {
        if (!source.TryGetValue(srcKey, out Tensor? fused)) return;

        int inDim = (int)fused.Shape[1];
        long rowBytes = (long)inDim * fused.DType.SizeInBytes;
        TensorShape qkvShape = new TensorShape(InnerDim, inDim);
        TensorShape mlpShape = new TensorShape(MlpDim, inDim);

        Tensor qW = new Tensor(qkvShape, fused.DType);
        Tensor kW = new Tensor(qkvShape, fused.DType);
        Tensor vW = new Tensor(qkvShape, fused.DType);
        Tensor mlpW = new Tensor(mlpShape, fused.DType);
        qW.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        kW.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        vW.Fp8ScaleFactor = fused.Fp8ScaleFactor;
        mlpW.Fp8ScaleFactor = fused.Fp8ScaleFactor;

        byte* src = (byte*)fused.DataPointer;
        long qkvChunkBytes = (long)InnerDim * rowBytes;
        long mlpChunkBytes = (long)MlpDim * rowBytes;

        Buffer.MemoryCopy(src, (void*)qW.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + qkvChunkBytes, (void*)kW.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + 2 * qkvChunkBytes, (void*)vW.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + 3 * qkvChunkBytes, (void*)mlpW.DataPointer, mlpChunkBytes, mlpChunkBytes);

        output[$"{dstPrefix}.attn.to_q.weight"] = qW;
        output[$"{dstPrefix}.attn.to_k.weight"] = kW;
        output[$"{dstPrefix}.attn.to_v.weight"] = vW;
        output[$"{dstPrefix}.proj_mlp.weight"] = mlpW;
    }

    private static unsafe void SplitFusedLinear1Bias(Dictionary<string, Tensor> source, string srcKey,
        string dstPrefix, Dictionary<string, Tensor> output)
    {
        if (!source.TryGetValue(srcKey, out Tensor? fused)) return;

        long elemBytes = fused.DType.SizeInBytes;
        TensorShape qkvShape = new TensorShape(InnerDim);
        TensorShape mlpShape = new TensorShape(MlpDim);

        Tensor qB = new Tensor(qkvShape, fused.DType);
        Tensor kB = new Tensor(qkvShape, fused.DType);
        Tensor vB = new Tensor(qkvShape, fused.DType);
        Tensor mlpB = new Tensor(mlpShape, fused.DType);

        byte* src = (byte*)fused.DataPointer;
        long qkvChunkBytes = (long)InnerDim * elemBytes;
        long mlpChunkBytes = (long)MlpDim * elemBytes;

        Buffer.MemoryCopy(src, (void*)qB.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + qkvChunkBytes, (void*)kB.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + 2 * qkvChunkBytes, (void*)vB.DataPointer, qkvChunkBytes, qkvChunkBytes);
        Buffer.MemoryCopy(src + 3 * qkvChunkBytes, (void*)mlpB.DataPointer, mlpChunkBytes, mlpChunkBytes);

        output[$"{dstPrefix}.attn.to_q.bias"] = qB;
        output[$"{dstPrefix}.attn.to_k.bias"] = kB;
        output[$"{dstPrefix}.attn.to_v.bias"] = vB;
        output[$"{dstPrefix}.proj_mlp.bias"] = mlpB;
    }
}
